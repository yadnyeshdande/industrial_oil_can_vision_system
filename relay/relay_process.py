"""
relay/relay_process.py  v3.0
=============================
Dual-backend relay process: USB (pyhid_usb_relay) + Ethernet (Waveshare Modbus TCP).

Architecture
============
  Only ONE backend is active at a time.
  Backend is selectable at runtime via GUI radio buttons without restarting
  detection, cameras, or the supervisor.

  Supervisor
  └── Relay Process
        ├── USB Backend      (pyhid_usb_relay)
        └── Modbus Backend   (Waveshare Modbus TCP)

Safe switching sequence (Step 11 of the design guide):
  1  Freeze writes           (self._switching = True)
  2  Turn all outputs OFF    (old_backend.all_off)
  3  Disconnect old backend  (old_backend.disconnect)
  4  Create new backend
  5  Connect new backend
  6  Health check            → rollback if failed
  7  Restore cached states   (re-energise any relay that was ON)
  8  Resume writes           (self._switching = False)

Failure rollback:
  If the new backend fails to connect or fails health_check,
  the process automatically reconnects the old backend before resuming.

Preserved from v2.2:
  • 9-relay support with per-camera mapping
  • Periodic re-sync of ON relays (silent hardware failure guard)
  • Consecutive-failure reinit
  • Memory limit watchdog
  • Heartbeat with relay states
"""
from __future__ import annotations
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path
from multiprocessing import Queue
from typing import List, Optional

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config_loader import VisionSystemConfig, RelayConfig
from core.ipc_schema import (
    MessageType, ProcessSource,
    RelayStateMessage,
    make_heartbeat, make_error,
)
from core.logging_setup import setup_process_logging, setup_crash_handler
from core.resource_monitor import get_process_memory_mb, is_memory_over_limit
from relay.backends.base_backend import RelayBackend

logger = logging.getLogger(__name__)

# Seconds between unconditional re-writes of every cached-ON relay.
_RESYNC_INTERVAL = 30.0


# ── Simulated backend (local fallback — never imported from backends/) ─────────
class _SimulatedBackend(RelayBackend):
    """
    In-memory backend used when hardware is unavailable.
    Keeps the relay process alive so the system degrades gracefully.
    """
    def __init__(self, relay_count: int):
        self._n      = relay_count
        self._states = [False] * relay_count
        self._err    = ""

    @property
    def last_error(self) -> str:
        return self._err

    def connect(self) -> bool:
        logger.warning("[SimBackend] SIMULATED relay — physical outputs will NOT fire")
        return True

    def disconnect(self):
        logger.info("[SimBackend] Disconnected (simulated)")

    def relay_on(self, idx: int) -> bool:
        if 0 <= idx < self._n:
            self._states[idx] = True
            logger.debug("[SimBackend] Relay %d ON (simulated)", idx + 1)
            return True
        return False

    def relay_off(self, idx: int) -> bool:
        if 0 <= idx < self._n:
            self._states[idx] = False
            logger.debug("[SimBackend] Relay %d OFF (simulated)", idx + 1)
            return True
        return False

    def read_relay(self, idx: int):
        return self._states[idx] if 0 <= idx < self._n else None

    def all_off(self) -> bool:
        self._states = [False] * self._n
        return True

    def health_check(self) -> bool:
        return True


# ── Backend factory ────────────────────────────────────────────────────────────
def _create_backend(name: str, rcfg: RelayConfig) -> RelayBackend:
    """
    Instantiate a backend by name.
    Falls back to _SimulatedBackend if the import fails or config disables it.
    """
    name = name.lower()

    if name == "usb":
        try:
            from relay.backends.usb_backend import USBRelayBackend
            return USBRelayBackend(relay_count=rcfg.relay_count)
        except Exception as e:
            logger.error("[RelayFactory] USB backend unavailable: %s — using simulated", e)
            return _SimulatedBackend(rcfg.relay_count)

    if name == "modbus":
        try:
            from relay.backends.modbus_backend import ModbusRelayBackend
            mcfg = rcfg.modbus
            return ModbusRelayBackend(
                ip=mcfg.ip,
                port=mcfg.port,
                device_id=mcfg.device_id,
                timeout_seconds=mcfg.timeout_seconds,
                relay_count=rcfg.relay_count,
                retry_attempts=rcfg.retry_attempts,
            )
        except Exception as e:
            logger.error("[RelayFactory] Modbus backend unavailable: %s — using simulated", e)
            return _SimulatedBackend(rcfg.relay_count)

    logger.error("[RelayFactory] Unknown backend '%s' — using simulated", name)
    return _SimulatedBackend(rcfg.relay_count)


# ── Main worker ────────────────────────────────────────────────────────────────
class RelayWorker:

    def __init__(
        self,
        cfg:              VisionSystemConfig,
        result_queue:     Queue,
        state_out_queue:  Queue,
        heartbeat_queue:  Queue,
        stop_event,
    ):
        self.cfg           = cfg
        self.rcfg: RelayConfig = cfg.relay
        self.result_queue  = result_queue
        self.state_out_queue = state_out_queue
        self.heartbeat_queue = heartbeat_queue
        self.stop_event    = stop_event

        self.pid  = os.getpid()
        self.name = "Relay"

        # ── Active backend state ──────────────────────────────────────────
        self.active_backend_name: str = self.rcfg.active_backend   # "usb" or "modbus"
        self.backend: Optional[RelayBackend] = None

        # ── Relay state cache — INDEPENDENT of backend ───────────────────
        # Survives backend switches.  Without this, switching clears relay
        # knowledge and results in false alarms.
        self._cached_states: List[bool] = [False] * self.rcfg.relay_count

        # ── Switching guard ───────────────────────────────────────────────
        self._switching        = False          # freeze writes during switch
        self._last_switch_time = 0.0            # for cooldown enforcement

        # ── Backend health tracking ───────────────────────────────────────
        self._backend_healthy    = False
        self._last_backend_error = ""

        # ── Failure / reinit counters ─────────────────────────────────────
        self._consecutive_failures = 0

        # ── Timers ────────────────────────────────────────────────────────
        self._last_heartbeat     = time.time()
        self._last_resync        = time.time()
        self._last_health_check  = time.time()

    # ══════════════════════════════════════════════════════════════════════
    # Lifecycle
    # ══════════════════════════════════════════════════════════════════════
    def run(self):
        logger.info(
            "[Relay] PID=%d starting  backend=%s  relays=%d",
            self.pid, self.active_backend_name, self.rcfg.relay_count,
        )

        # Startup — connect chosen backend and clear all outputs
        self.backend = _create_backend(self.active_backend_name, self.rcfg)
        if self.backend.connect():
            self.backend.all_off()
            self._backend_healthy    = True
            self._last_backend_error = ""
            logger.info("[Relay] Backend '%s' connected and outputs cleared",
                        self.active_backend_name)
            self._blink_test()
        else:
            self._backend_healthy    = False
            self._last_backend_error = self.backend.last_error
            logger.warning("[Relay] Backend '%s' connect failed at startup — "
                           "outputs will NOT fire", self.active_backend_name)

        self._send_backend_status()

        while not self.stop_event.is_set():
            try:
                msg_dict = self.result_queue.get(timeout=0.5)
            except Exception:
                self._maybe_heartbeat()
                self._check_memory()
                self._maybe_resync()
                self._maybe_health_check()
                continue

            mtype = msg_dict.get("type")

            if mtype == MessageType.SHUTDOWN.value:
                break

            elif mtype == MessageType.RELAY_BACKEND_CHANGE.value:
                self._handle_backend_change(msg_dict)

            elif mtype in (MessageType.DETECTION_RESULT.value, "pool_result"):
                self._handle_detection_result(msg_dict)

            self._maybe_heartbeat()
            self._check_memory()
            self._maybe_resync()
            self._maybe_health_check()

        # ── Graceful shutdown ─────────────────────────────────────────────
        self._all_off()
        if self.backend:
            self.backend.disconnect()
        logger.info("[Relay] Exiting cleanly")

    # ══════════════════════════════════════════════════════════════════════
    # Detection result handler
    # ══════════════════════════════════════════════════════════════════════
    def _handle_detection_result(self, msg: dict):
        if self._switching:
            return          # silently drop during backend switch

        camera_id   = msg.get("camera_id", 0)
        pair_results = msg.get("pair_results", [])
        relay_indices = self.rcfg.get_relay_indices(camera_id)

        # Warn when fewer pairs arrive than expected
        if len(pair_results) < len(relay_indices):
            logger.warning(
                "[Relay] cam%d: %d pair_results received but %d expected "
                "(indices=%s). Relays %s will not update this cycle. "
                "Check camera_%d_boundaries.json has equal OC/BH counts.",
                camera_id,
                len(pair_results), len(relay_indices), relay_indices,
                relay_indices[len(pair_results):], camera_id,
            )

        # Build new global state (copy current, overlay this camera's relays)
        new_states = list(self._cached_states)
        for i, pr in enumerate(pair_results):
            if i >= len(relay_indices):
                break
            gi = relay_indices[i]
            if 0 <= gi < self.rcfg.relay_count:
                new_states[gi] = bool(pr.get("relay_active", False))

        # Write only changed relays
        for i in range(self.rcfg.relay_count):
            if new_states[i] != self._cached_states[i]:
                self._write_relay(i, new_states[i])

        # Broadcast current state to GUI
        state_msg = RelayStateMessage(
            source=ProcessSource.RELAY,
            camera_id=camera_id,
            relay_states=list(self._cached_states),
        )
        try:
            self.state_out_queue.put_nowait(state_msg.to_dict())
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    # Backend switching
    # ══════════════════════════════════════════════════════════════════════
    def _handle_backend_change(self, msg: dict):
        new_name = msg.get("backend", "usb").lower()

        if new_name == self.active_backend_name:
            logger.info("[Relay] Backend is already '%s' — ignoring switch request",
                        new_name)
            return

        # Cooldown guard — prevent rapid repeated switching
        min_interval = getattr(self.rcfg, "minimum_switch_interval_seconds", 5)
        elapsed      = time.time() - self._last_switch_time
        if elapsed < min_interval:
            remaining = min_interval - elapsed
            logger.warning(
                "[Relay] Backend switch cooldown active — %.1fs remaining",
                remaining,
            )
            return

        if not getattr(self.rcfg, "allow_runtime_switching", True):
            logger.warning("[Relay] Runtime backend switching is disabled in config")
            return

        self._switch_backend(new_name)

    def _switch_backend(self, new_name: str):
        """
        Full safe-switching sequence.  The 8 numbered steps correspond
        to Steps 1–8 in the architecture guide (Doc 1).
        """
        old_name    = self.active_backend_name
        old_backend = self.backend

        logger.info("[RelayBackend] Switching  %s → %s", old_name, new_name)

        # ── Step 1: Freeze writes ─────────────────────────────────────────
        self._switching = True

        try:
            # ── Step 2: Turn all outputs OFF ──────────────────────────────
            if old_backend is not None:
                try:
                    old_backend.all_off()
                    logger.debug("[RelayBackend] Old backend outputs cleared")
                except Exception as e:
                    logger.warning("[RelayBackend] all_off on old backend failed: %s", e)

            # ── Step 3: Disconnect old backend ────────────────────────────
            if old_backend is not None:
                try:
                    old_backend.disconnect()
                except Exception as e:
                    logger.warning("[RelayBackend] disconnect old backend failed: %s", e)

            # ── Steps 4 & 5: Create and connect new backend ───────────────
            new_backend = _create_backend(new_name, self.rcfg)
            if not new_backend.connect():
                logger.error(
                    "[RelayBackend] New backend '%s' connect FAILED → rolling back to '%s'",
                    new_name, old_name,
                )
                self._last_backend_error = (
                    f"Switch to {new_name} failed (connect): "
                    + new_backend.last_error
                )
                self._rollback(old_name, old_backend)
                return

            # ── Step 6: Health check ──────────────────────────────────────
            if not new_backend.health_check():
                logger.error(
                    "[RelayBackend] New backend '%s' health check FAILED → rolling back to '%s'",
                    new_name, old_name,
                )
                self._last_backend_error = (
                    f"Switch to {new_name} failed (health check): "
                    + new_backend.last_error
                )
                try:
                    new_backend.disconnect()
                except Exception:
                    pass
                self._rollback(old_name, old_backend)
                return

            # ── Switch accepted ───────────────────────────────────────────
            self.backend             = new_backend
            self.active_backend_name = new_name
            self._last_switch_time   = time.time()
            self._backend_healthy    = True
            self._last_backend_error = ""
            self._consecutive_failures = 0

            # ── Step 7: Restore relay states ──────────────────────────────
            on_relays = [i for i, s in enumerate(self._cached_states) if s]
            if on_relays:
                logger.info("[RelayBackend] Restoring %d ON relays after switch: %s",
                            len(on_relays), [i + 1 for i in on_relays])
                for idx in on_relays:
                    try:
                        new_backend.relay_on(idx)
                    except Exception as e:
                        logger.warning(
                            "[RelayBackend] State restore relay %d failed: %s", idx + 1, e
                        )

            logger.info("[RelayBackend] Switch to '%s' COMPLETE", new_name)

        finally:
            # ── Step 8: Resume writes ─────────────────────────────────────
            self._switching = False
            self._send_backend_status()

    def _rollback(self, old_name: str, old_backend: Optional[RelayBackend]):
        """
        Reconnect the previous backend after a failed switch.
        If rollback also fails, fall through to simulated backend so
        the relay process stays alive.
        """
        logger.warning("[RelayBackend] Rolling back to '%s'", old_name)

        if old_backend is not None:
            try:
                if old_backend.connect():
                    # Restore ON states onto the old backend
                    for idx, state in enumerate(self._cached_states):
                        if state:
                            try:
                                old_backend.relay_on(idx)
                            except Exception:
                                pass
                    self.backend             = old_backend
                    self.active_backend_name = old_name
                    self._backend_healthy    = True
                    logger.info("[RelayBackend] Rollback to '%s' SUCCESSFUL", old_name)
                    return
            except Exception as e:
                logger.error("[RelayBackend] Rollback connect error: %s", e)

        # Last resort: fall back to simulated so the process stays up
        logger.error(
            "[RelayBackend] Both '%s' and rollback failed — using SIMULATED backend. "
            "Physical relay outputs will NOT fire until manual recovery.",
            old_name,
        )
        sim = _SimulatedBackend(self.rcfg.relay_count)
        sim.connect()
        self.backend             = sim
        self.active_backend_name = "simulated"
        self._backend_healthy    = False
        if not self._last_backend_error:
            self._last_backend_error = "Switch + rollback both failed — using simulated"

    # ══════════════════════════════════════════════════════════════════════
    # Low-level relay write (with retry + reinit on repeated failures)
    # ══════════════════════════════════════════════════════════════════════
    def _write_relay(self, index: int, state: bool):
        if self._switching:
            return

        for attempt in range(self.rcfg.retry_attempts):
            try:
                ok = (
                    self.backend.relay_on(index)
                    if state
                    else self.backend.relay_off(index)
                )
                if ok:
                    self._cached_states[index] = state
                    self._backend_healthy    = True
                    self._last_backend_error = ""
                    logger.info("[Relay] Relay %d → %s", index + 1,
                                "ON" if state else "OFF")
                    self._consecutive_failures = 0
                    return
            except Exception as e:
                logger.warning("[Relay] Write relay %d attempt %d failed: %s",
                               index + 1, attempt + 1, e)
            time.sleep(self.rcfg.retry_delay_seconds)

        self._consecutive_failures += 1
        self._backend_healthy    = False
        self._last_backend_error = (
            f"Relay {index + 1} write failed after "
            f"{self.rcfg.retry_attempts} attempts"
        )
        logger.error("[Relay] %s", self._last_backend_error)

        if self._consecutive_failures >= self.rcfg.reinit_after_failures:
            logger.warning("[Relay] %d consecutive failures — reinitialising backend",
                           self._consecutive_failures)
            self._reinit_backend()

    # ══════════════════════════════════════════════════════════════════════
    # Periodic re-sync of ON relays (v2.2 silent-failure guard)
    # ══════════════════════════════════════════════════════════════════════
    def _maybe_resync(self):
        if self._switching:
            return
        now = time.time()
        if now - self._last_resync < _RESYNC_INTERVAL:
            return
        self._last_resync = now

        on_relays = [i for i, s in enumerate(self._cached_states) if s]
        if not on_relays:
            return

        logger.debug("[Relay] Periodic re-sync: forcing ON for relays %s",
                     [i + 1 for i in on_relays])
        for i in on_relays:
            if self.backend:
                try:
                    ok = self.backend.relay_on(i)
                    if not ok:
                        logger.warning(
                            "[Relay] Re-sync: relay %d failed to set ON — "
                            "hardware may be disconnected", i + 1
                        )
                except Exception as e:
                    logger.warning("[Relay] Re-sync error relay %d: %s", i + 1, e)

    # ══════════════════════════════════════════════════════════════════════
    # Backend reinit (after repeated write failures)
    # ══════════════════════════════════════════════════════════════════════
    def _reinit_backend(self):
        """
        Disconnect and reconnect the current backend after repeated failures.
        Resets _cached_states because physical relay state is unknown after
        a hard disconnect.
        """
        # Safe shutdown of current backend
        try:
            if self.backend:
                for i in range(self.rcfg.relay_count):
                    try:
                        self.backend.relay_off(i)
                    except Exception:
                        pass
                self.backend.disconnect()
        except Exception:
            pass
        time.sleep(1.0)

        # Reconnect
        new_backend = _create_backend(self.active_backend_name, self.rcfg)
        if new_backend.connect():
            self.backend          = new_backend
            self._backend_healthy = True
            self._last_backend_error = ""
            logger.info("[Relay] Reinit backend '%s' successful",
                        self.active_backend_name)
            self._blink_test()
        else:
            self._backend_healthy    = False
            self._last_backend_error = new_backend.last_error
            logger.warning("[Relay] Reinit backend '%s' failed — using simulated",
                           self.active_backend_name)
            sim = _SimulatedBackend(self.rcfg.relay_count)
            sim.connect()
            self.backend = sim

        self._consecutive_failures = 0
        self._cached_states        = [False] * self.rcfg.relay_count
        self._last_resync          = time.time()

    # ══════════════════════════════════════════════════════════════════════
    # Shutdown helpers
    # ══════════════════════════════════════════════════════════════════════
    def _all_off(self):
        if self.backend:
            try:
                self.backend.all_off()
            except Exception as e:
                logger.warning("[Relay] all_off on shutdown failed: %s", e)
                # Fallback: try relay by relay
                for i in range(self.rcfg.relay_count):
                    try:
                        self.backend.relay_off(i)
                    except Exception:
                        pass

    # ══════════════════════════════════════════════════════════════════════
    # Heartbeat & status broadcasts
    # ══════════════════════════════════════════════════════════════════════
    def _maybe_heartbeat(self):
        now = time.time()
        if now - self._last_heartbeat < self.rcfg.heartbeat_interval_seconds:
            return
        self._last_heartbeat = now

        hb = make_heartbeat(
            source=ProcessSource.RELAY,
            camera_id=None,
            process_name=self.name,
            pid=self.pid,
            memory_mb=get_process_memory_mb(),
            fps=0.0,
            status="running",
            extra={
                "relay_states":        self._cached_states,
                "relay_count":         self.rcfg.relay_count,
                "failures":            self._consecutive_failures,
                # v3.0 additions — read by supervisor health broadcast → GUI
                "active_backend":      self.active_backend_name,
                "backend_healthy":     self._backend_healthy,
                "last_backend_error":  self._last_backend_error,
            },
        )
        try:
            self.heartbeat_queue.put_nowait(hb.to_dict())
        except Exception:
            pass

        # Also push a lightweight backend-status message directly to the
        # state_out_queue so the GUI sees it immediately without waiting
        # for the supervisor's 3-second health-snapshot cycle.
        self._send_backend_status()

    def _send_backend_status(self):
        """Push a relay_backend_status message to the GUI via state_out_queue."""
        msg = {
            "type":               MessageType.RELAY_BACKEND_STATUS.value,
            "source":             ProcessSource.RELAY.value,
            "camera_id":          None,
            "timestamp":          time.time(),
            "active_backend":     self.active_backend_name,
            "backend_healthy":    self._backend_healthy,
            "last_backend_error": self._last_backend_error,
        }
        try:
            self.state_out_queue.put_nowait(msg)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    # Resource monitor
    # ══════════════════════════════════════════════════════════════════════
    def _maybe_health_check(self):
        """
        Probe the active backend every 10 seconds.
        Updates _backend_healthy so the heartbeat reports real status.
        Triggers reinit when consecutive failures reach the threshold.
        """
        now = time.time()
        if now - self._last_health_check < 10.0:
            return
        self._last_health_check = now

        if self._switching or self.backend is None:
            return

        ok = False
        try:
            ok = self.backend.health_check()
        except Exception as e:
            self._last_backend_error = str(e)

        if ok:
            if not self._backend_healthy:
                logger.info("[Relay] Backend '%s' is healthy again",
                            self.active_backend_name)
            self._backend_healthy    = True
            self._last_backend_error = ""
            self._consecutive_failures = 0
        else:
            self._backend_healthy    = False
            self._last_backend_error = (
                self.backend.last_error or
                f"Health check failed on '{self.active_backend_name}'"
            )
            self._consecutive_failures += 1
            logger.warning(
                "[Relay] Backend '%s' health check FAILED (consecutive=%d): %s",
                self.active_backend_name,
                self._consecutive_failures,
                self._last_backend_error,
            )
            if self._consecutive_failures >= self.rcfg.reinit_after_failures:
                logger.warning("[Relay] Reinitialising after %d health failures",
                               self._consecutive_failures)
                self._reinit_backend()

    def _blink_test(self):
        """
        Fire every relay ON then OFF in sequence on connect/reconnect.
        Gives visual confirmation that the hardware and wiring are alive.
        Also exercises the full coil range so any dead channel is obvious.
        """
        logger.info("[Relay] Running blink test on '%s' (%d relays)...",
                    self.active_backend_name, self.rcfg.relay_count)
        try:
            for i in range(self.rcfg.relay_count):
                self.backend.relay_on(i)
                time.sleep(0.1)
                self.backend.relay_off(i)
                time.sleep(0.05)
            logger.info("[Relay] Blink test complete — all %d relays cycled",
                        self.rcfg.relay_count)
        except Exception as e:
            logger.warning("[Relay] Blink test error: %s", e)

    def _check_memory(self):
        if is_memory_over_limit(self.rcfg.memory_limit_mb):
            logger.critical("[Relay] Memory limit exceeded — requesting process stop")
            self.stop_event.set()


# ── Process entry point ────────────────────────────────────────────────────────
def relay_process_entry(
    config_path:     str,
    result_queue:    Queue,
    state_out_queue: Queue,
    heartbeat_queue: Queue,
    stop_event,
    log_dir: str = "logs",
):
    from core.config_loader import VisionSystemConfig

    cfg = VisionSystemConfig(config_path)
    setup_process_logging(
        "relay", log_dir, cfg.system.log_level,
        cfg.logging.max_bytes, cfg.logging.backup_count,
    )
    setup_crash_handler("relay", log_dir)

    def _sig(s, f):
        stop_event.set()
    signal.signal(signal.SIGTERM, _sig)

    worker = RelayWorker(cfg, result_queue, state_out_queue, heartbeat_queue, stop_event)
    try:
        worker.run()
    except Exception as e:
        logger.critical("[relay] Fatal: %s", e, exc_info=True)
        sys.exit(1)
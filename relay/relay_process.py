"""
relay/relay_process.py  v4.0
=============================
Ethernet-only relay process (Waveshare Modbus TCP Relay).

v4.0 — Ethernet relay is the sole, final backend:
  • USB relay backend removed completely.
  • Simulated/software relay backend removed completely — there is no
    fallback. If the Ethernet relay hardware is unreachable, the process
    reports it as disconnected and keeps retrying the SAME Ethernet
    target. It never fabricates relay state and never pretends outputs
    fired.
  • Runtime backend switching removed — there is nothing to switch to.

Failsafe behaviour:
  1. On startup, connect to the Ethernet relay. If that fails, the relay
     process stays alive, marks itself DISCONNECTED, and reports this to
     the GUI (which shows a "No Relay Connected" warning).
  2. Every 10s, a health check probes the Ethernet relay.
  3. After `reinit_after_failures` consecutive failed health checks, the
     process tears down and recreates the Modbus connection from scratch
     and tries again — always against the same configured Ethernet
     target, never a different backend.
  4. Detection results keep arriving and are cached (`_cached_states`)
     even while disconnected, so relay outputs are correctly re-applied
     the moment the Ethernet relay comes back online.

Preserved from earlier versions:
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
from relay.backends.modbus_backend import ModbusRelayBackend

logger = logging.getLogger(__name__)

# Seconds between unconditional re-writes of every cached-ON relay.
_RESYNC_INTERVAL = 30.0


# ── Backend factory ────────────────────────────────────────────────────────────
def _create_modbus_backend(rcfg: RelayConfig) -> ModbusRelayBackend:
    """
    Build a fresh ModbusRelayBackend from config. Always targets the same
    configured Ethernet relay — there is no other backend to fall back to.
    """
    mcfg = rcfg.modbus
    return ModbusRelayBackend(
        ip=mcfg.ip,
        port=mcfg.port,
        device_id=mcfg.device_id,
        timeout_seconds=mcfg.timeout_seconds,
        relay_count=rcfg.relay_count,
        retry_attempts=rcfg.retry_attempts,
    )


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

        # ── The one and only backend ────────────────────────────────────
        self.backend: Optional[ModbusRelayBackend] = None

        # ── Relay state cache — INDEPENDENT of connection state ──────────
        # Survives disconnects/reinits so relay knowledge isn't lost the
        # moment the Ethernet cable is unplugged.
        self._cached_states: List[bool] = [False] * self.rcfg.relay_count

        # ── Reinit guard ───────────────────────────────────────────────────
        self._reiniting = False          # freeze writes while reconnecting

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
            "[Relay] PID=%d starting  backend=ethernet(modbus)  relays=%d",
            self.pid, self.rcfg.relay_count,
        )

        # Startup — connect the Ethernet relay and clear all outputs
        self.backend = _create_modbus_backend(self.rcfg)
        if self.backend.connect():
            self.backend.all_off()
            self._backend_healthy    = True
            self._last_backend_error = ""
            logger.info("[Relay] Ethernet relay connected and outputs cleared")
            self._blink_test()
        else:
            self._backend_healthy    = False
            self._last_backend_error = self.backend.last_error
            logger.warning(
                "[Relay] Ethernet relay connect FAILED at startup (%s) — "
                "outputs will NOT fire. No relay connected.",
                self._last_backend_error,
            )

        self._send_relay_status()

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
        if self._reiniting:
            return          # silently drop while reconnecting

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
    # Low-level relay write (with retry + reinit on repeated failures)
    # ══════════════════════════════════════════════════════════════════════
    def _write_relay(self, index: int, state: bool):
        if self._reiniting:
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
            logger.warning("[Relay] %d consecutive failures — reinitialising Ethernet relay",
                           self._consecutive_failures)
            self._reinit_backend()

    # ══════════════════════════════════════════════════════════════════════
    # Periodic re-sync of ON relays (silent-failure guard)
    # ══════════════════════════════════════════════════════════════════════
    def _maybe_resync(self):
        if self._reiniting:
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
    # Backend reinit (after repeated write/health-check failures)
    # ══════════════════════════════════════════════════════════════════════
    def _reinit_backend(self):
        """
        Disconnect and reconnect the Ethernet relay after repeated failures.
        Always targets the same configured Ethernet relay — there is no
        other backend to fall back to. Resets _cached_states because
        physical relay state is unknown after a hard disconnect.
        """
        self._reiniting = True
        try:
            # Safe shutdown of current connection
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

            # Reconnect — fresh Modbus client, same Ethernet target
            new_backend = _create_modbus_backend(self.rcfg)
            if new_backend.connect():
                self.backend              = new_backend
                self._backend_healthy     = True
                self._last_backend_error  = ""
                logger.info("[Relay] Ethernet relay reinit successful")
                self._blink_test()
            else:
                self.backend              = new_backend
                self._backend_healthy     = False
                self._last_backend_error  = new_backend.last_error
                logger.warning(
                    "[Relay] Ethernet relay reinit FAILED (%s) — No relay "
                    "connected. Will keep retrying the same Ethernet target.",
                    self._last_backend_error,
                )

            self._consecutive_failures = 0
            self._cached_states        = [False] * self.rcfg.relay_count
            self._last_resync          = time.time()
        finally:
            self._reiniting = False
            self._send_relay_status()

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
                "backend_healthy":     self._backend_healthy,
                "last_backend_error":  self._last_backend_error,
            },
        )
        try:
            self.heartbeat_queue.put_nowait(hb.to_dict())
        except Exception:
            pass

        # Also push a lightweight relay-status message directly to the
        # state_out_queue so the GUI sees it immediately without waiting
        # for the supervisor's 3-second health-snapshot cycle.
        self._send_relay_status()

    def _send_relay_status(self):
        """Push a relay_backend_status message to the GUI via state_out_queue."""
        msg = {
            "type":               MessageType.RELAY_BACKEND_STATUS.value,
            "source":             ProcessSource.RELAY.value,
            "camera_id":          None,
            "timestamp":          time.time(),
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
        Probe the Ethernet relay every 10 seconds.
        Updates _backend_healthy so the heartbeat reports real status.
        Triggers reinit when consecutive failures reach the threshold —
        always reconnecting to the SAME Ethernet target.
        """
        now = time.time()
        if now - self._last_health_check < 10.0:
            return
        self._last_health_check = now

        if self._reiniting or self.backend is None:
            return

        ok = False
        try:
            ok = self.backend.health_check()
        except Exception as e:
            self._last_backend_error = str(e)

        if ok:
            if not self._backend_healthy:
                logger.info("[Relay] Ethernet relay is healthy again")
            self._backend_healthy    = True
            self._last_backend_error = ""
            self._consecutive_failures = 0
        else:
            self._backend_healthy    = False
            self._last_backend_error = (
                self.backend.last_error or "Ethernet relay health check failed"
            )
            self._consecutive_failures += 1
            logger.warning(
                "[Relay] Ethernet relay health check FAILED (consecutive=%d): %s",
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
        logger.info("[Relay] Running blink test on Ethernet relay (%d relays)...",
                    self.rcfg.relay_count)
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

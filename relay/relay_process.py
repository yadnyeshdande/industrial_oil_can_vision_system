"""
relay/relay_process.py  v2.1
=============================
9-relay process with per-camera mapping.
All v1 behaviour preserved: retry, state caching, reinit on failure.
v2.1: relay_count = 9, relay_mapping per camera from config.
"""
from __future__ import annotations
import logging, multiprocessing as mp, os, signal, sys, time
from pathlib import Path
from multiprocessing import Queue
from typing import List, Optional, Dict

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))

from core.config_loader import VisionSystemConfig, RelayConfig
from core.ipc_schema import (MessageType, ProcessSource, RelayStateMessage,
    make_heartbeat, make_error)
from core.logging_setup import setup_process_logging, setup_crash_handler
from core.resource_monitor import get_process_memory_mb, is_memory_over_limit

logger = logging.getLogger(__name__)


class RelayDriver:
    def __init__(self, relay_count): self.relay_count = relay_count
    def initialize(self) -> bool: raise NotImplementedError
    def set_relay(self, index, state) -> bool: raise NotImplementedError
    def close(self): pass


class SimulatedRelayDriver(RelayDriver):
    def __init__(self, relay_count):
        super().__init__(relay_count); self._states = [False]*relay_count
    def initialize(self):
        logger.info("[SimRelay] Initialized %d relays (SIMULATED)", self.relay_count); return True
    def set_relay(self, index, state):
        if 0 <= index < self.relay_count:
            self._states[index] = state
            logger.debug("[SimRelay] Relay %d → %s", index+1, "ON" if state else "OFF"); return True
        return False
    def close(self): logger.info("[SimRelay] Closed")


class PyhidRelayDriver(RelayDriver):
    def __init__(self, relay_count):
        super().__init__(relay_count); self._device = None
    def initialize(self):
        try:
            from pyhid_usb_relay import find_relay
            self._device = find_relay()
            if self._device is None:
                logger.error("[PyhidRelay] No device found"); return False
            logger.info("[PyhidRelay] Device found: %s", self._device); return True
        except ImportError:
            logger.error("[PyhidRelay] pyhid_usb_relay not installed"); return False
        except Exception as e:
            logger.error("[PyhidRelay] Init error: %s", e); return False
    def set_relay(self, index, state):
        try:
            relay_num = index+1
            if state: self._device.turn_on(relay_num)
            else: self._device.turn_off(relay_num)
            return True
        except Exception as e:
            logger.error("[PyhidRelay] set_relay(%d,%s) error: %s", index, state, e); return False
    def close(self):
        if self._device:
            for i in range(self.relay_count):
                try: self._device.turn_off(i+1)
                except Exception: pass


def build_relay_driver(library, relay_count) -> RelayDriver:
    if library == "pyhid_usb_relay":
        drv = PyhidRelayDriver(relay_count)
        if drv.initialize(): return drv
        logger.warning("Falling back to simulated relay")
    drv = SimulatedRelayDriver(relay_count)
    drv.initialize()
    return drv


class RelayWorker:
    def __init__(self, cfg: VisionSystemConfig, result_queue: Queue,
                 state_out_queue: Queue, heartbeat_queue: Queue, stop_event):
        self.cfg = cfg; self.rcfg: RelayConfig = cfg.relay
        self.result_queue = result_queue; self.state_out_queue = state_out_queue
        self.heartbeat_queue = heartbeat_queue; self.stop_event = stop_event
        self.pid = os.getpid(); self.name = "Relay"
        self._driver: Optional[RelayDriver] = None
        self._cached_states: List[bool] = [False]*self.rcfg.relay_count
        self._consecutive_failures = 0; self._last_heartbeat = time.time()

    def run(self):
        logger.info("[Relay] PID=%d starting (%d relays)", self.pid, self.rcfg.relay_count)
        self._driver = build_relay_driver(self.rcfg.library, self.rcfg.relay_count)

        while not self.stop_event.is_set():
            try:
                msg_dict = self.result_queue.get(timeout=0.5)
            except Exception:
                self._maybe_heartbeat(); self._check_memory(); continue

            if msg_dict.get("type") == MessageType.SHUTDOWN.value: break
            if msg_dict.get("type") in (MessageType.DETECTION_RESULT.value, "pool_result"):
                self._handle_detection_result(msg_dict)
            self._maybe_heartbeat(); self._check_memory()

        self._all_off()
        if self._driver: self._driver.close()
        logger.info("[Relay] Exiting cleanly")

    def _handle_detection_result(self, msg: dict):
        camera_id = msg.get("camera_id", 0)
        pair_results = msg.get("pair_results", [])

        # Get relay indices for this camera
        relay_indices = self.rcfg.get_relay_indices(camera_id)

        # Build new global relay state (copy current, update for this camera's relays)
        new_states = list(self._cached_states)
        for i, pr in enumerate(pair_results):
            if i >= len(relay_indices): break
            global_relay_idx = relay_indices[i]
            if 0 <= global_relay_idx < self.rcfg.relay_count:
                new_states[global_relay_idx] = bool(pr.get("relay_active", False))

        # Write only changed relays
        for i in range(self.rcfg.relay_count):
            if new_states[i] != self._cached_states[i]:
                self._write_relay(i, new_states[i])

        # Broadcast relay state
        state_msg = RelayStateMessage(
            source=ProcessSource.RELAY,
            camera_id=camera_id,
            relay_states=list(self._cached_states),
        )
        try: self.state_out_queue.put_nowait(state_msg.to_dict())
        except Exception: pass

    def _write_relay(self, index, state):
        for attempt in range(self.rcfg.retry_attempts):
            try:
                if self._driver.set_relay(index, state):
                    self._cached_states[index] = state
                    logger.info("[Relay] Relay %d → %s", index+1, "ON" if state else "OFF")
                    self._consecutive_failures = 0; return
            except Exception as e:
                logger.warning("[Relay] Attempt %d failed: %s", attempt+1, e)
            time.sleep(self.rcfg.retry_delay_seconds)
        self._consecutive_failures += 1
        logger.error("[Relay] Failed to set relay %d after %d attempts", index, self.rcfg.retry_attempts)
        if self._consecutive_failures >= self.rcfg.reinit_after_failures:
            logger.warning("[Relay] Reinitializing device after %d failures", self._consecutive_failures)
            self._reinit_device()

    def _reinit_device(self):
        try:
            if self._driver: self._driver.close()
        except Exception: pass
        time.sleep(1.0)
        self._driver = build_relay_driver(self.rcfg.library, self.rcfg.relay_count)
        self._consecutive_failures = 0
        self._cached_states = [False]*self.rcfg.relay_count

    def _all_off(self):
        if self._driver:
            for i in range(self.rcfg.relay_count):
                try: self._driver.set_relay(i, False)
                except Exception: pass

    def _maybe_heartbeat(self):
        now = time.time()
        if now - self._last_heartbeat >= self.rcfg.heartbeat_interval_seconds:
            hb = make_heartbeat(ProcessSource.RELAY, None, self.name, self.pid,
                get_process_memory_mb(), 0.0, extra={
                    "relay_states": self._cached_states,
                    "relay_count": self.rcfg.relay_count,
                    "failures": self._consecutive_failures})
            try: self.heartbeat_queue.put_nowait(hb.to_dict())
            except Exception: pass
            self._last_heartbeat = now

    def _check_memory(self):
        if is_memory_over_limit(self.rcfg.memory_limit_mb):
            logger.critical("[Relay] Memory limit exceeded"); self.stop_event.set()


def relay_process_entry(config_path, result_queue, state_out_queue,
                         heartbeat_queue, stop_event, log_dir="logs"):
    from core.config_loader import VisionSystemConfig
    cfg = VisionSystemConfig(config_path)
    setup_process_logging("relay", log_dir, cfg.system.log_level,
                          cfg.logging.max_bytes, cfg.logging.backup_count)
    setup_crash_handler("relay", log_dir)
    def _sig(s,f): stop_event.set()
    signal.signal(signal.SIGTERM, _sig)
    worker = RelayWorker(cfg, result_queue, state_out_queue, heartbeat_queue, stop_event)
    try: worker.run()
    except Exception as e:
        logger.critical("[relay] Fatal: %s", e, exc_info=True); sys.exit(1)

"""
supervisor/gpu_monitor.py
=========================
GPU health monitor process.
- Polls VRAM, temperature, utilization every N seconds
- Sends GPUStatsMessage to supervisor queue
- Triggers detection restart on VRAM or temperature violations
- Throttles detection FPS on overheat
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

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config_loader import VisionSystemConfig
from core.ipc_schema import (
    GPUStatsMessage, ProcessSource, MessageType,
    make_heartbeat, make_error
)
from core.logging_setup import setup_process_logging, setup_crash_handler
from core.resource_monitor import get_gpu_stats, get_process_memory_mb, shutdown_nvml

logger = logging.getLogger(__name__)


class GPUMonitorWorker:

    def __init__(self,
                 cfg: VisionSystemConfig,
                 supervisor_queue: Queue,
                 heartbeat_queue: Queue,
                 stop_event: mp.Event):
        self.cfg = cfg
        self.gcfg = cfg.gpu_monitor
        self.supervisor_queue = supervisor_queue
        self.heartbeat_queue = heartbeat_queue
        self.stop_event = stop_event
        self.pid = os.getpid()
        self.name = "GPUMonitor"

        self._overheat_start: float = 0.0
        self._overheating: bool = False
        self._last_heartbeat = time.time()

    def run(self):
        logger.info("[GPUMonitor] PID=%d starting", self.pid)

        if not self.gcfg.enabled:
            logger.info("[GPUMonitor] Disabled in config, exiting")
            return

        while not self.stop_event.is_set():
            stats = get_gpu_stats(device_index=0)

            if stats:
                vram_mb = stats.get("vram_used_mb", 0)
                temp_c = stats.get("temperature_c", 0)
                util_pct = stats.get("utilization_pct", 0)
                power_w = stats.get("power_w", 0)

                throttle_fps = None

                # Temperature handling
                if temp_c >= self.gcfg.temperature_threshold_celsius:
                    if not self._overheating:
                        self._overheating = True
                        self._overheat_start = time.time()
                        logger.warning("[GPUMonitor] GPU overheating: %.1f°C", temp_c)
                        throttle_fps = self.gcfg.fps_throttle_on_overheat

                    overheat_duration = time.time() - self._overheat_start
                    if (overheat_duration >= self.gcfg.overheat_duration_seconds
                            and self.gcfg.restart_on_persistent_overheat):
                        logger.critical("[GPUMonitor] Persistent overheat %.1f°C for %.0fs → restart detection",
                                        temp_c, overheat_duration)
                        self._send_restart_command("overheat")
                        self._overheat_start = time.time()  # reset timer
                else:
                    if self._overheating:
                        logger.info("[GPUMonitor] GPU temperature normalised: %.1f°C", temp_c)
                    self._overheating = False

                # VRAM threshold
                if vram_mb > self.gcfg.vram_threshold_mb:
                    logger.critical("[GPUMonitor] VRAM exceeded %.1f/%.1fMB → restart detection",
                                    vram_mb, self.gcfg.vram_threshold_mb)
                    self._send_restart_command("vram_exceeded")

                # Build and send stats message
                gpu_msg = GPUStatsMessage(
                    vram_used_mb=vram_mb,
                    vram_total_mb=stats.get("vram_total_mb", 6144),
                    temperature_c=temp_c,
                    utilization_pct=util_pct,
                    power_w=power_w,
                    throttle_fps=throttle_fps,
                )
                try:
                    self.supervisor_queue.put_nowait(gpu_msg.to_dict())
                except Exception:
                    pass

                logger.debug("[GPUMonitor] VRAM=%.1fMB Temp=%.1f°C Util=%.0f%%",
                             vram_mb, temp_c, util_pct)
            else:
                logger.debug("[GPUMonitor] No GPU stats available")

            # Heartbeat
            now = time.time()
            if now - self._last_heartbeat >= 5.0:
                hb = make_heartbeat(
                    source=ProcessSource.GPU_MONITOR,
                    camera_id=None,
                    process_name=self.name,
                    pid=self.pid,
                    memory_mb=get_process_memory_mb(),
                    fps=0.0,
                    status="running",
                )
                try:
                    self.heartbeat_queue.put_nowait(hb.to_dict())
                except Exception:
                    pass
                self._last_heartbeat = now

            # Poll interval
            for _ in range(int(self.gcfg.poll_interval_seconds * 10)):
                if self.stop_event.is_set():
                    break
                time.sleep(0.1)

        shutdown_nvml()
        logger.info("[GPUMonitor] Exiting cleanly")

    def _send_restart_command(self, reason: str):
        msg = {
            "type": MessageType.RESTART.value,
            "source": ProcessSource.GPU_MONITOR.value,
            "camera_id": None,
            "reason": reason,
            "target": "detection_all",
        }
        try:
            self.supervisor_queue.put_nowait(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------

def gpu_monitor_process_entry(config_path: str,
                               supervisor_queue: Queue,
                               heartbeat_queue: Queue,
                               stop_event: mp.Event,
                               log_dir: str = "logs"):
    from core.config_loader import VisionSystemConfig
    cfg = VisionSystemConfig(config_path)

    setup_process_logging(
        process_name="gpu_monitor",
        log_dir=log_dir,
        log_level=cfg.system.log_level,
        max_bytes=cfg.logging.max_bytes,
        backup_count=cfg.logging.backup_count,
    )
    setup_crash_handler("gpu_monitor", log_dir)

    def _sigterm(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGTERM, _sigterm)

    worker = GPUMonitorWorker(
        cfg=cfg,
        supervisor_queue=supervisor_queue,
        heartbeat_queue=heartbeat_queue,
        stop_event=stop_event,
    )
    try:
        worker.run()
    except Exception as e:
        logger.critical("[gpu_monitor] Fatal: %s", e, exc_info=True)
        sys.exit(1)

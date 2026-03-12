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

        self._overheat_start: float  = 0.0
        self._overheating:    bool   = False
        self._last_heartbeat: float  = time.time()

        # ── VRAM restart storm guard ────────────────────────────────────────
        # Bug: without these guards the monitor fires a restart on EVERY 5-second
        # poll whenever VRAM > threshold, causing an infinite restart loop.
        #
        # Three-layer protection:
        #  1. Sustained violation: VRAM must stay over threshold for
        #     `vram_sustained_seconds` (default 45s) before we act.
        #     This absorbs transient spikes during model warmup (~30s).
        #
        #  2. Cooldown: after triggering a restart we wait
        #     `vram_restart_cooldown_seconds` (default 120s) before
        #     considering another one. This gives the pool time to
        #     reload and VRAM to settle.
        #
        #  3. Max retries: after `vram_max_restarts` (default 5)
        #     consecutive VRAM-triggered restarts we stop and emit
        #     CRITICAL. The operator must intervene (reduce pool_size
        #     or lower vram_threshold_mb in config).
        self._vram_over_since:    float = 0.0   # epoch when first exceeded
        self._vram_is_over:       bool  = False
        self._last_vram_restart:  float = 0.0   # epoch of last restart trigger
        self._vram_restart_count: int   = 0     # consecutive VRAM restarts

        # Read optional fields with safe defaults for older configs
        self._vram_sustained  = getattr(self.gcfg, "vram_sustained_seconds",    45)
        self._vram_cooldown   = getattr(self.gcfg, "vram_restart_cooldown_seconds", 120)
        self._vram_max        = getattr(self.gcfg, "vram_max_restarts",          5)

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

                # ── VRAM threshold — sustained violation + cooldown ─────────
                # OLD (buggy): fired a restart on every single 5-second poll.
                # NEW: requires sustained violation AND enforces cooldown between
                #      consecutive restarts to prevent restart storms.
                if vram_mb > self.gcfg.vram_threshold_mb:
                    now = time.time()
                    if not self._vram_is_over:
                        # First poll to exceed threshold — start the clock
                        self._vram_is_over    = True
                        self._vram_over_since = now
                        logger.warning(
                            "[GPUMonitor] VRAM %.1f/%.1f MB exceeded — "
                            "waiting %.0fs sustained before restart",
                            vram_mb, self.gcfg.vram_threshold_mb, self._vram_sustained)
                    else:
                        sustained = now - self._vram_over_since
                        cooldown_ok = (now - self._last_vram_restart) >= self._vram_cooldown

                        if sustained >= self._vram_sustained and cooldown_ok:
                            if self._vram_restart_count >= self._vram_max:
                                # Storm guard tripped — stop triggering, require manual fix
                                logger.critical(
                                    "[GPUMonitor] VRAM restart limit reached (%d/%d). "
                                    "VRAM=%.1fMB — reduce pool_size or raise vram_threshold_mb. "
                                    "Detection disabled until restart.",
                                    self._vram_restart_count, self._vram_max, vram_mb)
                            else:
                                self._vram_restart_count += 1
                                self._last_vram_restart   = now
                                self._vram_over_since     = now  # reset sustain window
                                logger.critical(
                                    "[GPUMonitor] VRAM %.1f/%.1f MB sustained %.0fs "
                                    "→ restart detection (#%d/%d)",
                                    vram_mb, self.gcfg.vram_threshold_mb,
                                    sustained, self._vram_restart_count, self._vram_max)
                                self._send_restart_command("vram_exceeded")
                        elif not cooldown_ok:
                            remaining = self._vram_cooldown - (now - self._last_vram_restart)
                            logger.warning(
                                "[GPUMonitor] VRAM %.1f MB still high — "
                                "cooldown %.0fs remaining before next restart",
                                vram_mb, remaining)
                        # else: still in sustained window — log progress
                        elif sustained < self._vram_sustained:
                            logger.info(
                                "[GPUMonitor] VRAM %.1f MB over threshold — "
                                "%.0f/%.0fs sustained",
                                vram_mb, sustained, self._vram_sustained)
                else:
                    # VRAM back under threshold
                    if self._vram_is_over:
                        logger.info("[GPUMonitor] VRAM normalised: %.1f/%.1f MB",
                                    vram_mb, self.gcfg.vram_threshold_mb)
                        self._vram_restart_count = 0   # reset consecutive counter
                    self._vram_is_over    = False
                    self._vram_over_since = 0.0

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

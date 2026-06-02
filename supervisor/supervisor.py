"""
supervisor/supervisor.py  v3.0
================================
v3.0 — Bulletproof 24/7 industrial operation

CHANGES FROM v2.2
─────────────────

Task 1 — Deadlock-safe IPC  (core/safe_queue.py)
  All multiprocessing.Queue reads in the supervisor now go through
  SafeQueueReader bridge threads.  The supervisor main loop reads ONLY
  from thread-safe threading.queue.Queue objects that can NEVER deadlock,
  regardless of how worker processes die.

  Old path (deadlocks):
      _drain_hb()  →  self._heartbeat_q.get_nowait()   ← blocks if _rlock stuck

  New path (immune):
      _drain_hb()  →  self._hb_reader.drain()          ← reads local_q, never blocks
      bridge thread →  self._heartbeat_q.get(timeout)  ← may block, but it's a daemon

Task 2 — Hardware watchdog thread  (_WatchdogThread)
  Runs independently on a 2-second tick.  If any managed process has not
  sent a heartbeat for KILL_TIMEOUT_S (10 s), the watchdog:
    1. Sets mproc.stop_event
    2. Calls process.terminate(), then process.kill() if necessary
    3. On Windows uses process.kill(); on Linux escalates to SIGKILL
    4. Enqueues the process key into self._watchdog_restart_q
  The supervision main loop drains _watchdog_restart_q and calls
  _restart_process() — keeping all restart logic single-threaded.

Task 3 — Decoupled daily-restart timer  (_DailyRestartThread)
  Owns the 24-hour restart logic entirely.  Sleeps in 60-second ticks
  (responsive to stop()).  Signals via _daily_restart_q (thread-safe
  queue.Queue) — the main loop drains it every second.  The timer is
  now 100% immune to IPC stalls, queue deadlocks, and hung workers.

Tasks 4 & 5 — handled in gpu_worker_pool.py, detection_process.py,
  camera_process.py, gui_process.py (see those files for OOM handling
  and sys.exit(1) on resource-limit violations).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue as _tqueue   # thread-safe, in-process — alias to avoid name clash
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from multiprocessing import Queue, Process, Event, Value
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config_loader import VisionSystemConfig
from core.ipc_schema import MessageType, ProcessSource, HealthSnapshotMessage
from core.logging_setup import setup_process_logging, setup_crash_handler
from core.safe_queue import SafeQueueReader, safe_put   # Task 1

logger = logging.getLogger(__name__)

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Hardware Watchdog Thread
# ─────────────────────────────────────────────────────────────────────────────

class _WatchdogThread(threading.Thread):
    """
    Heartbeat watchdog — completely decoupled from the IPC drain loop.

    Runs on a fixed 2-second tick.  Checks mproc.last_heartbeat for every
    managed process.  If stale for KILL_TIMEOUT_S:
      1. Escalating kill: stop_event → terminate → kill (SIGKILL / process.kill())
      2. Enqueues the process key into restart_q for the main loop to act on.

    The main supervision loop drains restart_q and calls _restart_process(),
    so all restart bookkeeping remains single-threaded.

    Grace-period logic:
      We only watchdog a process that has sent at least one heartbeat
      (last_heartbeat > supervisor_start + 5s).  This avoids false positives
      during model loading / CUDA init which can take 60-120 s.
    """

    KILL_TIMEOUT_S   = 10.0   # seconds without heartbeat → force-kill
    CHECK_INTERVAL_S =  2.0   # watchdog tick rate

    def __init__(self,
                 processes:  Dict[str, "ManagedProcess"],
                 restart_q:  _tqueue.Queue,
                 start_time: float,
                 grace_s:    float):
        super().__init__(name="Watchdog", daemon=True)
        self._procs      = processes
        self._restart_q  = restart_q
        self._start_time = start_time
        self._grace_s    = grace_s
        self._stop       = threading.Event()
        self._pending: set[str] = set()   # keys already queued for restart

    def stop(self):
        self._stop.set()

    def run(self):
        logger.info("[Watchdog] started  kill_timeout=%.0fs  tick=%.0fs",
                    self.KILL_TIMEOUT_S, self.CHECK_INTERVAL_S)
        while not self._stop.wait(timeout=self.CHECK_INTERVAL_S):
            now    = time.time()
            uptime = now - self._start_time

            for key, mproc in list(self._procs.items()):
                if mproc.storm_disabled:
                    continue

                # Extra grace for processes with heavy initialisation
                extra_grace = 120 if key == "gpu_pool" else 30 if key == "gui" else 0
                if uptime < self._grace_s + extra_grace:
                    continue

                # Only watchdog if at least one heartbeat has been received.
                # Prevents false triggers during slow startup.
                hb_received = mproc.last_heartbeat > (self._start_time + 5)
                if not hb_received:
                    continue

                hb_age = now - mproc.last_heartbeat

                if hb_age > self.KILL_TIMEOUT_S and key not in self._pending:
                    logger.error(
                        "[Watchdog] %s heartbeat stale %.1f s  PID=%s — force-killing",
                        key, hb_age, mproc.pid)
                    self._pending.add(key)
                    self._force_kill(mproc)
                    try:
                        self._restart_q.put_nowait(key)
                    except _tqueue.Full:
                        pass   # restart_q should never be full; drop safely

                elif hb_age <= self.KILL_TIMEOUT_S / 2:
                    # Heartbeats resumed (e.g. after a restart) — clear pending
                    self._pending.discard(key)

        logger.info("[Watchdog] stopped")

    @staticmethod
    def _force_kill(mproc: "ManagedProcess"):
        """
        Escalating termination: stop_event → SIGTERM → SIGKILL/process.kill().

        Tolerates the case where process.pid is None (already dead).
        All exceptions are caught and logged — we must not raise here.
        """
        if not mproc.process:
            return

        pid = mproc.process.pid
        mproc.stop_event.set()

        if not mproc.process.is_alive():
            return   # already dead — nothing to do

        # Stage 1 — graceful SIGTERM
        try:
            mproc.process.terminate()
        except Exception as e:
            logger.debug("[Watchdog] terminate(%s) error: %s", pid, e)

        mproc.process.join(timeout=3.0)
        if not mproc.process.is_alive():
            return

        # Stage 2 — hard kill
        logger.warning("[Watchdog] PID %s did not respond to terminate — sending kill", pid)
        try:
            if sys.platform == "win32":
                mproc.process.kill()               # Windows: TerminateProcess
            elif pid:
                os.kill(pid, signal.SIGKILL)       # Linux: immediate
        except Exception as e:
            logger.error("[Watchdog] kill failed for PID %s: %s", pid, e)

        mproc.process.join(timeout=2.0)
        if mproc.process.is_alive():
            logger.critical(
                "[Watchdog] PID %s could not be killed — manual intervention required", pid)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Daily Restart Thread
# ─────────────────────────────────────────────────────────────────────────────

class _DailyRestartThread(threading.Thread):
    """
    24-hour rolling restart timer, completely decoupled from IPC.

    Design goals:
      • Pure clock-based: never reads from any multiprocessing.Queue.
      • Responsive stop(): sleeps in 60-second ticks, checks stop event.
      • Communicates via a thread-safe queue.Queue (not mp.Queue).
      • Main supervision loop drains _daily_restart_q each second.

    This thread cannot be affected by queue deadlocks, stuck workers,
    or any IPC failure — it only touches time.time() and a local queue.
    """

    def __init__(self, restart_q: _tqueue.Queue, interval_hours: float):
        super().__init__(name="DailyRestart", daemon=True)
        self._restart_q   = restart_q
        self._interval_s  = interval_hours * 3600.0
        self._stop        = threading.Event()
        self._last_restart = time.time()

    def stop(self):
        self._stop.set()

    def run(self):
        logger.info("[DailyRestart] started  interval=%.1f h", self._interval_s / 3600)
        while not self._stop.wait(timeout=60.0):   # wake every 60 s to check
            if time.time() - self._last_restart >= self._interval_s:
                self._last_restart = time.time()
                logger.info("[DailyRestart] interval elapsed — signalling daily restart")
                try:
                    self._restart_q.put_nowait("daily")
                except _tqueue.Full:
                    pass
        logger.info("[DailyRestart] stopped")


# ─────────────────────────────────────────────────────────────────────────────
# ManagedProcess (unchanged from v2.2)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ManagedProcess:
    name:          str
    process_key:   str
    process:       Optional[Process]
    stop_event:    Event
    last_heartbeat:   float = field(default_factory=time.time)
    last_fps:         float = 0.0
    last_fps_nonzero: float = field(default_factory=time.time)
    memory_mb:     float = 0.0
    restart_count: int   = 0
    status:        str   = "starting"
    extra:         dict  = field(default_factory=dict)
    camera_id:     Optional[int] = None
    pid_val:       int   = 0
    _restart_times: list = field(default_factory=list)
    storm_disabled: bool = False

    @property
    def pid(self):
        return self.process.pid if self.process and self.process.is_alive() else None

    def is_alive(self):
        return self.process is not None and self.process.is_alive()

    def record_restart(self, storm_window: float, storm_max: int) -> bool:
        now = time.time()
        self._restart_times = [t for t in self._restart_times if now - t <= storm_window]
        self._restart_times.append(now)
        if len(self._restart_times) > storm_max:
            self.storm_disabled = True
        return not self.storm_disabled


# ─────────────────────────────────────────────────────────────────────────────
# Supervisor
# ─────────────────────────────────────────────────────────────────────────────

class Supervisor:
    def __init__(self, config_path):
        self.config_path = config_path
        self.cfg = VisionSystemConfig(config_path)
        self.log_dir = self.cfg.logging.log_dir

        # ── multiprocessing queues (written by child processes) ──────────────
        self._heartbeat_q   = mp.Queue(maxsize=500)
        self._supervisor_q  = mp.Queue(maxsize=200)
        self._gui_cmd_q     = mp.Queue(maxsize=100)
        self._health_q      = mp.Queue(maxsize=50)
        self._inference_q   = mp.Queue(maxsize=60)
        self._result_q      = mp.Queue(maxsize=400)
        self._relay_result_q = mp.Queue(maxsize=200)
        self._gui_result_q  = mp.Queue(maxsize=200)
        self._relay_state_q = mp.Queue(maxsize=100)

        # ── Task 1: SafeQueueReader bridges — supervisor main loop reads ONLY
        #    from these thread-safe local queues, never from mp.Queue directly.
        #    If _heartbeat_q deadlocks due to a worker dying mid-write, only
        #    the bridge daemon thread gets stuck — not the main loop.
        # Replace with:
        self._hb_reader      = SafeQueueReader(
            self._heartbeat_q,  name="HBReader",     maxsize=500)
        self._sup_reader     = SafeQueueReader(
            self._supervisor_q, name="SupReader",    maxsize=200)
        # Deadlock guard: GUI process writes here; if it dies mid-write on
        # Windows the mutex is abandoned.  Same risk as _heartbeat_q.
        self._gui_cmd_reader = SafeQueueReader(
            self._gui_cmd_q,    name="GuiCmdReader", maxsize=100)

        # ── Task 2 & 3: thread-safe signal queues (queue.Queue, not mp.Queue)
        #    used by watchdog thread and daily-restart thread to tell the main
        #    loop what to do.  These are in-process and can never deadlock.
        self._watchdog_restart_q: _tqueue.Queue = _tqueue.Queue(maxsize=50)
        self._daily_restart_q:    _tqueue.Queue = _tqueue.Queue(maxsize=5)

        _preview_init = 1 if self.cfg.gui.start_in_preview_mode else 0
        self._preview_mode = Value('i', _preview_init)
         # Add immediately after:
        # Shared Value read by camera and gpu_pool processes every frame loop tick.
        # 0.0  = use each process's own config fps_limit (normal operation).
        # >0.0 = hard cap to this FPS (thermal throttle active).
        self._throttle_fps = mp.Value('d', 0.0)

        self._processes: Dict[str, ManagedProcess] = {}
        self._running           = False
        self._start_time        = time.time()
        self._last_health_log   = time.time()
        self._last_daily_restart = time.time()
        self._last_health_broadcast = time.time()
        self._last_gpu_stats    = {}
        self._shutdown_requested = False

        # Watchdog and daily-restart threads (started in start())
        self._watchdog_thread:       Optional[_WatchdogThread]     = None
        self._daily_restart_thread:  Optional[_DailyRestartThread] = None

        signal.signal(signal.SIGTERM, self._on_sigterm)
        signal.signal(signal.SIGINT,  self._on_sigterm)

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        logger.info("=" * 60)
        logger.info("Supervisor v%s — %s", self.cfg.system.version, self.cfg.system.name)
        logger.info("=" * 60)

        if self.cfg.supervisor.validate_model_on_startup:
            if not self.cfg.validate_model():
                logger.critical("MODEL NOT FOUND: %s", self.cfg.model.path)
                self._start_all(detection_ok=False)
            else:
                logger.info("Model OK: %s", self.cfg.model.path)
                self._start_all(detection_ok=True)
        else:
            self._start_all(detection_ok=True)

        # Task 2 — start hardware watchdog thread
        self._watchdog_thread = _WatchdogThread(
            processes  = self._processes,
            restart_q  = self._watchdog_restart_q,
            start_time = self._start_time,
            grace_s    = self.cfg.system.startup_grace_period_seconds,
        )
        self._watchdog_thread.start()
        logger.info("[Supervisor] Watchdog thread started")

        # Task 3 — start daily restart thread
        self._daily_restart_thread = _DailyRestartThread(
            restart_q     = self._daily_restart_q,
            interval_hours = self.cfg.system.daily_restart_interval_hours,
        )
        self._daily_restart_thread.start()
        logger.info("[Supervisor] DailyRestart thread started")

        self._running = True
        self._supervision_loop()

    # ─── Fan-out ──────────────────────────────────────────────────────────────

    def _start_fanout(self):
        """Copy each message from _result_q to both relay and GUI queues."""
        def _loop():
            while True:
                try:
                    msg = self._result_q.get(timeout=0.5)
                except Exception:
                    continue
                safe_put(self._relay_result_q, msg)
                safe_put(self._gui_result_q,   msg)

        t = threading.Thread(target=_loop, daemon=True, name="ResultFanout")
        t.start()
        logger.info("[Supervisor] Result fanout thread started")

    # ─── Process launchers ────────────────────────────────────────────────────

    def _start_all(self, detection_ok):
        self._start_fanout()
        for cam_id in self.cfg.camera_ids:
            self._start_camera(cam_id)
        if detection_ok:
            if self.cfg.gpu_pool.enabled:
                self._start_gpu_pool()
            else:
                for cam_id in self.cfg.camera_ids:
                    time.sleep(1.5)
                    self._start_detection(cam_id)
        self._start_relay()
        if self.cfg.gui.enabled:
            self._start_gui()
        if self.cfg.gpu_monitor.enabled:
            self._start_gpu_monitor()
        logger.info("All processes started.")

    def _start_camera(self, camera_id):
        from camera.camera_process import camera_process_entry
        key = f"camera_{camera_id}"
        stop_ev = Event()
        p = Process(
            target=camera_process_entry,
            args=(camera_id, self.config_path, self._inference_q,
                  self._heartbeat_q, stop_ev, self._preview_mode,
                  self._throttle_fps, self.log_dir),
            name=key, daemon=True)
        p.start()
        self._processes[key] = ManagedProcess(
            f"Camera {camera_id}", key, p, stop_ev, camera_id=camera_id)
        logger.info("Started %s PID=%d", key, p.pid)

    def _start_gpu_pool(self):
        from detection.gpu_worker_pool import pool_manager_process_entry
        key = "gpu_pool"
        stop_ev = Event()
        p = Process(
            target=pool_manager_process_entry,
            args=(self.config_path, self._inference_q, self._result_q,
                  self._heartbeat_q, stop_ev, self._throttle_fps, self.log_dir),
            name=key, daemon=False)   # daemon=False: spawns GPU worker children
        p.start()
        self._processes[key] = ManagedProcess("GPU Pool", key, p, stop_ev)
        logger.info("Started gpu_pool PID=%d (daemon=False)", p.pid)

    def _start_detection(self, camera_id):
        from detection.detection_process import detection_process_entry
        key = f"detection_{camera_id}"
        stop_ev = Event()
        # Replace with:
        p = Process(
            target=detection_process_entry,
            args=(camera_id, self.config_path, self._result_q,
                  self._heartbeat_q, stop_ev, self._throttle_fps, self.log_dir),
            name=key, daemon=True)
        p.start()
        self._processes[key] = ManagedProcess(
            f"Detection {camera_id}", key, p, stop_ev, camera_id=camera_id)
        logger.info("Started %s PID=%d", key, p.pid)

    def _start_relay(self):
        from relay.relay_process import relay_process_entry
        key = "relay"
        stop_ev = Event()
        p = Process(
            target=relay_process_entry,
            args=(self.config_path, self._relay_result_q, self._relay_state_q,
                  self._heartbeat_q, stop_ev, self.log_dir),
            name=key, daemon=True)
        p.start()
        self._processes[key] = ManagedProcess("Relay", key, p, stop_ev)
        logger.info("Started relay PID=%d", p.pid)

    def _start_gui(self):
        from gui.unified_gui import gui_process_entry
        key = "gui"
        stop_ev = Event()
        p = Process(
            target=gui_process_entry,
            args=(self.config_path, self._gui_result_q, self._relay_state_q,
                  self._heartbeat_q, self._gui_cmd_q, self._health_q,
                  stop_ev, self._preview_mode, self.log_dir),
            name=key, daemon=True)
        p.start()
        self._processes[key] = ManagedProcess("GUI", key, p, stop_ev)
        logger.info("Started gui PID=%d", p.pid)

    def _start_gpu_monitor(self):
        from supervisor.gpu_monitor import gpu_monitor_process_entry
        key = "gpu_monitor"
        stop_ev = Event()
        p = Process(
            target=gpu_monitor_process_entry,
            args=(self.config_path, self._supervisor_q, self._heartbeat_q,
                  stop_ev, self.log_dir),
            name=key, daemon=True)
        p.start()
        self._processes[key] = ManagedProcess("GPUMonitor", key, p, stop_ev)
        logger.info("Started gpu_monitor PID=%d", p.pid)

    # ─── Main supervision loop ────────────────────────────────────────────────

    def _supervision_loop(self):
        """
        Main loop — only ever reads from thread-safe queue.Queue objects.
        Can never deadlock, regardless of worker-process failures.

        Tasks performed each second:
          • Drain heartbeat messages      (_drain_hb via SafeQueueReader bridge)
          • Drain supervisor messages     (_drain_sup via SafeQueueReader bridge)
          • Drain GUI commands            (_drain_gui_cmd — put_nowait in workers)
          • Apply watchdog restart requests (_drain_watchdog_restarts)
          • Apply daily restart signal    (_drain_daily_restarts)
          • Liveness + memory check       (_check_health)
          • Log and broadcast health      (_maybe_log_health, _maybe_broadcast_health)
        """
        logger.info("[Supervisor] Supervision loop running")
        while self._running and not self._shutdown_requested:
            self._drain_hb()
            self._drain_sup()
            self._drain_gui_cmd()
            self._drain_watchdog_restarts()   # Task 2
            self._drain_daily_restarts()       # Task 3
            self._check_health()
            self._maybe_log_health()
            self._maybe_broadcast_health()
            time.sleep(1.0)

        logger.info("[Supervisor] Supervision loop ending")
        self._shutdown_all()

    # ─── Queue drains ────────────────────────────────────────────────────────

    def _drain_hb(self):
        """
        Task 1: drain via SafeQueueReader — reads local_q (thread-safe).
        Also checks bridge thread health and respawns it if it crashed.
        """
        for msg in self._hb_reader.drain(max_items=150):
            self._handle_msg(msg)

        # Bridge thread health guard: if the thread died (should be rare),
        # respawn it so we don't silently lose heartbeats.
        if not self._hb_reader.is_thread_alive():
            self._hb_reader.respawn_thread()

    def _drain_sup(self):
        """Task 1: drain supervisor queue via SafeQueueReader."""
        for msg in self._sup_reader.drain(max_items=50):
            self._handle_sup(msg)

        if not self._sup_reader.is_thread_alive():
            self._sup_reader.respawn_thread()

    def _drain_watchdog_restarts(self):
        """
        Task 2: apply restart requests from the watchdog thread.
        The watchdog does the killing; we do the respawn bookkeeping.
        This runs in the main thread so _restart_process() stays single-threaded.
        """
        while True:
            try:
                key = self._watchdog_restart_q.get_nowait()
            except _tqueue.Empty:
                break
            logger.warning("[Supervisor] Watchdog requested restart of %s", key)
            self._restart_process(key)

    def _drain_daily_restarts(self):
        """
        Task 3: apply daily-restart signal from _DailyRestartThread.
        Runs in the main thread, but the TIMER lives in the dedicated thread.
        The main loop can be stuck for minutes and the timer will still fire.
        """
        while True:
            try:
                self._daily_restart_q.get_nowait()
            except _tqueue.Empty:
                break
            logger.info("[Supervisor] Daily restart triggered by timer thread")
            self._last_daily_restart = time.time()
            self._restart_all_detection()
            for cam_id in self.cfg.camera_ids:
                self._restart_process(f"camera_{cam_id}")
                time.sleep(2)
            self._restart_process("relay")
            time.sleep(2)
            if "gui" in self._processes:
                self._restart_process("gui")

    # Replace with:
    def _drain_gui_cmd(self):
        """
        Route GUI commands via SafeQueueReader bridge (Task 1 loophole fix).

        _gui_cmd_q is written by the GUI process.  If the GUI dies mid-write
        on Windows, the internal mutex is abandoned and any direct get_nowait()
        call in the supervisor would block forever — identical to the original
        heartbeat queue deadlock.  Using the SafeQueueReader bridge isolates
        this risk to a daemon thread.
        """
        for msg in self._gui_cmd_reader.drain(max_items=30):
            mtype = msg.get("type")

            if mtype == MessageType.BOUNDARY_RELOAD.value:
                safe_put(self._inference_q, msg)

            elif mtype == MessageType.GUI_COMMAND.value:
                cmd = msg.get("command", "")
                if cmd == "start_detection":
                    self._preview_mode.value = 0
                    logger.info("[Supervisor] Detection STARTED by operator")
                elif cmd == "stop_detection":
                    self._preview_mode.value = 1
                    logger.info("[Supervisor] Detection PAUSED by operator")

            elif mtype == MessageType.RELAY_BACKEND_CHANGE.value:
                backend = msg.get("backend", "usb")
                logger.info("[Supervisor] Routing backend change → relay: %s", backend)
                safe_put(self._relay_result_q, msg)

        if not self._gui_cmd_reader.is_thread_alive():
            self._gui_cmd_reader.respawn_thread()

    # ─── Message handlers ────────────────────────────────────────────────────

    def _handle_msg(self, msg):
        mtype  = msg.get("type")
        src    = msg.get("source", "")
        cam_id = msg.get("camera_id")

        if mtype == MessageType.HEARTBEAT.value:
            key = self._find_key(src, cam_id)
            if key and key in self._processes:
                p = self._processes[key]
                p.last_heartbeat = time.time()
                p.memory_mb      = msg.get("memory_mb", 0)
                p.last_fps       = msg.get("fps", 0)
                p.status         = msg.get("status", "running")
                p.extra          = msg.get("extra", {})
                p.pid_val        = msg.get("pid", 0)
                if p.last_fps > 0:
                    p.last_fps_nonzero = time.time()

        elif mtype == MessageType.ERROR.value:
            fn = logger.critical if msg.get("severity") == "critical" else logger.error
            fn("[%s] cam=%s %s: %s", src, cam_id,
               msg.get("error_type", "?"), msg.get("error_msg", "?"))

        elif mtype == MessageType.GPU_STATS.value:
            self._last_gpu_stats = msg

    def _handle_sup(self, msg):
        mtype = msg.get("type")
        if mtype == MessageType.RESTART.value:
            target = msg.get("target", "")
            if target == "detection_all":
                self._restart_all_detection()

        elif mtype == MessageType.GPU_STATS.value:
            self._last_gpu_stats = msg
            # Task 3 FIX: propagate throttle_fps to all workers via shared Value.
            # Previously this branch only stored the message for the GUI.
            throttle = msg.get("throttle_fps")
            if throttle is not None and throttle > 0:
                if self._throttle_fps.value != float(throttle):
                    logger.warning(
                        "[Supervisor] Thermal throttle ACTIVE: %.0f FPS  "
                        "(GPU=%.1f°C)",
                        throttle, msg.get("temperature_c", 0))
                    self._throttle_fps.value = float(throttle)

        elif mtype == MessageType.FPS_LIMIT_UPDATE.value:
            # Sent by gpu_monitor when temperature normalises (reason="restore").
            fps    = msg.get("fps_limit", 0.0)
            reason = msg.get("reason", "")
            if reason == "restore" and self._throttle_fps.value != 0.0:
                logger.info(
                    "[Supervisor] Thermal throttle LIFTED — "
                    "restoring config FPS in all workers")
                self._throttle_fps.value = 0.0

    # ─── Health checks ────────────────────────────────────────────────────────

    def _check_health(self):
        """
        Checks process liveness and memory usage.
        Heartbeat-timeout check has been MOVED to _WatchdogThread (Task 2).
        This method only handles: dead process → restart, and memory over limit.
        """
        now    = time.time()
        uptime = now - self._start_time

        for key, mproc in list(self._processes.items()):
            if mproc.storm_disabled:
                if int(now) % 300 == 0:
                    logger.critical(
                        "[Supervisor] %s storm guard ACTIVE — manual restart required", key)
                continue

            max_r = self._restart_limit(key)
            if mproc.restart_count >= max_r:
                if int(now) % 300 == 0:
                    logger.critical(
                        "[Supervisor] %s restart limit %d/%d — manual intervention required",
                        key, mproc.restart_count, max_r)
                continue

            extra_grace    = 120 if key == "gpu_pool" else 30 if key == "gui" else 0
            effective_grace = self.cfg.system.startup_grace_period_seconds + extra_grace

            # Reap zombie handles
            if mproc.process and not mproc.process.is_alive():
                try:
                    mproc.process.join(timeout=0)
                except Exception:
                    pass

            # Liveness check — heartbeat-timeout is handled by watchdog thread
            if not mproc.is_alive() and uptime > effective_grace:
                logger.error("[Supervisor] %s not alive → restart", key)
                self._restart_process(key)
                continue

            self._check_proc_mem(key, mproc)

    def _check_proc_mem(self, key, mproc):
        limits = {f"camera_{c.id}": c.memory_limit_mb for c in self.cfg.cameras}
        limits.update({
            "gpu_pool": self.cfg.gpu_pool.memory_limit_mb,
            "relay":    self.cfg.relay.memory_limit_mb,
            "gui":      self.cfg.gui.memory_limit_mb,
        })
        for c in self.cfg.cameras:
            limits[f"detection_{c.id}"] = self.cfg.detection.memory_limit_mb
        limit = limits.get(key)
        if limit and mproc.memory_mb > limit:
            logger.warning(
                "[Supervisor] %s mem %.1f MB > limit %.1f MB → restart",
                key, mproc.memory_mb, limit)
            self._restart_process(key)

    # ─── Restart helpers ──────────────────────────────────────────────────────

    def _restart_limit(self, key: str) -> int:
        if key == "gui":         return 20
        if key == "gpu_monitor": return 10
        return 15

    def _restart_process(self, key):
        mproc = self._processes.get(key)
        if not mproc:
            return

        mproc.restart_count += 1
        logger.info("[Supervisor] Restarting %s (#%d)", key, mproc.restart_count)

        allowed = mproc.record_restart(
            self.cfg.supervisor.storm_window_seconds,
            self.cfg.supervisor.storm_max_restarts,
        )
        if not allowed:
            logger.critical(
                "[Supervisor] %s STORM GUARD TRIPPED — %d restarts in %ds. Stopped.",
                key, self.cfg.supervisor.storm_max_restarts,
                self.cfg.supervisor.storm_window_seconds)
            return

        mproc.stop_event.set()
        if mproc.process:
            if mproc.process.is_alive():
                mproc.process.terminate()
                mproc.process.join(timeout=8.0)
                if mproc.process.is_alive():
                    logger.warning("[Supervisor] %s did not terminate — killing", key)
                    try:
                        if sys.platform == "win32":
                            mproc.process.kill()
                        elif mproc.process.pid:
                            os.kill(mproc.process.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    mproc.process.join(timeout=3.0)
            else:
                mproc.process.join(timeout=0)

        mproc.stop_event.clear()

        max_delay = 10 if key == "gui" else 60
        delay = min(self.cfg.supervisor.restart_backoff_seconds * mproc.restart_count, max_delay)
        if delay > 0:
            logger.info("[Supervisor] %s backoff %.1fs", key, delay)
            time.sleep(delay)

        # If this process writes to _heartbeat_q, and the queue might be
        # corrupted (worker died mid-write), replace the reader's mp.Queue
        # so the bridge thread is not permanently stuck.
        # Note: we cannot cheaply give the new queue to OTHER alive processes,
        # so we only do this if all writers for this queue have been restarted.
        # For now, just respawn the bridge thread — it will reconnect to the
        # same (possibly recovered) mp.Queue.
        if key in ("gpu_pool", "gui", "relay") or key.startswith(("camera_", "detection_")):
            if not self._hb_reader.is_thread_alive():
                logger.warning(
                    "[Supervisor] heartbeat reader thread dead after %s crash — respawning", key)
                self._hb_reader.respawn_thread()

        self._respawn(key)

    def _respawn(self, key):
        """
        Spawn a replacement process.
        Preserves restart bookkeeping fields from the old ManagedProcess.
        """
        old = self._processes.get(key)
        saved_restarts       = old.restart_count      if old else 0
        saved_hb             = old.last_heartbeat     if old else time.time()
        saved_fps_nonzero    = old.last_fps_nonzero   if old else time.time()
        saved_restart_times  = list(old._restart_times) if old else []
        saved_storm_disabled = old.storm_disabled     if old else False

        if   key.startswith("camera_"):      self._start_camera(int(key.split("_")[1]))
        elif key == "gpu_pool":              self._start_gpu_pool()
        elif key.startswith("detection_"):   self._start_detection(int(key.split("_")[1]))
        elif key == "relay":                 self._start_relay()
        elif key == "gui":                   self._start_gui()
        elif key == "gpu_monitor":           self._start_gpu_monitor()
        else:
            logger.error("[Supervisor] Unknown process key: %s", key)
            return

        new = self._processes.get(key)
        if new:
            new.restart_count    = saved_restarts
            new.last_heartbeat   = saved_hb
            new.last_fps_nonzero = saved_fps_nonzero
            new._restart_times   = saved_restart_times
            new.storm_disabled   = saved_storm_disabled
            logger.debug("[Supervisor] %s respawned restart_count=%d storm=%s",
                         key, new.restart_count, new.storm_disabled)

    def _restart_all_detection(self):
        if "gpu_pool" in self._processes:
            self._restart_process("gpu_pool")
        else:
            for cam_id in self.cfg.camera_ids:
                self._restart_process(f"detection_{cam_id}")
                time.sleep(self.cfg.supervisor.sequential_detection_restart_delay)

    # ─── Health logging / broadcast ───────────────────────────────────────────

    def _maybe_log_health(self):
        now = time.time()
        if now - self._last_health_log < self.cfg.supervisor.health_log_interval_seconds:
            return
        self._last_health_log = now
        uptime = now - self._start_time
        h, m, s = int(uptime // 3600), int((uptime % 3600) // 60), int(uptime % 60)
        logger.info("[Health] Uptime %02dh%02dm%02ds", h, m, s)
        for key, mproc in self._processes.items():
            logger.info(
                "  %-22s alive=%-5s restarts=%-3d mem=%.1fMB fps=%.1f",
                key, mproc.is_alive(), mproc.restart_count, mproc.memory_mb, mproc.last_fps)

    def _maybe_broadcast_health(self):
        now = time.time()
        if now - self._last_health_broadcast < self.cfg.supervisor.health_broadcast_interval_seconds:
            return
        self._last_health_broadcast = now
        cpu_pct = ram_mb = 0.0
        if _PSUTIL:
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
                ram_mb  = psutil.virtual_memory().used / (1024 * 1024)
            except Exception:
                pass
        procs_list    = []
        total_restarts = 0
        for key, mproc in self._processes.items():
            total_restarts += mproc.restart_count
            procs_list.append({
                "key": key, "name": mproc.name, "pid": mproc.pid_val,
                "status": mproc.status, "restart_count": mproc.restart_count,
                "memory_mb": mproc.memory_mb, "fps": mproc.last_fps,
                "alive": mproc.is_alive(),
            })
        snap = HealthSnapshotMessage(
            processes=procs_list, gpu_stats=self._last_gpu_stats,
            cpu_pct=cpu_pct, ram_used_mb=ram_mb,
            system_uptime_s=now - self._start_time,
            total_restarts=total_restarts,
        )
        safe_put(self._health_q, snap.to_dict())

    # ─── Utilities ────────────────────────────────────────────────────────────

    def _find_key(self, source, cam_id):
        if source == ProcessSource.CAMERA.value     and cam_id is not None: return f"camera_{cam_id}"
        if source == ProcessSource.DETECTION.value:  return f"detection_{cam_id}" if cam_id is not None else None
        if source == ProcessSource.GPU_POOL.value:   return "gpu_pool"
        if source == ProcessSource.RELAY.value:      return "relay"
        if source == ProcessSource.GUI.value:        return "gui"
        if source == ProcessSource.GPU_MONITOR.value: return "gpu_monitor"
        return None

    def _on_sigterm(self, sig, frame):
        logger.info("[Supervisor] Signal %d → shutdown", sig)
        self._shutdown_requested = True
        self._running = False

    def _shutdown_all(self):
        logger.info("[Supervisor] Shutting down...")

        # Stop background threads
        if self._watchdog_thread:
            self._watchdog_thread.stop()
        if self._daily_restart_thread:
            self._daily_restart_thread.stop()

        for key, mproc in self._processes.items():
            mproc.stop_event.set()
            if mproc.process and mproc.process.is_alive():
                mproc.process.terminate()

        for key, mproc in self._processes.items():
            if mproc.process:
                mproc.process.join(timeout=10.0)
                if mproc.process.is_alive():
                    logger.warning("[Supervisor] Force-killing %s", key)
                    try:
                        mproc.process.kill()
                    except Exception:
                        pass

        logger.info("[Supervisor] Shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_supervisor(config_path):
    cfg = VisionSystemConfig(config_path)
    setup_process_logging(
        "supervisor", cfg.logging.log_dir, cfg.system.log_level,
        cfg.logging.max_bytes, cfg.logging.backup_count)
    setup_crash_handler("supervisor", cfg.logging.log_dir)
    Supervisor(config_path).start()

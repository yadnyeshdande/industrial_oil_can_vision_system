"""
supervisor/supervisor.py  v2.2
================================
v2.2 fixes:
  - gpu_pool process is daemon=False (Python daemon processes cannot spawn children)
  - All other processes remain daemon=True for clean exit
  - nvidia-ml-py replaces pynvml import alias
"""
from __future__ import annotations
import logging, multiprocessing as mp, os, signal, sys, time
from dataclasses import dataclass, field
from multiprocessing import Queue, Process, Event, Value
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))

from core.config_loader import VisionSystemConfig
from core.ipc_schema import MessageType, ProcessSource, HealthSnapshotMessage
from core.logging_setup import setup_process_logging, setup_crash_handler

logger = logging.getLogger(__name__)
try:
    import psutil; _PSUTIL = True
except ImportError:
    _PSUTIL = False


@dataclass
class ManagedProcess:
    name: str; process_key: str; process: Optional[Process]; stop_event: Event
    last_heartbeat: float = field(default_factory=time.time)
    last_fps: float = 0.0; last_fps_nonzero: float = field(default_factory=time.time)
    memory_mb: float = 0.0; restart_count: int = 0; status: str = "starting"
    extra: dict = field(default_factory=dict); camera_id: Optional[int] = None; pid_val: int = 0
    @property
    def pid(self): return self.process.pid if self.process and self.process.is_alive() else None
    def is_alive(self): return self.process is not None and self.process.is_alive()


class Supervisor:
    def __init__(self, config_path):
        self.config_path = config_path; self.cfg = VisionSystemConfig(config_path)
        self.log_dir = self.cfg.logging.log_dir
        self._heartbeat_q = mp.Queue(maxsize=500); self._supervisor_q = mp.Queue(maxsize=200)
        self._gui_cmd_q = mp.Queue(maxsize=100); self._health_q = mp.Queue(maxsize=50)
        self._inference_q = mp.Queue(maxsize=60); self._result_q = mp.Queue(maxsize=200)
        self._relay_state_q = mp.Queue(maxsize=100); self._preview_mode = Value('i', 1)  # start in preview
        self._processes: Dict[str, ManagedProcess] = {}
        self._running = False; self._start_time = time.time()
        self._last_health_log = time.time(); self._last_daily_restart = time.time()
        self._last_health_broadcast = time.time(); self._last_gpu_stats = {}
        self._shutdown_requested = False
        signal.signal(signal.SIGTERM, self._on_sigterm); signal.signal(signal.SIGINT, self._on_sigterm)

    def start(self):
        logger.info("="*60); logger.info("Supervisor v%s — %s", self.cfg.system.version, self.cfg.system.name); logger.info("="*60)
        if self.cfg.supervisor.validate_model_on_startup:
            if not self.cfg.validate_model():
                logger.critical("MODEL NOT FOUND: %s — detection will NOT start", self.cfg.model.path)
                self._start_all(detection_ok=False)
            else:
                logger.info("Model OK: %s", self.cfg.model.path); self._start_all(detection_ok=True)
        else:
            self._start_all(detection_ok=True)
        self._running = True; self._supervision_loop()

    def _start_all(self, detection_ok):
        for cam_id in self.cfg.camera_ids: self._start_camera(cam_id)
        if detection_ok:
            if self.cfg.gpu_pool.enabled: self._start_gpu_pool()
            else:
                for cam_id in self.cfg.camera_ids: time.sleep(1.5); self._start_detection(cam_id)
        self._start_relay()
        if self.cfg.gui.enabled: self._start_gui()
        if self.cfg.gpu_monitor.enabled: self._start_gpu_monitor()
        logger.info("All processes started.")

    def _start_camera(self, camera_id):
        from camera.camera_process import camera_process_entry
        key = f"camera_{camera_id}"; stop_ev = Event()
        p = Process(target=camera_process_entry,
            args=(camera_id, self.config_path, self._inference_q,
                  self._heartbeat_q, stop_ev, self._preview_mode, self.log_dir),
            name=key, daemon=True)  # daemon=True OK — no children
        p.start()
        self._processes[key] = ManagedProcess(f"Camera {camera_id}", key, p, stop_ev, camera_id=camera_id)
        logger.info("Started %s PID=%d", key, p.pid)

    def _start_gpu_pool(self):
        from detection.gpu_worker_pool import pool_manager_process_entry
        key = "gpu_pool"; stop_ev = Event()
        # CRITICAL: daemon=False because PoolManager spawns GPUWorker subprocesses.
        # Python does not allow daemon processes to have children.
        # Supervisor manually terminates this process on shutdown.
        p = Process(target=pool_manager_process_entry,
            args=(self.config_path, self._inference_q, self._result_q,
                  self._heartbeat_q, stop_ev, self.log_dir),
            name=key, daemon=False)
        p.start()
        self._processes[key] = ManagedProcess("GPU Pool", key, p, stop_ev)
        logger.info("Started gpu_pool PID=%d (daemon=False, spawns GPU worker children)", p.pid)

    def _start_detection(self, camera_id):
        from detection.detection_process import detection_process_entry
        key = f"detection_{camera_id}"; stop_ev = Event()
        p = Process(target=detection_process_entry,
            args=(camera_id, self.config_path, self._result_q,
                  self._heartbeat_q, stop_ev, self.log_dir),
            name=key, daemon=True)
        p.start()
        self._processes[key] = ManagedProcess(f"Detection {camera_id}", key, p, stop_ev, camera_id=camera_id)
        logger.info("Started %s PID=%d", key, p.pid)

    def _start_relay(self):
        from relay.relay_process import relay_process_entry
        key = "relay"; stop_ev = Event()
        p = Process(target=relay_process_entry,
            args=(self.config_path, self._result_q, self._relay_state_q,
                  self._heartbeat_q, stop_ev, self.log_dir),
            name=key, daemon=True)
        p.start()
        self._processes[key] = ManagedProcess("Relay", key, p, stop_ev)
        logger.info("Started relay PID=%d", p.pid)

    def _start_gui(self):
        from gui.unified_gui import gui_process_entry
        key = "gui"; stop_ev = Event()
        p = Process(target=gui_process_entry,
            args=(self.config_path, self._result_q, self._relay_state_q,
                  self._heartbeat_q, self._gui_cmd_q, self._health_q,
                  stop_ev, self._preview_mode, self.log_dir),
            name=key, daemon=True)
        p.start()
        self._processes[key] = ManagedProcess("GUI", key, p, stop_ev)
        logger.info("Started gui PID=%d", p.pid)

    def _start_gpu_monitor(self):
        from supervisor.gpu_monitor import gpu_monitor_process_entry
        key = "gpu_monitor"; stop_ev = Event()
        p = Process(target=gpu_monitor_process_entry,
            args=(self.config_path, self._supervisor_q, self._heartbeat_q,
                  stop_ev, self.log_dir),
            name=key, daemon=True)
        p.start()
        self._processes[key] = ManagedProcess("GPUMonitor", key, p, stop_ev)
        logger.info("Started gpu_monitor PID=%d", p.pid)

    def _supervision_loop(self):
        logger.info("Supervision loop running")
        while self._running and not self._shutdown_requested:
            self._drain_hb(); self._drain_sup(); self._drain_gui_cmd()
            self._check_health(); self._check_daily_restart()
            self._maybe_log_health(); self._maybe_broadcast_health()
            time.sleep(1.0)
        logger.info("Supervision loop ending"); self._shutdown_all()

    def _drain_hb(self):
        for _ in range(150):
            try: self._handle_msg(self._heartbeat_q.get_nowait())
            except Exception: break

    def _drain_sup(self):
        for _ in range(50):
            try: self._handle_sup(self._supervisor_q.get_nowait())
            except Exception: break

    def _drain_gui_cmd(self):
        for _ in range(30):
            try: msg = self._gui_cmd_q.get_nowait()
            except Exception: break
            mtype = msg.get("type")
            if mtype == MessageType.BOUNDARY_RELOAD.value:
                try: self._inference_q.put_nowait(msg)
                except Exception: pass
            elif mtype == MessageType.GUI_COMMAND.value:
                cmd = msg.get("command","")
                if cmd == "start_detection":
                    self._preview_mode.value = 0; logger.info("[Supervisor] Detection STARTED by operator")
                elif cmd == "stop_detection":
                    self._preview_mode.value = 1; logger.info("[Supervisor] Detection PAUSED by operator")

    def _handle_msg(self, msg):
        mtype = msg.get("type"); src = msg.get("source",""); cam_id = msg.get("camera_id")
        if mtype == MessageType.HEARTBEAT.value:
            key = self._find_key(src, cam_id)
            if key and key in self._processes:
                p = self._processes[key]
                p.last_heartbeat = time.time(); p.memory_mb = msg.get("memory_mb",0)
                p.last_fps = msg.get("fps",0); p.status = msg.get("status","running")
                p.extra = msg.get("extra",{}); p.pid_val = msg.get("pid",0)
                if p.last_fps > 0: p.last_fps_nonzero = time.time()
        elif mtype == MessageType.ERROR.value:
            fn = logger.critical if msg.get("severity")=="critical" else logger.error
            fn("[%s] cam=%s %s: %s", src, cam_id, msg.get("error_type","?"), msg.get("error_msg","?"))
        elif mtype == MessageType.GPU_STATS.value:
            self._last_gpu_stats = msg

    def _handle_sup(self, msg):
        mtype = msg.get("type")
        if mtype == MessageType.RESTART.value:
            target = msg.get("target","")
            if target == "detection_all": self._restart_all_detection()
        elif mtype == MessageType.GPU_STATS.value:
            self._last_gpu_stats = msg

    def _check_health(self):
        now = time.time()
        for key, mproc in list(self._processes.items()):
            if key == "gui": continue
            age = now - mproc.last_heartbeat; uptime = now - self._start_time
            grace = self.cfg.system.startup_grace_period_seconds
            hb_timeout = self.cfg.system.heartbeat_timeout_seconds
            # Extra grace for gpu_pool (model loading takes time)
            effective_grace = grace + 120 if key == "gpu_pool" else grace
            if age > hb_timeout and uptime > effective_grace:
                if not mproc.is_alive():
                    logger.error("[Supervisor] %s died → restart", key); self._restart_process(key); continue
            if not mproc.is_alive():
                if uptime > effective_grace:
                    logger.error("[Supervisor] %s not alive → restart", key); self._restart_process(key); continue
            self._check_proc_mem(key, mproc)

    def _check_proc_mem(self, key, mproc):
        limits = {f"camera_{c.id}": c.memory_limit_mb for c in self.cfg.cameras}
        limits.update({"gpu_pool": self.cfg.gpu_pool.memory_limit_mb,
                        "relay": self.cfg.relay.memory_limit_mb, "gui": self.cfg.gui.memory_limit_mb})
        for c in self.cfg.cameras: limits[f"detection_{c.id}"] = self.cfg.detection.memory_limit_mb
        limit = limits.get(key)
        if limit and mproc.memory_mb > limit:
            logger.warning("[Supervisor] %s mem %.1fMB > %.1fMB → restart", key, mproc.memory_mb, limit)
            self._restart_process(key)

    def _restart_process(self, key):
        mproc = self._processes.get(key)
        if not mproc: return
        mproc.restart_count += 1; logger.info("[Supervisor] Restarting %s (#%d)", key, mproc.restart_count)
        mproc.stop_event.set()
        if mproc.process and mproc.process.is_alive():
            mproc.process.terminate(); mproc.process.join(timeout=8.0)
            if mproc.process.is_alive(): mproc.process.kill()
        delay = min(self.cfg.supervisor.restart_backoff_seconds * mproc.restart_count, 60)
        time.sleep(delay); self._respawn(key)

    def _respawn(self, key):
        if key.startswith("camera_"): self._start_camera(int(key.split("_")[1]))
        elif key == "gpu_pool": self._start_gpu_pool()
        elif key.startswith("detection_"): self._start_detection(int(key.split("_")[1]))
        elif key == "relay": self._start_relay()
        elif key == "gui": self._start_gui()
        elif key == "gpu_monitor": self._start_gpu_monitor()

    def _restart_all_detection(self):
        if "gpu_pool" in self._processes: self._restart_process("gpu_pool")
        else:
            for cam_id in self.cfg.camera_ids:
                self._restart_process(f"detection_{cam_id}")
                time.sleep(self.cfg.supervisor.sequential_detection_restart_delay)

    def _check_daily_restart(self):
        if time.time() - self._last_daily_restart >= self.cfg.system.daily_restart_interval_hours * 3600:
            logger.info("[Supervisor] 24h restart"); self._last_daily_restart = time.time()
            self._restart_all_detection()
            for cam_id in self.cfg.camera_ids: self._restart_process(f"camera_{cam_id}"); time.sleep(2)
            self._restart_process("relay"); time.sleep(2)
            if "gui" in self._processes: self._restart_process("gui")

    def _maybe_broadcast_health(self):
        now = time.time()
        if now - self._last_health_broadcast < self.cfg.supervisor.health_broadcast_interval_seconds: return
        self._last_health_broadcast = now
        cpu_pct = 0.0; ram_mb = 0.0
        if _PSUTIL:
            try: cpu_pct = psutil.cpu_percent(interval=None); ram_mb = psutil.virtual_memory().used/(1024*1024)
            except Exception: pass
        procs_list = []; total_restarts = 0
        for key, mproc in self._processes.items():
            total_restarts += mproc.restart_count
            procs_list.append({"key":key,"name":mproc.name,"pid":mproc.pid_val,
                "status":mproc.status,"restart_count":mproc.restart_count,
                "memory_mb":mproc.memory_mb,"fps":mproc.last_fps,"alive":mproc.is_alive()})
        snap = HealthSnapshotMessage(processes=procs_list, gpu_stats=self._last_gpu_stats,
            cpu_pct=cpu_pct, ram_used_mb=ram_mb,
            system_uptime_s=now-self._start_time, total_restarts=total_restarts)
        try: self._health_q.put_nowait(snap.to_dict())
        except Exception: pass

    def _maybe_log_health(self):
        now = time.time()
        if now - self._last_health_log < self.cfg.supervisor.health_log_interval_seconds: return
        self._last_health_log = now
        uptime = now - self._start_time; h,m,s = int(uptime//3600),int((uptime%3600)//60),int(uptime%60)
        logger.info("[Health] Uptime %02dh%02dm%02ds", h, m, s)
        for key, mproc in self._processes.items():
            logger.info("  %-22s alive=%-5s restarts=%-3d mem=%.1fMB fps=%.1f",
                key, mproc.is_alive(), mproc.restart_count, mproc.memory_mb, mproc.last_fps)

    def _find_key(self, source, cam_id):
        if source == ProcessSource.CAMERA.value and cam_id is not None: return f"camera_{cam_id}"
        if source == ProcessSource.DETECTION.value: return f"detection_{cam_id}" if cam_id is not None else None
        if source == ProcessSource.GPU_POOL.value: return "gpu_pool"
        if source == ProcessSource.RELAY.value: return "relay"
        if source == ProcessSource.GUI.value: return "gui"
        if source == ProcessSource.GPU_MONITOR.value: return "gpu_monitor"
        return None

    def _on_sigterm(self, sig, frame): logger.info("[Supervisor] Signal %d → shutdown", sig); self._shutdown_requested = True; self._running = False

    def _shutdown_all(self):
        logger.info("[Supervisor] Shutting down all...")
        for key, mproc in self._processes.items():
            mproc.stop_event.set()
            if mproc.process and mproc.process.is_alive(): mproc.process.terminate()
        for key, mproc in self._processes.items():
            if mproc.process:
                mproc.process.join(timeout=10.0)
                if mproc.process.is_alive(): mproc.process.kill(); logger.warning("[Supervisor] Force-killed %s", key)
        logger.info("[Supervisor] Shutdown complete")


def run_supervisor(config_path):
    cfg = VisionSystemConfig(config_path)
    setup_process_logging("supervisor", cfg.logging.log_dir, cfg.system.log_level,
                          cfg.logging.max_bytes, cfg.logging.backup_count)
    setup_crash_handler("supervisor", cfg.logging.log_dir)
    Supervisor(config_path).start()

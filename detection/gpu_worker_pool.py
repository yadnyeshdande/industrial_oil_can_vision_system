"""
detection/gpu_worker_pool.py  v3.4
====================================
GPU Worker Pool — Shared-Model Architecture.

Previous architecture (v2.1):
    PoolManager spawns N GPUWorker SUBPROCESSES.
    Each subprocess loads a full copy of the YOLO model.
    With pool_size=2: 2 model copies × ~1.5 GB each = ~3 GB VRAM just for weights,
    plus 2 separate CUDA contexts and 2 rounds of torch/cuBLAS/cuFFT DLL loading.

New architecture (v3.4):
    PoolManager loads YOLO ONCE inside its own process.
    N worker THREADS share that single model via an inference lock.
    CUDA serialises concurrent kernels anyway, so there is no real throughput
    loss — YOLO on a single RTX 3050 at 1280×720 is already GPU-bound.

Benefits:
    • ~1.5 GB VRAM saved (one fewer model copy)
    • One fewer CUDA context → less driver overhead
    • One fewer round of cuBLAS/cuFFT DLL loads → less virtual memory pressure
    • No inter-process communication between workers → lower latency
    • Simpler shutdown (threads stop with the process)
    • Restart storm is less severe (only one CUDA init per pool restart)

Architecture:
    Camera 0 → SharedMemory[cam0] ─┐
    Camera 1 → SharedMemory[cam1] ─┼──> inference_queue ──> PoolManager
    Camera 2 → SharedMemory[cam2] ─┘                           │
                                                    ┌───────────┤
                                                    │   YOLO    │  ← loaded ONCE
                                                    │  (shared) │
                                                    └─── lock ──┤
                                                    Thread-0 ───┤
                                                    Thread-1 ───┘──> result_queue

IPC (unchanged from v2.1):
    Camera processes   → inference_queue   (InferenceRequestMessage)
    PoolManager process → result_queue     (DetectionResultMessage) → relay + GUI
    GUI cmd_queue      → inference_queue   (BOUNDARY_RELOAD)
    Heartbeats         → heartbeat_queue

pool_size in config now controls the number of INFERENCE THREADS (not processes).
Recommended: pool_size=2 is fine. With shared model, VRAM does not increase
proportionally — only thread overhead (~50MB RAM each) is added.
"""

from __future__ import annotations
import logging
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from multiprocessing import Queue, Process, Event
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config_loader import VisionSystemConfig, GPUPoolConfig, ModelConfig
from core.ipc_schema import (
    MessageType, ProcessSource,
    DetectionObject, DetectionResultMessage,
    InferenceRequestMessage,
    make_heartbeat, make_error,
)
from core.logging_setup import setup_process_logging, setup_crash_handler
from core.resource_monitor import get_process_memory_mb, is_memory_over_limit, get_gpu_stats, is_vram_over_limit
from core.shared_frame import SharedFrameReader
from core.boundary_engine import CameraBoundarySet

logger = logging.getLogger(__name__)


# ── Inference worker thread ────────────────────────────────────────────────────
class _InferenceThread(threading.Thread):
    """
    Pulls tasks from _task_q, runs inference through the shared model,
    posts results to result_queue. Shares the model object and inference_lock
    with all sibling threads (CUDA serialises concurrent kernel launches anyway).
    """

    def __init__(self, thread_id: int, cfg: VisionSystemConfig,
                 task_q: queue.Queue,
                 result_queue: Queue,
                 heartbeat_queue: Queue,
                 model,              # shared YOLO — do NOT reload, just call .predict()
                 inference_lock: threading.Lock,
                 stop_event: Event,
                 boundary_sets: Dict[int, Optional[CameraBoundarySet]],
                 boundary_lock: threading.Lock):
        super().__init__(name=f"InferThread-{thread_id}", daemon=True)
        self.tid           = thread_id
        self.cfg           = cfg
        self.mcfg          = cfg.model
        self.pcfg          = cfg.gpu_pool
        self.task_q        = task_q
        self.result_queue  = result_queue
        self.hb_q          = heartbeat_queue
        self.model         = model
        self.lock          = inference_lock
        self.stop_event    = stop_event
        self.b_sets        = boundary_sets
        self.b_lock        = boundary_lock

        self._readers:      Dict[int, SharedFrameReader] = {}
        self._fps_acc:      Dict[int, float]             = {}
        self._fps_ts:       Dict[int, float]             = {}
        self._inference_ms: float                        = 0.0
        self._last_hb:      float                        = time.time()

    def run(self):
        logger.info("[InferThread-%d] started", self.tid)
        while not self.stop_event.is_set():
            try:
                task = self.task_q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                self._process(task)
            except Exception as e:
                logger.error("[InferThread-%d] task error: %s", self.tid, e, exc_info=True)
        # Release shared-memory readers
        for r in self._readers.values():
            try: r.close()
            except Exception: pass
        logger.info("[InferThread-%d] stopped", self.tid)

    def _process(self, task: dict):
        cam_id  = task.get("camera_id")
        shm     = task.get("shared_memory_name", "")
        w       = task.get("frame_width",  1280)
        h       = task.get("frame_height", 720)
        t_req   = task.get("timestamp",    time.time())

        # Read frame from shared memory
        frame = self._read_frame(cam_id, shm, w, h)
        if frame is None:
            return

        # Run inference under the shared lock
        t0 = time.time()
        with self.lock:
            detections = self._infer(frame)
        self._inference_ms = (time.time() - t0) * 1000

        # Apply boundary logic
        pair_results, relay_states, problem_count = \
            self._apply_boundaries(cam_id, frame, detections)

        # FPS accounting
        fps = self._update_fps(cam_id)

        # Encode thumbnail
        frame_data = self._encode(frame)

        # Build result
        det_dicts = [
            {"class_id": d.class_id, "class_name": d.class_name,
             "confidence": d.confidence, "bbox": list(d.bbox)}
            for d in detections
        ]
        sr = 0.0
        msg: dict = DetectionResultMessage(
            camera_id=cam_id,
            detections=det_dicts,
            pair_results=pair_results,
            relay_states=relay_states,
            frame_data=frame_data,
            inference_time_ms=self._inference_ms,
            fps=fps,
            problem_count=problem_count,
            success_rate=sr,
        ).to_dict()
        try:
            self.result_queue.put_nowait(msg)
        except Exception:
            pass

        self._maybe_heartbeat(cam_id, fps)

    def _read_frame(self, cam_id, shm_name, w, h) -> Optional[np.ndarray]:
        if cam_id not in self._readers:
            r = SharedFrameReader(shm_name, w, h)
            if not r.connect(timeout=5.0):
                logger.warning("[InferThread-%d] Cannot connect to shm %s", self.tid, shm_name)
                return None
            self._readers[cam_id] = r
        try:
            frame, _ = self._readers[cam_id].read()
            return frame
        except Exception as e:
            logger.debug("[InferThread-%d] shm read error cam%d: %s", self.tid, cam_id, e)
            self._readers.pop(cam_id, None)
            return None

    def _infer(self, frame: np.ndarray) -> List[DetectionObject]:
        import torch
        with torch.no_grad():
            results = self.model.predict(
                frame, conf=self.mcfg.confidence,
                iou=self.mcfg.iou, verbose=False)
        dets = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id   = int(box.cls[0].item())
                conf     = float(box.conf[0].item())
                x1,y1,x2,y2 = box.xyxyn[0].tolist()
                cls_name = (self.mcfg.class_names[cls_id]
                            if cls_id < len(self.mcfg.class_names) else str(cls_id))
                dets.append(DetectionObject(
                    class_id=cls_id, class_name=cls_name,
                    confidence=conf, bbox=(x1,y1,x2,y2)))
        return dets

    def _apply_boundaries(self, cam_id, frame, detections):
        with self.b_lock:
            bset = self.b_sets.get(cam_id)
        if bset is None:
            return [], [], 0
        try:
            pair_results  = bset.evaluate(detections)
            relay_states  = [p.get("relay_active", False) for p in pair_results]
            problem_count = sum(1 for p in pair_results if p.get("relay_active", False))
            return pair_results, relay_states, problem_count
        except Exception as e:
            logger.debug("[InferThread-%d] boundary eval error: %s", self.tid, e)
            return [], [], 0

    def _update_fps(self, cam_id) -> float:
        now = time.time()
        prev = self._fps_ts.get(cam_id, now - 1)
        dt   = now - prev
        self._fps_ts[cam_id] = now
        instant  = 1.0 / max(dt, 0.001)
        smoothed = self._fps_acc.get(cam_id, instant)
        smoothed = 0.9 * smoothed + 0.1 * instant
        self._fps_acc[cam_id] = smoothed
        return round(smoothed, 1)

    def _encode(self, frame: np.ndarray) -> Optional[bytes]:
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buf.tobytes()
        except Exception:
            return None

    def _maybe_heartbeat(self, cam_id, fps):
        now = time.time()
        if now - self._last_hb < self.pcfg.heartbeat_interval_seconds:
            return
        self._last_hb = now
        avg_fps = sum(self._fps_acc.values()) / max(len(self._fps_acc), 1)
        vram    = get_gpu_stats() or {}
        hb = make_heartbeat(
            source=ProcessSource.GPU_POOL,
            camera_id=None,
            process_name=f"InferThread-{self.tid}",
            pid=os.getpid(),
            memory_mb=get_process_memory_mb(),
            fps=avg_fps,
            status="running",
            extra={
                "worker_id":   self.tid,
                "vram_mb":     round(vram.get("vram_used_mb", 0), 1),
                "inference_ms": round(self._inference_ms, 1),
                "shared_model": True,
            },
        )
        try:
            self.hb_q.put_nowait(hb.to_dict())
        except Exception:
            pass


# ── Pool Manager — loads model once, runs shared inference threads ─────────────
class PoolManager:
    """
    Single process that owns the YOLO model.
    Spawns pool_size inference threads that share the model via a lock.
    Boundary reload signals from the GUI are applied without restarting.
    """

    def __init__(self, cfg: VisionSystemConfig,
                 inference_queue: Queue,
                 result_queue:    Queue,
                 heartbeat_queue: Queue,
                 stop_event:      Event):
        self.cfg            = cfg
        self.pcfg           = cfg.gpu_pool
        self.mcfg           = cfg.model
        self.inference_queue = inference_queue
        self.result_queue   = result_queue
        self.hb_q           = heartbeat_queue
        self.stop_event     = stop_event
        self.pid            = os.getpid()
        self.name           = "GPUPoolManager"

        self._model         = None          # loaded once below
        self._inference_lock = threading.Lock()
        self._boundary_sets:  Dict[int, Optional[CameraBoundarySet]] = {}
        self._boundary_mtimes: Dict[int, float] = {}
        self._boundary_lock  = threading.Lock()
        self._last_boundary_poll = 0.0
        self._threads: List[_InferenceThread] = []
        self._task_q:  queue.Queue = queue.Queue(maxsize=60)
        self._last_hb  = time.time()
        self._last_vram_check = 0.0

    def run(self):
        logger.info("[PoolManager] PID=%d — shared-model pool, %d inference threads",
                    self.pid, self.pcfg.pool_size)

        # Load model ONCE — all threads will share this object
        if not self._load_model():
            logger.critical("[PoolManager] Model load failed — pool cannot start")
            return

        # Load initial boundaries for all cameras
        for cam in self.cfg.cameras:
            self._load_boundaries(cam.id)

        # Start inference threads (not processes — shared model, no VRAM duplication)
        for i in range(self.pcfg.pool_size):
            t = _InferenceThread(
                thread_id      = i,
                cfg            = self.cfg,
                task_q         = self._task_q,
                result_queue   = self.result_queue,
                heartbeat_queue= self.hb_q,
                model          = self._model,
                inference_lock = self._inference_lock,
                stop_event     = self.stop_event,
                boundary_sets  = self._boundary_sets,
                boundary_lock  = self._boundary_lock,
            )
            t.start()
            self._threads.append(t)
            logger.info("[PoolManager] Started InferThread-%d", i)

        # Dispatch loop
        while not self.stop_event.is_set():
            # Route incoming messages
            try:
                msg = self.inference_queue.get(timeout=0.5)
            except Exception:
                self._poll_boundaries()
                self._maybe_manager_hb()
                self._check_vram()
                continue

            mtype = msg.get("type")
            if mtype == MessageType.SHUTDOWN.value:
                break
            elif mtype == MessageType.BOUNDARY_RELOAD.value:
                cam_id = msg.get("camera_id")
                if cam_id is not None:
                    logger.info("[PoolManager] Boundary reload for cam %d", cam_id)
                    self._load_boundaries(cam_id)
                continue
            elif mtype == MessageType.INFERENCE_REQUEST.value:
                try:
                    self._task_q.put_nowait(msg)
                except queue.Full:
                    pass  # drop — latest-frame-only model

            self._poll_boundaries()
            self._maybe_manager_hb()
            self._check_vram()

        # Shutdown: signal threads (they are daemon threads, will stop with process)
        logger.info("[PoolManager] Stopping %d inference threads", len(self._threads))
        for t in self._threads:
            t.join(timeout=5.0)
        self._cleanup()

    def _load_model(self) -> bool:
        """Load YOLO model once into this process. All threads will share it."""
        try:
            import torch
            from ultralytics import YOLO
            torch.backends.cudnn.benchmark = False

            device_str = self.pcfg.device
            if device_str == "cuda" and not torch.cuda.is_available():
                logger.warning("[PoolManager] CUDA unavailable, using CPU")
                device_str = "cpu"

            model_path = Path(self.mcfg.path)
            if not model_path.is_absolute():
                model_path = _ROOT / model_path
            if not model_path.exists():
                alt = Path(self.cfg.detection.model_path)
                if not alt.is_absolute():
                    alt = _ROOT / alt
                if alt.exists():
                    model_path = alt
                else:
                    logger.error("[PoolManager] Model not found at %s or %s", model_path, alt)
                    return False

            logger.info("[PoolManager] Loading shared model %s on %s", model_path, device_str)
            self._model = YOLO(str(model_path))
            self._model.to(torch.device(device_str))

            # Warmup in fp32 first — triggers model.fuse() safely
            cam   = self.cfg.cameras[0]
            dummy = np.zeros((cam.frame_height, cam.frame_width, 3), dtype=np.uint8)
            logger.info("[PoolManager] Warmup (fp32)...")
            self._model.predict(dummy, conf=self.mcfg.confidence,
                                iou=self.mcfg.iou, verbose=False)

            # Now switch to fp16 if requested — fuse() has already run
            if self.pcfg.use_fp16 and device_str == "cuda":
                try:
                    self._model.model.half()
                    self._model.predict(dummy, conf=self.mcfg.confidence,
                                        iou=self.mcfg.iou, verbose=False)
                    logger.info("[PoolManager] FP16 enabled and verified")
                except Exception as e:
                    logger.warning("[PoolManager] FP16 failed (%s) — running fp32", e)
                    self._model.model.float()

            logger.info("[PoolManager] Shared model ready — VRAM ~%.0f MB",
                        get_gpu_stats().get("vram_used_mb", 0) if get_gpu_stats() else 0)
            return True
        except Exception as e:
            logger.critical("[PoolManager] Model load error: %s", e, exc_info=True)
            return False

    def _load_boundaries(self, cam_id: int):
        """
        Load (or reload) boundary definitions for one camera.

        Uses self._boundary_lock — the PoolManager-owned lock.
        This was previously self.b_lock which only exists on _InferenceThread,
        causing AttributeError: 'PoolManager' object has no attribute 'b_lock'.

        Also wrapped in a broad try/except per the industrial rule:
        detection must survive configuration failures.
        """
        bd_path = self.cfg.get_boundary_path(cam_id)
        try:
            if not bd_path.exists():
                with self._boundary_lock:
                    self._boundary_sets[cam_id] = None
                logger.debug("[PoolManager] No boundary file for cam %d — detection continues without boundaries", cam_id)
                return
            import json
            with open(bd_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            norm = _normalize_boundary_data(data, cam_id)
            bset = CameraBoundarySet(norm)
            mtime = bd_path.stat().st_mtime
            with self._boundary_lock:
                self._boundary_sets[cam_id] = bset
                self._boundary_mtimes[cam_id] = mtime
            logger.info("[PoolManager] Loaded boundaries cam %d", cam_id)
        except Exception as e:
            # Detection must never die because of boundary config errors.
            # Log it, mark as None (pass-through), and keep running.
            logger.error("[PoolManager] Boundary load error cam %d: %s — "
                         "detection continues without boundaries", cam_id, e)
            try:
                with self._boundary_lock:
                    self._boundary_sets[cam_id] = None
            except Exception:
                pass

    def _poll_boundaries(self):
        now = time.time()
        if now - self._last_boundary_poll < self.pcfg.boundary_poll_interval_seconds:
            return
        self._last_boundary_poll = now
        for cam in self.cfg.cameras:
            bd_path = self.cfg.get_boundary_path(cam.id)
            try:
                mtime = bd_path.stat().st_mtime
                if mtime != self._boundary_mtimes.get(cam.id, 0):
                    logger.info("[PoolManager] Boundary file changed cam %d", cam.id)
                    self._load_boundaries(cam.id)
            except Exception:
                pass

    def _maybe_manager_hb(self):
        now = time.time()
        if now - self._last_hb < self.pcfg.heartbeat_interval_seconds:
            return
        self._last_hb = now
        vram = get_gpu_stats() or {}
        hb = make_heartbeat(
            source=ProcessSource.GPU_POOL,
            camera_id=None,
            process_name=self.name,
            pid=self.pid,
            memory_mb=get_process_memory_mb(),
            fps=0.0,
            status="running",
            extra={
                "shared_model":  True,
                "pool_size":     self.pcfg.pool_size,
                "vram_mb":       round(vram.get("vram_used_mb", 0), 1),
                "threads_alive": sum(1 for t in self._threads if t.is_alive()),
            },
        )
        try:
            self.hb_q.put_nowait(hb.to_dict())
        except Exception:
            pass

    def _check_vram(self):
        """Emit an error heartbeat if VRAM is over the configured limit."""
        now = time.time()
        if now - self._last_vram_check < 10.0:
            return
        self._last_vram_check = now
        if is_vram_over_limit(self.pcfg.vram_limit_mb):
            vram = get_gpu_stats() or {}
            logger.critical("[PoolManager] VRAM %.1f/%.1f MB exceeded",
                            vram.get("vram_used_mb", 0), self.pcfg.vram_limit_mb)
            err = make_error(ProcessSource.GPU_POOL, None, "VRAMLimitExceeded",
                             f"VRAM > {self.pcfg.vram_limit_mb}MB", severity="critical")
            try:
                self.hb_q.put_nowait(err.to_dict())
            except Exception:
                pass

    def _cleanup(self):
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[PoolManager] Cleaned up")


# ── Normalise boundary JSON (v1 → v2 if needed) ───────────────────────────────
def _normalize_boundary_data(data: dict, camera_id: int) -> dict:
    if "boundaries" in data and "pairs" in data:
        return data

    oc_raw = data.get("oil_can", [])
    bh_raw = data.get("bunk_hole", [])

    def to_boundary(item, idx):
        bid     = item.get("id", f"B{idx}")
        polygon = item.get("polygon", item.get("points", []))
        return {"id": bid, "name": bid, "type": "polygon", "points": polygon, "pair": ""}

    oc_boundaries = [to_boundary(b, i) for i, b in enumerate(oc_raw)]
    bh_boundaries = [to_boundary(b, i) for i, b in enumerate(bh_raw)]

    relay_mapping = None
    try:
        from core.config_loader import get_config
        c = get_config()
        relay_mapping = c.relay.get_relay_indices(camera_id)
    except Exception:
        pass

    pairs = []
    for i in range(min(len(oc_boundaries), len(bh_boundaries))):
        oc_id = oc_boundaries[i]["id"]
        bh_id = bh_boundaries[i]["id"]
        oc_boundaries[i]["pair"] = bh_id
        bh_boundaries[i]["pair"] = oc_id
        relay_idx = (relay_mapping[i] if relay_mapping and i < len(relay_mapping)
                     else camera_id * 3 + i)
        pairs.append({
            "id": i,
            "name": f"Pair {i + 1}",
            "oil_can_boundary":   oc_id,
            "bunk_hole_boundary": bh_id,
            "relay_index":        relay_idx,
        })

    return {
        "camera_id": camera_id,
        "boundaries": {"oil_can": oc_boundaries, "bunk_hole": bh_boundaries},
        "pairs": pairs,
    }


# ── Process entry point ────────────────────────────────────────────────────────
def pool_manager_process_entry(config_path: str,
                                inference_queue: Queue,
                                result_queue:    Queue,
                                heartbeat_queue: Queue,
                                stop_event:      Event,
                                log_dir: str = "logs"):
    from core.config_loader import VisionSystemConfig
    cfg = VisionSystemConfig(config_path)
    setup_process_logging("gpu_pool", log_dir, cfg.system.log_level,
                          cfg.logging.max_bytes, cfg.logging.backup_count)
    setup_crash_handler("gpu_pool", log_dir)

    def _sig(s, f): stop_event.set()
    signal.signal(signal.SIGTERM, _sig)

    manager = PoolManager(cfg, inference_queue, result_queue, heartbeat_queue, stop_event)
    try:
        manager.run()
    except Exception as e:
        logger.critical("[gpu_pool] Fatal: %s", e, exc_info=True)
        sys.exit(1)


# ── Backward compat stubs used by tests ───────────────────────────────────────
def _gpu_worker_entry(*args, **kwargs):
    """Stub — replaced by shared-model thread architecture in v3.4."""
    raise RuntimeError("_gpu_worker_entry is no longer used. "
                       "PoolManager now uses inference threads internally.")

"""
detection/gpu_worker_pool.py
=============================
GPU Worker Pool — v2.1

Architecture:
    Camera 0 → SharedMemory[cam0] ─┐
    Camera 1 → SharedMemory[cam1] ─┼──> inference_queue ──> GPUWorker[0] ──> result_queue
    Camera 2 → SharedMemory[cam2] ─┘                    ──> GPUWorker[1] ─┘

Key design decisions vs per-camera detection processes:
  • Only 1–2 YOLO models in VRAM instead of 3 (saves ~3GB)
  • No CUDA context fragmentation over days of runtime
  • VRAM stays ~1.6 GB stable vs ~4.5 GB with 3 workers
  • Each GPU worker handles ALL cameras round-robin
  • Boundary hot-reload: file watcher signals reload without restart

Processes spawned:
  • PoolManagerProcess — receives inference requests, dispatches to workers
  • GPUWorkerProcess × pool_size — each loads model once, processes frames

IPC:
  Camera processes → inference_queue (InferenceRequestMessage)
  Pool manager → worker_queues (per worker)
  GPU workers → result_queue (DetectionResultMessage) → relay + GUI
  Boundary file watcher → reload_event (signals all workers)
"""

from __future__ import annotations
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
import threading
from multiprocessing import Queue, Process, Event, Value
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# GPU Worker — one per pool_size
# ---------------------------------------------------------------------------

class GPUWorker:
    """
    Single GPU worker. Loads YOLO once, processes frames from its own queue.
    Handles all cameras. Reloads boundaries on signal.
    """

    def __init__(self,
                 worker_id: int,
                 cfg: VisionSystemConfig,
                 task_queue: Queue,        # receives InferenceRequestMessage dicts
                 result_queue: Queue,      # sends DetectionResultMessage dicts
                 heartbeat_queue: Queue,
                 stop_event: Event,
                 reload_events: Dict[int, Event]):   # camera_id → boundary reload event
        self.worker_id = worker_id
        self.cfg = cfg
        self.pcfg: GPUPoolConfig = cfg.gpu_pool
        self.mcfg: ModelConfig = cfg.model
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.heartbeat_queue = heartbeat_queue
        self.stop_event = stop_event
        self.reload_events = reload_events

        self.pid = os.getpid()
        self.name = f"GPUWorker_{worker_id}"

        # Per-camera shared memory readers (lazy init)
        self._readers: Dict[int, SharedFrameReader] = {}
        # Per-camera boundary sets
        self._boundary_sets: Dict[int, Optional[CameraBoundarySet]] = {}
        # Per-camera stats
        self._fps_counters: Dict[int, float] = {}
        self._fps_timestamps: Dict[int, float] = {}
        self._frame_counts: Dict[int, int] = {}
        self._problem_counts: Dict[int, int] = {}
        self._ok_counts: Dict[int, int] = {}
        self._total_dets: Dict[int, int] = {}

        self._model = None
        self._device = None
        self._last_heartbeat = time.time()
        self._last_boundary_check: Dict[int, float] = {}
        self._boundary_mtimes: Dict[int, float] = {}
        self._inference_ms = 0.0

    # ------------------------------------------------------------------
    def run(self):
        logger.info("[%s] PID=%d starting", self.name, self.pid)
        if not self._load_model():
            logger.critical("[%s] Model load failed, exiting", self.name)
            return

        # Load boundaries for all cameras
        for cam in self.cfg.cameras:
            self._load_boundaries(cam.id)
            self._last_boundary_check[cam.id] = time.time()

        logger.info("[%s] Ready — serving %d cameras", self.name, len(self.cfg.cameras))

        while not self.stop_event.is_set():
            # Check boundary reload signals
            self._check_boundary_reloads()

            # Get next task
            try:
                task = self.task_queue.get(timeout=0.2)
            except Exception:
                self._maybe_heartbeat()
                continue

            if task.get("type") == MessageType.SHUTDOWN.value:
                break

            self._process_task(task)
            self._maybe_heartbeat()

        self._cleanup()
        logger.info("[%s] Exiting cleanly", self.name)

    # ------------------------------------------------------------------
    def _process_task(self, task: dict):
        camera_id = task.get("camera_id")
        shm_name = task.get("shm_name", "")
        frame_shape = task.get("frame_shape", (720, 1280, 3))

        # Get or create shared memory reader for this camera
        reader = self._get_reader(camera_id, shm_name, frame_shape)
        if reader is None:
            return

        frame, frame_idx = reader.read()
        if frame is None:
            return  # No new frame yet

        # Run YOLO inference
        try:
            detections = self._run_inference(frame)
        except Exception as e:
            logger.error("[%s] Inference error cam=%d: %s", self.name, camera_id, e)
            return

        # Boundary pairing
        h, w = frame.shape[:2]
        pair_results = []
        bset = self._boundary_sets.get(camera_id)
        if bset:
            try:
                pair_results = bset.evaluate(
                    detections, w, h,
                    self.mcfg.oil_can_class_id,
                    self.mcfg.bunk_hole_class_id,
                )
            except Exception as e:
                logger.error("[%s] Boundary eval error cam=%d: %s", self.name, camera_id, e)

        # Update per-camera stats
        cid = camera_id
        self._frame_counts[cid] = self._frame_counts.get(cid, 0) + 1
        self._total_dets[cid] = self._total_dets.get(cid, 0) + len(detections)
        problems = sum(1 for p in pair_results if p.relay_active)
        if problems:
            self._problem_counts[cid] = self._problem_counts.get(cid, 0) + 1
        else:
            self._ok_counts[cid] = self._ok_counts.get(cid, 0) + 1

        # FPS
        now = time.time()
        ts = self._fps_timestamps.get(cid, now)
        elapsed = now - ts
        if elapsed >= 2.0:
            fps = self._frame_counts.get(cid, 0) / elapsed
            self._fps_counters[cid] = fps
            self._frame_counts[cid] = 0
            self._fps_timestamps[cid] = now
        fps = self._fps_counters.get(cid, 0.0)

        # Success rate
        total = self._problem_counts.get(cid, 0) + self._ok_counts.get(cid, 0)
        success_rate = (self._ok_counts.get(cid, 0) / total * 100) if total > 0 else 100.0

        # Encode frame for GUI
        frame_jpeg = self._encode_frame(frame)

        # Build result message
        result = DetectionResultMessage(
            source=ProcessSource.GPU_POOL,
            camera_id=camera_id,
            detections=[d.to_dict() for d in detections],
            pair_results=[p.to_dict() for p in pair_results],
            inference_time_ms=self._inference_ms,
            fps=fps,
            total_detections=self._total_dets.get(cid, 0),
            problem_count=self._problem_counts.get(cid, 0),
            success_rate=success_rate,
            frame_shape=(h, w, 3),
            frame_data=frame_jpeg,
        )
        try:
            self.result_queue.put_nowait(result.to_dict())
        except Exception:
            pass

        # VRAM guard
        if is_vram_over_limit(self.pcfg.vram_limit_mb):
            logger.critical("[%s] VRAM exceeded, requesting restart", self.name)
            err = make_error(ProcessSource.GPU_POOL, None, "VRAMLimitExceeded",
                             f"VRAM > {self.pcfg.vram_limit_mb}MB", severity="critical")
            try:
                self.heartbeat_queue.put_nowait(err.to_dict())
            except Exception:
                pass
            self.stop_event.set()

    # ------------------------------------------------------------------
    def _load_model(self) -> bool:
        try:
            import torch
            from ultralytics import YOLO
            torch.backends.cudnn.benchmark = False

            device_str = self.pcfg.device
            if device_str == "cuda" and not torch.cuda.is_available():
                logger.warning("[%s] CUDA unavailable, using CPU", self.name)
                device_str = "cpu"

            self._device = torch.device(device_str)
            model_path = Path(self.mcfg.path)
            if not model_path.is_absolute():
                model_path = _ROOT / model_path

            if not model_path.exists():
                # Try detection.model_path as fallback
                alt = Path(self.cfg.detection.model_path)
                if not alt.is_absolute():
                    alt = _ROOT / alt
                if alt.exists():
                    model_path = alt
                else:
                    logger.error("[%s] Model not found at %s or %s", self.name, model_path, alt)
                    return False

            logger.info("[%s] Loading %s on %s", self.name, model_path, device_str)
            self._model = YOLO(str(model_path))
            self._model.to(self._device)

            # CRITICAL: warmup MUST run in float32 before calling .half().
            # Calling model.model.half() before warmup triggers Ultralytics
            # model.fuse() which fails with:
            #   RuntimeError: expected scalar type Half but found Float
            # Correct order: load → warmup (fp32) → half()
            cam = self.cfg.cameras[0]
            dummy = np.zeros((cam.frame_height, cam.frame_width, 3), dtype=np.uint8)
            logger.info("[%s] Warming up (fp32)...", self.name)
            self._model.predict(dummy, conf=self.mcfg.confidence,
                                iou=self.mcfg.iou, verbose=False)

            # NOW safe to switch to fp16 — fusion already happened during warmup
            if self.pcfg.use_fp16 and device_str == "cuda":
                try:
                    self._model.model.half()
                    # Verify with a second pass in fp16
                    self._model.predict(dummy, conf=self.mcfg.confidence,
                                        iou=self.mcfg.iou, verbose=False)
                    logger.info("[%s] FP16 enabled and verified", self.name)
                except Exception as fp16_err:
                    logger.warning("[%s] FP16 failed (%s), falling back to fp32", self.name, fp16_err)
                    self._model.model.float()

            logger.info("[%s] Model ready", self.name)
            return True
        except Exception as e:
            logger.critical("[%s] Model load error: %s", self.name, e, exc_info=True)
            return False

    def _run_inference(self, frame: np.ndarray) -> List[DetectionObject]:
        import torch
        t0 = time.time()
        with torch.no_grad():
            results = self._model.predict(
                frame,
                conf=self.mcfg.confidence,
                iou=self.mcfg.iou,
                verbose=False,
            )
        self._inference_ms = (time.time() - t0) * 1000
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                cls_name = (self.mcfg.class_names[cls_id]
                            if cls_id < len(self.mcfg.class_names) else str(cls_id))
                detections.append(DetectionObject(
                    class_id=cls_id, class_name=cls_name,
                    confidence=conf, bbox=(x1, y1, x2, y2)
                ))
        return detections

    def _get_reader(self, camera_id: int, shm_name: str,
                    frame_shape: tuple) -> Optional[SharedFrameReader]:
        if camera_id not in self._readers:
            cam = self.cfg.get_camera(camera_id)
            if cam is None:
                return None
            reader = SharedFrameReader(
                name=shm_name,
                width=cam.frame_width,
                height=cam.frame_height,
            )
            if not reader.connect(timeout=5.0):
                logger.warning("[%s] Cannot connect to shm %s", self.name, shm_name)
                return None
            self._readers[camera_id] = reader
            self._fps_timestamps[camera_id] = time.time()
        return self._readers[camera_id]

    def _load_boundaries(self, camera_id: int):
        data = self.cfg.load_boundaries(camera_id)
        if data:
            # Support both v1 and v2 boundary JSON formats
            normalized = _normalize_boundary_data(data, camera_id)
            self._boundary_sets[camera_id] = CameraBoundarySet(
                normalized,
                strict_mode=self.mcfg.strict_boundary_mode
            )
            bd_path = self.cfg.get_boundary_path(camera_id)
            try:
                self._boundary_mtimes[camera_id] = bd_path.stat().st_mtime
            except Exception:
                self._boundary_mtimes[camera_id] = 0.0
            logger.info("[%s] Boundaries loaded for camera %d", self.name, camera_id)
        else:
            self._boundary_sets[camera_id] = None

    def _check_boundary_reloads(self):
        """
        Two-mechanism hot-reload:
        1. Event signal from pool manager (boundary file changed)
        2. Periodic file mtime polling as fallback
        """
        for cam in self.cfg.cameras:
            cid = cam.id
            # Event-based reload
            ev = self.reload_events.get(cid)
            if ev and ev.is_set():
                logger.info("[%s] Boundary reload signal for cam %d", self.name, cid)
                self._load_boundaries(cid)
                ev.clear()
                continue
            # Mtime polling
            now = time.time()
            last_check = self._last_boundary_check.get(cid, 0)
            if now - last_check >= self.cfg.gpu_pool.boundary_poll_interval_seconds:
                self._last_boundary_check[cid] = now
                bd_path = self.cfg.get_boundary_path(cid)
                try:
                    mtime = bd_path.stat().st_mtime
                    if mtime != self._boundary_mtimes.get(cid, 0):
                        logger.info("[%s] Boundary file changed for cam %d (mtime)", self.name, cid)
                        self._load_boundaries(cid)
                except Exception:
                    pass

    def _encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buf.tobytes()
        except Exception:
            return None

    def _maybe_heartbeat(self):
        now = time.time()
        if now - self._last_heartbeat >= self.pcfg.heartbeat_interval_seconds:
            vram = get_gpu_stats()
            avg_fps = sum(self._fps_counters.values()) / max(len(self._fps_counters), 1)
            hb = make_heartbeat(
                source=ProcessSource.GPU_POOL,
                camera_id=None,
                process_name=self.name,
                pid=self.pid,
                memory_mb=get_process_memory_mb(),
                fps=avg_fps,
                status="running",
                extra={
                    "worker_id": self.worker_id,
                    "vram_mb": round(vram.get("vram_used_mb", 0), 1),
                    "inference_ms": round(self._inference_ms, 1),
                    "cameras": list(self._fps_counters.keys()),
                },
            )
            try:
                self.heartbeat_queue.put_nowait(hb.to_dict())
            except Exception:
                pass
            self._last_heartbeat = now

    def _cleanup(self):
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        for r in self._readers.values():
            r.close()
        logger.info("[%s] CUDA cache cleared, readers closed", self.name)


# ---------------------------------------------------------------------------
# Pool Manager — routes requests to worker queues round-robin
# ---------------------------------------------------------------------------

class PoolManager:
    """
    Receives InferenceRequestMessages from all camera processes.
    Routes to GPU workers round-robin.
    Monitors boundary file changes and signals workers.
    """

    def __init__(self,
                 cfg: VisionSystemConfig,
                 inference_queue: Queue,
                 result_queue: Queue,
                 heartbeat_queue: Queue,
                 stop_event: Event):
        self.cfg = cfg
        self.inference_queue = inference_queue
        self.result_queue = result_queue
        self.heartbeat_queue = heartbeat_queue
        self.stop_event = stop_event
        self.pid = os.getpid()
        self.name = "GPUPoolManager"

        self._worker_queues: List[Queue] = []
        self._workers: List[Process] = []
        self._reload_events: Dict[int, Event] = {}   # camera_id → Event
        self._rr_index = 0    # round-robin counter

    def run(self):
        logger.info("[PoolManager] PID=%d starting with pool_size=%d",
                    self.pid, self.cfg.gpu_pool.pool_size)

        # Create reload events for each camera
        for cam in self.cfg.cameras:
            self._reload_events[cam.id] = Event()

        # Start GPU worker subprocesses
        for i in range(self.cfg.gpu_pool.pool_size):
            q: Queue = mp.Queue(maxsize=30)
            self._worker_queues.append(q)
            p = Process(
                target=_gpu_worker_entry,
                args=(i, str(self.cfg._path), q, self.result_queue,
                      self.heartbeat_queue, self.stop_event,
                      {cid: ev for cid, ev in self._reload_events.items()}),
                name=f"gpu_worker_{i}",
                daemon=True,
            )
            p.start()
            self._workers.append(p)
            logger.info("[PoolManager] Started GPUWorker_%d PID=%d", i, p.pid)
            time.sleep(3.0)   # stagger GPU model loading

        # Dispatch loop
        while not self.stop_event.is_set():
            try:
                task = self.inference_queue.get(timeout=0.5)
            except Exception:
                continue

            if task.get("type") == MessageType.SHUTDOWN.value:
                break

            if task.get("type") == MessageType.BOUNDARY_RELOAD.value:
                cam_id = task.get("camera_id")
                if cam_id is not None and cam_id in self._reload_events:
                    logger.info("[PoolManager] Boundary reload signal for cam %d", cam_id)
                    self._reload_events[cam_id].set()
                continue

            # Route to next worker round-robin
            if self._worker_queues:
                target_q = self._worker_queues[self._rr_index % len(self._worker_queues)]
                self._rr_index += 1
                try:
                    target_q.put_nowait(task)
                except Exception:
                    pass  # Worker busy — drop frame (latest-frame-only model)

        # Shutdown all workers
        shutdown_msg = {"type": MessageType.SHUTDOWN.value, "reason": "pool_stop"}
        for q in self._worker_queues:
            try:
                q.put_nowait(shutdown_msg)
            except Exception:
                pass

        for p in self._workers:
            p.join(timeout=8.0)
            if p.is_alive():
                p.kill()

        logger.info("[PoolManager] All GPU workers stopped")


# ---------------------------------------------------------------------------
# Normalise boundary JSON (v1 → v2 if needed)
# ---------------------------------------------------------------------------

def _normalize_boundary_data(data: dict, camera_id: int) -> dict:
    """
    Accept both old format (with 'boundaries'.'oil_can' list) and
    new flat format (with 'oil_can' list directly).
    Returns always the old format that CameraBoundarySet expects.
    """
    # Already in correct format
    if "boundaries" in data and "pairs" in data:
        return data

    # New flat format from boundary editor
    oc_raw = data.get("oil_can", [])
    bh_raw = data.get("bunk_hole", [])

    def to_boundary(item, idx):
        bid = item.get("id", f"B{idx}")
        polygon = item.get("polygon", item.get("points", []))
        return {"id": bid, "name": bid, "type": "polygon", "points": polygon, "pair": ""}

    oc_boundaries = [to_boundary(b, i) for i, b in enumerate(oc_raw)]
    bh_boundaries = [to_boundary(b, i) for i, b in enumerate(bh_raw)]

    # Auto-pair OC1↔BH1, OC2↔BH2, OC3↔BH3
    pairs = []
    n_pairs = min(len(oc_boundaries), len(bh_boundaries))
    relay_mapping = None
    try:
        from core.config_loader import get_config
        c = get_config()
        relay_mapping = c.relay.get_relay_indices(camera_id)
    except Exception:
        pass

    for i in range(n_pairs):
        oc_id = oc_boundaries[i]["id"]
        bh_id = bh_boundaries[i]["id"]
        oc_boundaries[i]["pair"] = bh_id
        bh_boundaries[i]["pair"] = oc_id
        relay_idx = relay_mapping[i] if relay_mapping and i < len(relay_mapping) else camera_id * 3 + i
        pairs.append({
            "id": i,
            "name": f"Pair {i + 1}",
            "oil_can_boundary": oc_id,
            "bunk_hole_boundary": bh_id,
            "relay_index": relay_idx,
        })

    return {
        "camera_id": camera_id,
        "boundaries": {"oil_can": oc_boundaries, "bunk_hole": bh_boundaries},
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Process entry points
# ---------------------------------------------------------------------------

def _gpu_worker_entry(worker_id, config_path, task_queue, result_queue,
                      heartbeat_queue, stop_event, reload_events):
    """Subprocess entry for each GPU worker."""
    from core.config_loader import VisionSystemConfig
    cfg = VisionSystemConfig(config_path)
    pname = f"gpu_worker_{worker_id}"
    setup_process_logging(pname, cfg.logging.log_dir, cfg.system.log_level,
                          cfg.logging.max_bytes, cfg.logging.backup_count)
    setup_crash_handler(pname, cfg.logging.log_dir)

    def _sig(s, f): stop_event.set()
    signal.signal(signal.SIGTERM, _sig)

    worker = GPUWorker(worker_id, cfg, task_queue, result_queue,
                       heartbeat_queue, stop_event, reload_events)
    try:
        worker.run()
    except Exception as e:
        logger.critical("[gpu_worker_%d] Fatal: %s", worker_id, e, exc_info=True)
        sys.exit(1)


def pool_manager_process_entry(config_path: str,
                                inference_queue: Queue,
                                result_queue: Queue,
                                heartbeat_queue: Queue,
                                stop_event: Event,
                                log_dir: str = "logs"):
    """Main entry point for the PoolManager process."""
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

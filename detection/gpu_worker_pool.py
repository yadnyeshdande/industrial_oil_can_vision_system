"""
detection/gpu_worker_pool.py  v3.6
====================================
v3.6 changes (Tasks 4 & 5 from the 24/7 hardening pass)
─────────────────────────────────────────────────────────

Task 4 — Aggressive VRAM management in every inference path
  • _infer() wraps the model.predict() call in try/except for
    torch.cuda.OutOfMemoryError.  On OOM: clears cache, returns [].
    The inference thread keeps running — it does NOT crash the pool.
  • _process_batch() has the same OOM guard on the batched GPU call.
  • Both paths delete result tensors in a finally block immediately
    after use, preventing VRAM fragmentation from accumulating over
    long runtimes.
  • torch.cuda.empty_cache() is called:
      – On every OOM exception (immediate recovery)
      – By PoolManager every 5 minutes as a routine defragmentation step

Task 5 — Correct resource-limit self-termination
  OLD (bad): resource limit → self.stop_event.set()
    This called stop on the SHARED event, signalling ALL processes
    including cameras and relay to shut down.  The entire system died.

  NEW (correct): resource limit → sys.exit(1)
    The bloated PoolManager process hard-exits.  The OS immediately
    reclaims all its VRAM, heap, and virtual-memory pages (including
    the WinError 1455 paging-file allocation).  The supervisor detects
    the dead process (via watchdog or liveness check) and restarts only
    the gpu_pool — cameras and relay keep running uninterrupted.

  PoolManager now checks BOTH System RAM and VRAM limits.
  InferenceThreads do NOT sys.exit() — they are threads inside the
  PoolManager process; only the PoolManager decides to exit.
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
from core.resource_monitor import (
    get_process_memory_mb, is_memory_over_limit,
    get_gpu_stats, is_vram_over_limit,
)
from core.shared_frame import SharedFrameReader
from core.boundary_engine import CameraBoundarySet

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary normaliser (unchanged from v3.5)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_boundary_data(data: dict, camera_id: int,
                              relay_mapping=None) -> dict:
    if "boundaries" in data and "pairs" in data:
        return data
    oc_raw = data.get("oil_can", [])
    bh_raw = data.get("bunk_hole", [])
    if len(oc_raw) != len(bh_raw):
        logger.error(
            "[NormBoundary] cam=%d: oil_can count (%d) != bunk_hole count (%d). "
            "%d extra %s zone(s) ignored. Fix camera_%d_boundaries.json.",
            camera_id, len(oc_raw), len(bh_raw),
            abs(len(oc_raw) - len(bh_raw)),
            "oil_can" if len(oc_raw) > len(bh_raw) else "bunk_hole",
            camera_id)

    def _to_boundary(item, idx):
        bid     = item.get("id", f"B{idx}")
        polygon = item.get("polygon", item.get("points", []))
        return {"id": bid, "name": bid, "type": "polygon", "points": polygon, "pair": ""}

    oc_b = [_to_boundary(b, i) for i, b in enumerate(oc_raw)]
    bh_b = [_to_boundary(b, i) for i, b in enumerate(bh_raw)]
    pairs = []
    for i in range(min(len(oc_b), len(bh_b))):
        oc_id = oc_b[i]["id"]; bh_id = bh_b[i]["id"]
        oc_b[i]["pair"] = bh_id; bh_b[i]["pair"] = oc_id
        relay_idx = (relay_mapping[i] if relay_mapping and i < len(relay_mapping)
                     else camera_id * 3 + i)
        pairs.append({"id": i, "name": f"Pair {i+1}",
                      "oil_can_boundary": oc_id, "bunk_hole_boundary": bh_id,
                      "relay_index": relay_idx})
    return {"camera_id": camera_id,
            "boundaries": {"oil_can": oc_b, "bunk_hole": bh_b},
            "pairs": pairs}


# ─────────────────────────────────────────────────────────────────────────────
# Inference worker thread
# ─────────────────────────────────────────────────────────────────────────────

class _InferenceThread(threading.Thread):
    """
    Pulls tasks from _task_q and runs inference through the shared model.
    Shares the YOLO model object and inference_lock with sibling threads.

    OOM policy (Task 4):
        Catches torch.cuda.OutOfMemoryError per inference call.
        On OOM: clears GPU cache, logs the event, returns empty detections.
        The thread CONTINUES running — it does NOT crash the pool process.
        The PoolManager's resource-monitor decides whether to exit.
    """

    def __init__(self, thread_id, cfg, task_q, result_queue, heartbeat_queue,
                 model, inference_lock, stop_event, boundary_sets, boundary_lock):
        super().__init__(name=f"InferThread-{thread_id}", daemon=True)
        self.tid            = thread_id
        self.cfg            = cfg
        self.mcfg           = cfg.model
        self.pcfg           = cfg.gpu_pool
        self.task_q         = task_q
        self.result_queue   = result_queue
        self.hb_q           = heartbeat_queue
        self.model          = model
        self.lock           = inference_lock
        self.stop_event     = stop_event
        self.b_sets         = boundary_sets
        self.b_lock         = boundary_lock
        self._readers:       Dict[int, SharedFrameReader] = {}
        self._fps_acc:       Dict[int, float]             = {}
        self._fps_ts:        Dict[int, float]             = {}
        self._inference_ms:  float                        = 0.0
        self._last_hb:       float                        = time.time()

    def run(self):
        logger.info("[InferThread-%d] started", self.tid)
        while not self.stop_event.is_set():
            tasks = []
            try:
                tasks.append(self.task_q.get(timeout=0.3))
            except queue.Empty:
                continue

            while len(tasks) < 3:
                try:
                    tasks.append(self.task_q.get_nowait())
                except queue.Empty:
                    break

            try:
                if len(tasks) == 1:
                    self._process(tasks[0])
                else:
                    self._process_batch(tasks)
            except Exception as e:
                logger.error("[InferThread-%d] task error: %s", self.tid, e, exc_info=True)

        for r in self._readers.values():
            try:
                r.close()
            except Exception:
                pass
        logger.info("[InferThread-%d] stopped", self.tid)

    # ── Single-frame path ─────────────────────────────────────────────────────

    def _process(self, task: dict):
        cam_id = task.get("camera_id")
        shm    = task.get("shm_name") or task.get("shared_memory_name", "")
        w      = task.get("frame_width",  1280)
        h      = task.get("frame_height", 720)

        if not shm:
            return

        frame = self._read_frame(cam_id, shm, w, h)
        if frame is None:
            return

        t0 = time.time()
        with self.lock:
            detections = self._infer(frame)
        self._inference_ms = (time.time() - t0) * 1000

        del frame   # release numpy array immediately after inference

        pair_results, relay_states, problem_count = \
            self._apply_boundaries(cam_id, None, detections)

        fps = self._update_fps(cam_id)
        det_dicts = [{"class_id": d.class_id, "class_name": d.class_name,
                      "confidence": d.confidence, "bbox": list(d.bbox)}
                     for d in detections]

        msg = DetectionResultMessage(
            source=ProcessSource.GPU_POOL, camera_id=cam_id,
            detections=det_dicts, pair_results=pair_results,
            relay_states=relay_states, frame_data=None,
            inference_time_ms=self._inference_ms, fps=fps,
            problem_count=problem_count, success_rate=0.0,
        ).to_dict()
        try:
            self.result_queue.put_nowait(msg)
        except Exception:
            pass

        self._maybe_heartbeat(cam_id, fps)

    # ── Batch path ────────────────────────────────────────────────────────────

    def _process_batch(self, tasks: list):
        """
        Task 4: batched inference with OOM guard and explicit tensor cleanup.
        """
        import torch

        valid = []
        for task in tasks:
            cam_id = task.get("camera_id")
            shm    = task.get("shm_name") or task.get("shared_memory_name", "")
            w      = task.get("frame_width",  1280)
            h      = task.get("frame_height", 720)
            frame  = self._read_frame(cam_id, shm, w, h)
            if frame is not None:
                valid.append((task, frame))

        if not valid:
            return

        frames = [f for _, f in valid]
        batch_results = None

        # ── GPU call — OOM guarded ────────────────────────────────────────────
        t0 = time.time()
        try:
            with self.lock:
                with torch.no_grad():
                    batch_results = self.model.predict(
                        frames,
                        conf=self.mcfg.confidence,
                        iou=self.mcfg.iou,
                        verbose=False,
                    )
        except torch.cuda.OutOfMemoryError as oom:
            # Task 4: OOM recovery — clear cache, drop this batch, keep running
            logger.error(
                "[InferThread-%d] CUDA OOM in batch of %d frames: %s — "
                "clearing cache, dropping batch",
                self.tid, len(frames), oom)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return   # thread continues; PoolManager decides whether to exit
        except Exception as e:
            logger.error("[InferThread-%d] batch inference error: %s", self.tid, e, exc_info=True)
            return
        finally:
            # Task 4: explicit frame array cleanup — release BEFORE processing results
            try:
                del frames
            except Exception:
                pass

        self._inference_ms = (time.time() - t0) * 1000

        # ── Distribute results ────────────────────────────────────────────────
        try:
            for i, (task, frame) in enumerate(valid):
                cam_id = task.get("camera_id")
                result = batch_results[i]

                dets = []
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id       = int(box.cls[0].item())
                        conf         = float(box.conf[0].item())
                        x1,y1,x2,y2 = box.xyxyn[0].tolist()
                        cls_name     = (self.mcfg.class_names[cls_id]
                                        if cls_id < len(self.mcfg.class_names) else str(cls_id))
                        dets.append(DetectionObject(
                            class_id=cls_id, class_name=cls_name,
                            confidence=conf, bbox=(x1,y1,x2,y2)))

                pair_results, relay_states, problem_count = \
                    self._apply_boundaries(cam_id, frame, dets)

                del frame   # release 2.7 MB numpy array immediately

                fps = self._update_fps(cam_id)
                det_dicts = [{"class_id": d.class_id, "class_name": d.class_name,
                              "confidence": d.confidence, "bbox": list(d.bbox)}
                             for d in dets]
                msg = DetectionResultMessage(
                    source=ProcessSource.GPU_POOL, camera_id=cam_id,
                    detections=det_dicts, pair_results=pair_results,
                    relay_states=relay_states, frame_data=None,
                    inference_time_ms=self._inference_ms, fps=fps,
                    problem_count=problem_count, success_rate=0.0,
                ).to_dict()
                try:
                    self.result_queue.put_nowait(msg)
                except Exception:
                    pass
                self._maybe_heartbeat(cam_id, fps)
        finally:
            # Task 4: explicit batch result cleanup
            if batch_results is not None:
                try:
                    del batch_results
                except Exception:
                    pass

    # ── Core inference ────────────────────────────────────────────────────────

    def _infer(self, frame: np.ndarray) -> List[DetectionObject]:
        """
        Task 4: OOM guard + explicit result tensor cleanup.

        On torch.cuda.OutOfMemoryError:
          • Empties the CUDA cache (reclaims fragmented blocks)
          • Returns [] so the calling code gets valid (empty) output
          • Does NOT raise — the thread continues processing new frames

        On any other exception:
          • Logs and returns [] — prevents one bad frame from killing the thread
        """
        import torch
        results = None
        try:
            with torch.no_grad():
                results = self.model.predict(
                    frame, conf=self.mcfg.confidence,
                    iou=self.mcfg.iou, verbose=False)

            dets = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id       = int(box.cls[0].item())
                    conf         = float(box.conf[0].item())
                    x1,y1,x2,y2 = box.xyxyn[0].tolist()
                    cls_name     = (self.mcfg.class_names[cls_id]
                                    if cls_id < len(self.mcfg.class_names) else str(cls_id))
                    dets.append(DetectionObject(
                        class_id=cls_id, class_name=cls_name,
                        confidence=conf, bbox=(x1,y1,x2,y2)))
            return dets

        except torch.cuda.OutOfMemoryError as oom:
            # Task 4: OOM recovery
            logger.error(
                "[InferThread-%d] CUDA OOM on single frame: %s — "
                "clearing cache, returning empty detections",
                self.tid, oom)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return []

        except Exception as e:
            logger.error("[InferThread-%d] inference error: %s", self.tid, e, exc_info=True)
            return []

        finally:
            # Task 4: release prediction result tensors immediately
            if results is not None:
                try:
                    del results
                except Exception:
                    pass

    # ── Boundary evaluation ───────────────────────────────────────────────────

    def _apply_boundaries(self, cam_id, frame, detections):
        with self.b_lock:
            bset = self.b_sets.get(cam_id)
        if bset is None:
            return [], [], 0
        try:
            # frame may be None when called from single-frame path (frame already deleted)
            if frame is not None:
                frame_h, frame_w = frame.shape[:2]
            else:
                cam_cfg = self.cfg.get_camera(cam_id)
                frame_w = cam_cfg.frame_width if cam_cfg else 1280
                frame_h = cam_cfg.frame_height if cam_cfg else 720

            pair_results = bset.evaluate(
                detections, frame_w, frame_h,
                self.mcfg.oil_can_class_id, self.mcfg.bunk_hole_class_id)
            relay_states  = [p.relay_active for p in pair_results]
            problem_count = sum(1 for p in pair_results if p.relay_active)
            pair_dicts    = [
                {"pair_id": p.pair_id, "pair_name": p.pair_name,
                 "oil_can_present": p.oil_can_present, "bunk_hole_present": p.bunk_hole_present,
                 "status": p.status.value, "relay_index": p.relay_index,
                 "relay_active": p.relay_active}
                for p in pair_results]
            return pair_dicts, relay_states, problem_count
        except Exception as e:
            logger.error("[InferThread-%d] boundary eval error: %s", self.tid, e, exc_info=True)
            return [], [], 0

    # ── Utilities ─────────────────────────────────────────────────────────────

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

    def _update_fps(self, cam_id) -> float:
        now  = time.time()
        prev = self._fps_ts.get(cam_id, now - 1)
        dt   = now - prev
        self._fps_ts[cam_id] = now
        instant  = 1.0 / max(dt, 0.001)
        smoothed = 0.9 * self._fps_acc.get(cam_id, instant) + 0.1 * instant
        self._fps_acc[cam_id] = smoothed
        return round(smoothed, 1)

    def _maybe_heartbeat(self, cam_id, fps):
        now = time.time()
        if now - self._last_hb < self.pcfg.heartbeat_interval_seconds:
            return
        self._last_hb = now
        avg_fps = sum(self._fps_acc.values()) / max(len(self._fps_acc), 1)
        vram    = get_gpu_stats() or {}
        hb = make_heartbeat(
            source=ProcessSource.GPU_POOL, camera_id=None,
            process_name=f"InferThread-{self.tid}", pid=os.getpid(),
            memory_mb=get_process_memory_mb(), fps=avg_fps, status="running",
            extra={"worker_id": self.tid,
                   "vram_mb": round(vram.get("vram_used_mb", 0), 1),
                   "inference_ms": round(self._inference_ms, 1),
                   "shared_model": True})
        try:
            self.hb_q.put_nowait(hb.to_dict())
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Pool Manager
# ─────────────────────────────────────────────────────────────────────────────

class PoolManager:
    """
    Loads YOLO once, shares it across N inference threads.

    Task 5 — Resource-limit self-termination:
      _check_resources() checks both VRAM and System RAM.
      On violation: logs CRITICAL, calls torch.cuda.empty_cache(), then
      sys.exit(1).  This is correct — it lets the OS instantly reclaim
      ALL resources (VRAM, heap, Win32 paging-file allocation) from this
      process.  The supervisor detects the dead process and restarts only
      the gpu_pool — cameras and relay are unaffected.

    Task 4 — Routine VRAM defragmentation:
      Calls torch.cuda.empty_cache() every CACHE_CLEAR_INTERVAL_S (300 s)
      as a proactive step to prevent long-running fragmentation.
    """

    CACHE_CLEAR_INTERVAL_S = 300.0   # routine empty_cache() period

    def __init__(self, cfg, inference_queue, result_queue, heartbeat_queue, stop_event):
        self.cfg             = cfg
        self.pcfg            = cfg.gpu_pool
        self.mcfg            = cfg.model
        self.inference_queue = inference_queue
        self.result_queue    = result_queue
        self.hb_q            = heartbeat_queue
        self.stop_event      = stop_event
        self.pid             = os.getpid()
        self.name            = "GPUPoolManager"

        self._model           = None
        self._inference_lock  = threading.Lock()
        self._boundary_sets:  Dict[int, Optional[CameraBoundarySet]] = {}
        self._boundary_mtimes: Dict[int, float] = {}
        self._boundary_lock   = threading.Lock()
        self._last_boundary_poll = 0.0
        self._threads: List[_InferenceThread] = []
        self._task_q  = queue.Queue(maxsize=6)
        self._last_hb = time.time()
        self._last_resource_check = 0.0
        self._last_cache_clear    = time.time()

    def run(self):
        logger.info("[PoolManager] PID=%d — shared-model pool, %d inference threads",
                    self.pid, self.pcfg.pool_size)

        if not self._load_model():
            logger.critical("[PoolManager] Model load failed — exiting")
            sys.exit(1)

        for cam in self.cfg.cameras:
            self._load_boundaries(cam.id)

        for i in range(self.pcfg.pool_size):
            t = _InferenceThread(
                thread_id=i, cfg=self.cfg, task_q=self._task_q,
                result_queue=self.result_queue, heartbeat_queue=self.hb_q,
                model=self._model, inference_lock=self._inference_lock,
                stop_event=self.stop_event, boundary_sets=self._boundary_sets,
                boundary_lock=self._boundary_lock)
            t.start()
            self._threads.append(t)
            logger.info("[PoolManager] Started InferThread-%d", i)

        while not self.stop_event.is_set():
            try:
                msg = self.inference_queue.get(timeout=0.5)
            except Exception:
                self._poll_boundaries()
                self._maybe_manager_hb()
                self._check_resources()       # Task 5
                self._maybe_clear_cache()     # Task 4
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
                    pass   # drop frame — prefer dropping to blocking
            else:
                try:
                    self._task_q.put_nowait(msg)
                except queue.Full:
                    pass

            self._check_resources()
            self._maybe_clear_cache()

        # Clean shutdown
        logger.info("[PoolManager] Stopping inference threads...")
        for t in self._threads:
            t.join(timeout=5.0)
        self._clear_cuda_cache()
        logger.info("[PoolManager] Exiting cleanly")

    # ── Task 5: resource-limit self-termination ───────────────────────────────

    def _check_resources(self):
        """
        Check VRAM and System RAM.  On violation: clean up and hard-exit.

        sys.exit(1) is correct here — it lets the OS reclaim ALL resources
        from this process instantly.  The supervisor (which owns the stop_event)
        detects the dead process and restarts it cleanly.

        We deliberately do NOT call self.stop_event.set() — that would signal
        the cameras and relay to shut down, taking the whole system offline.
        """
        now = time.time()
        if now - self._last_resource_check < 10.0:
            return
        self._last_resource_check = now

        # Check VRAM
        if is_vram_over_limit(self.pcfg.vram_limit_mb):
            logger.critical(
                "[PoolManager] VRAM limit %.0f MB exceeded — "
                "hard exit so supervisor can restart this process cleanly. "
                "(stop_event NOT set — cameras and relay continue running)",
                self.pcfg.vram_limit_mb)
            self._clear_cuda_cache()
            sys.exit(1)   # Task 5: hard exit, NOT stop_event.set()

        # Check System RAM
        if is_memory_over_limit(self.pcfg.memory_limit_mb):
            logger.critical(
                "[PoolManager] RAM limit %.0f MB exceeded — "
                "hard exit so supervisor can restart cleanly.",
                self.pcfg.memory_limit_mb)
            self._clear_cuda_cache()
            sys.exit(1)   # Task 5: hard exit

    # ── Task 4: routine VRAM defragmentation ─────────────────────────────────

    def _maybe_clear_cache(self):
        """
        Proactively defragment the CUDA memory pool every 5 minutes.
        Prevents long-running VRAM fragmentation from accumulating even
        when no OOM exception has occurred.
        """
        if time.time() - self._last_cache_clear < self.CACHE_CLEAR_INTERVAL_S:
            return
        self._last_cache_clear = time.time()
        self._clear_cuda_cache()
        logger.debug("[PoolManager] Routine CUDA cache cleared")

    def _clear_cuda_cache(self):
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> bool:
        try:
            import torch
            from ultralytics import YOLO

            model_path = Path(self.mcfg.path)
            if not model_path.is_absolute():
                model_path = _ROOT / model_path
            if not model_path.exists():
                logger.error("[PoolManager] Model not found: %s", model_path)
                return False

            torch.backends.cudnn.benchmark = False

            device_str = self.mcfg.device
            if device_str == "cuda" and not torch.cuda.is_available():
                logger.warning("[PoolManager] CUDA unavailable — falling back to CPU")
                device_str = "cpu"

            logger.info("[PoolManager] Loading YOLO on %s", device_str)
            self._model = YOLO(str(model_path))
            self._model.to(torch.device(device_str))

            if getattr(self.mcfg, "use_fp16", False) and device_str == "cuda":
                self._model.model.half()
                logger.info("[PoolManager] FP16 enabled")

            # Warm-up
            cam = self.cfg.cameras[0] if self.cfg.cameras else None
            h = cam.frame_height if cam else 720
            w = cam.frame_width  if cam else 1280
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            self._model.predict(dummy, conf=self.mcfg.confidence,
                                iou=self.mcfg.iou, verbose=False)
            logger.info("[PoolManager] YOLO model loaded and warmed up")
            return True

        except Exception as e:
            logger.critical("[PoolManager] Model load error: %s", e, exc_info=True)
            return False

    # ── Boundary loading / polling ────────────────────────────────────────────

    def _load_boundaries(self, camera_id: int):
        data = self.cfg.load_boundaries(camera_id)
        if not data:
            logger.warning("[PoolManager] No boundary data for cam %d", camera_id)
            with self._boundary_lock:
                self._boundary_sets[camera_id] = None
            return
        try:
            relay_indices = self.cfg.relay.get_relay_indices(camera_id)
        except Exception:
            relay_indices = None
        data = _normalize_boundary_data(data, camera_id, relay_indices)
        bset = CameraBoundarySet(data, strict_mode=getattr(self.mcfg, "strict_boundary_mode", False))
        with self._boundary_lock:
            self._boundary_sets[camera_id] = bset
        logger.info("[PoolManager] Boundaries loaded for cam %d (%d pairs)",
                    camera_id, len(data.get("pairs", [])))

    def _poll_boundaries(self):
        """Reload boundary files if they have changed on disk."""
        now = time.time()
        if now - self._last_boundary_poll < 30.0:
            return
        self._last_boundary_poll = now
        for cam in self.cfg.cameras:
            try:
                path = self.cfg.get_boundary_path(cam.id)
                if path and path.exists():
                    mtime = path.stat().st_mtime
                    if mtime != self._boundary_mtimes.get(cam.id):
                        self._boundary_mtimes[cam.id] = mtime
                        logger.info("[PoolManager] Boundary file changed — reloading cam %d", cam.id)
                        self._load_boundaries(cam.id)
            except Exception:
                pass

    # ── Heartbeat ────────────────────────────────────────────────────────────

    def _maybe_manager_hb(self):
        now = time.time()
        if now - self._last_hb < self.pcfg.heartbeat_interval_seconds:
            return
        self._last_hb = now
        vram = get_gpu_stats() or {}
        hb = make_heartbeat(
            source=ProcessSource.GPU_POOL, camera_id=None,
            process_name=self.name, pid=self.pid,
            memory_mb=get_process_memory_mb(), fps=0.0, status="running",
            extra={"vram_mb": round(vram.get("vram_used_mb", 0), 1),
                   "shared_model": True,
                   "pool_size": self.pcfg.pool_size})
        try:
            self.hb_q.put_nowait(hb.to_dict())
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Process entry point
# ─────────────────────────────────────────────────────────────────────────────

def pool_manager_process_entry(config_path, inference_queue, result_queue,
                                heartbeat_queue, stop_event, log_dir="logs"):
    cfg = VisionSystemConfig(config_path)
    setup_process_logging("gpu_pool", log_dir, cfg.system.log_level,
                          cfg.logging.max_bytes, cfg.logging.backup_count)
    setup_crash_handler("gpu_pool", log_dir)

    def _sig(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT,  _sig)

    manager = PoolManager(cfg, inference_queue, result_queue,
                          heartbeat_queue, stop_event)
    try:
        manager.run()
    except SystemExit:
        raise   # propagate sys.exit() calls — do not catch here
    except Exception as e:
        logger.critical("[gpu_pool] Fatal: %s", e, exc_info=True)
        sys.exit(1)

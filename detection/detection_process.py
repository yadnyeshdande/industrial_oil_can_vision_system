"""
detection/detection_process.py  v2.3
======================================
v2.3 changes (Tasks 4 & 5 from the 24/7 hardening pass)
─────────────────────────────────────────────────────────

Task 4 — OOM handling in _run_inference()
  torch.cuda.OutOfMemoryError is caught per-frame.  On OOM:
    • Calls torch.cuda.empty_cache() immediately
    • Returns [] (empty detections) — frame is silently dropped
    • The detection loop CONTINUES — the process does not crash
  Result tensors are held in a local `results` variable and explicitly
  deleted in a `finally` block so VRAM is reclaimed after every frame.

Task 5 — Correct resource-limit self-termination
  OLD (bad): _check_resources() → self.stop_event.set()
    stop_event is SHARED across processes.  Setting it shuts down the
    cameras and relay too, taking the entire system offline.

  NEW (correct): _check_resources() → sys.exit(1)
    The detection process hard-exits.  The OS instantly reclaims its
    VRAM and heap.  The supervisor detects the dead process (watchdog
    or liveness check) and restarts only this detection process.
    Cameras and relay keep running uninterrupted.

v2.2 fixes (carried forward)
────────────────────────────
  * _relay_states sized from cfg.relay.relay_count (was hardcoded to 3).
  * relay_states guard corrected to relay_count.
  * _normalize_boundary_data validates equal oil_can / bunk_hole counts.
  * Boundary normalisation step added (was missing, causing 0 pairs).
  * Model path resolved relative to _ROOT (was breaking on CWD mismatch).
"""

from __future__ import annotations
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from multiprocessing import Queue
from typing import Dict, List, Optional

import cv2
import numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config_loader import VisionSystemConfig, DetectionConfig, CameraConfig
from core.ipc_schema import (
    DetectionObject, DetectionResultMessage,
    ProcessSource, PairStatus,
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
# Boundary data normaliser (unchanged from v2.2)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_boundary_data(data: dict, camera_id: int,
                              relay_mapping: Optional[List[int]] = None) -> dict:
    """Return boundary data in the nested form CameraBoundarySet expects."""
    if "boundaries" in data and "pairs" in data:
        return data

    oc_raw = data.get("oil_can", [])
    bh_raw = data.get("bunk_hole", [])

    if len(oc_raw) != len(bh_raw):
        logger.error(
            "[NormBoundary] cam=%d: oil_can boundary count (%d) != "
            "bunk_hole boundary count (%d).  The %d extra %s zone(s) will be "
            "ignored and their relays will NEVER fire.  Fix "
            "camera_%d_boundaries.json so both lists have the same length.",
            camera_id, len(oc_raw), len(bh_raw),
            abs(len(oc_raw) - len(bh_raw)),
            "oil_can" if len(oc_raw) > len(bh_raw) else "bunk_hole",
            camera_id)

    def _to_boundary(item: dict, idx: int) -> dict:
        bid     = item.get("id", f"B{idx}")
        polygon = item.get("polygon", item.get("points", []))
        return {"id": bid, "name": bid, "type": "polygon", "points": polygon, "pair": ""}

    oc_boundaries = [_to_boundary(b, i) for i, b in enumerate(oc_raw)]
    bh_boundaries = [_to_boundary(b, i) for i, b in enumerate(bh_raw)]

    pairs: List[dict] = []
    for i in range(min(len(oc_boundaries), len(bh_boundaries))):
        oc_id = oc_boundaries[i]["id"]
        bh_id = bh_boundaries[i]["id"]
        oc_boundaries[i]["pair"] = bh_id
        bh_boundaries[i]["pair"] = oc_id
        relay_idx = (relay_mapping[i] if relay_mapping and i < len(relay_mapping)
                     else camera_id * 3 + i)
        pairs.append({
            "id":                 i,
            "name":               f"Pair {i + 1}",
            "oil_can_boundary":   oc_id,
            "bunk_hole_boundary": bh_id,
            "relay_index":        relay_idx,
        })

    return {
        "camera_id":  camera_id,
        "boundaries": {"oil_can": oc_boundaries, "bunk_hole": bh_boundaries},
        "pairs":      pairs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Detection worker
# ─────────────────────────────────────────────────────────────────────────────

class DetectionWorker:
    """
    Core detection loop for one camera.
    Reads from shared memory, runs YOLO, applies boundary logic, sends results.
    """

    def __init__(self,
                 camera_id:       int,
                 cfg:             VisionSystemConfig,
                 result_queue:    Queue,
                 heartbeat_queue: Queue,
                 stop_event:      mp.Event):
        self.camera_id       = camera_id
        self.cfg             = cfg
        self.dcfg:           DetectionConfig = cfg.detection
        self.cam_cfg:        CameraConfig    = cfg.get_camera(camera_id)
        self.result_queue    = result_queue
        self.heartbeat_queue = heartbeat_queue
        self.stop_event      = stop_event

        self.pid  = os.getpid()
        self.name = f"Detection_{camera_id}"

        # Stats
        self._fps              = 0.0
        self._inference_ms     = 0.0
        self._total_detections = 0
        self._problem_count    = 0
        self._ok_count         = 0
        self._frame_count      = 0
        self._last_fps_time    = time.time()
        self._last_heartbeat   = time.time()
        self._last_vram_check  = time.time()
        self._start_time       = time.time()

        self._model  = None
        self._device = None

        self._reader:       Optional[SharedFrameReader]  = None
        self._boundary_set: Optional[CameraBoundarySet]  = None

        # v2.2: sized from relay_count, not hardcoded 3
        self._relay_states: List[bool] = [False] * cfg.relay.relay_count

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def run(self):
        logger.info("[%s] PID=%d starting", self.name, self.pid)

        if not self._load_model():
            logger.critical("[%s] Model load failed, exiting", self.name)
            return

        # Load and normalise boundaries
        boundary_data = self.cfg.load_boundaries(self.camera_id)
        if boundary_data:
            try:
                relay_indices = self.cfg.relay.get_relay_indices(self.camera_id)
            except Exception:
                relay_indices = None
            boundary_data = _normalize_boundary_data(
                boundary_data, self.camera_id, relay_indices)
            self._boundary_set = CameraBoundarySet(
                boundary_data, strict_mode=self.dcfg.strict_boundary_mode)
        else:
            logger.warning("[%s] No boundary data, running without boundaries", self.name)

        # Connect to shared memory
        self._reader = SharedFrameReader(
            name=self.cam_cfg.shared_memory_name,
            width=self.cam_cfg.frame_width,
            height=self.cam_cfg.frame_height,
        )
        if not self._reader.connect(timeout=30.0):
            logger.critical("[%s] Cannot connect to shared memory, exiting", self.name)
            return

        interval = 1.0 / max(self.dcfg.fps_limit, 1)
        logger.info("[%s] Detection loop started (fps_limit=%d)", self.name, self.dcfg.fps_limit)

        while not self.stop_event.is_set():
            loop_start = time.time()

            frame, frame_idx = self._reader.read()
            if frame is None:
                time.sleep(0.005)
                self._maybe_heartbeat()
                continue

            # Run inference
            try:
                detections = self._run_inference(frame)
            except Exception as e:
                logger.error("[%s] Inference error: %s", self.name, e, exc_info=True)
                self._send_error("InferenceError", str(e), traceback.format_exc())
                time.sleep(0.1)
                continue

            # Boundary pairing
            pair_results = []
            if self._boundary_set:
                try:
                    pair_results = self._boundary_set.evaluate(
                        detections,
                        self.cam_cfg.frame_width,
                        self.cam_cfg.frame_height,
                        self.dcfg.oil_can_class_id,
                        self.dcfg.bunk_hole_class_id,
                    )
                except Exception as e:
                    logger.error("[%s] Boundary eval error: %s", self.name, e)

            # Update stats
            self._frame_count      += 1
            self._total_detections += len(detections)
            problems = sum(1 for p in pair_results if p.relay_active)
            if problems:
                self._problem_count += 1
            else:
                self._ok_count += 1

            relay_count = self.cfg.relay.relay_count
            relay_states = [False] * relay_count
            for pr in pair_results:
                ri = pr.relay_index
                if 0 <= ri < relay_count:
                    relay_states[ri] = pr.relay_active
            self._relay_states = relay_states

            now     = time.time()
            elapsed = now - self._last_fps_time
            if elapsed >= 2.0:
                self._fps          = self._frame_count / elapsed
                self._frame_count  = 0
                self._last_fps_time = now

            total_evals  = self._problem_count + self._ok_count
            success_rate = (self._ok_count / total_evals * 100) if total_evals > 0 else 100.0

            frame_jpeg = self._encode_frame(frame)

            result_msg = DetectionResultMessage(
                source=ProcessSource.DETECTION,
                camera_id=self.camera_id,
                detections=[d.to_dict() for d in detections],
                pair_results=[p.to_dict() for p in pair_results],
                inference_time_ms=self._inference_ms,
                fps=self._fps,
                total_detections=self._total_detections,
                problem_count=self._problem_count,
                success_rate=success_rate,
                frame_shape=(self.cam_cfg.frame_height, self.cam_cfg.frame_width, 3),
                frame_data=frame_jpeg,
            )

            try:
                self.result_queue.put_nowait(result_msg.to_dict())
            except Exception:
                pass

            self._maybe_heartbeat()

            # Resource check every 10 s
            if now - self._last_vram_check >= 10.0:
                self._check_resources()
                self._last_vram_check = now

            elapsed_loop = time.time() - loop_start
            sleep_t = interval - elapsed_loop
            if sleep_t > 0:
                time.sleep(sleep_t)

        self._cleanup()
        logger.info("[%s] Detection process exiting cleanly", self.name)

    # ─── YOLO model ───────────────────────────────────────────────────────────

    def _load_model(self) -> bool:
        try:
            import torch
            from ultralytics import YOLO

            model_path = Path(self.dcfg.model_path)
            if not model_path.is_absolute():
                model_path = _ROOT / model_path
            if not model_path.exists():
                alt = Path(self.cfg.model.path) if hasattr(self.cfg, "model") else model_path
                if not alt.is_absolute():
                    alt = _ROOT / alt
                if alt.exists():
                    model_path = alt
                else:
                    logger.error("[%s] Model not found at %s or %s", self.name, model_path, alt)
                    return False

            torch.backends.cudnn.benchmark = False

            device_str = self.dcfg.device
            if device_str == "cuda" and not torch.cuda.is_available():
                logger.warning("[%s] CUDA not available, falling back to CPU", self.name)
                device_str = "cpu"

            self._device = torch.device(device_str)
            logger.info("[%s] Loading model on device: %s", self.name, device_str)

            self._model = YOLO(str(model_path))
            self._model.to(self._device)

            if self.dcfg.use_fp16 and device_str == "cuda":
                self._model.model.half()
                logger.info("[%s] FP16 enabled", self.name)

            dummy = np.zeros(
                (self.cam_cfg.frame_height, self.cam_cfg.frame_width, 3), dtype=np.uint8)
            self._model.predict(
                dummy, conf=self.dcfg.confidence_threshold,
                iou=self.dcfg.iou_threshold, verbose=False)
            logger.info("[%s] YOLO model loaded and warmed up", self.name)
            return True

        except Exception as e:
            logger.critical("[%s] Model load error: %s", self.name, e, exc_info=True)
            return False

    # ─── Inference ────────────────────────────────────────────────────────────

    def _run_inference(self, frame: np.ndarray) -> List[DetectionObject]:
        """
        Task 4: OOM guard + explicit tensor cleanup.

        On torch.cuda.OutOfMemoryError:
          • Empties the CUDA cache immediately
          • Returns [] — frame is dropped, loop continues
          • Does NOT raise — the process stays alive

        Result tensors are held in `results` and deleted in finally.
        """
        import torch

        results = None
        t0      = time.time()

        try:
            with torch.no_grad():
                results = self._model.predict(
                    frame,
                    conf=self.dcfg.confidence_threshold,
                    iou=self.dcfg.iou_threshold,
                    verbose=False,
                )
            self._inference_ms = (time.time() - t0) * 1000

            detections: List[DetectionObject] = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id       = int(box.cls[0].item())
                    conf         = float(box.conf[0].item())
                    x1,y1,x2,y2 = box.xyxyn[0].tolist()
                    cls_name     = (self.dcfg.class_names[cls_id]
                                    if cls_id < len(self.dcfg.class_names) else str(cls_id))
                    detections.append(DetectionObject(
                        class_id=cls_id, class_name=cls_name,
                        confidence=conf, bbox=(x1, y1, x2, y2)))
            return detections

        except torch.cuda.OutOfMemoryError as oom:
            # Task 4: OOM recovery — clear cache, return empty, keep running
            logger.error(
                "[%s] CUDA OOM: %s — clearing cache, dropping frame",
                self.name, oom)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self._inference_ms = (time.time() - t0) * 1000
            return []

        finally:
            # Task 4: release result tensors regardless of success or OOM
            if results is not None:
                try:
                    del results
                except Exception:
                    pass

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buf.tobytes()
        except Exception:
            return None

    def _maybe_heartbeat(self):
        now = time.time()
        if now - self._last_heartbeat >= 2.0:
            vram_stats = get_gpu_stats()
            hb = make_heartbeat(
                source=ProcessSource.DETECTION,
                camera_id=self.camera_id,
                process_name=self.name,
                pid=self.pid,
                memory_mb=get_process_memory_mb(),
                fps=self._fps,
                status="running",
                extra={
                    "inference_ms":      round(self._inference_ms, 1),
                    "total_detections":  self._total_detections,
                    "problem_count":     self._problem_count,
                    "vram_mb":           round(vram_stats.get("vram_used_mb", 0), 1),
                    "gpu_temp":          vram_stats.get("temperature_c", 0),
                },
            )
            try:
                self.heartbeat_queue.put_nowait(hb.to_dict())
            except Exception:
                pass
            self._last_heartbeat = now

    def _check_resources(self):
        """
        Task 5: resource-limit self-termination via sys.exit(1).

        Does NOT call self.stop_event.set() — that would shut down the
        whole system.  Instead, hard-exits this process so the OS
        reclaims its resources instantly and the supervisor can restart
        only this detection process.
        """
        if is_memory_over_limit(self.dcfg.memory_limit_mb):
            logger.critical(
                "[%s] RAM limit %.0f MB exceeded — "
                "hard exit so supervisor can restart this process cleanly. "
                "(stop_event NOT set — cameras and relay continue running)",
                self.name, self.dcfg.memory_limit_mb)
            self._send_error(
                "MemoryLimitExceeded",
                f"RAM exceeded {self.dcfg.memory_limit_mb} MB",
                severity="critical")
            self._cleanup()
            sys.exit(1)   # Task 5: hard exit, NOT stop_event.set()

        if is_vram_over_limit(self.dcfg.vram_limit_mb):
            logger.critical(
                "[%s] VRAM limit %.0f MB exceeded — hard exit.",
                self.name, self.dcfg.vram_limit_mb)
            self._send_error(
                "VRAMLimitExceeded",
                f"VRAM exceeded {self.dcfg.vram_limit_mb} MB",
                severity="critical")
            self._cleanup()
            sys.exit(1)   # Task 5: hard exit

    def _send_error(self, error_type: str, error_msg: str,
                    tb: str = "", severity: str = "error"):
        err = make_error(
            source=ProcessSource.DETECTION,
            camera_id=self.camera_id,
            error_type=error_type,
            error_msg=error_msg,
            traceback=tb,
            severity=severity,
        )
        try:
            self.heartbeat_queue.put_nowait(err.to_dict())
        except Exception:
            pass

    def _cleanup(self):
        """Release CUDA cache and shared memory before exiting."""
        try:
            import torch
            torch.cuda.empty_cache()
            logger.info("[%s] CUDA cache cleared", self.name)
        except Exception:
            pass
        if self._reader:
            try:
                self._reader.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Process entry point
# ─────────────────────────────────────────────────────────────────────────────

def detection_process_entry(camera_id: int,
                             config_path: str,
                             result_queue: Queue,
                             heartbeat_queue: Queue,
                             stop_event: mp.Event,
                             log_dir: str = "logs"):
    cfg     = VisionSystemConfig(config_path)
    cam_cfg = cfg.get_camera(camera_id)
    if cam_cfg is None:
        raise ValueError(f"Camera {camera_id} not found in config")

    pname = f"detection_{camera_id}"
    setup_process_logging(pname, log_dir, cfg.system.log_level,
                          cfg.logging.max_bytes, cfg.logging.backup_count)
    setup_crash_handler(pname, log_dir)

    def _sig(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT,  _sig)

    worker = DetectionWorker(camera_id, cfg, result_queue, heartbeat_queue, stop_event)
    try:
        worker.run()
    except SystemExit:
        raise   # propagate sys.exit() — do not swallow
    except Exception as e:
        logger.critical("[%s] Fatal: %s", pname, e, exc_info=True)
        sys.exit(1)

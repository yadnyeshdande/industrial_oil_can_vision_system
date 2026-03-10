"""
detection/detection_process.py
================================
Isolated detection process per camera.
- Loads YOLO model once onto GPU
- FP16 inference with torch.no_grad()
- Reads frames from shared memory
- Applies full boundary pairing logic
- Sends DetectionResultMessage to relay and GUI queues
- Monitors RAM and VRAM
- Sends heartbeat every 2 seconds
- Self-terminates on resource violations
"""

from __future__ import annotations
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from io import BytesIO
from pathlib import Path
from multiprocessing import Queue
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config_loader import VisionSystemConfig, DetectionConfig, CameraConfig
from core.ipc_schema import (
    DetectionObject, DetectionResultMessage,
    ProcessSource, PairStatus,
    make_heartbeat, make_error
)
from core.logging_setup import setup_process_logging, setup_crash_handler
from core.resource_monitor import (
    get_process_memory_mb, is_memory_over_limit,
    get_gpu_stats, is_vram_over_limit
)
from core.shared_frame import SharedFrameReader
from core.boundary_engine import CameraBoundarySet

logger = logging.getLogger(__name__)


class DetectionWorker:
    """
    Core detection loop for one camera.
    Reads from shared memory, runs YOLO, applies boundary logic, sends results.
    """

    def __init__(self,
                 camera_id: int,
                 cfg: VisionSystemConfig,
                 result_queue: Queue,       # → relay + GUI
                 heartbeat_queue: Queue,
                 stop_event: mp.Event):
        self.camera_id = camera_id
        self.cfg = cfg
        self.dcfg: DetectionConfig = cfg.detection
        self.cam_cfg: CameraConfig = cfg.get_camera(camera_id)
        self.result_queue = result_queue
        self.heartbeat_queue = heartbeat_queue
        self.stop_event = stop_event

        self.pid = os.getpid()
        self.name = f"Detection_{camera_id}"

        # Stats
        self._fps = 0.0
        self._inference_ms = 0.0
        self._total_detections = 0
        self._problem_count = 0
        self._ok_count = 0
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._last_heartbeat = time.time()
        self._last_vram_check = time.time()
        self._start_time = time.time()

        # YOLO model (loaded in run())
        self._model = None
        self._device = None

        # Shared frame reader
        self._reader: Optional[SharedFrameReader] = None

        # Boundary set
        self._boundary_set: Optional[CameraBoundarySet] = None

        # Relay states cache
        self._relay_states: List[bool] = [False, False, False]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self):
        logger.info("[%s] PID=%d starting", self.name, self.pid)

        # Load YOLO
        if not self._load_model():
            logger.critical("[%s] Model load failed, exiting", self.name)
            return

        # Load boundaries
        boundary_data = self.cfg.load_boundaries(self.camera_id)
        if boundary_data:
            self._boundary_set = CameraBoundarySet(
                boundary_data,
                strict_mode=self.dcfg.strict_boundary_mode
            )
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

            # Read latest frame
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
            self._frame_count += 1
            self._total_detections += len(detections)
            problems = sum(1 for p in pair_results if p.relay_active)
            if problems:
                self._problem_count += 1
            else:
                self._ok_count += 1

            # Relay states
            relay_states = [False, False, False]
            for pr in pair_results:
                ri = pr.relay_index
                if 0 <= ri < 3:
                    relay_states[ri] = pr.relay_active
            self._relay_states = relay_states

            # Update FPS
            now = time.time()
            elapsed = now - self._last_fps_time
            if elapsed >= 2.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now

            # Success rate
            total_evals = self._problem_count + self._ok_count
            success_rate = (self._ok_count / total_evals * 100) if total_evals > 0 else 100.0

            # Encode frame for GUI (JPEG, low quality for speed)
            frame_jpeg = self._encode_frame(frame)

            # Build result message
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

            # Send to result queue (relay + GUI subscribe to this)
            try:
                self.result_queue.put_nowait(result_msg.to_dict())
            except Exception:
                pass  # Drop if full

            # Heartbeat
            self._maybe_heartbeat()

            # Resource checks every 10s
            if now - self._last_vram_check >= 10.0:
                self._check_resources()
                self._last_vram_check = now

            # FPS governor
            elapsed_loop = time.time() - loop_start
            sleep_t = interval - elapsed_loop
            if sleep_t > 0:
                time.sleep(sleep_t)

        # Cleanup
        self._cleanup()
        logger.info("[%s] Detection process exiting cleanly", self.name)

    # ------------------------------------------------------------------
    # YOLO model
    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        try:
            import torch
            from ultralytics import YOLO

            model_path = Path(self.dcfg.model_path)
            if not model_path.exists():
                logger.error("[%s] Model not found: %s", self.name, model_path)
                return False

            # Stability: disable cudnn benchmark
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

            # Warm-up inference
            dummy = np.zeros(
                (self.cam_cfg.frame_height, self.cam_cfg.frame_width, 3), dtype=np.uint8
            )
            self._model.predict(
                dummy,
                conf=self.dcfg.confidence_threshold,
                iou=self.dcfg.iou_threshold,
                verbose=False,
            )
            logger.info("[%s] YOLO model loaded and warmed up", self.name)
            return True

        except Exception as e:
            logger.critical("[%s] Model load error: %s", self.name, e, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self, frame: np.ndarray) -> List[DetectionObject]:
        import torch

        t0 = time.time()
        with torch.no_grad():
            results = self._model.predict(
                frame,
                conf=self.dcfg.confidence_threshold,
                iou=self.dcfg.iou_threshold,
                verbose=False,
            )
        self._inference_ms = (time.time() - t0) * 1000

        detections: List[DetectionObject] = []
        h, w = frame.shape[:2]

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                # Normalised xyxy
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                cls_name = self.dcfg.class_names[cls_id] if cls_id < len(self.dcfg.class_names) else str(cls_id)
                detections.append(DetectionObject(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                ))

        return detections

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
                    "inference_ms": round(self._inference_ms, 1),
                    "total_detections": self._total_detections,
                    "problem_count": self._problem_count,
                    "vram_mb": round(vram_stats.get("vram_used_mb", 0), 1),
                    "gpu_temp": vram_stats.get("temperature_c", 0),
                },
            )
            try:
                self.heartbeat_queue.put_nowait(hb.to_dict())
            except Exception:
                pass
            self._last_heartbeat = now

    def _check_resources(self):
        if is_memory_over_limit(self.dcfg.memory_limit_mb):
            logger.critical("[%s] RAM limit exceeded, exiting", self.name)
            self._send_error("MemoryLimitExceeded",
                             f"RAM exceeded {self.dcfg.memory_limit_mb}MB", severity="critical")
            self.stop_event.set()
            return

        if is_vram_over_limit(self.dcfg.vram_limit_mb):
            logger.critical("[%s] VRAM limit exceeded, exiting", self.name)
            self._send_error("VRAMLimitExceeded",
                             f"VRAM exceeded {self.dcfg.vram_limit_mb}MB", severity="critical")
            self.stop_event.set()

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
        try:
            import torch
            torch.cuda.empty_cache()
            logger.info("[%s] CUDA cache cleared", self.name)
        except Exception:
            pass
        if self._reader:
            self._reader.close()


# ---------------------------------------------------------------------------
# Process entry point
# ---------------------------------------------------------------------------

def detection_process_entry(camera_id: int,
                              config_path: str,
                              result_queue: Queue,
                              heartbeat_queue: Queue,
                              stop_event: mp.Event,
                              log_dir: str = "logs"):
    from core.config_loader import VisionSystemConfig

    cfg = VisionSystemConfig(config_path)
    process_name = f"detection_{camera_id}"

    setup_process_logging(
        process_name=process_name,
        log_dir=log_dir,
        log_level=cfg.system.log_level,
        max_bytes=cfg.logging.max_bytes,
        backup_count=cfg.logging.backup_count,
    )
    setup_crash_handler(process_name, log_dir)

    def _sigterm(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGTERM, _sigterm)

    worker = DetectionWorker(
        camera_id=camera_id,
        cfg=cfg,
        result_queue=result_queue,
        heartbeat_queue=heartbeat_queue,
        stop_event=stop_event,
    )
    try:
        worker.run()
    except Exception as e:
        logger.critical("[detection_%d] Fatal: %s", camera_id, e, exc_info=True)
        sys.exit(1)

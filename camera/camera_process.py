"""
camera/camera_process.py  v2.1
==============================
Isolated camera process.
v2.1: sends InferenceRequestMessage to shared inference_queue (GPU pool)
      instead of per-camera detection queue.
All v1 behaviour preserved: RTSP capture, reconnect, FPS governor, heartbeat.
"""
from __future__ import annotations
import logging, multiprocessing as mp, os, signal, sys, time
from pathlib import Path
from multiprocessing import Queue
from typing import Optional
import cv2, numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config_loader import VisionSystemConfig, CameraConfig
from core.ipc_schema import (ProcessSource, MessageType,
    FrameReadyMessage, make_heartbeat, make_error, make_inference_request)
from core.logging_setup import setup_process_logging, setup_crash_handler
from core.resource_monitor import get_process_memory_mb, is_memory_over_limit
from core.shared_frame import SharedFrameWriter

logger = logging.getLogger(__name__)


class CameraWorker:
    def __init__(self, cam_cfg: CameraConfig, inference_queue: Queue,
                 heartbeat_queue: Queue, stop_event: mp.Event,
                 preview_mode: mp.Value):
        self.cfg = cam_cfg
        self.inference_queue = inference_queue
        self.heartbeat_queue = heartbeat_queue
        self.stop_event = stop_event
        self.preview_mode = preview_mode   # shared Value: 1=preview only, 0=send to pool
        self.pid = os.getpid()
        self.camera_id = cam_cfg.id
        self.name = f"Camera_{cam_cfg.id}"
        self._cap: Optional[cv2.VideoCapture] = None
        self._shm_writer: Optional[SharedFrameWriter] = None
        self._frame_count = 0; self._fps = 0.0
        self._last_fps_time = time.time(); self._last_heartbeat = time.time()
        self._reconnect_delay = cam_cfg.reconnect_base_delay
        self._reconnect_attempts = 0

    def run(self):
        logger.info("[%s] PID=%d starting", self.name, self.pid)
        self._shm_writer = SharedFrameWriter(
            name=self.cfg.shared_memory_name,
            width=self.cfg.frame_width, height=self.cfg.frame_height)
        interval = 1.0 / max(self.cfg.fps_limit, 1)

        while not self.stop_event.is_set():
            if not self._connect():
                if self.stop_event.is_set(): break
                self._wait_reconnect(); continue

            logger.info("[%s] Connected to RTSP", self.name)
            self._reconnect_delay = self.cfg.reconnect_base_delay
            self._reconnect_attempts = 0
            consecutive_failures = 0

            while not self.stop_event.is_set():
                loop_start = time.time()
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        logger.error("[%s] Too many read failures, reconnecting", self.name)
                        break
                    time.sleep(0.05); continue
                consecutive_failures = 0

                if frame.shape[1] != self.cfg.frame_width or frame.shape[0] != self.cfg.frame_height:
                    frame = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height))

                self._shm_writer.write(frame)
                self._frame_count += 1

                # Only send inference requests when NOT in preview mode
                if not self.preview_mode.value:
                    req = make_inference_request(
                        camera_id=self.camera_id,
                        shm_name=self.cfg.shared_memory_name,
                        frame_shape=(self.cfg.frame_height, self.cfg.frame_width, 3),
                        frame_index=self._frame_count,
                    )
                    try:
                        self.inference_queue.put_nowait(req.to_dict())
                    except Exception:
                        pass

                now = time.time()
                if now - self._last_fps_time >= 2.0:
                    self._fps = self._frame_count / (now - self._last_fps_time)
                    self._frame_count = 0; self._last_fps_time = now

                if now - self._last_heartbeat >= 2.0:
                    self._send_heartbeat(); self._last_heartbeat = now

                if is_memory_over_limit(self.cfg.memory_limit_mb):
                    logger.critical("[%s] Memory limit exceeded, exiting", self.name)
                    self.stop_event.set(); break

                sleep_t = interval - (time.time() - loop_start)
                if sleep_t > 0: time.sleep(sleep_t)

            self._release_cap()

        self._release_cap()
        if self._shm_writer: self._shm_writer.close()
        logger.info("[%s] Exiting cleanly", self.name)

    def _connect(self) -> bool:
        self._release_cap()
        logger.info("[%s] Connecting: %s", self.name, self.cfg.rtsp_url)
        cap = cv2.VideoCapture(self.cfg.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cfg.buffer_size)
        time.sleep(0.5)
        if not cap.isOpened():
            cap.release(); self._reconnect_attempts += 1
            self._send_error("ConnectionFailed", f"RTSP failed: {self.cfg.rtsp_url}"); return False
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release(); return False
        self._cap = cap; return True

    def _release_cap(self):
        if self._cap:
            try: self._cap.release()
            except Exception: pass
            self._cap = None

    def _wait_reconnect(self):
        delay = min(self._reconnect_delay, self.cfg.reconnect_max_delay)
        logger.info("[%s] Reconnecting in %.1fs", self.name, delay)
        deadline = time.time() + delay
        while time.time() < deadline and not self.stop_event.is_set(): time.sleep(0.1)
        self._reconnect_delay = min(self._reconnect_delay * 2, self.cfg.reconnect_max_delay)

    def _send_heartbeat(self):
        hb = make_heartbeat(ProcessSource.CAMERA, self.camera_id, self.name,
            self.pid, get_process_memory_mb(), self._fps, extra={
                "reconnect_attempts": self._reconnect_attempts,
                "preview_mode": bool(self.preview_mode.value)})
        try: self.heartbeat_queue.put_nowait(hb.to_dict())
        except Exception: pass

    def _send_error(self, etype, emsg, severity="error"):
        err = make_error(ProcessSource.CAMERA, self.camera_id, etype, emsg, severity=severity)
        try: self.heartbeat_queue.put_nowait(err.to_dict())
        except Exception: pass


def camera_process_entry(camera_id, config_path, inference_queue, heartbeat_queue,
                          stop_event, preview_mode, log_dir="logs"):
    from core.config_loader import VisionSystemConfig
    cfg = VisionSystemConfig(config_path)
    cam_cfg = cfg.get_camera(camera_id)
    if cam_cfg is None: raise ValueError(f"Camera {camera_id} not found")
    pname = f"camera_{camera_id}"
    setup_process_logging(pname, log_dir, cfg.system.log_level,
                          cfg.logging.max_bytes, cfg.logging.backup_count)
    setup_crash_handler(pname, log_dir)
    def _sig(s,f): stop_event.set()
    signal.signal(signal.SIGTERM, _sig)
    worker = CameraWorker(cam_cfg, inference_queue, heartbeat_queue, stop_event, preview_mode)
    try: worker.run()
    except Exception as e:
        logger.critical("[camera_%d] Fatal: %s", camera_id, e, exc_info=True); sys.exit(1)

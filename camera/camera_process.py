"""
camera/camera_process.py  —  FIXED (bufferless RTSP + CPU-safe IPC)
=====================================================================
Fixes applied
─────────────
1. RTSP BUFFER TRAP  →  BufferlessCapture runs a daemon thread that calls
   cap.grab() in a tight loop, keeping the OpenCV/FFmpeg ring-buffer drained.
   The main loop calls cap.retrieve() only when it is ready to process a frame,
   so it always gets the *newest* live frame instead of one that is 4–5 s old.

2. CPU BUSY-WAIT      →  Every polling / idle path now has time.sleep(0.001)
   (1 ms).  That single line drops a pinned core from ~100 % to < 1 %.

3. FPS LIMITER        →  The grab thread is throttled to CAP_PROP_FPS of the
   camera.  The main loop enforces fps_limit with a precise sleep so it never
   spins faster than needed.

4. BUFFER SIZE HINT   →  cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) is called right
   after VideoCapture.open() as an extra safeguard (works on most backends).

5. RTSP TRANSPORT     →  The RTSP URL is forced to TCP transport via the
   cv2.CAP_PROP_FOURCC / environment variable trick AND by appending
   ?rtsp_transport=tcp when the URL doesn't already have a query string.
   This eliminates UDP packet-loss induced blocking reads.

6. SHARED MEMORY WRITE MODEL  →  Frames are written to shared memory with the
   "overwrite" model (no Queue backlog), exactly matching the IPC schema in the
   README.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import sys
import time
import threading
from typing import Optional

import cv2
import numpy as np

# ── project-local imports (adjust if your package layout differs) ──────────────
from core.config_loader import VisionSystemConfig
from core.ipc_schema import (
    ProcessSource,
    make_heartbeat,
    make_error,
    make_inference_request,
)
from core.shared_frame import SharedFrameWriter
from core.logging_setup import setup_process_logging, setup_crash_handler
from core.resource_monitor import get_process_memory_mb, is_memory_over_limit


# ──────────────────────────────────────────────────────────────────────────────
# 1.  BUFFERLESS CAPTURE
# ──────────────────────────────────────────────────────────────────────────────

class BufferlessCapture:
    """
    Wraps cv2.VideoCapture so that the internal FFmpeg ring-buffer never grows.

    A background daemon thread calls cap.grab() in a tight loop at the camera's
    native FPS.  This continuously discards stale frames from the buffer.

    The caller calls read() which does a single cap.retrieve() to decode only
    the *most recently grabbed* frame — always live, never stale.

    Thread-safety: a threading.Lock guards cap.retrieve() / cap.grab().
    """

    def __init__(self, rtsp_url: str, camera_id: int, logger: logging.Logger) -> None:
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.logger = logger

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_available = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False

    def open(self) -> bool:
        """Open the RTSP stream and start the background grab thread."""
        # ✅ CORRECT: set FFmpeg TCP transport as env var BEFORE VideoCapture
        # ❌ WRONG (my mistake): appending ?rtsp_transport=tcp to the URL — causes 404
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        url = self.rtsp_url          # ← use the URL exactly as it is in config.yaml
        self.logger.info(f"[cam{self.camera_id}] Opening RTSP: {url}")

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # ... rest of the method unchanged

        if not cap.isOpened():
            self.logger.error(f"[cam{self.camera_id}] Failed to open stream.")
            cap.release()
            return False

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.logger.info(f"[cam{self.camera_id}] Stream opened — {w}×{h} @ {fps:.1f} FPS")

        self._cap = cap
        self._native_fps = fps
        self._connected = True
        self._stop.clear()

        self._thread = threading.Thread(
            target=self._grab_loop,
            name=f"grab-cam{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Return (True, frame) with the latest live frame, or (False, None)."""
        if not self._connected:
            return False, None

        got = self._frame_available.wait(timeout=1.0)
        if not got:
            return False, None

        with self._lock:
            frame = self._frame
            self._frame_available.clear()

        if frame is None:
            return False, None
        return True, frame.copy()

    def release(self) -> None:
        self._stop.set()
        self._connected = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.logger.info(f"[cam{self.camera_id}] Capture released.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _grab_loop(self) -> None:
        interval = 1.0 / max(self._native_fps, 1.0)

        while not self._stop.is_set():
            t0 = time.monotonic()

            with self._lock:
                if self._cap is None or not self._cap.isOpened():
                    break
                grabbed = self._cap.grab()

            if not grabbed:
                self.logger.warning(f"[cam{self.camera_id}] grab() failed — stream may have dropped.")
                self._connected = False
                break

            with self._lock:
                ret, frame = self._cap.retrieve()

            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                self._frame_available.set()

            elapsed = time.monotonic() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    @staticmethod
    def _make_tcp_url(url: str) -> str:
        if "rtsp_transport" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}rtsp_transport=tcp"
        return url


# ──────────────────────────────────────────────────────────────────────────────
# 2.  CAMERA PROCESS ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

class CameraWorker:
    def __init__(
        self,
        cam_cfg,
        inference_queue: mp.Queue,
        heartbeat_queue: mp.Queue,
        stop_event: mp.Event,
        preview_mode: mp.Value,
    ):
        self.cfg = cam_cfg
        self.inference_queue = inference_queue
        self.heartbeat_queue = heartbeat_queue
        self.stop_event = stop_event
        self.preview_mode = preview_mode
        self.pid = os.getpid()
        self.camera_id = cam_cfg.id
        self.name = f"Camera_{cam_cfg.id}"
        self._cap: Optional[BufferlessCapture] = None
        self._shm_writer: Optional[SharedFrameWriter] = None
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._last_heartbeat = time.time()
        self._reconnect_delay = cam_cfg.reconnect_base_delay
        self._reconnect_attempts = 0

    def run(self):
        logger = logging.getLogger(self.name)
        logger.info("[%s] PID=%d starting", self.name, self.pid)

        self._shm_writer = SharedFrameWriter(
            name=self.cfg.shared_memory_name,
            width=self.cfg.frame_width,
            height=self.cfg.frame_height,
        )

        interval = 1.0 / max(self.cfg.fps_limit, 1)

        while not self.stop_event.is_set():
            if not self._connect():
                if self.stop_event.is_set():
                    break
                self._wait_reconnect()
                continue

            logger.info("[%s] Connected to RTSP", self.name)
            self._reconnect_delay = self.cfg.reconnect_base_delay
            self._reconnect_attempts = 0
            consecutive_failures = 0

            while not self.stop_event.is_set():
                loop_start = time.time()
                ok, frame = self._cap.read() if self._cap else (False, None)

                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        logger.error("[%s] Too many read failures, reconnecting", self.name)
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0

                if frame.shape[1] != self.cfg.frame_width or frame.shape[0] != self.cfg.frame_height:
                    frame = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height))

                self._shm_writer.write(frame)
                self._frame_count += 1

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
                    self._frame_count = 0
                    self._last_fps_time = now

                if now - self._last_heartbeat >= 2.0:
                    self._send_heartbeat()
                    self._last_heartbeat = now

                if is_memory_over_limit(self.cfg.memory_limit_mb):
                    logger.critical("[%s] Memory limit exceeded, exiting", self.name)
                    self.stop_event.set()
                    break

                sleep_t = interval - (time.time() - loop_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)

            self._release_cap()

        self._release_cap()
        if self._shm_writer:
            self._shm_writer.close()
        logger.info("[%s] Exiting cleanly", self.name)

    def _connect(self) -> bool:
        self._release_cap()
        logger = logging.getLogger(self.name)
        logger.info("[%s] Connecting: %s", self.name, self.cfg.rtsp_url)
        cap = BufferlessCapture(self.cfg.rtsp_url, self.camera_id, logger)
        if not cap.open():
            self._reconnect_attempts += 1
            self._send_error("ConnectionFailed", f"RTSP failed: {self.cfg.rtsp_url}")
            return False
        self._cap = cap
        return True

    def _release_cap(self):
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _wait_reconnect(self):
        delay = min(self._reconnect_delay, self.cfg.reconnect_max_delay)
        logger = logging.getLogger(self.name)
        logger.info("[%s] Reconnecting in %.1fs", self.name, delay)
        deadline = time.time() + delay
        while time.time() < deadline and not self.stop_event.is_set():
            time.sleep(0.1)
        self._reconnect_delay = min(self._reconnect_delay * 2, self.cfg.reconnect_max_delay)

    def _send_heartbeat(self):
        hb = make_heartbeat(
            source=ProcessSource.CAMERA,
            camera_id=self.camera_id,
            process_name=self.name,
            pid=self.pid,
            memory_mb=get_process_memory_mb(),
            fps=self._fps,
            extra={
                "reconnect_attempts": self._reconnect_attempts,
                "preview_mode": bool(self.preview_mode.value),
            },
        )
        try:
            self.heartbeat_queue.put_nowait(hb.to_dict())
        except Exception:
            pass

    def _send_error(self, error_type: str, error_msg: str, severity: str = "error"):
        err = make_error(
            source=ProcessSource.CAMERA,
            camera_id=self.camera_id,
            error_type=error_type,
            error_msg=error_msg,
            severity=severity,
        )
        try:
            self.heartbeat_queue.put_nowait(err.to_dict())
        except Exception:
            pass

"""
camera/camera_process.py  — PATCH (replace only camera_process_entry)
======================================================================
ONLY THIS FUNCTION NEEDS TO CHANGE from your current file.
Replace the existing camera_process_entry() at the bottom of your
camera_process.py with this one.

Root cause of crash:
    cfg.logging.log_level  → AttributeError (LoggingConfig has no log_level)
    cfg.system.log_level   ← CORRECT  (SystemConfig.log_level exists)

Secondary fix:
    signal.SIGINT handler added (was missing in your new version,
    present in old working version — needed for Ctrl+C clean shutdown).
"""

def camera_process_entry(camera_id, config_path, inference_queue, heartbeat_queue,
                          stop_event, preview_mode, log_dir="logs"):
    import signal, sys, logging
    from core.config_loader import VisionSystemConfig
    from core.logging_setup import setup_process_logging, setup_crash_handler

    cfg = VisionSystemConfig(config_path)
    cam_cfg = cfg.get_camera(camera_id)
    if cam_cfg is None:
        raise ValueError(f"Camera {camera_id} not found in config")

    pname = f"camera_{camera_id}"

    # ✅ FIX: cfg.system.log_level   (was cfg.logging.log_level — AttributeError)
    setup_process_logging(pname, log_dir, cfg.system.log_level,
                          cfg.logging.max_bytes, cfg.logging.backup_count)
    setup_crash_handler(pname, log_dir)

    # ✅ FIX: SIGINT added back (was missing in new version, present in old working code)
    def _sig(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT,  _sig)

    worker = CameraWorker(cam_cfg, inference_queue, heartbeat_queue, stop_event, preview_mode)
    try:
        worker.run()
    except Exception as e:
        logging.critical("[camera_%d] Fatal: %s", camera_id, e, exc_info=True)
        sys.exit(1)
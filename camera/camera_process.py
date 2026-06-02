"""
camera/camera_process.py  v1.3
================================
v1.3 changes (Task 5 from the 24/7 hardening pass)
────────────────────────────────────────────────────

Task 5 — Correct resource-limit self-termination
  OLD (bad): is_memory_over_limit → self.stop_event.set()
    stop_event is shared.  Setting it shuts down detection and relay too,
    taking the entire system offline just because one camera process grew
    too large.

  NEW (correct): is_memory_over_limit → _cleanup() → sys.exit(1)
    The camera process hard-exits.  The OS reclaims its memory instantly.
    The supervisor detects the dead process and restarts only this camera.
    All other processes — detection, relay, GUI — keep running.

  _cleanup_local() centralises resource release (RTSP capture + shared
  memory writer) so it can be called from both the normal exit path and
  the emergency resource-limit exit path.

All other logic is unchanged from the v1.2 bufferless / CPU-safe fix.
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


# ─────────────────────────────────────────────────────────────────────────────
# Bufferless RTSP capture (unchanged from v1.2)
# ─────────────────────────────────────────────────────────────────────────────

class BufferlessCapture:
    """
    Wraps cv2.VideoCapture so the internal FFmpeg ring-buffer never grows.

    A background daemon thread calls cap.grab() continuously, discarding
    stale frames.  The main loop calls read() which does a single
    cap.retrieve() to decode only the most-recently-grabbed frame.
    """

    def __init__(self, rtsp_url: str, camera_id: int,
                 logger: logging.Logger) -> None:
        self.rtsp_url  = rtsp_url
        self.camera_id = camera_id
        self.logger    = logger

        self._cap:              Optional[cv2.VideoCapture] = None
        self._lock              = threading.Lock()
        self._frame:            Optional[np.ndarray]       = None
        self._frame_available   = threading.Event()
        self._stop              = threading.Event()
        self._thread:           Optional[threading.Thread] = None
        self._connected         = False

    def open(self) -> bool:
        # Set TCP transport via env var BEFORE VideoCapture — not via URL suffix
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        url = self.rtsp_url
        self.logger.info("[cam%d] Opening RTSP: %s", self.camera_id, url)

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.logger.error("[cam%d] Failed to open stream.", self.camera_id)
            cap.release()
            return False

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.logger.info("[cam%d] Stream opened — %dx%d @ %.1f FPS",
                         self.camera_id, w, h, fps)

        self._cap          = cap
        self._native_fps   = fps
        self._connected    = True
        self._stop.clear()

        self._thread = threading.Thread(
            target=self._grab_loop,
            name=f"grab-cam{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
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

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _grab_loop(self) -> None:
        """
        Drain the OpenCV/FFmpeg ring-buffer without sleeping.
        cap.grab() blocks on the network when the buffer is empty —
        that IS the natural pacing.  Lock is held only for retrieve().
        """
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._connected = False
                break

            # grab() — do NOT hold lock here (blocks ~40 ms on network I/O)
            grabbed = self._cap.grab()
            if not grabbed:
                self.logger.warning("[cam%d] grab() failed — stream dropped.",
                                    self.camera_id)
                self._connected = False
                break

            # retrieve() — hold lock only for this brief memory operation (~1 ms)
            with self._lock:
                ret, frame = self._cap.retrieve()
                if ret and frame is not None:
                    self._frame = frame
                    self._frame_available.set()


# ─────────────────────────────────────────────────────────────────────────────
# Camera worker
# ─────────────────────────────────────────────────────────────────────────────

class CameraWorker:
    # Replace with:
    def __init__(self, cam_cfg, inference_queue, heartbeat_queue,
                 stop_event, preview_mode, throttle_fps_value=None):
        self.cfg             = cam_cfg
        self.inference_queue = inference_queue
        self.heartbeat_queue = heartbeat_queue
        self.stop_event      = stop_event
        self.preview_mode    = preview_mode
        self.pid             = os.getpid()
        self.camera_id       = cam_cfg.id
        self.name            = f"Camera_{cam_cfg.id}"

        self._cap:                Optional[BufferlessCapture] = None
        self._shm_writer:         Optional[SharedFrameWriter] = None
        self._frame_count         = 0
        self._fps                 = 0.0
        self._last_fps_time       = time.time()
        self._last_heartbeat      = time.time()
        self._reconnect_delay     = cam_cfg.reconnect_base_delay
        self._reconnect_attempts  = 0
        # Add immediately after:
        self._throttle_fps_value = throttle_fps_value   # mp.Value('d') from supervisor

    def run(self):
        log = logging.getLogger(self.name)
        log.info("[%s] PID=%d starting", self.name, self.pid)

        self._shm_writer = SharedFrameWriter(
            name=self.cfg.shared_memory_name,
            width=self.cfg.frame_width,
            height=self.cfg.frame_height,
        )
        # Find and DELETE this line (it's before the outer while loop):
        # interval = 1.0 / max(self.cfg.fps_limit, 1)

        while not self.stop_event.is_set():
            if not self._connect():
                if self.stop_event.is_set():
                    break
                self._wait_reconnect()
                continue

            log.info("[%s] Connected to RTSP", self.name)
            self._reconnect_delay    = self.cfg.reconnect_base_delay
            self._reconnect_attempts = 0
            consecutive_failures     = 0

            while not self.stop_event.is_set():
                loop_start = time.time()
                ok, frame  = self._cap.read() if self._cap else (False, None)

                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        log.error("[%s] Too many read failures — reconnecting", self.name)
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0

                if (frame.shape[1] != self.cfg.frame_width
                        or frame.shape[0] != self.cfg.frame_height):
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
                    self._fps             = self._frame_count / (now - self._last_fps_time)
                    self._frame_count     = 0
                    self._last_fps_time   = now

                if now - self._last_heartbeat >= 2.0:
                    self._send_heartbeat()
                    self._last_heartbeat  = now

                # ── Task 5: hard exit on memory violation ─────────────────────
                # Do NOT call stop_event.set() — that shuts down the whole system.
                # sys.exit(1) lets the OS reclaim this process's memory instantly
                # and allows the supervisor to restart only this camera process.
                if is_memory_over_limit(self.cfg.memory_limit_mb):
                    log.critical(
                        "[%s] RAM limit %.0f MB exceeded — "
                        "hard exit so supervisor can restart this process cleanly. "
                        "(stop_event NOT set — detection and relay continue running)",
                        self.name, self.cfg.memory_limit_mb)
                    self._send_error("MemoryLimitExceeded",
                                     f"RAM exceeded {self.cfg.memory_limit_mb} MB",
                                     severity="critical")
                    self._cleanup_local()
                    sys.exit(1)   # Task 5: hard exit, NOT stop_event.set()

                # Task 5: dynamic interval — re-read throttle Value every frame.
                # 0.0 = no throttle; use config fps_limit.
                # >0.0 = thermal cap active; enforce it by sleeping longer.
                _throttle = (self._throttle_fps_value.value
                             if self._throttle_fps_value is not None else 0.0)
                _active_fps = _throttle if _throttle > 0.0 else self.cfg.fps_limit
                _interval   = 1.0 / max(_active_fps, 1)
                sleep_t = _interval - (time.time() - loop_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)
            self._release_cap()

        self._cleanup_local()
        log.info("[%s] Exiting cleanly", self.name)

    # ── Connection management ─────────────────────────────────────────────────

    def _connect(self) -> bool:
        self._release_cap()
        log = logging.getLogger(self.name)
        log.info("[%s] Connecting: %s", self.name, self.cfg.rtsp_url)
        cap = BufferlessCapture(self.cfg.rtsp_url, self.camera_id, log)
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
        log   = logging.getLogger(self.name)
        log.info("[%s] Reconnecting in %.1fs", self.name, delay)
        deadline = time.time() + delay
        while time.time() < deadline and not self.stop_event.is_set():
            time.sleep(0.1)
        self._reconnect_delay = min(
            self._reconnect_delay * 2, self.cfg.reconnect_max_delay)

    # ── Task 5: centralised local cleanup ────────────────────────────────────

    def _cleanup_local(self):
        """
        Release local resources (RTSP capture + shared memory writer).
        Called from both the normal exit path and the emergency sys.exit() path.
        """
        self._release_cap()
        if self._shm_writer:
            try:
                self._shm_writer.close()
            except Exception:
                pass
            self._shm_writer = None

    # ── IPC helpers ───────────────────────────────────────────────────────────

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
                "preview_mode":       bool(self.preview_mode.value),
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


# ─────────────────────────────────────────────────────────────────────────────
# Process entry point
# ─────────────────────────────────────────────────────────────────────────────

# Replace with:
def camera_process_entry(camera_id, config_path, inference_queue, heartbeat_queue,
                              stop_event, preview_mode,
                              throttle_fps_value=None, log_dir="logs"):
    cfg     = VisionSystemConfig(config_path)
    cam_cfg = cfg.get_camera(camera_id)
    if cam_cfg is None:
        raise ValueError(f"Camera {camera_id} not found in config")

    pname = f"camera_{camera_id}"
    setup_process_logging(pname, log_dir, cfg.system.log_level,
                          cfg.logging.max_bytes, cfg.logging.backup_count)
    setup_crash_handler(pname, log_dir)

    def _sig(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT,  _sig)

    # Replace with:
    worker = CameraWorker(cam_cfg, inference_queue, heartbeat_queue,
                           stop_event, preview_mode,
                           throttle_fps_value=throttle_fps_value)   # Task 5: pass throttle_fps_value to CameraWorker
    try:
        worker.run()
    except SystemExit:
        raise   # propagate sys.exit() — do not swallow
    except Exception as e:
        logging.critical("[%s] Fatal: %s", pname, e, exc_info=True)
        sys.exit(1)

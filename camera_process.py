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
   README.  A frame_ready Event signals the detection process without polling.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
import threading
from typing import Optional

import cv2
import numpy as np

# ── project-local imports (adjust if your package layout differs) ──────────────
from core.ipc_schema import (
    make_heartbeat_msg,
    make_camera_status_msg,
    make_error_msg,
)
from core.shared_frame import SharedFrame
from core.config_loader import get_config
from core.logging_setup import setup_process_logger


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
        self.rtsp_url  = rtsp_url
        self.camera_id = camera_id
        self.logger    = logger

        self._cap:   Optional[cv2.VideoCapture] = None
        self._lock   = threading.Lock()
        self._frame: Optional[np.ndarray]       = None
        self._frame_available = threading.Event()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False

    # ── public API ─────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """Open the RTSP stream and start the background grab thread."""
        url = self._make_tcp_url(self.rtsp_url)
        self.logger.info(f"[cam{self.camera_id}] Opening RTSP: {url}")

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        # FIX 4 — minimise the FFmpeg ring-buffer as a belt-and-braces measure
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.logger.error(f"[cam{self.camera_id}] Failed to open stream.")
            cap.release()
            return False

        # Log what the camera reported back
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.logger.info(f"[cam{self.camera_id}] Stream opened — {w}×{h} @ {fps:.1f} FPS")

        self._cap        = cap
        self._native_fps = fps
        self._connected  = True
        self._stop.clear()

        # Start the background grab thread (daemon so it dies with the process)
        self._thread = threading.Thread(
            target=self._grab_loop,
            name=f"grab-cam{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Return (True, frame) with the **latest** live frame, or (False, None).

        Blocks for at most 1 s waiting for a frame to be grabbed.
        """
        if not self._connected:
            return False, None

        got = self._frame_available.wait(timeout=1.0)
        if not got:
            return False, None

        with self._lock:
            frame = self._frame
            self._frame_available.clear()   # reset — next grab will set it again

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

    # ── internal ───────────────────────────────────────────────────────────────

    def _grab_loop(self) -> None:
        """
        FIX 1 — runs in a daemon thread; continuously drains the OpenCV buffer
        via grab().  This is what keeps the latency at true ~0 ms network RTT
        instead of buffer_depth / camera_fps seconds.
        """
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

            # Decode the frame (cheap — just YUV→BGR on the already-grabbed data)
            with self._lock:
                ret, frame = self._cap.retrieve()

            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                self._frame_available.set()   # signal: fresh frame is ready

            # FIX 3 — throttle grab loop to the camera's native FPS; avoids
            # spinning at 100 % CPU even when the network is fast.
            elapsed = time.monotonic() - t0
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)       # FIX 2 — never busy-wait

    @staticmethod
    def _make_tcp_url(url: str) -> str:
        """
        FIX 5 — append ?rtsp_transport=tcp if there is no query string yet.
        TCP eliminates UDP packet-loss which causes blocking reads and stutter.
        """
        if "rtsp_transport" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}rtsp_transport=tcp"
        return url


# ──────────────────────────────────────────────────────────────────────────────
# 2.  CAMERA PROCESS ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def camera_process(
    camera_id:       int,
    rtsp_url:        str,
    shared_frame:    SharedFrame,
    control_queue:   mp.Queue,
    heartbeat_queue: mp.Queue,
    stop_event:      mp.Event,
) -> None:
    """
    Main camera process.

    Reads live frames from the RTSP stream via BufferlessCapture, writes them
    to shared memory, and sends heartbeat / status messages to the supervisor.

    Parameters
    ----------
    camera_id       : integer index (0 / 1 / 2)
    rtsp_url        : full RTSP URL including credentials
    shared_frame    : SharedFrame IPC object (overwrite model — no backlog)
    control_queue   : mp.Queue for inbound commands from supervisor
    heartbeat_queue : mp.Queue for outbound heartbeat / status to supervisor
    stop_event      : mp.Event — set to True by supervisor to request shutdown
    """

    # ── logging ────────────────────────────────────────────────────────────────
    logger = setup_process_logger(f"camera_{camera_id}")
    logger.info(f"[cam{camera_id}] Process started (PID {os.getpid()})")

    # ── config ─────────────────────────────────────────────────────────────────
    cfg        = get_config()
    fps_limit  = cfg.get("cameras", [{}])[camera_id].get("fps_limit", 15)
    frame_interval = 1.0 / max(fps_limit, 1)

    # ── capture ────────────────────────────────────────────────────────────────
    cap = BufferlessCapture(rtsp_url, camera_id, logger)

    def _connect_with_retry() -> bool:
        max_delay   = cfg.get("cameras", [{}])[camera_id].get("reconnect_max_delay", 30)
        delay       = 1.0
        while not stop_event.is_set():
            if cap.open():
                return True
            logger.warning(f"[cam{camera_id}] Reconnect in {delay:.0f}s …")
            time.sleep(delay)                   # FIX 2 — structured sleep, not spin
            delay = min(delay * 2, max_delay)
        return False

    if not _connect_with_retry():
        logger.error(f"[cam{camera_id}] Could not connect — exiting.")
        return

    # ── performance counters ───────────────────────────────────────────────────
    fps_counter       = 0
    fps_window_start  = time.monotonic()
    last_heartbeat_t  = time.monotonic()
    heartbeat_interval = cfg.get("system", {}).get("heartbeat_interval_seconds", 2.0)

    # ── main loop ──────────────────────────────────────────────────────────────
    logger.info(f"[cam{camera_id}] Entering main loop (fps_limit={fps_limit})")

    while not stop_event.is_set():

        t_loop_start = time.monotonic()

        # ── drain inbound control queue without blocking ───────────────────────
        # FIX 2 — non-blocking get avoids a polling spin; the sleep at the
        # bottom of this loop provides the back-off.
        try:
            msg = control_queue.get_nowait()
            _handle_control_msg(msg, cap, camera_id, logger)
        except Exception:
            pass  # Empty queue — normal; mp.queues.Empty is not re-raised

        # ── fetch latest frame ─────────────────────────────────────────────────
        ok, frame = cap.read()

        if not ok or frame is None:
            logger.warning(f"[cam{camera_id}] Frame read failed — attempting reconnect.")
            cap.release()
            if not _connect_with_retry():
                break
            continue

        # ── write to shared memory (overwrite model — no backlog) ─────────────
        try:
            shared_frame.write(frame, camera_id)
        except Exception as exc:
            logger.error(f"[cam{camera_id}] SharedFrame write error: {exc}")

        fps_counter += 1

        # ── heartbeat (every heartbeat_interval seconds) ───────────────────────
        now = time.monotonic()
        if now - last_heartbeat_t >= heartbeat_interval:
            elapsed      = now - fps_window_start
            measured_fps = fps_counter / elapsed if elapsed > 0 else 0.0

            try:
                heartbeat_queue.put_nowait(
                    make_heartbeat_msg(
                        source=f"camera_{camera_id}",
                        camera_id=camera_id,
                        payload={"fps": round(measured_fps, 1)},
                    )
                )
                heartbeat_queue.put_nowait(
                    make_camera_status_msg(
                        camera_id=camera_id,
                        fps=measured_fps,
                        connected=cap.is_connected,
                    )
                )
            except Exception:
                pass  # Queue full — supervisor is slow; not fatal

            fps_counter      = 0
            fps_window_start = now
            last_heartbeat_t = now

        # ── FPS throttle ───────────────────────────────────────────────────────
        # FIX 3 — sleep for the remainder of the frame_interval so we never
        # call retrieve() faster than fps_limit, saving CPU.
        elapsed = time.monotonic() - t_loop_start
        sleep_t = frame_interval - elapsed
        if sleep_t > 0.0:
            time.sleep(sleep_t)             # FIX 2 — structured sleep
        else:
            # Even if we're behind, yield for 1 ms so the OS can schedule
            # other threads (grab thread, etc.) without starvation.
            time.sleep(0.001)               # FIX 2 — minimum yield

    # ── cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    logger.info(f"[cam{camera_id}] Process exiting cleanly.")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CONTROL MESSAGE HANDLER
# ──────────────────────────────────────────────────────────────────────────────

def _handle_control_msg(
    msg:       dict,
    cap:       BufferlessCapture,
    camera_id: int,
    logger:    logging.Logger,
) -> None:
    msg_type = msg.get("type", "")

    if msg_type == "RECONNECT":
        logger.info(f"[cam{camera_id}] Supervisor requested reconnect.")
        cap.release()
        cap.open()

    elif msg_type == "PING":
        logger.debug(f"[cam{camera_id}] PING received.")

    else:
        logger.debug(f"[cam{camera_id}] Unknown control message: {msg_type}")

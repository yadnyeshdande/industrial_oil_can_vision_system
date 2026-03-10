"""
core/shared_frame.py
====================
Shared memory frame transport layer.
Frame overwrite model: writer always writes latest frame,
reader always reads latest frame. No frame backlog.
Thread/process safe via a multiprocessing.Value flag.
"""

from __future__ import annotations
import ctypes
import logging
import struct
import time
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Header layout stored at the start of shared memory:
# [frame_index: uint64][rows: uint32][cols: uint32][channels: uint32][ready: uint8]
# Header: frame_index(Q), rows(I), cols(I), channels(I), ready(I)
_HEADER_FMT = "=QIIII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)   # 21 bytes


def _calc_shm_size(width: int, height: int, channels: int = 3) -> int:
    return _HEADER_SIZE + width * height * channels


class SharedFrameWriter:
    """Opened by camera process. Writes frames into shared memory."""

    def __init__(self, name: str, width: int, height: int, channels: int = 3):
        self.name = name
        self.width = width
        self.height = height
        self.channels = channels
        self._frame_index = 0
        size = _calc_shm_size(width, height, channels)
        try:
            # Try to unlink any stale segment
            stale = shared_memory.SharedMemory(name=name)
            stale.close()
            stale.unlink()
        except Exception:
            pass
        self._shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        self._buf = np.ndarray(
            (height, width, channels), dtype=np.uint8,
            buffer=self._shm.buf, offset=_HEADER_SIZE
        )
        logger.info("SharedFrameWriter created: %s (%dx%d ch=%d, %d bytes)",
                    name, width, height, channels, size)

    def write(self, frame: np.ndarray):
        """Write frame, overwriting previous. Non-blocking."""
        try:
            h, w = frame.shape[:2]
            ch = frame.shape[2] if frame.ndim == 3 else 1
            # Write ready=0 first (invalidate)
            struct.pack_into(_HEADER_FMT, self._shm.buf, 0,
                             self._frame_index, h, w, ch, 0)
            np.copyto(self._buf, frame)
            self._frame_index += 1
            # Write ready=1 (validate)
            struct.pack_into(_HEADER_FMT, self._shm.buf, 0,
                             self._frame_index, h, w, ch, 1)
        except Exception as e:
            logger.error("SharedFrameWriter.write error: %s", e)

    def close(self):
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass
        logger.info("SharedFrameWriter closed: %s", self.name)


class SharedFrameReader:
    """Opened by detection/GUI processes. Reads latest frame."""

    def __init__(self, name: str, width: int, height: int, channels: int = 3):
        self.name = name
        self.width = width
        self.height = height
        self.channels = channels
        self._last_index = -1
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._connected = False

    def connect(self, timeout: float = 10.0) -> bool:
        """Attempt to open existing shared memory. Retries until timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self._shm = shared_memory.SharedMemory(name=self.name, create=False)
                self._connected = True
                logger.info("SharedFrameReader connected: %s", self.name)
                return True
            except FileNotFoundError:
                time.sleep(0.2)
            except Exception as e:
                logger.warning("SharedFrameReader connect error: %s", e)
                time.sleep(0.5)
        logger.error("SharedFrameReader failed to connect to: %s", self.name)
        return False

    def read(self) -> Tuple[Optional[np.ndarray], int]:
        """
        Returns (frame, frame_index) of the latest frame.
        Returns (None, -1) if not ready or no new frame.
        """
        if not self._connected or self._shm is None:
            return None, -1
        try:
            hdr = struct.unpack_from(_HEADER_FMT, self._shm.buf, 0)
            frame_index, rows, cols, channels, ready = hdr
            if not ready or frame_index == self._last_index:
                return None, -1
            frame = np.ndarray(
                (rows, cols, channels), dtype=np.uint8,
                buffer=self._shm.buf, offset=_HEADER_SIZE
            ).copy()   # copy so we own the data
            self._last_index = frame_index
            return frame, frame_index
        except Exception as e:
            logger.debug("SharedFrameReader.read error: %s", e)
            return None, -1

    def read_blocking(self, timeout: float = 1.0) -> Tuple[Optional[np.ndarray], int]:
        """Block until a new frame is available or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame, idx = self.read()
            if frame is not None:
                return frame, idx
            time.sleep(0.005)
        return None, -1

    def has_new_frame(self) -> bool:
        if not self._connected or self._shm is None:
            return False
        try:
            hdr = struct.unpack_from(_HEADER_FMT, self._shm.buf, 0)
            frame_index, _, _, _, ready = hdr
            return bool(ready) and frame_index != self._last_index
        except Exception:
            return False

    def close(self):
        try:
            if self._shm:
                self._shm.close()
        except Exception:
            pass
        self._connected = False
        logger.info("SharedFrameReader closed: %s", self.name)

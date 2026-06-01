"""
core/safe_queue.py
==================
Deadlock-safe multiprocessing queue bridge for Windows industrial environments.

ROOT CAUSE of the zombie / frozen-supervisor state
───────────────────────────────────────────────────
On Windows, multiprocessing.Queue uses a named Win32 mutex (_rlock) internally
to serialise concurrent writers.  When a worker process is killed mid-write
(e.g. during a CUDA OOM crash that triggers WinError 1455), it can die while
holding that mutex.

Any subsequent call to queue.get() in the supervisor process will block
FOREVER waiting for the dead process to release the mutex.  Because the
supervisor calls _drain_hb() inside its main supervision_loop(), the entire
loop freezes — including the heartbeat-timeout check AND the 24-hour daily
restart timer.

SOLUTION — Bridge-Thread Pattern
──────────────────────────────────
SafeQueueReader spawns a single daemon thread that owns ALL blocking reads
from the (potentially dangerous) multiprocessing.Queue.  Results are
transferred to a standard threading.queue.Queue that lives entirely inside
the supervisor process.  The supervisor main loop reads ONLY from that
in-process queue, which can never deadlock.

  ┌──────────────────┐         ┌──────────────────────────┐
  │ worker processes │──mp.Q──▶│  bridge daemon thread    │──▶ local_q (safe)
  └──────────────────┘ (risky) └──────────────────────────┘    (main loop reads)

If the mp.Queue deadlocks (_rlock stuck), only the bridge daemon thread gets
stuck.  Daemon threads do NOT prevent process exit, and they do NOT block the
supervisor's main loop, watchdog thread, or daily-restart thread.

Recovery after process restart
────────────────────────────────
Call replace_queue(new_q) after recreating a queue that may be corrupted.
This abandons the stuck daemon thread (harmless — daemon) and spawns a fresh
one on the clean queue.  Call is_thread_alive() to detect a stuck bridge.

ALSO PROVIDED
─────────────
  safe_put()      — put_nowait() wrapper; silently drops on full/error.
                    NEVER blocks.  Safe to call from any process/thread.
  safe_get_nowait — get_nowait() wrapper; returns None on empty/error.
"""
from __future__ import annotations

import logging
import queue as _tqueue
import threading
import time
from multiprocessing import Queue as MPQueue
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helpers — safe wrappers for mp.Queue put / get
# ─────────────────────────────────────────────────────────────────────────────

def safe_put(q: MPQueue, item: Any, *, label: str = "") -> bool:
    """
    Non-blocking put.  Returns True on success, False on silent drop.

    Uses put_nowait() so it NEVER blocks even if the reader process is dead
    or the queue is full.  Swallows ALL exceptions — a dropped message is
    always preferable to a hung process in an industrial loop.
    """
    try:
        q.put_nowait(item)
        return True
    except Exception:
        return False


def safe_get_nowait(q: MPQueue) -> Optional[Any]:
    """Non-blocking get.  Returns None on empty or any error."""
    try:
        return q.get_nowait()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SafeQueueReader — bridge-thread pattern
# ─────────────────────────────────────────────────────────────────────────────

class SafeQueueReader:
    """
    Bridges a multiprocessing.Queue to a thread-safe threading.queue.Queue.

    The bridge daemon thread performs all potentially-blocking reads from the
    mp.Queue.  The supervisor main loop reads only from self.local_q, which
    is a standard threading.queue.Queue and is immune to mutex deadlocks.

    Parameters
    ──────────
    mp_queue     : The multiprocessing.Queue to read from.
    name         : Thread name (appears in logs and debuggers).
    maxsize      : Maximum items in the local in-process queue.
                   Old items are dropped (not the newest) when full.
    read_timeout : Seconds to wait in mp_queue.get() before looping.
                   This keeps the bridge thread responsive to self._stop.
                   On a deadlocked queue this call never returns, but that
                   only affects this daemon thread — not the main loop.

    Usage
    ─────
        reader = SafeQueueReader(heartbeat_mp_queue, name="HBReader")

        # In supervisor main loop — never deadlocks:
        for msg in reader.drain(max_items=150):
            handle(msg)

        # Or single-item poll:
        msg = reader.get_nowait()   # returns None immediately if empty

    Thread health
    ─────────────
        reader.is_thread_alive()   → False if bridge thread crashed (rare)
        reader.respawn_thread()    → restart bridge thread (call if not alive)

    Queue replacement after worker restart
    ──────────────────────────────────────
        reader.replace_queue(new_mp_queue)
        # Abandons old (possibly stuck) bridge thread; starts fresh one.
    """

    def __init__(self,
                 mp_queue:     MPQueue,
                 name:         str   = "SafeQueueReader",
                 maxsize:      int   = 1000,
                 read_timeout: float = 0.5):
        self._mp_q        = mp_queue
        self._name        = name
        self._read_timeout = read_timeout
        self.local_q: _tqueue.Queue = _tqueue.Queue(maxsize=maxsize)
        self._stop        = threading.Event()
        self._thread      = self._spawn_thread()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_nowait(self) -> Optional[Any]:
        """
        Non-blocking read from the safe local queue.
        Returns None immediately if the queue is empty.
        NEVER deadlocks — local_q is a pure Python in-process queue.
        """
        try:
            return self.local_q.get_nowait()
        except _tqueue.Empty:
            return None

    def drain(self, max_items: int = 200) -> Generator[Any, None, None]:
        """
        Yield up to max_items messages without any blocking.
        Stops early if the local queue is empty.
        """
        for _ in range(max_items):
            msg = self.get_nowait()
            if msg is None:
                return
            yield msg

    def is_thread_alive(self) -> bool:
        """Return True if the bridge daemon thread is still running."""
        return self._thread.is_alive()

    def respawn_thread(self):
        """
        Restart the bridge thread.  Call if is_thread_alive() returns False.
        The old stuck/dead thread is abandoned (daemon — harmless).
        """
        logger.warning("[%s] bridge thread is dead — respawning", self._name)
        self._stop = threading.Event()
        self._thread = self._spawn_thread()

    def replace_queue(self, new_mp_queue: MPQueue):
        """
        Swap in a fresh multiprocessing.Queue.

        The old bridge thread may be permanently stuck on a deadlocked queue.
        We signal it to stop (it may not respond if blocked), abandon it
        (daemon — collected on process exit), and start a new bridge thread
        on the clean queue.
        """
        logger.info("[%s] replacing mp.Queue — old bridge thread abandoned if stuck", self._name)
        self._stop.set()                  # signal old thread (may not respond)
        self._mp_q   = new_mp_queue
        self._stop   = threading.Event()  # fresh event for new thread
        self._thread = self._spawn_thread()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _spawn_thread(self) -> threading.Thread:
        t = threading.Thread(
            target=self._run,
            name=self._name,
            daemon=True,   # CRITICAL: daemon=True so supervisor can exit freely
        )
        t.start()
        logger.debug("[%s] bridge thread started (tid=%d)", self._name, t.ident or 0)
        return t

    def _run(self):
        """
        Bridge loop — lives in a daemon thread.

        On a healthy system: mp_queue.get(timeout) wakes up quickly, we copy
        the message to local_q, and loop.

        On a deadlocked queue (_rlock abandoned): mp_queue.get(timeout) blocks
        forever.  Only THIS thread hangs — the main loop is unaffected because
        it never touches mp_queue directly.
        """
        while not self._stop.is_set():
            try:
                # Timeout lets us check self._stop periodically.
                # If the queue is deadlocked, this call blocks indefinitely —
                # but that is acceptable because this is a daemon thread.
                msg = self._mp_q.get(timeout=self._read_timeout)
            except Exception:
                # queue.Empty (normal timeout), OSError, EOFError (pipe broken
                # when a writer process died), etc.  All are safe to ignore.
                continue

            try:
                self.local_q.put_nowait(msg)
            except _tqueue.Full:
                # Local queue full — drop oldest to make room for newest.
                # This keeps the supervisor responsive under message bursts.
                try:
                    self.local_q.get_nowait()
                    self.local_q.put_nowait(msg)
                except Exception:
                    pass

        logger.debug("[%s] bridge thread exiting", self._name)

"""
gui/gui_process.py  v1.2
==========================
v1.2 changes (Task 5 from the 24/7 hardening pass)
────────────────────────────────────────────────────

Task 5 — Correct resource-limit self-termination
  OLD (bad): is_memory_over_limit → break  (exits run() loop quietly)
    A silent break means the GUI process exits with code 0. On some
    Python versions the multiprocessing infrastructure misclassifies a
    clean exit as "intentional" and the supervisor does not restart it,
    leaving operators without visibility.

  NEW (correct): is_memory_over_limit → sys.exit(1)
    Non-zero exit code guarantees the supervisor's liveness check
    (`not mproc.is_alive()`) triggers a restart, regardless of whether
    a heartbeat-timeout fires first.  The OS reclaims GUI memory
    instantly, the detection pipeline is completely unaffected.

  _cleanup_local() centralises cv2 window / headless resource release
  so it can be called from both the normal exit path and sys.exit(1).

All other rendering and display logic is unchanged from v1.1.
"""

from __future__ import annotations
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path
from multiprocessing import Queue
from typing import Dict, List, Optional

import cv2
import numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config_loader import VisionSystemConfig
from core.ipc_schema import (
    MessageType, ProcessSource, DetectionObject,
    PairResult, PairStatus, make_heartbeat,
)
from core.logging_setup import setup_process_logging, setup_crash_handler
from core.resource_monitor import get_process_memory_mb, is_memory_over_limit
from core.boundary_engine import CameraBoundarySet

logger = logging.getLogger(__name__)

# ─── Colours ─────────────────────────────────────────────────────────────────
COL_OK       = (0, 255, 0)
COL_PROBLEM  = (0, 0, 255)
COL_WARNING  = (0, 165, 255)
COL_INFO     = (255, 255, 0)
COL_WHITE    = (255, 255, 255)
COL_BLACK    = (0, 0, 0)
COL_OIL_CAN  = (0, 200, 255)
COL_BUNK_HOLE= (200, 0, 255)
COL_OC_BOUNDARY = (0, 180, 255)
COL_BH_BOUNDARY = (180, 0, 255)


# ─────────────────────────────────────────────────────────────────────────────
# Per-camera display state
# ─────────────────────────────────────────────────────────────────────────────

class CameraDisplayState:
    def __init__(self, camera_id: int, cfg: VisionSystemConfig):
        self.camera_id  = camera_id
        self.cfg        = cfg
        self.cam_cfg    = cfg.get_camera(camera_id)

        self.frame:           Optional[np.ndarray] = None
        self.detections:      List[dict]           = []
        self.pair_results:    List[dict]           = []
        self.fps:             float                = 0.0
        self.inference_ms:    float                = 0.0
        self.total_detections: int                 = 0
        self.problem_count:   int                  = 0
        self.success_rate:    float                = 100.0
        self.relay_states:    List[bool]           = [False, False, False]
        self.last_update:     float                = time.time()

        boundary_data = cfg.load_boundaries(camera_id)
        self.boundary_set: Optional[CameraBoundarySet] = None
        if boundary_data:
            self.boundary_set = CameraBoundarySet(boundary_data, strict_mode=False)

    def update_from_detection(self, msg: dict):
        frame_data = msg.get("frame_data")
        if frame_data:
            buf     = np.frombuffer(frame_data, dtype=np.uint8)
            decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if decoded is not None:
                self.frame = decoded

        self.detections       = msg.get("detections", [])
        self.pair_results     = msg.get("pair_results", [])
        self.fps              = msg.get("fps", 0.0)
        self.inference_ms     = msg.get("inference_time_ms", 0.0)
        self.total_detections = msg.get("total_detections", 0)
        self.problem_count    = msg.get("problem_count", 0)
        self.success_rate     = msg.get("success_rate", 100.0)
        self.last_update      = time.time()

    def update_relay(self, states: List[bool]):
        self.relay_states = states


# ─────────────────────────────────────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────────────────────────────────────

class FrameRenderer:
    def __init__(self, cfg: VisionSystemConfig):
        self.cfg        = cfg
        self.gcfg       = cfg.gui
        self._start_time = time.time()

    def render(self, state: CameraDisplayState, system_fps: float) -> np.ndarray:
        if state.frame is None:
            h      = state.cam_cfg.frame_height if state.cam_cfg else 720
            w      = state.cam_cfg.frame_width  if state.cam_cfg else 1280
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(canvas, f"Camera {state.camera_id} — Waiting...",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_WHITE, 2)
            return canvas

        frame  = state.frame.copy()
        h, w   = frame.shape[:2]
        self._draw_boundaries(frame, state, w, h)
        self._draw_detections(frame, state, w, h)
        self._draw_pair_status(frame, state, w, h)
        self._draw_hud(frame, state, w, h)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (w - 250, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_INFO, 1, cv2.LINE_AA)
        return frame

    def _draw_boundaries(self, frame, state, w, h):
        if state.boundary_set is None:
            return
        overlay   = frame.copy()
        alpha     = self.gcfg.overlay_alpha
        thickness = self.gcfg.boundary_thickness
        bsets     = state.boundary_set.get_all_boundaries()

        for boundary in bsets.get("oil_can", []):
            pts = boundary.get_draw_points()
            if pts is not None:
                cv2.polylines(frame, [pts.reshape(-1, 1, 2)], True, COL_OC_BOUNDARY, thickness)
                cv2.fillPoly(overlay, [pts.reshape(-1, 1, 2)], COL_OC_BOUNDARY)
                cx, cy = pts.mean(axis=0).astype(int)
                cv2.putText(frame, boundary.id, (cx - 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_OC_BOUNDARY, 1)

        for boundary in bsets.get("bunk_hole", []):
            pts = boundary.get_draw_points()
            if pts is not None:
                cv2.polylines(frame, [pts.reshape(-1, 1, 2)], True, COL_BH_BOUNDARY, thickness)
                cv2.fillPoly(overlay, [pts.reshape(-1, 1, 2)], COL_BH_BOUNDARY)
                cx, cy = pts.mean(axis=0).astype(int)
                cv2.putText(frame, boundary.id, (cx - 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_BH_BOUNDARY, 1)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_detections(self, frame, state, w, h):
        for det in state.detections:
            x1n, y1n, x2n, y2n = det["bbox"]
            x1 = int(x1n * w); y1 = int(y1n * h)
            x2 = int(x2n * w); y2 = int(y2n * h)
            cls_name = det.get("class_name", "?")
            conf     = det.get("confidence", 0)
            bid      = det.get("boundary_id", "")
            color    = COL_OIL_CAN if "oil" in cls_name else COL_BUNK_HOLE
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.gcfg.bounding_box_thickness)
            label = f"{cls_name}:{conf:.2f}"
            if bid:
                label += f" [{bid}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_BLACK, 1, cv2.LINE_AA)

    def _draw_pair_status(self, frame, state, w, h):
        y_start = 120
        for pr in state.pair_results:
            relay_active = pr.get("relay_active", False)
            pair_name    = pr.get("pair_name", f"Pair {pr.get('pair_id', '?')}")
            oc_present   = pr.get("oil_can_present", False)
            bh_present   = pr.get("bunk_hole_present", False)
            relay_idx    = pr.get("relay_index", 0)
            status       = pr.get("status", "UNKNOWN")
            color        = COL_PROBLEM if relay_active else COL_OK
            relay_state  = (state.relay_states[relay_idx]
                            if relay_idx < len(state.relay_states) else False)
            line1 = f"{pair_name}: {status}"
            line2 = (f"  OC={'Y' if oc_present else 'N'}  "
                     f"BH={'Y' if bh_present else 'N'}  "
                     f"R{relay_idx+1}={'ON' if relay_state else 'OFF'}")
            cv2.putText(frame, line1, (10, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, self.gcfg.font_scale, color, 2, cv2.LINE_AA)
            cv2.putText(frame, line2, (10, y_start + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, self.gcfg.font_scale * 0.85, color, 1, cv2.LINE_AA)
            y_start += 50

    def _draw_hud(self, frame, state, w, h):
        uptime     = time.time() - self._start_time
        uptime_str = f"{int(uptime//3600)}h{int((uptime%3600)//60)}m{int(uptime%60)}s"
        lines      = [f"CAM {state.camera_id} | {state.cam_cfg.name if state.cam_cfg else '?'}"]
        if self.gcfg.show_fps:
            lines.append(f"FPS: {state.fps:.1f}  Inf: {state.inference_ms:.1f}ms")
        if self.gcfg.show_uptime:
            lines.append(f"Uptime: {uptime_str}")
        if self.gcfg.show_success_rate:
            lines.append(f"Success: {state.success_rate:.1f}%  Problems: {state.problem_count}")
        if self.gcfg.show_relay_state:
            rs = " | ".join([f"R{i+1}:{'ON' if s else 'OFF'}"
                             for i, s in enumerate(state.relay_states)])
            lines.append(f"Relays: {rs}")

        bg_h = len(lines) * 22 + 10
        cv2.rectangle(frame, (0, 0), (340, bg_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (340, bg_h), COL_INFO, 1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (6, 18 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, self.gcfg.font_scale,
                        COL_WHITE, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# GUI worker
# ─────────────────────────────────────────────────────────────────────────────

class GUIWorker:
    """OpenCV-based GUI worker.  Falls back gracefully if no display available."""

    def __init__(self, cfg, result_queue, relay_state_queue,
                 heartbeat_queue, stop_event):
        self.cfg               = cfg
        self.gcfg              = cfg.gui
        self.result_queue      = result_queue
        self.relay_state_queue = relay_state_queue
        self.heartbeat_queue   = heartbeat_queue
        self.stop_event        = stop_event
        self.pid               = os.getpid()
        self.name              = "GUI"
        self._start_time       = time.time()
        self._last_heartbeat   = time.time()

        self._camera_states: Dict[int, CameraDisplayState] = {
            cam.id: CameraDisplayState(cam.id, cfg) for cam in cfg.cameras
        }
        self._renderer          = FrameRenderer(cfg)
        self._display_available = False

    def run(self):
        logger.info("[GUI] PID=%d starting", self.pid)

        try:
            cv2.namedWindow("_test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("_test")
            self._display_available = True
            logger.info("[GUI] Display available, launching windows")
        except Exception as e:
            logger.warning("[GUI] No display: %s — running headless", e)
            self._run_headless()
            return

        for cam_id in self._camera_states:
            wname = f"Camera {cam_id} — {self.cfg.get_camera(cam_id).name}"
            cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(wname, 1280, 720)

        update_interval = self.gcfg.update_interval_ms / 1000.0

        while not self.stop_event.is_set():
            t0 = time.time()
            self._drain_queues()

            for cam_id, state in self._camera_states.items():
                wname    = f"Camera {cam_id} — {self.cfg.get_camera(cam_id).name}"
                rendered = self._renderer.render(state, state.fps)
                try:
                    cv2.imshow(wname, rendered)
                except Exception as e:
                    logger.error("[GUI] imshow error: %s", e)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("[GUI] User requested close")
                break

            self._maybe_heartbeat()

            # ── Task 5: hard exit on memory violation ─────────────────────────
            # A clean sys.exit(1) gives the supervisor a non-zero exit code
            # so the liveness/watchdog check unambiguously triggers a restart.
            # The GUI dying does NOT affect cameras, detection, or relay.
            if is_memory_over_limit(self.gcfg.memory_limit_mb):
                logger.critical(
                    "[GUI] RAM limit %.0f MB exceeded — "
                    "hard exit so supervisor can restart this process cleanly.",
                    self.gcfg.memory_limit_mb)
                self._cleanup_local()
                sys.exit(1)   # Task 5: hard exit, NOT break

            elapsed = time.time() - t0
            sleep_t = update_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        self._cleanup_local()
        logger.info("[GUI] Exiting cleanly")

    def _run_headless(self):
        """Drain queues and send heartbeats without a display."""
        logger.info("[GUI] Running in headless mode")
        while not self.stop_event.is_set():
            self._drain_queues()
            self._maybe_heartbeat()

            # Task 5: same hard-exit policy in headless mode
            if is_memory_over_limit(self.gcfg.memory_limit_mb):
                logger.critical(
                    "[GUI] RAM limit exceeded (headless) — hard exit.")
                sys.exit(1)

            time.sleep(0.1)
        logger.info("[GUI] Headless mode exiting")

    # ── Task 5: centralised local cleanup ────────────────────────────────────

    def _cleanup_local(self):
        """
        Release OpenCV windows.
        Called from both the normal exit path and the emergency sys.exit() path.
        """
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # ── Queue drains ─────────────────────────────────────────────────────────

    def _drain_queues(self):
        for _ in range(30):
            try:
                msg = self.result_queue.get_nowait()
            except Exception:
                break
            if msg.get("type") == MessageType.DETECTION_RESULT.value:
                cam_id = msg.get("camera_id")
                if cam_id is not None and cam_id in self._camera_states:
                    self._camera_states[cam_id].update_from_detection(msg)

        for _ in range(20):
            try:
                msg = self.relay_state_queue.get_nowait()
            except Exception:
                break
            if msg.get("type") == MessageType.RELAY_STATE.value:
                states = msg.get("relay_states", [False, False, False])
                for state in self._camera_states.values():
                    state.update_relay(states)

    def _maybe_heartbeat(self):
        now = time.time()
        if now - self._last_heartbeat >= 2.0:
            hb = make_heartbeat(
                source=ProcessSource.GUI, camera_id=None,
                process_name=self.name, pid=self.pid,
                memory_mb=get_process_memory_mb(),
                fps=0.0, status="running")
            try:
                self.heartbeat_queue.put_nowait(hb.to_dict())
            except Exception:
                pass
            self._last_heartbeat = now


# ─────────────────────────────────────────────────────────────────────────────
# Process entry point
# ─────────────────────────────────────────────────────────────────────────────

def gui_process_entry(config_path, result_queue, relay_state_queue,
                       heartbeat_queue, stop_event, log_dir="logs"):
    cfg = VisionSystemConfig(config_path)
    setup_process_logging(
        "gui", log_dir, cfg.system.log_level,
        cfg.logging.max_bytes, cfg.logging.backup_count)
    setup_crash_handler("gui", log_dir)

    def _sigterm(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGTERM, _sigterm)

    worker = GUIWorker(
        cfg=cfg, result_queue=result_queue,
        relay_state_queue=relay_state_queue,
        heartbeat_queue=heartbeat_queue, stop_event=stop_event)
    try:
        worker.run()
    except SystemExit:
        raise   # propagate sys.exit() — do not swallow
    except Exception as e:
        logger.critical("[gui] Fatal: %s", e, exc_info=True)
        sys.exit(1)   # non-zero so supervisor restarts; GUI crash is non-fatal

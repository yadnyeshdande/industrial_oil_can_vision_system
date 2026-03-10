"""
gui/unified_gui.py  v3.2
========================
Industrial Operator Dashboard — 4-tab layout.

Tabs:
  [1] Dashboard     — all cameras, relay grid, health bars
  [2] Camera View   — single camera with [CAM1][CAM2][CAM3] buttons,
                       [Edit Boundaries] button, [Start/Stop Detection] button
  [3] System Health — process table + resource bars
  [4] Logs Viewer   — tail of supervisor/pool/relay logs

Improvements over v3.1:
  • Boundary Editor TAB removed — editing lives in Camera View
  • Camera selector [CAM 1][CAM 2][CAM 3] buttons, plus arrow keys
  • [Edit Boundaries] button → freeze frame → launch editor in new window
  • Mouse callback restored after editor closes (no '+' cursor bleed)
  • Failsafe: if ANY camera is missing its boundary file,
      - Detection cannot be started (button disabled)
      - Amber banner: "Boundaries not configured — Camera X"
  • Camera status chip: CONNECTED (green) / STALE (amber) / OFFLINE (red)
  • Detection state pill: DETECTING (green) / PREVIEW MODE (amber)
  • Hover highlight on all interactive buttons
  • Flashing red alarm banner when relay is active
  • Mouse-clickable tabs + fullscreen on startup
"""
from __future__ import annotations
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from multiprocessing import Event, Queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config_loader import VisionSystemConfig
from core.ipc_schema import (
    MessageType, ProcessSource,
    GUICommandMessage, BoundaryReloadMessage, make_heartbeat,
)
from core.logging_setup import setup_process_logging, setup_crash_handler
from core.resource_monitor import get_process_memory_mb, is_memory_over_limit

logger = logging.getLogger(__name__)

# ── Design tokens ──────────────────────────────────────────────────────────────
BG      = ( 21,  20,  18)
PANEL   = ( 35,  33,  30)
PANEL2  = ( 48,  46,  43)
BORDER  = ( 65,  63,  58)
ACCENT  = ( 76, 175,  80)
ACC_B   = ( 33, 150, 243)
WARN    = ( 50, 193, 255)
ERR     = ( 60,  60, 220)
TEXT    = (224, 224, 224)
DIM     = (130, 128, 124)
BRIGHT  = (255, 255, 255)
T_ACT   = ( 60, 110,  76)
T_IDLE  = ( 42,  42,  40)
C_OC    = ( 60, 165, 255)
C_BH    = ( 76, 175,  80)
R_ON    = ( 55,  55, 210)
R_OFF   = ( 55,  55,  50)

FONT    = cv2.FONT_HERSHEY_SIMPLEX
FONT2   = cv2.FONT_HERSHEY_DUPLEX
TABS    = ["Dashboard", "Camera View", "System Health", "Logs Viewer"]
TICONS  = ["[1]", "[2]", "[3]", "[4]"]
TAB_H   = 42
STAT_H  = 28


# ── Drawing primitives ─────────────────────────────────────────────────────────
def _t(img, txt, pos, sc=0.50, col=TEXT, th=1):
    cv2.putText(img, str(txt), pos, FONT, sc, col, th, cv2.LINE_AA)

def _tb(img, txt, pos, sc=0.58, col=BRIGHT, th=2):
    cv2.putText(img, str(txt), pos, FONT2, sc, col, th, cv2.LINE_AA)

def _fill(img, x1, y1, x2, y2, c):
    cv2.rectangle(img, (x1, y1), (x2, y2), c, -1)

def _box(img, x1, y1, x2, y2, c, th=1):
    cv2.rectangle(img, (x1, y1), (x2, y2), c, th)

def _hline(img, y, x1, x2, c=BORDER, th=1):
    cv2.line(img, (x1, y), (x2, y), c, th)

def _dot(img, cx, cy, r, c):
    cv2.circle(img, (cx, cy), r, c, -1, cv2.LINE_AA)

def _rr(img, x1, y1, x2, y2, r, c):
    """Rounded-rect fill."""
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), c, -1)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), c, -1)
    for px, py in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (px, py), r, c, -1, cv2.LINE_AA)

def _bar(img, x, y, w, h, val, maxv, c=ACCENT, bg=PANEL2):
    f = int(w * min(max(val / max(maxv, 1), 0), 1))
    _fill(img, x, y, x+w, y+h, bg)
    if f > 0:
        _fill(img, x, y, x+f, y+h, c)
    _box(img, x, y, x+w, y+h, BORDER)

def _fmt_up(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

def _sc(status: str):
    s = status.upper()
    return ACCENT if s == "OK" else ERR if s in ("BOTH_ABSENT","BH_ONLY","OC_ONLY","PROBLEM") else WARN


# ── Button ─────────────────────────────────────────────────────────────────────
class Btn:
    """Rectangular button with hover + active state."""
    def __init__(self, x1, y1, x2, y2, label, normal_bg=(50,50,48), active_bg=(40,90,50)):
        self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2
        self.label = label; self.nbg = normal_bg; self.abg = active_bg

    def draw(self, canvas, hover=False, active=False, border_col=BORDER):
        bg = self.abg if active else ((70,70,68) if hover else self.nbg)
        _rr(canvas, self.x1, self.y1, self.x2, self.y2, 4, bg)
        _box(canvas, self.x1, self.y1, self.x2, self.y2, border_col if active else BORDER, 1)
        (lw, lh), _ = cv2.getTextSize(self.label, FONT2, 0.46, 1)
        tx = self.x1 + (self.x2 - self.x1 - lw) // 2
        ty = self.y1 + (self.y2 - self.y1 + lh) // 2
        _tb(canvas, self.label, (tx, ty), sc=0.46, col=BRIGHT if active else TEXT)

    def hit(self, x, y) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


# ── Mouse state ────────────────────────────────────────────────────────────────
class MouseState:
    def __init__(self):
        self.x = 0; self.y = 0
        self._cx = 0; self._cy = 0; self._clicked = False

    def make_cb(self):
        def _cb(event, x, y, flags, param):
            self.x = x; self.y = y
            if event == cv2.EVENT_LBUTTONDOWN:
                self._cx = x; self._cy = y; self._clicked = True
        return _cb

    def consume(self) -> Optional[Tuple[int, int]]:
        if self._clicked:
            self._clicked = False
            return (self._cx, self._cy)
        return None


# ── Camera state ───────────────────────────────────────────────────────────────
class CameraState:
    def __init__(self, cam_id: int, cfg: VisionSystemConfig):
        self.cam_id  = cam_id
        self.cfg     = cfg
        self.cam_cfg = cfg.get_camera(cam_id)
        self.frame: Optional[np.ndarray] = None
        self.detections:   List[dict] = []
        self.pair_results: List[dict] = []
        self.fps           = 0.0
        self.inference_ms  = 0.0
        self.problem_count = 0
        self.success_rate  = 100.0
        self.last_update   = 0.0
        self.connected     = False
        self._bset         = None
        self._shm_reader   = None
        self._shm_ok       = False
        self._load_bd()

    def _load_bd(self):
        try:
            bd = self.cfg.load_boundaries(self.cam_id)
            if bd:
                from detection.gpu_worker_pool import _normalize_boundary_data
                from core.boundary_engine import CameraBoundarySet
                self._bset = CameraBoundarySet(_normalize_boundary_data(bd, self.cam_id))
        except Exception:
            self._bset = None

    def reload_bd(self):
        self._load_bd()

    def has_boundaries(self) -> bool:
        return self.cfg.get_boundary_path(self.cam_id).exists()

    def update(self, msg: dict):
        fd = msg.get("frame_data")
        if fd:
            try:
                buf = np.frombuffer(fd, dtype=np.uint8)
                dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if dec is not None:
                    self.frame = dec
            except Exception:
                pass
        self.detections   = msg.get("detections", [])
        self.pair_results = msg.get("pair_results", [])
        self.fps          = msg.get("fps", 0.0)
        self.inference_ms = msg.get("inference_time_ms", 0.0)
        self.problem_count= msg.get("problem_count", 0)
        self.success_rate = msg.get("success_rate", 100.0)
        self.last_update  = time.time()
        self.connected    = True

    @property
    def has_problem(self) -> bool:
        return any(p.get("relay_active", False) for p in self.pair_results)

    @property
    def stale(self) -> bool:
        return time.time() - self.last_update > 5.0

    def read_preview(self) -> Optional[np.ndarray]:
        try:
            if not self._shm_ok:
                from core.shared_frame import SharedFrameReader
                cc = self.cfg.get_camera(self.cam_id)
                if cc is None:
                    return None
                self._shm_reader = SharedFrameReader(
                    cc.shared_memory_name, cc.frame_width, cc.frame_height)
                self._shm_ok = self._shm_reader.connect(timeout=1.5)
            if self._shm_ok and self._shm_reader:
                frame, _ = self._shm_reader.read()
                return frame
        except Exception:
            self._shm_ok = False
        return None


# ── Tab hit-test ───────────────────────────────────────────────────────────────
def _tab_hit(x: int, y: int, W: int) -> Optional[int]:
    if y > TAB_H:
        return None
    tw = W // len(TABS)
    for i in range(len(TABS)):
        if i * tw <= x < (i + 1) * tw:
            return i
    return None


# ── Common bars ────────────────────────────────────────────────────────────────
def _draw_tabs(canvas, active: int, W: int, ms: MouseState):
    _fill(canvas, 0, 0, W, TAB_H, (28, 27, 25))
    _hline(canvas, TAB_H - 1, 0, W, BORDER, 2)
    n = len(TABS); tw = W // n
    for i, name in enumerate(TABS):
        x1 = i * tw; x2 = x1 + tw - 2
        is_a = (i == active)
        hov  = (x1 <= ms.x < x2 and ms.y <= TAB_H and not is_a)
        bg   = T_ACT if is_a else ((55, 55, 52) if hov else T_IDLE)
        _fill(canvas, x1 + 1, 1, x2, TAB_H - 2, bg)
        if is_a:
            _hline(canvas, TAB_H - 2, x1 + 1, x2, ACCENT, 3)
        lbl = f"{TICONS[i]} {name}"
        (lw, _), _ = cv2.getTextSize(lbl, FONT, 0.50, 1)
        _t(canvas, lbl, (x1 + (tw - lw) // 2, TAB_H - 14),
           sc=0.50, col=BRIGHT if is_a else DIM)


def _draw_statusbar(canvas, W: int, H: int, health: dict,
                    preview: bool, start_t: float):
    y1 = H - STAT_H
    _fill(canvas, 0, y1, W, H, (22, 21, 19))
    _hline(canvas, y1, 0, W, BORDER)
    up = time.time() - start_t
    _t(canvas,
       f"{time.strftime('%Y-%m-%d %H:%M:%S')}   Uptime {_fmt_up(up)}",
       (10, y1 + 18), sc=0.46, col=DIM)
    g  = health.get("gpu_stats", {})
    cx = W // 2 - 260
    for lbl, val, warn in [
        ("CPU",  health.get("cpu_pct", 0), 70),
        ("GPU",  g.get("utilization_pct", 0), 85),
        ("VRAM", g.get("vram_used_mb", 0), 5000),
        ("Temp", g.get("temperature_c", 0), 75),
    ]:
        unit = "MB" if lbl == "VRAM" else "°C" if lbl == "Temp" else "%"
        col  = ERR if (lbl == "Temp" and val > 85) else WARN if val > warn else DIM
        _t(canvas, f"{lbl} {val:.0f}{unit}", (cx, y1 + 18), sc=0.46, col=col)
        cx += 130
    mode = "PREVIEW MODE" if preview else "DETECTING"
    col  = WARN if preview else ACCENT
    (tw, _), _ = cv2.getTextSize(mode, FONT, 0.50, 1)
    mx = W - tw - 22
    _rr(canvas, mx - 8, y1 + 4, W - 6, H - 4, 4, PANEL2)
    _box(canvas, mx - 8, y1 + 4, W - 6, H - 4, col, 1)
    _t(canvas, mode, (mx, y1 + 19), sc=0.50, col=col)


def _draw_alarm(canvas, W: int, body_y: int, cam_states: Dict) -> int:
    """Returns banner height used (0 if no alarm)."""
    probs = []
    for cid, st in cam_states.items():
        for pr in st.pair_results:
            if pr.get("relay_active"):
                probs.append(f"CAM {cid}  {pr.get('pair_name','?')}: {pr.get('status','?')}")
    if not probs:
        return 0
    bh    = 32
    flash = int(time.time() * 2) % 2 == 0
    _fill(canvas, 0, body_y, W, body_y + bh, ERR if flash else (70, 50, 50))
    _t(canvas, "  \u26a0  " + "   |   ".join(probs[:4]),
       (12, body_y + 22), sc=0.56, col=BRIGHT, th=1)
    return bh


def _draw_bd_banner(canvas, W: int, body_y: int, missing: List[int]) -> int:
    """Amber 'Boundaries not configured' banner."""
    bh  = 32
    _fill(canvas, 0, body_y, W, body_y + bh, (28, 25, 10))
    _box(canvas,  0, body_y, W, body_y + bh, WARN, 1)
    cams = ", ".join(f"Camera {c}" for c in sorted(missing))
    _t(canvas,
       f"  \u26a0  Boundaries not configured \u2014 {cams}"
       "   |   Go to Camera View and press [Edit Boundaries]",
       (12, body_y + 22), sc=0.50, col=WARN)
    return bh


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
def _render_dashboard(canvas, W, H, by, bh,
                      cam_states: Dict, relay_states: List[bool],
                      health: dict, cfg):
    PAD     = 10
    cam_ids = sorted(cam_states.keys())
    n       = len(cam_ids)
    grid_h  = int(bh * 0.52)
    cw      = (W - PAD * (n + 1)) // max(n, 1)
    ch      = grid_h - PAD * 2

    for idx, cid in enumerate(cam_ids):
        st  = cam_states[cid]
        cx  = PAD + idx * (cw + PAD)
        cy  = by + PAD

        # card
        border = ERR if st.has_problem else (ACCENT if (st.connected and not st.stale) else BORDER)
        _fill(canvas, cx, cy, cx+cw, cy+ch, PANEL)
        _box(canvas,  cx, cy, cx+cw, cy+ch, border, 2)

        # video
        fh2 = ch - 52; fw2 = cw - 4; fx2 = cx + 2; fy2 = cy + 26
        if st.frame is not None and not st.stale:
            try:
                canvas[fy2:fy2+fh2, fx2:fx2+fw2] = cv2.resize(st.frame, (fw2, fh2))
            except Exception:
                _fill(canvas, fx2, fy2, fx2+fw2, fy2+fh2, (18, 18, 18))
        else:
            _fill(canvas, fx2, fy2, fx2+fw2, fy2+fh2, (18, 18, 18))
            msg = "No Signal" if not st.connected else "Stale"
            (tw2, _), _ = cv2.getTextSize(msg, FONT, 0.55, 1)
            _t(canvas, msg, (fx2 + (fw2-tw2)//2, fy2 + fh2//2), sc=0.55, col=DIM)

        _overlay_bd_small(canvas, st._bset, fx2, fy2, fw2, fh2)

        # name bar
        name = st.cam_cfg.name if st.cam_cfg else f"Camera {cid}"
        _fill(canvas, cx, cy, cx+cw, cy+24, PANEL2)
        _tb(canvas, name, (cx+6, cy+17), sc=0.50)

        # status chip
        bd_ok = st.has_boundaries()
        chip  = ("OFFLINE" if not st.connected
                 else "STALE"  if st.stale
                 else "ERR"    if st.has_problem
                 else "NO BD"  if not bd_ok
                 else "OK")
        chip_col = (DIM     if not st.connected
                    else WARN   if st.stale
                    else ERR    if st.has_problem
                    else WARN   if not bd_ok
                    else ACCENT)
        (cw2, _), _ = cv2.getTextSize(chip, FONT, 0.40, 1)
        sx = cx + cw - cw2 - 18
        _rr(canvas, sx-4, cy+4, sx+cw2+8, cy+20, 3, chip_col)
        _t(canvas, chip, (sx, cy+17), sc=0.40, col=BRIGHT)

        # bottom strip
        by2 = cy + ch - 24
        _fill(canvas, cx, by2, cx+cw, cy+ch-2, (14, 14, 14))
        _t(canvas, f"FPS {st.fps:.1f}",  (cx+6,   by2+16), sc=0.44, col=DIM)
        _t(canvas, f"{st.inference_ms:.0f}ms", (cx+70, by2+16), sc=0.44, col=DIM)
        sr_col = ACCENT if st.success_rate >= 99 else WARN if st.success_rate > 80 else ERR
        _t(canvas, f"SR {st.success_rate:.0f}%", (cx+136, by2+16), sc=0.44, col=sr_col)

    # pair row
    ps_y = by + grid_h + PAD
    _fill(canvas, PAD, ps_y, W-PAD, ps_y+26, PANEL)
    _t(canvas, "PAIRS", (PAD+8, ps_y+18), sc=0.44, col=DIM)
    px = PAD + 72
    for cid in cam_ids:
        for pr in cam_states[cid].pair_results:
            sc2 = _sc(pr.get("status", "?"))
            _dot(canvas, px+5, ps_y+13, 5, sc2)
            _t(canvas, f"C{cid}-{pr.get('pair_name','?')}: {pr.get('status','?')}",
               (px+14, ps_y+18), sc=0.42, col=sc2)
            px += 165

    # relay grid
    ry = ps_y + 34
    _t(canvas, "RELAY OUTPUTS", (PAD+6, ry+20), sc=0.44, col=DIM)
    rx = PAD + 150; rs = 50; rg = 6
    for i in range(9):
        ex  = rx + i * (rs + rg) + (8 if i > 0 and i % 3 == 0 else 0)
        on  = relay_states[i] if i < len(relay_states) else False
        _rr(canvas, ex, ry, ex+rs, ry+rs-4, 6, R_ON if on else R_OFF)
        _box(canvas, ex, ry, ex+rs, ry+rs-4, BORDER, 1)
        dot = (80, 80, 255) if on else (50, 50, 50)
        _dot(canvas, ex+rs//2, ry+13, 7, dot)
        cv2.circle(canvas, (ex+rs//2, ry+13), 7, BRIGHT if on else BORDER, 1, cv2.LINE_AA)
        _tb(canvas, f"R{i+1}", (ex+rs//2-10, ry+29), sc=0.46, col=BRIGHT if on else DIM)
        _t(canvas, "ON" if on else "off", (ex+rs//2-8, ry+43),
           sc=0.36, col=(80, 80, 255) if on else DIM)
    for i, cid in enumerate(cam_ids):
        gx = rx + i * 3 * (rs + rg) + i * 8
        _t(canvas, f"Cam {cid}", (gx + rs//2, ry - 4), sc=0.38, col=DIM)

    # health mini
    hy     = ry + rs + 12
    gstats = health.get("gpu_stats", {})
    bx     = PAD + 6
    for lbl, val, maxv, col in [
        ("CPU",  health.get("cpu_pct", 0),  100,  WARN if health.get("cpu_pct",0)>70 else ACCENT),
        ("RAM",  health.get("ram_used_mb", 0), 16384, ACCENT),
        ("GPU",  gstats.get("utilization_pct", 0), 100, ACCENT),
        ("VRAM", gstats.get("vram_used_mb", 0),
                 gstats.get("vram_total_mb", 6144), ACCENT),
        ("TEMP", gstats.get("temperature_c", 0), 100,
                 ERR if gstats.get("temperature_c",0)>85
                 else WARN if gstats.get("temperature_c",0)>70 else ACCENT),
    ]:
        _t(canvas, lbl, (bx, hy+13), sc=0.42, col=DIM); bx += 38
        _bar(canvas, bx, hy+4, 110, 8, val, maxv, col)
        _t(canvas, f"{val:.0f}", (bx+116, hy+13), sc=0.38, col=DIM)
        bx += 162


def _overlay_bd_small(canvas, bset, fx, fy, fw, fh):
    if bset is None:
        return
    try:
        bdata = bset.get_all_boundaries()
        for b in bdata.get("oil_can", []) + bdata.get("bunk_hole", []):
            pts = b.get_draw_points()
            if pts is not None and len(pts) >= 3:
                col = C_OC if b.id.upper().startswith("OC") else C_BH
                sp  = pts.copy().astype(float)
                sp[:, 0] *= fw / 1280; sp[:, 1] *= fh / 720
                sp  = sp.astype(np.int32) + [fx, fy]
                cv2.polylines(canvas, [sp.reshape(-1, 1, 2)], True, col, 1, cv2.LINE_AA)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Camera View
# ═══════════════════════════════════════════════════════════════════════════════
def _render_camera_view(canvas, W, H, by, bh,
                        state: CameraState,
                        relay_states: List[bool],
                        cfg,
                        preview: bool,
                        ms: MouseState,
                        cam_ids: List[int],
                        detection_allowed: bool,
                        hit_regions: dict):
    """
    Renders camera view tab.
    hit_regions is populated with button objects for click handling.
    """
    PAD    = 12
    SP     = 240
    fw     = W - SP - PAD * 3
    fh     = min(int(fw * 9 / 16), bh - 96)
    fx     = PAD
    fy     = by + PAD

    # ── video feed ─────────────────────────────────────────────────────────────
    _fill(canvas, fx, fy, fx+fw, fy+fh, (10, 10, 10))
    _box(canvas,  fx, fy, fx+fw, fy+fh, BORDER, 1)

    if state.frame is not None and not state.stale:
        try:
            canvas[fy:fy+fh, fx:fx+fw] = cv2.resize(state.frame, (fw, fh))
        except Exception:
            pass
    else:
        msg = "No Signal" if not state.connected else "Waiting for frame..."
        (tw2, _), _ = cv2.getTextSize(msg, FONT, 0.7, 1)
        _t(canvas, msg, (fx + (fw-tw2)//2, fy + fh//2), sc=0.7, col=DIM)

    _overlay_bd_large(canvas, state._bset, fx, fy, fw, fh)

    for det in state.detections:
        x1n, y1n, x2n, y2n = det["bbox"]
        x1 = int(x1n*fw)+fx; y1 = int(y1n*fh)+fy
        x2 = int(x2n*fw)+fx; y2 = int(y2n*fh)+fy
        col = C_OC if "oil" in det.get("class_name", "") else C_BH
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
        lbl = f"{det.get('class_name','?')} {det.get('confidence',0):.0%}"
        (lw2, lh2), _ = cv2.getTextSize(lbl, FONT, 0.44, 1)
        _fill(canvas, x1, y1-lh2-6, x1+lw2+8, y1, col)
        _t(canvas, lbl, (x1+4, y1-4), sc=0.44, col=(0,0,0))

    # ── button row ─────────────────────────────────────────────────────────────
    bar_y  = fy + fh + 8
    btn_h  = 30
    bx     = fx

    # Camera selector buttons
    cam_btns = []
    for cid in cam_ids:
        active   = (cid == state.cam_id)
        lbl      = f"CAM {cid + 1}"
        btn      = Btn(bx, bar_y, bx + 74, bar_y + btn_h, lbl,
                       (40, 90, 50) if active else (50, 50, 48),
                       (40, 90, 50))
        hover    = btn.hit(ms.x, ms.y) and not active
        border_c = ACCENT if active else BORDER
        _rr(canvas, btn.x1, btn.y1, btn.x2, btn.y2, 4,
            btn.abg if active else ((70, 70, 68) if hover else btn.nbg))
        _box(canvas, btn.x1, btn.y1, btn.x2, btn.y2, border_c, 1)
        (lw2, lh2), _ = cv2.getTextSize(lbl, FONT2, 0.46, 1)
        tx = btn.x1 + (74 - lw2) // 2
        ty = btn.y1 + (btn_h + lh2) // 2
        _tb(canvas, lbl, (tx, ty), sc=0.46, col=BRIGHT if active else TEXT)
        cam_btns.append((cid, btn))
        bx += 78

    hit_regions["cam_btns"] = cam_btns

    bx += 8
    # Edit Boundaries button
    edit_btn = Btn(bx, bar_y, bx + 162, bar_y + btn_h, "Edit Boundaries",
                   (45, 45, 35), (60, 55, 30))
    bd_ok    = state.has_boundaries()
    e_col    = WARN if not bd_ok else ACC_B
    hover_e  = edit_btn.hit(ms.x, ms.y)
    _rr(canvas, edit_btn.x1, edit_btn.y1, edit_btn.x2, edit_btn.y2, 4,
        (65, 60, 38) if hover_e else edit_btn.nbg)
    _box(canvas, edit_btn.x1, edit_btn.y1, edit_btn.x2, edit_btn.y2, e_col, 1)
    (lw2, lh2), _ = cv2.getTextSize("Edit Boundaries", FONT2, 0.46, 1)
    tx = edit_btn.x1 + (162 - lw2) // 2
    ty = edit_btn.y1 + (btn_h + lh2) // 2
    _tb(canvas, "Edit Boundaries", (tx, ty), sc=0.46, col=e_col)
    hit_regions["edit_btn"] = edit_btn
    bx += 170

    # Start / Stop Detection button
    if not preview:
        det_lbl = "Stop Detection"
        det_nbg = (60, 35, 35)
        det_bc  = ERR
    elif detection_allowed:
        det_lbl = "Start Detection"
        det_nbg = (35, 55, 35)
        det_bc  = ACCENT
    else:
        det_lbl = "Detection Locked"
        det_nbg = (40, 40, 40)
        det_bc  = DIM

    det_btn  = Btn(bx, bar_y, bx + 162, bar_y + btn_h, det_lbl, det_nbg, det_nbg)
    hover_d  = det_btn.hit(ms.x, ms.y) and (not preview or detection_allowed)
    _rr(canvas, det_btn.x1, det_btn.y1, det_btn.x2, det_btn.y2, 4,
        tuple(min(c+14, 255) for c in det_nbg) if hover_d else det_nbg)
    _box(canvas, det_btn.x1, det_btn.y1, det_btn.x2, det_btn.y2, det_bc, 1)
    (lw2, lh2), _ = cv2.getTextSize(det_lbl, FONT2, 0.46, 1)
    tx = det_btn.x1 + (162 - lw2) // 2
    ty = det_btn.y1 + (btn_h + lh2) // 2
    _tb(canvas, det_lbl, (tx, ty), sc=0.46, col=det_bc)
    hit_regions["det_btn"] = det_btn

    # info line below buttons
    info_y = bar_y + btn_h + 14
    if not bd_ok:
        _t(canvas, "  \u26a0  No boundaries — press [Edit Boundaries] to draw them",
           (fx, info_y), sc=0.46, col=WARN)
    elif preview:
        _t(canvas, "PREVIEW MODE — press [Start Detection] to activate",
           (fx, info_y), sc=0.46, col=WARN)
    else:
        name = state.cam_cfg.name if state.cam_cfg else f"Camera {state.cam_id}"
        _t(canvas,
           f"{name}   FPS {state.fps:.1f}   {state.inference_ms:.0f}ms   SR {state.success_rate:.0f}%",
           (fx, info_y), sc=0.44, col=DIM)
    _t(canvas, "\u2190 \u2192  change camera   E  edit boundaries   D  detect   P  preview",
       (fx, info_y + 18), sc=0.38, col=DIM)

    # ── side panel ─────────────────────────────────────────────────────────────
    sx = fx + fw + PAD; sy = by + PAD; sw = SP - PAD
    _fill(canvas, sx, sy, sx+sw, sy+bh-20, PANEL)
    _box(canvas,  sx, sy, sx+sw, sy+bh-20, BORDER, 1)

    oy = sy + 14
    _tb(canvas, "PAIR STATUS", (sx+10, oy), sc=0.54); oy += 26
    _hline(canvas, oy, sx, sx+sw, BORDER); oy += 12

    cam_relays = cfg.relay.get_relay_indices(state.cam_id)
    for i, pr in enumerate(state.pair_results):
        status = pr.get("status", "?")
        sc2    = _sc(status)
        ri     = cam_relays[i] if i < len(cam_relays) else 0
        r_on   = relay_states[ri] if ri < len(relay_states) else False

        _dot(canvas, sx+13, oy+7, 7, sc2)
        _tb(canvas, pr.get("pair_name", f"P{i+1}"), (sx+27, oy+12), sc=0.50)
        oy += 24

        def _row(lbl2, val2, col2, _oy=None):
            nonlocal oy
            _t(canvas, lbl2, (sx+14, oy), sc=0.40, col=DIM)
            _t(canvas, val2, (sx+88, oy), sc=0.40, col=col2)
            oy += 16

        _row("Status:", status, sc2)
        _row("OilCan:",  "✓" if pr.get("oil_can_present")  else "✗",
             ACCENT if pr.get("oil_can_present")  else ERR)
        _row("BunkHole:", "✓" if pr.get("bunk_hole_present") else "✗",
             ACCENT if pr.get("bunk_hole_present") else ERR)
        _row(f"R{ri+1}:", "ON" if r_on else "off", ERR if r_on else ACCENT)
        oy += 8
        _hline(canvas, oy, sx+8, sx+sw-8, BORDER); oy += 10

    # connection + detection status at bottom of side panel
    bot_y = sy + bh - 80
    conn_col = ACCENT if (state.connected and not state.stale) else WARN if state.stale else ERR
    conn_txt = "CONNECTED" if (state.connected and not state.stale) else "STALE" if state.stale else "OFFLINE"
    _dot(canvas, sx+15, bot_y+8,  6, conn_col)
    _t(canvas, conn_txt, (sx+26, bot_y+13), sc=0.44, col=conn_col)

    det_col = ACCENT if not preview else WARN
    det_txt = "DETECTING" if not preview else "PREVIEW"
    _dot(canvas, sx+15, bot_y+28, 6, det_col)
    _t(canvas, det_txt, (sx+26, bot_y+33), sc=0.44, col=det_col)


def _overlay_bd_large(canvas, bset, fx, fy, fw, fh):
    if bset is None:
        return
    try:
        bdata = bset.get_all_boundaries()
        ov    = canvas.copy()
        for group, col in [("oil_can", C_OC), ("bunk_hole", C_BH)]:
            for b in bdata.get(group, []):
                pts = b.get_draw_points()
                if pts is not None and len(pts) >= 3:
                    sp = pts.copy().astype(float)
                    sp[:, 0] *= fw / 1280; sp[:, 1] *= fh / 720
                    sp = sp.astype(np.int32) + [fx, fy]
                    cv2.fillPoly(ov, [sp.reshape(-1, 1, 2)], col)
                    cv2.polylines(canvas, [sp.reshape(-1, 1, 2)], True, col, 2, cv2.LINE_AA)
                    cx2, cy2 = sp.mean(0).astype(int)
                    _t(canvas, b.id, (cx2-14, cy2), sc=0.55, col=col, th=2)
        cv2.addWeighted(ov, 0.15, canvas, 0.85, 0, canvas)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — System Health
# ═══════════════════════════════════════════════════════════════════════════════
def _render_health(canvas, W, H, by, bh, health: dict):
    PAD = 14; oy = by + PAD
    _tb(canvas, "SYSTEM HEALTH", (PAD, oy+20), sc=0.70); oy += 50
    g = health.get("gpu_stats", {}); BW = 280; BHR = 12

    for lbl, val, maxv, unit, col in [
        ("CPU Usage",  health.get("cpu_pct", 0), 100, "%",
         WARN if health.get("cpu_pct", 0) > 70 else ACCENT),
        ("RAM",        health.get("ram_used_mb", 0), 16384, "MB", ACCENT),
        ("GPU Usage",  g.get("utilization_pct", 0), 100, "%", ACCENT),
        ("VRAM",       g.get("vram_used_mb", 0), g.get("vram_total_mb", 6144), "MB",
         ERR if g.get("vram_used_mb", 0) > 5000 else ACCENT),
        ("GPU Temp",   g.get("temperature_c", 0), 100, "°C",
         ERR if g.get("temperature_c", 0) > 85
         else WARN if g.get("temperature_c", 0) > 70 else ACCENT),
    ]:
        _t(canvas, lbl, (PAD, oy+10), sc=0.48, col=DIM)
        _bar(canvas, PAD+140, oy, BW, BHR, val, maxv, col)
        _t(canvas, f"{val:.0f}{unit}", (PAD+146+BW, oy+10), sc=0.46, col=col)
        oy += 28

    oy += 6
    _t(canvas,
       f"Uptime: {_fmt_up(health.get('system_uptime_s', 0))}"
       f"   Total Restarts: {health.get('total_restarts', 0)}",
       (PAD, oy), sc=0.46, col=DIM); oy += 28
    _hline(canvas, oy, PAD, W-PAD, BORDER); oy += 12
    _tb(canvas, "PROCESSES", (PAD, oy+12), sc=0.52, col=DIM); oy += 26

    cols = [("Name",160),("PID",80),("Status",100),("Restarts",90),("Mem",80),("FPS",60)]
    hx = PAD
    for h2, cw2 in cols:
        _t(canvas, h2, (hx, oy), sc=0.42, col=DIM); hx += cw2
    oy += 6; _hline(canvas, oy, PAD, W-PAD, BORDER); oy += 10

    for proc in health.get("processes", []):
        status = proc.get("status", "?"); rc = proc.get("restart_count", 0)
        rows = [
            (proc.get("name","?"),        TEXT),
            (str(proc.get("pid", "-")),   DIM),
            (status, ACCENT if status == "running" else ERR),
            (str(rc), ERR if rc > 3 else WARN if rc > 0 else ACCENT),
            (f"{proc.get('memory_mb',0):.0f}", DIM),
            (f"{proc.get('fps',0):.1f}",  DIM),
        ]
        rx = PAD
        for (val2, c2), (_, cw2) in zip(rows, cols):
            _t(canvas, val2, (rx, oy+12), sc=0.44, col=c2); rx += cw2
        oy += 20
        if oy > by + bh - 20:
            break


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Logs
# ═══════════════════════════════════════════════════════════════════════════════
def _render_logs(canvas, W, H, by, bh, log_lines: deque):
    PAD = 14; oy = by + PAD
    _tb(canvas, "LOGS VIEWER", (PAD, oy+16), sc=0.65); oy += 40
    _t(canvas, "supervisor.log  ·  gpu_pool.log  ·  relay.log",
       (PAD, oy), sc=0.42, col=DIM); oy += 18
    _hline(canvas, oy, PAD, W-PAD, BORDER); oy += 10
    max_l = (by + bh - oy) // 18
    for line in list(log_lines)[-max_l:]:
        col = (ERR if "CRITICAL" in line or "FATAL" in line
               else (80,100,220) if "ERROR" in line
               else WARN if "WARNING" in line
               else DIM)
        _t(canvas, line[:140], (PAD, oy), sc=0.40, col=col)
        oy += 18


def _refresh_logs(log_lines: deque, cfg):
    log_dir = Path(cfg.logging.log_dir)
    for name in ("supervisor.log","gpu_pool.log","relay.log","camera_0.log"):
        p = log_dir / name
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for line in f.readlines()[-20:]:
                    log_lines.append(line.strip())
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main GUI
# ═══════════════════════════════════════════════════════════════════════════════
class UnifiedGUI:
    def __init__(self, cfg, result_q, relay_q, hb_q, cmd_q, health_q,
                 stop_ev, preview):
        self.cfg        = cfg
        self.gcfg       = cfg.gui
        self.result_q   = result_q
        self.relay_q    = relay_q
        self.hb_q       = hb_q
        self.cmd_q      = cmd_q
        self.health_q   = health_q
        self.stop_ev    = stop_ev
        self.preview    = preview
        self.pid        = os.getpid()
        self.cam_ids    = sorted(c.id for c in cfg.cameras)
        self.cam_states = {cid: CameraState(cid, cfg) for cid in self.cam_ids}
        self.relay_states: List[bool] = [False] * 9
        self.health     = {}
        self.log_lines  = deque(maxlen=300)
        self._tab       = 0
        self._cam_idx   = 0
        self._start     = time.time()
        self._last_hb   = time.time()
        self._last_log  = time.time()
        self._in_editor = False
        self._ms        = MouseState()
        self._hit       = {}          # buttons drawn last frame

    # ── helpers ────────────────────────────────────────────────────────────────
    def _missing_bd(self) -> List[int]:
        return [cid for cid in self.cam_ids
                if not self.cam_states[cid].has_boundaries()]

    def _can_detect(self) -> bool:
        return len(self._missing_bd()) == 0

    def _active_state(self) -> CameraState:
        cid = self.cam_ids[self._cam_idx % len(self.cam_ids)]
        return self.cam_states[cid]

    # ── main loop ──────────────────────────────────────────────────────────────
    def run(self):
        logger.info("[GUI v3.2] PID=%d", self.pid)
        WNAME = "Industrial Vision System v3.2"
        try:
            cv2.namedWindow(WNAME, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(WNAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(WNAME, self._ms.make_cb())
        except Exception as e:
            logger.warning("[GUI] No display: %s", e)
            self._headless()
            return

        W, H     = self.gcfg.window_width, self.gcfg.window_height
        interval = self.gcfg.update_interval_ms / 1000.0

        while not self.stop_ev.is_set():
            t0    = time.time()
            click = self._ms.consume()
            if click:
                self._on_click(*click, W, H, WNAME)

            self._drain()

            try:
                r = cv2.getWindowImageRect(WNAME)
                if r[2] > 100 and r[3] > 100:
                    W, H = r[2], r[3]
            except Exception:
                pass

            canvas  = np.full((H, W, 3), BG, dtype=np.uint8)
            missing = self._missing_bd()

            # alarm / banner
            banner_h = 0
            if missing and not self._in_editor:
                banner_h = _draw_bd_banner(canvas, W, TAB_H, missing)
            elif any(st.has_problem for st in self.cam_states.values()):
                banner_h = _draw_alarm(canvas, W, TAB_H, self.cam_states)

            body_y = TAB_H + banner_h + 2
            body_h = H - body_y - STAT_H - 4

            _draw_tabs(canvas, self._tab, W, self._ms)
            _draw_statusbar(canvas, W, H, self.health,
                            bool(self.preview.value), self._start)
            self._render(canvas, W, H, body_y, body_h)

            try:
                cv2.imshow(WNAME, canvas)
            except Exception:
                break

            key = cv2.waitKey(1) & 0xFF
            if not self._on_key(key, WNAME):
                break

            self._heartbeat()
            if is_memory_over_limit(self.gcfg.memory_limit_mb):
                break

            dt = time.time() - t0
            if interval - dt > 0:
                time.sleep(interval - dt)

        cv2.destroyAllWindows()

    def _headless(self):
        while not self.stop_ev.is_set():
            self._drain(); self._heartbeat()
            if is_memory_over_limit(self.gcfg.memory_limit_mb):
                break
            time.sleep(0.1)

    # ── click handler ──────────────────────────────────────────────────────────
    def _on_click(self, x: int, y: int, W: int, H: int, wname: str):
        # Tab bar
        hit = _tab_hit(x, y, W)
        if hit is not None:
            self._tab = hit
            return

        # Camera View buttons
        if self._tab != 1:
            return

        # Camera selector
        for cid, btn in self._hit.get("cam_btns", []):
            if btn.hit(x, y):
                if cid in self.cam_ids:
                    self._cam_idx = self.cam_ids.index(cid)
                return

        # Edit Boundaries
        eb = self._hit.get("edit_btn")
        if eb and eb.hit(x, y):
            if not self._in_editor:
                self._open_editor(wname)
            return

        # Start / Stop Detection
        db = self._hit.get("det_btn")
        if db and db.hit(x, y):
            if not bool(self.preview.value):
                # Currently detecting → stop
                self.preview.value = 1
                self._cmd("stop_detection")
            elif self._can_detect():
                self.preview.value = 0
                self._cmd("start_detection")
            else:
                logger.warning("[GUI] Detection blocked — configure boundaries first")
            return

    # ── key handler ────────────────────────────────────────────────────────────
    def _on_key(self, key: int, wname: str) -> bool:
        if key in (ord('q'), ord('Q'), 27):
            return False
        if key == ord('1'):   self._tab = 0
        elif key == ord('2'): self._tab = 1
        elif key == ord('3'): self._tab = 2
        elif key == ord('4'): self._tab = 3
        elif key in (81, ord('a')):   # left arrow
            self._cam_idx = (self._cam_idx - 1) % max(len(self.cam_ids), 1)
        elif key in (83, ord('f')):   # right arrow
            self._cam_idx = (self._cam_idx + 1) % max(len(self.cam_ids), 1)
        elif key in (ord('e'), ord('E')):
            if not self._in_editor:
                self._open_editor(wname)
        elif key in (ord('d'), ord('D')):
            if self._can_detect():
                self.preview.value = 0
                self._cmd("start_detection")
            else:
                logger.warning("[GUI] Detection blocked — configure boundaries first")
        elif key in (ord('p'), ord('P')):
            self.preview.value = 1
            self._cmd("stop_detection")
        return True

    # ── tab renderer ───────────────────────────────────────────────────────────
    def _render(self, canvas, W, H, body_y, body_h):
        t = self._tab
        if t == 0:
            _render_dashboard(canvas, W, H, body_y, body_h,
                              self.cam_states, self.relay_states,
                              self.health, self.cfg)
        elif t == 1:
            if self.cam_ids:
                st = self._active_state()
                if bool(self.preview.value):
                    pf = st.read_preview()
                    if pf is not None:
                        st.frame = pf
                hit = {}
                _render_camera_view(canvas, W, H, body_y, body_h,
                                    st, self.relay_states, self.cfg,
                                    bool(self.preview.value),
                                    self._ms, self.cam_ids,
                                    self._can_detect(), hit)
                self._hit = hit
        elif t == 2:
            _render_health(canvas, W, H, body_y, body_h, self.health)
        elif t == 3:
            if time.time() - self._last_log > 3.0:
                _refresh_logs(self.log_lines, self.cfg)
                self._last_log = time.time()
            _render_logs(canvas, W, H, body_y, body_h, self.log_lines)

    # ── boundary editor ────────────────────────────────────────────────────────
    def _open_editor(self, main_wname: str):
        st    = self._active_state()
        frame = st.frame or st.read_preview()
        if frame is None:
            cc = self.cfg.get_camera(st.cam_id)
            h  = cc.frame_height if cc else 720
            w  = cc.frame_width  if cc else 1280
            frame = np.full((h, w, 3), 30, dtype=np.uint8)
            cv2.putText(frame,
                        "No live feed — drawing on blank frame",
                        (20, h // 2), FONT, 0.9, TEXT, 2)

        out_path = self.cfg.get_boundary_path(st.cam_id)
        self._in_editor = True

        # Clear main-window mouse callback so '+' cursor does not bleed back
        try:
            cv2.setMouseCallback(main_wname, lambda *a: None)
        except Exception:
            pass

        try:
            from gui.boundary_editor import run_boundary_editor
            saved = run_boundary_editor(st.cam_id, frame, out_path)
            if saved:
                st.reload_bd()
                self._send_reload(st.cam_id, str(out_path))
                logger.info("[GUI] Boundaries saved cam %d, reload signalled",
                            st.cam_id)
        except Exception as e:
            logger.error("[GUI] Editor error: %s", e, exc_info=True)
        finally:
            self._in_editor = False
            # Restore main-window mouse callback
            try:
                cv2.setMouseCallback(main_wname, self._ms.make_cb())
            except Exception:
                pass

    # ── queues ─────────────────────────────────────────────────────────────────
    def _drain(self):
        for _ in range(60):
            try:
                msg = self.result_q.get_nowait()
            except Exception:
                break
            if msg.get("type") in (MessageType.DETECTION_RESULT.value, "pool_result"):
                cid = msg.get("camera_id")
                if cid in self.cam_states:
                    self.cam_states[cid].update(msg)

        for _ in range(20):
            try:
                msg = self.relay_q.get_nowait()
            except Exception:
                break
            if msg.get("type") == MessageType.RELAY_STATE.value:
                s = msg.get("relay_states", [])
                self.relay_states = s + [False] * (9 - len(s))

        for _ in range(5):
            try:
                msg = self.health_q.get_nowait()
            except Exception:
                break
            if msg.get("type") == MessageType.HEALTH_SNAPSHOT.value:
                self.health = msg

    def _cmd(self, cmd: str):
        try:
            self.cmd_q.put_nowait(
                GUICommandMessage(camera_id=None, command=cmd).to_dict())
        except Exception:
            pass

    def _send_reload(self, cam_id: int, path: str):
        try:
            self.cmd_q.put_nowait(
                BoundaryReloadMessage(camera_id=cam_id, boundary_file=path).to_dict())
        except Exception:
            pass

    def _heartbeat(self):
        now = time.time()
        if now - self._last_hb >= 2.0:
            try:
                self.hb_q.put_nowait(
                    make_heartbeat(ProcessSource.GUI, None, "GUI",
                                   self.pid, get_process_memory_mb(), 0.0).to_dict())
            except Exception:
                pass
            self._last_hb = now


# ── process entry point ────────────────────────────────────────────────────────
def gui_process_entry(config_path, result_queue, relay_state_queue,
                      heartbeat_queue, gui_cmd_queue, health_queue,
                      stop_event, preview_mode_flag, log_dir="logs"):
    from core.config_loader import VisionSystemConfig
    cfg = VisionSystemConfig(config_path)
    setup_process_logging("gui", log_dir, cfg.system.log_level,
                          cfg.logging.max_bytes, cfg.logging.backup_count)
    setup_crash_handler("gui", log_dir)

    def _sig(s, f):
        stop_event.set()
    signal.signal(signal.SIGTERM, _sig)

    gui = UnifiedGUI(cfg, result_queue, relay_state_queue, heartbeat_queue,
                     gui_cmd_queue, health_queue, stop_event, preview_mode_flag)
    try:
        gui.run()
    except Exception as e:
        logger.critical("[gui] Fatal: %s", e, exc_info=True)
        sys.exit(0)   # GUI crash is non-fatal to production

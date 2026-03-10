"""
gui/boundary_editor.py  v3.2
=============================
Interactive OpenCV boundary editor.

Public API (used by tests and GUI):
  BoundaryEditorState       - mutable editor state
  _BOUNDARY_SEQUENCE        - [(id, type), ...] for OC1-3, BH1-3
  _save_boundaries(state, camera_id, path)  - serialize to JSON (overwrite)
  run_boundary_editor(camera_id, frame, output_path) -> bool

New in v3.2:
  • Undo last point  [U] or [Ctrl+Z]
  • Clear WIP points  [C]
  • Clear ALL boundaries  [X]
  • Replace-on-redraw: drawing an already-defined ID replaces it (no duplicates)
  • Overwrite save (never appends)
  • Cross-hair cursor only inside the editor window
  • Highlighted active boundary row in the side panel
  • Jump to any slot with [1]-[6]
  • Flashing "ALL DONE — press S" when all 6 are defined
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Colours ────────────────────────────────────────────────────────────────────
_COL_OC     = (60, 165, 255)   # orange (BGR)
_COL_BH     = (60, 200,  80)   # green
_COL_CURSOR = (0,  230, 255)   # yellow
_COL_TEXT   = (220, 220, 220)
_COL_LABEL  = (200, 200,  50)
_COL_OK     = ( 60, 210,  60)
_COL_WARN   = ( 40, 180, 220)
_COL_PANEL  = ( 22,  22,  20)
_COL_BORDER = ( 55,  55,  52)
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT2      = cv2.FONT_HERSHEY_DUPLEX

# ── Boundary sequence (public — used by tests) ─────────────────────────────────
_BOUNDARY_SEQUENCE: List[Tuple[str, str]] = [
    ("OC1", "oil_can"),
    ("OC2", "oil_can"),
    ("OC3", "oil_can"),
    ("BH1", "bunk_hole"),
    ("BH2", "bunk_hole"),
    ("BH3", "bunk_hole"),
]


# ── State class (public — used by tests) ───────────────────────────────────────
class BoundaryEditorState:
    """
    Mutable state for the boundary editor.
    Public attributes accessed by tests:
      .current_idx       int   index into _BOUNDARY_SEQUENCE
      .current_bid       str   e.g. "OC1"
      .current_type      str   e.g. "oil_can"
      .closed_boundaries list  [{"id","type_class","polygon":[...]}, ...]
      .all_defined()     bool  True when all 6 IDs are present
    """

    def __init__(self) -> None:
        self.current_points:    List[Tuple[int, int]] = []   # WIP polygon
        self.closed_boundaries: List[Dict]            = []   # completed polygons
        self.current_idx:       int                   = 0
        self.mouse_pos:         Tuple[int, int]       = (0, 0)
        self.done:              bool                  = False
        self.saved:             bool                  = False

    # ── derived properties ─────────────────────────────────────────────────────

    @property
    def current_bid(self) -> str:
        if self.current_idx < len(_BOUNDARY_SEQUENCE):
            return _BOUNDARY_SEQUENCE[self.current_idx][0]
        return f"B{self.current_idx}"

    @property
    def current_type(self) -> str:
        if self.current_idx < len(_BOUNDARY_SEQUENCE):
            return _BOUNDARY_SEQUENCE[self.current_idx][1]
        return "oil_can"

    def all_defined(self) -> bool:
        have = {b["id"] for b in self.closed_boundaries}
        return all(bid in have for bid, _ in _BOUNDARY_SEQUENCE)

    def get_boundary_by_id(self, bid: str) -> Optional[Dict]:
        return next((b for b in self.closed_boundaries if b["id"] == bid), None)

    # ── mutation helpers ───────────────────────────────────────────────────────

    def replace_or_add(self, bid: str, btype: str, pts: List) -> None:
        """Store polygon, replacing any existing entry with the same id."""
        self.closed_boundaries = [b for b in self.closed_boundaries if b["id"] != bid]
        self.closed_boundaries.append({
            "id":         bid,
            "type_class": btype,
            "polygon":    [list(p) for p in pts],
        })

    def delete_current(self) -> None:
        bid = self.current_bid
        self.closed_boundaries = [b for b in self.closed_boundaries if b["id"] != bid]
        self.current_points = []

    def undo_point(self) -> None:
        if self.current_points:
            self.current_points.pop()

    def clear_wip(self) -> None:
        self.current_points = []

    def clear_all(self) -> None:
        self.closed_boundaries.clear()
        self.current_points = []
        self.current_idx = 0


# ── Mouse callback ─────────────────────────────────────────────────────────────

def _mouse_callback(event, x, y, flags, param: BoundaryEditorState) -> None:
    param.mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        param.current_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        param.undo_point()


# ── Rendering ──────────────────────────────────────────────────────────────────

def _draw_editor_frame(base: np.ndarray, state: BoundaryEditorState) -> np.ndarray:
    canvas    = base.copy()
    h, w      = canvas.shape[:2]
    PANEL_W   = 320

    # ── closed boundaries ──────────────────────────────────────────────────────
    for b in state.closed_boundaries:
        pts   = np.array(b["polygon"], dtype=np.int32)
        col   = _COL_OC if b.get("type_class") == "oil_can" else _COL_BH
        ov    = canvas.copy()
        cv2.fillPoly(ov, [pts.reshape(-1, 1, 2)], col)
        cv2.addWeighted(ov, 0.18, canvas, 0.82, 0, canvas)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], True, col, 2, cv2.LINE_AA)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(canvas, b["id"], (cx - 18, cy + 1), _FONT2, 0.70, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, b["id"], (cx - 18, cy),     _FONT2, 0.70, col,       2, cv2.LINE_AA)

    # ── WIP polygon ────────────────────────────────────────────────────────────
    cur_col = _COL_OC if state.current_type == "oil_can" else _COL_BH
    pts_list = state.current_points
    if pts_list:
        for i, pt in enumerate(pts_list):
            cv2.circle(canvas, pt, 5, cur_col,       -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 5, (255,255,255),  1, cv2.LINE_AA)
            if i > 0:
                cv2.line(canvas, pts_list[i-1], pt, cur_col, 2, cv2.LINE_AA)
        cv2.line(canvas, pts_list[-1], state.mouse_pos, _COL_CURSOR, 1, cv2.LINE_AA)
        if len(pts_list) >= 2:
            cv2.line(canvas, pts_list[0], state.mouse_pos, cur_col, 1, cv2.LINE_AA)

    # ── cross-hair cursor ──────────────────────────────────────────────────────
    mx, my = state.mouse_pos
    cv2.line(canvas, (mx - 14, my), (mx + 14, my), _COL_CURSOR, 1, cv2.LINE_AA)
    cv2.line(canvas, (mx, my - 14), (mx, my + 14), _COL_CURSOR, 1, cv2.LINE_AA)
    cv2.circle(canvas, (mx, my), 3, _COL_CURSOR, -1, cv2.LINE_AA)

    # ── right panel ────────────────────────────────────────────────────────────
    panel = np.full((h, PANEL_W, 3), _COL_PANEL, dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, h - 1), _COL_BORDER, 1)

    def _pt(text, x, y, scale=0.46, color=_COL_TEXT, thick=1, font=_FONT):
        cv2.putText(panel, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

    oy = 24
    _pt("BOUNDARY EDITOR", 10, oy, 0.56, _COL_LABEL, 1, _FONT2);  oy += 28
    cv2.line(panel, (8, oy), (PANEL_W - 8, oy), _COL_BORDER, 1);  oy += 14

    target_col = _COL_OC if state.current_type == "oil_can" else _COL_BH
    _pt(f"Drawing: {state.current_bid}", 10, oy, 0.52, target_col, 1, _FONT2); oy += 20
    _pt(f"({state.current_type.replace('_',' ')})  pts: {len(state.current_points)}",
        10, oy, 0.40, (140, 140, 140)); oy += 22
    cv2.line(panel, (8, oy), (PANEL_W - 8, oy), _COL_BORDER, 1); oy += 12

    controls = [
        ("L-Click",   "Add point"),
        ("R-Click/U", "Undo last point"),
        ("ENTER",     "Close polygon (>=3 pts)"),
        ("1-6",       "Jump to OC1-3 / BH1-3"),
        ("N",         "Next boundary slot"),
        ("D",         "Delete current slot"),
        ("C",         "Clear WIP points"),
        ("X",         "Clear ALL boundaries"),
        ("S",         "Save & exit"),
        ("ESC",       "Cancel"),
    ]
    _pt("CONTROLS", 10, oy, 0.42, _COL_LABEL); oy += 18
    for key_lbl, desc in controls:
        _pt(key_lbl, 12, oy, 0.37, (100, 180, 255))
        _pt(desc, 90, oy, 0.37, (150, 150, 150))
        oy += 16

    cv2.line(panel, (8, oy), (PANEL_W - 8, oy), _COL_BORDER, 1); oy += 12
    _pt("BOUNDARIES", 10, oy, 0.42, _COL_LABEL); oy += 18

    for seq_i, (bid, btype) in enumerate(_BOUNDARY_SEQUENCE):
        b          = state.get_boundary_by_id(bid)
        is_active  = (seq_i == state.current_idx)
        col        = _COL_OC if btype == "oil_can" else _COL_BH
        dim        = (70, 70, 70)

        if is_active:
            cv2.rectangle(panel, (6, oy - 13), (PANEL_W - 6, oy + 6), (42, 42, 40), -1)
            cv2.rectangle(panel, (6, oy - 13), (PANEL_W - 6, oy + 6), col, 1)

        tick     = ">" if is_active and not b else ("✓" if b else " ")
        pts_info = f"({len(b['polygon'])} pts)" if b else ""
        tick_col = col if (b or is_active) else dim

        _pt(f"{tick} {bid}", 12, oy, 0.48, tick_col, 1, _FONT2)
        _pt(pts_info,        76, oy, 0.38, (115, 115, 115))
        oy += 22

    oy += 8
    cv2.line(panel, (8, oy), (PANEL_W - 8, oy), _COL_BORDER, 1); oy += 14

    if state.all_defined():
        flash = int(time.time() * 2) % 2 == 0
        bg    = _COL_OK if flash else (35, 110, 35)
        cv2.rectangle(panel, (8, oy - 14), (PANEL_W - 8, oy + 8), bg, -1)
        _pt("ALL DONE  -- press S to save!", 12, oy, 0.44, (0, 0, 0), 1, _FONT2)
    else:
        missing = [bid for bid, _ in _BOUNDARY_SEQUENCE if not state.get_boundary_by_id(bid)]
        _pt(f"Missing: {', '.join(missing)}", 10, oy, 0.38, _COL_WARN)

    # ── combine ────────────────────────────────────────────────────────────────
    out = np.zeros((h, w + PANEL_W, 3), dtype=np.uint8)
    out[:, :w]  = canvas
    out[:, w:]  = panel
    return out


# ── Save helper (public — used by tests) ───────────────────────────────────────

def _save_boundaries(state: BoundaryEditorState,
                     camera_id: int,
                     output_path: Path) -> bool:
    """
    Serialise state.closed_boundaries to v2 flat JSON.
    Always overwrites the file — no appending, no duplicates.
    """
    oil_can:   List[Dict] = []
    bunk_hole: List[Dict] = []

    for b in state.closed_boundaries:
        entry = {"id": b["id"], "polygon": b["polygon"]}
        if b.get("type_class") == "oil_can":
            oil_can.append(entry)
        else:
            bunk_hole.append(entry)

    data = {
        "camera_id": camera_id,
        "oil_can":   oil_can,
        "bunk_hole": bunk_hole,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Boundaries saved → %s  (%d OC, %d BH)",
                    output_path, len(oil_can), len(bunk_hole))
        return True
    except Exception as e:
        logger.error("Failed to save boundaries: %s", e)
        return False


# ── Load helper ────────────────────────────────────────────────────────────────

def _load_existing_boundaries(state: BoundaryEditorState, bd_path: Path) -> None:
    """
    Pre-populate state from an existing boundary file.
    Uses replace_or_add to guarantee no duplicates.
    Advances current_idx to the first undefined slot.
    """
    if not bd_path.exists():
        return
    try:
        with open(bd_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Cannot load existing boundaries: %s", e)
        return

    for item in data.get("oil_can", []):
        state.replace_or_add(item["id"], "oil_can",
                             item.get("polygon", item.get("points", [])))
    for item in data.get("bunk_hole", []):
        state.replace_or_add(item["id"], "bunk_hole",
                             item.get("polygon", item.get("points", [])))

    # Advance cursor to first undefined slot
    defined = {b["id"] for b in state.closed_boundaries}
    for i, (bid, _) in enumerate(_BOUNDARY_SEQUENCE):
        if bid not in defined:
            state.current_idx = i
            return
    # All already defined — stay on last
    state.current_idx = len(_BOUNDARY_SEQUENCE) - 1

    logger.info("Loaded %d boundaries from %s", len(state.closed_boundaries), bd_path)


# ── Public entry point ─────────────────────────────────────────────────────────

def run_boundary_editor(camera_id: int,
                        frame: np.ndarray,
                        output_path: Path,
                        window_name: str = "Boundary Editor") -> bool:
    """
    Open the interactive boundary editor on a frozen frame.
    Returns True if the operator saved, False if cancelled.
    The output file is always overwritten (never appended to).
    """
    state = BoundaryEditorState()
    _load_existing_boundaries(state, Path(output_path))

    wname = f"{window_name} — Camera {camera_id}"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wname, 1600, 900)
    # Cross-hair cursor only exists inside this editor window
    cv2.setMouseCallback(wname, _mouse_callback, state)

    logger.info("Boundary editor open — camera %d", camera_id)

    while not state.done:
        display = _draw_editor_frame(frame, state)
        cv2.imshow(wname, display)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:      # ESC — cancel
            state.saved = False
            state.done  = True

        elif key == 13:    # ENTER — close polygon
            if len(state.current_points) >= 3:
                state.replace_or_add(state.current_bid, state.current_type,
                                     state.current_points)
                logger.info("Closed polygon %s (%d pts)",
                            state.current_bid, len(state.current_points))
                state.current_points = []
                if state.current_idx < len(_BOUNDARY_SEQUENCE) - 1:
                    state.current_idx += 1
            else:
                logger.warning("Need ≥3 points (have %d)", len(state.current_points))

        elif key in (ord('u'), ord('U'), 26):   # U / Ctrl+Z
            state.undo_point()

        elif key in (ord('c'), ord('C')):
            state.clear_wip()

        elif key in (ord('x'), ord('X')):
            state.clear_all()
            logger.info("All boundaries cleared")

        elif key in (ord('n'), ord('N')):
            state.current_points = []
            state.current_idx = min(state.current_idx + 1, len(_BOUNDARY_SEQUENCE) - 1)

        elif key in (ord('d'), ord('D')):
            state.delete_current()

        elif key in (ord('s'), ord('S')):
            if state.all_defined():
                ok = _save_boundaries(state, camera_id, Path(output_path))
                state.saved = ok
                state.done  = True
                if ok:
                    confirm = _draw_editor_frame(frame, state)
                    cv2.putText(confirm, "SAVED!", (50, 70),
                                _FONT2, 2.2, (0, 230, 80), 3, cv2.LINE_AA)
                    cv2.imshow(wname, confirm)
                    cv2.waitKey(1200)
            else:
                missing = [bid for bid, _ in _BOUNDARY_SEQUENCE
                           if not state.get_boundary_by_id(bid)]
                logger.warning("Cannot save — missing: %s", missing)

        elif ord('1') <= key <= ord('6'):
            state.current_idx    = key - ord('1')
            state.current_points = []

    cv2.destroyWindow(wname)
    return state.saved


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out = Path(__file__).parent.parent / "boundaries" / f"camera_{cam}_boundaries.json"
    img = np.full((720, 1280, 3), 32, dtype=np.uint8)
    cv2.putText(img, f"Camera {cam} — test frame",
                (40, 360), _FONT2, 1.2, (200, 200, 200), 2, cv2.LINE_AA)
    saved = run_boundary_editor(cam, img, out)
    print(f"saved={saved}")
    if saved and out.exists():
        with open(out) as f:
            print(json.dumps(json.load(f), indent=2))

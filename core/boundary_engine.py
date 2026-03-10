"""
core/boundary_engine.py
=======================
Full boundary pairing logic engine.
Supports:
  - Polygon boundaries (arbitrary convex/concave)
  - Rectangular fallback boundaries
  - Strict and relaxed detection modes
  - Pair status evaluation (OK / PROBLEM / BOTH_ABSENT / OC_ONLY / BH_ONLY)
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.ipc_schema import DetectionObject, PairResult, PairStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Return centre (cx, cy) of a bbox in pixel or normalised coords."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_in_polygon(px: float, py: float, polygon: List[Tuple[int, int]]) -> bool:
    """
    Test if point (px, py) is inside a polygon using OpenCV pointPolygonTest.
    Returns True if inside or on boundary.
    """
    pts = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    result = cv2.pointPolygonTest(pts, (float(px), float(py)), measureDist=False)
    return result >= 0


def point_in_rect(px: float, py: float,
                  x1: float, y1: float,
                  x2: float, y2: float) -> bool:
    return x1 <= px <= x2 and y1 <= py <= y2


def bbox_iou_with_polygon(bbox: Tuple[float, float, float, float],
                           polygon: List[Tuple[int, int]]) -> float:
    """
    Approximate IoU between a bbox rect and a polygon
    by checking if the bbox centre is inside the polygon.
    For strict mode, we check all 4 corners + centre.
    """
    x1, y1, x2, y2 = bbox
    test_points = [
        bbox_center(bbox),
        (x1, y1), (x2, y1), (x1, y2), (x2, y2),
    ]
    hits = sum(1 for px, py in test_points if point_in_polygon(px, py, polygon))
    return hits / len(test_points)


# ---------------------------------------------------------------------------
# Boundary definition
# ---------------------------------------------------------------------------

class Boundary:
    def __init__(self, boundary_def: Dict):
        self.id: str = boundary_def["id"]
        self.name: str = boundary_def.get("name", self.id)
        self.type: str = boundary_def.get("type", "polygon")   # polygon | rect
        self.pair: str = boundary_def.get("pair", "")

        if self.type == "polygon":
            self.points: List[Tuple[int, int]] = [
                (int(p[0]), int(p[1])) for p in boundary_def["points"]
            ]
            self.rect: Optional[Tuple[int, int, int, int]] = None
        else:
            pts = boundary_def.get("points", [[0, 0], [100, 100]])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            self.rect = (min(xs), min(ys), max(xs), max(ys))
            self.points = []

    def contains_detection(self, det: DetectionObject,
                            frame_w: int, frame_h: int,
                            strict: bool = True) -> bool:
        """
        Check if a detection's bounding box intersects this boundary.
        bbox is normalised [0,1] → convert to pixels.
        """
        x1n, y1n, x2n, y2n = det.bbox
        x1 = x1n * frame_w
        y1 = y1n * frame_h
        x2 = x2n * frame_w
        y2 = y2n * frame_h
        px, py = (x1 + x2) / 2, (y1 + y2) / 2

        if self.type == "polygon":
            if strict:
                return point_in_polygon(px, py, self.points)
            else:
                # Relaxed: any corner inside counts
                return bbox_iou_with_polygon((x1, y1, x2, y2), self.points) > 0
        else:
            rx1, ry1, rx2, ry2 = self.rect
            return point_in_rect(px, py, rx1, ry1, rx2, ry2)

    def get_draw_points(self) -> Optional[np.ndarray]:
        if self.type == "polygon" and self.points:
            return np.array(self.points, dtype=np.int32)
        elif self.rect:
            x1, y1, x2, y2 = self.rect
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        return None


# ---------------------------------------------------------------------------
# Boundary set for one camera
# ---------------------------------------------------------------------------

class CameraBoundarySet:
    def __init__(self, boundary_data: Dict, strict_mode: bool = True):
        self.camera_id: int = boundary_data.get("camera_id", 0)
        self.strict_mode: bool = strict_mode

        raw_oc = boundary_data.get("boundaries", {}).get("oil_can", [])
        raw_bh = boundary_data.get("boundaries", {}).get("bunk_hole", [])

        self.oc_boundaries: Dict[str, Boundary] = {
            b["id"]: Boundary(b) for b in raw_oc
        }
        self.bh_boundaries: Dict[str, Boundary] = {
            b["id"]: Boundary(b) for b in raw_bh
        }
        self.pairs: List[Dict] = boundary_data.get("pairs", [])
        logger.info("CameraBoundarySet loaded: cam=%d oc=%d bh=%d pairs=%d",
                    self.camera_id,
                    len(self.oc_boundaries),
                    len(self.bh_boundaries),
                    len(self.pairs))

    def evaluate(self,
                 detections: List[DetectionObject],
                 frame_w: int,
                 frame_h: int,
                 oc_class_id: int,
                 bh_class_id: int) -> List[PairResult]:
        """
        Run full boundary pairing logic against detected objects.
        Returns one PairResult per configured pair.
        """
        # Separate detections by class
        oc_dets = [d for d in detections if d.class_id == oc_class_id]
        bh_dets = [d for d in detections if d.class_id == bh_class_id]

        results: List[PairResult] = []
        for pair_def in self.pairs:
            pair_id = pair_def["id"]
            oc_bid = pair_def["oil_can_boundary"]
            bh_bid = pair_def["bunk_hole_boundary"]
            relay_idx = pair_def.get("relay_index", pair_id)

            oc_boundary = self.oc_boundaries.get(oc_bid)
            bh_boundary = self.bh_boundaries.get(bh_bid)

            oc_present = False
            bh_present = False

            if oc_boundary:
                for det in oc_dets:
                    if oc_boundary.contains_detection(det, frame_w, frame_h, self.strict_mode):
                        oc_present = True
                        det.boundary_id = oc_bid
                        break

            if bh_boundary:
                for det in bh_dets:
                    if bh_boundary.contains_detection(det, frame_w, frame_h, self.strict_mode):
                        bh_present = True
                        det.boundary_id = bh_bid
                        break

            status = self._evaluate_pair(oc_present, bh_present)
            relay_active = status != PairStatus.OK

            results.append(PairResult(
                pair_id=pair_id,
                pair_name=pair_def.get("name", f"Pair {pair_id + 1}"),
                oil_can_boundary=oc_bid,
                bunk_hole_boundary=bh_bid,
                oil_can_present=oc_present,
                bunk_hole_present=bh_present,
                status=status,
                relay_index=relay_idx,
                relay_active=relay_active,
            ))

        return results

    @staticmethod
    def _evaluate_pair(oc_present: bool, bh_present: bool) -> PairStatus:
        """
        Pair evaluation logic:
          OK            → both present
          BOTH_ABSENT   → neither present (PROBLEM)
          OC_ONLY       → oil_can present, bunk_hole missing (PROBLEM)
          BH_ONLY       → bunk_hole present, oil_can missing (PROBLEM)
        """
        if oc_present and bh_present:
            return PairStatus.OK
        elif not oc_present and not bh_present:
            return PairStatus.BOTH_ABSENT
        elif oc_present and not bh_present:
            return PairStatus.OC_ONLY
        else:
            return PairStatus.BH_ONLY

    def get_all_boundaries(self) -> Dict[str, List[Boundary]]:
        return {
            "oil_can": list(self.oc_boundaries.values()),
            "bunk_hole": list(self.bh_boundaries.values()),
        }

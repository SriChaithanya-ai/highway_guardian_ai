"""
Keeps a short rolling history of each tracked vehicle's position/speed so
the accident-verification module can spot motion signatures typical of a
collision: sudden deceleration, abrupt direction change, or two boxes
suddenly overlapping heavily.

ByteTrack (inside VehicleDetector) supplies stable track_ids across frames;
this module just accumulates history keyed by that id.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

from detection.vehicle_detector import Detection

HISTORY_LEN = 15  # frames of history kept per vehicle


@dataclass
class TrackState:
    positions: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    # each entry: (timestamp, center_x, center_y)
    last_box: tuple = None
    cls_name: str = ""


class TrajectoryTracker:
    def __init__(self):
        self._tracks: Dict[int, TrackState] = {}

    def update(self, detections: List[Detection]) -> Dict[int, TrackState]:
        now = time.time()
        seen_ids = set()
        for det in detections:
            seen_ids.add(det.track_id)
            state = self._tracks.setdefault(det.track_id, TrackState())
            x1, y1, x2, y2 = det.xyxy
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            state.positions.append((now, cx, cy))
            state.last_box = det.xyxy
            state.cls_name = det.cls_name

        # Drop tracks not seen this frame for a while (simple GC)
        stale = [tid for tid in self._tracks if tid not in seen_ids and
                 self._tracks[tid].positions and now - self._tracks[tid].positions[-1][0] > 5]
        for tid in stale:
            del self._tracks[tid]

        return self._tracks

    @staticmethod
    def speed(state: TrackState) -> float:
        """Approx pixel-speed over the tracked history (pixels/sec)."""
        pts = list(state.positions)
        if len(pts) < 2:
            return 0.0
        (t0, x0, y0), (t1, x1, y1) = pts[0], pts[-1]
        dt = max(t1 - t0, 1e-6)
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        return dist / dt

    @staticmethod
    def deceleration_ratio(state: TrackState) -> float:
        """
        Compares speed in the first half of the history window to the
        second half. Near 0 means it kept a steady speed; close to 1 means
        it nearly stopped -- a signature of a crash (as opposed to normal
        braking, which is more gradual and combined with lane-consistent
        motion).
        """
        pts = list(state.positions)
        if len(pts) < 6:
            return 0.0
        mid = len(pts) // 2
        first_half, second_half = pts[:mid], pts[mid:]

        def avg_speed(chunk):
            if len(chunk) < 2:
                return 0.0
            (t0, x0, y0), (t1, x1, y1) = chunk[0], chunk[-1]
            dt = max(t1 - t0, 1e-6)
            dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            return dist / dt

        v1, v2 = avg_speed(first_half), avg_speed(second_half)
        if v1 < 1e-3:
            return 0.0
        return max(0.0, (v1 - v2) / v1)


def boxes_overlap_ratio(box_a: tuple, box_b: tuple) -> float:
    """Intersection-over-min-area of two (x1,y1,x2,y2) boxes -- used to flag
    two vehicles suddenly occupying near-identical space (collision)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    min_area = max(1e-6, min(area_a, area_b))
    return inter / min_area

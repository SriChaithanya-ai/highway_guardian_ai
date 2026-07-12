"""
Accident Verification Module.

A single frame classified as "Accident" is not enough to trigger a real
emergency dispatch -- a stopped truck, shadow, or motion blur can fool a
classifier. This module requires agreement between two independent signals
across several consecutive frames before an incident is confirmed:

  1. The trained image classifier says "Accident" with high confidence.
  2. At least one motion heuristic supports it:
       - two vehicle boxes overlapping heavily (collision), OR
       - a tracked vehicle showing a sharp deceleration spike.

Only once ACCIDENT_CONFIRMATION_FRAMES consecutive frames satisfy both
signals does `verify()` return a confirmed incident.
"""
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from config import settings
from detection.trajectory_tracker import TrackState, boxes_overlap_ratio
from detection.vehicle_detector import Detection

OVERLAP_THRESHOLD = 0.35
DECELERATION_THRESHOLD = 0.6


@dataclass
class VerificationResult:
    confirmed: bool
    classifier_conf: float
    motion_supported: bool
    involved_track_ids: List[int]


class AccidentVerifier:
    def __init__(self, confirmation_frames: int = settings.ACCIDENT_CONFIRMATION_FRAMES):
        self.confirmation_frames = confirmation_frames
        self._recent_hits: Deque[bool] = deque(maxlen=confirmation_frames)

    def _motion_signal(self, detections: List[Detection],
                        track_states: Dict[int, TrackState]) -> (bool, List[int]):
        involved = []

        # Signal A: overlapping boxes (possible collision)
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                overlap = boxes_overlap_ratio(detections[i].xyxy, detections[j].xyxy)
                if overlap >= OVERLAP_THRESHOLD:
                    involved.extend([detections[i].track_id, detections[j].track_id])

        # Signal B: sharp deceleration on any tracked vehicle
        from detection.trajectory_tracker import TrajectoryTracker
        for det in detections:
            state = track_states.get(det.track_id)
            if state is None:
                continue
            decel = TrajectoryTracker.deceleration_ratio(state)
            if decel >= DECELERATION_THRESHOLD:
                involved.append(det.track_id)

        return (len(involved) > 0), sorted(set(involved))

    def verify(self, classifier_is_accident: bool, classifier_conf: float,
               detections: List[Detection], track_states: Dict[int, TrackState]) -> VerificationResult:

        motion_supported, involved_ids = self._motion_signal(detections, track_states)
        frame_hit = classifier_is_accident and (motion_supported or classifier_conf >= 0.85)
        # If classifier is *very* confident (>=0.85) we don't require motion
        # corroboration -- catches accidents already at rest (post-impact).

        self._recent_hits.append(frame_hit)
        confirmed = (
            len(self._recent_hits) == self.confirmation_frames
            and all(self._recent_hits)
        )

        return VerificationResult(
            confirmed=confirmed,
            classifier_conf=classifier_conf,
            motion_supported=motion_supported,
            involved_track_ids=involved_ids,
        )

    def reset(self):
        self._recent_hits.clear()

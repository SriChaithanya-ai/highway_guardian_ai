"""
inference_engine.py
--------------------
This module does NOT duplicate any detection/classification logic --
it directly reuses the same classes the production pipeline uses
(detection/vehicle_detector.py, accident/accident_classifier.py,
accident/severity_classifier.py, accident/verification.py,
detection/trajectory_tracker.py). That's what makes this a genuine
end-to-end test of the real pipeline, not a mocked-up demo.

The only thing this module does NOT do is call emergency services --
that's intentionally left out of the default path since this tool is for
pre-deployment testing. A `simulate_dispatch` flag is available if you want
to test the full chain including notifier.py in dry-run (Twilio/SMTP will
only actually send if you've configured real credentials in .env; without
them, dispatch calls are logged as skipped, same as in production).
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from accident.accident_classifier import AccidentClassifier
from accident.severity_classifier import SeverityClassifier
from accident.verification import AccidentVerifier
from detection.trajectory_tracker import TrajectoryTracker
from detection.vehicle_detector import Detection, VehicleDetector

logger = logging.getLogger("testing_module.inference_engine")


@dataclass
class FrameResult:
    detections: List[Detection] = field(default_factory=list)
    is_accident_frame: bool = False
    classifier_conf: float = 0.0
    classifier_label: str = ""
    motion_supported: bool = False
    confirmed: bool = False           # True only once temporal confirmation passes
    severity_label: Optional[str] = None
    severity_conf: Optional[float] = None


class InferenceEngine:
    """
    Loads all trained/pretrained models ONCE (expensive) and exposes a
    single process_frame() call that runs the full detect -> track ->
    classify -> verify -> (severity) chain on one frame, mirroring
    pipeline/main_pipeline.py exactly.
    """

    def __init__(self):
        logger.info("Loading models for testing module (this can take a few seconds)...")
        self.vehicle_detector = VehicleDetector()
        self.accident_classifier = AccidentClassifier()
        self.severity_classifier = SeverityClassifier()
        self.tracker = TrajectoryTracker()
        self.verifier = AccidentVerifier()
        logger.info("Models loaded. Testing module ready.")

    def process_frame(self, frame) -> FrameResult:
        detections = self.vehicle_detector.track(frame)
        track_states = self.tracker.update(detections)

        is_accident, conf, label = self.accident_classifier.predict(frame)
        verification = self.verifier.verify(is_accident, conf, detections, track_states)

        result = FrameResult(
            detections=detections,
            is_accident_frame=is_accident,
            classifier_conf=conf,
            classifier_label=label,
            motion_supported=verification.motion_supported,
            confirmed=verification.confirmed,
        )

        if verification.confirmed:
            severity_label, severity_conf = self.severity_classifier.predict(frame)
            result.severity_label = severity_label
            result.severity_conf = severity_conf
            self.verifier.reset()  # matches production behavior: reset after confirming

        return result

    def reset(self):
        """Call this whenever the input source is switched, so leftover
        track history / confirmation streaks from the old source don't
        bleed into the new one."""
        self.tracker = TrajectoryTracker()
        self.verifier.reset()

"""
Wraps the trained severity classification model
(Minor Impact / Substantial Impact / Critical Impact).
"""
import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from config import settings

logger = logging.getLogger("accident.severity_classifier")


class SeverityClassifier:
    def __init__(self, weights: str = settings.SEVERITY_CLASSIFIER_WEIGHTS):
        self.model = None
        self.names = {}
        if not Path(weights).exists():
            logger.warning(
                "Severity classifier weights not found at %s. "
                "Train it first with training/train_severity_classifier.py -- "
                "severity will report 'Unknown' until then.", weights,
            )
            return
        self.model = YOLO(weights)
        self.names = self.model.names  # e.g. {0: 'Critical', 1: 'Minor', 2: 'Substantial'}

    def predict(self, frame: np.ndarray) -> tuple:
        """
        Returns (label: str, confidence: float)
        label is one of "Minor Impact", "Substantial Impact", "Critical Impact",
        or "Unknown" if no trained weights are loaded yet.
        """
        if self.model is None:
            return "Unknown", 0.0

        results = self.model.predict(frame, verbose=False)
        r = results[0]
        probs = r.probs
        top1_idx = int(probs.top1)
        confidence = float(probs.top1conf)
        raw_label = self.names[top1_idx]  # "Minor" / "Substantial" / "Critical"

        label_map = {
            "Minor": "Minor Impact",
            "Substantial": "Substantial Impact",
            "Critical": "Critical Impact",
        }
        label = label_map.get(raw_label, raw_label)
        return label, confidence

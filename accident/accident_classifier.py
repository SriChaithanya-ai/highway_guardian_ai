"""
Wraps the trained Accident/NonAccident YOLO classification model.
Runs on the full frame (or a cropped region around a cluster of vehicles).
"""
import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from config import settings

logger = logging.getLogger("accident.accident_classifier")


class AccidentClassifier:
    def __init__(self, weights: str = settings.ACCIDENT_CLASSIFIER_WEIGHTS):
        self.model = None
        self.names = {}
        if not Path(weights).exists():
            logger.warning(
                "Accident classifier weights not found at %s. "
                "Train it first with training/train_accident_classifier.py -- "
                "running in vehicle-detection-only mode until then (accident "
                "detection will always report 'NotTrained').", weights,
            )
            return
        self.model = YOLO(weights)
        # model.names e.g. {0: 'Accident', 1: 'NonAccident'}
        self.names = self.model.names

    def predict(self, frame: np.ndarray) -> tuple:
        """
        Returns (is_accident: bool, confidence: float, label: str)
        If no trained weights are loaded, always returns a safe "no accident"
        result labeled "NotTrained" rather than crashing.
        """
        if self.model is None:
            return False, 0.0, "NotTrained"

        results = self.model.predict(frame, verbose=False)
        r = results[0]
        probs = r.probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        label = self.names[top1_idx]
        is_accident = (label.lower() == "accident" and top1_conf >= settings.ACCIDENT_CONF_THRESHOLD)
        return is_accident, top1_conf, label

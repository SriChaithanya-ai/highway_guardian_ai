"""
Vehicle detection using a pretrained (COCO) YOLOv8 model, filtered to
vehicle classes only: car, motorcycle, bus, truck.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

from config import settings


@dataclass
class Detection:
    track_id: int
    cls_name: str
    conf: float
    xyxy: tuple  # (x1, y1, x2, y2) in pixel coords


class VehicleDetector:
    """
    Thin wrapper around Ultralytics YOLO that also runs multi-object
    tracking (ByteTrack, built into Ultralytics) so each vehicle keeps a
    stable ID across frames -- required for accident heuristics that look
    at trajectory changes (sudden stop, collision overlap, etc.).
    """

    def __init__(self, weights: str = settings.VEHICLE_DETECTOR_WEIGHTS,
                 conf_threshold: float = settings.VEHICLE_CONF_THRESHOLD):
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold
        self.vehicle_classes = settings.VEHICLE_CLASSES  # {2: car, 3: motorcycle, 5: bus, 7: truck}

    def track(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs detection + ByteTrack tracking on a single frame.
        Returns only vehicle-class detections above the confidence threshold.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            classes=list(self.vehicle_classes.keys()),
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int)

        for box, conf, cls_id, track_id in zip(boxes, confs, clss, ids):
            cls_name = self.vehicle_classes.get(int(cls_id), "vehicle")
            detections.append(
                Detection(
                    track_id=int(track_id),
                    cls_name=cls_name,
                    conf=float(conf),
                    xyxy=tuple(box.tolist()),
                )
            )
        return detections

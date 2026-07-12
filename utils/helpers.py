import datetime as dt
from pathlib import Path

import cv2
import numpy as np

from config import settings


def save_snapshot(frame: np.ndarray, camera_id: str) -> str:
    """Saves an incident snapshot to disk and returns its path (used as
    photographic evidence attached to the incident record)."""
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{camera_id}_{timestamp}.jpg"
    path = settings.SNAPSHOT_DIR / filename
    cv2.imwrite(str(path), frame)
    return str(path)


def draw_detections(frame: np.ndarray, detections, color=(0, 255, 0)) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.xyxy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"#{det.track_id} {det.cls_name} {det.conf:.2f}"
        cv2.putText(annotated, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated


def draw_incident_banner(frame: np.ndarray, text: str) -> np.ndarray:
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    cv2.rectangle(annotated, (0, 0), (w, 40), (0, 0, 255), -1)
    cv2.putText(annotated, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return annotated

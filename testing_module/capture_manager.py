"""
capture_manager.py
-------------------
Saves a screenshot the moment an accident is CONFIRMED (not on every
low-confidence flicker). Screenshots go into testing_module/test_captures/
-- deliberately separate from static/snapshots/ (the production evidence
folder) so test runs never mix with real incident evidence.
"""
import datetime as dt
from pathlib import Path

import cv2

CAPTURE_DIR = Path(__file__).resolve().parent / "test_captures"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


def save_accident_frame(frame, source_label: str) -> str:
    """
    Saves the given frame as a JPEG evidence screenshot.
    Returns the path (relative to testing_module/) for logging/display.
    """
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_source = "".join(c if c.isalnum() else "_" for c in source_label)[:40]
    filename = f"accident_{safe_source}_{timestamp}.jpg"
    path = CAPTURE_DIR / filename
    cv2.imwrite(str(path), frame)
    return filename  # just the filename; served via /test_captures/<filename>

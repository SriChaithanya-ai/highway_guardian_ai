"""
test_logger.py
----------------
Two responsibilities:

1. Durable logging: every confirmed accident during a test run gets
   appended to a dated CSV file under testing_module/test_logs/, so you
   have a persistent audit trail of test sessions.
2. In-memory history: a thread-safe rolling deque the Flask app can read
   from to populate the "Detection History" table in the GUI without
   re-reading the CSV on every poll.

Kept deliberately separate from database/incident_db.py (the production
SQLite log) so test runs never get mixed in with real incident records.
"""
import csv
import datetime as dt
import threading
from collections import deque
from pathlib import Path
from typing import List

LOG_DIR = Path(__file__).resolve().parent / "test_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_HISTORY_MAXLEN = 200
_history = deque(maxlen=_HISTORY_MAXLEN)
_lock = threading.Lock()

_CSV_FIELDS = ["timestamp", "source", "severity", "severity_confidence",
               "classifier_confidence", "vehicle_count", "screenshot"]


def _csv_path_for_today() -> Path:
    return LOG_DIR / f"test_session_{dt.date.today().isoformat()}.csv"


def log_detection(source_label: str, severity: str, severity_confidence: float,
                   classifier_confidence: float, vehicle_count: int, screenshot_filename: str) -> dict:
    """Records one confirmed-accident event to both the CSV file and the
    in-memory history used by the GUI. Returns the record dict (includes an
    auto-incrementing id used by the front-end to detect "new" alerts)."""
    timestamp = dt.datetime.now().isoformat(timespec="seconds")

    record = {
        "timestamp": timestamp,
        "source": source_label,
        "severity": severity,
        "severity_confidence": round(severity_confidence, 3),
        "classifier_confidence": round(classifier_confidence, 3),
        "vehicle_count": vehicle_count,
        "screenshot": screenshot_filename,
    }

    csv_path = _csv_path_for_today()
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    with _lock:
        record_with_id = {"id": len(_history) + 1, **record}
        _history.append(record_with_id)

    return record_with_id


def get_recent(limit: int = 50) -> List[dict]:
    with _lock:
        return list(_history)[-limit:][::-1]  # most recent first


def get_last() -> dict:
    with _lock:
        return _history[-1] if _history else None


def clear_history():
    with _lock:
        _history.clear()

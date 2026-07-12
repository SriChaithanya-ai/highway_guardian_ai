"""
Central configuration for Highway Guardian AI.
Loads everything from environment variables (.env file) so no secret
ever needs to be hard-coded in source.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val not in (None, "") else default


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val not in (None, "") else default


# ---------------- Paths ----------------
MODELS_DIR = ROOT_DIR / "models"
SNAPSHOT_DIR = ROOT_DIR / "static" / "snapshots"
DB_PATH = ROOT_DIR / "database" / "incidents.db"
CAMERAS_CONFIG_PATH = ROOT_DIR / "config" / "cameras.json"
POLICE_DIRECTORY_PATH = ROOT_DIR / "config" / "local_emergency_directory.json"

VEHICLE_DETECTOR_WEIGHTS = str(MODELS_DIR / "yolov8n.pt")  # pretrained COCO model, auto-downloaded
ACCIDENT_CLASSIFIER_WEIGHTS = str(MODELS_DIR / "accident_classifier" / "weights" / "best.pt")
SEVERITY_CLASSIFIER_WEIGHTS = str(MODELS_DIR / "severity_classifier" / "weights" / "best.pt")

# ---------------- Camera / stream ----------------
RTSP_URL = os.getenv("RTSP_URL", "0")  # "0" = default webcam for local testing
CAMERA_ID = os.getenv("CAMERA_ID", "CAM_DEFAULT")
CAMERA_LAT = _get_float("CAMERA_LAT", 0.0)
CAMERA_LON = _get_float("CAMERA_LON", 0.0)
FRAME_SAMPLE_RATE = _get_int("FRAME_SAMPLE_RATE", 5)  # process every Nth frame for speed

# ---------------- Detection thresholds ----------------
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO class ids
VEHICLE_CONF_THRESHOLD = 0.35
ACCIDENT_CONF_THRESHOLD = 0.55
ACCIDENT_CONFIRMATION_FRAMES = _get_int("ACCIDENT_CONFIRMATION_FRAMES", 5)
INCIDENT_COOLDOWN_SECONDS = _get_int("INCIDENT_COOLDOWN_SECONDS", 300)

SEVERITY_LABELS = {0: "Minor Impact", 1: "Substantial Impact", 2: "Critical Impact"}

# ---------------- Emergency services ----------------
NEAREST_SERVICE_RADIUS_M = _get_int("NEAREST_SERVICE_RADIUS_M", 8000)
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"

# ---------------- Twilio ----------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")

# ---------------- Email ----------------
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = _get_int("SMTP_PORT", 587)
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "")

for _d in (MODELS_DIR, SNAPSHOT_DIR, DB_PATH.parent):
    _d.mkdir(parents=True, exist_ok=True)

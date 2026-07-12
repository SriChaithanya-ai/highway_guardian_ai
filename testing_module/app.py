"""
testing_module/app.py
-----------------------
Real-Time Camera Testing Module for Highway Guardian AI.

Purpose: let you verify the FULL pipeline -- camera input -> vehicle
detection -> accident classification -> temporal verification -> severity
classification -> alerting -- works correctly using a webcam, a phone
camera (DroidCam / IP Webcam), or a recorded test video, before pointing
any of this at a real highway CCTV feed.

Run from the project root:
    python -m testing_module.app

Then open http://localhost:5001 in a browser.

Architecture notes:
  - video_sources.py    -> swappable input (webcam / video file / IP camera).
                            Later, pointing this at an RTSP CCTV URL is a
                            one-line change (source_type="ip_camera",
                            source_value="rtsp://...").
  - inference_engine.py -> reuses the EXACT SAME model classes as
                            pipeline/main_pipeline.py, so a pass here means
                            the real pipeline logic works, not a mock of it.
  - capture_manager.py  -> saves accident screenshots to a dedicated
                            test-only folder (kept separate from production
                            evidence).
  - test_logger.py      -> CSV logging + in-memory history for the GUI.
  - This module intentionally does NOT call real emergency services unless
    you explicitly tick "Simulate emergency dispatch" in the GUI -- and even
    then, calls/SMS only actually go out if Twilio credentials are set in
    .env (same safety behavior as production).
"""
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory, render_template

# Make the project root importable (this file lives in testing_module/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import settings  # noqa: E402
from testing_module import capture_manager, test_logger  # noqa: E402
from testing_module.fps_counter import FPSCounter  # noqa: E402
from testing_module.inference_engine import InferenceEngine  # noqa: E402
from testing_module.video_sources import VideoSource  # noqa: E402
from utils.helpers import draw_detections, draw_incident_banner  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("testing_module.app")

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------------------------------
# Global (single-session) state.
# This tool is meant for one engineer testing one camera at a time, so a
# simple guarded global state is intentional -- not built for concurrent
# multi-user access.
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_state = {
    "video_source": None,          # VideoSource instance
    "source_label": "no source",
    "running": False,               # generator loop should keep going
    "source_dirty": False,          # a new source is waiting to be opened
    "pending_source": None,         # (source_type, source_value) tuple
    "simulate_dispatch": False,
    "fps": 0.0,
    "last_result": None,            # most recent FrameResult-derived dict
    "last_alert": None,             # most recent confirmed-accident record
    "frame_count": 0,
}

logger.info("Loading inference engine (models load once at startup)...")
_engine = InferenceEngine()
_fps_counter = FPSCounter()


# ---------------------------------------------------------------------------
# Routes: page + control API
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/source", methods=["POST"])
def set_source():
    """
    Switches the active input source without restarting the server.
    JSON body: { "source_type": "webcam"|"video_file"|"ip_camera",
                 "source_value": "0" | "/path/to/video.mp4" | "http://phone-ip:8080/video",
                 "simulate_dispatch": true|false }
    """
    payload = request.get_json(force=True)
    source_type = payload.get("source_type")
    source_value = payload.get("source_value")
    simulate_dispatch = bool(payload.get("simulate_dispatch", False))

    if source_type not in ("webcam", "video_file", "ip_camera"):
        return jsonify({"ok": False, "error": "Invalid source_type"}), 400
    if not source_value and source_type != "webcam":
        return jsonify({"ok": False, "error": "source_value is required"}), 400
    if source_type == "webcam" and source_value in (None, ""):
        source_value = 0

    with _state_lock:
        _state["pending_source"] = (source_type, source_value)
        _state["source_dirty"] = True
        _state["running"] = True
        _state["simulate_dispatch"] = simulate_dispatch
        _state["source_label"] = f"{source_type}:{source_value}"

    logger.info("Source switch requested -> %s", _state["source_label"])
    return jsonify({"ok": True, "source_label": _state["source_label"]})


@app.route("/api/stop", methods=["POST"])
def stop():
    with _state_lock:
        _state["running"] = False
        vs = _state["video_source"]
        _state["video_source"] = None
    if vs is not None:
        vs.release()
    logger.info("Stream stopped and source released.")
    return jsonify({"ok": True})


@app.route("/api/status")
def status():
    with _state_lock:
        return jsonify({
            "running": _state["running"],
            "source_label": _state["source_label"],
            "fps": round(_state["fps"], 1),
            "last_result": _state["last_result"],
            "last_alert": _state["last_alert"],
            "frame_count": _state["frame_count"],
        })


@app.route("/api/history")
def history():
    limit = int(request.args.get("limit", 50))
    return jsonify(test_logger.get_recent(limit=limit))


@app.route("/test_captures/<path:filename>")
def serve_capture(filename):
    return send_from_directory(capture_manager.CAPTURE_DIR, filename)


# ---------------------------------------------------------------------------
# Live video stream (MJPEG) + inference loop
# ---------------------------------------------------------------------------

def _open_pending_source_if_needed():
    """Called inside the streaming loop; (re)opens the video source if the
    user just switched it via /api/source."""
    with _state_lock:
        dirty = _state["source_dirty"]
        pending = _state["pending_source"]

    if not dirty or pending is None:
        return

    source_type, source_value = pending
    old_vs = None
    with _state_lock:
        old_vs = _state["video_source"]

    if old_vs is not None:
        old_vs.release()

    new_vs = VideoSource(source_type, source_value)
    opened = new_vs.open()

    with _state_lock:
        _state["video_source"] = new_vs if opened else None
        _state["source_dirty"] = False
        _state["running"] = opened
        if not opened:
            _state["source_label"] = f"FAILED to open {source_type}:{source_value}"

    _engine.reset()  # don't let old track history bleed into the new source
    _fps_counter.__init__()  # reset FPS window


def _handle_confirmed_accident(frame, result, source_label: str):
    """Screenshot + log + (optionally) dry-run the real emergency dispatch,
    exactly mirroring what pipeline/main_pipeline.py does on confirmation --
    minus ever silently pretending a call was placed."""
    screenshot_filename = capture_manager.save_accident_frame(frame, source_label)

    record = test_logger.log_detection(
        source_label=source_label,
        severity=result.severity_label or "Unknown",
        severity_confidence=result.severity_conf or 0.0,
        classifier_confidence=result.classifier_conf,
        vehicle_count=len(result.detections),
        screenshot_filename=screenshot_filename,
    )

    with _state_lock:
        _state["last_alert"] = record
        simulate = _state["simulate_dispatch"]

    logger.warning("TEST ACCIDENT CONFIRMED | source=%s severity=%s vehicles=%d screenshot=%s",
                    source_label, result.severity_label, len(result.detections), screenshot_filename)

    if simulate:
        # Dry-run of the real dispatch chain. Twilio/SMTP only actually send
        # if real credentials are configured in .env -- otherwise this logs
        # "skipped_no_credentials" per channel, same as production.
        try:
            from emergency import notifier, service_finder
            from location import geolocation

            loc = geolocation.reverse_geocode(settings.CAMERA_LAT, settings.CAMERA_LON)
            police = service_finder.find_nearest(settings.CAMERA_LAT, settings.CAMERA_LON, "police", loc.region)
            hospital = service_finder.find_nearest(settings.CAMERA_LAT, settings.CAMERA_LON, "hospital", loc.region)
            incident_payload = {
                "camera_id": f"TEST-{source_label}",
                "lat": settings.CAMERA_LAT,
                "lon": settings.CAMERA_LON,
                "address": loc.address,
                "severity": result.severity_label,
                "vehicle_count": len(result.detections),
            }
            notifier.dispatch_all(incident_payload, police, hospital)
        except Exception:
            logger.exception("Simulated dispatch failed (this is test-mode only, non-fatal).")


def _generate_frames():
    """The MJPEG generator: opens/reopens the source as needed, runs
    inference on every frame, draws overlays, updates shared state, and
    yields JPEG-encoded frames for the browser <img> tag."""
    while True:
        _open_pending_source_if_needed()

        with _state_lock:
            running = _state["running"]
            vs = _state["video_source"]

        if not running or vs is None:
            time.sleep(0.2)
            continue

        ok, frame = vs.read()
        if not ok or frame is None:
            # Likely end of a video file, or a dropped IP-camera frame.
            time.sleep(0.1)
            continue

        result = _engine.process_frame(frame)
        fps = _fps_counter.tick()

        annotated = draw_detections(frame, result.detections)
        overlay_text = f"FPS: {fps:.1f} | Vehicles: {len(result.detections)} | " \
                        f"Accident conf: {result.classifier_conf * 100:.1f}%"
        cv2.putText(annotated, overlay_text, (10, annotated.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        if result.confirmed:
            annotated = draw_incident_banner(
                annotated, f"ACCIDENT CONFIRMED - {result.severity_label} ({result.severity_conf * 100:.0f}%)"
            )
        elif result.is_accident_frame:
            annotated = draw_incident_banner(
                annotated, f"Possible accident - verifying ({result.classifier_conf * 100:.0f}%)"
            )

        with _state_lock:
            _state["fps"] = fps
            _state["frame_count"] += 1
            _state["last_result"] = {
                "vehicle_count": len(result.detections),
                "classifier_label": result.classifier_label,
                "classifier_conf": round(result.classifier_conf, 3),
                "motion_supported": result.motion_supported,
                "confirmed": result.confirmed,
                "severity_label": result.severity_label,
                "severity_conf": round(result.severity_conf, 3) if result.severity_conf else None,
            }
            source_label = _state["source_label"]

        if result.confirmed:
            _handle_confirmed_accident(frame, result, source_label)

        ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(_generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    # threaded=True is required: /video_feed holds a long-lived connection
    # while /api/status, /api/history etc. need to be served concurrently.
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)

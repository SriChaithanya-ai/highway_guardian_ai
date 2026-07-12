"""
Highway Guardian AI -- End-to-end real-time pipeline.

    CCTV / RTSP stream
        -> frame extraction (sampled every N frames)
        -> vehicle detection (YOLOv8)
        -> multi-object tracking (ByteTrack, trajectory history)
        -> accident classification (trained YOLO-cls model)
        -> accident verification (classifier + motion heuristics, temporal confirmation)
        -> severity classification (Minor / Substantial / Critical)
        -> location identification (camera GPS -> reverse-geocoded address)
        -> automated emergency notification (Twilio call + SMS + email, no human step)
        -> incident logging (SQLite) + snapshot saved as evidence

Usage:
    python -m pipeline.main_pipeline --source 0
    python -m pipeline.main_pipeline --source rtsp://... --camera_id CAM_NH44_KM231
    python -m pipeline.main_pipeline --source path/to/video.mp4 --display
"""
import argparse
import datetime as dt
import logging
import time
from dataclasses import asdict

import cv2

from accident.accident_classifier import AccidentClassifier
from accident.severity_classifier import SeverityClassifier
from accident.verification import AccidentVerifier
from config import settings
from database import incident_db
from detection.trajectory_tracker import TrajectoryTracker
from detection.vehicle_detector import VehicleDetector
from emergency import notifier, service_finder
from location import geolocation
from utils.helpers import draw_detections, draw_incident_banner, save_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


class HighwayGuardianPipeline:
    def __init__(self, camera_id: str, source: str, display: bool = False,
                 control_room_email: str = None):
        self.camera_id = camera_id
        self.source = source
        self.display = display
        self.control_room_email = control_room_email

        cam = geolocation.load_camera(camera_id)
        self.cam_lat = cam.get("lat", settings.CAMERA_LAT)
        self.cam_lon = cam.get("lon", settings.CAMERA_LON)

        logger.info("Loading models...")
        self.vehicle_detector = VehicleDetector()
        self.accident_classifier = AccidentClassifier()
        self.severity_classifier = SeverityClassifier()
        self.verifier = AccidentVerifier()
        self.tracker = TrajectoryTracker()

        self._last_dispatch_time = None
        logger.info("Pipeline ready for camera %s at (%s, %s)", camera_id, self.cam_lat, self.cam_lon)

    def _cooldown_active(self) -> bool:
        if self._last_dispatch_time is None:
            return False
        return (time.time() - self._last_dispatch_time) < settings.INCIDENT_COOLDOWN_SECONDS

    def _handle_confirmed_incident(self, frame, detections, verification):
        snapshot_path = save_snapshot(frame, self.camera_id)

        severity_label, severity_conf = self.severity_classifier.predict(frame)
        loc = geolocation.reverse_geocode(self.cam_lat, self.cam_lon)

        police = service_finder.find_nearest(self.cam_lat, self.cam_lon, "police", loc.region)
        hospital = service_finder.find_nearest(self.cam_lat, self.cam_lon, "hospital", loc.region)

        incident_payload = {
            "camera_id": self.camera_id,
            "lat": self.cam_lat,
            "lon": self.cam_lon,
            "address": loc.address,
            "severity": severity_label,
            "vehicle_count": len(detections),
        }

        logger.warning("ACCIDENT CONFIRMED | camera=%s severity=%s vehicles=%d address=%s",
                        self.camera_id, severity_label, len(detections), loc.address)

        # ---- Fully automated dispatch, no human confirmation step ----
        dispatch_records = notifier.dispatch_all(
            incident_payload, police, hospital, control_room_email=self.control_room_email
        )

        incident_db.log_incident(
            camera_id=self.camera_id, lat=self.cam_lat, lon=self.cam_lon, address=loc.address,
            severity=severity_label, severity_confidence=severity_conf,
            classifier_confidence=verification.classifier_conf,
            vehicle_count=len(detections),
            vehicle_classes=[d.cls_name for d in detections],
            snapshot_path=snapshot_path,
            police_contact=asdict(police), hospital_contact=asdict(hospital),
            dispatch_log=[asdict(r) for r in dispatch_records],
        )

        self._last_dispatch_time = time.time()
        self.verifier.reset()

    def run(self):
        cap = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.info("Stream ended or frame read failed.")
                    break

                frame_idx += 1
                if frame_idx % settings.FRAME_SAMPLE_RATE != 0:
                    continue  # skip frames for real-time performance

                detections = self.vehicle_detector.track(frame)
                track_states = self.tracker.update(detections)

                is_accident, conf, label = self.accident_classifier.predict(frame)
                verification = self.verifier.verify(is_accident, conf, detections, track_states)

                annotated = draw_detections(frame, detections)

                if verification.confirmed and not self._cooldown_active():
                    annotated = draw_incident_banner(annotated, f"ACCIDENT DETECTED - dispatching emergency services")
                    self._handle_confirmed_incident(frame, detections, verification)
                elif is_accident:
                    annotated = draw_incident_banner(annotated, f"Possible accident - verifying ({conf:.2f})")

                if self.display:
                    cv2.imshow("Highway Guardian AI", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if self.display:
                cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=settings.RTSP_URL,
                     help="RTSP URL, video file path, or webcam index (e.g. 0)")
    ap.add_argument("--camera_id", default=settings.CAMERA_ID)
    ap.add_argument("--display", action="store_true", help="Show annotated video window")
    ap.add_argument("--control_room_email", default=None)
    args = ap.parse_args()

    pipeline = HighwayGuardianPipeline(
        camera_id=args.camera_id, source=args.source,
        display=args.display, control_room_email=args.control_room_email,
    )
    pipeline.run()


if __name__ == "__main__":
    main()

"""
video_sources.py
----------------
Modular input-source abstraction for the testing module.

The whole point of this class is that swapping the input later for a real
highway CCTV camera means changing ONE thing -- the source_type/source_value
passed in here -- and nothing else in the inference/logging/alerting code
has to change. RTSP CCTV cameras, DroidCam/IP Webcam phone streams, local
video files, and laptop webcams are all just an OpenCV VideoCapture target
under the hood.

Supported source types:
    "webcam"     -> source_value = integer device index, e.g. 0
    "video_file" -> source_value = path to a local .mp4/.avi/etc.
    "ip_camera"  -> source_value = an HTTP/RTSP URL, e.g.:
                      IP Webcam (Android app):  http://<phone-ip>:8080/video
                      DroidCam:                 http://<phone-ip>:4747/video
                      Any RTSP CCTV camera:      rtsp://user:pass@ip:554/stream1
"""
import logging
import threading

import cv2

logger = logging.getLogger("testing_module.video_sources")


class VideoSource:
    """Thread-safe-ish wrapper around cv2.VideoCapture with a uniform
    interface regardless of where the frames are actually coming from."""

    def __init__(self, source_type: str, source_value):
        self.source_type = source_type  # "webcam" | "video_file" | "ip_camera"
        self.source_value = source_value
        self._cap = None
        self._lock = threading.Lock()

    def open(self) -> bool:
        with self._lock:
            self._release_locked()

            if self.source_type == "webcam":
                target = int(self.source_value)
            else:
                # video_file and ip_camera both resolve to a path/URL string
                # that OpenCV's FFMPEG backend can open directly.
                target = str(self.source_value)

            self._cap = cv2.VideoCapture(target)

            # Small buffer size reduces latency on IP-camera/RTSP streams by
            # preventing OpenCV from queuing up several stale frames.
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass  # not all backends support this property

            opened = self._cap.isOpened()
            if not opened:
                logger.error("Failed to open source: type=%s value=%s", self.source_type, self.source_value)
            else:
                logger.info("Opened source: type=%s value=%s", self.source_type, self.source_value)
            return opened

    def read(self):
        """Returns (success: bool, frame or None)."""
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return False, None
            return self._cap.read()

    def is_opened(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def release(self):
        with self._lock:
            self._release_locked()

    def _release_locked(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> str:
        return f"{self.source_type}:{self.source_value}"

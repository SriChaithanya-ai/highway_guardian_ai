"""
fps_counter.py
--------------
Tiny rolling-average FPS counter so the on-screen readout doesn't jitter
wildly frame to frame (which a naive 1/dt calculation would).
"""
import time
from collections import deque


class FPSCounter:
    def __init__(self, window: int = 20):
        self._timestamps = deque(maxlen=window)

    def tick(self) -> float:
        """Call once per processed frame. Returns the current smoothed FPS."""
        now = time.time()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

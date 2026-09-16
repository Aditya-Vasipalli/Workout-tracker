"""Camera sources, including a phone over WiFi.

The old tracker opened two cameras, ran pose on both, and then used one and
discarded the other (`display_dual_camera_feed` picked a single keypoint set).
This module keeps sources honest: a `CameraSource` yields timestamped frames,
and `SyncedPair` only reports a pair as usable when the two frames are actually
close enough in time to describe the same instant.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue

import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    source: str


class CameraSource:
    """A camera read on a background thread.

    Threading matters here: `VideoCapture.read()` blocks, so reading two
    cameras serially on one thread makes their frames a frame-time apart before
    anything else goes wrong. A network camera makes that far worse.
    """

    def __init__(self, spec: int | str, name: str = "", width: int = 1280, height: int = 720):
        self.spec = spec
        self.name = name or f"cam{spec}"
        self.width, self.height = width, height
        self._queue: Queue[Frame] = Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap = None
        self.frames_read = 0
        self.frames_dropped = 0

    def open(self) -> bool:
        import cv2

        self._cap = cv2.VideoCapture(self.spec)
        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Keep the driver buffer tiny so we get the newest frame, not a backlog.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = Frame(image=image, timestamp=time.monotonic(), source=self.name)
            self.frames_read += 1
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self.frames_dropped += 1
                except Empty:
                    pass
            self._queue.put(frame)

    def read(self, timeout: float = 0.5) -> Frame | None:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()


def phone_camera(url: str, name: str = "phone") -> CameraSource:
    """A phone streaming MJPEG/RTSP over WiFi.

    Works with the common 'IP Webcam' style apps. Typical URLs:
        http://192.168.1.42:8080/video      (MJPEG)
        rtsp://192.168.1.42:8554/live       (RTSP)

    Expect 100-300ms of latency over WiFi. That is why `SyncedPair` below
    enforces a tolerance rather than assuming two streams are aligned.
    """
    return CameraSource(url, name=name)


class SyncedPair:
    """Two sources paired by timestamp, with an explicit tolerance.

    Reports `skew` so the caller can tell the user their phone stream has
    drifted, instead of silently fusing frames from different moments -- which
    would corrupt any triangulated 3D far more than using one camera would.
    """

    def __init__(self, primary: CameraSource, secondary: CameraSource, tolerance_s: float = 0.05):
        self.primary = primary
        self.secondary = secondary
        self.tolerance_s = tolerance_s
        self._pending: Frame | None = None
        self.pairs_matched = 0
        self.pairs_rejected = 0

    def open(self) -> bool:
        if not self.primary.open():
            return False
        if not self.secondary.open():
            self.primary.close()
            return False
        return True

    def read(self) -> tuple[Frame, Frame, float] | None:
        """Return (primary, secondary, skew) or None if no usable pair."""
        a = self.primary.read()
        if a is None:
            return None

        best: Frame | None = self._pending
        best_skew = abs(best.timestamp - a.timestamp) if best else float("inf")

        # Drain the secondary for its closest frame in time.
        for _ in range(3):
            b = self.secondary.read(timeout=0.05)
            if b is None:
                break
            skew = abs(b.timestamp - a.timestamp)
            if skew < best_skew:
                best, best_skew = b, skew
            if b.timestamp > a.timestamp:
                self._pending = b
                break

        if best is None:
            return None
        if best_skew > self.tolerance_s:
            self.pairs_rejected += 1
            return None

        self.pairs_matched += 1
        return a, best, best_skew

    @property
    def sync_health(self) -> float:
        total = self.pairs_matched + self.pairs_rejected
        return self.pairs_matched / total if total else 0.0

    def close(self) -> None:
        self.primary.close()
        self.secondary.close()


def list_cameras(max_index: int = 5) -> list[int]:
    """Indices that actually open. Probing is noisy on some drivers."""
    import cv2

    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found

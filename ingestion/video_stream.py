"""
ingestion/video_stream.py - Universal OpenCV Video Stream Ingestion
Part of AIEcoLightswitcher Modular Architecture (Milestone 1)
"""

import logging
import os
import sys
import threading
import time
from typing import Any, Optional, Tuple
import cv2
import numpy as np

from .base_stream import BaseVideoStream, FeedStatus

logger = logging.getLogger("ingestion.video_stream")


def open_capture(source: Any, timeout_ms: int = 2000) -> cv2.VideoCapture:
    """
    Factory function configuring low-latency VideoCapture backends and timeouts.
    - Windows webcams use DirectShow (cv2.CAP_DSHOW) for sub-second startup.
    - Network streams (RTSP/HTTP) use TCP transport and CAP_PROP socket timeouts.
    - Sets CAP_PROP_BUFFERSIZE = 1 to eliminate internal driver buffering.
    """
    is_windows = sys.platform.startswith("win")

    # Normalize integer source (e.g. "0" -> 0)
    if isinstance(source, str) and source.strip().isdigit():
        source = int(source.strip())

    # 1. Local USB Camera Index
    if isinstance(source, int):
        if is_windows and hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)

        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap

    # 2. Network Stream (RTSP or HTTP/DroidCam)
    src_str = str(source).strip()
    is_network = src_str.lower().startswith(("rtsp://", "http://", "https://"))

    if is_network:
        # Enforce TCP transport for RTSP to prevent packet drops
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;2000000")

        # Pass non-blocking socket timeouts to FFmpeg backend
        has_timeouts = hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC") and hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC")
        if has_timeouts and hasattr(cv2, "CAP_FFMPEG"):
            try:
                params = [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms
                ]
                cap = cv2.VideoCapture(src_str, cv2.CAP_FFMPEG, params)
                if cap.isOpened():
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    return cap
            except Exception as e:
                logger.debug(f"CAP_FFMPEG params not supported: {e}")

        # Fallback network open
        cap = cv2.VideoCapture(src_str)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap

    # 3. File or general capture
    cap = cv2.VideoCapture(src_str)
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    return cap


class VideoStream(BaseVideoStream):
    """
    Threaded universal OpenCV video capture stream.
    Eliminates internal frame buffering lag and supports graceful start/stop/read.
    """

    def __init__(self, source: Any = 0, timeout_ms: int = 2000, is_simulation_fallback: bool = False):
        self.source = source
        self.timeout_ms = timeout_ms
        self.is_simulation_fallback = is_simulation_fallback

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.is_simulation = False

        # Legacy simulation animation state
        self._sim_x = 50
        self._sim_dx = 4

    def start(self) -> "VideoStream":
        """Opens capture and starts internal frame ingestion loop."""
        self.running = True
        self.is_simulation = False

        # Handle explicit mock/simulation requests for backwards compatibility
        if self.source in ("simulation", "mock", -1):
            self.is_simulation = True
            self.cap = None
        else:
            try:
                self.cap = open_capture(self.source, timeout_ms=self.timeout_ms)
                if not self.cap.isOpened():
                    if self.is_simulation_fallback:
                        self.is_simulation = True
                    self.cap = None
            except Exception as e:
                logger.warning(f"Error opening VideoCapture for {self.source}: {e}")
                if self.is_simulation_fallback:
                    self.is_simulation = True
                self.cap = None

        self.thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name=f"VideoStream-{self.source}"
        )
        self.thread.start()
        return self

    def _update_loop(self) -> None:
        """Internal capture loop grabbing latest frames."""
        while self.running:
            if not self.is_simulation and self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.01)
            elif self.is_simulation:
                # Backwards-compatible legacy animated simulation frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(frame, (20, 20), (620, 460), (40, 40, 40), -1)
                cv2.putText(
                    frame, "CAMERA SIMULATION MODE", (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                )
                self._sim_x += self._sim_dx
                if self._sim_x > 500 or self._sim_x < 50:
                    self._sim_dx *= -1
                cv2.circle(frame, (self._sim_x + 30, 240), 25, (0, 200, 255), -1)
                with self.lock:
                    self.frame = frame
                time.sleep(0.03)
            else:
                with self.lock:
                    self.frame = None
                time.sleep(0.05)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the latest captured frame."""
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self) -> None:
        """Stops the capture thread and releases the OpenCV VideoCapture."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.frame = None

    @property
    def is_opened(self) -> bool:
        if self.is_simulation:
            return True
        return (self.cap is not None) and self.cap.isOpened()

    @property
    def status(self) -> FeedStatus:
        if self.is_opened:
            return FeedStatus.ONLINE
        return FeedStatus.OFFLINE

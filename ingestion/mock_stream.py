"""
ingestion/mock_stream.py - Deterministic Multi-Feed Synthetic Video Generator
Part of AIEcoLightswitcher Modular Architecture (Milestone 1)
"""

import os
import threading
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    from ultralytics.utils import ASSETS
except Exception:
    ASSETS = None

from .base_stream import BaseVideoStream, FeedStatus


class MockVideoSource(BaseVideoStream):
    """
    Deterministic frame generator utilizing local Ultralytics bus.jpg asset
    (verified 3-4 person detections) and synthetic frames for automated headless verification.
    Fully backwards-compatible with legacy eco_engine.MockVideoSource.
    """
    def __init__(
        self,
        default_scenario: str = "PERSON_IN_ROI",
        fps: float = 30.0,
        width: int = 640,
        height: int = 480,
        feed_id: str = "mock_feed",
        simulate_fps: bool = False
    ):
        self.feed_id = feed_id
        self.fps = max(0.1, float(fps))
        self.width = int(width)
        self.height = int(height)
        self._simulate_fps = bool(simulate_fps)

        self._is_opened = True
        self.is_simulation = True
        self.lock = threading.Lock()

        self._last_frame_time = time.perf_counter()
        self._frame_count = 0
        self._artificial_latency = 0.0
        self._drop_interval = 0  # 0 = no drops; N = drop every Nth frame

        # 1. Multi-path asset resolution for bus.jpg
        bus_candidate = None
        if ASSETS and os.path.exists(os.path.join(ASSETS, "bus.jpg")):
            bus_candidate = os.path.join(ASSETS, "bus.jpg")
        else:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(module_dir, "bus.jpg"),
                os.path.join(module_dir, "..", "bus.jpg"),
                os.path.join(module_dir, "..", "Model Demo", "bus.jpg"),
            ]
            for p in candidates:
                if os.path.exists(p):
                    bus_candidate = p
                    break

        if bus_candidate and os.path.exists(bus_candidate):
            loaded = cv2.imread(bus_candidate)
            if loaded is not None:
                self._bus_img = cv2.resize(loaded, (self.width, self.height))
            else:
                self._bus_img = self._create_synthetic_fallback()
        else:
            self._bus_img = self._create_synthetic_fallback()

        h, w = self._bus_img.shape[:2]

        # Right-only frame: mask out left 60% so only person on the far right (x > 0.60) is detected
        self._right_person_img = self._bus_img.copy()
        self._right_person_img[:, :int(w * 0.60)] = 0

        # Left-only frame: mask out right 50% so only persons on the left (x < 0.50) are detected
        self._left_persons_img = self._bus_img.copy()
        self._left_persons_img[:, int(w * 0.50):] = 0

        # Empty frame: all zeros -> 0 detections (strict count_nonzero == 0 guarantee)
        self._empty_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self._scenario = default_scenario.upper().strip()
        self._current_frame = self._bus_img.copy()
        self.set_scenario(default_scenario)

    def _create_synthetic_fallback(self) -> np.ndarray:
        """Generates high-contrast synthetic image if bus.jpg is unavailable."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 400), (255, 255, 255), -1)
        return img

    def set_scenario(self, scenario: str) -> None:
        """
        Switches the deterministic frame scenario:
        - 'PERSON_IN_ROI' / 'BUS' / 'PRESENT': Full image (occupants present).
        - 'PERSON_OUTSIDE_ROI' / 'RIGHT_ONLY' / 'OUTSIDE': Occupant at x > 0.60.
        - 'LEFT_PERSONS' / 'LEFT_ONLY': Occupants at x < 0.50.
        - 'EMPTY_FRAME' / 'EMPTY' / 'ABSENT' / 'BLANK': Zeroed array (0 occupants).
        - 'OFFLINE_STALLED' / 'OFFLINE' / 'STALLED': Stream severed (read returns False).
        - 'INTERMITTENT_DROP' / 'PACKET_LOSS': Simulates wireless packet drop every 5th frame.
        - 'LATENCY_SPIKE' / 'LAG': Injects 0.5s artificial latency.
        """
        sc = scenario.upper().strip()
        with self.lock:
            self._scenario = sc
            if sc in ("PERSON_IN_ROI", "BUS", "PRESENT"):
                self._current_frame = self._bus_img.copy()
            elif sc in ("PERSON_OUTSIDE_ROI", "RIGHT_ONLY", "OUTSIDE"):
                self._current_frame = self._right_person_img.copy()
            elif sc in ("LEFT_PERSONS", "LEFT_ONLY"):
                self._current_frame = self._left_persons_img.copy()
            elif sc in ("EMPTY_FRAME", "EMPTY", "ABSENT", "BLANK"):
                self._current_frame = self._empty_img.copy()
            elif sc in ("OFFLINE_STALLED", "OFFLINE", "STALLED"):
                self._current_frame = None
            elif sc in ("INTERMITTENT_DROP", "PACKET_LOSS"):
                self._current_frame = self._bus_img.copy()
                self._drop_interval = 5
            elif sc in ("LATENCY_SPIKE", "LAG"):
                self._current_frame = self._bus_img.copy()
                self._artificial_latency = 0.5
            else:
                self._current_frame = self._empty_img.copy()

    def set_custom_frame(self, frame: Optional[np.ndarray]) -> None:
        """Sets an arbitrary numpy frame for custom test scenarios."""
        with self.lock:
            if frame is not None and hasattr(frame, "copy"):
                self._current_frame = frame.copy()
            else:
                self._current_frame = None

    def set_artificial_latency(self, seconds: float) -> None:
        """Configures simulated network latency in seconds."""
        with self.lock:
            self._artificial_latency = max(0.0, float(seconds))

    def set_drop_interval(self, interval: int) -> None:
        """Configures packet drop frequency (drop every N frames, 0 to disable)."""
        with self.lock:
            self._drop_interval = max(0, int(interval))

    def start(self) -> "MockVideoSource":
        """Starts or re-opens mock video source."""
        with self.lock:
            self._is_opened = True
            self._last_frame_time = time.perf_counter()
        return self

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads next deterministic frame.
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success_flag, frame_bgr)
        """
        with self.lock:
            if not self._is_opened:
                return False, None

            if self._scenario in ("OFFLINE_STALLED", "OFFLINE", "STALLED"):
                return False, None

            # Artificial latency simulation
            if self._artificial_latency > 0:
                time.sleep(self._artificial_latency)

            # Packet drop simulation
            self._frame_count += 1
            if self._drop_interval > 0 and (self._frame_count % self._drop_interval == 0):
                return False, None

            # Optional FPS simulation
            if self._simulate_fps and self.fps > 0:
                target_delta = 1.0 / self.fps
                now = time.perf_counter()
                elapsed = now - self._last_frame_time
                if elapsed < target_delta:
                    time.sleep(target_delta - elapsed)
                self._last_frame_time = time.perf_counter()

            if self._current_frame is not None and hasattr(self._current_frame, "copy"):
                return True, self._current_frame.copy()
            elif self._current_frame is not None:
                return True, self._current_frame
            return False, None

    def stop(self) -> None:
        """Stops the mock video stream."""
        with self.lock:
            self._is_opened = False

    @property
    def is_opened(self) -> bool:
        """Returns True if the stream is currently open."""
        with self.lock:
            return self._is_opened

    @property
    def status(self) -> FeedStatus:
        """Returns FeedStatus based on current operational state."""
        with self.lock:
            if not self._is_opened or self._scenario in ("OFFLINE_STALLED", "OFFLINE", "STALLED"):
                return FeedStatus.OFFLINE
            return FeedStatus.ONLINE


def create_mock_feed_grid(
    scenarios: Optional[List[str]] = None,
    fps: float = 30.0,
    simulate_fps: bool = False
) -> List[MockVideoSource]:
    """
    Factory helper generating 4 (or N) distinct MockVideoSource instances
    for multi-feed parallel testing.
    Default grid:
      Feed 0: PERSON_IN_ROI (positive presence)
      Feed 1: PERSON_OUTSIDE_ROI (presence outside zone)
      Feed 2: EMPTY_FRAME (zero occupants)
      Feed 3: OFFLINE_STALLED (disconnected camera)
    """
    defaults = [
        "PERSON_IN_ROI",
        "PERSON_OUTSIDE_ROI",
        "EMPTY_FRAME",
        "OFFLINE_STALLED"
    ]
    target_scenarios = scenarios if scenarios is not None else defaults
    feeds = []
    for idx, sc in enumerate(target_scenarios):
        feed = MockVideoSource(
            default_scenario=sc,
            fps=fps,
            feed_id=f"mock_feed_{idx + 1}",
            simulate_fps=simulate_fps
        )
        feeds.append(feed)
    return feeds

"""
ingestion/base_stream.py - Abstract Video Stream Contract & Feed Status Telemetry
Part of AIEcoLightswitcher Modular Architecture (Milestone 1)
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np


class FeedStatus(str, Enum):
    """Lifecycle states for video stream feeds."""
    UNASSIGNED = "UNASSIGNED"
    CONNECTING = "CONNECTING"
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    RECONNECTING = "RECONNECTING"
    DISABLED = "DISABLED"
    ERROR = "ERROR"


@dataclass
class StreamTelemetry:
    """Standardized real-time telemetry structure for a video stream."""
    feed_id: str = ""
    status: FeedStatus = FeedStatus.UNASSIGNED
    fps: float = 0.0
    latency_ms: float = 0.0
    frame_id: int = 0
    reconnect_attempts: int = 0
    reconnect_countdown: float = 0.0
    last_frame_time: float = 0.0
    resolution: Tuple[int, int] = (0, 0)  # (width, height)
    source: Any = None
    error_msg: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes telemetry for UI, logging, and JSON dispatch."""
        data = asdict(self)
        data["status"] = self.status
        return data


class BaseVideoStream(ABC):
    """Abstract interface contract for all video stream sources."""

    @abstractmethod
    def start(self) -> "BaseVideoStream":
        """Initializes and opens the video stream."""
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a single frame from the stream. Returns (success, frame)."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops the stream and releases all underlying resources."""
        pass

    @property
    @abstractmethod
    def is_opened(self) -> bool:
        """Checks if the stream is actively opened and valid."""
        pass

    @property
    def status(self) -> FeedStatus:
        """Default status based on is_opened state."""
        return FeedStatus.ONLINE if self.is_opened else FeedStatus.OFFLINE

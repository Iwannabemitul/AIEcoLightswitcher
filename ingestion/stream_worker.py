"""
ingestion/stream_worker.py - Dedicated Background Stream Worker & Atomic Frame Buffer
Part of AIEcoLightswitcher Modular Architecture (Milestone 1)
"""

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from .base_stream import BaseVideoStream, FeedStatus, StreamTelemetry

logger = logging.getLogger("ingestion.stream_worker")


@dataclass(frozen=True)
class BufferedFrame:
    """Immutable single-slot frame container."""
    frame: np.ndarray
    frame_id: int
    timestamp: float
    latency_ms: float = 0.0


class SingleFrameBuffer:
    """
    Thread-safe, non-blocking, single-slot atomic frame buffer.
    - Drop-oldest semantics: newer frames unconditionally overwrite unconsumed frames.
    - Minimal lock holding time (<100 nanoseconds) for reference swap.
    - Provides O(1) zero-latency access to the freshest frame.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._slot: Optional[BufferedFrame] = None
        self._frame_count = 0
        self._drop_count = 0
        self._last_read_id = -1

    def put(self, frame: np.ndarray, timestamp: Optional[float] = None) -> int:
        """Stores a new frame, dropping any existing unconsumed frame."""
        now = timestamp if timestamp is not None else time.time()
        with self._lock:
            self._frame_count += 1
            if self._slot is not None and self._slot.frame_id != self._last_read_id:
                self._drop_count += 1
            self._slot = BufferedFrame(
                frame=frame,
                frame_id=self._frame_count,
                timestamp=now
            )
            return self._frame_count

    def get(self, copy_frame: bool = False) -> Tuple[bool, Optional[np.ndarray], float, int]:
        """
        Retrieves the latest available frame.
        Returns: (success, frame, timestamp, frame_id)
        """
        with self._lock:
            slot = self._slot
            if slot is None:
                return False, None, 0.0, 0
            self._last_read_id = slot.frame_id
            frame_out = slot.frame.copy() if copy_frame else slot.frame
            return True, frame_out, slot.timestamp, slot.frame_id

    def clear(self) -> None:
        """Clears the buffer slot."""
        with self._lock:
            self._slot = None

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._slot is None

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def drop_count(self) -> int:
        with self._lock:
            return self._drop_count

    @property
    def age_ms(self) -> float:
        """Returns the age of the current slot frame in milliseconds."""
        with self._lock:
            if self._slot is None:
                return 0.0
            return max(0.0, (time.time() - self._slot.timestamp) * 1000.0)


class StreamWorker:
    """
    Dedicated background worker thread for a single video feed.
    - Runs decoupled capture loop at native camera speed.
    - Updates SingleFrameBuffer atomically.
    - Detects stream timeouts (< 2.0s).
    - Executes exponential backoff auto-reconnect (2s, 4s, 8s, max 15s).
    - Responds to enable/disable and clean stop within 100ms.
    """

    def __init__(
        self,
        source: Any,
        feed_id: str = "feed_0",
        name: str = "",
        reconnect_interval: float = 2.0,
        max_reconnect_interval: float = 15.0,
        read_timeout: float = 2.0,
        stream_factory: Optional[Callable[[Any], BaseVideoStream]] = None
    ):
        self.source = source
        self.feed_id = feed_id
        self.name = name or feed_id
        self.reconnect_interval = max(0.5, float(reconnect_interval))
        self.max_reconnect_interval = max(self.reconnect_interval, float(max_reconnect_interval))
        self.read_timeout = max(0.5, float(read_timeout))
        self.stream_factory = stream_factory

        self._buffer = SingleFrameBuffer()
        self._status = FeedStatus.UNASSIGNED
        self._status_lock = threading.Lock()

        # Telemetry metrics
        self._fps_ema = 0.0
        self._fps_alpha = 0.15
        self._last_frame_time = 0.0
        self._resolution = (0, 0)
        self._reconnect_attempts = 0
        self._reconnect_countdown = 0.0
        self._error_msg: Optional[str] = None

        # Thread synchronization
        self._running = False
        self._enabled = True
        self._stop_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._thread: Optional[threading.Thread] = None
        self._active_stream: Optional[BaseVideoStream] = None

    def _set_status(self, status: FeedStatus, error_msg: Optional[str] = None) -> None:
        with self._status_lock:
            self._status = status
            if error_msg is not None:
                self._error_msg = error_msg

    @property
    def status(self) -> FeedStatus:
        with self._status_lock:
            return self._status

    @property
    def is_online(self) -> bool:
        return self.status == FeedStatus.ONLINE

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def fps(self) -> float:
        return round(self._fps_ema, 1)

    def start(self) -> "StreamWorker":
        """Spawns the background ingestion worker thread."""
        if self._running:
            return self
        self._running = True
        self._stop_event.clear()
        self._set_status(FeedStatus.CONNECTING)
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"StreamWorker-{self.feed_id}"
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 1.0) -> None:
        """Stops the worker thread and releases resources."""
        self._running = False
        self._stop_event.set()
        self._resume_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._close_active_stream()
        self._set_status(FeedStatus.OFFLINE)

    def enable(self) -> None:
        """Enables feed ingestion."""
        self._enabled = True
        self._resume_event.set()

    def disable(self) -> None:
        """Disables feed ingestion and pauses stream."""
        self._enabled = False
        self._resume_event.clear()
        self._close_active_stream()
        self._set_status(FeedStatus.DISABLED)

    def _create_stream(self) -> BaseVideoStream:
        """Creates appropriate BaseVideoStream instance for the source."""
        if self.stream_factory is not None:
            return self.stream_factory(self.source)

        if isinstance(self.source, BaseVideoStream) or (
            hasattr(self.source, "read") and hasattr(self.source, "start")
        ):
            return self.source

        src_str = str(self.source).strip()
        if src_str.lower().startswith("mock") or src_str in ("simulation", "-1"):
            from .mock_stream import MockVideoSource
            return MockVideoSource(feed_id=self.feed_id)

        # Lazy import to avoid circular dependency
        from .video_stream import VideoStream
        return VideoStream(source=self.source, timeout_ms=int(self.read_timeout * 1000))

    def _close_active_stream(self) -> None:
        if self._active_stream is not None:
            try:
                self._active_stream.stop()
            except Exception as e:
                logger.warning(f"[{self.feed_id}] Error stopping active stream: {e}")
            finally:
                self._active_stream = None

    def _calculate_backoff(self) -> float:
        """Calculates exponential backoff delay (2s, 4s, 8s, max 15s)."""
        power = min(self._reconnect_attempts, 4)
        delay = self.reconnect_interval * (2 ** power)
        return min(delay, self.max_reconnect_interval)

    def _wait_interruptible(self, duration: float) -> bool:
        """Waits for duration in small slices, updating countdown. Returns False if stopped."""
        t_end = time.time() + duration
        while time.time() < t_end:
            if self._stop_event.is_set() or not self._running:
                return False
            remaining = max(0.0, t_end - time.time())
            self._reconnect_countdown = round(remaining, 1)
            self._stop_event.wait(timeout=min(0.05, remaining))
        self._reconnect_countdown = 0.0
        return True

    def _worker_loop(self) -> None:
        """Main background ingestion and reconnect lifecycle loop."""
        while self._running:
            # Check for disabled pause
            if not self._enabled:
                self._set_status(FeedStatus.DISABLED)
                self._resume_event.wait(timeout=0.2)
                continue

            # Connection attempt
            self._set_status(FeedStatus.CONNECTING)
            self._close_active_stream()

            try:
                stream = self._create_stream()
                stream.start()
                if not stream.is_opened:
                    raise IOError(f"Failed to open source: {self.source}")
                self._active_stream = stream
            except Exception as e:
                logger.warning(f"[{self.feed_id}] Connect failure: {e}")
                self._handle_disconnect(str(e))
                continue

            # Connected: enter read loop
            consecutive_failures = 0
            self._set_status(FeedStatus.ONLINE)
            self._reconnect_attempts = 0
            self._last_frame_time = time.time()

            while self._running and self._enabled:
                loop_start = time.time()
                try:
                    ret, frame = self._active_stream.read()
                except Exception as e:
                    logger.warning(f"[{self.feed_id}] Exception during read: {e}")
                    ret, frame = False, None

                now = time.time()

                if ret and frame is not None and getattr(frame, "size", 0) > 0:
                    consecutive_failures = 0
                    self._last_frame_time = now
                    h, w = frame.shape[:2]
                    self._resolution = (w, h)

                    # Update EMA FPS
                    dt = now - loop_start
                    if dt > 0:
                        instant_fps = 1.0 / dt
                        self._fps_ema = (self._fps_alpha * instant_fps) + ((1.0 - self._fps_alpha) * self._fps_ema)

                    self._buffer.put(frame, now)
                    if self.status != FeedStatus.ONLINE:
                        self._set_status(FeedStatus.ONLINE)
                    time.sleep(0.002)

                else:
                    consecutive_failures += 1
                    time.sleep(0.01)

                # Timeout detection: > read_timeout without valid frame or 5 consecutive failures
                time_since_frame = now - self._last_frame_time
                if time_since_frame >= self.read_timeout or consecutive_failures >= 5:
                    logger.warning(
                        f"[{self.feed_id}] Stream stalled (no frame for {time_since_frame:.2f}s, "
                        f"failures={consecutive_failures}). Triggering reconnect."
                    )
                    break

            # Stream lost or broken
            if self._running and self._enabled:
                self._handle_disconnect("Stream read timeout or socket disconnect")

    def _handle_disconnect(self, reason: str) -> None:
        """Transitions to OFFLINE, then RECONNECTING with exponential backoff."""
        self._close_active_stream()
        self._set_status(FeedStatus.OFFLINE, error_msg=reason)
        self._fps_ema = 0.0

        backoff = self._calculate_backoff()
        self._reconnect_attempts += 1
        self._set_status(FeedStatus.RECONNECTING, error_msg=reason)

        logger.info(f"[{self.feed_id}] Reconnecting in {backoff:.1f}s (Attempt #{self._reconnect_attempts})")
        if not self._wait_interruptible(backoff):
            return

    def get_latest_frame(self, copy_frame: bool = False) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Public consumer API for freshest frame and structured telemetry.
        Returns: (frame_or_none, telemetry_dict)
        """
        success, frame, timestamp, frame_id = self._buffer.get(copy_frame=copy_frame)
        latency = max(0.0, (time.time() - timestamp) * 1000.0) if success else 0.0

        telemetry = StreamTelemetry(
            feed_id=self.feed_id,
            status=self.status,
            fps=self.fps,
            latency_ms=round(latency, 2),
            frame_id=frame_id,
            reconnect_attempts=self._reconnect_attempts,
            reconnect_countdown=self._reconnect_countdown,
            last_frame_time=self._last_frame_time,
            resolution=self._resolution,
            source=str(self.source),
            error_msg=self._error_msg
        ).to_dict()

        if not success or self.status != FeedStatus.ONLINE:
            return None, telemetry

        return frame, telemetry

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Backwards compatibility contract conforming to BaseVideoStream."""
        frame, _ = self.get_latest_frame(copy_frame=False)
        return (frame is not None), frame

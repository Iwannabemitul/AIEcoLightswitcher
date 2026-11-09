"""
eco_engine.py - Core Detection Engine & Hardware Abstraction for AIEcoLightswitcher
Showcase Edition (Milestone 1 Implementation)

Provides:
- EngineConfig: Thread-safe runtime configuration dataclass.
- Device Detection & Warmup: Safe PyTorch CUDA auto-detection with graceful CPU fallback.
- SpatialROIFilter: Normalized coordinate ROI polygon containment using bottom-center (foot)
  and centroid anchors with translucent visualization overlays.
- TemporalPresenceSmoother: Asymmetric dual-threshold hysteresis state machine (M frames ON,
  N frames OFF) and minimum switching interval guard clamp to eliminate relay chattering.
- DutyCycleManager: Dynamic energy-saving evaluation vs. sleep duty cycle scheduler.
- Hardware Abstraction Layer: BaseHardwareController, physical USB serial HardwareController,
  and in-memory MockHardwareController with timestamped command recording.
- Video Stream Abstraction: BaseVideoStream, threaded OpenCV VideoStream, and deterministic
  MockVideoSource using local Ultralytics bus.jpg asset.
- AdvancedDetectionEngine: Decoupled multi-threaded inference engine with zero UI stutter.
- EcoVisionSystem: Backwards-compatibility wrapper for seamless integration.
"""

import copy
from dataclasses import asdict, dataclass, field, replace
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import serial
import serial.tools.list_ports
import torch
from ultralytics import YOLO

try:
    from ultralytics.utils import ASSETS
except Exception:
    ASSETS = None

# Default Target Classes for YOLOv8 (COCO dataset)
# 0 = person (human occupant), 77 = teddy bear (doll / tabletop test proxy)
DEFAULT_TARGET_CLASSES = {0: "person", 77: "doll/teddy"}
logger = logging.getLogger("eco_engine")


# ============================================================================
# 1. Connection Helper
# ============================================================================

class ConnectionResult(tuple):
    """
    Subclass of tuple (bool, str) allowing usage both as a boolean
    (if ctrl.connect(): ...) and as an unpacked pair (success, msg = ctrl.connect()).
    """
    def __new__(cls, success: bool, message: str = ""):
        return super().__new__(cls, (bool(success), str(message)))

    def __bool__(self) -> bool:
        return bool(self[0])


# ============================================================================
# 2. Compute Device Detection & Model Warmup (Requirement R4)
# ============================================================================

def detect_best_device(requested_device: str = "auto") -> Tuple[str, str]:
    """
    Safely resolves compute device with validation and graceful CPU fallback.

    Args:
        requested_device: "auto", "0", "cuda", "cuda:0", or "cpu".

    Returns:
        Tuple[str, str]: (device_identifier, human_readable_display_name)
    """
    req = str(requested_device).lower().strip()
    if req in ("cpu", "host"):
        return "cpu", "CPU (Host Processing)"

    has_cuda = False
    try:
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if req in ("auto", "0", "cuda", "cuda:0", "gpu"):
        if has_cuda:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return "0", f"CUDA:0 ({gpu_name}, {vram_gb:.1f} GB)"
            except Exception as e:
                return "cpu", f"CPU (Fallback: CUDA query failed: {e})"
        else:
            return "cpu", "CPU (Host Processing)"

    return "cpu", "CPU (Host Processing)"


def get_inference_device() -> Tuple[str, str]:
    """Backwards-compatibility alias for detect_best_device('auto')."""
    return detect_best_device("auto")


def warmup_engine(model: Any, device: str = "cpu") -> None:
    """
    Warms up the YOLO model graph to absorb the ~2.4s first inference latency
    before real processing begins.
    """
    dummy_input = np.zeros((320, 320, 3), dtype=np.uint8)
    try:
        model(dummy_input, device=device, verbose=False, imgsz=320)
    except Exception:
        if device != "cpu":
            try:
                model(dummy_input, device="cpu", verbose=False, imgsz=320)
            except Exception:
                pass


# ============================================================================
# 3. Engine Configuration (Requirement R2, R3)
# ============================================================================

@dataclass
class EngineConfig:
    """
    Thread-safe configuration dataclass for runtime parameter adjustments.
    """
    conf_threshold: float = 0.50          # Range: 0.01 to 1.00
    iou_threshold: float = 0.45           # Range: 0.10 to 0.90
    eval_duration: float = 2.0            # Seconds; range 0.5 to 10.0
    sleep_duration: float = 5.0           # Seconds; range 0.0 to 30.0 (0.0 = continuous)
    duty_cycle_enabled: bool = True       # Enable/disable duty cycling
    roi_enabled: bool = False             # Enable/disable ROI filtering
    roi_polygon: list = field(default_factory=list)  # Normalized [(u1, v1), ...] in [0.0, 1.0]
    roi_mode: str = "BOTTOM_CENTER"       # "BOTTOM_CENTER" (foot point) or "CENTER"
    temporal_smoothing_enabled: bool = True
    hysteresis_on_frames: int = 3         # Consecutive frames to confirm ON
    hysteresis_off_frames: int = 15       # Consecutive frames to confirm OFF
    min_switch_interval: float = 1.5      # Minimum seconds between serial toggles
    override_mode: str = "AUTO"           # "AUTO", "FORCE_ON", "FORCE_OFF"
    target_classes: list = field(default_factory=lambda: [0, 77])

    def to_dict(self) -> Dict[str, Any]:
        """Returns configuration as a dictionary."""
        return asdict(self)

    def copy(self) -> "EngineConfig":
        """Creates a deep copy of the configuration."""
        return replace(
            self,
            roi_polygon=copy.deepcopy(self.roi_polygon),
            target_classes=list(self.target_classes)
        )

    def update(self, **kwargs) -> None:
        """Updates attributes in-place with validation."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key == "conf_threshold":
                    value = max(0.01, min(1.00, float(value)))
                elif key == "iou_threshold":
                    value = max(0.05, min(0.95, float(value)))
                elif key == "eval_duration":
                    value = max(0.1, float(value))
                elif key == "sleep_duration":
                    value = max(0.0, float(value))
                elif key == "hysteresis_on_frames":
                    value = max(1, int(value))
                elif key == "hysteresis_off_frames":
                    value = max(1, int(value))
                elif key == "min_switch_interval":
                    value = max(0.0, float(value))
                elif key == "override_mode":
                    value = str(value).upper().strip()
                elif key == "roi_mode":
                    value = str(value).upper().strip()
                setattr(self, key, value)


# ============================================================================
# 4. Spatial ROI Filter (Requirement R3)
# ============================================================================

class SpatialROIFilter:
    """
    Manages normalized polygon ROI definition and evaluates spatial containment
    using OpenCV pointPolygonTest with resolution-independent coordinates.
    """
    def __init__(
        self,
        normalized_vertices: Optional[List[Tuple[float, float]]] = None,
        mode: str = "BOTTOM_CENTER",
        enabled: bool = False
    ):
        self.mode = mode.upper().strip()
        self.enabled = bool(enabled)
        self.vertices = self._sanitize_vertices(normalized_vertices)
        self._cached_poly_px = None
        self._last_shape = None

    def _sanitize_vertices(
        self, vertices: Optional[List[Tuple[float, float]]]
    ) -> List[Tuple[float, float]]:
        """Ensures vertices are valid normalized coordinates in [0.0, 1.0]."""
        if not vertices or len(vertices) < 3:
            return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        return [(max(0.0, min(1.0, float(u))), max(0.0, min(1.0, float(v)))) for u, v in vertices]

    def update_polygon(self, normalized_vertices: Optional[List[Tuple[float, float]]]) -> None:
        """Updates the ROI polygon vertices."""
        self.vertices = self._sanitize_vertices(normalized_vertices)
        self._cached_poly_px = None

    def set_mode(self, mode: str) -> None:
        """Sets the anchor mode: 'BOTTOM_CENTER' (feet) or 'CENTER' (centroid)."""
        self.mode = mode.upper().strip()

    def set_enabled(self, enabled: bool) -> None:
        """Enables or disables ROI filtering."""
        self.enabled = bool(enabled)

    def get_anchor_point(
        self, xyxy: Union[np.ndarray, List[int], Tuple[int, ...]]
    ) -> Tuple[int, int]:
        """
        Computes the spatial anchor point for a bounding box [x1, y1, x2, y2].
        BOTTOM_CENTER: ((x1 + x2) // 2, y2) -> physically represents floor/foot contact.
        CENTER: ((x1 + x2) // 2, (y1 + y2) // 2) -> body center.
        """
        x1, y1, x2, y2 = xyxy
        mid_x = int((x1 + x2) / 2)
        if self.mode in ("CENTER", "CENTROID"):
            mid_y = int((y1 + y2) / 2)
            return mid_x, mid_y
        return mid_x, int(y2)

    def _get_pixel_polygon(self, width: int, height: int) -> np.ndarray:
        """Converts normalized vertices to pixel coordinates for the specified frame dimensions."""
        cached = self._cached_poly_px
        if cached is not None and self._last_shape == (width, height):
            return cached
        pts = np.array(
            [[int(u * width), int(v * height)] for u, v in self.vertices],
            dtype=np.int32
        ).reshape((-1, 1, 2))
        self._cached_poly_px = pts
        self._last_shape = (width, height)
        return pts

    def is_point_in_roi(self, point: Tuple[int, int], width: int, height: int) -> bool:
        """Evaluates whether point (x, y) is inside or on the boundary of the polygon."""
        if width <= 0 or height <= 0:
            return False
        px = min(max(0, int(point[0])), width - 1)
        py = min(max(0, int(point[1])), height - 1)
        poly = self._get_pixel_polygon(width, height)
        res = cv2.pointPolygonTest(poly, (float(px), float(py)), measureDist=False)
        return res >= 0

    def filter_detections(
        self,
        detections: List[Dict[str, Any]],
        width: int,
        height: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filters detections by testing spatial anchor containment in the polygon.

        Returns:
            Tuple[List[Dict], List[Dict]]: (inside_detections, outside_detections)
        """
        if not self.enabled:
            for det in detections:
                if isinstance(det, dict):
                    det["in_roi"] = True
                    fx, fy = self.get_anchor_point(det["xyxy"])
                    det["anchor"] = (min(max(0, int(fx)), width - 1), min(max(0, int(fy)), height - 1))
            return list(detections), []

        inside = []
        outside = []
        for det in detections:
            xyxy = det["xyxy"] if isinstance(det, dict) else det[0]
            fx, fy = self.get_anchor_point(xyxy)
            anchor = (min(max(0, int(fx)), width - 1), min(max(0, int(fy)), height - 1))
            in_zone = self.is_point_in_roi(anchor, width, height)
            if isinstance(det, dict):
                det["anchor"] = anchor
                det["in_roi"] = in_zone
            if in_zone:
                inside.append(det)
            else:
                outside.append(det)
        return inside, outside

    def draw_roi_overlay(
        self,
        frame: np.ndarray,
        is_occupied: bool = False,
        label: Optional[str] = None
    ) -> np.ndarray:
        """
        Draws a translucent polygon overlay on the frame.
        - Active/Occupied: Emerald green tint (alpha 0.22) with vibrant outline.
        - Idle/Empty: Cyan/Blue tint (alpha 0.12) with solid outline.
        """
        if not self.enabled:
            return frame

        h, w = frame.shape[:2]
        pts = self._get_pixel_polygon(w, h)
        overlay = frame.copy()

        # Color in BGR: emerald green if occupied, cyan/blue if idle
        color = (50, 220, 50) if is_occupied else (255, 180, 0)
        alpha = 0.22 if is_occupied else 0.12

        cv2.fillPoly(overlay, [pts], color)
        annotated = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

        # Draw status badge at top-left of polygon
        pts_xy = pts.reshape(-1, 2)
        min_x = int(np.min(pts_xy[:, 0]))
        min_y = int(np.min(pts_xy[:, 1]))
        badge_text = label or ("ROI: OCCUPIED" if is_occupied else "ROI: ACTIVE (EMPTY)")
        cv2.putText(
            annotated, badge_text,
            (max(10, min_x + 5), max(25, min_y + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
        return annotated


# ============================================================================
# 5. Temporal Presence Smoother & Relay Debouncer (Requirement R3)
# ============================================================================

class TemporalPresenceSmoother:
    """
    Asymmetric dual-threshold hysteresis state machine:
    - Requires M consecutive positive frames (default 3) to switch ON.
    - Requires N consecutive negative frames (default 15) to switch OFF.
    - Enforces min_switch_interval guard clamp (default 1.5s) to eliminate relay chattering.
    """
    def __init__(
        self,
        on_frames: int = 3,
        off_frames: int = 15,
        min_switch_interval: float = 1.5,
        enabled: bool = True
    ):
        self.on_frames = max(1, int(on_frames))
        self.off_frames = max(1, int(off_frames))
        self.min_switch_interval = max(0.0, float(min_switch_interval))
        self.enabled = bool(enabled)
        self.current_state = False
        self.consecutive_positive = 0
        self.consecutive_negative = 0
        self.last_switch_time = 0.0

    def update_params(
        self,
        on_frames: Optional[int] = None,
        off_frames: Optional[int] = None,
        min_switch_interval: Optional[float] = None,
        enabled: Optional[bool] = None
    ) -> None:
        """Updates smoother parameters dynamically."""
        if on_frames is not None:
            self.on_frames = max(1, int(on_frames))
        if off_frames is not None:
            self.off_frames = max(1, int(off_frames))
        if min_switch_interval is not None:
            self.min_switch_interval = max(0.0, float(min_switch_interval))
        if enabled is not None:
            self.enabled = bool(enabled)

    def reset(self, state: bool = False, now: Optional[float] = None) -> None:
        """Resets the state machine counters."""
        self.current_state = bool(state)
        self.consecutive_positive = 0
        self.consecutive_negative = 0
        self.last_switch_time = now if now is not None else time.time()

    def update(
        self, raw_presence: Any, now: Optional[float] = None
    ) -> Tuple[bool, bool]:
        """
        Updates the filter state.

        Returns:
            Tuple[bool, bool]: (current_smoothed_state, state_has_changed)
        """
        raw_presence = bool(raw_presence)
        if now is None:
            now = time.time()

        state_changed = False

        if not self.enabled:
            if raw_presence != self.current_state:
                if (now - self.last_switch_time) >= self.min_switch_interval:
                    self.current_state = raw_presence
                    self.last_switch_time = now
                    state_changed = True
            return self.current_state, state_changed

        if raw_presence:
            self.consecutive_positive += 1
            self.consecutive_negative = 0
            if not self.current_state and self.consecutive_positive >= self.on_frames:
                if (now - self.last_switch_time) >= self.min_switch_interval:
                    self.current_state = True
                    self.last_switch_time = now
                    state_changed = True
        else:
            self.consecutive_negative += 1
            self.consecutive_positive = 0
            if self.current_state and self.consecutive_negative >= self.off_frames:
                if (now - self.last_switch_time) >= self.min_switch_interval:
                    self.current_state = False
                    self.last_switch_time = now
                    state_changed = True

        return self.current_state, state_changed


# Survey alias
TemporalPresenceDebouncer = TemporalPresenceSmoother


# ============================================================================
# 6. Duty Cycle Manager (Requirement R3)
# ============================================================================

class DutyCycleManager:
    """
    Manages EVALUATING and SLEEPING duty cycle phases to conserve compute power.
    When sleep_duration == 0.0 or duty_cycle_enabled is False, operates in continuous mode.
    """
    def __init__(
        self,
        eval_duration: float = 2.0,
        sleep_duration: float = 5.0,
        enabled: bool = True
    ):
        self.eval_duration = max(0.1, float(eval_duration))
        self.sleep_duration = max(0.0, float(sleep_duration))
        self.enabled = bool(enabled)
        self.current_state = "EVALUATING"
        self.timer_start = 0.0
        self._initialized = False

    def update_durations(
        self,
        eval_duration: Optional[float] = None,
        sleep_duration: Optional[float] = None,
        enabled: Optional[bool] = None
    ) -> None:
        """Dynamically adjusts duty cycle timings at runtime."""
        if eval_duration is not None:
            self.eval_duration = max(0.1, float(eval_duration))
        if sleep_duration is not None:
            self.sleep_duration = max(0.0, float(sleep_duration))
        if enabled is not None:
            self.enabled = bool(enabled)

    def reset(self, phase: str = "EVALUATING", now: Optional[float] = None) -> None:
        """Resets the duty cycle timer to the specified phase."""
        self.current_state = phase.upper()
        self.timer_start = now if now is not None else time.time()
        self._initialized = True

    def should_infer(self) -> bool:
        """Returns True if YOLO inference should run in the current phase."""
        if not self.enabled or self.sleep_duration <= 0.0:
            return True
        return self.current_state == "EVALUATING"

    def update(
        self, now: Optional[float] = None
    ) -> Tuple[str, float, bool]:
        """
        Updates duty cycle timers and handles phase transitions.

        Returns:
            Tuple[str, float, bool]: (current_state, countdown_seconds, phase_transitioned)
        """
        if now is None:
            now = time.time()

        if not self._initialized:
            self.timer_start = now
            self._initialized = True

        if not self.enabled or self.sleep_duration <= 0.0:
            self.current_state = "EVALUATING"
            return "EVALUATING", 0.0, False

        elapsed = now - self.timer_start
        transitioned = False

        if self.current_state == "EVALUATING":
            if elapsed >= self.eval_duration:
                self.current_state = "SLEEPING"
                self.timer_start = now
                transitioned = True
                countdown = self.sleep_duration
            else:
                countdown = max(0.0, self.eval_duration - elapsed)
        else:
            if elapsed >= self.sleep_duration:
                self.current_state = "EVALUATING"
                self.timer_start = now
                transitioned = True
                countdown = self.eval_duration
            else:
                countdown = max(0.0, self.sleep_duration - elapsed)

        return self.current_state, countdown, transitioned


# ============================================================================
# 7. Hardware Abstraction Layer (Requirement R4)
# ============================================================================

class BaseHardwareController:
    """Abstract interface contract for physical and simulated hardware controllers."""
    def connect(self, port: Optional[str] = None) -> Any:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def set_light(self, state: bool) -> bool:
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        raise NotImplementedError

    @property
    def current_state(self) -> bool:
        raise NotImplementedError

    @property
    def port(self) -> Optional[str]:
        return None


class HardwareController(BaseHardwareController):
    """
    Manages robust USB serial communication with Arduino Uno/Nano at 9600 baud.
    Includes auto-detection of COM ports, clean buffer handling, and rate limiting.
    """
    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 9600,
        min_switch_interval: float = 1.5
    ):
        self._port = port
        self.baudrate = baudrate
        self.min_switch_interval = min_switch_interval
        self.serial_conn = None
        self._is_connected = False
        self._current_state = False
        self._last_switch_time = 0.0
        self.lock = threading.Lock()

    @staticmethod
    def list_available_ports() -> List[str]:
        """Returns list of detected serial COM ports."""
        try:
            ports = serial.tools.list_ports.comports()
            return [p.device for p in ports]
        except Exception:
            return []

    @property
    def is_connected(self) -> bool:
        with self.lock:
            return self._is_connected and (self.serial_conn is not None) and self.serial_conn.is_open

    @property
    def current_state(self) -> bool:
        with self.lock:
            return self._current_state

    @property
    def port(self) -> Optional[str]:
        return self._port

    def connect(self, port: Optional[str] = None) -> ConnectionResult:
        """Connects to serial port, draining inbound buffers and setting initial state."""
        with self.lock:
            if port:
                self._port = port
            if not self._port:
                available = self.list_available_ports()
                if available:
                    self._port = available[0]
                else:
                    return ConnectionResult(False, "No serial COM ports found on system")

            try:
                if self.serial_conn and self.serial_conn.is_open:
                    self.serial_conn.close()
                self.serial_conn = serial.Serial(
                    self._port, self.baudrate, timeout=0.5, write_timeout=0.5
                )
                time.sleep(1.0)  # Allow Arduino bootloader to initialize
                # Drain any unread bytes (e.g. "READY\r\n")
                try:
                    self.serial_conn.reset_input_buffer()
                except Exception:
                    pass

                self._is_connected = True
                self._send_raw("0")
                self._current_state = False
                return ConnectionResult(True, f"Connected to {self._port}")
            except Exception as e:
                self._is_connected = False
                self.serial_conn = None
                return ConnectionResult(False, f"Failed connecting to {self._port}: {e}")

    def disconnect(self) -> None:
        """Gracefully disconnects serial link, turning off relay first."""
        with self.lock:
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    self._send_raw("0")
                    self.serial_conn.close()
                except Exception:
                    pass
            self._is_connected = False
            self.serial_conn = None

    def _send_raw(self, char: str) -> None:
        """Internal helper to write byte and flush."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write(char.encode("ascii"))
            self.serial_conn.flush()
            # Drain inbound response bytes to prevent buffer overflow
            try:
                if self.serial_conn.in_waiting > 0:
                    self.serial_conn.read_all()
            except Exception:
                pass

    def set_light(self, turn_on: bool) -> bool:
        """
        Sends command '1' (ON) or '0' (OFF) with edge detection and rate clamp.
        """
        with self.lock:
            now = time.time()
            if turn_on != self._current_state:
                if (now - self._last_switch_time) < self.min_switch_interval:
                    return False  # Rate limit clamp active

                self._current_state = turn_on
                self._last_switch_time = now
                cmd = "1" if turn_on else "0"
                if self.serial_conn and self.serial_conn.is_open:
                    try:
                        self._send_raw(cmd)
                        return True
                    except Exception as e:
                        print(f"[HardwareController] Serial write error: {e}")
                        self._is_connected = False
                        return False
                return False
            return True


class MockHardwareController(BaseHardwareController):
    """
    In-memory hardware controller recording timestamps and commands ('1' and '0')
    for headless automated testing without physical Arduino hardware.
    """
    def __init__(self, port: str = "MOCK_COM1", auto_connect: bool = True):
        self._port = port
        self._is_connected = bool(auto_connect)
        self._current_state = False
        self.command_log: List[Tuple[float, str]] = []
        self.lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        with self.lock:
            return self._is_connected

    @property
    def current_state(self) -> bool:
        with self.lock:
            return self._current_state

    @property
    def port(self) -> str:
        return self._port

    @property
    def command_history(self) -> List[str]:
        """Returns sequence of commands ['1', '0', ...] without timestamps."""
        with self.lock:
            return [cmd for _, cmd in self.command_log]

    def connect(self, port: Optional[str] = None) -> ConnectionResult:
        with self.lock:
            if port:
                self._port = port
            self._is_connected = True
            return ConnectionResult(True, f"Connected to {self._port}")

    def disconnect(self) -> None:
        with self.lock:
            self._is_connected = False

    def set_light(self, state: bool) -> bool:
        """Records state transition command ('1' or '0') on edge change."""
        with self.lock:
            if not self._is_connected:
                return False
            if state != self._current_state:
                cmd = "1" if state else "0"
                self.command_log.append((time.time(), cmd))
                self._current_state = state
            return True

    def get_command_log(self) -> List[Tuple[float, str]]:
        """Returns copy of recorded [(timestamp, '1'), ...] command log."""
        with self.lock:
            return list(self.command_log)

    def reset_log(self) -> None:
        """Clears recorded command history."""
        with self.lock:
            self.command_log.clear()


# ============================================================================
# 8. Video Stream Abstraction (Requirement R4)
# ============================================================================

class BaseVideoStream:
    """Abstract interface contract for video streams."""
    def start(self) -> "BaseVideoStream":
        raise NotImplementedError

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    @property
    def is_opened(self) -> bool:
        raise NotImplementedError


class VideoStream(BaseVideoStream):
    """
    Threaded OpenCV video capture to eliminate frame buffering lag.
    Supports webcam index, DroidCam URL, video files, and fallback simulation.
    """
    def __init__(self, source: Any = 0):
        self.source = source
        self.cap = None
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.is_simulation = False
        self._sim_x = 50
        self._sim_dx = 4

    def start(self) -> "VideoStream":
        self.running = True
        self.is_simulation = False

        try:
            if isinstance(self.source, str) and self.source.isdigit():
                src = int(self.source)
            elif self.source in ("simulation", "mock", -1):
                self.is_simulation = True
                src = None
            else:
                src = self.source

            if src is not None:
                self.cap = cv2.VideoCapture(src)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not self.cap.isOpened():
                    self.is_simulation = True
                    self.cap = None
        except Exception:
            self.is_simulation = True
            self.cap = None

        self.thread = threading.Thread(target=self._update_loop, daemon=True, name="VideoStreamThread")
        self.thread.start()
        return self

    def _update_loop(self) -> None:
        while self.running:
            if not self.is_simulation and self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.01)
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(frame, (20, 20), (620, 460), (40, 40, 40), -1)
                cv2.putText(
                    frame, "CAMERA SIMULATION MODE", (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                )
                cv2.putText(
                    frame, "Select valid webcam or DroidCam source", (60, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1
                )

                self._sim_x += self._sim_dx
                if self._sim_x > 500 or self._sim_x < 50:
                    self._sim_dx *= -1

                cv2.circle(frame, (self._sim_x + 30, 240), 25, (0, 200, 255), -1)
                cv2.rectangle(frame, (self._sim_x, 265), (self._sim_x + 60, 360), (0, 150, 220), -1)
                cv2.putText(
                    frame, "Simulated Subject", (self._sim_x - 20, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )

                with self.lock:
                    self.frame = frame
                time.sleep(0.03)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def is_opened(self) -> bool:
        if self.is_simulation:
            return True
        return (self.cap is not None) and self.cap.isOpened()


class MockVideoSource(BaseVideoStream):
    """
    Deterministic frame generator utilizing local Ultralytics bus.jpg asset
    (verified 3-4 person detections) and blank frames for automated headless verification.
    """
    def __init__(self, default_scenario: str = "PERSON_IN_ROI"):
        self._is_opened = True
        self.is_simulation = True
        self.lock = threading.Lock()

        # Locate bus.jpg asset
        bus_candidate = None
        if ASSETS and os.path.exists(os.path.join(ASSETS, "bus.jpg")):
            bus_candidate = os.path.join(ASSETS, "bus.jpg")
        else:
            local_paths = [
                os.path.join(os.path.dirname(__file__), "bus.jpg"),
                os.path.join(os.path.dirname(__file__), "Model Demo", "bus.jpg"),
            ]
            for p in local_paths:
                if os.path.exists(p):
                    bus_candidate = p
                    break

        if bus_candidate and os.path.exists(bus_candidate):
            self._bus_img = cv2.imread(bus_candidate)
        else:
            # Fallback high-contrast synthetic image
            self._bus_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(self._bus_img, (100, 100), (300, 400), (255, 255, 255), -1)

        h, w = self._bus_img.shape[:2]
        # Right-only frame: mask out left 60% so only person on the far right (x > 0.65) is detected
        self._right_person_img = self._bus_img.copy()
        self._right_person_img[:, :int(w * 0.60)] = 0

        # Left-only frame: mask out right 50% so only persons on the left (x < 0.50) are detected
        self._left_persons_img = self._bus_img.copy()
        self._left_persons_img[:, int(w * 0.50):] = 0

        # Empty frame: all zeros -> 0 detections
        self._empty_img = np.zeros((480, 640, 3), dtype=np.uint8)

        self._current_frame = self._bus_img.copy()
        self.set_scenario(default_scenario)

    def set_scenario(self, scenario: str) -> None:
        """
        Switches the deterministic frame scenario:
        - 'PERSON_IN_ROI': Yields full bus.jpg (3 persons detected).
        - 'PERSON_OUTSIDE_ROI': Yields right_only frame (person at x=0.91, outside left-half ROIs).
        - 'EMPTY_FRAME': Yields blank frame (0 persons detected).
        """
        sc = scenario.upper().strip()
        with self.lock:
            if sc in ("PERSON_IN_ROI", "BUS", "PRESENT"):
                self._current_frame = self._bus_img.copy()
            elif sc in ("PERSON_OUTSIDE_ROI", "RIGHT_ONLY", "OUTSIDE"):
                self._current_frame = self._right_person_img.copy()
            elif sc in ("LEFT_PERSONS", "LEFT_ONLY"):
                self._current_frame = self._left_persons_img.copy()
            elif sc in ("EMPTY_FRAME", "EMPTY", "ABSENT", "BLANK"):
                self._current_frame = self._empty_img.copy()
            else:
                self._current_frame = self._empty_img.copy()

    def set_custom_frame(self, frame: Optional[np.ndarray]) -> None:
        """Sets an arbitrary numpy frame for custom test scenarios."""
        with self.lock:
            if frame is not None and hasattr(frame, "copy"):
                self._current_frame = frame.copy()
            else:
                self._current_frame = None

    def start(self) -> "MockVideoSource":
        self._is_opened = True
        return self

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self._current_frame is not None and hasattr(self._current_frame, "copy"):
                return True, self._current_frame.copy()
            elif self._current_frame is not None:
                return True, self._current_frame
            return False, None

    def stop(self) -> None:
        self._is_opened = False

    @property
    def is_opened(self) -> bool:
        return self._is_opened


# ============================================================================
# 9. Advanced Detection Engine (Requirements R1, R2, R3, R4)
# ============================================================================

class AdvancedDetectionEngine:
    """
    High-performance decoupled multi-threaded detection engine for AIEcoLightswitcher.
    Runs YOLOv8 inference, Spatial ROI filtering, temporal presence smoothing,
    duty-cycle power management, and hardware relay switching in a background thread.
    """
    def __init__(
        self,
        source: Any = 0,
        port: Optional[str] = None,
        config: Optional[EngineConfig] = None,
        model_path: str = "yolov8n.pt",
        hardware: Optional[BaseHardwareController] = None,
        video_stream: Optional[BaseVideoStream] = None
    ):
        self.device, self.device_info = detect_best_device()
        self.model_path = model_path
        self.model = YOLO(model_path)
        warmup_engine(self.model, self.device)

        self.config = config.copy() if config else EngineConfig()
        self.target_class_names = DEFAULT_TARGET_CLASSES

        # Instantiate algorithm modules
        self.roi_filter = SpatialROIFilter(
            normalized_vertices=self.config.roi_polygon,
            mode=self.config.roi_mode,
            enabled=self.config.roi_enabled
        )
        self.smoother = TemporalPresenceSmoother(
            on_frames=self.config.hysteresis_on_frames,
            off_frames=self.config.hysteresis_off_frames,
            min_switch_interval=self.config.min_switch_interval,
            enabled=self.config.temporal_smoothing_enabled
        )
        self.duty_cycle = DutyCycleManager(
            eval_duration=self.config.eval_duration,
            sleep_duration=self.config.sleep_duration,
            enabled=self.config.duty_cycle_enabled
        )

        # Hardware setup
        if hardware is not None:
            self.hardware = hardware
        elif port and ("mock" in str(port).lower()):
            self.hardware = MockHardwareController(port=str(port))
        else:
            self.hardware = HardwareController(
                port=port,
                min_switch_interval=self.config.min_switch_interval
            )

        # Video stream setup
        if video_stream is not None:
            self.stream = video_stream
        elif source in ("mock", "simulation") or isinstance(source, MockVideoSource):
            self.stream = source if isinstance(source, MockVideoSource) else MockVideoSource()
        else:
            self.stream = VideoStream(source=source)

        # Threading and synchronization state
        self._running = False
        self._worker_thread = None
        self._lock = threading.Lock()
        self._config_lock = threading.RLock()

        self._latest_annotated_frame = None
        self._latest_telemetry: Dict[str, Any] = {}
        self._fps_ema = 0.0
        self._last_raw_occupants = 0
        self._last_roi_occupants = 0
        self._max_cycle_occupants = 0
        self._last_detections = ([], [])
        self._last_latency_ms = 0.0
        self._first_sync = True

    def start(self) -> None:
        """Starts video capture, serial connection, and decoupled inference worker."""
        if self._running:
            return

        self.stream.start()
        if self.hardware.port and not self.hardware.is_connected:
            self.hardware.connect()

        self.duty_cycle.reset("EVALUATING")
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="DetectionWorker"
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """Stops the worker thread, video capture, and hardware serial link."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self.stream.stop()
        self.hardware.disconnect()

    def update_config(self, **kwargs) -> None:
        """Thread-safe update of runtime detection parameters."""
        with self._config_lock:
            self.config.update(**kwargs)
            if "roi_polygon" in kwargs:
                self.roi_filter.update_polygon(self.config.roi_polygon)
            if "roi_enabled" in kwargs:
                self.roi_filter.set_enabled(self.config.roi_enabled)
            if "roi_mode" in kwargs:
                self.roi_filter.set_mode(self.config.roi_mode)
            if any(k in kwargs for k in ("hysteresis_on_frames", "hysteresis_off_frames",
                                         "min_switch_interval", "temporal_smoothing_enabled")):
                self.smoother.update_params(
                    on_frames=self.config.hysteresis_on_frames,
                    off_frames=self.config.hysteresis_off_frames,
                    min_switch_interval=self.config.min_switch_interval,
                    enabled=self.config.temporal_smoothing_enabled
                )
            if any(k in kwargs for k in ("eval_duration", "sleep_duration", "duty_cycle_enabled")):
                self.duty_cycle.update_durations(
                    eval_duration=self.config.eval_duration,
                    sleep_duration=self.config.sleep_duration,
                    enabled=self.config.duty_cycle_enabled
                )
        if "override_mode" in kwargs:
            self.set_override(self.config.override_mode)

    def get_config(self) -> EngineConfig:
        """Returns snapshot of current configuration."""
        with self._config_lock:
            return self.config.copy()

    def set_override(self, mode: str) -> None:
        """Applies manual override: 'AUTO', 'FORCE_ON', or 'FORCE_OFF'."""
        mode = str(mode).upper().strip()
        with self._config_lock:
            self.config.override_mode = mode

        if mode == "FORCE_ON":
            self.hardware.set_light(True)
        elif mode == "FORCE_OFF":
            self.hardware.set_light(False)
        elif mode == "AUTO":
            self.hardware.set_light(self.smoother.current_state)

    def connect_serial(self, port: str) -> bool:
        """Connects serial hardware to the specified COM port or mock."""
        if "mock" in str(port).lower():
            self.hardware = MockHardwareController(port=port)
            return bool(self.hardware.connect())
        if not isinstance(self.hardware, HardwareController):
            self.hardware = HardwareController(
                port=port,
                min_switch_interval=self.config.min_switch_interval
            )
        res = self.hardware.connect(port)
        return bool(res)

    def disconnect_serial(self) -> None:
        """Disconnects serial hardware cleanly."""
        self.hardware.disconnect()

    def switch_source(self, new_source: Any) -> bool:
        """Safely switches video source without interrupting the worker loop."""
        if new_source in ("mock", "simulation") or isinstance(new_source, MockVideoSource):
            new_stream = new_source if isinstance(new_source, MockVideoSource) else MockVideoSource()
        else:
            new_stream = VideoStream(source=new_source)

        new_stream.start()
        old_stream = self.stream
        self.stream = new_stream
        old_stream.stop()
        return self.stream.is_opened

    def set_camera_source(self, new_source: Any) -> bool:
        """Backwards compatibility alias for switch_source."""
        return self.switch_source(new_source)

    def _execute_frame_step(
        self, raw_frame: np.ndarray, now: float, cfg: EngineConfig
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Executes a single end-to-end processing step on a frame."""
        if not isinstance(raw_frame, np.ndarray) or raw_frame.ndim < 2 or raw_frame.size == 0:
            raise ValueError(
                f"Invalid frame format: expected non-empty 2D/3D ndarray, got {type(raw_frame)} "
                f"with shape {getattr(raw_frame, 'shape', None)}"
            )
        h, w = raw_frame.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid frame dimensions: {w}x{h}")

        # 1. Duty cycle update
        cycle_state, cycle_countdown, phase_transitioned = self.duty_cycle.update(now)
        if phase_transitioned and cycle_state == "EVALUATING":
            self._max_cycle_occupants = 0

        should_infer = self.duty_cycle.should_infer()

        detections = []
        raw_occupants = 0
        roi_occupants = 0
        latency_ms = 0.0

        # 2. YOLOv8 Inference
        if should_infer:
            t0 = time.perf_counter()
            try:
                results = self.model(
                    raw_frame,
                    device=self.device,
                    verbose=False,
                    conf=cfg.conf_threshold,
                    iou=cfg.iou_threshold,
                    classes=cfg.target_classes
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0

                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    label = f"{self.target_class_names.get(cls_id, str(cls_id))} {conf:.2f}"
                    detections.append({
                        "xyxy": xyxy,
                        "conf": conf,
                        "cls_id": cls_id,
                        "label": label
                    })
                del results
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = "cpu"
                latency_ms = 0.0
            except Exception:
                latency_ms = 0.0

            raw_occupants = len(detections)
            inside_dets, outside_dets = self.roi_filter.filter_detections(detections, w, h)
            roi_occupants = len(inside_dets) if cfg.roi_enabled else raw_occupants

            self._last_raw_occupants = raw_occupants
            self._last_roi_occupants = roi_occupants
            self._last_detections = (inside_dets, outside_dets)
            self._last_latency_ms = latency_ms

            if roi_occupants > self._max_cycle_occupants:
                self._max_cycle_occupants = roi_occupants
        else:
            raw_occupants = self._last_raw_occupants
            roi_occupants = self._last_roi_occupants
            latency_ms = self._last_latency_ms
            inside_dets, outside_dets = self._last_detections

        # 3. Temporal Presence Smoothing
        raw_presence = (roi_occupants > 0)
        smoothed_presence, state_changed = self.smoother.update(raw_presence, now)

        # 4. Hardware Relay Switching
        if cfg.override_mode == "FORCE_ON":
            if not self.hardware.current_state:
                self.hardware.set_light(True)
        elif cfg.override_mode == "FORCE_OFF":
            if self.hardware.current_state:
                self.hardware.set_light(False)
        elif cfg.override_mode == "AUTO":
            if state_changed or (self.hardware.current_state != smoothed_presence):
                self.hardware.set_light(smoothed_presence)
                self._first_sync = False

        # 5. Visual Rendering & Overlays
        annotated_frame = raw_frame.copy()

        if cfg.roi_enabled:
            annotated_frame = self.roi_filter.draw_roi_overlay(
                annotated_frame, is_occupied=(roi_occupants > 0)
            )

        # Render Detections Inside ROI (green)
        for det in inside_dets:
            xyxy = det["xyxy"]
            color = (0, 255, 0) if det["cls_id"] == 0 else (255, 200, 0)
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(
                annotated_frame, det["label"],
                (xyxy[0], max(20, xyxy[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            anchor = det.get("anchor")
            if anchor:
                cv2.circle(annotated_frame, anchor, 4, (0, 0, 255), -1)

        # Render Detections Outside ROI (dim gray)
        for det in outside_dets:
            xyxy = det["xyxy"]
            color = (130, 130, 130)
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 1)
            cv2.putText(
                annotated_frame, f"{det['label']} (Outside ROI)",
                (xyxy[0], max(20, xyxy[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
            )
            anchor = det.get("anchor")
            if anchor:
                cv2.circle(annotated_frame, anchor, 3, (100, 100, 100), -1)

        # Sleep Phase Visual Dimming Overlay
        if cycle_state == "SLEEPING":
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.35, annotated_frame, 0.65, 0, annotated_frame)
            cv2.putText(
                annotated_frame,
                f"POWER SAVING IDLE (Resumes in {cycle_countdown:.1f}s)",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2
            )

        # 6. FPS Calculation
        frame_time = time.time() - now
        current_fps = 1.0 / max(frame_time, 0.001)
        self._fps_ema = (
            0.9 * self._fps_ema + 0.1 * current_fps if self._fps_ema > 0 else current_fps
        )

        telemetry = {
            "fps": round(self._fps_ema, 1),
            "latency_ms": round(latency_ms, 1),
            "raw_occupants": raw_occupants,
            "roi_occupants": roi_occupants,
            "occupants": roi_occupants,
            "max_occupants": self._max_cycle_occupants,
            "smoothed_presence": smoothed_presence,
            "cycle_state": cycle_state,
            "cycle_countdown": round(cycle_countdown, 1),
            "cycle_time_left": round(cycle_countdown, 1),
            "relay_state": self.hardware.current_state,
            "light_state": self.hardware.current_state,
            "device": self.device,
            "device_info": self.device_info,
            "override_mode": cfg.override_mode,
            "hardware_connected": self.hardware.is_connected,
            "is_connected": self.hardware.is_connected,
            "camera_connected": self.stream.is_opened,
            "com_port": self.hardware.port,
            "is_simulation": getattr(self.stream, "is_simulation", False)
        }

        return annotated_frame, telemetry

    def _worker_loop(self) -> None:
        """Background thread worker loop running inference and state management."""
        while self._running:
            with self._config_lock:
                cfg = self.config.copy()

            try:
                ret, raw_frame = self.stream.read()
            except Exception as exc:
                logger.error(f"[eco_engine] Stream read error: {exc}")
                time.sleep(0.01)
                continue

            if not ret or raw_frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            try:
                annotated_frame, telemetry = self._execute_frame_step(raw_frame, now, cfg)
                with self._lock:
                    self._latest_annotated_frame = annotated_frame
                    self._latest_telemetry = telemetry
            except Exception as exc:
                logger.error(f"[eco_engine] Worker loop frame step error: {exc}")
                time.sleep(0.01)
                continue

            time.sleep(0.005)

    def get_latest_frame_and_telemetry(
        self
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Retrieves the latest annotated frame and telemetry dictionary.
        Thread-safe and non-blocking when the background worker is active.
        """
        if not self._running:
            return self.process_frame()

        with self._lock:
            if self._latest_annotated_frame is not None:
                return self._latest_annotated_frame.copy(), dict(self._latest_telemetry)

        ret, frame = self.stream.read()
        return frame, dict(self._latest_telemetry)

    def process_frame(self) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Processes a single frame. If the worker thread is running, returns latest frame.
        Otherwise executes a single frame step synchronously.
        """
        if self._running:
            return self.get_latest_frame_and_telemetry()

        ret, raw_frame = self.stream.read()
        if not ret or raw_frame is None:
            return None, {}

        with self._config_lock:
            cfg = self.config.copy()

        now = time.time()
        try:
            annotated_frame, telemetry = self._execute_frame_step(raw_frame, now, cfg)
            with self._lock:
                self._latest_annotated_frame = annotated_frame
                self._latest_telemetry = telemetry
            return annotated_frame, telemetry
        except Exception as exc:
            logger.error(f"[eco_engine] Synchronous process_frame error: {exc}")
            return None, {}


# ============================================================================
# 10. Backwards Compatibility Wrapper (EcoVisionSystem)
# ============================================================================

class EcoVisionSystem(AdvancedDetectionEngine):
    """
    Backwards-compatible wrapper providing full interface compatibility
    with existing code while leveraging the new decoupled pipeline.
    """
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        source: Any = 0,
        arduino_port: Optional[str] = None
    ):
        super().__init__(
            source=source,
            port=arduino_port,
            model_path=model_path
        )

    @property
    def light_state(self) -> bool:
        return self.hardware.current_state

    @light_state.setter
    def light_state(self, val: bool) -> None:
        self.hardware.set_light(val)

    @property
    def override_mode(self) -> str:
        return self.config.override_mode

    @override_mode.setter
    def override_mode(self, val: str) -> None:
        self.set_override(val)

    @property
    def duty_cycle_enabled(self) -> bool:
        return self.config.duty_cycle_enabled

    @duty_cycle_enabled.setter
    def duty_cycle_enabled(self, val: bool) -> None:
        self.update_config(duty_cycle_enabled=val)

    @property
    def eval_duration(self) -> float:
        return self.config.eval_duration

    @eval_duration.setter
    def eval_duration(self, val: float) -> None:
        self.update_config(eval_duration=val)

    @property
    def sleep_duration(self) -> float:
        return self.config.sleep_duration

    @sleep_duration.setter
    def sleep_duration(self, val: float) -> None:
        self.update_config(sleep_duration=val)

    @property
    def conf_threshold(self) -> float:
        return self.config.conf_threshold

    @conf_threshold.setter
    def conf_threshold(self, val: float) -> None:
        self.update_config(conf_threshold=val)

    @property
    def current_cycle_state(self) -> str:
        return self.duty_cycle.current_state

    @property
    def fps(self) -> float:
        return self._fps_ema

    @property
    def last_detection_count(self) -> int:
        return self._last_roi_occupants

    @property
    def max_cycle_occupants(self) -> int:
        return self._max_cycle_occupants

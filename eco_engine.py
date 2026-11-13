"""
eco_engine.py - Core Detection Engine & Backwards-Compatibility Facade
AIEcoLightswitcher Showcase Edition (Milestone 1 Implementation)

Acts as a high-level facade re-exporting modular components from:
- core/ (config, device, roi_filter, smoother, duty_cycle)
- ingestion/ (base_stream, stream_worker, video_stream, mock_stream)

Retains:
- BaseHardwareController, HardwareController, MockHardwareController (migrating in M2)
- AdvancedDetectionEngine, EcoVisionSystem
Maintains 100% backwards compatibility for legacy callers and automated test suites.
"""

import copy
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

# ============================================================================
# Re-exports from modular core/ package
# ============================================================================
from core.config import (
    ConnectionResult,
    DEFAULT_TARGET_CLASSES,
    EngineConfig,
    FeedConfig,
)
from core.device import (
    detect_best_device,
    get_inference_device,
    warmup_engine,
)
from core.duty_cycle import DutyCycleManager
from core.roi_filter import SpatialROIFilter
from core.smoother import (
    TemporalPresenceDebouncer,
    TemporalPresenceSmoother,
)

# ============================================================================
# Re-exports from modular ingestion/ package
# ============================================================================
from ingestion.base_stream import (
    BaseVideoStream,
    FeedStatus,
    StreamTelemetry,
)
from ingestion.mock_stream import (
    MockVideoSource,
    create_mock_feed_grid,
)
from ingestion.stream_worker import (
    BufferedFrame,
    SingleFrameBuffer,
    StreamWorker,
)
from ingestion.video_stream import (
    VideoStream,
    open_capture,
)

logger = logging.getLogger("eco_engine")


# ============================================================================
# Hardware Abstraction Layer (Preserved for M1; refactored in M2)
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
# Core Detection Engine (Preserving internal attributes for test compatibility)
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

        # Instantiate algorithm modules from core/
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

        # Video stream setup from ingestion/
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


# ============================================================================
# Facade __all__ Export Manifest
# ============================================================================

__all__ = [
    # Core Algorithms & Config
    "ConnectionResult",
    "DEFAULT_TARGET_CLASSES",
    "DutyCycleManager",
    "EngineConfig",
    "FeedConfig",
    "SpatialROIFilter",
    "TemporalPresenceDebouncer",
    "TemporalPresenceSmoother",
    "detect_best_device",
    "get_inference_device",
    "warmup_engine",
    # Ingestion & Video Streams
    "BaseVideoStream",
    "BufferedFrame",
    "FeedStatus",
    "MockVideoSource",
    "SingleFrameBuffer",
    "StreamTelemetry",
    "StreamWorker",
    "VideoStream",
    "create_mock_feed_grid",
    "open_capture",
    # Hardware Controllers
    "BaseHardwareController",
    "HardwareController",
    "MockHardwareController",
    # Detection Pipeline & Facade
    "AdvancedDetectionEngine",
    "EcoVisionSystem",
]

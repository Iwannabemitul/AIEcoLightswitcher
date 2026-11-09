# Project: AIEcoLightswitcher — Showcase Edition

## Architecture
AIEcoLightswitcher is a high-performance computer vision smart energy system that optimizes electrical lighting based on real-time human presence detection. The architecture employs a 3-tier decoupled multi-threaded pipeline to ensure zero UI stutter and 60 FPS responsiveness on desktop environments:
1. **Core Detection Engine (`eco_engine.py`)**:
   - `AdvancedDetectionEngine`: Manages asynchronous YOLOv8 inference, engine warmup, memory management, and duty cycling.
   - `SpatialROIFilter`: Resolution-independent normalized polygon ROI containment using OpenCV `pointPolygonTest` and foot-point / center-point spatial anchors.
   - `TemporalPresenceSmoother`: Asymmetric dual-threshold hysteresis state machine ($M=3$ frames ON, $N=15$ frames OFF) and minimum relay switching interval guard ($\Delta t_{\min}=1.5$s) to eliminate relay chattering and flicker.
   - `DutyCycleManager`: Manages `EVALUATING` $\leftrightarrow$ `SLEEPING` state transitions with dynamic duration adjustment and continuous mode.
   - `EngineConfig`: Thread-safe configuration dataclass for runtime parameter adjustments without restarting threads.
2. **Hardware & Simulation Layer (`hardware/` & `eco_engine.py`)**:
   - `HardwareController`: Physical USB serial Arduino communication (commands `'1'` and `'0'`), bidirectional port handshake, and connection state management.
   - `MockHardwareController`: In-memory serial simulator recording command history and state transitions for headless testing.
   - `VideoStream`: Threaded OpenCV camera ingestion supporting Webcams, DroidCam (IP/URL), and video files.
   - `MockVideoSource`: Deterministic synthetic video feed yielding known person detections (via Ultralytics `bus.jpg` asset) or empty frames for testing presence inside/outside ROIs.
3. **Showcase User Interface (`Model Demo/main.py` / `gui/`)**:
   - Built on CustomTkinter 6.0 in modern engineering dark mode (`#121417` deep slate, `#1E222B` card surfaces, emerald `#2ECC71` active status).
   - High-framerate double-buffered video canvas with bounding boxes, ROI overlays, and duty-cycle sleep banner.
   - Live 6-metric telemetry bar: FPS, Inference Latency (ms), Target Confidence, Smoothed Presence, Cycle Countdown, Serial Relay State.
   - Interactive control panels: runtime sliders for confidence (0.01–1.00), eval duration (0.5–10.0s), sleep duration (1.0–30.0s), ROI toggle, and manual overrides (`AUTO`, `FORCE_ON`, `FORCE_OFF`).
   - Hardware connection hub: Camera source dropdown, Serial port auto-detection / Mock mode selector, Connect/Disconnect buttons, and live status badges.
4. **E2E Testing & Verification Track (`tests/` & `run_verification.py`)**:
   - Automated test suite using Python standard library `unittest` executing headlessly without physical camera or Arduino hardware.
   - Tests Tiers 1–4: Feature coverage, Boundary/Corner cases, Cross-feature interactions, and Real-world application scenarios.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Modern Dashboard UI | Professional CustomTkinter dark-theme layout with responsive grid, card panels, and clean typography | M2 | R1 |
| 2 | Telemetry HUD Bar | Real-time display of FPS, latency (ms), detection confidence, smoothed presence state, duty cycle countdown, and serial relay command state | M2 | R1 |
| 3 | Video Canvas & Overlays | Double-buffered canvas rendering live feed, bounding boxes, labels with confidence, ROI dashed outline, and duty cycle sleep banner | M2 | R1 |
| 4 | Decoupled Threading Pipeline | 3-tier threading (UI thread, Inference worker thread, Serial worker thread) with thread-safe buffers ensuring 0 UI freezing | M1 | R1, R2 |
| 5 | Live Interactive Parameter Sliders | Dynamic runtime tuning of confidence threshold (0.01–1.00), evaluation duration (0.5–10.0s), and sleep duration (1.0–30.0s) without restarting | M2 | R2 |
| 6 | Manual Hardware Overrides | Runtime mode selection: `AUTO`, `FORCE_ON` (command '1'), `FORCE_OFF` (command '0') with visual feedback | M2 | R2 |
| 7 | Hardware Connection Hub | Dropdown controls for camera source selection (webcam, DroidCam, mock) and serial port selection (COM ports, mock serial) with connect/disconnect | M2 | R2 |
| 8 | CUDA Auto-Detection & Fallback | `detect_best_device` checking `torch.cuda.is_available()`, device selection ('0' vs 'cpu'), startup warmup pass, and graceful CPU fallback | M1 | R4 |
| 9 | Configurable Spatial Zones (ROI) | Normalized coordinate polygon/box ROI filtering with bottom-center foot point and center point containment via `cv2.pointPolygonTest` | M1 | R3 |
| 10 | Temporal Presence Smoothing | Asymmetric hysteresis state machine ($M=3$ ON, $N=15$ OFF) and minimum switching guard ($\Delta t_{\min}=1.5$s) preventing relay chattering | M1 | R3 |
| 11 | Dynamic Duty Cycle Logic | Evaluation vs. sleep cycle management with dynamic timing adjustment and continuous evaluation mode | M1 | R3 |
| 12 | Mock Hardware & Video Drivers | `MockHardwareController` (command recording & query) and `MockVideoSource` (deterministic person & empty frame generator) | M1 | R4, AC |
| 13 | E2E Mock Verification Suite | Headless test suite (`run_verification.py`) executing Tiers 1–4 tests validating presence $\rightarrow$ '1', absence $\rightarrow$ '0', duty cycle, and ROI without hardware | E2E Track | AC |
| 14 | Adversarial Hardening (Tier 5) | White-box edge-case and boundary verification with automated Challenger testing | M3 | System Quality |

## Milestones
| # | Track | Name | Scope | Dependencies | Status |
|---|-------|------|-------|-------------|--------|
| E2E | Testing | E2E Test Suite Track | Implement `tests/test_verification_suite.py` and `run_verification.py` covering Tiers 1–4 test cases using mock video and mock serial; publish `TEST_READY.md` | Survey | DONE |
| M1 | Implementation | Core Detection Engine & Hardware Layer | Refactor `eco_engine.py`: implement `detect_best_device`, `SpatialROIFilter`, `TemporalPresenceSmoother`, `DutyCycleManager`, `EngineConfig`, `MockHardwareController`, `MockVideoSource`, and decoupled `AdvancedDetectionEngine` | Survey | DONE |
| M2 | Implementation | Modernized UI Dashboard & Interactive Controls | Refactor `Model Demo/main.py`: implement modern CustomTkinter dark-mode UI, telemetry HUD, interactive sliders, ROI overlay, manual overrides, and hardware hub connected to `EngineConfig` | M1 | READY |
| M3 | Implementation | Integration, 100% E2E Test Pass & Hardening | Full integration verification: run E2E test suite until 100% pass (Phase 1), perform Tier 5 adversarial hardening (Phase 2), and complete forensic integrity audit | M1, M2, E2E | PLANNED |

## Feature Inventory Cross-Check
- Total Features: 14
- Assigned Features:
  - M1: F4, F8, F9, F10, F11, F12 (6 features)
  - M2: F1, F2, F3, F5, F6, F7 (6 features)
  - E2E Track: F13 (1 feature)
  - M3: F14 (1 feature)
- Unassigned Features: 0 (Feature Inventory cross-check PASSED)

## Code Layout
```
d:\Study\ECEPROJO\
├── eco_engine.py               # Core detection engine, spatial ROI, temporal smoothing, duty cycle, hardware & mock drivers (Owned by M1)
├── Model Demo/
│   ├── main.py                 # Modernized CustomTkinter showcase dashboard and interactive control panels (Owned by M2)
│   └── classroom_lights.ino    # Arduino sketch (Reference/Unchanged)
├── tests/
│   ├── __init__.py             # Test package initialization (Owned by E2E Track)
│   └── test_verification_suite.py # Automated unittest suite covering Tiers 1-4 (Owned by E2E Track)
├── run_verification.py         # Top-level standalone verification runner (Owned by E2E Track)
├── TEST_INFRA.md               # Test infrastructure specification (Owned by E2E Track)
├── TEST_READY.md               # E2E test readiness publication (Owned by E2E Track)
├── PROJECT.md                  # Project architecture and milestones (Owned by Orchestrator)
└── requirements.txt            # Project dependencies
```

## Interface Contracts

### 1. `EngineConfig` ↔ Detection Engine & UI
```python
@dataclass
class EngineConfig:
    conf_threshold: float = 0.50          # Range: 0.01 to 1.00
    iou_threshold: float = 0.45           # Range: 0.10 to 0.90
    eval_duration: float = 2.0            # Seconds; range 0.5 to 10.0
    sleep_duration: float = 5.0           # Seconds; range 0.0 to 30.0 (0.0 = continuous)
    duty_cycle_enabled: bool = True       # Enable/disable duty cycling
    roi_enabled: bool = False             # Enable/disable ROI filtering
    roi_polygon: list = field(default_factory=list) # List of normalized points [(u1, v1), ...] in [0.0, 1.0]
    temporal_smoothing_enabled: bool = True
    hysteresis_on_frames: int = 3         # Frames to confirm ON
    hysteresis_off_frames: int = 15       # Frames to confirm OFF
    min_switch_interval: float = 1.5      # Minimum seconds between serial toggles
    override_mode: str = "AUTO"           # "AUTO", "FORCE_ON", "FORCE_OFF"
```

### 2. `AdvancedDetectionEngine` ↔ UI Dashboard
```python
class AdvancedDetectionEngine:
    def __init__(self, source=0, port=None, config: Optional[EngineConfig] = None): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def update_config(self, **kwargs) -> None: ...
    def get_latest_frame_and_telemetry(self) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        # Returns: (annotated_bgr_frame, {
        #   "fps": float,
        #   "latency_ms": float,
        #   "raw_occupants": int,
        #   "roi_occupants": int,
        #   "smoothed_presence": bool,
        #   "cycle_state": "EVALUATING" | "SLEEPING",
        #   "cycle_countdown": float,
        #   "relay_state": bool,
        #   "device": str ("cuda:0" | "cpu"),
        #   "override_mode": str ("AUTO" | "FORCE_ON" | "FORCE_OFF"),
        #   "hardware_connected": bool,
        #   "camera_connected": bool
        # })
    def switch_source(self, new_source) -> bool: ...
    def connect_serial(self, port: str) -> bool: ...
    def disconnect_serial(self) -> None: ...
    def set_override(self, mode: str) -> None: ... # mode in ("AUTO", "FORCE_ON", "FORCE_OFF")
```

### 3. `HardwareController` & `MockHardwareController` Contract
```python
class BaseHardwareController:
    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def set_light(self, state: bool) -> bool: ... # True sends '1', False sends '0'
    @property
    def is_connected(self) -> bool: ...
    @property
    def current_state(self) -> bool: ...

class MockHardwareController(BaseHardwareController):
    def get_command_log(self) -> List[Tuple[float, str]]: ... # [(timestamp, '1'), ...]
    def reset_log(self) -> None: ...
```

### 4. `VideoStream` & `MockVideoSource` Contract
```python
class BaseVideoStream:
    def start(self) -> None: ...
    def read(self) -> Tuple[bool, Optional[np.ndarray]]: ...
    def stop(self) -> None: ...
    @property
    def is_opened(self) -> bool: ...

class MockVideoSource(BaseVideoStream):
    def set_scenario(self, scenario: str) -> None: # "PERSON_IN_ROI", "PERSON_OUTSIDE_ROI", "EMPTY_FRAME"
    def set_custom_frame(self, frame: np.ndarray) -> None: ...
```

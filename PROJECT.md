# Project: AIEcoLightswitcher — Multi-Feed Academic & Showcase Edition

## Architecture
AIEcoLightswitcher is a high-performance computer vision smart energy system that optimizes electrical lighting based on real-time human presence detection across multiple concurrent camera feeds. The architecture employs a 4-tier decoupled multi-threaded pipeline:

1. **Ingestion & Stream Layer (`ingestion/` & `core/`)**:
   - `BaseVideoStream`: Abstract stream contract with `read()`, `start()`, `stop()`, and `is_opened`.
   - `StreamWorker`: Dedicated per-feed ingestion thread with non-blocking single-slot frame buffer (`SingleFrameBuffer`, drop-oldest semantics) ensuring $O(1)$ zero-latency frame access.
   - `VideoStream`: Universal OpenCV video ingestion supporting local USB webcams (DirectShow `cv2.CAP_DSHOW`), DroidCam over Wi-Fi HTTP, and RTSP IP streams with non-blocking network socket timeouts.
   - `MockVideoSource`: Deterministic synthetic video generator supporting multi-feed scenarios (person in ROI, person outside ROI, empty frame, offline/stalled feed).
   - `AutoReconnect`: 5-state lifecycle (`CONNECTING`, `ONLINE`, `OFFLINE`, `RECONNECTING`, `DISABLED`) with exponential backoff reconnect loops without crashing video processing.

2. **Detection & Pipeline Layer (`pipeline/` & `core/`)**:
   - `MultiFeedManager`: Central coordinator managing active feed registry, dynamic lifecycle (`add_feed`, `remove_feed`, `enable_feed`, `disable_feed`), and centralized YOLOv8 inference dispatch.
   - `FeedPipeline`: Per-feed processing encapsulation combining `StreamWorker`, `SpatialROIFilter` (normalized polygon containment via `cv2.pointPolygonTest`), `TemporalPresenceSmoother` (asymmetric hysteresis state machine $M=3$ ON, $N=15$ OFF), `DutyCycleManager` (evaluating vs sleeping), and assigned Arduino pin mapping.
   - Centralized Inference Dispatcher: Single YOLOv8 model instance evaluated across active feeds sequentially, preventing GPU context collisions and VRAM exhaustion.

3. **Hardware & Serial Communication Layer (`hardware/` & Arduino Firmware)**:
   - `MultiPinHardwareController`: Physical USB serial communication with Arduino Uno/Nano driving independent digital GPIO pins (Pins 2–19).
   - Framed ASCII Protocol: `SET <pin> <state>\n`, `GET <pin>\n`, `STATUS\n`, `PING\n`, `IDENT\n`, with 100% backwards-compatibility for legacy single-pin `'1'`/`'0'` commands.
   - Delta-Update Optimization: Only transmits serial commands on state changes, reducing steady-state UART traffic by >99%.
   - Per-Pin Rate Limiting: Individual `min_switch_interval` clamps per pin, preventing relay chattering without cross-feed blocking.
   - Hardware Resilience: Automatic fallback to virtual simulation if serial port is disconnected/unavailable; background auto-reconnection and full pin state resynchronization (`force_sync`) upon reconnection.
   - Updated Arduino Firmware (`Model Demo/classroom_lights/classroom_lights.ino`): Zero dynamic heap allocations (0 bytes `String`/`malloc`), static 32-byte circular/linear buffer, fast C-string parsing, and watchdog failsafe.

4. **Showcase Multi-Feed User Interface (`gui/` & `Model Demo/`)**:
   - Responsive Multi-Feed Dashboard adhering to modern web guidance (`css-layout`, `interactions-in-complex-layouts`, `responsive-design`).
   - Persistent Minimum 4-Card Grid Layout: Displays active feeds alongside informative offline placeholder cards (showing reconnection countdown and attempt counters), dynamically reflowing (2x2, 2x3, 3x3) when $N > 4$ feeds are active.
   - Modern Web Best Practices: CSS Grid with `repeat(auto-fit, minmax(280px, 1fr))`, intrinsic `aspect-ratio: 16 / 9` on video containers, CSS containment (`content-visibility: auto; contain-intrinsic-size: auto 320px auto 240px; contain: layout style paint;`) for zero-stutter isolated card rendering, and container-relative HUD badge scaling via `@container` queries.
   - Interactive Per-Feed & Global Controls: Real-time enable/disable toggles, per-feed manual overrides (`AUTO`, `FORCE_ON`, `FORCE_OFF`), pin assignment badges, global Add Feed modal, and serial port connection hub.

5. **Automated Verification & E2E Testing Track (`tests/` & `run_verification.py`)**:
   - Headless automated test suite running on CPU without physical cameras, GPUs, or microcontrollers.
   - Backwards compatibility: Existing 35/35 tests in `run_verification.py` pass without regression via `eco_engine.py` backwards-compatibility facade.
   - Multi-Feed Test Suite: Tiers 1–4 requirement-driven tests + Tier 5 adversarial tests validating parallel concurrency, dynamic feed management, stream disconnect recovery, and multi-pin serial dispatch.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Universal Video Ingestion | Support local USB webcams (DirectShow), DroidCam Wi-Fi HTTP, RTSP IP cameras, and synthetic mock streams | M1 | R1 |
| 2 | Decoupled Stream Workers | Dedicated per-feed thread with non-blocking single-slot frame buffer (`drop-oldest`), zero cross-feed latency coupling | M1 | R1, R4 |
| 3 | Immediate Offline Detection & Recovery | Transition failed/disconnected streams to `OFFLINE` state immediately; exponential backoff background auto-reconnection | M1 | R1, AC |
| 4 | Dynamic Feed Lifecycle Management | Runtime `add_feed`, `remove_feed`, `enable_feed`, `disable_feed` with safe thread joining and resource release | M3 | R1, AC |
| 5 | Multi-Pin Serial Protocol | Framed ASCII protocol (`SET <pin> <state>\n`) supporting pins 2–19, query commands, and legacy `'1'`/`'0'` fallback | M2 | R2 |
| 6 | Delta-Update & Per-Pin Rate Limiting | Transmit serial commands only on state transitions; independent per-pin `min_switch_interval` anti-chatter clamp | M2 | R2 |
| 7 | Hardware Auto-Fallback & Resilience | Graceful switch to virtual simulation mode if Arduino is unavailable/disconnected; background auto-reconnect & resync | M2 | R2, AC |
| 8 | Updated Multi-Pin Arduino Firmware | ATmega328P `.ino` firmware with static buffer parsing, 0 dynamic heap allocations, pin clamping, and failsafe reset | M2 | R2, AC |
| 9 | Persistent Minimum 4-Card Grid | Minimum 4-card grid layout with active stream tiles and structured offline placeholder cards, reflowing when $N > 4$ | M4 | R3, AC |
| 10 | Modern Web Layout & CSS Containment | CSS Grid `repeat(auto-fit, minmax(280px, 1fr))`, `aspect-ratio: 16/9`, `content-visibility: auto` rendering isolation | M4 | R3, Directive |
| 11 | Container-Relative HUD Badges | Dynamic HUD telemetry badges (FPS, pin status, confidence, override mode) scaled relative to card dimensions via container queries | M4 | R3, Directive |
| 12 | Informative Offline Placeholder Cards | Informative card rendering stream status, reconnection retry timer, attempt counter, and per-feed toggle controls | M4 | R3, AC |
| 13 | Interactive Per-Feed & Global Controls | Runtime per-feed toggles, manual overrides (`AUTO`/`FORCE_ON`/`FORCE_OFF`), pin configuration, and global Add Feed modal | M4 | R3, AC |
| 14 | Shared YOLO Inference Dispatcher | Single model instance evaluated across active feeds with per-feed ROI polygon filtering and temporal presence smoothing | M3 | R1, R3 |
| 15 | Modular Architecture & Facade | Clean separation into `core/`, `ingestion/`, `hardware/`, `pipeline/`, `gui/` with `eco_engine.py` facade for 100% test compatibility | M1 | R4, AC |
| 16 | Multi-Feed E2E Test Suite | Automated headless tests (Tiers 1–4) validating parallel ingestion, dynamic lifecycle, offline recovery, and multi-pin serial dispatch | E2E Track | Verification |
| 17 | Adversarial Hardening (Tier 5) | White-box stress testing of thread race conditions, buffer overruns, disconnect bursts, and rapid toggle floods | M5 | Verification |

---

## Milestones
| # | Track | Name | Scope | Dependencies | Status |
|---|-------|------|-------|-------------|--------|
| E2E | Testing | Multi-Feed E2E Test Suite Track | Build `tests/test_multifeed_suite.py` implementing Tiers 1–4 tests for parallel execution, dynamic feed addition/removal, stream failure recovery, and multi-pin serial dispatch; update `TEST_READY.md` | Survey | READY |
| M1 | Implementation | Modular Core & Parallel Ingestion Engine | Implement `core/` (`config.py`, `device.py`, `roi_filter.py`, `smoother.py`, `duty_cycle.py`) and `ingestion/` (`base_stream.py`, `stream_worker.py`, `video_stream.py`, `mock_stream.py`); update `eco_engine.py` facade | Survey | READY |
| M2 | Implementation | Multi-Pin Hardware Routing & Arduino Firmware | Implement `hardware/` (`base_controller.py`, `serial_controller.py`, `mock_controller.py`) and updated `Model Demo/classroom_lights/classroom_lights.ino` with multi-pin ASCII protocol and simulation fallback | M1 | PLANNED |
| M3 | Implementation | Multi-Feed Pipeline & Dynamic Management | Implement `pipeline/` (`feed_pipeline.py`, `multi_feed_manager.py`) with dynamic add/remove/enable/disable, shared YOLO inference dispatch, and per-feed pin routing | M1, M2 | PLANNED |
| M4 | Implementation | Responsive Multi-Feed UI Dashboard | Implement `gui/` (`bento_theme.py`, `web_dashboard.py` / `multi_feed_grid.py`) complying with user directive (CSS Grid, intrinsic 16:9, CSS containment, container queries, min 4 cards, offline placeholders, controls) | M3 | PLANNED |
| M5 | Implementation | Full Integration, 100% E2E Pass & Hardening | Full system integration: run complete E2E test suite until 100% pass (Phase 1), perform Tier 5 adversarial stress testing (Phase 2), and complete forensic integrity audit | M1–M4, E2E | PLANNED |

---

## Feature Inventory Cross-Check
- Total Features: 17
- Assigned Features:
  - M1: F1, F2, F3, F15 (4 features)
  - M2: F5, F6, F7, F8 (4 features)
  - M3: F4, F14 (2 features)
  - M4: F9, F10, F11, F12, F13 (5 features)
  - E2E Track: F16 (1 feature)
  - M5: F17 (1 feature)
- Unassigned Features: 0 (Feature Inventory cross-check PASSED)

---

## Code Layout
```
d:\Study\ECEPROJO\
├── core/
│   ├── __init__.py
│   ├── config.py                 # EngineConfig, FeedConfig dataclasses
│   ├── device.py                 # detect_best_device, warmup_engine
│   ├── roi_filter.py             # SpatialROIFilter with polygon containment
│   ├── smoother.py               # TemporalPresenceSmoother (hysteresis state machine)
│   └── duty_cycle.py             # DutyCycleManager (evaluating vs sleeping)
│
├── ingestion/
│   ├── __init__.py
│   ├── base_stream.py            # BaseVideoStream, FeedStatus enum, StreamMetadata
│   ├── stream_worker.py          # StreamWorker (dedicated thread, atomic buffer, auto-reconnect)
│   ├── video_stream.py           # VideoStream (DirectShow, RTSP, DroidCam HTTP)
│   └── mock_stream.py            # MockVideoSource (multi-feed scenario synthetic generator)
│
├── hardware/
│   ├── __init__.py
│   ├── base_controller.py        # BaseHardwareController interface contract
│   ├── serial_controller.py      # MultiPinHardwareController (framed ASCII protocol, delta-updates)
│   └── mock_controller.py        # MockMultiPinHardwareController (multi-pin command logging)
│
├── pipeline/
│   ├── __init__.py
│   ├── feed_pipeline.py          # FeedPipeline (per-feed stream, ROI, smoother, pin mapping)
│   └── multi_feed_manager.py     # MultiFeedManager (dynamic lifecycle, shared YOLO dispatch)
│
├── gui/
│   ├── __init__.py
│   ├── bento_theme.py            # Dark telemetry palette, color tokens
│   ├── web_dashboard.py          # Modern web dashboard server with CSS Grid & WebSockets/MJPEG
│   ├── static/                   # HTML5, CSS Grid (auto-fit, aspect-ratio: 16/9), containment
│   └── desktop_grid.py           # Desktop CustomTkinter responsive grid fallback
│
├── Model Demo/
│   ├── main.py                   # Multi-feed showcase launcher
│   └── classroom_lights/
│       └── classroom_lights.ino  # Updated multi-pin Arduino firmware (Pins 2-19, ASCII protocol)
│
├── eco_engine.py                 # Backwards-compatibility facade re-exporting all components
├── run_verification.py           # Unified automated test runner (existing 35 tests + multi-feed suite)
├── tests/
│   ├── __init__.py
│   ├── test_verification_suite.py # Existing 35 single-feed tests (Tiers 1-4 baseline)
│   └── test_multifeed_suite.py   # Multi-feed parallel execution, dynamic lifecycle & pin dispatch tests
├── TEST_INFRA.md                 # Test infrastructure specification
├── TEST_READY.md                 # E2E test suite readiness attestation
└── requirements.txt              # Project dependencies
```

---

## Interface Contracts

### 1. `FeedStatus` & `FeedConfig` Contract (`core/config.py` & `ingestion/base_stream.py`)
```python
class FeedStatus(enum.Enum):
    UNASSIGNED = "UNASSIGNED"
    CONNECTING = "CONNECTING"
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    RECONNECTING = "RECONNECTING"
    DISABLED = "DISABLED"

@dataclass
class FeedConfig:
    feed_id: str                          # Unique identifier e.g. "feed_1"
    name: str                             # Human-readable name e.g. "Front Entrance"
    source: Any                           # Camera index (0, 1), URL (HTTP/RTSP), or "MOCK:..."
    pin: int = 13                         # Arduino digital output pin (2–19)
    enabled: bool = True                  # Feed active processing flag
    conf_threshold: float = 0.50          # Detection confidence threshold
    eval_duration: float = 2.0            # Duty cycle eval duration in seconds
    sleep_duration: float = 5.0           # Duty cycle sleep duration in seconds
    roi_enabled: bool = False             # Spatial ROI enable flag
    roi_polygon: list = field(default_factory=list) # Normalized [(u, v), ...] coordinates
    temporal_smoothing_enabled: bool = True
    override_mode: str = "AUTO"           # "AUTO", "FORCE_ON", "FORCE_OFF"
```

### 2. `StreamWorker` Contract (`ingestion/stream_worker.py`)
```python
class StreamWorker:
    def __init__(self, source: Any, name: str = "", reconnect_interval: float = 2.0): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        # Returns (frame, {
        #   "status": FeedStatus,
        #   "fps": float,
        #   "latency_ms": float,
        #   "frame_id": int,
        #   "reconnect_attempts": int,
        #   "reconnect_countdown": float
        # })
    @property
    def status(self) -> FeedStatus: ...
    @property
    def is_alive(self) -> bool: ...
```

### 3. `MultiPinHardwareController` Contract (`hardware/base_controller.py`)
```python
class BaseMultiPinHardwareController(abc.ABC):
    @abc.abstractmethod
    def connect(self) -> bool: ...
    @abc.abstractmethod
    def disconnect(self) -> None: ...
    @abc.abstractmethod
    def set_pin(self, pin: int, state: bool) -> bool: ... # SET <pin> <state>\n
    @abc.abstractmethod
    def get_pin(self, pin: int) -> bool: ...
    @abc.abstractmethod
    def set_all(self, state: bool) -> bool: ...
    @abc.abstractmethod
    def force_sync(self) -> bool: ...
    @property
    @abc.abstractmethod
    def is_connected(self) -> bool: ...
    @property
    @abc.abstractmethod
    def is_simulation(self) -> bool: ...

    # Backwards compatibility with legacy single-pin interface
    def set_light(self, state: bool) -> bool:
        return self.set_pin(13, state)
```

### 4. `MultiFeedManager` Contract (`pipeline/multi_feed_manager.py`)
```python
class MultiFeedManager:
    def __init__(self, hardware: Optional[BaseMultiPinHardwareController] = None): ...
    def add_feed(self, config: FeedConfig) -> bool: ...
    def remove_feed(self, feed_id: str) -> bool: ...
    def enable_feed(self, feed_id: str) -> bool: ...
    def disable_feed(self, feed_id: str) -> bool: ...
    def update_feed_config(self, feed_id: str, **kwargs) -> bool: ...
    def get_feed_telemetry(self, feed_id: str) -> Dict[str, Any]: ...
    def get_all_feeds_telemetry(self) -> Dict[str, Dict[str, Any]]: ...
    def get_feed_frame(self, feed_id: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
```

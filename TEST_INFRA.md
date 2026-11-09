# AIEcoLightswitcher — Testing Infrastructure & Verification Specification

**Specification Document**: `SPEC-TEST-INFRA-001`  
**Target Systems**: AIEcoLightswitcher Core Engine, Spatial ROI, Temporal Smoothing, Duty Cycle, Hardware Abstraction  
**Verification Framework**: Python Standard Library `unittest` (Zero external test runner dependencies)  
**Execution Environment**: 100% Headless CPU Execution (No physical camera, GPU, or Arduino required)  

---

## 1. Test Philosophy & Design Principles

The verification infrastructure for AIEcoLightswitcher is engineered to ensure total software reliability, deterministic evaluation, and repeatable validation under academic and industry examination:

1. **Opaque-Box & Interface-Driven Verification**:
   Tests evaluate behavioral specifications against the defined public interface contracts (`EngineConfig`, `detect_best_device`, `SpatialROIFilter`, `TemporalPresenceSmoother`, `DutyCycleManager`, `BaseHardwareController`, `MockHardwareController`, `MockVideoSource`, `AdvancedDetectionEngine`) rather than private implementation details.
2. **Zero External Hardware Dependency**:
   The entire test harness executes headlessly on standard CPU architectures without requiring a connected webcam, DroidCam phone, NVIDIA CUDA GPU, or physical USB-serial Arduino microcontroller.
3. **Deterministic Simulation & Ground Truth**:
   Instead of stochastic or uncontrolled video streams, tests utilize `MockVideoSource` (yielding deterministic blank frames or verified Ultralytics COCO person assets) and `MockHardwareController` (capturing timestamped `'1'` and `'0'` serial commands into an inspectable in-memory event bus).
4. **Strict Safety & Hysteresis Guards**:
   Tests verify electrical contact preservation by asserting that single-frame false positives or confidence jitter cannot cause rapid relay toggling (relay chattering), and that minimum switching intervals ($\Delta t_{\min} = 1.5$s) are enforced.
5. **Fast Execution & CI/CD Readiness**:
   The entire test suite completes execution in under 5 seconds, making it ideal for automated pre-commit validation and continuous integration pipelines.

---

## 2. Test Architecture & Runner Command

### 2.1 Invocation Commands
The primary top-level verification runner is:
```bash
python run_verification.py
```
Alternatively, Python's built-in `unittest` runner can discover and execute the suite directly:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 2.2 Directory Layout
```
d:\Study\ECEPROJO\
├── run_verification.py              # Standalone CLI runner with formatted colored summary
├── TEST_INFRA.md                    # Verification architecture & matrix specification
├── TEST_READY.md                    # Readiness publication & feature checklist
└── tests/
    ├── __init__.py                  # Package initialization
    └── test_verification_suite.py   # Comprehensive 4-tier automated test suite
```

### 2.3 Test Suite Structure
The test suite is structured into four distinct classes within `tests/test_verification_suite.py`:
- `TestTier1FeatureCoverage`: Tests core feature capabilities in isolation (TC_T1_01 through TC_T1_10).
- `TestTier2BoundaryCornerCases`: Stresses edge conditions, extremes, and error handling (TC_T2_01 through TC_T2_05).
- `TestTier3CrossFeatureCombinations`: Validates inter-module data pipelines and state interactions (TC_T3_01 through TC_T3_04).
- `TestTier4RealWorldScenarios`: End-to-end simulation of physical deployment environments (TC_T4_01 through TC_T4_05).

---

## 3. Test Matrix & Coverage Tiers

| Tier | Test ID | Target Component | Description | Expected Verification |
|---|---|---|---|---|
| **Tier 1** | `TC_T1_01` | `EngineConfig` | Engine initialization with default and custom configurations | Default parameters match contract; custom overrides persist |
| **Tier 1** | `TC_T1_02` | `detect_best_device` | CUDA/CPU device resolution without crashing | Safely returns `(device, display_name)`; graceful CPU fallback |
| **Tier 1** | `TC_T1_03` | `MockHardwareController` | In-memory serial command recording & state tracking | Records '1' and '0'; tracks `current_state` and command log |
| **Tier 1** | `TC_T1_04` | `MockVideoSource` | Synthetic video stream & scenario switching | Generates valid BGR frames; handles scenario changes |
| **Tier 1** | `TC_T1_05` | `SpatialROIFilter` | Normalized polygon containment test | Correctly identifies points inside vs outside polygon |
| **Tier 1** | `TC_T1_06` | `TemporalPresenceSmoother` | Turn-ON hysteresis state transition | Requires $M$ consecutive positive frames before switching ON |
| **Tier 1** | `TC_T1_07` | `TemporalPresenceSmoother` | Turn-OFF hold-off state transition | Requires $N$ consecutive negative frames before switching OFF |
| **Tier 1** | `TC_T1_08` | `DutyCycleManager` | `EVALUATING` $\leftrightarrow$ `SLEEPING` state transitions | Transitions at configured intervals; calculates countdown |
| **Tier 1** | `TC_T1_09` | `DutyCycleManager` | Continuous evaluation mode | When sleep is 0.0 or disabled, stays continuously in `EVALUATING` |
| **Tier 1** | `TC_T1_10` | `EngineConfig` | Dynamic runtime parameter updating | Parameter adjustments take effect immediately in active components |
| **Tier 2** | `TC_T2_01` | `SpatialROIFilter` | Empty or degenerate polygon boundary handling | Safely falls back to full-frame or 0 occupants without crashing |
| **Tier 2** | `TC_T2_02` | Inference / Engine | Extreme confidence thresholds ($0.01$ and $1.00$) | Rejects invalid ranges or filters correctly without numerical error |
| **Tier 2** | `TC_T2_03` | `TemporalPresenceSmoother` | Rapid single-frame detection oscillation | Alternating 1/0 detections do not chatter relay; 0 false switches |
| **Tier 2** | `TC_T2_04` | `DutyCycleManager` | Extreme duty cycle intervals ($10$s/$30$s and $0.5$s/$0$s) | Correct countdown calculation without timing freeze or div-by-zero |
| **Tier 2** | `TC_T2_05` | Hardware / Stream | Non-existent COM port or camera failure | Graceful error handling, no unhandled exceptions or crashes |
| **Tier 3** | `TC_T3_01` | ROI + Smoother | Spatial ROI filtered output fed into Temporal Smoother | Persons outside ROI never activate smoother; inside ROI activates after $M$ |
| **Tier 3** | `TC_T3_02` | Duty Cycle + Config | Dynamic parameter tuning across duty cycle states | Changing confidence during sleep takes effect on next evaluation |
| **Tier 3** | `TC_T3_03` | Overrides + Duty Cycle | Manual overrides (`FORCE_ON`, `FORCE_OFF`, `AUTO`) | Overrides bypass duty cycle sensor logic; restore correctly on `AUTO` |
| **Tier 3** | `TC_T3_04` | Spatial ROI + Movement | Person moving from outside ROI into ROI | Outside: lights stay OFF; Enters ROI: lights turn ON after $M$ frames |
| **Tier 4** | `TC_T4_01` | End-to-End System | Classroom Lecture Scenario | Students arrive $\rightarrow$ lights ON; class ends $\rightarrow$ lights OFF after hold-off |
| **Tier 4** | `TC_T4_02` | End-to-End System | Corridor Transient Passerby Scenario | Passerby outside room ROI does not trigger room lights |
| **Tier 4** | `TC_T4_03` | End-to-End System | Study Session Micro-Movements Scenario | Momentary 1-2 frame occlusions do not flicker lighting relay |
| **Tier 4** | `TC_T4_04` | End-to-End System | Energy Saver Duty Cycle Scenario | Lights ON, duty cycle sleeps inference to save compute power, resumes cleanly |
| **Tier 4** | `TC_T4_05` | End-to-End System | Emergency Manual Override Scenario | Presentation requires room darkness: `FORCE_OFF` overrides detected occupants |

---

## 4. Coverage Thresholds & Quality Metrics

1. **Feature Coverage**: 100% of the functional requirements outlined in R3, R4, and the Interface Contracts are covered.
2. **Deterministic Output**: All test cases use deterministic synthetic data (numpy image buffers, simulated timers, and mock serial logs) guaranteeing 0 flaky or order-dependent test outcomes.
3. **Execution Exit Code**:
   - Exit code `0`: All tests passed.
   - Exit code `1`: One or more tests failed or encountered errors.

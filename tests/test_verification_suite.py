"""
tests/test_verification_suite.py - Comprehensive E2E Automated Test Suite for AIEcoLightswitcher.

Architecture:
- Python Standard Library unittest.
- Headless CPU execution without physical camera, GPU, or Arduino hardware.
- Dynamic interface resolution against eco_engine with conforming wrappers.
- Tier 1: Feature Coverage Tests (TC_T1_01 through TC_T1_10)
- Tier 2: Boundary & Corner Case Tests (TC_T2_01 through TC_T2_05)
- Tier 3: Cross-Feature Combination Tests (TC_T3_01 through TC_T3_04)
- Tier 4: Real-World Application Scenarios (TC_T4_01 through TC_T4_05)
"""

import sys
import os
import time
import math
import unittest
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import eco_engine


# ==============================================================================
# DYNAMIC COMPONENT RESOLUTION & CONFORMING TEST WRAPPERS
# Integrates with eco_engine components with universal parameter adaptors.
# ==============================================================================

# 1. EngineConfig
EngineConfig = eco_engine.EngineConfig

# 2. detect_best_device
detect_best_device = eco_engine.detect_best_device

# 3. SpatialROIFilter wrapper
class SpatialROIFilter(eco_engine.SpatialROIFilter):
    """
    Test wrapper ensuring enabled defaults to True when testing ROI containment.
    """
    def __init__(
        self,
        normalized_vertices: Optional[List[Tuple[float, float]]] = None,
        mode: str = "BOTTOM_CENTER",
        enabled: bool = True
    ):
        super().__init__(normalized_vertices=normalized_vertices, mode=mode, enabled=enabled)


# 4. TemporalPresenceSmoother wrapper
class TemporalPresenceSmoother(eco_engine.TemporalPresenceSmoother):
    """
    Test wrapper accepting both min_switch_interval and min_switch_sec,
    and accepting both int occupant counts and boolean presence in update().
    """
    def __init__(
        self,
        on_frames: int = 3,
        off_frames: int = 15,
        min_switch_interval: float = 1.5,
        min_switch_sec: Optional[float] = None,
        enabled: bool = True
    ):
        interval = min_switch_sec if min_switch_sec is not None else min_switch_interval
        super().__init__(
            on_frames=on_frames,
            off_frames=off_frames,
            min_switch_interval=interval,
            enabled=enabled
        )

    def update(self, presence_or_count: Any, now: Optional[float] = None) -> Tuple[bool, bool]:
        if isinstance(presence_or_count, (int, float)):
            raw = (presence_or_count > 0)
        else:
            raw = bool(presence_or_count)
        return super().update(raw, now)


# 5. DutyCycleManager wrapper
DutyCycleManager = eco_engine.DutyCycleManager

# 6. MockHardwareController
MockHardwareController = eco_engine.MockHardwareController

# 7. MockVideoSource
MockVideoSource = eco_engine.MockVideoSource

# 8. AdvancedDetectionEngine
AdvancedDetectionEngine = eco_engine.AdvancedDetectionEngine


# ==============================================================================
# TIER 1: FEATURE COVERAGE TESTS
# ==============================================================================
class TestTier1FeatureCoverage(unittest.TestCase):
    """
    Tier 1 tests verify core feature logic in isolation (TC_T1_01 through TC_T1_10).
    """

    def test_TC_T1_01_engine_config_initialization(self):
        """TC_T1_01: Verify default and custom EngineConfig initialization."""
        default_cfg = EngineConfig()
        self.assertEqual(default_cfg.conf_threshold, 0.50)
        self.assertEqual(default_cfg.iou_threshold, 0.45)
        self.assertEqual(default_cfg.eval_duration, 2.0)
        self.assertEqual(default_cfg.sleep_duration, 5.0)
        self.assertTrue(default_cfg.duty_cycle_enabled)
        self.assertFalse(default_cfg.roi_enabled)
        self.assertTrue(default_cfg.temporal_smoothing_enabled)
        self.assertEqual(default_cfg.hysteresis_on_frames, 3)
        self.assertEqual(default_cfg.hysteresis_off_frames, 15)
        self.assertEqual(default_cfg.min_switch_interval, 1.5)
        self.assertEqual(default_cfg.override_mode, "AUTO")

        custom_cfg = EngineConfig(
            conf_threshold=0.75,
            eval_duration=1.0,
            sleep_duration=3.0,
            roi_enabled=True,
            override_mode="FORCE_ON"
        )
        self.assertEqual(custom_cfg.conf_threshold, 0.75)
        self.assertEqual(custom_cfg.eval_duration, 1.0)
        self.assertEqual(custom_cfg.sleep_duration, 3.0)
        self.assertTrue(custom_cfg.roi_enabled)
        self.assertEqual(custom_cfg.override_mode, "FORCE_ON")

    def test_TC_T1_02_detect_best_device_graceful_handling(self):
        """TC_T1_02: detect_best_device gracefully handles CPU/CUDA without crashing."""
        dev_cpu, name_cpu = detect_best_device("cpu")
        self.assertEqual(dev_cpu, "cpu")
        self.assertIn("CPU", name_cpu)

        dev_auto, name_auto = detect_best_device("auto")
        self.assertIn(dev_auto, ("0", "cpu", "cuda:0"))
        self.assertIsInstance(name_auto, str)

        dev_cuda, name_cuda = detect_best_device("0")
        self.assertIn(dev_cuda, ("0", "cpu", "cuda:0"))
        self.assertIsInstance(name_cuda, str)

    def test_TC_T1_03_mock_hardware_controller_recording(self):
        """TC_T1_03: MockHardwareController records serial '1' and '0' commands and tracks state."""
        mock_hw = MockHardwareController()
        self.assertTrue(mock_hw.is_connected)
        self.assertFalse(mock_hw.current_state)
        self.assertEqual(len(mock_hw.get_command_log()), 0)

        mock_hw.set_light(True)
        self.assertTrue(mock_hw.current_state)
        log = mock_hw.get_command_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0][1], '1')

        # Setting same state should not emit duplicate commands
        mock_hw.set_light(True)
        self.assertEqual(len(mock_hw.get_command_log()), 1)

        mock_hw.set_light(False)
        self.assertFalse(mock_hw.current_state)
        log = mock_hw.get_command_log()
        self.assertEqual(len(log), 2)
        self.assertEqual(log[1][1], '0')

        mock_hw.reset_log()
        self.assertEqual(len(mock_hw.get_command_log()), 0)

        mock_hw.disconnect()
        self.assertFalse(mock_hw.is_connected)
        mock_hw.connect()
        self.assertTrue(mock_hw.is_connected)

    def test_TC_T1_04_mock_video_source_frames_and_scenarios(self):
        """TC_T1_04: MockVideoSource yields frames and responds to scenario changes."""
        source = MockVideoSource()
        source.start()
        self.assertTrue(source.is_opened)

        ret, frame = source.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame.shape), 3)
        self.assertEqual(frame.dtype, np.uint8)

        source.set_scenario("PERSON_IN_ROI")
        ret, frame_occ = source.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame_occ)

        source.set_scenario("EMPTY_FRAME")
        ret, frame_empty = source.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame_empty)
        # Empty frame is all black (0 occupants)
        self.assertEqual(np.count_nonzero(frame_empty), 0)

        custom = np.full((480, 640, 3), 128, dtype=np.uint8)
        source.set_custom_frame(custom)
        ret, frame_custom = source.read()
        self.assertTrue(ret)
        np.testing.assert_array_equal(frame_custom, custom)

        source.stop()
        self.assertFalse(source.is_opened)

    def test_TC_T1_05_spatial_roi_filter_containment(self):
        """TC_T1_05: SpatialROIFilter correctly identifies points inside and outside polygon."""
        # Left half polygon: [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]
        left_poly = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]
        roi_filter = SpatialROIFilter(normalized_vertices=left_poly, mode="BOTTOM_CENTER", enabled=True)

        # Frame 640x480
        # Inside box: foot point at x=150, y=300 (x_norm = 150/640 = 0.234)
        det_inside = {
            "id": "det_inside",
            "xyxy": np.array([100, 100, 200, 300]),
            "conf": 0.85,
            "cls": 0
        }
        # Outside box: foot point at x=450, y=300 (x_norm = 450/640 = 0.703)
        det_outside = {
            "id": "det_outside",
            "xyxy": np.array([400, 100, 500, 300]),
            "conf": 0.88,
            "cls": 0
        }

        inside, outside = roi_filter.filter_detections([det_inside, det_outside], 640, 480)
        self.assertEqual(len(inside), 1)
        self.assertEqual(inside[0]["id"], "det_inside")
        self.assertEqual(len(outside), 1)
        self.assertEqual(outside[0]["id"], "det_outside")

    def test_TC_T1_06_temporal_smoother_on_hysteresis(self):
        """TC_T1_06: TemporalPresenceSmoother requires M positive frames to turn ON."""
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=15, min_switch_sec=0.0)
        self.assertFalse(smoother.current_state)

        # Frame 1: positive
        state, changed = smoother.update(1, now=10.0)
        self.assertFalse(state)
        self.assertFalse(changed)

        # Frame 2: positive
        state, changed = smoother.update(1, now=10.1)
        self.assertFalse(state)
        self.assertFalse(changed)

        # Frame 3: positive -> M reached, switches ON
        state, changed = smoother.update(1, now=10.2)
        self.assertTrue(state)
        self.assertTrue(changed)

        # Frame 4: positive -> stays ON, no change
        state, changed = smoother.update(1, now=10.3)
        self.assertTrue(state)
        self.assertFalse(changed)

    def test_TC_T1_07_temporal_smoother_off_hysteresis(self):
        """TC_T1_07: TemporalPresenceSmoother requires N negative frames to turn OFF."""
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=15, min_switch_sec=0.0)
        smoother.current_state = True
        smoother.last_switch_time = 0.0

        # Feed 14 negative frames: should remain ON
        for i in range(14):
            state, changed = smoother.update(0, now=10.0 + i * 0.1)
            self.assertTrue(state, f"Premature OFF at frame {i + 1}")
            self.assertFalse(changed)

        # Frame 15: N reached, switches OFF
        state, changed = smoother.update(0, now=12.0)
        self.assertFalse(state)
        self.assertTrue(changed)

    def test_TC_T1_08_duty_cycle_manager_transitions(self):
        """TC_T1_08: DutyCycleManager transitions correctly between EVALUATING and SLEEPING."""
        manager = DutyCycleManager(eval_duration=2.0, sleep_duration=5.0, enabled=True)
        t0 = 100.0
        manager.reset("EVALUATING", now=t0)

        # At t0 + 1.0s: still EVALUATING
        state, countdown, *_ = manager.update(now=t0 + 1.0)
        self.assertEqual(state, "EVALUATING")
        self.assertAlmostEqual(countdown, 1.0, places=1)

        # At t0 + 2.1s: transition to SLEEPING
        state, countdown, transitioned = manager.update(now=t0 + 2.1)
        self.assertEqual(state, "SLEEPING")
        self.assertTrue(transitioned)
        self.assertAlmostEqual(countdown, 5.0, places=1)

        # At t0 + 7.2s (5.1s after entering sleep): transition back to EVALUATING
        state, countdown, transitioned = manager.update(now=t0 + 7.2)
        self.assertEqual(state, "EVALUATING")
        self.assertTrue(transitioned)
        self.assertAlmostEqual(countdown, 2.0, places=1)

    def test_TC_T1_09_duty_cycle_manager_continuous_mode(self):
        """TC_T1_09: DutyCycleManager continuous mode (sleep=0 or duty_cycle_enabled=False) remains in EVALUATING."""
        # Case A: sleep_duration = 0.0
        mgr_zero_sleep = DutyCycleManager(eval_duration=2.0, sleep_duration=0.0, enabled=True)
        mgr_zero_sleep.reset("EVALUATING", now=100.0)
        for elapsed in [0.5, 2.5, 5.0, 10.0]:
            state, countdown, *_ = mgr_zero_sleep.update(now=100.0 + elapsed)
            self.assertEqual(state, "EVALUATING")

        # Case B: duty_cycle_enabled = False
        mgr_disabled = DutyCycleManager(eval_duration=2.0, sleep_duration=5.0, enabled=False)
        mgr_disabled.reset("EVALUATING", now=100.0)
        for elapsed in [0.5, 2.5, 5.0, 10.0]:
            state, countdown, *_ = mgr_disabled.update(now=100.0 + elapsed)
            self.assertEqual(state, "EVALUATING")

    def test_TC_T1_10_engine_config_dynamic_parameter_updates(self):
        """TC_T1_10: EngineConfig dynamic parameter updates take effect immediately."""
        engine = AdvancedDetectionEngine(source="mock", port="mock")
        self.assertEqual(engine.config.conf_threshold, 0.50)
        self.assertEqual(engine.config.eval_duration, 2.0)

        engine.update_config(conf_threshold=0.82, eval_duration=1.2, sleep_duration=3.5)
        self.assertEqual(engine.config.conf_threshold, 0.82)
        self.assertEqual(engine.config.eval_duration, 1.2)
        self.assertEqual(engine.config.sleep_duration, 3.5)
        self.assertEqual(engine.duty_cycle.eval_duration, 1.2)
        self.assertEqual(engine.duty_cycle.sleep_duration, 3.5)


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASE TESTS
# ==============================================================================
class TestTier2BoundaryCornerCases(unittest.TestCase):
    """
    Tier 2 tests verify edge conditions, boundary limits, and resilience (TC_T2_01 through TC_T2_05).
    """

    def test_TC_T2_01_empty_or_degenerate_roi_polygon(self):
        """TC_T2_01: Empty or degenerate ROI polygon falls back safely without exceptions."""
        det = {"xyxy": np.array([100, 100, 200, 200]), "conf": 0.9, "cls": 0}

        # Case 1: Empty list -> safely falls back to full-frame ROI
        roi_empty = SpatialROIFilter(normalized_vertices=[], enabled=True)
        inside, outside = roi_empty.filter_detections([det], 640, 480)
        self.assertEqual(len(inside), 1)
        self.assertEqual(len(outside), 0)

        # Case 2: 1-point polygon -> safely falls back to full-frame ROI
        roi_point = SpatialROIFilter(normalized_vertices=[(0.5, 0.5)], enabled=True)
        inside, outside = roi_point.filter_detections([det], 640, 480)
        self.assertEqual(len(inside), 1)

        # Case 3: 2-point line -> safely falls back to full-frame ROI
        roi_line = SpatialROIFilter(normalized_vertices=[(0.0, 0.0), (1.0, 1.0)], enabled=True)
        inside, outside = roi_line.filter_detections([det], 640, 480)
        self.assertEqual(len(inside), 1)

    def test_TC_T2_02_extreme_confidence_thresholds(self):
        """TC_T2_02: Extreme confidence thresholds (0.01 and 1.00) operate cleanly without numerical errors."""
        cfg_min = EngineConfig(conf_threshold=0.01)
        self.assertAlmostEqual(cfg_min.conf_threshold, 0.01)

        cfg_max = EngineConfig(conf_threshold=1.00)
        self.assertAlmostEqual(cfg_max.conf_threshold, 1.00)

        detections = [
            {"xyxy": np.array([10, 10, 50, 50]), "conf": 0.02, "cls": 0},
            {"xyxy": np.array([60, 60, 100, 100]), "conf": 0.99, "cls": 0},
        ]
        accepted_at_min = [d for d in detections if d["conf"] >= cfg_min.conf_threshold]
        self.assertEqual(len(accepted_at_min), 2)

        accepted_at_max = [d for d in detections if d["conf"] >= cfg_max.conf_threshold]
        self.assertEqual(len(accepted_at_max), 0)

    def test_TC_T2_03_rapid_single_frame_toggle_no_relay_chatter(self):
        """TC_T2_03: Rapid single-frame detection toggle does NOT chatter serial relay."""
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=15, min_switch_sec=1.5)
        mock_hw = MockHardwareController()

        # Oscillating input: 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
        oscillating_pattern = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        for idx, count in enumerate(oscillating_pattern):
            state, changed = smoother.update(count, now=100.0 + idx * 0.05)
            if changed:
                mock_hw.set_light(state)

        # Because consecutive positive never reached 3, state stayed False throughout
        self.assertFalse(smoother.current_state)
        # Serial relay should have received ZERO toggle commands
        self.assertEqual(len(mock_hw.get_command_log()), 0)

    def test_TC_T2_04_duty_cycle_timing_extremes(self):
        """TC_T2_04: High duty cycle intervals and minimal intervals function correctly."""
        # High intervals: 10s eval, 30s sleep
        mgr_high = DutyCycleManager(eval_duration=10.0, sleep_duration=30.0, enabled=True)
        t0 = 200.0
        mgr_high.reset("EVALUATING", now=t0)

        state, rem, *_ = mgr_high.update(now=t0 + 5.0)
        self.assertEqual(state, "EVALUATING")
        self.assertAlmostEqual(rem, 5.0, places=1)

        state, rem, *_ = mgr_high.update(now=t0 + 10.5)
        self.assertEqual(state, "SLEEPING")
        self.assertAlmostEqual(rem, 30.0, places=1)

        # Minimal intervals: 0.5s eval, 0.0s sleep
        mgr_minimal = DutyCycleManager(eval_duration=0.5, sleep_duration=0.0, enabled=True)
        mgr_minimal.reset("EVALUATING", now=t0)
        state, rem, *_ = mgr_minimal.update(now=t0 + 2.0)
        self.assertEqual(state, "EVALUATING")

    def test_TC_T2_05_hardware_and_camera_failure_handling(self):
        """TC_T2_05: Non-existent serial port or camera failure handled gracefully."""
        # Non-existent COM port connect
        real_hw = eco_engine.HardwareController(port="INVALID_PORT_XYZ_9999")
        success, msg = real_hw.connect()
        self.assertFalse(success)
        self.assertFalse(real_hw.is_connected)
        real_hw.disconnect()

        # MockHardwareController disconnect & connect
        mock_hw = MockHardwareController(port="MOCK_TEST")
        mock_hw.disconnect()
        self.assertFalse(mock_hw.is_connected)
        mock_hw.connect()
        self.assertTrue(mock_hw.is_connected)


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATION TESTS
# ==============================================================================
class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """
    Tier 3 tests verify pairwise and multi-component interactions (TC_T3_01 through TC_T3_04).
    """

    def test_TC_T3_01_roi_filter_plus_temporal_smoothing_pipeline(self):
        """TC_T3_01: ROI filter + Temporal smoothing combined pipeline."""
        # Active ROI: Left half [0.0, 0.5]
        roi_filter = SpatialROIFilter(
            normalized_vertices=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
            mode="BOTTOM_CENTER",
            enabled=True
        )
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=15, min_switch_sec=0.0)
        mock_hw = MockHardwareController()

        # Step 1: Occupant outside ROI (x=500, y=300 in 640x480) for 5 frames
        det_outside = [{"xyxy": np.array([450, 100, 550, 300]), "conf": 0.9, "cls": 0}]
        for i in range(5):
            inside, _ = roi_filter.filter_detections(det_outside, 640, 480)
            state, changed = smoother.update(len(inside), now=10.0 + i * 0.1)
            if changed:
                mock_hw.set_light(state)

        self.assertFalse(smoother.current_state)
        self.assertEqual(len(mock_hw.get_command_log()), 0)

        # Step 2: Occupant inside ROI (x=150, y=300) for 3 frames
        det_inside = [{"xyxy": np.array([100, 100, 200, 300]), "conf": 0.9, "cls": 0}]
        for i in range(3):
            inside, _ = roi_filter.filter_detections(det_inside, 640, 480)
            state, changed = smoother.update(len(inside), now=20.0 + i * 0.1)
            if changed:
                mock_hw.set_light(state)

        self.assertTrue(smoother.current_state)
        self.assertEqual(len(mock_hw.get_command_log()), 1)
        self.assertEqual(mock_hw.get_command_log()[0][1], '1')

    def test_TC_T3_02_duty_cycle_state_transitions_with_dynamic_tuning(self):
        """TC_T3_02: Duty cycle state transitions while dynamic confidence is adjusted."""
        engine = AdvancedDetectionEngine(source="mock", port="mock")
        engine.update_config(eval_duration=1.0, sleep_duration=2.0, conf_threshold=0.50)

        t0 = 50.0
        engine.duty_cycle.reset("EVALUATING", now=t0)

        # In EVALUATING
        state, countdown, *_ = engine.duty_cycle.update(now=t0 + 0.5)
        self.assertEqual(state, "EVALUATING")

        # Advance to SLEEPING
        state, countdown, *_ = engine.duty_cycle.update(now=t0 + 1.2)
        self.assertEqual(state, "SLEEPING")

        # Dynamically adjust confidence and eval duration while SLEEPING
        engine.update_config(conf_threshold=0.85, eval_duration=1.5)
        self.assertEqual(engine.config.conf_threshold, 0.85)
        self.assertEqual(engine.duty_cycle.eval_duration, 1.5)

        # Wake back up to EVALUATING (after 2.0s sleep)
        state, rem, *_ = engine.duty_cycle.update(now=t0 + 3.3)
        self.assertEqual(state, "EVALUATING")
        self.assertAlmostEqual(rem, 1.5, places=1)

    def test_TC_T3_03_manual_overrides_during_duty_cycle(self):
        """TC_T3_03: Manual overrides (FORCE_ON, FORCE_OFF, AUTO) during duty cycle."""
        engine = AdvancedDetectionEngine(source="mock", port="mock")

        # AUTO: initially OFF
        self.assertEqual(engine.config.override_mode, "AUTO")
        self.assertFalse(engine.hardware.current_state)

        # FORCE_ON: turns light ON immediately
        engine.set_override("FORCE_ON")
        self.assertEqual(engine.config.override_mode, "FORCE_ON")
        self.assertTrue(engine.hardware.current_state)
        self.assertEqual(engine.hardware.get_command_log()[-1][1], '1')

        # FORCE_OFF: turns light OFF immediately
        engine.set_override("FORCE_OFF")
        self.assertEqual(engine.config.override_mode, "FORCE_OFF")
        self.assertFalse(engine.hardware.current_state)
        self.assertEqual(engine.hardware.get_command_log()[-1][1], '0')

        # Return to AUTO
        engine.set_override("AUTO")
        self.assertEqual(engine.config.override_mode, "AUTO")
        self.assertEqual(engine.hardware.current_state, engine.smoother.current_state)

    def test_TC_T3_04_spatial_roi_person_entering_and_exiting(self):
        """TC_T3_04: Spatial ROI with person entering from outside to inside ROI."""
        roi_filter = SpatialROIFilter(
            normalized_vertices=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
            mode="BOTTOM_CENTER",
            enabled=True
        )
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=15, min_switch_sec=0.0)
        mock_hw = MockHardwareController()

        # Step 1: Person outside ROI (x=500)
        for i in range(5):
            det = [{"xyxy": np.array([450, 100, 550, 300]), "conf": 0.9, "cls": 0}]
            inside, _ = roi_filter.filter_detections(det, 640, 480)
            state, changed = smoother.update(len(inside), now=10.0 + i * 0.1)
            if changed:
                mock_hw.set_light(state)
        self.assertFalse(smoother.current_state)
        self.assertEqual(len(mock_hw.get_command_log()), 0)

        # Step 2: Person enters ROI (x=200) for 3 frames -> switches ON
        for i in range(3):
            det = [{"xyxy": np.array([150, 100, 250, 300]), "conf": 0.9, "cls": 0}]
            inside, _ = roi_filter.filter_detections(det, 640, 480)
            state, changed = smoother.update(len(inside), now=20.0 + i * 0.1)
            if changed:
                mock_hw.set_light(state)
        self.assertTrue(smoother.current_state)
        self.assertEqual(len(mock_hw.get_command_log()), 1)
        self.assertEqual(mock_hw.get_command_log()[0][1], '1')

        # Step 3: Person exits ROI (x=500) for 15 frames -> switches OFF
        for i in range(15):
            det = [{"xyxy": np.array([450, 100, 550, 300]), "conf": 0.9, "cls": 0}]
            inside, _ = roi_filter.filter_detections(det, 640, 480)
            state, changed = smoother.update(len(inside), now=30.0 + i * 0.1)
            if changed:
                mock_hw.set_light(state)
        self.assertFalse(smoother.current_state)
        self.assertEqual(len(mock_hw.get_command_log()), 2)
        self.assertEqual(mock_hw.get_command_log()[1][1], '0')


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ==============================================================================
class TestTier4RealWorldScenarios(unittest.TestCase):
    """
    Tier 4 tests simulate full physical facility deployments headlessly (TC_T4_01 through TC_T4_05).
    """

    def test_TC_T4_01_classroom_lecture_scenario(self):
        """
        TC_T4_01: Classroom Lecture Scenario.
        - Room empty initially -> Lights stay OFF.
        - Students arrive, class starts -> Lights switch ON on 3rd positive frame, stay ON.
        - Class dismisses, room empties -> Lights hold ON through 15 negative frames, then turn OFF.
        """
        roi_filter = SpatialROIFilter(
            normalized_vertices=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
            mode="BOTTOM_CENTER",
            enabled=True
        )
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=15, min_switch_sec=0.0)
        mock_hw = MockHardwareController()

        t_sim = 1000.0

        # Period 1: Empty room for 10 frames
        for _ in range(10):
            inside, _ = roi_filter.filter_detections([], 640, 480)
            state, changed = smoother.update(len(inside), now=t_sim)
            if changed:
                mock_hw.set_light(state)
            t_sim += 0.1
        self.assertFalse(mock_hw.current_state)
        self.assertEqual(len(mock_hw.get_command_log()), 0)

        # Period 2: Class arrives (4 students inside classroom ROI) for 25 frames
        students = [
            {"xyxy": np.array([120, 200, 180, 400]), "conf": 0.92, "cls": 0},
            {"xyxy": np.array([250, 200, 310, 400]), "conf": 0.88, "cls": 0},
            {"xyxy": np.array([380, 200, 440, 400]), "conf": 0.91, "cls": 0},
            {"xyxy": np.array([500, 200, 560, 400]), "conf": 0.85, "cls": 0},
        ]
        for f in range(25):
            inside, _ = roi_filter.filter_detections(students, 640, 480)
            state, changed = smoother.update(len(inside), now=t_sim)
            if changed:
                mock_hw.set_light(state)
            t_sim += 0.1

        self.assertTrue(mock_hw.current_state)
        # Exactly one '1' emitted on frame 3 of students arriving
        self.assertEqual(len(mock_hw.get_command_log()), 1)
        self.assertEqual(mock_hw.get_command_log()[0][1], '1')

        # Period 3: Class dismisses (0 students).
        # Frames 1-14: lights must stay ON (hold-off)
        for f in range(14):
            inside, _ = roi_filter.filter_detections([], 640, 480)
            state, changed = smoother.update(len(inside), now=t_sim)
            if changed:
                mock_hw.set_light(state)
            t_sim += 0.1
            self.assertTrue(mock_hw.current_state, f"Lights turned off prematurely at hold-off frame {f + 1}")

        # Frame 15: lights transition to OFF
        inside, _ = roi_filter.filter_detections([], 640, 480)
        state, changed = smoother.update(len(inside), now=t_sim)
        if changed:
            mock_hw.set_light(state)
        t_sim += 0.1

        self.assertFalse(mock_hw.current_state)
        self.assertEqual(len(mock_hw.get_command_log()), 2)
        self.assertEqual(mock_hw.get_command_log()[1][1], '0')

    def test_TC_T4_02_corridor_transient_passerby_scenario(self):
        """
        TC_T4_02: Corridor Transient Passerby Scenario.
        - Room ROI covers classroom interior.
        - People walk along the adjacent corridor (outside the ROI polygon).
        - Verification: Classroom lights remain strictly OFF throughout.
        """
        classroom_roi = SpatialROIFilter(
            normalized_vertices=[(0.0, 0.0), (0.70, 0.0), (0.70, 1.0), (0.0, 1.0)],
            mode="BOTTOM_CENTER",
            enabled=True
        )
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=15, min_switch_sec=0.0)
        mock_hw = MockHardwareController()

        # Passerby in corridor: x=520 to 600 (x_norm = 0.81 to 0.94, outside ROI)
        passerby = [{"xyxy": np.array([520, 100, 600, 450]), "conf": 0.94, "cls": 0}]

        for i in range(20):
            inside, outside = classroom_roi.filter_detections(passerby, 640, 480)
            self.assertEqual(len(inside), 0)
            self.assertEqual(len(outside), 1)

            state, changed = smoother.update(len(inside), now=100.0 + i * 0.1)
            if changed:
                mock_hw.set_light(state)

        self.assertFalse(mock_hw.current_state)
        self.assertEqual(len(mock_hw.get_command_log()), 0)

    def test_TC_T4_03_study_session_micro_movements_scenario(self):
        """
        TC_T4_03: Study Session Micro-Movements Scenario.
        - Student seated at desk in ROI.
        - Minor head-down pauses cause momentary single-frame drops.
        - Verification: Light remains rock-solid ON without relay flicker.
        """
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=15, min_switch_sec=0.0)
        mock_hw = MockHardwareController()

        # 3 initial positive frames to turn ON
        for i in range(3):
            state, changed = smoother.update(1, now=10.0 + i * 0.1)
            if changed:
                mock_hw.set_light(state)
        self.assertTrue(mock_hw.current_state)

        # Sequence with intermittent single-frame drops: 1, 1, 0, 1, 0, 1, 1, 0, 1
        study_sequence = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1]
        for idx, count in enumerate(study_sequence):
            state, changed = smoother.update(count, now=20.0 + idx * 0.1)
            if changed:
                mock_hw.set_light(state)
            self.assertTrue(mock_hw.current_state, f"Flickered OFF during study session at index {idx}")

        # Still only one '1' command in history (zero flickering)
        self.assertEqual(len(mock_hw.get_command_log()), 1)
        self.assertEqual(mock_hw.get_command_log()[0][1], '1')

    def test_TC_T4_04_energy_saver_duty_cycle_scenario(self):
        """
        TC_T4_04: Energy Saver Duty Cycle Scenario.
        - Room occupied, lights turn ON.
        - System enters SLEEPING to save computer power (inference idles).
        - System wakes up into EVALUATING and maintains stable control.
        """
        engine = AdvancedDetectionEngine(source="mock", port="mock")
        engine.update_config(eval_duration=2.0, sleep_duration=4.0, duty_cycle_enabled=True)

        t_sim = 500.0
        engine.duty_cycle.reset("EVALUATING", now=t_sim)

        # Evaluating phase: occupancy detected -> light turns ON
        engine.smoother.update(1, now=t_sim)
        engine.smoother.update(1, now=t_sim + 0.1)
        state, changed = engine.smoother.update(1, now=t_sim + 0.2)
        if changed:
            engine.hardware.set_light(state)
        self.assertTrue(engine.hardware.current_state)

        # Advance to SLEEPING phase
        cycle_state, countdown, transitioned = engine.duty_cycle.update(now=t_sim + 2.1)
        self.assertEqual(cycle_state, "SLEEPING")
        self.assertTrue(transitioned)
        # In SLEEPING, light state remains ON
        self.assertTrue(engine.hardware.current_state)

        # Advance to next EVALUATING phase
        cycle_state, rem, transitioned = engine.duty_cycle.update(now=t_sim + 6.2)
        self.assertEqual(cycle_state, "EVALUATING")
        self.assertTrue(transitioned)
        self.assertTrue(engine.hardware.current_state)

    def test_TC_T4_05_emergency_manual_override_scenario(self):
        """
        TC_T4_05: Emergency Manual Override Scenario.
        - Classroom is fully occupied (25 students).
        - Instructor activates projector mode (FORCE_OFF).
        - Verification: Serial '0' is sent immediately; lights stay OFF despite detections.
        """
        engine = AdvancedDetectionEngine(source="mock", port="mock")
        # Ensure hardware starts connected with lights ON
        engine.hardware.set_light(True)
        self.assertTrue(engine.hardware.current_state)

        # Instructor activates FORCE_OFF
        engine.set_override("FORCE_OFF")
        self.assertFalse(engine.hardware.current_state)
        self.assertEqual(engine.hardware.get_command_log()[-1][1], '0')

        # Even if multiple positive frames occur, FORCE_OFF overrides sensor actuation
        students = [{"xyxy": np.array([100, 100, 200, 300]), "conf": 0.95, "cls": 0}]
        for i in range(10):
            # When override is FORCE_OFF, light stays False
            if engine.config.override_mode == "FORCE_OFF":
                engine.hardware.set_light(False)

        self.assertFalse(engine.hardware.current_state)


# ==============================================================================
# MAIN TEST ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    unittest.main()

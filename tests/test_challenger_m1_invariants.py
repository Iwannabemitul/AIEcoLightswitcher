"""
tests/test_challenger_m1_invariants.py

Empirical invariant, boundary condition, and edge-case challenge suite
for Milestone 1 (core/ algorithms, ingestion/ components, and eco_engine.py facade).
Authored by challenger_m1_2.
"""

import math
import time
import unittest
from typing import Any, Dict, List, Tuple
import numpy as np

# Verify both direct package imports and facade re-exports
import core
import eco_engine
from core.config import EngineConfig, FeedConfig, ConnectionResult, DEFAULT_TARGET_CLASSES
from core.device import detect_best_device, get_inference_device, warmup_engine
from core.duty_cycle import DutyCycleManager
from core.roi_filter import SpatialROIFilter
from core.smoother import TemporalPresenceSmoother, TemporalPresenceDebouncer
from ingestion.base_stream import BaseVideoStream, FeedStatus, StreamTelemetry
from ingestion.mock_stream import MockVideoSource
from ingestion.stream_worker import StreamWorker, SingleFrameBuffer, BufferedFrame


class TestFacadeParityAndExportIntegrity(unittest.TestCase):
    """
    Empirically verifies that eco_engine facade exports all core and ingestion
    symbols with 100% identity and parity.
    """

    def test_core_symbols_identity(self):
        """All core symbols exported in eco_engine must be identical objects to core/."""
        expected_core_symbols = [
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
        ]
        for sym in expected_core_symbols:
            self.assertTrue(
                hasattr(eco_engine, sym),
                f"eco_engine is missing core symbol: {sym}"
            )
            core_obj = getattr(core, sym)
            facade_obj = getattr(eco_engine, sym)
            self.assertIs(
                core_obj, facade_obj,
                f"Symbol {sym} in eco_engine is not identical to core.{sym}"
            )

    def test_ingestion_symbols_identity(self):
        """All critical ingestion symbols must exist in eco_engine."""
        expected_ingestion_symbols = [
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
        ]
        for sym in expected_ingestion_symbols:
            self.assertTrue(
                hasattr(eco_engine, sym),
                f"eco_engine is missing ingestion symbol: {sym}"
            )

    def test_legacy_pipeline_symbols_exist(self):
        """eco_engine must retain legacy hardware and pipeline symbols."""
        legacy_symbols = [
            "BaseHardwareController",
            "HardwareController",
            "MockHardwareController",
            "AdvancedDetectionEngine",
            "EcoVisionSystem",
        ]
        for sym in legacy_symbols:
            self.assertTrue(
                hasattr(eco_engine, sym),
                f"eco_engine missing legacy symbol: {sym}"
            )

    def test_engine_config_facade_behavior(self):
        """EngineConfig instantiated from eco_engine behaves identically."""
        cfg1 = core.EngineConfig(conf_threshold=0.65, eval_duration=3.5)
        cfg2 = eco_engine.EngineConfig(conf_threshold=0.65, eval_duration=3.5)
        self.assertEqual(cfg1.to_dict(), cfg2.to_dict())
        self.assertEqual(cfg1.conf_threshold, 0.65)
        self.assertEqual(cfg2.eval_duration, 3.5)


class TestSpatialROIFilterDegenerateAndEdgeCases(unittest.TestCase):
    """
    Adversarial challenge for SpatialROIFilter against degenerate polygons,
    boundary coordinates, out-of-bounds bounding boxes, and zero-size frames.
    """

    def test_degenerate_vertices_empty_and_insufficient(self):
        """Filter must safely sanitize empty or < 3 vertex definitions to unit box."""
        # None -> default unit box
        f_none = SpatialROIFilter(normalized_vertices=None)
        self.assertEqual(f_none.vertices, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

        # Empty list -> default unit box
        f_empty = SpatialROIFilter(normalized_vertices=[])
        self.assertEqual(f_empty.vertices, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

        # 1 vertex -> default unit box
        f_one = SpatialROIFilter(normalized_vertices=[(0.5, 0.5)])
        self.assertEqual(f_one.vertices, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

        # 2 vertices (line segment, not polygon) -> default unit box
        f_two = SpatialROIFilter(normalized_vertices=[(0.2, 0.2), (0.8, 0.8)])
        self.assertEqual(f_two.vertices, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    def test_out_of_bounds_normalized_vertices_clamped(self):
        """Vertices outside [0.0, 1.0] must be clamped strictly to [0.0, 1.0]."""
        raw_verts = [(-0.5, -0.2), (1.5, 0.0), (1.2, 1.8), (-0.1, 1.1)]
        rf = SpatialROIFilter(normalized_vertices=raw_verts)
        self.assertEqual(rf.vertices, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    def test_collinear_vertices_does_not_crash(self):
        """Collinear polygon points (zero-area triangle) evaluate safely without exception."""
        collinear = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]
        rf = SpatialROIFilter(normalized_vertices=collinear, enabled=True)
        # Should evaluate without crash
        in_res = rf.is_point_in_roi((320, 240), 640, 480)
        self.assertIsInstance(in_res, bool)

    def test_zero_and_negative_frame_dimensions(self):
        """Zero or negative frame dimensions must return False without throwing cv2 error."""
        rf = SpatialROIFilter(enabled=True)
        self.assertFalse(rf.is_point_in_roi((50, 50), 0, 480))
        self.assertFalse(rf.is_point_in_roi((50, 50), 640, 0))
        self.assertFalse(rf.is_point_in_roi((50, 50), -640, 480))

    def test_boundary_points_containment(self):
        """Points exactly on the polygon perimeter and vertices evaluate as contained (res >= 0)."""
        square = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        rf = SpatialROIFilter(normalized_vertices=square, enabled=True)
        w, h = 1000, 1000

        # Exact corners: (200, 200), (800, 200), (800, 800), (200, 800)
        self.assertTrue(rf.is_point_in_roi((200, 200), w, h), "Top-left vertex must be in ROI")
        self.assertTrue(rf.is_point_in_roi((800, 200), w, h), "Top-right vertex must be in ROI")
        self.assertTrue(rf.is_point_in_roi((800, 800), w, h), "Bottom-right vertex must be in ROI")
        self.assertTrue(rf.is_point_in_roi((200, 800), w, h), "Bottom-left vertex must be in ROI")

        # Exact edges: (500, 200), (800, 500), (500, 800), (200, 500)
        self.assertTrue(rf.is_point_in_roi((500, 200), w, h), "Top edge midpoint must be in ROI")
        self.assertTrue(rf.is_point_in_roi((800, 500), w, h), "Right edge midpoint must be in ROI")
        self.assertTrue(rf.is_point_in_roi((500, 800), w, h), "Bottom edge midpoint must be in ROI")
        self.assertTrue(rf.is_point_in_roi((200, 500), w, h), "Left edge midpoint must be in ROI")

        # Interior vs Exterior
        self.assertTrue(rf.is_point_in_roi((500, 500), w, h), "Center must be in ROI")
        self.assertFalse(rf.is_point_in_roi((100, 100), w, h), "Exterior point must not be in ROI")
        self.assertFalse(rf.is_point_in_roi((900, 900), w, h), "Exterior point must not be in ROI")

    def test_extreme_and_out_of_bounds_bounding_boxes(self):
        """Bounding boxes far outside frame dimensions clamp correctly to frame margins."""
        rf = SpatialROIFilter(
            normalized_vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            mode="BOTTOM_CENTER",
            enabled=True
        )
        w, h = 640, 480

        # Negative bounding box: [-200, -200, -100, -100]
        neg_det = [{"xyxy": [-200, -200, -100, -100], "conf": 0.9}]
        inside, outside = rf.filter_detections(neg_det, w, h)
        # Clamped anchor is (0, 0), which is on the boundary of the unit box
        self.assertEqual(neg_det[0]["anchor"], (0, 0))
        self.assertEqual(len(inside), 1)

        # Huge out-of-frame bounding box: [1000, 1000, 2000, 2000]
        huge_det = [{"xyxy": [1000, 1000, 2000, 2000], "conf": 0.9}]
        inside, outside = rf.filter_detections(huge_det, w, h)
        # Clamped anchor is (w-1, h-1) = (639, 479)
        self.assertEqual(huge_det[0]["anchor"], (639, 479))
        self.assertEqual(len(inside), 1)

    def test_disabled_filter_passes_all_detections(self):
        """When enabled=False, all detections are marked in_roi=True and returned inside."""
        rf = SpatialROIFilter(
            normalized_vertices=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)],
            enabled=False
        )
        dets = [
            {"xyxy": [10, 10, 50, 50], "conf": 0.9},
            {"xyxy": [500, 400, 600, 470], "conf": 0.8},
        ]
        inside, outside = rf.filter_detections(dets, 640, 480)
        self.assertEqual(len(inside), 2)
        self.assertEqual(len(outside), 0)
        for d in inside:
            self.assertTrue(d["in_roi"])

    def test_anchor_mode_switching(self):
        """Switching between BOTTOM_CENTER and CENTER produces mathematically expected anchors."""
        rf = SpatialROIFilter()
        xyxy = [100, 200, 300, 400]
        # BOTTOM_CENTER: mid_x = 200, y2 = 400
        rf.set_mode("BOTTOM_CENTER")
        self.assertEqual(rf.get_anchor_point(xyxy), (200, 400))

        # CENTER: mid_x = 200, mid_y = 300
        rf.set_mode("CENTER")
        self.assertEqual(rf.get_anchor_point(xyxy), (200, 300))

        # CENTROID alias: mid_x = 200, mid_y = 300
        rf.set_mode("centroid")
        self.assertEqual(rf.get_anchor_point(xyxy), (200, 300))

    def test_inverted_bounding_box_coordinates(self):
        """Inverted coordinates [x2, y2, x1, y1] should still produce midpoint without crashing."""
        rf = SpatialROIFilter(mode="CENTER")
        # Inverted coordinates: x1=400, y1=500, x2=200, y2=100
        xyxy = [400, 500, 200, 100]
        anchor = rf.get_anchor_point(xyxy)
        self.assertEqual(anchor, (300, 300))

    def test_draw_roi_overlay_resilience(self):
        """draw_roi_overlay executes safely on active and empty frames."""
        rf = SpatialROIFilter(
            normalized_vertices=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
            enabled=True
        )
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw empty overlay
        out1 = rf.draw_roi_overlay(dummy_frame, is_occupied=False)
        self.assertEqual(out1.shape, (480, 640, 3))

        # Draw occupied overlay
        out2 = rf.draw_roi_overlay(dummy_frame, is_occupied=True, label="ALERT: PERSON")
        self.assertEqual(out2.shape, (480, 640, 3))

        # When disabled, returns original frame unchanged
        rf.set_enabled(False)
        out3 = rf.draw_roi_overlay(dummy_frame, is_occupied=True)
        self.assertIs(out3, dummy_frame)


class TestTemporalPresenceSmootherHysteresis(unittest.TestCase):
    """
    Adversarial verification of asymmetric hysteresis ($M=3$ ON, $N=15$ OFF),
    state transitions, and counter resets.
    """

    def test_exact_m3_positive_transition(self):
        """Requires exactly M=3 consecutive positive updates to transition from OFF to ON."""
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=15, min_switch_interval=0.0)
        smoother.reset(state=False, now=0.0)

        # Frame 1: positive
        st, ch = smoother.update(True, now=1.0)
        self.assertFalse(st)
        self.assertFalse(ch)
        self.assertEqual(smoother.consecutive_positive, 1)

        # Frame 2: positive
        st, ch = smoother.update(True, now=2.0)
        self.assertFalse(st)
        self.assertFalse(ch)
        self.assertEqual(smoother.consecutive_positive, 2)

        # Frame 3: positive -> should switch ON!
        st, ch = smoother.update(True, now=3.0)
        self.assertTrue(st)
        self.assertTrue(ch)
        self.assertEqual(smoother.consecutive_positive, 3)

        # Frame 4: positive -> stays ON, no change
        st, ch = smoother.update(True, now=4.0)
        self.assertTrue(st)
        self.assertFalse(ch)

    def test_interrupted_positive_streak_resets_counter(self):
        """A single negative frame during the ON streak resets consecutive_positive to 0."""
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=15, min_switch_interval=0.0)
        smoother.reset(state=False, now=0.0)

        smoother.update(True, now=1.0)  # pos=1
        smoother.update(True, now=2.0)  # pos=2
        st, ch = smoother.update(False, now=3.0)  # interrupted!
        self.assertFalse(st)
        self.assertFalse(ch)
        self.assertEqual(smoother.consecutive_positive, 0)
        self.assertEqual(smoother.consecutive_negative, 1)

        # Needs 3 fresh positive frames to trigger ON
        smoother.update(True, now=4.0)  # pos=1
        smoother.update(True, now=5.0)  # pos=2
        st, ch = smoother.update(True, now=6.0)  # pos=3 -> ON
        self.assertTrue(st)
        self.assertTrue(ch)

    def test_exact_n15_negative_transition(self):
        """Requires exactly N=15 consecutive negative updates to transition from ON to OFF."""
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=15, min_switch_interval=0.0)
        smoother.reset(state=True, now=0.0)

        for i in range(1, 15):
            st, ch = smoother.update(False, now=float(i))
            self.assertTrue(st, f"At step {i}, state should still be True (hysteresis)")
            self.assertFalse(ch, f"At step {i}, no change should be reported")
            self.assertEqual(smoother.consecutive_negative, i)

        # Frame 15: switches OFF!
        st, ch = smoother.update(False, now=15.0)
        self.assertFalse(st, "At step 15, state should transition to False")
        self.assertTrue(ch, "At step 15, state_changed should be True")
        self.assertEqual(smoother.consecutive_negative, 15)

    def test_interrupted_negative_streak_resets_counter(self):
        """A single positive frame at frame 14 resets the countdown, preserving ON state."""
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=15, min_switch_interval=0.0)
        smoother.reset(state=True, now=0.0)

        # 14 negative frames
        for i in range(1, 15):
            smoother.update(False, now=float(i))
        self.assertEqual(smoother.consecutive_negative, 14)

        # Interrupted at 14 by 1 positive frame
        st, ch = smoother.update(True, now=15.0)
        self.assertTrue(st)
        self.assertFalse(ch)
        self.assertEqual(smoother.consecutive_negative, 0)
        self.assertEqual(smoother.consecutive_positive, 1)

        # Needs full 15 negative frames again
        for i in range(1, 15):
            st, ch = smoother.update(False, now=15.0 + float(i))
            self.assertTrue(st)
            self.assertFalse(ch)
        st, ch = smoother.update(False, now=30.0)
        self.assertFalse(st)
        self.assertTrue(ch)

    def test_numeric_occupant_counts(self):
        """Supports numeric occupant counts (counts > 0 treated as positive)."""
        smoother = TemporalPresenceSmoother(on_frames=2, off_frames=2, min_switch_interval=0.0)
        smoother.reset(state=False, now=0.0)

        smoother.update(3, now=1.0)
        st, ch = smoother.update(1, now=2.0)
        self.assertTrue(st)
        self.assertTrue(ch)

        smoother.update(0, now=3.0)
        st, ch = smoother.update(0.0, now=4.0)
        self.assertFalse(st)
        self.assertTrue(ch)


class TestTemporalPresenceSmootherAntiChatterAndClamps(unittest.TestCase):
    """
    Adversarial challenge for min_switch_interval anti-chatter clamp,
    high-frequency oscillation, and dynamic parameter adjustments.
    """

    def test_anti_chatter_clamp_blocks_rapid_turnoff(self):
        """Even with 15 negative frames, transition to OFF is clamped until min_switch_interval elapses."""
        min_sec = 2.0
        smoother = TemporalPresenceSmoother(on_frames=1, off_frames=15, min_switch_interval=min_sec)
        # Reset at t=0.0
        smoother.reset(state=False, now=0.0)

        # Switch ON at t=5.0 (5.0s > 2.0s, so switch ON succeeds)
        st, ch = smoother.update(True, now=5.0)
        self.assertTrue(st, "Initial switch ON at t=5.0 must succeed")
        self.assertTrue(ch)
        self.assertEqual(smoother.last_switch_time, 5.0)

        # In rapid succession, 15 negative frames arrive at t=5.5 (only 0.5s after ON switch)
        for i in range(15):
            st, ch = smoother.update(False, now=5.5)
            self.assertTrue(st, "Anti-chatter clamp must keep state ON before min_switch_interval")
            self.assertFalse(ch)

        # At t=6.9 (still < 5.0 + 2.0 = 7.0), state remains locked ON
        st, ch = smoother.update(False, now=6.9)
        self.assertTrue(st, "Anti-chatter clamp must keep state ON at t=6.9")
        self.assertFalse(ch)

        # At t=7.01 (>= 7.0), anti-chatter clamp expires -> transition to OFF succeeds
        st, ch = smoother.update(False, now=7.01)
        self.assertFalse(st, "Must transition to OFF once min_switch_interval has elapsed")
        self.assertTrue(ch)
        self.assertEqual(smoother.last_switch_time, 7.01)

    def test_anti_chatter_clamp_blocks_rapid_turnon(self):
        """Turn ON is clamped if min_switch_interval has not elapsed since turn OFF."""
        min_sec = 2.0
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=1, min_switch_interval=min_sec)
        # Reset at t=0.0 with state=True
        smoother.reset(state=True, now=0.0)

        # Switch OFF at t=5.0 (5.0s > 2.0s, so switch OFF succeeds)
        st, ch = smoother.update(False, now=5.0)
        self.assertFalse(st, "Initial switch OFF at t=5.0 must succeed")
        self.assertTrue(ch)
        self.assertEqual(smoother.last_switch_time, 5.0)

        # 3 positive frames arrive at t=5.8 (only 0.8s < 2.0s since OFF switch)
        smoother.update(True, now=5.8)
        smoother.update(True, now=5.8)
        st, ch = smoother.update(True, now=5.8)
        self.assertFalse(st, "Anti-chatter clamp must prevent turning back ON prematurely at t=5.8")
        self.assertFalse(ch)

        # At t=7.05 (>= 5.0 + 2.0 = 7.0), positive update arrives -> turns ON
        st, ch = smoother.update(True, now=7.05)
        self.assertTrue(st, "Must transition to ON once min_switch_interval has elapsed")
        self.assertTrue(ch)
        self.assertEqual(smoother.last_switch_time, 7.05)

    def test_rapid_oscillation_suppression(self):
        """Under rapid alternating presence (True, False, True, False...), state changes must be < 1%."""
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=15, min_switch_interval=1.5)
        smoother.reset(state=False, now=0.0)

        total_switches = 0
        current_time = 0.0
        for i in range(1000):
            current_time += 0.033  # ~30 FPS
            raw = (i % 2 == 0)
            st, ch = smoother.update(raw, now=current_time)
            if ch:
                total_switches += 1

        # Because raw alternates every frame, consecutive_positive never reaches 3 and
        # consecutive_negative never reaches 15. Total state switches must be exactly 0!
        self.assertEqual(total_switches, 0, "Alternating noise must produce 0 state switches")

    def test_dynamic_param_updates(self):
        """Updating parameters via update_params alters behavior immediately."""
        smoother = TemporalPresenceSmoother(on_frames=10, off_frames=50, min_switch_interval=5.0)
        smoother.reset(state=False, now=0.0)

        # Update to eager parameters: on_frames=1, off_frames=1, min_switch_sec=0.0
        smoother.update_params(on_frames=1, off_frames=1, min_switch_sec=0.0)
        self.assertEqual(smoother.on_frames, 1)
        self.assertEqual(smoother.off_frames, 1)
        self.assertEqual(smoother.min_switch_interval, 0.0)

        # Single update immediately flips ON
        st, ch = smoother.update(True, now=0.1)
        self.assertTrue(st)
        self.assertTrue(ch)

    def test_disabled_smoother_still_respects_anti_chatter(self):
        """When enabled=False, smoother passes through raw state but STILL respects anti-chatter clamp."""
        smoother = TemporalPresenceSmoother(min_switch_interval=2.0, enabled=False)
        smoother.reset(state=False, now=0.0)

        # At t=0.5 (< 2.0s), presence arrives -> should NOT switch because anti-chatter is active
        st, ch = smoother.update(True, now=0.5)
        self.assertFalse(st)
        self.assertFalse(ch)

        # At t=2.1 (>= 2.0s), presence arrives -> switches ON immediately (0 hysteresis delay)
        st, ch = smoother.update(True, now=2.1)
        self.assertTrue(st)
        self.assertTrue(ch)

    def test_negative_occupant_counts_treated_as_false(self):
        """Negative counts e.g. -1, -100 are treated as zero/False."""
        smoother = TemporalPresenceSmoother(on_frames=1, off_frames=1, min_switch_interval=0.0)
        smoother.reset(state=True, now=0.0)

        st, ch = smoother.update(-5, now=1.0)
        self.assertFalse(st)
        self.assertTrue(ch)

    def test_non_monotonic_time_jump_backwards(self):
        """If clock jumps backwards, smoother refuses to switch until time catches up."""
        smoother = TemporalPresenceSmoother(on_frames=1, off_frames=1, min_switch_interval=2.0)
        smoother.reset(state=False, now=0.0)

        # Switch ON at t=10.0
        smoother.update(True, now=10.0)
        self.assertEqual(smoother.last_switch_time, 10.0)

        # Clock jumps backward to t=5.0
        st, ch = smoother.update(False, now=5.0)
        self.assertTrue(st, "Must not switch when now < last_switch_time")
        self.assertFalse(ch)


class TestDutyCycleManagerTransitionsAndContinuousMode(unittest.TestCase):
    """
    Adversarial verification of DutyCycleManager evaluating/sleeping phases,
    continuous mode (sleep_duration=0.0), and timer resets.
    """

    def test_nominal_phase_cycle(self):
        """Verifies exact deterministic phase cycle between EVALUATING and SLEEPING."""
        eval_dur = 2.0
        sleep_dur = 4.0
        dc = DutyCycleManager(eval_duration=eval_dur, sleep_duration=sleep_dur, enabled=True)
        dc.reset(phase="EVALUATING", now=100.0)

        # Mid-evaluating at t=101.0
        phase, countdown, trans = dc.update(now=101.0)
        self.assertEqual(phase, "EVALUATING")
        self.assertAlmostEqual(countdown, 1.0, places=2)
        self.assertFalse(trans)
        self.assertTrue(dc.should_infer())

        # Exact transition to SLEEPING at t=102.0
        phase, countdown, trans = dc.update(now=102.0)
        self.assertEqual(phase, "SLEEPING")
        self.assertAlmostEqual(countdown, 4.0, places=2)
        self.assertTrue(trans)
        self.assertFalse(dc.should_infer())

        # Mid-sleeping at t=104.0
        phase, countdown, trans = dc.update(now=104.0)
        self.assertEqual(phase, "SLEEPING")
        self.assertAlmostEqual(countdown, 2.0, places=2)
        self.assertFalse(trans)
        self.assertFalse(dc.should_infer())

        # Transition back to EVALUATING at t=106.0
        phase, countdown, trans = dc.update(now=106.0)
        self.assertEqual(phase, "EVALUATING")
        self.assertAlmostEqual(countdown, 2.0, places=2)
        self.assertTrue(trans)
        self.assertTrue(dc.should_infer())

    def test_continuous_mode_zero_sleep_duration(self):
        """Setting sleep_duration=0.0 locks system into continuous evaluation."""
        dc = DutyCycleManager(eval_duration=2.0, sleep_duration=0.0, enabled=True)
        dc.reset(now=0.0)

        # Over 50 simulated seconds, phase is ALWAYS EVALUATING, should_infer ALWAYS True
        for t in range(0, 50):
            phase, countdown, trans = dc.update(now=float(t))
            self.assertEqual(phase, "EVALUATING")
            self.assertEqual(countdown, 0.0)
            self.assertFalse(trans)
            self.assertTrue(dc.should_infer())

    def test_continuous_mode_disabled_duty_cycle(self):
        """Setting enabled=False keeps system in continuous mode even if sleep_duration > 0."""
        dc = DutyCycleManager(eval_duration=2.0, sleep_duration=10.0, enabled=False)
        dc.reset(now=0.0)

        phase, countdown, trans = dc.update(now=5.0)
        self.assertEqual(phase, "EVALUATING")
        self.assertEqual(countdown, 0.0)
        self.assertFalse(trans)
        self.assertTrue(dc.should_infer())

    def test_dynamic_transition_to_continuous_mode_during_sleep(self):
        """Switching sleep_duration to 0.0 while in SLEEPING phase forces immediate transition to EVALUATING."""
        dc = DutyCycleManager(eval_duration=2.0, sleep_duration=10.0, enabled=True)
        dc.reset(phase="EVALUATING", now=0.0)

        # Transition to SLEEPING at t=2.0
        phase, _, _ = dc.update(now=2.0)
        self.assertEqual(phase, "SLEEPING")
        self.assertFalse(dc.should_infer())

        # Dynamically set sleep_duration to 0.0 at t=3.0
        dc.update_durations(sleep_duration=0.0)
        phase, countdown, trans = dc.update(now=3.1)
        self.assertEqual(phase, "EVALUATING", "Must immediately restore EVALUATING state")
        self.assertEqual(countdown, 0.0)
        self.assertTrue(dc.should_infer())

    def test_eval_duration_lower_bound_clamp(self):
        """eval_duration cannot be set below 0.1s."""
        dc = DutyCycleManager(eval_duration=-5.0, sleep_duration=-1.0)
        self.assertGreaterEqual(dc.eval_duration, 0.1)
        self.assertGreaterEqual(dc.sleep_duration, 0.0)

        dc.update_durations(eval_duration=0.0)
        self.assertGreaterEqual(dc.eval_duration, 0.1)


class TestCrossAlgorithmicIntegrationInvariants(unittest.TestCase):
    """
    Tests cross-algorithm pipeline interactions:
    SpatialROIFilter -> TemporalPresenceSmoother -> DutyCycleManager.
    """

    def test_roi_to_smoother_presence_pipeline(self):
        """
        Simulate an object moving outside ROI, entering ROI, remaining, then leaving.
        Verify end-to-end presence state pipeline.
        """
        roi = SpatialROIFilter(
            normalized_vertices=[(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)],
            enabled=True
        )
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=5, min_switch_interval=0.0)
        smoother.reset(state=False, now=0.0)

        w, h = 1000, 1000

        # Step 1: 5 frames outside ROI (person at x=100, y=100)
        for i in range(5):
            det = [{"xyxy": [80, 80, 120, 120], "conf": 0.9}]
            inside, _ = roi.filter_detections(det, w, h)
            st, ch = smoother.update(len(inside) > 0, now=float(i))
            self.assertFalse(st)

        # Step 2: Person enters ROI (x=500, y=500)
        # Frames 1 and 2 in ROI: smoother still OFF
        for i in range(5, 7):
            det = [{"xyxy": [480, 480, 520, 520], "conf": 0.9}]
            inside, _ = roi.filter_detections(det, w, h)
            self.assertEqual(len(inside), 1)
            st, ch = smoother.update(len(inside) > 0, now=float(i))
            self.assertFalse(st)

        # Frame 3 in ROI (step 7): smoother turns ON
        det = [{"xyxy": [480, 480, 520, 520], "conf": 0.9}]
        inside, _ = roi.filter_detections(det, w, h)
        st, ch = smoother.update(len(inside) > 0, now=7.0)
        self.assertTrue(st)
        self.assertTrue(ch)

        # Step 3: Person leaves ROI
        # Frames 1-4 outside: smoother stays ON
        for i in range(8, 12):
            det = [{"xyxy": [80, 80, 120, 120], "conf": 0.9}]
            inside, _ = roi.filter_detections(det, w, h)
            self.assertEqual(len(inside), 0)
            st, ch = smoother.update(len(inside) > 0, now=float(i))
            self.assertTrue(st)

        # Frame 5 outside (step 12): smoother turns OFF
        det = [{"xyxy": [80, 80, 120, 120], "conf": 0.9}]
        inside, _ = roi.filter_detections(det, w, h)
        st, ch = smoother.update(len(inside) > 0, now=12.0)
        self.assertFalse(st)
        self.assertTrue(ch)


if __name__ == "__main__":
    unittest.main()

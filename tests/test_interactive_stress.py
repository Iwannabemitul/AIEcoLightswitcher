"""
tests/test_interactive_stress.py - Adversarial Stress & Interactive Controls Verification Suite
Milenger 2 Verification Track (Challenger: challenger_m2_1)

Stress-tests:
1. Rapid Slider Oscillations (100 updates each: conf 0.01-1.00, eval 0.5-10.0s, sleep 0.0-30.0s)
2. Continuous Evaluation Mode (sleep = 0.0s disables duty cycle without stalling)
3. Malformed ROI Input Injection (empty, odd counts, negatives, >1.0, non-numeric, 1-pt, 2-pt, adversarial)
4. Rapid Preset Switching ('Classroom', 'Study Lab', 'Eco Maximum' 100 iterations)
5. Spatial Zone Preset Cycling ('Full Frame', 'Center Focus', 'Entrance / Doorway')
6. Live Multi-Threaded Concurrency Stress (active inference loop + control hammering)
"""

import os
import sys
import time
import unittest
from typing import Dict, Any, List, Tuple

# Ensure project root and 'Model Demo' are in path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODEL_DEMO = os.path.join(_PROJECT_ROOT, "Model Demo")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _MODEL_DEMO not in sys.path:
    sys.path.insert(0, _MODEL_DEMO)

from eco_engine import (
    AdvancedDetectionEngine,
    EngineConfig,
    MockHardwareController,
    MockVideoSource,
)
from main import AIEcoLightSwitcher, parse_and_validate_roi


class TestSliderOscillations(unittest.TestCase):
    """Stress-tests rapid interactive slider adjustments on the active GUI and Engine."""

    def setUp(self):
        self.mock_hw = MockHardwareController()
        self.mock_vs = MockVideoSource("PERSON_IN_ROI")
        self.engine = AdvancedDetectionEngine(
            video_stream=self.mock_vs,
            hardware=self.mock_hw
        )
        self.app = AIEcoLightSwitcher(engine=self.engine, headless=True)
        self.app.update()

    def tearDown(self):
        self.app.on_closing()

    def test_rapid_confidence_slider_oscillations(self):
        """Oscillates confidence threshold rapidly across 100 steps from 0.01 to 1.00."""
        iterations = 100
        t0 = time.time()
        for i in range(iterations):
            val = round(0.01 + (i % iterations) * (0.99 / (iterations - 1)), 2)
            self.app.on_conf_slider_change(val)
            self.assertEqual(
                round(self.engine.config.conf_threshold, 2),
                val,
                f"Confidence threshold mismatch at iteration {i}"
            )
            # Verify UI label reflects percentage
            lbl_text = self.app.lbl_conf_val.cget("text")
            self.assertIn(f"{int(val * 100)}%", lbl_text)
            self.assertIn(f"{val:.2f}", lbl_text)
        elapsed = time.time() - t0
        throughput = iterations / elapsed
        print(f"\n[STRESS] 100 Conf Slider Updates: {elapsed:.4f}s ({throughput:.1f} ops/sec)")
        self.assertGreater(throughput, 200.0, "Slider update throughput is unexpectedly low")

    def test_rapid_eval_duration_slider_oscillations(self):
        """Oscillates evaluation duration slider across 100 steps from 0.5s to 10.0s."""
        iterations = 100
        t0 = time.time()
        for i in range(iterations):
            val = round(0.5 + (i % iterations) * (9.5 / (iterations - 1)), 1)
            self.app.on_eval_slider_change(val)
            self.assertEqual(
                round(self.engine.config.eval_duration, 1),
                val,
                f"Eval duration mismatch at iteration {i}"
            )
            lbl_text = self.app.lbl_eval_val.cget("text")
            self.assertEqual(lbl_text, f"{val:.1f}s")
        elapsed = time.time() - t0
        throughput = iterations / elapsed
        print(f"[STRESS] 100 Eval Duration Updates: {elapsed:.4f}s ({throughput:.1f} ops/sec)")
        self.assertGreater(throughput, 200.0)

    def test_rapid_sleep_duration_oscillations_and_continuous_mode(self):
        """Oscillates sleep duration from 0.0s to 30.0s and verifies continuous mode switching."""
        iterations = 100
        t0 = time.time()
        continuous_count = 0
        duty_cycle_count = 0

        for i in range(iterations):
            val = round(0.0 + (i % iterations) * (30.0 / (iterations - 1)), 1)
            self.app.on_sleep_slider_change(val)
            if val <= 0.0:
                continuous_count += 1
                self.assertFalse(
                    self.engine.config.duty_cycle_enabled,
                    f"Duty cycle should be disabled at sleep=0.0 (iter {i})"
                )
                self.assertEqual(self.engine.config.sleep_duration, 0.0)
                self.assertIn("Continuous", self.app.lbl_sleep_val.cget("text"))
            else:
                duty_cycle_count += 1
                self.assertTrue(
                    self.engine.config.duty_cycle_enabled,
                    f"Duty cycle should be enabled at sleep={val} (iter {i})"
                )
                self.assertEqual(round(self.engine.config.sleep_duration, 1), val)
                self.assertEqual(self.app.lbl_sleep_val.cget("text"), f"{val:.1f}s")

        elapsed = time.time() - t0
        throughput = iterations / elapsed
        print(f"[STRESS] 100 Sleep Duration Updates: {elapsed:.4f}s ({throughput:.1f} ops/sec)")
        print(f"         Continuous transitions: {continuous_count}, Duty-cycle transitions: {duty_cycle_count}")
        self.assertGreater(throughput, 200.0)

    def test_continuous_mode_execution_semantics(self):
        """Verifies that setting sleep=0.0s keeps the duty cycle engine continuously evaluating."""
        self.app.on_sleep_slider_change(0.0)
        self.assertFalse(self.engine.config.duty_cycle_enabled)
        self.assertEqual(self.engine.config.sleep_duration, 0.0)

        # DutyCycleManager should always report state as EVALUATING and should_infer() True
        for _ in range(10):
            state, countdown, transitioned = self.engine.duty_cycle.update()
            self.assertEqual(state, "EVALUATING")
            self.assertEqual(countdown, 0.0)
            self.assertFalse(transitioned)
            self.assertTrue(self.engine.duty_cycle.should_infer())

        # Switch back to duty cycle mode (e.g. 5.0s)
        self.app.on_sleep_slider_change(5.0)
        self.assertTrue(self.engine.config.duty_cycle_enabled)
        self.assertEqual(self.engine.config.sleep_duration, 5.0)


class TestPresetSwitching(unittest.TestCase):
    """Stress-tests rapid preset switching and verifies 100% synchronization."""

    def setUp(self):
        self.mock_hw = MockHardwareController()
        self.mock_vs = MockVideoSource("PERSON_IN_ROI")
        self.engine = AdvancedDetectionEngine(
            video_stream=self.mock_vs,
            hardware=self.mock_hw
        )
        self.app = AIEcoLightSwitcher(engine=self.engine, headless=True)
        self.app.update()

    def tearDown(self):
        self.app.on_closing()

    def test_rapid_preset_switching_synchronization(self):
        """Switches between Classroom, Study Lab, and Eco Maximum 100 times, asserting sync."""
        presets = ["Classroom", "Study Lab", "Eco Maximum"]
        preset_configs = {
            "Classroom":   {"conf": 0.50, "eval": 2.0, "sleep": 5.0},
            "Study Lab":   {"conf": 0.35, "eval": 3.0, "sleep": 10.0},
            "Eco Maximum": {"conf": 0.60, "eval": 1.0, "sleep": 15.0},
        }

        iterations = 100
        t0 = time.time()
        for i in range(iterations):
            pname = presets[i % 3]
            self.app.apply_preset(pname)
            expected = preset_configs[pname]

            # 1. Engine config sync
            cfg = self.engine.config
            self.assertEqual(round(cfg.conf_threshold, 2), expected["conf"])
            self.assertEqual(round(cfg.eval_duration, 1), expected["eval"])
            self.assertEqual(round(cfg.sleep_duration, 1), expected["sleep"])
            self.assertTrue(cfg.duty_cycle_enabled)

            # 2. Sliders sync
            self.assertEqual(round(self.app.slider_conf.get(), 2), expected["conf"])
            self.assertEqual(round(self.app.slider_eval.get(), 1), expected["eval"])
            self.assertEqual(round(self.app.slider_sleep.get(), 1), expected["sleep"])

            # 3. Label text sync
            self.assertIn(f"{int(expected['conf'] * 100)}%", self.app.lbl_conf_val.cget("text"))
            self.assertEqual(self.app.lbl_eval_val.cget("text"), f"{expected['eval']:.1f}s")
            self.assertEqual(self.app.lbl_sleep_val.cget("text"), f"{expected['sleep']:.1f}s")

            # 4. Stat tile sync
            self.assertIn(f"{int(expected['conf'] * 100)}%", self.app.lbl_tile1_val.cget("text"))

        elapsed = time.time() - t0
        throughput = iterations / elapsed
        print(f"\n[STRESS] 100 Preset Switches: {elapsed:.4f}s ({throughput:.1f} switches/sec)")
        self.assertGreater(throughput, 25.0)

    def test_invalid_preset_graceful_handling(self):
        """Applying an unknown preset should not crash or corrupt existing state."""
        # Save baseline config
        baseline_conf = self.engine.config.conf_threshold
        baseline_eval = self.engine.config.eval_duration
        baseline_sleep = self.engine.config.sleep_duration

        # Attempt to apply invalid presets
        for bad_name in ["InvalidPreset", "", "   ", "UNKNOWN_ZONE", "NULL"]:
            self.app.apply_preset(bad_name)
            # Config must remain intact
            self.assertEqual(self.engine.config.conf_threshold, baseline_conf)
            self.assertEqual(self.engine.config.eval_duration, baseline_eval)
            self.assertEqual(self.engine.config.sleep_duration, baseline_sleep)


class TestMalformedROIInputInjection(unittest.TestCase):
    """Exhaustively stress-tests coordinate validation and malformed input resilience."""

    def setUp(self):
        self.mock_hw = MockHardwareController()
        self.mock_vs = MockVideoSource("PERSON_IN_ROI")
        self.engine = AdvancedDetectionEngine(
            video_stream=self.mock_vs,
            hardware=self.mock_hw
        )
        self.app = AIEcoLightSwitcher(engine=self.engine, headless=True)
        self.app.update()

    def tearDown(self):
        self.app.on_closing()

    def test_parse_and_validate_roi_malformed_inputs(self):
        """Tests pure function parse_and_validate_roi against all adversarial inputs."""
        malformed_cases = [
            ("", "empty string"),
            ("   ", "whitespace only"),
            ("0.5", "single float"),
            ("(0.5, 0.5)", "single 2D point (2 floats)"),
            ("(0.1, 0.1), (0.9, 0.9)", "two 2D points (4 floats)"),
            ("0.1, 0.2, 0.3, 0.4, 0.5", "odd number of floats (5 floats)"),
            ("0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7", "odd number of floats (7 floats)"),
            ("(-0.1, 0.2), (0.3, 0.4), (0.5, 0.6)", "negative float u"),
            ("(0.1, -0.5), (0.3, 0.4), (0.5, 0.6)", "negative float v"),
            ("(1.5, 0.2), (0.3, 0.4), (0.5, 0.6)", "float u > 1.0"),
            ("(0.1, 2.0), (0.3, 0.4), (0.5, 0.6)", "float v > 1.0"),
            ("hello world abc def", "non-numeric string"),
            ("<script>alert('xss')</script>", "HTML/script injection"),
            ("; DROP TABLE roi; --", "SQL injection"),
            ("NaN, Inf, -Inf, NaN, Inf, -Inf", "NaN/Inf strings without valid floats"),
            ("💥🔥🚀✨🎉🎃", "Unicode emojis"),
            ("((((((((", "Only punctuation"),
            ("10.0, 20.0, 30.0, 40.0, 50.0, 60.0", "All coordinates drastically > 1.0"),
            ("-1.0, -1.0, -2.0, -2.0, -3.0, -3.0", "All coordinates negative"),
        ]

        for text, desc in malformed_cases:
            valid, result = parse_and_validate_roi(text)
            self.assertFalse(
                valid,
                f"Expected rejection for malformed input '{desc}': text={text!r}"
            )
            self.assertIsInstance(
                result, str,
                f"Expected descriptive error string for '{desc}'"
            )
            self.assertGreater(len(result), 0)

    def test_parse_and_validate_roi_valid_inputs(self):
        """Verifies valid boundary coordinate strings parse correctly."""
        valid_cases = [
            # Minimal 3 vertices
            ("(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)", 3),
            # Standard 4-point rectangle
            ("(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)", 4),
            # Full frame normalized extremes
            ("0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0", 4),
            # Irregular 5-point polygon
            ("(0.2, 0.1), (0.8, 0.1), (0.9, 0.6), (0.5, 0.95), (0.1, 0.6)", 5),
        ]

        for text, vertex_count in valid_cases:
            valid, result = parse_and_validate_roi(text)
            self.assertTrue(valid, f"Failed to parse valid input: {text}")
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), vertex_count)
            for u, v in result:
                self.assertTrue(0.0 <= u <= 1.0, f"u coordinate {u} out of [0, 1]")
                self.assertTrue(0.0 <= v <= 1.0, f"v coordinate {v} out of [0, 1]")

    def test_ui_custom_roi_injection_resilience(self):
        """Injects malformed inputs via UI Entry widget and asserts UI & Engine safety."""
        # Set a known valid baseline ROI
        baseline_poly = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        self.engine.update_config(roi_polygon=baseline_poly, roi_enabled=True)

        bad_inputs = [
            "",
            "   ",
            "0.1, 0.2, 0.3, 0.4, 0.5",
            "(-0.5, 0.2), (0.5, 0.2), (0.5, 0.8)",
            "(1.5, 0.2), (0.5, 0.2), (0.5, 0.8)",
            "malformed non-numeric text",
            "(0.5, 0.5)",
            "(0.1, 0.1), (0.9, 0.9)",
        ]

        for bad in bad_inputs:
            self.app.entry_roi_coords.delete(0, "end")
            self.app.entry_roi_coords.insert(0, bad)
            self.app.apply_custom_roi()

            # Verify UI feedback label shows 'Error'
            fb = self.app.lbl_roi_feedback.cget("text")
            self.assertTrue(
                fb.startswith("Error:"),
                f"Expected 'Error:' prefix in feedback for input {bad!r}, got {fb!r}"
            )
            # Verify engine polygon was NOT corrupted
            self.assertEqual(
                self.engine.config.roi_polygon,
                baseline_poly,
                f"Engine ROI polygon was corrupted by invalid input {bad!r}"
            )


class TestROIPresetCycling(unittest.TestCase):
    """Stress-tests cycling through spatial zone presets."""

    def setUp(self):
        self.mock_hw = MockHardwareController()
        self.mock_vs = MockVideoSource("PERSON_IN_ROI")
        self.engine = AdvancedDetectionEngine(
            video_stream=self.mock_vs,
            hardware=self.mock_hw
        )
        self.app = AIEcoLightSwitcher(engine=self.engine, headless=True)
        self.app.update()

    def tearDown(self):
        self.app.on_closing()

    def test_rapid_roi_preset_cycling(self):
        """Cycles through Full Frame, Center Focus, and Entrance / Doorway presets 30 times."""
        presets = ["Full Frame", "Center Focus", "Entrance / Doorway"]
        expected_counts = {
            "Full Frame": 4,
            "Center Focus": 4,
            "Entrance / Doorway": 4,
        }

        iterations = 30
        t0 = time.time()
        for i in range(iterations):
            pname = presets[i % 3]
            self.app.apply_roi_preset(pname)

            self.assertTrue(self.engine.config.roi_enabled)
            self.assertEqual(len(self.engine.config.roi_polygon), expected_counts[pname])
            self.assertEqual(self.app.switch_roi.get(), 1)
            self.assertIn("ACTIVE", self.app.lbl_roi_status.cget("text"))
            self.assertIn(pname, self.app.lbl_roi_feedback.cget("text"))

        elapsed = time.time() - t0
        throughput = iterations / elapsed
        print(f"\n[STRESS] 30 ROI Preset Cycles: {elapsed:.4f}s ({throughput:.1f} cycles/sec)")
        self.assertGreater(throughput, 50.0)


class TestLiveEngineConcurrencyStress(unittest.TestCase):
    """Stress-tests controls while the detection engine background worker thread is running."""

    def setUp(self):
        self.mock_hw = MockHardwareController()
        self.mock_vs = MockVideoSource("PERSON_IN_ROI")
        self.engine = AdvancedDetectionEngine(
            video_stream=self.mock_vs,
            hardware=self.mock_hw
        )
        self.engine.start()
        self.app = AIEcoLightSwitcher(engine=self.engine, headless=True)
        self.app.update()

    def tearDown(self):
        self.app.on_closing()

    def test_concurrent_control_hammering_during_inference(self):
        """Hammers sliders, presets, and ROI controls while background inference loop is active."""
        # Allow worker thread to produce initial frames
        time.sleep(0.1)

        # Hammer controls concurrently
        for i in range(50):
            # 1. Oscillate conf
            conf_val = round(0.10 + (i % 50) * (0.80 / 49.0), 2)
            self.app.on_conf_slider_change(conf_val)

            # 2. Toggle sleep
            sleep_val = 0.0 if (i % 5 == 0) else round(1.0 + (i % 10) * 1.5, 1)
            self.app.on_sleep_slider_change(sleep_val)

            # 3. Apply preset occasionally
            if i % 10 == 0:
                pname = ["Classroom", "Study Lab", "Eco Maximum"][(i // 10) % 3]
                self.app.apply_preset(pname)

            # 4. Cycle ROI occasionally
            if i % 15 == 0:
                roi_name = ["Full Frame", "Center Focus", "Entrance / Doorway"][(i // 15) % 3]
                self.app.apply_roi_preset(roi_name)

            # 5. Let UI event loop tick
            self.app.update()

        # Check telemetry retrieval under high concurrency
        frame, telemetry = self.engine.get_latest_frame_and_telemetry()
        self.assertIsNotNone(telemetry)
        self.assertIn("fps", telemetry)
        self.assertIn("latency_ms", telemetry)
        self.assertIn("raw_occupants", telemetry)
        self.assertIn("cycle_state", telemetry)

        # Verify event logger ring buffer did not exceed max capacity (100)
        entries = self.app.event_logger.get_entries()
        self.assertLessEqual(len(entries), 100)
        self.assertGreater(len(entries), 10)
        print(f"\n[STRESS] Concurrent stress completed successfully. Event buffer: {len(entries)}/100 entries.")


if __name__ == "__main__":
    unittest.main(verbosity=2)

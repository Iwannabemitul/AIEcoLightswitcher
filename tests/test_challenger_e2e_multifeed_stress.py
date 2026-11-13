"""
tests/test_challenger_e2e_multifeed_stress.py - Adversarial Stress & Verification Harness
for Multi-Feed Test Suite, Rapid Disconnect/Reconnect Resilience, and Multi-Pin Routing (Pins 2-19).

Authored by: challenger_e2e_2 (Empirical Challenger)
"""

import sys
import os
import time
import math
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_multifeed_suite import (
    FeedStatus,
    FeedConfig,
    MockMultiPinHardwareController,
    MockMultiFeedSource,
    StreamWorker,
    SingleFrameBuffer,
    MultiFeedManager,
    compute_grid_geometry,
    compute_placeholders,
    calculate_backoff,
)


class TestMultiPinRoutingPins2To19(unittest.TestCase):
    """
    Adversarial verification of Multi-Pin Serial Routing across Pins 2 through 19.
    """

    def setUp(self):
        self.hw = MockMultiPinHardwareController(auto_connect=True, min_switch_interval=0.0)

    def test_all_18_pins_independent_actuation(self):
        """Verify each of Pins 2 through 19 can be set HIGH and LOW independently."""
        all_pins = list(range(2, 20))  # 2 to 19 inclusive (18 pins)
        self.assertEqual(len(all_pins), 18)

        # Initially all pins are False
        for p in all_pins:
            self.assertFalse(self.hw.get_pin(p), f"Pin {p} should initially be False")

        # Sequentially turn each pin ON and verify independence
        for i, pin in enumerate(all_pins):
            success = self.hw.set_pin(pin, True)
            self.assertTrue(success, f"Failed to set pin {pin} HIGH")
            self.assertTrue(self.hw.get_pin(pin), f"Pin {pin} did not report HIGH")

            # Check that preceding pins are still HIGH and subsequent pins are still LOW
            for prev_pin in all_pins[:i]:
                self.assertTrue(self.hw.get_pin(prev_pin), f"Prior pin {prev_pin} was disturbed when setting pin {pin}")
            for next_pin in all_pins[i + 1:]:
                self.assertFalse(self.hw.get_pin(next_pin), f"Future pin {next_pin} was disturbed when setting pin {pin}")

        # Verify command log contains exactly 18 commands
        cmds = self.hw.get_raw_commands()
        self.assertEqual(len(cmds), 18)
        for pin in all_pins:
            self.assertIn(f"SET {pin} 1\n", cmds)

        # Sequentially turn each pin OFF in reverse order
        for i, pin in enumerate(reversed(all_pins)):
            success = self.hw.set_pin(pin, False)
            self.assertTrue(success, f"Failed to set pin {pin} LOW")
            self.assertFalse(self.hw.get_pin(pin), f"Pin {pin} did not report LOW")

        self.assertEqual(len(self.hw.get_raw_commands()), 36)
        for pin in all_pins:
            self.assertFalse(self.hw.get_pin(pin))

    def test_reserved_and_boundary_pin_rejection(self):
        """Verify strict rejection of UART RX/TX (pins 0, 1), negative, and out-of-range pins."""
        invalid_pins = [-100, -1, 0, 1, 20, 21, 50, 100, 255]
        for p in invalid_pins:
            self.assertFalse(self.hw.set_pin(p, True), f"Invalid pin {p} was unexpectedly accepted")
            self.assertFalse(self.hw.set_pin(p, False), f"Invalid pin {p} was unexpectedly accepted")
            self.assertFalse(self.hw.get_pin(p), f"Invalid pin {p} returned True")

        # Command log must be empty
        self.assertEqual(len(self.hw.get_command_log()), 0)

    def test_delta_updates_across_all_18_pins(self):
        """Verify that 100 redundant writes across all 18 pins produce zero redundant UART traffic."""
        all_pins = list(range(2, 20))

        # Initial turn ON: 18 writes -> 18 commands
        for p in all_pins:
            self.hw.set_pin(p, True)
        self.assertEqual(len(self.hw.get_command_log()), 18)

        # 100 redundant writes to each pin: 1800 calls -> 0 new commands
        for _ in range(100):
            for p in all_pins:
                self.hw.set_pin(p, True)
        self.assertEqual(len(self.hw.get_command_log()), 18)

        # Turn OFF pin 10 only -> exactly 1 new command
        self.hw.set_pin(10, False)
        self.assertEqual(len(self.hw.get_command_log()), 19)
        self.assertEqual(self.hw.get_raw_commands()[-1], "SET 10 0\n")

    def test_per_pin_rate_limiting_isolation_pins_2_to_19(self):
        """Verify that rate limiting on Pin A never blocks or interferes with Pin B."""
        hw = MockMultiPinHardwareController(min_switch_interval=1.0)
        t_base = 5000.0

        # Turn Pin 2 ON at t=5000.0
        self.assertTrue(hw.set_pin(2, True, now=t_base))
        # Turn Pin 2 OFF at t=5000.1 (clamped, fails)
        self.assertFalse(hw.set_pin(2, False, now=t_base + 0.1))

        # Simultaneously turn Pins 3..19 ON at t=5000.1 (each must succeed!)
        for p in range(3, 20):
            self.assertTrue(hw.set_pin(p, True, now=t_base + 0.1), f"Pin {p} blocked by Pin 2 clamp")

        # At t=5001.05, Pin 2 OFF now succeeds
        self.assertTrue(hw.set_pin(2, False, now=t_base + 1.05))
        self.assertFalse(hw.get_pin(2))

    def test_concurrent_hammering_all_18_pins(self):
        """Stress 18 concurrent threads hammering pins 2..19 simultaneously."""
        all_pins = list(range(2, 20))
        iterations = 100
        errors = []

        def worker(pin: int):
            try:
                for i in range(iterations):
                    desired_state = (i % 2 == 0)
                    ret = self.hw.set_pin(pin, desired_state, force=True)
                    if not ret:
                        errors.append(f"Pin {pin} set_pin failed on iter {i}")
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(f"Pin {pin} threw exception: {e}")

        threads = [threading.Thread(target=worker, args=(p,)) for p in all_pins]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(errors), 0, f"Concurrent pin stress errors: {errors}")
        # Verify all pins have valid boolean states
        for p in all_pins:
            self.assertIsInstance(self.hw.get_pin(p), bool)

    def test_force_sync_full_reconstruction_all_18_pins(self):
        """Verify force_sync accurately resynchronizes physical firmware pin states after reconnect."""
        all_pins = list(range(2, 20))
        # Set arbitrary pattern: even pins ON, odd pins OFF
        for p in all_pins:
            self.hw.set_pin(p, (p % 2 == 0))

        # Simulate disconnect and reconnect
        self.hw.simulate_disconnect()
        self.assertFalse(self.hw.is_connected)
        self.hw.simulate_reconnect()
        self.assertTrue(self.hw.is_connected)

        self.hw.reset_log()
        self.assertTrue(self.hw.force_sync())

        raw_cmds = self.hw.get_raw_commands()
        self.assertEqual(len(raw_cmds), 18)
        for p in all_pins:
            expected_val = 1 if (p % 2 == 0) else 0
            self.assertIn(f"SET {p} {expected_val}\n", raw_cmds)


class TestRapidDisconnectReconnectStress(unittest.TestCase):
    """
    Adversarial verification of rapid disconnect/reconnect cycles under high load.
    """

    def test_single_feed_rapid_disconnect_burst_50_cycles(self):
        """Stress 50 rapid disconnect/reconnect cycles on a single StreamWorker."""
        src = MockMultiFeedSource("burst_cam", default_scenario="PERSON_IN_ROI")
        worker = StreamWorker(source=src, feed_id="burst_cam", reconnect_interval=0.5, read_timeout=0.5)
        worker.start()

        # Let it establish initial online state
        t_deadline = time.time() + 1.0
        while time.time() < t_deadline and worker.status != FeedStatus.ONLINE:
            time.sleep(0.01)
        self.assertEqual(worker.status, FeedStatus.ONLINE)

        # Rapidly toggle disconnect/reconnect 50 times
        for i in range(50):
            src.simulate_disconnect()
            time.sleep(0.005)
            src.simulate_reconnect()
            time.sleep(0.005)

        # Worker thread must still be alive and not deadlocked
        self.assertTrue(worker.is_alive)

        # Allow stream to settle back to ONLINE
        settle_deadline = time.time() + 2.0
        while time.time() < settle_deadline and worker.status != FeedStatus.ONLINE:
            time.sleep(0.02)

        self.assertEqual(worker.status, FeedStatus.ONLINE)
        frame, telem = worker.get_latest_frame()
        self.assertIsNotNone(frame)
        self.assertGreater(telem["frame_id"], 0)
        worker.stop()
        self.assertFalse(worker.is_alive)

    def test_multi_feed_asynchronous_flapping_8_streams(self):
        """Simulate 8 concurrent streams experiencing randomized asynchronous network flapping."""
        sources = [MockMultiFeedSource(f"flap_{i}") for i in range(8)]
        workers = [
            StreamWorker(source=sources[i], feed_id=f"flap_{i}", reconnect_interval=0.5, read_timeout=0.5)
            for i in range(8)
        ]

        for w in workers:
            w.start()

        time.sleep(0.1)

        # Threaded flapper to disconnect and reconnect randomly
        stop_flapping = threading.Event()
        flap_errors = []

        def flapper(idx: int):
            src = sources[idx]
            try:
                cycles = 0
                while not stop_flapping.is_set() and cycles < 20:
                    src.simulate_disconnect()
                    time.sleep(0.01)
                    src.simulate_reconnect()
                    time.sleep(0.01)
                    cycles += 1
            except Exception as e:
                flap_errors.append(f"Worker {idx} flapper error: {e}")

        flapper_threads = [threading.Thread(target=flapper, args=(i,)) for i in range(8)]
        for t in flapper_threads:
            t.start()

        # Concurrently query frames from all 8 workers during flapping
        t_query_end = time.time() + 0.5
        query_count = 0
        while time.time() < t_query_end:
            for w in workers:
                f, t = w.get_latest_frame()
                self.assertIsInstance(t, dict)
                self.assertIn("status", t)
                query_count += 1
            time.sleep(0.01)

        stop_flapping.set()
        for t in flapper_threads:
            t.join(timeout=2.0)

        self.assertEqual(len(flap_errors), 0, f"Flap errors: {flap_errors}")
        self.assertGreater(query_count, 50)

        # All workers must still be alive
        for i, w in enumerate(workers):
            self.assertTrue(w.is_alive, f"Worker {i} died during flapping")
            w.stop()
            self.assertFalse(w.is_alive)

    def test_single_frame_buffer_under_concurrent_put_and_get_race(self):
        """Stress SingleFrameBuffer with simultaneous high-frequency puts and gets."""
        buf = SingleFrameBuffer()
        stop_event = threading.Event()
        put_count = [0]
        get_count = [0]
        anomalies = []

        def writer():
            idx = 0
            while not stop_event.is_set():
                frame = np.full((32, 32, 3), idx % 256, dtype=np.uint8)
                fid = buf.put(frame, timestamp=time.time())
                put_count[0] += 1
                idx += 1
                time.sleep(0.0001)

        def reader():
            while not stop_event.is_set():
                ret, frame, ts, fid = buf.get()
                if ret:
                    get_count[0] += 1
                    if frame is None or frame.shape != (32, 32, 3):
                        anomalies.append("Corrupted frame retrieved")
                time.sleep(0.0001)

        w_thread = threading.Thread(target=writer)
        r_thread = threading.Thread(target=reader)

        w_thread.start()
        r_thread.start()

        time.sleep(0.3)
        stop_event.set()

        w_thread.join(timeout=1.0)
        r_thread.join(timeout=1.0)

        self.assertEqual(len(anomalies), 0)
        self.assertGreater(put_count[0], 100)
        self.assertGreater(get_count[0], 50)
        self.assertGreaterEqual(buf.drop_count, 0)


class TestDynamicMultiFeedSerialIntegration(unittest.TestCase):
    """
    Adversarial verification of MultiFeedManager integrating dynamic feeds and multi-pin serial.
    """

    def setUp(self):
        self.hw = MockMultiPinHardwareController()
        self.mgr = MultiFeedManager(hardware=self.hw)
        self.mgr.start()

    def tearDown(self):
        self.mgr.stop()

    def test_full_18_feed_to_18_pin_mapping(self):
        """Map 18 separate feeds to all 18 pins (Pins 2 to 19) and verify zero cross-talk."""
        pins = list(range(2, 20))
        for p in pins:
            fid = f"cam_pin_{p}"
            cfg = FeedConfig(
                feed_id=fid,
                name=f"Camera for Pin {p}",
                source=MockMultiFeedSource(fid),
                pin=p,
                enabled=True
            )
            self.assertTrue(self.mgr.add_feed(cfg))

        time.sleep(0.05)
        all_telems = self.mgr.get_all_feeds_telemetry()
        self.assertEqual(len(all_telems), 18)

        # Test individual actuation and pin state verification
        for p in pins:
            fid = f"cam_pin_{p}"
            self.hw.set_pin(p, True)
            self.assertTrue(self.hw.get_pin(p))
            telem = self.mgr.get_feed_telemetry(fid)
            self.assertEqual(telem["pin"], p)
            self.assertTrue(telem["pin_state"])

        # Test disabling feeds resets respective pins without disturbing others
        for p in pins[:9]:  # Disable first 9 feeds (pins 2 to 10)
            fid = f"cam_pin_{p}"
            self.mgr.disable_feed(fid)
            self.assertFalse(self.hw.get_pin(p), f"Pin {p} not reset upon disable")

        # Pins 11 to 19 must still be True!
        for p in pins[9:]:
            self.assertTrue(self.hw.get_pin(p), f"Pin {p} was unexpectedly altered")

        # Re-enable first 9 feeds
        for p in pins[:9]:
            fid = f"cam_pin_{p}"
            self.mgr.enable_feed(fid)
            self.assertTrue(self.mgr.get_feed_telemetry(fid)["enabled"])

    def test_dynamic_add_remove_stress_under_continuous_queries(self):
        """Dynamically add and remove feeds while querying telemetry in tight loop."""
        stop_event = threading.Event()
        query_errors = []

        def querier():
            while not stop_event.is_set():
                try:
                    telems = self.mgr.get_all_feeds_telemetry()
                    for fid, t in telems.items():
                        _ = t.get("status")
                        _ = t.get("pin")
                except Exception as e:
                    query_errors.append(f"Query error: {e}")
                time.sleep(0.002)

        q_thread = threading.Thread(target=querier)
        q_thread.start()

        # Add and remove feeds repeatedly
        for cycle in range(10):
            fid = f"temp_feed_{cycle}"
            cfg = FeedConfig(feed_id=fid, name=fid, source=MockMultiFeedSource(fid), pin=13)
            self.assertTrue(self.mgr.add_feed(cfg))
            time.sleep(0.01)
            self.assertTrue(self.mgr.remove_feed(fid))
            time.sleep(0.01)

        stop_event.set()
        q_thread.join(timeout=2.0)

        self.assertEqual(len(query_errors), 0, f"Errors during dynamic churn: {query_errors}")

    def test_duplicate_feed_id_rejection(self):
        """Verify that adding a feed with duplicate feed_id is cleanly rejected without overwriting."""
        cfg1 = FeedConfig(feed_id="same_id", name="Feed 1", source=0, pin=13)
        cfg2 = FeedConfig(feed_id="same_id", name="Feed 2 Duplicate", source=1, pin=12)

        self.assertTrue(self.mgr.add_feed(cfg1))
        self.assertFalse(self.mgr.add_feed(cfg2))  # Must reject!

        telem = self.mgr.get_feed_telemetry("same_id")
        self.assertEqual(telem["name"], "Feed 1")
        self.assertEqual(telem["pin"], 13)


class TestGridGeometryAndPlaceholderStress(unittest.TestCase):
    """
    Adversarial verification of responsive grid geometry and placeholder math invariants.
    """

    def test_invariants_across_large_feed_range(self):
        """Verify mathematical invariants for N = 0 up to N = 500."""
        for n in range(0, 501):
            rows, cols = compute_grid_geometry(n)
            placeholders = compute_placeholders(n)
            capacity = rows * cols

            # Invariant 1: Minimum 4 capacity
            self.assertGreaterEqual(capacity, 4, f"Capacity {capacity} < 4 for N={n}")
            # Invariant 2: Capacity is sufficient to hold all feeds
            self.assertGreaterEqual(capacity, n, f"Capacity {capacity} < N={n}")
            # Invariant 3: Rows and cols are at least 2
            self.assertGreaterEqual(rows, 2, f"Rows {rows} < 2 for N={n}")
            self.assertGreaterEqual(cols, 2, f"Cols {cols} < 2 for N={n}")
            # Invariant 4: Capacity == effective_n + placeholders
            effective_n = max(0, n)
            self.assertEqual(capacity, effective_n + placeholders, f"Capacity mismatch for N={n}")

    def test_backoff_monotonicity_and_clamp(self):
        """Verify exponential backoff calculation monotonicity and ceiling clamp."""
        prev = 0.0
        for attempt in range(0, 30):
            b = calculate_backoff(attempt, base=1.0, max_backoff=30.0)
            self.assertGreaterEqual(b, prev, f"Backoff decreased at attempt {attempt}")
            self.assertLessEqual(b, 30.0, f"Backoff exceeded max clamp at attempt {attempt}")
            prev = b
        self.assertEqual(calculate_backoff(100), 30.0)


if __name__ == "__main__":
    unittest.main()

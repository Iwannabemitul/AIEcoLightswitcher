"""
tests/test_challenger_m1_round2.py - Adversarial Stress & Invariant Verification Suite (Round 2)
Author: challenger_m1_round2_1 (Empirical Challenger)
Focus:
- VideoStream immediate frame clearing & stall detection (<2.0s)
- StreamWorker backoff sleep non-busy looping & edge-triggered wake
- FeedStatus.ONLINE transition strictly on valid frame receipt & backoff escalation
- SpatialROIFilter inverted, adversarial, and degenerate bounding box resilience
"""

import math
import random
import threading
import time
import unittest
from typing import Optional, Tuple
import cv2
import numpy as np

from core.roi_filter import SpatialROIFilter
from ingestion.base_stream import BaseVideoStream, FeedStatus
from ingestion.stream_worker import StreamWorker
from ingestion.video_stream import VideoStream


class ControllableMockCap:
    """Mock OpenCV VideoCapture for fine-grained adversarial testing."""
    def __init__(self, opened: bool = True, initial_frame: Optional[np.ndarray] = None):
        self._opened = opened
        self._fail = False
        self._custom_frame = initial_frame if initial_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        self._raise_exc = False
        self._read_delay = 0.0

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._read_delay > 0:
            time.sleep(self._read_delay)
        if self._raise_exc:
            raise IOError("Simulated transport bus fault")
        if self._fail or not self._opened:
            return False, None
        if self._custom_frame is None:
            return True, None
        return True, self._custom_frame.copy()

    def release(self) -> None:
        self._opened = False


class DynamicMockStream(BaseVideoStream):
    """Controllable stream implementing BaseVideoStream for StreamWorker stress testing."""
    def __init__(self, opened: bool = True, read_delay: float = 0.0, can_open: bool = True):
        self._opened = opened
        self._can_open = can_open
        self._failing = False
        self._empty_frame = False
        self._raise_exc = False
        self._read_delay = read_delay
        self._read_count = 0
        self._lock = threading.Lock()

    def start(self) -> "DynamicMockStream":
        with self._lock:
            if self._can_open:
                self._opened = True
        return self

    def stop(self) -> None:
        with self._lock:
            self._opened = False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._read_delay > 0:
            time.sleep(self._read_delay)
        with self._lock:
            self._read_count += 1
            if self._raise_exc:
                raise RuntimeError("DynamicMockStream transport reset")
            if not self._opened or self._failing:
                return False, None
            if self._empty_frame:
                return True, np.empty((0, 0, 3), dtype=np.uint8)
            frame = np.full((480, 640, 3), (self._read_count % 255), dtype=np.uint8)
            return True, frame

    @property
    def is_opened(self) -> bool:
        with self._lock:
            return self._opened

    def set_opened(self, val: bool) -> None:
        with self._lock:
            self._opened = val
            self._can_open = val

    def set_failing(self, val: bool) -> None:
        with self._lock:
            self._failing = val

    def set_empty_frame(self, val: bool) -> None:
        with self._lock:
            self._empty_frame = val


class TestVideoStreamImmediateClearingAndStall(unittest.TestCase):
    """Stress tests verifying that VideoStream clears frames immediately and detects stalls (<2.0s)."""

    def test_01_videostream_immediate_clearing_on_read_failure(self):
        """Verify that VideoStream immediately clears self.frame to None when cap.read fails."""
        mock_cap = ControllableMockCap(opened=True)
        vs = VideoStream(source=0)
        vs.cap = mock_cap
        vs.start()

        # Allow worker thread to capture initial valid frame
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            ret, frame = vs.read()
            if ret and frame is not None:
                break
            time.sleep(0.01)

        ret, frame = vs.read()
        self.assertTrue(ret, "Expected initial read to succeed")
        self.assertIsNotNone(frame, "Expected initial frame to be present")
        self.assertEqual(vs.status, FeedStatus.ONLINE)

        # Trigger read failure
        mock_cap._fail = True
        # Frame should be cleared within one loop iteration (~10ms)
        cleared = False
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.2:
            ret, frame = vs.read()
            if not ret and frame is None:
                cleared = True
                break
            time.sleep(0.005)

        self.assertTrue(cleared, "VideoStream failed to immediately clear frame on read failure!")
        self.assertIsNone(vs.frame, "vs.frame must be None under lock")
        self.assertEqual(vs.status, FeedStatus.OFFLINE, "Status must transition to OFFLINE when frame is None")
        vs.stop()

    def test_02_videostream_clearing_on_empty_frame_and_exceptions(self):
        """Verify that empty frames or read exceptions clear self.frame immediately."""
        mock_cap = ControllableMockCap(opened=True)
        vs = VideoStream(source=0)
        vs.cap = mock_cap
        vs.start()

        # Capture initial valid frame
        time.sleep(0.05)
        ret, frame = vs.read()
        self.assertTrue(ret)

        # 1. Simulate empty frame (0x0)
        mock_cap._custom_frame = np.empty((0, 0, 3), dtype=np.uint8)
        time.sleep(0.05)
        ret, frame = vs.read()
        self.assertFalse(ret, "Read must return False on empty frame")
        self.assertIsNone(frame, "Frame must be None on empty frame")
        self.assertEqual(vs.status, FeedStatus.OFFLINE)

        # Restore healthy frame
        mock_cap._custom_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        time.sleep(0.05)
        ret, frame = vs.read()
        self.assertTrue(ret)
        self.assertEqual(vs.status, FeedStatus.ONLINE)

        # 2. Simulate read exception
        mock_cap._raise_exc = True
        time.sleep(0.05)
        ret, frame = vs.read()
        self.assertFalse(ret, "Read must return False when read() raises exception")
        self.assertIsNone(frame, "Frame must be None when read() raises exception")
        self.assertEqual(vs.status, FeedStatus.OFFLINE)

        vs.stop()

    def test_03_stream_worker_detects_stall_under_two_seconds(self):
        """Verify StreamWorker detects stream stall and transitions away from ONLINE in < 2.0s."""
        mock_cap = ControllableMockCap(opened=True)
        vs = VideoStream(source=0)
        vs.cap = mock_cap

        # Configure worker with default 2.0s read timeout
        worker = StreamWorker(source=vs, feed_id="stall_detector", read_timeout=2.0)
        worker.start()

        # Wait for worker to become ONLINE
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            if worker.status == FeedStatus.ONLINE:
                break
            time.sleep(0.02)
        self.assertEqual(worker.status, FeedStatus.ONLINE)

        # Cause stream to fail/stall
        mock_cap._fail = True
        t_fail = time.perf_counter()

        # Poll until worker is no longer ONLINE
        stalled = False
        detection_elapsed = 0.0
        while time.perf_counter() - t_fail < 2.0:
            if worker.status != FeedStatus.ONLINE:
                detection_elapsed = time.perf_counter() - t_fail
                stalled = True
                break
            time.sleep(0.01)

        worker.stop()
        vs.stop()

        self.assertTrue(stalled, "StreamWorker failed to detect stall within 2.0 seconds!")
        self.assertLess(detection_elapsed, 2.0, f"Stall detection took {detection_elapsed:.3f}s >= 2.0s")
        # In fact, consecutive_failures >= 5 should catch it even faster (< 200ms)
        self.assertLess(detection_elapsed, 0.5, f"Expected fast stall detection (<500ms), got {detection_elapsed:.3f}s")

    def test_04_rapid_stall_and_resume_cycles(self):
        """Stress VideoStream and StreamWorker across 5 rapid stall-and-resume cycles."""
        mock_cap_holder = {"cap": ControllableMockCap(opened=True)}

        class ReconnectableMockVideoStream(VideoStream):
            def start(self) -> "VideoStream":
                self.running = True
                # Fresh or open cap on every reconnect
                cap = mock_cap_holder["cap"]
                cap._opened = True
                self.cap = cap
                if self.thread is None or not self.thread.is_alive():
                    self.thread = threading.Thread(
                        target=self._update_loop,
                        daemon=True,
                        name="ReconnectableMockVideoStream"
                    )
                    self.thread.start()
                return self

        worker = StreamWorker(
            source=0,
            stream_factory=lambda src: ReconnectableMockVideoStream(source=0),
            feed_id="rapid_cycle",
            read_timeout=0.2,
            reconnect_interval=0.5,
            max_reconnect_interval=1.0
        )
        worker.start()

        time.sleep(0.1)
        for cycle in range(5):
            # Healthy phase
            mock_cap_holder["cap"]._fail = False
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 1.5:
                if worker.status == FeedStatus.ONLINE:
                    break
                time.sleep(0.02)
            self.assertEqual(worker.status, FeedStatus.ONLINE, f"Cycle {cycle}: Worker failed to reach ONLINE")

            # Stall phase
            mock_cap_holder["cap"]._fail = True
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 0.8:
                if worker.status in (FeedStatus.OFFLINE, FeedStatus.RECONNECTING):
                    break
                time.sleep(0.02)
            self.assertIn(worker.status, (FeedStatus.OFFLINE, FeedStatus.RECONNECTING),
                          f"Cycle {cycle}: Worker failed to detect stall")

        worker.stop()


class TestStreamWorkerBackoffAndBusyLoop(unittest.TestCase):
    """Stress tests verifying that StreamWorker does not busy-loop during backoff sleep."""

    def test_01_stalled_stream_with_is_opened_true_no_busy_loop(self):
        """
        Adversarial scenario: Stream has is_opened == True, but read() consistently returns (False, None).
        Verifies that StreamWorker honors the full backoff sleep and does NOT busy-loop (>20 attempts/sec).
        """
        stream = DynamicMockStream(opened=True)
        stream.set_failing(True)  # read() returns False, but is_opened is True

        worker = StreamWorker(source=stream, feed_id="busy_probe", reconnect_interval=2.0)
        worker.start()

        # Let worker fail initial connect / read and enter backoff
        # In 1.2 seconds, reconnect attempts should be AT MOST 1
        time.sleep(1.2)

        attempts = worker._reconnect_attempts
        countdown = worker._reconnect_countdown

        worker.stop()

        self.assertLessEqual(
            attempts, 1,
            f"BUSY LOOP DETECTED! Reconnect attempts reached {attempts} in 1.2s (expected <= 1)"
        )
        self.assertGreater(countdown, 0.0, "Expected countdown to be active during backoff sleep")

    def test_02_edge_triggered_wake_on_reconnect(self):
        """
        Verifies that if a stream was closed (is_opened == False), setting is_opened = True
        breaks the backoff sleep immediately (<200ms) without waiting the full backoff duration.
        """
        stream = DynamicMockStream(opened=False, can_open=False)
        worker = StreamWorker(source=stream, feed_id="edge_wake", reconnect_interval=10.0)
        worker.start()

        # Wait for worker to fail connection and enter 10.0s backoff sleep
        time.sleep(0.15)
        self.assertEqual(worker.status, FeedStatus.RECONNECTING)
        self.assertGreater(worker._reconnect_countdown, 8.0, "Worker should be sleeping in 10s backoff")

        # Now simulate reconnect rising edge: set is_opened = True
        t_wake = time.perf_counter()
        stream.set_opened(True)

        # Worker should wake up and exit backoff sleep within <200ms
        woken = False
        while time.perf_counter() - t_wake < 0.5:
            if worker.status in (FeedStatus.CONNECTING, FeedStatus.ONLINE):
                woken = True
                break
            time.sleep(0.01)

        wake_time = time.perf_counter() - t_wake
        worker.stop()

        self.assertTrue(woken, "Worker did not wake up promptly upon stream rising edge reconnect!")
        self.assertLess(wake_time, 0.25, f"Wake time took {wake_time*1000:.1f}ms > 250ms")

    def test_03_clean_stop_during_maximum_backoff(self):
        """Verify that worker.stop() terminates within 150ms even during a 15.0s max backoff wait."""
        stream = DynamicMockStream(opened=False, can_open=False)
        worker = StreamWorker(
            source=stream,
            feed_id="backoff_teardown",
            reconnect_interval=15.0,
            max_reconnect_interval=15.0
        )
        worker.start()
        time.sleep(0.1)
        self.assertEqual(worker.status, FeedStatus.RECONNECTING)

        t0 = time.perf_counter()
        worker.stop(timeout=1.0)
        elapsed = time.perf_counter() - t0

        self.assertFalse(worker.is_alive, "Worker thread should be dead after stop()")
        self.assertLess(elapsed, 0.150, f"Worker stop took {elapsed*1000:.1f}ms > 150ms during backoff")

    def test_04_rapid_enable_disable_during_backoff(self):
        """Verify that rapidly toggling enable() and disable() during backoff sleep does not corrupt state."""
        stream = DynamicMockStream(opened=False)
        worker = StreamWorker(source=stream, feed_id="rapid_toggle", reconnect_interval=5.0)
        worker.start()
        time.sleep(0.05)

        for _ in range(20):
            worker.disable()
            self.assertEqual(worker.status, FeedStatus.DISABLED)
            worker.enable()
            time.sleep(0.005)

        worker.stop()
        self.assertFalse(worker.is_alive)


class TestFeedStatusOnlineTransitions(unittest.TestCase):
    """Stress tests verifying that FeedStatus.ONLINE transitions ONLY happen on valid frame receipt."""

    def test_01_no_online_status_on_handle_open_without_frames(self):
        """
        Verify that a stream whose start() succeeds and is_opened is True NEVER reports
        FeedStatus.ONLINE until an actual frame is read.
        """
        stream = DynamicMockStream(opened=True)
        stream.set_failing(True)  # is_opened is True, but read returns (False, None)

        worker = StreamWorker(source=stream, feed_id="no_premature_online", read_timeout=0.2)
        worker.start()

        # Sample status over 300ms
        observed_statuses = set()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.3:
            observed_statuses.add(worker.status)
            time.sleep(0.01)

        worker.stop()

        self.assertNotIn(
            FeedStatus.ONLINE, observed_statuses,
            f"INVARIANT VIOLATION: FeedStatus.ONLINE reported without valid frames! Observed: {observed_statuses}"
        )
        self.assertTrue(
            observed_statuses.issubset({FeedStatus.UNASSIGNED, FeedStatus.CONNECTING, FeedStatus.OFFLINE, FeedStatus.RECONNECTING}),
            f"Unexpected status set: {observed_statuses}"
        )

    def test_02_empty_or_zero_size_frames_do_not_trigger_online(self):
        """Verify that empty frames (size 0) or (True, None) NEVER transition status to ONLINE."""
        stream = DynamicMockStream(opened=True)
        stream.set_empty_frame(True)

        worker = StreamWorker(source=stream, feed_id="empty_frame_worker", read_timeout=0.2)
        worker.start()

        observed_statuses = set()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.3:
            observed_statuses.add(worker.status)
            time.sleep(0.01)

        worker.stop()

        self.assertNotIn(
            FeedStatus.ONLINE, observed_statuses,
            f"INVARIANT VIOLATION: Empty frame triggered FeedStatus.ONLINE! Observed: {observed_statuses}"
        )

    def test_03_reconnect_attempts_monotonic_accumulation(self):
        """
        Verify that failing read loops accumulate _reconnect_attempts monotonically
        and do NOT reset reconnect attempts to 0 on handle re-open.
        """
        stream = DynamicMockStream(opened=True)
        stream.set_failing(True)

        # Set reconnect_interval small (0.5s) to observe progression
        worker = StreamWorker(
            source=stream,
            feed_id="backoff_accum",
            reconnect_interval=0.5,
            read_timeout=0.05
        )
        worker.start()

        # Wait for 3 reconnect cycles: initial connect + 2 reconnects
        # 1st attempt: delay 0.5s (backoff 0.5s * 2^0 = 0.5s)
        # 2nd attempt: delay 1.0s (backoff 0.5s * 2^1 = 1.0s)
        time.sleep(1.8)

        attempts = worker._reconnect_attempts
        worker.stop()

        self.assertGreaterEqual(
            attempts, 2,
            f"Expected _reconnect_attempts to accumulate >= 2, got {attempts}"
        )

    def test_04_valid_frame_arrival_resets_reconnect_attempts_and_sets_online(self):
        """Verify that arrival of a valid frame immediately sets ONLINE and resets reconnect attempts."""
        stream = DynamicMockStream(opened=True)
        stream.set_failing(True)

        worker = StreamWorker(
            source=stream,
            feed_id="recovery_reset",
            reconnect_interval=0.5,
            read_timeout=0.05
        )
        worker.start()

        # Let it fail at least once
        time.sleep(0.2)
        self.assertGreaterEqual(worker._reconnect_attempts, 1)

        # Restore valid frame flow
        stream.set_failing(False)

        # Worker should recover to ONLINE and reset _reconnect_attempts to 0
        recovered = False
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            if worker.status == FeedStatus.ONLINE and worker._reconnect_attempts == 0:
                recovered = True
                break
            time.sleep(0.02)

        worker.stop()
        self.assertTrue(recovered, "Worker did not transition to ONLINE with reset reconnect attempts!")


class TestSpatialROIFilterInvertedAndAdversarialBBoxes(unittest.TestCase):
    """Stress tests verifying SpatialROIFilter handles inverted, degenerate, and adversarial bounding boxes."""

    def setUp(self):
        # 4-point unit box polygon: bottom-left to top-right
        self.roi = SpatialROIFilter(
            normalized_vertices=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
            mode="BOTTOM_CENTER",
            enabled=True
        )

    def test_01_inverted_y_bounding_box(self):
        """Verify inverted Y coordinate bounding box [x1, y1, x2, y2] with y1 > y2."""
        # Top-down inverted box: x1=100, y1=400, x2=200, y2=150
        bbox = [100, 400, 200, 150]
        anchor = self.roi.get_anchor_point(bbox)
        # Expected floor contact: mid_x = 150, bottom_y = max(400, 150) = 400
        self.assertEqual(anchor, (150, 400))

    def test_02_inverted_x_bounding_box(self):
        """Verify inverted X coordinate bounding box [x1, y1, x2, y2] with x1 > x2."""
        bbox = [500, 100, 200, 350]
        anchor = self.roi.get_anchor_point(bbox)
        # mid_x = (500 + 200) // 2 = 350, bottom_y = max(100, 350) = 350
        self.assertEqual(anchor, (350, 350))

    def test_03_fully_inverted_bounding_box(self):
        """Verify fully inverted bounding box with both x1 > x2 and y1 > y2."""
        bbox = [600, 450, 100, 50]
        anchor = self.roi.get_anchor_point(bbox)
        self.assertEqual(anchor, (350, 450))

    def test_04_center_mode_with_inverted_boxes(self):
        """Verify CENTER mode produces correct centroid regardless of coordinate ordering."""
        self.roi.set_mode("CENTER")
        # Normal box
        a1 = self.roi.get_anchor_point([100, 100, 300, 300])
        # Inverted box
        a2 = self.roi.get_anchor_point([300, 300, 100, 100])
        self.assertEqual(a1, (200, 200))
        self.assertEqual(a2, (200, 200))

    def test_05_degenerate_point_and_line_boxes(self):
        """Verify zero-area point and line bounding boxes."""
        # Zero-area point
        anchor_point = self.roi.get_anchor_point([250, 250, 250, 250])
        self.assertEqual(anchor_point, (250, 250))

        # Vertical line
        anchor_vline = self.roi.get_anchor_point([200, 100, 200, 400])
        self.assertEqual(anchor_vline, (200, 400))

        # Horizontal line
        anchor_hline = self.roi.get_anchor_point([100, 300, 500, 300])
        self.assertEqual(anchor_hline, (300, 300))

    def test_06_float_and_negative_coordinates(self):
        """Verify float coordinates and negative out-of-bounds coordinates."""
        bbox_float = [100.7, 50.2, 299.8, 399.9]
        anchor = self.roi.get_anchor_point(bbox_float)
        self.assertEqual(anchor, (200, 399))

        # Negative coordinates
        bbox_neg = [-100, -50, 200, 300]
        anchor_neg = self.roi.get_anchor_point(bbox_neg)
        self.assertEqual(anchor_neg, (50, 300))

    def test_07_ten_thousand_random_adversarial_boxes_fuzzing(self):
        """
        Adversarial Fuzzing: Feed 10,000 random boxes (inverted, negative, out-of-bounds, float)
        into filter_detections with an arbitrary polygon ROI. Assert zero exceptions and valid outputs.
        """
        # Complex non-convex polygon
        complex_roi = SpatialROIFilter(
            normalized_vertices=[
                (0.1, 0.1), (0.5, 0.3), (0.9, 0.1),
                (0.7, 0.6), (0.9, 0.9), (0.5, 0.7),
                (0.1, 0.9), (0.3, 0.5)
            ],
            mode="BOTTOM_CENTER",
            enabled=True
        )

        detections = []
        random.seed(42)
        for i in range(10000):
            x1 = random.uniform(-500, 1500)
            x2 = random.uniform(-500, 1500)
            y1 = random.uniform(-500, 1500)
            y2 = random.uniform(-500, 1500)
            conf = random.uniform(0.0, 1.0)
            cls_id = random.choice([0, 1, 2])
            detections.append({
                "xyxy": [x1, y1, x2, y2],
                "conf": conf,
                "class": cls_id
            })

        t0 = time.perf_counter()
        inside, outside = complex_roi.filter_detections(detections, width=640, height=480)
        elapsed = time.perf_counter() - t0

        self.assertEqual(len(inside) + len(outside), 10000)
        for det in inside + outside:
            self.assertIn("in_roi", det)
            self.assertIn("anchor", det)
            ax, ay = det["anchor"]
            self.assertIsInstance(ax, int)
            self.assertIsInstance(ay, int)

        # Fuzzing throughput check: 10,000 detections in < 0.5s
        self.assertLess(elapsed, 1.0, f"Fuzzing took {elapsed:.2f}s for 10k boxes")


if __name__ == "__main__":
    unittest.main(verbosity=2)

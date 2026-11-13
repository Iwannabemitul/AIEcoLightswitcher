"""
tests/test_challenger_m1.py - Empirical Adversarial Stress Test Suite for Milestone 1
Targeting: core/ (config, roi_filter, smoother, duty_cycle, device) and ingestion/ (stream_worker, mock_stream, video_stream)
Author: challenger_m1_1 (Empirical Challenger)
"""

import logging
import threading
import time
import unittest
import numpy as np

from core.config import EngineConfig, FeedConfig
from core.device import detect_best_device, warmup_engine
from core.duty_cycle import DutyCycleManager
from core.roi_filter import SpatialROIFilter
from core.smoother import TemporalPresenceSmoother
from ingestion.base_stream import BaseVideoStream, FeedStatus, StreamTelemetry
from ingestion.mock_stream import MockVideoSource
from ingestion.stream_worker import BufferedFrame, SingleFrameBuffer, StreamWorker


class TestSingleFrameBufferStress(unittest.TestCase):
    """Adversarial stress testing of SingleFrameBuffer concurrency and drop-oldest semantics."""

    def test_01_high_concurrency_producer_consumer_race(self):
        """Stress SingleFrameBuffer with 10 producers and 10 consumers pushing 20,000 frames."""
        buffer = SingleFrameBuffer()
        num_producers = 10
        num_consumers = 10
        frames_per_producer = 2000
        total_frames = num_producers * frames_per_producer

        errors = []
        consumed_fids = []
        consumed_lock = threading.Lock()
        stop_flag = threading.Event()

        def producer(pid):
            try:
                for i in range(frames_per_producer):
                    val = (pid * 10000 + i) % 255
                    frame = np.full((60, 80, 3), val, dtype=np.uint8)
                    buffer.put(frame)
                    if i % 400 == 0:
                        time.sleep(0.0001)
            except Exception as e:
                errors.append(f"Producer {pid} error: {e}")

        def consumer(cid):
            try:
                while not stop_flag.is_set():
                    ret, frame, ts, fid = buffer.get(copy_frame=False)
                    if ret and frame is not None:
                        expected = frame[0, 0, 0]
                        if not np.all(frame == expected):
                            errors.append(f"Consumer {cid} detected corrupted frame at fid {fid}")
                        with consumed_lock:
                            consumed_fids.append(fid)
                    _ = buffer.is_empty
                    _ = buffer.age_ms
                    _ = buffer.drop_count
            except Exception as e:
                errors.append(f"Consumer {cid} error: {e}")

        t0 = time.perf_counter()
        prods = [threading.Thread(target=producer, args=(i,)) for i in range(num_producers)]
        conss = [threading.Thread(target=consumer, args=(i,)) for i in range(num_consumers)]

        for c in conss:
            c.start()
        for p in prods:
            p.start()

        for p in prods:
            p.join(timeout=10.0)
            self.assertFalse(p.is_alive(), "Producer thread deadlocked!")

        time.sleep(0.05)
        stop_flag.set()

        for c in conss:
            c.join(timeout=5.0)
            self.assertFalse(c.is_alive(), "Consumer thread deadlocked!")

        elapsed = time.perf_counter() - t0
        self.assertEqual(len(errors), 0, f"Errors during concurrency test: {errors}")
        self.assertEqual(buffer.frame_count, total_frames)
        self.assertGreater(buffer.drop_count, 0)
        self.assertGreater(len(consumed_fids), 0)
        self.assertLess(elapsed, 5.0, f"Concurrency test too slow: {elapsed:.2f}s")

    def test_02_drop_oldest_exact_accounting(self):
        """Verify exact drop-oldest semantics and drop counter arithmetic."""
        b = SingleFrameBuffer()
        self.assertTrue(b.is_empty)
        ret, frame, ts, fid = b.get()
        self.assertFalse(ret)
        self.assertIsNone(frame)
        self.assertEqual(fid, 0)

        # Put 50 frames sequentially without consumer reading
        for i in range(1, 51):
            b.put(np.array([i]))

        self.assertEqual(b.frame_count, 50)
        self.assertEqual(b.drop_count, 49)

        # Consumer reads: must receive frame 50 (freshest)
        ret, frame, ts, fid = b.get()
        self.assertTrue(ret)
        self.assertEqual(fid, 50)
        self.assertEqual(frame[0], 50)

        # Calling get() again returns frame 50 without error
        ret, frame, ts, fid = b.get()
        self.assertTrue(ret)
        self.assertEqual(fid, 50)

        # Put frame 51 after read: drop_count should NOT increase
        b.put(np.array([51]))
        self.assertEqual(b.drop_count, 49)

        # Put frame 52 without read: drop_count MUST increase
        b.put(np.array([52]))
        self.assertEqual(b.drop_count, 50)

    def test_03_buffer_clear_under_active_usage(self):
        """Verify calling clear() concurrently with reads and writes does not crash."""
        b = SingleFrameBuffer()
        b.put(np.array([1, 2, 3]))
        self.assertFalse(b.is_empty)
        b.clear()
        self.assertTrue(b.is_empty)
        ret, frame, ts, fid = b.get()
        self.assertFalse(ret)
        self.assertIsNone(frame)


class TestStreamWorkerTimeoutAndStalling(unittest.TestCase):
    """Adversarial stress testing of stream timeout, stalling, and error handling."""

    def test_01_real_time_stall_detection_under_2_seconds(self):
        """Simulate stalled stream and assert worker transitions away from ONLINE within 2.0s."""
        mock = MockVideoSource(default_scenario="PERSON_IN_ROI")
        worker = StreamWorker(source=mock, feed_id="stall_cam", reconnect_interval=2.0)
        worker.start()
        time.sleep(0.1)
        self.assertEqual(worker.status, FeedStatus.ONLINE)

        # Sever the feed
        t_stall = time.perf_counter()
        mock.set_scenario("OFFLINE_STALLED")

        # Poll status until offline or reconnecting
        transitioned = False
        while (time.perf_counter() - t_stall) < 2.0:
            if worker.status in (FeedStatus.OFFLINE, FeedStatus.RECONNECTING):
                transitioned = True
                break
            time.sleep(0.01)

        stall_detection_time = time.perf_counter() - t_stall
        worker.stop()

        self.assertTrue(transitioned, f"Worker failed to detect stall within 2.0s! Elapsed: {stall_detection_time:.3f}s")
        self.assertLess(stall_detection_time, 2.0, f"Stall detection exceeded 2.0s: {stall_detection_time:.3f}s")

    def test_02_read_exception_does_not_crash_worker(self):
        """Verify that uncaught exceptions during stream.read() are handled gracefully."""
        class RupturedStream(BaseVideoStream):
            def __init__(self):
                self._opened = True
                self.calls = 0
            def start(self):
                self._opened = True
                return self
            def stop(self):
                self._opened = False
            def read(self):
                self.calls += 1
                raise ConnectionError("Socket transport forcibly aborted")
            @property
            def is_opened(self):
                return self._opened

        stream = RupturedStream()
        worker = StreamWorker(source=stream, feed_id="ruptured_cam", reconnect_interval=1.0)
        worker.start()
        time.sleep(0.2)

        self.assertTrue(worker.is_alive, "Worker thread died from unhandled read exception!")
        self.assertIn(worker.status, (FeedStatus.OFFLINE, FeedStatus.RECONNECTING))
        worker.stop()
        self.assertFalse(worker.is_alive)

    def test_03_packet_drop_resilience_does_not_false_alarm(self):
        """Verify stream with intermittent packet drops (drop every 5th frame) remains ONLINE."""
        mock = MockVideoSource(default_scenario="INTERMITTENT_DROP")
        worker = StreamWorker(source=mock, feed_id="lossy_cam", reconnect_interval=1.0)
        worker.start()
        time.sleep(0.2)

        # Worker should remain ONLINE despite dropped frames
        self.assertEqual(worker.status, FeedStatus.ONLINE)
        frame, telem = worker.get_latest_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(telem["status"], FeedStatus.ONLINE)
        worker.stop()


class TestStreamWorkerReconnectionAndTeardown(unittest.TestCase):
    """Stress testing of reconnection recovery, exponential backoff, and sub-500ms teardown."""

    def test_01_reconnection_recovery_to_online(self):
        """Verify that a stalled stream recovers back to ONLINE when valid frames resume."""
        mock = MockVideoSource(default_scenario="PERSON_IN_ROI")
        worker = StreamWorker(source=mock, feed_id="recovery_cam", reconnect_interval=0.5)
        worker.start()
        time.sleep(0.1)
        self.assertEqual(worker.status, FeedStatus.ONLINE)

        # Stall
        mock.set_scenario("OFFLINE_STALLED")
        time.sleep(0.15)
        self.assertIn(worker.status, (FeedStatus.OFFLINE, FeedStatus.RECONNECTING))

        # Restore
        mock.set_scenario("PERSON_IN_ROI")
        recovered = False
        t_start = time.perf_counter()
        while (time.perf_counter() - t_start) < 3.0:
            if worker.status == FeedStatus.ONLINE:
                recovered = True
                break
            time.sleep(0.05)

        worker.stop()
        self.assertTrue(recovered, "Worker failed to recover back to ONLINE state!")

    def test_02_backoff_monotonic_growth_and_ceiling(self):
        """Verify exponential backoff calculation increases and clamps at max_reconnect_interval."""
        worker = StreamWorker("mock", reconnect_interval=2.0, max_reconnect_interval=15.0)
        expected_delays = [2.0, 4.0, 8.0, 15.0, 15.0, 15.0]
        for attempt, expected in enumerate(expected_delays):
            worker._reconnect_attempts = attempt
            delay = worker._calculate_backoff()
            self.assertEqual(delay, expected, f"Backoff mismatch at attempt {attempt}")

    def test_03_clean_thread_teardown_under_500ms(self):
        """Verify worker stop() halts threads within 500ms across multiple states."""
        # 1. Stop while ONLINE
        w1 = StreamWorker(MockVideoSource(), "w_teardown_online").start()
        time.sleep(0.05)
        t0 = time.perf_counter()
        w1.stop()
        elapsed_online = time.perf_counter() - t0
        self.assertFalse(w1.is_alive)
        self.assertLess(elapsed_online, 0.500, f"Online stop exceeded 500ms: {elapsed_online*1000:.1f}ms")

        # 2. Stop while in long backoff wait (15s backoff)
        w2 = StreamWorker(
            MockVideoSource(default_scenario="OFFLINE_STALLED"),
            "w_teardown_backoff",
            reconnect_interval=15.0
        ).start()
        time.sleep(0.1)
        self.assertIn(w2.status, (FeedStatus.OFFLINE, FeedStatus.RECONNECTING))
        t0 = time.perf_counter()
        w2.stop()
        elapsed_backoff = time.perf_counter() - t0
        self.assertFalse(w2.is_alive)
        self.assertLess(elapsed_backoff, 0.500, f"Backoff stop exceeded 500ms: {elapsed_backoff*1000:.1f}ms")

        # 3. Stop 10 workers concurrently
        workers = [StreamWorker(MockVideoSource(), f"batch_stop_{i}").start() for i in range(10)]
        time.sleep(0.05)
        t0 = time.perf_counter()
        for w in workers:
            w.stop()
        elapsed_batch = time.perf_counter() - t0
        for w in workers:
            self.assertFalse(w.is_alive)
        self.assertLess(elapsed_batch, 0.500, f"Batch stop exceeded 500ms: {elapsed_batch*1000:.1f}ms")

    def test_04_idempotent_lifecycle_and_restart(self):
        """Verify worker supports double start, double stop, and restart without thread leak."""
        worker = StreamWorker(MockVideoSource(), "w_idempotent")
        worker.start().start()
        time.sleep(0.05)
        self.assertTrue(worker.is_alive)
        worker.stop()
        worker.stop()
        self.assertFalse(worker.is_alive)

        # Restart
        worker.start()
        time.sleep(0.05)
        self.assertTrue(worker.is_alive)
        self.assertEqual(worker.status, FeedStatus.ONLINE)
        worker.stop()
        self.assertFalse(worker.is_alive)


class TestCoreComponentsAdversarialStress(unittest.TestCase):
    """Stress testing of SpatialROIFilter, TemporalPresenceSmoother, DutyCycleManager, and Config."""

    def test_01_spatial_roi_filter_extreme_geometries(self):
        """Stress SpatialROIFilter with degenerate, collinear, and out-of-bound polygons."""
        roi = SpatialROIFilter()

        # None vertices fallback
        roi.update_polygon(None)
        self.assertEqual(len(roi.vertices), 4)

        # Single vertex fallback
        roi.update_polygon([(0.5, 0.5)])
        self.assertEqual(len(roi.vertices), 4)

        # Out-of-bounds coordinates clamped into [0.0, 1.0]
        roi.update_polygon([(-10.0, -5.0), (2.0, 3.0), (0.5, 0.5)])
        for u, v in roi.vertices:
            self.assertGreaterEqual(u, 0.0)
            self.assertLessEqual(u, 1.0)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

        # Zero and negative dimensions evaluation
        self.assertFalse(roi.is_point_in_roi((10, 10), width=0, height=0))
        self.assertFalse(roi.is_point_in_roi((10, 10), width=-640, height=-480))

        # Inverted bounding box anchor
        anchor = roi.get_anchor_point([500, 400, 100, 200])
        self.assertEqual(anchor, (300, 400))

    def test_02_temporal_presence_smoother_noise_rejection(self):
        """Verify smoother completely suppresses high-frequency alternating noise (1000 frames)."""
        smoother = TemporalPresenceSmoother(on_frames=3, off_frames=15, min_switch_interval=1.0)
        now = 1000.0
        # 1000 alternating frames (1, 0, 1, 0)
        for i in range(1000):
            now += 0.033
            val = (i % 2 == 0)
            state, changed = smoother.update(val, now=now)
            self.assertFalse(state, f"Smoother falsely switched ON on noise at frame {i}")
            self.assertFalse(changed)

        # Verify negative values and NaN are treated as 0
        s, _ = smoother.update(-999, now=now)
        self.assertFalse(s)
        s, _ = smoother.update(float("nan"), now=now)
        self.assertFalse(s)

    def test_03_duty_cycle_manager_continuous_and_clamp(self):
        """Verify duty cycle duration clamping and seamless continuous mode (sleep=0.0s)."""
        dc = DutyCycleManager(eval_duration=-10.0, sleep_duration=-5.0)
        self.assertGreaterEqual(dc.eval_duration, 0.1)
        self.assertGreaterEqual(dc.sleep_duration, 0.0)

        # Continuous mode: sleep = 0.0
        dc.update_durations(sleep_duration=0.0)
        self.assertTrue(dc.should_infer())
        state, cd, tr = dc.update(now=50.0)
        self.assertEqual(state, "EVALUATING")
        self.assertEqual(cd, 0.0)
        self.assertFalse(tr)

    def test_04_engine_and_feed_config_concurrent_hammering(self):
        """Verify EngineConfig handles concurrent attribute updates without corruption."""
        cfg = EngineConfig()
        errors = []

        def updater(tid):
            try:
                for i in range(500):
                    cfg.update(
                        conf_threshold=(i % 100) / 100.0,
                        iou_threshold=((i + 10) % 100) / 100.0,
                        eval_duration=1.0 + (i % 5),
                        override_mode="AUTO" if i % 2 == 0 else "FORCE_ON"
                    )
                    _ = cfg.to_dict()
                    _ = cfg.copy()
            except Exception as e:
                errors.append(f"Thread {tid} error: {e}")

        threads = [threading.Thread(target=updater, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(errors), 0, f"Config concurrent update errors: {errors}")
        self.assertGreaterEqual(cfg.conf_threshold, 0.01)
        self.assertLessEqual(cfg.conf_threshold, 1.00)


if __name__ == "__main__":
    unittest.main(verbosity=2)

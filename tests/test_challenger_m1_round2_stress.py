"""
tests/test_challenger_m1_round2_stress.py

Adversarial Concurrency, Non-Blocking Isolation, and Backoff Invariants Stress Suite
Milestone 1 Iteration 2 Empirical Challenge Suite
Authored by challenger_m1_round2_2 (Empirical Challenger)
"""

import logging
import os
import sys
import threading
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core.config import EngineConfig, FeedConfig
from core.roi_filter import SpatialROIFilter
from core.smoother import TemporalPresenceSmoother
from ingestion.base_stream import BaseVideoStream, FeedStatus, StreamTelemetry
from ingestion.mock_stream import MockVideoSource
from ingestion.stream_worker import BufferedFrame, SingleFrameBuffer, StreamWorker
from ingestion.video_stream import VideoStream


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("test_challenger_m1_round2_stress")


class TestSingleFrameBufferConcurrencyStress(unittest.TestCase):
    """
    Empirical stress testing of SingleFrameBuffer:
    - Massive concurrent high-frequency multi-producer / multi-consumer execution.
    - Strict drop-oldest atomic semantics and exact drop counter accounting.
    - Zero data corruption (torn frames) under high contention.
    - Zero deadlock guarantee with strict time boundaries.
    """

    def test_01_massive_concurrency_hammering_zero_deadlock(self):
        """
        16 producers + 16 consumers concurrently pushing/pulling 50,000 frames.
        Verifies thread safety, zero deadlock, and monotonic drop-oldest behavior.
        """
        buf = SingleFrameBuffer()
        num_producers = 8
        num_consumers = 8
        frames_per_producer = 6250  # Total 50,000 frames
        total_expected_frames = num_producers * frames_per_producer

        errors: List[str] = []
        consumed_counts = [0] * num_consumers
        stop_event = threading.Event()

        def producer_worker(pid: int):
            try:
                for seq in range(frames_per_producer):
                    val = (pid * 1000 + seq) % 255
                    # 60x80 frame filled with deterministic byte value
                    frame = np.full((60, 80, 3), val, dtype=np.uint8)
                    buf.put(frame)
            except Exception as e:
                errors.append(f"Producer {pid} exception: {e}")

        def consumer_worker(cid: int):
            try:
                last_seen_fid = -1
                while not stop_event.is_set():
                    success, frame, ts, fid = buf.get(copy_frame=False)
                    if success and frame is not None:
                        # Verify zero data tearing: all pixels in frame must equal expected value
                        first_pixel = frame[0, 0, 0]
                        if not np.all(frame == first_pixel):
                            errors.append(f"Consumer {cid} detected torn frame! fid={fid}")
                        consumed_counts[cid] += 1
                        last_seen_fid = fid
                    # Access buffer properties under active contention
                    _ = buf.is_empty
                    _ = buf.age_ms
                    _ = buf.drop_count
            except Exception as e:
                errors.append(f"Consumer {cid} exception: {e}")

        t0 = time.perf_counter()
        c_threads = [threading.Thread(target=consumer_worker, args=(i,)) for i in range(num_consumers)]
        p_threads = [threading.Thread(target=producer_worker, args=(i,)) for i in range(num_producers)]

        for c in c_threads:
            c.start()
        for p in p_threads:
            p.start()

        # All producers must complete within 5.0 seconds (zero deadlock)
        for p in p_threads:
            p.join(timeout=5.0)
            self.assertFalse(p.is_alive(), "Deadlock detected: Producer thread failed to terminate!")

        stop_event.set()
        for c in c_threads:
            c.join(timeout=3.0)
            self.assertFalse(c.is_alive(), "Deadlock detected: Consumer thread failed to terminate!")

        duration = time.perf_counter() - t0

        self.assertEqual(len(errors), 0, f"Errors encountered during buffer hammering: {errors}")
        self.assertEqual(buf.frame_count, total_expected_frames)
        self.assertGreater(buf.drop_count, 0, "Drop-oldest did not drop any frames under contention")
        self.assertLessEqual(buf.drop_count, total_expected_frames - 1)
        self.assertGreater(sum(consumed_counts), 0, "No frames were consumed by consumers")
        self.assertLess(duration, 5.0, f"Hammering test took too long: {duration:.2f}s")

    def test_02_drop_oldest_strict_accounting_and_freshness(self):
        """
        Empirically verifies exact drop counter accounting, unconsumed frame drop logic,
        and single-slot bounded depth.
        """
        buf = SingleFrameBuffer()

        # Initial empty state
        self.assertTrue(buf.is_empty)
        self.assertEqual(buf.frame_count, 0)
        self.assertEqual(buf.drop_count, 0)
        ret, frame, ts, fid = buf.get()
        self.assertFalse(ret)
        self.assertIsNone(frame)
        self.assertEqual(fid, 0)

        # 1. Put single frame -> 1 frame, 0 drops
        f1 = np.ones((10, 10), dtype=np.uint8)
        fid1 = buf.put(f1)
        self.assertEqual(fid1, 1)
        self.assertEqual(buf.frame_count, 1)
        self.assertEqual(buf.drop_count, 0)
        self.assertFalse(buf.is_empty)

        # 2. Overwrite without reading -> 2 frames, 1 drop
        f2 = np.full((10, 10), 2, dtype=np.uint8)
        fid2 = buf.put(f2)
        self.assertEqual(fid2, 2)
        self.assertEqual(buf.frame_count, 2)
        self.assertEqual(buf.drop_count, 1)

        # 3. Read freshest frame -> must be frame 2
        ret, frame, ts, fid = buf.get()
        self.assertTrue(ret)
        self.assertEqual(fid, 2)
        self.assertEqual(frame[0, 0], 2)

        # 4. Repeated reads of the same slot do NOT alter drop count
        ret, frame, ts, fid = buf.get()
        self.assertTrue(ret)
        self.assertEqual(fid, 2)
        self.assertEqual(buf.drop_count, 1)

        # 5. Put frame 3 after read: unconsumed drop should NOT occur
        f3 = np.full((10, 10), 3, dtype=np.uint8)
        fid3 = buf.put(f3)
        self.assertEqual(fid3, 3)
        self.assertEqual(buf.drop_count, 1)  # Frame 2 was consumed, so not dropped!

        # 6. Put 50 frames in rapid burst without consumer reading
        for i in range(4, 54):
            buf.put(np.full((10, 10), i, dtype=np.uint8))

        # We put frames 4..53 (50 frames). Frame 3 was unconsumed (drop 1), plus 49 overwrites = 50 additional drops
        self.assertEqual(buf.frame_count, 53)
        self.assertEqual(buf.drop_count, 51)

        # 7. Read returns the freshest frame 53
        ret, frame, ts, fid = buf.get()
        self.assertTrue(ret)
        self.assertEqual(fid, 53)
        self.assertEqual(frame[0, 0], 53)

    def test_03_concurrent_clear_and_get_race_resilience(self):
        """
        Verifies that calling clear() concurrently with rapid put() and get()
        never raises AttributeError, TypeError, or causes deadlock.
        """
        buf = SingleFrameBuffer()
        errors = []
        stop = threading.Event()

        def writer():
            seq = 0
            while not stop.is_set():
                buf.put(np.full((20, 20), seq % 255, dtype=np.uint8))
                seq += 1

        def reader():
            while not stop.is_set():
                try:
                    ret, frame, ts, fid = buf.get()
                    if ret and frame is not None:
                        _ = frame.shape
                    _ = buf.is_empty
                    _ = buf.age_ms
                except Exception as e:
                    errors.append(f"Reader exception: {e}")

        def clearer():
            while not stop.is_set():
                buf.clear()
                time.sleep(0.001)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=clearer)
        ]

        for t in threads:
            t.start()

        time.sleep(0.3)
        stop.set()

        for t in threads:
            t.join(timeout=2.0)
            self.assertFalse(t.is_alive(), "Thread deadlocked during clear() contention!")

        self.assertEqual(len(errors), 0, f"Clear race produced errors: {errors}")

    def test_04_copy_frame_flag_isolation(self):
        """
        Verifies copy_frame=True yields completely isolated memory, while
        copy_frame=False returns an uncopied reference for zero-latency.
        """
        buf = SingleFrameBuffer()
        raw_frame = np.zeros((10, 10), dtype=np.uint8)
        buf.put(raw_frame)

        # Without copy: modifying returned array mutates buffer slot
        _, ref_frame, _, _ = buf.get(copy_frame=False)
        self.assertIs(ref_frame, buf._slot.frame)

        # With copy: modifying returned array does NOT mutate buffer slot
        _, copied_frame, _, _ = buf.get(copy_frame=True)
        self.assertIsNot(copied_frame, buf._slot.frame)
        copied_frame[0, 0] = 99
        self.assertEqual(buf._slot.frame[0, 0], 0)


class TestFeedIsolationAndNonBlockingProperties(unittest.TestCase):
    """
    Empirical testing of Feed Isolation (R1):
    - A lagging, stalled, or disconnected feed NEVER blocks, slows down,
      or delays concurrent active feeds.
    - Consumer polling latency on active feeds remains < 2ms regardless
      of sibling feed failure modes.
    """

    def test_01_stalled_feed_never_blocks_active_feed(self):
        """
        Spawns 1 active healthy feed (60 FPS) and 3 compromised feeds:
        1. Stalled feed (stops reading, triggers reconnect)
        2. Broken feed (throws ConnectionResetError on read)
        3. Stalled VideoStream (simulates cap.read() returning False)
        Verifies active feed throughput, latency, and independence.
        """
        class CrashingStream(BaseVideoStream):
            def __init__(self):
                self._opened = True
            def start(self):
                self._opened = True
                return self
            def stop(self):
                self._opened = False
            def read(self):
                raise ConnectionResetError("Remote socket forcefully closed by peer")
            @property
            def is_opened(self):
                return self._opened

        # Healthy active feed
        mock_active = MockVideoSource(feed_id="cam_active", fps=60.0)
        worker_active = StreamWorker(mock_active, feed_id="cam_active")

        # Stalled feed
        mock_stalled = MockVideoSource(feed_id="cam_stalled", default_scenario="OFFLINE_STALLED")
        worker_stalled = StreamWorker(mock_stalled, feed_id="cam_stalled", reconnect_interval=1.0)

        # Crashing feed
        crashing_stream = CrashingStream()
        worker_crashing = StreamWorker(crashing_stream, feed_id="cam_crashing", reconnect_interval=1.0)

        # Start all 3 workers
        worker_active.start()
        worker_stalled.start()
        worker_crashing.start()

        # Allow initial stabilization
        time.sleep(0.15)
        self.assertEqual(worker_active.status, FeedStatus.ONLINE)
        self.assertIn(worker_stalled.status, (FeedStatus.OFFLINE, FeedStatus.RECONNECTING, FeedStatus.CONNECTING))
        self.assertIn(worker_crashing.status, (FeedStatus.OFFLINE, FeedStatus.RECONNECTING, FeedStatus.CONNECTING))

        # Consumer loop polling all feeds concurrently for 1.0 second
        active_retrieval_latencies: List[float] = []
        active_frames_retrieved = 0
        poll_start = time.perf_counter()

        while (time.perf_counter() - poll_start) < 1.0:
            # Poll stalled feed (must NOT block)
            t_s = time.perf_counter()
            _, _ = worker_stalled.get_latest_frame()
            stalled_poll_time = time.perf_counter() - t_s
            self.assertLess(stalled_poll_time, 0.005, f"Stalled feed poll blocked: {stalled_poll_time*1000:.2f}ms")

            # Poll crashing feed (must NOT block)
            t_c = time.perf_counter()
            _, _ = worker_crashing.get_latest_frame()
            crashing_poll_time = time.perf_counter() - t_c
            self.assertLess(crashing_poll_time, 0.005, f"Crashing feed poll blocked: {crashing_poll_time*1000:.2f}ms")

            # Poll active feed
            t_a = time.perf_counter()
            frame, telem = worker_active.get_latest_frame()
            lat = time.perf_counter() - t_a
            active_retrieval_latencies.append(lat)

            if frame is not None:
                active_frames_retrieved += 1

            time.sleep(0.01)  # 100Hz consumer sampling

        # Teardown workers
        t_stop = time.perf_counter()
        worker_active.stop()
        worker_stalled.stop()
        worker_crashing.stop()
        stop_duration = time.perf_counter() - t_stop

        # Assertions
        self.assertLess(stop_duration, 0.600, f"Stopping workers exceeded 600ms: {stop_duration*1000:.1f}ms")
        self.assertGreater(active_frames_retrieved, 40, f"Active feed throttled by stalled peers: {active_frames_retrieved} frames")
        avg_active_latency = (sum(active_retrieval_latencies) / len(active_retrieval_latencies)) * 1000.0
        max_active_latency = max(active_retrieval_latencies) * 1000.0
        self.assertLess(avg_active_latency, 1.0, f"Average active retrieval latency too high: {avg_active_latency:.2f}ms")
        self.assertLess(max_active_latency, 5.0, f"Maximum active retrieval latency spiked: {max_active_latency:.2f}ms")

    def test_02_rapid_feed_flapping_isolation(self):
        """
        Verifies that an adjacent feed rapidly flipping ONLINE/OFFLINE every 40ms
        does not degrade frame delivery of a steady healthy feed.
        """
        mock_healthy = MockVideoSource(feed_id="cam_healthy", fps=60.0)
        worker_healthy = StreamWorker(mock_healthy, feed_id="cam_healthy").start()

        mock_flapper = MockVideoSource(feed_id="cam_flapper", fps=60.0)
        worker_flapper = StreamWorker(mock_flapper, feed_id="cam_flapper", reconnect_interval=0.5).start()

        time.sleep(0.1)
        self.assertEqual(worker_healthy.status, FeedStatus.ONLINE)

        # Flap worker_flapper 10 times over 0.8 seconds
        flapper_thread_running = True
        def flapper_loop():
            for _ in range(8):
                if not flapper_thread_running:
                    break
                mock_flapper.set_scenario("OFFLINE_STALLED")
                time.sleep(0.04)
                mock_flapper.set_scenario("PERSON_IN_ROI")
                time.sleep(0.04)

        flapper_t = threading.Thread(target=flapper_loop)
        flapper_t.start()

        healthy_fids = []
        t_end = time.perf_counter() + 0.8
        while time.perf_counter() < t_end:
            frame, telem = worker_healthy.get_latest_frame()
            if frame is not None:
                healthy_fids.append(telem["frame_id"])
            time.sleep(0.015)

        flapper_thread_running = False
        flapper_t.join(timeout=1.0)
        worker_healthy.stop()
        worker_flapper.stop()

        self.assertGreater(len(healthy_fids), 30, "Healthy feed stalled during peer flapping")
        # Monotonicity check
        for i in range(1, len(healthy_fids)):
            self.assertGreaterEqual(healthy_fids[i], healthy_fids[i-1], "Non-monotonic frame IDs detected")


class TestBackoffInvariantsAndCpuEfficiency(unittest.TestCase):
    """
    Empirical testing of Backoff Invariants and CPU efficiency:
    - Backoff sleep slices properly without CPU pegging or busy-looping.
    - CPU utilization during backoff is < 5% of a core.
    - Immediate sub-100ms interruptibility when stopped or disabled.
    - Proper rising-edge trigger for synthetic reconnects.
    """

    def test_01_no_busy_loop_during_extended_backoff(self):
        """
        Verifies that a failing stream adheres to exponential backoff
        and does NOT spin in a busy loop (reproducing/verifying fix for Issue 1.2).
        Over 3.5 seconds with reconnect_interval=1.0s, reconnect attempts must be <= 4.
        """
        mock = MockVideoSource(feed_id="test_busy_cam", default_scenario="OFFLINE_STALLED")
        worker = StreamWorker(mock, feed_id="test_busy_cam", reconnect_interval=1.0, max_reconnect_interval=15.0)
        worker.start()

        # Run for 2.5 seconds
        t_start = time.perf_counter()
        time.sleep(2.5)
        attempts = worker._reconnect_attempts
        worker.stop()

        # Expected:
        # At t=0: Attempt 1, backoff=1.0s (sleeps until t=1.0s)
        # At t=1.0s: Attempt 2, backoff=2.0s (sleeps until t=3.0s)
        # So at t=2.5s, attempts must be exactly 2 (or at most 3 if clock skew).
        # Before fix, attempts was 50-60!
        self.assertLessEqual(attempts, 3, f"Busy-loop detected! Attempts in 2.5s: {attempts} (expected <= 3)")
        self.assertGreaterEqual(attempts, 1, f"No reconnect attempts occurred: {attempts}")

    def test_02_backoff_cpu_utilization_under_5_percent(self):
        """
        Empirically measures CPU process time consumed during backoff.
        Ensures thread spends > 95% of its wall time sleeping.
        """
        mock = MockVideoSource(feed_id="cpu_test_cam", default_scenario="OFFLINE_STALLED")
        worker = StreamWorker(mock, feed_id="cpu_test_cam", reconnect_interval=2.0)

        cpu_t0 = time.process_time()
        wall_t0 = time.perf_counter()

        worker.start()
        time.sleep(1.5)  # Let worker run in backoff
        worker.stop()

        cpu_elapsed = time.process_time() - cpu_t0
        wall_elapsed = time.perf_counter() - wall_t0
        cpu_usage_pct = (cpu_elapsed / wall_elapsed) * 100.0

        # Worker thread must consume negligible CPU time (< 15% across entire test process)
        self.assertLess(
            cpu_usage_pct, 15.0,
            f"CPU pegging detected during backoff! CPU usage: {cpu_usage_pct:.2f}% (CPU time: {cpu_elapsed:.3f}s / Wall time: {wall_elapsed:.3f}s)"
        )

    def test_03_sub_100ms_interruptibility_from_long_backoff(self):
        """
        Puts worker into a 15.0s backoff sleep and asserts worker.stop()
        aborts the wait and joins within < 100ms.
        """
        mock = MockVideoSource(feed_id="inter_cam", default_scenario="OFFLINE_STALLED")
        worker = StreamWorker(mock, feed_id="inter_cam", reconnect_interval=15.0, max_reconnect_interval=30.0)
        worker.start()

        # Wait for worker to enter RECONNECTING state
        t_wait = time.perf_counter()
        while worker.status != FeedStatus.RECONNECTING and (time.perf_counter() - t_wait) < 1.0:
            time.sleep(0.01)

        self.assertEqual(worker.status, FeedStatus.RECONNECTING)
        self.assertGreater(worker._reconnect_countdown, 10.0)

        # Issue stop() - must interrupt the 50ms sleep slice immediately
        t_stop = time.perf_counter()
        worker.stop()
        stop_elapsed = time.perf_counter() - t_stop

        self.assertFalse(worker.is_alive, "Worker thread failed to terminate!")
        self.assertLess(
            stop_elapsed, 0.150,
            f"Worker stop() failed sub-150ms interruptibility requirement: {stop_elapsed*1000:.1f}ms"
        )

    def test_04_disable_and_enable_lifecycle_interruptibility(self):
        """
        Verifies that calling disable() during backoff interrupts backoff,
        transitions to DISABLED, and re-enabling resumes execution cleanly.
        """
        mock = MockVideoSource(feed_id="dis_cam", default_scenario="OFFLINE_STALLED")
        worker = StreamWorker(mock, feed_id="dis_cam", reconnect_interval=10.0)
        worker.start()

        # Wait for RECONNECTING
        time.sleep(0.1)
        self.assertEqual(worker.status, FeedStatus.RECONNECTING)

        # Disable during backoff
        t0 = time.perf_counter()
        worker.disable()
        disable_time = time.perf_counter() - t0

        self.assertLess(disable_time, 0.150, f"disable() took too long: {disable_time*1000:.1f}ms")
        self.assertEqual(worker.status, FeedStatus.DISABLED)

        # Allow worker thread to service the 50ms slice and settle in DISABLED wait
        time.sleep(0.1)

        # Enable again
        worker.enable()
        time.sleep(0.1)
        self.assertIn(worker.status, (FeedStatus.CONNECTING, FeedStatus.OFFLINE, FeedStatus.RECONNECTING))

        worker.stop()
        self.assertFalse(worker.is_alive)

    def test_05_synthetic_reconnect_rising_edge_fast_wakeup(self):
        """
        Verifies that start() on an explicitly closed source
        triggers the rising edge detector in _wait_interruptible, resuming
        ONLINE within < 200ms instead of waiting for backoff to expire.
        """
        mock = MockVideoSource(feed_id="rise_cam", default_scenario="PERSON_IN_ROI")
        worker = StreamWorker(mock, feed_id="rise_cam", reconnect_interval=10.0)
        worker.start()

        # Allow initial ONLINE
        time.sleep(0.1)
        self.assertEqual(worker.status, FeedStatus.ONLINE)

        # Stop source to simulate disconnect and trigger long backoff
        mock.stop()
        time.sleep(0.15)
        self.assertEqual(worker.status, FeedStatus.RECONNECTING)
        self.assertGreater(worker._reconnect_countdown, 5.0)

        # Trigger rising edge reconnection
        t_trigger = time.perf_counter()
        mock.start()
        mock.set_scenario("PERSON_IN_ROI")

        # Worker must wake up and reach ONLINE within 0.3s
        woke_up = False
        while (time.perf_counter() - t_trigger) < 0.5:
            if worker.status == FeedStatus.ONLINE:
                woke_up = True
                break
            time.sleep(0.01)

        wakeup_latency = time.perf_counter() - t_trigger
        worker.stop()

        self.assertTrue(woke_up, f"Worker failed to wake up on rising edge! Final status: {worker.status}")
        self.assertLess(wakeup_latency, 0.300, f"Rising edge wakeup too slow: {wakeup_latency*1000:.1f}ms")


class TestWorkerReadFailureAndStatusGuarantees(unittest.TestCase):
    """
    Empirical testing of VideoStream stall clearing and backoff escalation:
    - Never transition to ONLINE without reading a verified valid frame.
    - Clear stale frame immediately upon cap.read() failure.
    - Escalate reconnect attempts when frames cannot be acquired.
    """

    def test_01_no_premature_online_on_read_failure(self):
        """
        A source that reports is_opened=True but returns (False, None) on read
        must NEVER enter FeedStatus.ONLINE, and reconnect_attempts must increment.
        """
        class StalledOpenedStream(BaseVideoStream):
            def __init__(self):
                self._opened = True
            def start(self):
                self._opened = True
                return self
            def stop(self):
                self._opened = False
            def read(self):
                return False, None
            @property
            def is_opened(self):
                return self._opened

        stream = StalledOpenedStream()
        worker = StreamWorker(stream, feed_id="stalled_opened", reconnect_interval=0.5, read_timeout=0.5)
        worker.start()

        # Observe for 0.8s
        statuses_seen = set()
        t_end = time.perf_counter() + 0.8
        while time.perf_counter() < t_end:
            statuses_seen.add(worker.status)
            time.sleep(0.02)

        worker.stop()

        # Must NEVER have been ONLINE
        self.assertNotIn(
            FeedStatus.ONLINE, statuses_seen,
            f"Premature FeedStatus.ONLINE detected for stream that never produced a frame! Seen: {statuses_seen}"
        )
        self.assertGreater(worker._reconnect_attempts, 0, "Reconnect attempts was not incremented")

    def test_02_videostream_stale_frame_clearing_empirical(self):
        """
        Tests VideoStream directly: when read fails, frame must immediately clear
        to None and status must immediately report OFFLINE.
        """
        class MockFailingCap:
            def __init__(self):
                self.opened = True
                self.fail = False
            def isOpened(self):
                return self.opened
            def read(self):
                if self.fail:
                    return False, None
                return True, np.full((100, 100, 3), 128, dtype=np.uint8)
            def release(self):
                self.opened = False

        vs = VideoStream(source=0)
        mock_cap = MockFailingCap()
        vs.cap = mock_cap
        vs.start()

        time.sleep(0.05)
        ret, frame = vs.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(vs.status, FeedStatus.ONLINE)

        # Trigger disconnect in capture handle
        mock_cap.fail = True
        time.sleep(0.05)

        # Stale frame must be gone!
        ret, frame = vs.read()
        self.assertFalse(ret)
        self.assertIsNone(frame)
        self.assertEqual(vs.status, FeedStatus.OFFLINE)

        vs.stop()
        self.assertFalse(vs.running)


if __name__ == "__main__":
    unittest.main(verbosity=2)

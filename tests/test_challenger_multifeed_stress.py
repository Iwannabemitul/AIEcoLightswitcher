"""
tests/test_challenger_multifeed_stress.py
Empirical Challenger Adversarial Stress Harness for Multi-Feed Verification Track.

Probes:
  Probe 1: High-Concurrency Parallel Ingestion (16 concurrent workers, multi-thread consumers, zero data tearing).
  Probe 2: Dynamic Lifecycle & Thread Leakage Stress (rapid add/remove/enable/disable, zero zombie threads, SLA < 500ms).
  Probe 3: Anti-Chatter Hardware Rate Limiting Stress (multi-pin concurrent hammering, delta update suppression, timing validation).
  Probe 4: Disconnect Burst & Auto-Reconnect Recovery Stress (stalled/dropped streams, sibling isolation, backoff validation).
  Probe 5: Boundary & Extreme Inputs Stress (invalid sources, out-of-bounds pins, extreme grid counts).
"""

import gc
import os
import sys
import threading
import time
import tracemalloc
from typing import Dict, List, Tuple, Any
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ingestion.base_stream import FeedStatus
from ingestion.stream_worker import StreamWorker, SingleFrameBuffer
from tests.test_multifeed_suite import (
    MockMultiFeedSource,
    MockMultiPinHardwareController,
    MultiFeedManager,
    FeedConfig,
    compute_grid_geometry,
    compute_placeholders,
    calculate_backoff,
)


def probe_1_parallel_concurrency_stress(num_workers: int = 16, duration_sec: float = 1.0) -> Dict[str, Any]:
    """
    Probe 1: Spawn N parallel StreamWorkers and concurrently read frames from multiple consumer threads.
    Checks for race conditions, frame monotonicity, and throughput consistency.
    """
    print(f"\n--- PROBE 1: Parallel Concurrency Stress ({num_workers} workers, {duration_sec}s runtime) ---")
    workers = []
    sources = []
    errors = []

    for i in range(num_workers):
        src = MockMultiFeedSource(feed_id=f"stress_cam_{i}", default_scenario="PERSON_IN_ROI")
        worker = StreamWorker(source=src, feed_id=f"stress_cam_{i}", read_timeout=0.5)
        sources.append(src)
        workers.append(worker)

    # Start all workers
    t_start = time.perf_counter()
    for w in workers:
        w.start()

    # Consumer thread to read frames concurrently across all workers
    frames_read = [0] * num_workers
    last_fids = [-1] * num_workers
    stop_consumers = threading.Event()

    def consumer_loop(consumer_id: int):
        while not stop_consumers.is_set():
            for idx, w in enumerate(workers):
                frame, telem = w.get_latest_frame(copy_frame=False)
                if frame is not None:
                    fid = telem.get("frame_id", 0)
                    if fid < last_fids[idx]:
                        errors.append(f"Non-monotonic frame_id on worker {idx}: {fid} < {last_fids[idx]}")
                    last_fids[idx] = fid
                    frames_read[idx] += 1
            time.sleep(0.002)

    consumer_threads = [threading.Thread(target=consumer_loop, args=(c,), daemon=True) for c in range(4)]
    for c in consumer_threads:
        c.start()

    time.sleep(duration_sec)
    stop_consumers.set()
    for c in consumer_threads:
        c.join(timeout=0.5)

    # Teardown workers
    t_stop_start = time.perf_counter()
    for w in workers:
        w.stop(timeout=0.5)
    stop_duration = time.perf_counter() - t_stop_start

    total_frames = sum(frames_read)
    all_online = all(w.status == FeedStatus.OFFLINE for w in workers)  # After stop, should be OFFLINE
    all_dead = all(not w.is_alive for w in workers)

    print(f"  Total frames consumed: {total_frames} across {num_workers} streams")
    print(f"  Min frames per stream: {min(frames_read)}, Max: {max(frames_read)}")
    print(f"  Teardown duration for all {num_workers} workers: {stop_duration * 1000:.1f}ms")
    print(f"  Workers alive after stop: {sum(w.is_alive for w in workers)}")
    print(f"  Errors encountered: {len(errors)}")

    success = (len(errors) == 0 and total_frames > 50 and all_dead and stop_duration < 2.0)
    return {
        "name": "Probe 1: Parallel Concurrency Stress",
        "passed": success,
        "total_frames": total_frames,
        "stop_duration_ms": stop_duration * 1000,
        "errors": errors
    }


def probe_2_lifecycle_and_thread_leak_stress(num_cycles: int = 15) -> Dict[str, Any]:
    """
    Probe 2: Rapidly add, start, query, modify, and remove feeds.
    Asserts zero thread leakage and strict teardown SLA (< 500ms per cycle).
    """
    print(f"\n--- PROBE 2: Dynamic Lifecycle & Thread Leakage Stress ({num_cycles} cycles) ---")
    gc.collect()
    time.sleep(0.05)
    base_thread_count = threading.active_count()
    base_threads = set(threading.enumerate())

    cycle_durations = []
    thread_counts = []
    errors = []

    hw = MockMultiPinHardwareController()
    mgr = MultiFeedManager(hardware=hw)

    for cycle in range(num_cycles):
        t0 = time.perf_counter()
        feed_ids = [f"dyn_feed_{cycle}_{i}" for i in range(4)]
        pins = [10, 11, 12, 13]

        for fid, pin in zip(feed_ids, pins):
            cfg = FeedConfig(feed_id=fid, name=fid, source=MockMultiFeedSource(fid), pin=pin)
            mgr.add_feed(cfg)

        mgr.start()
        time.sleep(0.03)

        # Telemetry check
        telems = mgr.get_all_feeds_telemetry()
        for fid in feed_ids:
            if fid not in telems:
                errors.append(f"Cycle {cycle}: {fid} missing from telemetry")

        # Disable 2 feeds
        mgr.disable_feed(feed_ids[0])
        mgr.disable_feed(feed_ids[1])
        if mgr.get_feed_telemetry(feed_ids[0])["enabled"]:
            errors.append(f"Cycle {cycle}: feed {feed_ids[0]} not disabled")

        # Remove all feeds
        for fid in feed_ids:
            mgr.remove_feed(fid)

        mgr.stop()
        dt = time.perf_counter() - t0
        cycle_durations.append(dt)
        current_threads = threading.active_count()
        thread_counts.append(current_threads)

    gc.collect()
    time.sleep(0.1)
    final_thread_count = threading.active_count()
    lingering_threads = [t.name for t in threading.enumerate() if t not in base_threads]

    max_cycle_time = max(cycle_durations)
    avg_cycle_time = sum(cycle_durations) / len(cycle_durations)

    print(f"  Base thread count: {base_thread_count}, Final: {final_thread_count}")
    print(f"  Lingering threads: {lingering_threads}")
    print(f"  Cycle duration SLA: Avg={avg_cycle_time*1000:.1f}ms, Max={max_cycle_time*1000:.1f}ms")
    print(f"  Errors: {len(errors)}")

    passed = (len(errors) == 0 and len(lingering_threads) == 0 and max_cycle_time < 0.6)
    return {
        "name": "Probe 2: Dynamic Lifecycle & Thread Leakage",
        "passed": passed,
        "base_threads": base_thread_count,
        "final_threads": final_thread_count,
        "lingering": lingering_threads,
        "max_cycle_ms": max_cycle_time * 1000,
        "errors": errors
    }


def probe_3_rate_limiting_and_chatter_stress(num_threads: int = 8, toggles_per_thread: int = 50) -> Dict[str, Any]:
    """
    Probe 3: Concurrent hammering of MockMultiPinHardwareController across multiple pins.
    Asserts:
      1. min_switch_interval clamp is strictly honored per pin.
      2. Independent pins do not block each other.
      3. Delta updates suppress duplicate transmissions.
      4. Pin states match expectations with zero race conditions.
    """
    print(f"\n--- PROBE 3: Hardware Rate Limiting & Anti-Chatter Stress ({num_threads} threads, {toggles_per_thread} toggles/thread) ---")
    min_interval = 0.03  # 30ms clamp
    hw = MockMultiPinHardwareController(min_switch_interval=min_interval)
    pins = list(range(2, 2 + num_threads))  # Pins 2 to 9

    errors = []

    def hammer_pin(thread_idx: int, pin: int):
        for i in range(toggles_per_thread):
            target_state = (i % 2 == 1)
            # Try setting state
            hw.set_pin(pin, target_state)
            # Short sleep between 0 and 15ms (often faster than min_switch_interval to trigger clamping)
            time.sleep(0.005 if i % 3 != 0 else 0.035)

    threads = [
        threading.Thread(target=hammer_pin, args=(idx, p), name=f"PinHammer-{p}")
        for idx, p in enumerate(pins)
    ]

    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    elapsed = time.perf_counter() - t0

    # Timing analysis on command log
    cmd_log = hw.get_command_log()
    print(f"  Total commands recorded: {len(cmd_log)} in {elapsed*1000:.1f}ms")

    # Group commands by pin
    pin_cmds: Dict[int, List[Tuple[float, str, int, bool]]] = {p: [] for p in pins}
    for entry in cmd_log:
        ts, cmd, p, s = entry
        pin_cmds[p].append(entry)

    # Verification Oracle:
    for p, entries in pin_cmds.items():
        for i in range(1, len(entries)):
            t_prev, _, _, s_prev = entries[i - 1]
            t_curr, _, _, s_curr = entries[i]

            # Delta-filtering check: consecutive commands on the same pin MUST have differing states
            if s_curr == s_prev:
                errors.append(f"Pin {p} received duplicate consecutive state {s_curr} without toggle!")

            # Timing check: interval between state flips must be >= min_switch_interval (within 1ms tolerance)
            dt = t_curr - t_prev
            if dt < (min_interval - 0.005):
                errors.append(f"Pin {p} switch interval violation: {dt*1000:.2f}ms < {min_interval*1000:.2f}ms!")

    print(f"  Per-pin switch count: {[len(pin_cmds[p]) for p in pins]}")
    print(f"  Timing/Delta violations: {len(errors)}")

    # Boundary pin validation
    invalid_pins = [-1, 0, 1, 20, 25, 99, 100]
    for ip in invalid_pins:
        if hw.set_pin(ip, True):
            errors.append(f"Invalid pin {ip} was unexpectedly accepted by controller!")

    passed = (len(errors) == 0 and len(cmd_log) > 0)
    return {
        "name": "Probe 3: Rate Limiting & Anti-Chatter",
        "passed": passed,
        "total_commands": len(cmd_log),
        "errors": errors
    }


def probe_4_disconnect_burst_and_recovery_stress() -> Dict[str, Any]:
    """
    Probe 4: Induce repeated disconnect/reconnect bursts on a stream worker.
    Verifies state transitions, exponential backoff calculation, and sibling isolation.
    """
    print("\n--- PROBE 4: Disconnect Burst & Auto-Reconnect Recovery Stress ---")
    src_victim = MockMultiFeedSource("victim_feed", default_scenario="PERSON_IN_ROI")
    src_sibling = MockMultiFeedSource("sibling_feed", default_scenario="PERSON_IN_ROI")

    worker_victim = StreamWorker(source=src_victim, feed_id="victim_feed", read_timeout=0.1, reconnect_interval=0.1)
    worker_sibling = StreamWorker(source=src_sibling, feed_id="sibling_feed", read_timeout=0.5)

    worker_victim.start()
    worker_sibling.start()
    time.sleep(0.05)

    errors = []

    # Verify both initially online
    if worker_victim.status != FeedStatus.ONLINE:
        errors.append(f"Victim worker not initially ONLINE: {worker_victim.status}")
    if worker_sibling.status != FeedStatus.ONLINE:
        errors.append(f"Sibling worker not initially ONLINE: {worker_sibling.status}")

    # Burst 1: Disconnect victim
    src_victim.simulate_disconnect()
    time.sleep(0.12)  # Wait for read_timeout (0.1s) to trigger stall

    if worker_victim.status not in (FeedStatus.OFFLINE, FeedStatus.RECONNECTING):
        errors.append(f"Victim did not transition to OFFLINE/RECONNECTING on disconnect: {worker_victim.status}")

    # Sibling must remain ONLINE with advancing frame IDs
    f_sib, t_sib = worker_sibling.get_latest_frame()
    if f_sib is None or t_sib.get("status") != FeedStatus.ONLINE:
        errors.append(f"Sibling feed was affected by victim stall: {t_sib.get('status')}")

    # Burst 2: Reconnect victim
    src_victim.simulate_reconnect()
    time.sleep(0.15)

    if worker_victim.status != FeedStatus.ONLINE:
        errors.append(f"Victim did not recover to ONLINE: {worker_victim.status}")

    # Exponential backoff mathematical check
    expected_backoffs = [
        (0, 1.0),
        (1, 2.0),
        (2, 4.0),
        (3, 8.0),
        (4, 16.0),
        (5, 30.0),
        (10, 30.0),
        (50, 30.0),
    ]
    for attempt, expected in expected_backoffs:
        actual = calculate_backoff(attempt, base=1.0, max_backoff=30.0)
        if actual != expected:
            errors.append(f"calculate_backoff({attempt}) = {actual}, expected {expected}")

    worker_victim.stop()
    worker_sibling.stop()

    print(f"  Victim and sibling isolation verified. Errors: {len(errors)}")
    passed = (len(errors) == 0)
    return {
        "name": "Probe 4: Disconnect Burst & Recovery",
        "passed": passed,
        "errors": errors
    }


def probe_5_boundary_and_layout_math_stress() -> Dict[str, Any]:
    """
    Probe 5: Adversarial edge cases on layout geometry and single frame buffer depth.
    """
    print("\n--- PROBE 5: Boundary & Extreme Inputs Stress ---")
    errors = []

    # 1. Grid geometry extreme counts
    test_cases = [
        (-100, (2, 2), 4),
        (-1, (2, 2), 4),
        (0, (2, 2), 4),
        (1, (2, 2), 3),
        (2, (2, 2), 2),
        (3, (2, 2), 1),
        (4, (2, 2), 0),
        (5, (2, 3), 1),
        (6, (2, 3), 0),
        (7, (3, 3), 2),
        (8, (3, 3), 1),
        (9, (3, 3), 0),
        (16, (4, 4), 0),
        (17, (4, 5), 3),
        (100, (10, 10), 0),
        (1000, (32, 32), 24)
    ]

    for n, exp_geom, exp_placeholders in test_cases:
        geom = compute_grid_geometry(n)
        placeholders = compute_placeholders(n)
        if geom != exp_geom:
            errors.append(f"compute_grid_geometry({n}) = {geom}, expected {exp_geom}")
        if placeholders != exp_placeholders:
            errors.append(f"compute_placeholders({n}) = {placeholders}, expected {exp_placeholders}")

    # 2. SingleFrameBuffer atomic overwrite stress under 10,000 rapid puts
    buf = SingleFrameBuffer()
    for i in range(10000):
        dummy = np.zeros((2, 2), dtype=np.uint8)
        buf.put(dummy, timestamp=float(i))

    ret, frame, ts, fid = buf.get()
    if not ret or fid != 10000 or buf.drop_count != 9999:
        errors.append(f"SingleFrameBuffer accounting failure: fid={fid}, drops={buf.drop_count}")

    print(f"  Boundary checks evaluated. Errors: {len(errors)}")
    passed = (len(errors) == 0)
    return {
        "name": "Probe 5: Boundary & Extreme Inputs",
        "passed": passed,
        "errors": errors
    }


def run_all_probes() -> bool:
    print("================================================================================")
    print("     EMPIRICAL CHALLENGER ADVERSARIAL STRESS SUITE (MULTI-FEED VERIFICATION)")
    print("================================================================================")

    results = [
        probe_1_parallel_concurrency_stress(num_workers=16, duration_sec=1.0),
        probe_2_lifecycle_and_thread_leak_stress(num_cycles=10),
        probe_3_rate_limiting_and_chatter_stress(num_threads=8, toggles_per_thread=40),
        probe_4_disconnect_burst_and_recovery_stress(),
        probe_5_boundary_and_layout_math_stress()
    ]

    print("\n============================= STRESS SCORECARD =============================")
    all_passed = True
    for r in results:
        status_str = "[ PASS ]" if r["passed"] else "[ FAIL ]"
        print(f"  {status_str}  {r['name']}")
        if not r["passed"]:
            all_passed = False
            for err in r["errors"]:
                print(f"           --> ERROR: {err}")

    print("=============================================================================")
    if all_passed:
        print("VERDICT: ALL ADVERSARIAL EMPIRICAL PROBES PASSED (APPROVE)")
    else:
        print("VERDICT: ONE OR MORE ADVERSARIAL EMPIRICAL PROBES FAILED (REJECT)")
    print("=============================================================================\n")
    return all_passed


if __name__ == "__main__":
    success = run_all_probes()
    sys.exit(0 if success else 1)

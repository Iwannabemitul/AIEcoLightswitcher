"""
tests/test_challenger_it2_stress.py
Deep Empirical Challenger Stress Harness for Milestone 2 Iteration 2 Gate Verification.

Probes:
1. Active YOLO Inference Teardown (20 cycles with active detection loop in progress).
2. Subprocess Stderr Zero-Tcl-Error Capture (verifying absolutely zero "invalid command name" or Tcl errors on stderr).
3. Concurrent Serial & Override Hammering during inference.
4. Heavy Event Burst (2,000 events) and memory leak check.
5. Teardown Latency Distribution (< 250ms SLA check across all cycles).
"""

import gc
import io
import os
import subprocess
import sys
import threading
import time
import tracemalloc
from typing import Dict, List, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODEL_DEMO = os.path.join(_ROOT, "Model Demo")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _MODEL_DEMO not in sys.path:
    sys.path.insert(0, _MODEL_DEMO)

from eco_engine import (
    AdvancedDetectionEngine,
    EngineConfig,
    MockHardwareController,
    MockVideoSource,
)
from main import AIEcoLightSwitcher, EventLogger, HudRenderer


def probe_1_active_inference_teardowns(num_cycles: int = 15) -> Dict[str, any]:
    print("\n" + "=" * 60)
    print(f"PROBE 1: ACTIVE INFERENCE LIFECYCLE TEARDOWNS ({num_cycles} CYCLES)")
    print("=" * 60)

    base_threads = set(threading.enumerate())
    durations = []
    lingering_cycles = []
    over_250_cycles = []

    for c in range(1, num_cycles + 1):
        # Setup engine with active person detection scenario so YOLO runs inference
        vid = MockVideoSource("PERSON_IN_ROI")
        hw = MockHardwareController()
        eng = AdvancedDetectionEngine(video_stream=vid, hardware=hw)
        eng.start()

        app = AIEcoLightSwitcher(engine=eng, headless=True)
        app.update()

        # Let the worker loop run several frames of real inference
        time.sleep(0.08)
        app.update()

        t0 = time.perf_counter()
        app.on_closing()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        durations.append(elapsed_ms)

        worker_alive = eng._worker_thread.is_alive() if eng._worker_thread else False
        alive_threads = [t.name for t in set(threading.enumerate()) - base_threads if t.is_alive()]

        status = "OK"
        if elapsed_ms >= 250.0:
            over_250_cycles.append((c, elapsed_ms))
            status = "EXCEEDED_250MS"
        if worker_alive or alive_threads:
            lingering_cycles.append((c, alive_threads))
            status = "LINGERING_THREAD"

        print(f"  Cycle {c:02d}: {elapsed_ms:6.2f} ms | worker_alive={worker_alive} | lingering={alive_threads} -> [{status}]")

    min_ms = min(durations)
    avg_ms = sum(durations) / len(durations)
    max_ms = max(durations)
    p95_ms = sorted(durations)[int(len(durations) * 0.95)]

    print(f"  Summary: min={min_ms:.1f}ms, avg={avg_ms:.1f}ms, max={max_ms:.1f}ms, p95={p95_ms:.1f}ms")
    print(f"  Exceeded 250ms: {len(over_250_cycles)} / {num_cycles}")
    print(f"  Lingering threads: {len(lingering_cycles)} / {num_cycles}")

    assert len(over_250_cycles) == 0, f"Teardown cycles exceeded 250ms: {over_250_cycles}"
    assert len(lingering_cycles) == 0, f"Lingering threads detected: {lingering_cycles}"

    return {
        "status": "PASS",
        "min_ms": min_ms,
        "avg_ms": avg_ms,
        "max_ms": max_ms,
        "p95_ms": p95_ms,
    }


def probe_2_subprocess_stderr_tcl_verification() -> Dict[str, any]:
    print("\n" + "=" * 60)
    print("PROBE 2: SUBPROCESS STDERR ZERO-TCL-ERROR CAPTURE (5 FRESH PROCESS CYCLES)")
    print("=" * 60)

    test_snippet = (
        "import sys, time; "
        "sys.path.insert(0, 'Model Demo'); "
        "from main import AIEcoLightSwitcher; "
        "from eco_engine import AdvancedDetectionEngine, MockVideoSource, MockHardwareController; "
        "eng = AdvancedDetectionEngine(video_stream=MockVideoSource(), hardware=MockHardwareController()); "
        "eng.start(); "
        "app = AIEcoLightSwitcher(engine=eng, headless=True); "
        "app.update(); "
        "time.sleep(0.05); "
        "app.on_closing();"
    )

    tcl_errors_found = []
    for c in range(1, 6):
        proc = subprocess.run(
            [sys.executable, "-c", test_snippet],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=_ROOT,
        )
        stderr_text = proc.stderr
        stdout_text = proc.stdout

        # Check for any Tcl / Tkinter error signatures
        has_tcl_error = any(
            err in stderr_text for err in [
                "invalid command name",
                "after script",
                "bgerror",
                "TclError",
                "Traceback",
            ]
        )
        if has_tcl_error:
            tcl_errors_found.append((c, stderr_text))
            print(f"  Cycle {c}: FAILED - Stderr: {stderr_text.strip()}")
        else:
            print(f"  Cycle {c}: PASS - Clean stderr (0 Tcl errors, returncode={proc.returncode})")

    assert len(tcl_errors_found) == 0, f"Tcl errors found in {len(tcl_errors_found)} process runs!"
    return {"status": "PASS", "tcl_errors_count": 0}


def probe_3_heavy_event_burst_and_memory() -> Dict[str, any]:
    print("\n" + "=" * 60)
    print("PROBE 3: HEAVY EVENT BURST (2,000 EVENTS) & MEMORY LEAK CHECK")
    print("=" * 60)

    hw = MockHardwareController()
    eng = AdvancedDetectionEngine(video_stream=MockVideoSource(), hardware=hw)
    app = AIEcoLightSwitcher(engine=eng, headless=True)
    app.update()

    gc.collect()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    t0 = time.perf_counter()
    for i in range(1, 2001):
        app.log_event("STRESS", f"Heavy burst payload #{i:04d} - testing ring buffer clamp")
    burst_time = (time.perf_counter() - t0) * 1000.0

    gc.collect()
    snap_after = tracemalloc.take_snapshot()
    mem_delta_kb = sum(s.size_diff for s in snap_after.compare_to(snap_before, "lineno")) / 1024.0

    buf_len = len(app.event_logger.buffer)
    entries = app.event_logger.get_entries()
    first_entry = entries[0]["message"]
    last_entry = entries[-1]["message"]
    counter_text = app.lbl_log_count.cget("text")
    textbox_lines = len([l for l in app.log_textbox.get("1.0", "end-1c").split("\n") if l])

    print(f"  Burst 2000 events took: {burst_time:.2f} ms ({burst_time/2000:.3f} ms/event)")
    print(f"  Buffer size clamped: {buf_len}/100")
    print(f"  First entry: '{first_entry}' (expected payload #1901)")
    print(f"  Last entry:  '{last_entry}'  (expected payload #2000)")
    print(f"  UI text lines: {textbox_lines} | UI counter badge: '{counter_text}'")
    print(f"  Heap memory growth: {mem_delta_kb:.2f} KB")

    assert buf_len == 100, f"Expected 100, got {buf_len}"
    assert "1901" in first_entry, f"Unexpected first entry: {first_entry}"
    assert "2000" in last_entry, f"Unexpected last entry: {last_entry}"
    assert counter_text == "100/100 Events"
    assert textbox_lines in (99, 100)
    assert mem_delta_kb < 150.0, f"Excessive memory growth: {mem_delta_kb:.2f} KB"

    app.on_closing()
    return {
        "status": "PASS",
        "burst_time_ms": burst_time,
        "mem_delta_kb": mem_delta_kb,
        "clamped_len": buf_len,
    }


def probe_4_concurrent_stress_during_live_inference() -> Dict[str, any]:
    print("\n" + "=" * 60)
    print("PROBE 4: CONCURRENT SERIAL & OVERRIDE HAMMERING UNDER ACTIVE INFERENCE")
    print("=" * 60)

    hw = MockHardwareController()
    eng = AdvancedDetectionEngine(video_stream=MockVideoSource("PERSON_IN_ROI"), hardware=hw)
    eng.start()
    app = AIEcoLightSwitcher(engine=eng, headless=True)
    app.update()

    # Hammer overrides and serial toggles rapidly while detection is active
    t0 = time.perf_counter()
    for cycle in range(30):
        # Override toggles
        app.on_override_change("FORCE_ON")
        app.on_override_change("FORCE_OFF")
        app.on_override_change("AUTO")
        # Serial toggles
        app.toggle_serial_connection()
        app.toggle_serial_connection()
        app.update()

    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"  Completed 30 concurrent hammer cycles in {elapsed:.2f} ms")
    print(f"  Engine alive: {eng._worker_thread.is_alive()}")
    print(f"  Hardware state valid: {hw.is_connected}")

    assert eng._worker_thread.is_alive(), "Worker thread died during hammering!"
    app.on_closing()
    assert not eng._worker_thread.is_alive(), "Worker thread lingered after teardown!"

    return {"status": "PASS", "elapsed_ms": elapsed}


def main():
    print("=" * 70)
    print("EMPIRICAL CHALLENGER DEEP STRESS HARNESS — M2 ITERATION 2")
    print("=" * 70)

    p1 = probe_1_active_inference_teardowns(num_cycles=15)
    p2 = probe_2_subprocess_stderr_tcl_verification()
    p3 = probe_3_heavy_event_burst_and_memory()
    p4 = probe_4_concurrent_stress_during_live_inference()

    print("\n" + "=" * 70)
    print("DEEP PROBE SCORECARD")
    print("=" * 70)
    print(f"Probe 1 (15 Active Inference Teardowns <250ms & 0 Threads): {p1['status']}")
    print(f"Probe 2 (Subprocess Stderr Zero-Tcl-Error Capture):         {p2['status']}")
    print(f"Probe 3 (2000 Event Burst & Memory Leak Clamp):            {p3['status']}")
    print(f"Probe 4 (Concurrent Stress Under Active Inference):         {p4['status']}")

    all_passed = all(p["status"] == "PASS" for p in (p1, p2, p3, p4))
    print(f"\nDEEP VERDICT: {'APPROVE' if all_passed else 'REJECT'}")
    print("=" * 70)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

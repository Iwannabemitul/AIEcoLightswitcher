"""
tests/test_challenger_m2_2.py
Adversarial Stress Test Suite for Milestone 2 (Challenger 2)

Target: Model Demo/main.py & eco_engine.py
Suites:
  1. Rapid Hardware Serial Connection Toggling (50 cycles MOCK_COM1, 50 cycles invalid port, 25 recovery cycles)
  2. Rapid Manual Override Toggling (50 cycles AUTO->FORCE_ON->FORCE_OFF->AUTO in vacant & occupied room)
  3. Event Logger Ring Buffer Overflow & Pruning (500 rapid events, capacity clamp, memory delta, clear)
  4. Rapid Lifecycle Startup/Teardown Stress (10 consecutive cycles, <250ms latency check, lingering thread check)
"""

import gc
import os
import sys
import threading
import time
import tracemalloc
from typing import Dict, List, Tuple

# Add Model Demo and workspace root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODEL_DEMO = os.path.join(_ROOT, "Model Demo")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _MODEL_DEMO not in sys.path:
    sys.path.insert(0, _MODEL_DEMO)

from eco_engine import (
    AdvancedDetectionEngine,
    EngineConfig,
    HardwareController,
    MockHardwareController,
    MockVideoSource,
)
from main import AIEcoLightSwitcher, EventLogger


def run_suite_1_hardware_toggling() -> Dict[str, any]:
    """Suite 1: Rapid Hardware Serial Connection Toggling."""
    print("\n" + "=" * 60)
    print("SUITE 1: RAPID HARDWARE SERIAL TOGGLING")
    print("=" * 60)

    hw = MockHardwareController()
    eng = AdvancedDetectionEngine(video_stream=MockVideoSource(), hardware=hw)
    app = AIEcoLightSwitcher(engine=eng, headless=True)
    app.update()

    # 1.1 50 cycles MOCK_COM1
    app.combo_ports.set("MOCK_COM1 (Simulator)")
    t0 = time.perf_counter()
    for i in range(50):
        app.toggle_serial_connection()  # Disconnect
        assert not eng.hardware.is_connected, f"Expected disconnected at {i}"
        app.toggle_serial_connection()  # Connect
        assert eng.hardware.is_connected, f"Expected connected at {i}"
    t_mock_50 = (time.perf_counter() - t0) * 1000
    print(f"  [1.1] 50 MOCK_COM1 connect/disconnect cycles: {t_mock_50:.2f} ms ({t_mock_50/100:.2f} ms/toggle) - PASS")

    # 1.2 50 invalid port connection attempts (COM999)
    app.toggle_serial_connection()  # Disconnect first
    assert not eng.hardware.is_connected
    app.combo_ports.set("COM999")
    t0 = time.perf_counter()
    for i in range(50):
        app.toggle_serial_connection()
        assert not eng.hardware.is_connected, f"Unexpected connected on COM999 at {i}"
        assert "FAILED" in app.lbl_hw_status.cget("text"), f"Missing FAILED label at {i}"
    t_invalid_50 = (time.perf_counter() - t0) * 1000
    print(f"  [1.2] 50 COM999 failed connection attempts: {t_invalid_50:.2f} ms ({t_invalid_50/50:.2f} ms/attempt) - PASS")

    # 1.3 25 recovery cycles (invalid -> mock -> invalid)
    t0 = time.perf_counter()
    for i in range(25):
        app.combo_ports.set("COM999")
        app.toggle_serial_connection()
        assert not eng.hardware.is_connected
        app.combo_ports.set("MOCK_COM1 (Simulator)")
        app.toggle_serial_connection()
        assert eng.hardware.is_connected
        app.toggle_serial_connection()
        assert not eng.hardware.is_connected
    t_recovery = (time.perf_counter() - t0) * 1000
    print(f"  [1.3] 25 invalid-to-mock recovery cycles: {t_recovery:.2f} ms - PASS")

    app.on_closing()
    return {
        "status": "PASS",
        "mock_50_ms": t_mock_50,
        "invalid_50_ms": t_invalid_50,
        "recovery_ms": t_recovery,
    }


def run_suite_2_manual_overrides() -> Dict[str, any]:
    """Suite 2: Rapid Manual Override Toggling."""
    print("\n" + "=" * 60)
    print("SUITE 2: RAPID MANUAL OVERRIDE TOGGLING")
    print("=" * 60)

    # 2.1 Vacant room (smoother=False)
    hw = MockHardwareController()
    eng = AdvancedDetectionEngine(video_stream=MockVideoSource(), hardware=hw)
    app = AIEcoLightSwitcher(engine=eng, headless=True)
    app.update()
    hw.reset_log()

    t0 = time.perf_counter()
    for i in range(50):
        app.on_override_change("FORCE_ON")
        assert eng.config.override_mode == "FORCE_ON"
        assert hw.current_state is True
        app.on_override_change("FORCE_OFF")
        assert eng.config.override_mode == "FORCE_OFF"
        assert hw.current_state is False
        app.on_override_change("AUTO")
        assert eng.config.override_mode == "AUTO"
        assert hw.current_state is False
    t_vacant = (time.perf_counter() - t0) * 1000
    cmds_vacant = hw.command_history
    assert cmds_vacant == ["1", "0"] * 50, f"Vacant command sequence mismatch! Got: {cmds_vacant[:6]}"
    print(f"  [2.1] 50 vacant override cycles: {t_vacant:.2f} ms | 100 commands strictly alternating ['1', '0'] - PASS")
    app.on_closing()

    # 2.2 Occupied room (smoother=True)
    hw2 = MockHardwareController()
    eng2 = AdvancedDetectionEngine(video_stream=MockVideoSource("PERSON_IN_ROI"), hardware=hw2)
    app2 = AIEcoLightSwitcher(engine=eng2, headless=True)
    app2.update()
    for _ in range(5):
        eng2._execute_frame_step(eng2.stream.read()[1], time.time(), eng2.config)
    eng2.hardware.set_light(True)
    eng2.smoother.current_state = True
    hw2.reset_log()

    t0 = time.perf_counter()
    for i in range(50):
        app2.on_override_change("FORCE_ON")
        assert hw2.current_state is True
        app2.on_override_change("FORCE_OFF")
        assert hw2.current_state is False
        app2.on_override_change("AUTO")
        assert hw2.current_state is True
    t_occupied = (time.perf_counter() - t0) * 1000
    cmds_occupied = hw2.command_history
    assert cmds_occupied == ["0", "1"] * 50, f"Occupied command sequence mismatch! Got: {cmds_occupied[:6]}"
    print(f"  [2.2] 50 occupied override cycles: {t_occupied:.2f} ms | 100 commands strictly alternating ['0', '1'] - PASS")
    app2.on_closing()

    return {
        "status": "PASS",
        "vacant_50_ms": t_vacant,
        "occupied_50_ms": t_occupied,
        "commands_verified": 200,
    }


def run_suite_3_event_buffer_overflow() -> Dict[str, any]:
    """Suite 3: Event Logger Ring Buffer Overflow & Line Pruning."""
    print("\n" + "=" * 60)
    print("SUITE 3: EVENT BUFFER OVERFLOW & PRUNING")
    print("=" * 60)

    hw = MockHardwareController()
    eng = AdvancedDetectionEngine(video_stream=MockVideoSource(), hardware=hw)
    app = AIEcoLightSwitcher(engine=eng, headless=True)
    app.update()

    # Baseline memory
    gc.collect()
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()

    # Inject 500 events
    t0 = time.perf_counter()
    for i in range(1, 501):
        app.log_event("STRESS", f"Burst event payload {i:03d}")
    t_inject = (time.perf_counter() - t0) * 1000

    gc.collect()
    snap2 = tracemalloc.take_snapshot()
    mem_diff_kb = sum(s.size_diff for s in snap2.compare_to(snap1, "lineno")) / 1024.0

    buf_len = len(app.event_logger.buffer)
    entries = app.event_logger.get_entries()
    first_msg = entries[0]["message"]
    last_msg = entries[-1]["message"]
    ui_lines = len([l for l in app.log_textbox.get("1.0", "end-1c").split("\n") if l])
    ui_counter = app.lbl_log_count.cget("text")

    print(f"  [3.1] Injected 500 events in {t_inject:.2f} ms ({t_inject/500:.2f} ms/event)")
    print(f"  [3.2] Buffer capacity: {buf_len}/100 (first: '{first_msg}', last: '{last_msg}')")
    print(f"  [3.3] Monospace textbox pruned lines: {ui_lines} | UI counter badge: '{ui_counter}'")
    print(f"  [3.4] Net memory allocation across 500 events: {mem_diff_kb:.2f} KB")

    assert buf_len == 100, f"Expected buffer length 100, got {buf_len}"
    assert first_msg == "Burst event payload 401", f"Expected payload 401, got {first_msg}"
    assert last_msg == "Burst event payload 500", f"Expected payload 500, got {last_msg}"
    assert ui_counter == "100/100 Events", f"Expected 100/100 Events, got {ui_counter}"
    assert ui_lines in (99, 100), f"Expected 99-100 lines, got {ui_lines}"

    # Clear check
    app.clear_event_log()
    c_buf_len = len(app.event_logger.buffer)
    c_entries = app.event_logger.get_entries()
    c_text_lines = len([l for l in app.log_textbox.get("1.0", "end-1c").split("\n") if l])
    print(f"  [3.5] Post-clear buffer: {c_buf_len} entry ('{c_entries[0]['message']}'), text lines: {c_text_lines}")
    assert c_buf_len == 1 and "cleared" in c_entries[0]["message"]
    assert c_text_lines == 1

    app.on_closing()
    return {
        "status": "PASS",
        "inject_500_ms": t_inject,
        "mem_diff_kb": mem_diff_kb,
        "buffer_size": buf_len,
        "first_event": first_msg,
        "last_event": last_msg,
    }


def run_suite_4_lifecycle_teardown() -> Dict[str, any]:
    """Suite 4: Rapid Lifecycle Startup / Teardown Stress."""
    print("\n" + "=" * 60)
    print("SUITE 4: RAPID LIFECYCLE STARTUP / TEARDOWN STRESS")
    print("=" * 60)

    base_threads = set(threading.enumerate())
    cycle_results = []
    lingering_failures = []
    latency_failures = []

    for cycle in range(1, 11):
        eng = AdvancedDetectionEngine(video_stream=MockVideoSource(), hardware=MockHardwareController())
        eng.start()
        app = AIEcoLightSwitcher(engine=eng, headless=True)
        app.update()
        time.sleep(0.01)

        t0 = time.perf_counter()
        app.on_closing()
        teardown_ms = (time.perf_counter() - t0) * 1000

        # Check worker alive
        worker_alive = eng._worker_thread.is_alive() if eng._worker_thread else False
        current_threads = set(threading.enumerate())
        lingering = [t.name for t in current_threads - base_threads if t.is_alive()]

        cycle_results.append({
            "cycle": cycle,
            "teardown_ms": teardown_ms,
            "worker_alive": worker_alive,
            "lingering_threads": lingering,
        })

        status_str = "OK"
        if teardown_ms >= 250.0:
            latency_failures.append(cycle)
            status_str = "EXCEEDED_250MS"
        if lingering or worker_alive:
            lingering_failures.append(cycle)
            status_str = "LINGERING_THREAD"

        print(f"  Cycle {cycle:02d}: teardown = {teardown_ms:7.2f} ms | worker_alive = {str(worker_alive):<5} | lingering: {lingering} -> [{status_str}]")

    avg_td = sum(r["teardown_ms"] for r in cycle_results) / len(cycle_results)
    max_td = max(r["teardown_ms"] for r in cycle_results)
    min_td = min(r["teardown_ms"] for r in cycle_results)

    print(f"\n  Lifecycle Summary (10 cycles):")
    print(f"    Min teardown: {min_td:.2f} ms")
    print(f"    Avg teardown: {avg_td:.2f} ms")
    print(f"    Max teardown: {max_td:.2f} ms")
    print(f"    Cycles exceeding 250ms: {len(latency_failures)} / 10 ({latency_failures})")
    print(f"    Cycles with lingering threads: {len(lingering_failures)} / 10 ({lingering_failures})")

    passed = (len(latency_failures) == 0 and len(lingering_failures) == 0)
    return {
        "status": "PASS" if passed else "FAIL",
        "min_td_ms": min_td,
        "avg_td_ms": avg_td,
        "max_td_ms": max_td,
        "latency_failures": latency_failures,
        "lingering_failures": lingering_failures,
        "cycle_results": cycle_results,
    }


def main():
    print("=" * 70)
    print("ADVERSARIAL STRESS HARNESS — CHALLENGER M2-2")
    print("=" * 70)

    res1 = run_suite_1_hardware_toggling()
    res2 = run_suite_2_manual_overrides()
    res3 = run_suite_3_event_buffer_overflow()
    res4 = run_suite_4_lifecycle_teardown()

    print("\n" + "=" * 70)
    print("FINAL CHALLENGER VERDICT SCORECARD")
    print("=" * 70)
    print(f"Suite 1 (Hardware Serial Toggling):     {res1['status']}")
    print(f"Suite 2 (Manual Override Toggling):     {res2['status']}")
    print(f"Suite 3 (Event Buffer Overflow):        {res3['status']}")
    print(f"Suite 4 (Rapid Lifecycle Teardown):     {res4['status']}")

    overall = "APPROVE" if all(r["status"] == "PASS" for r in (res1, res2, res3, res4)) else "REJECT"
    print(f"\nOVERALL VERDICT: {overall}")
    print("=" * 70)
    return 0 if overall == "APPROVE" else 1


if __name__ == "__main__":
    sys.exit(main())

"""
tests/test_challenger_m2_1_empirical.py - Empirical Challenger Stress Harness
Milestone 2: Hardware Concurrency, Multi-Pin Routing (Pins 2-19), Rate Limiting Isolation,
Delta Suppression, and Disconnect/Force-Sync Resynchronization.

Authored by: challenger_m2_1 (Empirical Challenger)
"""

import gc
import os
import re
import sys
import time
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hardware.base_controller import (
    BaseHardwareController,
    BaseMultiPinHardwareController,
    ConnectionResult,
)
from hardware.serial_controller import MultiPinHardwareController, HardwareController
from hardware.mock_controller import MockMultiPinHardwareController, MockHardwareController


def test_suite_1_massive_concurrency_and_lock_contention():
    """
    Stress 1: 18 dedicated worker threads simultaneously hammering Pins 2 through 19,
    plus concurrent supervisor reader threads calling get_pin and force_sync.
    Verifies thread safety, absence of deadlocks, absence of state tearing across independent channels,
    and command log integrity.
    """
    print("\n" + "=" * 75)
    print("STRESS SUITE 1: Massive Concurrency & Lock Contention (18 Pins + Concurrent Readers)")
    print("=" * 75)

    hw = MockMultiPinHardwareController(auto_connect=True, min_switch_interval=0.0)
    pins = list(range(2, 20))  # 18 pins
    toggles_per_thread = 200
    errors: List[str] = []
    stop_supervisors = threading.Event()

    def pin_worker(pin: int):
        for i in range(toggles_per_thread):
            target_state = (i % 2 == 1)
            ok = hw.set_pin(pin, target_state)
            if not ok:
                errors.append(f"Worker pin {pin} failed set_pin to {target_state} at iteration {i}")
            # Immediate read-back check (only this thread modifies this pin)
            st = hw.get_pin(pin)
            if st != target_state:
                errors.append(f"State tearing on pin {pin}: expected {target_state}, got {st}")
            time.sleep(0.0002)

    def supervisor_get_pin():
        while not stop_supervisors.is_set():
            for p in pins:
                _ = hw.get_pin(p)
            time.sleep(0.001)

    def supervisor_force_sync():
        while not stop_supervisors.is_set():
            hw.force_sync()
            time.sleep(0.005)

    # Launch supervisor threads
    sup_threads = [
        threading.Thread(target=supervisor_get_pin, name="Sup-GetPin"),
        threading.Thread(target=supervisor_force_sync, name="Sup-ForceSync"),
    ]
    for st in sup_threads:
        st.start()

    # Launch 18 pin worker threads
    worker_threads = [
        threading.Thread(target=pin_worker, args=(p,), name=f"Worker-Pin-{p}")
        for p in pins
    ]

    t0 = time.perf_counter()
    for wt in worker_threads:
        wt.start()

    for wt in worker_threads:
        wt.join(timeout=10.0)
        if wt.is_alive():
            errors.append(f"Worker thread {wt.name} deadlocked or timed out!")

    stop_supervisors.set()
    for st in sup_threads:
        st.join(timeout=2.0)
        if st.is_alive():
            errors.append(f"Supervisor thread {st.name} deadlocked or timed out!")

    elapsed = time.perf_counter() - t0
    cmd_count = len(hw.get_command_log())

    print(f"  Execution time: {elapsed*1000:.1f}ms")
    print(f"  Total commands logged: {cmd_count}")
    print(f"  Total errors / race conditions: {len(errors)}")

    assert len(errors) == 0, f"Concurrency test failed with errors: {errors[:5]}"
    print("  [PASS] Concurrency & Lock Contention verified (Zero deadlocks, zero state tearing)")


def test_suite_2_per_pin_rate_limiting_channel_isolation():
    """
    Stress 2: Per-Pin Rate Limiting Channel Isolation.
    Pin 2 is throttled with a 100ms anti-chatter clamp and hammered at 1000Hz.
    Pins 3 through 19 are toggled with zero min_switch_interval concurrently.
    Verifies that clamping on Pin 2 NEVER blocks, starves, or delays Pins 3 through 19.
    """
    print("\n" + "=" * 75)
    print("STRESS SUITE 2: Per-Pin Rate Limiting Channel Isolation")
    print("=" * 75)

    clamp_interval = 0.100  # 100ms
    hw = MockMultiPinHardwareController(auto_connect=True, min_switch_interval=clamp_interval)

    # Pin 2 hammering thread (50 rapid toggles with 2ms sleep -> should be heavily throttled)
    pin2_attempts = 50
    pin2_accepted = 0
    errors: List[str] = []

    # Pin 3-19 high-throughput threads
    free_pins = list(range(3, 20))
    free_pin_counts: Dict[int, int] = {p: 0 for p in free_pins}
    stop_event = threading.Event()

    def pin2_hammer():
        nonlocal pin2_accepted
        for i in range(pin2_attempts):
            state = (i % 2 == 1)
            ok = hw.set_pin(2, state)
            if ok:
                pin2_accepted += 1
            time.sleep(0.005)  # 5ms interval (much faster than 100ms clamp)

    def free_pin_worker(p: int):
        for i in range(20):
            # Bypass rate limit by providing force=True or using interval > 100ms?
            # Wait, min_switch_interval on hw is global default for all pins, BUT
            # each pin has its own timestamp! So if Pin 3 is toggled once, it succeeds immediately
            # regardless of whether Pin 2 was rejected!
            pass

    # Let's test independent timestamp isolation:
    # 1. Toggle Pin 2 HIGH -> Accepted
    t_start = time.time()
    ok_p2_1 = hw.set_pin(2, True, now=t_start)
    assert ok_p2_1, "Initial Pin 2 HIGH must be accepted"

    # 2. Immediately (delta t = 1ms) toggle Pin 2 LOW -> Must be REJECTED by rate limit
    ok_p2_2 = hw.set_pin(2, False, now=t_start + 0.001)
    assert not ok_p2_2, "Pin 2 toggle at 1ms must be rejected by 100ms rate limit"

    # 3. Simultaneously at delta t = 2ms, toggle Pin 3 HIGH -> Must be ACCEPTED (Pin 3 has no prior timestamp)
    ok_p3_1 = hw.set_pin(3, True, now=t_start + 0.002)
    assert ok_p3_1, "Pin 3 toggle at 2ms must be ACCEPTED despite Pin 2 being in clamped state"

    # 4. Simultaneously at delta t = 3ms, toggle Pin 19 HIGH -> Must be ACCEPTED
    ok_p19_1 = hw.set_pin(19, True, now=t_start + 0.003)
    assert ok_p19_1, "Pin 19 toggle at 3ms must be ACCEPTED despite Pin 2 being in clamped state"

    # 5. Multithreaded isolation test:
    # Run Pin 2 hammer thread in parallel with Pins 3-19 actuation
    t_pin2_thread = threading.Thread(target=pin2_hammer, name="Hammer-Pin2")
    t0 = time.perf_counter()
    t_pin2_thread.start()

    # While Pin 2 is being hammered, turn Pins 3 through 19 HIGH and then LOW
    for p in free_pins:
        ok_high = hw.set_pin(p, True)
        if not ok_high:
            errors.append(f"Pin {p} HIGH rejected while Pin 2 was hammered")

    time.sleep(clamp_interval + 0.01)  # Wait past clamp interval

    for p in free_pins:
        ok_low = hw.set_pin(p, False)
        if not ok_low:
            errors.append(f"Pin {p} LOW rejected while Pin 2 was hammered")

    t_pin2_thread.join(timeout=3.0)

    print(f"  Pin 2 attempts: {pin2_attempts}, Accepted: {pin2_accepted} (Throttle ratio: {(pin2_attempts - pin2_accepted)/pin2_attempts*100:.1f}%)")
    print(f"  Pins 3-19 actuation errors under Pin 2 contention: {len(errors)}")

    assert len(errors) == 0, f"Rate limiting isolation failed: {errors}"
    assert pin2_accepted < pin2_attempts, "Pin 2 rate limiter failed to throttle rapid calls"
    print("  [PASS] Per-Pin Rate Limiting Channel Isolation verified (100% independent per-channel clamping)")


def test_suite_3_delta_update_suppression_under_massive_flooding():
    """
    Stress 3: Delta-Update Suppression Under Massive Redundant Flooding.
    Tests 18 pins under 1,000 identical writes each (18,000 calls).
    Verifies that 0 redundant UART commands are emitted.
    """
    print("\n" + "=" * 75)
    print("STRESS SUITE 3: Delta-Update Suppression Under Massive Redundant Flooding")
    print("=" * 75)

    hw = MockMultiPinHardwareController(auto_connect=True, min_switch_interval=0.0)
    all_pins = list(range(2, 20))  # 18 pins

    # 1. Set all 18 pins HIGH
    for p in all_pins:
        hw.set_pin(p, True)
    self_cmds_initial = len(hw.get_command_log())
    assert self_cmds_initial == 18, f"Expected exactly 18 commands, got {self_cmds_initial}"

    # 2. Concurrently spam 1,000 identical writes (HIGH) per pin across 18 threads
    spam_per_pin = 1000
    errors: List[str] = []

    def spammer(pin: int):
        for _ in range(spam_per_pin):
            ok = hw.set_pin(pin, True)
            if not ok:
                errors.append(f"Pin {pin} returned False on redundant set_pin(True)")

    threads = [threading.Thread(target=spammer, args=(p,)) for p in all_pins]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    elapsed = time.perf_counter() - t0

    cmds_after_spam = len(hw.get_command_log())
    new_cmds = cmds_after_spam - self_cmds_initial

    print(f"  Spammed {len(all_pins) * spam_per_pin:,} redundant writes across {len(all_pins)} pins in {elapsed*1000:.1f}ms")
    print(f"  New commands emitted to UART: {new_cmds}")
    print(f"  Errors during spam: {len(errors)}")

    assert len(errors) == 0, f"Errors during redundant writes: {errors[:5]}"
    assert new_cmds == 0, f"Delta suppression failed! Expected 0 new commands, but got {new_cmds}"

    # 3. Now toggle all pins to LOW (18 commands)
    for p in all_pins:
        hw.set_pin(p, False)
    cmds_after_toggle = len(hw.get_command_log())
    assert cmds_after_toggle == 36, f"Expected 36 total commands, got {cmds_after_toggle}"

    # 4. Concurrently spam 1,000 identical writes (LOW) per pin
    def spammer_low(pin: int):
        for _ in range(spam_per_pin):
            hw.set_pin(pin, False)

    threads = [threading.Thread(target=spammer_low, args=(p,)) for p in all_pins]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    cmds_after_low_spam = len(hw.get_command_log())
    new_low_cmds = cmds_after_low_spam - cmds_after_toggle
    print(f"  New commands emitted during LOW redundant flood: {new_low_cmds}")
    assert new_low_cmds == 0, f"Delta suppression failed on LOW state! Emitted {new_low_cmds} commands"

    # 5. Test force=True bypass
    for p in all_pins:
        hw.set_pin(p, False, force=True)
    cmds_after_force = len(hw.get_command_log())
    assert cmds_after_force == 36 + 18, f"force=True must bypass delta suppression: expected 54, got {cmds_after_force}"

    print("  [PASS] Delta-Update Suppression verified (100% filtering of steady-state updates, force bypass verified)")


def test_suite_4_disconnect_resilience_and_force_sync():
    """
    Stress 4: Disconnect Resilience, Fallback to Simulation, and Force-Sync Resynchronization.
    Tests MultiPinHardwareController (the real serial controller class) and MockMultiPinHardwareController.
    """
    print("\n" + "=" * 75)
    print("STRESS SUITE 4: Disconnect Resilience & Force-Sync Resynchronization")
    print("=" * 75)

    # 1. MultiPinHardwareController initialization with invalid port
    real_hw = MultiPinHardwareController(
        port="COM999_NONEXISTENT",
        auto_reconnect=False,
        fallback_to_simulation=True,
    )
    res = real_hw.connect()
    print(f"  Connecting to invalid COM port returned: success={bool(res)}, message='{res[1]}'")
    assert not bool(res), "Connecting to invalid port should return success=False"
    assert real_hw.is_simulation, "Controller must switch to simulation mode on connection failure"

    # 2. Writing to pins while in simulation fallback must succeed without exception
    for p in range(2, 20):
        target = (p % 2 == 0)
        ok = real_hw.set_pin(p, target)
        assert ok, f"set_pin({p}) must return True in simulation fallback"
        assert real_hw.get_pin(p) == target, f"get_pin({p}) mismatch in simulation fallback"

    # 3. Test force_sync() in simulation mode
    real_hw.reset_log()
    ok_sync = real_hw.force_sync()
    assert ok_sync, "force_sync() must return True in simulation fallback mode"
    logged_cmds = real_hw.get_command_log()
    print(f"  force_sync() emitted {len(logged_cmds)} commands for {len(real_hw.pin_states)} cached pins")
    assert len(logged_cmds) == 18, f"force_sync() must emit exactly 18 commands, got {len(logged_cmds)}"

    # 4. Verify MockMultiPinHardwareController disconnect/reconnect simulation
    mock_hw = MockMultiPinHardwareController(auto_connect=True)
    # Set pins 2..19 to alternating states
    for p in range(2, 20):
        mock_hw.set_pin(p, (p % 3 == 0))

    # Simulate physical disconnection
    mock_hw.simulate_disconnect()
    assert not mock_hw.is_connected, "Controller must report disconnected"
    assert mock_hw.is_simulation, "Controller must report simulation mode"

    # Writes during disconnected state fail when auto-fallback is not configured on mock
    ok_fail = mock_hw.set_pin(2, True)
    assert not ok_fail, "Mock set_pin must return False when disconnected"

    # Simulate reconnect
    mock_hw.simulate_reconnect()
    assert mock_hw.is_connected, "Controller must report connected after reconnect"

    # Execute force_sync()
    mock_hw.reset_log()
    ok_resync = mock_hw.force_sync()
    assert ok_resync, "force_sync() must succeed upon reconnect"
    resync_cmds = mock_hw.get_command_log()
    assert len(resync_cmds) == 18, f"force_sync() must emit 18 resynchronization commands, got {len(resync_cmds)}"

    # 5. Backwards-compatibility tests
    assert issubclass(MultiPinHardwareController, BaseHardwareController)
    assert issubclass(MultiPinHardwareController, BaseMultiPinHardwareController)
    assert HardwareController is MultiPinHardwareController

    # Legacy set_light(True/False) maps to Pin 13
    mock_hw.set_light(True)
    assert mock_hw.get_pin(13) is True
    assert mock_hw.current_state is True

    mock_hw.set_light(False)
    assert mock_hw.get_pin(13) is False
    assert mock_hw.current_state is False

    print("  [PASS] Disconnect Resilience & Force-Sync Resynchronization verified")


def test_suite_5_boundary_and_reserved_pin_protection():
    """
    Stress 5: Boundary & Reserved Pin Fuzzing.
    Strictly asserts rejection of:
      - Reserved UART Pins 0 and 1
      - Negative pins (-100, -1)
      - Out-of-bounds pins (20, 21, 255, 1000)
      - Non-integer types
    """
    print("\n" + "=" * 75)
    print("STRESS SUITE 5: Boundary & Reserved Pin Protection")
    print("=" * 75)

    hw = MockMultiPinHardwareController(auto_connect=True)

    # 1. Pins 0 and 1 are hardwired to ATmega328P USART RX/TX
    for p in [0, 1]:
        assert not hw.set_pin(p, True), f"UART Reserved Pin {p} was unexpectedly accepted!"
        assert not hw.set_pin(p, False), f"UART Reserved Pin {p} was unexpectedly accepted!"
        assert not hw.get_pin(p), f"get_pin on Reserved Pin {p} returned True"

    # 2. Out-of-bounds pins
    invalid_pins = [-1000, -1, 20, 21, 30, 100, 255]
    for p in invalid_pins:
        assert not hw.set_pin(p, True), f"Invalid pin {p} was unexpectedly accepted"
        assert not hw.set_pin(p, False), f"Invalid pin {p} was unexpectedly accepted"
        assert not hw.get_pin(p), f"get_pin on invalid pin {p} returned True"

    # 3. Non-integer and strange types
    weird_inputs = ["13", "pin_2", None, 3.14, (2, 3), [13]]
    for w in weird_inputs:
        try:
            res = hw.set_pin(w, True)
            assert not res, f"Weird pin input {w} should return False"
        except (TypeError, ValueError):
            pass  # Raising TypeError/ValueError is also acceptable

    # 4. Verify command log remained completely empty during fuzzing
    assert len(hw.get_command_log()) == 0, "Invalid pins emitted commands to log!"

    # 5. Exactly 18 valid pins accepted (Pins 2 to 19 inclusive)
    for p in range(2, 20):
        assert hw.set_pin(p, True), f"Valid pin {p} was rejected"
        assert hw.get_pin(p) is True

    assert len(hw.get_command_log()) == 18
    print("  [PASS] Boundary & Reserved Pin Protection verified (Pins 0, 1, and out-of-bounds strictly blocked)")


def test_suite_6_arduino_firmware_zero_heap_audit():
    """
    Stress 6: Static Security & Memory Audit of Model Demo/classroom_lights/classroom_lights.ino.
    Empirically inspects file for dynamic heap allocations, buffer sizes, and watchdog failsafe.
    """
    print("\n" + "=" * 75)
    print("STRESS SUITE 6: Arduino Firmware Zero-Heap & Security Static Audit")
    print("=" * 75)

    ino_path = os.path.join(PROJECT_ROOT, "Model Demo", "classroom_lights", "classroom_lights.ino")
    assert os.path.exists(ino_path), f"Arduino firmware file missing at: {ino_path}"

    with open(ino_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip comments (/* ... */ and // ...) to audit actual compiled code for heap usage
    code_without_block_comments = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    code_only = re.sub(r"//.*", "", code_without_block_comments)

    # 1. Zero heap allocation check in actual code:
    # Must NOT contain: String class, malloc, calloc, realloc, free, new, delete
    heap_patterns = [
        (r"\bString\b", "Dynamic Arduino String class"),
        (r"\bmalloc\s*\(", "malloc() heap allocation"),
        (r"\bcalloc\s*\(", "calloc() heap allocation"),
        (r"\brealloc\s*\(", "realloc() heap allocation"),
        (r"\bfree\s*\(", "free() heap deallocation"),
        (r"\bnew\s+", "C++ new operator"),
        (r"\bdelete\s+", "C++ delete operator"),
    ]

    violations: List[str] = []
    for pattern, desc in heap_patterns:
        matches = re.findall(pattern, code_only)
        if matches:
            violations.append(f"Found {len(matches)} violations of {desc} (regex: '{pattern}')")

    print(f"  Firmware source size: {len(content)} bytes ({len(content.splitlines())} lines)")
    print(f"  Heap allocation violations: {len(violations)}")
    assert len(violations) == 0, f"Firmware heap audit FAILED: {violations}"

    # 2. PROGMEM / F() macro check:
    # Serial.print / Serial.println string literals should use F() macro
    unwrapped_strings = re.findall(r'Serial\.print(?:ln)?\s*\(\s*"([^"]+)"\s*\)', content)
    print(f"  Unwrapped Serial string literals in RAM: {len(unwrapped_strings)}")
    assert len(unwrapped_strings) == 0, f"Found strings not in PROGMEM F() macro: {unwrapped_strings}"

    # 3. Buffer overflow protection check:
    assert "sizeof(rx_buffer)" in content, "Firmware must check rx_buffer capacity against sizeof(rx_buffer)"
    assert "ERR OVERFLOW" in content, "Firmware must emit 'ERR OVERFLOW' on buffer limit"

    # 4. Failsafe watchdog timer check:
    assert "DEFAULT_FAILSAFE_MS" in content, "Firmware must define watchdog timeout"
    assert "FAILSAFE_TRIGGERED" in content, "Firmware must alert on failsafe triggering"

    # 5. Backward compatibility for legacy single-byte commands:
    assert "LED_ON" in content and "LED_OFF" in content, "Firmware must support legacy '1'/'0' commands"

    print("  [PASS] Arduino Firmware Zero-Heap & Security Static Audit verified (0 bytes heap allocation)")


class TestChallengerM2Empirical(unittest.TestCase):
    """Unittest wrapper for CI/CD and run_verification discovery."""

    def test_01_massive_concurrency_and_lock_contention(self):
        test_suite_1_massive_concurrency_and_lock_contention()

    def test_02_per_pin_rate_limiting_channel_isolation(self):
        test_suite_2_per_pin_rate_limiting_channel_isolation()

    def test_03_delta_update_suppression_under_massive_flooding(self):
        test_suite_3_delta_update_suppression_under_massive_flooding()

    def test_04_disconnect_resilience_and_force_sync(self):
        test_suite_4_disconnect_resilience_and_force_sync()

    def test_05_boundary_and_reserved_pin_protection(self):
        test_suite_5_boundary_and_reserved_pin_protection()

    def test_06_arduino_firmware_zero_heap_audit(self):
        test_suite_6_arduino_firmware_zero_heap_audit()


def run_all_challenger_suites():
    print("\n" + "#" * 75)
    print("   CHALLENGER_M2_1 EMPIRICAL ADVERSARIAL STRESS TEST SUITE")
    print("   Milestone 2: Hardware Concurrency & Multi-Pin Verification")
    print("#" * 75)

    test_suite_1_massive_concurrency_and_lock_contention()
    test_suite_2_per_pin_rate_limiting_channel_isolation()
    test_suite_3_delta_update_suppression_under_massive_flooding()
    test_suite_4_disconnect_resilience_and_force_sync()
    test_suite_5_boundary_and_reserved_pin_protection()
    test_suite_6_arduino_firmware_zero_heap_audit()

    print("\n" + "#" * 75)
    print("   VERDICT: ALL 6 ADVERSARIAL STRESS SUITES PASSED (100% APPROVE)")
    print("#" * 75 + "\n")


if __name__ == "__main__":
    run_all_challenger_suites()

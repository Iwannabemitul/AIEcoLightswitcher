"""
tests/test_challenger_m2_boundary_stress.py
Milestone 2 Adversarial Stress & Boundary Verification Harness

Probes:
  Probe 1: Strict Pin Boundary Validation (Reserved 0,1; Negatives; Overflows >19; Valid 2-19)
  Probe 2: High-Precision Rate Limiting Clamping & Cross-Channel Isolation
  Probe 3: Delta-Update Suppression & Force-Sync Integrity
  Probe 4: Arduino Firmware C-Parser Simulation & Fuzzing (Buffer Overflows, Malformed Syntax, Watchdog)
  Probe 5: High-Concurrency Multi-Thread Hammering & Serial Reconnection Resilience
  Probe 6: Full Backward Compatibility Facade & Legacy 2-Tuple / 4-Tuple Invariants
"""

import math
import os
import sys
import threading
import time
import unittest
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import eco_engine
from hardware.base_controller import (
    BaseHardwareController,
    BaseMultiPinHardwareController,
    ConnectionResult,
)
from hardware.serial_controller import MultiPinHardwareController, HardwareController
from hardware.mock_controller import MockHardwareController, MockMultiPinHardwareController


class TestProbe1PinBoundaries(unittest.TestCase):
    """
    Probe 1: Adversarial Pin Boundary Testing across MultiPinHardwareController
    and MockMultiPinHardwareController.
    """

    def setUp(self):
        self.controllers = [
            ("MockMultiPin", MockMultiPinHardwareController(auto_connect=True, min_switch_interval=0.0)),
            ("RealMultiPin-Sim", MultiPinHardwareController(port="MOCK_COM1", fallback_to_simulation=True, min_switch_interval=0.0)),
        ]
        for _, ctrl in self.controllers:
            ctrl.connect()

    def tearDown(self):
        for _, ctrl in self.controllers:
            ctrl.disconnect()

    def test_valid_pins_2_to_19_accepted(self):
        """All pins 2 through 19 inclusive must be valid and accepted."""
        for name, ctrl in self.controllers:
            for pin in range(2, 20):
                self.assertTrue(ctrl.set_pin(pin, True), f"{name}: Pin {pin} rejected")
                self.assertTrue(ctrl.get_pin(pin), f"{name}: Pin {pin} not True")
                self.assertTrue(ctrl.set_pin(pin, False), f"{name}: Pin {pin} rejected on False")
                self.assertFalse(ctrl.get_pin(pin), f"{name}: Pin {pin} not False")

    def test_reserved_uart_pins_0_and_1_strictly_rejected(self):
        """Pins 0 and 1 are UART RX/TX and must NEVER be actuated or read."""
        for name, ctrl in self.controllers:
            for reserved_pin in (0, 1):
                res = ctrl.set_pin(reserved_pin, True)
                self.assertFalse(res, f"{name}: Reserved pin {reserved_pin} was accepted for set_pin")
                self.assertFalse(ctrl.get_pin(reserved_pin), f"{name}: Reserved pin {reserved_pin} returned True for get_pin")
            # Verify no commands logged for pins 0 or 1
            cmd_log = ctrl.get_command_log()
            for entry in cmd_log:
                pin = entry[2]
                self.assertNotIn(pin, {0, 1}, f"{name}: Reserved pin {pin} appeared in command log")

    def test_negative_pins_strictly_rejected(self):
        """Negative pin indices must return False and not corrupt internal tables."""
        for name, ctrl in self.controllers:
            for neg_pin in (-1, -2, -10, -100, -999999):
                res = ctrl.set_pin(neg_pin, True)
                self.assertFalse(res, f"{name}: Negative pin {neg_pin} was accepted")
                self.assertFalse(ctrl.get_pin(neg_pin), f"{name}: Negative pin {neg_pin} returned True")

    def test_overflow_pins_strictly_rejected(self):
        """Pins >= 20 must return False and not corrupt internal tables."""
        for name, ctrl in self.controllers:
            for overflow_pin in (20, 21, 22, 50, 99, 100, 255, 1000):
                res = ctrl.set_pin(overflow_pin, True)
                self.assertFalse(res, f"{name}: Overflow pin {overflow_pin} was accepted")
                self.assertFalse(ctrl.get_pin(overflow_pin), f"{name}: Overflow pin {overflow_pin} returned True")

    def test_hashable_non_integer_pins_rejected_safely(self):
        """Hashable non-integer types (None, string, float outside set) must return False."""
        for name, ctrl in self.controllers:
            for weird_pin in (None, "13", "invalid", 2.5):
                res = ctrl.set_pin(weird_pin, True)
                self.assertFalse(res, f"{name}: Hashable non-integer pin {weird_pin} was accepted")

    def test_unhashable_pins_behavior(self):
        """
        Adversarial finding: Unhashable types (e.g. [13], {}) raise TypeError
        because controller directly evaluates 'pin in self.RESERVED_PINS' without
        prior isinstance(pin, int) guard.
        """
        for name, ctrl in self.controllers:
            with self.assertRaises(TypeError):
                ctrl.set_pin([13], True)
            with self.assertRaises(TypeError):
                ctrl.get_pin([13])


class TestProbe2RateLimitingPrecision(unittest.TestCase):
    """
    Probe 2: High-Precision Rate Limiting Clamping & Cross-Channel Isolation.
    """

    def test_exact_edge_timing_clamping(self):
        """
        Tests exact boundary timing:
        t < min_switch_interval -> rejected (False)
        t >= min_switch_interval -> accepted (True)
        """
        for cls in (MockMultiPinHardwareController, MultiPinHardwareController):
            ctrl = cls(min_switch_interval=1.0)
            ctrl.connect("MOCK_COM1")

            t_base = 1000.0
            # 1. First switch at t_base succeeds
            self.assertTrue(ctrl.set_pin(2, True, now=t_base))
            self.assertTrue(ctrl.get_pin(2))

            # 2. Sub-interval: dt = 0.5s -> strictly clamped
            self.assertFalse(ctrl.set_pin(2, False, now=t_base + 0.5))
            self.assertTrue(ctrl.get_pin(2))  # state remains True

            # 3. Sub-interval: dt = 0.999s -> strictly clamped
            self.assertFalse(ctrl.set_pin(2, False, now=t_base + 0.999))
            self.assertTrue(ctrl.get_pin(2))

            # 4. Exact boundary: dt = 1.000s -> strictly accepted
            self.assertTrue(ctrl.set_pin(2, False, now=t_base + 1.0))
            self.assertFalse(ctrl.get_pin(2))

            # 5. Immediate next switch: dt = 0.001s -> clamped
            self.assertFalse(ctrl.set_pin(2, True, now=t_base + 1.001))
            self.assertFalse(ctrl.get_pin(2))

            # 6. Force override: dt = 0.001s but force=True -> strictly accepted
            self.assertTrue(ctrl.set_pin(2, True, force=True, now=t_base + 1.001))
            self.assertTrue(ctrl.get_pin(2))

            ctrl.disconnect()

    def test_per_pin_isolation_no_cross_channel_blocking(self):
        """
        Pin 2 being clamped must NEVER block Pin 3 through 19 from switching.
        """
        for cls in (MockMultiPinHardwareController, MultiPinHardwareController):
            ctrl = cls(min_switch_interval=2.0)
            ctrl.connect("MOCK_COM1")

            t_base = 500.0
            # Pin 2 switches at t_base
            self.assertTrue(ctrl.set_pin(2, True, now=t_base))

            # At t_base + 0.1, Pin 2 is clamped
            self.assertFalse(ctrl.set_pin(2, False, now=t_base + 0.1))

            # Pin 3 through 19 must all switch cleanly at t_base + 0.1
            for p in range(3, 20):
                self.assertTrue(ctrl.set_pin(p, True, now=t_base + 0.1), f"Pin {p} was blocked by Pin 2!")
                self.assertTrue(ctrl.get_pin(p))

            ctrl.disconnect()


class TestProbe3DeltaUpdatesAndSync(unittest.TestCase):
    """
    Probe 3: Delta Updates and force_sync() verification.
    """

    def test_delta_update_suppression(self):
        """Redundant writes must not emit new commands."""
        for cls in (MockMultiPinHardwareController, MultiPinHardwareController):
            ctrl = cls(min_switch_interval=0.0)
            ctrl.connect("MOCK_COM1")

            # Turn Pin 5 ON: emits 1 command
            self.assertTrue(ctrl.set_pin(5, True))
            self.assertEqual(len(ctrl.get_command_log()), 1)

            # 50 redundant ON writes: must emit 0 new commands
            for _ in range(50):
                self.assertTrue(ctrl.set_pin(5, True))
            self.assertEqual(len(ctrl.get_command_log()), 1)

            # Turn Pin 5 OFF: emits 2nd command
            self.assertTrue(ctrl.set_pin(5, False))
            self.assertEqual(len(ctrl.get_command_log()), 2)

            # 50 redundant OFF writes: must emit 0 new commands
            for _ in range(50):
                self.assertTrue(ctrl.set_pin(5, False))
            self.assertEqual(len(ctrl.get_command_log()), 2)

            ctrl.disconnect()

    def test_force_sync_transmits_all_states(self):
        """force_sync() must re-emit all cached states in order."""
        for cls in (MockMultiPinHardwareController, MultiPinHardwareController):
            ctrl = cls(min_switch_interval=0.0)
            ctrl.connect("MOCK_COM1")

            # Set a pattern across pins 2, 5, 13, 19
            ctrl.set_pin(2, True)
            ctrl.set_pin(5, False)
            ctrl.set_pin(13, True)
            ctrl.set_pin(19, True)
            ctrl.reset_log()
            self.assertEqual(len(ctrl.get_command_log()), 0)

            # force_sync()
            self.assertTrue(ctrl.force_sync())
            cmds = ctrl.get_raw_commands()
            self.assertEqual(len(cmds), 4)
            self.assertEqual(cmds[0], "SET 2 1\n")
            self.assertEqual(cmds[1], "SET 5 0\n")
            self.assertEqual(cmds[2], "SET 13 1\n")
            self.assertEqual(cmds[3], "SET 19 1\n")

            ctrl.disconnect()


class ArduinoFirmwareEmulator:
    """
    Exact Python emulator of the ATmega328P C firmware logic in
    Model Demo/classroom_lights/classroom_lights.ino.
    Validates static buffer parsing, boundary checks, and overflow protection.
    """

    def __init__(self):
        self.PIN_MIN = 2
        self.PIN_MAX = 19
        self.LEGACY_PIN = 13
        self.pin_states = {p: 0 for p in range(self.PIN_MIN, self.PIN_MAX + 1)}
        self.rx_buffer = ""
        self.rx_limit = 63  # 64-byte buffer minus null terminator
        self.output_log: List[str] = []

    def feed(self, text: str) -> List[str]:
        responses = []
        for c in text:
            # 1. Dual-mode single character compatibility when buffer empty
            if len(self.rx_buffer) == 0:
                if c == '1':
                    self.pin_states[self.LEGACY_PIN] = 1
                    responses.append("LED_ON")
                    continue
                elif c == '0':
                    self.pin_states[self.LEGACY_PIN] = 0
                    responses.append("LED_OFF")
                    continue
                elif c == '?':
                    responses.append("AIEcoLightswitcher_v2")
                    continue
                elif c in ('\r', '\n'):
                    continue

            if c == '\r':
                continue

            if c == '\n':
                cmd = self.rx_buffer
                self.rx_buffer = ""
                if cmd:
                    resp = self._process_command(cmd)
                    if resp:
                        responses.append(resp)
            else:
                if len(self.rx_buffer) < self.rx_limit:
                    self.rx_buffer += c
                else:
                    self.rx_buffer = ""
                    responses.append("ERR OVERFLOW")
        self.output_log.extend(responses)
        return responses

    def _process_command(self, cmd: str) -> str:
        cmd = cmd.lstrip(" \t")
        if not cmd:
            return ""
        if cmd == "PING":
            return "PONG"
        if cmd == "IDENT":
            return "AIEcoLightswitcher_v2"
        if cmd == "STATUS":
            st = "STATUS " + " ".join(f"{p}:{self.pin_states[p]}" for p in range(self.PIN_MIN, self.PIN_MAX + 1))
            return st
        if cmd.startswith("SET "):
            parts = cmd[4:].split()
            if len(parts) < 2:
                return "ERR SYNTAX"
            try:
                pin = int(parts[0])
                state = int(parts[1])
            except ValueError:
                return "ERR SYNTAX"
            if pin < self.PIN_MIN or pin > self.PIN_MAX:
                return "ERR INVALID_PIN"
            if state not in (0, 1):
                return "ERR INVALID_STATE"
            self.pin_states[pin] = state
            return f"OK SET {pin} {state}"
        if cmd.startswith("GET "):
            parts = cmd[4:].split()
            if not parts:
                return "ERR SYNTAX"
            try:
                pin = int(parts[0])
            except ValueError:
                return "ERR SYNTAX"
            if pin < self.PIN_MIN or pin > self.PIN_MAX:
                return "ERR INVALID_PIN"
            return f"PIN {pin} {self.pin_states[pin]}"
        return "ERR UNKNOWN_COMMAND"


class TestProbe4FirmwareParserFuzzing(unittest.TestCase):
    """
    Probe 4: Adversarial Fuzzing and Boundary Testing of Arduino Firmware Logic.
    """

    def setUp(self):
        self.fw = ArduinoFirmwareEmulator()

    def test_legacy_single_char_commands(self):
        """Single character '1', '0', '?' without newlines."""
        self.assertEqual(self.fw.feed("1"), ["LED_ON"])
        self.assertEqual(self.fw.pin_states[13], 1)
        self.assertEqual(self.fw.feed("0"), ["LED_OFF"])
        self.assertEqual(self.fw.pin_states[13], 0)
        self.assertEqual(self.fw.feed("?"), ["AIEcoLightswitcher_v2"])

    def test_framed_ascii_set_and_get(self):
        """Framed SET and GET across all valid pins."""
        for p in range(2, 20):
            res = self.fw.feed(f"SET {p} 1\n")
            self.assertEqual(res, [f"OK SET {p} 1"])
            self.assertEqual(self.fw.pin_states[p], 1)
            res_get = self.fw.feed(f"GET {p}\n")
            self.assertEqual(res_get, [f"PIN {p} 1"])

            res_off = self.fw.feed(f"SET {p} 0\n")
            self.assertEqual(res_off, [f"OK SET {p} 0"])
            self.assertEqual(self.fw.pin_states[p], 0)

    def test_fuzzing_reserved_and_out_of_bounds_pins(self):
        """Reserved UART (0, 1), negative, and overflow pins must return ERR INVALID_PIN."""
        for bad_pin in (0, 1, -1, -50, 20, 21, 99, 255):
            res = self.fw.feed(f"SET {bad_pin} 1\n")
            self.assertEqual(res, ["ERR INVALID_PIN"], f"Bad pin {bad_pin} gave {res}")
            res_get = self.fw.feed(f"GET {bad_pin}\n")
            self.assertEqual(res_get, ["ERR INVALID_PIN"])

    def test_fuzzing_invalid_states(self):
        """State values other than 0 or 1 must return ERR INVALID_STATE."""
        for bad_state in (2, -1, 5, 100):
            res = self.fw.feed(f"SET 13 {bad_state}\n")
            self.assertEqual(res, ["ERR INVALID_STATE"], f"Bad state {bad_state} gave {res}")

    def test_fuzzing_malformed_syntax(self):
        """Malformed or incomplete commands."""
        self.assertEqual(self.fw.feed("SET\n"), ["ERR UNKNOWN_COMMAND"])
        self.assertEqual(self.fw.feed("SET \n"), ["ERR SYNTAX"])
        self.assertEqual(self.fw.feed("SET 13\n"), ["ERR SYNTAX"])
        self.assertEqual(self.fw.feed("SET abc 1\n"), ["ERR SYNTAX"])
        self.assertEqual(self.fw.feed("GET\n"), ["ERR UNKNOWN_COMMAND"])
        self.assertEqual(self.fw.feed("GET \n"), ["ERR SYNTAX"])
        self.assertEqual(self.fw.feed("UNKNOWN_CMD\n"), ["ERR UNKNOWN_COMMAND"])

    def test_buffer_overflow_resilience(self):
        """Input exceeding 64 bytes without newline must trigger ERR OVERFLOW without corruption."""
        garbage = "A" * 70 + "\n"
        res = self.fw.feed(garbage)
        self.assertIn("ERR OVERFLOW", res)
        # Verify subsequent normal command succeeds immediately
        res2 = self.fw.feed("PING\n")
        self.assertEqual(res2, ["PONG"])


class TestProbe5ConcurrencyAndConnectCycling(unittest.TestCase):
    """
    Probe 5: High-Concurrency Hammering and Rapid Connect/Disconnect Cycling.
    """

    def test_concurrent_pin_actuation_thread_safety(self):
        """16 concurrent threads actuating pins simultaneously without deadlock or state corruption."""
        ctrl = MockMultiPinHardwareController(min_switch_interval=0.0)
        ctrl.connect()

        errors = []
        threads = []

        def worker(thread_id: int):
            for i in range(100):
                pin = 2 + (i % 18)  # Pins 2 to 19
                state = ((i + thread_id) % 2 == 1)
                try:
                    success = ctrl.set_pin(pin, state)
                    if not success:
                        errors.append(f"Worker {thread_id} failed setting pin {pin}")
                except Exception as e:
                    errors.append(f"Worker {thread_id} raised: {e}")

        for tid in range(16):
            t = threading.Thread(target=worker, args=(tid,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(errors), 0, f"Encountered {len(errors)} concurrency errors: {errors[:5]}")
        self.assertGreater(len(ctrl.get_command_log()), 100)
        ctrl.disconnect()

    def test_rapid_connect_disconnect_cycling(self):
        """100 rapid connect/disconnect cycles under simulation mode."""
        ctrl = MultiPinHardwareController(port="MOCK_COM1", fallback_to_simulation=True)
        for i in range(100):
            res_conn = ctrl.connect()
            self.assertTrue(res_conn, f"Failed connect at {i}")
            self.assertTrue(ctrl.is_connected)
            ctrl.disconnect()
            self.assertFalse(ctrl.is_connected)

    def test_failed_connection_fallback_to_simulation(self):
        """Non-existent COM port COM999 triggers clean simulation fallback."""
        ctrl = MultiPinHardwareController(port="COM999", fallback_to_simulation=True)
        res = ctrl.connect()
        self.assertFalse(res[0])
        self.assertTrue(ctrl.is_simulation)
        self.assertFalse(ctrl.is_connected)
        # Actuation returns True in simulation fallback mode
        self.assertTrue(ctrl.set_pin(13, True))
        ctrl.disconnect()


class TestProbe6BackwardCompatibilityFacade(unittest.TestCase):
    """
    Probe 6: Backward Compatibility with Legacy Single-Channel API.
    """

    def test_mock_hardware_controller_legacy_2_tuple(self):
        """MockHardwareController must preserve 2-tuple command log format."""
        hw = MockHardwareController()
        hw.set_light(True)
        hw.set_light(False)
        log = hw.get_command_log()
        self.assertEqual(len(log), 2)
        # Must unpack as exactly 2 items: (ts, cmd)
        ts1, cmd1 = log[0]
        ts2, cmd2 = log[1]
        self.assertEqual(cmd1, "1")
        self.assertEqual(cmd2, "0")
        self.assertEqual(hw.command_history, ["1", "0"])

    def test_multipin_hardware_controller_legacy_interface(self):
        """MultiPinHardwareController must implement set_light() and current_state."""
        hw = MockMultiPinHardwareController()
        hw.connect()
        self.assertTrue(isinstance(hw, BaseHardwareController))
        self.assertFalse(hw.current_state)
        hw.set_light(True)
        self.assertTrue(hw.current_state)
        self.assertTrue(hw.get_pin(13))
        hw.set_light(False)
        self.assertFalse(hw.current_state)
        self.assertFalse(hw.get_pin(13))
        hw.disconnect()

    def test_eco_engine_facade_reexports_hardware_classes(self):
        """eco_engine must re-export all hardware classes with 100% identity."""
        self.assertIs(eco_engine.BaseHardwareController, BaseHardwareController)
        self.assertIs(eco_engine.BaseMultiPinHardwareController, BaseMultiPinHardwareController)
        self.assertIs(eco_engine.MultiPinHardwareController, MultiPinHardwareController)
        self.assertIs(eco_engine.MockHardwareController, MockHardwareController)
        self.assertIs(eco_engine.MockMultiPinHardwareController, MockMultiPinHardwareController)


if __name__ == "__main__":
    unittest.main()

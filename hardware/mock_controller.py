"""
hardware/mock_controller.py - In-Memory Hardware Controller Test Doubles
Part of AIEcoLightswitcher Modular Architecture (Milestone 2)

Provides:
  1. MockHardwareController: Legacy single-channel controller with 2-tuple command log
     (timestamp, '1'/'0') for 100% backward compatibility with test_verification_suite.py.
  2. MockMultiPinHardwareController: High-performance multi-pin controller simulator
     with 4-tuple command log (timestamp, cmd, pin, state), delta updates, per-pin rate limiting,
     and fault injection controls for multi-feed testing.
"""

import random
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from hardware.base_controller import (
    BaseHardwareController,
    BaseMultiPinHardwareController,
    ConnectionResult,
)


class MockHardwareController(BaseHardwareController):
    """
    In-memory hardware controller recording timestamps and single-character
    commands ('1' and '0') for legacy automated testing without physical Arduino hardware.
    Preserves exact 2-tuple command log format for test_verification_suite.py.
    """

    def __init__(self, port: str = "MOCK_COM1", auto_connect: bool = True):
        self._port = port
        self._is_connected = bool(auto_connect)
        self._current_state = False
        self.command_log: List[Tuple[float, str]] = []
        self.lock = threading.RLock()

    @property
    def is_connected(self) -> bool:
        with self.lock:
            return self._is_connected

    @property
    def current_state(self) -> bool:
        with self.lock:
            return self._current_state

    @property
    def is_simulation(self) -> bool:
        return True

    @property
    def command_history(self) -> List[str]:
        """Returns sequence of commands ['1', '0', ...] without timestamps."""
        with self.lock:
            return [cmd for _, cmd in self.command_log]

    def connect(self, port: Optional[str] = None) -> ConnectionResult:
        with self.lock:
            if port:
                self._port = port
            self._is_connected = True
            return ConnectionResult(True, f"Connected to {self._port}")

    def disconnect(self) -> None:
        with self.lock:
            self._is_connected = False

    def set_light(self, state: bool) -> bool:
        """Records state transition command ('1' or '0') on edge change."""
        with self.lock:
            if not self._is_connected:
                return False
            if state != self._current_state:
                cmd = "1" if state else "0"
                self.command_log.append((time.time(), cmd))
                self._current_state = state
            return True

    def get_command_log(self) -> List[Tuple[float, str]]:
        """Returns copy of recorded [(timestamp, '1'), ...] command log."""
        with self.lock:
            return list(self.command_log)

    def reset_log(self) -> None:
        """Clears recorded command history."""
        with self.lock:
            self.command_log.clear()


class MockMultiPinHardwareController(BaseMultiPinHardwareController):
    """
    In-memory multi-pin hardware controller simulator conforming to
    BaseMultiPinHardwareController (PROJECT.md:190-215).
    Maintains 4-tuple command log (timestamp, cmd, pin, state), supports
    delta-update filtering, independent per-pin rate limiting, and fault injection.
    """

    VALID_PINS = set(range(2, 20))  # Pins 2 to 19 (including 14-19 / A0-A5)
    RESERVED_PINS = {0, 1}          # Pins 0 and 1 reserved for UART RX/TX

    def __init__(
        self,
        port: str = "MOCK_COM1",
        auto_connect: bool = True,
        min_switch_interval: float = 0.0,
        simulated_latency_ms: float = 0.0,
        error_rate: float = 0.0,
    ):
        self._port = port
        self._is_connected = bool(auto_connect)
        self._is_simulation = True
        self.min_switch_interval = float(min_switch_interval)
        self.simulated_latency_ms = float(simulated_latency_ms)
        self.error_rate = float(error_rate)
        self.injected_errors_count: int = 0

        self.pin_states: Dict[int, bool] = {}
        self.last_switch_time: Dict[int, float] = {}
        self.command_log: List[Tuple[float, str, int, bool]] = []
        self.lock = threading.RLock()

    def connect(self, port: Optional[str] = None) -> ConnectionResult:
        with self.lock:
            if port:
                self._port = port
            self._is_connected = True
            return ConnectionResult(True, f"Connected to {self._port}")

    def disconnect(self) -> None:
        with self.lock:
            self._is_connected = False

    def set_pin(
        self,
        pin: int,
        state: Union[bool, int],
        force: bool = False,
        now: Optional[float] = None,
    ) -> bool:
        with self.lock:
            # 1. Pin validation: strictly reject reserved pins 0/1 and out-of-bounds pins
            if pin in self.RESERVED_PINS or pin not in self.VALID_PINS:
                return False

            # 2. Connection check
            if not self._is_connected:
                return False

            # 3. Fault injection hooks
            if self.injected_errors_count > 0:
                self.injected_errors_count -= 1
                return False
            if self.error_rate > 0.0 and random.random() < self.error_rate:
                return False

            # 4. Latency simulation
            if self.simulated_latency_ms > 0:
                time.sleep(self.simulated_latency_ms / 1000.0)

            bool_state = bool(state)
            t = time.time() if now is None else float(now)
            curr = self.pin_states.get(pin, False)

            if not force:
                # 5. Delta-update check: suppress redundant command if state hasn't changed
                if pin in self.pin_states and bool_state == curr:
                    return True

                # 6. Per-pin rate limiting clamp
                last_t = self.last_switch_time.get(pin, None)
                if last_t is not None and (t - last_t) < self.min_switch_interval:
                    return False

            # 7. Commit state and log command
            self.pin_states[pin] = bool_state
            self.last_switch_time[pin] = t
            cmd = f"SET {pin} {1 if bool_state else 0}\n"
            self.command_log.append((t, cmd, pin, bool_state))
            return True

    def get_pin(self, pin: int) -> bool:
        with self.lock:
            if pin in self.RESERVED_PINS or pin not in self.VALID_PINS:
                return False
            return self.pin_states.get(pin, False)

    def set_all(self, state: Union[bool, int, Dict[int, Union[bool, int]]]) -> bool:
        with self.lock:
            if not self._is_connected:
                return False
            if isinstance(state, dict):
                for pin, val in state.items():
                    self.set_pin(pin, bool(val), force=True)
            else:
                targets = sorted(self.pin_states.keys()) if self.pin_states else sorted(self.VALID_PINS)
                for pin in targets:
                    self.set_pin(pin, bool(state), force=True)
            return True

    def force_sync(self) -> bool:
        with self.lock:
            if not self._is_connected:
                return False
            for pin in sorted(self.pin_states.keys()):
                st = self.pin_states[pin]
                cmd = f"SET {pin} {1 if st else 0}\n"
                self.command_log.append((time.time(), cmd, pin, st))
            return True

    @property
    def is_connected(self) -> bool:
        with self.lock:
            return self._is_connected

    @property
    def is_simulation(self) -> bool:
        return self._is_simulation

    def get_command_log(self) -> List[Tuple[float, str, int, bool]]:
        with self.lock:
            return list(self.command_log)

    def get_raw_commands(self) -> List[str]:
        with self.lock:
            return [cmd for _, cmd, _, _ in self.command_log]

    @property
    def command_history(self) -> List[str]:
        """Returns sequence of raw command strings without timestamps."""
        return self.get_raw_commands()

    def reset_log(self) -> None:
        with self.lock:
            self.command_log.clear()

    def simulate_disconnect(self) -> None:
        with self.lock:
            self._is_connected = False
            self._is_simulation = True

    def simulate_reconnect(self) -> None:
        with self.lock:
            self._is_connected = True
            self._is_simulation = False

    def inject_error(self, count: int = 1) -> None:
        with self.lock:
            self.injected_errors_count += count

    def set_latency(self, latency_ms: float) -> None:
        with self.lock:
            self.simulated_latency_ms = float(latency_ms)

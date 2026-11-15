"""
hardware/serial_controller.py - Multi-Pin USB Serial Hardware Controller
Part of AIEcoLightswitcher Modular Architecture (Milestone 2)

Manages robust USB serial communication with Arduino Uno/Nano at 9600 baud.
Features:
  - Multi-pin routing across Pins 2–19 (standard D2–D13, A0–A5 as D14–D19).
  - Framed ASCII protocol (SET <pin> <state>\n) + legacy single-byte compatibility.
  - Delta-update optimization: suppresses UART writes for unchanging states.
  - Per-pin rate limiting: independent anti-chatter clamps per pin without cross-channel blocking.
  - Thread safety: re-entrant RLock protecting all serial operations and state tables.
  - Resilient simulation fallback: seamless fallback when hardware is unavailable.
  - Background auto-reconnect: automatic reconnection thread with full force_sync() recovery.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import serial
    import serial.tools.list_ports
    _PYSERIAL_AVAILABLE = True
except ImportError:
    serial = None
    _PYSERIAL_AVAILABLE = False

from hardware.base_controller import (
    BaseMultiPinHardwareController,
    ConnectionResult,
)

logger = logging.getLogger(__name__)


class MultiPinHardwareController(BaseMultiPinHardwareController):
    """
    Manages robust multi-pin USB serial communication with Arduino Uno/Nano at 9600 baud.
    Conforms to BaseMultiPinHardwareController and provides complete simulation fallback.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 9600,
        min_switch_interval: float = 1.5,
        auto_reconnect: bool = True,
        reconnect_interval: float = 2.0,
        timeout: float = 0.5,
        write_timeout: float = 0.5,
        fallback_to_simulation: bool = True,
    ):
        self._port: Optional[str] = port
        self.baudrate: int = int(baudrate)
        self.min_switch_interval: float = float(min_switch_interval)
        self.auto_reconnect: bool = bool(auto_reconnect)
        self.reconnect_interval: float = float(reconnect_interval)
        self.timeout: float = float(timeout)
        self.write_timeout: float = float(write_timeout)
        self.fallback_to_simulation: bool = bool(fallback_to_simulation)

        self.serial_conn: Optional[Any] = None
        self._is_connected: bool = False
        self._is_simulation: bool = False
        self._user_disconnected: bool = False

        # State tracking dictionaries
        self.pin_states: Dict[int, bool] = {}
        self.last_switch_time: Dict[int, float] = {}
        self.command_log: List[Tuple[float, str, int, bool]] = []

        # Thread synchronization
        self.lock = threading.RLock()
        self._stop_reconnect_event = threading.Event()
        self._reconnect_thread: Optional[threading.Thread] = None

    # ----------------------------------------------------------------------
    # Port Discovery & Diagnostics
    # ----------------------------------------------------------------------

    @staticmethod
    def list_available_ports() -> List[str]:
        """Returns detected serial COM ports on the system."""
        if not _PYSERIAL_AVAILABLE:
            return []
        try:
            ports = serial.tools.list_ports.comports()
            return [p.device for p in ports]
        except Exception as e:
            logger.debug(f"Error enumerating serial ports: {e}")
            return []

    # ----------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        with self.lock:
            if self._is_simulation:
                return self._is_connected
            return (
                self._is_connected
                and (self.serial_conn is not None)
                and getattr(self.serial_conn, "is_open", False)
            )

    @property
    def is_simulation(self) -> bool:
        with self.lock:
            return self._is_simulation

    @property
    def port(self) -> Optional[str]:
        with self.lock:
            return self._port

    @port.setter
    def port(self, val: Optional[str]) -> None:
        with self.lock:
            self._port = val

    @property
    def command_history(self) -> List[str]:
        """Returns sequence of raw command strings without timestamps."""
        return self.get_raw_commands()

    # ----------------------------------------------------------------------
    # Connection Lifecycle
    # ----------------------------------------------------------------------

    def connect(self, port: Optional[str] = None) -> ConnectionResult:
        """
        Connects to the physical hardware or simulation backend.
        Returns a ConnectionResult tuple (success, message).
        """
        with self.lock:
            self._user_disconnected = False
            if port:
                self._port = port

            # Handle Mock / Simulator ports
            if self._port and (
                self._port.startswith("MOCK") or self._port.upper() == "SIMULATION"
            ):
                self._is_connected = True
                self._is_simulation = True
                return ConnectionResult(True, f"Connected to {self._port} (Simulation)")

            # Auto-detect port if none specified
            if not self._port:
                available = self.list_available_ports()
                if available:
                    self._port = available[0]
                else:
                    self._is_connected = False
                    self._is_simulation = True
                    return ConnectionResult(
                        False,
                        "No serial COM ports found on system (Simulation fallback active)",
                    )

            # Check pySerial availability
            if not _PYSERIAL_AVAILABLE:
                self._is_connected = False
                self._is_simulation = True
                return ConnectionResult(
                    False, "pySerial is not installed (Simulation fallback active)"
                )

            # Attempt physical UART connection
            try:
                if self.serial_conn and getattr(self.serial_conn, "is_open", False):
                    try:
                        self.serial_conn.close()
                    except Exception:
                        pass

                self.serial_conn = serial.Serial(
                    self._port,
                    self.baudrate,
                    timeout=self.timeout,
                    write_timeout=self.write_timeout,
                )
                time.sleep(1.0)  # Bootloader stabilization delay

                # Drain inbound bootloader banner (e.g. "READY\r\n")
                try:
                    self.serial_conn.reset_input_buffer()
                except Exception:
                    pass

                self._is_connected = True
                self._is_simulation = False

                # Resynchronize pin states to hardware
                self.force_sync()

                # Start auto-reconnect monitoring thread if enabled
                if self.auto_reconnect:
                    self._start_reconnect_thread()

                return ConnectionResult(True, f"Connected to {self._port}")

            except Exception as e:
                self._is_connected = False
                self._is_simulation = True
                self.serial_conn = None
                return ConnectionResult(
                    False, f"Failed connecting to {self._port}: {e}"
                )

    def disconnect(self) -> None:
        """Gracefully disconnects hardware and cleans up active resources."""
        with self.lock:
            self._user_disconnected = True
            if self.serial_conn and getattr(self.serial_conn, "is_open", False):
                try:
                    # Turn off managed pins on clean disconnect
                    for pin in list(self.pin_states.keys()):
                        self._send_command_raw(f"SET {pin} 0\n")
                    self.serial_conn.close()
                except Exception:
                    pass
            self.serial_conn = None
            self._is_connected = False

    def close(self) -> None:
        """Closes hardware controller and stops background reconnect thread."""
        self._stop_reconnect_thread()
        self.disconnect()

    # ----------------------------------------------------------------------
    # Pin Actuation & Rate Limiting
    # ----------------------------------------------------------------------

    def set_pin(
        self,
        pin: int,
        state: Union[bool, int],
        force: bool = False,
        now: Optional[float] = None,
    ) -> bool:
        """
        Actuates an individual GPIO pin (Pins 2–19).
        Applies delta-update filtering and per-pin rate limiting unless force=True.
        """
        with self.lock:
            # 1. Pin validation: strictly reject reserved pins 0/1 and out-of-bounds pins
            if pin in self.RESERVED_PINS or pin not in self.VALID_PINS:
                return False

            bool_state = bool(state)
            t = time.time() if now is None else float(now)
            curr = self.pin_states.get(pin, False)

            if not force:
                # 2. Delta-update check: suppress redundant UART write if state is unchanged
                if pin in self.pin_states and bool_state == curr:
                    return True

                # 3. Independent per-pin rate limiting clamp
                last_t = self.last_switch_time.get(pin, None)
                if last_t is not None and (t - last_t) < self.min_switch_interval:
                    return False

            # 4. Update software pin state and rate limit timestamp
            self.pin_states[pin] = bool_state
            self.last_switch_time[pin] = t
            cmd = f"SET {pin} {1 if bool_state else 0}\n"
            self.command_log.append((t, cmd, pin, bool_state))

            # 5. Actuate hardware or simulation
            if self._is_connected and not self._is_simulation:
                success = self._send_command_raw(cmd)
                if not success:
                    return False
            elif not self._is_connected:
                if self._is_simulation and self.fallback_to_simulation:
                    return True
                return False

            return True

    def get_pin(self, pin: int) -> bool:
        """Queries the current known logical state of an individual pin."""
        with self.lock:
            if pin in self.RESERVED_PINS or pin not in self.VALID_PINS:
                return False
            return self.pin_states.get(pin, False)

    def set_all(self, state: Union[bool, int, Dict[int, Union[bool, int]]]) -> bool:
        """Sets all tracked/valid pins to the specified logical state."""
        with self.lock:
            if isinstance(state, dict):
                for pin, val in state.items():
                    self.set_pin(pin, bool(val), force=True)
            else:
                targets = sorted(self.pin_states.keys()) if self.pin_states else sorted(self.VALID_PINS)
                for pin in targets:
                    self.set_pin(pin, bool(state), force=True)
            return True

    def force_sync(self) -> bool:
        """Forces all cached pin states to be re-transmitted over the transport link."""
        with self.lock:
            if not self.is_connected and not (self._is_simulation and self.fallback_to_simulation):
                return False
            for pin in sorted(self.pin_states.keys()):
                st = self.pin_states[pin]
                cmd = f"SET {pin} {1 if st else 0}\n"
                self.command_log.append((time.time(), cmd, pin, st))
                if self.is_connected and not self._is_simulation:
                    self._send_command_raw(cmd)
            return True

    # ----------------------------------------------------------------------
    # Internal Helpers & Background Reconnect
    # ----------------------------------------------------------------------

    def _send_command_raw(self, cmd: str) -> bool:
        """Internal helper to transmit ASCII command over serial UART."""
        if not (self.serial_conn and getattr(self.serial_conn, "is_open", False)):
            return False
        try:
            self.serial_conn.write(cmd.encode("ascii"))
            self.serial_conn.flush()
            if self.serial_conn.in_waiting > 0:
                self.serial_conn.read_all()
            return True
        except Exception as e:
            logger.warning(f"[MultiPinHardwareController] UART write error: {e}")
            self._is_connected = False
            self._is_simulation = True
            return False

    def _start_reconnect_thread(self) -> None:
        """Launches daemon thread for automatic serial reconnection."""
        if self._reconnect_thread is None or not self._reconnect_thread.is_alive():
            self._stop_reconnect_event.clear()
            self._reconnect_thread = threading.Thread(
                target=self._auto_reconnect_loop,
                daemon=True,
                name="SerialAutoReconnectThread",
            )
            self._reconnect_thread.start()

    def _stop_reconnect_thread(self) -> None:
        """Signals and joins reconnect thread cleanly."""
        self._stop_reconnect_event.set()
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            self._reconnect_thread.join(timeout=0.2)
            self._reconnect_thread = None

    def _auto_reconnect_loop(self) -> None:
        """Background loop attempting to reconnect dropped serial link."""
        while not self._stop_reconnect_event.is_set():
            self._stop_reconnect_event.wait(self.reconnect_interval)
            if self._stop_reconnect_event.is_set():
                break

            with self.lock:
                if self._is_connected or self._user_disconnected:
                    continue
                if self._port and self._port.startswith("MOCK"):
                    continue
                target_port = self._port

            if not target_port:
                ports = self.list_available_ports()
                target_port = ports[0] if ports else None

            if target_port:
                try:
                    res = self.connect(target_port)
                    if res:
                        logger.info(f"Auto-reconnected to {target_port}; force_sync completed.")
                except Exception as e:
                    logger.debug(f"Auto-reconnect attempt failed on {target_port}: {e}")

    # ----------------------------------------------------------------------
    # Diagnostics & Command Logs
    # ----------------------------------------------------------------------

    def get_command_log(self) -> List[Tuple[float, str, int, bool]]:
        """Returns copy of recorded 4-tuple command log."""
        with self.lock:
            return list(self.command_log)

    def get_raw_commands(self) -> List[str]:
        """Returns list of raw command strings from log."""
        with self.lock:
            return [cmd for _, cmd, _, _ in self.command_log]

    def reset_log(self) -> None:
        """Clears recorded command history."""
        with self.lock:
            self.command_log.clear()


# Backward-compatible alias
HardwareController = MultiPinHardwareController

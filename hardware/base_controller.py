"""
hardware/base_controller.py - Abstract Multi-Pin Hardware Controller Contract
Part of AIEcoLightswitcher Modular Architecture (Milestone 2)

Defines the hardware abstraction layer interface contracts for both legacy
single-channel systems (Pin 13) and multi-pin digital GPIO routing (Pins 2-19).
"""

import abc
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from core.config import ConnectionResult
except ImportError:
    class ConnectionResult(tuple):
        def __new__(cls, success: bool, message: str = ""):
            return super().__new__(cls, (bool(success), str(message)))

        def __bool__(self) -> bool:
            return bool(self[0])


class BaseHardwareController(abc.ABC):
    """
    Abstract interface contract for physical and simulated hardware controllers.
    Preserved as legacy base class for 100% backward compatibility.
    """

    @abc.abstractmethod
    def connect(self, port: Optional[str] = None) -> Any:
        """Connects to the hardware controller or simulation backend."""
        raise NotImplementedError

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnects the hardware controller and releases resources."""
        raise NotImplementedError

    @abc.abstractmethod
    def set_light(self, state: bool) -> bool:
        """Actuates the primary output channel (Pin 13)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """True if the controller is actively connected."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def current_state(self) -> bool:
        """Returns the logical state of the primary output channel."""
        raise NotImplementedError

    @property
    def port(self) -> Optional[str]:
        """Returns the configured or active serial COM port."""
        return getattr(self, "_port", None)

    @port.setter
    def port(self, val: Optional[str]) -> None:
        self._port = val

    def close(self) -> None:
        """Alias for disconnect() to ensure standard resource cleanup."""
        self.disconnect()


class BaseMultiPinHardwareController(BaseHardwareController):
    """
    Abstract interface contract for physical and simulated multi-pin
    hardware controllers (PROJECT.md § Interface Contracts).

    Supports independent actuation across digital pins 2 through 19,
    including analog pins A0-A5 addressed as digital pins 14-19.
    """

    # Pins 2 to 19 inclusive (18 independent controllable channels: D2-D13, A0-A5 as D14-D19)
    VALID_PINS: Set[int] = set(range(2, 20))
    # Pins 0 and 1 are strictly reserved for ATmega328P hardware UART (RX/TX)
    RESERVED_PINS: Set[int] = {0, 1}

    @abc.abstractmethod
    def connect(self, port: Optional[str] = None) -> Union[bool, ConnectionResult]:
        """Connects to the hardware link or simulation backend."""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnects the controller and cleans up active resources."""
        pass

    @abc.abstractmethod
    def set_pin(
        self,
        pin: int,
        state: bool,
        force: bool = False,
        now: Optional[float] = None,
    ) -> bool:
        """
        Actuates an individual GPIO pin (Pins 2–19).
        Applies delta-update filtering and per-pin rate limiting unless force=True.
        """
        pass

    @abc.abstractmethod
    def get_pin(self, pin: int) -> bool:
        """Queries the current known logical state of an individual pin."""
        pass

    @abc.abstractmethod
    def set_all(self, state: Union[bool, int, Dict[int, Union[bool, int]]]) -> bool:
        """Sets all tracked/valid pins to the specified logical state."""
        pass

    @abc.abstractmethod
    def force_sync(self) -> bool:
        """Forces all cached pin states to be re-transmitted over the transport link."""
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """True if actively connected to the hardware link."""
        pass

    @property
    @abc.abstractmethod
    def is_simulation(self) -> bool:
        """True if operating in virtual simulation mode."""
        pass

    @property
    def current_state(self) -> bool:
        """Legacy compatibility property reporting Pin 13 state."""
        return self.get_pin(13)

    def set_light(self, state: bool) -> bool:
        """Legacy compatibility method mapping single-light switch to Pin 13."""
        return self.set_pin(13, state)

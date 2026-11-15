"""
hardware - Multi-Pin Hardware Routing & Serial Abstraction Package
Part of AIEcoLightswitcher Modular Architecture (Milestone 2)

Provides abstract contracts, physical USB serial controllers, and
test mocks for multi-pin Arduino hardware routing (Pins 2-19).
"""

from hardware.base_controller import (
    BaseHardwareController,
    BaseMultiPinHardwareController,
    ConnectionResult,
)
from hardware.serial_controller import (
    HardwareController,
    MultiPinHardwareController,
)
from hardware.mock_controller import (
    MockHardwareController,
    MockMultiPinHardwareController,
)

__all__ = [
    "BaseHardwareController",
    "BaseMultiPinHardwareController",
    "ConnectionResult",
    "HardwareController",
    "MultiPinHardwareController",
    "MockHardwareController",
    "MockMultiPinHardwareController",
]

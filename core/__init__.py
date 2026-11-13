"""
core - Algorithmic and Configuration Foundation for AIEcoLightswitcher
Showcase Edition (Milestone 1 Implementation)
"""

from core.config import (
    ConnectionResult,
    DEFAULT_TARGET_CLASSES,
    EngineConfig,
    FeedConfig,
)
from core.device import (
    detect_best_device,
    get_inference_device,
    warmup_engine,
)
from core.duty_cycle import DutyCycleManager
from core.roi_filter import SpatialROIFilter
from core.smoother import (
    TemporalPresenceDebouncer,
    TemporalPresenceSmoother,
)

__all__ = [
    "ConnectionResult",
    "DEFAULT_TARGET_CLASSES",
    "DutyCycleManager",
    "EngineConfig",
    "FeedConfig",
    "SpatialROIFilter",
    "TemporalPresenceDebouncer",
    "TemporalPresenceSmoother",
    "detect_best_device",
    "get_inference_device",
    "warmup_engine",
]

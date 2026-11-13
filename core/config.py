"""
core/config.py - Configuration Dataclasses & Global Constants
Part of AIEcoLightswitcher Modular Architecture (Milestone 1)
"""

import copy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple, Union


# Default Target Classes for YOLOv8 (COCO dataset)
# 0 = person (human occupant), 77 = teddy bear (doll / tabletop test proxy)
DEFAULT_TARGET_CLASSES: Dict[int, str] = {0: "person", 77: "doll/teddy"}


class ConnectionResult(tuple):
    """
    Subclass of tuple (bool, str) allowing usage both as a boolean
    (if ctrl.connect(): ...) and as an unpacked pair (success, msg = ctrl.connect()).
    """
    def __new__(cls, success: bool, message: str = ""):
        return super().__new__(cls, (bool(success), str(message)))

    def __bool__(self) -> bool:
        return bool(self[0])


@dataclass
class EngineConfig:
    """
    Thread-safe configuration dataclass for runtime parameter adjustments.
    """
    conf_threshold: float = 0.50          # Range: 0.01 to 1.00
    iou_threshold: float = 0.45           # Range: 0.10 to 0.90
    eval_duration: float = 2.0            # Seconds; range 0.5 to 10.0
    sleep_duration: float = 5.0           # Seconds; range 0.0 to 30.0 (0.0 = continuous)
    duty_cycle_enabled: bool = True       # Enable/disable duty cycling
    roi_enabled: bool = False             # Enable/disable ROI filtering
    roi_polygon: list = field(default_factory=list)  # Normalized [(u1, v1), ...] in [0.0, 1.0]
    roi_mode: str = "BOTTOM_CENTER"       # "BOTTOM_CENTER" (foot point) or "CENTER"
    temporal_smoothing_enabled: bool = True
    hysteresis_on_frames: int = 3         # Consecutive frames to confirm ON
    hysteresis_off_frames: int = 15       # Consecutive frames to confirm OFF
    min_switch_interval: float = 1.5      # Minimum seconds between serial toggles
    override_mode: str = "AUTO"           # "AUTO", "FORCE_ON", "FORCE_OFF"
    target_classes: list = field(default_factory=lambda: [0, 77])

    def to_dict(self) -> Dict[str, Any]:
        """Returns configuration as a dictionary."""
        return asdict(self)

    def copy(self) -> "EngineConfig":
        """Creates a deep copy of the configuration."""
        return replace(
            self,
            roi_polygon=copy.deepcopy(self.roi_polygon),
            target_classes=list(self.target_classes)
        )

    def update(self, **kwargs) -> None:
        """Updates attributes in-place with validation and clamping."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key == "conf_threshold":
                    value = max(0.01, min(1.00, float(value)))
                elif key == "iou_threshold":
                    value = max(0.05, min(0.95, float(value)))
                elif key == "eval_duration":
                    value = max(0.1, float(value))
                elif key == "sleep_duration":
                    value = max(0.0, float(value))
                elif key == "hysteresis_on_frames":
                    value = max(1, int(value))
                elif key == "hysteresis_off_frames":
                    value = max(1, int(value))
                elif key == "min_switch_interval":
                    value = max(0.0, float(value))
                elif key == "override_mode":
                    value = str(value).upper().strip()
                elif key == "roi_mode":
                    value = str(value).upper().strip()
                setattr(self, key, value)


@dataclass
class FeedConfig:
    """
    Per-feed configuration descriptor conforming to PROJECT.md multi-feed contract.
    Integrates seamlessly with EngineConfig for inference parameters.
    """
    feed_id: str                                  # Unique identifier e.g. "feed_1"
    name: str                                     # Human-readable name e.g. "Front Entrance"
    source: Any                                   # Camera index (0, 1), URL (HTTP/RTSP), or "MOCK:..."
    pin: int = 13                                 # Arduino digital output pin (2–19)
    enabled: bool = True                          # Feed active processing flag
    conf_threshold: float = 0.50                  # Detection confidence threshold
    iou_threshold: float = 0.45                   # NMS IOU threshold
    eval_duration: float = 2.0                    # Duty cycle eval duration in seconds
    sleep_duration: float = 5.0                   # Duty cycle sleep duration in seconds
    duty_cycle_enabled: bool = True               # Enable/disable duty cycle
    roi_enabled: bool = False                     # Spatial ROI enable flag
    roi_polygon: list = field(default_factory=list) # Normalized [(u, v), ...] coordinates
    roi_mode: str = "BOTTOM_CENTER"               # "BOTTOM_CENTER" or "CENTER"
    temporal_smoothing_enabled: bool = True       # Enable temporal hysteresis
    hysteresis_on_frames: int = 3                 # M frames ON
    hysteresis_off_frames: int = 15               # N frames OFF
    min_switch_interval: float = 1.5              # Anti-chatter guard clamp (seconds)
    override_mode: str = "AUTO"                   # "AUTO", "FORCE_ON", "FORCE_OFF"
    target_classes: list = field(default_factory=lambda: [0, 77])
    engine_config: Optional[EngineConfig] = None

    def __post_init__(self):
        if self.engine_config is not None:
            # Synchronize from provided EngineConfig
            self.conf_threshold = self.engine_config.conf_threshold
            self.iou_threshold = self.engine_config.iou_threshold
            self.eval_duration = self.engine_config.eval_duration
            self.sleep_duration = self.engine_config.sleep_duration
            self.duty_cycle_enabled = self.engine_config.duty_cycle_enabled
            self.roi_enabled = self.engine_config.roi_enabled
            self.roi_polygon = copy.deepcopy(self.engine_config.roi_polygon)
            self.roi_mode = self.engine_config.roi_mode
            self.temporal_smoothing_enabled = self.engine_config.temporal_smoothing_enabled
            self.hysteresis_on_frames = self.engine_config.hysteresis_on_frames
            self.hysteresis_off_frames = self.engine_config.hysteresis_off_frames
            self.min_switch_interval = self.engine_config.min_switch_interval
            self.override_mode = self.engine_config.override_mode
            self.target_classes = list(self.engine_config.target_classes)
        else:
            self.engine_config = self.to_engine_config()

    def to_engine_config(self) -> EngineConfig:
        """Converts feed settings to an EngineConfig instance."""
        return EngineConfig(
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            eval_duration=self.eval_duration,
            sleep_duration=self.sleep_duration,
            duty_cycle_enabled=self.duty_cycle_enabled,
            roi_enabled=self.roi_enabled,
            roi_polygon=copy.deepcopy(self.roi_polygon),
            roi_mode=self.roi_mode,
            temporal_smoothing_enabled=self.temporal_smoothing_enabled,
            hysteresis_on_frames=self.hysteresis_on_frames,
            hysteresis_off_frames=self.hysteresis_off_frames,
            min_switch_interval=self.min_switch_interval,
            override_mode=self.override_mode,
            target_classes=list(self.target_classes)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Returns feed configuration as a dictionary."""
        d = asdict(self)
        if "engine_config" in d and d["engine_config"] is not None:
            d["engine_config"] = asdict(self.engine_config)
        return d

    def copy(self) -> "FeedConfig":
        """Creates a deep copy of the feed configuration."""
        return replace(
            self,
            roi_polygon=copy.deepcopy(self.roi_polygon),
            target_classes=list(self.target_classes),
            engine_config=self.engine_config.copy() if self.engine_config else None
        )

    def update(self, **kwargs) -> None:
        """Updates attributes in-place and keeps engine_config synchronized."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.engine_config = self.to_engine_config()

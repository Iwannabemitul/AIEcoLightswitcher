"""
core/smoother.py - Asymmetric Temporal Presence Smoother & Debouncer
Part of AIEcoLightswitcher Modular Architecture (Milestone 1)
"""

import time
from typing import Any, Optional, Tuple


class TemporalPresenceSmoother:
    """
    Asymmetric dual-threshold hysteresis state machine:
    - Requires M consecutive positive frames (default 3) to switch ON.
    - Requires N consecutive negative frames (default 15) to switch OFF.
    - Enforces min_switch_interval guard clamp (default 1.5s) to eliminate relay chattering.
    """
    def __init__(
        self,
        on_frames: int = 3,
        off_frames: int = 15,
        min_switch_interval: float = 1.5,
        min_switch_sec: Optional[float] = None,
        enabled: bool = True
    ):
        interval = min_switch_sec if min_switch_sec is not None else min_switch_interval
        self.on_frames = max(1, int(on_frames))
        self.off_frames = max(1, int(off_frames))
        self.min_switch_interval = max(0.0, float(interval))
        self.enabled = bool(enabled)
        self.current_state = False
        self.consecutive_positive = 0
        self.consecutive_negative = 0
        self.last_switch_time = 0.0

    def update_params(
        self,
        on_frames: Optional[int] = None,
        off_frames: Optional[int] = None,
        min_switch_interval: Optional[float] = None,
        min_switch_sec: Optional[float] = None,
        enabled: Optional[bool] = None
    ) -> None:
        """Updates smoother parameters dynamically."""
        if on_frames is not None:
            self.on_frames = max(1, int(on_frames))
        if off_frames is not None:
            self.off_frames = max(1, int(off_frames))
        interval = min_switch_sec if min_switch_sec is not None else min_switch_interval
        if interval is not None:
            self.min_switch_interval = max(0.0, float(interval))
        if enabled is not None:
            self.enabled = bool(enabled)

    def reset(self, state: bool = False, now: Optional[float] = None) -> None:
        """Resets the state machine counters."""
        self.current_state = bool(state)
        self.consecutive_positive = 0
        self.consecutive_negative = 0
        self.last_switch_time = now if now is not None else time.time()

    def update(
        self, raw_presence: Any, now: Optional[float] = None
    ) -> Tuple[bool, bool]:
        """
        Updates the filter state.
        Accepts boolean presence or numeric occupant counts (> 0).

        Returns:
            Tuple[bool, bool]: (current_smoothed_state, state_has_changed)
        """
        if isinstance(raw_presence, (int, float)):
            presence_bool = (raw_presence > 0)
        else:
            presence_bool = bool(raw_presence)

        if now is None:
            now = time.time()

        state_changed = False

        if not self.enabled:
            if presence_bool != self.current_state:
                if (now - self.last_switch_time) >= self.min_switch_interval:
                    self.current_state = presence_bool
                    self.last_switch_time = now
                    state_changed = True
            return self.current_state, state_changed

        if presence_bool:
            self.consecutive_positive += 1
            self.consecutive_negative = 0
            if not self.current_state and self.consecutive_positive >= self.on_frames:
                if (now - self.last_switch_time) >= self.min_switch_interval:
                    self.current_state = True
                    self.last_switch_time = now
                    state_changed = True
        else:
            self.consecutive_negative += 1
            self.consecutive_positive = 0
            if self.current_state and self.consecutive_negative >= self.off_frames:
                if (now - self.last_switch_time) >= self.min_switch_interval:
                    self.current_state = False
                    self.last_switch_time = now
                    state_changed = True

        return self.current_state, state_changed


# Survey & legacy alias
TemporalPresenceDebouncer = TemporalPresenceSmoother

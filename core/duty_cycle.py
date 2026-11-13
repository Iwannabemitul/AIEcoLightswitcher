"""
core/duty_cycle.py - Duty Cycle Energy Management
Part of AIEcoLightswitcher Modular Architecture (Milestone 1)
"""

import time
from typing import Optional, Tuple


class DutyCycleManager:
    """
    Manages EVALUATING and SLEEPING duty cycle phases to conserve compute power.
    When sleep_duration == 0.0 or duty_cycle_enabled is False, operates in continuous mode.
    """
    def __init__(
        self,
        eval_duration: float = 2.0,
        sleep_duration: float = 5.0,
        enabled: bool = True
    ):
        self.eval_duration = max(0.1, float(eval_duration))
        self.sleep_duration = max(0.0, float(sleep_duration))
        self.enabled = bool(enabled)
        self.current_state = "EVALUATING"
        self.timer_start = 0.0
        self._initialized = False

    def update_durations(
        self,
        eval_duration: Optional[float] = None,
        sleep_duration: Optional[float] = None,
        enabled: Optional[bool] = None
    ) -> None:
        """Dynamically adjusts duty cycle timings at runtime."""
        if eval_duration is not None:
            self.eval_duration = max(0.1, float(eval_duration))
        if sleep_duration is not None:
            self.sleep_duration = max(0.0, float(sleep_duration))
        if enabled is not None:
            self.enabled = bool(enabled)

    def reset(self, phase: str = "EVALUATING", now: Optional[float] = None) -> None:
        """Resets the duty cycle timer to the specified phase."""
        self.current_state = phase.upper()
        self.timer_start = now if now is not None else time.time()
        self._initialized = True

    def should_infer(self) -> bool:
        """Returns True if YOLO inference should run in the current phase."""
        if not self.enabled or self.sleep_duration <= 0.0:
            return True
        return self.current_state == "EVALUATING"

    def update(
        self, now: Optional[float] = None
    ) -> Tuple[str, float, bool]:
        """
        Updates duty cycle timers and handles phase transitions.

        Returns:
            Tuple[str, float, bool]: (current_state, countdown_seconds, phase_transitioned)
        """
        if now is None:
            now = time.time()

        if not self._initialized:
            self.timer_start = now
            self._initialized = True

        if not self.enabled or self.sleep_duration <= 0.0:
            self.current_state = "EVALUATING"
            return "EVALUATING", 0.0, False

        elapsed = now - self.timer_start
        transitioned = False

        if self.current_state == "EVALUATING":
            if elapsed >= self.eval_duration:
                self.current_state = "SLEEPING"
                self.timer_start = now
                transitioned = True
                countdown = self.sleep_duration
            else:
                countdown = max(0.0, self.eval_duration - elapsed)
        else:
            if elapsed >= self.sleep_duration:
                self.current_state = "EVALUATING"
                self.timer_start = now
                transitioned = True
                countdown = self.eval_duration
            else:
                countdown = max(0.0, self.sleep_duration - elapsed)

        return self.current_state, countdown, transitioned

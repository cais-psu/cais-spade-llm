"""Simulation progress uses ROS time; liveness and cancellation use wall time."""

from __future__ import annotations

import math
import time
from collections.abc import Callable


class MotionDeadline:
    """A simulation progress deadline with independent wall-clock liveness limits."""

    def __init__(self, duration: float, now: Callable[[], float], cancelled: Callable[[], bool],
                 *, wall_timeout: float = 300., stall_timeout: float = 30.) -> None:
        if not math.isfinite(duration) or duration < 0:
            raise ValueError('Simulation duration must be finite and non-negative')
        self.now, self.cancelled = now, cancelled
        self.origin = self.previous = now()
        if not math.isfinite(self.origin):
            raise ValueError('Simulation clock is unavailable')
        self.started = self.progressed = time.monotonic()
        self.duration, self.wall_timeout, self.stall_timeout = duration, wall_timeout, stall_timeout

    def pending(self) -> bool:
        """Reject Stop/reset/stall before checking the intended motion duration."""
        if self.cancelled():
            raise InterruptedError('Simulation motion cancelled')
        current, wall = self.now(), time.monotonic()
        if not math.isfinite(current) or current < self.previous:
            raise RuntimeError('Simulation clock reset during execution')
        if current > self.previous:
            self.progressed = wall
        self.previous = current
        if wall - self.progressed > self.stall_timeout or wall - self.started > self.wall_timeout:
            raise TimeoutError('Simulation motion wall-clock watchdog expired')
        return current - self.origin < self.duration


def wait_for_simulation(
    duration: float, *, now: Callable[[], float], cancelled: Callable[[], bool],
    poll: Callable[[float], None] = time.sleep, stall_timeout: float = 30.0,
) -> None:
    """Wait for simulated progress without hiding a paused/reset clock or Stop."""
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Simulation duration must be finite and non-negative")
    previous = origin = now()
    if not math.isfinite(origin):
        raise ValueError('Simulation clock is unavailable')
    last_progress = time.monotonic()
    while previous - origin < duration:
        if cancelled():
            raise InterruptedError("Simulation wait cancelled")
        poll(.01)
        current = now()
        if not math.isfinite(current) or current < previous:
            raise RuntimeError("Simulation clock reset during execution")
        if current > previous:
            last_progress = time.monotonic()
        elif time.monotonic() - last_progress >= stall_timeout:
            raise TimeoutError("Simulation clock is paused or unavailable")
        previous = current

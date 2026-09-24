"""The twin's clock.

Every time-dependent part of the twin (stage motion, holder thermal response,
in-situ specimen evolution, frame pacing) reads time from one clock, so the
whole twin can run in real time or be fast-forwarded for automation testing.
"""

from __future__ import annotations

import threading
import time


class Clock:
    """Real-time clock, optionally scaled.

    ``time_scale > 1`` makes simulated time run faster than wall time: an
    in-situ anneal that takes 10 minutes on the microscope can be replayed in
    one minute with ``time_scale=10``. Frame pacing still sleeps in *wall*
    time (``frame_time / time_scale``), so the whole twin speeds up consistently.
    """

    def __init__(self, time_scale: float = 1.0):
        if time_scale <= 0:
            raise ValueError("time_scale must be > 0")
        self.time_scale = float(time_scale)
        self._t0_wall = time.perf_counter()
        self._t0_sim = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        return self.now()

    def now(self) -> float:
        """Simulated seconds since the clock was created."""
        with self._lock:
            return self._t0_sim + (time.perf_counter() - self._t0_wall) * self.time_scale

    def set_time_scale(self, time_scale: float) -> None:
        with self._lock:
            now_wall = time.perf_counter()
            self._t0_sim += (now_wall - self._t0_wall) * self.time_scale
            self._t0_wall = now_wall
            self.time_scale = float(time_scale)

    def sleep(self, sim_seconds: float) -> None:
        if sim_seconds > 0:
            time.sleep(sim_seconds / self.time_scale)

    def sleep_until(self, sim_time: float) -> None:
        self.sleep(sim_time - self.now())

    @property
    def virtual(self) -> bool:
        return False


class ManualClock(Clock):
    """Clock that only moves when told to (deterministic tests, offline studies).

    ``sleep`` advances the clock instead of blocking, so paced frame loops run
    as fast as the CPU allows while still seeing correct simulated time.
    """

    def __init__(self, start: float = 0.0):
        self._t = float(start)
        self._lock = threading.Lock()
        self.time_scale = float("inf")

    def now(self) -> float:
        with self._lock:
            return self._t

    def advance(self, seconds: float) -> float:
        with self._lock:
            self._t += max(0.0, float(seconds))
            return self._t

    def set(self, t: float) -> None:
        with self._lock:
            self._t = float(t)

    def set_time_scale(self, time_scale: float) -> None:  # pragma: no cover - no-op
        pass

    def sleep(self, sim_seconds: float) -> None:
        if sim_seconds > 0:
            self.advance(sim_seconds)

    @property
    def virtual(self) -> bool:
        return True

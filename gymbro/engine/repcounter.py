"""Rep counting by hysteresis on a normalized 1D movement signal.

The old `validate_progressive_movement` tried to force movement through four
ordered angle thresholds. That failed for three reasons, all of which this
design avoids:

  * Stage 0's threshold was often already satisfied at rest, so the machine
    burned stages before the user moved.
  * The "jerky movement" guard compared stage gaps that were always exactly 1
    apart, so it could never fire.
  * Absolute angle thresholds ignore that people have different ranges of
    motion. A 165-degree hip extension is a full lockout for one person and a
    partial rep for another.

Instead: calibrate the user's own range on the first reps, normalize the signal
to [0, 1] against *their* range, and count with Schmitt-trigger hysteresis plus
a refractory period. Partial reps are detected and reported, not silently
dropped or silently counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Phase(str, Enum):
    IDLE = "idle"
    CONCENTRIC = "concentric"   # working against resistance
    HOLD = "hold"               # squeeze at peak contraction
    ECCENTRIC = "eccentric"     # controlled return


@dataclass
class RepEvent:
    index: int
    start_time: float
    end_time: float
    peak_value: float           # normalized [0,1] depth reached
    concentric_s: float
    eccentric_s: float
    hold_s: float
    full_range: bool
    tempo_ok: bool
    notes: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class RangeCalibration:
    """The user's observed range for one exercise, learned then frozen."""

    low: float
    high: float
    samples: int = 0
    frozen: bool = False

    @property
    def span(self) -> float:
        return self.high - self.low

    def normalize(self, value: float) -> float:
        if self.span < 1e-6:
            return 0.0
        return float(np.clip((value - self.low) / self.span, 0.0, 1.0))


class RepCounter:
    """Counts reps on a scalar signal where higher = deeper into the rep.

    Args:
        enter_threshold: normalized level that starts a rep.
        exit_threshold: normalized level that completes it. Must be below
            `enter_threshold`; the gap is the hysteresis band that stops
            jitter around a single level from double-counting.
        full_range_threshold: peak needed for the rep to count as full range.
        min_rep_s: refractory period. Anything faster is momentum, not a rep.
        max_rep_s: beyond this the rep is abandoned (user stopped mid-way).
        min_hold_s: required pause at peak, for exercises that demand a squeeze.
    """

    def __init__(
        self,
        enter_threshold: float = 0.70,
        exit_threshold: float = 0.30,
        full_range_threshold: float = 0.85,
        min_rep_s: float = 0.6,
        max_rep_s: float = 20.0,
        min_hold_s: float = 0.0,
        calibration_reps: int = 2,
    ) -> None:
        if not 0.0 < exit_threshold < enter_threshold < 1.0:
            raise ValueError("require 0 < exit < enter < 1")

        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.full_range_threshold = full_range_threshold
        self.min_rep_s = min_rep_s
        self.max_rep_s = max_rep_s
        self.min_hold_s = min_hold_s
        self.calibration_reps = calibration_reps

        self.reps: list[RepEvent] = []
        self.phase = Phase.IDLE
        self.calibration: RangeCalibration | None = None

        self._rep_start: float | None = None
        self._peak_value = 0.0
        self._peak_time: float | None = None
        self._hold_start: float | None = None
        self._hold_accum = 0.0
        self._last_norm = 0.0
        self._raw_window: list[float] = []
        self._partial_count = 0

    # ------------------------------------------------------------------ range

    def seed_calibration(self, low: float, high: float) -> None:
        """Provide a known range (e.g. from a previous session) and freeze it."""
        self.calibration = RangeCalibration(low=low, high=high, frozen=True)

    def _update_calibration(self, raw: float) -> None:
        """Widen the observed range during the calibration window."""
        if self.calibration is None:
            self.calibration = RangeCalibration(low=raw, high=raw)
        cal = self.calibration
        if cal.frozen:
            return

        cal.low = min(cal.low, raw)
        cal.high = max(cal.high, raw)
        cal.samples += 1

        # Freeze once we have enough reps to trust the range.
        if len(self.reps) >= self.calibration_reps and cal.span > 1e-3:
            cal.frozen = True

    # ------------------------------------------------------------------ update

    def update(self, raw_value: float, timestamp: float) -> RepEvent | None:
        """Feed one frame. Returns a RepEvent on the frame a rep completes."""
        if raw_value is None or not np.isfinite(raw_value):
            return None

        self._raw_window.append(raw_value)
        if len(self._raw_window) > 300:
            self._raw_window.pop(0)

        self._update_calibration(raw_value)
        cal = self.calibration
        assert cal is not None

        # Until we have a real span, we cannot normalize meaningfully.
        if cal.span < 1e-3:
            return None

        norm = cal.normalize(raw_value)
        completed: RepEvent | None = None

        if self.phase is Phase.IDLE:
            if norm >= self.enter_threshold:
                self.phase = Phase.CONCENTRIC
                self._rep_start = timestamp
                self._peak_value = norm
                self._peak_time = timestamp
                self._hold_accum = 0.0
                self._hold_start = None

        elif self.phase in (Phase.CONCENTRIC, Phase.HOLD):
            if norm > self._peak_value:
                self._peak_value = norm
                self._peak_time = timestamp

            # Accumulate time spent near the peak as "hold".
            near_peak = norm >= self._peak_value - 0.08
            if near_peak:
                if self._hold_start is None:
                    self._hold_start = timestamp
                    self.phase = Phase.HOLD
                self._hold_accum = timestamp - self._hold_start
            else:
                self._hold_start = None
                if norm < self._last_norm:
                    self.phase = Phase.ECCENTRIC

            if self._rep_start is not None and timestamp - self._rep_start > self.max_rep_s:
                self._abandon("held too long without completing the rep")

        elif self.phase is Phase.ECCENTRIC:
            if norm > self._peak_value:  # pushed deeper again -> same rep
                self._peak_value = norm
                self._peak_time = timestamp
                self.phase = Phase.CONCENTRIC
            elif norm <= self.exit_threshold:
                completed = self._complete(timestamp)
            elif self._rep_start is not None and timestamp - self._rep_start > self.max_rep_s:
                self._abandon("rep exceeded maximum duration")

        self._last_norm = norm
        return completed

    # --------------------------------------------------------------- lifecycle

    def _complete(self, timestamp: float) -> RepEvent | None:
        assert self._rep_start is not None and self._peak_time is not None
        duration = timestamp - self._rep_start

        if duration < self.min_rep_s:
            # Too fast to be a controlled rep -- almost always bounced momentum.
            self._partial_count += 1
            self._reset_rep()
            return None

        notes: list[str] = []
        full_range = self._peak_value >= self.full_range_threshold
        if not full_range:
            notes.append(
                f"partial range: reached {self._peak_value * 100:.0f}% of your range"
            )

        concentric = self._peak_time - self._rep_start
        eccentric = timestamp - self._peak_time

        tempo_ok = True
        if eccentric < 0.4:
            tempo_ok = False
            notes.append("lowering too fast - control the negative")
        if self.min_hold_s > 0 and self._hold_accum < self.min_hold_s:
            tempo_ok = False
            notes.append(
                f"hold the squeeze: {self._hold_accum:.1f}s of {self.min_hold_s:.1f}s"
            )

        event = RepEvent(
            index=len(self.reps) + 1,
            start_time=self._rep_start,
            end_time=timestamp,
            peak_value=self._peak_value,
            concentric_s=concentric,
            eccentric_s=eccentric,
            hold_s=self._hold_accum,
            full_range=full_range,
            tempo_ok=tempo_ok,
            notes=notes,
        )
        self.reps.append(event)
        self._reset_rep()
        return event

    def _abandon(self, reason: str) -> None:
        self._partial_count += 1
        self._reset_rep()

    def _reset_rep(self) -> None:
        self.phase = Phase.IDLE
        self._rep_start = None
        self._peak_value = 0.0
        self._peak_time = None
        self._hold_start = None
        self._hold_accum = 0.0

    # ------------------------------------------------------------------ status

    @property
    def count(self) -> int:
        return len(self.reps)

    @property
    def good_count(self) -> int:
        """Reps that were both full range and correct tempo."""
        return sum(1 for r in self.reps if r.full_range and r.tempo_ok)

    @property
    def partial_count(self) -> int:
        return self._partial_count

    def reset(self) -> None:
        self.reps.clear()
        self._partial_count = 0
        self._reset_rep()
        self._raw_window.clear()
        self._last_norm = 0.0
        if self.calibration and not self.calibration.frozen:
            self.calibration = None

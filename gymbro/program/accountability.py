"""Streaks, training debt and locked progression.

A note on what this is and isn't. Software cannot make you train. What it can
do is make skipping visible and make progress contingent on work actually done,
so the program stops quietly pretending you did sessions you didn't. That's the
honest version of "forcing you", and it's what this module implements.

The design rule throughout: a session counts only when the vision system
verified the reps. There is no self-report path and no skip key.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

# A session must hit this fraction of its prescribed *good* reps to count.
COMPLETION_THRESHOLD = 0.80

# Grace: one scheduled day can be missed without breaking a streak.
STREAK_GRACE_DAYS = 1

# Debt beyond this means the plan gets easier, not harder -- chasing an
# impossible backlog is how people quit.
MAX_DEBT_SESSIONS = 3

# Consecutive clean clears required before load goes up.
CLEARS_TO_PROGRESS = 2


@dataclass
class SessionVerdict:
    completed: bool
    completion_ratio: float
    verified_reps: int
    prescribed_reps: int
    reason: str


def judge_session(set_records: list[dict]) -> SessionVerdict:
    """Decide whether a finished session counts.

    `set_records` carry `target_reps` and `good_reps` -- good_reps being reps
    the vision system confirmed at full range and acceptable tempo. Partial and
    bounced reps deliberately do not count toward completion.
    """
    prescribed = sum(int(s.get("target_reps", 0)) for s in set_records)
    verified = sum(int(s.get("good_reps", 0)) for s in set_records)

    if prescribed == 0:
        return SessionVerdict(False, 0.0, 0, 0, "no sets were prescribed")

    ratio = verified / prescribed
    if ratio >= COMPLETION_THRESHOLD:
        return SessionVerdict(True, ratio, verified, prescribed, "session complete")

    return SessionVerdict(
        False, ratio, verified, prescribed,
        f"only {verified}/{prescribed} verified reps "
        f"({ratio * 100:.0f}%, need {COMPLETION_THRESHOLD * 100:.0f}%)",
    )


@dataclass
class StreakState:
    current: int
    longest: int
    debt: int
    last_session_date: date | None

    @property
    def in_debt(self) -> bool:
        return self.debt > 0


def update_streak(
    state: StreakState,
    session_date: date,
    completed: bool,
    scheduled_days: set[int] | None = None,
) -> StreakState:
    """Advance streak state after a session.

    `scheduled_days` are weekday numbers (0=Mon) the user committed to. Missing
    an unscheduled day is not a miss.
    """
    if not completed:
        return StreakState(
            current=0,
            longest=state.longest,
            debt=min(state.debt + 1, MAX_DEBT_SESSIONS),
            last_session_date=state.last_session_date,
        )

    last = state.last_session_date
    if last is None:
        current = 1
    else:
        gap = (session_date - last).days
        if gap <= 0:
            current = state.current  # same day; no double credit
        elif gap <= 1 + STREAK_GRACE_DAYS:
            current = state.current + 1
        else:
            missed = _scheduled_days_between(last, session_date, scheduled_days)
            current = 1 if missed > STREAK_GRACE_DAYS else state.current + 1

    # Completing a session pays off one unit of debt.
    return StreakState(
        current=current,
        longest=max(state.longest, current),
        debt=max(0, state.debt - 1),
        last_session_date=session_date,
    )


def _scheduled_days_between(
    start: date, end: date, scheduled_days: set[int] | None
) -> int:
    """Count scheduled training days strictly between two dates."""
    if scheduled_days is None:
        return max(0, (end - start).days - 1)
    count = 0
    day = start + timedelta(days=1)
    while day < end:
        if day.weekday() in scheduled_days:
            count += 1
        day += timedelta(days=1)
    return count


def accrue_missed_days(
    state: StreakState, today: date, scheduled_days: set[int]
) -> StreakState:
    """Add debt for scheduled sessions that passed without a session.

    Called at plan-generation time so debt appears without the user opening
    anything -- the backlog builds whether or not they look at it.
    """
    if state.last_session_date is None:
        return state

    missed = _scheduled_days_between(state.last_session_date, today, scheduled_days)
    if missed <= 0:
        return state

    return StreakState(
        current=0 if missed > STREAK_GRACE_DAYS else state.current,
        longest=state.longest,
        debt=min(state.debt + missed, MAX_DEBT_SESSIONS),
        last_session_date=state.last_session_date,
    )


# ------------------------------------------------------------- progression


@dataclass
class ProgressionDecision:
    load_kg: float | None
    target_reps: int
    target_sets: int
    consecutive_clears: int
    changed: bool
    message: str


def next_prescription(
    *,
    current_load_kg: float | None,
    target_reps: int,
    target_sets: int,
    consecutive_clears: int,
    last_good_reps: int,
    last_prescribed_reps: int,
    last_form_score: float | None,
    rep_ceiling: int = 15,
    rep_floor: int = 8,
    load_increment_kg: float = 2.0,
    available_loads: tuple[float, ...] | None = None,
) -> ProgressionDecision:
    """Double progression, gated on verified reps *and* form.

    Reps climb to a ceiling, then load steps up and reps reset to the floor.
    Load never increases on a session with poor form -- which is how the old
    approach of "count the rep, move on" turns a technique problem into an
    injury. A session that fails outright holds everything where it is.
    """
    cleared = (
        last_good_reps >= last_prescribed_reps
        and (last_form_score is None or last_form_score >= 0.85)
    )

    if not cleared:
        reason = "form needs work before adding load" if (
            last_form_score is not None and last_form_score < 0.85
        ) else "finish all prescribed reps to progress"
        return ProgressionDecision(
            current_load_kg, target_reps, target_sets, 0, False,
            f"Holding steady - {reason}.",
        )

    clears = consecutive_clears + 1
    if clears < CLEARS_TO_PROGRESS:
        return ProgressionDecision(
            current_load_kg, target_reps, target_sets, clears, False,
            f"Clean session ({clears}/{CLEARS_TO_PROGRESS}). "
            f"One more and you move up.",
        )

    # Earned a step up.
    if target_reps < rep_ceiling:
        return ProgressionDecision(
            current_load_kg, target_reps + 1, target_sets, 0, True,
            f"Earned it - going to {target_reps + 1} reps.",
        )

    new_load = _next_load(current_load_kg, load_increment_kg, available_loads)
    if new_load is None:
        # Out of load: add volume instead of stalling.
        return ProgressionDecision(
            current_load_kg, rep_ceiling, target_sets + 1, 0, True,
            f"No heavier dumbbell available - adding a {target_sets + 1}th set instead.",
        )

    return ProgressionDecision(
        new_load, rep_floor, target_sets, 0, True,
        f"Earned it - up to {new_load:g}kg, back to {rep_floor} reps.",
    )


def _next_load(
    current: float | None, increment: float, available: tuple[float, ...] | None
) -> float | None:
    """Next usable weight, honouring which dumbbells actually exist."""
    if current is None:
        return None
    if not available:
        return current + increment
    heavier = sorted(w for w in available if w > current + 1e-9)
    return heavier[0] if heavier else None

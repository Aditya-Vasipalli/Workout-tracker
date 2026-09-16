"""The coach: plan generation, session orchestration, verdicts.

Camera-free. `gymbro/cli.py` drives this with real video; the tests drive it
with recorded poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .engine.session import SetResult
from .exercises.loader import get_exercise
from .program.accountability import (
    StreakState, accrue_missed_days, judge_session, next_prescription, update_streak,
)
from .program.generator import DayPlan, WorkoutGenerator
from .store.db import Store


@dataclass
class CoachConfig:
    scheduled_days: frozenset[int] = frozenset({0, 1, 2, 3, 4})   # Mon-Fri
    available_loads: tuple[float, ...] = ()
    difficulty: str = "intermediate"
    session_minutes: int = 45


class Coach:
    def __init__(self, store: Store, config: CoachConfig | None = None, seed: int | None = None):
        self.store = store
        self.config = config or CoachConfig()
        self.generator = WorkoutGenerator(
            difficulty=self.config.difficulty,
            session_minutes=self.config.session_minutes,
            seed=seed,
        )

    # ----------------------------------------------------------------- state

    def streak_state(self) -> StreakState:
        row = self.store.get_streak()
        last = row["last_session_date"]
        return StreakState(
            current=row["current"], longest=row["longest"], debt=row["debt_sessions"],
            last_session_date=date.fromisoformat(last) if last else None,
        )

    def _save_streak(self, state: StreakState) -> None:
        self.store.update_streak(
            current=state.current, longest=state.longest, debt_sessions=state.debt,
            last_session_date=(
                state.last_session_date.isoformat() if state.last_session_date else None
            ),
        )

    # ------------------------------------------------------------- planning

    def plan_for(self, day: date, regenerate: bool = False) -> DayPlan:
        """Today's workout. Stable across calls unless regenerated."""
        if not regenerate:
            cached = self.store.get_plan(day.isoformat())
            if cached:
                return DayPlan.from_dict(cached)

        # Accrue debt for scheduled days that passed without a session, so a
        # backlog shows up whether or not the app was opened.
        state = accrue_missed_days(self.streak_state(), day, set(self.config.scheduled_days))
        self._save_streak(state)

        plan = self.generator.generate(
            day,
            recent_volume=self.store.muscle_volume(since=date.fromordinal(day.toordinal() - 10)),
            debt=state.debt,
            progression_lookup=self._progression_for,
        )
        self.store.save_plan(day.isoformat(), plan.to_dict())
        return plan

    def _progression_for(self, exercise_id: str) -> dict | None:
        row = self.store.get_progression(exercise_id)
        return dict(row) if row else None

    # -------------------------------------------------------------- sessions

    def start_session(self, plan: DayPlan, backend: str = "") -> int:
        return self.store.start_session(plan.plan_date, plan.name, backend)

    def record_set(self, session_id: int, result: SetResult, set_index: int,
                   load_kg: float | None = None) -> None:
        set_id = self.store.record_set(
            session_id,
            exercise_id=result.exercise_id, set_index=set_index, side=result.side or "",
            target_reps=result.target_reps, counted_reps=result.counted_reps,
            good_reps=result.good_reps, partial_reps=result.partial_reps,
            load_kg=load_kg, form_score=result.form_score,
            form_coverage=result.form_coverage, rom_low=result.rom_low,
            rom_high=result.rom_high, duration_s=result.duration_s,
            completed=int(result.completed),
        )
        self.store.record_reps(set_id, result.reps)

        # Persist the learned range so future sessions start calibrated.
        if result.rom_low is not None and result.rom_high is not None:
            if result.tracking_quality > 0.7 and result.counted_reps >= 3:
                self.store.save_calibration(
                    result.exercise_id, result.rom_low, result.rom_high, result.side or ""
                )

    def finish_session(self, session_id: int, session_date: date) -> dict:
        """Judge the session, update streak and progression. Returns a summary."""
        sets = [dict(r) for r in self.store.session_sets(session_id)]
        verdict = judge_session(sets)

        self.store.finish_session(
            session_id,
            "completed" if verdict.completed else ("partial" if verdict.verified_reps else "abandoned"),
            verdict.reason,
        )

        state = update_streak(
            self.streak_state(), session_date, verdict.completed,
            set(self.config.scheduled_days),
        )
        self._save_streak(state)

        progression_notes = []
        if verdict.completed:
            progression_notes = self._advance_progression(sets)

        return {
            "verdict": verdict,
            "streak": state,
            "progression": progression_notes,
            "sets": sets,
        }

    def _advance_progression(self, sets: list[dict]) -> list[str]:
        """Apply double progression per exercise, gated on form."""
        by_exercise: dict[str, list[dict]] = {}
        for s in sets:
            by_exercise.setdefault(s["exercise_id"], []).append(s)

        notes = []
        for exercise_id, records in by_exercise.items():
            current = self._progression_for(exercise_id) or {}
            scores = [r["form_score"] for r in records if r["form_score"] is not None]

            decision = next_prescription(
                current_load_kg=current.get("load_kg"),
                target_reps=current.get("target_reps", records[0]["target_reps"]),
                target_sets=current.get("target_sets", len(records)),
                consecutive_clears=current.get("consecutive_clears", 0),
                last_good_reps=sum(r["good_reps"] for r in records),
                last_prescribed_reps=sum(r["target_reps"] for r in records),
                last_form_score=(sum(scores) / len(scores)) if scores else None,
                available_loads=self.config.available_loads or None,
            )
            self.store.upsert_progression(
                exercise_id,
                load_kg=decision.load_kg, target_reps=decision.target_reps,
                target_sets=decision.target_sets,
                consecutive_clears=decision.consecutive_clears,
            )
            if decision.changed:
                notes.append(f"{get_exercise(exercise_id).name}: {decision.message}")

        return notes

    # -------------------------------------------------------------- reporting

    def status(self) -> dict:
        state = self.streak_state()
        recent = self.store.recent_sessions(limit=10)
        return {
            "streak": state.current,
            "longest": state.longest,
            "debt": state.debt,
            "last_session": state.last_session_date,
            "recent": [dict(r) for r in recent],
        }

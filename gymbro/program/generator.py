"""Daily workout generation.

Priorities, in order:
  1. Glutes. This is the stated goal, so glute work gets the first slots of
     every session and the highest weekly frequency.
  2. Recovery. The same muscle does not get hammered two days running; the
     generator reads recent volume from the store and rotates.
  3. What's actually available. Only exercises whose equipment you own.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from ..exercises.loader import find, get_exercise, load_library
from ..exercises.schema import Equipment, Exercise
from .accountability import MAX_DEBT_SESSIONS


@dataclass
class PrescribedSet:
    exercise_id: str
    sets: int
    reps: int
    load_kg: float | None = None
    side: str | None = None       # for unilateral work
    rest_s: int = 60
    note: str = ""


@dataclass
class DayPlan:
    plan_date: str
    name: str
    focus: str
    blocks: list[PrescribedSet] = field(default_factory=list)
    estimated_minutes: int = 0
    warning: str = ""

    def total_prescribed_reps(self) -> int:
        return sum(b.sets * b.reps for b in self.blocks)

    def to_dict(self) -> dict:
        return {
            "plan_date": self.plan_date,
            "name": self.name,
            "focus": self.focus,
            "estimated_minutes": self.estimated_minutes,
            "warning": self.warning,
            "blocks": [vars(b) for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DayPlan":
        plan = cls(
            plan_date=d["plan_date"], name=d["name"], focus=d["focus"],
            estimated_minutes=d.get("estimated_minutes", 0),
            warning=d.get("warning", ""),
        )
        plan.blocks = [PrescribedSet(**b) for b in d["blocks"]]
        return plan


# A week that hits glutes three times without stacking them back to back.
ROTATION: tuple[tuple[str, str], ...] = (
    ("Glute Focus",        "glutes"),
    ("Upper Body",         "upper"),
    ("Glutes + Pilates",   "glutes_pilates"),
    ("Pilates & Core",     "pilates"),
    ("Glutes + Legs",      "glutes_legs"),
    ("Active Recovery",    "recovery"),
)


class WorkoutGenerator:
    def __init__(
        self,
        equipment: set[Equipment] | None = None,
        difficulty: str = "intermediate",
        session_minutes: int = 45,
        seed: int | None = None,
    ) -> None:
        self.equipment = equipment or {
            Equipment.BODYWEIGHT, Equipment.DUMBBELL, Equipment.BAND, Equipment.MAT,
        }
        self.difficulty = difficulty
        self.session_minutes = session_minutes
        self._rng = random.Random(seed)

    # ------------------------------------------------------------ selection

    def _available(self, **kwargs) -> list[Exercise]:
        kwargs.setdefault("max_difficulty", self.difficulty)
        return find(equipment=self.equipment, **kwargs)

    def _pick(
        self,
        candidates: list[Exercise],
        count: int,
        recent_volume: dict[str, int],
        exclude: set[str],
    ) -> list[Exercise]:
        """Prefer movements that are fresh, with a little randomness."""
        pool = [e for e in candidates if e.id not in exclude]
        if not pool:
            return []

        def freshness(ex: Exercise) -> float:
            # Lower recent volume = higher priority. Jitter avoids a fixed order.
            return recent_volume.get(ex.id, 0) + self._rng.random() * 10

        pool.sort(key=freshness)
        return pool[:count]

    # ------------------------------------------------------------- building

    def generate(
        self,
        plan_date: date,
        recent_volume: dict[str, int] | None = None,
        debt: int = 0,
        day_index: int | None = None,
        progression_lookup=None,
    ) -> DayPlan:
        recent_volume = recent_volume or {}
        idx = day_index if day_index is not None else plan_date.toordinal() % len(ROTATION)
        name, focus = ROTATION[idx % len(ROTATION)]

        plan = DayPlan(plan_date=plan_date.isoformat(), name=name, focus=focus)
        chosen: set[str] = set()

        builder = {
            "glutes": self._build_glute_day,
            "upper": self._build_upper_day,
            "glutes_pilates": self._build_glute_pilates_day,
            "pilates": self._build_pilates_day,
            "glutes_legs": self._build_glute_legs_day,
            "recovery": self._build_recovery_day,
        }[focus]
        builder(plan, recent_volume, chosen)

        # Debt makes the plan *shorter*, not longer. A backlog you can't clear
        # is the fastest way to stop training altogether.
        if debt > 0:
            keep = max(2, len(plan.blocks) - debt)
            plan.blocks = plan.blocks[:keep]
            plan.warning = (
                f"You have {debt} missed session{'s' if debt > 1 else ''}. "
                f"Today is trimmed to {keep} movements - clear the backlog by "
                f"finishing short sessions, not by skipping more."
            )
            if debt >= MAX_DEBT_SESSIONS:
                plan.warning += " Progression is paused until you complete one."

        if progression_lookup:
            for block in plan.blocks:
                prog = progression_lookup(block.exercise_id)
                if prog:
                    block.sets = prog.get("target_sets", block.sets)
                    block.reps = prog.get("target_reps", block.reps)
                    block.load_kg = prog.get("load_kg", block.load_kg)

        plan.estimated_minutes = self._estimate_minutes(plan)
        return plan

    def _add(
        self, plan: DayPlan, exercises: list[Exercise], sets: int, reps: int,
        chosen: set[str], rest: int = 60,
    ) -> None:
        for ex in exercises:
            chosen.add(ex.id)
            if ex.unilateral:
                # Unilateral work is prescribed per side.
                for side in ("left", "right"):
                    plan.blocks.append(PrescribedSet(
                        exercise_id=ex.id, sets=sets, reps=reps, side=side,
                        rest_s=rest if side == "right" else 15,
                    ))
            elif ex.is_isometric:
                plan.blocks.append(PrescribedSet(
                    exercise_id=ex.id, sets=sets, reps=reps, rest_s=rest,
                    note="hold for the prescribed seconds",
                ))
            else:
                plan.blocks.append(PrescribedSet(
                    exercise_id=ex.id, sets=sets, reps=reps, rest_s=rest,
                ))

    # ------------------------------------------------------------ day types

    def _build_glute_day(self, plan, vol, chosen) -> None:
        heavy = self._available(category="glutes", min_glute_emphasis=0.85)
        self._add(plan, self._pick(heavy, 2, vol, chosen), 4, 10, chosen, rest=90)

        accessory = self._available(category="glutes", min_glute_emphasis=0.6)
        self._add(plan, self._pick(accessory, 2, vol, chosen), 3, 12, chosen, rest=60)

        burnout = self._available(category="glutes", min_glute_emphasis=0.8)
        self._add(plan, self._pick(burnout, 1, vol, chosen), 2, 20, chosen, rest=45)

    def _build_upper_day(self, plan, vol, chosen) -> None:
        push = self._available(category="upper_push")
        pull = self._available(category="upper_pull")
        self._add(plan, self._pick(push, 2, vol, chosen), 3, 10, chosen)
        self._add(plan, self._pick(pull, 2, vol, chosen), 3, 10, chosen)
        # Glutes still get touched on upper day.
        glute = self._available(category="glutes", min_glute_emphasis=0.85)
        self._add(plan, self._pick(glute, 1, vol, chosen), 3, 15, chosen, rest=45)

    def _build_glute_pilates_day(self, plan, vol, chosen) -> None:
        glute = self._available(category="glutes", min_glute_emphasis=0.85)
        self._add(plan, self._pick(glute, 2, vol, chosen), 3, 12, chosen, rest=75)
        pilates = self._available(category="pilates")
        self._add(plan, self._pick(pilates, 3, vol, chosen), 3, 10, chosen, rest=45)

    def _build_pilates_day(self, plan, vol, chosen) -> None:
        pilates = self._available(category="pilates")
        self._add(plan, self._pick(pilates, 5, vol, chosen), 3, 10, chosen, rest=45)

    def _build_glute_legs_day(self, plan, vol, chosen) -> None:
        glute = self._available(category="glutes", min_glute_emphasis=0.85)
        self._add(plan, self._pick(glute, 2, vol, chosen), 4, 10, chosen, rest=90)
        legs = self._available(category="legs")
        self._add(plan, self._pick(legs, 2, vol, chosen), 3, 12, chosen, rest=75)

    def _build_recovery_day(self, plan, vol, chosen) -> None:
        pilates = self._available(category="pilates", max_difficulty="beginner")
        self._add(plan, self._pick(pilates, 3, vol, chosen), 2, 10, chosen, rest=30)
        glute = self._available(category="glutes", min_glute_emphasis=0.8)
        self._add(plan, self._pick(glute, 1, vol, chosen), 2, 15, chosen, rest=30)

    # ------------------------------------------------------------- estimate

    def _estimate_minutes(self, plan: DayPlan) -> int:
        seconds = 0.0
        for block in plan.blocks:
            ex = get_exercise(block.exercise_id)
            per_rep = 3.0 + ex.criteria.min_hold_s
            work = block.reps * (block.reps if ex.is_isometric else per_rep)
            seconds += block.sets * (work + block.rest_s)
        return max(5, round(seconds / 60))

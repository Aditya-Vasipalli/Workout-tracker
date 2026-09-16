"""Multi-day integration: does the coach actually hold you to the program?"""
from datetime import date, timedelta

import pytest

from gymbro.coach import Coach, CoachConfig
from gymbro.engine.session import SetResult
from gymbro.store.db import Store


@pytest.fixture
def coach():
    store = Store(":memory:")
    yield Coach(store, CoachConfig(available_loads=(5.0, 10.0, 15.0)), seed=11)
    store.close()


def fake_set(exercise_id, target=10, good=10, score=0.95, side=None):
    return SetResult(
        exercise_id=exercise_id, side=side, target_reps=target, counted_reps=good,
        good_reps=good, partial_reps=0, form_score=score, form_coverage=1.0,
        duration_s=40.0, completed=good >= target, rom_low=138.0, rom_high=176.0,
        tracking_quality=0.95,
    )


def run_day(coach, day, quality=1.0, form=0.95):
    """Run a full day at a given fraction of prescribed reps."""
    plan = coach.plan_for(day)
    sid = coach.start_session(plan)
    for i, block in enumerate(plan.blocks):
        for s in range(block.sets):
            good = int(round(block.reps * quality))
            coach.record_set(
                sid, fake_set(block.exercise_id, block.reps, good, form, block.side),
                set_index=s + 1,
            )
    return coach.finish_session(sid, day)


class TestVerification:
    def test_full_session_completes_and_builds_streak(self, coach):
        out = run_day(coach, date(2026, 9, 16))
        assert out["verdict"].completed
        assert out["streak"].current == 1

    def test_half_finished_session_does_not_count(self, coach):
        out = run_day(coach, date(2026, 9, 16), quality=0.5)
        assert not out["verdict"].completed
        assert out["streak"].current == 0
        assert out["streak"].debt == 1

    def test_sloppy_form_blocks_load_increase(self, coach):
        """Full reps but bad form: session counts, load must not go up."""
        day = date(2026, 9, 16)
        for i in range(3):
            run_day(coach, day + timedelta(days=i), quality=1.0, form=0.55)
        rows = [coach.store.get_progression(e) for e in ("glute_bridge", "hip_thrust")]
        for row in rows:
            if row is not None:
                assert row["consecutive_clears"] == 0

    def test_clean_sessions_advance_progression(self, coach):
        """Primary lifts must recur week to week, or nothing can ever progress."""
        day = date(2026, 9, 16)
        first = coach.plan_for(day)
        ex_id = first.blocks[0].exercise_id
        start_reps = first.blocks[0].reps

        # Same slot in the rotation, three weeks running.
        for week in range(3):
            run_day(coach, day + timedelta(days=week * 6))

        prog = coach.store.get_progression(ex_id)
        assert prog is not None
        assert prog["target_reps"] > start_reps, "clean work should raise the target"

    def test_primary_lift_recurs_across_weeks(self, coach):
        day = date(2026, 9, 16)
        anchors = [coach.plan_for(day + timedelta(days=w * 6)).blocks[0].exercise_id
                   for w in range(3)]
        assert len(set(anchors)) == 1, f"primary lift drifted: {anchors}"


class TestDebtAndStreaks:
    def test_debt_accrues_from_silence(self, coach):
        """Miss two weeks without opening the app; debt is waiting."""
        run_day(coach, date(2026, 9, 1))
        coach.plan_for(date(2026, 9, 21))
        assert coach.streak_state().debt > 0

    def test_debt_shortens_the_plan(self, coach):
        run_day(coach, date(2026, 9, 1))
        plan = coach.plan_for(date(2026, 9, 21))
        assert plan.warning
        assert "missed session" in plan.warning

    def test_completing_pays_down_debt(self, coach):
        run_day(coach, date(2026, 9, 1))
        coach.plan_for(date(2026, 9, 21))
        before = coach.streak_state().debt
        out = run_day(coach, date(2026, 9, 21))
        assert out["streak"].debt < before

    def test_longest_streak_survives_a_break(self, coach):
        day = date(2026, 9, 16)
        for i in range(5):
            run_day(coach, day + timedelta(days=i))
        assert coach.streak_state().longest == 5
        run_day(coach, day + timedelta(days=30), quality=0.2)
        state = coach.streak_state()
        assert state.current == 0 and state.longest == 5


class TestPlanning:
    def test_plan_is_stable_across_calls(self, coach):
        """Reopening the app must not reroll today's workout into something easier."""
        day = date(2026, 9, 16)
        a = coach.plan_for(day)
        b = coach.plan_for(day)
        assert [x.exercise_id for x in a.blocks] == [x.exercise_id for x in b.blocks]

    def test_glutes_appear_in_most_sessions(self, coach):
        from gymbro.exercises.loader import get_exercise
        day = date(2026, 9, 16)
        glute_days = 0
        for i in range(6):
            plan = coach.plan_for(day + timedelta(days=i))
            if any(get_exercise(b.exercise_id).glute_emphasis >= 0.5 for b in plan.blocks):
                glute_days += 1
        assert glute_days >= 5, f"glutes are the priority; only {glute_days}/6 days had them"

    def test_calibration_persists_between_sessions(self, coach):
        day = date(2026, 9, 16)
        plan = coach.plan_for(day)
        sid = coach.start_session(plan)
        block = plan.blocks[0]
        coach.record_set(sid, fake_set(block.exercise_id, 10, 10), set_index=1)
        assert coach.store.get_calibration(block.exercise_id, "") == (138.0, 176.0)

    def test_low_quality_tracking_does_not_poison_calibration(self, coach):
        day = date(2026, 9, 16)
        plan = coach.plan_for(day)
        sid = coach.start_session(plan)
        bad = fake_set(plan.blocks[0].exercise_id, 10, 10)
        bad.tracking_quality = 0.3
        coach.record_set(sid, bad, set_index=1)
        assert coach.store.get_calibration(plan.blocks[0].exercise_id, "") is None


class TestUnilateralAccounting:
    """Left and right are separate blocks; sets must not compound across sides."""

    def test_set_count_does_not_double_for_unilateral_work(self, coach):
        day = date(2026, 9, 16)
        plan = coach.plan_for(day)
        unilateral = [b for b in plan.blocks if b.side]
        assert unilateral, "expected some unilateral work in the plan"
        target = unilateral[0]
        prescribed_per_side = target.sets

        sid = coach.start_session(plan)
        for block in plan.blocks:
            for s in range(block.sets):
                coach.record_set(
                    sid, fake_set(block.exercise_id, block.reps, block.reps,
                                  side=block.side),
                    set_index=s + 1,
                )
        coach.finish_session(sid, day)

        stored = coach.store.get_progression(target.exercise_id)
        assert stored["target_sets"] == prescribed_per_side, (
            f"sets per side inflated from {prescribed_per_side} "
            f"to {stored['target_sets']}"
        )

    def test_sets_stay_bounded_over_many_weeks(self, coach):
        """The real symptom: runaway volume after repeated sessions."""
        day = date(2026, 9, 16)
        for week in range(5):
            run_day(coach, day + timedelta(days=week * 6))
        for row in coach.store._conn.execute("SELECT * FROM progression"):
            assert row["target_sets"] <= 6, (
                f"{row['exercise_id']} ballooned to {row['target_sets']} sets"
            )


class TestGraduation:
    """Outgrowing a movement should change the program, not just print a note."""

    def _max_out(self, coach, exercise_id, day):
        from gymbro.program.accountability import MAX_SETS
        coach.store.upsert_progression(
            exercise_id, load_kg=None, target_reps=15, target_sets=MAX_SETS,
            consecutive_clears=1, unlocked=1,
        )

    def test_maxed_bodyweight_exercise_graduates(self, coach):
        day = date(2026, 9, 16)
        self._max_out(coach, "glute_bridge", day)

        sid = coach.start_session(coach.plan_for(day))
        coach.record_set(sid, fake_set("glute_bridge", 15, 15), set_index=1)
        out = coach.finish_session(sid, day)

        assert any("outgrown" in n for n in out["progression"]), out["progression"]
        assert "glute_bridge" in coach.graduated_exercises()

    def test_harder_variant_is_seeded_sanely(self, coach):
        day = date(2026, 9, 16)
        self._max_out(coach, "glute_bridge", day)
        sid = coach.start_session(coach.plan_for(day))
        coach.record_set(sid, fake_set("glute_bridge", 15, 15), set_index=1)
        coach.finish_session(sid, day)

        seeded = coach.store.get_progression("single_leg_glute_bridge")
        assert seeded is not None
        assert seeded["target_reps"] == 8 and seeded["target_sets"] == 3

    def test_graduated_exercise_stops_being_prescribed(self, coach):
        day = date(2026, 9, 16)
        self._max_out(coach, "glute_bridge", day)
        sid = coach.start_session(coach.plan_for(day))
        coach.record_set(sid, fake_set("glute_bridge", 15, 15), set_index=1)
        coach.finish_session(sid, day)

        for i in range(1, 13):
            plan = coach.plan_for(day + timedelta(days=i), regenerate=True)
            assert all(b.exercise_id != "glute_bridge" for b in plan.blocks)

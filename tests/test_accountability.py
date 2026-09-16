from datetime import date

import pytest

from gymbro.program.accountability import (
    COMPLETION_THRESHOLD, MAX_DEBT_SESSIONS, StreakState, accrue_missed_days,
    judge_session, next_prescription, update_streak,
)


class TestJudgeSession:
    def test_full_completion(self):
        v = judge_session([{"target_reps": 10, "good_reps": 10}] * 3)
        assert v.completed and v.completion_ratio == 1.0

    def test_partial_below_threshold_fails(self):
        v = judge_session([{"target_reps": 10, "good_reps": 5}] * 3)
        assert not v.completed
        assert "15/30" in v.reason

    def test_at_threshold_passes(self):
        v = judge_session([{"target_reps": 10, "good_reps": 8}])
        assert v.completed and v.completion_ratio == pytest.approx(COMPLETION_THRESHOLD)

    def test_counted_but_sloppy_reps_do_not_count(self):
        """good_reps excludes partials -- 10 counted but 4 good still fails."""
        v = judge_session([{"target_reps": 10, "good_reps": 4, "counted_reps": 10}])
        assert not v.completed

    def test_empty_plan(self):
        assert not judge_session([]).completed


class TestStreak:
    def test_first_session_starts_streak(self):
        s = update_streak(StreakState(0, 0, 0, None), date(2026, 9, 16), True)
        assert s.current == 1 and s.longest == 1

    def test_consecutive_days_build(self):
        s = StreakState(0, 0, 0, None)
        for d in range(16, 21):
            s = update_streak(s, date(2026, 9, d), True)
        assert s.current == 5 and s.longest == 5

    def test_one_day_grace_preserves_streak(self):
        s = StreakState(4, 4, 0, date(2026, 9, 16))
        s = update_streak(s, date(2026, 9, 18), True)   # skipped the 17th
        assert s.current == 5

    def test_long_gap_breaks_streak(self):
        s = StreakState(10, 10, 0, date(2026, 9, 1))
        s = update_streak(s, date(2026, 9, 16), True, scheduled_days={0,1,2,3,4,5,6})
        assert s.current == 1
        assert s.longest == 10   # longest is never lost

    def test_failed_session_resets_and_adds_debt(self):
        s = update_streak(StreakState(7, 7, 0, date(2026, 9, 15)), date(2026, 9, 16), False)
        assert s.current == 0 and s.debt == 1 and s.longest == 7

    def test_completing_pays_down_debt(self):
        s = update_streak(StreakState(0, 5, 2, date(2026, 9, 15)), date(2026, 9, 16), True)
        assert s.debt == 1

    def test_debt_is_capped(self):
        s = StreakState(0, 0, 0, date(2026, 9, 1))
        for d in range(2, 12):
            s = update_streak(s, date(2026, 9, d), False)
        assert s.debt == MAX_DEBT_SESSIONS

    def test_unscheduled_days_are_not_misses(self):
        """Training Mon/Wed/Fri: a weekend gap must not break the streak."""
        s = StreakState(3, 3, 0, date(2026, 9, 11))      # Friday
        s = update_streak(s, date(2026, 9, 14), True, scheduled_days={0, 2, 4})
        assert s.current == 4

    def test_missed_days_accrue_without_a_session(self):
        s = accrue_missed_days(
            StreakState(5, 5, 0, date(2026, 9, 1)), date(2026, 9, 16),
            scheduled_days={0, 2, 4},
        )
        assert s.debt == MAX_DEBT_SESSIONS and s.current == 0

    def test_no_accrual_without_history(self):
        s = StreakState(0, 0, 0, None)
        assert accrue_missed_days(s, date(2026, 9, 16), {0, 2, 4}).debt == 0


class TestProgression:
    def _clear(self, **over):
        base = dict(
            current_load_kg=10.0, target_reps=10, target_sets=3,
            consecutive_clears=0, last_good_reps=10, last_prescribed_reps=10,
            last_form_score=0.95,
        )
        base.update(over)
        return next_prescription(**base)

    def test_missed_reps_holds_everything(self):
        d = self._clear(last_good_reps=7)
        assert not d.changed and d.target_reps == 10 and d.consecutive_clears == 0
        assert "finish all prescribed reps" in d.message

    def test_bad_form_blocks_progression_even_at_full_reps(self):
        """The key safety property: never add load on top of broken technique."""
        d = self._clear(last_form_score=0.6)
        assert not d.changed and d.load_kg == 10.0
        assert "form" in d.message.lower()

    def test_one_clean_session_is_not_enough(self):
        d = self._clear()
        assert not d.changed and d.consecutive_clears == 1

    def test_two_clean_sessions_add_a_rep(self):
        d = self._clear(consecutive_clears=1)
        assert d.changed and d.target_reps == 11

    def test_rep_ceiling_converts_to_load(self):
        d = self._clear(consecutive_clears=1, target_reps=15)
        assert d.changed and d.load_kg == 12.0 and d.target_reps == 8

    def test_respects_available_dumbbells(self):
        d = next_prescription(
            current_load_kg=10.0, target_reps=15, target_sets=3, consecutive_clears=1,
            last_good_reps=15, last_prescribed_reps=15, last_form_score=0.95,
            available_loads=(5.0, 10.0, 17.5),
        )
        assert d.load_kg == 17.5   # not 12.0 -- you don't own a 12

    def test_out_of_load_adds_a_set_instead_of_stalling(self):
        d = next_prescription(
            current_load_kg=17.5, target_reps=15, target_sets=3, consecutive_clears=1,
            last_good_reps=15, last_prescribed_reps=15, last_form_score=0.95,
            available_loads=(5.0, 10.0, 17.5),
        )
        assert d.target_sets == 4 and d.load_kg == 17.5
        assert "adding a 4th set" in d.message

    def test_bodyweight_exercise_progresses_by_reps(self):
        d = next_prescription(
            current_load_kg=None, target_reps=12, target_sets=3, consecutive_clears=1,
            last_good_reps=12, last_prescribed_reps=12, last_form_score=0.9,
        )
        assert d.target_reps == 13 and d.load_kg is None


class TestVolumeCeiling:
    def _maxed(self, sets, progressions=()):
        return next_prescription(
            current_load_kg=None, target_reps=15, target_sets=sets,
            consecutive_clears=1, last_good_reps=15, last_prescribed_reps=15,
            last_form_score=0.95, progressions=progressions,
        )

    def test_bodyweight_adds_sets_up_to_the_cap(self):
        d = self._maxed(3)
        assert d.target_sets == 4 and d.changed

    def test_sets_stop_at_the_cap(self):
        from gymbro.program.accountability import MAX_SETS
        d = self._maxed(MAX_SETS)
        assert d.target_sets == MAX_SETS

    def test_maxed_out_exercise_graduates_to_a_harder_variant(self):
        from gymbro.program.accountability import MAX_SETS
        d = self._maxed(MAX_SETS, progressions=("single_leg_glute_bridge",))
        assert d.graduate_to == "single_leg_glute_bridge"
        assert "outgrown" in d.message

    def test_no_variant_available_suggests_tempo_instead(self):
        from gymbro.program.accountability import MAX_SETS
        d = self._maxed(MAX_SETS)
        assert d.graduate_to is None
        assert "tempo" in d.message

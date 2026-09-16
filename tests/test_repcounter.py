import numpy as np
import pytest

from gymbro.engine.repcounter import RepCounter, Phase


def simulate(counter, values, fps=30.0, start=0.0):
    """Feed a signal; return the list of completed reps."""
    out = []
    for i, v in enumerate(values):
        ev = counter.update(v, start + i / fps)
        if ev is not None:
            out.append(ev)
    return out


def rep_signal(n_reps, low=140.0, high=175.0, fps=30.0, period=2.5, hold=0.0):
    """A clean sinusoid-ish rep train with optional hold at the top."""
    vals = []
    frames = int(period * fps)
    hold_frames = int(hold * fps)
    for _ in range(n_reps):
        # up
        for i in range(frames // 2):
            vals.append(low + (high - low) * (i / (frames // 2)))
        for _ in range(hold_frames):
            vals.append(high)
        # down
        for i in range(frames // 2):
            vals.append(high - (high - low) * (i / (frames // 2)))
    return vals


def test_counts_clean_reps():
    c = RepCounter()
    # Prime the range with one rep, then count.
    reps = simulate(c, rep_signal(8))
    # First rep(s) establish calibration; we should still land close to 8.
    assert 6 <= c.count <= 8, f"got {c.count}"


def test_seeded_calibration_counts_exactly():
    c = RepCounter()
    c.seed_calibration(low=140.0, high=175.0)
    simulate(c, rep_signal(10))
    assert c.count == 10


def test_jitter_does_not_double_count():
    """Noise around the threshold must not produce phantom reps."""
    c = RepCounter()
    c.seed_calibration(140.0, 175.0)
    rng = np.random.default_rng(1)
    vals = np.array(rep_signal(5)) + rng.normal(0, 1.5, len(rep_signal(5)))
    simulate(c, vals)
    assert c.count == 5, f"hysteresis failed: {c.count}"


def test_partial_reps_flagged_not_counted_as_good():
    c = RepCounter()
    c.seed_calibration(140.0, 175.0)
    # Only reaching 165 of a 140-175 range = 71% depth: a rep, but partial.
    simulate(c, rep_signal(4, low=140.0, high=165.0))
    assert c.count == 4
    assert c.good_count == 0
    assert all(not r.full_range for r in c.reps)
    assert any("partial range" in n for r in c.reps for n in r.notes)


def test_bounced_reps_rejected():
    """Sub-min_rep_s movement is momentum, not a rep."""
    c = RepCounter(min_rep_s=0.6)
    c.seed_calibration(140.0, 175.0)
    simulate(c, rep_signal(6, period=0.4))
    assert c.count == 0
    assert c.partial_count == 6


def test_hold_requirement_enforced():
    c = RepCounter(min_hold_s=2.0)
    c.seed_calibration(140.0, 175.0)
    simulate(c, rep_signal(3, hold=0.2))
    assert c.count == 3
    assert all(not r.tempo_ok for r in c.reps)
    assert any("hold the squeeze" in n for r in c.reps for n in r.notes)


def test_hold_requirement_satisfied():
    c = RepCounter(min_hold_s=1.0)
    c.seed_calibration(140.0, 175.0)
    simulate(c, rep_signal(3, hold=1.5))
    assert c.count == 3
    assert all(r.hold_s >= 1.0 for r in c.reps), [r.hold_s for r in c.reps]


def test_no_movement_yields_no_reps():
    c = RepCounter()
    simulate(c, [150.0] * 300)
    assert c.count == 0


def test_abandoned_rep_does_not_count():
    """Go up, stay up forever -- never returns, so never a rep."""
    c = RepCounter(max_rep_s=2.0)
    c.seed_calibration(140.0, 175.0)
    vals = list(np.linspace(140, 175, 30)) + [175.0] * 200
    simulate(c, vals)
    assert c.count == 0


def test_range_calibration_adapts_to_user():
    """Someone with a smaller ROM still gets full credit for their own range."""
    c = RepCounter()
    simulate(c, rep_signal(10, low=150.0, high=168.0))
    assert c.count >= 8
    cal = c.calibration
    assert cal is not None and cal.frozen
    assert 149 <= cal.low <= 151 and 167 <= cal.high <= 169


def test_nan_and_dropouts_ignored():
    c = RepCounter()
    c.seed_calibration(140.0, 175.0)
    vals = rep_signal(4)
    # Simulate tracking dropouts
    vals = [float("nan") if i % 37 == 0 else v for i, v in enumerate(vals)]
    simulate(c, vals)
    assert c.count == 4


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        RepCounter(enter_threshold=0.3, exit_threshold=0.7)

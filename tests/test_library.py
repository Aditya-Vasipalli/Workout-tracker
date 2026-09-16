"""The exercise library is data, so validate it like data."""
import pytest

from gymbro.exercises.loader import LibraryError, find, get_exercise, load_library
from gymbro.exercises.schema import Equipment
from gymbro.pose.skeleton import JOINT_INDEX
from tests.synthetic import glute_bridge_pose


@pytest.fixture(scope="module")
def library():
    return load_library()


class TestIntegrity:
    def test_library_loads(self, library):
        assert len(library) >= 40

    def test_every_signal_computes_on_a_real_pose(self, library):
        """No exercise may crash or return NaN on a fully visible body."""
        pose = glute_bridge_pose(160.0)
        for ex in library.values():
            side = "left" if ex.unilateral else None
            value = ex.signal.compute(pose, side)
            assert value == value, f"{ex.id} produced NaN"

    def test_every_rule_evaluates(self, library):
        pose = glute_bridge_pose(160.0)
        for ex in library.values():
            for rule in ex.rules:
                result = rule.evaluate(pose, {"reference_positions": {}})
                assert result.name

    def test_all_joints_are_canonical(self, library):
        for ex in library.values():
            for joint in ex.required_joints("left"):
                assert joint in JOINT_INDEX, f"{ex.id}: bad joint {joint}"

    def test_thresholds_are_ordered(self, library):
        for ex in library.values():
            c = ex.criteria
            assert 0 < c.exit_threshold < c.enter_threshold < 1, ex.id
            assert c.min_rep_s < c.max_rep_s, ex.id

    def test_unilateral_exercises_use_side_placeholders(self, library):
        for ex in library.values():
            if ex.unilateral and ex.signal.kind != "joint_distance":
                assert any("{side}" in j for j in ex.signal.joints), (
                    f"{ex.id} is unilateral but its signal does not follow the side"
                )

    def test_cross_references_resolve(self, library):
        for ex in library.values():
            for ref in ex.progressions + ex.regressions:
                assert ref in library

    def test_aliases_are_unique(self, library):
        seen = set()
        for ex in library.values():
            for alias in ex.aliases:
                assert alias not in seen and alias not in library
                seen.add(alias)


class TestGlutePriority:
    def test_glutes_are_the_largest_category(self, library):
        from collections import Counter
        counts = Counter(e.category for e in library.values())
        assert counts["glutes"] >= 15

    def test_plenty_of_high_emphasis_glute_work(self):
        assert len(find(min_glute_emphasis=0.85)) >= 8

    def test_glute_work_available_without_a_bench(self):
        """Not everyone has a bench; the priority goal must not depend on one."""
        home = {Equipment.BODYWEIGHT, Equipment.DUMBBELL, Equipment.BAND, Equipment.MAT}
        assert len(find(equipment=home, min_glute_emphasis=0.85)) >= 5

    def test_pilates_is_covered(self, library):
        assert len([e for e in library.values() if e.category == "pilates"]) >= 10


class TestLookup:
    def test_by_id(self):
        assert get_exercise("hip_thrust").name == "Dumbbell Hip Thrust"

    def test_by_alias(self):
        assert get_exercise("bridge").id == "glute_bridge"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_exercise("nonexistent_lift")

    def test_equipment_filter_is_what_you_own(self):
        """A bench exercise must not appear when you have no bench."""
        home = {Equipment.BODYWEIGHT, Equipment.DUMBBELL, Equipment.MAT}
        for ex in find(equipment=home):
            assert Equipment.BENCH not in ex.equipment

    def test_difficulty_ceiling_respected(self):
        assert all(e.difficulty == "beginner" for e in find(max_difficulty="beginner"))


class TestValidation:
    """The loader must reject malformed data loudly, at load time."""

    def _write(self, tmp_path, body):
        (tmp_path / "bad.yaml").write_text(body)
        load_library.cache_clear()
        return tmp_path

    def test_rejects_unknown_joint(self, tmp_path):
        d = self._write(tmp_path, """
- id: bad
  name: Bad
  category: test
  signal: {kind: joint_angle, joints: [left_shoulder, left_hip, left_tentacle]}
""")
        with pytest.raises(LibraryError, match="unknown joint"):
            load_library(str(d))
        load_library.cache_clear()

    def test_rejects_wrong_joint_count(self, tmp_path):
        d = self._write(tmp_path, """
- id: bad
  name: Bad
  category: test
  signal: {kind: joint_angle, joints: [left_hip, left_knee]}
""")
        with pytest.raises(LibraryError, match="needs 3 joints"):
            load_library(str(d))
        load_library.cache_clear()

    def test_rejects_inverted_thresholds(self, tmp_path):
        d = self._write(tmp_path, """
- id: bad
  name: Bad
  category: test
  signal: {kind: joint_angle, joints: [left_shoulder, left_hip, left_knee]}
  criteria: {enter_threshold: 0.2, exit_threshold: 0.8}
""")
        with pytest.raises(LibraryError, match="exit_threshold < enter_threshold"):
            load_library(str(d))
        load_library.cache_clear()

    def test_rejects_unknown_rule_type(self, tmp_path):
        d = self._write(tmp_path, """
- id: bad
  name: Bad
  category: test
  signal: {kind: joint_angle, joints: [left_shoulder, left_hip, left_knee]}
  rules:
    - type: vibes_check
""")
        with pytest.raises(LibraryError, match="bad form rule"):
            load_library(str(d))
        load_library.cache_clear()

    def test_rejects_out_of_range_glute_emphasis(self, tmp_path):
        d = self._write(tmp_path, """
- id: bad
  name: Bad
  category: test
  glute_emphasis: 1.7
  signal: {kind: joint_angle, joints: [left_shoulder, left_hip, left_knee]}
""")
        with pytest.raises(LibraryError, match="glute_emphasis"):
            load_library(str(d))
        load_library.cache_clear()

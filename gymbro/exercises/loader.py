"""Load and validate the exercise library."""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

from ..pose.skeleton import JOINT_INDEX
from .schema import (
    CameraView, Equipment, Exercise, Position, RepCriteria, SignalSpec, build_rule,
)

LIBRARY_DIR = Path(__file__).parent / "library"

_VALID_SIGNAL_KINDS = {"joint_angle", "joint_distance", "height", "abduction"}


class LibraryError(ValueError):
    """Raised when the library is malformed. Fail loudly at load, not mid-set."""


def _validate_joints(joints: tuple[str, ...], where: str) -> None:
    for j in joints:
        canonical = j.replace("{side}", "left")
        if canonical not in JOINT_INDEX:
            raise LibraryError(f"{where}: unknown joint {j!r}")


def _parse_signal(raw: dict, where: str) -> SignalSpec:
    kind = raw.get("kind")
    if kind not in _VALID_SIGNAL_KINDS:
        raise LibraryError(f"{where}: bad signal kind {kind!r}")

    joints = tuple(raw.get("joints", ()))
    reference = tuple(raw.get("reference", ()))
    _validate_joints(joints + reference, where)

    expected = {"joint_angle": 3, "joint_distance": 2, "height": 1, "abduction": 2}[kind]
    if len(joints) != expected:
        raise LibraryError(
            f"{where}: signal kind {kind!r} needs {expected} joints, got {len(joints)}"
        )

    return SignalSpec(
        kind=kind,
        joints=joints,
        reference=reference,
        invert=bool(raw.get("invert", False)),
    )


def _parse_exercise(raw: dict, source: Path) -> Exercise:
    where = f"{source.name}:{raw.get('id', '<no id>')}"

    for required in ("id", "name", "category", "signal"):
        if required not in raw:
            raise LibraryError(f"{where}: missing required field {required!r}")

    criteria_raw = dict(raw.get("criteria", {}))
    unknown = set(criteria_raw) - set(RepCriteria.__dataclass_fields__)
    if unknown:
        raise LibraryError(f"{where}: unknown criteria keys {sorted(unknown)}")
    criteria = RepCriteria(**criteria_raw)

    if not 0.0 < criteria.exit_threshold < criteria.enter_threshold < 1.0:
        raise LibraryError(
            f"{where}: require 0 < exit_threshold < enter_threshold < 1, got "
            f"exit={criteria.exit_threshold} enter={criteria.enter_threshold}"
        )

    try:
        rules = [build_rule(r) for r in raw.get("rules", [])]
    except (ValueError, TypeError) as exc:
        raise LibraryError(f"{where}: bad form rule: {exc}") from exc

    for rule in rules:
        _validate_joints(rule.joints, f"{where}:rule:{rule.name}")

    try:
        equipment = tuple(Equipment(e) for e in raw.get("equipment", ["bodyweight"]))
        position = Position(raw.get("position", "standing"))
        view = CameraView(raw.get("preferred_view", "side"))
    except ValueError as exc:
        raise LibraryError(f"{where}: {exc}") from exc

    emphasis = float(raw.get("glute_emphasis", 0.0))
    if not 0.0 <= emphasis <= 1.0:
        raise LibraryError(f"{where}: glute_emphasis must be in [0,1], got {emphasis}")

    return Exercise(
        id=raw["id"],
        name=raw["name"],
        category=raw["category"],
        primary_muscles=tuple(raw.get("primary_muscles", ())),
        secondary_muscles=tuple(raw.get("secondary_muscles", ())),
        equipment=equipment,
        position=position,
        signal=_parse_signal(raw["signal"], where),
        criteria=criteria,
        rules=rules,
        unilateral=bool(raw.get("unilateral", False)),
        preferred_view=view,
        difficulty=raw.get("difficulty", "beginner"),
        setup=raw.get("setup", "").strip(),
        cues=tuple(raw.get("cues", ())),
        glute_emphasis=emphasis,
        aliases=tuple(raw.get("aliases", ())),
        progressions=tuple(raw.get("progressions", ())),
        regressions=tuple(raw.get("regressions", ())),
    )


@functools.lru_cache(maxsize=1)
def load_library(directory: str | None = None) -> dict[str, Exercise]:
    """Load every exercise. Cached; call `load_library.cache_clear()` to reload."""
    path = Path(directory) if directory else LIBRARY_DIR
    exercises: dict[str, Exercise] = {}

    for yaml_file in sorted(path.glob("*.yaml")):
        raw = yaml.safe_load(yaml_file.read_text()) or []
        for entry in raw:
            ex = _parse_exercise(entry, yaml_file)
            if ex.id in exercises:
                raise LibraryError(f"duplicate exercise id {ex.id!r} in {yaml_file.name}")
            exercises[ex.id] = ex

    if not exercises:
        raise LibraryError(f"no exercises found in {path}")

    # Cross-reference progressions/regressions now, so a typo surfaces at load.
    for ex in exercises.values():
        for ref in ex.progressions + ex.regressions:
            if ref not in exercises:
                raise LibraryError(f"{ex.id}: references unknown exercise {ref!r}")

    return exercises


def get_exercise(exercise_id: str) -> Exercise:
    library = load_library()
    if exercise_id in library:
        return library[exercise_id]
    for ex in library.values():
        if exercise_id in ex.aliases:
            return ex
    raise KeyError(f"unknown exercise: {exercise_id!r}")


def find(
    *,
    category: str | None = None,
    equipment: set[Equipment] | None = None,
    max_difficulty: str | None = None,
    min_glute_emphasis: float = 0.0,
) -> list[Exercise]:
    """Query the library. `equipment` is what you *have*, not what's required."""
    order = {"beginner": 0, "intermediate": 1, "advanced": 2}
    out = []

    for ex in load_library().values():
        if category and ex.category != category:
            continue
        if ex.glute_emphasis < min_glute_emphasis:
            continue
        if max_difficulty and order.get(ex.difficulty, 0) > order.get(max_difficulty, 2):
            continue
        if equipment is not None and not set(ex.equipment) <= equipment:
            continue
        out.append(ex)

    return sorted(out, key=lambda e: (-e.glute_emphasis, e.name))

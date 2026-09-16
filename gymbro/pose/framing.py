"""Is the camera actually seeing enough to coach from?

Relevant when there is no room to back the camera off: a laptop on a table with
the user on the floor underneath sees a steep, close, partly-cropped view. The
honest thing is to say which joints are missing and what to change, rather than
silently scoring form from half a body.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .skeleton import JOINT_INDEX, Pose

# Fraction of the frame edge within which a joint counts as "nearly cropped".
EDGE_MARGIN = 0.04


@dataclass
class FramingReport:
    visible: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    at_edge: list[str] = field(default_factory=list)
    coverage: float = 0.0
    body_fill: float = 0.0
    advice: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.missing and self.coverage >= 0.8


def check_framing(
    pose: Pose | None,
    frame_width: int,
    frame_height: int,
    required_joints: tuple[str, ...],
    confidence_threshold: float = 0.4,
) -> FramingReport:
    """Report whether `required_joints` are usably in frame."""
    report = FramingReport()

    if pose is None:
        report.advice.append("Nobody detected - step into the camera's view.")
        return report

    mx, my = frame_width * EDGE_MARGIN, frame_height * EDGE_MARGIN

    for name in required_joints:
        idx = JOINT_INDEX[name]
        conf = float(pose.confidence[idx])
        x, y = pose.pixels[idx]

        if conf < confidence_threshold:
            report.missing.append(name)
            continue

        report.visible.append(name)
        if x < mx or x > frame_width - mx or y < my or y > frame_height - my:
            report.at_edge.append(name)

    total = len(required_joints)
    report.coverage = len(report.visible) / total if total else 0.0

    # How much of the frame the body occupies: too small and the pose model
    # has little to work with; too large and limbs leave the frame mid-rep.
    confident = pose.confidence >= confidence_threshold
    if confident.sum() >= 4:
        pts = pose.pixels[confident]
        span_x = float(pts[:, 0].max() - pts[:, 0].min())
        span_y = float(pts[:, 1].max() - pts[:, 1].min())
        report.body_fill = max(
            span_x / max(frame_width, 1), span_y / max(frame_height, 1)
        )

    report.advice = _advise(report, pose, frame_width, frame_height)
    return report


def _advise(
    report: FramingReport, pose: Pose, width: int, height: int
) -> list[str]:
    advice: list[str] = []

    if report.missing:
        joints = ", ".join(n.replace("_", " ") for n in report.missing[:3])
        advice.append(f"Can't see your {joints}.")

        # Say which way to move, from where the visible joints actually sit.
        confident = pose.confidence >= 0.4
        if confident.sum() >= 3:
            centre_y = float(pose.pixels[confident][:, 1].mean())
            if centre_y < height * 0.4:
                advice.append("Move further from the camera, or tilt it down.")
            elif centre_y > height * 0.6:
                advice.append("Move further from the camera, or tilt it up.")
            else:
                advice.append("Back up so your whole body fits in frame.")

    if report.at_edge and not report.missing:
        joints = ", ".join(n.replace("_", " ") for n in report.at_edge[:2])
        advice.append(f"Your {joints} are right at the edge - back up a little.")

    if report.body_fill > 0.95:
        advice.append("You fill the whole frame; limbs will leave it mid-rep.")
    elif 0 < report.body_fill < 0.35:
        advice.append("You're small in frame - move closer for better tracking.")

    if not advice and report.usable:
        advice.append("Framing looks good.")
    return advice

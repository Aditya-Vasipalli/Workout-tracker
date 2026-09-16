"""Export to open formats.

Google closed Fit API signups in May 2024 and sunsets the Fit APIs at the end
of 2026, so there is no supported path to push this data there. These exports
are the portable alternative: CSV for spreadsheets, JSON for anything else, and
TCX, which Strava, Garmin Connect and most fitness apps import directly.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from .db import Store

TCX_NS = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"


def export_sessions_csv(store: Store, path: Path) -> Path:
    rows = store._conn.execute(
        "SELECT se.plan_date, se.plan_name, se.status, se.started_at, se.finished_at, "
        "s.exercise_id, s.side, s.set_index, s.target_reps, s.counted_reps, "
        "s.good_reps, s.partial_reps, s.load_kg, s.form_score, s.form_coverage, "
        "s.duration_s "
        "FROM sets s JOIN sessions se ON se.id = s.session_id "
        "ORDER BY se.plan_date, s.id"
    ).fetchall()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        if not rows:
            fh.write("")
            return path
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))
    return path


def export_json(store: Store, path: Path) -> Path:
    sessions = []
    for s in store._conn.execute("SELECT * FROM sessions ORDER BY started_at"):
        session = dict(s)
        session["sets"] = [dict(x) for x in store.session_sets(s["id"])]
        sessions.append(session)

    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "format_version": 1,
        "sessions": sessions,
        "calibration": [
            dict(r) for r in store._conn.execute("SELECT * FROM calibration")
        ],
        "progression": [
            dict(r) for r in store._conn.execute("SELECT * FROM progression")
        ],
        "streak": dict(store.get_streak()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def export_tcx(store: Store, path: Path) -> Path:
    """TCX, which Strava and Garmin Connect import directly."""
    ET.register_namespace("", TCX_NS)
    root = ET.Element(f"{{{TCX_NS}}}TrainingCenterDatabase")
    activities = ET.SubElement(root, f"{{{TCX_NS}}}Activities")

    for s in store._conn.execute(
        "SELECT * FROM sessions WHERE status = 'completed' ORDER BY started_at"
    ):
        activity = ET.SubElement(activities, f"{{{TCX_NS}}}Activity", Sport="Other")
        ET.SubElement(activity, f"{{{TCX_NS}}}Id").text = s["started_at"]

        lap = ET.SubElement(activity, f"{{{TCX_NS}}}Lap", StartTime=s["started_at"])
        sets = store.session_sets(s["id"])
        total = sum((x["duration_s"] or 0) for x in sets)
        ET.SubElement(lap, f"{{{TCX_NS}}}TotalTimeSeconds").text = f"{total:.0f}"
        ET.SubElement(lap, f"{{{TCX_NS}}}DistanceMeters").text = "0"
        ET.SubElement(lap, f"{{{TCX_NS}}}Calories").text = "0"
        ET.SubElement(lap, f"{{{TCX_NS}}}Intensity").text = "Active"
        ET.SubElement(lap, f"{{{TCX_NS}}}TriggerMethod").text = "Manual"

        notes = "; ".join(
            f"{x['exercise_id']} {x['counted_reps']}/{x['target_reps']}" for x in sets
        )
        ET.SubElement(activity, f"{{{TCX_NS}}}Notes").text = f"{s['plan_name']}: {notes}"

    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def export_all(store: Store, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    stamp = datetime.now().strftime("%Y%m%d")
    return [
        export_sessions_csv(store, out / f"gymbro_sets_{stamp}.csv"),
        export_json(store, out / f"gymbro_full_{stamp}.json"),
        export_tcx(store, out / f"gymbro_{stamp}.tcx"),
    ]

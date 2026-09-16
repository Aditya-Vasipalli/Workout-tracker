"""Local SQLite store.

Everything lives here and nothing depends on a third-party fitness API. That is
a deliberate choice: Google closed Fit API signups in May 2024 and the whole Fit
API family sunsets at the end of 2026, so building on it was never an option.
`gymbro/store/export.py` writes open formats for anything that needs to leave.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

DEFAULT_DB = Path.home() / ".gymbro" / "gymbro.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    plan_date     TEXT NOT NULL,
    plan_name     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'in_progress',
    backend       TEXT,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS sets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    exercise_id   TEXT NOT NULL,
    set_index     INTEGER NOT NULL,
    side          TEXT,
    target_reps   INTEGER NOT NULL,
    counted_reps  INTEGER NOT NULL DEFAULT 0,
    good_reps     INTEGER NOT NULL DEFAULT 0,
    partial_reps  INTEGER NOT NULL DEFAULT 0,
    load_kg       REAL,
    form_score    REAL,
    form_coverage REAL,
    rom_low       REAL,
    rom_high      REAL,
    duration_s    REAL,
    completed     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reps (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id        INTEGER NOT NULL REFERENCES sets(id) ON DELETE CASCADE,
    rep_index     INTEGER NOT NULL,
    peak_value    REAL,
    concentric_s  REAL,
    eccentric_s   REAL,
    hold_s        REAL,
    full_range    INTEGER,
    tempo_ok      INTEGER,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS calibration (
    exercise_id   TEXT NOT NULL,
    side          TEXT NOT NULL DEFAULT '',
    rom_low       REAL NOT NULL,
    rom_high      REAL NOT NULL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (exercise_id, side)
);

CREATE TABLE IF NOT EXISTS progression (
    exercise_id   TEXT PRIMARY KEY,
    load_kg       REAL,
    target_reps   INTEGER NOT NULL DEFAULT 10,
    target_sets   INTEGER NOT NULL DEFAULT 3,
    consecutive_clears INTEGER NOT NULL DEFAULT 0,
    unlocked      INTEGER NOT NULL DEFAULT 1,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS streak (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    current       INTEGER NOT NULL DEFAULT 0,
    longest       INTEGER NOT NULL DEFAULT 0,
    last_session_date TEXT,
    debt_sessions INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS plans (
    plan_date     TEXT PRIMARY KEY,
    payload       TEXT NOT NULL,
    generated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sets_session  ON sets(session_id);
CREATE INDEX IF NOT EXISTS idx_sets_exercise ON sets(exercise_id);
CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(plan_date);
"""


class Store:
    def __init__(self, path: str | Path = DEFAULT_DB) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.execute("INSERT OR IGNORE INTO streak (id) VALUES (1)")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def _tx(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------- sessions

    def start_session(self, plan_date: str, plan_name: str, backend: str = "") -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO sessions (started_at, plan_date, plan_name, backend) "
                "VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), plan_date, plan_name, backend),
            )
            return int(cur.lastrowid)

    def finish_session(self, session_id: int, status: str, notes: str = "") -> None:
        if status not in ("completed", "partial", "abandoned"):
            raise ValueError(f"bad session status: {status!r}")
        with self._tx() as c:
            c.execute(
                "UPDATE sessions SET finished_at = ?, status = ?, notes = ? WHERE id = ?",
                (datetime.now().isoformat(timespec="seconds"), status, notes, session_id),
            )

    def record_set(self, session_id: int, **fields) -> int:
        fields.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        fields["session_id"] = session_id
        cols = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        with self._tx() as c:
            cur = c.execute(
                f"INSERT INTO sets ({cols}) VALUES ({placeholders})", tuple(fields.values())
            )
            return int(cur.lastrowid)

    def record_reps(self, set_id: int, reps) -> None:
        with self._tx() as c:
            c.executemany(
                "INSERT INTO reps (set_id, rep_index, peak_value, concentric_s, "
                "eccentric_s, hold_s, full_range, tempo_ok, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (set_id, r.index, r.peak_value, r.concentric_s, r.eccentric_s,
                     r.hold_s, int(r.full_range), int(r.tempo_ok), "; ".join(r.notes))
                    for r in reps
                ],
            )

    def session_sets(self, session_id: int) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT * FROM sets WHERE session_id = ? ORDER BY id", (session_id,)
        ))

    def recent_sessions(self, limit: int = 30) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
        ))

    # ---------------------------------------------------------- calibration

    def save_calibration(self, exercise_id: str, low: float, high: float, side: str = "") -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO calibration (exercise_id, side, rom_low, rom_high, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(exercise_id, side) DO UPDATE SET "
                "rom_low = excluded.rom_low, rom_high = excluded.rom_high, "
                "updated_at = excluded.updated_at",
                (exercise_id, side, low, high, datetime.now().isoformat(timespec="seconds")),
            )

    def get_calibration(self, exercise_id: str, side: str = "") -> tuple[float, float] | None:
        row = self._conn.execute(
            "SELECT rom_low, rom_high FROM calibration WHERE exercise_id = ? AND side = ?",
            (exercise_id, side),
        ).fetchone()
        return (row["rom_low"], row["rom_high"]) if row else None

    # ---------------------------------------------------------- progression

    def get_progression(self, exercise_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM progression WHERE exercise_id = ?", (exercise_id,)
        ).fetchone()

    def upsert_progression(self, exercise_id: str, **fields) -> None:
        fields["updated_at"] = datetime.now().isoformat(timespec="seconds")
        existing = self.get_progression(exercise_id)
        with self._tx() as c:
            if existing is None:
                fields["exercise_id"] = exercise_id
                cols = ", ".join(fields)
                ph = ", ".join("?" for _ in fields)
                c.execute(f"INSERT INTO progression ({cols}) VALUES ({ph})", tuple(fields.values()))
            else:
                assignments = ", ".join(f"{k} = ?" for k in fields)
                c.execute(
                    f"UPDATE progression SET {assignments} WHERE exercise_id = ?",
                    (*fields.values(), exercise_id),
                )

    def exercise_history(self, exercise_id: str, limit: int = 20) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT s.*, se.plan_date FROM sets s "
            "JOIN sessions se ON se.id = s.session_id "
            "WHERE s.exercise_id = ? ORDER BY s.id DESC LIMIT ?",
            (exercise_id, limit),
        ))

    # --------------------------------------------------------------- streak

    def get_streak(self) -> sqlite3.Row:
        return self._conn.execute("SELECT * FROM streak WHERE id = 1").fetchone()

    def update_streak(self, **fields) -> None:
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._tx() as c:
            c.execute(f"UPDATE streak SET {assignments} WHERE id = 1", tuple(fields.values()))

    # ---------------------------------------------------------------- plans

    def save_plan(self, plan_date: str, payload: dict) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO plans (plan_date, payload, generated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(plan_date) DO UPDATE SET payload = excluded.payload",
                (plan_date, json.dumps(payload), datetime.now().isoformat(timespec="seconds")),
            )

    def get_plan(self, plan_date: str) -> dict | None:
        row = self._conn.execute(
            "SELECT payload FROM plans WHERE plan_date = ?", (plan_date,)
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def sessions_in_range(self, start: date, end: date) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT * FROM sessions WHERE plan_date BETWEEN ? AND ? ORDER BY plan_date",
            (start.isoformat(), end.isoformat()),
        ))

    def muscle_volume(self, since: date) -> dict[str, int]:
        """Completed reps per exercise since a date -- drives balanced programming."""
        rows = self._conn.execute(
            "SELECT s.exercise_id, SUM(s.good_reps) AS vol FROM sets s "
            "JOIN sessions se ON se.id = s.session_id "
            "WHERE se.plan_date >= ? AND s.completed = 1 GROUP BY s.exercise_id",
            (since.isoformat(),),
        )
        return {r["exercise_id"]: int(r["vol"] or 0) for r in rows}

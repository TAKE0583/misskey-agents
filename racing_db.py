#!/usr/bin/env python3
"""Local racing database (SQLite) for misskey-agents.

Two layers, per the 2026-11-09 design (goal files:
data-db-design-20261109.html):

  1. Horse career cache — db.netkeiba career rows for each runner,
     fetched at most once a day per horse, so a fixture build can
     render detailed recent form (margin / passing / final 3F /
     horse weight) without depending on the day's scrape of the
     shutuba_past page.
  2. Results archive — every ingested results fixture lands here
     (races / runners / payouts), so results accumulate in one
     queryable store instead of scattered per-race JSON.

The DB is a single file at data/racing.db (override with the
RACING_DB_PATH environment variable, used by the tests). Everything
is upsert-based and idempotent: re-ingesting a race or a career
never duplicates rows.

Career-sourced rows are keyed by the db.netkeiba race id. For JRA
races that id is identical to this project's race_id scheme; for
NAR races the two schemes differ, so the recent-form query dedupes
by (date, race name) to hide any double-counting between the
career layer and the archive layer.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent / "data" / "racing.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    race_id     TEXT PRIMARY KEY,
    date        TEXT,
    jurisdiction TEXT,
    venue       TEXT,
    race_number INTEGER,
    name        TEXT,
    grade       TEXT,
    surface     TEXT,
    distance_m  INTEGER,
    going       TEXT,
    weather     TEXT,
    field_size  INTEGER
);
CREATE TABLE IF NOT EXISTS runners (
    race_id         TEXT NOT NULL,
    number          INTEGER NOT NULL,
    bracket         INTEGER,
    horse_id        TEXT,
    name            TEXT,
    sex_age         TEXT,
    weight_kg       REAL,
    jockey          TEXT,
    jockey_id       TEXT,
    trainer         TEXT,
    trainer_id      TEXT,
    position        INTEGER,
    status          TEXT,
    time            TEXT,
    margin          TEXT,
    popularity      INTEGER,
    win_odds        REAL,
    final_3f        REAL,
    passing         TEXT,
    horse_weight    INTEGER,
    horse_weight_diff INTEGER,
    PRIMARY KEY (race_id, number)
);
CREATE INDEX IF NOT EXISTS idx_runners_horse ON runners(horse_id);
CREATE TABLE IF NOT EXISTS payouts (
    race_id TEXT NOT NULL,
    ticket  TEXT NOT NULL,
    combo   TEXT NOT NULL,
    payout  INTEGER NOT NULL,
    PRIMARY KEY (race_id, ticket, combo)
);
CREATE TABLE IF NOT EXISTS horses (
    horse_id          TEXT PRIMARY KEY,
    name              TEXT,
    career_fetched_at TEXT,
    career_through    TEXT
);
CREATE TABLE IF NOT EXISTS ingest_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT,
    target     TEXT,
    fetched_at TEXT,
    rows       INTEGER,
    ok         INTEGER,
    detail     TEXT
);
"""


def db_path() -> Path:
    override = os.environ.get("RACING_DB_PATH")
    return Path(override) if override else DEFAULT_DB_PATH


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = Path(path) if path else db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log_ingest(conn: sqlite3.Connection, source: str, target: str,
               rows: int, ok: bool, detail: str = "") -> None:
    conn.execute(
        "INSERT INTO ingest_log (source, target, fetched_at, rows, ok,"
        " detail) VALUES (?, ?, ?, ?, ?, ?)",
        (source, target, _now(), rows, 1 if ok else 0, detail),
    )


def upsert_race(conn: sqlite3.Connection, race: dict) -> None:
    conn.execute(
        """
        INSERT INTO races (race_id, date, jurisdiction, venue,
            race_number, name, grade, surface, distance_m, going,
            weather, field_size)
        VALUES (:race_id, :date, :jurisdiction, :venue, :race_number,
            :name, :grade, :surface, :distance_m, :going, :weather,
            :field_size)
        ON CONFLICT(race_id) DO UPDATE SET
            date=COALESCE(excluded.date, races.date),
            jurisdiction=COALESCE(excluded.jurisdiction,
                                  races.jurisdiction),
            venue=COALESCE(NULLIF(excluded.venue, ''), races.venue),
            race_number=COALESCE(excluded.race_number,
                                 races.race_number),
            name=COALESCE(NULLIF(excluded.name, ''), races.name),
            grade=COALESCE(NULLIF(excluded.grade, ''), races.grade),
            surface=COALESCE(NULLIF(excluded.surface, ''),
                             races.surface),
            distance_m=COALESCE(excluded.distance_m, races.distance_m),
            going=COALESCE(NULLIF(excluded.going, ''), races.going),
            weather=COALESCE(NULLIF(excluded.weather, ''),
                             races.weather),
            field_size=COALESCE(excluded.field_size, races.field_size)
        """,
        {
            "race_id": race.get("race_id"),
            "date": race.get("date"),
            "jurisdiction": race.get("jurisdiction"),
            "venue": race.get("venue"),
            "race_number": race.get("race_number"),
            "name": race.get("name"),
            "grade": race.get("grade"),
            "surface": race.get("surface"),
            "distance_m": race.get("distance_m"),
            "going": race.get("going"),
            "weather": race.get("weather"),
            "field_size": race.get("field_size"),
        },
    )


_RUNNER_COLS = (
    "race_id", "number", "bracket", "horse_id", "name", "sex_age",
    "weight_kg", "jockey", "jockey_id", "trainer", "trainer_id",
    "position", "status", "time", "margin", "popularity", "win_odds",
    "final_3f", "passing", "horse_weight", "horse_weight_diff",
)


def upsert_runner(conn: sqlite3.Connection, runner: dict) -> None:
    cols = ", ".join(_RUNNER_COLS)
    marks = ", ".join(f":{c}" for c in _RUNNER_COLS)
    # COALESCE merge: multiple sources (career page, results
    # fixture, archive) hold overlapping subsets, and a NULL from
    # one must never erase a value another source provided.
    updates = ", ".join(
        f"{c}=COALESCE(excluded.{c}, runners.{c})" for c in _RUNNER_COLS
        if c not in ("race_id", "number")
    )
    conn.execute(
        f"INSERT INTO runners ({cols}) VALUES ({marks}) "
        f"ON CONFLICT(race_id, number) DO UPDATE SET {updates}",
        {c: runner.get(c) for c in _RUNNER_COLS},
    )


def ingest_results_fixture(conn: sqlite3.Connection, fixture: dict,
                           results: dict, source: str = "fixture") -> int:
    """Store one prediction fixture + its results fixture.

    The runners come from the results' finish table joined back to
    the fixture's entry data (bracket, horse_id, connections). Older
    results fixtures lack the detail fields (passing, final_3f,
    horse weight); those columns simply stay NULL.
    Returns the number of runner rows written.
    """
    race = fixture.get("race") or {}
    race_id = race.get("race_id") or results.get("race_id")
    if not race_id:
        return 0
    course = str(race.get("course") or "")
    surface = course.split("・")[0] if course else None
    upsert_race(conn, {
        "race_id": race_id,
        "date": race.get("date"),
        "jurisdiction": "NAR" if _is_nar_race_id(race_id) else "JRA",
        "venue": race.get("venue"),
        "race_number": race.get("race_number"),
        "name": race.get("name"),
        "grade": race.get("grade"),
        "surface": surface,
        "distance_m": race.get("distance_m"),
        "going": race.get("going"),
        "weather": race.get("weather"),
        "field_size": race.get("field_size"),
    })
    entries = {h["number"]: h for h in fixture.get("horses") or []}
    written = 0
    for fin in results.get("finish") or []:
        entry = entries.get(fin.get("number")) or {}
        upsert_runner(conn, {
            "race_id": race_id,
            "number": fin.get("number"),
            "bracket": fin.get("bracket", entry.get("bracket")),
            "horse_id": fin.get("horse_id") or entry.get("horse_id"),
            "name": fin.get("name") or entry.get("name"),
            "sex_age": fin.get("sex_age") or entry.get("sex_age"),
            "weight_kg": fin.get("weight_kg") or entry.get("weight_kg"),
            "jockey": fin.get("jockey") or entry.get("jockey"),
            "jockey_id": entry.get("jockey_id"),
            "trainer": fin.get("trainer") or entry.get("trainer"),
            "trainer_id": entry.get("trainer_id"),
            "position": fin.get("position"),
            "status": fin.get("status"),
            "time": fin.get("time"),
            "margin": fin.get("margin"),
            "popularity": fin.get("popularity"),
            "win_odds": fin.get("win_odds"),
            "final_3f": fin.get("final_3f"),
            "passing": fin.get("passing"),
            "horse_weight": fin.get("horse_weight"),
            "horse_weight_diff": fin.get("horse_weight_diff"),
        })
        written += 1
    payouts = results.get("payouts_jpy") or results.get("payouts") or {}
    for ticket, combos in payouts.items():
        for combo, amount in (combos or {}).items():
            conn.execute(
                "INSERT INTO payouts (race_id, ticket, combo, payout)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(race_id, ticket, combo)"
                " DO UPDATE SET payout=excluded.payout",
                (race_id, ticket, combo, int(amount)),
            )
    log_ingest(conn, source, race_id, written, True)
    conn.commit()
    return written


def _is_nar_race_id(race_id: str) -> bool:
    # Both schemes are 12 digits: YYYY + venue(2) + ... The venue
    # codes never overlap (JRA 01-10, NAR 30-65), so positions 4:6
    # discriminate cleanly.
    try:
        return int(str(race_id)[4:6]) >= 30
    except (TypeError, ValueError):
        return False


def ingest_career(conn: sqlite3.Connection, horse_id: str,
                  horse_name: str | None, rows: list[dict],
                  source: str = "career") -> int:
    """Store parsed db.netkeiba career rows for one horse.

    Each row is a normalized dict as produced by
    build_fixture.fetch_horse_career(). Races are upserted from the
    row's own race-level fields; the runner row is keyed by the
    db race id (identical to the project race_id for JRA).
    """
    written = 0
    through = None
    for row in rows:
        race_id = row.get("race_id")
        if not race_id:
            continue
        upsert_race(conn, {
            "race_id": race_id,
            "date": row.get("date"),
            "jurisdiction": row.get("jurisdiction"),
            "venue": row.get("venue"),
            "race_number": row.get("race_number"),
            "name": row.get("race_name"),
            "grade": row.get("grade"),
            "surface": row.get("surface"),
            "distance_m": row.get("distance_m"),
            "going": row.get("going"),
            "weather": row.get("weather"),
            "field_size": row.get("field_size"),
        })
        upsert_runner(conn, {
            "race_id": race_id,
            "number": row.get("number"),
            "bracket": row.get("bracket"),
            "horse_id": horse_id,
            "name": horse_name,
            "sex_age": None,
            "weight_kg": row.get("weight_kg"),
            "jockey": row.get("jockey"),
            "jockey_id": row.get("jockey_id"),
            "trainer": None,
            "trainer_id": None,
            "position": row.get("position"),
            "status": row.get("status"),
            "time": row.get("time"),
            "margin": row.get("margin"),
            "popularity": row.get("popularity"),
            "win_odds": row.get("win_odds"),
            "final_3f": row.get("final_3f"),
            "passing": row.get("passing"),
            "horse_weight": row.get("horse_weight"),
            "horse_weight_diff": row.get("horse_weight_diff"),
        })
        written += 1
        if row.get("date") and (through is None or row["date"] > through):
            through = row["date"]
    conn.execute(
        "INSERT INTO horses (horse_id, name, career_fetched_at,"
        " career_through) VALUES (?, ?, ?, ?)"
        " ON CONFLICT(horse_id) DO UPDATE SET"
        " name=excluded.name,"
        " career_fetched_at=excluded.career_fetched_at,"
        " career_through=excluded.career_through",
        (horse_id, horse_name, _now(), through),
    )
    log_ingest(conn, source, horse_id, written, True)
    conn.commit()
    return written


def career_fetched_today(conn: sqlite3.Connection, horse_id: str) -> bool:
    row = conn.execute(
        "SELECT career_fetched_at FROM horses WHERE horse_id = ?",
        (horse_id,),
    ).fetchone()
    if not row or not row["career_fetched_at"]:
        return False
    return row["career_fetched_at"][:10] == _now()[:10]


def recent_runs(conn: sqlite3.Connection, horse_id: str,
                before_date: str, limit: int = 5,
                exclude_race_id: str | None = None) -> list[dict]:
    """The horse's most recent runs strictly before `before_date`.

    Deduped by (date, race name): the career layer and the archive
    layer can hold the same NAR race under different ids.
    """
    cur = conn.execute(
        """
        SELECT r.*, ra.date AS race_date, ra.venue AS race_venue,
               ra.name AS race_name, ra.surface AS race_surface,
               ra.distance_m AS race_distance
        FROM runners r JOIN races ra ON ra.race_id = r.race_id
        WHERE r.horse_id = ?
          AND (ra.date < ? OR (ra.date = ? AND r.race_id != ?))
        ORDER BY ra.date DESC, r.race_id DESC
        """,
        (horse_id, before_date, before_date, exclude_race_id or ""),
    )
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in cur.fetchall():
        key = (row["race_date"], row["race_name"] or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
        if len(out) >= limit:
            break
    return out


def format_form_run(run: dict) -> str:
    """One detailed recent-form string, e.g.
    '2026-10-18 京都 芝2000 エリザベス女王杯 3着（着差0.2 / 通過5-5-4 / 上り34.1 / 馬体重480(+2)）'.
    Missing details are omitted, never invented."""
    date = str(run.get("race_date") or "")
    venue = str(run.get("race_venue") or "").replace("競馬場", "")
    surface = run.get("race_surface") or ""
    distance = run.get("race_distance")
    course = f"{surface}{distance}" if distance else surface
    name = str(run.get("race_name") or "")
    head = " ".join(p for p in (date, venue, course, name) if p)
    position = run.get("position")
    if position:
        head += f" {position}着"
    elif run.get("status"):
        head += f" {run['status']}"
    details = []
    margin = run.get("margin")
    if margin not in (None, "", "-"):
        details.append(f"着差{margin}")
    if run.get("passing"):
        details.append(f"通過{run['passing']}")
    if run.get("final_3f") is not None:
        details.append(f"上り{run['final_3f']:.1f}")
    weight = run.get("horse_weight")
    if weight is not None:
        diff = run.get("horse_weight_diff")
        if diff is not None:
            details.append(f"馬体重{weight}({diff:+d})")
        else:
            details.append(f"馬体重{weight}")
    if details:
        head += "（" + " / ".join(details) + "）"
    return head


def race_ingest_complete(conn: sqlite3.Connection, race_id: str) -> bool:
    # A successful ingest log entry is the primary signal (results
    # fixtures omit non-finishers, so a runner-count check alone can
    # never reach field_size for races with scratchings/exclusions).
    logged = conn.execute(
        "SELECT 1 FROM ingest_log WHERE target = ? AND ok = 1 LIMIT 1",
        (race_id,),
    ).fetchone()
    if logged:
        return True
    row = conn.execute(
        "SELECT field_size FROM races WHERE race_id = ?", (race_id,)
    ).fetchone()
    if not row:
        return False
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM runners WHERE race_id = ?",
        (race_id,),
    ).fetchone()["c"]
    expected = row["field_size"] or 0
    return expected > 0 and count >= expected


def _load_pair(pred_path: Path) -> tuple[dict, dict] | None:
    results_path = pred_path.with_name(
        pred_path.name.replace(".json", "-results.json")
    )
    if not results_path.exists():
        return None
    fixture = json.loads(pred_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    return fixture, results


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="racing_db maintenance")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest-fixtures",
                         help="ingest prediction+results fixture pairs")
    ing.add_argument("fixtures", nargs="+",
                     help="prediction fixture paths (results pair by name)")
    sub.add_parser("stats", help="row counts")
    args = ap.parse_args()

    conn = connect()
    if args.cmd == "ingest-fixtures":
        total = 0
        for name in args.fixtures:
            pair = _load_pair(Path(name))
            if pair is None:
                print(f"skip (no results pair): {name}")
                continue
            n = ingest_results_fixture(conn, pair[0], pair[1])
            total += n
            print(f"ingested {name}: {n} runners")
        print(f"total runners written: {total}")
    elif args.cmd == "stats":
        for table in ("races", "runners", "payouts", "horses"):
            c = conn.execute(
                f"SELECT COUNT(*) AS c FROM {table}"
            ).fetchone()["c"]
            print(f"{table}: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

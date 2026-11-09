#!/usr/bin/env python3
"""Nightly results archiver for the racing DB (design layer 2).

Enumerates a day's races from netkeiba's race-list pages and stores
every result into data/racing.db via racing_db.ingest_results_fixture:

  - JRA: all races of the day.
  - NAR: the main race of each venue only (the day's highest race
    number), matching the design's recommended scope — NAR runs
    ~12 races x 14 venues a day, most of which nothing here consumes.

The primary path is results-only (finish table + payouts from the
result page); the full prediction fixture is the fallback for race
metadata. Races we actually predict arrive richer through their own
fixtures and COALESCE-merge over the archive rows. Ingestion is
idempotent, and races already recorded are skipped, so re-running
a day is cheap gap-fill.

Each race is ingested in a child process with a hard timeout: the
vendor scrapers issue requests without timeouts, and one hung
connection must never stall the whole night.

Usage:
  archive_results.py [--date YYYY-MM-DD] [--days N] [--sleep SEC]
                      [--race-timeout SEC]
Exits non-zero when any race failed after both paths.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import sys
import time
from datetime import date, timedelta

os.environ.setdefault("RACING_DB_READONLY", "1")  # no career fetches

import requests  # noqa: E402

import build_fixture  # noqa: E402
import racing_db  # noqa: E402

JRA_TOP = "https://race.netkeiba.com/top"
NAR_TOP = "https://nar.netkeiba.com/top"

JRA_VENUES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}
NAR_VENUES = {
    "30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和",
    "43": "船橋", "44": "大井", "45": "川崎", "46": "金沢",
    "47": "笠松", "48": "名古屋", "50": "園田", "51": "姫路",
    "54": "高知", "55": "佐賀", "65": "帯広",
}


def discover(date_iso: str, nar: bool) -> list[str]:
    host = NAR_TOP if nar else JRA_TOP
    ymd = date_iso.replace("-", "")
    resp = requests.get(
        f"{host}/race_list_sub.html?kaisai_date={ymd}",
        headers=build_fixture.HEADERS, timeout=30,
    )
    resp.raise_for_status()
    html = resp.content.decode("utf-8", errors="replace")
    ids = sorted(set(re.findall(r"race_id=(\d{12})", html)))
    if nar:
        mains: dict[str, str] = {}
        for rid in ids:
            venue = rid[4:6]
            if venue not in mains or rid[-2:] > mains[venue][-2:]:
                mains[venue] = rid
        ids = sorted(mains.values())
    return ids


def _results_fixture(race_id: str, date_iso: str, nar: bool) -> dict:
    """Minimal fixture for the results-only path, with the race name
    and going/weather lifted from the result page when available."""
    venues = NAR_VENUES if nar else JRA_VENUES
    name, going, weather = "", None, None
    try:
        host = (
            build_fixture.NAR_RACE_URL if nar
            else "https://race.netkeiba.com"
        )
        resp = requests.get(
            f"{host}/race/result.html?race_id={race_id}",
            headers=build_fixture.HEADERS, timeout=30,
        )
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(
            resp.content.decode("utf-8", errors="replace"), "html.parser"
        )
        el = soup.find(class_="RaceName")
        if el is not None:
            name = el.get_text(strip=True)
        data01 = soup.find("div", class_="RaceData01")
        if data01 is not None:
            text = data01.get_text("", strip=True)
            m = re.search(r"馬場:([^\s/]+)", text)
            if m:
                going = m.group(1)
            m = re.search(r"天候:([^\s/]+)", text)
            if m:
                weather = m.group(1)
    except Exception:  # noqa: BLE001 - metadata is a nice-to-have
        pass
    return {
        "race": {
            "race_id": race_id,
            "edition": 0,
            "name": name,
            "grade": "",
            "date": date_iso,
            "venue": venues.get(race_id[4:6], ""),
            "race_number": int(race_id[-2:]),
            "course": "",
            "distance_m": None,
            "going": going,
            "weather": weather,
            "field_size": None,
        },
        "horses": [],
    }


def _ingest_child(race_id: str, date_iso: str, nar: bool, queue) -> None:
    """One race, in a child process: results-only path first, the
    full fixture build as fallback. Reports through the queue."""
    try:
        conn = racing_db.connect()
        try:
            fixture = _results_fixture(race_id, date_iso, nar)
            results = build_fixture.build_results(
                race_id, fixture, nar=nar
            )
        except Exception:  # noqa: BLE001 - fall back to fixture path
            fixture = build_fixture.build_fixture(
                race_id, 0, "", nar=nar
            )
            results = build_fixture.build_results(
                race_id, fixture, nar=nar
            )
        n = racing_db.ingest_results_fixture(
            conn, fixture, results, source="archive"
        )
        queue.put(("ok", n))
    except Exception as exc:  # noqa: BLE001
        queue.put(("err", f"{type(exc).__name__}: {exc}"))


def archive_race(race_id: str, date_iso: str, nar: bool,
                 timeout_s: float) -> int:
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_ingest_child, args=(race_id, date_iso, nar, queue)
    )
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        raise TimeoutError(f"race ingest exceeded {timeout_s:.0f}s")
    if queue.empty():
        raise RuntimeError(f"child exited rc={proc.exitcode} silently")
    kind, payload = queue.get()
    if kind == "err":
        raise RuntimeError(payload)
    return int(payload)


def archive_day(conn, date_iso: str, sleep_s: float,
                race_timeout: float) -> tuple[int, int, int]:
    archived = skipped = failed = 0
    plan: list[tuple[str, bool]] = []
    for nar in (False, True):
        try:
            plan += [(rid, nar) for rid in discover(date_iso, nar)]
        except Exception as exc:  # noqa: BLE001
            print(f"{date_iso} discovery nar={nar} failed: {exc}",
                  flush=True)
            failed += 1
    for race_id, nar in plan:
        if racing_db.race_ingest_complete(conn, race_id):
            skipped += 1
            continue
        try:
            n = archive_race(race_id, date_iso, nar, race_timeout)
            print(f"{date_iso} {race_id} nar={nar}: {n} runners",
                  flush=True)
            archived += 1
        except Exception as exc:  # noqa: BLE001
            print(f"{date_iso} {race_id} nar={nar}: FAILED {exc}",
                  flush=True)
            racing_db.log_ingest(
                conn, "archive", race_id, 0, False, str(exc)
            )
            conn.commit()
            failed += 1
        time.sleep(sleep_s)
    return archived, skipped, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--days", type=int, default=1,
                    help="also walk back this many days (gap-fill)")
    ap.add_argument("--sleep", type=float, default=1.2,
                    help="seconds between race ingests")
    ap.add_argument("--race-timeout", type=float, default=180,
                    help="hard per-race timeout in seconds")
    args = ap.parse_args()

    first = (
        date.fromisoformat(args.date) if args.date else date.today()
    )
    conn = racing_db.connect()
    total_failed = 0
    for back in range(args.days):
        day = (first - timedelta(days=back)).isoformat()
        archived, skipped, failed = archive_day(
            conn, day, args.sleep, args.race_timeout
        )
        print(
            f"{day}: archived={archived} skipped={skipped} "
            f"failed={failed}",
            flush=True,
        )
        total_failed += failed
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())

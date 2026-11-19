#!/usr/bin/env python3
"""Run state, race identity checks and locking (extracted from orchestrator.py, 2026-11-19 split; behavior unchanged)."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

# --- Run state, locking, identity ------------------------------------------
# Live runs and dry runs keep separate state namespaces: a dry run must
# never overwrite production state (external review 2026-11-04, finding 1).
# Live posting is incremental and resumable: generated texts and every
# posted note ID are saved as they happen, so a crash or provider failure
# mid-run resumes where it stopped instead of re-posting (and re-billing)
# everything (finding 2). A per-race lock keeps concurrent runs out.


def fixture_race_id(fixture: dict) -> str | None:
    return (fixture.get("race") or {}).get("race_id")


def _legacy_results_mismatch(fixture: dict, results: dict) -> str | None:
    """For files without a race_id: prove sameness, don't assume it.
    Date + venue alone cannot distinguish two races on the same card
    (re-review 2026-11-04, finding 3), so also require the race name and
    a runner-by-runner match of the finish against the fixture. Returns
    a mismatch reason, or None when the pair provably is the same race."""
    fr = fixture["race"]
    if fr.get("date") != results.get("date"):
        return f"date: fixture {fr.get('date')} vs results {results.get('date')}"
    if fr.get("venue") != results.get("venue"):
        return f"venue: fixture {fr.get('venue')} vs results {results.get('venue')}"
    if not fr.get("name") or fr["name"] not in (results.get("race") or ""):
        return f"race name {fr.get('name')} not found in results {results.get('race')}"
    names = {h["number"]: h["name"] for h in fixture["horses"]}
    finish = results.get("finish") or []
    if not finish:
        return "results have no finish list to verify against"
    for f in finish:
        if names.get(f.get("number")) != f.get("name"):
            return (
                f"runner mismatch: number {f.get('number')} is "
                f"{f.get('name')} in results but {names.get(f.get('number'))} in fixture"
            )
    return None


def _legacy_state_mismatch(fixture: dict, state: dict) -> str | None:
    """A state without a race_id must at least name the same race."""
    expected = f"第{fixture['race'].get('edition')}回 {fixture['race'].get('name')}"
    if state.get("race") != expected:
        return f"state is for {state.get('race')}, fixture is {expected}"
    return None


def check_race_identity(fixture: dict, results: dict | None,
                        state: dict | None) -> None:
    """Refuse to combine data from different races (finding 8). Uses the
    shared race_id when present; legacy files must PROVE sameness via
    name and field composition, and anything indeterminate stops the run
    (re-review finding 3)."""
    fid = fixture_race_id(fixture)
    if results is not None:
        rid = results.get("race_id")
        if fid and rid:
            if fid != rid:
                raise SystemExit(
                    f"race mismatch: fixture is {fid} but results are {rid}"
                )
        else:
            reason = _legacy_results_mismatch(fixture, results)
            if reason:
                raise SystemExit(f"race mismatch (legacy check): {reason}")
    if state is not None:
        sid = state.get("race_id")
        if fid and sid:
            if fid != sid:
                raise SystemExit(
                    f"race mismatch: fixture is {fid} but state is {sid}"
                )
        else:
            reason = _legacy_state_mismatch(fixture, state)
            if reason:
                raise SystemExit(f"race mismatch (legacy state check): {reason}")


def _migrate_legacy_state(data: dict) -> dict:
    """Upgrade a v1 state in place (re-review 2026-11-04, finding 2).

    A v1 state holds predictions/marks/conclusion but no stage, texts, or
    note IDs. Anything that evidently ran is treated as COMPLETE: the new
    code must never regenerate and repost it (regenerated text differs,
    so exact-text reconciliation alone cannot prevent duplicates). A v1
    review_posted_at becomes a legacy review record whose note ID is
    unknown — run_review refuses to touch it without --force-review."""
    data = dict(data)
    data["version"] = 2
    data["legacy"] = True
    data.setdefault("note_ids", {})
    if data.get("predictions") or data.get("marks") or data.get("conclusion_ranking"):
        data["stage"] = "complete"
    rp = data.get("review_posted_at")
    if rp and "review" not in data:
        data["review"] = {
            "note_id": None,
            "legacy_posted_at": rp,
            "reflections": {},
        }
    return data


class RunState:
    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data

    @classmethod
    def open(cls, path: Path, fixture: dict, mode: str) -> "RunState":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            state = cls(path, data)
            if data.get("version") != 2:
                state.data = _migrate_legacy_state(data)
                state.save()  # make the upgrade durable immediately
            return state
        import uuid

        data = {
            "version": 2,
            "race_id": fixture_race_id(fixture),
            "mode": mode,
            "run_id": uuid.uuid4().hex[:12],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "stage": "new",
            "note_ids": {},
        }
        state = cls(path, data)
        state.save()
        return state

    def save(self) -> None:
        self.data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.path)


def state_path_for(fixture_path: str, live: bool, review_only: bool = False) -> Path:
    stem = Path(fixture_path).stem
    base = Path("runs") if live else Path("runs") / "dryrun"
    suffix = "-review-state.json" if review_only else "-state.json"
    return base / f"{stem}{suffix}"


class RaceLock:
    """Exclusive lock per fixture stem (live runs only), via flock(2):
    the OS holds the lock on the open file description, so there is no
    create-then-write window and no stale-file judgement call — a dead
    holder's lock is released by the kernel (re-review 2026-11-04,
    finding 6). The pid written into the file is informational only."""

    def __init__(self, fixture_path: str):
        stem = Path(fixture_path).stem
        self.path = Path("runs") / f"{stem}.lock"
        self._fd: int | None = None

    def __enter__(self) -> "RaceLock":
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = os.read(fd, 64).decode(errors="replace").strip()
            os.close(fd)
            raise SystemExit(
                "another run holds the lock for this race "
                f"({holder or 'pid unknown'}); refusing to run concurrently"
            )
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()}\n".encode())
        self._fd = fd
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            import fcntl

            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


def _backfill_race_id(store: "RunState", fixture: dict) -> None:
    """Once identity is proven, stamp the fixture's race_id into a state
    that lacks one, so later runs use the strong ID check (finding 3)."""
    fid = fixture_race_id(fixture)
    if fid and not store.data.get("race_id"):
        store.data["race_id"] = fid
        store.save()

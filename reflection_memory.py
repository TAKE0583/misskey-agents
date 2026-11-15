"""Per-AI reflection memory for the predictors.

After each race, every AI writes a short reflection on its own
prediction (generated in orchestrator.run_review and posted as its own
note). This module records those reflections per AI, and builds a
short digest block -- that AI's most recent reflections -- which the
orchestrator injects into the AI's OWN prediction prompt for the next
races.

Design (2026-11-16, user decision): the experiment's ground rule is
that all AIs receive the SAME external information and argue on equal
footing. A reflection memory does not break that rule: every AI gets
exactly the same treatment, and what it sees is only its own past
words -- the asymmetry is "one's own past", not an information
advantage handed to one model. The moderator / conclusion account
never predicts and gets no memory; the guest account is outside the
system. Whether carrying a memory actually changes (or improves) an
AI's judgment is part of what the experiment observes.

Store: one JSON file, runs/reflection-memory.json for live runs and
runs/dryrun/reflection-memory.json for dry runs, mirroring the points
ledger and A/B ledger separation -- a dry run must never write into
the live memory. Recording is idempotent per (username, race_id): a
forced re-review replaces the entry. Reading is fail-safe: a missing
or corrupt store yields no block, and prediction proceeds exactly as
before.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

MAX_STORED_PER_AI = 40  # entries kept per account (oldest dropped)
INJECT_COUNT = 5  # reflections shown in a prediction prompt
MAX_ENTRY_CHARS = 400  # stored body cap (reflections run 200-300 chars)

HEADER = (
    "【あなたの反省メモ（直近のレースであなた自身が述べた反省です。"
    "同じ見誤りを繰り返さないよう、今回の予想に活かしてください）】"
)


def memory_path(live: bool) -> Path:
    return (
        Path("runs/reflection-memory.json")
        if live
        else Path("runs/dryrun/reflection-memory.json")
    )


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("accounts"), dict):
            accounts = {
                u: e for u, e in data["accounts"].items() if isinstance(e, list)
            }
            return {"version": 1, "accounts": accounts}
    except Exception:  # noqa: BLE001 - fail-safe: memory is optional
        pass
    return {"version": 1, "accounts": {}}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    tmp.replace(path)


def strip_header(text: str) -> str:
    """Drop the posted note's reflection header line, keeping the body."""
    body = (text or "").strip()
    if body.startswith("【反省】"):
        nl = body.find("\n")
        if nl >= 0:
            body = body[nl + 1:].strip()
    return body


def record(
    username: str,
    race_id: str | None,
    date: str | None,
    label: str,
    text: str,
    *,
    live: bool,
    path: Path | None = None,
) -> bool:
    """Record one AI's reflection for one race. Returns True when the
    store was updated. Idempotent per race_id: re-recording the same
    race replaces the old entry (and moves it to the newest slot)."""
    body = strip_header(text)
    if not username or not race_id or not body:
        return False
    path = path or memory_path(live)
    data = _load(path)
    entries = data["accounts"].setdefault(username, [])
    entries[:] = [e for e in entries if e.get("race_id") != race_id]
    entries.append(
        {
            "race_id": race_id,
            "date": date or "",
            "label": label or "",
            "text": body[:MAX_ENTRY_CHARS],
            "recorded_at": _now(),
        }
    )
    del entries[:-MAX_STORED_PER_AI]
    data["updated_at"] = _now()
    _save(path, data)
    return True


def block_for(
    username: str,
    *,
    live: bool = True,
    path: Path | None = None,
    limit: int = INJECT_COUNT,
) -> str:
    """The prompt block carrying the AI's most recent reflections, or
    "" when it has none. Starts with a blank line so it concatenates
    directly into the prediction user prompt, like the market note."""
    path = path or memory_path(live)
    entries = _load(path)["accounts"].get(username) or []
    if not entries:
        return ""
    lines = [HEADER]
    for e in entries[-limit:]:
        head = " ".join(
            x for x in [e.get("date") or "", e.get("label") or ""] if x
        )
        text = e.get("text") or ""
        lines.append("・" + head + "：" + text if head else "・" + text)
    return "\n\n" + "\n".join(lines)


def blocks_for(
    usernames, *, live: bool, path: Path | None = None
) -> dict[str, str]:
    """{username: block} for the accounts that have any memory -- the
    snapshot run_prediction hands to generation, mirroring the points
    snapshot. Accounts without memory are absent, not empty."""
    out: dict[str, str] = {}
    for u in usernames:
        block = block_for(u, live=live, path=path)
        if block:
            out[u] = block
    return out

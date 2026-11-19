#!/usr/bin/env python3
"""Top-grade model selection and OpenAI pool allocation (extracted from orchestrator.py, 2026-11-19 split; behavior unchanged)."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- Model selection: top-grade flagships + OpenAI pool allocation --------
# Top-grade rule (user decision 2026-11-07): for G1 / JpnI races (incl.
# the J.G1 jump grade), @claude runs its flagship; reviews of top-grade
# races follow the same rule (same fixture). @gemini already runs its
# flagship (gemini-3.1-pro-preview) and @local is capped by take-pc's
# VRAM (qwen3:14b), so neither ever changes.
#
# @chatgpt (user decisions 2026-11-11, effective 2026-11-12): the
# everyday default IS the flagship gpt-6-astra — gpt-5-mini, the old
# default, leaves the API on 2026-12-11 (OpenAI notice 2026-11-11).
# Astra draws on the data-sharing FLAGSHIP pool (250K tokens/day for
# this org; the pool day resets 00:00 UTC = 09:00 JST), while the small
# pool holds 2.5M/day. A request that would cross the flagship cap is
# billed IN FULL at list price, and the org balance is small ($3.00,
# auto-reload off), so astra is ALLOCATED by grade: the daily cron first
# writes a pool plan (runs/pool-plan-<pool date>.json) for the day's
# races; the highest grades get astra until the day's budget (pool minus
# actual usage minus a safety reserve) runs out, and the rest run on
# gpt-5.6-terra, which sits in the small pool. (gpt-6.1-terra is in
# NEITHER pool per OpenAI's published list, so it is never a free
# fallback.) Assignments are refreshed at every race start from the
# actual usage log, so an over-running day demotes the lowest-grade
# unstarted races first; races that already started keep their model.
# Top-grade races always run astra. Without a plan (manual runs,
# dedicated crons), a dynamic guard falls back to terra when the
# remaining flagship budget drops below POOL_GUARD_THRESHOLD.
# Reviews use the models recorded in the prediction state, so a race
# predicted on terra is also reflected on with terra.
TOP_GRADE_MODELS = {
    "chatgpt": "gpt-6-astra",      # OpenAI flagship ($10/$50 per MTok)
    "claude": "claude-fable-5-2",  # Anthropic flagship ($8/$40 per MTok; 5.2 since 2026-11-10, user decision)
    "gemini": "gemini-4-neon",   # Google efficiency tier released 2026-11-18 ($3/$15 per MTok regular pricing). User decision 2026-11-19 after a live eval (tests/neon_eval.py): all 6 calls passed validation with zero format errors, ~2-4x faster and cheaper than Carbon, which intermittently returned empty first responses and needed a 2x-budget retry. Carbon was used 2026-11-18 only; Argon 2026-11-11 to 11-17.
}
_TOP_GRADE_KEYS = {"G1", "JPN1", "JG1"}  # JG1: jump G1 (Nakayama Grand Jump)
_GRADE_TRANSLATE = str.maketrans({"Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3"})
_ROMAN_DIGITS = {"I": "1", "II": "2", "III": "3"}


def _grade_key(grade: str) -> str:
    """Canonical grade key: 'JpnI' / 'JpnⅠ' / 'Jpn1' all become 'JPN1',
    'GⅠ' / 'GI' become 'G1', 'J.G1' becomes 'JG1'. (The roman numeral
    is a digit in G1 but a letter in JpnI, so both forms are unified
    to the digit form.)"""
    g = grade.strip().upper().replace(".", "").translate(_GRADE_TRANSLATE)
    for prefix in ("JPN", "JG", "G"):
        if g.startswith(prefix):
            rest = g[len(prefix):]
            return prefix + _ROMAN_DIGITS.get(rest, rest)
    return g


def is_top_grade(fixture: dict) -> bool:
    """True for G1 / JpnI fixtures (incl. the J.G1 jump grade), whatever
    spelling the fixture carries ('G1', 'GⅠ', 'JpnI', 'JpnⅠ', 'J.G1')."""
    grade = str((fixture.get("race") or {}).get("grade") or "")
    return _grade_key(grade) in _TOP_GRADE_KEYS


POOL_FLAGSHIP_DAILY = 250_000
POOL_SAFETY_RESERVE = 30_000   # also covers a dedicated-cron G1 the same pool day
POOL_GUARD_THRESHOLD = 75_000  # no-plan fallback line for the flagship pool
POOL_FALLBACK_MODEL = "gpt-5.6-terra"  # small pool (2.5M tokens/day)
POOL_ASTRA_MODEL = TOP_GRADE_MODELS["chatgpt"]
POOL_SMALL_MODELS = {
    "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.4-mini", "gpt-5.4-nano",
    "gpt-5-mini", "gpt-5-nano", "gpt-4o-mini", "gpt-4.1-mini",
    "gpt-4.1-nano",
}
_JST = timezone(timedelta(hours=9))
USAGE_LOG = Path("runs/usage.log")
RUNS_DIR = Path("runs")


def _pool_day_key_for(moment: datetime) -> str:
    """Pool days run 09:00 JST to 09:00 JST (the complimentary-token
    counters reset at 00:00 UTC); the key is the JST calendar date on
    which the pool day STARTS."""
    local = moment.astimezone(_JST)
    return (local - timedelta(hours=9)).date().isoformat()


def _fixture_pool_key(fixture: dict) -> str:
    """Pool-day key of a race, from its own date + start time (JST)."""
    race = fixture.get("race") or {}
    try:
        hh, mm = (int(x) for x in
                  str(race.get("start_time") or "12:00").split(":")[:2])
        moment = datetime.fromisoformat(
            str(race.get("date"))).replace(hour=hh, minute=mm, tzinfo=_JST)
        return _pool_day_key_for(moment)
    except (ValueError, TypeError):
        return _pool_day_key_for(datetime.now(_JST))


def _pool_window(key: str) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(key).replace(hour=9, tzinfo=_JST)
    return start, start + timedelta(days=1)


def flagship_tokens_used(key: str, usage_path: Path = USAGE_LOG) -> int:
    """OpenAI flagship-pool tokens consumed in pool day `key`, summed
    from the usage log (small-pool models excluded). The log is written
    by the connector helpers per call; timestamps arrive in mixed
    offsets and are normalised to JST here."""
    path = Path(usage_path)
    if not path.exists():
        return 0
    start, end = _pool_window(key)
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("provider") != "openai":
            continue
        if entry.get("model") in POOL_SMALL_MODELS:
            continue
        try:
            ts = datetime.fromisoformat(str(entry.get("ts")))
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if start <= ts.astimezone(_JST) < end:
            total += int(entry.get("input_tokens") or 0)
            total += int(entry.get("output_tokens") or 0)
    return total


def grade_rank(fixture: dict) -> int:
    """Coarse grade ordering for pool allocation (higher = earlier in
    line for astra). Graded races top the list; NAR class races
    (A > B > C) sit below open / listed races."""
    grade = str((fixture.get("race") or {}).get("grade") or "")
    key = _grade_key(grade)
    if key in _TOP_GRADE_KEYS:
        return 100
    if key in ("G2", "JPN2"):
        return 90
    if key in ("G3", "JPN3"):
        return 80
    if "重賞" in grade:
        return 70
    if key.startswith("L") or "リステッド" in grade:
        return 60
    if "OP" in grade.upper() or "オープン" in grade:
        return 50
    m = re.match(r"\s*([ABC])", grade)
    if m:
        return {"A": 40, "B": 30, "C": 20}[m.group(1)]
    return 10


def estimate_race_tokens(fixture: dict) -> int:
    """Estimated flagship tokens for one race's @chatgpt prediction
    stages. Measured use is ~16-21K per race; the estimate stays
    conservative because crossing the cap bills a whole request."""
    race = fixture.get("race") or {}
    field = race.get("field_size") or len(fixture.get("horses") or [])
    return 15_000 + 600 * int(field or 0)


def _plan_path(key: str, runs_dir: Path) -> Path:
    return Path(runs_dir) / f"pool-plan-{key}.json"


def _save_plan(plan: dict, runs_dir: Path) -> Path:
    path = _plan_path(plan["pool_date"], runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(path)
    return path


def refresh_pool_plan(plan: dict, runs_dir: Path = RUNS_DIR,
                      usage_path: Path = USAGE_LOG) -> dict:
    """Recompute astra/terra assignments for the plan's UNSTARTED races
    from the actual flagship usage so far. Started races (a state file
    exists) keep the model they began with — a finished race is frozen
    at the model its state recorded."""
    used = flagship_tokens_used(plan["pool_date"], usage_path)
    budget = POOL_FLAGSHIP_DAILY - used - POOL_SAFETY_RESERVE
    plan["flagship_used"] = used
    plan["budget_remaining"] = budget
    pending = []
    for entry in plan["races"]:
        state = Path(runs_dir) / f"{entry['stem']}-state.json"
        if state.exists():
            entry["started"] = True
            try:
                recorded = json.loads(
                    state.read_text(encoding="utf-8")).get("models") or {}
            except (ValueError, OSError):
                recorded = {}
            if recorded.get("chatgpt") == POOL_FALLBACK_MODEL:
                entry["assigned"] = "terra"
            elif recorded.get("chatgpt"):
                entry["assigned"] = "astra"
        else:
            entry["started"] = False
            pending.append(entry)
    pending.sort(key=lambda e: (-e["rank"], e["start_time"], e["stem"]))
    for entry in pending:
        if budget >= entry["estimate"]:
            entry["assigned"] = "astra"
            budget -= entry["estimate"]
        else:
            entry["assigned"] = "terra"
    plan["budget_after_estimates"] = budget
    return plan


def create_pool_plan(items, runs_dir: Path = RUNS_DIR,
                     usage_path: Path = USAGE_LOG) -> list[dict]:
    """Build (and save) the pool plan(s) for a day's races. `items` is a
    list of (fixture stem, fixture) pairs. Returns the saved plans."""
    by_key: dict[str, list] = {}
    for stem, fixture in items:
        by_key.setdefault(_fixture_pool_key(fixture), []).append(
            (stem, fixture))
    plans = []
    for key, group in sorted(by_key.items()):
        races = []
        for stem, fixture in group:
            race = fixture.get("race") or {}
            races.append({
                "stem": stem,
                "race_id": str(race.get("race_id") or ""),
                "grade": str(race.get("grade") or ""),
                "rank": grade_rank(fixture),
                "start_time": str(race.get("start_time") or ""),
                "estimate": estimate_race_tokens(fixture),
                "assigned": "terra",
                "started": False,
            })
        plan = {"pool_date": key, "races": races}
        refresh_pool_plan(plan, runs_dir=runs_dir, usage_path=usage_path)
        _save_plan(plan, runs_dir)
        plans.append(plan)
    return plans


def _pool_decision(fixture: dict, *, runs_dir: Path,
                   usage_path: Path) -> tuple[str, dict]:
    """@chatgpt's model for a non-top-grade live race: the day plan if
    this race is in one, else the dynamic remaining-budget guard."""
    key = _fixture_pool_key(fixture)
    race_id = str((fixture.get("race") or {}).get("race_id") or "")
    path = _plan_path(key, runs_dir)
    if path.exists():
        try:
            plan = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            plan = None
        if plan is not None and any(
                e.get("race_id") == race_id for e in plan.get("races", [])):
            refresh_pool_plan(plan, runs_dir=runs_dir,
                              usage_path=usage_path)
            _save_plan(plan, runs_dir)
            entry = next(e for e in plan["races"]
                         if e.get("race_id") == race_id)
            model = (POOL_ASTRA_MODEL if entry["assigned"] == "astra"
                     else POOL_FALLBACK_MODEL)
            return model, {"source": "plan",
                           "remaining": plan.get("budget_remaining")}
    remaining = POOL_FLAGSHIP_DAILY - flagship_tokens_used(key, usage_path)
    if remaining < POOL_GUARD_THRESHOLD:
        return POOL_FALLBACK_MODEL, {"source": "guard",
                                     "remaining": remaining}
    return POOL_ASTRA_MODEL, {"source": "default", "remaining": remaining}


def apply_top_grade_models(fixture: dict, built, *, live: bool = False,
                           stem: str | None = None,
                           recorded: dict | None = None,
                           runs_dir: Path = RUNS_DIR,
                           usage_path: Path = USAGE_LOG,
                           info_out: dict | None = None) -> dict:
    """Resolve every persona's effective model for this run.

    Top-grade races swap @chatgpt/@claude/@gemini to their flagships. For other
    live races, @chatgpt's model comes from the pool allocation (see
    the block comment above). `recorded` — the prediction state's saved
    models, passed on review runs — wins over every rule, so reflections
    use the models that made the predictions. Returns {username: model};
    when `info_out` is given it also receives the @chatgpt pool decision
    as {"source": ..., "remaining": ...}."""
    effective = {}
    info = {"source": "default", "remaining": None}
    top = is_top_grade(fixture)
    for persona, prov in built:
        username = persona["username"]
        model = getattr(prov, "model", None) or getattr(
            prov, "default_model", None)
        if recorded and recorded.get(username):
            prov.model = recorded[username]
            model = prov.model
            if username == "chatgpt":
                info = {"source": "recorded", "remaining": None}
        elif (top and username in TOP_GRADE_MODELS
                and getattr(prov, "default_model", None)):
            prov.model = TOP_GRADE_MODELS[username]
            model = prov.model
            if username == "chatgpt":
                info = {"source": "top-grade", "remaining": None}
        elif (username == "chatgpt" and live
                and getattr(prov, "default_model", None)):
            model, info = _pool_decision(
                fixture, runs_dir=runs_dir, usage_path=usage_path)
            prov.model = model
        effective[username] = model
    if info_out is not None:
        info_out.update(info)
    return effective

#!/usr/bin/env python3
"""Debate orchestrator for the AI議論サーバー (Misskey) racing predictions.

Flow (one race):
  1. @shikai (MC) posts the race thread opening.
  2. @chatgpt / @claude / @gemini each post a comprehensive prediction
     (marks + reasoning) as a reply.
  3. One debate round: each AI reads the other two predictions and posts
     a response (agreement / objection / mark revisions).
  4. @shikai consolidates a single final conclusion from all marks.
  5. Later, with --review and a results file, @shikai posts a review
     comparing the conclusion and each AI's marks with the actual result.

Safety:
  * Default is --dry-run: everything is printed, nothing is posted, and
    no credentials are fetched.
  * Posting requires the explicit --live flag.
  * The prediction flow NEVER reads a results file. Results are only
    loaded inside the --review branch.
  * Credentials are fetched in memory only (see misskey_client.fetch_tokens)
    and only in --live mode.

Usage:
  python3 orchestrator.py --dry-run --mock
  python3 orchestrator.py --live                      # needs all providers ready
  python3 orchestrator.py --review --results fixtures/2025-elizabeth-cup-results.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import providers  # noqa: E402
import points as points_mod  # noqa: E402
from misskey_client import MisskeyClient, fetch_tokens  # noqa: E402

MC_USERNAME = "shikai"
PERSONAS = [
    {"username": "chatgpt", "display": "ChatGPT", "provider": providers.OpenAIProvider},
    {"username": "claude", "display": "Claude", "provider": providers.AnthropicProvider},
    {"username": "gemini", "display": "Gemini", "provider": providers.GeminiProvider},
]
LOCAL_PERSONA = {
    "username": "local",
    "display": "ローカルLLM（Qwen3）",
    "provider": providers.OllamaProvider,
    "structured": True,
}
MARKS = ["◎", "○", "▲", "△"]
MARK_SCORE = {"◎": 4, "○": 3, "▲": 2, "△": 1}

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

    Top-grade races swap @chatgpt/@claude to their flagships. For other
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


# --- Market anchor, conclusion gate, A/B measurement (2026-11-05) --------
# Adopted from the prior-art survey (research report §4, items 1/2/4).
# The closing market is a strong, well-calibrated forecast, so:
# (1) fixtures carry a de-vigged market win probability per horse, the
#     prompts ask each AI to justify its win% AS A DEVIATION from the
#     market, and the market itself votes as one extra ballot;
# (2) the conclusion account bets only when the panel's aggregate win
#     probability for its anchor clears the market by EDGE_MARGIN, and
#     a horse the market prices below LONGSHOT_MIN_MARKET_PROB can
#     never anchor the conclusion bet (the longshot cap);
# (4) pre-debate vs post-debate aggregates are scored permanently
#     (A/B ledger) so the debate's value is measured, not assumed.
EDGE_MARGIN = 1.15               # aggregate win prob must reach market x this
LONGSHOT_MIN_MARKET_PROB = 0.02  # anchor needs at least this market prob
MARKET_USERNAME = "market"       # pseudo-voter carrying the market ballot
AB_JUDGE_AT_RACES = 200          # the debate default is judged at this count
AB_STANDINGS_MIN_RACES = 10      # review notes show A/B standings from here
MARK_RE = re.compile(r"([◎○▲△])\s*(?:馬番)?\s*(\d{1,2})\s*番?\s+([ァ-ヶー・]+)")

# --- Hashtags on the moderator's posts (2026-11-05) ----------------------
# Every @shikai note (thread opener, conclusion, review) ends with one
# hashtag line so any post of a race's thread identifies the race:
#   #AI競馬予想 #門別競馬場 #2026道営記念
# The line is derived deterministically from the fixture alone, so the
# three notes of one race always carry the identical line. Texts of
# already-posted notes live in the run state and are never regenerated,
# so adding tags here cannot disturb reconciliation of older runs; only
# newly generated @shikai texts gain the line.
PROJECT_TAG = "#AI競馬予想"

# Bracketed grade/class badges the netkeiba scrapes glue onto race
# names (道営記念〔H1〕重賞, カムイユカラスプリント(A2)). A bracketed
# segment is dropped whole only when its entire content is one of these
# annotations; any other parenthetical (天皇賞(秋)) is part of the race
# name and its content survives (the bracket characters themselves are
# removed by the character filter below).
_TAG_ANNOTATION = re.compile(
    r"(?:G[123]|Jpn[ⅠⅡⅢI123]|H[123]|[A-C][123]|OP|L|重賞|条件|新馬|未勝利)"
)
_TAG_BRACKETED = re.compile(
    r"〔([^〕]*)〕|（([^）]*)）|\(([^)]*)\)|［([^］]*)］|\[([^\]]*)\]"
)
# Characters allowed in a Misskey hashtag fragment: kana/kanji (with
# iteration marks and the long-vowel mark), ASCII letters/digits and
# underscore. Everything else — whitespace, brackets, symbols and the
# katakana middle dot ・ — is stripped: MFM hashtag parsing stops at
# characters like ・, so keeping one would silently truncate the tag.
_TAG_INVALID = re.compile(r"[^0-9A-Za-z_ぁ-んァ-ヶー一-龯々〆]")


def _tag_fragment(raw: str) -> str:
    """Sanitise a race name into a hashtag-safe fragment.

    Rules: drop bracketed grade/class annotations (brackets AND
    content); keep the content of other bracketed segments; an opening
    bracket that never closes is scrape glue — drop it and everything
    after it; remove all whitespace; drop a trailing 重賞 descriptor the
    NAR scrape glues on (道営記念〔H1〕重賞 -> 道営記念); finally strip
    every character invalid in a Misskey hashtag (see _TAG_INVALID).
    """
    def _sub(m: re.Match) -> str:
        content = next(g for g in m.groups() if g is not None)
        if _TAG_ANNOTATION.fullmatch(content.strip()):
            return ""
        return content

    s = _TAG_BRACKETED.sub(_sub, raw or "")
    unmatched = re.search(r"[〔（(［\[]", s)
    if unmatched:
        s = s[: unmatched.start()]
    s = re.sub(r"\s+", "", s)
    if s.endswith("重賞") and len(s) > 2:
        s = s[:-2]
    return _TAG_INVALID.sub("", s)


def race_hashtag(fixture: dict) -> str | None:
    """'#2026道営記念' from the fixture's race name + date year."""
    r = fixture.get("race") or {}
    name = _tag_fragment(str(r.get("name") or ""))
    if not name:
        return None
    year = re.search(r"\d{4}", str(r.get("date") or ""))
    return f"#{year.group(0)}{name}" if year else f"#{name}"


def venue_hashtag(fixture: dict) -> str | None:
    """'#門別競馬場' from the fixture's venue ('門別競馬場' -> same tag)."""
    r = fixture.get("race") or {}
    venue = re.sub(r"\s+", "", str(r.get("venue") or ""))
    if not venue:
        return None
    base = venue[:-3] if venue.endswith("競馬場") else venue
    base = _TAG_INVALID.sub("", base)
    return f"#{base}競馬場" if base else None


def hashtag_line(fixture: dict) -> str:
    """The final line of every @shikai note: project, venue, race tags."""
    tags = [PROJECT_TAG]
    for tag in (venue_hashtag(fixture), race_hashtag(fixture)):
        if tag:
            tags.append(tag)
    return " ".join(tags)

SYSTEM_TEMPLATE = """あなたは {display} です。競馬予想AIとして、他のAIと議論して最終的に一つの結論に収束させる場に参加しています。
予想は特定の観点に偏らず、過去の走破内容・騎手・枠順・馬場状態・脚質と想定ペース・近走の勢いを総合的に判断してください。
出力は日本語で、必ず次の形式の印を含めてください（各行：印、馬番、馬名、勝率、3着内率、その後に簡潔な理由）:
◎ 馬番 馬名 勝率○% 3着内○% — 理由
○ 馬番 馬名 勝率○% 3着内○% — 理由
▲ 馬番 馬名 勝率○% 3着内○% — 理由
△ 馬番 馬名 勝率○% 3着内○% — 理由
勝率と3着内率はあなたの推定です。勝率は1〜90%、3着内率は勝率以上99%以下にし、4頭の勝率の合計は100%を超えないでください。人気だけに引きずられず、「勝つ確率が高いのに評価が割れそうな馬」を見抜くことが、確率推定の精度を上げる近道です。
出走馬データに「市場勝率○%」と併記されている数値は、単勝オッズから算出した市場の勝率評価です（併記がないレースでは市場データなし）。あなたの勝率は市場との差（乖離）として捉え、各印の理由では「市場が何を見落としているのか」を、その差の大きさ（ポイント）とともに具体的に述べてください。市場との乖離が大きいほど強い根拠が必要です。根拠が弱い場合は、無理に乖離を作らず市場寄りの評価にしてください。
出走馬データに「騎手（中央・今季）勝率○% 連対○% 複勝○%」のように併記されている数値は、その騎手・調教師の当該主催（中央/地方）における今季成績です（「通算」とある場合は今季の記録がないための通算成績）。併記がない人物は成績データなしです。騎手の評価は名前の印象ではなく、この数字を根拠にしてください。
印のあとに、あなた自身が馬券を買うならどう買うかも、必ず次の形式で書いてください（券種・組み合わせ・金額を1行に1点。金額は点数（100点単位）で、上限は別途指示されます）:
【買い目】
単勝 7 500
3連複 1-7-11 500
全体で800字程度に収めてください。"""


def person_stats_text(horse: dict) -> str:
    """Compact jockey/trainer season record for one runner line, e.g.
    '騎手（地方・今季）勝率12.8% 連対21.9% 複勝34.0% / 調教師（地方・今季）…'.
    The stats come from the fixture's jockey_stats / trainer_stats
    (db.netkeiba season tables for the race's own jurisdiction;
    career=True renders 通算 instead of 今季）. A person without
    stats contributes nothing, and a runner with neither renders the
    empty string, leaving the line exactly as it was before the
    fixture carried stats."""
    parts = []
    for key, label in (("jockey_stats", "騎手"), ("trainer_stats", "調教師")):
        st = horse.get(key)
        if not st:
            continue
        span = "通算" if st.get("career") else "今季"
        parts.append(
            f"{label}（{st['scope']}・{span}）"
            f"勝率{st['win_rate']:.1f}% 連対{st['top2_rate']:.1f}% "
            f"複勝{st['top3_rate']:.1f}%"
        )
    return " / ".join(parts)


def race_text(fixture: dict, form_limit: int | None = None) -> str:
    r = fixture["race"]
    # Bracket and weather have always been in the fixture but were not
    # rendered into the prompt until 2026-11-10 (the prompts ask the
    # AIs to weigh 枠順, so the data has to actually reach them). The
    # .get guards keep older fixtures rendering as before.
    header = (
        f"{r['date']} {r['venue']} {r['race_number']}R / {r['course']} "
        f"{r['distance_m']}m / 馬場：{r['going']}"
    )
    weather = str(r.get("weather") or "").strip()
    if weather:
        header += f" / 天候：{weather}"
    header += f" / {r['conditions']} / {r['field_size']}頭立て"
    lines = [
        f"第{r['edition']}回 {r['name']}（{r['grade']}）",
        header,
        "",
        "【出走馬】",
    ]
    for h in fixture["horses"]:
        recent = h["recent_form"][:form_limit] if form_limit else h["recent_form"]
        form = " / ".join(recent)
        bracket = h.get("bracket")
        bracket_part = f"{bracket}枠・" if bracket else ""
        line = (
            f"{h['number']} {h['name']}（{bracket_part}{h['sex_age']}・{h['weight_kg']}kg・{h['jockey']}・{h['trainer']}厩舎）近走：{form}"
        )
        mkt = h.get("market") or {}
        if mkt.get("win_prob") is not None:
            line += f" / 市場勝率{mkt['win_prob'] * 100:.1f}%（単勝{mkt.get('odds')}倍）"
        stats = person_stats_text(h)
        if stats:
            line += f" / {stats}"
        lines.append(line)
    return "\n".join(lines)


def parse_marks(text: str) -> dict[str, tuple[int, str]]:
    marks: dict[str, tuple[int, str]] = {}
    for mark, num, name in MARK_RE.findall(text or ""):
        marks.setdefault(mark, (int(num), name))
    return marks


BET_SECTION_RE = re.compile(r"【?買い目[^\n】]*】?\s*\n?(.*)", re.S)

# --- Structured prediction path (local LLM) -------------------------------
# Small local models collapse when asked for formatted prose. Instead the
# local predictor returns JSON under a schema constraint (Ollama `format`),
# the orchestrator validates it against the fixture (with one error-feedback
# retry), and renders the Misskey post itself. Horse names are rendered
# from the fixture by number, so name/number mix-ups cannot occur.

LOCAL_SYSTEM = """あなたは ローカルLLM（Qwen3） です。競馬予想AIとして、他のAIと議論して最終的に一つの結論に収束させる場に参加しています。
予想は特定の観点に偏らず、過去の走破内容・騎手・枠順・馬場状態・脚質と想定ペース・近走の勢いを総合的に判断してください。
出力は指定されたJSON形式のみです。出走データに「市場勝率○%」とある数値は単勝オッズ由来の市場の勝率評価です。各印の理由では、自分の勝率と市場勝率との差（乖離、ポイント）と、その根拠（市場が見落としている情報）を短く述べてください。印は◎○▲△の4つを、すべて異なる馬番で選んでください。各印には勝率（win_pct、1〜90の整数）と3着内率（top3_pct、勝率以上99以下の整数）の推定を付けてください。4つの勝率の合計は100を超えないでください。買い目の金額は点数で、合計は指示された上限以内にしてください。出走データに「騎手（中央・今季）勝率○% 連対○% 複勝○%」とある数値は、その騎手・調教師の当該主催（中央/地方）での今季成績です（「通算」は今季の記録がない場合の通算成績）。併記がない人物は成績データなしです。騎手の評価は名前の印象ではなく、この数字を根拠にしてください。理由の中でレース名や馬名に触れるときは、入力データの表記（漢字・かなの別を含む）をそのままコピーしてください。
出力例: {"marks": [{"mark": "◎", "number": 7, "win_pct": 30, "top3_pct": 60, "reason": "近走の内容が安定"}, {"mark": "○", "number": 1, "win_pct": 22, "top3_pct": 50, "reason": "コース適性が高い"}, {"mark": "▲", "number": 11, "win_pct": 15, "top3_pct": 40, "reason": "脚質が展開に合う"}, {"mark": "△", "number": 4, "win_pct": 10, "top3_pct": 30, "reason": "人気薄だが一発がある"}], "bets": [{"type": "単勝", "combo": "7", "amount": 500}, {"type": "3連複", "combo": "1-7-11", "amount": 500}], "comment": "全体のまとめを1〜2文で。"}"""

PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "marks": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "mark": {"type": "string", "enum": MARKS},
                    "number": {"type": "integer"},
                    "win_pct": {"type": "integer", "minimum": 1, "maximum": 90},
                    "top3_pct": {"type": "integer", "minimum": 1, "maximum": 99},
                    "reason": {"type": "string"},
                },
                "required": ["mark", "number", "win_pct", "top3_pct", "reason"],
            },
        },
        "bets": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "combo": {"type": "string"},
                    "amount": {"type": "integer"},
                },
                "required": ["type", "combo", "amount"],
            },
        },
        "comment": {"type": "string"},
    },
    "required": ["marks", "bets", "comment"],
}


# --- Shared prediction validation (all providers) --------------------------
# Every prediction and final ballot, whatever the provider, must pass the
# same validation before anything is posted: the four marks must name four
# different real runners, and the bets must be well-formed per ticket type
# and total exactly 1,000 yen. (External review 2026-11-04: previously only
# the local LLM was validated, and only loosely — a non-existent #99 could
# become the conclusion's honmei.)

# --- Probability estimates (expected-value layer) --------------------------
# Every prediction must state, for each marked horse, the AI's own win
# probability and top-3 probability. Probabilities are what make expected
# value computable: the orchestrator derives fair odds (100 / win_pct) at
# prediction time, and after the race multiplies the stated win_pct by
# the actual closing win odds to judge whether the AI priced the horse
# better than the market did (EV > 1 means it found value), independent
# of whether the horse actually won.

PROB_LINE_RE = re.compile(
    r"([◎○▲△])\s*(\d{1,2})\s+\S+\s+勝率\s*(\d{1,3})\s*%\s+3着内\s*(\d{1,3})\s*%"
)


def parse_probabilities(text: str) -> dict[str, tuple[int, int, int]]:
    """{mark: (number, win_pct, top3_pct)} parsed from prediction mark
    lines of the form '◎ 7 馬名 勝率30% 3着内60% — ...'."""
    normalized = (text or "").replace("％", "%").replace("　", " ")
    probs: dict[str, tuple[int, int, int]] = {}
    for mark, num, win, top3 in PROB_LINE_RE.findall(normalized):
        probs.setdefault(mark, (int(num), int(win), int(top3)))
    return probs


def validate_probabilities(
    probs: dict[str, tuple[int, int, int]],
    marks: dict[str, tuple[int, str]],
) -> list[str]:
    """Probabilities must cover all four marks (on the same horses the
    marks name), be in range with top3 >= win, and the four win_pct must
    not sum past 100 — the AI is asserting a probability distribution
    over its own picks, and a >100% total is incoherent."""
    errors: list[str] = []
    total = 0
    for mk in MARKS:
        if mk not in probs:
            errors.append(f"{mk} の確率推定（勝率・3着内率）がありません")
            continue
        num, win, top3 = probs[mk]
        if mk in marks and marks[mk][0] != num:
            errors.append(f"{mk} の確率推定の馬番 {num} が印の馬番と一致しません")
        if not 1 <= win <= 90:
            errors.append(f"{mk} の勝率 {win}% が範囲外です（1〜90%）")
        if not win <= top3 <= 99:
            errors.append(f"{mk} の3着内率 {top3}% は勝率以上99%以下にしてください")
        total += win
    if total > 100:
        errors.append(f"勝率の合計が {total}% で100%を超えています")
    return errors


def fair_odds(win_pct: int) -> float:
    """The odds at which a bet on this horse breaks even at win_pct."""
    return 100.0 / win_pct


# --- Market data (the fixture's odds snapshot) ------------------------------
# build_fixture stamps every horse with a market block {odds, win_prob,
# snapshot_time}: the win odds published at fixture-build time and the
# de-vigged win probability derived from them. Before betting opens
# there are no odds, the block is nulls, and every consumer below must
# degrade to the pre-market behaviour (no prompt note, no annotation,
# no market ballot, no conclusion gate).

def fixture_market(fixture: dict) -> dict[int, dict]:
    """Usable market entries {number: {"odds":…, "win_prob":…}} — only
    horses whose de-vigged probability is present."""
    out: dict[int, dict] = {}
    for h in fixture.get("horses") or []:
        m = h.get("market") or {}
        if m.get("win_prob") is not None:
            out[h["number"]] = {
                "odds": m.get("odds"),
                "win_prob": float(m["win_prob"]),
            }
    return out


def market_win_prob(fixture: dict, number: int) -> float | None:
    entry = fixture_market(fixture).get(number)
    return entry["win_prob"] if entry else None


def market_ballot_marks(fixture: dict) -> dict | None:
    """The market's own ballot: ◎○▲△ on the four runners with the
    highest market win probabilities. None without usable market data
    (or with fewer than four priced runners)."""
    market = fixture_market(fixture)
    if len(market) < 4:
        return None
    names = {h["number"]: h["name"] for h in fixture["horses"]}
    top = sorted(
        market.items(), key=lambda kv: (-kv[1]["win_prob"], kv[0])
    )[:4]
    return {
        mk: (num, names.get(num, ""))
        for mk, (num, _entry) in zip(MARKS, top)
    }


def market_prompt_note(fixture: dict) -> str:
    """The deviation-forecasting instruction appended to prompts when
    (and only when) the race card carries market probabilities."""
    if not fixture_market(fixture):
        return ""
    return (
        "\n\n【市場との乖離について】出走データの「市場勝率」は単勝オッズから"
        "算出した市場の評価です。印を付けた馬については、自分の勝率と市場勝率"
        "との差（乖離、ポイント）を意識し、その差を正当化する具体的な根拠"
        "（市場が見落としている情報）を理由の中で数値とともに述べてください。"
    )


_MARKET_ANNOTATION_RE = re.compile(r"市場\s*\d+(?:\.\d+)?\s*%")


def annotate_market(text: str, fixture: dict) -> str:
    """Show the market win% alongside each marked horse's own estimates
    in a prediction text: '… 3着内60%（市場24%）'. Deterministic and
    idempotent. The mark lines were validated against PROB_LINE_RE, so
    splicing the market figure in right after the top-3 estimate keeps
    every later parser working (they all match the line prefix)."""
    market = fixture_market(fixture)
    if not market:
        return text

    def repl(m: re.Match) -> str:
        seg = m.group(0)
        # The annotation lands right AFTER the matched segment, so
        # idempotency means checking the text that follows the match.
        following = m.string[m.end():m.end() + 16]
        if _MARKET_ANNOTATION_RE.search(seg) or following.startswith("（市場"):
            return seg
        entry = market.get(int(m.group(2)))
        if entry is None:
            return seg
        return f"{seg}（市場{entry['win_prob'] * 100:.0f}%）"

    return PROB_LINE_RE.sub(repl, text)


BET_TYPES: dict[str, tuple[int, bool]] = {
    # ticket type -> (number of horses, ordered?)
    "単勝": (1, False),
    "複勝": (1, False),
    "枠連": (2, False),
    "馬連": (2, False),
    "ワイド": (2, False),
    "馬単": (2, True),
    "3連複": (3, False),
    "3連単": (3, True),
}
BET_LINE_RE = re.compile(
    r"^(単勝|複勝|枠連|馬連|ワイド|馬単|3連複|3連単)\s+(\d+(?:-\d+)*)\s+(\d+)\s*(?:円|点)?$"
)


def validate_marks(
    marks: dict[str, tuple[int, str]], fixture: dict
) -> list[str]:
    """marks: {mark: (number, name)} — must be exactly ◎○▲△ on four
    different runners that exist in the fixture."""
    errors: list[str] = []
    valid = {h["number"] for h in fixture["horses"]}
    for mk in MARKS:
        if mk not in marks:
            errors.append(f"印 {mk} がありません")
    seen: set[int] = set()
    for mk, (num, _name) in marks.items():
        if mk not in MARKS:
            errors.append(f"未知の印 '{mk}' があります")
            continue
        if num not in valid:
            errors.append(f"{mk} の馬番 {num} は出走馬に存在しません")
        elif num in seen:
            errors.append(f"馬番 {num} が複数の印で重複しています")
        seen.add(num)
    return errors


def validate_bets(bets: list[dict], fixture: dict,
                  max_total: int | None = None,
                  expected_total: int | None = None) -> list[str]:
    """bets: [{type, numbers: [int], amount: int}] — per-type shape, real
    runners, no duplicates inside a combo, and positive 100-point units.
    Total rule: exactly `expected_total` when given (the moderator's
    deterministic final bet), otherwise at most `max_total` (an AI's
    points cap), otherwise the legacy fixed total of exactly 1,000.
    枠連 is the exception: its numbers are BRACKET numbers, and a
    same-bracket pair (ゾロ目) is valid when that bracket fields at least
    two horses (re-review 2026-11-04, finding 5)."""
    errors: list[str] = []
    valid = {h["number"] for h in fixture["horses"]}
    brackets: dict[int, int] = {}
    for h in fixture["horses"]:
        brackets[h["bracket"]] = brackets.get(h["bracket"], 0) + 1
    if not bets:
        return ["買い目がありません"]
    total = 0
    for b in bets:
        kind = b.get("type")
        numbers = b.get("numbers") or []
        amount = b.get("amount")
        if kind not in BET_TYPES:
            errors.append(f"未知の券種 '{kind}' です")
            continue
        want, _ordered = BET_TYPES[kind]
        if len(numbers) != want:
            errors.append(f"{kind} は {want} 頭の組み合わせが必要です（{len(numbers)} 頭でした）")
        if kind == "枠連":
            for num in numbers:
                if num not in brackets:
                    errors.append(f"枠連 の枠番 {num} は存在しません")
            if len(numbers) == 2 and numbers[0] == numbers[1]:
                if brackets.get(numbers[0], 0) < 2:
                    errors.append(
                        f"枠連 {numbers[0]}-{numbers[1]}（ゾロ目）は同枠に2頭以上必要です"
                    )
        else:
            if len(set(numbers)) != len(numbers):
                errors.append(f"{kind} {numbers} に同じ馬が重複しています")
            for num in numbers:
                if num not in valid:
                    errors.append(f"{kind} の馬番 {num} は出走馬に存在しません")
        if not isinstance(amount, int) or amount <= 0 or amount % 100 != 0:
            errors.append(f"{kind} の金額 {amount} は正の100単位にしてください")
        else:
            total += amount
    if expected_total is not None:
        if total != expected_total:
            errors.append(
                f"買い目の合計が {total} です。合計{expected_total}にしてください"
            )
    elif max_total is not None:
        if total > max_total:
            errors.append(
                f"買い目の合計が {total} で上限 {max_total} を超えています"
            )
    elif total != 1000:
        errors.append(f"買い目の合計が {total} 円です。必ず合計1,000円にしてください")
    return errors


def parse_bet_lines(section: str) -> tuple[list[dict], list[str]]:
    """Parse a free-text 買い目 section into structured bets. Every
    non-empty line must match the canonical '券種 組番 金額円' shape."""
    bets: list[dict] = []
    errors: list[str] = []
    for raw in (section or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = BET_LINE_RE.match(line)
        if not m:
            errors.append(f"買い目の行を解釈できません: {line}")
            continue
        bets.append(
            {
                "type": m.group(1),
                "numbers": [int(x) for x in m.group(2).split("-")],
                "amount": int(m.group(3)),
            }
        )
    return bets, errors


def prediction_errors(text: str, fixture: dict,
                      max_total: int | None = None) -> list[str]:
    """Full validation of a free-text prediction (marks + probabilities
    + bets). With max_total, bets are points capped by the flat
    per-race cap; without it, the legacy fixed total of 1,000 yen
    applies."""
    marks = parse_marks(text)
    errors = validate_marks(marks, fixture)
    errors += validate_probabilities(parse_probabilities(text), marks)
    section = extract_bets(text, limit=100000)
    if not section:
        errors.append("【買い目】のセクションがありません")
        return errors
    bets, perr = parse_bet_lines(section)
    errors += perr
    errors += validate_bets(bets, fixture, max_total=max_total)
    return errors


def validate_local_prediction(obj: object, fixture: dict,
                              max_total: int | None = None) -> list[str]:
    if not isinstance(obj, dict):
        return ["出力がJSONオブジェクトではありません"]
    marks: dict[str, tuple[int, str]] = {}
    probs: dict[str, tuple[int, int, int]] = {}
    shape_errors: list[str] = []
    for m in obj.get("marks") or []:
        if not isinstance(m, dict) or m.get("mark") not in MARKS:
            shape_errors.append(f"印の形式が不正です: {m}")
            continue
        if m["mark"] in marks:
            shape_errors.append(f"印 {m['mark']} が重複しています")
            continue
        marks[m["mark"]] = (m.get("number"), "")
        win, top3 = m.get("win_pct"), m.get("top3_pct")
        if isinstance(win, int) and isinstance(top3, int):
            probs[m["mark"]] = (m.get("number"), win, top3)
        else:
            shape_errors.append(f"印 {m['mark']} に勝率・3着内率がありません")
    bets: list[dict] = []
    for b in obj.get("bets") or []:
        if not isinstance(b, dict):
            shape_errors.append(f"買い目の形式が不正です: {b}")
            continue
        try:
            numbers = [int(x) for x in str(b.get("combo", "")).split("-")]
        except ValueError:
            shape_errors.append(f"買い目の組番を解釈できません: {b.get('combo')}")
            continue
        bets.append(
            {"type": b.get("type"), "numbers": numbers, "amount": b.get("amount")}
        )
    return (
        shape_errors
        + validate_marks(marks, fixture)
        + validate_probabilities(probs, marks)
        + validate_bets(bets, fixture, max_total=max_total)
    )


def render_local_prediction(obj: dict, fixture: dict) -> str:
    names = {h["number"]: h["name"] for h in fixture["horses"]}
    lines = []
    ordered = sorted(
        obj["marks"], key=lambda m: MARKS.index(m["mark"]) if m["mark"] in MARKS else 9
    )
    for m in ordered:
        lines.append(
            f"{m['mark']} {m['number']} {names.get(m['number'], '')} "
            f"勝率{m['win_pct']}% 3着内{m['top3_pct']}% — {str(m['reason']).strip()[:140]}"
        )
    comment = str(obj.get("comment") or "").strip()
    if comment:
        lines.append(comment[:300])
    lines.append("")
    lines.append("【買い目】")
    # Validation guarantees the bets are well-formed and within the
    # points cap at this point; there is deliberately no silent amount
    # repair (external review 2026-11-04: the old fold-the-difference
    # repair could itself fail and post a wrong total).
    for b in obj["bets"]:
        lines.append(f"{b['type']} {b['combo']} {b['amount']}点")
    return "\n".join(lines)


def _structured_with_retry(prov, system: str, user: str, schema: dict,
                           validate, what: str):
    """chat_json + one error-feedback retry. The result is adopted ONLY if
    it validates with zero errors; otherwise RuntimeError — invalid data
    must never reach rendering, aggregation, or posting."""
    obj = prov.chat_json(system, user, schema)
    errors = validate(obj)
    if not errors:
        return obj
    print(f"[structured retry ({what}): {errors}]", file=sys.stderr)
    retry_user = (
        user
        + "\n\n【重要】前回の出力に次の問題がありました: "
        + " / ".join(errors)
        + "\n問題をすべて直して、JSON形式で再度出力してください。"
    )
    obj = prov.chat_json(system, retry_user, schema)
    errors = validate(obj)
    if errors:
        raise RuntimeError(f"{what}: validation failed after retry: {errors}")
    return obj


def budget_line(cap: int | None) -> str:
    """The staking instruction appended to a prediction prompt. In
    points mode it states ONLY the flat per-race cap: no balance is
    ever shown at generation time (decoupled scoreboard, 2026-11-08 —
    a losing streak must not change how an AI bets or estimates).
    The legacy wording (no points context) keeps the fixed
    1,000-yen rule."""
    if cap is not None:
        return (
            f"\n\n買い目の合計は {cap:,} 点以内にし、100点単位で配分して"
            "ください。自信のある買い目ほど厚く張ってください。"
        )
    return "\n\n買い目の合計は必ず1,000円ちょうどにしてください。"


def generate_structured_prediction(prov, fixture: dict,
                                   cap: int | None = None) -> str:
    body = race_text(fixture, form_limit=3)
    user = (
        f"次のレースの予想をしてください。\n\n{body}"
        + market_prompt_note(fixture)
        + budget_line(cap)
    )
    obj = _structured_with_retry(
        prov, LOCAL_SYSTEM, user, PREDICTION_SCHEMA,
        lambda o: validate_local_prediction(o, fixture, max_total=cap),
        "local prediction",
    )
    return render_local_prediction(obj, fixture)


BALLOT_SCHEMA = {
    "type": "object",
    "properties": {
        "marks": PREDICTION_SCHEMA["properties"]["marks"],
        "comment": {"type": "string"},
    },
    "required": ["marks"],
}


def validate_ballot_obj(obj: object, fixture: dict) -> list[str]:
    if not isinstance(obj, dict):
        return ["出力がJSONオブジェクトではありません"]
    marks: dict[str, tuple[int, str]] = {}
    errors: list[str] = []
    for m in obj.get("marks") or []:
        if not isinstance(m, dict) or m.get("mark") not in MARKS:
            errors.append(f"印の形式が不正です: {m}")
            continue
        if m["mark"] in marks:
            errors.append(f"印 {m['mark']} が重複しています")
            continue
        marks[m["mark"]] = (m.get("number"), "")
    return errors + validate_marks(marks, fixture)


def render_ballot_block(marks: dict, probs: dict | None = None) -> str:
    """The canonical 【最終票】 block for a validated ballot. The published
    debate text, the saved state, and the aggregation must all carry the
    SAME ballot (re-review 2026-11-04, finding 1: the local AI's prose
    ballot and its structured ballot could diverge). When the ballot's
    win percentages are known they are rendered with the marks, like
    the cloud personas' ballot lines."""
    lines = ["【最終票】"]
    for mk in MARKS:
        if mk in marks:
            num, name = marks[mk]
            pct = f" 勝率{probs[num]}%" if probs and num in probs else ""
            lines.append(f"{mk} {num} {name}{pct}")
    return "\n".join(lines)


def splice_ballot(text: str, marks: dict, probs: dict | None = None) -> str:
    """Replace (or append) the ballot section of a debate text with the
    canonical block rendered from the validated ballot."""
    head = text.rfind("【最終票】")
    base = text[:head].rstrip() if head >= 0 else text.rstrip()
    return base + "\n\n" + render_ballot_block(marks, probs)


def generate_structured_ballot(prov, fixture: dict, context: str):
    """The local LLM's final ballot after the debate, as
    ({mark: (num, name)}, {number: win_pct}) with fixture-canonical
    names. The ballot schema carries win_pct per mark; percentages are
    captured when present (older/looser outputs may omit them, and the
    caller falls back to the initial estimates)."""
    names = {h["number"]: h["name"] for h in fixture["horses"]}
    obj = _structured_with_retry(
        prov, LOCAL_SYSTEM, context, BALLOT_SCHEMA,
        lambda o: validate_ballot_obj(o, fixture),
        "local ballot",
    )
    marks = {
        m["mark"]: (m["number"], names.get(m["number"], ""))
        for m in obj["marks"]
    }
    probs: dict[int, int] = {}
    for m in obj["marks"]:
        win = m.get("win_pct")
        if isinstance(win, int):
            probs[m["number"]] = win
    return marks, probs


BALLOT_PROB_RE = re.compile(
    r"([◎○▲△])\s*(\d{1,2})\s+\S+\s+勝率\s*(\d{1,3})\s*%"
)


def parse_ballot_probs(text: str) -> dict[int, int]:
    """{number: win_pct} stated in the 【最終票】 section of a debate
    text. Ballot lines carry win percentages since the market-anchor
    change; texts without them yield {} and callers fall back."""
    idx = (text or "").rfind("【最終票】")
    if idx < 0:
        return {}
    out: dict[int, int] = {}
    for _mk, num, win in BALLOT_PROB_RE.findall(text[idx:]):
        out.setdefault(int(num), int(win))
    return out


def _resolve_ballot_probs(marks: dict, stated: dict[int, int],
                          prediction_text: str) -> dict[int, int]:
    """Win percentages for the final marks, best source first: what the
    ballot itself stated, then the AI's initial estimate for the same
    horse, then its initial estimate at the same mark. A mark with no
    resolvable percentage stays absent — the aggregate builder covers
    it through the residual probability mass."""
    init = parse_probabilities(prediction_text)
    by_number = {num: win for num, win, _t in init.values()}
    by_mark = {mk: win for mk, (_n, win, _t) in init.items()}
    out: dict[int, int] = {}
    for mk, (num, _name) in marks.items():
        if num in stated:
            out[num] = stated[num]
        elif num in by_number:
            out[num] = by_number[num]
        elif mk in by_mark:
            out[num] = by_mark[mk]
    return out


def canonical_marks(text: str, fixture: dict) -> dict[str, tuple[int, str]]:
    """parse_marks + fix name typos: the horse number is authoritative,
    so replace the name with the fixture's name for that number (models —
    notably the local 14B — sometimes pair a number with the wrong name).
    """
    names = {h["number"]: h["name"] for h in fixture["horses"]}
    return {
        mark: (num, names.get(num, name))
        for mark, (num, name) in parse_marks(text).items()
    }


def extract_bets(text: str, limit: int = 220) -> str:
    """Pull the betting section (買い目) out of a prediction, if present."""
    m = BET_SECTION_RE.search(text or "")
    if not m:
        return ""
    section = m.group(1).strip()
    # Drop trailing meta lines the models sometimes append after the bets.
    section = re.split(r"\n\s*(?:以上|まとめ|総合判断)", section)[0].strip()
    return section[:limit]


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


def build_final_bet(ranking: list, fixture: dict,
                    total: int = 1000) -> list[dict] | None:
    """The moderator's consolidated bet, built deterministically from the
    final ranking so it is valid by construction (finding 6). `total` is
    the conclusion account's stake for this race (its points cap applies;
    the two legs split it as evenly as 100-point units allow)."""
    nums = [num for num, _info in ranking]
    bets: list[dict] | None = None
    first = (total // 2 // 100) * 100
    rest = total - first
    if len(nums) >= 3 and first > 0:
        bets = [
            {"type": "単勝", "numbers": [nums[0]], "amount": first},
            {"type": "3連複", "numbers": sorted(nums[:3]), "amount": rest},
        ]
    elif len(nums) == 2 and first > 0:
        bets = [
            {"type": "単勝", "numbers": [nums[0]], "amount": first},
            {"type": "馬連", "numbers": sorted(nums[:2]), "amount": rest},
        ]
    elif len(nums) >= 1 and total > 0:
        bets = [{"type": "単勝", "numbers": [nums[0]], "amount": total}]
    if bets is not None:
        errors = validate_bets(bets, fixture, expected_total=total)
        assert not errors, f"final bet must validate: {errors}"
    return bets


def format_bets(bets: list[dict]) -> str:
    return "\n".join(
        f"{b['type']} {'-'.join(str(n) for n in b['numbers'])} {b['amount']}点"
        for b in bets
    )


class Emitter:
    """Prints in dry-run mode; posts to Misskey in live mode."""

    def __init__(self, live: bool, usernames: list[str] | None = None):
        self.live = live
        self.client = MisskeyClient() if live else None
        roster = usernames or [p["username"] for p in PERSONAS]
        self.tokens = (
            fetch_tokens([MC_USERNAME] + roster)
            if live
            else {}
        )

    def emit(self, username: str, text: str, reply_to: str | None) -> str | None:
        if not self.live:
            target = f" (reply to {reply_to})" if reply_to else ""
            print(f"\n===== @{username}{target} =====")
            print(text)
            return f"dryrun-{username}-{abs(hash(text)) % 10**8}"
        # Reconcile before posting: if an identical note by this author
        # already exists (e.g. a previous run posted it but the response
        # was lost), adopt it instead of posting a duplicate.
        existing = self.client.find_note_by_text(
            self.tokens[username], text, reply_to
        )
        if existing:
            print(
                f"[adopted existing note {existing} for @{username}]",
                file=sys.stderr,
            )
            return existing
        note = self.client.post_note(self.tokens[username], text, reply_id=reply_to)
        time.sleep(1)
        return note.get("id")


def build_providers(mock: bool, roster: list[dict] | None = None) -> list[tuple[dict, providers.BaseProvider]]:
    built = []
    for persona in (roster or PERSONAS):
        if mock:
            built.append((persona, providers.MockProvider(persona["display"])))
        else:
            built.append((persona, persona["provider"]()))
    return built


def preflight(built, live: bool) -> None:
    """Fail early (before any post) if a real provider is not wired yet."""
    problems = []
    for persona, prov in built:
        if isinstance(prov, providers._HelperProvider):
            if prov._helper is None or not (prov.model or prov.default_model):
                problems.append(persona["display"])
    if problems and live:
        raise SystemExit(
            "Cannot run live: providers not configured yet: "
            + ", ".join(problems)
            + ". Register their keys in the Secure Vault first (or use --mock)."
        )


def _ballot_errors(text: str, fixture: dict) -> tuple[dict, list[str]]:
    idx = text.rfind("【最終票】")
    if idx < 0:
        return {}, ["【最終票】のセクションがありません"]
    marks = parse_marks(text[idx:])
    return marks, validate_marks(marks, fixture)


def _score_view(points: dict | None, built, display_of: dict) -> str:
    """One-line scoreboard for posts, as cumulative P&L (累積収支):
    'ChatGPT +29,430点 / ...'. The raw balance is never displayed."""
    if not points:
        return ""
    parts = [
        f"{display_of.get(u, u)} {points_mod.format_score(v['balance'])}点"
        for u, v in points.items()
    ]
    return " / ".join(parts)


class _StageFailure(RuntimeError):
    """A generation-stage failure carrying the stage name, so the caller
    can record WHERE an excluded AI fell out (prediction/debate/ballot)."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def _generate_text_prediction(prov, persona: dict, fixture: dict, body: str,
                              cap: int | None) -> str:
    """One cloud persona's free-text prediction, fully validated, with
    the standard one error-feedback retry. Raises RuntimeError when the
    retry still fails validation."""
    username = persona["username"]
    system = SYSTEM_TEMPLATE.format(display=persona["display"])
    user = f"次のレースの予想をしてください。\n\n{body}"
    user += market_prompt_note(fixture)
    user += budget_line(cap)
    text = prov.chat(system, user)
    errors = prediction_errors(text, fixture, max_total=cap) if text.strip() else ["出力が空でした"]
    if errors:
        print(f"[prediction retry for {username}: {errors}]", file=sys.stderr)
        retry = prov.chat(
            system,
            user
            + "\n\n【重要】前回の出力に次の問題がありました: "
            + " / ".join(errors)
            + "\nすべて直して、印4つ（◎○▲△、すべて異なる実在の馬、各印に勝率と3着内率）と"
            "【買い目】（券種ごとの頭数を守り、金額は100単位・上限以内）"
            "を含む完全な予想を出し直してください。",
        )
        errors = prediction_errors(retry, fixture, max_total=cap) if retry.strip() else ["再生成の出力が空でした"]
        if errors:
            raise RuntimeError(
                f"prediction from {username} failed validation after retry: {errors}"
            )
        text = retry
    return text


def _generate_debate_and_ballot(prov, persona: dict, fixture: dict, body: str,
                                predictions: dict, roster: list):
    """One persona's debate reply plus its validated final ballot, as
    (debate text, {mark: (number, name)}, {number: win_pct}). Raises
    _StageFailure with the failing stage on irrecoverable problems,
    RuntimeError otherwise."""
    username = persona["username"]
    others = "\n\n".join(
        f"--- {p['display']} の予想 ---\n{predictions[p['username']]}"
        for p, _ in roster
        if p["username"] != username and p["username"] in predictions
    )
    # The local model generates slowly and its context is precious:
    # for the structured persona, trim the debate prompt (shorter
    # form lines, other predictions excerpted) and cap the reply
    # budget. Cloud personas keep the full context. (The untrimmed
    # prompt plus a 4096-token context made the local debate call
    # hit its 600s timeout in the 2026-11-05 Monbetsu 12R dry run.)
    prompt_body, prompt_others, chat_budget = body, others, None
    if persona.get("structured"):
        prompt_body = race_text(fixture, form_limit=3)
        prompt_others = "\n\n".join(
            f"--- {p['display']} の予想 ---\n{predictions[p['username']][:500]}"
            for p, _ in roster
            if p["username"] != username and p["username"] in predictions
        )
        chat_budget = 2500
    system = SYSTEM_TEMPLATE.format(display=persona["display"])
    user = (
        "他のAIの予想が出そろいました。元の出走データと各AIの予想を踏まえて、"
        "議論として応答してください。最も同意できる点と最も異論のある点を"
        "具体的に挙げてください。\n"
        "最後に、議論を踏まえたあなたの最終票を必ず次の形式で書いてください"
        "（買い目は初期予想から変えません。最終票はすべて異なる実在の馬にし、"
        "各印には議論後のあなたの勝率推定も併記してください）:\n"
        "【最終票】\n◎ 馬番 馬名 勝率○%\n○ 馬番 馬名 勝率○%\n"
        "▲ 馬番 馬名 勝率○%\n△ 馬番 馬名 勝率○%\n"
        + market_prompt_note(fixture)
        + f"\n\n【元の出走データ】\n{prompt_body}\n\n"
        f"【自分の予想】\n{predictions[username]}\n\n【他のAIの予想】\n{prompt_others}"
    )
    try:
        text = prov.chat(system, user, max_tokens=chat_budget)
    except Exception as exc:  # noqa: BLE001 - re-raised with stage info
        raise _StageFailure("debate", f"debate generation failed: {exc}") from exc
    if not text.strip():
        raise _StageFailure("debate", f"empty debate reply from {username}")
    if persona.get("structured") and hasattr(prov, "chat_json"):
        context = (
            "次のレースについて、他のAIとの議論を踏まえた最終票をJSONで出してください。\n\n"
            f"【元の出走データ】\n{race_text(fixture, form_limit=3)}\n\n"
            f"【自分の予想】\n{predictions[username]}\n\n"
            f"【他のAIの予想（抜粋）】\n{prompt_others}\n\n【自分の議論】\n{text[:800]}"
        )
        try:
            ballot, stated = generate_structured_ballot(prov, fixture, context)
        except Exception as exc:  # noqa: BLE001 - re-raised with stage info
            raise _StageFailure("ballot", f"structured ballot failed: {exc}") from exc
        probs = _resolve_ballot_probs(ballot, stated, predictions[username])
        # Publish the validated structured ballot, not whatever the
        # free-form debate prose happened to claim (finding 1).
        return splice_ballot(text, ballot, probs), ballot, probs
    marks, errors = _ballot_errors(text, fixture)
    if errors:
        print(f"[ballot retry for {username}: {errors}]", file=sys.stderr)
        try:
            retry = prov.chat(
                system,
                "最終票だけを出し直してください。前回の問題: "
                + " / ".join(errors)
                + "\n次の形式のみで出力してください（各印に勝率も併記）:\n"
                "【最終票】\n◎ 馬番 馬名 勝率○%\n○ 馬番 馬名 勝率○%\n"
                "▲ 馬番 馬名 勝率○%\n△ 馬番 馬名 勝率○%\n\n"
                f"【元の出走データ】\n{body}",
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with stage info
            raise _StageFailure("ballot", f"ballot retry failed: {exc}") from exc
        marks, errors = _ballot_errors(retry, fixture)
        if errors:
            raise _StageFailure(
                "ballot",
                f"final ballot from {username} failed validation after retry: {errors}",
            )
        head = text.rfind("【最終票】")
        base = text[:head].rstrip() if head >= 0 else text.rstrip()
        text = base + "\n\n" + retry.strip()
    names = {h["number"]: h["name"] for h in fixture["horses"]}
    canonical = {
        mk: (num, names.get(num, name)) for mk, (num, name) in marks.items()
    }
    probs = _resolve_ballot_probs(
        canonical, parse_ballot_probs(text), predictions[username]
    )
    return text, canonical, probs


# --- Bet/final-mark reconciliation (2026-11-05) ----------------------------
# Bets are fixed at prediction time, but the debate can move an AI's
# marks afterwards — the Funabashi 11R case: @local's final ◎ flipped
# 14 -> 5 in the debate while every one of its bets still rode the
# initial ◎14, a horse its final ballot no longer marked at all. After
# the final ballots are known, each AI's bets are checked against its
# FINAL marks; a contradiction triggers a bets-only regeneration, and
# if that also fails the bets are rebuilt deterministically from the
# final marks (the same builder as the moderator's final bet). The
# race never aborts over a stale bet slip, and settlement always uses
# the reconciled bets.

BETS_REGEN_SCHEMA = {
    "type": "object",
    "properties": {"bets": PREDICTION_SCHEMA["properties"]["bets"]},
    "required": ["bets"],
}


def bet_consistency_errors(bets: list[dict], final_marks: dict,
                           fixture: dict) -> list[str]:
    """Bets contradict the final marks when a leg names a horse outside
    the final four marks, or when no bet covers the final ◎ at all.
    枠連 legs are bracket numbers, so a bracket leg contradicts only
    when its bracket contains no marked horse, and it covers the ◎
    when the ◎'s own bracket is among the legs."""
    errors: list[str] = []
    marked = {num for num, _name in final_marks.values()}
    honmei = (final_marks.get("◎") or (None, ""))[0]
    bracket_of = {h["number"]: h["bracket"] for h in fixture["horses"]}
    honmei_covered = False
    for b in bets:
        numbers = b.get("numbers") or []
        if b.get("type") == "枠連":
            for br in numbers:
                in_bracket = {
                    h["number"] for h in fixture["horses"] if h["bracket"] == br
                }
                if not in_bracket & marked:
                    errors.append(f"枠連 の枠 {br} には最終印の馬が1頭もいません")
            if honmei is not None and bracket_of.get(honmei) in numbers:
                honmei_covered = True
        else:
            for num in numbers:
                if num not in marked:
                    errors.append(
                        f"{b.get('type')} の馬番 {num} は最終印（◎○▲△）に含まれません"
                    )
            if honmei is not None and honmei in numbers:
                honmei_covered = True
    if honmei is not None and not honmei_covered:
        errors.append(f"最終◎ {honmei} を含む買い目が1つもありません")
    return errors


def _final_marks_block(final_marks: dict, probs_by_number: dict) -> str:
    lines = []
    for mk in MARKS:
        if mk in final_marks:
            num, name = final_marks[mk]
            p = probs_by_number.get(num)
            pct = f" 勝率{p[0]}% 3着内{p[1]}%" if p else ""
            lines.append(f"{mk} {num} {name}{pct}")
    return "\n".join(lines)


def _regen_prompt(persona: dict, final_marks: dict, probs_by_number: dict,
                  errors: list[str], cap: int | None) -> str:
    limit = (
        f"合計は {cap:,} 点以内" if cap is not None else "合計は必ず1,000円ちょうど"
    )
    return (
        "討論の結果、あなたの最終票は次のとおり確定しましたが、初期予想の買い目が"
        "最終票と矛盾しています。\n\n【最終票】\n"
        + _final_marks_block(final_marks, probs_by_number)
        + "\n\n【矛盾の内容】\n"
        + "\n".join(f"- {e}" for e in errors)
        + "\n\n買い目だけを出し直してください。最終票の4頭だけを使って組み立て"
        "（枠連は枠番で構いません）、最終◎を含む買い目を必ず1つ以上入れてください。"
        f"金額は100単位で、{limit}にしてください。次の形式のみで出力してください:\n"
        "【買い目】\n単勝 7 500\n3連複 1-7-11 500"
    )


def _regen_bets_cloud(prov, persona: dict, fixture: dict, final_marks: dict,
                      probs_by_number: dict, errors: list[str],
                      cap: int | None) -> list[dict] | None:
    """Bets-only regeneration for a free-text persona: one
    error-feedback retry, then None (the caller falls back)."""
    system = (
        f"あなたは {persona['display']} です。競馬予想AIとして、確定した"
        "最終票に整合する買い目だけを、指定の形式で出し直します。"
    )
    user = _regen_prompt(persona, final_marks, probs_by_number, errors, cap)
    for _attempt in range(2):
        try:
            text = prov.chat(system, user)
        except Exception as exc:  # noqa: BLE001 - regeneration is best-effort
            print(f"[bets regen failed for {persona['username']}: {exc}]",
                  file=sys.stderr)
            return None
        section = extract_bets(text, limit=100000)
        if not section:
            errs = ["【買い目】のセクションがありません"]
            bets: list[dict] = []
        else:
            bets, perr = parse_bet_lines(section)
            errs = (
                perr
                + validate_bets(bets, fixture, max_total=cap)
                + bet_consistency_errors(bets, final_marks, fixture)
            )
        if not errs:
            return bets
        user += (
            "\n\n【重要】前回の出力に次の問題がありました: "
            + " / ".join(errs)
            + "\nすべて直して買い目だけを出し直してください。"
        )
    return None


def _regen_bets_structured(prov, fixture: dict, final_marks: dict,
                           probs_by_number: dict, errors: list[str],
                           cap: int | None) -> list[dict] | None:
    """Bets-only regeneration for the structured (local) persona, via
    the same chat_json + error-feedback path as its predictions."""
    user = _regen_prompt({}, final_marks, probs_by_number, errors, cap) \
        + "\n買い目は指定されたJSON形式で出力してください。"
    holder: dict = {}

    def validate(obj: object) -> list[str]:
        if not isinstance(obj, dict):
            return ["出力がJSONオブジェクトではありません"]
        bets: list[dict] = []
        errs: list[str] = []
        for b in obj.get("bets") or []:
            if not isinstance(b, dict):
                errs.append(f"買い目の形式が不正です: {b}")
                continue
            try:
                numbers = [int(x) for x in str(b.get("combo", "")).split("-")]
            except ValueError:
                errs.append(f"買い目の組番を解釈できません: {b.get('combo')}")
                continue
            bets.append(
                {"type": b.get("type"), "numbers": numbers, "amount": b.get("amount")}
            )
        errs += validate_bets(bets, fixture, max_total=cap)
        errs += bet_consistency_errors(bets, final_marks, fixture)
        if not errs:
            holder["bets"] = bets
        return errs

    try:
        _structured_with_retry(
            prov, LOCAL_SYSTEM, user, BETS_REGEN_SCHEMA, validate,
            "bets regeneration",
        )
    except Exception as exc:  # noqa: BLE001 - regeneration is best-effort
        print(f"[bets regen failed (structured): {exc}]", file=sys.stderr)
        return None
    return holder.get("bets")


def _reconcile_bets(fixture: dict, active: list, predictions: dict,
                    final_marks: dict, points: dict | None,
                    cache: dict, save) -> tuple[dict, dict]:
    """Check every active AI's bets against its final marks and repair
    the contradicting ones (regenerate, else deterministic rebuild).
    Returns (final_bets, revisions): final_bets maps every active
    username to its authoritative structured bets; revisions records,
    per changed AI, the original bets, the reason, and the method —
    the audit trail for the swap. Results are cached per AI so a
    resumed run never re-bills a regeneration."""
    cache.setdefault("final_bets", {})
    cache.setdefault("bet_revisions", {})
    final_bets: dict[str, list[dict]] = {}
    revisions: dict[str, dict] = {}
    for persona, prov in active:
        username = persona["username"]
        if username in cache["final_bets"]:
            final_bets[username] = cache["final_bets"][username]
            if username in cache["bet_revisions"]:
                revisions[username] = cache["bet_revisions"][username]
            continue
        bets, perr = parse_bet_lines(
            extract_bets(predictions[username], limit=100000)
        )
        if perr:  # cannot happen: predictions were validated on entry
            raise RuntimeError(f"saved bets for {username} do not parse: {perr}")
        errors = bet_consistency_errors(bets, final_marks[username], fixture)
        if not errors:
            final_bets[username] = bets
            cache["final_bets"][username] = bets
            save()
            continue
        cap = (points or {}).get(username, {}).get("cap")
        probs = parse_probabilities(predictions[username])
        probs_by_number = {
            num: (win, top3) for _mk, (num, win, top3) in probs.items()
        }
        if persona.get("structured") and hasattr(prov, "chat_json"):
            new_bets = _regen_bets_structured(
                prov, fixture, final_marks[username], probs_by_number, errors, cap
            )
        else:
            new_bets = _regen_bets_cloud(
                prov, persona, fixture, final_marks[username],
                probs_by_number, errors, cap,
            )
        method = "regenerated"
        if new_bets is None:
            marks = final_marks[username]
            ranking = [
                (marks[mk][0], {"name": marks[mk][1]})
                for mk in MARKS if mk in marks
            ]
            total = sum(b["amount"] for b in bets) or (cap or 1000)
            new_bets = build_final_bet(ranking, fixture, total=total)
            method = "deterministic-fallback"
        revisions[username] = {
            "original_bets": bets,
            "final_bets": new_bets,
            "reason": " / ".join(errors),
            "method": method,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        final_bets[username] = new_bets
        cache["final_bets"][username] = new_bets
        cache["bet_revisions"][username] = revisions[username]
        save()
    return final_bets, revisions


# --- Aggregate probability vectors + the conclusion gate -------------------
# Items 2 and 4 share one construction: turn per-AI stated win
# percentages (which cover only the marked horses) into a full-field
# probability vector. The Item 2 gate judges the conclusion bet from
# the post-debate vector; the Item 4 A/B scoring builds the same
# vector from the initial predictions and from the final ballots and
# compares the two.

def build_probability_vector(contribs: list[dict], fixture: dict) -> dict[int, float]:
    """Average full-field win-probability vector (percent) from per-AI
    stated win percentages {horse number: win%}.

    Each AI states percentages only for its marked horses; its residual
    mass (100 − stated total) is spread over the horses it did NOT
    state, proportionally to the market win probability when the
    fixture has market data, uniformly otherwise. Each AI's vector is
    normalised to sum to 100, so the average does too."""
    numbers = [h["number"] for h in fixture["horses"]]
    market = fixture_market(fixture)
    vectors: list[dict[int, float]] = []
    for contrib in contribs:
        stated = {
            int(n): float(p) for n, p in (contrib or {}).items()
            if int(n) in numbers
        }
        residual = max(0.0, 100.0 - sum(stated.values()))
        rest = [n for n in numbers if n not in stated]
        vec = dict(stated)
        if rest and residual > 0:
            weights = {
                n: (market.get(n) or {}).get("win_prob") or 0.0 for n in rest
            }
            wsum = sum(weights.values())
            if wsum > 0:
                for n in rest:
                    vec[n] = residual * weights[n] / wsum
            else:
                for n in rest:
                    vec[n] = residual / len(rest)
        total = sum(vec.values())
        if total > 0:
            vec = {n: v * 100.0 / total for n, v in vec.items()}
        vectors.append(vec)
    if not vectors:
        return {n: 0.0 for n in numbers}
    return {
        n: sum(v.get(n, 0.0) for v in vectors) / len(vectors)
        for n in numbers
    }


def evaluate_conclusion_gate(fixture: dict, contribs: list[dict]) -> dict:
    """Decide whether the conclusion account bets, and on which anchor.

    Returns a JSON-friendly record:
      "bet"     — the anchor's aggregate win probability reaches
                  market × EDGE_MARGIN; bet, anchored on `anchor`.
      "pass"    — 見送り: no clear edge over the market; stake 0.
                  A fixture with no market data at all also lands
                  here (flagged via market_missing): beating the
                  market is the bet's premise, and without a market
                  snapshot that premise cannot be verified, so the
                  conclusion account stands down (user decision
                  2026-11-10, applied uniformly — including fixtures
                  built before betting opens, e.g. a Thursday-built
                  G1). "No data" stays distinguishable from "no edge"
                  via the market_missing flag.
    The anchor is the aggregate's top horse among those whose market
    probability is at least LONGSHOT_MIN_MARKET_PROB: a horse the
    market prices under that floor may top the panel's ranking but
    may not anchor the conclusion bet (its number is kept in
    `skipped_longshot` when it was the aggregate's top)."""
    names = {h["number"]: h["name"] for h in fixture["horses"]}
    market = fixture_market(fixture)
    vector = build_probability_vector(contribs, fixture)
    record: dict = {
        "edge_margin": EDGE_MARGIN,
        "longshot_min_market_prob": LONGSHOT_MIN_MARKET_PROB,
        "market_missing": not market,
        "aggregate_probs": {
            str(n): round(p, 2) for n, p in sorted(vector.items())
        },
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if not market:
        record.update({
            "decision": "pass",
            "anchor": None,
            "reason": "市場データがないため、市場を上回ることを検証できず見送り",
        })
        return record
    order = sorted(vector.items(), key=lambda kv: (-kv[1], kv[0]))
    anchor = None
    for num, _p in order:
        mp = (market.get(num) or {}).get("win_prob")
        if mp is not None and mp >= LONGSHOT_MIN_MARKET_PROB:
            anchor = num
            break
    if anchor is None:
        top = order[0][0] if order else None
        record.update({
            "decision": "pass",
            "anchor": None,
            "reason": (
                f"集計首位の {top} {names.get(top, '')} を含め、市場"
                f"{LONGSHOT_MIN_MARKET_PROB * 100:.0f}%以上の馬がおらず"
                "アンカーにできる馬がいない"
            ),
        })
        return record
    agg = vector[anchor]
    mp_pct = market[anchor]["win_prob"] * 100.0
    required = mp_pct * EDGE_MARGIN
    record.update({
        "anchor": anchor,
        "aggregate_prob": round(agg, 2),
        "market_prob": round(mp_pct, 2),
        "required_prob": round(required, 2),
        "edge": round(agg / mp_pct, 3) if mp_pct > 0 else None,
        "skipped_longshot": (
            order[0][0] if order and order[0][0] != anchor else None
        ),
    })
    head = f"{anchor} {names.get(anchor, '')}: 集計勝率 {agg:.1f}% vs 市場 {mp_pct:.1f}%"
    if agg >= required:
        record["decision"] = "bet"
        record["reason"] = (
            head + f" × {EDGE_MARGIN}（={required:.1f}%）を上回るため賭ける"
        )
    else:
        record["decision"] = "pass"
        record["reason"] = (
            head + f" × {EDGE_MARGIN}（={required:.1f}%）に届かず、"
            "対市場の優位が不足しているため見送り"
        )
    return record


def _generate_prediction_texts(fixture: dict, built, mc_provider=None,
                               cache: dict | None = None, save=None,
                               points: dict | None = None) -> dict:
    """Generate every text (predictions, debate, final ballots, conclusion)
    before anything is posted.

    Failure policy (2026-11-05 user decisions): an AI whose generation
    fails irrecoverably at any stage is EXCLUDED from the race and the
    rest proceed; only an exclusion that would leave fewer than two
    active AIs aborts the run (before anything is posted, as before).
    After the final ballots, each AI's bets are reconciled against its
    final marks (see _reconcile_bets).

    Generation is incremental (re-review 2026-11-04, finding 4): each AI's
    prediction, debate+ballot, and the MC prose are stored into `cache`
    (and persisted via `save()`) as soon as they are produced, so a
    failure mid-generation never re-bills the steps that already
    succeeded — a re-run reuses them and generates only what is missing.
    """
    if cache is None:
        cache = {"predictions": {}, "debate": {}, "final_marks": {},
                 "final_probs": {}}
    else:
        cache.setdefault("predictions", {})
        cache.setdefault("debate", {})
        cache.setdefault("final_marks", {})
        cache.setdefault("final_probs", {})

    def _save() -> None:
        if save is not None:
            save()

    body = race_text(fixture)
    r = fixture["race"]
    opening = (
        f"【予想スレッド】第{r['edition']}回 {r['name']}（{r['grade']}）\n"
        f"{r['date']} {r['venue']} / {r['course']}{r['distance_m']}m / 馬場 {r['going']} / {r['field_size']}頭\n\n"
        + "\n".join(
            f"{h['number']} {h['name']}（{h['jockey']}）" for h in fixture["horses"]
        )
        + "\n\n各AIの皆さん、総合判断での予想をお願いします。議論ののち、最終結論を一つにまとめます。"
    )
    if points:
        display_of = {p["username"]: p["display"] for p, _ in built}
        display_of[MC_USERNAME] = "結論口座"
        opening += (
            "\n\n【累積収支（戦績）】\n"
            + _score_view(points, built, display_of)
            + "\n（買い目は残高によらない一律の上限で賭け、精算結果は累積収支として記録されます）"
        )
    opening += "\n\n" + hashtag_line(fixture)

    # --- roster management: an AI whose generation fails irrecoverably
    # is EXCLUDED from this race instead of aborting the whole run
    # (2026-11-05 user decision: one AI's bad output must not kill a
    # race the others completed fine). The floor is two active AIs: an
    # exclusion that would leave fewer aborts the run as before.
    # Exclusions are decided per generation pass and recorded in the
    # returned dict (hence in the state); they are deliberately NOT
    # cached in `gen` — a resumed run gives a previously failing AI
    # another chance, since provider failures are often transient.
    active: list = list(built)
    excluded: list[dict] = []

    def _exclude(persona, stage: str, reason: str) -> None:
        username = persona["username"]
        active[:] = [p for p in active if p[0]["username"] != username]
        excluded.append(
            {
                "username": username,
                "display": persona["display"],
                "stage": stage,
                "reason": reason,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )
        print(f"[excluded {username} at {stage}: {reason}]", file=sys.stderr)
        if not active or (len(active) < 2 and len(built) >= 2):
            raise RuntimeError(
                f"aborting: excluding {username} ({stage}: {reason}) leaves "
                f"{len(active)} active AI(s)"
            )

    # --- initial predictions (validated for every provider) ---
    predictions: dict[str, str] = {}
    for persona, prov in built:
        username = persona["username"]
        if username in cache["predictions"]:
            predictions[username] = cache["predictions"][username]
            continue
        # Only the flat cap reaches generation; the balance is a
        # scoreboard figure and is never an input (decoupled, 2026-11-08).
        cap = None
        if points and username in points:
            cap = points[username]["cap"]
        try:
            if persona.get("structured") and hasattr(prov, "chat_json"):
                text = generate_structured_prediction(
                    prov, fixture, cap=cap
                )
            else:
                text = _generate_text_prediction(
                    prov, persona, fixture, body, cap
                )
        except Exception as exc:  # noqa: BLE001 - exclusion, not abort
            _exclude(persona, "prediction", f"{type(exc).__name__}: {exc}")
            continue
        # Posted texts show the market win% next to the AI's own
        # estimates for its marked horses (deterministic annotation).
        text = annotate_market(text, fixture)
        predictions[username] = text
        cache["predictions"][username] = text
        _save()

    initial_marks = {
        p["username"]: canonical_marks(predictions[p["username"]], fixture)
        for p, _ in active
    }

    # --- debate + final ballots ---
    debate: dict[str, str] = {}
    final_marks: dict[str, dict] = {}
    final_probs: dict[str, dict] = {}
    for persona, prov in built:
        username = persona["username"]
        if username not in predictions:
            continue  # excluded at the prediction stage
        if username in cache["debate"] and username in cache["final_marks"]:
            debate[username] = cache["debate"][username]
            final_marks[username] = {
                mk: tuple(v) for mk, v in cache["final_marks"][username].items()
            }
            raw_probs = (cache.get("final_probs") or {}).get(username) or {}
            final_probs[username] = {
                int(n): int(p) for n, p in raw_probs.items()
            }
            continue
        try:
            text, marks, probs = _generate_debate_and_ballot(
                prov, persona, fixture, body, predictions, built
            )
        except Exception as exc:  # noqa: BLE001 - exclusion, not abort
            stage = getattr(exc, "stage", "debate")
            _exclude(persona, stage, f"{type(exc).__name__}: {exc}")
            continue
        debate[username] = text
        final_marks[username] = marks
        final_probs[username] = probs
        cache["debate"][username] = text
        cache["final_marks"][username] = {
            mk: list(v) for mk, v in marks.items()
        }
        cache["final_probs"][username] = {
            str(n): p for n, p in probs.items()
        }
        _save()

    # --- reconcile bets against the final marks ---
    final_bets, bet_revisions = _reconcile_bets(
        fixture, active, predictions, final_marks, points, cache, _save
    )

    # --- aggregation over the FINAL ballots (initial scores break ties) ---
    # The market votes as one extra ballot with the same weight as one
    # AI ballot (Item 1); it has no bets and no points account, and it
    # is recorded in the state and shown in the conclusion note.
    display_of_all = {p["username"]: p["display"] for p, _ in built}
    display_of_all[MARKET_USERNAME] = "市場"

    def _scores(marks_by_user: dict) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for username, marks in marks_by_user.items():
            for mark, (num, name) in marks.items():
                entry = out.setdefault(num, {"name": name, "score": 0, "marks": {}})
                entry["score"] += MARK_SCORE[mark]
                entry["marks"][display_of_all.get(username, username)] = mark
        return out

    market_marks = market_ballot_marks(fixture)
    initial_scores = _scores(initial_marks)
    scoring_marks = dict(final_marks)
    if market_marks:
        scoring_marks[MARKET_USERNAME] = market_marks
    final_scores = _scores(scoring_marks)
    ranking = sorted(
        final_scores.items(),
        key=lambda kv: (
            -kv[1]["score"],
            -initial_scores.get(kv[0], {"score": 0})["score"],
            kv[0],
        ),
    )

    bets_digest = "\n".join(
        f"{p['display']}: {format_bets(final_bets[p['username']]).replace(chr(10), ' / ')}"
        for p, _ in active
        if final_bets.get(p["username"])
    )
    # --- the conclusion gate (Item 2): the aggregate probability
    # vector over the final ballots decides whether the conclusion
    # account bets at all, and which horse anchors its win leg. An
    # AI whose ballot percentages are unavailable (e.g. a gen cache
    # from before they were recorded) contributes its initial
    # estimates instead, so the gate never runs on an empty panel.
    init_probs = {
        u: {num: win for num, win, _t in parse_probabilities(text).values()}
        for u, text in predictions.items()
    }
    contribs: list[dict] = []
    for persona, _prov in active:
        u = persona["username"]
        fp = final_probs.get(u) or init_probs.get(u) or {}
        contribs.append({int(n): int(p) for n, p in fp.items()})
    gate = evaluate_conclusion_gate(fixture, contribs)
    # The conclusion account stakes a flat 1,000 points per race,
    # never derived from its balance (decoupled, 2026-11-08).
    final_total = 1000
    final_bet = None
    if gate["decision"] != "pass" and final_total > 0 and ranking:
        bet_ranking = ranking
        anchor = gate.get("anchor")
        if (
            gate["decision"] == "bet"
            and anchor is not None
            and ranking[0][0] != anchor
        ):
            names = {h["number"]: h["name"] for h in fixture["horses"]}
            bet_ranking = [
                (anchor, {"name": names.get(anchor, ""), "score": 0})
            ] + [entry for entry in ranking if entry[0] != anchor]
        final_bet = build_final_bet(bet_ranking, fixture, total=final_total)

    lines = [f"【最終結論】第{r['edition']}回 {r['name']}", "（討論後の最終票を集計）", ""]
    if ranking:
        top_num, top = ranking[0]
        lines.append(f"◎ {top_num} {top['name']}（総意の本命）")
        for num, info in ranking[1:4]:
            init = initial_scores.get(num, {"score": 0})["score"]
            lines.append(f"・ {num} {info['name']}（最終スコア {info['score']} / 初期 {init}）")
        lines.append("")
        lines.append("各AIの最終本命：")
        for persona, _ in active:
            m = final_marks[persona["username"]].get("◎")
            if m:
                lines.append(f"  {persona['display']}: ◎ {m[0]} {m[1]}")
        if market_marks:
            mkt = fixture_market(fixture)
            lines.append("")
            lines.append("市場（単勝オッズ）の票：")
            for mk in MARKS:
                num, name = market_marks[mk]
                lines.append(
                    f"  {mk} {num} {name}（市場勝率{mkt[num]['win_prob'] * 100:.1f}%）"
                )
        changed = []
        for persona, _ in active:
            before = initial_marks[persona["username"]].get("◎")
            after = final_marks[persona["username"]].get("◎")
            if before and after and before[0] != after[0]:
                changed.append(
                    f"  {persona['display']}: ◎ {before[0]} {before[1]} → {after[0]} {after[1]}"
                )
        lines.append("")
        if changed:
            lines.append("討論での本命の変動：")
            lines.extend(changed)
        else:
            lines.append("討論でも各AIの本命は変わりませんでした。")
        if excluded:
            lines.append("")
            lines.append("不参加のAI（生成エラーにより除外）:")
            for e in excluded:
                lines.append(f"  {e['display']}（{e['stage']}段階で除外）")
        fair_lines = []
        for persona, _ in active:
            probs = parse_probabilities(predictions[persona["username"]])
            honmei = initial_marks[persona["username"]].get("◎")
            p0 = probs.get("◎")
            if honmei and p0:
                dev = ""
                mp = market_win_prob(fixture, honmei[0])
                if mp is not None:
                    dev = (
                        f"・市場{mp * 100:.1f}%"
                        f"（乖離{p0[1] - mp * 100:+.0f}pt）"
                    )
                fair_lines.append(
                    f"  {persona['display']}: ◎ {honmei[0]} {honmei[1]} "
                    f"勝率{p0[1]}%{dev}（フェアオッズ {fair_odds(p0[1]):.1f}倍相当）"
                )
        if fair_lines:
            lines.append("")
            lines.append("各AIの本命評価（初期予想の勝率から算出した妥当オッズ）:")
            lines.extend(fair_lines)
    conclusion = "\n".join(lines)
    if mc_provider is not None and ranking:
        # The local LLM writes the moderator's summary prose; the ranking
        # above stays the deterministic ground truth. Any failure falls
        # back to the ranking-only conclusion. The prose no longer carries
        # a bet: the final bet below is built deterministically. The
        # outcome (including a fallback) is cached so a re-run does not
        # re-bill it.
        if "mc_prose" in cache:
            prose = cache["mc_prose"]
        else:
            prose = ""
            try:
                debate_digest = "\n\n".join(
                    f"--- {p['display']} の議論 ---\n{debate[p['username']][:400]}"
                    for p, _ in active
                )
                prose = mc_provider.chat(
                    "あなたは競馬予想の議論スレッドの司会AIです。各AIの議論を読み、"
                    "最終結論への総括を書きます。新しい印や順位は作らず、既に決まった"
                    "結論の根拠と、議論で浮かんだ注意点、討論で印が動いた点だけを"
                    "述べてください。買い目には触れないでください。",
                    f"レース: 第{r['edition']}回 {r['name']}\n"
                    f"最終結論の順位:\n" + "\n".join(lines[3:7]) + "\n\n"
                    f"各AIの議論（抜粋）:\n{debate_digest}\n\n"
                    "総括は500字以内で書いてください。",
                ).strip()
            except Exception as exc:  # noqa: BLE001 - prose is optional garnish
                print(f"[mc prose skipped: {exc}]", file=sys.stderr)
                prose = ""
            cache["mc_prose"] = prose
            _save()
        if prose:
            conclusion += "\n\n【司会総括（ローカルLLM）】\n" + prose
    if bets_digest:
        conclusion += "\n\n【各AIの買い目】\n" + bets_digest
    if bet_revisions:
        conclusion += "\n\n【買い目の差し替え】\n" + "\n".join(
            f"  {display_of_all.get(u, u)}: 最終票と矛盾したため買い目を再構成しました"
            f"（{rev['method']}）"
            for u, rev in bet_revisions.items()
        )
    if final_bet:
        conclusion += "\n\n【最終買い目（結論口座）】\n" + format_bets(final_bet)
        if gate["decision"] == "bet":
            conclusion += f"\n（結論口座の対市場判定: {gate['reason']}）"
    elif gate["decision"] == "pass":
        conclusion += (
            "\n\n【結論口座】見送り（このレースは賭けません）"
            f"\n理由: {gate['reason']}"
        )
    conclusion += "\n\n結果が出たら回顧します。"
    conclusion += "\n\n" + hashtag_line(fixture)

    serialize = lambda marks_by_user: {  # noqa: E731
        u: {k: list(v) for k, v in marks.items()}
        for u, marks in marks_by_user.items()
    }
    return {
        "race": f"第{r['edition']}回 {r['name']}",
        "texts": {
            "opening": opening,
            "predictions": predictions,
            "debate": debate,
            "conclusion": conclusion,
        },
        "initial_marks": serialize(initial_marks),
        "final_marks": serialize(final_marks),
        # Back-compat: "marks" is what reviews historically read; it now
        # holds the FINAL marks, which are what the conclusion used.
        "marks": serialize(final_marks),
        "conclusion_ranking": [
            {"number": num, "name": info["name"], "score": info["score"]}
            for num, info in ranking
        ],
        # "bets" (text form) and "final_bets" (structured) both carry
        # the RECONCILED bets — the ones consistent with the final
        # marks. Settlement reads final_bets; the original slips remain
        # visible in the prediction posts themselves and, for changed
        # AIs, in bet_revisions.
        "bets": {u: format_bets(bs) for u, bs in final_bets.items()},
        "final_bets": final_bets,
        "bet_revisions": bet_revisions,
        "excluded": excluded,
        "final_bet": format_bets(final_bet) if final_bet else None,
        # Market anchor (Item 1): the market's own ballot, recorded so
        # reviews can show what the odds said. No bets, no account.
        "market_ballot": (
            {"marks": serialize({MARKET_USERNAME: market_marks})[MARKET_USERNAME]}
            if market_marks else None
        ),
        # Ballot win percentages per AI (Item 2/4 inputs), keyed by
        # horse number as a string for JSON.
        "final_probs": {
            u: {str(n): p for n, p in probs.items()}
            for u, probs in final_probs.items()
        },
        # The conclusion gate record (Item 2): decision, anchor,
        # aggregate vs market numbers, and the full aggregate vector.
        "conclusion_gate": gate,
    }


def run_prediction(fixture: dict, emitter: Emitter, built, store: RunState,
                   mc_provider=None) -> dict:
    """Two-phase run: generate everything (validating as we go), persist,
    then post note by note, recording each note ID so an interrupted live
    run resumes instead of duplicating."""
    d = store.data
    if not d.get("texts"):
        # Points: snapshot every account before generation. Only the
        # flat cap reaches the prompts; the balances are kept for the
        # opening note's scoreboard (shown as 累積収支) and the state
        # audit trail. No relief top-up exists any more (decoupled,
        # 2026-11-08): a low balance changes nothing about a run.
        ledger = points_mod.PointsLedger(points_mod.ledger_path(emitter.live))
        usernames = [p["username"] for p, _ in built] + [MC_USERNAME]
        snap = ledger.snapshot(usernames)
        cache = d.setdefault(
            "gen", {"predictions": {}, "debate": {}, "final_marks": {},
                    "final_probs": {}}
        )
        generated = _generate_prediction_texts(
            fixture, built, mc_provider=mc_provider,
            cache=cache, save=store.save, points=snap,
        )
        d.update(generated)
        d["points"] = {
            "enabled": True,
            "balances_before": {u: v["balance"] for u, v in snap.items()},
            "caps": {u: v["cap"] for u, v in snap.items()},
        }
        d["stage"] = "generated"
        store.save()
    texts = d["texts"]
    ids = d.setdefault("note_ids", {})

    def post_once(key: str, username: str, text: str, reply_to) -> str:
        if key in ids:
            return ids[key]
        note_id = emitter.emit(username, text, reply_to)
        ids[key] = note_id
        store.save()
        return note_id

    root_id = post_once("root", MC_USERNAME, texts["opening"], None)
    ids.setdefault("predictions", {})
    ids.setdefault("debate", {})
    for persona, _ in built:
        u = persona["username"]
        if u not in texts["predictions"]:
            continue  # excluded during generation: nothing to post
        if u not in ids["predictions"]:
            ids["predictions"][u] = emitter.emit(u, texts["predictions"][u], root_id)
            store.save()
    for persona, _ in built:
        u = persona["username"]
        if u not in texts["debate"]:
            continue  # excluded during generation: nothing to post
        if u not in ids["debate"]:
            ids["debate"][u] = emitter.emit(u, texts["debate"][u], root_id)
            store.save()
    post_once("conclusion", MC_USERNAME, texts["conclusion"], root_id)
    d["stage"] = "complete"
    store.save()
    return d



REFLECTION_SYSTEM = """あなたは {display} です。競馬予想AIとして予想スレッドに参加し、レースが終わりました。
結果と自分の予想を振り返り、反省を短く述べてください。言い訳はせず、何を見誤ったか、上位に来た馬の何を過小評価したか、次回同じタイプのレースで何を変えるかを具体的に書いてください。自分の買い目が当たったかどうかにも正直に触れること。日本語で200〜300字程度に収めてください。"""


def reflection_user_prompt(
    fixture: dict, results: dict, state: dict, username: str
) -> str:
    actual = {f["number"]: f for f in results["finish"]}
    lines = [
        f"【レース結果】{results['race']}（勝ちタイム {results['winning_time']}）",
        "着順（上位5頭）:",
    ]
    for f in results["finish"][:5]:
        lines.append(
            f"  {f['position']}着 {f['number']} {f['name']}（{f['popularity']}番人気、単勝{f['win_odds']}倍）"
        )
    payouts = results.get("payouts_jpy", {})
    if payouts.get("win") and payouts.get("trio"):
        lines.append(
            f"払戻: 単勝 {list(payouts['win'].values())[0]}円 / "
            f"3連複 {list(payouts['trio'].keys())[0]} {list(payouts['trio'].values())[0]}円"
        )
    lines.append("")
    lines.append("【あなたの印と実際の着順】")
    marks_map = state.get("final_marks") or state.get("marks") or {}
    for mark, (num, name) in (marks_map.get(username) or {}).items():
        fin = actual.get(num)
        lines.append(
            f"  {mark} {num} {name} → {fin['position']}着" if fin else f"  {mark} {num} {name} → 着順不明"
        )
    bets = (state.get("bets") or {}).get(username)
    if bets:
        lines.append("")
        lines.append(f"【あなたの買い目】\n{bets}")
    ev_rows = ev_summary(fixture, results, state).get(username) or []
    if ev_rows:
        lines.append("")
        lines.append("【あなたの確率推定と期待値（勝率 × 最終単勝オッズ、1.0超は割安の見抜き）】")
        for r in ev_rows:
            if r["ev"] is not None:
                lines.append(
                    f"  {r['mark']} {r['number']} {r['name']}: 勝率{r['win_pct']}% × "
                    f"{r['odds']}倍 = EV {r['ev']:.2f}（{r['position']}着）"
                )
    rec = ((state.get("review") or {}).get("points") or {}).get(username)
    if rec:
        lines.append("")
        lines.append(
            f"【精算】賭け {rec['staked']:,}点 → 払戻 {rec['returned']:,}点、"
            f"累積収支は {points_mod.format_score(rec['balance'])}点になりました。"
        )
    lines.append("")
    lines.append("この結果を受けて、あなたの反省を述べてください。")
    return "\n".join(lines)


def build_review_text(fixture: dict, results: dict, state: dict | None) -> str:
    top = results["finish"][:5]
    lines = [
        f"【回顧】{results['race']}",
        f"勝ちタイム {results['winning_time']} / {results.get('race_summary', '')}",
        "",
        "着順：",
    ]
    for f in top:
        lines.append(f"  {f['position']}着 {f['number']} {f['name']}（{f['popularity']}番人気）")
    payouts = results.get("payouts_jpy", {})
    if payouts.get("win") and payouts.get("trio"):
        lines.append(
            f"払戻：単勝 {list(payouts['win'].values())[0]}円 / "
            f"3連複 {list(payouts['trio'].values())[0]}円"
        )
    lines.append("")
    if state and state.get("excluded"):
        lines.append(
            "不参加のAI（生成段階で除外され、精算・反省の対象外）: "
            + "、".join(
                f"@{e['username']}（{e.get('stage', '?')}段階）"
                for e in state["excluded"]
            )
        )
    if state and state.get("conclusion_ranking"):
        best = state["conclusion_ranking"][0]
        actual = {f["number"]: f["position"] for f in results["finish"]}
        pos = actual.get(best["number"])
        verdict = (
            "本命が優勝！" if pos == 1 else f"最終結論の本命 {best['name']} は {pos}着でした。"
        )
        lines.append(f"最終結論の検証：{verdict}")
        marks_map = state.get("final_marks") or state.get("marks") or {}
        for username, marks in marks_map.items():
            honmei = marks.get("◎")
            if honmei:
                lines.append(
                    f"  @{username} の◎ {honmei[1]} は {actual.get(honmei[0], '?')}着"
                )
    else:
        lines.append("（予想スレッドの記録がないため、結果のみの回顧です）")
    gate = (state or {}).get("conclusion_gate") or {}
    if gate.get("decision") == "pass":
        lines.append(
            f"結論口座は見送りでした（{gate.get('reason', '対市場の優位が不足')}）。"
        )
    ev = ev_summary(fixture, results, state)
    if ev:
        lines.append("")
        lines.append(
            "期待値の検証（各AIの◎の勝率 × 最終単勝オッズ。1.0を超えていれば、"
            "結果はともあれ市場の最終評価より割安に見抜いていました）:"
        )
        for username, rows in ev.items():
            honmei = next((r for r in rows if r["mark"] == "◎"), None)
            if honmei and honmei["ev"] is not None:
                lines.append(
                    f"  {display_name(username)} ◎ {honmei['name']}: "
                    f"勝率{honmei['win_pct']}% × {honmei['odds']}倍 = "
                    f"EV {honmei['ev']:.2f}（{honmei['position']}着）"
                )
    points_rec = (state.get("review") or {}).get("points")
    if points_rec:
        lines.append("")
        lines.append("精算（累積収支）：")
        for username, rec in points_rec.items():
            profit = rec["returned"] - rec["staked"]
            lines.append(
                f"  {display_name(username)}: 賭け {rec['staked']:,}点 → "
                f"払戻 {rec['returned']:,}点（{profit:+,}点）"
                f"累積収支 {points_mod.format_score(rec['balance'])}点"
            )
        standings = (state.get("review") or {}).get("standings") or {}
        if standings:
            order = sorted(standings.items(), key=lambda kv: -kv[1])
            lines.append(
                "現在の順位（累積収支）: "
                + " / ".join(
                    f"{i + 1}位 {display_name(u)} "
                    f"{points_mod.format_score(b)}点"
                    for i, (u, b) in enumerate(order)
                )
            )
    ab_line = ab_standings_text(
        ((state.get("review") or {}).get("ab") or {}).get("totals")
    )
    if ab_line:
        lines.append("")
        lines.append(ab_line)
    lines.append("次回の予想に活かします。")
    lines.append("")
    lines.append(hashtag_line(fixture))
    return "\n".join(lines)


def display_name(username: str) -> str:
    for p in PERSONAS + [LOCAL_PERSONA]:
        if p["username"] == username:
            return p["display"]
    return "結論口座" if username == MC_USERNAME else username


# --- Points settlement and expected-value review ----------------------------
# Settlement uses the real payouts (per 100 points) parsed into the
# results fixture. A bet that WINS but whose payout is missing from the
# fixture returns None: the caller must fail loudly (regenerate the
# results with the full payout table) instead of silently paying 0.

PAYOUT_KEY = {
    "単勝": "win", "複勝": "place", "枠連": "bracket", "馬連": "quinella",
    "ワイド": "wide", "馬単": "exacta", "3連複": "trio", "3連単": "trifecta",
}


def bet_return(bet: dict, fixture: dict, results: dict) -> int | None:
    finish = results.get("finish") or []
    if len(finish) < 3:
        return None
    pos = {f["number"]: f["position"] for f in finish}
    first, second, third = finish[0]["number"], finish[1]["number"], finish[2]["number"]
    brackets = {h["number"]: h["bracket"] for h in fixture["horses"]}
    kind = bet["type"]
    nums = list(bet["numbers"])
    won = False
    key = None
    if kind == "単勝":
        won = nums[0] == first
        key = str(nums[0])
    elif kind == "複勝":
        places = 2 if len(fixture["horses"]) <= 7 else 3
        won = pos.get(nums[0], 99) <= places
        key = str(nums[0])
    elif kind == "枠連":
        pair = sorted([brackets.get(first, 0), brackets.get(second, 0)])
        won = pair == sorted(nums)
        key = "-".join(str(n) for n in sorted(nums))
    elif kind == "馬連":
        won = sorted(nums) == sorted([first, second])
        key = "-".join(str(n) for n in sorted(nums))
    elif kind == "ワイド":
        won = all(pos.get(n, 99) <= 3 for n in nums)
        key = "-".join(str(n) for n in sorted(nums))
    elif kind == "馬単":
        won = nums == [first, second]
        key = "-".join(str(n) for n in nums)
    elif kind == "3連複":
        won = sorted(nums) == sorted([first, second, third])
        key = "-".join(str(n) for n in sorted(nums))
    elif kind == "3連単":
        won = nums == [first, second, third]
        key = "-".join(str(n) for n in nums)
    if not won:
        return 0
    table = (results.get("payouts_jpy") or {}).get(PAYOUT_KEY.get(kind, ""))
    if table is None or key not in table:
        return None
    return bet["amount"] // 100 * int(table[key])


def settle_race_points(fixture: dict, results: dict, state: dict,
                       ledger: "points_mod.PointsLedger") -> dict:
    """Settle every saved bet of this race against the real payouts.
    Returns {username: settlement record}. Raises RuntimeError when a
    winning bet cannot be priced from the results fixture. AIs listed
    in state["excluded"] are never settled, at any exclusion stage."""
    race_id = fixture_race_id(fixture) or state.get("race_id") or ""
    race_name = state.get("race") or ""
    records: dict[str, dict] = {}
    texts = (state.get("texts") or {}).get("predictions") or {}
    # An AI excluded at ANY stage is out of settlement for this race,
    # even when it still has a saved prediction text: debate/ballot
    # exclusions happen after the prediction was posted, so the initial
    # bets exist but must never be settled (2026-11-08 Saga bug: @local,
    # excluded at the debate stage, lost 3,100 points this way).
    # Prediction-stage exclusions have no text and are skipped either
    # way; this guard is what covers the later stages.
    excluded = {e.get("username") for e in (state.get("excluded") or [])}
    # The reconciled bets (state["final_bets"]) are authoritative: they
    # are the slips consistent with each AI's final marks. States saved
    # before reconciliation existed fall back to parsing the prediction
    # text, as before.
    final_bets_map = state.get("final_bets") or {}
    for username, text in texts.items():
        if username in excluded:
            continue  # excluded at some stage: no bets, no settlement
        if username in final_bets_map:
            bets = [
                {
                    "type": b["type"],
                    "numbers": list(b["numbers"]),
                    "amount": int(b["amount"]),
                }
                for b in final_bets_map[username]
            ]
        else:
            bets, perr = parse_bet_lines(extract_bets(text, limit=100000))
            if perr:
                raise RuntimeError(f"saved bets for {username} do not parse: {perr}")
        total_return = 0
        for b in bets:
            ret = bet_return(b, fixture, results)
            if ret is None:
                raise RuntimeError(
                    f"cannot settle {username}'s bet {b}: payout data missing "
                    "from the results fixture (regenerate it with the full "
                    "payout table)"
                )
            total_return += ret
        records[username] = ledger.settle(
            race_id, race_name, username,
            sum(b["amount"] for b in bets), total_return,
        )
    final_bet_text = state.get("final_bet")
    if final_bet_text:
        bets, perr = parse_bet_lines(final_bet_text)
        if perr:
            raise RuntimeError(f"saved final bet does not parse: {perr}")
        total_return = 0
        for b in bets:
            ret = bet_return(b, fixture, results)
            if ret is None:
                raise RuntimeError(
                    f"cannot settle the final bet {b}: payout data missing"
                )
            total_return += ret
        records[MC_USERNAME] = ledger.settle(
            race_id, race_name, MC_USERNAME,
            sum(b["amount"] for b in bets), total_return,
        )
    elif (state.get("conclusion_gate") or {}).get("decision") == "pass":
        # 見送り (Item 2 gate): the conclusion account staked nothing.
        # Record a zero-stake settlement so the pass is on the ledger
        # and in the review's standings without moving any balance;
        # idempotent like every other settlement.
        records[MC_USERNAME] = ledger.settle(
            race_id, race_name, MC_USERNAME, 0, 0
        )
    return records


def ev_summary(fixture: dict, results: dict, state: dict) -> dict[str, list[dict]]:
    """Per AI, per marked horse: the stated win probability against the
    actual closing win odds. ev = win_pct/100 x closing odds; ev > 1
    means the AI priced the horse shorter than the market closed — it
    found value whether or not the horse won."""
    odds = {f["number"]: f.get("win_odds") for f in results.get("finish") or []}
    pos = {f["number"]: f["position"] for f in results.get("finish") or []}
    names = {h["number"]: h["name"] for h in fixture["horses"]}
    texts = (state.get("texts") or {}).get("predictions") or {}
    out: dict[str, list[dict]] = {}
    for username, text in texts.items():
        probs = parse_probabilities(text)
        rows = []
        for mk in MARKS:
            if mk not in probs:
                continue
            num, win_pct, _top3 = probs[mk]
            o = odds.get(num)
            rows.append({
                "mark": mk, "number": num, "name": names.get(num, ""),
                "win_pct": win_pct, "odds": o,
                "ev": (win_pct / 100.0 * o) if o else None,
                "position": pos.get(num),
            })
        if rows:
            out[username] = rows
    return out


# --- A/B measurement: pre-debate vs post-debate aggregates (Item 4) --------
# Every settled points-era race scores three predictors against the
# actual result: PRE (the aggregate built from the initial
# predictions), POST (the aggregate built from the final ballots) and
# CONCL (the conclusion as posted). PRE/POST carry full-field
# probability vectors (build_probability_vector), so they are scored
# by log-loss against the winner and by the ROI of a flat 100-point
# win bet on their top pick. CONCL's ranking is a mark-score aggregate
# (final ballots + the market ballot), NOT a probability vector, so
# CONCL is scored by its bet outcome only: a flat 100-point win bet on
# its anchor pick — or "no bet" (roi None) when the Item 2 gate passed
# the race. Records live in runs/ab-ledger.json (dry runs:
# runs/dryrun/ab-ledger.json), keyed by race_id, so re-running a
# review never double-counts. Review notes carry the standings once
# AB_STANDINGS_MIN_RACES races are recorded; whether the conclusion
# bet should default to the PRE aggregate is the user's decision at
# AB_JUDGE_AT_RACES races — until then this is measurement only and
# the conclusion keeps using the post-debate ballots.

def ab_ledger_path(live: bool) -> Path:
    return (
        Path("runs/ab-ledger.json") if live
        else Path("runs/dryrun/ab-ledger.json")
    )


class ABLedger:
    """Per-race A/B records with running totals, mirroring the points
    ledger's file discipline. A missing file starts empty; a corrupt
    file starts fresh (stderr note + a flag in the data) — measurement
    must never break settlement or a review."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.recovered = False
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or not isinstance(
                    data.get("races"), dict
                ):
                    raise ValueError("unexpected A/B ledger shape")
                self.data = data
            except (ValueError, OSError) as exc:
                print(
                    f"[A/B ledger at {self.path} unreadable ({exc}); "
                    "starting fresh]",
                    file=sys.stderr,
                )
                self.data = {
                    "version": 1,
                    "races": {},
                    "recovered_from_corrupt": True,
                }
                self.recovered = True
        else:
            self.data = {"version": 1, "races": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def record(self, record: dict) -> dict:
        """Store one race's record. Idempotent per race_id: a second
        call returns the stored record unchanged."""
        rid = record["race_id"]
        if rid in self.data["races"]:
            return self.data["races"][rid]
        self.data["races"][rid] = record
        self.save()
        return record

    def totals(self) -> dict:
        recs = list(self.data["races"].values())

        def _mean(values: list) -> float | None:
            vals = [v for v in values if v is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        out: dict = {"n": len(recs)}
        for side in ("pre", "post"):
            out[f"{side}_log_loss"] = _mean(
                [(r.get(side) or {}).get("log_loss") for r in recs]
            )
            out[f"{side}_roi"] = _mean(
                [(r.get(side) or {}).get("roi") for r in recs]
            )
        concl_rois = [
            (r.get("concl") or {}).get("roi") for r in recs
        ]
        out["concl_roi"] = _mean(concl_rois)
        out["concl_bets"] = sum(1 for v in concl_rois if v is not None)
        out["concl_passes"] = sum(
            1 for r in recs
            if (r.get("concl") or {}).get("decision") == "pass"
        )
        return out


def _flat_win_roi(pick: int | None, winner: int,
                  odds_map: dict) -> float | None:
    """Return ratio (payout / stake) of a flat win bet on `pick` at
    closing odds: the odds themselves on a hit, 0.0 on a miss, None
    when the bet cannot be priced."""
    if pick is None:
        return None
    if pick != winner:
        return 0.0
    odds = odds_map.get(pick)
    return float(odds) if odds else None


def _score_vector(vector: dict, winner: int, odds_map: dict) -> dict:
    """Log-loss of a full-field probability vector (percent) against
    the actual winner, plus the flat-bet ROI of its top pick."""
    if not vector:
        return {"pick": None, "log_loss": None, "roi": None}
    pick = max(vector.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    p = max(vector.get(winner, 0.0) / 100.0, 1e-6)
    return {
        "pick": pick,
        "log_loss": round(-math.log(p), 4),
        "roi": _flat_win_roi(pick, winner, odds_map),
    }


def score_ab_race(fixture: dict, results: dict, state: dict) -> dict:
    """Build this race's A/B record — or {"skipped": reason} when the
    state lacks what scoring needs. States saved before final_probs
    existed (e.g. the 2026-11-05 Monbetsu/Funabashi live states) skip
    cleanly: their reviews proceed without A/B rather than crashing."""
    finish = results.get("finish") or []
    if not finish:
        return {"skipped": "結果の着順表がありません"}
    winner = finish[0]["number"]
    odds_map = {f["number"]: f.get("win_odds") for f in finish}
    texts = (state.get("texts") or {}).get("predictions") or {}
    initial_marks = state.get("initial_marks") or {}
    final_probs = state.get("final_probs") or {}
    if not texts or not initial_marks:
        return {"skipped": "初期予想の記録がない旧形式stateのためA/B対象外"}
    if not final_probs:
        return {"skipped": "最終票の勝率が記録されていない旧形式stateのためA/B対象外"}
    pre_contribs = []
    for username in initial_marks:
        probs = parse_probabilities(texts.get(username, ""))
        pre_contribs.append(
            {num: win for num, win, _t in probs.values()}
        )
    post_contribs = [
        {int(n): int(p) for n, p in (final_probs.get(u) or {}).items()}
        for u in (state.get("final_marks") or state.get("marks") or {})
    ]
    gate = state.get("conclusion_gate") or {}
    ranking = state.get("conclusion_ranking") or []
    top = ranking[0]["number"] if ranking else None
    if gate.get("decision") == "pass":
        concl = {
            "pick": gate.get("anchor") or top,
            "decision": "pass",
            "roi": None,
        }
    elif gate.get("decision") == "bet":
        pick = gate.get("anchor") or top
        concl = {
            "pick": pick,
            "decision": "bet",
            "roi": _flat_win_roi(pick, winner, odds_map),
        }
    elif state.get("final_bet"):
        concl = {
            "pick": top,
            "decision": "ungated",
            "roi": _flat_win_roi(top, winner, odds_map),
        }
    else:
        concl = {"pick": top, "decision": "none", "roi": None}
    return {
        "race_id": fixture_race_id(fixture) or state.get("race_id") or "",
        "race": state.get("race") or "",
        "winner": winner,
        "market_missing": not bool(fixture_market(fixture)),
        "pre": _score_vector(
            build_probability_vector(pre_contribs, fixture), winner, odds_map
        ),
        "post": _score_vector(
            build_probability_vector(post_contribs, fixture), winner, odds_map
        ),
        "concl": concl,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def record_ab_for_race(fixture: dict, results: dict, state: dict,
                       live: bool) -> dict:
    """Score this race into the A/B ledger (idempotent per race) and
    return {"record":…, "totals":…} for the state's review section, or
    {"skipped": reason} for states that cannot be scored."""
    scored = score_ab_race(fixture, results, state)
    if "skipped" in scored:
        return scored
    ledger = ABLedger(ab_ledger_path(live))
    record = ledger.record(scored)
    return {"record": record, "totals": ledger.totals()}


def ab_standings_text(totals: dict | None) -> str | None:
    """The compact A/B standings line for review notes; None until at
    least AB_STANDINGS_MIN_RACES races are recorded."""
    if not totals or totals.get("n", 0) < AB_STANDINGS_MIN_RACES:
        return None

    def _ll(x) -> str:
        return f"{x:.3f}" if x is not None else "—"

    def _pct(x) -> str:
        return f"{x * 100:.0f}%" if x is not None else "—"

    return (
        f"【討論のA/B測定】{totals['n']}レース時点: "
        f"log-loss 討論前 {_ll(totals.get('pre_log_loss'))} / "
        f"討論後 {_ll(totals.get('post_log_loss'))}、"
        f"単勝回収率 討論前 {_pct(totals.get('pre_roi'))} / "
        f"討論後 {_pct(totals.get('post_roi'))}"
        f"（結論の賭け方針の判定は{AB_JUDGE_AT_RACES}レースで実施）"
    )


def run_review(
    fixture: dict,
    results: dict,
    store: RunState,
    emitter: Emitter,
    built: list | None = None,
    force: bool = False,
) -> None:
    """Post the review note and each AI's reflection, with a per-item
    ledger in the state file. Completion is tracked per reflection, so a
    partially failed review resumes by posting only what is missing
    (findings 7 and 9); generation results are cached in the ledger so a
    resume never re-bills a provider for a text it already produced."""
    state = store.data
    if force:
        state.pop("review", None)
    review = state.setdefault("review", {})
    if review.get("legacy_posted_at") and not review.get("note_id"):
        raise SystemExit(
            "legacy state: a review was already posted at "
            f"{review['legacy_posted_at']} by the previous version, and its "
            "note ID is unknown, so reflections cannot be threaded onto "
            "it. Leaving it untouched (a full repost needs --force-review)."
        )
    # Points settlement (points-era states only), before anything is
    # generated or posted: if a winning bet cannot be priced from the
    # results fixture, the run stops here with nothing public, and the
    # operator regenerates the results and re-runs.
    if (state.get("points") or {}).get("enabled") and not review.get("points"):
        points_ledger = points_mod.PointsLedger(
            points_mod.ledger_path(emitter.live)
        )
        try:
            records = settle_race_points(fixture, results, state, points_ledger)
        except RuntimeError as e:
            raise SystemExit(f"points settlement failed: {e}")
        review["points"] = {
            u: {
                "staked": r["staked"],
                "returned": r["returned"],
                "balance": r["balance"],
            }
            for u, r in records.items()
        }
        review["standings"] = {
            u: points_ledger.balance(u) for u in points_ledger.data["accounts"]
        }
        store.save()

    # A/B measurement (Item 4), recorded before the review note is
    # built so the note can carry the standings line. Points-era
    # states only; idempotent per race; legacy states record a skip
    # reason. Measurement must never break a review, hence the guard.
    if (state.get("points") or {}).get("enabled") and "ab" not in review:
        try:
            review["ab"] = record_ab_for_race(
                fixture, results, state, live=emitter.live
            )
        except Exception as e:  # noqa: BLE001 - measurement only
            review["ab"] = {"skipped": f"A/B記録に失敗しました: {e}"}
        store.save()

    ledger: dict = review.setdefault("reflections", {})

    marks_map = state.get("final_marks") or state.get("marks") or {}
    participants = []
    if built:
        for persona, prov in built:
            username = persona["username"]
            if username in marks_map:
                participants.append((persona, prov))

    def _all_posted() -> bool:
        return bool(review.get("note_id")) and all(
            ledger.get(p["username"], {}).get("status") == "posted"
            for p, _ in participants
        )

    if review.get("note_id") and _all_posted():
        raise SystemExit(
            "review already fully posted (note "
            f"{review['note_id']}); use --force-review to post it all again"
        )

    failures: list[str] = []

    # Generate any missing reflection texts first (cached in the ledger).
    for persona, prov in participants:
        username = persona["username"]
        entry = ledger.setdefault(username, {})
        if entry.get("status") == "posted" or entry.get("text"):
            continue
        try:
            text = prov.chat(
                REFLECTION_SYSTEM.format(display=persona["display"]),
                reflection_user_prompt(fixture, results, state, username),
            )
        except Exception as e:  # noqa: BLE001 - record and continue with others
            entry["status"] = "failed"
            entry["error"] = f"generation: {e}"
            failures.append(username)
            store.save()
            continue
        if text and text.strip():
            entry["text"] = f"【反省】{results['race']}\n\n{text.strip()}"
            entry["status"] = "generated"
        else:
            entry["status"] = "failed"
            entry["error"] = "generation: empty response"
            failures.append(username)
        store.save()

    # Post the review note itself.
    if not review.get("note_id"):
        review_id = emitter.emit(
            MC_USERNAME, build_review_text(fixture, results, state), None
        )
        review["note_id"] = review_id
        review["posted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        state["review_posted_at"] = review["posted_at"]  # legacy field
        store.save()
    review_id = review["note_id"]

    # Post the reflections that are ready.
    for persona, _prov in participants:
        username = persona["username"]
        entry = ledger.get(username, {})
        if entry.get("status") == "posted" or not entry.get("text"):
            continue
        try:
            entry["note_id"] = emitter.emit(username, entry["text"], review_id)
            entry["status"] = "posted"
            entry.pop("error", None)
        except Exception as e:  # noqa: BLE001
            entry["status"] = "failed"
            entry["error"] = f"posting: {e}"
            if username not in failures:
                failures.append(username)
        store.save()

    if failures:
        raise SystemExit(
            "review posted, but reflections are incomplete for: "
            + ", ".join(failures)
            + ". Re-run the same command to post only the missing ones."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", default="fixtures/2025-elizabeth-cup.json")
    ap.add_argument("--mock", action="store_true", help="force MockProvider for all AIs")
    ap.add_argument("--live", action="store_true", help="actually post to Misskey")
    ap.add_argument("--dry-run", action="store_true", help="print posts instead (default)")
    ap.add_argument("--review", action="store_true", help="post a review using --results")
    ap.add_argument("--results", default=None, help="results JSON (review mode only)")
    ap.add_argument("--force-review", action="store_true",
                    help="post a review again even if the ledger says it is complete")
    ap.add_argument("--force", action="store_true",
                    help="run even if the prediction state is marked complete "
                         "(the resume logic still avoids duplicate posts; "
                         "for LEGACY states, whose note IDs were never "
                         "recorded, a forced re-run can duplicate posts)")
    ap.add_argument("--local", action="store_true",
                    help="add the local LLM (Ollama on take-pc) as a fourth predictor "
                         "and let it write the moderator summary prose")
    ap.add_argument("--pool-plan", nargs="*", default=None,
                    help="write the OpenAI pool plan for the given fixture "
                         "files (the day's races) and exit; combine with --live "
                         "to write the live plan")
    args = ap.parse_args()

    live = bool(args.live) and not args.dry_run
    if args.pool_plan is not None:
        if not args.pool_plan:
            raise SystemExit("--pool-plan needs at least one fixture path")
        items = [(Path(p).stem,
                  json.loads(Path(p).read_text(encoding="utf-8")))
                 for p in args.pool_plan]
        plans = create_pool_plan(
            items,
            runs_dir=Path("runs") if live else Path("runs") / "dryrun")
        for plan in plans:
            print(f"[pool plan {plan['pool_date']}: flagship used "
                  f"{plan['flagship_used']}, budget "
                  f"{plan['budget_remaining']}]", file=sys.stderr)
            for e in sorted(plan["races"],
                            key=lambda e: (-e["rank"], e["start_time"])):
                print(f"  {e['start_time']} {e['stem']} "
                      f"({e['grade'] or '?'}) -> {e['assigned']}")
        return 0
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    roster = PERSONAS + ([LOCAL_PERSONA] if args.local else [])
    built = build_providers(mock=args.mock, roster=roster)
    preflight(built, live=live)
    pool_info: dict = {}
    effective_models = apply_top_grade_models(
        fixture, built, live=live, stem=Path(args.fixture).stem,
        runs_dir=Path("runs") if live else Path("runs") / "dryrun",
        info_out=pool_info)
    if is_top_grade(fixture):
        print(f"[top-grade race: models {effective_models}]",
              file=sys.stderr)
    elif pool_info.get("source") in ("plan", "guard"):
        print(f"[pool allocation: chatgpt={effective_models.get('chatgpt')} "
              f"({pool_info['source']}, flagship budget remaining "
              f"~{pool_info.get('remaining')})]", file=sys.stderr)

    if args.review:
        if not args.results:
            raise SystemExit("--review requires --results <file>")
        results = json.loads(Path(args.results).read_text(encoding="utf-8"))
        check_race_identity(fixture, results, None)
        pred_path = state_path_for(args.fixture, live)
        path = pred_path if pred_path.exists() else state_path_for(
            args.fixture, live, review_only=True
        )
        lock = RaceLock(args.fixture) if live else None
        if lock:
            lock.__enter__()
        try:
            store = RunState.open(path, fixture, "live" if live else "dryrun")
            check_race_identity(fixture, None, store.data)
            _backfill_race_id(store, fixture)
            recorded = store.data.get("models")
            if recorded:
                # Reflections use the models that made the predictions.
                apply_top_grade_models(
                    fixture, built, live=live,
                    stem=Path(args.fixture).stem,
                    runs_dir=Path("runs") if live
                    else Path("runs") / "dryrun",
                    recorded=recorded, info_out=pool_info)
            emitter = Emitter(live=live, usernames=[p["username"] for p in roster])
            run_review(fixture, results, store, emitter, built=built,
                       force=args.force_review)
        finally:
            if lock:
                lock.__exit__()
        return 0

    mc_provider = None
    if args.local and not args.mock:
        mc_provider = providers.OllamaProvider()
    lock = RaceLock(args.fixture) if live else None
    if lock:
        lock.__enter__()
    try:
        store = RunState.open(
            state_path_for(args.fixture, live), fixture,
            "live" if live else "dryrun",
        )
        check_race_identity(fixture, None, store.data)
        _backfill_race_id(store, fixture)
        _changed = False
        if store.data.get("models") != effective_models:
            store.data["models"] = effective_models
            _changed = True
        if store.data.get("pool") != pool_info:
            store.data["pool"] = pool_info
            _changed = True
        if _changed:
            store.save()
        if store.data.get("stage") == "complete" and not args.force:
            raise SystemExit(
                "prediction already completed for this race "
                f"(state: {store.path}); use --force to re-enter — the "
                "resume logic will still avoid duplicate posts"
            )
        emitter = Emitter(live=live, usernames=[p["username"] for p in roster])
        run_prediction(fixture, emitter, built, store, mc_provider=mc_provider)
    finally:
        if lock:
            lock.__exit__()
    print(f"\n[state saved: {store.path}]", file=sys.stderr)
    return 0




if __name__ == "__main__":
    raise SystemExit(main())

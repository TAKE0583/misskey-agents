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
import os
import re
import sys
import time
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
MARK_RE = re.compile(r"([◎○▲△])\s*(?:馬番)?\s*(\d{1,2})\s*番?\s+([ァ-ヶー・]+)")

SYSTEM_TEMPLATE = """あなたは {display} です。競馬予想AIとして、他のAIと議論して最終的に一つの結論に収束させる場に参加しています。
予想は特定の観点に偏らず、過去の走破内容・騎手・枠順・馬場状態・脚質と想定ペース・近走の勢いを総合的に判断してください。
出力は日本語で、必ず次の形式の印を含めてください（各行：印、馬番、馬名、勝率、3着内率、その後に簡潔な理由）:
◎ 馬番 馬名 勝率○% 3着内○% — 理由
○ 馬番 馬名 勝率○% 3着内○% — 理由
▲ 馬番 馬名 勝率○% 3着内○% — 理由
△ 馬番 馬名 勝率○% 3着内○% — 理由
勝率と3着内率はあなたの推定です。勝率は1〜90%、3着内率は勝率以上99%以下にし、4頭の勝率の合計は100%を超えないでください。人気だけに引きずられず、「勝つ確率が高いのに評価が割れそうな馬」を見抜くことが持ち点を増やす近道です。
印のあとに、あなた自身が馬券を買うならどう買うかも、必ず次の形式で書いてください（券種・組み合わせ・金額を1行に1点。金額は持ち点から賭ける点数で、上限は別途指示されます）:
【買い目】
単勝 7 500
3連複 1-7-11 500
全体で800字程度に収めてください。"""


def race_text(fixture: dict, form_limit: int | None = None) -> str:
    r = fixture["race"]
    lines = [
        f"第{r['edition']}回 {r['name']}（{r['grade']}）",
        f"{r['date']} {r['venue']} {r['race_number']}R / {r['course']} {r['distance_m']}m / 馬場：{r['going']} / {r['conditions']} / {r['field_size']}頭立て",
        "",
        "【出走馬】",
    ]
    for h in fixture["horses"]:
        recent = h["recent_form"][:form_limit] if form_limit else h["recent_form"]
        form = " / ".join(recent)
        lines.append(
            f"{h['number']} {h['name']}（{h['sex_age']}・{h['weight_kg']}kg・{h['jockey']}・{h['trainer']}厩舎）近走：{form}"
        )
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
出力は指定されたJSON形式のみです。印は◎○▲△の4つを、すべて異なる馬番で選んでください。各印には勝率（win_pct、1〜90の整数）と3着内率（top3_pct、勝率以上99以下の整数）の推定を付けてください。4つの勝率の合計は100を超えないでください。買い目の金額は持ち点から賭ける点数で、合計は指示された上限以内にしてください。理由の中でレース名や馬名に触れるときは、入力データの表記（漢字・かなの別を含む）をそのままコピーしてください。
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
    + bets). With max_total, bets are points capped by the AI's balance;
    without it, the legacy fixed total of 1,000 yen applies."""
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


def budget_line(balance: int | None, cap: int | None) -> str:
    """The staking instruction appended to a prediction prompt. In
    points mode the AI stakes from its own balance up to its cap; the
    legacy wording (no points context) keeps the fixed 1,000-yen rule."""
    if balance is not None and cap is not None:
        return (
            f"\n\nあなたの現在の持ち点は {balance:,} 点です。買い目の合計は "
            f"{cap:,} 点以内にし、100点単位で配分してください。"
            "自信のある買い目ほど厚く張ってください（当たれば実際の配当で"
            "持ち点が増え、外れれば賭けた分が減ります）。"
        )
    return "\n\n買い目の合計は必ず1,000円ちょうどにしてください。"


def generate_structured_prediction(prov, fixture: dict,
                                   balance: int | None = None,
                                   cap: int | None = None) -> str:
    body = race_text(fixture, form_limit=3)
    user = f"次のレースの予想をしてください。\n\n{body}" + budget_line(balance, cap)
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


def render_ballot_block(marks: dict) -> str:
    """The canonical 【最終票】 block for a validated ballot. The published
    debate text, the saved state, and the aggregation must all carry the
    SAME ballot (re-review 2026-11-04, finding 1: the local AI's prose
    ballot and its structured ballot could diverge)."""
    lines = ["【最終票】"]
    for mk in MARKS:
        if mk in marks:
            num, name = marks[mk]
            lines.append(f"{mk} {num} {name}")
    return "\n".join(lines)


def splice_ballot(text: str, marks: dict) -> str:
    """Replace (or append) the ballot section of a debate text with the
    canonical block rendered from the validated ballot."""
    head = text.rfind("【最終票】")
    base = text[:head].rstrip() if head >= 0 else text.rstrip()
    return base + "\n\n" + render_ballot_block(marks)


def generate_structured_ballot(prov, fixture: dict, context: str) -> dict[str, tuple[int, str]]:
    """The local LLM's final ballot after the debate, as {mark: (num, name)}
    with fixture-canonical names."""
    names = {h["number"]: h["name"] for h in fixture["horses"]}
    obj = _structured_with_retry(
        prov, LOCAL_SYSTEM, context, BALLOT_SCHEMA,
        lambda o: validate_ballot_obj(o, fixture),
        "local ballot",
    )
    return {
        m["mark"]: (m["number"], names.get(m["number"], ""))
        for m in obj["marks"]
    }


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


def _points_view(points: dict | None, built, display_of: dict) -> str:
    """One-line standings for posts: 'ChatGPT 10,000点 / ...'."""
    if not points:
        return ""
    parts = [
        f"{display_of.get(u, u)} {v['balance']:,}点" for u, v in points.items()
    ]
    return " / ".join(parts)


def _generate_prediction_texts(fixture: dict, built, mc_provider=None,
                               cache: dict | None = None, save=None,
                               points: dict | None = None) -> dict:
    """Generate every text (predictions, debate, final ballots, conclusion)
    before anything is posted. Any validation failure raises, so an invalid
    run posts nothing at all.

    Generation is incremental (re-review 2026-11-04, finding 4): each AI's
    prediction, debate+ballot, and the MC prose are stored into `cache`
    (and persisted via `save()`) as soon as they are produced, so a
    failure mid-generation never re-bills the steps that already
    succeeded — a re-run reuses them and generates only what is missing.
    """
    if cache is None:
        cache = {"predictions": {}, "debate": {}, "final_marks": {}}
    else:
        cache.setdefault("predictions", {})
        cache.setdefault("debate", {})
        cache.setdefault("final_marks", {})

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
            "\n\n【現在の持ち点】\n"
            + _points_view(points, built, display_of)
            + "\n（買い目は持ち点から賭けます。当たれば実際の配当で増え、外れれば賭けた分が減ります）"
        )

    # --- initial predictions (validated for every provider) ---
    predictions: dict[str, str] = {}
    for persona, prov in built:
        username = persona["username"]
        if username in cache["predictions"]:
            predictions[username] = cache["predictions"][username]
            continue
        system = SYSTEM_TEMPLATE.format(display=persona["display"])
        user = f"次のレースの予想をしてください。\n\n{body}"
        balance = cap = None
        if points and username in points:
            balance = points[username]["balance"]
            cap = points[username]["cap"]
        user += budget_line(balance, cap)
        if persona.get("structured") and hasattr(prov, "chat_json"):
            predictions[username] = generate_structured_prediction(
                prov, fixture, balance=balance, cap=cap
            )
            cache["predictions"][username] = predictions[username]
            _save()
            continue
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
        predictions[username] = text
        cache["predictions"][username] = text
        _save()

    initial_marks = {
        p["username"]: canonical_marks(predictions[p["username"]], fixture)
        for p, _ in built
    }

    # --- debate + final ballots ---
    debate: dict[str, str] = {}
    final_marks: dict[str, dict] = {}
    for persona, prov in built:
        username = persona["username"]
        if username in cache["debate"] and username in cache["final_marks"]:
            debate[username] = cache["debate"][username]
            final_marks[username] = {
                mk: tuple(v) for mk, v in cache["final_marks"][username].items()
            }
            continue
        others = "\n\n".join(
            f"--- {p['display']} の予想 ---\n{predictions[p['username']]}"
            for p, _ in built
            if p["username"] != username
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
                for p, _ in built
                if p["username"] != username
            )
            chat_budget = 2500
        system = SYSTEM_TEMPLATE.format(display=persona["display"])
        user = (
            "他のAIの予想が出そろいました。元の出走データと各AIの予想を踏まえて、"
            "議論として応答してください。最も同意できる点と最も異論のある点を"
            "具体的に挙げてください。\n"
            "最後に、議論を踏まえたあなたの最終票を必ず次の形式で書いてください"
            "（買い目は初期予想から変えません。最終票はすべて異なる実在の馬にしてください）:\n"
            "【最終票】\n◎ 馬番 馬名\n○ 馬番 馬名\n▲ 馬番 馬名\n△ 馬番 馬名\n\n"
            f"【元の出走データ】\n{prompt_body}\n\n"
            f"【自分の予想】\n{predictions[username]}\n\n【他のAIの予想】\n{prompt_others}"
        )
        text = prov.chat(system, user, max_tokens=chat_budget)
        if not text.strip():
            raise RuntimeError(f"empty debate reply from {username}")
        debate[username] = text
        if persona.get("structured") and hasattr(prov, "chat_json"):
            context = (
                "次のレースについて、他のAIとの議論を踏まえた最終票をJSONで出してください。\n\n"
                f"【元の出走データ】\n{race_text(fixture, form_limit=3)}\n\n"
                f"【自分の予想】\n{predictions[username]}\n\n"
                f"【他のAIの予想（抜粋）】\n{prompt_others}\n\n【自分の議論】\n{text[:800]}"
            )
            ballot = generate_structured_ballot(prov, fixture, context)
            final_marks[username] = ballot
            # Publish the validated structured ballot, not whatever the
            # free-form debate prose happened to claim (finding 1).
            debate[username] = splice_ballot(text, ballot)
            cache["debate"][username] = debate[username]
            cache["final_marks"][username] = {
                mk: list(v) for mk, v in ballot.items()
            }
            _save()
            continue
        marks, errors = _ballot_errors(text, fixture)
        if errors:
            print(f"[ballot retry for {username}: {errors}]", file=sys.stderr)
            retry = prov.chat(
                system,
                "最終票だけを出し直してください。前回の問題: "
                + " / ".join(errors)
                + "\n次の形式のみで出力してください:\n"
                "【最終票】\n◎ 馬番 馬名\n○ 馬番 馬名\n▲ 馬番 馬名\n△ 馬番 馬名\n\n"
                f"【元の出走データ】\n{body}",
            )
            marks, errors = _ballot_errors(retry, fixture)
            if errors:
                raise RuntimeError(
                    f"final ballot from {username} failed validation after retry: {errors}"
                )
            head = text.rfind("【最終票】")
            base = text[:head].rstrip() if head >= 0 else text.rstrip()
            debate[username] = base + "\n\n" + retry.strip()
        names = {h["number"]: h["name"] for h in fixture["horses"]}
        final_marks[username] = {
            mk: (num, names.get(num, name)) for mk, (num, name) in marks.items()
        }
        cache["debate"][username] = debate[username]
        cache["final_marks"][username] = {
            mk: list(v) for mk, v in final_marks[username].items()
        }
        _save()

    # --- aggregation over the FINAL ballots (initial scores break ties) ---
    def _scores(marks_by_user: dict) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for p, _ in built:
            for mark, (num, name) in marks_by_user[p["username"]].items():
                entry = out.setdefault(num, {"name": name, "score": 0, "marks": {}})
                entry["score"] += MARK_SCORE[mark]
                entry["marks"][p["display"]] = mark
        return out

    initial_scores = _scores(initial_marks)
    final_scores = _scores(final_marks)
    ranking = sorted(
        final_scores.items(),
        key=lambda kv: (
            -kv[1]["score"],
            -initial_scores.get(kv[0], {"score": 0})["score"],
            kv[0],
        ),
    )

    bets = {p["username"]: extract_bets(predictions[p["username"]]) for p, _ in built}
    bets_digest = "\n".join(
        f"{p['display']}: {bets[p['username']].replace(chr(10), ' / ')}"
        for p, _ in built
        if bets[p["username"]]
    )
    final_total = 1000
    if points and MC_USERNAME in points:
        final_total = min(1000, points[MC_USERNAME]["cap"])
    final_bet = (
        build_final_bet(ranking, fixture, total=final_total)
        if final_total > 0 else None
    )

    lines = [f"【最終結論】第{r['edition']}回 {r['name']}", "（討論後の最終票を集計）", ""]
    if ranking:
        top_num, top = ranking[0]
        lines.append(f"◎ {top_num} {top['name']}（総意の本命）")
        for num, info in ranking[1:4]:
            init = initial_scores.get(num, {"score": 0})["score"]
            lines.append(f"・ {num} {info['name']}（最終スコア {info['score']} / 初期 {init}）")
        lines.append("")
        lines.append("各AIの最終本命：")
        for persona, _ in built:
            m = final_marks[persona["username"]].get("◎")
            if m:
                lines.append(f"  {persona['display']}: ◎ {m[0]} {m[1]}")
        changed = []
        for persona, _ in built:
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
        fair_lines = []
        for persona, _ in built:
            probs = parse_probabilities(predictions[persona["username"]])
            honmei = initial_marks[persona["username"]].get("◎")
            p0 = probs.get("◎")
            if honmei and p0:
                fair_lines.append(
                    f"  {persona['display']}: ◎ {honmei[0]} {honmei[1]} "
                    f"勝率{p0[1]}%（フェアオッズ {fair_odds(p0[1]):.1f}倍相当）"
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
                    for p, _ in built
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
    if final_bet:
        conclusion += "\n\n【最終買い目（結論口座）】\n" + format_bets(final_bet)
    if points:
        display_of = {p["username"]: p["display"] for p, _ in built}
        display_of[MC_USERNAME] = "結論口座"
        conclusion += (
            "\n\n【持ち点（予想時点）】\n" + _points_view(points, built, display_of)
        )
    conclusion += "\n\n結果が出たら回顧します。"

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
        "bets": bets,
        "final_bet": format_bets(final_bet) if final_bet else None,
    }


def run_prediction(fixture: dict, emitter: Emitter, built, store: RunState,
                   mc_provider=None) -> dict:
    """Two-phase run: generate everything (validating as we go), persist,
    then post note by note, recording each note ID so an interrupted live
    run resumes instead of duplicating."""
    d = store.data
    if not d.get("texts"):
        # Points: snapshot every account's balance/cap before generation
        # (the prompts stake against these numbers), applying the relief
        # top-up first so no account is priced out of betting entirely.
        ledger = points_mod.PointsLedger(points_mod.ledger_path(emitter.live))
        race_id = fixture_race_id(fixture) or ""
        usernames = [p["username"] for p, _ in built] + [MC_USERNAME]
        for u in usernames:
            ledger.ensure_bettable(u, race_id)
        snap = ledger.snapshot(usernames)
        cache = d.setdefault(
            "gen", {"predictions": {}, "debate": {}, "final_marks": {}}
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
        if u not in ids["predictions"]:
            ids["predictions"][u] = emitter.emit(u, texts["predictions"][u], root_id)
            store.save()
    for persona, _ in built:
        u = persona["username"]
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
            f"【持ち点の精算】賭け {rec['staked']:,}点 → 払戻 {rec['returned']:,}点、"
            f"残高は {rec['balance']:,}点になりました。"
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
        lines.append("持ち点の精算：")
        for username, rec in points_rec.items():
            profit = rec["returned"] - rec["staked"]
            lines.append(
                f"  {display_name(username)}: 賭け {rec['staked']:,}点 → "
                f"払戻 {rec['returned']:,}点（{profit:+,}点）残高 {rec['balance']:,}点"
            )
        standings = (state.get("review") or {}).get("standings") or {}
        if standings:
            order = sorted(standings.items(), key=lambda kv: -kv[1])
            lines.append(
                "現在の順位: "
                + " / ".join(
                    f"{i + 1}位 {display_name(u)} {b:,}点"
                    for i, (u, b) in enumerate(order)
                )
            )
    lines.append("次回の予想に活かします。")
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
    winning bet cannot be priced from the results fixture."""
    race_id = fixture_race_id(fixture) or state.get("race_id") or ""
    race_name = state.get("race") or ""
    records: dict[str, dict] = {}
    texts = (state.get("texts") or {}).get("predictions") or {}
    for username, text in texts.items():
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
    args = ap.parse_args()

    live = bool(args.live) and not args.dry_run
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    roster = PERSONAS + ([LOCAL_PERSONA] if args.local else [])
    built = build_providers(mock=args.mock, roster=roster)
    preflight(built, live=live)

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

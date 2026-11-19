#!/usr/bin/env python3
"""Prediction validation: probabilities, marks, bets (extracted from orchestrator.py, 2026-11-19 split; behavior unchanged)."""
from __future__ import annotations

import re

from common import MARKS, PROB_LINE_RE
from race_text import parse_marks


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


BET_HEADING_RE = re.compile(r"【?買い目[^\n】]*】?\s*\n?")


def extract_bets(text: str, limit: int = 220) -> str:
    """Pull the betting section (買い目) out of a prediction, if present.

    A prediction's prose can mention 買い目 before the real section
    heading (the local renderer's rationale does), so the first
    textual occurrence is not necessarily the section. Heading
    candidates are tried in order and the first whose body parses as
    bet lines wins; a clean-but-empty body is next best. When no
    candidate parses, the first candidate is returned unchanged so
    upstream validators report the same errors as before (the
    error-feedback retry depends on them)."""
    text = text or ""
    first = ""
    first_clean = ""
    for m in BET_HEADING_RE.finditer(text):
        section = text[m.end():].strip()
        # Drop trailing meta lines the models sometimes append after the bets.
        section = re.split(r"\n\s*(?:以上|まとめ|総合判断)", section)[0].strip()
        if not first:
            first = section
        bets, errors = parse_bet_lines(section)
        if bets and not errors:
            return section[:limit]
        if not errors and not first_clean:
            first_clean = section
    return (first_clean or first)[:limit]

#!/usr/bin/env python3
"""Structured prediction/ballot generation and prompt templates (extracted from orchestrator.py, 2026-11-19 split; behavior unchanged)."""
from __future__ import annotations

import re
import sys
import time

from common import MARKS
from market import market_prompt_note
from race_text import parse_marks, race_text
from validation import (
    parse_probabilities, render_local_prediction, validate_local_prediction,
    validate_marks,
)


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
                                   cap: int | None = None,
                                   memory: str = "") -> str:
    body = race_text(fixture, form_limit=3)
    user = (
        f"次のレースの予想をしてください。\n\n{body}"
        + market_prompt_note(fixture)
        + memory
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

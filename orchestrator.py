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
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import providers  # noqa: E402
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
出力は日本語で、必ず次の形式の印を含めてください（各行：印、馬番、馬名、その後に簡潔な理由）:
◎ 馬番 馬名 — 理由
○ 馬番 馬名 — 理由
▲ 馬番 馬名 — 理由
△ 馬番 馬名 — 理由
印のあとに、予算1,000円であなた自身が馬券を買うならどう買うかも、必ず次の形式で書いてください（券種・組み合わせ・金額を1行に1点、合計が必ず1,000円ちょうどになること）:
【買い目（予算1,000円）】
単勝 7 500円
3連複 1-7-11 500円
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
出力は指定されたJSON形式のみです。印は◎○▲△の4つを、すべて異なる馬番で選んでください。買い目は合計が必ず1,000円ちょうどになるようにしてください。理由の中でレース名や馬名に触れるときは、入力データの表記（漢字・かなの別を含む）をそのままコピーしてください。
出力例: {"marks": [{"mark": "◎", "number": 7, "reason": "近走の内容が安定"}, {"mark": "○", "number": 1, "reason": "コース適性が高い"}, {"mark": "▲", "number": 11, "reason": "脚質が展開に合う"}, {"mark": "△", "number": 4, "reason": "人気薄だが一発がある"}], "bets": [{"type": "単勝", "combo": "7", "amount": 500}, {"type": "3連複", "combo": "1-7-11", "amount": 500}], "comment": "全体のまとめを1〜2文で。"}"""

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
                    "reason": {"type": "string"},
                },
                "required": ["mark", "number", "reason"],
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


def validate_local_prediction(obj: object, fixture: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["出力がJSONオブジェクトではありません"]
    valid_numbers = {h["number"] for h in fixture["horses"]}
    marks = obj.get("marks") or []
    seen_marks: set[str] = set()
    seen_numbers: set[int] = set()
    for m in marks:
        if not isinstance(m, dict):
            errors.append("marks の要素がオブジェクトではありません")
            continue
        if m.get("mark") not in MARKS:
            errors.append(f"印 '{m.get('mark')}' は ◎○▲△ のいずれかではありません")
        elif m["mark"] in seen_marks:
            errors.append(f"印 {m['mark']} が重複しています")
        seen_marks.add(m.get("mark"))
        num = m.get("number")
        if num not in valid_numbers:
            errors.append(f"馬番 {num} は出走馬に存在しません")
        elif num in seen_numbers:
            errors.append(f"馬番 {num} が重複して指名されています")
        seen_numbers.add(num)
    missing = [mk for mk in MARKS if mk not in seen_marks]
    if missing:
        errors.append(f"印が不足しています: {'・'.join(missing)}")
    bets = obj.get("bets") or []
    total = sum(b.get("amount", 0) for b in bets if isinstance(b, dict))
    if total != 1000:
        errors.append(f"買い目の合計が {total} 円です。必ず合計1,000円にしてください")
    return errors


def render_local_prediction(obj: dict, fixture: dict) -> str:
    names = {h["number"]: h["name"] for h in fixture["horses"]}
    lines = []
    ordered = sorted(
        obj["marks"], key=lambda m: MARKS.index(m["mark"]) if m["mark"] in MARKS else 9
    )
    for m in ordered:
        lines.append(
            f"{m['mark']} {m['number']} {names.get(m['number'], '')} — {str(m['reason']).strip()[:140]}"
        )
    comment = str(obj.get("comment") or "").strip()
    if comment:
        lines.append(comment[:300])
    lines.append("")
    lines.append("【買い目（予算1,000円）】")
    total = sum(b["amount"] for b in obj["bets"])
    for b in obj["bets"]:
        lines.append(f"{b['type']} {b['combo']} {b['amount']}円")
    if total != 1000 and obj["bets"]:
        # Deterministic repair (after a failed retry): fold the difference
        # into the largest bet so the posted total is always exactly 1,000.
        diff = 1000 - total
        idx = max(range(len(obj["bets"])), key=lambda i: obj["bets"][i]["amount"])
        if obj["bets"][idx]["amount"] + diff > 0:
            obj["bets"][idx]["amount"] += diff
            lines[-len(obj["bets"]) + idx] = (
                f"{obj['bets'][idx]['type']} {obj['bets'][idx]['combo']} "
                f"{obj['bets'][idx]['amount']}円"
            )
    return "\n".join(lines)


def generate_structured_prediction(prov, fixture: dict) -> str:
    body = race_text(fixture, form_limit=3)
    user = f"次のレースの予想をしてください。\n\n{body}"
    obj = prov.chat_json(LOCAL_SYSTEM, user, PREDICTION_SCHEMA)
    errors = validate_local_prediction(obj, fixture)
    if errors:
        print(f"[local structured retry: {errors}]", file=sys.stderr)
        retry_user = (
            user
            + "\n\n【重要】前回の出力に次の問題がありました: "
            + " / ".join(errors)
            + "\n問題をすべて直して、JSON形式で再度出力してください。"
        )
        obj2 = prov.chat_json(LOCAL_SYSTEM, retry_user, PREDICTION_SCHEMA)
        if len(validate_local_prediction(obj2, fixture)) <= len(errors):
            obj = obj2
    return render_local_prediction(obj, fixture)


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


def run_prediction(fixture: dict, emitter: Emitter, built, mc_provider=None) -> dict:
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
    root_id = emitter.emit(MC_USERNAME, opening, None)

    predictions: dict[str, str] = {}
    for persona, prov in built:
        system = SYSTEM_TEMPLATE.format(display=persona["display"])
        user = f"次のレースの予想をしてください。\n\n{body}"
        if persona.get("structured") and hasattr(prov, "chat_json"):
            # Local LLM: schema-constrained JSON, validated and rendered
            # by the orchestrator (see generate_structured_prediction).
            text = generate_structured_prediction(prov, fixture)
            predictions[persona["username"]] = text
            continue
        text = prov.chat(system, user)
        quality = lambda t: (len(parse_marks(t)), 1 if extract_bets(t) else 0)  # noqa: E731
        if quality(text) < (len(MARKS), 1):
            # Format insurance: some models (notably the local 14B) drop the
            # required mark lines or the bets section. Re-ask once with an
            # explicit correction, keeping whichever output is more complete.
            retry = prov.chat(
                system,
                user
                + "\n\n【重要】前回の出力は形式が崩れていました。必ず次の4行を"
                "そのままの形式で含めてください（番号付きリストや見出しは禁止）:\n"
                "◎ 馬番 馬名 — 理由\n○ 馬番 馬名 — 理由\n"
                "▲ 馬番 馬名 — 理由\n△ 馬番 馬名 — 理由\n"
                "さらに【買い目（予算1,000円）】のセクションも必ず含めてください。",
            )
            if quality(retry) > quality(text):
                text = retry
            print(
                f"[format retry for {persona['username']}: "
                f"marks={len(parse_marks(text))}, bets={'y' if extract_bets(text) else 'n'}]",
                file=sys.stderr,
            )
        if not text.strip():
            raise RuntimeError(f"empty prediction from {persona['username']}")
        predictions[persona["username"]] = text
    for persona, _ in built:
        emitter.emit(
            persona["username"], predictions[persona["username"]], root_id
        )

    # --- debate round ---
    debate: dict[str, str] = {}
    for persona, prov in built:
        others = "\n\n".join(
            f"--- {p['display']} の予想 ---\n{predictions[p['username']]}"
            for p, _ in built
            if p["username"] != persona["username"]
        )
        system = SYSTEM_TEMPLATE.format(display=persona["display"])
        user = (
            "他のAIの予想が出そろいました。議論として応答してください。"
            "最も同意できる点と最も異論のある点を具体的に挙げ、必要なら自分の印の修正案も示してください。\n\n"
            f"【自分の予想】\n{predictions[persona['username']]}\n\n【他のAIの予想】\n{others}"
        )
        text = prov.chat(system, user)
        if not text.strip():
            raise RuntimeError(f"empty debate reply from {persona['username']}")
        debate[persona["username"]] = text
    for persona, _ in built:
        emitter.emit(persona["username"], debate[persona["username"]], root_id)

    # --- MC conclusion ---
    scores: dict[int, dict] = {}
    for persona, _ in built:
        for mark, (num, name) in canonical_marks(
            predictions[persona["username"]], fixture
        ).items():
            entry = scores.setdefault(num, {"name": name, "score": 0, "marks": {}})
            entry["score"] += MARK_SCORE[mark]
            entry["marks"][persona["display"]] = mark
    ranking = sorted(scores.items(), key=lambda kv: -kv[1]["score"])
    lines = [f"【最終結論】第{r['edition']}回 {r['name']}", ""]
    if ranking:
        top_num, top = ranking[0]
        lines.append(f"◎ {top_num} {top['name']}（総意の本命）")
        for num, info in ranking[1:4]:
            lines.append(f"・ {num} {info['name']}（スコア {info['score']}）")
        lines.append("")
        lines.append("各AIの本命：")
        for persona, _ in built:
            m = canonical_marks(predictions[persona["username"]], fixture).get("◎")
            if m:
                lines.append(f"  {persona['display']}: ◎ {m[0]} {m[1]}")
    lines.append("")
    lines.append("議論を踏まえた集約です。結果が出たら回顧します。")
    conclusion = "\n".join(lines)
    bets = {
        p["username"]: extract_bets(predictions[p["username"]]) for p, _ in built
    }
    bets_digest = "\n".join(
        f"{p['display']}: {bets[p['username']].replace(chr(10), ' / ')}"
        for p, _ in built
        if bets[p["username"]]
    )
    if mc_provider is not None and ranking:
        # The local LLM writes the moderator's summary prose; the ranking
        # above stays the deterministic ground truth. Any failure falls
        # back to the ranking-only conclusion.
        try:
            debate_digest = "\n\n".join(
                f"--- {p['display']} の議論 ---\n{debate[p['username']][:400]}"
                for p, _ in built
            )
            prose = mc_provider.chat(
                "あなたは競馬予想の議論スレッドの司会AIです。各AIの議論を読み、"
                "最終結論への総括を書きます。新しい印や順位は作らず、既に決まった"
                "結論の根拠と、議論で浮かんだ注意点だけを述べてください。"
                "最後に、集約としての最終買い目を予算1,000円で1〜3点"
                "（券種・組み合わせ・金額、合計1,000円ちょうど）提案してください。",
                f"レース: 第{r['edition']}回 {r['name']}\n"
                f"最終結論の順位:\n" + "\n".join(lines[2:6]) + "\n\n"
                f"各AIの買い目:\n{bets_digest}\n\n"
                f"各AIの議論（抜粋）:\n{debate_digest}\n\n"
                "総括は500字以内で書いてください。",
            )
            if prose.strip():
                conclusion += "\n\n【司会総括（ローカルLLM）】\n" + prose.strip()
        except Exception as exc:  # noqa: BLE001 - prose is optional garnish
            print(f"[mc prose skipped: {exc}]", file=sys.stderr)
    if bets_digest:
        conclusion += "\n\n【各AIの買い目（予算1,000円）】\n" + bets_digest
    emitter.emit(MC_USERNAME, conclusion, root_id)

    return {
        "race": f"第{r['edition']}回 {r['name']}",
        "predictions": predictions,
        "debate": debate,
        "marks": {
            p["username"]: {
                k: list(v)
                for k, v in canonical_marks(
                    predictions[p["username"]], fixture
                ).items()
            }
            for p, _ in built
        },
        "conclusion_ranking": [
            {"number": num, "name": info["name"], "score": info["score"]}
            for num, info in ranking
        ],
        "bets": bets,
    }


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
    for mark, (num, name) in (state.get("marks", {}).get(username) or {}).items():
        fin = actual.get(num)
        lines.append(
            f"  {mark} {num} {name} → {fin['position']}着" if fin else f"  {mark} {num} {name} → 着順不明"
        )
    bets = (state.get("bets") or {}).get(username)
    if bets:
        lines.append("")
        lines.append(f"【あなたの買い目】\n{bets}")
    lines.append("")
    lines.append("この結果を受けて、あなたの反省を述べてください。")
    return "\n".join(lines)


def run_review(
    fixture: dict,
    results: dict,
    state: dict | None,
    emitter: Emitter,
    built: list | None = None,
) -> None:
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
        for username, marks in (state.get("marks") or {}).items():
            honmei = marks.get("◎")
            if honmei:
                lines.append(
                    f"  @{username} の◎ {honmei[1]} は {actual.get(honmei[0], '?')}着"
                )
    else:
        lines.append("（予想スレッドの記録がないため、結果のみの回顧です）")
    lines.append("次回の予想に活かします。")
    review_text = "\n".join(lines)

    # Per-AI reflections: generate every text BEFORE posting anything, so a
    # provider failure cannot leave a half-finished review thread.
    reflections: list[tuple[str, str]] = []
    if built and state and state.get("marks"):
        for persona, prov in built:
            username = persona["username"]
            if username not in (state.get("marks") or {}):
                continue  # this AI did not predict this race
            try:
                text = prov.chat(
                    REFLECTION_SYSTEM.format(display=persona["display"]),
                    reflection_user_prompt(fixture, results, state, username),
                )
            except Exception as e:  # noqa: BLE001 - skip this AI, keep others
                print(f"[reflection failed for {username}: {e}]", file=sys.stderr)
                continue
            if text and text.strip():
                reflections.append((username, f"【反省】{results['race']}\n\n{text.strip()}"))

    review_id = emitter.emit(MC_USERNAME, review_text, None)
    for username, text in reflections:
        emitter.emit(username, text, review_id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", default="fixtures/2025-elizabeth-cup.json")
    ap.add_argument("--mock", action="store_true", help="force MockProvider for all AIs")
    ap.add_argument("--live", action="store_true", help="actually post to Misskey")
    ap.add_argument("--dry-run", action="store_true", help="print posts instead (default)")
    ap.add_argument("--review", action="store_true", help="post a review using --results")
    ap.add_argument("--results", default=None, help="results JSON (review mode only)")
    ap.add_argument("--force-review", action="store_true",
                    help="post a review even if the state file says one was posted")
    ap.add_argument("--local", action="store_true",
                    help="add the local LLM (Ollama on take-pc) as a fourth predictor "
                         "and let it write the moderator summary prose")
    args = ap.parse_args()

    live = bool(args.live) and not args.dry_run
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    roster = PERSONAS + ([LOCAL_PERSONA] if args.local else [])
    emitter = Emitter(live=live, usernames=[p["username"] for p in roster])

    if args.review:
        if not args.results:
            raise SystemExit("--review requires --results <file>")
        results = json.loads(Path(args.results).read_text(encoding="utf-8"))
        state_path = Path("runs") / (Path(args.fixture).stem + "-state.json")
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.exists()
            else None
        )
        if live and state and state.get("review_posted_at") and not args.force_review:
            raise SystemExit(
                f"review already posted at {state['review_posted_at']} "
                "(live mode prints nothing on success, so check the state "
                "file before re-running; use --force-review to post again)"
            )
        built = build_providers(mock=args.mock, roster=roster)
        preflight(built, live=live)
        run_review(fixture, results, state, emitter, built=built)
        if live and state is not None:
            from datetime import datetime

            state["review_posted_at"] = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )
            state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return 0

    built = build_providers(mock=args.mock, roster=roster)
    preflight(built, live=live)
    mc_provider = None
    if args.local and not args.mock:
        mc_provider = providers.OllamaProvider()
    state = run_prediction(fixture, emitter, built, mc_provider=mc_provider)
    state_path = Path("runs") / (Path(args.fixture).stem + "-state.json")
    state_path.parent.mkdir(exist_ok=True)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[state saved: {state_path}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

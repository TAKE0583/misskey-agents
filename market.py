#!/usr/bin/env python3
"""Market odds snapshot handling (extracted from orchestrator.py, 2026-11-19 split; behavior unchanged)."""
from __future__ import annotations

import re

from common import MARKS, PROB_LINE_RE


# --- Market data (the fixture's odds snapshot) ------------------------------
# build_fixture stamps every horse with a market block {odds, win_prob,
# snapshot_time}: the win odds published at fixture-build time and the
# de-vigged win probability derived from them. Before betting opens
# there are no odds, the block is nulls, and every consumer below must
# degrade to the pre-market behaviour (no prompt note, no annotation,
# no market ballot, no conclusion gate).
#
# A snapshot can also be DISTORTED rather than missing: the odds API
# answers before the market has really formed and stamps unpriced
# runners with its 999.9 sentinel odds. De-vigging normalises over
# what is there, so the unpriced runners' probability mass lands on
# the few priced ones — the 2026 QE Cup fixture (built Friday 19:00
# for a Sunday G1) priced the field at 63.3% / 3.9% for the top two
# while the real market was 23% / 12%, and every consumer downstream
# (market ballot, gate, the moderator's narrative) trusted the
# invented numbers. A distorted snapshot is worse than a missing one,
# so when sentinel-priced runners make up a material share of the
# field the snapshot degrades to "missing" for every consumer, with
# the distortion kept on record (see the conclusion gate).

MARKET_SENTINEL_ODDS = 999.0      # odds at/above this mean "not priced"
MARKET_DISTORT_MIN_COUNT = 2      # at least this many sentinel runners …
MARKET_DISTORT_MIN_RATIO = 0.15   # … and at least this share of the field


def market_distortion(fixture: dict) -> dict | None:
    """Distortion report for the fixture's market snapshot, or None
    when the snapshot is usable. Absence of market data is NOT
    distortion (that is the plain missing case): distortion needs
    priced entries whose odds are the API's not-yet-priced sentinel
    on a material share of the field."""
    horses = fixture.get("horses") or []
    field = len(horses)
    if not field:
        return None
    priced = 0
    sentinel = 0
    for h in horses:
        m = h.get("market") or {}
        if m.get("win_prob") is None:
            continue
        priced += 1
        odds = m.get("odds")
        if odds is not None and float(odds) >= MARKET_SENTINEL_ODDS:
            sentinel += 1
    if not priced:
        return None
    if (sentinel >= MARKET_DISTORT_MIN_COUNT
            and sentinel / field >= MARKET_DISTORT_MIN_RATIO):
        return {
            "sentinel_count": sentinel,
            "field": field,
            "priced": priced,
            "ratio": round(sentinel / field, 3),
        }
    return None


def fixture_market(fixture: dict) -> dict[int, dict]:
    """Usable market entries {number: {"odds":…, "win_prob":…}} — only
    horses whose de-vigged probability is present. A distorted
    snapshot (see market_distortion) yields no usable market at all,
    so every consumer degrades exactly as for a missing one."""
    if market_distortion(fixture) is not None:
        return {}
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

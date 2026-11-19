#!/usr/bin/env python3
"""Aggregate probability vectors and the conclusion gate (extracted from orchestrator.py, 2026-11-19 split; behavior unchanged)."""
from __future__ import annotations

import time

from market import fixture_market, market_distortion

EDGE_MARGIN = 1.15               # aggregate win prob must reach market x this
LONGSHOT_MIN_MARKET_PROB = 0.02  # anchor needs at least this market prob
MARKET_USERNAME = "market"       # pseudo-voter carrying the market ballot


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
                  via the market_missing flag. A distorted snapshot
                  (see market_distortion) also lands here, flagged via
                  market_distorted: its numbers exist but cannot be
                  trusted, which is worse than no data.
    The anchor is the aggregate's top horse among those whose market
    probability is at least LONGSHOT_MIN_MARKET_PROB: a horse the
    market prices under that floor may top the panel's ranking but
    may not anchor the conclusion bet (its number is kept in
    `skipped_longshot` when it was the aggregate's top)."""
    names = {h["number"]: h["name"] for h in fixture["horses"]}
    market = fixture_market(fixture)
    distortion = market_distortion(fixture)
    vector = build_probability_vector(contribs, fixture)
    record: dict = {
        "edge_margin": EDGE_MARGIN,
        "longshot_min_market_prob": LONGSHOT_MIN_MARKET_PROB,
        "market_missing": not market,
        "market_distorted": distortion is not None,
        "aggregate_probs": {
            str(n): round(p, 2) for n, p in sorted(vector.items())
        },
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if distortion is not None:
        record["market_distortion"] = distortion
    if not market:
        record.update({
            "decision": "pass",
            "anchor": None,
            "reason": (
                "市場データが不完全（未確定オッズが多数を占める）のため、"
                "市場を上回ることを検証できず見送り"
                if distortion is not None else
                "市場データがないため、市場を上回ることを検証できず見送り"
            ),
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

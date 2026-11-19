#!/usr/bin/env python3
"""A/B measurement ledger: pre-debate vs post-debate aggregates (extracted from orchestrator.py, 2026-11-19 split; behavior unchanged)."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from gate import build_probability_vector
from market import fixture_market
from runstate import fixture_race_id
from validation import parse_probabilities

AB_JUDGE_AT_RACES = 200          # the debate default is judged at this count
AB_STANDINGS_MIN_RACES = 10      # review notes show A/B standings from here


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

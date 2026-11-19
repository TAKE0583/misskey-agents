#!/usr/bin/env python3
"""Points settlement and expected-value review (extracted from orchestrator.py, 2026-11-19 split; behavior unchanged)."""
from __future__ import annotations

import points as points_mod

from common import MARKS, MC_USERNAME
from runstate import fixture_race_id
from validation import extract_bets, parse_bet_lines, parse_probabilities

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

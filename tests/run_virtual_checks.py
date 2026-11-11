"""Offline checks for the virtual settlement tool (virtual_settle.py).

No network: `python tests/run_virtual_checks.py` must end with
"ALL CHECKS PASSED". Synthetic races pin the counterfactual rules
(gate margins judged from the recorded gate values only, bet shapes,
unpriceable handling, the ungated era); a smoke check then ties the
tool to the real archive: replaying the current rules over every
recorded live race must reproduce the recorded gate decisions and
the recorded settlements exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import virtual_settle as vs  # noqa: E402

fails = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("PASS" if cond else "FAIL"), name, detail if not cond else "")
    if not cond:
        fails.append(name)


# --- synthetic race ---------------------------------------------------------
# 6 runners; market favourite is #1, the panel's anchor is #3.
FIXTURE = {
    "race": "仮想テストレース",
    "horses": [
        {"number": n, "name": f"馬{n}", "bracket": (n + 1) // 2,
         "market": {"odds": o, "win_prob": p}}
        for n, o, p in [
            (1, 2.0, 0.40), (2, 5.0, 0.16), (3, 4.0, 0.20),
            (4, 10.0, 0.08), (5, 20.0, 0.04), (6, 30.0, 0.027),
        ]
    ],
}
# Finish: 3 - 1 - 2 (the anchor wins, the trio 1-2-3 lands).
RESULTS = {
    "finish": [
        {"position": 1, "number": 3, "win_odds": 4.0},
        {"position": 2, "number": 1, "win_odds": 2.0},
        {"position": 3, "number": 2, "win_odds": 5.0},
        {"position": 4, "number": 4, "win_odds": 10.0},
    ],
    "payouts_jpy": {
        "win": {"3": 400},
        "place": {"3": 150, "1": 110, "2": 180},
        "trio": {"1-2-3": 1200},
    },
}
GATE = {
    "decision": "bet", "anchor": 3,
    "aggregate_prob": 30.0, "market_prob": 20.0, "market_missing": False,
    "aggregate_probs": {"1": 25.0, "2": 10.0, "3": 30.0,
                        "4": 5.0, "5": 3.0, "6": 2.0},
}
STATE = {
    "race": "仮想テストレース",
    "conclusion_gate": GATE,
    "conclusion_ranking": [
        {"number": 1, "name": "馬1", "score": 20},
        {"number": 3, "name": "馬3", "score": 18},
        {"number": 2, "name": "馬2", "score": 10},
        {"number": 4, "name": "馬4", "score": 5},
    ],
    # the actual slip: anchored on 3, as the current rules build it
    "final_bet": "単勝 3 500点\n3連複 1-2-3 500点",
    "review": {"points": {"shikai": {"staked": 1000, "returned": 8000,
                                     "balance": 0}}},
}
RACE = {
    "stem": "synthetic", "name": "仮想テストレース",
    "fixture": FIXTURE, "results": RESULTS, "state": STATE,
    "gate": GATE,
    "vector": {int(n): float(p) for n, p in GATE["aggregate_probs"].items()},
    "ranking": [1, 3, 2, 4],
    "winner": 3,
}

# --- decide_recorded ---------------------------------------------------------
check("decide: edge 1.50 bets at 1.15",
      vs.decide_recorded(GATE, 1.15) == (True, 3))
check("decide: boundary 1.50 bets at exactly 1.50",
      vs.decide_recorded(GATE, 1.50) == (True, 3))
check("decide: passes above the edge",
      vs.decide_recorded(GATE, 1.51) == (False, 3))
check("decide: margin 0 (always) bets",
      vs.decide_recorded(GATE, 0.0) == (True, 3))
check("decide: market missing passes even at margin 0",
      vs.decide_recorded({**GATE, "market_missing": True,
                          "anchor": None}, 0.0) == (False, None))
check("decide: ungated record (no anchor) passes",
      vs.decide_recorded({"decision": "ungated", "anchor": None},
                         1.15) == (False, None))

# --- make_bets ---------------------------------------------------------------
order = vs.anchor_order(3, [1, 3, 2, 4])
check("anchor_order puts the anchor first", order == [3, 1, 2, 4])
split = vs.make_bets(order, "split", FIXTURE)
check("split shape", split == [
    {"type": "単勝", "numbers": [3], "amount": 500},
    {"type": "3連複", "numbers": [1, 2, 3], "amount": 500},
], str(split))
check("win shape", vs.make_bets(order, "win", FIXTURE) == [
    {"type": "単勝", "numbers": [3], "amount": 1000}])
check("trio shape", vs.make_bets(order, "trio", FIXTURE) == [
    {"type": "3連複", "numbers": [1, 2, 3], "amount": 1000}])
check("winplace shape", vs.make_bets(order, "winplace", FIXTURE) == [
    {"type": "単勝", "numbers": [3], "amount": 500},
    {"type": "複勝", "numbers": [3], "amount": 500}])

# --- settle_bets -------------------------------------------------------------
settled = vs.settle_bets(split, FIXTURE, RESULTS)
check("settle split: win 400x5 + trio 1200x5",
      settled == {"staked": 1000, "returned": 2000 + 6000}, str(settled))
no_trio = {"finish": RESULTS["finish"],
           "payouts_jpy": {"win": {"3": 400}}}
check("settle: winning leg without a payout table is unpriceable",
      vs.settle_bets(split, FIXTURE, no_trio) is None)
loser = vs.settle_bets(
    [{"type": "単勝", "numbers": [1], "amount": 1000}], FIXTURE, RESULTS)
check("settle: losing bet returns 0, still priceable",
      loser == {"staked": 1000, "returned": 0})

# --- evaluate_race -----------------------------------------------------------
ev = vs.evaluate_race(RACE)
check("eval: gated flag", ev["gated"] is True)
check("eval: current rules bet and return 8000",
      ev["margins"]["1.15"]["bet"] is True
      and ev["margins"]["1.15"]["returned"] == 8000)
check("eval: margin 1.30 still bets (edge 1.50)",
      ev["margins"]["1.30"]["bet"] is True)
check("eval: ranking axis bets the ◎ (#1) and loses the win leg",
      ev["axis_ranking"]["order"][0] == 1
      and ev["axis_ranking"]["returned"] == 6000,
      str(ev["axis_ranking"]))  # trio 1-2-3 still lands: 1200x5
check("eval: win shape doubles the win return",
      ev["shapes"]["win"]["returned"] == 4000)
check("eval: trio shape doubles the trio return",
      ev["shapes"]["trio"]["returned"] == 12000)
check("eval: winplace = win 2000 + place 750",
      ev["shapes"]["winplace"]["returned"] == 2750)
check("eval: actual reproduces the recorded settlement",
      ev["actual"] == {"bet": True, "staked": 1000, "returned": 8000})
check("eval: no initial predictions -> no PRE variant", ev["pre"] is None)

ungated_state = {**STATE, "conclusion_gate": {
    "decision": "ungated", "anchor": None,
    "aggregate_probs": GATE["aggregate_probs"]}}
ungated_race = {**RACE, "state": ungated_state,
                "gate": ungated_state["conclusion_gate"]}
ev_u = vs.evaluate_race(ungated_race)
check("eval: ungated era has no variants but settles its actual bet",
      ev_u["gated"] is False
      and all(v is None for v in ev_u["margins"].values())
      and ev_u["actual"]["returned"] == 8000)

# --- aggregate ---------------------------------------------------------------
agg = vs.aggregate([ev], lambda e: e["margins"]["1.15"])
check("aggregate totals",
      agg["bets"] == 1 and agg["staked"] == 1000
      and agg["returned"] == 8000 and agg["roi"] == 8.0)

# --- smoke: the real archive must replay exactly ------------------------------
races, _excluded = vs.load_races()
if races:
    evals = [vs.evaluate_race(r) for r in races]
    replay_bad = [
        ev["stem"] for ev in evals
        if ev["recorded_gate_decision"] in ("bet", "pass")
        and ("bet" if ev["margins"]["1.15"].get("bet") else "pass")
        != ev["recorded_gate_decision"]
    ]
    check("archive: 1.15 replay reproduces every recorded gate decision",
          not replay_bad, ",".join(replay_bad))
    actual_bad = [
        ev["stem"] for ev in evals if ev["recorded"] is not None
        and (ev["actual"]["staked"] != ev["recorded"].get("staked")
             or ev["actual"]["returned"] != ev["recorded"].get("returned"))
    ]
    check("archive: actual slips reproduce every recorded settlement",
          not actual_bad, ",".join(actual_bad))
else:
    print("SKIP archive smoke checks (no qualifying races found)")

print()
if fails:
    print("FAILED:", ", ".join(fails))
    sys.exit(1)
print("ALL CHECKS PASSED")

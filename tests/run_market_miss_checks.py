"""Offline checks for market_miss.py (market calibration analysis).

No network: `python tests/run_market_miss_checks.py` must end with
"ALL CHECKS PASSED". Lives in its own file (same style as
run_checks.py) because run_checks.py is edited by other work; the
checks here cover only the analysis layer — de-vigging, the Harville
top-3 model, bucket aggregation math, and row building on a real
repo fixture pair.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import market_miss as mm  # noqa: E402

FIXTURES = Path(__file__).parent.parent / "fixtures"
monbetsu = json.loads((FIXTURES / "2026-monbetsu-1105-12r.json").read_text(encoding="utf-8"))
monbetsu_results = json.loads((FIXTURES / "2026-monbetsu-1105-12r-results.json").read_text(encoding="utf-8"))

fails = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("PASS" if cond else "FAIL"), name, detail if not cond else "")
    if not cond:
        fails.append(name)


# --- devig_win_probs ---
probs = mm.devig_win_probs({1: 2.0, 2: 4.0, 3: 4.0})
check("devig: normalised to 1", abs(sum(probs.values()) - 1.0) < 1e-9)
check("devig: proportional to implied", abs(probs[1] - 0.5) < 1e-9 and abs(probs[2] - 0.25) < 1e-9)
p10 = mm.devig_win_probs({1: 1.0, 2: 9.0})
check("devig: odds 1.0 (money-back) kept, not dropped",
      1 in p10 and p10[1] > p10[2])
check("devig: junk odds excluded",
      mm.devig_win_probs({1: 0.0, 2: -3.0, 3: None}) == {})

# --- harville_top3_probs ---
uni = mm.harville_top3_probs({i: 0.25 for i in range(1, 5)})
check("harville: uniform 4-runner top3 = 0.75 each",
      all(abs(v - 0.75) < 1e-9 for v in uni.values()))
dom = mm.harville_top3_probs({1: 0.9, 2: 0.05, 3: 0.05})
check("harville: probabilities sum to the 3 slots",
      abs(sum(dom.values()) - 3.0) < 1e-6)
check("harville: dominant horse almost surely top3", dom[1] > 0.99)

# --- build_rows on a real pair (closing odds preferred) ---
rows, meta = mm.build_rows(monbetsu, monbetsu_results)
check("rows: one per finisher", len(rows) == len(monbetsu_results["finish"]))
check("rows: closing odds labelled", meta.get("odds_source") == "closing")
check("rows: market probs sum to 1 per race",
      abs(sum(r["market_prob"] for r in rows) - 1.0) < 1e-6)
winners = [r for r in rows if r["won"]]
check("rows: exactly the winner flagged", len(winners) == 1 and winners[0]["position"] == 1)
check("rows: top3 flags count 3", sum(1 for r in rows if r["top3"]) == 3)
check("rows: top3 probs sum to 3",
      abs(sum(r["top3_prob"] for r in rows) - 3.0) < 1e-6)
check("rows: jurisdiction classified NAR", meta["jurisdiction"] == "NAR")
check("rows: person stats carried through",
      any(r["jockey_win_rate"] is not None for r in rows))

# --- build_rows: snapshot fallback labelling ---
fx_snap = json.loads(json.dumps(monbetsu))
res_no_odds = json.loads(json.dumps(monbetsu_results))
for f in res_no_odds["finish"]:
    f["win_odds"] = None
for h in fx_snap["horses"]:
    h["market"] = {"odds": 5.0, "win_prob": 1.0 / len(fx_snap["horses"]),
                   "snapshot_time": "2026-11-05T00:00:00+0900"}
rows2, meta2 = mm.build_rows(fx_snap, res_no_odds)
check("rows: snapshot fallback used and labelled",
      meta2.get("odds_source") == "snapshot" and len(rows2) == len(rows))
fx_none = json.loads(json.dumps(fx_snap))
for h in fx_none["horses"]:
    h["market"] = {"odds": None, "win_prob": None, "snapshot_time": None}
rows3, meta3 = mm.build_rows(fx_none, res_no_odds)
check("rows: no market anywhere -> skipped with reason",
      rows3 == [] and "skip" in meta3)

# --- aggregate math on synthetic rows ---
def _row(prob, won, top3p=0.0, top3=False, pop=1):
    return {"market_prob": prob, "won": won, "top3_prob": top3p,
            "top3": top3, "popularity": pop, "jockey_win_rate": None,
            "jockey_top3_rate": None, "trainer_win_rate": None,
            "trainer_top3_rate": None, "jurisdiction": "NAR",
            "venue": "門別競馬場", "field_size": 10, "distance_m": 1800,
            "going": "良", "surface": "ダ"}

syn = [_row(0.5, True), _row(0.3, False), _row(0.2, False)]
table = dict(mm.aggregate(syn, "popularity"))
b = table["1番人気"]
check("aggregate: expected wins = sum of probs", b["exp_wins"] == 1.0)
check("aggregate: actual wins counted", b["act_wins"] == 1)
check("aggregate: ratio 1.0 when calibrated", b["win_ratio"] == 1.0)
check("aggregate: z = 0 when act == exp", b["win_z"] == 0.0)
check("aggregate: small n flagged unreliable", b["unreliable"] is True)
check("aggregate: canonical bucket order",
      [k for k, _ in mm.aggregate(syn + [_row(0.1, False, pop=12)], "popularity")]
      == ["1番人気", "10番人気以下"])

# --- log loss ---
ll = mm.overall_log_loss([_row(0.5, True), _row(0.5, False)])
import math
check("log loss: coin-flip pair = ln2", abs(ll - math.log(2)) < 1e-9)
check("log loss: empty input -> None", mm.overall_log_loss([]) is None)

# --- discovery helpers ---
check("class rank: graded beats A-class beats C-class",
      mm.class_rank("11R 重賞 ダ1600m") > mm.class_rank("11R 特別(A1) ダ1600m")
      > mm.class_rank("5R 3歳以上 (C4) ダ1200m"))
check("pick_main: graded race wins over later low-class race",
      mm.pick_main([("202600010111", "11R (C1) ダ1400m"),
                    ("202600010109", "9R 重賞 ダ1600m")])[0] == "202600010109")

# --- bucket log-loss + whole-race partitions ---
check("aggregate: bucket carries mean log-loss",
      isinstance(b.get("log_loss"), float) and b["log_loss"] > 0)
race_a = [_row(0.6, True), _row(0.4, False)]
race_b = [_row(0.5, False), _row(0.5, True)]
jur = dict(mm.aggregate(race_a + race_b, "jurisdiction"))["NAR"]
check("aggregate: whole-race partition has exp == act by construction",
      jur["exp_wins"] == jur["act_wins"] == 2.0)

# --- race-list parsing: JRA header + grade icon (live layout) ---
JRA_HTML = """
<dl class="RaceList_DataList"><dt>4回 東京 2日目 払戻一覧</dt>
<dd><ul>
<li class="RaceList_DataItem"><a href="../race/shutuba.html?race_id=202605040212">12R 3歳以上2勝クラス 16:30 ダ1600m 16頭</a></li>
<li class="RaceList_DataItem"><span class="Icon_GradeType2"></span><a href="../race/shutuba.html?race_id=202605040211">11R 毎日王冠 15:45 芝1800m 17頭</a></li>
</ul></dd></dl>"""
parsed = mm.parse_race_list(JRA_HTML)
check("race list: JRA 'N回 venue N日目' header parsed", list(parsed) == ["東京"])
check("race list: grade icon makes 11R the main, not the later 12R",
      mm.pick_main(parsed["東京"])[0] == "202605040211")

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED: {fails}")
    raise SystemExit(1)
print("ALL CHECKS PASSED")

"""Offline checks for win5.py (WIN5 tickets for the conclusion account).

No network, no posting, no ledger writes: the JRA racelist page is the
real page saved on 2026-11-08 (tests/fixtures/), leg fixtures/states
are cloned from the real 2026-11-08 Tokyo/Kyoto/Fukushima runs into
temp directories, and settlement results are injected.
`python tests/run_win5_checks.py` must end with "ALL CHECKS PASSED".
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import win5  # noqa: E402

ROOT = Path(__file__).parent.parent
RACELIST_HTML = (
    Path(__file__).parent / "fixtures" / "jra-win5-racelist-2026-11-08.html"
).read_bytes().decode("cp932", errors="replace")

fails = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("PASS" if cond else "FAIL"), name, detail if not cond else "")
    if not cond:
        fails.append(name)


# --- parse_yen ------------------------------------------------------
check("yen: man format", win5.parse_yen("713万2610円") == 7_132_610)
check("yen: comma format", win5.parse_yen("18,586,810円") == 18_586_810)
check("yen: oku format", win5.parse_yen("1億2345万6789円") == 123_456_789)
check("yen: plain", win5.parse_yen("8,540円") == 8_540)
check("yen: zero", win5.parse_yen("0円") == 0)

# --- racelist parser -------------------------------------------------
d = datetime(2026, 11, 8).date()
info = win5.parse_racelist(RACELIST_HTML, d)
check("racelist: 2026-11-08 found", info is not None)
check("racelist: deadline 14:50", info["deadline"] == "14:50",
      str(info["deadline"]))
check(
    "racelist: legs 2026-11-08",
    [(l["venue"], l["race_number"]) for l in info["legs"]]
    == [("東京", 10), ("京都", 10), ("福島", 11), ("東京", 11), ("京都", 11)],
    str(info["legs"]),
)
arima = win5.parse_racelist(RACELIST_HTML, datetime(2026, 12, 27).date())
check(
    "racelist: Arima day legs",
    [(l["venue"], l["race_number"]) for l in arima["legs"]]
    == [("中山", 9), ("阪神", 10), ("中山", 10), ("阪神", 11), ("中山", 11)],
    str(arima["legs"]),
)
check("racelist: Arima deadline 14:10", arima["deadline"] == "14:10")
check(
    "racelist: unlisted date -> None",
    win5.parse_racelist(RACELIST_HTML, datetime(2026, 11, 4).date()) is None,
)
# 2026-11-30 IS listed by JRA (Monday make-up day) but the user's
# scope is weekends only: win5_day must refuse it before parsing.
check(
    "win5_day: Monday refused despite listing",
    win5.win5_day(datetime(2026, 11, 30).date(), RACELIST_HTML) is None,
)
check(
    "win5_day: Sunday accepted",
    win5.win5_day(d, RACELIST_HTML) is not None,
)

# --- netkeiba result parser ------------------------------------------
RESULT_HTML = """
<table class="RaceCommon_Table Win5_Table"><tbody>
<tr><th>的中馬番</th><td><ul>
<li class="w5 Waku1">3</li><li class="w5 Waku2">5</li>
<li class="w5 Waku3">7</li><li class="w5 Waku4">2</li>
<li class="w5 Waku5">11</li></ul></td></tr>
<tr><th>払戻金</th><td>713万2610円</td></tr>
<tr><th>的中票数</th><td>80票</td></tr>
</tbody></table>
"""
res = win5.parse_win5_result(RESULT_HTML)
check("result: winners", res["winners"] == [3, 5, 7, 2, 11], str(res))
check("result: payout", res["payout_yen"] == 7_132_610)
check("result: tickets", res["tickets"] == 80)
check("result: not estimated", res["estimated"] is False)
est = win5.parse_win5_result(RESULT_HTML.replace("払戻金", "想定払い戻し"))
check("result: estimated flagged", est["estimated"] is True)

# --- allocator ---------------------------------------------------------
VEC = {1: 40.0, 2: 25.0, 3: 15.0, 4: 10.0, 5: 6.0, 6: 4.0}
vecs = [dict(VEC) for _ in range(5)]
alloc = win5.allocate_win5(vecs, 2000)
check("alloc: within budget", alloc["stake"] <= 2000, str(alloc["stake"]))
check("alloc: stake == combos*100",
      alloc["stake"] == alloc["combinations"] * 100)
check("alloc: every leg >= 1 pick",
      all(len(p) >= 1 for p in alloc["picks"]))
check("alloc: per-leg cap 4", all(len(p) <= 4 for p in alloc["picks"]))
check("alloc: top horse always picked",
      all(p[0] == 1 or 1 in p for p in alloc["picks"]))
import math  # noqa: E402

expect_hit = 1.0
for cov in alloc["coverage_pct"]:
    expect_hit *= cov / 100.0
check("alloc: hit prob = product of coverage",
      abs(alloc["hit_prob_pct"] - round(expect_hit * 100, 4)) < 0.01,
      f"{alloc['hit_prob_pct']} vs {expect_hit * 100}")
alloc_small = win5.allocate_win5(vecs, 300)
check("alloc: small budget respected",
      alloc_small["combinations"] <= 3, str(alloc_small["combinations"]))
check("alloc: budget < 100 -> None", win5.allocate_win5(vecs, 50) is None)
alloc_again = win5.allocate_win5(vecs, 2000)
check("alloc: deterministic", alloc_again == alloc)


# --- leg resolution + run/settle flows in a temp workspace -------------
def clone_leg(tmp: Path, src_fixture: str, src_state: str,
              fixture_stem: str, state_stem: str,
              race_number: int | None = None,
              race_id: str | None = None) -> None:
    fx = json.loads((ROOT / "fixtures" / src_fixture).read_text("utf-8"))
    st = json.loads((ROOT / "runs" / src_state).read_text("utf-8"))
    if race_number is not None:
        fx["race"]["race_number"] = race_number
    if race_id is not None:
        fx["race"]["race_id"] = race_id
        st["race_id"] = race_id
    (tmp / "fixtures").mkdir(parents=True, exist_ok=True)
    (tmp / "runs").mkdir(parents=True, exist_ok=True)
    (tmp / "fixtures" / f"{fixture_stem}.json").write_text(
        json.dumps(fx, ensure_ascii=False), encoding="utf-8")
    (tmp / "runs" / f"{state_stem}-state.json").write_text(
        json.dumps(st, ensure_ascii=False), encoding="utf-8")


def base_workspace(tmp: Path) -> None:
    # The three real 11R legs of 2026-11-08. The Kyoto pair is written
    # under the QE-Cup-style stem on purpose: resolution must work by
    # fixture content + race_id, never by filename stem.
    clone_leg(tmp, "2026-tokyo-1108-11r.json", "2026-tokyo-1108-11r-state.json",
              "2026-tokyo-1108-11r", "2026-tokyo-1108-11r")
    clone_leg(tmp, "2026-kyoto-1108-11r.json", "2026-kyoto-1108-11r-state.json",
              "2026-elizabeth-cup", "2026-elizabeth-cup")
    clone_leg(tmp, "2026-fukushima-1108-11r.json",
              "2026-fukushima-1108-11r-state.json",
              "2026-fukushima-1108-11r", "2026-fukushima-1108-11r")


def add_10r_legs(tmp: Path) -> None:
    clone_leg(tmp, "2026-tokyo-1108-11r.json", "2026-tokyo-1108-11r-state.json",
              "2026-tokyo-1108-10r", "2026-tokyo-1108-10r",
              race_number=10, race_id="202605050210")
    clone_leg(tmp, "2026-kyoto-1108-11r.json", "2026-kyoto-1108-11r-state.json",
              "2026-kyoto-1108-10r", "2026-kyoto-1108-10r",
              race_number=10, race_id="202608050210")


JST = win5.JST
BEFORE = datetime(2026, 11, 8, 13, 0, tzinfo=JST)
AFTER = datetime(2026, 11, 8, 15, 0, tzinfo=JST)
cwd = os.getcwd()

# legs_incomplete: only the three 11R legs exist
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    base_workspace(tmp)
    legs = win5.resolve_legs("2026-11-08", info["legs"],
                             tmp / "fixtures", tmp / "runs")
    ok = [win5.leg_label(l) for l in legs if l["status"] == "ok"]
    ng = [win5.leg_label(l) for l in legs if l["status"] != "ok"]
    check("resolve: 3 legs ok by race_id (stem differs for Kyoto)",
          ok == ["福島11R", "東京11R", "京都11R"], str(ok))
    check("resolve: 10R legs missing",
          ng == ["東京10R", "京都10R"], str(ng))
    os.chdir(tmp)
    try:
        data = win5.run_win5("2026-11-08", False, html=RACELIST_HTML,
                             now=BEFORE, legs_runs_dir=tmp / "runs",
                             fixtures_dir=tmp / "fixtures")
    finally:
        os.chdir(cwd)
    check("run: legs_incomplete stage", data["stage"] == "legs_incomplete",
          data["stage"])
    check("run: missing named exactly",
          data["missing_legs"] == ["東京10R", "京都10R"],
          str(data["missing_legs"]))
    check("run: no ticket created", data["ticket"] is None)
    check("run: state written under runs/dryrun",
          (tmp / "runs" / "dryrun" / "win5-2026-11-08-state.json").exists())

# missed deadline: all 5 legs usable, but the clock is past 14:50
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    base_workspace(tmp)
    add_10r_legs(tmp)
    os.chdir(tmp)
    try:
        data = win5.run_win5("2026-11-08", False, html=RACELIST_HTML,
                             now=AFTER, legs_runs_dir=tmp / "runs",
                             fixtures_dir=tmp / "fixtures")
    finally:
        os.chdir(cwd)
    check("run: missed_deadline stage", data["stage"] == "missed_deadline",
          data["stage"])
    check("run: missed deadline -> no ticket", data["ticket"] is None)

# full generation before the deadline, then settlement math (dry-run)
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    base_workspace(tmp)
    add_10r_legs(tmp)
    os.chdir(tmp)
    try:
        data = win5.run_win5("2026-11-08", False, html=RACELIST_HTML,
                             now=BEFORE, legs_runs_dir=tmp / "runs",
                             fixtures_dir=tmp / "fixtures")
        check("run: ticket generated", data["stage"] == "ticket_generated",
              data["stage"])
        ticket = data["ticket"]
        check("run: stake within 2000 budget", ticket["stake"] <= 2000,
              str(ticket["stake"]))
        check("run: every leg has picks",
              all(leg["picks"] for leg in ticket["legs"]))
        data2 = win5.run_win5("2026-11-08", False, html=RACELIST_HTML,
                              now=BEFORE, legs_runs_dir=tmp / "runs",
                              fixtures_dir=tmp / "fixtures")
        check("run: idempotent re-entry keeps the same ticket",
              data2["ticket"]["generated_at"] == ticket["generated_at"])

        winners = [leg["picks"][0]["number"] for leg in ticket["legs"]]
        win_result = {"winners": winners, "payout_yen": 7_132_610,
                      "tickets": 80, "estimated": False,
                      "leg_race_ids": [], "url": "test"}
        rec = win5.settle_win5("2026-11-08", False, result=win_result,
                               fixtures_dir=tmp / "fixtures")
        check("settle: hit pays the official payout",
              rec["hit"] and rec["returned"] == 7_132_610, str(rec))
        check("settle: staked equals ticket stake",
              rec["staked"] == ticket["stake"])

        lose_winners = list(winners)
        all_picks = {p["number"] for leg in ticket["legs"]
                     for p in leg["picks"]}
        lose_winners[0] = next(
            n for n in range(1, 19) if n not in {
                p["number"] for p in ticket["legs"][0]["picks"]})
        lose_result = dict(win_result, winners=lose_winners)
        rec2 = win5.settle_win5("2026-11-08", False, result=lose_result,
                                fixtures_dir=tmp / "fixtures")
        check("settle: a missed leg pays 0",
              not rec2["hit"] and rec2["returned"] == 0, str(rec2))

        try:
            win5.settle_win5(
                "2026-11-08", False,
                result=dict(win_result, payout_yen=None),
                fixtures_dir=tmp / "fixtures")
            check("settle: hit without payout fails loudly", False)
        except RuntimeError:
            check("settle: hit without payout fails loudly", True)
        try:
            win5.settle_win5(
                "2026-11-08", False,
                result=dict(win_result, estimated=True),
                fixtures_dir=tmp / "fixtures")
            check("settle: estimated (not final) result refused", False)
        except RuntimeError:
            check("settle: estimated (not final) result refused", True)

        # results-fixture cross-check: a fixture winner that disagrees
        # with the WIN5 page must refuse settlement
        rid = ticket["legs"][0]["race_id"]
        bad_fx = {"race_id": rid, "finish": [
            {"position": 1, "number": lose_winners[0], "name": "ダミー"}]}
        (tmp / "fixtures" / "dummy-results.json").write_text(
            json.dumps(bad_fx, ensure_ascii=False), encoding="utf-8")
        try:
            win5.settle_win5("2026-11-08", False, result=win_result,
                             fixtures_dir=tmp / "fixtures")
            check("settle: fixture/page winner mismatch refused", False)
        except RuntimeError:
            check("settle: fixture/page winner mismatch refused", True)
    finally:
        os.chdir(cwd)

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED: {fails}")
    sys.exit(1)
print("ALL CHECKS PASSED")

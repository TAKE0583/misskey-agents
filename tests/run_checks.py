"""Offline checks for the orchestrator's validation / state logic.

No network, no providers: `python tests/run_checks.py` must end with
"ALL CHECKS PASSED". Covers the regression cases from the 2026-11-04
external review (findings 1, 3, 4, 6, 8, 11).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import orchestrator as o  # noqa: E402
import providers  # noqa: E402

FIXTURES = Path(__file__).parent.parent / "fixtures"
monbetsu = json.loads((FIXTURES / "2026-monbetsu-1104-6r.json").read_text(encoding="utf-8"))
monbetsu_results = json.loads((FIXTURES / "2026-monbetsu-1104-6r-results.json").read_text(encoding="utf-8"))
jbc = json.loads((FIXTURES / "2026-jbc-classic.json").read_text(encoding="utf-8"))
jbc_results = json.loads((FIXTURES / "2026-jbc-classic-results.json").read_text(encoding="utf-8"))

fails = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("PASS" if cond else "FAIL"), name, detail if not cond else "")
    if not cond:
        fails.append(name)


# --- validate_marks (finding 4: #99 must not pass) ---
good = {"◎": (10, "タイムエッジ"), "○": (4, "ダムドローズ"), "▲": (1, "エリオラ"), "△": (2, "ラップドアップ")}
check("marks: valid set passes", o.validate_marks(good, monbetsu) == [])
bad99 = dict(good)
bad99["◎"] = (99, "ゴースト")
check("marks: non-existent #99 rejected", o.validate_marks(bad99, monbetsu) != [])
dup = {"◎": (10, "a"), "○": (10, "a"), "▲": (1, "b"), "△": (2, "c")}
check("marks: duplicate horse rejected", o.validate_marks(dup, monbetsu) != [])
missing = {"◎": (10, "a"), "○": (4, "b"), "▲": (1, "c")}
check("marks: missing mark rejected", o.validate_marks(missing, monbetsu) != [])

# --- validate_bets (finding 6) ---
good_bets = [
    {"type": "単勝", "numbers": [10], "amount": 400},
    {"type": "3連複", "numbers": [1, 4, 10], "amount": 600},
]
check("bets: valid set passes", o.validate_bets(good_bets, monbetsu) == [])
check("bets: unknown type rejected",
      o.validate_bets([{"type": "WIN5", "numbers": [1], "amount": 1000}], monbetsu) != [])
check("bets: wrong count rejected",
      o.validate_bets([{"type": "馬連", "numbers": [1, 2, 3], "amount": 1000}], monbetsu) != [])
check("bets: duplicate in combo rejected",
      o.validate_bets([{"type": "3連複", "numbers": [1, 1, 2], "amount": 1000}], monbetsu) != [])
check("bets: non-existent horse rejected",
      o.validate_bets([{"type": "単勝", "numbers": [99], "amount": 1000}], monbetsu) != [])
check("bets: non-100-unit amount rejected",
      o.validate_bets([{"type": "単勝", "numbers": [1], "amount": 150},
                       {"type": "単勝", "numbers": [2], "amount": 850}], monbetsu) != [])
check("bets: total != 1000 rejected",
      o.validate_bets([{"type": "単勝", "numbers": [1], "amount": 900}], monbetsu) != [])
neg = [{"type": "単勝", "numbers": [1], "amount": -500},
       {"type": "単勝", "numbers": [2], "amount": 1500}]
check("bets: negative amount rejected even if total is 1000",
      o.validate_bets(neg, monbetsu) != [])

# --- parse_bet_lines ---
bets, errs = o.parse_bet_lines("単勝 7 500円\n3連複 1-7-11 500円")
check("bet lines: canonical parse", errs == [] and len(bets) == 2 and bets[1]["numbers"] == [1, 7, 11])
_b, errs = o.parse_bet_lines("単勝 7 五百円")
check("bet lines: unparseable line is an error", errs != [])

# --- prediction_errors end to end ---
text_ok = (
    "◎ 10 タイムエッジ 勝率30% 3着内60% — 本命。\n○ 4 ダムドローズ 勝率22% 3着内50% — 対抗。\n"
    "▲ 1 エリオラ 勝率15% 3着内40% — 単穴。\n△ 2 ラップドアップ 勝率10% 3着内30% — 連下。\n\n"
    "【買い目】\n単勝 10 400点\n3連複 1-4-10 600点"
)
check("prediction: valid text passes", o.prediction_errors(text_ok, monbetsu) == [])
text_99 = text_ok.replace("10 タイムエッジ", "99 ゴースト").replace("単勝 10", "単勝 99").replace("1-4-10", "1-4-99")
check("prediction: #99 everywhere rejected", o.prediction_errors(text_99, monbetsu) != [])

# --- MockProvider: distinct picks + ballot (finding 11) ---
for n in (5, 8, 12, 16, 18):
    horses = [(i + 1, "テスト") for i in range(n)]
    m = providers.MockProvider("モック")
    picks = m._distinct_picks(horses, 3)
    check(f"mock: 4 distinct picks at n={n}", len({p[0] for p in picks}) == 4)
mock = providers.MockProvider("ChatGPT")
body = o.race_text(monbetsu)
pred = mock.chat("sys", f"次のレースの予想をしてください。\n\n{body}")
check("mock: prediction validates", o.prediction_errors(pred, monbetsu) == [],
      str(o.prediction_errors(pred, monbetsu)))
debate = mock.chat(
    "sys",
    "他のAIの予想が出そろいました。議論として応答してください。\n\n"
    f"【元の出走データ】\n{body}\n\n【自分の予想】\n{pred}\n\n【他のAIの予想】\nなし",
)
_marks, berrs = o._ballot_errors(debate, monbetsu)
check("mock: debate carries a valid final ballot", berrs == [], str(berrs))

# --- race identity (finding 8) ---
try:
    o.check_race_identity(monbetsu, monbetsu_results, None)
    check("identity: matching race_id passes", True)
except SystemExit:
    check("identity: matching race_id passes", False)
try:
    o.check_race_identity(monbetsu, jbc_results, None)
    check("identity: different race rejected", False)
except SystemExit:
    check("identity: different race rejected", True)
legacy_fixture = json.loads(json.dumps(monbetsu))
legacy_fixture["race"].pop("race_id")
legacy_results = json.loads(json.dumps(monbetsu_results))
legacy_results.pop("race_id")
try:
    o.check_race_identity(legacy_fixture, legacy_results, None)
    check("identity: legacy date+venue fallback passes", True)
except SystemExit:
    check("identity: legacy date+venue fallback passes", False)
legacy_results_bad = json.loads(json.dumps(jbc_results))
legacy_results_bad.pop("race_id")
try:
    o.check_race_identity(legacy_fixture, legacy_results_bad, None)
    check("identity: legacy mismatch rejected", False)
except SystemExit:
    check("identity: legacy mismatch rejected", True)

# --- final bet construction (finding 6) ---
ranking = [(10, {"name": "a", "score": 9}), (4, {"name": "b", "score": 7}),
           (1, {"name": "c", "score": 5}), (2, {"name": "d", "score": 3})]
fb = o.build_final_bet(ranking, monbetsu)
check("final bet: valid by construction", fb is not None and o.validate_bets(fb, monbetsu) == [])
fb1 = o.build_final_bet(ranking[:1], monbetsu)
check("final bet: single-horse fallback", fb1 is not None and o.validate_bets(fb1, monbetsu) == [])

# --- state separation (finding 1) ---
live_p = o.state_path_for("fixtures/x.json", True)
dry_p = o.state_path_for("fixtures/x.json", False)
check("state: live and dry-run paths differ", live_p != dry_p and dry_p.parent.name == "dryrun")

# --- client URL policy ---
try:
    from misskey_client import MisskeyClient
    MisskeyClient("http://example.com:3001")
    check("client: plain HTTP to non-loopback refused", False)
except ValueError:
    check("client: plain HTTP to non-loopback refused", True)
MisskeyClient("http://127.0.0.1:3001")
check("client: loopback HTTP allowed", True)


# --- structured local path: zero-error adoption (finding 5) ---
class StubJSON:
    """chat_json stub: first call returns an invalid object (#99), the
    error-feedback retry returns a valid one."""
    def __init__(self, always_bad=False):
        self.calls = 0
        self.always_bad = always_bad

    def chat_json(self, system, user, schema, model=None, max_tokens=None):
        self.calls += 1
        if self.always_bad or self.calls == 1:
            return {"marks": [{"mark": m, "number": 99, "reason": "x"} for m in ("◎", "○", "▲", "△")],
                    "bets": [{"type": "単勝", "combo": "99", "amount": 1000}], "comment": ""}
        return {"marks": [{"mark": "◎", "number": 10, "win_pct": 30, "top3_pct": 60, "reason": "本命"},
                          {"mark": "○", "number": 4, "win_pct": 22, "top3_pct": 50, "reason": "対抗"},
                          {"mark": "▲", "number": 1, "win_pct": 15, "top3_pct": 40, "reason": "単穴"},
                          {"mark": "△", "number": 2, "win_pct": 10, "top3_pct": 30, "reason": "連下"}],
                "bets": [{"type": "単勝", "combo": "10", "amount": 1000}], "comment": ""}


stub = StubJSON()
rendered = o.generate_structured_prediction(stub, monbetsu)
check("structured: retry with zero errors is adopted and rendered",
      "◎ 10 タイムエッジ" in rendered and "99" not in rendered)
try:
    o.generate_structured_prediction(StubJSON(always_bad=True), monbetsu)
    check("structured: persistently invalid output raises", False)
except RuntimeError:
    check("structured: persistently invalid output raises", True)

# --- Re-review (80aafaa) regression cases --------------------------------
import os as _os
import tempfile

# #5: 枠連 is validated by bracket, not horse number
bracket_counts = {}
for h in monbetsu["horses"]:
    bracket_counts[h["bracket"]] = bracket_counts.get(h["bracket"], 0) + 1
multi = next(b for b, c in bracket_counts.items() if c >= 2)
single = next(b for b, c in bracket_counts.items() if c == 1)
check("wakuren: non-existent bracket rejected",
      o.validate_bets([{"type": "枠連", "numbers": [1, 12], "amount": 1000}], monbetsu) != [])
check("wakuren: same-bracket pair valid when bracket has 2+ horses",
      o.validate_bets([{"type": "枠連", "numbers": [multi, multi], "amount": 1000}], monbetsu) == [])
check("wakuren: same-bracket pair rejected when bracket has 1 horse",
      o.validate_bets([{"type": "枠連", "numbers": [single, single], "amount": 1000}], monbetsu) != [])
check("wakuren: two distinct brackets valid",
      o.validate_bets([{"type": "枠連", "numbers": [multi, single], "amount": 1000}], monbetsu) == [])

# #1: the local AI's published ballot must equal the aggregated ballot
class _StructuredStub:
    name = "stub"

    def chat(self, system, user, model=None, max_tokens=None):
        return (
            "議論します。他AIの見解は参考になりますが、私は押し切ります。\n\n"
            "【最終票】\n◎ 10 タイムエッジ\n○ 4 ダムドローズ\n▲ 1 エリオラ\n△ 2 ラップドアップ"
        )

    def chat_json(self, system, user, schema, model=None, max_tokens=None):
        if "bets" in (schema.get("required") or []):
            return {"marks": [{"mark": "◎", "number": 10, "win_pct": 30, "top3_pct": 60, "reason": "本命"},
                              {"mark": "○", "number": 4, "win_pct": 22, "top3_pct": 50, "reason": "対抗"},
                              {"mark": "▲", "number": 1, "win_pct": 15, "top3_pct": 40, "reason": "単穴"},
                              {"mark": "△", "number": 2, "win_pct": 10, "top3_pct": 30, "reason": "連下"}],
                    "bets": [{"type": "単勝", "combo": "10", "amount": 1000}],
                    "comment": ""}
        return {"marks": [{"mark": "◎", "number": 2, "reason": "討論で変更"},
                          {"mark": "○", "number": 10, "reason": "対抗"},
                          {"mark": "▲", "number": 4, "reason": "単穴"},
                          {"mark": "△", "number": 1, "reason": "連下"}],
                "comment": ""}


_local_persona = {"username": "local", "display": "ローカル", "structured": True}
_gen = o._generate_prediction_texts(monbetsu, [(_local_persona, _StructuredStub())])
_debate_text = _gen["texts"]["debate"]["local"]
_ballot_idx = _debate_text.rfind("【最終票】")
_published = o.parse_marks(_debate_text[_ballot_idx:])
check("local ballot: published debate ballot equals aggregated ballot",
      _published.get("◎", (None,))[0] == 2
      and _gen["final_marks"]["local"]["◎"][0] == 2)

# #2: legacy v1 state migration
with tempfile.TemporaryDirectory() as td:
    legacy_path = Path(td) / "legacy-state.json"
    legacy_path.write_text(json.dumps({
        "race": "第1回 2歳 未勝利",
        "predictions": {"chatgpt": "…"},
        "marks": {"chatgpt": {"◎": [10, "タイムエッジ"]}},
        "conclusion_ranking": [{"number": 10, "name": "タイムエッジ", "score": 9}],
        "review_posted_at": "2026-11-04T17:10:00+09:00",
    }, ensure_ascii=False), encoding="utf-8")
    store = o.RunState.open(legacy_path, monbetsu, "live")
    check("legacy state: migrated to version 2 and marked complete",
          store.data.get("version") == 2 and store.data.get("stage") == "complete")
    check("legacy state: old review timestamp becomes a legacy review record",
          (store.data.get("review") or {}).get("legacy_posted_at") is not None
          and (store.data.get("review") or {}).get("note_id") is None)
    try:
        o.run_review(monbetsu, monbetsu_results, store, o.Emitter(live=False), built=[])
        check("legacy state: review refuses to repost", False)
    except SystemExit:
        check("legacy state: review refuses to repost", True)

# #3: legacy identity must distinguish same-day same-venue races
other_results = json.loads(json.dumps(monbetsu_results))
other_results.pop("race_id")
other_results["race"] = "第3回 別のレース（条件）"
for f in other_results["finish"]:
    f["name"] = "ベツウマ"
try:
    o.check_race_identity(legacy_fixture, other_results, None)
    check("identity: same-day same-venue different race rejected", False)
except SystemExit:
    check("identity: same-day same-venue different race rejected", True)
try:
    o.check_race_identity(monbetsu, None, {"race": "第99回 別レース"})
    check("identity: legacy state for another race rejected", False)
except SystemExit:
    check("identity: legacy state for another race rejected", True)

# #4: generation cache — a mid-generation failure must not re-bill successes
class _CountingMock(providers.MockProvider):
    def __init__(self, display):
        super().__init__(display)
        self.calls = 0
        self.users = []

    def chat(self, system, user, model=None, max_tokens=None):
        self.calls += 1
        self.users.append(user)
        return super().chat(system, user, model=model, max_tokens=max_tokens)


class _FlakyOnce:
    name = "flaky"

    def __init__(self, display):
        self.inner = _CountingMock(display)
        self.failed = False

    def chat(self, system, user, model=None, max_tokens=None):
        if not self.failed:
            self.failed = True
            raise RuntimeError("simulated outage during prediction")
        return self.inner.chat(system, user, model=model, max_tokens=max_tokens)


prov_a = _CountingMock(o.PERSONAS[0]["display"])
prov_b = _FlakyOnce(o.PERSONAS[1]["display"])
built2 = [(o.PERSONAS[0], prov_a), (o.PERSONAS[1], prov_b)]
cache = {"predictions": {}, "debate": {}, "final_marks": {}}
try:
    o._generate_prediction_texts(monbetsu, built2, cache=cache)
    check("gen cache: first run fails on provider B", False)
except RuntimeError:
    check("gen cache: first run fails on provider B", True)
a_calls_after_fail = prov_a.calls
cache = json.loads(json.dumps(cache))  # simulate the state save/load cycle
o._generate_prediction_texts(monbetsu, built2, cache=cache)
# Since 2026-11-05 the mock's rotated debate ballot can contradict its
# prediction bets, which legitimately costs A bets-regeneration calls;
# the invariant this check guards is narrower: A's PREDICTION prompt
# was sent exactly once across both runs (the cached text was reused).
check("gen cache: provider A not re-billed for its cached prediction",
      sum(1 for u in prov_a.users if "次のレースの予想をしてください" in u) == 1
      and prov_a.calls > a_calls_after_fail)  # debate (+maybe regen) did run

# #6: flock-based race lock excludes a second holder, then frees
_cwd = _os.getcwd()
with tempfile.TemporaryDirectory() as td:
    _os.chdir(td)
    try:
        lock1 = o.RaceLock("some-fixture.json")
        lock1.__enter__()
        try:
            o.RaceLock("some-fixture.json").__enter__()
            check("lock: second holder refused", False)
        except SystemExit:
            check("lock: second holder refused", True)
        lock1.__exit__()
        lock3 = o.RaceLock("some-fixture.json")
        lock3.__enter__()
        lock3.__exit__()
        check("lock: re-acquirable after release", True)
    finally:
        _os.chdir(_cwd)

# --- Probabilities + points (2026-11-05 feature) -------------------------
import points as points_mod  # noqa: E402

_marks_ok = o.parse_marks(text_ok)
check("probs: parse from mark lines",
      o.parse_probabilities(text_ok).get("◎") == (10, 30, 60))
check("probs: valid set passes",
      o.validate_probabilities(o.parse_probabilities(text_ok), _marks_ok) == [])
_over = text_ok.replace("勝率30%", "勝率80%")
check("probs: win sum over 100 rejected",
      o.validate_probabilities(o.parse_probabilities(_over), o.parse_marks(_over)) != [])
_missing = text_ok.replace(" 勝率10% 3着内30%", "")
check("probs: missing estimate rejected",
      o.validate_probabilities(o.parse_probabilities(_missing), o.parse_marks(_missing)) != [])
_inverted = text_ok.replace("勝率30% 3着内60%", "勝率60% 3着内30%")
check("probs: top3 below win rejected",
      o.validate_probabilities(o.parse_probabilities(_inverted), o.parse_marks(_inverted)) != [])
check("probs: prediction without probabilities rejected",
      o.prediction_errors(text_ok.replace(" 勝率30% 3着内60%", ""), monbetsu) != [])
check("fair odds: 25% -> 4.0", abs(o.fair_odds(25) - 4.0) < 1e-9)

# bets under a points cap: any total up to the cap, no fixed 1,000
check("bets: under-cap total passes",
      o.validate_bets([{"type": "単勝", "numbers": [1], "amount": 700}],
                      monbetsu, max_total=3000) == [])
check("bets: over-cap total rejected",
      o.validate_bets([{"type": "単勝", "numbers": [1], "amount": 3100}],
                      monbetsu, max_total=3000) != [])
check("bets: prediction with cap accepts non-1000 total",
      o.prediction_errors(text_ok, monbetsu, max_total=3000) == [])

fb600 = o.build_final_bet(ranking, monbetsu, total=600)
check("final bet: total=600 splits evenly and validates",
      fb600 is not None and sum(b["amount"] for b in fb600) == 600
      and o.validate_bets(fb600, monbetsu, expected_total=600) == [])

# --- settlement math (synthetic full payout table over the Monbetsu finish:
# 1st=2, 2nd=4, 3rd=1) ---
synth_results = json.loads(json.dumps(monbetsu_results))
_brackets = {h["number"]: h["bracket"] for h in monbetsu["horses"]}
_win_pair = sorted([_brackets[2], _brackets[4]])
synth_results["payouts_jpy"] = {
    "win": {"2": 290},
    "place": {"2": 120, "4": 160, "1": 180},
    "bracket": {"-".join(map(str, _win_pair)): 670},
    "quinella": {"2-4": 690},
    "wide": {"2-4": 280, "1-2": 290, "1-4": 470},
    "exacta": {"2-4": 1250},
    "trio": {"1-2-4": 1170},
    "trifecta": {"2-4-1": 5020},
}
_b = lambda t, nums, amt: {"type": t, "numbers": nums, "amount": amt}  # noqa: E731
check("settle: winning win-bet pays per 100",
      o.bet_return(_b("単勝", [2], 200), monbetsu, synth_results) == 580)
check("settle: losing win-bet pays 0",
      o.bet_return(_b("単勝", [10], 200), monbetsu, synth_results) == 0)
check("settle: exacta is order-sensitive",
      o.bet_return(_b("馬単", [2, 4], 100), monbetsu, synth_results) == 1250
      and o.bet_return(_b("馬単", [4, 2], 100), monbetsu, synth_results) == 0)
check("settle: trifecta ordered / trio unordered",
      o.bet_return(_b("3連単", [2, 4, 1], 100), monbetsu, synth_results) == 5020
      and o.bet_return(_b("3連複", [4, 1, 2], 100), monbetsu, synth_results) == 1170)
check("settle: wide pays when both are top-3",
      o.bet_return(_b("ワイド", [1, 4], 100), monbetsu, synth_results) == 470)
check("settle: place pays for 3rd, not for 4th",
      o.bet_return(_b("複勝", [1], 100), monbetsu, synth_results) == 180
      and o.bet_return(_b("複勝", [10], 100), monbetsu, synth_results) == 0)
_no_exacta = json.loads(json.dumps(synth_results))
del _no_exacta["payouts_jpy"]["exacta"]
check("settle: winning bet with missing payout data returns None",
      o.bet_return(_b("馬単", [2, 4], 100), monbetsu, _no_exacta) is None)

# --- ledger: init, cap, idempotent settle, relief ---
with tempfile.TemporaryDirectory() as td:
    led = points_mod.PointsLedger(Path(td) / "points.json")
    check("ledger: initial balance and 30% cap",
          led.balance("chatgpt") == 10000 and led.cap("chatgpt") == 3000)
    rec = led.settle("r1", "レース1", "chatgpt", 1000, 2500)
    check("ledger: settle moves balance by returned - staked",
          rec["balance"] == 11500 and led.balance("chatgpt") == 11500)
    rec2 = led.settle("r1", "レース1", "chatgpt", 1000, 2500)
    check("ledger: settle is idempotent per race+account",
          rec2["balance"] == 11500 and led.balance("chatgpt") == 11500)
    led.settle("r2", "レース2", "claude", 9950, 0)
    check("ledger: balance can fall below the minimum bet",
          led.balance("claude") == 50)
    check("ledger: relief tops up once per race",
          led.ensure_bettable("claude", "r3") is True
          and led.balance("claude") == 1000
          and led.ensure_bettable("claude", "r3") is False)

# --- settle_race_points end to end ---
with tempfile.TemporaryDirectory() as td:
    led = points_mod.PointsLedger(Path(td) / "points.json")
    state_syn = {
        "race": "第1回 2歳 未勝利",
        "race_id": monbetsu["race"]["race_id"],
        "texts": {"predictions": {"chatgpt": text_ok}},
        "final_bet": "単勝 2 500点\n3連複 1-2-4 500点",
    }
    recs = o.settle_race_points(monbetsu, synth_results, state_syn, led)
    check("settle race: AI staked 1000 returned 0",
          recs["chatgpt"]["staked"] == 1000 and recs["chatgpt"]["returned"] == 0
          and led.balance("chatgpt") == 9000)
    check("settle race: conclusion account settles the final bet",
          recs["shikai"]["returned"] == 1450 + 5850
          and led.balance("shikai") == 10000 - 1000 + 7300)

# --- EV summary ---
_ev_state = {"texts": {"predictions": {"chatgpt": text_ok}}}
_ev = o.ev_summary(monbetsu, monbetsu_results, _ev_state)
_odds2 = next(f["win_odds"] for f in monbetsu_results["finish"] if f["number"] == 2)
_row = next(r for r in _ev["chatgpt"] if r["mark"] == "△")
check("ev: win_pct x closing odds",
      _row["ev"] is not None and abs(_row["ev"] - 0.10 * _odds2) < 1e-9)

# --- generation with a points snapshot ---
_snap = {
    p["username"]: {"balance": 10000, "cap": 3000} for p in o.PERSONAS
}
_snap["shikai"] = {"balance": 10000, "cap": 3000}
_gen_pts = o._generate_prediction_texts(
    monbetsu, o.build_providers(mock=True), points=_snap
)
check("gen points: opening and conclusion carry balances and fair odds",
      "持ち点" in _gen_pts["texts"]["opening"]
      and "フェアオッズ" in _gen_pts["texts"]["conclusion"]
      and "結論口座" in _gen_pts["texts"]["conclusion"])

# --- Exclusion + bet reconciliation (2026-11-05 user decisions) ---------
# (1) An AI that fails validation twice is excluded and the rest of the
# field completes the race; (2) bets that contradict the final marks
# are regenerated (or rebuilt deterministically) so settlement always
# uses slips consistent with the final ballot.

check("consistency: bets within final marks pass",
      o.bet_consistency_errors(good_bets, good, monbetsu) == [])
_marks_flip = {"◎": (2, "ラップドアップ"), "○": (10, "タイムエッジ"),
               "▲": (4, "ダムドローズ"), "△": (1, "エリオラ")}
check("consistency: no bet on the final honmei flagged",
      o.bet_consistency_errors(good_bets, _marks_flip, monbetsu) != [])
check("consistency: leg outside the final marks flagged",
      o.bet_consistency_errors(
          [{"type": "単勝", "numbers": [3], "amount": 1000}], good, monbetsu) != [])
_b_of = {h["number"]: h["bracket"] for h in monbetsu["horses"]}
check("consistency: wakuren over marked brackets passes",
      o.bet_consistency_errors(
          [{"type": "枠連", "numbers": [_b_of[10], _b_of[4]], "amount": 1000}],
          good, monbetsu) == [])
_unmarked_bracket = next(
    b for b in set(_b_of.values())
    if all(_b_of[n] != b for n in (10, 4, 1, 2))
)
check("consistency: wakuren on a bracket with no marked horse flagged",
      o.bet_consistency_errors(
          [{"type": "枠連", "numbers": [_unmarked_bracket, _b_of[10]], "amount": 1000}],
          good, monbetsu) != [])


class _AlwaysInvalid:
    """Provider whose every reply fails validation (no marks, no bets)."""
    name = "bad"

    def chat(self, system, user, model=None, max_tokens=None):
        return "よく分かりませんが、適当にこの馬が来ると思います。"


_bad_persona = {"username": "badai", "display": "バッドAI"}

# (a) one AI invalid twice -> excluded, remaining three complete
_built_a = [(p, providers.MockProvider(p["display"])) for p in o.PERSONAS]
_built_a.append((_bad_persona, _AlwaysInvalid()))
_gen_a = o._generate_prediction_texts(monbetsu, _built_a)
check("exclusion: race completes with the remaining AIs",
      bool(_gen_a["texts"]["conclusion"])
      and len(_gen_a["texts"]["predictions"]) == 3)
check("exclusion: recorded in state with stage and reason",
      len(_gen_a["excluded"]) == 1
      and _gen_a["excluded"][0]["username"] == "badai"
      and _gen_a["excluded"][0]["stage"] == "prediction"
      and bool(_gen_a["excluded"][0]["reason"]))
check("exclusion: excluded AI has no prediction, marks, or bets",
      "badai" not in _gen_a["texts"]["predictions"]
      and "badai" not in _gen_a["final_marks"]
      and "badai" not in _gen_a["final_bets"])
check("exclusion: conclusion and review name the absent AI",
      "バッドAI" in _gen_a["texts"]["conclusion"]
      and "badai" in o.build_review_text(monbetsu, monbetsu_results, _gen_a))
with tempfile.TemporaryDirectory() as td:
    _led_a = points_mod.PointsLedger(Path(td) / "points.json")
    _recs_a = o.settle_race_points(monbetsu, synth_results, _gen_a, _led_a)
    check("exclusion: settlement skips the excluded AI",
          "badai" not in _recs_a
          and set(_recs_a) == {"chatgpt", "claude", "gemini", "shikai"})

# (c) exclusions that leave fewer than two AIs abort the run
try:
    o._generate_prediction_texts(
        monbetsu,
        [(o.PERSONAS[0], providers.MockProvider("x")),
         (_bad_persona, _AlwaysInvalid())],
    )
    check("exclusion: dropping to one active AI aborts", False)
except RuntimeError:
    check("exclusion: dropping to one active AI aborts", True)
try:
    o._generate_prediction_texts(
        monbetsu,
        [(o.PERSONAS[0], providers.MockProvider("x")),
         (_bad_persona, _AlwaysInvalid()),
         ({"username": "badai2", "display": "バッドAI2"}, _AlwaysInvalid())],
    )
    check("exclusion: two failures out of three abort", False)
except RuntimeError:
    check("exclusion: two failures out of three abort", True)


# (b) debate flips the honmei -> bets are regenerated onto the final marks
class _FlipStub:
    """Cloud stub: valid prediction on ◎10, debate ballot flips ◎ to 2,
    and the bets-only regeneration answers with slips on the new ◎."""
    name = "flip"

    def __init__(self, regen_text):
        self.regen_text = regen_text
        self.regen_calls = 0

    def chat(self, system, user, model=None, max_tokens=None):
        if "買い目だけを出し直して" in user:
            self.regen_calls += 1
            return self.regen_text
        if "議論" in user or "他のAI" in user:
            return (
                "議論の結果、本命を入れ替えます。\n\n【最終票】\n"
                "◎ 2 ラップドアップ\n○ 10 タイムエッジ\n"
                "▲ 4 ダムドローズ\n△ 1 エリオラ"
            )
        return text_ok


_flip_persona = {"username": "flip", "display": "フリップ"}
_stub_b = _FlipStub("【買い目】\n単勝 2 400点\n3連複 2-4-10 600点")
_gen_b = o._generate_prediction_texts(monbetsu, [(_flip_persona, _stub_b)])
check("reconcile: flipped honmei triggers a bets revision",
      _gen_b["final_marks"]["flip"]["◎"][0] == 2
      and _stub_b.regen_calls == 1
      and _gen_b["bet_revisions"]["flip"]["method"] == "regenerated"
      and _gen_b["bet_revisions"]["flip"]["original_bets"][0]["numbers"] == [10])
check("reconcile: regenerated bets are consistent with the final marks",
      _gen_b["final_bets"]["flip"] == [
          {"type": "単勝", "numbers": [2], "amount": 400},
          {"type": "3連複", "numbers": [2, 4, 10], "amount": 600},
      ]
      and o.bet_consistency_errors(
          _gen_b["final_bets"]["flip"],
          {mk: tuple(v) for mk, v in _gen_b["final_marks"]["flip"].items()},
          monbetsu) == []
      and "単勝 2" in _gen_b["bets"]["flip"]
      and "単勝 2" in _gen_b["texts"]["conclusion"])

# (b2) regeneration also fails -> deterministic rebuild from final marks
_stub_b2 = _FlipStub("【買い目】\n単勝 99 1000点")
_gen_b2 = o._generate_prediction_texts(monbetsu, [(_flip_persona, _stub_b2)])
check("reconcile: failed regeneration falls back to a deterministic rebuild",
      _gen_b2["bet_revisions"]["flip"]["method"] == "deterministic-fallback"
      and _gen_b2["final_bets"]["flip"] == [
          {"type": "単勝", "numbers": [2], "amount": 500},
          {"type": "3連複", "numbers": [2, 4, 10], "amount": 500},
      ]
      and o.bet_consistency_errors(
          _gen_b2["final_bets"]["flip"],
          {mk: tuple(v) for mk, v in _gen_b2["final_marks"]["flip"].items()},
          monbetsu) == [])


# (b3) the structured (local) persona goes through the same route
class _LocalFlipStub:
    name = "localstub"

    def chat(self, system, user, model=None, max_tokens=None):
        return "議論します。本命を入れ替えるべきだと考えます。"

    def chat_json(self, system, user, schema, model=None, max_tokens=None):
        props = schema.get("properties") or {}
        if "marks" in props and "bets" in props:  # prediction schema
            return {"marks": [
                        {"mark": "◎", "number": 10, "win_pct": 30, "top3_pct": 60, "reason": "本命"},
                        {"mark": "○", "number": 4, "win_pct": 22, "top3_pct": 50, "reason": "対抗"},
                        {"mark": "▲", "number": 1, "win_pct": 15, "top3_pct": 40, "reason": "単穴"},
                        {"mark": "△", "number": 2, "win_pct": 10, "top3_pct": 30, "reason": "連下"}],
                    "bets": [{"type": "単勝", "combo": "10", "amount": 1000}],
                    "comment": ""}
        if "marks" in props:  # ballot schema: honmei flips to 2
            return {"marks": [
                        {"mark": "◎", "number": 2, "reason": "討論で変更"},
                        {"mark": "○", "number": 10, "reason": "対抗"},
                        {"mark": "▲", "number": 4, "reason": "単穴"},
                        {"mark": "△", "number": 1, "reason": "連下"}],
                    "comment": ""}
        return {"bets": [{"type": "単勝", "combo": "2", "amount": 1000}]}


_local_flip_persona = {"username": "local", "display": "ローカル", "structured": True}
_gen_b3 = o._generate_prediction_texts(
    monbetsu, [(_local_flip_persona, _LocalFlipStub())]
)
check("reconcile: structured persona's bets are regenerated the same way",
      _gen_b3["final_marks"]["local"]["◎"][0] == 2
      and _gen_b3["final_bets"]["local"] == [
          {"type": "単勝", "numbers": [2], "amount": 1000}]
      and _gen_b3["bet_revisions"]["local"]["method"] == "regenerated")

# settlement prefers the reconciled final_bets over the prediction text
with tempfile.TemporaryDirectory() as td:
    _led_fb = points_mod.PointsLedger(Path(td) / "points.json")
    _state_fb = {
        "race": "第1回 2歳 未勝利",
        "race_id": monbetsu["race"]["race_id"],
        "texts": {"predictions": {"chatgpt": text_ok}},
        "final_bets": {"chatgpt": [{"type": "単勝", "numbers": [2], "amount": 1000}]},
        "final_bet": None,
    }
    _recs_fb = o.settle_race_points(monbetsu, synth_results, _state_fb, _led_fb)
    check("settle race: reconciled final_bets win over prediction text",
          _recs_fb["chatgpt"]["staked"] == 1000
          and _recs_fb["chatgpt"]["returned"] == 2900)

# --- Market anchor, conclusion gate, A/B measurement ---------------------
# (2026-11-05, prior-art items 1/2/4: the fixture carries de-vigged
# market probabilities, the market votes as an extra ballot, the
# conclusion account is gated on an aggregate edge over the market,
# and pre- vs post-debate aggregates are scored permanently.)

try:
    import build_fixture as _bf
except ImportError:  # system python lacks the scraping deps (bs4, ...)
    _bf = None

if _bf is not None:
    _dv = _bf.devig_win_probs({1: 2.0, 2: 4.0, 3: 4.0})
    check("devig: probabilities normalise to 1",
          abs(sum(_dv.values()) - 1.0) < 1e-9 and abs(_dv[1] - 0.5) < 1e-9)
    check("devig: unusable odds yield no probabilities",
          _bf.devig_win_probs({1: 0.0, 2: -3.0}) == {}
          and _bf.devig_win_probs({}) == {})
    check("odds api: win odds parsed, junk skipped",
          _bf.parse_win_odds(
              {"data": {"odds": {"1": {"7": ["2.9", "0", "1"], "3": ["---"]}}}})
          == {7: 2.9})
    check("odds api: pre-sale payload is empty, not an error",
          _bf.parse_win_odds({"data": ""}) == {}
          and _bf.parse_win_odds({}) == {})
else:
    print("SKIP odds checks (build_fixture needs the venv python)")

import copy as _copy

_names = {h["number"]: h["name"] for h in monbetsu["horses"]}


def _stamp_market(fx, odds):
    tot = sum(1.0 / x for x in odds.values())
    for h in fx["horses"]:
        od = odds.get(h["number"])
        h["market"] = {
            "odds": od,
            "win_prob": (1.0 / od / tot) if od else None,
            "snapshot_time": "2026-11-05T12:00:00+0900",
        }
    return fx


check("market: no market data without the odds snapshot",
      o.fixture_market(monbetsu) == {}
      and o.market_ballot_marks(monbetsu) is None)

_bet_fx = _stamp_market(
    _copy.deepcopy(monbetsu),
    {**{n: 30.0 for n in _names}, 10: 8.0, 2: 3.0, 4: 5.0, 1: 6.0},
)
_mb = o.market_ballot_marks(_bet_fx)
check("market ballot: top four by market probability",
      [_mb[m][0] for m in o.MARKS] == [2, 4, 1, 10])
check("market: race text carries the market win%",
      "市場勝率" in o.race_text(_bet_fx))
_ann = o.annotate_market(text_ok, _bet_fx)
check("market: annotation shows market% next to own% and stays valid",
      "（市場" in _ann and o.prediction_errors(_ann, _bet_fx) == [])
check("market: annotation is idempotent",
      o.annotate_market(_ann, _bet_fx) == _ann)
check("ballot probs: parsed from the final-ballot section",
      o.parse_ballot_probs(
          "議論。\n\n【最終票】\n◎ 10 タイムエッジ 勝率35%\n○ 4 ダムドローズ 勝率20%")
      == {10: 35, 4: 20})

_vec = o.build_probability_vector([{10: 40, 4: 20}], _bet_fx)
check("vector: full-field, sums to 100, residual follows the market",
      abs(sum(_vec.values()) - 100.0) < 1e-6
      and abs(_vec[10] - 40.0) < 1e-6 and _vec[2] > _vec[3])


class _FixedStub:
    """Cloud stub with a fixed prediction and a fixed final ballot
    (marks never move, so no bets reconciliation is triggered)."""
    name = "fixed"

    def __init__(self, pred, ballot):
        self.pred = pred
        self.ballot = ballot

    def chat(self, system, user, model=None, max_tokens=None):
        if "次のレースの予想をしてください" in user:
            return self.pred
        return self.ballot


def _pred_text(mp, bets):
    lines = [
        f"{mk} {n} {_names[n]} 勝率{w}% 3着内{t}% — 理由。"
        for mk, n, w, t in mp
    ]
    return "\n".join(lines + ["", "【買い目】"] + bets)


def _ballot_text(mp):
    lines = ["議論しました。結論は変わりません。", "", "【最終票】"]
    lines += [f"{mk} {n} {_names[n]} 勝率{w}%" for mk, n, w, t in mp]
    return "\n".join(lines)


def _fixed_gen(fx, mp, bets, count=2):
    built = [
        ({"username": f"fixed{i}", "display": f"固定{i}"},
         _FixedStub(_pred_text(mp, bets), _ballot_text(mp)))
        for i in range(count)
    ]
    return o._generate_prediction_texts(fx, built)


def _state_of(gen):
    return {
        "race": gen["race"],
        "race_id": monbetsu["race"]["race_id"],
        "texts": gen["texts"],
        "initial_marks": gen["initial_marks"],
        "final_marks": gen["final_marks"],
        "final_probs": gen["final_probs"],
        "final_bets": gen["final_bets"],
        "conclusion_ranking": gen["conclusion_ranking"],
        "conclusion_gate": gen["conclusion_gate"],
        "final_bet": gen["final_bet"],
        "points": {"enabled": True},
    }


# gate -> bet: the panel's 35% on #10 vs the market's ~11.5%
_mp_bet = [("◎", 10, 35, 55), ("○", 4, 20, 45), ("▲", 1, 15, 40), ("△", 2, 10, 30)]
_gen_mkt = _fixed_gen(_bet_fx, _mp_bet, ["単勝 10 400点", "3連複 1-4-10 600点"])
_gate = _gen_mkt["conclusion_gate"]
check("gate: a clear edge over the market bets, anchored on the panel top",
      _gate["decision"] == "bet" and _gate["anchor"] == 10
      and _gen_mkt["final_bet"].startswith("単勝 10"))
check("market ballot: recorded in state and shown in the conclusion",
      _gen_mkt["market_ballot"]["marks"]["◎"][0] == 2
      and "市場（単勝オッズ）の票" in _gen_mkt["texts"]["conclusion"])
check("market: ballot win percentages are recorded per AI",
      _gen_mkt["final_probs"]["fixed0"].get("10") == 35)
with tempfile.TemporaryDirectory() as td:
    _led_m = points_mod.PointsLedger(Path(td) / "points.json")
    _recs_m = o.settle_race_points(
        _bet_fx, synth_results, _state_of(_gen_mkt), _led_m
    )
    check("market ballot: no points account and never settled",
          "market" not in _recs_m
          and "market" not in _led_m.data["accounts"]
          and set(_recs_m) == {"fixed0", "fixed1", "shikai"})

# gate -> pass: the panel agrees with the market (30% vs ~31.7%)
_pass_fx = _stamp_market(
    _copy.deepcopy(monbetsu),
    {**{n: 25.0 for n in _names}, 10: 2.5, 2: 4.0, 4: 6.0, 1: 8.0},
)
_mp_pass = [("◎", 10, 30, 55), ("○", 2, 20, 45), ("▲", 4, 15, 40), ("△", 1, 10, 30)]
_gen_pass = _fixed_gen(_pass_fx, _mp_pass, ["単勝 10 400点", "3連複 1-2-10 600点"])
check("gate: agreement with the market is not an edge -> pass",
      _gen_pass["conclusion_gate"]["decision"] == "pass"
      and _gen_pass["final_bet"] is None
      and "見送り" in _gen_pass["texts"]["conclusion"])
with tempfile.TemporaryDirectory() as td:
    _led_p = points_mod.PointsLedger(Path(td) / "points.json")
    _recs_p = o.settle_race_points(
        _pass_fx, synth_results, _state_of(_gen_pass), _led_p
    )
    check("gate: a pass settles as a clean zero-stake record",
          _recs_p["shikai"]["staked"] == 0
          and _recs_p["shikai"]["returned"] == 0
          and _led_p.balance("shikai") == 10000)
    _recs_p2 = o.settle_race_points(
        _pass_fx, synth_results, _state_of(_gen_pass), _led_p
    )
    check("gate: the pass settlement is idempotent",
          _recs_p2["shikai"]["balance"] == 10000)

# gate -> longshot cap: the panel's top (#3 at 45%) is priced ~1% by
# the market, so the next eligible horse (#10) anchors instead
_long_fx = _stamp_market(
    _copy.deepcopy(monbetsu),
    {**{n: 20.0 for n in _names}, 3: 80.0, 10: 3.0, 2: 4.5, 4: 6.0, 1: 7.0},
)
_mp_long = [("◎", 3, 45, 70), ("○", 10, 35, 60), ("▲", 2, 10, 30), ("△", 4, 5, 20)]
_gen_long = _fixed_gen(
    _long_fx, _mp_long, ["単勝 3 400点", "3連複 2-3-10 600点"], count=4
)
_gate_l = _gen_long["conclusion_gate"]
check("gate: longshot top is skipped, next eligible horse anchors the bet",
      _gate_l["decision"] == "bet" and _gate_l["anchor"] == 10
      and _gate_l["skipped_longshot"] == 3
      and _gen_long["conclusion_ranking"][0]["number"] == 3
      and _gen_long["final_bet"].startswith("単勝 10"))

# gate -> ungated: the plain Monbetsu fixture has no market data
_gen_nomkt = o._generate_prediction_texts(monbetsu, o.build_providers(mock=True))
check("gate: missing market data bets ungated and records it",
      _gen_nomkt["conclusion_gate"]["decision"] == "ungated"
      and _gen_nomkt["final_bet"] is not None
      and _gen_nomkt["market_ballot"] is None)

# --- A/B scoring ---
_ab_state = _state_of(_gen_mkt)
_scored = o.score_ab_race(_bet_fx, synth_results, _ab_state)
check("ab: a settled race scores pre/post/concl",
      "skipped" not in _scored
      and _scored["pre"]["log_loss"] is not None
      and _scored["post"]["pick"] == 10
      and _scored["concl"]["decision"] == "bet"
      and _scored["concl"]["roi"] == 0.0)  # #10 lost; the winner was #2

_legacy_ab = {
    "texts": {"predictions": {"chatgpt": text_ok}},
    "initial_marks": {"chatgpt": {"◎": [10, "タイムエッジ"]}},
}
check("ab: legacy state without final_probs skips safely",
      "skipped" in o.score_ab_race(monbetsu, monbetsu_results, _legacy_ab)
      and "skipped" in o.record_ab_for_race(
          monbetsu, monbetsu_results, _legacy_ab, live=False))

_cwd_ab = _os.getcwd()
with tempfile.TemporaryDirectory() as td:
    _os.chdir(td)
    try:
        _out1 = o.record_ab_for_race(_bet_fx, synth_results, _ab_state, live=False)
        check("ab: the record lands in the dry-run ledger",
              Path("runs/dryrun/ab-ledger.json").exists()
              and _out1["totals"]["n"] == 1)
        _out2 = o.record_ab_for_race(_bet_fx, synth_results, _ab_state, live=False)
        check("ab: re-recording the same race does not double-count",
              _out2["totals"]["n"] == 1)
        _abl = o.ABLedger(o.ab_ledger_path(False))
        for i in range(9):
            _abl.record({
                "race_id": f"dummy-{i}", "race": "ダミー", "winner": 1,
                "pre": {"pick": 1, "log_loss": 2.0, "roi": 0.5},
                "post": {"pick": 1, "log_loss": 1.8, "roi": 0.6},
                "concl": {"pick": 1, "decision": "bet", "roi": 0.4},
            })
        _totals10 = _abl.totals()
        _line10 = o.ab_standings_text(_totals10)
        check("ab: standings appear at 10 races and cite the judge point",
              _totals10["n"] == 10 and _line10 is not None
              and "200" in _line10)
        _state_line = dict(_ab_state)
        _state_line["review"] = {"ab": {"totals": _totals10}}
        check("ab: the review note carries the standings line",
              "【討論のA/B測定】" in o.build_review_text(
                  _bet_fx, synth_results, _state_line))
        _abl_path = o.ab_ledger_path(False)
        _abl_path.write_text("{ not json", encoding="utf-8")
        _abl2 = o.ABLedger(_abl_path)
        check("ab: a corrupt ledger starts fresh instead of breaking",
              _abl2.recovered and _abl2.totals()["n"] == 0)
    finally:
        _os.chdir(_cwd_ab)

# --- hashtags on the moderator's posts (2026-11-05) -----------------------
# Every @shikai note (opener, conclusion, review) ends with one hashtag
# line: project tag, venue tag, race tag. The line is derived from the
# fixture alone, so all three stages of one race agree.
_eliz = json.loads((FIXTURES / "2025-elizabeth-cup.json").read_text(encoding="utf-8"))
_mon1105 = json.loads((FIXTURES / "2026-monbetsu-1105-12r.json").read_text(encoding="utf-8"))
_funa1105 = json.loads((FIXTURES / "2026-funabashi-1105-11r.json").read_text(encoding="utf-8"))
check("tags: line is project + venue + year-race, in order",
      o.hashtag_line(_eliz) == "#AI競馬予想 #京都競馬場 #2025エリザベス女王杯"
      and o.hashtag_line(_mon1105) == "#AI競馬予想 #門別競馬場 #2026道営記念"
      and o.hashtag_line(_funa1105)
      == "#AI競馬予想 #船橋競馬場 #2026カムイユカラスプリント")
check("tags: venue tag from the fixture venue (JRA and NAR)",
      o.venue_hashtag(monbetsu) == "#門別競馬場"
      and o.venue_hashtag(_eliz) == "#京都競馬場"
      and o.venue_hashtag(jbc) == "#金沢競馬場")

# Sanitisation of the scraped race names.
check("tags: bracketed grade badge and glued 重賞 dropped",
      o._tag_fragment("道営記念〔H1〕重賞") == "道営記念"
      and o._tag_fragment("カムイユカラスプリント(A2)") == "カムイユカラスプリント"
      and o._tag_fragment("テスト（G1）") == "テスト")
check("tags: non-annotation parenthetical content is part of the name",
      o._tag_fragment("天皇賞(秋)") == "天皇賞秋")
check("tags: whitespace (half/full width) removed",
      o._tag_fragment("2歳 未勝利") == "2歳未勝利"
      and o._tag_fragment("テスト　レース") == "テストレース")
check("tags: middle dot stripped (would truncate a Misskey hashtag)",
      o._tag_fragment("テスト・レース") == "テストレース")
check("tags: unmatched opening bracket truncates the scrape glue",
      o._tag_fragment("道営スプリント〔H1〕(ソウルラッシュ賞重賞") == "道営スプリント")

# The three @shikai stages of a mock run carry the identical final line;
# the AIs' own posts do not gain one.
_gen_tag = o._generate_prediction_texts(
    monbetsu, o.build_providers(mock=True), points=_snap
)
_tag_line = o.hashtag_line(monbetsu)
check("tags: opener and conclusion end with the tag line",
      _gen_tag["texts"]["opening"].splitlines()[-1] == _tag_line
      and _gen_tag["texts"]["conclusion"].splitlines()[-1] == _tag_line)
check("tags: review ends with the same tag line",
      o.build_review_text(monbetsu, monbetsu_results, _gen_tag)
      .splitlines()[-1] == _tag_line)
check("tags: AI prediction/debate posts carry no hashtag line",
      all(_tag_line not in t for t in _gen_tag["texts"]["predictions"].values())
      and all(_tag_line not in t for t in _gen_tag["texts"]["debate"].values()))

# Compatibility: a completed state stored by the OLD code (opener and
# conclusion posted without tags, note IDs recorded) must review
# cleanly tomorrow: prediction re-entry posts nothing, the stored texts
# stay untouched, and exactly one new @shikai note (the review, WITH
# tags) is emitted; a second review is refused instead of duplicating.


class _RecordingEmitter(o.Emitter):
    def __init__(self):
        super().__init__(live=False)
        self.calls: list = []

    def emit(self, username, text, reply_to):
        self.calls.append((username, text, reply_to))
        return super().emit(username, text, reply_to)


_cwd_tag = _os.getcwd()
with tempfile.TemporaryDirectory() as td:
    _os.chdir(td)
    try:
        _store_tag = o.RunState.open(
            o.state_path_for("fixtures/tag-compat.json", False),
            monbetsu, "dryrun",
        )
        _built_tag = o.build_providers(mock=True)
        o.run_prediction(
            monbetsu, o.Emitter(live=False), _built_tag, _store_tag
        )
        # Downgrade the stored texts to the old format (no tag line),
        # exactly what the 2026-11-05 live states hold.
        for _key in ("opening", "conclusion"):
            _t = _store_tag.data["texts"][_key]
            assert _t.endswith(_tag_line), _key
            _store_tag.data["texts"][_key] = _t[: -len(_tag_line)].rstrip("\n")
        _store_tag.save()
        _ids_before = json.loads(json.dumps(_store_tag.data["note_ids"]))
        _texts_before = json.loads(json.dumps(_store_tag.data["texts"]))

        _rec_pred = _RecordingEmitter()
        o.run_prediction(monbetsu, _rec_pred, _built_tag, _store_tag)
        check("tags compat: re-entering a completed old state posts nothing",
              _rec_pred.calls == []
              and _store_tag.data["note_ids"] == _ids_before
              and _store_tag.data["texts"] == _texts_before)

        _rec_rev = _RecordingEmitter()
        o.run_review(
            monbetsu, synth_results, _store_tag, _rec_rev, built=_built_tag
        )
        _mc_notes = [c for c in _rec_rev.calls if c[0] == o.MC_USERNAME]
        check("tags compat: old state review emits exactly one MC note",
              len(_mc_notes) == 1
              and bool((_store_tag.data.get("review") or {}).get("note_id")))
        check("tags compat: the new review note carries the tag line",
              _mc_notes[0][1].splitlines()[-1] == _tag_line)
        check("tags compat: stored old texts were not retro-tagged",
              o.PROJECT_TAG not in _store_tag.data["texts"]["opening"]
              and o.PROJECT_TAG not in _store_tag.data["texts"]["conclusion"])
        try:
            o.run_review(
                monbetsu, synth_results, _store_tag,
                _RecordingEmitter(), built=_built_tag,
            )
            _refused = False
        except SystemExit:
            _refused = True
        check("tags compat: a second review is refused, not duplicated",
              _refused)
    finally:
        _os.chdir(_cwd_tag)

# --- full mock dry-run end to end, now with gate + A/B wired in ---
_cwd_e2e = _os.getcwd()
with tempfile.TemporaryDirectory() as td:
    _os.chdir(td)
    try:
        _store_e2e = o.RunState.open(
            o.state_path_for("fixtures/e2e-mock.json", False),
            monbetsu, "dryrun",
        )
        _built_e2e = o.build_providers(mock=True)
        o.run_prediction(
            monbetsu, o.Emitter(live=False), _built_e2e, _store_e2e
        )
        check("e2e: mock prediction completes (ungated, no market data)",
              _store_e2e.data["stage"] == "complete"
              and _store_e2e.data["conclusion_gate"]["decision"] == "ungated")
        o.run_review(
            monbetsu, synth_results, _store_e2e,
            o.Emitter(live=False), built=_built_e2e,
        )
        _review_e2e = _store_e2e.data.get("review") or {}
        check("e2e: mock review settles points and records the A/B race",
              bool(_review_e2e.get("points"))
              and (_review_e2e.get("ab") or {}).get("totals", {}).get("n") == 1
              and Path("runs/dryrun/ab-ledger.json").exists())
    finally:
        _os.chdir(_cwd_e2e)

print()
if fails:
    print("FAILED:", ", ".join(fails))
    sys.exit(1)
print("ALL CHECKS PASSED")

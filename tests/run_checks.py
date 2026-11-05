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

print()
if fails:
    print("FAILED:", ", ".join(fails))
    sys.exit(1)
print("ALL CHECKS PASSED")

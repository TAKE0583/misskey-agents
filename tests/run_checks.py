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
    "◎ 10 タイムエッジ — 本命。\n○ 4 ダムドローズ — 対抗。\n"
    "▲ 1 エリオラ — 単穴。\n△ 2 ラップドアップ — 連下。\n\n"
    "【買い目（予算1,000円）】\n単勝 10 400円\n3連複 1-4-10 600円"
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
        return {"marks": [{"mark": "◎", "number": 10, "reason": "本命"},
                          {"mark": "○", "number": 4, "reason": "対抗"},
                          {"mark": "▲", "number": 1, "reason": "単穴"},
                          {"mark": "△", "number": 2, "reason": "連下"}],
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

print()
if fails:
    print("FAILED:", ", ".join(fails))
    sys.exit(1)
print("ALL CHECKS PASSED")

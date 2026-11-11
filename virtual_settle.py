"""仮想精算ツール (virtual settlement / strategy backtest).

過去のライブ実行 state と結果 fixture を使って、結論口座の戦略が
「もし違っていたらどうなっていたか」を実払戻で再精算する。
対象は結論口座のみ（各AIの買い目は各AIの判断なので対象外）。

比較軸:
  A. ゲート閾値スイープ — 集計勝率が市場×margin を超えたら賭ける。
     margin ∈ {常時（市場データがある限り賭ける）, 1.00, 1.05, 1.10,
     1.15（現行）, 1.20, 1.30}。軸はアンカー、買い目は現行 split。
  B. 軸ルール（仕様レビュー項目3）— margin 1.15・split のまま、
     買い目の軸を「ゲートのアンカー」（現行）にするか
     「スコアランキング首位（◎）」にするか。
  C. 買い目構成（同・項目1）— margin 1.15・アンカー軸のまま、
     split（単勝500+3連複500、現行）/ 単勝1000 / 3連複1000 /
     単勝500+複勝500 を比較する。
  D. PRE集計ゲート — アンカーと市場価格は記録どおりに固定し、
     討論前の初期予想だけの集計ベクトル（PRE）がそのアンカーに
     付けた勝率でゲート判定した場合（A/B計測の結論版）。
  E. 実績 — state の final_bet をそのまま精算し、記録済みの
     review.points と突き合わせて再現性を検証する。

前提と限界:
  - 市場価格と集計勝率は state の conclusion_gate 記録
    （予測時点で保存された値、小数2桁に丸め済み）を正とする。
    fixture ファイル側の市場スナップショットは、レビュー時の
    再生成で最終オッズに上書きされているレースが半数弱あり、
    予測時点の値ではないため使わない（2026-11-11 の初回実行で
    検出）。境界近傍では丸めにより本番判定とごく稀に食い違う
    可能性があり、A の 1.15 行と記録済み decision の一致率を
    整合性チェックとして表示する。
  - 市場データのないレースは全バリアントで見送り（現行の
    一律ルール 2026-11-10 を仮想側でも維持する）。
  - 払戻は結果 fixture の payouts_jpy（100点あたり）を使用。
    勝ち目なのに払戻表にない場合は「価格付け不能」として
    そのバリアントの合計から外し、件数を明示する（0円扱いしない）。
  - 取り消し時の返還は本番精算と同じく扱わない。
  - サンプルはまだ少なく（ゲート時代のライブは数十レース）、
    ROI の差は統計的に弱い。判断材料であって結論ではない。

使い方:
  .venv/bin/python virtual_settle.py [--out report.md] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import orchestrator as o

ROOT = Path(__file__).resolve().parent
TOTAL_STAKE = 1000  # 結論口座の1レース賭け金（現行の固定額）
MC = o.MC_USERNAME

MARGINS: list[tuple[str, float]] = [
    ("常時", 0.0),
    ("1.00", 1.00),
    ("1.05", 1.05),
    ("1.10", 1.10),
    ("1.15", 1.15),
    ("1.20", 1.20),
    ("1.30", 1.30),
]
SHAPES = ["split", "win", "trio", "winplace"]
SHAPE_LABEL = {
    "split": "単勝500+3連複500（現行）",
    "win": "単勝1000",
    "trio": "3連複1000",
    "winplace": "単勝500+複勝500",
}


# --- race loading -----------------------------------------------------------

def load_races(runs_dir: Path = ROOT / "runs",
               fixtures_dir: Path = ROOT / "fixtures"
               ) -> tuple[list[dict], list[tuple[str, str]]]:
    """Qualifying live races with everything a re-settlement needs.

    A race qualifies when it is a live run whose state carries a gate
    record with the aggregate vector, a conclusion ranking, and whose
    results fixture exists. Everything else is reported as excluded
    with a reason — never silently dropped."""
    races: list[dict] = []
    excluded: list[tuple[str, str]] = []
    for sp in sorted(runs_dir.glob("*-state.json")):
        stem = sp.name[: -len("-state.json")]
        state = json.loads(sp.read_text(encoding="utf-8"))
        if state.get("mode") != "live":
            excluded.append((stem, "ライブ実行ではない"))
            continue
        fx_path = fixtures_dir / f"{stem}.json"
        rs_path = fixtures_dir / f"{stem}-results.json"
        if not rs_path.exists():
            excluded.append((stem, "結果fixture未生成（精算待ち）"))
            continue
        gate = state.get("conclusion_gate") or {}
        if "aggregate_probs" not in gate:
            excluded.append((stem, "ゲート記録のない旧形式state"))
            continue
        ranking = [
            int(e["number"]) for e in (state.get("conclusion_ranking") or [])
        ]
        if not ranking:
            excluded.append((stem, "結論ランキングなし"))
            continue
        fixture = json.loads(fx_path.read_text(encoding="utf-8"))
        results = json.loads(rs_path.read_text(encoding="utf-8"))
        vector = {
            int(n): float(p) for n, p in gate["aggregate_probs"].items()
        }
        races.append({
            "stem": stem,
            "name": state.get("race") or stem,
            "fixture": fixture,
            "results": results,
            "state": state,
            "gate": gate,
            "vector": vector,
            "ranking": ranking,
            "winner": (results.get("finish") or [{}])[0].get("number"),
        })
    return races, excluded


# --- counterfactual machinery ----------------------------------------------

def decide_recorded(gate: dict, margin: float) -> tuple[bool, int | None]:
    """(bet?, anchor) under a gate margin, judged ONLY from the gate
    record saved at prediction time: its anchor, its aggregate
    probability for that anchor and its market probability. The
    fixture file's market snapshot is NOT used — for about half of
    the races it was overwritten with closing odds when the fixture
    was regenerated at review time. No market data (or the ungated
    era, which recorded no anchor) → always pass."""
    anchor = gate.get("anchor")
    if gate.get("market_missing") or anchor is None:
        return False, None
    anchor = int(anchor)
    if margin <= 0:
        return True, anchor
    agg = gate.get("aggregate_prob")
    mp = gate.get("market_prob")
    if agg is None or mp is None:
        return False, None
    return float(agg) >= float(mp) * margin, anchor


def anchor_order(anchor: int, ranking: list[int]) -> list[int]:
    return [anchor] + [n for n in ranking if n != anchor]


def make_bets(order: list[int], shape: str,
              fixture: dict) -> list[dict] | None:
    if shape == "split":
        return o.build_final_bet(
            [(n, {}) for n in order], fixture, total=TOTAL_STAKE)
    if shape == "win":
        return [{"type": "単勝", "numbers": [order[0]],
                 "amount": TOTAL_STAKE}]
    if shape == "trio":
        if len(order) < 3:
            return None
        return [{"type": "3連複", "numbers": sorted(order[:3]),
                 "amount": TOTAL_STAKE}]
    if shape == "winplace":
        half = TOTAL_STAKE // 2
        return [
            {"type": "単勝", "numbers": [order[0]], "amount": half},
            {"type": "複勝", "numbers": [order[0]], "amount": half},
        ]
    raise ValueError(f"unknown shape {shape}")


def settle_bets(bets: list[dict] | None, fixture: dict,
                results: dict) -> dict | None:
    """Settle a hypothetical slip with the real payouts. Returns None
    when a winning leg cannot be priced from the results fixture —
    the caller must surface that, never treat it as a 0 return."""
    if bets is None:
        return None
    staked = sum(b["amount"] for b in bets)
    returned = 0
    for b in bets:
        ret = o.bet_return(b, fixture, results)
        if ret is None:
            return None
        returned += ret
    return {"staked": staked, "returned": returned}


def pre_vector(race: dict) -> dict[int, float] | None:
    """The pre-debate aggregate, rebuilt exactly as score_ab_race does.
    None for states without the initial predictions recorded."""
    state = race["state"]
    texts = (state.get("texts") or {}).get("predictions") or {}
    initial_marks = state.get("initial_marks") or {}
    if not texts or not initial_marks:
        return None
    contribs = []
    for username in initial_marks:
        probs = o.parse_probabilities(texts.get(username, ""))
        contribs.append({num: win for num, win, _t in probs.values()})
    if not contribs:
        return None
    return o.build_probability_vector(contribs, race["fixture"])


def evaluate_race(race: dict) -> dict:
    """Every variant's outcome for one race."""
    fixture, results = race["fixture"], race["results"]
    ranking = race["ranking"]
    gate = race["gate"]
    out: dict = {"stem": race["stem"], "name": race["name"],
                 "winner": race["winner"]}
    # The ungated era recorded a vector but no anchor/market price, so
    # no gate variant can be judged for it; it still takes part in the
    # actual-settlement check below.
    gated = gate.get("decision") in ("bet", "pass")
    out["gated"] = gated

    def run(margin, axis, shape):
        bet, anchor = decide_recorded(gate, margin)
        if not bet or anchor is None:
            return {"bet": False, "staked": 0, "returned": 0}
        order = (anchor_order(anchor, ranking) if axis == "anchor"
                 else list(ranking))
        settled = settle_bets(make_bets(order, shape, fixture),
                              fixture, results)
        if settled is None:
            return {"bet": True, "unpriceable": True, "anchor": anchor,
                    "order": order}
        return {"bet": True, "anchor": anchor, "order": order, **settled}

    if gated:
        out["margins"] = {
            label: run(margin, "anchor", "split")
            for label, margin in MARGINS
        }
        out["axis_ranking"] = run(o.EDGE_MARGIN, "ranking", "split")
        out["shapes"] = {
            shape: run(o.EDGE_MARGIN, "anchor", shape)
            for shape in SHAPES if shape != "split"
        }
    else:
        out["margins"] = {label: None for label, _m in MARGINS}
        out["axis_ranking"] = None
        out["shapes"] = {shape: None for shape in SHAPES
                         if shape != "split"}

    # D: the same anchor and market price, but the PRE-debate vector
    # decides bet/pass — the controlled pre/post comparison.
    pv = pre_vector(race)
    anchor = gate.get("anchor")
    mp = gate.get("market_prob")
    if pv is None or anchor is None or mp is None:
        out["pre"] = None
    elif pv.get(int(anchor), 0.0) < float(mp) * o.EDGE_MARGIN:
        out["pre"] = {"bet": False, "staked": 0, "returned": 0}
    else:
        order = anchor_order(int(anchor), ranking)
        settled = settle_bets(make_bets(order, "split", fixture),
                              fixture, results)
        out["pre"] = (
            {"bet": True, "unpriceable": True, "anchor": int(anchor),
             "order": order}
            if settled is None else
            {"bet": True, "anchor": int(anchor), "order": order, **settled}
        )

    # E: the actual slip, and the recorded settlement to check against.
    state = race["state"]
    actual = {"bet": False, "staked": 0, "returned": 0}
    if state.get("final_bet"):
        bets, perr = o.parse_bet_lines(state["final_bet"])
        if not perr:
            settled = settle_bets(bets, fixture, results)
            if settled is not None:
                actual = {"bet": True, **settled}
    out["actual"] = actual
    recorded = ((state.get("review") or {}).get("points") or {}).get(MC)
    out["recorded"] = recorded
    out["recorded_gate_decision"] = race["gate"].get("decision")
    out["gate_anchor"] = race["gate"].get("anchor")
    return out


# --- aggregation and report --------------------------------------------------

def aggregate(evals: list[dict], key_path) -> dict:
    """Totals over the races where the variant could be priced."""
    bets = staked = returned = priced = unpriceable = 0
    for ev in evals:
        node = key_path(ev)
        if node is None:
            continue
        if not node.get("bet"):
            priced += 1
            continue
        if node.get("unpriceable"):
            unpriceable += 1
            continue
        priced += 1
        bets += 1
        staked += node["staked"]
        returned += node["returned"]
    return {
        "races": priced, "bets": bets, "staked": staked,
        "returned": returned,
        "roi": (returned / staked) if staked else None,
        "unpriceable": unpriceable,
    }


def _roi_text(agg: dict) -> str:
    if agg["roi"] is None:
        return "—"
    return f"{agg['roi']:.2f}"


def build_report(races: list[dict], excluded: list[tuple[str, str]],
                 evals: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# 仮想精算レポート（結論口座の戦略バックテスト）")
    lines.append("")
    gated_n = sum(1 for ev in evals if ev.get("gated"))
    lines.append(f"対象: ライブ {len(races)} レース"
                 f"（うちゲート適用 {gated_n} 件がバリアント比較の対象、"
                 f"1レース {TOTAL_STAKE} 点に正規化）")
    if excluded:
        lines.append(f"対象外 {len(excluded)} 件: " + "、".join(
            f"{stem}（{reason}）" for stem, reason in excluded))
    lines.append("")

    # Consistency: replaying the current rules must reproduce history.
    replay_ok, replay_ng = [], []
    actual_ok, actual_ng = [], []
    for ev in evals:
        replay = ev["margins"]["1.15"]
        recorded_decision = ev["recorded_gate_decision"]
        if recorded_decision in ("bet", "pass"):
            got = "bet" if replay.get("bet") else "pass"
            (replay_ok if got == recorded_decision else replay_ng).append(
                ev["stem"])
        rec = ev["recorded"]
        if rec is not None:
            same = (ev["actual"]["staked"] == rec.get("staked")
                    and ev["actual"]["returned"] == rec.get("returned"))
            (actual_ok if same else actual_ng).append(ev["stem"])
    lines.append("## 整合性チェック")
    lines.append(f"- 現行ルール（1.15・アンカー・split）の再演が記録済み"
                 f"ゲート判定と一致: {len(replay_ok)}/{len(replay_ok) + len(replay_ng)}"
                 + (f"（不一致: {', '.join(replay_ng)}）" if replay_ng else ""))
    lines.append(f"- 実買い目の再精算が記録済み精算と一致: "
                 f"{len(actual_ok)}/{len(actual_ok) + len(actual_ng)}"
                 + (f"（不一致: {', '.join(actual_ng)}）" if actual_ng else ""))
    lines.append("")

    lines.append("## A. ゲート閾値スイープ（軸=アンカー、買い目=split）")
    lines.append("| 閾値 | 賭けたレース | 賭け金 | 払戻 | ROI |")
    lines.append("|---|---|---|---|---|")
    for label, _m in MARGINS:
        agg = aggregate(evals, lambda ev, lb=label: ev["margins"][lb])
        lines.append(f"| {label} | {agg['bets']} | {agg['staked']} | "
                     f"{agg['returned']} | {_roi_text(agg)} |")
    lines.append("")

    lines.append("## B. 軸ルール比較（閾値1.15、買い目=split）— 項目3")
    agg_anchor = aggregate(evals, lambda ev: ev["margins"]["1.15"])
    agg_rank = aggregate(evals, lambda ev: ev["axis_ranking"])
    lines.append(f"- アンカー軸（現行）: {agg_anchor['bets']}レース、"
                 f"払戻 {agg_anchor['returned']} / 賭け {agg_anchor['staked']}"
                 f"、ROI {_roi_text(agg_anchor)}")
    lines.append(f"- ランキング◎軸: {agg_rank['bets']}レース、"
                 f"払戻 {agg_rank['returned']} / 賭け {agg_rank['staked']}"
                 f"、ROI {_roi_text(agg_rank)}")
    diffs = []
    for ev in evals:
        if not ev.get("gated"):
            continue
        a, r = ev["margins"]["1.15"], ev["axis_ranking"]
        if a.get("bet") and r.get("bet") and not a.get("unpriceable") \
                and not r.get("unpriceable") \
                and a.get("order", [None])[0] != r.get("order", [None])[0]:
            diffs.append(ev)
    lines.append(f"- 軸が割れたレース: {len(diffs)} 件")
    for ev in diffs:
        a, r = ev["margins"]["1.15"], ev["axis_ranking"]
        lines.append(f"  - {ev['name']}: アンカー {a['order'][0]}番"
                     f"（払戻 {a['returned']}）／ ◎ {r['order'][0]}番"
                     f"（払戻 {r['returned']}）、勝ち馬 {ev['winner']}番")
    lines.append("")

    lines.append("## C. 買い目構成比較（閾値1.15、軸=アンカー）— 項目1")
    lines.append("| 構成 | 賭けたレース | 賭け金 | 払戻 | ROI |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| {SHAPE_LABEL['split']} | {agg_anchor['bets']} | "
                 f"{agg_anchor['staked']} | {agg_anchor['returned']} | "
                 f"{_roi_text(agg_anchor)} |")
    for shape in SHAPES:
        if shape == "split":
            continue
        agg = aggregate(evals, lambda ev, sh=shape: ev["shapes"][sh])
        lines.append(f"| {SHAPE_LABEL[shape]} | {agg['bets']} | "
                     f"{agg['staked']} | {agg['returned']} | "
                     f"{_roi_text(agg)} |")
    lines.append("")

    lines.append("## D. PRE集計でゲート判定した場合（閾値1.15）")
    agg_pre = aggregate(evals, lambda ev: ev["pre"])
    lines.append(f"- PRE: {agg_pre['bets']}レース、払戻 {agg_pre['returned']}"
                 f" / 賭け {agg_pre['staked']}、ROI {_roi_text(agg_pre)}"
                 f"（対象 {agg_pre['races']} レース）")
    lines.append(f"- POST（現行・Aの1.15行と同一）: ROI {_roi_text(agg_anchor)}")
    lines.append("")

    lines.append("## 実績（記録どおり）")
    agg_actual = aggregate(evals, lambda ev: ev["actual"])
    lines.append(f"- 結論口座の実績再集計: 払戻 {agg_actual['returned']} / "
                 f"賭け {agg_actual['staked']}、ROI {_roi_text(agg_actual)}")
    lines.append("")
    lines.append("注意: サンプルは小さく、ROI差は統計的に弱い。"
                 "市場価格はゲート記録（予測時点の値）を使用し、"
                 "fixture側のスナップショット（半数弱のレースで"
                 "レビュー時に最終オッズへ上書き済み）は使っていない。"
                 "価格付け不能だったバリアントは合計から除外してある。")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="結論口座の仮想精算")
    ap.add_argument("--out", help="Markdownレポートの出力先")
    ap.add_argument("--json", help="全レース×全バリアントのJSON出力先")
    args = ap.parse_args()

    races, excluded = load_races()
    evals = [evaluate_race(r) for r in races]
    report = build_report(races, excluded, evals)
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(
            json.dumps(evals, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

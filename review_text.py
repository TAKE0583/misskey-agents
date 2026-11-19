#!/usr/bin/env python3
"""Review note text construction (extracted from orchestrator.py, 2026-11-19 split; behavior unchanged)."""
from __future__ import annotations

import points as points_mod

from ab_ledger import ab_standings_text
from common import LOCAL_PERSONA, MC_USERNAME, PERSONAS
from hashtags import hashtag_line
from settlement import ev_summary


def reflection_user_prompt(
    fixture: dict, results: dict, state: dict, username: str
) -> str:
    actual = {f["number"]: f for f in results["finish"]}
    lines = [
        f"【レース結果】{results['race']}（勝ちタイム {results['winning_time']}）",
        "着順（上位5頭）:",
    ]
    for f in results["finish"][:5]:
        lines.append(
            f"  {f['position']}着 {f['number']} {f['name']}（{f['popularity']}番人気、単勝{f['win_odds']}倍）"
        )
    payouts = results.get("payouts_jpy", {})
    if payouts.get("win") and payouts.get("trio"):
        lines.append(
            f"払戻: 単勝 {list(payouts['win'].values())[0]}円 / "
            f"3連複 {list(payouts['trio'].keys())[0]} {list(payouts['trio'].values())[0]}円"
        )
    lines.append("")
    lines.append("【あなたの印と実際の着順】")
    marks_map = state.get("final_marks") or state.get("marks") or {}
    for mark, (num, name) in (marks_map.get(username) or {}).items():
        fin = actual.get(num)
        lines.append(
            f"  {mark} {num} {name} → {fin['position']}着" if fin else f"  {mark} {num} {name} → 着順不明"
        )
    bets = (state.get("bets") or {}).get(username)
    if bets:
        lines.append("")
        lines.append(f"【あなたの買い目】\n{bets}")
    ev_rows = ev_summary(fixture, results, state).get(username) or []
    if ev_rows:
        lines.append("")
        lines.append("【あなたの確率推定と期待値（勝率 × 最終単勝オッズ、1.0超は割安の見抜き）】")
        for r in ev_rows:
            if r["ev"] is not None:
                lines.append(
                    f"  {r['mark']} {r['number']} {r['name']}: 勝率{r['win_pct']}% × "
                    f"{r['odds']}倍 = EV {r['ev']:.2f}（{r['position']}着）"
                )
    rec = ((state.get("review") or {}).get("points") or {}).get(username)
    if rec:
        lines.append("")
        lines.append(
            f"【精算】賭け {rec['staked']:,}点 → 払戻 {rec['returned']:,}点、"
            f"累積収支は {points_mod.format_score(rec['balance'])}点になりました。"
        )
    lines.append("")
    lines.append("この結果を受けて、あなたの反省を述べてください。")
    return "\n".join(lines)


def build_review_text(fixture: dict, results: dict, state: dict | None) -> str:
    top = results["finish"][:5]
    lines = [
        f"【回顧】{results['race']}",
        f"勝ちタイム {results['winning_time']} / {results.get('race_summary', '')}",
        "",
        "着順：",
    ]
    for f in top:
        lines.append(f"  {f['position']}着 {f['number']} {f['name']}（{f['popularity']}番人気）")
    payouts = results.get("payouts_jpy", {})
    if payouts.get("win") and payouts.get("trio"):
        lines.append(
            f"払戻：単勝 {list(payouts['win'].values())[0]}円 / "
            f"3連複 {list(payouts['trio'].values())[0]}円"
        )
    lines.append("")
    if state and state.get("excluded"):
        lines.append(
            "不参加のAI（生成段階で除外され、精算・反省の対象外）: "
            + "、".join(
                f"@{e['username']}（{e.get('stage', '?')}段階）"
                for e in state["excluded"]
            )
        )
    if state and state.get("conclusion_ranking"):
        best = state["conclusion_ranking"][0]
        actual = {f["number"]: f["position"] for f in results["finish"]}
        pos = actual.get(best["number"])
        verdict = (
            "本命が優勝！" if pos == 1 else f"最終結論の本命 {best['name']} は {pos}着でした。"
        )
        lines.append(f"最終結論の検証：{verdict}")
        marks_map = state.get("final_marks") or state.get("marks") or {}
        for username, marks in marks_map.items():
            honmei = marks.get("◎")
            if honmei:
                lines.append(
                    f"  @{username} の◎ {honmei[1]} は {actual.get(honmei[0], '?')}着"
                )
    else:
        lines.append("（予想スレッドの記録がないため、結果のみの回顧です）")
    gate = (state or {}).get("conclusion_gate") or {}
    if gate.get("decision") == "pass":
        lines.append(
            f"結論口座は見送りでした（{gate.get('reason', '対市場の優位が不足')}）。"
        )
    ev = ev_summary(fixture, results, state)
    if ev:
        lines.append("")
        lines.append(
            "期待値の検証（各AIの◎の勝率 × 最終単勝オッズ。1.0を超えていれば、"
            "結果はともあれ市場の最終評価より割安に見抜いていました）:"
        )
        for username, rows in ev.items():
            honmei = next((r for r in rows if r["mark"] == "◎"), None)
            if honmei and honmei["ev"] is not None:
                lines.append(
                    f"  {display_name(username)} ◎ {honmei['name']}: "
                    f"勝率{honmei['win_pct']}% × {honmei['odds']}倍 = "
                    f"EV {honmei['ev']:.2f}（{honmei['position']}着）"
                )
    points_rec = (state.get("review") or {}).get("points")
    if points_rec:
        lines.append("")
        lines.append("精算（累積収支）：")
        for username, rec in points_rec.items():
            profit = rec["returned"] - rec["staked"]
            lines.append(
                f"  {display_name(username)}: 賭け {rec['staked']:,}点 → "
                f"払戻 {rec['returned']:,}点（{profit:+,}点）"
                f"累積収支 {points_mod.format_score(rec['balance'])}点"
            )
        standings = (state.get("review") or {}).get("standings") or {}
        if standings:
            order = sorted(standings.items(), key=lambda kv: -kv[1])
            lines.append(
                "現在の順位（累積収支）: "
                + " / ".join(
                    f"{i + 1}位 {display_name(u)} "
                    f"{points_mod.format_score(b)}点"
                    for i, (u, b) in enumerate(order)
                )
            )
    ab_line = ab_standings_text(
        ((state.get("review") or {}).get("ab") or {}).get("totals")
    )
    if ab_line:
        lines.append("")
        lines.append(ab_line)
    lines.append("次回の予想に活かします。")
    lines.append("")
    lines.append(hashtag_line(fixture))
    return "\n".join(lines)


def display_name(username: str) -> str:
    for p in PERSONAS + [LOCAL_PERSONA]:
        if p["username"] == username:
            return p["display"]
    return "結論口座" if username == MC_USERNAME else username

#!/usr/bin/env python3
"""Fixture-to-text rendering and mark parsing (extracted from orchestrator.py, 2026-11-19 split; behavior unchanged)."""
from __future__ import annotations

import re
from market import market_distortion
MARK_RE = re.compile(r"([◎○▲△])\s*(?:馬番)?\s*(\d{1,2})\s*番?\s+([ァ-ヶー・]+)")

def person_stats_text(horse: dict) -> str:
    """Compact jockey/trainer season record for one runner line, e.g.
    '騎手（地方・今季）勝率12.8% 連対21.9% 複勝34.0% / 調教師（地方・今季）…'.
    The stats come from the fixture's jockey_stats / trainer_stats
    (db.netkeiba season tables for the race's own jurisdiction;
    career=True renders 通算 instead of 今季）. A person without
    stats contributes nothing, and a runner with neither renders the
    empty string, leaving the line exactly as it was before the
    fixture carried stats."""
    parts = []
    for key, label in (("jockey_stats", "騎手"), ("trainer_stats", "調教師")):
        st = horse.get(key)
        if not st:
            continue
        span = "通算" if st.get("career") else "今季"
        parts.append(
            f"{label}（{st['scope']}・{span}）"
            f"勝率{st['win_rate']:.1f}% 連対{st['top2_rate']:.1f}% "
            f"複勝{st['top3_rate']:.1f}%"
        )
    return " / ".join(parts)


def race_text(fixture: dict, form_limit: int | None = None) -> str:
    r = fixture["race"]
    # Bracket and weather have always been in the fixture but were not
    # rendered into the prompt until 2026-11-10 (the prompts ask the
    # AIs to weigh 枠順, so the data has to actually reach them). The
    # .get guards keep older fixtures rendering as before.
    header = (
        f"{r['date']} {r['venue']} {r['race_number']}R / {r['course']} "
        f"{r['distance_m']}m / 馬場：{r['going']}"
    )
    weather = str(r.get("weather") or "").strip()
    if weather:
        header += f" / 天候：{weather}"
    header += f" / {r['conditions']} / {r['field_size']}頭立て"
    lines = [
        f"第{r['edition']}回 {r['name']}（{r['grade']}）",
        header,
        "",
        "【出走馬】",
    ]
    distorted = market_distortion(fixture) is not None
    for h in fixture["horses"]:
        recent = h["recent_form"][:form_limit] if form_limit else h["recent_form"]
        form = " / ".join(recent)
        bracket = h.get("bracket")
        bracket_part = f"{bracket}枠・" if bracket else ""
        line = (
            f"{h['number']} {h['name']}（{bracket_part}{h['sex_age']}・{h['weight_kg']}kg・{h['jockey']}・{h['trainer']}厩舎）近走：{form}"
        )
        mkt = h.get("market") or {}
        if mkt.get("win_prob") is not None and not distorted:
            line += f" / 市場勝率{mkt['win_prob'] * 100:.1f}%（単勝{mkt.get('odds')}倍）"
        stats = person_stats_text(h)
        if stats:
            line += f" / {stats}"
        lines.append(line)
    return "\n".join(lines)


def parse_marks(text: str) -> dict[str, tuple[int, str]]:
    marks: dict[str, tuple[int, str]] = {}
    for mark, num, name in MARK_RE.findall(text or ""):
        marks.setdefault(mark, (int(num), name))
    return marks

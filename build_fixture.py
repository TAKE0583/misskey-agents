#!/usr/bin/env python3
"""Build orchestrator fixtures from netkeiba via keiba-scraping.

Replaces the JRDB plan (JRDB is a paid service): race entries come from
EntryPageScraper, recent form from the public shutuba_past page (parsed
with pandas.read_html), and results/payouts from ResultPageScraper.
Output matches the hand-built fixture schema used by orchestrator.py.

Usage (venv python):
  .venv/bin/python build_fixture.py --race-id 200906050810 \
      --edition 54 --grade G1 \
      --out fixtures/2009-arima-kinen.json \
      --results-out fixtures/2009-arima-kinen-results.json

NAR (regional racing) mode: add --nar. The scrapers then point at
nar.netkeiba.com; race info and the finish table are parsed by the
NAR-specific helpers below because the vendor parsers assume the JRA
page layout (payouts and entry rows work unmodified). NAR race IDs are
YYYY + venue(2) + MM + DD + R, e.g. the 2026 JBC Classic at Kanazawa
(venue 46) on Nov 3, race 9 = 202646110309.

Run with the project venv: system python lacks bs4/lxml/keiba-domain.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vendor" / "keiba-scraping"))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

import scraping.config as keiba_config  # noqa: E402
from scraping.config import ScrapingConfig  # noqa: E402
from scraping.entry_page import EntryPageScraper  # noqa: E402
from scraping.result_page import ResultPageScraper  # noqa: E402

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

JRA_RACE_URL = "https://race.netkeiba.com"
NAR_RACE_URL = "https://nar.netkeiba.com"

# NAR trainer affiliations are place names (first two chars of the stable
# string), outside the JRA validation set in keiba-scraping. Extending the
# shared set object lets EntryPageScraper validate NAR entries; JRA entries
# are unaffected because the added values never occur there.
NAR_AFFILIATIONS = {
    "北海", "門別", "岩手", "盛岡", "水沢", "金沢", "石川", "笠松", "岐阜",
    "愛知", "名古", "兵庫", "園田", "高知", "佐賀", "大井", "東京", "川崎",
    "神奈川", "船橋", "千葉", "浦和", "埼玉", "帯広",
}

FORM_RE = re.compile(
    r"(?P<y>\d{4})\.(?P<m>\d{2})\.(?P<d>\d{2})\s+(?P<venue>\S+)\s+"
    r"(?P<pos>\d+|降|除|中|取)\s+(?P<race>.+?)\s+"
    r"(?:(?P<grade>GI{1,3}|G[123]|J\.G[123]|Jpn[123]|OP|L)\s+)?"
    r"(?P<surf>芝|ダ|障)(?P<dist>\d+)"
)
GRADE_NORM = {"GI": "G1", "GII": "G2", "GIII": "G3", "G1": "G1", "G2": "G2", "G3": "G3"}


def parse_form_cell(cell: object) -> str | None:
    if not isinstance(cell, str):
        return None
    m = FORM_RE.search(cell.replace("\u3000", " "))
    if not m:
        return None
    g = m.groupdict()
    grade = GRADE_NORM.get(g["grade"] or "", g["grade"] or "")
    grade_part = f"（{grade}・" if grade else "（"
    return (
        f"{g['y']}-{g['m']}-{g['d']} {g['race']}"
        f"{grade_part}{g['venue']}{g['surf']}{g['dist']}m）{g['pos']}着"
    )


def fetch_recent_form(race_id: str, nar: bool = False) -> dict[int, list[str]]:
    host = NAR_RACE_URL if nar else JRA_RACE_URL
    url = f"{host}/race/shutuba_past.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.content.decode("utf-8", errors="replace")
    tables = pd.read_html(io.StringIO(html))
    main = tables[0]
    form: dict[int, list[str]] = {}
    run_cols = [c for c in main.columns if isinstance(c, str) and c.endswith("走")]
    for _, row in main.iterrows():
        try:
            number = int(row["馬番"])
        except (KeyError, TypeError, ValueError):
            continue
        runs = []
        for col in run_cols:
            parsed = parse_form_cell(row[col])
            if parsed:
                runs.append(parsed)
        form[number] = runs[:5]
    return form


def nar_race_info(soup: BeautifulSoup, race_id: str, field_size: int) -> dict:
    """Extract the same info dict shape as EntryPageScraper.get_race_info
    from a nar.netkeiba.com shutuba page, whose markup differs from JRA's."""
    name_el = soup.find(class_="RaceName")
    if name_el is None:
        raise ValueError("nar: RaceName not found")
    data01_el = soup.find("div", class_="RaceData01")
    data02_el = soup.find("div", class_="RaceData02")
    if data01_el is None or data02_el is None:
        raise ValueError("nar: RaceData01/02 not found")
    # No separator: nar splits the going value (e.g. 稍重) across text nodes,
    # and a " " separator would truncate it to 稍.
    d01 = data01_el.get_text("", strip=True)
    d02 = data02_el.get_text(" ", strip=True)

    def grab(pattern: str, text: str) -> str:
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"nar: pattern {pattern!r} not in {text!r}")
        return m.group(1)

    surf_dist = re.search(r"([芝ダ障])\s?(\d+)\s?m", d01)
    if not surf_dist:
        raise ValueError(f"nar: surface/distance not in {d01!r}")
    venue_m = re.search(r"\d+回\s+(\S+?)\s+\d+日目", d02)
    cond_m = re.search(r"日目\s+(.+)$", d02)
    conditions = cond_m.group(1).strip() if cond_m else ""
    conditions = re.split(r"\s+\d+頭\b", conditions)[0].strip()
    # The nar RaceName element glues the grade badge onto the name
    # (e.g. "JBCクラシックJpn1"); the grade is passed via --grade instead.
    name = re.sub(r"(Jpn[123]|G[123])$", "", name_el.get_text(strip=True))
    from datetime import date

    going = grab(r"馬場:([^\s/]+)", d01)
    if going == "稍":
        going = "稍重"  # nar abbreviates the going to one char in RaceData01
    return {
        "レース名": name,
        "日付": date(int(race_id[0:4]), int(race_id[6:8]), int(race_id[8:10])),
        "競馬場": venue_m.group(1) if venue_m else "",
        "発走時刻": grab(r"(\d{1,2}:\d{2})発走", d01),
        "芝ダ": surf_dist.group(1),
        "左右": grab(r"\((右|左)\)", d01),
        "内外": "",
        "距離": int(surf_dist.group(2)),
        "天候": grab(r"天候:([^\s/]+)", d01),
        "馬場": going,
        "競走条件": conditions,
        "頭数": field_size,
    }


def build_fixture(race_id: str, edition: int, grade: str, nar: bool = False) -> dict:
    if nar:
        keiba_config.VALID_AFFILIATIONS.update(NAR_AFFILIATIONS)
        cfg = ScrapingConfig(netkeiba_race_url=NAR_RACE_URL)
    else:
        cfg = ScrapingConfig()
    scraper = EntryPageScraper(race_id, cfg)
    entry = scraper.get_entry()
    if nar:
        info = nar_race_info(scraper.soup, race_id, len(entry))
    else:
        info = scraper.get_race_info().to_dict("records")[0]
    form = fetch_recent_form(race_id, nar=nar)

    course = f"{info['芝ダ']}・{info['左右']}"
    if info.get("内外"):
        course += f"・{info['内外']}"
    race = {
        "race_id": race_id,
        "edition": edition,
        "name": info["レース名"],
        "grade": grade,
        "date": info["日付"].isoformat(),
        "venue": f"{info['競馬場']}競馬場",
        "race_number": int(race_id[-2:]),
        "start_time": info["発走時刻"],
        "course": course,
        "distance_m": int(info["距離"]),
        "weather": info["天候"],
        "going": info["馬場"],
        "conditions": info["競走条件"],
        "field_size": int(info["頭数"]),
    }
    horses = []
    for _, row in entry.sort_values("馬番").iterrows():
        number = int(row["馬番"])
        horses.append({
            "bracket": int(row["枠"]),
            "number": number,
            "name": row["馬名"],
            "sex_age": f"{row['性別']}{int(row['年齢'])}",
            "weight_kg": float(row["斤量"]),
            "jockey": row["騎手"],
            "trainer": row["厩舎"],
            "recent_form": form.get(number, []),
        })
    return {"race": race, "horses": horses}


def nar_finish(race_id: str) -> list[dict]:
    """Parse the finish table from a nar result page with read_html; the
    vendor get_result() assumes JRA-only columns/ID links and fails on NAR."""
    url = f"{NAR_RACE_URL}/race/result.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.content.decode("utf-8", errors="replace")
    table = pd.read_html(io.StringIO(html))[0]
    table = table.rename(columns=lambda c: str(c).replace(" ", ""))
    finish = []
    for _, row in table.iterrows():
        pos = pd.to_numeric(row["着順"], errors="coerce")
        if pd.isna(pos):
            continue  # 競走中止・除外など
        margin = row.get("着差")
        finish.append({
            "position": int(pos),
            "number": int(row["馬番"]),
            "name": row["馬名"],
            "popularity": int(row["人気"]),
            "time": row["タイム"],
            "margin": "" if pd.isna(margin) else str(margin),
            "win_odds": float(row["単勝オッズ"]),
        })
    return sorted(finish, key=lambda f: f["position"])


def build_results(race_id: str, fixture: dict, nar: bool = False) -> dict:
    if nar:
        cfg = ScrapingConfig(netkeiba_race_url=NAR_RACE_URL)
    else:
        cfg = ScrapingConfig()
    scraper = ResultPageScraper(race_id, cfg)
    if nar:
        finish = nar_finish(race_id)
    else:
        result = scraper.get_result().sort_values("着順")
        finish = []
        for _, row in result.iterrows():
            if pd.isna(row["着順"]):
                continue  # 競走中止・除外など着順なしの馬は回顧の着順表に入れない
            margin = row.get("着差")
            finish.append({
                "position": int(row["着順"]),
                "number": int(row["馬番"]),
                "name": row["馬名"],
                "popularity": int(row["人気"]),
                "time": row["タイム"],
                "margin": "" if pd.isna(margin) else str(margin),
                "win_odds": float(row["単勝オッズ"]),
            })
    win = scraper.get_win_payoff().to_dict("records")[0]
    trio = scraper.get_trio_payoff().to_dict("records")[0]
    r = fixture["race"]
    return {
        "race_id": race_id,
        "race": f"第{r['edition']}回 {r['name']}（{r['grade']}）",
        "date": r["date"],
        "venue": r["venue"],
        "winning_time": finish[0]["time"],
        "race_summary": "",
        "finish": finish,
        "payouts_jpy": {
            "win": {str(int(win["単勝馬番_1"])): int(win["単勝払戻金_1"])},
            "trio": {
                "-".join(str(int(trio[f"3連複組番_1_{i}"])) for i in (1, 2, 3)): int(
                    trio["3連複払戻金_1"]
                )
            },
        },
        "sources": [f"netkeiba race_id={race_id} (keiba-scraping)"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--race-id", required=True)
    ap.add_argument("--edition", type=int, required=True)
    ap.add_argument("--grade", default="G1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--results-out", default=None)
    ap.add_argument("--nar", action="store_true", help="regional racing (nar.netkeiba.com)")
    args = ap.parse_args()

    fixture = build_fixture(args.race_id, args.edition, args.grade, nar=args.nar)
    Path(args.out).write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"fixture: {args.out} ({len(fixture['horses'])} horses)")
    time.sleep(1)
    if args.results_out:
        results = build_results(args.race_id, fixture, nar=args.nar)
        Path(args.results_out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        top = results["finish"][:3]
        print(
            "results:",
            ", ".join(f"{f['position']}着 {f['name']}" for f in top),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

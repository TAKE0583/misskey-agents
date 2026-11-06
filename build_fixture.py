#!/usr/bin/env python3
"""Build orchestrator fixtures from netkeiba via keiba-scraping.

Replaces the JRDB plan (JRDB is a paid service): race entries come from
EntryPageScraper, recent form from the public shutuba_past page (parsed
with pandas.read_html), and results/payouts from ResultPageScraper.
Each horse also gets a `market` block (current win odds + de-vigged
win probability + snapshot time) from the netkeiba odds API; before
betting opens the odds do not exist and the block is nulls, which the
orchestrator treats as "no market data". Output matches the hand-built
fixture schema used by orchestrator.py.

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
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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
# Jockey/trainer profile pages live on db.netkeiba.com for both JRA
# and NAR people (both shutuba pages link there).
DB_URL = "https://db.netkeiba.com"

# Person stats are cached per calendar day: the same jockeys and
# trainers recur across the day's venues, and their profile pages
# change at most daily. The cache file is runtime state, not a fixture
# (it is listed in .gitignore).
STATS_CACHE_PATH = Path(__file__).parent / "fixtures" / "person-stats-cache.json"

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


def parse_win_odds(payload: object) -> dict[int, float]:
    """Extract {horse number: win odds} from a netkeiba odds API payload.
    Two shapes are handled: the JRA shape (the same JSON the vendor
    odds parser reads: data.odds["1"] maps horse number ->
    [odds, ?, popularity]) and the NAR host shape (ary_odds maps
    zero-padded number -> {"Odds": ..., "Ninki": ...}). Anything
    unexpected — including the empty payload served before betting
    opens — yields an empty dict rather than an error."""
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    if isinstance(data, dict):
        tan = (data.get("odds") or {}).get("1") or {}
        out: dict[int, float] = {}
        for num, vals in tan.items():
            try:
                odds = float(vals[0])
                number = int(num)
            except (TypeError, ValueError, IndexError):
                continue
            if odds > 0:  # negative values mark scratched/excluded runners
                out[number] = odds
        if out:
            return out
    # NAR host shape (confirmed live 2026-11-06: api_get_nar_odds.html
    # answers with real odds, but in its own shape, not the JRA one):
    # {"status": "OK", "odds_status": "real",
    #  "ary_odds": {"01": {"Odds": "52.6", "Ninki": "10"}, ...}}
    ary = payload.get("ary_odds")
    if isinstance(ary, dict):
        out = {}
        for num, vals in ary.items():
            try:
                odds = float((vals or {}).get("Odds"))
                number = int(num)
            except (TypeError, ValueError, AttributeError):
                continue
            if odds > 0:
                out[number] = odds
        return out
    return {}


def devig_win_probs(odds_by_number: dict[int, float]) -> dict[int, float]:
    """De-vigged market win probabilities: implied probabilities
    (1 / odds) normalised to sum to 1 across the field. Odds of 1.0 or
    below carry no usable implied probability and are left out."""
    implied = {
        n: 1.0 / o for n, o in odds_by_number.items() if o and o > 1.0
    }
    total = sum(implied.values())
    if total <= 0:
        return {}
    return {n: v / total for n, v in implied.items()}


def fetch_win_odds(race_id: str, nar: bool = False) -> dict[int, float]:
    """Current win odds from the netkeiba odds API. Best-effort by
    design: odds legitimately do not exist yet for races priced before
    betting opens (e.g. a Sunday G1 built on Thursday), and a fixture
    must build fine without them — any failure returns {} and the
    caller stamps null market fields. The NAR host's API endpoint name
    could not be verified offline, so both candidate names are tried
    and whichever answers with data wins."""
    if nar:
        urls = [
            f"{NAR_RACE_URL}/api/api_get_nar_odds.html?race_id={race_id}&type=1",
            f"{NAR_RACE_URL}/api/api_get_jra_odds.html?race_id={race_id}&type=1",
        ]
    else:
        urls = [
            f"{JRA_RACE_URL}/api/api_get_jra_odds.html?race_id={race_id}&type=1"
        ]
    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            odds = parse_win_odds(resp.json())
        except Exception:  # noqa: BLE001 - odds are optional input
            continue
        if odds:
            return odds
    return {}


# ---------------------------------------------------------------------------
# Jockey / trainer performance data (db.netkeiba.com profile pages)
#
# Until 2026-11-05 the fixture carried people as bare names, so the AIs
# could only judge famous JRA names from baked-in knowledge. The
# profile pages publish per-jurisdiction season tables (年度別成績,
# split 中央 / 地方) for NAR people too, so both modes fetch from the
# same source. A per-venue breakdown exists only behind netkeiba's
# premium mask (the data_summary tables show '**' for non-members),
# so no venue split is attached. Identity comes from the shutuba
# hrefs, never from name matching: the vendor entry parser's own ID
# columns cannot be used because it slices the href assuming a
# trailing slash, which NAR hrefs lack (its NAR 騎手ID comes out as
# '/0559').
# ---------------------------------------------------------------------------

_PERSON_HREF_RE = {
    "jockey": re.compile(r"/jockey/result/recent/([0-9A-Za-z]+)/?$"),
    "trainer": re.compile(r"/trainer/result/recent/([0-9A-Za-z]+)/?$"),
}
# syusai (主催) query value on the profile page's 年度別 "more" links:
# absent = 中央 (JRA), 20 = 地方 (NAR). The third box (海外) has no
# such link and is not used.
_SCOPE_BY_SYUSAI = {"": "中央", "20": "地方"}
_SCOPES = ("中央", "地方")


def parse_person_ids(soup: BeautifulSoup) -> dict[int, dict[str, str | None]]:
    """Map horse number -> {jockey_id, trainer_id} from the shutuba
    page's own links. Rows are keyed by their Umaban cell, so the
    mapping cannot drift out of alignment with the entry table."""
    out: dict[int, dict[str, str | None]] = {}
    for row in soup.find_all("tr", class_="HorseList"):
        num_td = row.find("td", class_=re.compile(r"Umaban"))
        if num_td is None:
            continue
        try:
            number = int(num_td.get_text(strip=True))
        except ValueError:
            continue
        ids: dict[str, str | None] = {"jockey_id": None, "trainer_id": None}
        for kind, td_class in (("jockey", "Jockey"), ("trainer", "Trainer")):
            td = row.find("td", class_=td_class)
            anchor = td.find("a") if td is not None else None
            href = anchor.get("href") if anchor is not None else None
            if href:
                m = _PERSON_HREF_RE[kind].search(href)
                if m:
                    ids[f"{kind}_id"] = m.group(1)
        out[number] = ids
    return out


def _stats_row(cells: list[str]) -> dict | None:
    """One 年度別成績 body row -> counts, or None when the row is a
    placeholder (empty / '-' cells). Columns are positional: the
    layout (年度, 順位, 1着, 2着, 3着, 4着〜, 回数, ...) is shared by
    the jockey （騎乗回数） and trainer （出走回数） tables."""
    if len(cells) < 7:
        return None
    try:
        wins = int(cells[2].replace(",", ""))
        seconds = int(cells[3].replace(",", ""))
        thirds = int(cells[4].replace(",", ""))
        starts = int(cells[6].replace(",", ""))
    except ValueError:
        return None
    if starts <= 0:
        return None
    return {"starts": starts, "wins": wins, "seconds": seconds, "thirds": thirds}


def _with_rates(counts: dict, scope: str, year: int, career: bool) -> dict:
    starts = counts["starts"]
    w, s, t = counts["wins"], counts["seconds"], counts["thirds"]
    return {
        **counts,
        "win_rate": round(100.0 * w / starts, 1),
        "top2_rate": round(100.0 * (w + s) / starts, 1),
        "top3_rate": round(100.0 * (w + s + t) / starts, 1),
        "scope": scope,
        "year": year,
        "career": career,
    }


def _table_scope(table: BeautifulSoup) -> str | None:
    """Jurisdiction of one ResultsByYears table, from the syusai
    parameter of its ResultsBox 'more results' link."""
    box = table.find_parent("div", class_="ResultsBox")
    if box is None:
        return None
    for anchor in box.find_all("a", href=True):
        href = anchor["href"]
        if "result.html" not in href:
            continue
        syusai = (parse_qs(urlparse(href).query).get("syusai") or [""])[0]
        return _SCOPE_BY_SYUSAI.get(syusai)
    return None


def parse_person_page(html: str, year: int) -> dict[str, dict | None]:
    """Parse a db.netkeiba jockey/trainer profile page into
    {scope: stats-or-None} for 中央 and 地方. The season row for
    `year` wins; when a jurisdiction table has no row for the year,
    its 累計 (career) row is used and flagged career=True. Rates are
    computed from the counts (netkeiba's own percentages agree to
    rounding), so a mangled rate header cell cannot corrupt them."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="ResultsByYears")
    by_scope: dict[str, dict | None] = {scope: None for scope in _SCOPES}
    for idx, table in enumerate(tables):
        scope = _table_scope(table)
        if scope is None and idx < len(_SCOPES):
            scope = _SCOPES[idx]  # document order is 中央, 地方, 海外
        if scope not in by_scope:
            continue
        season: dict | None = None
        career: dict | None = None
        for tr in table.find_all("tr"):
            cells = [
                c.get_text(strip=True).replace("\u3000", " ")
                for c in tr.find_all(["th", "td"])
            ]
            if not cells:
                continue
            label = re.sub(r"\s+", "", cells[0])
            counts = _stats_row(cells)
            if counts is None:
                continue
            if label == str(year):
                season = counts
            elif label == "累計" and career is None:
                career = counts
        if season is not None:
            by_scope[scope] = _with_rates(season, scope, year, career=False)
        elif career is not None:
            by_scope[scope] = _with_rates(career, scope, year, career=True)
    return by_scope


def _decode_db_html(content: bytes) -> str:
    """db.netkeiba.com serves EUC-JP; fall back to UTF-8 if the
    年度 marker is missing (page format change)."""
    text = content.decode("euc-jp", errors="replace")
    if "年度" not in text:
        alt = content.decode("utf-8", errors="replace")
        if "年度" in alt:
            text = alt
    return text


def fetch_person_page_stats(
    kind: str, person_id: str, year: int
) -> dict[str, dict | None] | None:
    """Fetch + parse one profile page. Best-effort like the odds
    snapshot: any failure returns None and the caller degrades to
    name-only rendering for that person."""
    url = f"{DB_URL}/{kind}/{person_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception:  # noqa: BLE001 - stats are optional input
        return None
    time.sleep(0.3)  # keep the day's burst of profile fetches polite
    return parse_person_page(_decode_db_html(resp.content), year)


class PersonStatsCache:
    """Day-valid cache of parsed profile pages, keyed by person.

    One fetch per person per calendar day serves every race that day
    (both jurisdictions are parsed from the single page). A failed or
    empty fetch is cached too, so a db.netkeiba outage degrades the
    whole day to name-only instead of being retried per race.
    """

    def __init__(self, path: Path, today: str | None = None, fetcher=None):
        self.path = Path(path)
        self.today = today or date.today().isoformat()
        self.fetcher = fetcher or fetch_person_page_stats
        self.dirty = False
        self.data: dict = {}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data = loaded
        except (OSError, ValueError):
            self.data = {}

    def get(
        self, kind: str, person_id: str | None, scope: str, year: int
    ) -> dict | None:
        if not person_id:
            return None
        key = f"{kind}:{person_id}"
        entry = self.data.get(key)
        if (
            not isinstance(entry, dict)
            or entry.get("fetched_on") != self.today
            or entry.get("year") != year
        ):
            scopes = self.fetcher(kind, person_id, year)
            entry = {
                "fetched_on": self.today,
                "year": year,
                "scopes": scopes or {},
            }
            self.data[key] = entry
            self.dirty = True
        return (entry.get("scopes") or {}).get(scope)

    def save(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        self.dirty = False


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


def build_fixture(
    race_id: str,
    edition: int,
    grade: str,
    nar: bool = False,
    stats_cache: PersonStatsCache | None = None,
) -> dict:
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
    # Jockey/trainer season stats from the db.netkeiba profile pages,
    # keyed on the person IDs in the shutuba hrefs. Best-effort: a
    # person whose stats cannot be fetched keeps null stats and the
    # prompts render the name only, exactly as before this feature.
    person_ids = parse_person_ids(scraper.soup)
    stats_scope = "地方" if nar else "中央"
    stats_year = int(str(info["日付"])[:4])
    if stats_cache is None:
        stats_cache = PersonStatsCache(STATS_CACHE_PATH)
    # Market snapshot: current win odds -> de-vigged win probabilities.
    # Legitimately empty before betting opens; the nulls then degrade
    # every downstream consumer to the pre-market behaviour.
    win_odds = fetch_win_odds(race_id, nar=nar)
    market_probs = devig_win_probs(win_odds)
    snapshot_time = (
        time.strftime("%Y-%m-%dT%H:%M:%S%z") if win_odds else None
    )

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
        ids = person_ids.get(number) or {}
        horses.append({
            "bracket": int(row["枠"]),
            "number": number,
            "name": row["馬名"],
            "sex_age": f"{row['性別']}{int(row['年齢'])}",
            "weight_kg": float(row["斤量"]),
            "jockey": row["騎手"],
            "trainer": row["厩舎"],
            "jockey_id": ids.get("jockey_id"),
            "trainer_id": ids.get("trainer_id"),
            "jockey_stats": stats_cache.get(
                "jockey", ids.get("jockey_id"), stats_scope, stats_year
            ),
            "trainer_stats": stats_cache.get(
                "trainer", ids.get("trainer_id"), stats_scope, stats_year
            ),
            "recent_form": form.get(number, []),
            "market": {
                "odds": win_odds.get(number),
                "win_prob": market_probs.get(number),
                "snapshot_time": snapshot_time,
            },
        })
    stats_cache.save()
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


PAYOUT_TYPES = {
    # Japanese ticket type -> (payouts_jpy key, numbers per winning combo)
    "単勝": ("win", 1),
    "複勝": ("place", 1),
    "枠連": ("bracket", 2),
    "馬連": ("quinella", 2),
    "ワイド": ("wide", 2),
    "馬単": ("exacta", 2),
    "3連複": ("trio", 3),
    "3連単": ("trifecta", 3),
}
# Types whose combo key is order-insensitive (stored sorted); 馬単/3連単
# keep the finishing order. 枠単 (NAR only) is not a bettable type in the
# orchestrator and is skipped.
PAYOUT_SORTED = {"win", "place", "bracket", "quinella", "wide", "trio"}


def parse_payouts(soup: BeautifulSoup) -> dict:
    """Parse the full Payout_Detail_Table(s) of a result page into
    {type_key: {combo_key: yen_per_100_yen_ticket}}. Each table row packs
    all winning combos of one type into space-separated cells (組番 /
    払戻金 / 人気), so the numbers cell is chunked by the type's arity."""
    payouts: dict[str, dict[str, int]] = {}
    for table in soup.find_all("table", class_="Payout_Detail_Table"):
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if len(cells) < 3 or cells[0] not in PAYOUT_TYPES:
                continue
            key, arity = PAYOUT_TYPES[cells[0]]
            try:
                numbers = [int(x) for x in cells[1].split()]
                amounts = [
                    int(x.replace("円", "").replace(",", ""))
                    for x in cells[2].split()
                ]
            except ValueError:
                continue
            if len(numbers) % arity or len(numbers) // arity != len(amounts):
                continue  # unexpected shape; leave this type to the fallback
            entry: dict[str, int] = {}
            for i, amount in enumerate(amounts):
                combo = numbers[i * arity:(i + 1) * arity]
                if key in PAYOUT_SORTED:
                    combo = sorted(combo)
                entry["-".join(str(n) for n in combo)] = amount
            payouts[key] = entry
    return payouts


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
    payouts = parse_payouts(scraper.soup)
    if "win" not in payouts or "trio" not in payouts:
        # Fallback: the vendor getters for the two types the review flow
        # historically relied on.
        win = scraper.get_win_payoff().to_dict("records")[0]
        trio = scraper.get_trio_payoff().to_dict("records")[0]
        payouts.setdefault(
            "win", {str(int(win["単勝馬番_1"])): int(win["単勝払戻金_1"])}
        )
        payouts.setdefault(
            "trio",
            {
                "-".join(
                    str(int(trio[f"3連複組番_1_{i}"])) for i in (1, 2, 3)
                ): int(trio["3連複払戻金_1"])
            },
        )
    r = fixture["race"]
    return {
        "race_id": race_id,
        "race": f"第{r['edition']}回 {r['name']}（{r['grade']}）",
        "date": r["date"],
        "venue": r["venue"],
        "winning_time": finish[0]["time"],
        "race_summary": "",
        "finish": finish,
        "payouts_jpy": payouts,
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
    with_odds = sum(
        1 for h in fixture["horses"] if (h.get("market") or {}).get("odds")
    )
    if with_odds:
        print(f"market odds: snapshot for {with_odds} horses")
    else:
        print("market odds: not available yet (betting not open?)")
    with_stats = sum(
        1 for h in fixture["horses"]
        if h.get("jockey_stats") or h.get("trainer_stats")
    )
    if with_stats:
        print(f"person stats: season records attached for {with_stats} horses")
    else:
        print("person stats: not available (prompts render names only)")
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

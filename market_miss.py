#!/usr/bin/env python3
"""Market-miss analysis: conditional calibration of the betting market.

Concept (user-approved 2026-11-06): the closing market is a strong,
money-weighted aggregate forecast. Instead of asking whether a model
"beats the odds" overall, this tool asks WHERE the market's own
probabilities are systematically off: runners are grouped by a factor
(popularity band, jockey season win rate, venue, field size, ...) and
each bucket compares the market's de-vigged win probability against
what actually happened (expected wins = sum of market probs vs actual
wins, plus a Harville-derived top-3 comparison). A bucket whose actual
count sits far above expectation over an adequate sample is a
*candidate* market bias; everything else is reported as noise-level.
This is statistics, not story-generation: no bucket is called an edge
without n, expected, actual, ratio and a significance indication.

Data sources, in priority order per race:
  1. CLOSING win odds from the results fixture's finish table
     (every finisher carries 単勝オッズ) — de-vigged per race.
  2. The prediction fixture's market snapshot (prediction-time odds,
     de-vigged by build_fixture) — only when closing odds are
     unavailable, and the race is labelled accordingly.

Offline only: this tool never posts, never touches runs/ state, the
points ledger or the A/B ledger. The optional backfill subcommand
builds fixture/results pairs for past races with build_fixture's own
machinery and caches them under data/market-miss/ (git-ignored).

Usage (venv python):
  .venv/bin/python market_miss.py analyze [--json out.json]
  .venv/bin/python market_miss.py backfill --since 2026-10-10 \
      --until 2026-11-05 [--limit 130]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent
FIXTURES_DIR = REPO / "fixtures"
DATA_DIR = REPO / "data" / "market-miss"

JRA_VENUES = {"札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"}
NAR_VENUES = {
    "門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎", "名古屋",
    "笠松", "金沢", "園田", "姫路", "高知", "佐賀", "帯広",
}

MIN_RELIABLE_N = 30  # below this a bucket is flagged unreliable


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def devig_win_probs(odds_by_number: dict[int, float]) -> dict[int, float]:
    """De-vigged win probabilities from a {number: odds} map.

    Same normalisation as build_fixture.devig_win_probs, except that
    odds of exactly 1.0 are kept: at prediction time a 1.0 quote is a
    placeholder, but in a results table it is a real money-back price
    (it happens in NAR) and dropping the horse would corrupt the
    calibration accounting."""
    implied = {
        n: 1.0 / o for n, o in odds_by_number.items()
        if isinstance(o, (int, float)) and o >= 1.0
    }
    total = sum(implied.values())
    if total <= 0:
        return {}
    return {n: v / total for n, v in implied.items()}


def harville_top3_probs(win_probs: dict[int, float]) -> dict[int, float]:
    """P(runner finishes in the top 3) under the Harville model:
    finishing order drawn sequentially with probabilities proportional
    to the win probabilities. Exact O(n^3) enumeration; fields here are
    at most ~18 runners. The results sum to 3 (three top-3 slots)."""
    nums = list(win_probs)
    p = win_probs
    out: dict[int, float] = {}
    for i in nums:
        pi = p[i]
        total = pi  # wins outright
        for j in nums:  # j wins, i second
            if j == i:
                continue
            denom = 1.0 - p[j]
            if denom <= 0:
                continue
            total += p[j] * pi / denom
            for k in nums:  # j wins, k second, i third
                if k == i or k == j:
                    continue
                denom2 = 1.0 - p[j] - p[k]
                if denom2 <= 0:
                    continue
                total += p[j] * (p[k] / denom) * (pi / denom2)
        out[i] = total
    return out


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def jurisdiction_of(venue_name: str) -> str:
    base = venue_name.replace("競馬場", "")
    if base in JRA_VENUES:
        return "JRA"
    if base in NAR_VENUES:
        return "NAR"
    return "?"


def build_rows(fixture: dict, results: dict) -> tuple[list[dict], dict]:
    """Per-runner rows for one (fixture, results) pair, plus race meta.
    Returns ([], meta-with-skip-reason) when no usable market exists."""
    race = fixture.get("race") or {}
    finish = results.get("finish") or []
    meta = {
        "race_id": results.get("race_id") or race.get("race_id"),
        "date": results.get("date") or race.get("date"),
        "venue": results.get("venue") or race.get("venue"),
        "race_name": results.get("race") or race.get("name"),
    }
    meta["jurisdiction"] = jurisdiction_of(meta["venue"] or "")
    horses = {h["number"]: h for h in fixture.get("horses") or []}

    closing = {
        f["number"]: f.get("win_odds")
        for f in finish
        if isinstance(f.get("win_odds"), (int, float)) and f["win_odds"] >= 1.0
    }
    probs: dict[int, float] = {}
    source = None
    if finish and len(closing) == len(finish):
        probs = devig_win_probs(closing)
        source = "closing"
    if not probs:
        snap = {
            h["number"]: (h.get("market") or {}).get("win_prob")
            for h in horses.values()
            if isinstance((h.get("market") or {}).get("win_prob"), (int, float))
        }
        if finish and len(snap) == len(finish):
            total = sum(snap.values())
            if total > 0:
                probs = {n: v / total for n, v in snap.items()}
                source = "snapshot"
    if not probs:
        meta["skip"] = "no complete market (closing odds or snapshot)"
        return [], meta
    meta["odds_source"] = source
    top3 = harville_top3_probs(probs)

    course = race.get("course") or ""
    rows = []
    for f in finish:
        num = f["number"]
        h = horses.get(num) or {}
        js = h.get("jockey_stats") or {}
        ts = h.get("trainer_stats") or {}
        rows.append({
            "race_id": meta["race_id"],
            "date": meta["date"],
            "venue": meta["venue"],
            "jurisdiction": meta["jurisdiction"],
            "number": num,
            "name": f.get("name") or h.get("name"),
            "popularity": f.get("popularity"),
            "position": f.get("position"),
            "won": f.get("position") == 1,
            "top3": isinstance(f.get("position"), int) and f["position"] <= 3,
            "market_prob": probs.get(num),
            "top3_prob": top3.get(num),
            "field_size": len(finish),
            "distance_m": race.get("distance_m"),
            "going": race.get("going"),
            "surface": course.split("・")[0] if course else None,
            "jockey_win_rate": js.get("win_rate"),
            "jockey_top3_rate": js.get("top3_rate"),
            "trainer_win_rate": ts.get("win_rate"),
            "trainer_top3_rate": ts.get("top3_rate"),
        })
    return rows, meta


def discover_pairs(extra_dirs: list[Path] | None = None) -> list[tuple[Path, Path, str]]:
    """(fixture_path, results_path, origin) for every usable pair:
    the repo's fixtures/ stems and the backfill cache layout."""
    pairs: list[tuple[Path, Path, str]] = []
    if FIXTURES_DIR.is_dir():
        for res in sorted(FIXTURES_DIR.glob("*-results.json")):
            fx = res.with_name(res.name.replace("-results.json", ".json"))
            if fx.exists():
                pairs.append((fx, res, "repo-fixtures"))
    for base in extra_dirs or []:
        fdir, rdir = base / "fixtures", base / "results"
        if fdir.is_dir() and rdir.is_dir():
            for fx in sorted(fdir.glob("*.json")):
                res = rdir / fx.name
                if res.exists():
                    pairs.append((fx, res, "backfill"))
    return pairs


def load_all_rows(
    extra_dirs: list[Path] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """(rows, race_metas, skipped_metas) across every discovered pair."""
    rows: list[dict] = []
    metas: list[dict] = []
    skipped: list[dict] = []
    for fx_path, res_path, origin in discover_pairs(extra_dirs):
        try:
            fixture = json.loads(fx_path.read_text(encoding="utf-8"))
            results = json.loads(res_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        r, meta = build_rows(fixture, results)
        meta["origin"] = origin
        if r:
            rows.extend(r)
            metas.append(meta)
        else:
            skipped.append(meta)
    return rows, metas, skipped


# ---------------------------------------------------------------------------
# Bucketing + aggregation
# ---------------------------------------------------------------------------

def _band(value, cuts: list[tuple[float, str]], top_label: str):
    """cuts: ascending (upper_bound_exclusive, label) pairs."""
    if value is None:
        return None
    for bound, label in cuts:
        if value < bound:
            return label
    return top_label


FACTORS: dict[str, tuple[str, callable]] = {}


def _factor(name, label):
    def deco(fn):
        FACTORS[name] = (label, fn)
        return fn
    return deco


@_factor("popularity", "人気")
def _f_popularity(r):
    p = r["popularity"]
    if p is None:
        return None
    if p == 1:
        return "1番人気"
    if p <= 3:
        return "2-3番人気"
    if p <= 6:
        return "4-6番人気"
    if p <= 9:
        return "7-9番人気"
    return "10番人気以下"


@_factor("prob_band", "市場勝率帯")
def _f_prob_band(r):
    return _band(r["market_prob"], [
        (0.03, "~3%"), (0.08, "3-8%"), (0.15, "8-15%"), (0.30, "15-30%"),
    ], "30%~")


@_factor("jockey_win", "騎手・今季勝率")
def _f_jockey_win(r):
    return _band(r["jockey_win_rate"], [
        (5, "~5%"), (10, "5-10%"), (15, "10-15%"), (20, "15-20%"),
    ], "20%~")


@_factor("jockey_top3", "騎手・今季3着内率")
def _f_jockey_top3(r):
    return _band(r["jockey_top3_rate"], [
        (20, "~20%"), (30, "20-30%"), (40, "30-40%"), (50, "40-50%"),
    ], "50%~")


@_factor("trainer_win", "調教師・今季勝率")
def _f_trainer_win(r):
    return _band(r["trainer_win_rate"], [
        (5, "~5%"), (10, "5-10%"), (15, "10-15%"),
    ], "15%~")


@_factor("trainer_top3", "調教師・今季3着内率")
def _f_trainer_top3(r):
    return _band(r["trainer_top3_rate"], [
        (15, "~15%"), (25, "15-25%"), (35, "25-35%"),
    ], "35%~")


@_factor("jurisdiction", "主催")
def _f_jurisdiction(r):
    return r["jurisdiction"] if r["jurisdiction"] in ("JRA", "NAR") else None


@_factor("venue", "競馬場")
def _f_venue(r):
    return (r["venue"] or "").replace("競馬場", "") or None


@_factor("field_size", "頭数")
def _f_field_size(r):
    return _band(r["field_size"], [(10, "9頭以下"), (13, "10-12頭"), (16, "13-15頭")], "16頭~")


@_factor("distance", "距離帯")
def _f_distance(r):
    d = r["distance_m"]
    if d is None:
        return None
    return _band(d, [(1300, "~1200m"), (1700, "1300-1600m"), (2100, "1700-2000m")], "2100m~")


@_factor("going", "馬場状態")
def _f_going(r):
    return {"不": "不良", "稍": "稍重"}.get(r["going"], r["going"]) or None


@_factor("surface", "芝/ダート")
def _f_surface(r):
    return {"芝": "芝", "ダ": "ダート"}.get(r["surface"] or "")


def _bucket_stats(rows: list[dict]) -> dict:
    n = len(rows)
    exp_w = sum(r["market_prob"] for r in rows)
    act_w = sum(1 for r in rows if r["won"])
    var_w = sum(r["market_prob"] * (1 - r["market_prob"]) for r in rows)
    exp_t = sum(r["top3_prob"] for r in rows)
    act_t = sum(1 for r in rows if r["top3"])
    var_t = sum(r["top3_prob"] * (1 - r["top3_prob"]) for r in rows)

    def ratio(act, exp):
        return round(act / exp, 3) if exp > 0 else None

    def z(act, exp, var):
        return round((act - exp) / math.sqrt(var), 2) if var > 0 else None

    return {
        "n": n,
        "log_loss": (round(overall_log_loss(rows), 4) if rows else None),
        "exp_wins": round(exp_w, 2),
        "act_wins": act_w,
        "win_ratio": ratio(act_w, exp_w),
        "win_z": z(act_w, exp_w, var_w),
        "exp_top3": round(exp_t, 2),
        "act_top3": act_t,
        "top3_ratio": ratio(act_t, exp_t),
        "top3_z": z(act_t, exp_t, var_t),
        "unreliable": n < MIN_RELIABLE_N,
    }


# Canonical display order for banded factors (strongest first).
_BUCKET_ORDERS: dict[str, list[str]] = {
    "popularity": ["1番人気", "2-3番人気", "4-6番人気", "7-9番人気", "10番人気以下"],
    "prob_band": ["30%~", "15-30%", "8-15%", "3-8%", "~3%"],
    "jockey_win": ["20%~", "15-20%", "10-15%", "5-10%", "~5%"],
    "jockey_top3": ["50%~", "40-50%", "30-40%", "20-30%", "~20%"],
    "trainer_win": ["15%~", "10-15%", "5-10%", "~5%"],
    "trainer_top3": ["35%~", "25-35%", "15-25%", "~15%"],
    "jurisdiction": ["JRA", "NAR"],
    "field_size": ["9頭以下", "10-12頭", "13-15頭", "16頭~"],
    "distance": ["~1200m", "1300-1600m", "1700-2000m", "2100m~"],
    "going": ["良", "稍重", "重", "不良"],
    "surface": ["芝", "ダート"],
}


def aggregate(rows: list[dict], factor: str) -> list[tuple[str, dict]]:
    """Bucket table for one factor, in canonical band order (venues by
    descending n). Rows whose factor value is None are excluded."""
    _, fn = FACTORS[factor]
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        key = fn(r)
        if key is None:
            continue
        buckets.setdefault(key, []).append(r)
    items = [(k, _bucket_stats(v)) for k, v in buckets.items()]
    order = _BUCKET_ORDERS.get(factor)
    if order:
        rank = {label: i for i, label in enumerate(order)}
        items.sort(key=lambda kv: rank.get(kv[0], len(rank)))
    elif factor == "venue":
        items.sort(key=lambda kv: -kv[1]["n"])
    return items


def overall_log_loss(rows: list[dict]) -> float | None:
    """Mean binary log-loss of the market win probabilities."""
    if not rows:
        return None
    eps = 1e-6
    total = 0.0
    for r in rows:
        p = min(max(r["market_prob"], eps), 1 - eps)
        total += -(math.log(p) if r["won"] else math.log(1 - p))
    return total / len(rows)


def analyze(extra_dirs: list[Path] | None = None) -> dict:
    rows, metas, skipped = load_all_rows(extra_dirs)
    return {
        "races": metas,
        "skipped_races": skipped,
        "n_rows": len(rows),
        "log_loss": overall_log_loss(rows),
        "total_exp_wins": round(sum(r["market_prob"] for r in rows), 2),
        "total_act_wins": sum(1 for r in rows if r["won"]),
        "factors": {
            name: aggregate(rows, name) for name in FACTORS
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Backfill: discover past main races and build pairs via build_fixture
# ---------------------------------------------------------------------------

_GRADE_RANKS: list[tuple[str, int]] = [
    (r"Jpn1|JpnI\b|G1|Ｇ１", 100), (r"Jpn2|G2", 99), (r"Jpn3|G3", 98),
    (r"H1|S1", 97), (r"H2|S2", 96), (r"H3|S3", 95),
    (r"重賞", 94), (r"オープン|OP\b", 88),
    (r"3勝クラス", 70), (r"2勝クラス", 60), (r"1勝クラス", 50),
    (r"未勝利|新馬", 10),
]
_CLASS_RE = re.compile(r"\(([A-C])(\d)?\)")


def class_rank(item_text: str) -> int:
    """Rough class rank of one race-list item, for main-race picking:
    graded badges first, then NAR A/B/C class bands, then JRA win
    classes. 0 when nothing matches."""
    rank = 0
    for pattern, value in _GRADE_RANKS:
        if re.search(pattern, item_text):
            rank = max(rank, value)
    m = _CLASS_RE.search(item_text)
    if m:
        base = {"A": 85, "B": 75, "C": 65}[m.group(1)]
        digit = int(m.group(2)) if m.group(2) else 3
        rank = max(rank, base - (digit - 1))
    return rank


def parse_race_list(html: str) -> dict[str, list[tuple[str, str]]]:
    """race_list_sub HTML -> {venue base name: [(race_id, item text)]}.
    Shared by the JRA and NAR hosts (same list markup)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, list[tuple[str, str]]] = {}
    # One dl.RaceList_DataList per venue inside the day's RaceList_Box;
    # its dt header carries "○○競馬場TOP ...". Fall back to treating
    # each RaceList_Box as a venue block if the dl layout is absent.
    blocks = soup.select("dl.RaceList_DataList") or soup.select(".RaceList_Box")
    for block in blocks:
        head_text = (block.find("dt") or block).get_text(" ", strip=True)
        m = re.search(r"(\S+?)競馬場", head_text) or re.search(
            r"^\d+回\s+(\S+?)\s+\d+日目", head_text)
        if not m:
            continue
        venue = m.group(1)
        items: list[tuple[str, str]] = []
        for li in block.select("li.RaceList_DataItem"):
            anchor = li.find("a", href=re.compile(r"race_id="))
            if anchor is None:
                continue
            rid = re.search(r"race_id=(\d+)", anchor.get("href", ""))
            if rid:
                text = li.get_text(" ", strip=True)
                # The JRA list shows grades only as CSS icon classes
                # (Icon_GradeType*), never as text; surface them as a
                # 重賞 token so class_rank can see them.
                if li.find(class_=re.compile(r"Icon_GradeType")):
                    text += " 重賞"
                items.append((rid.group(1), text))
        if items:
            out[venue] = items
    return out


def pick_main(items: list[tuple[str, str]]) -> tuple[str, str] | None:
    """The venue-day's main race: highest class rank, latest race on
    ties (race number is the race_id's last two digits)."""
    if not items:
        return None
    return max(items, key=lambda it: (class_rank(it[1]), int(it[0][-2:])))


def discover_mains(day: str, nar: bool) -> dict[str, tuple[str, str]]:
    """{venue: (race_id, item text)} of main races for one date
    (YYYY-MM-DD). Best-effort: any failure means an empty dict."""
    import requests

    import build_fixture as bf

    host = bf.NAR_RACE_URL if nar else bf.JRA_RACE_URL
    url = f"{host}/top/race_list_sub.html?kaisai_date={day.replace('-', '')}"
    try:
        resp = requests.get(url, headers=bf.HEADERS, timeout=30)
        resp.raise_for_status()
        html = resp.content.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - discovery is best-effort
        return {}
    mains: dict[str, tuple[str, str]] = {}
    for venue, items in parse_race_list(html).items():
        main = pick_main(items)
        if main:
            mains[venue] = main
    return mains


def _known_race_ids() -> set[str]:
    known: set[str] = set()
    if FIXTURES_DIR.is_dir():
        for res in FIXTURES_DIR.glob("*-results.json"):
            try:
                known.add(json.loads(res.read_text(encoding="utf-8"))["race_id"])
            except (OSError, ValueError, KeyError):
                continue
    rdir = DATA_DIR / "results"
    if rdir.is_dir():
        known.update(p.stem for p in rdir.glob("*.json"))
    return known


def backfill(since: str, until: str, limit: int = 130) -> dict:
    """Build (fixture, results) pairs for past main races into
    DATA_DIR. Polite: sleeps between page fetches and race builds.
    Returns a summary dict; per-race failures are recorded, not fatal."""
    from datetime import date as _date, timedelta

    import build_fixture as bf

    (DATA_DIR / "fixtures").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "results").mkdir(parents=True, exist_ok=True)
    known = _known_race_ids()
    stats_cache = bf.PersonStatsCache(bf.STATS_CACHE_PATH)
    summary = {"built": [], "failed": [], "skipped_known": 0}

    day = _date.fromisoformat(since)
    last = _date.fromisoformat(until)
    while day <= last and len(summary["built"]) < limit:
        for nar in (True, False):
            if len(summary["built"]) >= limit:
                break
            mains = discover_mains(day.isoformat(), nar)
            time.sleep(1)
            for venue, (race_id, text) in sorted(mains.items()):
                if len(summary["built"]) >= limit:
                    break
                if venue == "帯広":
                    continue  # ban'ei is out of scope for this system
                if race_id in known:
                    summary["skipped_known"] += 1
                    continue
                try:
                    fixture = bf.build_fixture(
                        race_id, 1, "", nar=nar, stats_cache=stats_cache
                    )
                    results = bf.build_results(race_id, fixture, nar=nar)
                except Exception as exc:  # noqa: BLE001 - record and move on
                    summary["failed"].append(
                        {"race_id": race_id, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    continue
                (DATA_DIR / "fixtures" / f"{race_id}.json").write_text(
                    json.dumps(fixture, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (DATA_DIR / "results" / f"{race_id}.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                known.add(race_id)
                summary["built"].append(race_id)
                print(f"built {race_id} ({venue}, {'NAR' if nar else 'JRA'})",
                      flush=True)
                time.sleep(1)
        day += timedelta(days=1)
    stats_cache.save()
    return summary


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------

def _fmt_table(label: str, table: list[tuple[str, dict]]) -> str:
    lines = [f"### {label}"]
    lines.append("bucket | n | exp勝 | 実勝 | 比率 | z | exp3着内 | 実3着内 | 比率 | z")
    for bucket, s in table:
        flag = " ⚠n<30" if s["unreliable"] else ""
        lines.append(
            f"{bucket} | {s['n']} | {s['exp_wins']} | {s['act_wins']} | "
            f"{s['win_ratio']} | {s['win_z']} | {s['exp_top3']} | "
            f"{s['act_top3']} | {s['top3_ratio']} | {s['top3_z']}{flag}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    an = sub.add_parser("analyze", help="aggregate all known pairs")
    an.add_argument("--json", default=None, help="write structured result JSON")
    bk = sub.add_parser("backfill", help="build past main-race pairs")
    bk.add_argument("--since", required=True)
    bk.add_argument("--until", required=True)
    bk.add_argument("--limit", type=int, default=130)
    args = ap.parse_args()

    if args.cmd == "backfill":
        summary = backfill(args.since, args.until, args.limit)
        print(json.dumps(
            {"built": len(summary["built"]),
             "failed": summary["failed"],
             "skipped_known": summary["skipped_known"]},
            ensure_ascii=False, indent=2))
        return 0

    result = analyze([DATA_DIR])
    print(f"races: {len(result['races'])}  runners: {result['n_rows']}  "
          f"skipped: {len(result['skipped_races'])}")
    print(f"market log-loss (win): {result['log_loss']}")
    print(f"expected wins {result['total_exp_wins']} vs actual "
          f"{result['total_act_wins']}")
    for name, (label, _fn) in FACTORS.items():
        print()
        print(_fmt_table(label, result["factors"][name]))
    if args.json:
        out = {k: v for k, v in result.items() if k != "rows"}
        Path(args.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

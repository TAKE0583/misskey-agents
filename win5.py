"""WIN5 tickets for the conclusion account (2026-11-08 added).

Once the daily batch has predicted the day's JRA races, this module can
turn the per-leg conclusion aggregates into one WIN5 ticket for the
conclusion account (@shikai): JRA designates 5 races on weekend days,
and a ticket picks one or more horses per leg; every leg's winner must
be among the picks for the ticket to pay.

Design (user-approved 2026-11-08, conclusion account only, weekends
only):

- Target races come from JRA's official WIN5 list
  (https://www.jra.go.jp/kouza/win5/info/racelist.html): date, sales
  deadline and the 5 legs as venue + race number. A date that is not a
  Saturday/Sunday, or that JRA does not list, has no WIN5.
- Legs are resolved by race_id, never by filename stem: the fixture
  files under fixtures/ are indexed by their own race meta
  (date, venue, race number) -> race_id, and live run states under
  runs/ are indexed by race_id. A leg is usable only when its live
  state completed AND carries the conclusion gate's aggregate
  probability vector (state["conclusion_gate"]["aggregate_probs"],
  the exact vector the conclusion gate itself uses). This is what
  lets the 2026-11-15 Kyoto 11R leg resolve to the QE Cup state
  (fixtures/2026-elizabeth-cup.json) instead of a stem guess.
- Ticket generation (allocate_win5): budget is a flat 2,000 points
  (decoupled scoreboard, 2026-11-08 — never derived from the
  conclusion account's balance) at 100 points per combination.
  Every leg starts with its top-probability horse; a greedy allocator
  then repeatedly adds the pick that buys the most log(hit
  probability) per extra combination, at most 4 picks per leg, while
  the total stake fits the budget. Hit probability is the product of
  the legs' covered aggregate mass.
- Generation only happens before the official sales deadline; a
  ticket for a date is generated once and never regenerated.
- Settlement reads the official result from netkeiba's WIN5 page
  (https://race.netkeiba.com/top/win5.html?date=YYYYMMDD): the 5
  winning numbers and the payout per ¥100. Winners are cross-checked
  against the legs' results fixtures when those exist. Settlement
  goes through points.py under race_id "win5-YYYY-MM-DD", so it is
  idempotent per date, and it runs BEFORE the settlement note posts,
  matching the review flow's discipline. A winning ticket whose
  payout cannot be fetched/parised fails loudly and settles nothing.

Known limitation — no scratch refunds: real WIN5 refunds the
combinations that include a horse scratched (出走取消) or excluded
(競走除外) before the race. The legs' results fixtures do not expose
scratches at all (build_fixture.py keeps only finishers; a pulled-up
horse (競走中止), which is NOT refunded, is indistinguishable from a
scratched one), and the per-race settlement has no refund concept
either. WIN5 settlement here is therefore win/lose only; if a picked
horse is scratched, its combinations simply lose. Revisit if the
results fixtures ever record scratch status.

CLI (mirrors orchestrator.py's mode flags; default is dry-run):

  .venv/bin/python win5.py run    --date 2026-11-15 [--live]
  .venv/bin/python win5.py settle --date 2026-11-15 [--live]

State lives in runs/win5-YYYY-MM-DD-state.json (live) and
runs/dryrun/win5-YYYY-MM-DD-state.json (dry-run), under a per-date
flock. Dry-run never posts and never touches any points ledger;
generation reads no ledger at all (the budget is a flat constant),
and only live settlement writes to the ledger.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

import points as points_mod

JST = timezone(timedelta(hours=9))

RACELIST_URL = "https://www.jra.go.jp/kouza/win5/info/racelist.html"
WIN5_RESULT_URL = "https://race.netkeiba.com/top/win5.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

ACCOUNT = "shikai"  # the conclusion account (orchestrator.MC_USERNAME)
BUDGET_POINTS = 2000  # flat per-day WIN5 budget, balance-independent
UNIT_POINTS = 100
MAX_PICKS_PER_LEG = 4
N_LEGS = 5

WEEKDAY_JA = "月火水木金土日"  # index = datetime.weekday()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def now_jst() -> datetime:
    return datetime.now(JST)


# ------------------------------------------------------------------
# JRA official WIN5 list (racelist) parsing
# ------------------------------------------------------------------

def parse_yen(text: str) -> int:
    """Parse a JRA/netkeiba yen amount: '713万2610円', '1億2345万6789円',
    '18,586,810円', '8,540円', '0円' -> int yen."""
    t = text.replace(",", "").replace(" ", "").strip()
    m = re.fullmatch(r"(?:(\d+)億)?(?:(\d+)万)?(\d+)円", t)
    if not m:
        raise ValueError(f"cannot parse yen amount: {text!r}")
    oku, man, rest = (int(g) if g else 0 for g in m.groups())
    return oku * 100_000_000 + man * 10_000 + rest


def parse_racelist(html: str, target: datetime.date) -> dict | None:
    """Find `target` in JRA's WIN5 list page.

    Returns {"date", "weekday", "deadline": "HH:MM",
    "deadline_dt": aware datetime (JST),
    "legs": [{"venue": str, "race_number": int}, ...]} or None when the
    date is not listed. The page is an annual list (its <title> names
    the coverage year, e.g. '2026年12月27日までのWIN5対象レース');
    rows carry month/day + weekday only, so a row matches when its
    month/day in the coverage year IS the target date and the printed
    weekday agrees — the weekday check keeps a stale/next-year page
    from silently matching."""
    tm = re.search(
        r"(\d{4})年\d{1,2}月\d{1,2}日(?:（.曜）)?までのWIN5対象レース", html
    )
    if not tm:
        raise ValueError("racelist page: coverage-year title not found")
    year = int(tm.group(1))
    for row in re.findall(r"<tr>(.*?)</tr>", html, re.S):
        dm = re.search(r"(\d{1,2})月(\d{1,2})日", row)
        if not dm:
            continue
        month, day = int(dm.group(1)), int(dm.group(2))
        if (month, day) != (target.month, target.day) or year != target.year:
            continue
        ym = re.search(r"（(.)曜）", row)
        if ym and ym.group(1) != WEEKDAY_JA[target.weekday()]:
            continue  # same month/day but a different year's calendar
        cm = re.search(
            r'<span class="close">.*?<strong>(\d{1,2})時(\d{1,2})分</strong>',
            row, re.S,
        )
        legs = []
        for cell in re.findall(r'<span class="race">([^<]+)</span>', row):
            lm = re.fullmatch(r"(.+?)(\d{1,2})R", cell.strip())
            if lm:
                legs.append({
                    "venue": lm.group(1),
                    "race_number": int(lm.group(2)),
                })
        if not cm or len(legs) != N_LEGS:
            raise ValueError(
                f"racelist row for {target} is malformed "
                f"(deadline={'yes' if cm else 'missing'}, legs={len(legs)})"
            )
        hh, mm = int(cm.group(1)), int(cm.group(2))
        return {
            "date": target.isoformat(),
            "weekday": WEEKDAY_JA[target.weekday()],
            "deadline": f"{hh:02d}:{mm:02d}",
            "deadline_dt": datetime(
                year, month, day, hh, mm, tzinfo=JST
            ),
            "legs": legs,
        }
    return None


def fetch_racelist() -> str:
    resp = requests.get(RACELIST_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.content.decode("cp932", errors="replace")


def win5_day(target: datetime.date, html: str | None = None) -> dict | None:
    """The day's WIN5 info, or None: weekends only (user's scope), and
    only when JRA lists the date."""
    if target.weekday() not in (5, 6):
        return None
    if html is None:
        html = fetch_racelist()
    return parse_racelist(html, target)


# ------------------------------------------------------------------
# netkeiba WIN5 result parsing (settlement source)
# ------------------------------------------------------------------

def parse_win5_result(html: str) -> dict:
    """Parse netkeiba's WIN5 page for one date.

    Returns {"winners": [int, ...] (leg order, complete only when 5),
    "payout_yen": int | None (per ¥100, final only),
    "tickets": int | None (的中票数）, "estimated": bool (page still
    shows 想定払い戻し = racing in progress, not final),
    "leg_race_ids": [str, ...] (when the leg anchors parse cleanly)}.
    """
    out: dict = {
        "winners": [], "payout_yen": None, "tickets": None,
        "estimated": "想定払い戻し" in html, "leg_race_ids": [],
    }
    for tb in re.findall(
        r'<table class="RaceCommon_Table Win5_Table".*?</table>', html, re.S
    ):
        if "的中馬番" in tb:
            out["winners"] = [
                int(n) for n in re.findall(r'<li class="w5[^"]*">(\d+)</li>', tb)
            ]
        pm = re.search(r"払戻金</th>\s*<td>([^<]+)</td>", tb)
        if pm:
            out["payout_yen"] = parse_yen(pm.group(1))
        tm2 = re.search(r"的中票数</th>\s*<td>([\d,]+)票</td>", tb)
        if tm2:
            out["tickets"] = int(tm2.group(1).replace(",", ""))
    ids: list[str] = []
    for rid, _label, _name in re.findall(
        r'race_id=(\d+)">([^<]+)<br>([^<]+)</a>', html
    ):
        if rid not in ids:
            ids.append(rid)
    if len(ids) == N_LEGS:
        out["leg_race_ids"] = ids
    return out


def fetch_win5_result(date_str: str) -> dict:
    resp = requests.get(
        WIN5_RESULT_URL, params={"date": date_str.replace("-", "")},
        headers=HEADERS, timeout=30,
    )
    resp.raise_for_status()
    result = parse_win5_result(resp.text)
    result["url"] = (
        f"{WIN5_RESULT_URL}?date={date_str.replace('-', '')}"
    )
    return result


# ------------------------------------------------------------------
# Leg resolution: fixtures + live states, indexed by content
# ------------------------------------------------------------------

def _venue_short(venue: str) -> str:
    return venue[:-3] if venue.endswith("競馬場") else venue


def index_fixtures(fixtures_dir: Path) -> dict[tuple[str, str, int], dict]:
    """{(date, venue_short, race_number): fixture record} built from the
    fixture files' own race meta — never from filename stems."""
    index: dict[tuple[str, str, int], dict] = {}
    for path in sorted(fixtures_dir.glob("*.json")):
        if path.name.endswith("-results.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            meta = data["race"]
            key = (
                meta["date"],
                _venue_short(meta["venue"]),
                int(meta["race_number"]),
            )
            edition = meta.get("edition")
            index[key] = {
                "race_id": meta["race_id"],
                "race_name": (
                    f"第{edition}回 {meta['name']}"
                    if edition else meta["name"]
                ),
                "horses": {
                    int(h["number"]): h["name"]
                    for h in data.get("horses") or []
                },
                "path": str(path),
            }
        except Exception:
            continue  # legacy/mock fixtures without full race meta
    return index


def index_states(runs_dir: Path) -> dict[str, dict]:
    """{race_id: state record} for LIVE run states only (dry-run leg
    states never qualify a leg)."""
    index: dict[str, dict] = {}
    for path in sorted(runs_dir.glob("*-state.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        race_id = data.get("race_id")
        if not race_id or data.get("mode") != "live":
            continue
        index[race_id] = {"state": data, "path": str(path)}
    return index


def index_results(fixtures_dir: Path) -> dict[str, dict]:
    """{race_id: results fixture} for settlement cross-checks."""
    index: dict[str, dict] = {}
    for path in sorted(fixtures_dir.glob("*-results.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("race_id"):
            index[data["race_id"]] = data
    return index


def leg_label(leg: dict) -> str:
    return f"{leg['venue']}{leg['race_number']}R"


def resolve_legs(date_str: str, legs: list[dict], fixtures_dir: Path,
                 runs_dir: Path) -> list[dict]:
    """Resolve each designated leg to its live prediction state.

    A leg is usable ("ok") only when a fixture exists for
    (date, venue, race_number), a live state exists for that fixture's
    race_id, the state completed, and the conclusion gate's aggregate
    vector is present. Anything else is "missing" with a reason."""
    fixture_index = index_fixtures(fixtures_dir)
    state_index = index_states(runs_dir)
    resolved: list[dict] = []
    for i, leg in enumerate(legs, start=1):
        rec: dict = {
            "leg": i,
            "venue": leg["venue"],
            "race_number": leg["race_number"],
            "race_id": None,
            "race_name": None,
            "status": "missing",
            "reason": None,
            "aggregate_probs": None,
            "horses": {},
            "state_path": None,
        }
        fixture = fixture_index.get(
            (date_str, leg["venue"], leg["race_number"])
        )
        if fixture is None:
            rec["reason"] = "fixtureが無い（未予想のレース）"
            resolved.append(rec)
            continue
        rec["race_id"] = fixture["race_id"]
        rec["race_name"] = fixture["race_name"]
        rec["horses"] = fixture["horses"]
        entry = state_index.get(fixture["race_id"])
        if entry is None:
            rec["reason"] = "live予想stateが無い"
            resolved.append(rec)
            continue
        state = entry["state"]
        rec["state_path"] = entry["path"]
        if state.get("stage") != "complete":
            rec["reason"] = f"予想が未完了（stage={state.get('stage')}）"
            resolved.append(rec)
            continue
        gate = state.get("conclusion_gate") or {}
        probs = gate.get("aggregate_probs")
        if not probs:
            rec["reason"] = "結論ゲートの集計確率がstateに無い"
            resolved.append(rec)
            continue
        rec["aggregate_probs"] = {
            int(n): float(p) for n, p in probs.items()
        }
        rec["status"] = "ok"
        resolved.append(rec)
    return resolved


# ------------------------------------------------------------------
# Ticket allocation
# ------------------------------------------------------------------

def allocate_win5(vectors: list[dict[int, float]],
                  budget_points: int) -> dict | None:
    """Greedy coverage allocation over the 5 legs' aggregate vectors
    (percent). Returns None when even the minimum ticket (1 pick per
    leg, 100 points) exceeds the budget.

    Start: each leg's top-probability horse. Then repeatedly add the
    (leg, next horse) with the best gain in log(hit probability) per
    extra combination, among additions that keep the stake within the
    budget; a leg holds at most MAX_PICKS_PER_LEG picks. Deterministic
    tie-breaking: efficiency, then gain, then leg order, then horse
    number."""
    if len(vectors) != N_LEGS or budget_points < UNIT_POINTS:
        return None
    order: list[list[int]] = []
    for vec in vectors:
        ranked = sorted(
            (n for n, p in vec.items() if p > 0),
            key=lambda n: (-vec[n], n),
        )
        if not ranked:
            return None
        order.append(ranked)
    picks: list[list[int]] = [[ranked[0]] for ranked in order]
    coverage = [vectors[i][picks[i][0]] for i in range(N_LEGS)]

    def combos_of(p: list[list[int]]) -> int:
        c = 1
        for leg_picks in p:
            c *= len(leg_picks)
        return c

    while True:
        combos = combos_of(picks)
        best = None  # (efficiency, gain, -leg, -number, leg, number)
        for leg in range(N_LEGS):
            k = len(picks[leg])
            if k >= MAX_PICKS_PER_LEG or k >= len(order[leg]):
                continue
            nxt = order[leg][k]
            new_combos = combos // k * (k + 1)
            if new_combos * UNIT_POINTS > budget_points:
                continue
            gain = math.log(coverage[leg] + vectors[leg][nxt]) - math.log(
                coverage[leg]
            )
            if gain <= 0:
                continue
            eff = gain / (new_combos - combos)
            key = (eff, gain, -leg, -nxt)
            if best is None or key > best[0]:
                best = (key, leg, nxt)
        if best is None:
            break
        _key, leg, nxt = best
        picks[leg].append(nxt)
        coverage[leg] += vectors[leg][nxt]
    combos = combos_of(picks)
    hit = 1.0
    for c in coverage:
        hit *= c / 100.0
    return {
        "picks": [sorted(p) for p in picks],
        "combinations": combos,
        "stake": combos * UNIT_POINTS,
        "coverage_pct": [round(c, 2) for c in coverage],
        "hit_prob_pct": round(hit * 100, 4),
    }


# ------------------------------------------------------------------
# WIN5 run state (mirrors orchestrator.RunState discipline)
# ------------------------------------------------------------------

def win5_state_path(date_str: str, live: bool) -> Path:
    base = Path("runs") if live else Path("runs") / "dryrun"
    return base / f"win5-{date_str}-state.json"


class Win5State:
    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data

    @classmethod
    def open(cls, path: Path, date_str: str, live: bool) -> "Win5State":
        if path.exists():
            return cls(path, json.loads(path.read_text(encoding="utf-8")))
        data = {
            "version": 1,
            "kind": "win5",
            "date": date_str,
            "mode": "live" if live else "dry-run",
            "created_at": _now(),
            "stage": "new",
            "deadline": None,
            "legs": [],
            "missing_legs": [],
            "ticket": None,
            "ticket_text": None,
            "note_id": None,
            "settlement": None,
        }
        state = cls(path, data)
        state.save()
        return state

    def save(self) -> None:
        self.data["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)


class Win5Lock:
    """Per-date flock, same pattern as orchestrator.RaceLock: the OS
    holds the lock on the open file description, so a dead holder's
    lock is released by the kernel; the pid in the file is only
    informational."""

    def __init__(self, state_path: Path):
        self.path = state_path.with_suffix(".lock")
        self._fd: int | None = None

    def __enter__(self) -> "Win5Lock":
        import fcntl
        import os

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = os.read(fd, 64).decode(errors="replace").strip()
            os.close(fd)
            raise SystemExit(
                "another WIN5 run holds the lock for this date "
                f"({holder or 'pid unknown'}); refusing to run concurrently"
            )
        os.ftruncate(fd, 0)
        os.write(fd, f"pid {os.getpid()}\n".encode())
        self._fd = fd
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            import fcntl
            import os

            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


# ------------------------------------------------------------------
# Ticket text
# ------------------------------------------------------------------

def _date_label(date_str: str) -> str:
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    return f"{d.month}月{d.day}日（{WEEKDAY_JA[d.weekday()]}）"


def build_ticket_text(data: dict) -> str:
    ticket = data["ticket"]
    lines = [
        f"【WIN5】{_date_label(data['date'])} 結論口座の買い目",
        "",
    ]
    for leg in ticket["legs"]:
        picks = " / ".join(
            f"{p['number']}番 {p['name']}（{p['prob']:.1f}%）"
            for p in leg["picks"]
        )
        lines.append(
            f"{leg['leg']}. {leg['venue']}{leg['race_number']}R "
            f"{leg['race_name']}: {picks}"
            f"［カバー率 {leg['coverage_pct']:.1f}%］"
        )
    lines += [
        "",
        f"組合せ: {ticket['combinations']}通り × {UNIT_POINTS}点 "
        f"= {ticket['stake']}点",
        f"推定的中率: {ticket['hit_prob_pct']:.2f}%",
        "※結論口座のWIN5。各レースの結論集計確率から自動生成しています。",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------
# Generation (+ posting in live mode)
# ------------------------------------------------------------------

def run_win5(date_str: str, live: bool, *, html: str | None = None,
             now: datetime | None = None,
             legs_runs_dir: Path = Path("runs"),
             fixtures_dir: Path = Path("fixtures")) -> dict:
    """Generate (and in live mode post) the day's WIN5 ticket.

    Outcomes are recorded in the state and returned: stage is one of
    no_win5 / legs_incomplete / missed_deadline
    / ticket_generated / ticket_posted / settled. A date that already
    has a ticket is never regenerated; a ticket_generated live state
    whose note never posted (crash) is posted on re-entry, with the
    Emitter's exact-text reconciliation preventing a duplicate."""
    target = datetime.strptime(date_str, "%Y-%m-%d").date()
    path = win5_state_path(date_str, live)
    with Win5Lock(path):
        store = Win5State.open(path, date_str, live)
        data = store.data
        if data.get("settlement"):
            return data
        if data.get("ticket") is None:
            day = win5_day(target, html)
            if day is None:
                if data["stage"] in ("new", "no_win5"):
                    data["stage"] = "no_win5"
                    store.save()
                return data
            data["deadline"] = day["deadline_dt"].strftime(
                "%Y-%m-%dT%H:%M:%S%z"
            )
            resolved = resolve_legs(
                date_str, day["legs"], fixtures_dir, legs_runs_dir
            )
            data["legs"] = [
                {
                    "leg": r["leg"], "venue": r["venue"],
                    "race_number": r["race_number"],
                    "race_id": r["race_id"], "race_name": r["race_name"],
                    "status": r["status"], "reason": r["reason"],
                    "state_path": r["state_path"],
                }
                for r in resolved
            ]
            missing = [r for r in resolved if r["status"] != "ok"]
            data["missing_legs"] = [leg_label(r) for r in missing]
            if missing:
                data["stage"] = "legs_incomplete"
                store.save()
                return data
            moment = now or now_jst()
            if moment >= day["deadline_dt"]:
                data["stage"] = "missed_deadline"
                store.save()
                return data
            # Flat budget (decoupled scoreboard, 2026-11-08): the
            # conclusion account's balance is never consulted here.
            budget = BUDGET_POINTS
            vectors = [r["aggregate_probs"] for r in resolved]
            alloc = allocate_win5(vectors, budget)
            if alloc is None:
                raise RuntimeError(
                    "WIN5 allocator returned no ticket for 5 resolved "
                    f"legs at the flat {budget}-point budget"
                )
            ticket_legs = []
            for r, leg_picks, cov in zip(
                resolved, alloc["picks"], alloc["coverage_pct"]
            ):
                ticket_legs.append({
                    "leg": r["leg"], "venue": r["venue"],
                    "race_number": r["race_number"],
                    "race_id": r["race_id"], "race_name": r["race_name"],
                    "picks": [
                        {
                            "number": n,
                            "name": r["horses"].get(n, ""),
                            "prob": r["aggregate_probs"][n],
                        }
                        for n in leg_picks
                    ],
                    "coverage_pct": cov,
                })
            data["ticket"] = {
                "legs": ticket_legs,
                "combinations": alloc["combinations"],
                "stake": alloc["stake"],
                "hit_prob_pct": alloc["hit_prob_pct"],
                "budget": budget,
                "generated_at": _now(),
            }
            data["ticket_text"] = build_ticket_text(data)
            data["stage"] = "ticket_generated"
            store.save()

        if live:
            if not data.get("note_id"):
                import orchestrator

                emitter = orchestrator.Emitter(
                    live=True, usernames=[ACCOUNT]
                )
                note_id = emitter.emit(
                    ACCOUNT, data["ticket_text"], None
                )
                data["note_id"] = note_id
                data["stage"] = "ticket_posted"
                store.save()
        else:
            print(f"\n===== @{ACCOUNT} (WIN5 ticket, dry-run) =====")
            print(data["ticket_text"])
        return data


# ------------------------------------------------------------------
# Settlement
# ------------------------------------------------------------------

def build_settlement_text(data: dict, record: dict,
                          balance_after: int | None) -> str:
    lines = [f"【WIN5精算】{_date_label(data['date'])}", ""]
    for row in record["legs"]:
        mark = "的中" if row["in_picks"] else "不的中"
        lines.append(
            f"{row['leg']}. {row['venue']}{row['race_number']}R: "
            f"1着 {row['winner']}番 {row['winner_name']} … {mark}"
        )
    hit_count = sum(1 for row in record["legs"] if row["in_picks"])
    lines.append("")
    if record["hit"]:
        lines.append(
            f"結果: 的中！ 払戻金 {record['payout_yen']:,}円"
            f"（的中票数 {record['winning_tickets']}票）"
        )
    else:
        lines.append(f"結果: 不的中（{hit_count}/5レース的中）")
    net = record["returned"] - record["staked"]
    lines.append(
        f"賭け金 {record['staked']}点 / 払戻 {record['returned']}点 / "
        f"収支 {net:+}点"
    )
    if balance_after is not None:
        lines.append(
            f"結論口座の累積収支: {points_mod.format_score(balance_after)}点"
        )
    else:
        lines.append("（ドライラン: 精算は未実行）")
    return "\n".join(lines)


def settle_win5(date_str: str, live: bool, *,
                result: dict | None = None,
                fixtures_dir: Path = Path("fixtures")) -> dict:
    """Settle the day's WIN5 ticket against the official result.

    The ledger write (live) happens BEFORE the settlement note posts.
    A hit whose payout cannot be determined raises RuntimeError and
    settles nothing. Dry-run computes and prints only: no ledger, no
    state mutation, no posting."""
    path = win5_state_path(date_str, live)
    if not path.exists():
        raise SystemExit(
            f"no WIN5 state for {date_str} "
            f"({'live' if live else 'dry-run'}): {path}"
        )
    with Win5Lock(path):
        store = Win5State.open(path, date_str, live)
        data = store.data
        if data.get("settlement"):
            record = data["settlement"]
            if live and not record.get("note_id"):
                _post_settlement_note(store, record)
            return record
        ticket = data.get("ticket")
        if not ticket:
            raise SystemExit(
                f"WIN5 state for {date_str} has no ticket "
                f"(stage={data.get('stage')})"
            )
        if result is None:
            result = fetch_win5_result(date_str)
        if result.get("estimated"):
            raise RuntimeError(
                "WIN5 result is not final yet (page still shows "
                "想定払い戻し); refusing to settle"
            )
        winners = result.get("winners") or []
        if len(winners) != N_LEGS:
            raise RuntimeError(
                f"WIN5 result page gave {len(winners)} winning numbers, "
                f"expected {N_LEGS}; refusing to settle"
            )
        ticket_race_ids = [leg["race_id"] for leg in ticket["legs"]]
        page_ids = result.get("leg_race_ids") or []
        if page_ids and page_ids != ticket_race_ids:
            raise RuntimeError(
                f"WIN5 result page legs {page_ids} do not match the "
                f"ticket legs {ticket_race_ids}; refusing to settle"
            )
        # Cross-check winners against the legs' results fixtures when
        # they exist (the morning review rebuilds them anyway).
        results_index = index_results(fixtures_dir)
        fixture_index = index_fixtures(fixtures_dir)
        horses_by_race = {
            rec["race_id"]: rec["horses"]
            for rec in fixture_index.values()
        }
        rows = []
        for leg, winner in zip(ticket["legs"], winners):
            results_fx = results_index.get(leg["race_id"])
            if results_fx is not None:
                top = next(
                    (f for f in results_fx.get("finish") or []
                     if f.get("position") == 1),
                    None,
                )
                if top is not None and int(top["number"]) != winner:
                    raise RuntimeError(
                        f"winner mismatch on leg {leg['leg']} "
                        f"({leg['race_id']}): WIN5 page says {winner}, "
                        f"results fixture says {top['number']}; "
                        "refusing to settle"
                    )
            rows.append({
                "leg": leg["leg"], "venue": leg["venue"],
                "race_number": leg["race_number"],
                "winner": winner,
                "winner_name": horses_by_race.get(
                    leg["race_id"], {}
                ).get(winner, ""),
                "in_picks": winner in [
                    p["number"] for p in leg["picks"]
                ],
            })
        hit = all(row["in_picks"] for row in rows)
        payout_yen = result.get("payout_yen")
        if hit:
            if payout_yen is None:
                raise RuntimeError(
                    "the WIN5 ticket HIT but the official payout "
                    "could not be fetched/parsed; refusing to settle "
                    "at 0 — investigate the result page manually"
                )
            returned = int(payout_yen)
        else:
            returned = 0
        record: dict = {
            "legs": rows,
            "hit": hit,
            "payout_yen": payout_yen if hit else None,
            "winning_tickets": result.get("tickets"),
            "staked": int(ticket["stake"]),
            "returned": returned,
            "source_url": result.get("url"),
            "settled_at": _now(),
            "note_id": None,
        }
        if not live:
            print(build_settlement_text(data, record, None))
            return record
        ledger = points_mod.PointsLedger(points_mod.ledger_path(True))
        rec = ledger.settle(
            f"win5-{date_str}", f"WIN5（{date_str}）", ACCOUNT,
            record["staked"], record["returned"],
        )
        record["ledger"] = rec
        data["settlement"] = record
        data["stage"] = "settled"
        store.save()
        _post_settlement_note(store, record)
        return record


def _post_settlement_note(store: Win5State, record: dict) -> None:
    import orchestrator

    data = store.data
    ledger_rec = record.get("ledger") or {}
    text = build_settlement_text(
        data, record, ledger_rec.get("balance")
    )
    emitter = orchestrator.Emitter(live=True, usernames=[ACCOUNT])
    note_id = emitter.emit(ACCOUNT, text, data.get("note_id"))
    record["note_id"] = note_id
    store.save()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="WIN5 tickets for the conclusion account"
    )
    sub = ap.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "generate the day's ticket (post it with --live)"),
        ("settle", "settle the day's ticket (post the result with --live)"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--date", required=True, help="YYYY-MM-DD")
        p.add_argument("--live", action="store_true",
                       help="actually post / settle for real")
        p.add_argument("--dry-run", action="store_true",
                       help="print only (default)")
    args = ap.parse_args()
    live = bool(args.live)
    try:
        if args.command == "run":
            data = run_win5(args.date, live)
            print(
                f"[win5 run] {args.date}: stage={data.get('stage')}"
                + (
                    f" missing={data.get('missing_legs')}"
                    if data.get("missing_legs") else ""
                )
                + (
                    f" stake={data['ticket']['stake']} "
                    f"combos={data['ticket']['combinations']} "
                    f"hit%={data['ticket']['hit_prob_pct']}"
                    if data.get("ticket") else ""
                )
            )
        else:
            record = settle_win5(args.date, live)
            print(
                f"[win5 settle] {args.date}: "
                f"hit={record['hit']} staked={record['staked']} "
                f"returned={record['returned']}"
            )
    except (RuntimeError, ValueError) as exc:
        print(f"[win5] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline checks for racing_db.py + the career parser in
build_fixture.py (racing DB, 2026-11-10).

No network: the career page is a real db.netkeiba page saved on
2026-11-10 (tests/fixtures/), fixture pairs are the real 2026-11-08
Tokyo 11R files, and the DB lives in a temp directory.
`python tests/run_db_checks.py` must end with "ALL CHECKS PASSED".
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import build_fixture  # noqa: E402
import racing_db  # noqa: E402

ROOT = Path(__file__).parent.parent
CAREER_HTML = (
    Path(__file__).parent / "fixtures" / "career_sample.html"
).read_text(encoding="utf-8")

fails = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("PASS" if cond else "FAIL"), name, detail if not cond else "")
    if not cond:
        fails.append(name)


# --- _id_str ---------------------------------------------------------
check("id_str: float", build_fixture._id_str(2024106282.0) == "2024106282")
check("id_str: str", build_fixture._id_str("2024106282") == "2024106282")
check("id_str: none", build_fixture._id_str(None) is None)

# --- jurisdiction split ----------------------------------------------
check("nar id: saga", racing_db._is_nar_race_id("202655110811") is True)
check("nar id: jra tokyo", racing_db._is_nar_race_id("202605050211") is False)

# --- career parse ------------------------------------------------------
rows = build_fixture.parse_horse_career(CAREER_HTML)
check("career: 2 rows", len(rows) == 2, f"got {len(rows)}")
r0 = rows[0]
check("career: race_id", r0["race_id"] == "202604030207", r0["race_id"])
check("career: date", r0["date"] == "2026-08-23", r0["date"])
check("career: venue", r0["venue"] == "新潟", r0["venue"])
check("career: course", r0["surface"] == "芝" and r0["distance_m"] == 1600)
check("career: position", r0["position"] == 9)
check("career: margin", r0["margin"] == "2.7", str(r0["margin"]))
check("career: passing", r0["passing"] == "1-1", str(r0["passing"]))
check("career: final_3f", r0["final_3f"] == 34.1)
check("career: weight", r0["horse_weight"] == 488
      and r0["horse_weight_diff"] == 10)
check("career: jockey id", r0["jockey_id"] == "01163")
check("career: winner margin negative", rows[1]["margin"] == "-0.2")

# --- ingest + query + format -------------------------------------------
tmp = Path(tempfile.mkdtemp())
conn = racing_db.connect(tmp / "t.db")
n = racing_db.ingest_career(conn, "2024106282", "テストホース", rows)
check("ingest career: rows", n == 2)
runs = racing_db.recent_runs(conn, "2024106282", "2026-11-15", limit=5)
check("recent: 2 runs", len(runs) == 2)
check("recent: newest first", runs[0]["race_date"] == "2026-08-23")
s = racing_db.format_form_run(runs[0])
check("format: position", "9着" in s, s)
check("format: margin", "着差2.7" in s, s)
check("format: passing", "通過1-1" in s, s)
check("format: final3f", "上り34.1" in s, s)
check("format: weight", "馬体重488(+10)" in s, s)
check("career fetched today",
      racing_db.career_fetched_today(conn, "2024106282"))

# --- dedupe across layers ----------------------------------------------
dupe = dict(rows[0])
dupe["race_id"] = "999999999999"  # same date+name, foreign id scheme
racing_db.ingest_career(conn, "2024106282", "テストホース", [dupe],
                        source="career-dupe-test")
runs = racing_db.recent_runs(conn, "2024106282", "2026-11-15", limit=5)
check("recent: deduped by date+name", len(runs) == 2,
      f"got {len(runs)}")

# --- results fixture ingest --------------------------------------------
fixture = json.loads(
    (ROOT / "fixtures" / "2026-tokyo-1108-11r.json").read_text("utf-8"))
results = json.loads(
    (ROOT / "fixtures" / "2026-tokyo-1108-11r-results.json")
    .read_text("utf-8"))
written = racing_db.ingest_results_fixture(conn, fixture, results)
check("ingest results: runners == finish",
      written == len(results["finish"]),
      f"{written} vs {len(results['finish'])}")
check("ingest results: complete",
      racing_db.race_ingest_complete(conn, results["race_id"]))
pay = conn.execute(
    "SELECT COUNT(*) AS c FROM payouts WHERE race_id = ?",
    (results["race_id"],),
).fetchone()["c"]
check("ingest results: payouts stored", pay > 0, f"got {pay}")
again = racing_db.ingest_results_fixture(conn, fixture, results)
count = conn.execute(
    "SELECT COUNT(*) AS c FROM runners WHERE race_id = ?",
    (results["race_id"],),
).fetchone()["c"]
check("ingest results: idempotent", again == written
      and count == len(results["finish"]))

# --- COALESCE merge: sparse re-ingest keeps detail ----------------------
sparse = {
    "race": dict(fixture["race"]),
    "horses": [],
}
sparse_results = {
    "race_id": results["race_id"],
    "finish": [dict(results["finish"][0], final_3f=None,
                    passing=None, horse_weight=None)],
    "payouts_jpy": {},
}
target = results["finish"][0]["number"]
racing_db.ingest_career(conn, "9999999999", "マージ馬",
                        [dict(rows[0], race_id=results["race_id"],
                              date=results["date"],
                              race_name=fixture["race"]["name"],
                              number=target, final_3f=33.3,
                              passing="2-2", horse_weight=500,
                              horse_weight_diff=-4)],
                        source="career-merge-test")
racing_db.ingest_results_fixture(conn, sparse, sparse_results)
row = conn.execute(
    "SELECT final_3f, passing, horse_weight FROM runners"
    " WHERE race_id = ? AND number = ?",
    (results["race_id"], target),
).fetchone()
check("merge: detail survives sparse ingest",
      row["final_3f"] == 33.3 and row["passing"] == "2-2"
      and row["horse_weight"] == 500,
      str(dict(row)))

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED: {fails}")
    sys.exit(1)
print("ALL CHECKS PASSED")

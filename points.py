"""Points ledger for the AI predictors (持ち点).

Each AI account starts with INITIAL_BALANCE points and bets from its own
balance; settlements after a race move the balance by (returned - staked)
using the real JRA/NAR payouts. The ledger is a single JSON file, kept in
runs/ for live runs and runs/dryrun/ for dry runs, mirroring the run-state
separation: a dry run must never move real balances.

Settlement is idempotent per (race_id, username): re-running a review
never settles the same race twice. An account that falls below 100 points
cannot place even the minimum bet, so ensure_bettable tops it back up to
RELIEF_BALANCE once per race, recorded in its history as a relief.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

INITIAL_BALANCE = 10_000
RELIEF_BALANCE = 1_000
STAKE_CAP_RATIO = 0.30


def ledger_path(live: bool) -> Path:
    return Path("runs/points.json") if live else Path("runs/dryrun/points.json")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class PointsLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"version": 1, "accounts": {}, "settled": {}, "reliefs": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.path)

    def _account(self, username: str) -> dict:
        return self.data["accounts"].setdefault(
            username, {"balance": INITIAL_BALANCE, "history": []}
        )

    def balance(self, username: str) -> int:
        return int(self._account(username)["balance"])

    def cap(self, username: str) -> int:
        """Max total stake for one race: 30% of the balance, in 100-point
        units, never more than the balance itself."""
        b = self.balance(username)
        return min(b, int(b * STAKE_CAP_RATIO) // 100 * 100)

    def snapshot(self, usernames: list[str]) -> dict[str, dict]:
        return {
            u: {"balance": self.balance(u), "cap": self.cap(u)} for u in usernames
        }

    def ensure_bettable(self, username: str, race_id: str | None) -> bool:
        """Relief top-up for an account under the 100-point minimum bet.
        Once per race; returns True when a top-up was applied."""
        acct = self._account(username)
        key = f"{race_id}:{username}"
        if acct["balance"] >= 100 or key in self.data["reliefs"]:
            return False
        acct["balance"] = RELIEF_BALANCE
        acct["history"].append(
            {"kind": "relief", "race_id": race_id, "balance": acct["balance"],
             "at": _now()}
        )
        self.data["reliefs"][key] = True
        self.save()
        return True

    def settle(self, race_id: str, race_name: str, username: str,
               staked: int, returned: int) -> dict:
        """Apply one race's settlement. Idempotent: a second call for the
        same (race, account) returns the existing record unchanged."""
        key = f"{race_id}:{username}"
        if key in self.data["settled"]:
            return self.data["settled"][key]
        acct = self._account(username)
        acct["balance"] = int(acct["balance"]) + int(returned) - int(staked)
        rec = {
            "race_id": race_id,
            "race": race_name,
            "username": username,
            "staked": int(staked),
            "returned": int(returned),
            "balance": int(acct["balance"]),
            "at": _now(),
        }
        acct["history"].append({"kind": "settle", **rec})
        self.data["settled"][key] = rec
        self.save()
        return rec

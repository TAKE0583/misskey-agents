"""Points ledger for the AI predictors (持ち点).

Each AI account starts with INITIAL_BALANCE points and bets from its own
balance; settlements after a race move the balance by (returned - staked)
using the real JRA/NAR payouts. The ledger is a single JSON file, kept in
runs/ for live runs and runs/dryrun/ for dry runs, mirroring the run-state
separation: a dry run must never move real balances.

Settlement is idempotent per (race_id, username): re-running a review
never settles the same race twice.

Decoupled scoreboard (2026-11-08 user decision): the balance is a pure
scoreboard, never an input to prediction or betting. The per-race
stake cap is a flat STAKE_CAP for every account regardless of balance,
generation prompts carry no balance at all, and there is no relief
top-up — balances may fall to zero or below, and settlement records
the exact figure (no flooring). What readers see is the cumulative
P&L ("score"): balance - INITIAL_BALANCE, via format_score(). The old
relief entries remain in existing ledgers' history as historical
record; nothing writes them any more.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

INITIAL_BALANCE = 10_000
STAKE_CAP = 3_000


def score_of(balance: int) -> int:
    """Cumulative P&L behind a ledger balance: the scoreboard figure
    readers see. The internal balance keeps the INITIAL_BALANCE offset
    so all historical settlement math stays untouched."""
    return int(balance) - INITIAL_BALANCE


def format_score(balance: int) -> str:
    """Signed score for display, thousands-separated: a balance of
    39,430 renders '+29,430', 620 renders '-9,380', and exactly
    INITIAL_BALANCE renders '±0' (the agreed zero form)."""
    score = score_of(balance)
    if score == 0:
        return "±0"
    return f"{score:+,}"


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
            self.data = {"version": 1, "accounts": {}, "settled": {}}

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
        """Max total stake for one race: a flat STAKE_CAP for every
        account, whatever its balance (decoupled scoreboard — a losing
        streak no longer shrinks an AI's betting budget)."""
        return STAKE_CAP

    def score(self, username: str) -> int:
        """The account's cumulative P&L (balance - INITIAL_BALANCE)."""
        return score_of(self.balance(username))

    def snapshot(self, usernames: list[str]) -> dict[str, dict]:
        return {
            u: {"balance": self.balance(u), "cap": self.cap(u)} for u in usernames
        }

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

    def adjust(self, race_id: str, race_name: str, username: str,
               amount: int, reason: str) -> dict:
        """Apply a one-off auditable balance correction (for example
        undoing a settlement that should never have happened).
        Idempotent per (race, account): repeating the same correction
        returns the existing record and never moves the balance twice.
        The original history entries stay untouched; the correction is
        its own history entry, so the audit trail shows both."""
        corrections = self.data.setdefault("corrections", {})
        key = f"{race_id}:{username}"
        if key in corrections:
            return corrections[key]
        acct = self._account(username)
        acct["balance"] = int(acct["balance"]) + int(amount)
        rec = {
            "race_id": race_id,
            "race": race_name,
            "username": username,
            "amount": int(amount),
            "reason": reason,
            "balance": int(acct["balance"]),
            "at": _now(),
        }
        acct["history"].append({"kind": "adjust", **rec})
        corrections[key] = rec
        self.save()
        return rec

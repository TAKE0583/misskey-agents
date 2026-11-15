"""Offline checks for the per-AI reflection memory (2026-11-16).

No network, no providers: `python tests/run_memory_checks.py` must end
with "ALL CHECKS PASSED". Covers the store (record / dedupe / caps /
fail-safe reads), the prompt block format, the orchestrator wiring
(the digest reaches both prediction prompt builders and the generation
loop), and an end-to-end dry-run cycle: review records the memory,
and the snapshot for the next prediction reads it back.
"""
from __future__ import annotations

import json
import os as _os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import orchestrator as o  # noqa: E402
import providers  # noqa: E402
import reflection_memory as memory_mod  # noqa: E402

FIXTURES = Path(__file__).parent.parent / "fixtures"
monbetsu = json.loads(
    (FIXTURES / "2026-monbetsu-1104-6r.json").read_text(encoding="utf-8")
)
monbetsu_results = json.loads(
    (FIXTURES / "2026-monbetsu-1104-6r-results.json").read_text(encoding="utf-8")
)

fails = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("PASS" if cond else "FAIL"), name, detail if not cond else "")
    if not cond:
        fails.append(name)


# --- store basics -----------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "memory.json"
    check(
        "store: unknown account has no block",
        memory_mod.block_for("nobody", path=path) == "",
    )
    ok = memory_mod.record(
        "chatgpt", "R1", "2026-11-10", "東京11R テストS",
        "【反省】第1回 テストS\n\n軸馬の過信が敗因でした。",
        live=True, path=path,
    )
    check("store: record returns True", ok)
    block = memory_mod.block_for("chatgpt", path=path)
    check(
        "store: block carries header + body, note header stripped",
        "反省メモ" in block
        and "軸馬の過信が敗因でした" in block
        and "【反省】" not in block
        and "2026-11-10" in block
        and "東京11R テストS" in block,
    )
    check(
        "store: other accounts stay empty",
        memory_mod.block_for("claude", path=path) == "",
    )
    memory_mod.record(
        "chatgpt", "R1", "2026-11-10", "東京11R テストS",
        "書き直した反省です。", live=True, path=path,
    )
    entries = json.loads(path.read_text(encoding="utf-8"))["accounts"]["chatgpt"]
    check(
        "store: re-record replaces, not duplicates",
        len(entries) == 1 and entries[0]["text"] == "書き直した反省です。",
    )
    check(
        "store: empty text refused",
        memory_mod.record(
            "chatgpt", "R2", "2026-11-11", "x", "  ", live=True, path=path
        )
        is False,
    )
    check(
        "store: missing race_id refused",
        memory_mod.record(
            "chatgpt", None, "2026-11-11", "x", "本文", live=True, path=path
        )
        is False,
    )

# --- inject limit + store cap ----------------------------------------
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "memory.json"
    for i in range(7):
        memory_mod.record(
            "gemini", f"R{i}", f"2026-11-{10 + i:02d}", f"場{i}R レース{i}",
            f"反省その{i}。", live=True, path=path,
        )
    block = memory_mod.block_for("gemini", path=path)
    check(
        "inject: only the newest 5 are shown",
        "反省その6" in block
        and "反省その2" in block
        and "反省その1" not in block
        and "反省その0" not in block,
    )
    for i in range(7, 50):
        memory_mod.record(
            "gemini", f"R{i}", "2026-11-12", "場R レース",
            f"反省その{i}。", live=True, path=path,
        )
    stored = json.loads(path.read_text(encoding="utf-8"))["accounts"]["gemini"]
    check(
        "store: capped at MAX_STORED_PER_AI",
        len(stored) == memory_mod.MAX_STORED_PER_AI,
    )

# --- corrupt store is fail-safe ---------------------------------------
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "memory.json"
    path.write_text("{not json", encoding="utf-8")
    check(
        "store: corrupt file yields no block",
        memory_mod.block_for("chatgpt", path=path) == "",
    )
    check(
        "store: record recovers from corruption",
        memory_mod.record(
            "chatgpt", "R9", "2026-11-12", "場R", "復旧後の反省。",
            live=True, path=path,
        )
        and "復旧後の反省" in memory_mod.block_for("chatgpt", path=path),
    )

# --- blocks_for snapshot shape ----------------------------------------
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "memory.json"
    memory_mod.record(
        "local", "R1", "2026-11-12", "場R", "ローカルの反省。",
        live=True, path=path,
    )
    snap = memory_mod.blocks_for(["local", "chatgpt"], live=True, path=path)
    check(
        "blocks_for: only accounts with memory appear",
        set(snap) == {"local"} and "ローカルの反省" in snap["local"],
    )

# --- wiring: prediction prompts ---------------------------------------
text_ok = (
    "◎ 10 タイムエッジ 勝率30% 3着内60% — 本命。\n"
    "○ 4 ダムドローズ 勝率22% 3着内50% — 対抗。\n"
    "▲ 1 エリオラ 勝率15% 3着内40% — 単穴。\n"
    "△ 2 ラップドアップ 勝率10% 3着内30% — 連下。\n\n"
    "【買い目】\n単勝 10 400点\n3連複 1-4-10 600点"
)
SENTINEL = "\n\nMEMORY-BLOCK-SENTINEL"


class _CaptureCloud:
    name = "capture"

    def __init__(self):
        self.users: list[str] = []

    def chat(self, system, user, model=None, max_tokens=None):
        self.users.append(user)
        if "議論として応答" in user:
            return (
                "議論を踏まえ、最終票を出します。\n\n【最終票】\n"
                "◎ 10 タイムエッジ 勝率30%\n○ 4 ダムドローズ 勝率22%\n"
                "▲ 1 エリオラ 勝率15%\n△ 2 ラップドアップ 勝率10%"
            )
        return text_ok


_persona = {"username": "chatgpt", "display": "ChatGPT"}
_cap = _CaptureCloud()
_out = o._generate_text_prediction(
    _cap, _persona, monbetsu, o.race_text(monbetsu), 3000, memory=SENTINEL
)
check(
    "wiring: cloud prediction prompt includes the memory block",
    _out == text_ok and SENTINEL.strip() in _cap.users[0],
)
_cap2 = _CaptureCloud()
o._generate_text_prediction(
    _cap2, _persona, monbetsu, o.race_text(monbetsu), 3000
)
check(
    "wiring: no memory -> cloud prompt unchanged",
    SENTINEL.strip() not in _cap2.users[0],
)


class _CaptureLocal:
    name = "capturelocal"

    def __init__(self):
        self.users: list[str] = []

    def chat_json(self, system, user, schema, model=None, max_tokens=None):
        self.users.append(user)
        return {
            "marks": [
                {"mark": "◎", "number": 10, "win_pct": 30, "top3_pct": 60, "reason": "本命"},
                {"mark": "○", "number": 4, "win_pct": 22, "top3_pct": 50, "reason": "対抗"},
                {"mark": "▲", "number": 1, "win_pct": 15, "top3_pct": 40, "reason": "単穴"},
                {"mark": "△", "number": 2, "win_pct": 10, "top3_pct": 30, "reason": "連下"},
            ],
            "bets": [{"type": "単勝", "combo": "10", "amount": 1000}],
            "comment": "",
        }


_loc = _CaptureLocal()
o.generate_structured_prediction(_loc, monbetsu, cap=None, memory=SENTINEL)
check(
    "wiring: structured prediction prompt includes the memory block",
    SENTINEL.strip() in _loc.users[0],
)

# --- wiring: the generation loop routes the map per persona -----------
_cap3 = _CaptureCloud()
_gen = o._generate_prediction_texts(
    monbetsu,
    [(_persona, _cap3), (o.PERSONAS[1], providers.MockProvider("Claude"))],
    memory={"chatgpt": SENTINEL},
)
_pred_users = [u for u in _cap3.users if "次のレースの予想をしてください" in u]
check(
    "wiring: generation loop injects the map into the prediction call",
    bool(_pred_users) and SENTINEL.strip() in _pred_users[0],
)

# --- end to end: dry-run review records, snapshot reads back ----------
_cwd = _os.getcwd()
with tempfile.TemporaryDirectory() as td:
    _os.chdir(td)
    try:
        _store = o.RunState.open(
            o.state_path_for("fixtures/memory-e2e.json", False),
            monbetsu, "dryrun",
        )
        _built = o.build_providers(mock=True)
        o.run_prediction(monbetsu, o.Emitter(live=False), _built, _store)
        o.run_review(
            monbetsu, monbetsu_results, _store,
            o.Emitter(live=False), built=_built,
        )
        check(
            "e2e: dry-run review writes the dry-run memory store",
            Path("runs/dryrun/reflection-memory.json").exists(),
        )
        _usernames = [p["username"] for p, _ in _built]
        _snap = memory_mod.blocks_for(_usernames, live=False)
        check(
            "e2e: every participant has a memory block afterwards",
            set(_snap) == set(_usernames)
            and all("反省メモ" in b for b in _snap.values()),
        )
        _data = json.loads(
            Path("runs/dryrun/reflection-memory.json").read_text(encoding="utf-8")
        )
        check(
            "e2e: entries are keyed by the race_id and labelled by venue",
            all(
                _data["accounts"][u][0]["race_id"] == _store.data["race_id"]
                and "門別6R" in _data["accounts"][u][0]["label"]
                for u in _usernames
            ),
        )
        check(
            "e2e: the live store was not touched by the dry run",
            not Path("runs/reflection-memory.json").exists(),
        )
    finally:
        _os.chdir(_cwd)

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED: {fails}")
    sys.exit(1)
print("ALL CHECKS PASSED")

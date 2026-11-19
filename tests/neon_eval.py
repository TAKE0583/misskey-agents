#!/usr/bin/env python3
"""Compare Gemini models on the exact production prediction path.

Judges whether gemini-4-neon is a safe replacement for the current
@gemini default by replaying the byte-identical @gemini prediction
prompt (built with the same functions the live orchestrator path
uses) through several Gemini models, and recording wall time, token
usage (attributed from the JSON lines the connector helper appends to
runs/usage.log), cost, validation, and probability coverage for each
(fixture, model).

Run with cwd = repo root (fixture/ledger/log paths are cwd-relative,
exactly like the orchestrator's own CLI):

    python3 tests/neon_eval.py [--fixture fixtures/X.json ...]

Raw replies are saved to tests/neon_eval_out/<fixture-stem>__<model>.txt
and all details (including full validation error lists) to
tests/neon_eval_out/summary.json. Nothing outside tests/neon_eval_out
is written, and nothing is ever posted to Misskey.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import orchestrator  # noqa: E402
import points as points_mod  # noqa: E402
import providers  # noqa: E402
import reflection_memory as memory_mod  # noqa: E402

MODELS = [
    "gemini-4-neon",
    "gemini-4-carbon",
    "gemini-3.1-pro-preview",
]

# USD per 1M tokens (input, output). gemini-4-carbon becomes
# (4.0, 20.0) on 2026-12-01. Any model without an entry here reports
# cost n/a -- other prices are deliberately not guessed.
PRICE_PER_MTOK = {
    "gemini-4-neon": (3.0, 15.0),
    "gemini-4-carbon": (2.0, 10.0),  # (4.0, 20.0) from 2026-12-01
}
PRICE_NOTE = (
    "gemini-4-carbon becomes (4.0, 20.0) USD per 1M tokens on "
    "2026-12-01; models without a price entry are reported as n/a."
)

DEFAULT_FIXTURES = [
    "fixtures/2026-oi-1118-11r.json",
    "fixtures/2026-kasamatsu-1118-12r.json",
]

OUT_DIR = Path("tests/neon_eval_out")

# The connector helper appends one JSON line per API call here
# (fields: ts / provider / model / input_tokens / output_tokens);
# MISSKEY_USAGE_LOG overrides the location, so honor it the same way.
USAGE_LOG = Path(os.environ.get("MISSKEY_USAGE_LOG", orchestrator.USAGE_LOG))

# Production chat budget for a cloud prediction: _generate_text_prediction
# calls prov.chat(system, user) with no max_tokens, so _HelperProvider
# resolves the budget to default_max_tokens (6000). Reuse that exact
# budget -- do not invent one.
CHAT_BUDGET = providers.GeminiProvider.default_max_tokens

# The @gemini entry of orchestrator.PERSONAS.
GEMINI_PERSONA = next(p for p in orchestrator.PERSONAS if p["username"] == "gemini")

# Production validates with max_total = the account cap, which the
# points snapshot resolves to the flat STAKE_CAP = 3,000 -- exactly
# the 3,000 this eval is specified to validate with.
VALIDATION_MAX_TOTAL = 3000


def production_prompt(fixture: dict, memory_block: str) -> tuple[str, str, int]:
    """(system, user, cap) byte-identical to what the live prediction
    path builds for @gemini: run_prediction snapshots the points ledger
    (flat cap) and the reflection memory, and _generate_text_prediction
    concatenates the body, market note, memory and budget line in this
    exact order."""
    ledger = points_mod.PointsLedger(points_mod.ledger_path(live=True))
    cap = ledger.snapshot([GEMINI_PERSONA["username"]])[
        GEMINI_PERSONA["username"]
    ]["cap"]
    system = orchestrator.SYSTEM_TEMPLATE.format(
        display=GEMINI_PERSONA["display"]
    )
    user = f"次のレースの予想をしてください。\n\n{orchestrator.race_text(fixture)}"
    user += orchestrator.market_prompt_note(fixture)
    user += memory_block
    user += orchestrator.budget_line(cap)
    return system, user, cap


def usage_log_size() -> int:
    try:
        return USAGE_LOG.stat().st_size
    except OSError:
        return 0


def usage_lines_since(before: int) -> list[dict]:
    """The JSON lines the gemini helper appended after byte offset
    `before` (a provider retry appends one line per API call)."""
    if usage_log_size() <= before:
        return []
    with USAGE_LOG.open("r", encoding="utf-8", errors="replace") as fh:
        fh.seek(before)
        chunk = fh.read()
    out = []
    for raw in chunk.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def apply_usage(row: dict) -> None:
    """Sum the attributed usage lines into the row and price them."""
    in_tok = out_tok = 0
    seen = False
    for rec in row["usage_lines"]:
        if rec.get("input_tokens") is None and rec.get("output_tokens") is None:
            continue
        seen = True
        in_tok += int(rec.get("input_tokens") or 0)
        out_tok += int(rec.get("output_tokens") or 0)
    if seen:
        row["input_tokens"] = in_tok
        row["output_tokens"] = out_tok
        price = PRICE_PER_MTOK.get(row["model"])
        if price is not None:
            row["cost_usd"] = (in_tok * price[0] + out_tok * price[1]) / 1_000_000


HEADER = (
    f"{'fixture':<24} {'model':<26} {'sec':>5} "
    f"{'in/out tok':>13} {'cost':>9}  {'valid':<5} {'errs':>4}  probs"
)


def print_row(row: dict) -> None:
    sec = f"{row['seconds']:.1f}" if row["seconds"] is not None else "-"
    tok = (
        f"{row['input_tokens']}/{row['output_tokens']}"
        if row["input_tokens"] is not None
        else "-"
    )
    cost = f"${row['cost_usd']:.4f}" if row["cost_usd"] is not None else "n/a"
    line = (
        f"{row['fixture_stem']:<24} {row['model']:<26} {sec:>5} "
        f"{tok:>13} {cost:>9}  "
    )
    if row["status"] == "ERROR":
        print(line + "ERROR")
        print(" " * 52 + row["exception"])
        return
    probs = (
        f"{row['with_probabilities']}/{row['runner_count']} "
        f"Σ{row['win_probability_sum']}%"
    )
    print(
        line
        + f"{row['status']:<5} {row['error_count']:>4}  {probs}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compare Gemini models on the exact production @gemini "
            "prediction path (run with cwd = repo root)."
        )
    )
    ap.add_argument(
        "--fixture",
        action="append",
        metavar="PATH",
        help="fixture JSON, repeatable (default: " + " ".join(DEFAULT_FIXTURES) + ")",
    )
    args = ap.parse_args()
    fixture_paths = [Path(p) for p in (args.fixture or DEFAULT_FIXTURES)]
    missing = [str(p) for p in fixture_paths if not p.is_file()]
    if missing:
        raise SystemExit("fixture file(s) not found: " + ", ".join(missing))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reflection memory: a live run injects @gemini's own recent
    # reflections into its prediction prompt (block_for is fail-safe:
    # no store / no entries -> ""), so the eval replays exactly that.
    memory_block = memory_mod.block_for(GEMINI_PERSONA["username"], live=True)
    print(f"[neon_eval] models: {', '.join(MODELS)}")
    print(
        f"[neon_eval] chat budget: {CHAT_BUDGET} tokens "
        "(GeminiProvider.default_max_tokens, as the production "
        "prediction path resolves it)"
    )
    print(
        f"[neon_eval] reflection memory: {len(memory_block)} chars for "
        f"@{GEMINI_PERSONA['username']} (live store, as production)"
    )
    print(HEADER)

    provider = providers.GeminiProvider()
    rows: list[dict] = []
    prompts_meta: dict[str, dict] = {}
    for fx_path in fixture_paths:
        # The same load the orchestrator's CLI performs.
        fixture = json.loads(fx_path.read_text(encoding="utf-8"))
        stem = fx_path.stem
        system, user, cap = production_prompt(fixture, memory_block)
        prompts_meta[stem] = {
            "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "user_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
            "user_chars": len(user),
            "cap": cap,
        }
        runner_count = len(fixture["horses"])
        for model in MODELS:
            row = {
                "fixture": str(fx_path),
                "fixture_stem": stem,
                "model": model,
                "runner_count": runner_count,
                "cap": cap,
                "status": "ERROR",
                "seconds": None,
                "usage_lines": [],
                "input_tokens": None,
                "output_tokens": None,
                "cost_usd": None,
                "errors": [],
                "error_count": None,
                "with_probabilities": None,
                "win_probability_sum": None,
                "output_file": None,
                "exception": None,
            }
            before = usage_log_size()
            start = time.monotonic()
            try:
                text = provider.chat(
                    system, user, model=model, max_tokens=CHAT_BUDGET
                )
            except Exception as exc:  # noqa: BLE001 - ERROR row, run continues
                row["seconds"] = time.monotonic() - start
                row["exception"] = f"{type(exc).__name__}: {exc}"
                row["usage_lines"] = usage_lines_since(before)
                apply_usage(row)
                rows.append(row)
                print_row(row)
                continue
            row["seconds"] = time.monotonic() - start
            row["usage_lines"] = usage_lines_since(before)
            apply_usage(row)
            out_file = OUT_DIR / f"{stem}__{model}.txt"
            out_file.write_text(text, encoding="utf-8")
            row["output_file"] = str(out_file)
            errors = orchestrator.prediction_errors(
                text, fixture, max_total=VALIDATION_MAX_TOTAL
            )
            row["errors"] = errors
            row["error_count"] = len(errors)
            row["status"] = "PASS" if not errors else "FAIL"
            probs = orchestrator.parse_probabilities(text)
            row["with_probabilities"] = len(probs)
            row["win_probability_sum"] = sum(
                win for _num, win, _top3 in probs.values()
            )
            rows.append(row)
            print_row(row)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": MODELS,
        "fixtures": [str(p) for p in fixture_paths],
        "persona": {
            "username": GEMINI_PERSONA["username"],
            "display": GEMINI_PERSONA["display"],
        },
        "chat_budget_max_tokens": CHAT_BUDGET,
        "validation_max_total": VALIDATION_MAX_TOTAL,
        "reflection_memory_chars": len(memory_block),
        "price_per_mtok_usd": PRICE_PER_MTOK,
        "price_note": PRICE_NOTE,
        "prompts": prompts_meta,
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n[summary: {OUT_DIR / 'summary.json'}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

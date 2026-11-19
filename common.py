#!/usr/bin/env python3
"""Shared constants for the misskey-agents orchestrator modules (extracted 2026-11-19; behavior unchanged)."""
from __future__ import annotations

import re

MARKS = ["◎", "○", "▲", "△"]
MARK_SCORE = {"◎": 4, "○": 3, "▲": 2, "△": 1}
PROB_LINE_RE = re.compile(
    r"([◎○▲△])\s*(\d{1,2})\s+\S+\s+勝率\s*(\d{1,3})\s*%\s+3着内\s*(\d{1,3})\s*%"
)

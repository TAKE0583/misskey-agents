#!/usr/bin/env python3
"""Hashtag line construction for the moderator's posts (extracted from orchestrator.py, 2026-11-19 split; behavior unchanged)."""
from __future__ import annotations

import re

# --- Hashtags on the moderator's posts (2026-11-05) ----------------------
# Every @shikai note (thread opener, conclusion, review) ends with one
# hashtag line so any post of a race's thread identifies the race:
#   #AI競馬予想 #門別競馬場 #2026道営記念
# The line is derived deterministically from the fixture alone, so the
# three notes of one race always carry the identical line. Texts of
# already-posted notes live in the run state and are never regenerated,
# so adding tags here cannot disturb reconciliation of older runs; only
# newly generated @shikai texts gain the line.
PROJECT_TAG = "#AI競馬予想"

# Bracketed grade/class badges the netkeiba scrapes glue onto race
# names (道営記念〔H1〕重賞, カムイユカラスプリント(A2)). A bracketed
# segment is dropped whole only when its entire content is one of these
# annotations; any other parenthetical (天皇賞(秋)) is part of the race
# name and its content survives (the bracket characters themselves are
# removed by the character filter below).
_TAG_ANNOTATION = re.compile(
    r"(?:G[123]|Jpn[ⅠⅡⅢI123]|H[123]|[A-C][123]|OP|L|重賞|条件|新馬|未勝利)"
)
_TAG_BRACKETED = re.compile(
    r"〔([^〕]*)〕|（([^）]*)）|\(([^)]*)\)|［([^］]*)］|\[([^\]]*)\]"
)
# Characters allowed in a Misskey hashtag fragment: kana/kanji (with
# iteration marks and the long-vowel mark), ASCII letters/digits and
# underscore. Everything else — whitespace, brackets, symbols and the
# katakana middle dot ・ — is stripped: MFM hashtag parsing stops at
# characters like ・, so keeping one would silently truncate the tag.
_TAG_INVALID = re.compile(r"[^0-9A-Za-z_ぁ-んァ-ヶー一-龯々〆]")


def _tag_fragment(raw: str) -> str:
    """Sanitise a race name into a hashtag-safe fragment.

    Rules: drop bracketed grade/class annotations (brackets AND
    content); keep the content of other bracketed segments; an opening
    bracket that never closes is scrape glue — drop it and everything
    after it; remove all whitespace; drop a trailing 重賞 descriptor the
    NAR scrape glues on (道営記念〔H1〕重賞 -> 道営記念); finally strip
    every character invalid in a Misskey hashtag (see _TAG_INVALID).
    """
    def _sub(m: re.Match) -> str:
        content = next(g for g in m.groups() if g is not None)
        if _TAG_ANNOTATION.fullmatch(content.strip()):
            return ""
        return content

    s = _TAG_BRACKETED.sub(_sub, raw or "")
    unmatched = re.search(r"[〔（(［\[]", s)
    if unmatched:
        s = s[: unmatched.start()]
    s = re.sub(r"\s+", "", s)
    if s.endswith("重賞") and len(s) > 2:
        s = s[:-2]
    return _TAG_INVALID.sub("", s)


def race_hashtag(fixture: dict) -> str | None:
    """'#2026道営記念' from the fixture's race name + date year."""
    r = fixture.get("race") or {}
    name = _tag_fragment(str(r.get("name") or ""))
    if not name:
        return None
    year = re.search(r"\d{4}", str(r.get("date") or ""))
    return f"#{year.group(0)}{name}" if year else f"#{name}"


def venue_hashtag(fixture: dict) -> str | None:
    """'#門別競馬場' from the fixture's venue ('門別競馬場' -> same tag)."""
    r = fixture.get("race") or {}
    venue = re.sub(r"\s+", "", str(r.get("venue") or ""))
    if not venue:
        return None
    base = venue[:-3] if venue.endswith("競馬場") else venue
    base = _TAG_INVALID.sub("", base)
    return f"#{base}競馬場" if base else None


def hashtag_line(fixture: dict) -> str:
    """The final line of every @shikai note: project, venue, race tags."""
    tags = [PROJECT_TAG]
    for tag in (venue_hashtag(fixture), race_hashtag(fixture)):
        if tag:
            tags.append(tag)
    return " ".join(tags)

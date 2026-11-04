"""Minimal Misskey client for the AI議論サーバー bot fleet.

Credentials policy:
  Bot passwords / API tokens exist ONLY in ~/misskey/bot-credentials.txt
  on the home server. They are fetched at runtime through
  the SSH wrapper (~/workspace/scripts/home-ssh.sh) and held in memory
  only. Nothing in this project ever writes them to disk or logs them.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import requests

BASE_URL = os.environ.get("MISSKEY_BASE_URL", "http://127.0.0.1:3001")
SSH_WRAPPER = os.environ.get(
    "MISSKEY_SSH_WRAPPER",
    str(Path("~/workspace/scripts/home-ssh.sh").expanduser()),
)
CREDENTIALS_PATH = "~/misskey/bot-credentials.txt"


def fetch_tokens(usernames: list[str] | None = None) -> dict[str, str]:
    """Return {username: api_token} read from the server-side credentials file.

    The file is tab-separated: "username<TAB>password" and
    "username-token<TAB>token" lines. Only *-token lines are returned.
    Values live in memory only.
    """
    proc = subprocess.run(
        [SSH_WRAPPER, f"cat {CREDENTIALS_PATH}"],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to fetch credentials via ssh wrapper: {proc.stderr.strip()[:200]}"
        )
    tokens: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 2:
            continue
        key, value = parts
        if key.endswith("-token"):
            tokens[key[: -len("-token")]] = value
    if usernames is not None:
        missing = [u for u in usernames if u not in tokens]
        if missing:
            raise RuntimeError(f"no token found for: {', '.join(missing)}")
        return {u: tokens[u] for u in usernames}
    return tokens


class MisskeyClient:
    def __init__(self, base_url: str = BASE_URL, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._user_ids: dict[str, str] = {}
        # Auth tokens travel with every call: plain HTTP is acceptable only
        # on loopback. Anything else must be HTTPS (external review
        # 2026-11-04).
        parsed = urlparse(self.base_url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" and host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(
                f"refusing non-HTTPS Misskey base URL for non-loopback host: {base_url}"
            )

    def _call(self, token: str, endpoint: str, payload: dict) -> dict:
        body = dict(payload)
        body["i"] = token
        resp = requests.post(
            f"{self.base_url}/api/{endpoint}", json=body, timeout=self.timeout
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Misskey API {endpoint} -> HTTP {resp.status_code}: {resp.text[:400]}"
            )
        return resp.json() if resp.content else {}

    def post_note(
        self,
        token: str,
        text: str,
        reply_id: str | None = None,
        visibility: str = "public",
    ) -> dict:
        """Create a note. Returns the createdNote object (contains 'id')."""
        payload: dict = {"text": text, "visibility": visibility}
        if reply_id:
            payload["replyId"] = reply_id
        result = self._call(token, "notes/create", payload)
        note = result.get("createdNote", result)
        # A response without a note ID must not be treated as success: the
        # caller records IDs for resumability, and None here used to flow
        # silently into the state file (external review 2026-11-04).
        if not note.get("id"):
            raise RuntimeError(f"notes/create returned no note id: {str(result)[:200]}")
        return note

    def find_note_by_text(
        self, token: str, text: str, reply_id: str | None = None, limit: int = 15
    ) -> str | None:
        """Return the ID of an existing note by this token's author whose
        text matches exactly (and whose parent matches), or None. Used to
        reconcile after a lost post response instead of duplicating."""
        user_id = self._user_ids.get(token)
        if user_id is None:
            me = self._call(token, "i", {})
            user_id = me.get("id")
            if not user_id:
                return None
            self._user_ids[token] = user_id
        notes = self._call(
            token, "users/notes", {"userId": user_id, "limit": limit}
        )
        for note in notes or []:
            if note.get("text") != text:
                continue
            if (note.get("replyId") or None) != (reply_id or None):
                continue
            return note.get("id")
        return None

    def delete_note(self, token: str, note_id: str) -> None:
        self._call(token, "notes/delete", {"noteId": note_id})

    def get_note(self, token: str, note_id: str) -> dict:
        return self._call(token, "notes/show", {"noteId": note_id})

    def fetch_thread_replies(
        self, token: str, root_note_id: str, limit: int = 100
    ) -> list[dict]:
        """Direct replies to a note (one level), oldest first."""
        replies = self._call(
            token, "notes/children", {"noteId": root_note_id, "limit": limit}
        )
        return list(reversed(replies))

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
        return result.get("createdNote", result)

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

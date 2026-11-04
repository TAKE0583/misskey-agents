"""LLM provider layer for the Misskey racing-discussion agents.

Common interface:  provider.chat(system, user, model=None, max_tokens=...) -> str

Key policy (Secure Vault connectors):
  Real API keys are NEVER read from environment variables or files here.
  Hosted providers are invoked through the verified connector helper CLIs,
  which attach a surrogate credential that the runtime swaps for the real
  key only for the provider's own API host:

    OpenAI:    python3 ~/workspace/custom-connectors/custom.openai/bin/openai_api.py
                 chat --model M --system S --user U --max-tokens N
               (prints only the assistant reply text on stdout)
               Default model: gpt-5-mini (verified working).

    Anthropic: same pattern under
               ~/workspace/custom-connectors/custom.anthropic/bin/
               (not registered yet -> raises ProviderNotConfigured)
    Gemini:    same pattern under
               ~/workspace/custom-connectors/custom.google-gemini/bin/
               (not registered yet -> raises ProviderNotConfigured)

  Ollama is a direct HTTP call to the local inference server (take-pc).
  MockProvider returns canned Japanese text for offline flow testing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import requests

CONNECTORS_DIR = Path.home() / "workspace" / "custom-connectors"


class ProviderError(RuntimeError):
    pass


class ProviderNotConfigured(ProviderError):
    pass


def _find_helper(connector: str, preferred: str) -> Path | None:
    """Locate a connector helper CLI: bin/<preferred> or any bin/*_api.py."""
    bin_dir = CONNECTORS_DIR / connector / "bin"
    candidate = bin_dir / preferred
    if candidate.is_file():
        return candidate
    if bin_dir.is_dir():
        matches = sorted(bin_dir.glob("*_api.py"))
        if matches:
            return matches[0]
    return None


def _run_helper_chat(
    helper: Path, model: str, system: str, user: str, max_tokens: int
) -> str:
    proc = subprocess.run(
        [
            sys.executable, str(helper), "chat",
            "--model", model,
            "--system", system,
            "--user", user,
            "--max-tokens", str(max_tokens),
        ],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        detail = tail[-1][:300] if tail else "no output"
        raise ProviderError(f"helper {helper.name} failed: {detail}")
    return proc.stdout.strip()


class BaseProvider:
    name = "base"
    default_model: str | None = None

    def chat(
        self,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 1200,
    ) -> str:
        raise NotImplementedError


class _HelperProvider(BaseProvider):
    """Provider backed by a Secure Vault connector helper CLI."""

    connector = ""
    preferred_helper = ""
    not_ready_hint = ""
    # Reasoning models spend output tokens on hidden thinking before any
    # visible text appears; keep generous defaults so replies are not
    # truncated to empty strings.
    default_max_tokens = 6000

    def __init__(self, model: str | None = None):
        self.model = model or self.default_model
        self._helper = _find_helper(self.connector, self.preferred_helper)

    def chat(self, system, user, model=None, max_tokens=None):
        if self._helper is None:
            raise ProviderNotConfigured(
                f"{self.name}: connector {self.connector} is not registered yet. "
                f"{self.not_ready_hint}"
            )
        use_model = model or self.model
        if not use_model:
            raise ProviderNotConfigured(
                f"{self.name}: no model configured for this provider yet."
            )
        budget = max_tokens or self.default_max_tokens
        out = _run_helper_chat(self._helper, use_model, system, user, budget)
        if not out.strip():
            # Reasoning likely consumed the whole budget; retry once bigger.
            out = _run_helper_chat(
                self._helper, use_model, system, user, budget * 2
            )
        if not out.strip():
            raise ProviderError(
                f"{self.name}: empty response from {use_model} "
                f"even with max_tokens={budget * 2}"
            )
        return out


class OpenAIProvider(_HelperProvider):
    name = "openai"
    connector = "custom.openai"
    preferred_helper = "openai_api.py"
    default_model = "gpt-5-mini"  # verified working via the connector
    not_ready_hint = "Register the OpenAI key in the Secure Vault first."


class AnthropicProvider(_HelperProvider):
    name = "anthropic"
    connector = "custom.anthropic"
    preferred_helper = "anthropic_api.py"
    default_model = "claude-sonnet-5"  # verified working via the connector
    not_ready_hint = (
        "Anthropic key is not in the Secure Vault yet; once registered, a "
        "helper will appear under custom.anthropic/bin/ and this provider "
        "will work with the same chat interface."
    )


class GeminiProvider(_HelperProvider):
    name = "gemini"
    connector = "custom.google-gemini"
    preferred_helper = "gemini_api.py"
    default_model = "gemini-3.1-pro-preview"  # verified working via the connector
    not_ready_hint = (
        "Gemini key is not in the Secure Vault yet; once registered, a "
        "helper will appear under custom.google-gemini/bin/ and this "
        "provider will work with the same chat interface."
    )


class OllamaProvider(BaseProvider):
    """Direct HTTP call to the local Ollama server (on take-pc).

    Reachability: the orchestrator normally runs on the Muse VM, where
    take-pc is reached through an SSH tunnel, so the default base URL is
    the tunnel endpoint http://127.0.0.1:11434. Set OLLAMA_BASE_URL to
    override (for example http://<pc-lan-address>:11434 on the home LAN).
    """

    name = "ollama"
    default_model = "qwen3:14b"
    default_max_tokens = 6000

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or os.environ.get("OLLAMA_MODEL") or self.default_model
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")

    def chat(self, system, user, model=None, max_tokens=None):
        use_model = model or self.model
        if not use_model:
            raise ProviderNotConfigured(
                "ollama: no model specified (pass the model installed on take-pc)."
            )
        budget = max_tokens or self.default_max_tokens
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": use_model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "options": {"num_predict": budget},
                },
                timeout=600,
            )
        except requests.RequestException as exc:
            raise ProviderError(
                f"ollama unreachable at {self.base_url} (is take-pc awake?): {exc}"
            ) from exc
        resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content", "").strip()
        if not content:
            raise ProviderError(
                f"ollama: empty content from {use_model} "
                f"(num_predict={budget}; thinking may have consumed it)"
            )
        return content

    def chat_json(self, system, user, schema, model=None, max_tokens=None):
        """Chat with JSON-Schema-constrained output; returns the parsed object.

        Constrained decoding makes small local models reliably produce
        machine-readable predictions (marks/bets) instead of free prose
        whose format collapses. Raises ProviderError on transport failure
        or unparseable output.
        """
        use_model = model or self.model
        budget = max_tokens or self.default_max_tokens
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": use_model,
                    "stream": False,
                    "format": schema,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "options": {"num_predict": budget, "temperature": 0.4},
                },
                timeout=900,
            )
        except requests.RequestException as exc:
            raise ProviderError(
                f"ollama unreachable at {self.base_url} (is take-pc awake?): {exc}"
            ) from exc
        resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content", "").strip()
        try:
            return json.loads(content)
        except (ValueError, TypeError) as exc:
            raise ProviderError(
                f"ollama: structured output was not valid JSON: {content[:200]}"
            ) from exc


class MockProvider(BaseProvider):
    """Canned Japanese responses so the debate flow can be tested offline.

    Picks marks deterministically from the horse list found in the prompt,
    rotated per persona so the three mock AIs disagree with each other.
    """

    name = "mock"
    _MARKS = ["◎", "○", "▲", "△"]

    def __init__(self, persona: str = "Mock"):
        self.persona = persona
        self._calls = 0

    def _horses(self, user: str) -> list[tuple[int, str]]:
        found = re.findall(r"(?m)^\s*(\d{1,2})\s+([ァ-ヶー・]+)", user)
        return [(int(n), name) for n, name in found]

    def _seed(self) -> int:
        digest = hashlib.sha256(self.persona.encode()).hexdigest()
        return int(digest[:6], 16)

    def _distinct_picks(self, horses, start):
        """Pick 4 different horses, striding by 4 from `start` and skipping
        any index already taken (the plain (start + i*4) % n formula repeats
        the first horse when n == 12)."""
        n = len(horses)
        picks, seen = [], set()
        for i in range(4):
            idx = (start + i * 4) % n
            while idx in seen and len(seen) < n:
                idx = (idx + 1) % n
            seen.add(idx)
            picks.append(horses[idx])
        return picks

    def chat(self, system, user, model=None, max_tokens=1200):
        self._calls += 1
        horses = self._horses(user)
        if "回顧" in user or "反省" in user:
            return (
                f"【{self.persona}・回顧（モック）】"
                "結果と各AIの印を見比べると、本命の読みが当たった部分と、"
                "展開の想定が外れた部分が分かれました。次回はペース想定を"
                "もう一段詰めたいところです。（これはモック応答です）"
            )
        if "議論" in user or "他のAI" in user:
            own = re.search(r"◎\s*(\d{1,2})\s+([ァ-ヶー・]+)", user)
            if own:
                target = f"{own.group(1)}番{own.group(2)}"
            elif horses:
                target = f"{horses[0][0]}番{horses[0][1]}"
            else:
                target = "自分の本命"
            text = (
                f"【{self.persona}・議論（モック）】他AIの◎には一理ありますが、"
                f"私はやはり{target}の近走内容とコース適性を"
                "重視します。展開が速くなれば差し馬、落ち着けば好位組が残る"
                "はずで、その分岐でも本命は揺るぎません。"
                "対抗の評価だけ少し引き上げて応答します。（モック応答）"
            )
            if len(horses) >= 4:
                # Final ballot: keep the persona's own ◎ when it can be
                # found, fill the other marks with distinct horses.
                start = (self._seed() + (self._calls - 1) * 3) % len(horses)
                ballot = self._distinct_picks(horses, start)
                if own:
                    own_pair = (int(own.group(1)), own.group(2))
                    if own_pair in ballot:
                        ballot.remove(own_pair)
                    ballot.insert(0, own_pair)
                    ballot = ballot[:4]
                text += "\n\n【最終票】\n" + "\n".join(
                    f"{mark} {num} {name}"
                    for mark, (num, name) in zip(self._MARKS, ballot)
                )
            return text
        if not horses:
            return f"【{self.persona}（モック）】出走馬データを受け取りました。"
        n = len(horses)
        start = (self._seed() + (self._calls - 1) * 3) % n
        picks = self._distinct_picks(horses, start)
        reasons = [
            "近走の内容が安定し、距離・コース適性も高い",
            "騎手との相性と枠順の利がある",
            "脚質的に想定ペースへ対応しやすい",
            "人気ほどの信頼はないが一発の魅力がある",
        ]
        lines = [f"【{self.persona}の予想（モック）】"]
        for mark, (num, name), reason in zip(self._MARKS, picks, reasons):
            lines.append(f"{mark} {num} {name} — {reason}。")
        lines.append("総合判断として上位4頭に印を打ちました。（モック応答です）")
        lines.append("")
        lines.append("【買い目（予算1,000円）】")
        combo = "-".join(str(x) for x in sorted(p[0] for p in picks[:3]))
        lines.append(f"単勝 {picks[0][0]} 400円")
        lines.append(f"3連複 {combo} 600円")
        return "\n".join(lines)

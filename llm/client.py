"""Thin wrapper around multiple LLM backends — OpenAI, Anthropic, Ollama."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config import LLMConfig

import httpx


class LLMClient:
    """Unified chat-completions interface.

    Currently targets the OpenAI v1 chat/completions shape which most
    providers (Anthropic via proxy, Ollama, vLLM) also speak.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._client = httpx.Client(timeout=cfg.timeout_s)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Send a chat-completion request and return the JSON response body."""
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = self._auth_headers()

        r = self._client.post(
            self._chat_url(),
            headers=headers,
            json=payload,
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _chat_url(self) -> str:
        # Always use the configured base_url; only fall back to OpenAI
        # when literally nothing is set — prevents silent misrouting.
        base = self.cfg.base_url or "https://api.openai.com"
        # If the user already included /v1 in their base_url (e.g. DeepSeek),
        # don't double it up.
        if base.rstrip("/").endswith("/v1"):
            return f"{base.rstrip('/')}/chat/completions"
        return f"{base.rstrip('/')}/v1/chat/completions"

    def _auth_headers(self) -> dict[str, str]:
        key = self.cfg.api_key or "sk-placeholder"
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

"""Web-search tool powered by Tavily API with graceful fallback."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # 确保 .env 中的 TAVILY_API_KEY 被加载


class SearchTool:
    """Encapsulates Tavily search so callers don't touch the SDK directly."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self._client = None  # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        """Execute a web search and return structured results.

        Args:
            query: Search query string.
            max_results: Maximum number of results (default 5, max 10).

        Returns:
            ``{"query": str, "results": [...], "source": "tavily" | "fallback"}``
        """
        if not self._api_key:
            return _fallback(query, max_results, reason="TAVILY_API_KEY not set")

        try:
            return self._tavily_search(query, max_results)
        except Exception as exc:
            return _fallback(query, max_results, reason=str(exc))

    def is_available(self) -> bool:
        """Return True when a real search backend is configured."""
        return bool(self._api_key)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tavily_search(self, query: str, max_results: int) -> dict[str, Any]:
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self._api_key)

        response = self._client.search(
            query=query,
            max_results=min(max_results, 10),
            search_depth="basic",
        )

        results = []
        for item in response.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                }
            )

        return {
            "query": query,
            "results": results,
            "source": "tavily",
        }


# ----------------------------------------------------------------------
# Module-level function (kept backward-compatible with registry.py)
# ----------------------------------------------------------------------

_search_tool: SearchTool | None = None


def _get_tool() -> SearchTool:
    global _search_tool
    if _search_tool is None:
        _search_tool = SearchTool()
    return _search_tool


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Module-level search function — auto-detects Tavily vs fallback.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        A dict with ``{"query": ..., "results": [...], "source": ...}``.
    """
    return _get_tool().search(query=query, max_results=max_results)


# ----------------------------------------------------------------------
# Graceful fallback
# ----------------------------------------------------------------------

def _fallback(query: str, max_results: int, reason: str) -> dict[str, Any]:
    """Return a polite degradation message when search is unavailable."""
    return {
        "query": query,
        "results": [
            {
                "title": f"[Search unavailable] {reason}",
                "url": "",
                "snippet": (
                    f"Web search is currently unavailable ({reason}). "
                    f"Please set a valid TAVILY_API_KEY in your .env file to enable real search. "
                    f"Meanwhile, I'll do my best to answer based on my training knowledge."
                ),
            }
        ],
        "source": "fallback",
    }

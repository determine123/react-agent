"""Weather tool — powered by Tavily search with graceful fallback."""

from __future__ import annotations

from tools.search import _get_tool as _get_search_tool


def get_weather(city: str) -> dict:
    """Get current weather information for a city.

    Uses Tavily search under the hood. Falls back to a helpful message
    if the search API key is not configured.

    Args:
        city: City name (e.g. "Shanghai", "Beijing", "New York").

    Returns:
        Dict with ``city``, ``summary``, ``results``, ``source``.
    """
    search_tool = _get_search_tool()
    query = f"{city} weather today temperature forecast"

    if not search_tool.is_available():
        return {
            "city": city,
            "summary": (
                f"Search API is not configured. Set TAVILY_API_KEY in .env to "
                f"enable real weather lookups. I'll try to answer from my "
                f"training knowledge instead."
            ),
            "results": [],
            "source": "fallback",
        }

    result = search_tool.search(query=query, max_results=3)

    snippets = [r.get("snippet", "") for r in result.get("results", [])]
    summary = (
        f"Weather search results for '{city}':\n" + "\n---\n".join(snippets)
        if snippets
        else f"No weather data found for '{city}'."
    )

    return {
        "city": city,
        "summary": summary,
        "results": result.get("results", []),
        "source": result.get("source", "unknown"),
    }

"""Web fetch tool — retrieve and extract text from URLs."""

from __future__ import annotations

import re


def fetch_page(url: str, max_chars: int = 3000) -> dict:
    """Fetch and extract readable text from a web page.

    Args:
        url: Full URL to fetch (must include https://).
        max_chars: Maximum characters to return (default 3000).

    Returns:
        Dict with ``url``, ``title``, ``text``, ``status``.
    """
    if not url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://", "url": url, "status": "invalid"}

    # Try httpx first (already a project dependency), fall back to urllib
    html = None
    error_msg = ""

    try:
        import httpx

        with httpx.Client(timeout=15, follow_redirects=True) as client:
            r = client.get(url, headers={"User-Agent": "AgentProject/1.0"})
            r.raise_for_status()
            html = r.text
    except Exception as exc:
        error_msg = f"httpx error: {exc}"

    # Fallback: stdlib urllib
    if html is None:
        try:
            from urllib.request import Request, urlopen

            req = Request(url, headers={"User-Agent": "AgentProject/1.0"})
            with urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            error_msg += f" | urllib error: {exc}"
            return {"error": error_msg.strip(), "url": url, "status": "fetch_failed"}

    # Extract title and text
    title = _extract_title(html)
    text = _html_to_text(html)

    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars] + f"... [truncated, total {len(text)} chars]"

    return {
        "url": url,
        "title": title or "(no title)",
        "text": text,
        "char_count": len(text),
        "status": "success",
    }


# ----------------------------------------------------------------------
# HTML parsing helpers (stdlib only — no BeautifulSoup dependency)
# ----------------------------------------------------------------------

def _extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        return re.sub(r"\s+", " ", match.group(1).strip())
    return ""


def _html_to_text(html: str) -> str:
    """Crude but dependency-free HTML-to-text converter."""
    # Remove scripts and styles
    text = re.sub(
        r"<(script|style|noscript|iframe|svg)[^>]*>.*?</\1>",
        " ",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Replace block-level tags with newlines
    text = re.sub(
        r"</?(div|p|li|h[1-6]|tr|article|section|header|footer|table|ul|ol|br|hr)[^>]*>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    # Remove remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s+", "", text)
    return text.strip()

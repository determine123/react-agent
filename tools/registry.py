"""Tool registry — callable functions mapped to OpenAI tool schemas."""

from __future__ import annotations

from typing import Any, Callable

from tools.calculator import calculate
from tools.file_ops import list_notes, read_note, write_note
from tools.search import web_search
from tools.time_tool import days_between, get_current_time
from tools.weather import get_weather
from tools.web_fetch import fetch_page

# Each entry: (callable, openai_function_def)
ToolEntry = tuple[Callable[..., Any], dict[str, Any]]


class ToolRegistry:
    """Holds every tool the agent can call and handles dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}
        self._register_builtins()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Add a tool at runtime."""
        self._tools[name] = (
            fn,
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            },
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, name: str, **kwargs: Any) -> Any:
        """Call a tool by name, forwarding keyword arguments."""
        fn, _ = self._tools[name]
        return fn(**kwargs)

    def as_openai_schema(self) -> list[dict[str, Any]]:
        """Return the tool definitions in OpenAI function-calling format."""
        return [schema for _, schema in self._tools.values()]

    def list_tools(self) -> list[str]:
        """Return registered tool names."""
        return sorted(self._tools.keys())

    # ------------------------------------------------------------------
    # Built-in tools
    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        # --- Calculator ---
        self.register(
            "calculate",
            calculate,
            "Safely evaluate a mathematical expression. Supports +, -, *, /, //, "
            "%, **, and constants pi, e, tau. Example: '(3 + 5) * 2'.",
            {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate.",
                    }
                },
                "required": ["expression"],
            },
        )

        # --- Web Search ---
        self.register(
            "web_search",
            web_search,
            "Search the web and return snippets. Use for current events, "
            "fact-checking, and real-time information.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Max results (default 5, max 10).",
                    },
                },
                "required": ["query"],
            },
        )

        # --- Date & Time ---
        self.register(
            "get_current_time",
            get_current_time,
            "Get the current date and time. Optionally specify a timezone offset.",
            {
                "type": "object",
                "properties": {
                    "timezone_offset": {
                        "type": "string",
                        "description": "UTC offset like '+8' or '-5'. Omit for local time.",
                    }
                },
                "required": [],
            },
        )

        self.register(
            "days_between",
            days_between,
            "Calculate the number of days between two dates (YYYY-MM-DD format).",
            {
                "type": "object",
                "properties": {
                    "date1": {"type": "string", "description": "First date (YYYY-MM-DD)."},
                    "date2": {"type": "string", "description": "Second date (YYYY-MM-DD)."},
                },
                "required": ["date1", "date2"],
            },
        )

        # --- Weather ---
        self.register(
            "get_weather",
            get_weather,
            "Get current weather information for a city via web search.",
            {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Shanghai' or 'Beijing'.",
                    }
                },
                "required": ["city"],
            },
        )

        # --- Web Fetch ---
        self.register(
            "fetch_page",
            fetch_page,
            "Fetch and extract readable text from a web page URL. "
            "Use this to read articles, documentation, or any web content.",
            {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL (including https://) to fetch.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return (default 3000).",
                    },
                },
                "required": ["url"],
            },
        )

        # --- File Operations ---
        self.register(
            "write_note",
            write_note,
            "Save text content to a note file. Use this to persist information "
            "the user wants to keep for later.",
            {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "File name, e.g. 'meeting_notes.txt'.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to save.",
                    },
                },
                "required": ["filename", "content"],
            },
        )

        self.register(
            "read_note",
            read_note,
            "Read a previously saved note file.",
            {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the note file to read.",
                    }
                },
                "required": ["filename"],
            },
        )

        self.register(
            "list_notes",
            list_notes,
            "List all saved note files.",
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
        )

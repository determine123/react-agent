"""Tools module — calculator, search, weather, time, web fetch, file ops."""

from tools.calculator import calculate
from tools.file_ops import list_notes, read_note, write_note
from tools.registry import ToolRegistry
from tools.search import SearchTool, web_search
from tools.time_tool import days_between, get_current_time
from tools.weather import get_weather
from tools.web_fetch import fetch_page

__all__ = [
    "calculate",
    "days_between",
    "fetch_page",
    "get_current_time",
    "get_weather",
    "list_notes",
    "read_note",
    "SearchTool",
    "ToolRegistry",
    "web_search",
    "write_note",
]

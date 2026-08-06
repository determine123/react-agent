"""Unit tests for tools/"""

import pytest
from tools.calculator import calculate
from tools.registry import ToolRegistry
from tools.time_tool import get_current_time, days_between
from tools.file_ops import write_note, read_note, list_notes


class TestCalculator:
    def test_basic_addition(self):
        assert calculate("15 + 27") == 42.0

    def test_multiplication(self):
        assert calculate("(3 + 5) * 2") == 16.0

    def test_power(self):
        assert calculate("2 ** 10") == 1024.0

    def test_constants(self):
        assert calculate("pi * 1") == pytest.approx(3.141592653589793)

    def test_constants_e(self):
        assert calculate("e * 1") == pytest.approx(2.718281828459045)

    def test_floor_div(self):
        assert calculate("7 // 2") == 3.0

    def test_modulo(self):
        assert calculate("10 % 3") == 1.0

    def test_negative(self):
        assert calculate("-5 + 3") == -2.0

    def test_complex_expression(self):
        assert calculate("(15 + 27) * (10 - 5) / 2") == 105.0

    def test_error_returns_string(self):
        result = calculate("__import__('os').system('dir')")
        assert isinstance(result, str)
        assert "error" in result.lower() or "unsupported" in result.lower()

    def test_unknown_name(self):
        result = calculate("foo + 1")
        assert isinstance(result, str)


class TestToolRegistry:
    def test_all_tools_registered(self):
        reg = ToolRegistry()
        names = reg.list_tools()
        assert "calculate" in names
        assert "web_search" in names
        assert "get_current_time" in names
        assert "days_between" in names
        assert "get_weather" in names
        assert "fetch_page" in names
        assert "write_note" in names
        assert "read_note" in names
        assert "list_notes" in names

    def test_as_openai_schema(self):
        reg = ToolRegistry()
        schemas = reg.as_openai_schema()
        assert len(schemas) >= 9
        for s in schemas:
            assert s["type"] == "function"
            assert "name" in s["function"]
            assert "description" in s["function"]
            assert "parameters" in s["function"]

    def test_dispatch_calculator(self):
        reg = ToolRegistry()
        result = reg.dispatch("calculate", expression="1 + 2")
        assert result == 3.0

    def test_dispatch_unknown_tool(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.dispatch("nonexistent_tool")

    def test_register_custom_tool(self):
        reg = ToolRegistry()

        def custom_tool(x: int) -> int:
            return x * 2

        reg.register(
            "custom",
            custom_tool,
            "A custom tool",
            {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        )
        assert "custom" in reg.list_tools()
        assert reg.dispatch("custom", x=5) == 10


class TestTimeTool:
    def test_get_current_time(self):
        result = get_current_time()
        assert "datetime" in result
        assert "date" in result
        assert "time" in result
        assert "unix_ts" in result
        assert "day_of_week" in result

    def test_get_current_time_with_timezone(self):
        result = get_current_time(timezone_offset="+8")
        assert "datetime" in result

    def test_days_between(self):
        result = days_between("2024-01-01", "2024-01-10")
        assert result["days"] == 9
        assert "description" in result

    def test_days_between_same_date(self):
        result = days_between("2024-06-15", "2024-06-15")
        assert result["days"] == 0

    def test_days_between_invalid_format(self):
        result = days_between("01-01-2024", "2024-01-10")
        assert "error" in result


class TestFileOps:
    def test_write_and_read_note(self):
        r = write_note("pytest_test.txt", "Hello pytest!")
        assert r["status"] == "written"

        r = read_note("pytest_test.txt")
        assert r["status"] == "read"
        assert "Hello pytest!" in r["content"]

    def test_read_nonexistent_note(self):
        r = read_note("nonexistent_xyz.txt")
        assert "error" in r

    def test_list_notes(self):
        write_note("pytest_list_test.txt", "test")
        r = list_notes()
        assert r["count"] >= 1
        assert len(r["files"]) >= 1

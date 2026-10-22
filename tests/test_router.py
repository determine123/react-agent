"""Unit tests for core/tool_router.py"""

import pytest

from core.tool_router import Intent, RoutingResult, ToolRouter


class TestToolRouter:
    @pytest.fixture
    def router(self):
        return ToolRouter()

    # --- Math ---
    def test_math_expression(self, router):
        r = router.classify("3 + 5")
        assert r.intent == Intent.MATH
        assert r.tool_name == "calculate"
        assert r.confidence >= 0.85
        assert r.tool_args["expression"] == "3 + 5"

    def test_math_chinese(self, router):
        r = router.classify("计算 15 加 27")
        assert r.intent == Intent.MATH

    def test_math_keywords(self, router):
        r = router.classify("求圆的面积 半径5")
        assert r.intent == Intent.MATH

    # --- Weather ---
    def test_weather_chinese(self, router):
        r = router.classify("今天上海天气怎么样")
        assert r.intent == Intent.WEATHER
        assert r.tool_name == "get_weather"
        assert r.tool_args.get("city") == "上海"

    def test_weather_keywords(self, router):
        r = router.classify("明天会下雨吗")
        assert r.intent == Intent.WEATHER

    # --- Time ---
    def test_time_chinese(self, router):
        r = router.classify("现在几点了")
        assert r.intent == Intent.TIME
        assert r.tool_name == "get_current_time"

    def test_date_question(self, router):
        r = router.classify("今天星期几")
        assert r.intent == Intent.TIME

    # --- Search ---
    def test_search_intent(self, router):
        r = router.classify("搜索 AI Agent 最新进展")
        assert r.intent == Intent.SEARCH

    def test_search_question(self, router):
        r = router.classify("什么是 RAG 技术")
        assert r.intent == Intent.SEARCH

    # --- General ---
    def test_general_chat(self, router):
        r = router.classify("你好，请帮我分析一下这个项目")
        # Complex requests should fall through to ReAct
        assert r.intent == Intent.GENERAL or r.confidence < 0.9

    def test_complex_analysis(self, router):
        # Should not hit a high-confidence tool match
        assert not router.should_skip_llm("比较一下 React 和 Vue 的优缺点")

    # --- should_skip_llm ---
    def test_skip_llm_math(self, router):
        assert router.should_skip_llm("15 + 27 = ?") is True

    def test_skip_llm_weather(self, router):
        assert router.should_skip_llm("北京天气") is True

    def test_no_skip_complex(self, router):
        assert router.should_skip_llm("帮我写一篇关于AI的文章") is False

    def test_article_request_is_not_misrouted_as_note(self, router):
        r = router.classify("帮我写一篇关于AI的文章")
        assert r.intent == Intent.GENERAL
        assert r.tool_name is None

    # --- Argument extraction ---
    def test_extract_math_expression(self, router):
        r = router.classify("计算 (15 + 27) * 3")
        assert "expression" in r.tool_args
        assert "15" in r.tool_args["expression"] or "15" in str(r.tool_args)

    def test_extract_math_expression_with_constants(self, router):
        r = router.classify("计算 2 * pi + e")
        assert r.tool_args["expression"] == "2 * pi + e"

    def test_extract_city(self, router):
        r = router.classify("深圳今天天气")
        assert r.tool_args.get("city") == "深圳"

    def test_extract_search_query(self, router):
        r = router.classify("搜索最新的 AI 论文")
        assert "query" in r.tool_args
        assert "AI" in r.tool_args["query"]

    # --- Fetch ---
    def test_fetch_url(self, router):
        r = router.classify("帮我抓取 https://example.com 的内容")
        assert r.intent == Intent.FETCH
        assert r.tool_name == "fetch_page"

    # --- Notes ---
    def test_write_note(self, router):
        r = router.classify("帮我记下来：明天下午3点开会")
        assert r.intent == Intent.NOTE_WRITE
        assert r.tool_name == "write_note"

    def test_write_note_phrase_is_still_recognized(self, router):
        r = router.classify("帮我写笔记：项目计划")
        assert r.intent == Intent.NOTE_WRITE
        assert r.tool_name == "write_note"

    def test_read_note(self, router):
        r = router.classify("看看我之前记的笔记")
        assert r.intent == Intent.NOTE_READ
        assert r.tool_name == "read_note"

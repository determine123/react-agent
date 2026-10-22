"""Tool Router — lightweight intent classifier that routes user questions
to the most relevant tools BEFORE hitting the full ReAct loop.

Saves latency and tokens by avoiding unnecessary LLM reasoning steps:

    Before (no router):
        User → LLM (think) → call tool → LLM (observe) → answer
        Every question costs at least 2 LLM calls

    After (with router):
        Math question   → calculate tool → format answer        (0 LLM calls!)
        Weather query   → weather tool → format answer          (0 LLM calls!)
        Time question   → time tool → format answer             (0 LLM calls!)
        Complex query   → full ReAct loop                       (normal flow)

Architecture:
    Rule-based fast path (regex + keywords) → 0 LLM calls, <1ms
    Embedding-based semantic path (fallback) → 1 embedding, ~50ms
    Full ReAct (no match) → normal agent flow
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Intent(str, Enum):
    """Classification labels for user intent."""

    MATH = "math"            # arithmetic, algebra — route to calculate
    WEATHER = "weather"      # weather queries — route to get_weather
    TIME = "time"            # date / time queries — route to get_current_time
    DATE_DIFF = "date_diff"  # days-between queries — route to days_between
    SEARCH = "search"        # factual / current events — route to web_search
    FETCH = "fetch"          # URL reading — route to fetch_page
    NOTE_WRITE = "note_write"  # save / write / record — route to write_note
    NOTE_READ = "note_read"    # read / recall notes — route to read_note
    GENERAL = "general"      # conversation / complex — use full ReAct


# ---------------------------------------------------------------------------
# Rule patterns — fast, zero-cost classification
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    intent: Intent
    patterns: list[str]         # regex patterns (case-insensitive)
    keywords: list[str]         # literal substrings (case-insensitive)
    tool_name: str | None = None  # direct tool dispatch (None = use full ReAct)


_RULES: list[Rule] = [
    # --- Math ---
    Rule(
        Intent.MATH,
        patterns=[
            r"(计算|等于|多少|几分之|的平方|的立方|开方|次方|阶乘)",
            r"\d+\s*[\+\-\*\/\^\％]\s*\d+",
            r"(sin|cos|tan|log|sqrt|abs|pi|圆周率)",
        ],
        keywords=[
            "calculate", "compute", "evaluate", "solve",
            "等于多少", "计算", "求值", "求和", "算一下",
            "加", "减", "乘", "除以", "的平方", "的立方",
            "面积", "周长", "体积",
        ],
        tool_name="calculate",
    ),
    # --- Weather ---
    Rule(
        Intent.WEATHER,
        patterns=[
            r"(天气|气温|下雨|下雪|刮风|台风|湿度|空气质量|雾霾)",
            r"(今天|明天|后天|这周|周末).{0,5}(天气|冷|热|温度)",
        ],
        keywords=[
            "weather", "天气", "气温", "温度", "降雨",
            "多云", "晴天", "阴天", "刮风",
        ],
        tool_name="get_weather",
    ),
    # --- Time ---
    Rule(
        Intent.TIME,
        patterns=[
            r"(现在几点|今天几号|今天星期几|当前时间|现在时间|几点了)",
            r"(今天|明天|昨天|现在).{0,3}(日期|时间)",
        ],
        keywords=[
            "现在时间", "当前时间", "今天日期", "几点了",
            "星期几", "今天几号", "当前日期",
        ],
        tool_name="get_current_time",
    ),
    # --- Date difference ---
    Rule(
        Intent.DATE_DIFF,
        patterns=[
            r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}).{0,10}(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})",
            r"(还有多少天|还剩几天|几天后|几天前|相隔.*天|间隔)",
            r"(倒计时|countdown)",
        ],
        keywords=["天数", "间隔", "还有几天", "距离", "倒计时"],
        tool_name="days_between",
    ),
    # --- Web fetch ---
    Rule(
        Intent.FETCH,
        patterns=[
            r"https?://[^\s]+",
            r"(打开|读取|访问|抓取|fetch|read).{0,5}(网页|网站|页面|链接|url)",
        ],
        keywords=["fetch this", "read this url", "抓取", "打开链接", "读取网页"],
        tool_name="fetch_page",
    ),
    # --- Note write ---
    Rule(
        Intent.NOTE_WRITE,
        patterns=[
            r"(保存|记录|写下|存下来|记住|保存下来|写笔记)",
            r"(帮我记|帮我写笔记|帮我存|帮我保存)",
        ],
        keywords=["save note", "write note", "记住", "保存", "写笔记", "记录下来"],
        tool_name="write_note",
    ),
    # --- Note read ---
    Rule(
        Intent.NOTE_READ,
        patterns=[
            r"(查看笔记|读取笔记|我的笔记|之前记的|之前保存的)",
            r"(看看.{0,5}笔记|翻翻.{0,5}笔记)",
        ],
        keywords=["read note", "我的笔记", "之前记的", "查看笔记"],
        tool_name="read_note",
    ),
    # --- Search (fuzzy — low confidence, falls through to full ReAct if uncertain) ---
    Rule(
        Intent.SEARCH,
        patterns=[
            r"(搜索|搜一下|查一下|查找|search|find|look up)",
            r"(什么是|是谁|什么时候|在哪里|怎么样|为什么)",
            r"(最新|最近的|现在的|当前的|今年|本月)",
        ],
        keywords=[
            "search for", "搜索", "查找", "最新", "最近",
            "新闻", "事件", "消息",
        ],
        tool_name="web_search",
    ),
]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@dataclass
class RoutingResult:
    intent: Intent
    tool_name: str | None      # tool to call directly (None = full ReAct)
    tool_args: dict[str, Any]  # pre-filled arguments
    confidence: float           # 0.0–1.0
    explanation: str


@dataclass
class ToolRouter:
    """Fast intent classifier with rule-first, embedding-fallback strategy."""

    rules: list[Rule] = field(default_factory=lambda: _RULES.copy())
    _embedder: Any = None  # lazy-loaded SentenceTransformer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str) -> RoutingResult:
        """Classify user input and return routing decision.

        Args:
            text: Raw user input.

        Returns:
            RoutingResult with intent, tool_name, and pre-filled args.
        """
        text_lower = text.lower().strip()

        # Phase 1: Rule-based matching (fast path)
        for rule in self.rules:
            result = self._match_rule(rule, text, text_lower)
            if result:
                return result

        # Phase 2: Semantic embedding matching (medium path)
        semantic = self._semantic_match(text)
        if semantic and semantic.confidence >= 0.6:
            return semantic

        # Phase 3: Fall through to full ReAct
        return RoutingResult(
            intent=Intent.GENERAL,
            tool_name=None,
            tool_args={},
            confidence=0.0,
            explanation="No rule matched — using full ReAct loop",
        )

    def should_skip_llm(self, text: str) -> bool:
        """Return True if this query can be handled without an LLM call."""
        result = self.classify(text)
        return result.tool_name is not None and result.confidence >= 0.8

    # ------------------------------------------------------------------
    # Rule matching
    # ------------------------------------------------------------------

    def _match_rule(self, rule: Rule, text: str, text_lower: str) -> RoutingResult | None:
        # Check regex patterns
        for pat in rule.patterns:
            if re.search(pat, text, re.IGNORECASE):
                return RoutingResult(
                    intent=rule.intent,
                    tool_name=rule.tool_name,
                    tool_args=self._extract_args(rule.intent, text),
                    confidence=0.9,
                    explanation=f"Regex pattern matched: '{pat[:60]}'",
                )

        # Check keyword substrings
        for kw in rule.keywords:
            if kw.lower() in text_lower:
                return RoutingResult(
                    intent=rule.intent,
                    tool_name=rule.tool_name,
                    tool_args=self._extract_args(rule.intent, text),
                    confidence=0.85,
                    explanation=f"Keyword matched: '{kw}'",
                )

        return None

    # ------------------------------------------------------------------
    # Argument extraction
    # ------------------------------------------------------------------

    def _extract_args(self, intent: Intent, text: str) -> dict[str, Any]:
        """Extract tool arguments from the user's text."""
        if intent == Intent.MATH:
            # Try to extract a math expression
            expr_match = re.search(
                r"((?:[\d\s+*/^%.()pie-]|tau|sqrt|abs|sin|cos|tan|log)+)",
                text,
                re.IGNORECASE,
            )
            if expr_match:
                expr = expr_match.group(1).strip()
                if len(expr) > 2:  # has actual math content
                    return {"expression": expr}
            # Fall back to sending the whole query as expression
            return {"expression": text}

        if intent == Intent.WEATHER:
            # Extract city name
            city_match = re.search(
                r"(北京|上海|广州|深圳|杭州|南京|成都|武汉|重庆|西安|天津|苏州|"
                r"长沙|郑州|东莞|青岛|沈阳|宁波|昆明|大连|厦门|合肥|佛山|福州|"
                r"哈尔滨|济南|温州|长春|石家庄|常州|泉州|南宁|贵阳|南昌|太原|"
                r"烟台|嘉兴|南通|金华|珠海|惠州|徐州|海口|乌鲁木齐|兰州|呼和浩特|"
                r"银川|西宁|拉萨|Tokyo|London|Paris|New York|Beijing|Shanghai)",
                text,
            )
            return {"city": city_match.group(1) if city_match else text}

        if intent == Intent.SEARCH:
            # Clean up the query
            query = re.sub(r"(搜索|搜一下|查一下|帮我查|帮我搜)", "", text).strip()
            return {"query": query or text}

        if intent == Intent.FETCH:
            url_match = re.search(r"https?://[^\s]+", text)
            return {"url": url_match.group(0) if url_match else text}

        if intent in (Intent.NOTE_WRITE, Intent.NOTE_READ):
            return {"filename": "note.txt"}

        if intent == Intent.DATE_DIFF:
            dates = re.findall(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text)
            if len(dates) >= 2:
                return {"date1": dates[0], "date2": dates[1]}

        return {}

    # ------------------------------------------------------------------
    # Semantic fallback (embedding-based)
    # ------------------------------------------------------------------

    def _semantic_match(self, text: str) -> RoutingResult | None:
        """Use sentence embeddings to classify intent.

        Only invoked when rule-based matching fails.  Lazy-loads the
        embedding model on first use.
        """
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                # Use a small, fast model
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

            # Pre-computed embeddings for intent descriptions
            if not hasattr(self, "_intent_embs"):
                self._intent_labels = [
                    (Intent.MATH, "calculate math expression arithmetic number"),
                    (Intent.WEATHER, "weather temperature rain forecast today"),
                    (Intent.TIME, "current time date today now clock"),
                    (Intent.SEARCH, "search find information news facts lookup"),
                    (Intent.FETCH, "fetch read url webpage website link"),
                    (Intent.NOTE_WRITE, "save write note remember record"),
                ]
                self._intent_embs = self._embedder.encode(
                    [label for _, label in self._intent_labels]
                )

            text_emb = self._embedder.encode([text])[0]
            best_score = -1.0
            best_intent = Intent.GENERAL

            for (intent, _), intent_emb in zip(self._intent_labels, self._intent_embs):
                similarity = float(text_emb @ intent_emb) / (
                    float(text_emb.dot(text_emb) ** 0.5)
                    * float(intent_emb.dot(intent_emb) ** 0.5)
                    + 1e-8
                )
                if similarity > best_score:
                    best_score = similarity
                    best_intent = intent

            if best_score >= 0.4:  # lower threshold for semantic match
                return RoutingResult(
                    intent=best_intent,
                    tool_name=self._intent_to_tool(best_intent),
                    tool_args=self._extract_args(best_intent, text),
                    confidence=best_score,
                    explanation=f"Semantic match: {best_intent.value} (score={best_score:.2f})",
                )
        except ImportError:
            pass  # sentence-transformers not installed — skip semantic path

        return None

    def _intent_to_tool(self, intent: Intent) -> str | None:
        mapping = {
            Intent.MATH: "calculate",
            Intent.WEATHER: "get_weather",
            Intent.TIME: "get_current_time",
            Intent.SEARCH: "web_search",
            Intent.FETCH: "fetch_page",
            Intent.NOTE_WRITE: "write_note",
            Intent.NOTE_READ: "read_note",
        }
        return mapping.get(intent)

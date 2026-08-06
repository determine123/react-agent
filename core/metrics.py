"""Structured logging & performance metrics for the Agent.

Provides:
- JSON-formatted structured logging via python-json-logger
- Per-step metrics: LLM call duration, tool execution time, token usage
- Metrics persistence to logs/metrics.jsonl

Usage:
    from core.metrics import MetricsCollector

    metrics = MetricsCollector()
    with metrics.track_llm_call():
        response = llm.chat(...)
    metrics.flush()  # persist to disk
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Structured JSON logger
# ---------------------------------------------------------------------------

def setup_json_logger(name: str = "agent", log_file: str = "logs/agent.jsonl") -> logging.Logger:
    """Configure a logger that emits JSON lines to file + human-readable to console.

    Args:
        name: Logger name.
        log_file: Path to the JSON-lines log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler — human-readable
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    # File handler — structured JSON
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_JsonFormatter())
    logger.addHandler(file_handler)

    return logger


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])
        # Merge extra fields passed via `extra={...}`
        for key in dir(record):
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "module",
                "msecs", "message", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread",
                "threadName",
            } and not key.startswith("_"):
                val = getattr(record, key, None)
                if val is not None and key not in entry:
                    entry[key] = val
        return json.dumps(entry, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------


@dataclass
class StepMetrics:
    step: int
    llm_duration_ms: float = 0
    llm_model: str = ""
    tools_called: list[str] = field(default_factory=list)
    tools_duration_ms: float = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    error: str = ""


@dataclass
class SessionMetrics:
    session_id: str
    steps: list[StepMetrics] = field(default_factory=list)
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_duration_ms: float = 0
    start_ts: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "steps": [
                {
                    "step": s.step,
                    "llm_duration_ms": round(s.llm_duration_ms, 2),
                    "llm_model": s.llm_model,
                    "tools_called": s.tools_called,
                    "tools_duration_ms": round(s.tools_duration_ms, 2),
                    "tokens_prompt": s.tokens_prompt,
                    "tokens_completion": s.tokens_completion,
                }
                for s in self.steps
            ],
        }


class MetricsCollector:
    """Tracks per-step and per-session performance metrics.

    Usage::

        collector = MetricsCollector()
        collector.start_session()

        with collector.track_llm() as ctx:
            raw = llm.chat(...)
            ctx["tokens_prompt"] = raw.get("usage", {}).get("prompt_tokens", 0)
            ctx["tokens_completion"] = raw.get("usage", {}).get("completion_tokens", 0)

        collector.record_step(tools=["calculate"], tools_duration_ms=12.3)
        collector.end_session()
        collector.flush()
    """

    def __init__(self, metrics_path: str = "logs/metrics.jsonl") -> None:
        self._path = Path(metrics_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._session: SessionMetrics | None = None
        self._current_step: StepMetrics | None = None
        self._llm_start: float = 0
        self._tools_start: float = 0
        self._llm_context: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, session_id: str | None = None) -> str:
        sid = session_id or uuid.uuid4().hex[:12]
        self._session = SessionMetrics(
            session_id=sid,
            start_ts=time.perf_counter(),
        )
        return sid

    def end_session(self) -> None:
        if self._session:
            self._session.total_duration_ms = (time.perf_counter() - self._session.start_ts) * 1000

    # ------------------------------------------------------------------
    # Step tracking
    # ------------------------------------------------------------------

    def start_step(self, step_num: int) -> None:
        self._current_step = StepMetrics(step=step_num)

    def record_llm(self, model: str = "", duration_ms: float = 0, **kwargs) -> None:
        if self._current_step:
            self._current_step.llm_model = model
            self._current_step.llm_duration_ms = duration_ms
            for k, v in kwargs.items():
                setattr(self._current_step, k, v)

    def record_tools(self, names: list[str], duration_ms: float = 0) -> None:
        if self._current_step:
            self._current_step.tools_called = names
            self._current_step.tools_duration_ms = duration_ms

    def record_error(self, error: str) -> None:
        if self._current_step:
            self._current_step.error = error

    def commit_step(self) -> None:
        if self._session and self._current_step:
            self._session.steps.append(self._current_step)
            self._session.total_llm_calls += 1
            if self._current_step.tools_called:
                self._session.total_tool_calls += len(self._current_step.tools_called)
        self._current_step = None

    # ------------------------------------------------------------------
    # Context managers for timing
    # ------------------------------------------------------------------

    @contextmanager
    def track_llm(self):
        """Context manager that times an LLM call and captures metadata."""
        self._llm_context = {}
        self._llm_start = time.perf_counter()
        try:
            yield self._llm_context
        finally:
            elapsed = (time.perf_counter() - self._llm_start) * 1000
            if self._current_step:
                self._current_step.llm_duration_ms = elapsed
                for k, v in self._llm_context.items():
                    setattr(self._current_step, k, v)

    @contextmanager
    def track_tools(self):
        """Context manager that times tool execution."""
        self._tools_start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - self._tools_start) * 1000
            if self._current_step:
                self._current_step.tools_duration_ms = elapsed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write the current session metrics to disk."""
        if self._session and self._session.steps:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._session.to_dict(), ensure_ascii=False) + "\n")

    def get_summary(self) -> dict[str, Any] | None:
        if not self._session:
            return None
        return self._session.to_dict()

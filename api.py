"""FastAPI application — REST API for Agent and RAG services.

Endpoints:
    POST /chat          — ReAct agent chat (multi-turn with memory)
    POST /chat/stream   — Streaming agent chat (SSE)
    POST /rag           — RAG-based Q&A
    POST /rag/index     — Index documents from a directory
    GET  /tools         — List available tools
    GET  /health        — Health check

Usage:
    python main.py --api                # start server at 0.0.0.0:8000
    python main.py --api --port 8080    # custom port
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from core.session_store import AgentSessionStore

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", min_length=1)
    session_id: str | None = Field(
        None,
        description=(
            "Conversation id. Omit for a new isolated session; reuse the id "
            "returned by the API to continue the conversation."
        ),
        min_length=1,
        max_length=128,
    )
    system_prompt: str | None = Field(
        None, description="Custom system prompt (uses default if unset)"
    )
    clear_history: bool = Field(
        False, description="Reset conversation history before this message"
    )

    @field_validator("session_id")
    @classmethod
    def _normalise_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("session_id must not be blank")
        return value


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    tool_calls_made: list[str] = Field(default_factory=list)
    steps: int = 1
    duration_ms: float = 0


class RAGRequest(BaseModel):
    question: str = Field(..., description="Question to answer", min_length=1)
    top_k: int = Field(5, description="Number of chunks to retrieve", ge=1, le=20)
    return_sources: bool = Field(False, description="Include source citations")


class RAGResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = []
    chunk_count: int = 0


class IndexRequest(BaseModel):
    directory: str = Field(..., description="Path to document directory")


class IndexResponse(BaseModel):
    chunks_indexed: int
    directory: str


class ToolsResponse(BaseModel):
    tools: list[str]
    count: int


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

# Global state (lazy-initialised per worker)
_config = None
_qa = None


def _get_config():
    global _config
    if _config is None:
        from config import load_config

        _config = load_config()
    return _config


def _new_agent_state():
    from core.state import AgentState

    cfg = _get_config()
    return AgentState(system_prompt=cfg.system_prompt)


def _build_agent_loop(config, state):
    from core.agent_loop import AgentLoop

    return AgentLoop(config=config, state=state)


def _new_session_id() -> str:
    return f"anon-{secrets.token_urlsafe(24)}"


def _run_agent_turn(req: ChatRequest, state):
    """Execute one turn while the caller holds the session lock."""
    cfg = _get_config()
    history_start = 0 if req.clear_history else len(state.messages)

    if req.clear_history:
        state.messages.clear()
        state.scratchpad.clear()
        state.finished = False
        state.step_count = 0

    if req.system_prompt:
        state.system_prompt = req.system_prompt

    state.finished = False
    state.step_count = 0

    t0 = time.perf_counter()
    agent = _build_agent_loop(cfg, state)
    answer = agent.run(req.message)
    elapsed = (time.perf_counter() - t0) * 1000

    # Report tool calls made by this turn, not by earlier turns in the session.
    tool_calls_made = []
    for msg in state.messages[history_start:]:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append(tc["function"]["name"])

    return answer, tool_calls_made, state.step_count, round(elapsed, 2)


def _run_session_turn(
    session_store: AgentSessionStore,
    session_id: str,
    req: ChatRequest,
):
    with session_store.acquire(session_id) as state:
        return _run_agent_turn(req, state)


def _get_qa():
    global _qa
    if _qa is None:
        from rag.qa import RAGQA

        _qa = RAGQA(_get_config())
    return _qa


def _answer_rag_question(qa, question: str, top_k: int):
    """Run the synchronous RAG pipeline outside the API event loop."""
    chunk_count = qa.index_count()
    if chunk_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Use POST /rag/index first.",
        )
    result = qa.answer(question, top_k=top_k, return_sources=True)
    return chunk_count, result


def _index_rag_documents(qa, directory_text: str):
    """Validate and index a directory without blocking the event loop."""
    directory = Path(directory_text)
    if not directory.exists() or not directory.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Directory not found: {directory_text}",
        )
    return qa.index_documents(str(directory)), str(directory)


def create_app(*, session_store: AgentSessionStore | None = None) -> FastAPI:
    sessions = session_store or AgentSessionStore(_new_agent_state)

    app = FastAPI(
        title="Agent Project API",
        description="ReAct Agent + RAG Q&A with tool calling",
        version="1.0.0",
    )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": time.time()}

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        """Send a message to the ReAct agent. Multi-turn by default."""
        session_id = req.session_id or _new_session_id()
        answer, tool_calls_made, steps, duration_ms = await asyncio.to_thread(
            _run_session_turn, sessions, session_id, req
        )

        return ChatResponse(
            answer=answer,
            session_id=session_id,
            tool_calls_made=tool_calls_made,
            steps=steps,
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # Chat streaming (SSE)
    # ------------------------------------------------------------------

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        """Stream the agent's response via Server-Sent Events."""
        session_id = req.session_id or _new_session_id()

        async def event_generator():
            answer, _, _, _ = await asyncio.to_thread(
                _run_session_turn, sessions, session_id, req
            )

            # Yield the final answer as a single SSE event
            payload = {"answer": answer, "session_id": session_id}
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
        )

    # ------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------

    @app.post("/rag", response_model=RAGResponse)
    async def rag_qa(req: RAGRequest):
        """Answer a question using RAG over the indexed document corpus."""
        qa = _get_qa()
        chunk_count, result = await asyncio.to_thread(
            _answer_rag_question, qa, req.question, req.top_k
        )
        return RAGResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            chunk_count=chunk_count,
        )

    @app.post("/rag/index", response_model=IndexResponse)
    async def rag_index(req: IndexRequest):
        """Index documents from a directory into the vector store."""
        qa = _get_qa()
        count, directory = await asyncio.to_thread(
            _index_rag_documents, qa, req.directory
        )
        return IndexResponse(chunks_indexed=count, directory=directory)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @app.get("/tools", response_model=ToolsResponse)
    async def list_tools():
        """Return all registered tool names."""
        from tools.registry import ToolRegistry

        registry = ToolRegistry()
        names = registry.list_tools()
        return ToolsResponse(tools=names, count=len(names))

    return app

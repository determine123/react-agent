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

import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", min_length=1)
    system_prompt: str | None = Field(
        None, description="Custom system prompt (uses default if unset)"
    )
    clear_history: bool = Field(
        False, description="Reset conversation history before this message"
    )


class ChatResponse(BaseModel):
    answer: str
    tool_calls_made: list[str] = []
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
_agent_state = None
_qa = None


def _get_config():
    global _config
    if _config is None:
        from config import load_config

        _config = load_config()
    return _config


def _get_agent_state():
    global _agent_state
    if _agent_state is None:
        from core.state import AgentState

        cfg = _get_config()
        _agent_state = AgentState(system_prompt=cfg.system_prompt)
    return _agent_state


def _get_qa():
    global _qa
    if _qa is None:
        from rag.qa import RAGQA

        _qa = RAGQA(_get_config())
    return _qa


def create_app() -> FastAPI:
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
        from core.agent_loop import AgentLoop

        cfg = _get_config()
        state = _get_agent_state()

        if req.clear_history:
            state.messages.clear()
            state.finished = False
            state.step_count = 0

        if req.system_prompt:
            state.system_prompt = req.system_prompt

        state.finished = False
        state.step_count = 0

        t0 = time.perf_counter()
        agent = AgentLoop(config=cfg, state=state)
        answer = agent.run(req.message)
        elapsed = (time.perf_counter() - t0) * 1000

        # Collect tool call names from assistant messages
        tool_calls_made = []
        for msg in state.messages:
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_made.append(tc["function"]["name"])

        return ChatResponse(
            answer=answer,
            tool_calls_made=tool_calls_made,
            steps=state.step_count,
            duration_ms=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Chat streaming (SSE)
    # ------------------------------------------------------------------

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        """Stream the agent's response via Server-Sent Events."""

        async def event_generator():
            from core.agent_loop import AgentLoop

            cfg = _get_config()
            state = _get_agent_state()

            if req.clear_history:
                state.messages.clear()
            state.finished = False
            state.step_count = 0

            agent = AgentLoop(config=cfg, state=state)
            answer = agent.run(req.message)

            # Yield the final answer as a single SSE event
            yield f"data: {json.dumps({'answer': answer})}\n\n"
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

        if qa.index_count() == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents indexed. Use POST /rag/index first.",
            )

        result = qa.answer(req.question, top_k=req.top_k, return_sources=True)
        return RAGResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            chunk_count=qa.index_count(),
        )

    @app.post("/rag/index", response_model=IndexResponse)
    async def rag_index(req: IndexRequest):
        """Index documents from a directory into the vector store."""
        directory = Path(req.directory)
        if not directory.exists() or not directory.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Directory not found: {req.directory}",
            )

        qa = _get_qa()
        count = qa.index_documents(str(directory))
        return IndexResponse(chunks_indexed=count, directory=str(directory))

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

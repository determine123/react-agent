"""Multi-Agent collaboration via LangGraph — Planner + Executor pattern.

Architecture::

    User Request
        │
        ▼
    ┌──────────┐
    │ PLANNER  │ ← Decomposes the task into subtasks
    └────┬─────┘
         │ subtasks[]
         ▼
    ┌──────────┐
    │ EXECUTOR │ ← Executes each subtask (ReAct loop w/ tools)
    └────┬─────┘
         │ results[]
         ▼
    ┌──────────┐
    │ SYNTHESIS│ ← Combines results into final answer
    └────┬─────┘
         │
         ▼
      Final Answer

Usage::

    from graph.multi_agent import MultiAgentOrchestrator
    from config import load_config

    orch = MultiAgentOrchestrator(load_config())
    result = orch.run("Write a report on AI trends in 2025")
    print(result["final_answer"])
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class MultiAgentState(TypedDict, total=False):
    """Shared state across Planner, Executor, and Synthesis nodes."""

    messages: Annotated[list, add_messages]
    user_request: str

    # Planner output
    subtasks: list[str]

    # Executor output
    task_results: list[dict[str, Any]]  # [{"task": str, "answer": str, "tools": []}]

    # Routing
    next_step: str
    current_task_index: int

    # Final output
    final_answer: str
    error: str


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = (
    "You are a strategic task planner. Given a user's request, break it down "
    "into 2-5 independent subtasks. Each subtask should be a clear, "
    "self-contained instruction that can be executed by an AI assistant with "
    "access to web search, calculator, and other tools.\n\n"
    "Output ONLY a JSON array of strings. Example:\n"
    '["Search for the latest AI trends in 2025", '
    '"Summarize the top 3 trends", '
    '"Write a conclusion paragraph"]'
)

_EXECUTOR_SYSTEM = (
    "You are executing ONE subtask from a larger plan. Focus ONLY on this "
    "specific task. Be thorough but concise. Use tools (search, calculate, "
    "etc.) as needed."
)

_SYNTHESIS_SYSTEM = (
    "You are a report writer. Combine the results of individually executed "
    "subtasks into one comprehensive, well-structured answer that directly "
    "addresses the user's original request. Use markdown formatting for clarity."
)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class MultiAgentOrchestrator:
    """High-level API for running multi-agent workflows."""

    def __init__(self, config) -> None:
        self._config = config
        self._graph = build_multi_agent_graph()

    def run(self, user_request: str) -> dict[str, Any]:
        """Execute the multi-agent pipeline for a user request.

        Returns:
            Dict with ``final_answer``, ``subtasks``, ``task_results``.
        """
        initial: MultiAgentState = {
            "messages": [{"role": "user", "content": user_request}],
            "user_request": user_request,
            "subtasks": [],
            "task_results": [],
            "next_step": "planner",
            "current_task_index": 0,
            "final_answer": "",
            "error": "",
        }
        result = self._graph.invoke(initial)
        return result

    @property
    def graph(self):
        return self._graph


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def planner_node(state: MultiAgentState) -> dict:
    """Decompose user request into subtasks via LLM."""
    from llm.client import LLMClient
    from config import load_config

    cfg = load_config()
    llm = LLMClient(cfg.llm)

    request = state.get("user_request", "")
    logger.info("Planner: decomposing request: %s", request[:80])

    msgs = [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": request},
    ]
    raw = llm.chat(messages=msgs)
    response = raw["choices"][0]["message"].get("content", "[]")

    # Parse JSON array
    try:
        # Find JSON array in response
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            subtasks = json.loads(response[start:end])
        else:
            subtasks = [request]  # fallback: execute as single task
    except (json.JSONDecodeError, Exception):
        subtasks = [request]

    logger.info("Planner: %d subtasks created", len(subtasks))
    return {
        "subtasks": subtasks,
        "next_step": "executor",
        "current_task_index": 0,
        "task_results": [],
    }


def executor_node(state: MultiAgentState) -> dict:
    """Execute the current subtask using the ReAct agent."""
    from core.agent_loop import AgentLoop
    from core.state import AgentState
    from config import load_config

    cfg = load_config()
    subtasks = state.get("subtasks", [])
    idx = state.get("current_task_index", 0)
    task_results = list(state.get("task_results", []))

    if idx >= len(subtasks):
        return {"next_step": "synthesis"}

    task = subtasks[idx]
    logger.info("Executor: step %d/%d — %s", idx + 1, len(subtasks), task[:80])

    # Execute the subtask
    agent_state = AgentState(
        system_prompt=(
            f"{_EXECUTOR_SYSTEM}\n\n"
            f"Original user request: {state.get('user_request', '')}\n"
            f"Your subtask ({idx + 1}/{len(subtasks)}): {task}"
        )
    )
    agent = AgentLoop(config=cfg, state=agent_state)
    answer = agent.run(task)

    # Collect tools used
    tools_used = []
    for msg in agent_state.messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc["function"]["name"])

    task_results.append(
        {
            "task": task,
            "answer": answer,
            "tools_used": tools_used,
            "step": idx + 1,
        }
    )

    next_idx = idx + 1
    return {
        "task_results": task_results,
        "current_task_index": next_idx,
        "next_step": "executor" if next_idx < len(subtasks) else "synthesis",
    }


def synthesis_node(state: MultiAgentState) -> dict:
    """Combine all task results into a final answer."""
    from llm.client import LLMClient
    from config import load_config

    cfg = load_config()
    llm = LLMClient(cfg.llm)

    request = state.get("user_request", "")
    task_results = state.get("task_results", [])

    logger.info("Synthesis: combining %d subtask results", len(task_results))

    # Build context from task results
    results_text = ""
    for r in task_results:
        results_text += (
            f"## Subtask {r['step']}: {r['task']}\n\n{r['answer']}\n\n"
        )

    msgs = [
        {"role": "system", "content": _SYNTHESIS_SYSTEM},
        {
            "role": "user",
            "content": (
                f"**User's original request:** {request}\n\n"
                f"**Subtask results:**\n\n{results_text}\n\n"
                "Please synthesize these into a comprehensive final answer."
            ),
        },
    ]
    raw = llm.chat(messages=msgs)
    final_answer = raw["choices"][0]["message"].get("content", "")

    return {
        "final_answer": final_answer,
        "next_step": END,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_multi_agent_graph() -> StateGraph:
    """Construct the multi-agent workflow graph.

    Flow::

        planner ──▶ executor ──(loop)──▶ synthesis ──▶ END
    """
    wf = StateGraph(MultiAgentState)

    wf.add_node("planner", planner_node)
    wf.add_node("executor", executor_node)
    wf.add_node("synthesis", synthesis_node)

    wf.set_entry_point("planner")

    wf.add_conditional_edges(
        "planner",
        lambda s: s.get("next_step", "executor"),
        {"executor": "executor", END: END},
    )

    wf.add_conditional_edges(
        "executor",
        lambda s: s.get("next_step", "synthesis"),
        {
            "executor": "executor",  # loop for next subtask
            "synthesis": "synthesis",
            END: END,
        },
    )

    wf.add_edge("synthesis", END)

    return wf.compile()

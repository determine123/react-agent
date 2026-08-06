"""LangGraph workflow — state-machine view of the agent loop.

Orchestrates agent → tool → agent → finish with conditional routing,
providing a visualisable, traceable execution graph.

Usage::

    from graph.workflow import AgentWorkflow
    from config import load_config

    wf = AgentWorkflow(load_config())
    result = wf.run("What is (15 + 27) * 3?")
    print(result["final_answer"])
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """Shared state propagated through the graph nodes."""

    messages: Annotated[list, add_messages]
    next_step: str
    tool_results: dict[str, Any]
    final_answer: str
    error: str


# ---------------------------------------------------------------------------
# Workflow wrapper
# ---------------------------------------------------------------------------

class AgentWorkflow:
    """High-level wrapper that compiles and runs the LangGraph workflow.

    Internally delegates to the ReAct ``AgentLoop`` for LLM calls and tool
    execution while using LangGraph for routing and state tracking.
    """

    def __init__(self, config) -> None:
        self._config = config
        self._graph = build_workflow()

    def run(self, user_input: str) -> dict[str, Any]:
        """Execute the workflow for a single user turn.

        Returns:
            The final ``GraphState`` dict with ``final_answer`` populated.
        """
        initial: GraphState = {
            "messages": [{"role": "user", "content": user_input}],
            "next_step": "agent",
            "tool_results": {},
            "final_answer": "",
            "error": "",
        }
        result = self._graph.invoke(initial)
        return result

    @property
    def graph(self):
        """The compiled LangGraph state graph (for visualisation)."""
        return self._graph


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def agent_node(state: GraphState) -> dict:
    """LLM reasoning step — delegate to AgentLoop to decide next action.

    In production, this passes state["messages"] to the LLM and detects
    whether tool calls are requested.
    """
    from core.agent_loop import AgentLoop
    from core.state import AgentState
    from config import load_config

    cfg = load_config()
    agent_state = AgentState(system_prompt=cfg.system_prompt)

    # Reconstruct agent state from graph messages
    for msg in state.get("messages", []):
        role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        from core.state import Message

        agent_state.add_message(Message(role=role, content=content))

    agent = AgentLoop(config=cfg, state=agent_state)

    # Run one iteration to determine next step
    response = agent._call_llm()

    if agent.state.finished:
        return {
            "next_step": "finish",
            "messages": [{"role": "assistant", "content": agent.state.last_assistant_text()}],
        }

    # Extract tool calls for the tool node
    tool_calls = response.get("tool_calls", [])
    tool_names = [tc["function"]["name"] for tc in (tool_calls or [])]
    return {
        "next_step": "tools" if tool_calls else "finish",
        "messages": [
            {
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": tool_calls,
            }
        ],
        "tool_results": {"pending": tool_names},
    }


def tool_node(state: GraphState) -> dict:
    """Execute pending tool calls and attach results.

    Reads the last assistant message's tool_calls, dispatches each via
    ToolRegistry, and appends tool-result messages.
    """
    from tools.registry import ToolRegistry
    import json

    tools = ToolRegistry()
    messages: list[dict] = []
    results: dict[str, Any] = {}

    # Find the last assistant message with tool_calls
    for msg in reversed(state.get("messages", [])):
        msg_dict = msg if isinstance(msg, dict) else {"role": getattr(msg, "role", ""), "content": getattr(msg, "content", "")}
        tcs = msg_dict.get("tool_calls") or (
            getattr(msg, "tool_calls", None) if not isinstance(msg, dict) else None
        )
        if tcs:
            for tc in tcs:
                tc = tc if isinstance(tc, dict) else {}
                fn_name = tc.get("function", {}).get("name", "")
                fn_args_raw = tc.get("function", {}).get("arguments", "{}")
                tc_id = tc.get("id", "")

                try:
                    fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                except json.JSONDecodeError:
                    fn_args = {}

                try:
                    result = tools.dispatch(fn_name, **fn_args)
                except Exception as exc:
                    result = {"error": str(exc)}

                result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                results[fn_name] = result_str
                messages.append(
                    {
                        "role": "tool",
                        "content": result_str,
                        "tool_call_id": tc_id,
                    }
                )
            break

    return {
        "next_step": "agent",
        "messages": messages,
        "tool_results": results,
    }


def finish_node(state: GraphState) -> dict:
    """Produce the final answer from the last assistant message."""
    for msg in reversed(state.get("messages", [])):
        msg_dict = msg if isinstance(msg, dict) else {"role": getattr(msg, "role", ""), "content": getattr(msg, "content", "")}
        if msg_dict.get("role") == "assistant" and msg_dict.get("content"):
            return {
                "final_answer": msg_dict["content"],
                "next_step": END,
            }

    return {
        "final_answer": "I was unable to produce an answer.",
        "next_step": END,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_workflow() -> StateGraph:
    """Construct and compile the agent workflow graph.

    Flow::

        agent ──(needs tools?)──▶ tools ──▶ agent (loop)
          │
          └──(finished?)────────▶ finish ──▶ END
    """
    wf = StateGraph(GraphState)

    wf.add_node("agent", agent_node)
    wf.add_node("tools", tool_node)
    wf.add_node("finish", finish_node)

    wf.set_entry_point("agent")

    # Conditional edge: agent → tools | finish
    wf.add_conditional_edges(
        "agent",
        lambda s: s.get("next_step", "finish"),
        {
            "tools": "tools",
            "finish": "finish",
            END: END,
        },
    )
    wf.add_edge("tools", "agent")  # loop back after tool execution
    wf.add_edge("finish", END)

    return wf.compile()

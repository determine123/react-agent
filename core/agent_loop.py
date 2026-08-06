"""The main agent loop: perceive → reason → act → observe → repeat.

Implements a ReACT-style agent with robust multi-turn tool calling.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from core.state import AgentState, Message

if TYPE_CHECKING:
    from config import Config

from llm.client import LLMClient
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentLoop:
    """ReACT-style agent loop with tool-use support.

    The loop runs until the model emits a final answer (no tool_calls)
    or the step budget is exhausted.
    """

    def __init__(self, config: Config, state: AgentState) -> None:
        self.config = config
        self.state = state
        self.llm = LLMClient(config.llm)
        self.tools = ToolRegistry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str | None = None) -> str:
        """Run the agent loop to completion.

        Args:
            user_input: If given, appended as a user message before the loop
                        starts.  Pass ``None`` to resume from the last
                        assistant message.

        Returns:
            The final text answer from the assistant.
        """
        if user_input:
            self.state.add_message(Message(role="user", content=user_input))

        while self.state.step_count < self.config.max_steps and not self.state.finished:
            self.state.step_count += 1

            if self.config.verbose:
                logger.info(
                    "--- Step %d/%d ---",
                    self.state.step_count,
                    self.config.max_steps,
                )

            response = self._call_llm()

            if self.state.finished:
                break

            self._handle_tool_calls(response)

        if not self.state.finished and self.state.step_count >= self.config.max_steps:
            logger.warning("Agent hit max-steps limit (%d).", self.config.max_steps)
            return (
                "I've reached the maximum number of reasoning steps. "
                "Here's what I have so far:\n\n"
                + self._best_effort_answer()
            )

        return self.state.last_assistant_text() or ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self) -> dict[str, Any]:
        """Send the current message history to the LLM.

        Returns:
            The raw ``choices[0].message`` dict from the API response.
        """
        msgs = [{"role": "system", "content": self.state.system_prompt}]
        msgs += self.state.to_chat_format_with_tools()
        tool_defs = self.tools.as_openai_schema()

        raw = self.llm.chat(messages=msgs, tools=tool_defs or None)
        choice = raw["choices"][0]
        assistant_msg = choice["message"]

        content = assistant_msg.get("content") or ""
        tool_calls = assistant_msg.get("tool_calls")

        # Record the assistant turn in conversation history
        self.state.add_message(
            Message(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            )
        )

        # If the model didn't request any tools, this is a final answer
        if not tool_calls:
            self.state.finished = True
        elif self.config.verbose:
            names = [tc["function"]["name"] for tc in tool_calls]
            logger.info("Model requested tools: %s", names)

        return assistant_msg

    def _handle_tool_calls(self, assistant_msg: dict[str, Any]) -> None:
        """Execute every tool call the model requested and feed results back.

        Each tool result is appended as a ``role: "tool"`` message with the
        matching ``tool_call_id`` so the API can pair requests and responses.
        """
        for tc in assistant_msg.get("tool_calls", []) or []:
            tc_id = tc.get("id", "")
            fn_info = tc.get("function", {})
            fn_name = fn_info.get("name", "")
            fn_args_raw = fn_info.get("arguments", "{}")

            # Parse arguments safely
            try:
                fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
            except json.JSONDecodeError as exc:
                self.state.add_message(
                    Message(
                        role="tool",
                        content=json.dumps(
                            {"error": f"Invalid JSON arguments: {exc}"},
                            ensure_ascii=False,
                        ),
                        tool_call_id=tc_id,
                    )
                )
                continue

            # Dispatch and capture result (never crash on tool errors)
            try:
                result = self.tools.dispatch(fn_name, **fn_args)
            except Exception as exc:
                logger.exception("Tool '%s' raised an exception.", fn_name)
                result = {"error": f"Tool execution failed: {exc}"}

            # Ensure result is a JSON-serialisable string for the API
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)

            self.state.add_message(
                Message(
                    role="tool",
                    content=result,
                    tool_call_id=tc_id,
                )
            )

    def _best_effort_answer(self) -> str:
        """Return the last assistant content, or a polite fallback."""
        last = self.state.last_assistant()
        if last and last.content:
            return last.content
        return "I wasn't able to complete the reasoning process in time. Could you rephrase or simplify your question?"

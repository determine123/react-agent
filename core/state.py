"""Immutable-ish agent state carried across loop iterations.

Models the conversation history with first-class support for tool-call /
tool-result message pairs required by OpenAI-compatible APIs (DeepSeek, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A single turn in the conversation.

    Attributes:
        role: One of ``"system"``, ``"user"``, ``"assistant"``, ``"tool"``.
        content: The text body (may be empty for assistant messages that only
                 contain tool_calls).
        tool_call_id: Required for ``role="tool"`` — must match the ``id``
                      of the corresponding assistant ``tool_calls`` entry.
        tool_calls: Set on assistant messages that request tool invocations.
    """

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None
    tool_calls: list[dict] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Holds every piece of transient state the agent loop needs."""

    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    step_count: int = 0

    # Scratch-pad for tool results / intermediate reasoning
    scratchpad: dict[str, Any] = field(default_factory=dict)

    # Flag set when the agent emits a final answer (no tool_calls)
    finished: bool = False

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_message(self, msg: Message) -> None:
        self.messages.append(msg)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def last_assistant(self) -> Message | None:
        """Return the most recent assistant message (if any)."""
        for m in reversed(self.messages):
            if m.role == "assistant":
                return m
        return None

    def last_assistant_text(self) -> str:
        """Return the text content of the last assistant message."""
        m = self.last_assistant()
        return m.content if m else ""

    # ------------------------------------------------------------------
    # Serialisation for chat-completion APIs
    # ------------------------------------------------------------------

    def to_chat_format(self) -> list[dict[str, Any]]:
        """Convert internal messages to the basic chat-completion shape.

        Tool messages are excluded — use this for contexts that don't need
        function-calling history.
        """
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
            if m.role != "tool"
        ]

    def to_chat_format_with_tools(self) -> list[dict[str, Any]]:
        """Convert internal messages, preserving tool-call / tool-result pairs.

        The output strictly follows the format expected by DeepSeek and other
        OpenAI-compatible APIs:

        - Assistant messages with tool_calls include the ``tool_calls`` array.
        - Tool messages include ``tool_call_id`` to match the request.
        - ``content`` is always present (empty string if absent).
        """
        msgs: list[dict[str, Any]] = []
        for m in self.messages:
            d: dict[str, Any] = {
                "role": m.role,
                "content": m.content or "",
            }
            if m.tool_calls:
                d["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            msgs.append(d)
        return msgs

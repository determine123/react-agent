"""Unit tests for core/state.py and core/agent_loop.py"""

import pytest
from core.state import AgentState, Message


class TestMessage:
    def test_create_user_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_call_id is None
        assert msg.tool_calls is None

    def test_create_assistant_with_tool_calls(self):
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "calculate", "arguments": '{"expression": "1+1"}'},
            }
        ]
        msg = Message(role="assistant", content="", tool_calls=tool_calls)
        assert msg.role == "assistant"
        assert msg.tool_calls == tool_calls
        assert msg.content == ""

    def test_create_tool_message(self):
        msg = Message(role="tool", content="42", tool_call_id="call_1")
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_1"

    def test_metadata_default(self):
        msg = Message(role="user", content="hi")
        assert msg.metadata == {}


class TestAgentState:
    def test_add_messages(self):
        state = AgentState(system_prompt="You are helpful.")
        state.add_message(Message(role="user", content="Hello"))
        state.add_message(Message(role="assistant", content="Hi!"))
        assert len(state.messages) == 2

    def test_last_assistant(self):
        state = AgentState(system_prompt="test")
        state.add_message(Message(role="user", content="Q"))
        state.add_message(Message(role="assistant", content="A1"))
        state.add_message(Message(role="assistant", content="A2"))
        last = state.last_assistant()
        assert last.content == "A2"

    def test_last_assistant_none(self):
        state = AgentState(system_prompt="test")
        state.add_message(Message(role="user", content="Q"))
        assert state.last_assistant() is None

    def test_last_assistant_text(self):
        state = AgentState(system_prompt="test")
        state.add_message(Message(role="assistant", content="Answer"))
        assert state.last_assistant_text() == "Answer"

    def test_last_assistant_text_empty(self):
        state = AgentState(system_prompt="test")
        assert state.last_assistant_text() == ""

    def test_to_chat_format_excludes_tool(self):
        state = AgentState(system_prompt="test")
        state.add_message(Message(role="user", content="Q"))
        state.add_message(Message(role="tool", content="result", tool_call_id="c1"))
        msgs = state.to_chat_format()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_to_chat_format_with_tools_preserves_tool_calls(self):
        state = AgentState(system_prompt="test")
        state.add_message(Message(role="user", content="Q"))
        state.add_message(
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}],
            )
        )
        state.add_message(Message(role="tool", content="42", tool_call_id="c1"))
        msgs = state.to_chat_format_with_tools()
        assert len(msgs) == 3
        # Assistant message has tool_calls
        assert "tool_calls" in msgs[1]
        # Tool message has tool_call_id
        assert msgs[2]["tool_call_id"] == "c1"

    def test_finished_default(self):
        state = AgentState(system_prompt="test")
        assert state.finished is False

    def test_step_count_default(self):
        state = AgentState(system_prompt="test")
        assert state.step_count == 0

    def test_scratchpad_default(self):
        state = AgentState(system_prompt="test")
        assert state.scratchpad == {}

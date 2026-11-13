"""Integration tests for the FastAPI chat session boundaries."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api
from core.session_store import AgentSessionStore
from core.state import AgentState, Message


class FakeAgentLoop:
    """Deterministic agent which records the state visible to each turn."""

    def __init__(self, config, state: AgentState) -> None:
        self.config = config
        self.state = state

    def run(self, user_input: str) -> str:
        self.state.add_message(Message(role="user", content=user_input))

        if user_input.startswith("use tool:"):
            self.state.add_message(
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": '{"expression": "1 + 1"}',
                            },
                        }
                    ],
                )
            )
            self.state.add_message(
                Message(role="tool", content="2", tool_call_id="call-1")
            )

        history = [
            message.content
            for message in self.state.messages
            if message.role == "user"
        ]
        answer = json.dumps(
            {"prompt": self.state.system_prompt, "history": history},
            ensure_ascii=False,
        )
        self.state.add_message(Message(role="assistant", content=answer))
        self.state.finished = True
        self.state.step_count = 1
        return answer


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_get_config", lambda: SimpleNamespace(system_prompt="default prompt"))
    monkeypatch.setattr(
        api,
        "_build_agent_loop",
        lambda config, state: FakeAgentLoop(config, state),
    )
    store = AgentSessionStore(lambda: AgentState(system_prompt="default prompt"))
    with TestClient(api.create_app(session_store=store)) as test_client:
        yield test_client


def _payload(response):
    return json.loads(response.json()["answer"])


def test_chat_sessions_are_isolated_and_keep_history(client):
    alice_first = client.post(
        "/chat", json={"session_id": "alice", "message": "alice one"}
    )
    bob_first = client.post(
        "/chat", json={"session_id": "bob", "message": "bob one"}
    )
    alice_second = client.post(
        "/chat", json={"session_id": "alice", "message": "alice two"}
    )

    assert alice_first.json()["session_id"] == "alice"
    assert _payload(alice_first)["history"] == ["alice one"]
    assert _payload(bob_first)["history"] == ["bob one"]
    assert _payload(alice_second)["history"] == ["alice one", "alice two"]


def test_clear_history_only_resets_the_current_session(client):
    client.post("/chat", json={"session_id": "alice", "message": "alice one"})
    client.post("/chat", json={"session_id": "bob", "message": "bob one"})

    cleared = client.post(
        "/chat",
        json={
            "session_id": "alice",
            "message": "alice reset",
            "clear_history": True,
        },
    )
    bob_second = client.post(
        "/chat", json={"session_id": "bob", "message": "bob two"}
    )

    assert _payload(cleared)["history"] == ["alice reset"]
    assert _payload(bob_second)["history"] == ["bob one", "bob two"]


def test_missing_session_id_returns_a_new_isolated_session(client):
    first = client.post("/chat", json={"message": "first"})
    second = client.post("/chat", json={"message": "second"})
    continued = client.post(
        "/chat",
        json={
            "session_id": first.json()["session_id"],
            "message": "continued",
        },
    )

    assert first.json()["session_id"] != second.json()["session_id"]
    assert _payload(first)["history"] == ["first"]
    assert _payload(second)["history"] == ["second"]
    assert _payload(continued)["history"] == ["first", "continued"]


def test_tool_calls_are_reported_only_for_the_current_turn(client):
    first = client.post(
        "/chat",
        json={"session_id": "tools", "message": "use tool: calculate"},
    )
    second = client.post(
        "/chat",
        json={"session_id": "tools", "message": "next question"},
    )

    assert first.json()["tool_calls_made"] == ["calculate"]
    assert second.json()["tool_calls_made"] == []


def test_streaming_uses_system_prompt_and_shares_session_state(client):
    streamed = client.post(
        "/chat/stream",
        json={
            "session_id": "stream",
            "message": "first",
            "system_prompt": "stream prompt",
        },
    )
    event = next(
        line.removeprefix("data: ")
        for line in streamed.text.splitlines()
        if line.startswith("data: {")
    )
    payload = json.loads(event)
    streamed_answer = json.loads(payload["answer"])

    followed_up = client.post(
        "/chat", json={"session_id": "stream", "message": "second"}
    )

    assert payload["session_id"] == "stream"
    assert streamed_answer["prompt"] == "stream prompt"
    assert _payload(followed_up) == {
        "prompt": "stream prompt",
        "history": ["first", "second"],
    }

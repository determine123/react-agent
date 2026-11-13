"""Unit tests for bounded per-session AgentState storage."""

from threading import Event, Thread

import pytest

from core.session_store import AgentSessionStore
from core.state import AgentState, Message


def _new_state() -> AgentState:
    return AgentState(system_prompt="test")


def test_named_sessions_are_isolated_and_persistent():
    store = AgentSessionStore(_new_state)

    with store.acquire("alice") as state:
        state.add_message(Message(role="user", content="alice message"))
    with store.acquire("bob") as state:
        state.add_message(Message(role="user", content="bob message"))

    with store.acquire("alice") as state:
        assert [message.content for message in state.messages] == ["alice message"]
    with store.acquire("bob") as state:
        assert [message.content for message in state.messages] == ["bob message"]


def test_ephemeral_sessions_are_not_persisted():
    store = AgentSessionStore(_new_state)

    with store.acquire(None) as first:
        first.add_message(Message(role="user", content="one"))
    with store.acquire(None) as second:
        assert second.messages == []

    assert len(store) == 0


def test_expired_session_is_recreated():
    now = [0.0]
    store = AgentSessionStore(_new_state, ttl_seconds=10, clock=lambda: now[0])

    with store.acquire("alice") as state:
        state.add_message(Message(role="user", content="before expiry"))

    now[0] = 11
    with store.acquire("alice") as state:
        assert state.messages == []


def test_least_recently_used_session_is_evicted():
    store = AgentSessionStore(_new_state, max_sessions=2)

    for session_id in ("first", "second"):
        with store.acquire(session_id) as state:
            state.add_message(Message(role="user", content=session_id))
    with store.acquire("third"):
        pass

    assert len(store) == 2
    with store.acquire("first") as state:
        assert state.messages == []


def test_same_session_turns_are_serialised():
    store = AgentSessionStore(_new_state)
    first_entered = Event()
    release_first = Event()
    second_entered = Event()

    def first_turn():
        with store.acquire("shared"):
            first_entered.set()
            release_first.wait(timeout=2)

    def second_turn():
        first_entered.wait(timeout=2)
        with store.acquire("shared"):
            second_entered.set()

    first = Thread(target=first_turn)
    second = Thread(target=second_turn)
    first.start()
    assert first_entered.wait(timeout=1)
    second.start()

    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    assert second_entered.wait(timeout=1)

    first.join(timeout=1)
    second.join(timeout=1)
    assert not first.is_alive()
    assert not second.is_alive()


@pytest.mark.parametrize(
    ("ttl_seconds", "max_sessions"),
    [(0, 1), (1, 0)],
)
def test_store_requires_positive_limits(ttl_seconds, max_sessions):
    with pytest.raises(ValueError):
        AgentSessionStore(
            _new_state,
            ttl_seconds=ttl_seconds,
            max_sessions=max_sessions,
        )

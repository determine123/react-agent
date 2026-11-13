"""Bounded, process-local conversation state for the HTTP API.

The CLI uses one ``AgentState`` for one human.  The API serves many clients at
once, so each client needs its own state and requests for the same client must
not mutate that state concurrently.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Any

from core.state import AgentState


@dataclass
class _SessionRecord:
    state: AgentState
    last_access: float
    active: int = 0
    lock: Any = field(default_factory=RLock)


class AgentSessionStore:
    """Store conversation state by session id with TTL and LRU eviction.

    State is intentionally process-local.  Deployments with multiple worker
    processes should enable sticky routing or replace this store with a shared
    backend so a session always resolves to the same state.
    """

    def __init__(
        self,
        state_factory: Callable[[], AgentState],
        *,
        ttl_seconds: float = 3600,
        max_sessions: int = 1000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")

        self._state_factory = state_factory
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._clock = clock
        self._guard = Lock()
        self._sessions: OrderedDict[str, _SessionRecord] = OrderedDict()

    @contextmanager
    def acquire(self, session_id: str | None) -> Iterator[AgentState]:
        """Yield state for one turn, serialising turns for the same session.

        ``None`` creates a fresh ephemeral state which is never inserted into
        the store.  API callers receive a generated id when they omit one, so
        normal multi-turn clients can explicitly continue the conversation.
        """
        if session_id is None:
            yield self._state_factory()
            return

        now = self._clock()
        with self._guard:
            self._evict_expired(now)
            record = self._sessions.get(session_id)
            if record is None:
                record = _SessionRecord(
                    state=self._state_factory(),
                    last_access=now,
                )
                self._sessions[session_id] = record
            else:
                self._sessions.move_to_end(session_id)
                record.last_access = now
            record.active += 1
            self._enforce_capacity()

        try:
            with record.lock:
                yield record.state
        finally:
            with self._guard:
                record.active -= 1
                record.last_access = self._clock()
                self._enforce_capacity()

    def __len__(self) -> int:
        with self._guard:
            return len(self._sessions)

    def _evict_expired(self, now: float) -> None:
        expired = [
            session_id
            for session_id, record in self._sessions.items()
            if record.active == 0 and now - record.last_access >= self._ttl_seconds
        ]
        for session_id in expired:
            del self._sessions[session_id]

    def _enforce_capacity(self) -> None:
        while len(self._sessions) > self._max_sessions:
            for session_id, record in self._sessions.items():
                if record.active == 0:
                    del self._sessions[session_id]
                    break
            else:
                # Every tracked session is currently in use.  They will be
                # trimmed as soon as the next active turn releases its lock.
                break

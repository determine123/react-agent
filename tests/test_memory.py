"""Unit tests for memory/memory.py"""

import pytest
from pathlib import Path
from config import MemoryConfig
from memory.memory import Memory


class TestMemory:
    @pytest.fixture
    def memory(self):
        cfg = MemoryConfig(
            working_memory_limit=5,
            long_term_path="./logs/_pytest_memory.json",
        )
        mem = Memory(cfg)
        yield mem
        # Cleanup
        Path(cfg.long_term_path).unlink(missing_ok=True)

    def test_remember_and_recent(self, memory):
        memory.remember({"role": "user", "content": "hello"})
        recent = memory.recent(1)
        assert len(recent) == 1
        assert recent[0]["content"] == "hello"

    def test_recent_limit(self, memory):
        for i in range(10):
            memory.remember({"role": "user", "content": f"msg_{i}"})
        recent = memory.recent(3)
        assert len(recent) == 3
        # Should have the most recent messages
        assert recent[-1]["content"] == "msg_9"

    def test_working_memory_limit(self, memory):
        for i in range(10):
            memory.remember({"role": "user", "content": f"msg_{i}"})
        # Working memory is capped at 5
        assert len(memory.working) == 5

    def test_load_all_persists(self, memory):
        memory.remember({"role": "user", "content": "persistent"})
        all_entries = memory.load_all()
        assert len(all_entries) >= 1
        assert any(e["content"] == "persistent" for e in all_entries)

    def test_empty_memory(self, memory):
        assert memory.recent() == []

    def test_recent_default_n(self, memory):
        for i in range(3):
            memory.remember({"role": "user", "content": f"msg_{i}"})
        assert len(memory.recent()) == 3

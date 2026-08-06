"""Two-tier memory: working (in-context) + long-term (persisted JSON)."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import MemoryConfig


class Memory:
    """Sliding-window working memory backed by a long-term JSON journal."""

    def __init__(self, cfg: MemoryConfig) -> None:
        self.cfg = cfg
        self.working: deque[dict] = deque(maxlen=cfg.working_memory_limit)
        self._disk_path = Path(cfg.long_term_path)
        self._ensure_disk_file()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def remember(self, entry: dict) -> None:
        """Add an entry to both working and long-term memory."""
        self.working.append(entry)
        self._append_to_disk(entry)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def recent(self, n: int | None = None) -> list[dict]:
        """Return the N most recent working-memory entries."""
        items = list(self.working)
        return items[-n:] if n else items

    def load_all(self) -> list[dict]:
        """Read every entry from the long-term journal."""
        if not self._disk_path.exists():
            return []
        with open(self._disk_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_disk_file(self) -> None:
        self._disk_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._disk_path.exists():
            self._disk_path.write_text("", encoding="utf-8")

    def _append_to_disk(self, entry: dict) -> None:
        with open(self._disk_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

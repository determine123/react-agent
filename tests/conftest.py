"""Pytest configuration — shared fixtures and settings."""

import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def setup_logs_dir():
    """Ensure logs/ directory exists for test artifacts."""
    Path("logs").mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture
def temp_dir():
    """Create and clean up a temporary directory for tests."""
    import tempfile
    import shutil

    tmp = tempfile.mkdtemp(prefix="agent_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)

"""Ensure the workflow smoke check detects an incomplete graph."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

import smoke_test


@pytest.mark.parametrize("missing", [None, "agent", "tools", "finish"])
def test_workflow_smoke_requires_all_nodes(monkeypatch, missing):
    nodes = {"agent": object(), "tools": object(), "finish": object()}
    if missing:
        nodes.pop(missing)
    workflow_module = ModuleType("graph.workflow")
    workflow_module.build_workflow = lambda: SimpleNamespace(
        get_graph=lambda: SimpleNamespace(nodes=nodes)
    )
    monkeypatch.setitem(sys.modules, "graph.workflow", workflow_module)

    if missing:
        with pytest.raises(AssertionError, match=missing):
            smoke_test.test_workflow()
    else:
        smoke_test.test_workflow()

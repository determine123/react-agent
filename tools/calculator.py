"""Safe arithmetic evaluator exposed as an agent tool."""

from __future__ import annotations

import ast
import math
import operator

# Allowed binary operators — no function calls, no attribute access.
_SAFE_OPS: dict[type, callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a safe AST node."""
    match node:
        case ast.Expression(body=value):
            return _safe_eval(value)
        case ast.Constant(value=v):
            if isinstance(v, (int, float)):
                return float(v)
            raise ValueError(f"Unsupported constant: {v}")
        case ast.BinOp(left=l, op=op, right=r):
            return _SAFE_OPS[type(op)](_safe_eval(l), _safe_eval(r))
        case ast.UnaryOp(op=op, operand=o):
            return _SAFE_OPS[type(op)](_safe_eval(o))
        case ast.Name(id=name):
            if name in _SAFE_NAMES:
                return _SAFE_NAMES[name]
            raise ValueError(f"Unknown name: {name}")
        case _:
            raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def calculate(expression: str) -> float | str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A string like ``"(3 + 5) * 2"`` or ``"sqrt(pi * 2)"``.

    Returns:
        The numeric result, or an error string.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return _safe_eval(tree.body)
    except Exception as exc:
        return f"Calculator error: {exc}"

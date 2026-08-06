"""Evaluation runner with RAGAS integration.

Supports two evaluation modes:
1. **Quick** — substring matching (fast, no API calls)
2. **RAGAS** — semantic metrics (faithfulness, relevancy, precision, recall)

Usage::

    python -m eval.evaluator          # quick mode
    python -m eval.evaluator --ragas  # RAGAS mode (requires API)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.agent_loop import AgentLoop
from core.state import AgentState
from config import Config, load_config


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    question_id: str
    question: str
    answer: str
    duration_s: float
    tool_called: str | None
    passed: bool
    detail: str = ""

    # RAGAS fields (populated only in RAGAS mode)
    ragas_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "accuracy": round(self.accuracy, 4),
            "results": [
                {
                    "id": r.question_id,
                    "question": r.question,
                    "answer": r.answer[:500],
                    "duration_s": r.duration_s,
                    "passed": r.passed,
                    "detail": r.detail,
                    "ragas_scores": r.ragas_scores,
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# Quick evaluation (substring match)
# ---------------------------------------------------------------------------

def run_eval(questions_path: str | None = None) -> EvalReport:
    """Run the quick eval suite with substring matching.

    Args:
        questions_path: Path to questions.json (default: eval/questions.json).

    Returns:
        EvalReport with accuracy and per-question results.
    """
    path = Path(questions_path or str(Path(__file__).parent / "questions.json"))
    questions: list[dict] = json.loads(path.read_text(encoding="utf-8"))

    cfg = load_config()
    cfg.verbose = False
    report = EvalReport(total=len(questions))

    for q in questions:
        state = AgentState(system_prompt=cfg.system_prompt)
        agent = AgentLoop(config=cfg, state=state)

        t0 = time.perf_counter()
        answer = agent.run(q["question"])
        elapsed = time.perf_counter() - t0

        expected = q.get("expected_answer_contains", "")
        passed = expected.lower() in answer.lower() if expected else bool(answer.strip())

        report.results.append(
            EvalResult(
                question_id=q["id"],
                question=q["question"],
                answer=answer,
                duration_s=round(elapsed, 3),
                tool_called=q.get("expected_tool"),
                passed=passed,
                detail=(
                    f"Expected '{expected}' in answer"
                    if expected
                    else "Non-empty answer"
                ),
            )
        )
        if passed:
            report.passed += 1

    _persist_report(report, "logs/eval_quick_report.json")
    return report


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas_eval(
    questions_path: str | None = None,
    llm_model: str = "deepseek-chat",
) -> EvalReport:
    """Run evaluation with RAGAS semantic metrics.

    This requires the ``ragas`` and ``datasets`` packages.  Each question
    should have a ``context`` field pointing to reference material.

    Args:
        questions_path: Path to questions.json.
        llm_model: Model name passed to RAGAS for judgement (uses the same
                   API key / base URL from the project .env).

    Returns:
        EvalReport with RAGAS scores attached to each result.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "ragas and datasets are required for RAGAS evaluation. "
            "Install with: pip install ragas datasets"
        )

    path = Path(questions_path or str(Path(__file__).parent / "questions.json"))
    questions: list[dict] = json.loads(path.read_text(encoding="utf-8"))

    cfg = load_config()
    cfg.verbose = False
    report = EvalReport(total=len(questions))

    # Build dataset rows
    dataset_rows: list[dict] = []
    for q in questions:
        state = AgentState(system_prompt=cfg.system_prompt)
        agent = AgentLoop(config=cfg, state=state)

        t0 = time.perf_counter()
        answer = agent.run(q["question"])
        elapsed = time.perf_counter() - t0

        expected = q.get("expected_answer_contains", "")
        passed = expected.lower() in answer.lower() if expected else bool(answer.strip())

        report.results.append(
            EvalResult(
                question_id=q["id"],
                question=q["question"],
                answer=answer,
                duration_s=round(elapsed, 3),
                tool_called=q.get("expected_tool"),
                passed=passed,
                detail=(
                    f"Expected '{expected}' in answer"
                    if expected
                    else "Non-empty answer"
                ),
            )
        )
        if passed:
            report.passed += 1

        # Collect RAGAS data for RAG-type questions
        if q.get("type") == "rag" and answer.strip():
            dataset_rows.append(
                {
                    "question": q["question"],
                    "answer": answer,
                    "contexts": [q.get("context", "")],
                    "ground_truth": q.get("expected_answer_contains", ""),
                }
            )

    # Run RAGAS metrics if we have RAG questions
    if dataset_rows:
        try:
            dataset = Dataset.from_list(dataset_rows)
            ragas_result = ragas_evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )
            # Attach scores to RAG-type results
            rag_idx = 0
            ragas_df = ragas_result.to_pandas()
            for r in report.results:
                if r.question_id.startswith("rag") and rag_idx < len(ragas_df):
                    row = ragas_df.iloc[rag_idx]
                    r.ragas_scores = {
                        k: round(float(v), 4)
                        for k, v in row.items()
                        if not isinstance(v, str) and k != "question"
                    }
                    rag_idx += 1
        except Exception as exc:
            report.results.append(
                EvalResult(
                    question_id="ragas_error",
                    question="RAGAS evaluation",
                    answer="",
                    duration_s=0,
                    tool_called=None,
                    passed=False,
                    detail=f"RAGAS error: {exc}",
                )
            )

    _persist_report(report, "logs/eval_ragas_report.json")
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _persist_report(report: EvalReport, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Agent evaluation runner")
    p.add_argument(
        "--ragas",
        action="store_true",
        help="Run RAGAS semantic evaluation (requires ragas package)",
    )
    args = p.parse_args()

    if args.ragas:
        print("Running RAGAS evaluation...\n")
        report = run_ragas_eval()
    else:
        print("Running quick evaluation...\n")
        report = run_eval()

    print(f"\n{'=' * 60}")
    print(f"Eval complete: {report.passed}/{report.total} passed ({report.accuracy:.1%})")
    print(f"{'=' * 60}")
    for r in report.results:
        status = "✓" if r.passed else "✗"
        print(f"  {status} [{r.question_id}] {r.question}")
        print(f"       → {r.answer[:120]}{'...' if len(r.answer) > 120 else ''}")
        print(f"       ⏱ {r.duration_s}s  |  {r.detail}")
        if r.ragas_scores:
            scores_str = "  ".join(f"{k}={v:.3f}" for k, v in r.ragas_scores.items())
            print(f"       📊 RAGAS: {scores_str}")

"""Evaluation module — test runner, RAGAS metrics, and HTML reporting."""

from eval.evaluator import EvalReport, EvalResult, run_eval, run_ragas_eval
from eval.report import ReportConfig, generate_html_report

__all__ = [
    "EvalReport",
    "EvalResult",
    "ReportConfig",
    "generate_html_report",
    "run_eval",
    "run_ragas_eval",
]

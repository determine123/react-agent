"""HTML evaluation report generator.

Produces a self-contained HTML page with:
- Overall accuracy & summary stats
- Per-question results table
- RAGAS metric charts (when available)
- Pass/fail distribution pie chart
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ReportConfig:
    title: str = "Agent Evaluation Report"
    dark_mode: bool = True


def generate_html_report(
    report: Any,  # EvalReport
    config: ReportConfig | None = None,
    output_path: str = "logs/eval_report.html",
) -> str:
    """Generate a self-contained HTML evaluation report.

    Args:
        report: An EvalReport instance from eval.evaluator.
        config: Optional styling configuration.
        output_path: Where to write the HTML file.

    Returns:
        The absolute path to the generated HTML file.
    """
    cfg = config or ReportConfig()
    results = report.results
    total = report.total
    passed = report.passed
    accuracy = report.accuracy

    # Build result rows
    rows_html = ""
    for r in results:
        status_icon = "✓" if r.passed else "✗"
        status_class = "pass" if r.passed else "fail"
        ragas_cells = ""
        if r.ragas_scores:
            for k, v in r.ragas_scores.items():
                color = _score_color(v)
                ragas_cells += f'<td style="color:{color}">{v:.3f}</td>'
        else:
            ragas_cells = '<td colspan="4" style="color:#888">—</td>'

        rows_html += f"""
        <tr class="{status_class}">
            <td>{status_icon}</td>
            <td><code>{r.question_id}</code></td>
            <td>{_escape(r.question[:80])}{'...' if len(r.question) > 80 else ''}</td>
            <td>{_escape(r.answer[:120])}{'...' if len(r.answer) > 120 else ''}</td>
            <td>{r.duration_s}s</td>
            <td>{r.tool_called or '—'}</td>
            <td>{r.detail}</td>
            {ragas_cells}
        </tr>"""

    # RAGAS header columns
    has_ragas = any(r.ragas_scores for r in results)
    ragas_headers = ""
    if has_ragas:
        ragas_headers = "<th>Faithfulness</th><th>Relevancy</th><th>Precision</th><th>Recall</th>"

    # Build type distribution data for chart
    type_counts: dict[str, int] = {}
    for r in results:
        qtype = getattr(r, "question_id", "unknown").split("_")[0] if hasattr(r, "question_id") else "unknown"
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    bg = "#1a1a2e" if cfg.dark_mode else "#ffffff"
    text_color = "#e0e0e0" if cfg.dark_mode else "#333333"
    card_bg = "#16213e" if cfg.dark_mode else "#f5f5f5"
    pass_color = "#4caf50"
    fail_color = "#f44336"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{cfg.title}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: {bg};
    color: {text_color};
    padding: 24px;
    line-height: 1.6;
}}
h1 {{ font-size: 28px; margin-bottom: 8px; }}
h2 {{ font-size: 18px; margin: 24px 0 12px; color: #888; }}
.subtitle {{ color: #888; font-size: 14px; margin-bottom: 24px; }}

/* Summary cards */
.summary {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
}}
.card {{
    background: {card_bg};
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}}
.card .value {{ font-size: 36px; font-weight: 700; }}
.card .label {{ font-size: 13px; color: #888; margin-top: 4px; }}
.card.pass .value {{ color: {pass_color}; }}
.card.fail .value {{ color: {fail_color}; }}

/* Table */
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 12px;
}}
th {{
    background: {card_bg};
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    position: sticky;
    top: 0;
}}
td {{ padding: 8px 12px; border-bottom: 1px solid rgba(128,128,128,0.15); }}
tr.pass {{ background: rgba(76,175,80,0.08); }}
tr.fail {{ background: rgba(244,67,54,0.08); }}
tr:hover {{ background: rgba(128,128,128,0.1); }}
code {{ background: rgba(128,128,128,0.2); padding: 1px 6px; border-radius: 4px; font-size: 12px; }}

/* Chart placeholder */
.chart-container {{
    display: flex;
    gap: 32px;
    flex-wrap: wrap;
    margin: 16px 0;
}}
.bar-chart {{ flex: 1; min-width: 300px; }}
.bar-row {{
    display: flex;
    align-items: center;
    margin: 8px 0;
    gap: 10px;
}}
.bar-label {{ width: 100px; font-size: 13px; text-align: right; }}
.bar-track {{
    flex: 1;
    height: 24px;
    background: rgba(128,128,128,0.15);
    border-radius: 6px;
    overflow: hidden;
}}
.bar-fill {{
    height: 100%;
    border-radius: 6px;
    transition: width 0.3s;
}}
.bar-value {{ width: 50px; font-size: 13px; }}

footer {{ margin-top: 48px; text-align: center; color: #555; font-size: 12px; }}
</style>
</head>
<body>

<h1>📊 {cfg.title}</h1>
<p class="subtitle">Generated from {total} test cases</p>

<div class="summary">
    <div class="card pass">
        <div class="value">{passed}</div>
        <div class="label">Passed</div>
    </div>
    <div class="card fail">
        <div class="value">{total - passed}</div>
        <div class="label">Failed</div>
    </div>
    <div class="card">
        <div class="value">{accuracy:.1%}</div>
        <div class="label">Accuracy</div>
    </div>
    <div class="card">
        <div class="value">{_avg_duration(results):.1f}s</div>
        <div class="label">Avg Duration</div>
    </div>
</div>

<h2>📈 Pass/Fail Distribution</h2>
<div class="bar-chart">
    <div class="bar-row">
        <span class="bar-label">Passed</span>
        <div class="bar-track">
            <div class="bar-fill" style="width:{accuracy * 100}%;background:{pass_color}"></div>
        </div>
        <span class="bar-value">{accuracy:.1%}</span>
    </div>
    <div class="bar-row">
        <span class="bar-label">Failed</span>
        <div class="bar-track">
            <div class="bar-fill" style="width:{(1 - accuracy) * 100}%;background:{fail_color}"></div>
        </div>
        <span class="bar-value">{(1 - accuracy):.1%}</span>
    </div>
</div>

<h2>📋 Detailed Results</h2>
<div style="overflow-x:auto;">
<table>
<thead>
<tr>
    <th></th>
    <th>ID</th>
    <th>Question</th>
    <th>Answer</th>
    <th>Time</th>
    <th>Tool</th>
    <th>Detail</th>
    {ragas_headers}
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>

<footer>
    Generated by Agent Project Evaluation Framework &mdash; {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}
</footer>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    return str(Path(output_path).absolute())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _avg_duration(results: list) -> float:
    if not results:
        return 0.0
    return sum(getattr(r, "duration_s", 0) for r in results) / len(results)


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "#4caf50"
    if score >= 0.5:
        return "#ff9800"
    return "#f44336"

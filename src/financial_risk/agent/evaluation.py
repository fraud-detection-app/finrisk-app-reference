"""Measuring an agent: did it pick the right tool?

The RAG block measured three failure modes. Adding tools adds a fourth, and it sits *before*
the other three: if the router picks the wrong capability, retrieval quality and groundedness
are irrelevant -- the assistant answered a different question well.

Routing accuracy is the metric, and it is cheap: a labelled set of questions with the tool each
one should reach. It is also the first thing to re-run after adding a tool, because a new tool
competes with the existing ones for the same phrasings.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from financial_risk.agent.router import run_agent
from financial_risk.agent.tools import AgentContext


def routing_report(
    questions: pd.DataFrame,
    ctx: AgentContext,
    use_llm_router: bool = False,
    extra_policy_patterns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Per-question: which tool was expected, which was chosen, and whether it ran.

    ``questions`` needs ``question`` and ``expected_tool``; use the literal ``refuse`` as the
    expected tool for questions the compliance gate must stop before routing happens."""
    rows = []
    for _, item in questions.iterrows():
        expected = str(item["expected_tool"]).strip()
        result = run_agent(str(item["question"]), ctx, use_llm_router=use_llm_router,
                           extra_policy_patterns=extra_policy_patterns)
        chosen = "refuse" if result.refusal_reason == "out_of_policy" else (result.tool or "none")
        rows.append({
            "question": str(item["question"])[:44],
            "expected": expected,
            "chosen": chosen,
            "refused": result.refused,
            "pass": chosen == expected,
        })
    return pd.DataFrame(rows)


def routing_accuracy(report: pd.DataFrame) -> float:
    return float(report["pass"].mean())


def confusion(report: pd.DataFrame) -> pd.DataFrame:
    """Where the router sends each kind of question -- misroutes cluster, they are rarely random."""
    return pd.crosstab(report["expected"], report["chosen"])


def write_routing_report(report: pd.DataFrame, output_path: Path, accuracy: float | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    if accuracy is not None:
        header = "| " + " | ".join(report.columns) + " |\n|" + "|".join("---" for _ in report.columns) + "|\n"
        body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in report.itertuples(index=False))
        output_path.with_suffix(".md").write_text(
            f"# Agent routing evaluation\n\nRouting accuracy: **{accuracy:.0%}** "
            f"({int(report['pass'].sum())}/{len(report)})\n\n{header}{body}\n", encoding="utf-8")
    return output_path

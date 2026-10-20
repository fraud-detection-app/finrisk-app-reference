"""The tools a risk assistant can call, and what each one is allowed to claim.

A knowledge assistant retrieves prose. A *risk* assistant has to answer questions whose answers
do not live in prose at all -- a stored transaction, a fresh prediction, an aggregate over the
gold table. Those need different mechanisms, and the whole design rests on keeping them apart:

    "What is PR-AUC?"                     -> documents      -> vector search   (search_knowledge)
    "What happened with row 14532?"       -> the database   -> lookup          (lookup_transaction)
    "Score this new transfer"             -> the model      -> tool call       (score_new_transaction)
    "Show me cases like this one"         -> the feature space -> kNN          (find_similar_cases)
    "What is the fraud rate by type?"     -> the database   -> aggregation     (aggregate_stats)

Only the first is RAG. Treating the other four as RAG is the expensive mistake of the genre:
embedding transaction rows into a vector store makes exact filters approximate, aggregation
impossible, and the whole thing unauditable. Structured data belongs in SQL.

Every tool returns a dict with a ``provenance`` key naming where the answer came from, so the
sentence the LLM eventually writes can always be traced back to a mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from financial_risk.models.evaluation import load_latest_artifact
from financial_risk.models.inference import heuristic_reasons
from financial_risk.models.training import FEATURE_COLUMNS, NUMERIC_FEATURES
from financial_risk.rag.index import answer_question
from financial_risk.serving.contract import InvalidRequest, frame_to_instances, reason_sentences


@dataclass
class AgentContext:
    """Everything the tools need, loaded once instead of per call."""

    gold_path: Path
    model_dir: Path
    index_dir: Path
    # The service hands over the index it already loaded (service/artifacts.py); scripts and notebooks
    # leave this None and search_knowledge reads the folder in index_dir.
    index: Any = field(default=None, repr=False)
    _features: pd.DataFrame | None = field(default=None, repr=False)
    _artifact: dict | None = field(default=None, repr=False)
    # How search_knowledge answers. Week 9 leaves these alone (no model, stitched chunks); the
    # week-10 service injects llm.client.complete when a provider is configured, and asks for
    # the extractive offline answer otherwise.
    llm: Any = field(default=None, repr=False)
    offline_answer: str = "stitch"
    # Who scores (13b): any object with ``score(instances, explain, top_k)`` -- the service hands over the scorer it serves
    # with (in process, or a SageMaker endpoint). Scripts and notebooks leave it None and get a local one built from model_dir.
    _scorer: Any = field(default=None, repr=False)

    @property
    def features(self) -> pd.DataFrame:
        if self._features is None:
            self._features = pd.read_parquet(self.gold_path)
        return self._features

    @property
    def artifact(self) -> dict:
        if self._artifact is None:
            self._artifact = load_latest_artifact(self.model_dir)
        return self._artifact

    @property
    def scorer(self) -> Any:
        if self._scorer is None:
            # Imported here, not at the top: agent/ sits below service/ and only a script that built no scorer needs this one.
            from financial_risk.serving.runner import ModelRunner
            from financial_risk.service.scoring import LocalScorer

            features = self.features
            background = features.sample(min(200, len(features)), random_state=42) if len(features) >= 20 else None
            self._scorer = LocalScorer(ModelRunner(self.artifact, "local", background), "local")
        return self._scorer

    @scorer.setter
    def scorer(self, value: Any) -> None:
        self._scorer = value

    @property
    def threshold(self) -> float | None:
        """The threshold of the model that scores (an endpoint reports it with its first answer)."""
        threshold = self.scorer.threshold
        return float(threshold) if threshold is not None else None


def _score_with_reasons(ctx: AgentContext, row: pd.DataFrame) -> tuple[float, float, bool, list[str], str]:
    """One row through the scorer, asking for its reasons. -> (score, threshold, flagged, reason sentences, method)."""
    response = ctx.scorer.score(frame_to_instances(row), explain=True, top_k=4)
    prediction = response["predictions"][0]
    reasons = reason_sentences(prediction.get("reasons") or [])
    if reasons:
        method = "shap"
    else:                                   # this model type cannot explain itself: rules of thumb, and say so
        reasons, method = heuristic_reasons(row.iloc[0], float(prediction["score"])), "heuristic"
    return float(prediction["score"]), float(response["threshold"]), bool(prediction["flagged"]), reasons, method


@dataclass(frozen=True)
class Tool:
    """A capability the router can choose, described the way an LLM needs to see it."""

    name: str
    description: str
    parameters: dict[str, str]
    run: Callable[..., dict[str, object]]

    def schema(self) -> dict[str, object]:
        """The tool as a JSON schema fragment -- what goes into an LLM router's prompt."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


# --------------------------------------------------------------------------------- tools


def lookup_transaction(ctx: AgentContext, row_id: int) -> dict[str, object]:
    """A settled transaction from the gold table: what the label says AND what the model says.

    These are two different claims and the tool keeps them apart on purpose. ``confirmed_fraud``
    is ground truth from an investigation that already happened. ``model_score`` is a
    prediction. Conflating them teaches a habit that breaks in production, where labels for
    recent transactions do not exist yet -- they arrive weeks later, after someone investigates.
    """
    rows = ctx.features[ctx.features["row_id"] == int(row_id)]
    if rows.empty:
        return {"found": False, "row_id": int(row_id),
                "note": f"No transaction with row_id={row_id} in the gold table.",
                "provenance": "sql:gold.transaction_features"}

    row = rows.head(1)
    record = row.iloc[0].to_dict()
    score, threshold, flagged, reasons, method = _score_with_reasons(ctx, row)
    return {
        "found": True,
        "row_id": int(record["row_id"]),
        "transaction_type": record["transaction_type"],
        "amount": float(record["amount"]),
        "confirmed_fraud": bool(record["is_fraud"]),
        "label_note": "Ground truth from a completed investigation. Not available for recent transactions.",
        "model_score": round(score, 4),
        "model_threshold": round(threshold, 4),
        "model_says": "high" if flagged else "low",
        "reasons": reasons,
        "reason_method": method,
        "provenance": "sql:gold.transaction_features + model",
    }


def score_new_transaction(ctx: AgentContext, transaction: dict[str, object]) -> dict[str, object]:
    """Score a transaction that is not in the table -- the production path.

    No label exists and none is invented. The answer is a score, a threshold, and the reasons
    the model gives for it.

    Fields the caller omits are filled from a template row so the model always sees a complete
    feature vector. That is a convenience, not a free lunch: a half-specified transaction is a
    blend of the caller's values and someone else's, and the score reflects both. The result
    reports ``unspecified_fields`` so nobody reads such a score as a clean prediction -- in a
    real service these values arrive from the feature pipeline, never from a template."""
    template = ctx.features.iloc[[0]].copy()
    unknown = [k for k in transaction if k not in template.columns]
    if unknown:
        return {"scored": False, "note": f"Unknown fields: {unknown}", "provenance": "model"}
    try:
        for key, value in transaction.items():
            template.loc[template.index[0], key] = value
    except (TypeError, ValueError) as exc:    # e.g. amount="a lot" into a number column
        return {"scored": False, "note": f"Those values cannot be scored: {exc}", "provenance": "model"}

    try:
        score, threshold, flagged, reasons, method = _score_with_reasons(ctx, template)
    except InvalidRequest as exc:          # e.g. amount="a lot": the model contract refuses it; say so instead of crashing the agent
        return {"scored": False, "note": f"Those values cannot be scored: {exc}", "provenance": "model"}
    ignored = {"row_id", "is_fraud"}
    unspecified = [c for c in FEATURE_COLUMNS if c not in transaction and c not in ignored]
    return {
        "scored": True,
        "inputs": {k: transaction[k] for k in sorted(transaction)},
        "unspecified_fields": unspecified,
        "model_score": round(score, 4),
        "model_threshold": round(threshold, 4),
        "model_says": "high" if flagged else "low",
        "confirmed_fraud": None,
        "label_note": "No label exists for an unsettled transaction -- only a prediction.",
        "reasons": reasons,
        "reason_method": method,
        "provenance": "model",
    }


def search_knowledge(ctx: AgentContext, question: str) -> dict[str, object]:
    """The RAG tool: answer from the indexed documents, with citations, or refuse."""
    result = answer_question(question, ctx.index if ctx.index is not None else ctx.index_dir,
                             llm=ctx.llm, offline=ctx.offline_answer)
    return {
        "refused": bool(result["refused"]),
        "refusal_reason": result.get("refusal_reason"),
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "provenance": "rag:vector_index",
    }


def find_similar_cases(ctx: AgentContext, row_id: int, k: int = 5) -> dict[str, object]:
    """Nearest confirmed frauds in feature space -- the one retrieval that belongs on the data.

    This IS vector search, but over the model's own numeric feature space rather than text
    embeddings, and the neighbours are filtered to cases an investigation already confirmed.
    It answers the question an analyst actually asks: "what does this look like?"
    """
    rows = ctx.features[ctx.features["row_id"] == int(row_id)]
    if rows.empty:
        return {"found": False, "row_id": int(row_id), "provenance": "knn:feature_space"}

    frauds = ctx.features[(ctx.features["is_fraud"] == 1) & (ctx.features["row_id"] != int(row_id))]
    if frauds.empty:
        return {"found": True, "row_id": int(row_id), "neighbours": [], "provenance": "knn:feature_space"}

    scale = ctx.features[NUMERIC_FEATURES].std().replace(0, 1.0)
    target = (rows[NUMERIC_FEATURES].iloc[0] / scale).to_numpy(dtype=float)
    matrix = (frauds[NUMERIC_FEATURES] / scale).to_numpy(dtype=float)
    distance = np.linalg.norm(matrix - target, axis=1)
    order = np.argsort(distance)[:k]

    neighbours = [
        {
            "row_id": int(frauds.iloc[i]["row_id"]),
            "transaction_type": frauds.iloc[i]["transaction_type"],
            "amount": round(float(frauds.iloc[i]["amount"]), 2),
            "distance": round(float(distance[i]), 3),
        }
        for i in order
    ]
    return {"found": True, "row_id": int(row_id), "neighbours": neighbours,
            "note": "Neighbours are CONFIRMED frauds, ranked by distance in scaled feature space.",
            "provenance": "knn:feature_space"}


def aggregate_stats(ctx: AgentContext, group_by: str = "transaction_type") -> dict[str, object]:
    """Counts and fraud rates over the gold table -- the question vector search cannot answer.

    Aggregation is the most common analyst question and the one most obviously outside RAG's
    reach: no amount of chunk retrieval computes a group-by."""
    if group_by not in ctx.features.columns:
        return {"ok": False, "note": f"Unknown column: {group_by}", "provenance": "sql:gold.transaction_features"}

    grouped = ctx.features.groupby(group_by)["is_fraud"].agg(["count", "sum"])
    grouped["fraud_rate"] = (grouped["sum"] / grouped["count"]).round(5)
    grouped = grouped.rename(columns={"count": "transactions", "sum": "confirmed_frauds"})
    return {
        "ok": True,
        "group_by": group_by,
        "rows": grouped.reset_index().to_dict(orient="records"),
        "total_transactions": int(len(ctx.features)),
        "total_confirmed_frauds": int(ctx.features["is_fraud"].sum()),
        "provenance": "sql:gold.transaction_features",
    }


# ------------------------------------------------------------------------------- registry

TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in [
        Tool(
            name="lookup_transaction",
            description="Look up a transaction that already exists in the database by row_id. "
                        "Returns the confirmed investigation outcome AND the model's score.",
            parameters={"row_id": "integer - the row_id of a stored transaction"},
            run=lookup_transaction,
        ),
        Tool(
            name="score_new_transaction",
            description="Score a NEW transaction that is not in the database. Returns a model "
                        "score, the threshold, and the reasons. No label exists for it.",
            parameters={"transaction": "object - feature values, e.g. amount, transaction_type"},
            run=score_new_transaction,
        ),
        Tool(
            name="search_knowledge",
            description="Answer a question about the project, model, metrics, data or policy "
                        "from the indexed documentation, with citations.",
            parameters={"question": "string - the user's question"},
            run=search_knowledge,
        ),
        Tool(
            name="find_similar_cases",
            description="Find confirmed fraud cases that resemble a given transaction, by "
                        "distance in the model's feature space.",
            parameters={"row_id": "integer - the transaction to compare", "k": "integer - how many neighbours"},
            run=find_similar_cases,
        ),
        Tool(
            name="aggregate_stats",
            description="Count transactions and compute fraud rates grouped by a column, "
                        "e.g. transaction_type.",
            parameters={"group_by": "string - the column to group by"},
            run=aggregate_stats,
        ),
    ]
}


def tool_catalogue() -> str:
    """The registry rendered for an LLM router's prompt."""
    return "\n".join(
        f"- {tool.name}({', '.join(tool.parameters)}): {tool.description}" for tool in TOOLS.values()
    )

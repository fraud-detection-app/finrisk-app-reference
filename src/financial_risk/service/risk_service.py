"""The service layer: one place that knows how the parts fit together.

The parts of the risk assistant -- features, model, SHAP reasons, LLM summary, RAG, agent -- are separate
components. This module composes them, and it exists so that the *API* never has to:

    features (w6)  ->  model (w6/7)  ->  SHAP reasons (w7)  ->  LLM summary (w8)
                                                             ->  RAG / agent (w9)

Three rules make it a service rather than a script:

1. **Load once, answer many.** The ``ArtifactStore`` (service/artifacts.py) opens the feature table and the
   vector index a single time and keeps them in memory, and connects to the model: in 13b the model is a SageMaker
   endpoint (service/scoring.py), not a forest in this process. A handler that reloads things per request
   turns a 3 ms answer into a 2 s one, and it is the single most common mistake in a first API.
2. **Swap, don't restart.** Which model and index are served is decided by two SSM pointers that other pipelines
   move. The store loads a new version beside the running one and replaces it in one assignment; a request takes
   one snapshot (``model = ctx.model``) and finishes with it.
3. **Every dependency is optional at import time.** Nothing may be loaded yet. ``readiness()`` reports what is
   actually loaded, so the API can answer "alive but not ready" instead of crashing on the first request -- the
   difference between a liveness and a readiness probe, and the reason orchestrators can tell "restart me" from
   "don't send me traffic yet".

Transport lives in ``api/main.py``; this module knows nothing about HTTP, which is what makes it
testable without a server and reusable from the CLI, a notebook, or a Streamlit page.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from financial_risk.config import Settings, get_settings
from financial_risk.llm.client import complete, llm_status, record_calls
from financial_risk.models.inference import heuristic_reasons
from financial_risk.rag.index import LoadedIndex
from financial_risk.serving.contract import InvalidRequest, reason_sentences
from financial_risk.service.artifacts import ArtifactStore, IndexBundle, ModelBundle
from financial_risk.service.scoring import ModelUnavailable, score_frame  # noqa: F401 - ModelUnavailable is re-exported for the API layer

logger = logging.getLogger("financial_risk.service")


def _model_call():
    """The model the capabilities may use for THIS request, or None to stay offline.

    Resolved per call rather than at import, so a test (or a teacher) can flip LLM_PROVIDER
    without re-importing anything. With a provider configured, the knowledge base generates
    its answers and the agent routes with the model; either falls back on any failure."""
    return complete if llm_status()["ready"] else None


def _plain(value: Any) -> Any:
    """Tool output can carry numpy scalars; JSON cannot. Round-trip through json to be sure."""
    return json.loads(json.dumps(value, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


def _groundedness(answer: str, chunks: list[dict[str, Any]]) -> float | None:
    """Week 9's per-sentence support, averaged: how much of the answer the retrieved text backs.

    Citations like [faq.md § FAQ] are removed first -- they are labels, not claims."""
    from financial_risk.rag.evaluation import groundedness_report

    text = re.sub(r"\[[^\]]*\]", "", answer)
    if not text.strip() or not chunks:
        return None
    report = groundedness_report(text, chunks)
    return None if report.empty else round(float(report["support"].mean()), 2)


class ServiceNotReady(RuntimeError):
    """A required artifact is missing. Maps to HTTP 503 -- retry after the pipeline runs."""


class BadScoreRequest(ValueError):
    """POST /api/score: the rows break the endpoint contract. Maps to HTTP 422; ``problems`` lists each one."""

    def __init__(self, message: str, problems: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.problems = problems


class TransactionNotFound(LookupError):
    """No such row_id in the gold table. Maps to HTTP 404 -- retrying will not help."""


# These two exception types are the whole error taxonomy of the service, and the distinction is
# the teaching point: 503 says "the server is not ready, come back"; 404 says "your request is
# fine but that thing does not exist". Collapsing both into a generic 400, as the week-9 version
# of this API did, throws away the only information the caller could act on.


@dataclass
class ServiceContext:
    """Everything the service needs: the settings and the ArtifactStore that holds the model and the index.

    The store swaps a new model or index in while the service runs (service/artifacts.py), so a request
    takes ONE snapshot -- ``model = ctx.model`` -- and uses only that, instead of reading ``ctx.features``
    and ``ctx.artifact`` separately and risking half of each."""

    settings: Settings
    store: ArtifactStore = field(repr=False, default=None)  # type: ignore[assignment]
    _agent_ctx: Any | None = field(default=None, repr=False)
    _agent_key: tuple[int, int] | None = field(default=None, repr=False)
    loaded_at: float | None = field(default=None, repr=False)
    llm_check: dict[str, Any] | None = field(default=None, repr=False)  # set by the API at startup

    # ------------------------------------------------------------------ construction

    @classmethod
    def build(cls, settings: Settings | None = None, eager: bool = True, ssm_client: Any | None = None,
              s3_client: Any | None = None, embedder_factory: Any | None = None, scorer_factory: Any | None = None) -> "ServiceContext":
        settings = settings or get_settings()
        ctx = cls(settings=settings, store=ArtifactStore(settings, ssm_client=ssm_client, s3_client=s3_client,
                                                         embedder_factory=embedder_factory, scorer_factory=scorer_factory))
        if eager:
            ctx.warm()
        return ctx

    def warm(self) -> "ServiceContext":
        """Load the first versions now (pointer mode: wait a few seconds, the rest finishes in the background)."""
        self.store.start()
        if self.store.model.bundle is not None:
            self.loaded_at = time.time()
        return self

    def maybe_refresh(self) -> None:
        """Called per request: look at the pointers again if the last look is older than the TTL. On Lambda (inline mode)
        this call waits for the look and any load, up to INLINE_LOAD_TIMEOUT_SECONDS; elsewhere it returns at once."""
        self.store.maybe_refresh()

    # ------------------------------------------------------------------ what is being served

    @property
    def model(self) -> ModelBundle:
        self.store.ensure_started()
        bundle = self.store.model.bundle
        if bundle is None:
            raise ServiceNotReady(self._not_ready("model"))
        return bundle

    @property
    def index(self) -> LoadedIndex:
        self.store.ensure_started()
        bundle: IndexBundle | None = self.store.index.bundle
        if bundle is None:
            raise ServiceNotReady(self._not_ready("index"))
        return bundle.index

    def _not_ready(self, kind: str) -> str:
        slot = getattr(self.store, kind)
        if slot.error:
            return f"No {kind} loaded: {slot.error}"
        if self.store.pointers_enabled:
            return f"No {kind} loaded yet: waiting for the first load from {self.store.prefix}/{kind}."
        return f"No {kind} found. Run: python scripts/run_pipeline.py --demo-rows 5000"

    # Conveniences that read the CURRENT model. Anything that needs two of them at once should take
    # ``ctx.model`` once instead.

    @property
    def features(self) -> pd.DataFrame:
        """The service's own transaction table (what /api/predict/<row_id> looks rows up in)."""
        return self._features_of(self.model)

    @property
    def scorer(self):
        return self.model.scorer

    @staticmethod
    def _features_of(model: ModelBundle) -> pd.DataFrame:
        if model.features is None:
            raise ServiceNotReady("This service has no transaction table of its own to look rows up in (the baked demo feature table is "
                                  "missing), so /api/predict/<row_id> and the transaction tools cannot answer. POST /api/score still works.")
        return model.features

    @property
    def agent_context(self):
        """The week-9 agent, sharing the model, features and index being served right now. Rebuilt when
        either one is swapped, so it never answers from a version that is no longer served."""
        model, index = self.model, self.index
        key = (id(model), id(index))
        if self._agent_ctx is None or self._agent_key != key:
            from financial_risk.agent.tools import AgentContext

            # The agent scores through the SAME scorer the API uses (in process or the endpoint), and reads the same
            # transaction table; hand over what we have already paid to load, instead of loading it twice.
            agent = AgentContext(gold_path=model.gold_path or self.settings.gold_path / "transaction_features.parquet",
                                 model_dir=model.model_dir or self.settings.model_dir,
                                 index_dir=self.settings.vector_index_dir, index=index)
            agent._features = model.features
            agent.scorer = model.scorer
            agent.offline_answer = "extract"
            self._agent_ctx, self._agent_key = agent, key
        return self._agent_ctx

    # ------------------------------------------------------------------ readiness

    def readiness(self) -> dict[str, Any]:
        """What is loaded right now, and what went wrong. Never raises -- this is the diagnostic endpoint's source.

        ``ready`` is True only when a model AND an index are loaded. A failed attempt to load a NEWER
        version does not make it False (the old one still serves): it shows up as ``load_error``."""
        store = self.store
        m_slot, i_slot = store.model, store.index
        m: ModelBundle | None = m_slot.bundle
        i: IndexBundle | None = i_slot.bundle
        checks = {"gold_features": m is not None and m.features is not None, "model_artifact": m is not None, "vector_index": i is not None}
        missing = [name for name, ok in checks.items() if not ok]
        errors = [e for e in (m_slot.error, i_slot.error) if e]

        if errors and missing:
            hint = "A pointer names something that could not be loaded, and there is nothing older to serve. See load_error."
        elif errors:
            hint = "A newer version could not be loaded; the previous one keeps serving. See load_error."
        elif missing and store.pointers_enabled:
            hint = ("Waiting for the first model/index to load, or the pointers are null and nothing is baked in. "
                    f"Pointers: {store.prefix}/model and {store.prefix}/index.")
        elif missing:
            hint = "Run: python scripts/run_pipeline.py --demo-rows 5000"
        else:
            hint = None

        # model.source is "endpoint" (a SageMaker endpoint the pointer names) or "baked" (in process); endpoint, status
        # (DescribeEndpoint, cached 60 s) and error (the last scoring/describe problem) are null for the baked one.
        model_info: dict[str, Any] = {"version": None, "source": None, "endpoint": None, "status": None, "error": None,
                                      "loaded_at": None, "load_error": m_slot.error,
                                      "loading": m_slot.loading,
                                      "loading_note": m_slot.loading_note}
        if m is not None:
            model_info.update(version=m.version, source=m.source, loaded_at=m.loaded_at, **m.summary())
        index_info: dict[str, Any] = {"version": None, "embeddings": None, "docs": None, "chunks": None, "source": None,
                                      "loaded_at": None, "load_error": i_slot.error, "loading": i_slot.loading,
                                      "loading_note": i_slot.loading_note}
        if i is not None:
            index_info.update(version=i.version, source=i.source, loaded_at=i.loaded_at, **i.summary())
        return {
            "ready": not missing,
            "checks": checks,
            "missing": missing,
            "rows": int(len(m.features)) if m is not None and m.features is not None else None,
            "warmed": self.loaded_at is not None or m is not None,
            "load_error": errors[0] if errors else None,
            "hint": hint,
            "model": model_info,
            "index": index_info,
            "pointers_checked_at": store.pointers_checked_at,
            "pointer_prefix": store.prefix or None,
        }


# ---------------------------------------------------------------------------- operations


def assess_transaction(ctx: ServiceContext, row_id: int, use_llm: bool = False) -> dict[str, Any]:
    """Score one stored transaction and explain it: the full ML -> SHAP -> LLM chain.

    The row comes from the service's own transaction table; the SCORE comes from the model's scorer -- this process for
    the baked demo model, a SageMaker endpoint otherwise -- and so do the reasons (TreeSHAP computed by the model server).
    Raises ``TransactionNotFound`` (404), ``ServiceNotReady`` or ``ModelUnavailable`` (503); the API layer turns those
    into status codes and never leaks a traceback."""
    model = ctx.model                     # ONE snapshot: the feature table and the scorer belong together
    features = ctx._features_of(model)
    rows = features[features["row_id"] == int(row_id)]
    if rows.empty:
        raise TransactionNotFound(f"No transaction with row_id={row_id}")

    row = rows.head(1)
    started = time.perf_counter()
    calls: list[dict[str, Any]] = []
    response = score_frame(model.scorer, row, explain=True, top_k=4)
    prediction = response["predictions"][0]
    score = float(prediction["score"])
    threshold = float(response["threshold"])
    reasons = reason_sentences(prediction.get("reasons") or [])
    method = "shap"
    if not reasons:                       # this model type cannot explain itself: say so, and fall back to rules of thumb
        reasons, method = heuristic_reasons(row.iloc[0], score), "heuristic"
    record = row.iloc[0].to_dict()
    risk_level = "high" if prediction["flagged"] else "low"

    summary = None
    summary_source = "not_requested"
    summary_model = None
    if use_llm:
        from financial_risk.llm.client import generate_insight

        with record_calls() as calls:
            insight = generate_insight({
                "row_id": int(record["row_id"]),
                "transaction_type": record["transaction_type"],
                "amount": float(record["amount"]),
                "risk_score": score,
                "risk_level": risk_level,
                "threshold": threshold,
                "top_reasons": reasons,
            })
        summary = insight["summary"]
        summary_source = insight["source"]
        summary_model = insight.get("model")

    elapsed = round((time.perf_counter() - started) * 1000, 1)
    logger.info("predict: row %s · %s %.2f · score %.4f vs threshold %.4f -> %s · scored by %s%s · reasons via %s%s · %.0f ms",
                record["row_id"], record["transaction_type"], float(record["amount"]), score, threshold,
                risk_level, model.source, f" {model.endpoint}" if model.endpoint else "", method,
                f" · summary by {summary_source}" if use_llm else "", elapsed)
    return {
        "row_id": int(record["row_id"]),
        "transaction_type": str(record["transaction_type"]),
        "amount": round(float(record["amount"]), 2),
        "risk_score": round(score, 4),
        "threshold": round(threshold, 4),
        "risk_level": risk_level,
        "confirmed_fraud": bool(record["is_fraud"]) if "is_fraud" in record else None,
        "reasons": reasons,
        "reason_method": method,
        "scored_by": model.source,
        "model_version": str(response["model_version"]),
        "summary": summary,
        "summary_source": summary_source,
        "summary_model": summary_model,
        "llm_calls": calls,
        "elapsed_ms": elapsed,
    }


def score_transactions(ctx: ServiceContext, transactions: list[dict[str, Any]], explain: bool = False, top_k: int = 3) -> dict[str, Any]:
    """The production path (POST /api/score): rows that arrive with their features, scored in ONE call to the model.

    The caller (a payments system, the replay) already holds the features, so unlike /api/predict nothing is looked up.
    1 to 500 rows; each must hold exactly ``row_id`` + the model's features (serving/contract.py).
    Raises ``BadScoreRequest`` (422), ``ServiceNotReady`` / ``ModelUnavailable`` (503)."""
    model = ctx.model
    started = time.perf_counter()
    try:
        response = model.scorer.score(transactions, explain=explain, top_k=top_k)
    except InvalidRequest as exc:
        raise BadScoreRequest(str(exc), exc.problems) from exc
    flagged = sum(1 for p in response["predictions"] if p["flagged"])
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    logger.info("score: %d row(s) · %d flagged · explain=%s · scored by %s%s v%s · %.0f ms", len(transactions), flagged, explain,
                model.source, f" {model.endpoint}" if model.endpoint else "", response["model_version"], elapsed)
    return {
        "model_version": str(response["model_version"]),
        "threshold": response["threshold"],
        "scored_by": model.source,
        "results": response["predictions"],
        "elapsed_ms": elapsed,
    }


def answer_question(ctx: ServiceContext, question: str) -> dict[str, Any]:
    """Ask the week-9 knowledge assistant: a cited answer, or a refusal with its reason."""
    from financial_risk.rag.index import answer_question as rag_answer

    index = ctx.index                     # the loaded index being served (raises ServiceNotReady when there is none)

    started = time.perf_counter()
    logger.info('ask: "%s"', question[:160])
    with record_calls() as calls:
        result = rag_answer(question, index, llm=_model_call(), offline="extract")
    method = result.get("answer_method") or ("refused" if result["refused"] else "unknown")
    grounded = None if result["refused"] else _groundedness(str(result["answer"]), result.get("chunks", []))
    logger.info("ask: %s%s · %d citation(s) · %.0f ms",
                "refused (" + str(result.get("refusal_reason")) + ")" if result["refused"] else f"answer {method}",
                f" · grounded {grounded:.0%}" if grounded is not None else "",
                len(result.get("citations", [])), (time.perf_counter() - started) * 1000)
    return {
        "question": question,
        "answer": result["answer"],
        "answer_method": method,
        "groundedness": grounded,
        "refused": bool(result["refused"]),
        "refusal_reason": result.get("refusal_reason"),
        "citations": result.get("citations", []),
        "disclaimer": result["disclaimer"],
        "llm_calls": calls,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def ask_agent(ctx: ServiceContext, question: str) -> dict[str, Any]:
    """Ask the week-9 agent: it picks a tool, runs it, and returns the trace that proves it."""
    from financial_risk.agent.router import SERVICE_ROUTING_RULES, run_agent

    started = time.perf_counter()
    logger.info('agent: "%s"', question[:160])
    agent_ctx = ctx.agent_context
    model = _model_call()
    agent_ctx.llm = model  # the knowledge-base tool answers with the same model the router used
    with record_calls() as calls:
        result = run_agent(question, agent_ctx, extra_policy_patterns=POLICY_PATTERNS,
                           llm=model, rules=SERVICE_ROUTING_RULES)
    logger.info("agent: %s · %d stage(s) · %.0f ms",
                f"refused ({result.refusal_reason})" if result.refused else f"answered by {result.tool}",
                len(result.trace), (time.perf_counter() - started) * 1000)
    return {
        "question": question,
        "answer": result.answer,
        "tool": result.tool,
        "arguments": result.arguments,
        "refused": result.refused,
        "refusal_reason": result.refusal_reason,
        "route_method": result.route_method,
        "trace": result.trace,
        "data": _plain(result.data),
        "disclaimer": result.disclaimer,
        "llm_calls": calls,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


# Week 9 homework 1 made permanent: the third-person and account-action patterns the eval set
# found. Kept here as the service's policy overlay until they move into OUT_OF_POLICY_PATTERNS.
POLICY_PATTERNS: tuple[str, ...] = (
    r"\bshould (this|that|the) \w+\b",
    r"\b(waive|reverse|refund|unblock|cancel|close)\b.{0,30}\b(account|flag|transfer|charge|payment)\b",
)

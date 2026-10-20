"""The risk service as a FastAPI application.

This file is deliberately readable top-to-bottom as a lesson in what an API is made of:

    lifespan   -> load the model ONCE at startup, not per request
    schemas    -> Pydantic models that validate input and document output
    middleware -> a request id on every log line and every response
    handlers   -> our exceptions become status codes; tracebacks never reach a client
    routes     -> liveness, readiness, scoring (one stored row, or a batch) and two assistants
    static     -> a small HTML page so this is a web app, not just JSON

The rule that shapes all of it: **transport code does no thinking.** Every route is a thin
adapter over ``financial_risk.service.risk_service``, which knows nothing about HTTP. That is
why the service can be tested in a notebook with no server running, and why swapping FastAPI
for anything else would not touch a line of business logic.

Run it:

    uvicorn api.main:app --reload --port 8000        # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from financial_risk.config import get_settings
from financial_risk.llm.client import available_providers, llm_status, verify_provider
from financial_risk.rag.index import RetrievalUnavailable
from financial_risk.service.logs import REQUEST_ID, configure_logging
from financial_risk.serving.contract import DEFAULT_TOP_K, MAX_INSTANCES, MAX_TOP_K
from financial_risk.service.risk_service import (
    BadScoreRequest,
    ServiceContext,
    ServiceNotReady,
    TransactionNotFound,
    answer_question,
    ask_agent,
    assess_transaction,
    score_transactions,
)
from financial_risk.service.scoring import ModelUnavailable

configure_logging(get_settings().log_level)
logger = logging.getLogger("risk_api")

# Which capabilities use the configured model. Shown at startup, in /ready and in the web app.
LLM_USES = ["summaries (when requested)", "knowledge-base answers", "agent routing"]


def _llm_info(ctx: ServiceContext | None = None) -> dict:
    status = llm_status()
    check = (ctx.llm_check if ctx else None) or {}
    return {**status, "verified": check.get("verified"), "verify_detail": check.get("detail"),
            "uses": LLM_USES if status["ready"] else []}


def _startup_banner(context: ServiceContext, ready: dict, took_ms: float) -> None:
    """Say what is actually loaded and which model will answer -- the first thing to read."""
    lines = [f"{app.title} {app.version}"]
    store = context.store
    if store.pointers_enabled:
        lines.append(f"  pointers {store.prefix}/model, {store.prefix}/index · re-read every {store.ttl} s")
    else:
        lines.append("  pointers none (POINTER_PREFIX is empty) -> the baked demo model and index")
    model, index = ready["model"], ready["index"]
    if ready["checks"]["model_artifact"]:
        if model["source"] == "endpoint":
            where = f"endpoint {model['endpoint']} ({model['status'] or 'status unknown'})"
        else:
            where = "baked, scored in this process"
        threshold = model["threshold"] if model["threshold"] is not None else "n/a until the first score"
        rows = f" · {ready['rows']:,} transactions" if ready["rows"] else " · NO transaction table (/api/predict/<row_id> cannot answer)"
        facts = f" · {model['model_type']} · PR-AUC {model['pr_auc']}" if model.get("model_type") else ""
        lines.append(f"  model    version {model['version']} · {where}{facts} · threshold {threshold}{rows}")
    else:
        lines.append(f"  model    NOT LOADED{' · ' + model['load_error'] if model['load_error'] else ''}")
    if ready["checks"]["vector_index"]:
        lines.append(f"  index    version {index['version']} ({index['source']}) · {index['embeddings']} · {index['docs']} docs · {index['chunks']} chunks")
    else:
        lines.append(f"  index    NOT LOADED{' · ' + index['load_error'] if index['load_error'] else ''}")
    info = _llm_info(context)
    if info["provider"] == "none":
        lines.append("  llm      none -> template summaries, extractive answers, keyword routing")
    else:
        key = "" if info["key_set"] is None else (" · key set" if info["key_set"] else " · KEY MISSING")
        lines.append(f"  llm      {info['provider']} · model {info['model']}{key} · {info['endpoint']}")
        lines.append(f"  llm      check: {info['verify_detail']}")
        lines.append(f"  llm use  {' · '.join(info['uses']) or 'nothing (not ready)'}")
    lines.append(f"  ready    {'yes' if ready['ready'] else 'NO, missing ' + ', '.join(ready['missing'])}"
                 f" · loaded in {took_ms:.0f} ms")
    for line in lines:
        logger.info(line)

STATIC_DIR = Path(__file__).parent / "static"


def _app_version() -> str:
    """The release number from the VERSION file (bumped in the pull request that changes behaviour)."""
    try:
        return (ROOT / "VERSION").read_text(encoding="utf-8").strip() or "0.0.0"
    except OSError:
        return "0.0.0"


# ------------------------------------------------------------------------------ lifespan

state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown.

    The model, the feature table and the vector index are loaded here — once — and reused by
    every request. Put that work inside a route handler instead and each call pays it again:
    the difference between a few milliseconds and a couple of seconds per request.
    """
    started = time.perf_counter()
    context = ServiceContext.build(eager=True)
    state["service"] = context
    ready = context.readiness()
    took = (time.perf_counter() - started) * 1000
    context.llm_check = verify_provider()  # one cheap call: does the key work, is there balance?
    logger.info("startup: ready=%s missing=%s in %.0f ms", ready["ready"], ready["missing"] or "-", took)
    _startup_banner(context, ready, took)
    if not ready["ready"]:
        logger.warning("Serving without every artifact (loading goes on in the background). %s", ready["hint"])
    yield
    state.clear()
    logger.info("shutdown: released model and index")


app = FastAPI(
    title="Financial Fraud Risk Intelligence Assistant",
    description=(
        "Score a transaction, explain it, and ask the knowledge assistant — the week 1–9 "
        "project behind one HTTP interface."
    ),
    version=_app_version(),
    lifespan=lifespan,
)


def service() -> ServiceContext:
    """Dependency-style accessor. Returns the single context created at startup.

    Every call is also the moment the service looks at its pointers again if the last look is older than
    POINTER_TTL_SECONDS. Background mode (ECS, laptop): the look runs in a thread, the request is never slowed by it.
    Inline mode (Lambda): this request waits for the look and any load, at most INLINE_LOAD_TIMEOUT_SECONDS, because
    Lambda freezes a background thread between requests. /health never calls this function."""
    ctx = state.get("service")
    if ctx is None:  # only reachable if a route runs outside the lifespan
        ctx = ServiceContext.build(eager=False)
        state["service"] = ctx
    ctx.maybe_refresh()  # type: ignore[union-attr]
    return ctx  # type: ignore[return-value]


# ------------------------------------------------------------------------------- schemas


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    service: str
    version: str
    image_digest: str | None = Field(
        default=None, examples=["sha256:0123abcd..."],
        description="Which image is answering: the sha256 digest Terraform passed in as APP_IMAGE_DIGEST (null on a laptop). "
                    "The deploy pipeline's smoke test waits for the NEW digest, so an old task that still answers 200 is not mistaken for the new version.")


class LLMInfo(BaseModel):
    """What the service will use for language generation, and whether it was verified."""

    provider: str = Field(examples=["deepseek", "none"])
    model: str | None = Field(default=None, examples=["deepseek-flash"])
    ready: bool
    key_set: bool | None = None
    endpoint: str | None = None
    region: str | None = Field(default=None, description="Bedrock only: the AWS Region the model is called in.")
    credentials: bool | None = Field(default=None, description="Bedrock only: did boto3 find AWS credentials? (No model call.)")
    detail: str
    verified: bool | None = Field(default=None, description="Startup check result; None = not checked.")
    verify_detail: str | None = None
    uses: list[str] = []


class LLMCall(BaseModel):
    """One model call made while answering. `model` is what the provider reported, not our setting."""

    provider: str
    model: str | None
    purpose: str = Field(examples=["route", "rag_answer", "summary"])
    ok: bool
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    requested_model: str | None = None
    error: str | None = None


class ReadyResponse(BaseModel):
    ready: bool = Field(description="True only when a model AND an index are loaded.")
    image_digest: str | None = Field(default=None, description="The image's sha256 digest (APP_IMAGE_DIGEST); null when unset.")
    checks: dict[str, bool]
    missing: list[str]
    rows: int | None = None
    llm_providers: dict[str, bool]
    llm: LLMInfo
    load_error: str | None = Field(default=None, description="The first load problem (model or index), account ids masked. "
                                   "A failed NEWER version keeps the old one serving: ready stays true and this explains.")
    hint: str | None = None
    model: dict = Field(description="The model being served: version (the model pointer's version, or 'baked'), source (endpoint | baked), "
                                    "endpoint (the SageMaker endpoint name; null when baked), status (DescribeEndpoint, cached 60 s; null when baked), "
                                    "error (the last scoring or describe problem; null when fine), loaded_at, load_error (a pointer that could not be loaded), "
                                    "and its facts: model_type, trained_at_utc, threshold, pr_auc (baked) / the last threshold the endpoint reported (endpoint).")
    index: dict = Field(description="The knowledge index being served: version, embeddings (titan | tfidf), docs, chunks, source (s3 | baked), "
                                    "loaded_at, load_error.")
    pointers_checked_at: str | None = Field(default=None, description="When the SSM pointers were last read (UTC); null without POINTER_PREFIX.")
    pointer_prefix: str | None = Field(default=None, description="POINTER_PREFIX, e.g. /finrisk/dev; null = baked artifacts only.")


class RiskResponse(BaseModel):
    row_id: int
    transaction_type: str
    amount: float
    risk_score: float = Field(ge=0.0, le=1.0, description="Model probability of fraud.")
    threshold: float = Field(description="Chosen in week 7; above it a human reviews.")
    risk_level: str = Field(examples=["high", "low"])
    confirmed_fraud: bool | None = Field(
        default=None,
        description="Investigation outcome where one exists. NOT the prediction — see week 9.",
    )
    reasons: list[str]
    reason_method: str = Field(examples=["shap", "heuristic"])
    scored_by: str = Field(default="baked", examples=["endpoint", "baked"], description="Where the score was computed: a SageMaker endpoint, or the baked model in this process.")
    model_version: str | None = Field(default=None, examples=["3"], description="The model version that answered (the endpoint reports it; 'baked' in process).")
    summary: str | None = None
    summary_source: str = Field(examples=["template", "deepseek", "not_requested"])
    summary_model: str | None = None
    llm_calls: list[LLMCall] = []
    elapsed_ms: float


class ScoreRequest(BaseModel):
    """POST /api/score. The rows are checked against the endpoint contract (exactly ``row_id`` + the model's features); a row that
    does not fit is a 422 that lists every problem. Pydantic checks only the envelope."""

    transactions: list[dict] = Field(min_length=1, max_length=MAX_INSTANCES, description=f"1 to {MAX_INSTANCES} rows; each holds row_id and every model feature.")
    explain: bool = Field(default=False, description="Also return the top reasons (TreeSHAP) per row. Slower; leave off for bulk scoring.")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="How many reasons per row when explain is true.")


class ScoreReason(BaseModel):
    feature: str
    value: float | str
    contribution: float = Field(description="SHAP value in log-odds; positive pushes the score up.")


class ScoreResult(BaseModel):
    row_id: str
    score: float = Field(ge=0.0, le=1.0)
    flagged: bool = Field(description="score >= threshold")
    reasons: list[ScoreReason] | None = Field(default=None, description="Present only when explain was true.")


class ScoreResponse(BaseModel):
    model_version: str
    threshold: float
    scored_by: str = Field(examples=["endpoint", "baked"])
    results: list[ScoreResult]
    elapsed_ms: float


class QuestionRequest(BaseModel):
    """Pydantic validates this before the handler runs — a bad body never reaches our code."""

    question: str = Field(min_length=3, max_length=500, examples=["What is PR-AUC and why use it?"])


class Citation(BaseModel):
    source: str
    section: str | None = None
    score: float | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    answer_method: str = Field(examples=["generated", "extracted", "extracted_fallback", "refused"])
    groundedness: float | None = Field(default=None, description="Mean per-sentence support (week 9).")
    refused: bool
    refusal_reason: str | None = None
    citations: list[Citation] = []
    disclaimer: str
    llm_calls: list[LLMCall] = []
    elapsed_ms: float


class AgentResponse(BaseModel):
    question: str
    answer: str
    tool: str | None = None
    arguments: dict = {}
    refused: bool
    refusal_reason: str | None = None
    route_method: str
    trace: list[dict]
    data: dict = Field(default={}, description="The tool's structured result, for rich display.")
    disclaimer: str
    llm_calls: list[LLMCall] = []
    elapsed_ms: float


class ErrorResponse(BaseModel):
    """Every failure looks like this. A client can parse an error as reliably as a success."""

    error: str
    detail: str
    request_id: str | None = None
    hint: str | None = None


# ---------------------------------------------------------------------------- middleware


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tag each request so a log line and a client-side error can be tied together."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    token = REQUEST_ID.set(request_id)  # every log line made while serving this carries the id
    quiet = request.url.path.startswith("/static") or request.url.path == "/favicon.ico"
    target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    try:
        if not quiet:
            logger.info("-> %s %s", request.method, target)
        started = time.perf_counter()
        response = await call_next(request)
        duration = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{duration:.1f}"
        if not quiet:
            logger.log(logging.WARNING if response.status_code >= 500 else logging.INFO,
                       "<- %s %s %s in %.0f ms", response.status_code, request.method, target, duration)
        return response
    finally:
        REQUEST_ID.reset(token)


# ----------------------------------------------------------------------- error handlers


def _error(request: Request, code: int, error: str, detail: str, hint: str | None = None):
    return JSONResponse(
        status_code=code,
        content=ErrorResponse(
            error=error, detail=detail,
            request_id=getattr(request.state, "request_id", None), hint=hint,
        ).model_dump(),
    )


@app.exception_handler(TransactionNotFound)
async def handle_not_found(request: Request, exc: TransactionNotFound):
    # 404: the request was valid, the thing is not there. Retrying will not help.
    return _error(request, status.HTTP_404_NOT_FOUND, "transaction_not_found", str(exc),
                  "Try a row_id from GET /api/transactions/sample")


@app.exception_handler(ServiceNotReady)
async def handle_not_ready(request: Request, exc: ServiceNotReady):
    # 503: nothing wrong with the request; the server cannot serve it YET. Retry later.
    return _error(request, status.HTTP_503_SERVICE_UNAVAILABLE, "service_not_ready", str(exc),
                  "See GET /ready: model and index say what is loaded, load_error says what failed.")


@app.exception_handler(ModelUnavailable)
async def handle_model_unavailable(request: Request, exc: ModelUnavailable):
    # 503: the request is fine; the MODEL cannot answer right now (a cold Serverless endpoint, throttling, a missing endpoint).
    # There is no fallback to the baked model on purpose: see service/scoring.py. A cold endpoint wakes up in about a minute.
    response = _error(request, status.HTTP_503_SERVICE_UNAVAILABLE, "model_unavailable", str(exc),
                      "Retry in a few seconds." if exc.starting else "See GET /ready: model.status and model.error say what the endpoint is doing.")
    response.headers["Retry-After"] = "10" if exc.starting else "5"
    return response


@app.exception_handler(BadScoreRequest)
async def handle_bad_score_request(request: Request, exc: BadScoreRequest):
    # 422: the rows do not obey the endpoint contract. `problems` names the row and what is wrong with it.
    response = _error(request, 422, "invalid_transactions", str(exc),
                      f"Each transaction needs exactly row_id and the model's features (see docs/api_schema.md); at most {MAX_INSTANCES} per call.")
    body = json.loads(response.body)
    body["problems"] = exc.problems
    return JSONResponse(status_code=response.status_code, content=body)


@app.exception_handler(RetrievalUnavailable)
async def handle_retrieval_unavailable(request: Request, exc: RetrievalUnavailable):
    # 503: the question is fine, but the knowledge base cannot be searched right now (the embedding call failed).
    return _error(request, status.HTTP_503_SERVICE_UNAVAILABLE, "knowledge_base_unavailable", str(exc),
                  "Retry in a moment. /ready shows the index that is being served.")


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception):
    # The safety net. A stack trace in an HTTP response is an information leak and a bad
    # experience; it belongs in the server log, keyed by the request id the caller can quote.
    logger.exception("unhandled error rid=%s", getattr(request.state, "request_id", None))
    return _error(request, status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error",
                  "An unexpected error occurred. Quote the request id when reporting it.")


# ---------------------------------------------------------------------------- operations


def _image_digest() -> str | None:
    """The digest of the image this process runs from. Terraform sets APP_IMAGE_DIGEST on every deploy (deploy/terraform/main.tf);
    it is None on a laptop. Read at request time, so a test can set it."""
    return os.environ.get("APP_IMAGE_DIGEST") or None


@app.get("/health", response_model=HealthResponse, tags=["ops"], summary="Liveness probe")
def health() -> HealthResponse:
    """Is the process alive? Always 200 while it can answer at all.

    Deliberately does NOT touch the model: a liveness probe that checks dependencies will
    restart a healthy process because a disk was slow.
    """
    return HealthResponse(status="ok", service=app.title, version=app.version, image_digest=_image_digest())


@app.get("/ready", response_model=ReadyResponse, tags=["ops"], summary="Readiness probe")
def ready() -> JSONResponse:
    """Can it actually serve? 200 when every artifact is present, 503 when one is missing.

    Liveness vs readiness is the distinction: restart me, versus don't send me traffic yet.
    """
    info = service().readiness()
    payload = ReadyResponse(**info, image_digest=_image_digest(), llm_providers=available_providers(),
                            llm=_llm_info(service())).model_dump()
    code = status.HTTP_200_OK if info["ready"] else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=payload)


@app.get("/api/transactions/sample", tags=["risk"], summary="A few row_ids to try")
def sample_transactions(limit: int = 5, high_risk: bool = False) -> dict:
    """Convenience for the UI and for anyone poking at the API by hand."""
    ctx = service()
    frame = ctx.features
    if high_risk and "is_fraud" in frame.columns:
        frame = frame[frame["is_fraud"] == 1]
    picked = frame.head(max(1, min(limit, 50)))
    return {
        "row_ids": [int(r) for r in picked["row_id"].tolist()],
        "total_rows": int(len(ctx.features)),
    }


@app.get(
    "/api/predict/{row_id}",
    response_model=RiskResponse,
    tags=["risk"],
    summary="Score and explain one transaction",
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def predict(row_id: int, use_llm: bool = False) -> RiskResponse:
    """stored row -> model (a SageMaker endpoint, or the baked model) -> SHAP reasons -> (optional) LLM summary.

    ``row_id`` is typed as ``int``: FastAPI rejects ``/api/predict/abc`` with a 422 before this
    function runs. Validation you did not write is validation you cannot forget.
    """
    return RiskResponse(**assess_transaction(service(), row_id, use_llm=use_llm))


@app.post(
    "/api/score",
    response_model=ScoreResponse,
    response_model_exclude_none=True,
    tags=["risk"],
    summary="Score a batch of transactions (the production path)",
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def score(payload: ScoreRequest) -> ScoreResponse:
    """1 to 500 transactions that arrive WITH their features -> one call to the model -> a score and a flag each.

    This is how a payments system (or the replay in finrisk-ml) uses the service: nothing is looked up, so there is no
    ``row_id`` lookup and no label. With a SageMaker endpoint behind it, it is one ``InvokeEndpoint`` per request and the
    endpoint's Data Capture records it. A cold Serverless endpoint answers 503 "model endpoint is starting, retry"."""
    return ScoreResponse(**score_transactions(service(), payload.transactions, explain=payload.explain, top_k=payload.top_k))


@app.post(
    "/api/ask",
    response_model=AnswerResponse,
    tags=["assistant"],
    summary="Ask the knowledge base (RAG)",
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def ask(payload: QuestionRequest) -> AnswerResponse:
    """A cited answer from the indexed documents, or a refusal with its reason."""
    return AnswerResponse(**answer_question(service(), payload.question))


@app.post(
    "/api/agent",
    response_model=AgentResponse,
    tags=["assistant"],
    summary="Ask the agent (picks a tool)",
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def agent(payload: QuestionRequest) -> AgentResponse:
    """Week 9's agent: compliance gate, router, one of five tools, and the trace that proves it."""
    return AgentResponse(**ask_agent(service(), payload.question))


# --------------------------------------------------------------------------------- pages

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    """The web app. Same origin as the API, so the page can fetch() it without CORS."""
    page = STATIC_DIR / "index.html"
    if page.exists():
        return FileResponse(page)
    raise HTTPException(status_code=404, detail="UI not installed; the JSON API is at /docs")

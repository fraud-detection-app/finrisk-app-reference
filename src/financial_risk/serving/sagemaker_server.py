"""The SageMaker hosting server: the model, behind the two routes SageMaker requires.

SageMaker starts a container with the argument ``serve`` and expects a web server on port 8080 that answers

    GET  /ping          200 when the model is loaded and the container can take traffic (anything else = unhealthy)
    POST /invocations   the prediction request; whatever the client sent to InvokeEndpoint arrives here

before it will route a single request to it. The model files come from the model package (``model.tar.gz``), which
SageMaker has already unpacked into ``/opt/ml/model``. This file is only that hosting contract: loading
(models/package.py), validating (serving/contract.py) and scoring (serving/runner.py) are shared with the rest of
the code, so the endpoint and a laptop run the same lines.

    python -m financial_risk.serving.sagemaker_server      # what finrisk-ml's `serve` command runs

Settings (environment):

    SM_MODEL_DIR          where the unpacked package is. Default /opt/ml/model (SageMaker); tests and a laptop point it elsewhere.
    MODEL_VERSION         the model version this endpoint serves; echoed in every response (set on the SageMaker Model).
    SAGEMAKER_BIND_TO_PORT  the port (default 8080).

ONE worker, one copy of the model: a Serverless Inference endpoint has 1-6 GB for the whole container, and the
model is small. Scoring is serialised inside the runner, so the single worker is also the safe one for XGBoost.
Request bodies are never logged (they are transactions); the log says how many rows and how long.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from financial_risk.serving.contract import MAX_INSTANCES, InvalidRequest
from financial_risk.serving.runner import ModelRunner
from financial_risk.service.logs import REQUEST_ID, configure_logging

logger = logging.getLogger("financial_risk.serving")

DEFAULT_MODEL_DIR = "/opt/ml/model"
DEFAULT_PORT = 8080


def model_dir_from_env() -> Path:
    return Path(os.environ.get("SM_MODEL_DIR") or DEFAULT_MODEL_DIR)


def _background_if_needed(runner_artifact: dict[str, Any]) -> Any:
    """Only a model that is neither XGBoost nor logistic regression needs background rows to explain itself (the week 7
    ``shap.Explainer`` path). The release image bakes a small demo feature table; use a sample of it when it is there."""
    estimator = runner_artifact["model"].steps[-1][1]
    if hasattr(estimator, "get_booster") or type(estimator).__name__ == "LogisticRegression":
        return None
    try:
        import pandas as pd

        from financial_risk.config import get_settings

        path = get_settings().gold_path / "transaction_features.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            return frame.sample(min(200, len(frame)), random_state=42)
    except Exception as exc:  # noqa: BLE001 - no background means no reasons, never a failed start
        logger.warning("serve: no background rows for the explainer (%s)", exc)
    return None


def create_app(model_dir: Path | str | None = None, model_version: str | None = None) -> FastAPI:
    """Build the hosting app. The model is loaded when the server starts (not at import), so importing is free."""
    state: dict[str, Any] = {"runner": None, "error": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
        directory = Path(model_dir) if model_dir else model_dir_from_env()
        version = model_version or os.environ.get("MODEL_VERSION") or "unversioned"
        started = time.perf_counter()
        try:
            from financial_risk.models.package import load_model_package

            background = _background_if_needed(load_model_package(directory))
            state["runner"] = ModelRunner.from_package_dir(directory, version, background)
            facts = state["runner"].facts()
            logger.info("serve: model version %s loaded from %s in %.0f ms · %s · threshold %s · /ping is healthy",
                        version, directory, (time.perf_counter() - started) * 1000, facts["model_type"], facts["threshold"])
        except Exception as exc:  # noqa: BLE001 - stay up and say why: SageMaker reads /ping and the CloudWatch log
            state["error"] = f"{type(exc).__name__}: {exc}"
            logger.error("serve: could not load the model from %s -- %s. /ping will answer 503.", directory, state["error"])
        yield
        state["runner"] = None

    app = FastAPI(title="finrisk SageMaker endpoint", version="1", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)       # nothing but the two routes SageMaker uses
    app.state.serving = state

    @app.get("/ping")
    def ping() -> JSONResponse:
        """SageMaker's health check: 200 only when the model is loaded."""
        if state["runner"] is None:
            return JSONResponse(status_code=503, content={"status": "unhealthy", "error": state["error"] or "model not loaded yet"})
        return JSONResponse(content={"status": "ok", "model_version": state["runner"].model_version})

    @app.post("/invocations")
    async def invocations(request: Request) -> JSONResponse:
        """The prediction request (serving/contract.py). 400 lists every problem; the response never echoes the features back."""
        runner: ModelRunner | None = state["runner"]
        request_id = request.headers.get("x-amzn-sagemaker-inference-id") or uuid.uuid4().hex[:12]
        token = REQUEST_ID.set(request_id)
        try:
            if runner is None:
                return JSONResponse(status_code=503, content={"error": "model_not_loaded", "message": state["error"] or "model not loaded yet"})
            started = time.perf_counter()
            try:
                payload = json.loads(await request.body())
            except (ValueError, UnicodeDecodeError):
                return JSONResponse(status_code=400, content={
                    "error": "invalid_json", "message": "the body must be JSON: {\"instances\": [...], \"explain\": false}", "problems": []})
            try:
                response = await run_in_threadpool(runner.invoke, payload)
            except InvalidRequest as exc:
                logger.warning("invocations: 400 %s", str(exc)[:300])
                return JSONResponse(status_code=400, content={"error": "invalid_request", "message": str(exc), "problems": exc.problems,
                                                              "max_instances": MAX_INSTANCES})
            except Exception:  # noqa: BLE001 - the safety net: a traceback belongs in the log, not in the response
                logger.exception("invocations: unexpected error")
                return JSONResponse(status_code=500, content={"error": "internal_error", "message": "unexpected error; see the endpoint's CloudWatch log"})
            rows = len(response["predictions"])
            flagged = sum(1 for p in response["predictions"] if p["flagged"])
            logger.info("invocations: %d row(s) · %d flagged · explain=%s · model %s · %.0f ms",
                        rows, flagged, bool(payload.get("explain")), response["model_version"], (time.perf_counter() - started) * 1000)
            return JSONResponse(content=response)
        finally:
            REQUEST_ID.reset(token)

    return app


app = create_app()


def main() -> None:
    """Run the server the way SageMaker expects: all interfaces, port 8080, one worker."""
    import uvicorn

    port = int(os.environ.get("SAGEMAKER_BIND_TO_PORT") or DEFAULT_PORT)
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, log_level="warning")


if __name__ == "__main__":
    main()

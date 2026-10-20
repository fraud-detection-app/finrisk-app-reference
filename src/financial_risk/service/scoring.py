"""Who scores a transaction: this process, or a SageMaker endpoint. One interface, two implementations.

Until 13b the service loaded the model into its own memory. Now the model is *deployed* on its own: finrisk-ml
promotes a version to a SageMaker endpoint (`finrisk-<env>-fraud`) and writes the endpoint's name into the model
pointer. The service stops owning the model and asks the endpoint instead. The rest of the code (the API, the
agent, the LLM summary) must not care which, so both implementations speak the endpoint contract
(serving/contract.py):

    scorer.score(instances, explain=False, top_k=3) -> {"model_version", "threshold", "predictions": [...]}

    LocalScorer     the model in this process (the baked demo model: a laptop, the tests, a fresh environment). It is
                    the very ModelRunner the SageMaker container runs, so both give the same numbers.
    EndpointScorer  sagemaker-runtime InvokeEndpoint. Connect 5 s, read 40 s. A cold Serverless endpoint can need about
                    a minute to wake up: a timeout is reported as 503 "model endpoint is starting, retry", never hidden.

Which one is used is decided by the model pointer (service/artifacts.py): `{"version": null}` -> LocalScorer on the baked
model; `{"version": N, "serving": "endpoint", "endpoint": ...}` -> EndpointScorer. There is NO silent fallback from a
configured endpoint to the baked model: a model that answers differently from the one you deployed is worse than an
error you can see. A scorer that fails raises ``ModelUnavailable`` (HTTP 503) and /ready shows why.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Protocol, runtime_checkable

import pandas as pd

from financial_risk.serving.contract import (
    DEFAULT_TOP_K, InvalidRequest, build_request, frame_to_instances, parse_request, reason_sentences,
)
from financial_risk.serving.runner import ModelRunner
from financial_risk.service.masking import mask

logger = logging.getLogger("financial_risk.service")

__all__ = ["EndpointScorer", "InvalidRequest", "LocalScorer", "ModelUnavailable", "Scorer", "reason_sentences",
           "score_frame"]

STARTING_MESSAGE = "model endpoint is starting, retry"


class ModelUnavailable(RuntimeError):
    """The model cannot answer right now (endpoint cold, throttled, missing, denied ...). Maps to HTTP 503.

    ``starting`` is True when the likely cause is a cold Serverless endpoint: retrying in a few seconds will work."""

    def __init__(self, message: str, starting: bool = False, retryable: bool = False) -> None:
        super().__init__(message)
        self.starting = starting
        self.retryable = retryable             # worth another try within the same request (a fast failure, not a timeout)


@runtime_checkable
class Scorer(Protocol):
    """What the service needs from a model, wherever it runs."""

    source: str                        # "baked" | "endpoint"
    version: Any
    endpoint: str | None

    @property
    def threshold(self) -> float | None: ...

    def score(self, instances: list[dict[str, Any]], explain: bool = False, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]: ...

    def facts(self) -> dict[str, Any]: ...


# ------------------------------------------------------------------------------ in this process


class LocalScorer:
    """The model inside this process: today's behaviour, now behind the same interface as the endpoint."""

    source = "baked"
    endpoint = None

    def __init__(self, runner: ModelRunner, version: Any = "baked") -> None:
        self.runner = runner
        self.version = version

    @property
    def threshold(self) -> float:
        return self.runner.threshold

    def score(self, instances: list[dict[str, Any]], explain: bool = False, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        return self.runner.invoke(build_request(instances, explain, top_k))

    def facts(self) -> dict[str, Any]:
        return {**self.runner.facts(), "endpoint": None, "status": None, "error": None}


# ------------------------------------------------------------------------------ a SageMaker endpoint


class EndpointScorer:
    """Calls ``sagemaker-runtime`` ``InvokeEndpoint`` on the endpoint the model pointer names.

    Both boto3 clients are injectable (``runtime``, ``sagemaker``), which is how the tests run it without AWS.

    Retries (2, as the spec says), done here rather than by botocore so a total time budget can be enforced -- a Lambda
    function has 60 s, and a client that waits 40 s three times would be killed mid-retry. The timeouts are settings
    (ENDPOINT_CONNECT_TIMEOUT_SECONDS 5, ENDPOINT_READ_TIMEOUT_SECONDS 40; 25 on Lambda) and the budget derives from them
    (``budget`` = connect + read + 5 s slack):
      * retried, with a short pause:  a connection that fails fast, throttling, a 5xx from the service,
        ``ModelNotReadyException`` (a Serverless endpoint still loading its container)
      * NOT retried:  a read timeout (40 s already went by: report "starting" now and let the caller retry),
        anything that says the request or the permissions are wrong
    """

    source = "endpoint"
    _RETRYABLE_CODES = {"ThrottlingException", "ServiceUnavailable", "InternalFailure", "InternalDependencyException", "ModelNotReadyException"}
    _STARTING_CODES = {"ModelNotReadyException"}

    def __init__(
        self,
        endpoint: str,
        version: Any,
        region: str = "ap-southeast-2",
        runtime: Any | None = None,
        sagemaker: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        status_ttl: float = 60.0,
        retries: int = 2,
        budget_seconds: float | None = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 40.0,
        slack_seconds: float = 5.0,
    ) -> None:
        self.endpoint = endpoint
        self.version = version
        self.region = region
        self._runtime, self._sagemaker = runtime, sagemaker
        self._clock, self._sleep = clock, sleep
        self.connect_timeout, self.read_timeout = connect_timeout, read_timeout
        # THE BUDGET RULE: one score() call never takes longer than connect + read + slack (defaults 5 + 40 + 5 = 50 s;
        # on Lambda 5 + 25 + 5 = 35 s). A retry is only started when it could still finish inside that budget, so the
        # retries use the slack and never push the call past it. Pass budget_seconds to override (tests).
        self.budget = budget_seconds if budget_seconds is not None else connect_timeout + read_timeout + slack_seconds
        self.status_ttl, self.retries = status_ttl, retries
        self._lock = threading.Lock()
        self.last_error: str | None = None
        self.last_threshold: float | None = None
        self._status: tuple[str | None, str | None, float] | None = None       # (status, error, when)

    @property
    def threshold(self) -> float | None:
        """The threshold the endpoint reported in its last response (None until the first score)."""
        return self.last_threshold

    # ---- clients

    def _runtime_client(self) -> Any:
        if self._runtime is None:
            import boto3
            from botocore.config import Config

            # total_max_attempts=1: botocore does not retry; score() does, inside a time budget.
            self._runtime = boto3.client("sagemaker-runtime", region_name=self.region, config=Config(
                connect_timeout=self.connect_timeout, read_timeout=self.read_timeout, retries={"total_max_attempts": 1}))
        return self._runtime

    def _sagemaker_client(self) -> Any:
        if self._sagemaker is None:
            import boto3
            from botocore.config import Config

            self._sagemaker = boto3.client("sagemaker", region_name=self.region, config=Config(
                connect_timeout=3, read_timeout=5, retries={"total_max_attempts": 2}))
        return self._sagemaker

    # ---- scoring

    def score(self, instances: list[dict[str, Any]], explain: bool = False, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        request = build_request(instances, explain, top_k)
        parse_request(request)                       # a bad row is OUR 422, found before any AWS call (raises InvalidRequest)
        body = json.dumps(request)
        started = self._clock()
        attempt = 0
        while True:
            try:
                raw = self._invoke_once(body)
                break
            except ModelUnavailable as exc:
                pause = 0.4 * (2 ** attempt)
                # a retry may only start if the pause plus one more worst-case call (connect + read) still fits the budget
                if (not exc.retryable or attempt >= self.retries
                        or (self._clock() - started) + pause + self.connect_timeout + self.read_timeout > self.budget):
                    self._remember_error(exc)
                    raise
                logger.warning("endpoint %s: %s -- retry %d of %d in %.1f s", self.endpoint, exc, attempt + 1, self.retries, pause)
                self._sleep(pause)
                attempt += 1
        response = self._check(raw, len(instances))
        with self._lock:
            self.last_error = None
            self.last_threshold = float(response["threshold"])
        return response

    def _invoke_once(self, body: str) -> bytes:
        from botocore.exceptions import (
            BotoCoreError, ClientError, ConnectionClosedError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError,
        )

        try:
            result = self._runtime_client().invoke_endpoint(
                EndpointName=self.endpoint, ContentType="application/json", Accept="application/json", Body=body)
            return result["Body"].read()
        except ReadTimeoutError as exc:
            raise ModelUnavailable(STARTING_MESSAGE, starting=True) from exc
        except (ConnectTimeoutError, EndpointConnectionError, ConnectionClosedError) as exc:
            raise ModelUnavailable(mask(f"cannot reach the model endpoint: {type(exc).__name__}"), retryable=True) from exc
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            message = exc.response.get("Error", {}).get("Message", "")
            if code == "ModelError":
                original = exc.response.get("OriginalStatusCode")
                raise ModelUnavailable(mask(f"the model endpoint rejected the request ({original}): {exc.response.get('OriginalMessage') or message}"[:300])) from exc
            starting = code in self._STARTING_CODES
            raise ModelUnavailable(STARTING_MESSAGE if starting else mask(f"model endpoint error {code}: {message}"[:300]),
                                   starting=starting, retryable=code in self._RETRYABLE_CODES) from exc
        except BotoCoreError as exc:
            raise ModelUnavailable(mask(f"model endpoint call failed: {type(exc).__name__}: {exc}"[:300])) from exc

    def _check(self, raw: bytes, expected_rows: int) -> dict[str, Any]:
        """The endpoint answered; is it the contract's answer? A stale or wrong container fails here, with a reason."""
        try:
            response = json.loads(raw)
            predictions = response["predictions"]
            float(response["threshold"])
        except (ValueError, KeyError, TypeError) as exc:
            error = ModelUnavailable(f"the model endpoint answered something that is not the contract's response ({type(exc).__name__})")
            self._remember_error(error)
            raise error from exc
        if not isinstance(predictions, list) or len(predictions) != expected_rows:
            error = ModelUnavailable(f"the model endpoint scored {len(predictions) if isinstance(predictions, list) else '?'} row(s) for {expected_rows} sent")
            self._remember_error(error)
            raise error
        return response

    def _remember_error(self, exc: Exception) -> None:
        with self._lock:
            self.last_error = str(exc)

    # ---- /ready

    def verify(self) -> str:
        """DescribeEndpoint now (not cached). Used when the pointer moves: an endpoint that does not exist, or that this role
        may not describe, fails the load and the old model keeps serving. Returns the status."""
        status = self._sagemaker_client().describe_endpoint(EndpointName=self.endpoint)["EndpointStatus"]
        with self._lock:
            self._status = (status, None, self._clock())
        return status

    def status(self) -> tuple[str | None, str | None]:
        """-> (EndpointStatus, error). DescribeEndpoint, cached for ``status_ttl`` seconds so /ready can be polled."""
        with self._lock:
            cached = self._status
        if cached is not None and self._clock() - cached[2] < self.status_ttl:
            return cached[0], cached[1]
        try:
            status = self._sagemaker_client().describe_endpoint(EndpointName=self.endpoint)["EndpointStatus"]
            error = None
        except Exception as exc:  # noqa: BLE001 - /ready must always answer
            status, error = None, mask(f"cannot describe endpoint {self.endpoint}: {type(exc).__name__}: {exc}"[:300])
        with self._lock:
            self._status = (status, error, self._clock())
        return status, error

    def facts(self) -> dict[str, Any]:
        status, describe_error = self.status()
        return {"model_type": None, "trained_at_utc": None, "threshold": None if self.last_threshold is None else round(self.last_threshold, 4),
                "pr_auc": None, "endpoint": self.endpoint, "status": status, "error": self.last_error or describe_error}


# ------------------------------------------------------------------------------ helpers


def score_frame(scorer: Scorer, frame: pd.DataFrame, explain: bool = False, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Score the rows of a feature table (``row_id`` + the model's features) through any scorer."""
    return scorer.score(frame_to_instances(frame), explain=explain, top_k=top_k)

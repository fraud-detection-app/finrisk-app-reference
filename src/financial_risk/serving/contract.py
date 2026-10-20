"""The endpoint contract: what a scoring request and response look like, and the validation of a request.

ONE definition, used by everybody who talks to the model:

    the SageMaker endpoint     serving/sagemaker_server.py   POST /invocations
    the app (any scorer)       service/scoring.py            validates BEFORE it calls AWS, so a bad row is a clean 422 here
    the replay and the monitor finrisk-ml                    build requests and read captured ones with the same helpers
    the tests                  tests/test_serving.py

Request (application/json)::

    {"instances": [{"row_id": "4821377", "transaction_type": "TRANSFER", "amount": 181000.0, ...every FEATURE_COLUMNS key},
                   ...],                                   # 1 to 500 instances
     "explain": true,                                      # optional, default false: add the top reasons per row
     "top_k": 3}                                           # optional, default 3 (1..20): how many reasons

Response::

    {"model_version": "3", "threshold": 0.4812,
     "predictions": [{"row_id": "4821377", "score": 0.9132, "flagged": true,
                      "reasons": [{"feature": "origin_new_balance", "value": 0.0, "contribution": 6.21}, ...]}]}

``reasons`` is present only when ``explain`` was true. ``contribution`` is a SHAP value in log-odds: positive pushes the
score up. Each instance must hold EXACTLY the keys ``row_id`` + FEATURE_COLUMNS: a missing key is a 400 (the model would
guess), and so is an extra one (a typo such as ``amout`` would otherwise be ignored silently). The 400 lists every problem.

Why strict: this is the interface a monitor reads back from the capture files months later. A contract that quietly
accepts anything is a contract nobody can rely on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from financial_risk.models.training import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

MAX_INSTANCES = 500                 # well under SageMaker's 6 MB (real-time) / 4 MB (serverless) payload limits
DEFAULT_TOP_K = 3
MAX_TOP_K = 20
INSTANCE_KEYS: tuple[str, ...] = ("row_id", *FEATURE_COLUMNS)
REQUEST_KEYS = {"instances", "explain", "top_k"}
_MAX_PROBLEMS_LISTED = 20


class InvalidRequest(ValueError):
    """The request breaks the contract. ``problems`` is the machine-readable list (HTTP 400 / 422)."""

    def __init__(self, message: str, problems: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.problems = problems or []


@dataclass
class ParsedRequest:
    frame: pd.DataFrame                       # FEATURE_COLUMNS only, numeric columns as float64
    row_ids: list[str]
    explain: bool = False
    top_k: int = DEFAULT_TOP_K
    raw: list[dict[str, Any]] = field(default_factory=list, repr=False)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _problem_text(p: dict[str, Any]) -> str:
    parts = []
    if p.get("missing"):
        parts.append(f"missing {p['missing']}")
    if p.get("extra"):
        parts.append(f"extra {p['extra']}")
    if p.get("invalid"):
        parts.append("invalid " + "; ".join(f"{k}: {v}" for k, v in p["invalid"].items()))
    return f"instance {p['index']} (row_id {p.get('row_id')!r}): " + ", ".join(parts)


def validate_instances(instances: Any) -> tuple[pd.DataFrame, list[str]]:
    """-> (the feature frame, the row ids as strings). Raises ``InvalidRequest`` listing every problem found."""
    if not isinstance(instances, list) or not instances:
        raise InvalidRequest("'instances' must be a non-empty list of objects",
                             [{"index": None, "invalid": {"instances": "must be a non-empty list"}}])
    if len(instances) > MAX_INSTANCES:
        raise InvalidRequest(f"at most {MAX_INSTANCES} instances per request (got {len(instances)}); send several requests",
                             [{"index": None, "invalid": {"instances": f"{len(instances)} > {MAX_INSTANCES}"}}])

    problems: list[dict[str, Any]] = []
    for index, item in enumerate(instances):
        if not isinstance(item, dict):
            problems.append({"index": index, "row_id": None, "invalid": {"instance": f"must be an object, got {type(item).__name__}"}})
            continue
        missing = [k for k in INSTANCE_KEYS if k not in item]
        extra = sorted(k for k in item if k not in INSTANCE_KEYS)
        invalid: dict[str, str] = {}
        for key in NUMERIC_FEATURES:
            if key in item and not _is_number(item[key]):
                invalid[key] = f"must be a finite number, got {item[key]!r}"
        for key in CATEGORICAL_FEATURES:
            if key in item and not (isinstance(item[key], str) and item[key].strip()):
                invalid[key] = f"must be a non-empty string, got {item[key]!r}"
        row_id = item.get("row_id")
        if "row_id" in item and not ((isinstance(row_id, str) and row_id.strip()) or (isinstance(row_id, int) and not isinstance(row_id, bool))):
            invalid["row_id"] = f"must be a string or a whole number, got {row_id!r}"
        if missing or extra or invalid:
            problems.append({"index": index, "row_id": row_id if "row_id" in item else None,
                             **({"missing": missing} if missing else {}), **({"extra": extra} if extra else {}),
                             **({"invalid": invalid} if invalid else {})})
    if problems:
        shown = problems[:_MAX_PROBLEMS_LISTED]
        message = "; ".join(_problem_text(p) for p in shown)
        if len(problems) > len(shown):
            message += f"; ... and {len(problems) - len(shown)} more instance(s) with problems"
        raise InvalidRequest(message, shown)

    frame = pd.DataFrame([{k: item[k] for k in FEATURE_COLUMNS} for item in instances], columns=FEATURE_COLUMNS)
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].astype("float64")
    return frame, [str(item["row_id"]) for item in instances]


def parse_request(payload: Any) -> ParsedRequest:
    """Validate a whole request body (already JSON-decoded). Raises ``InvalidRequest``."""
    if not isinstance(payload, dict):
        raise InvalidRequest("the request body must be a JSON object with an 'instances' list",
                             [{"index": None, "invalid": {"body": f"got {type(payload).__name__}"}}])
    unknown = sorted(set(payload) - REQUEST_KEYS)
    if unknown:
        raise InvalidRequest(f"unknown top-level key(s) {unknown}; allowed: {sorted(REQUEST_KEYS)}",
                             [{"index": None, "extra": unknown}])
    explain = payload.get("explain", False)
    if not isinstance(explain, bool):
        raise InvalidRequest(f"'explain' must be true or false, got {explain!r}", [{"index": None, "invalid": {"explain": repr(explain)}}])
    top_k = payload.get("top_k", DEFAULT_TOP_K)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K:
        raise InvalidRequest(f"'top_k' must be a whole number from 1 to {MAX_TOP_K}, got {top_k!r}",
                             [{"index": None, "invalid": {"top_k": repr(top_k)}}])
    frame, row_ids = validate_instances(payload.get("instances"))
    return ParsedRequest(frame=frame, row_ids=row_ids, explain=explain, top_k=top_k, raw=payload["instances"])


def build_request(instances: list[dict[str, Any]], explain: bool = False, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """The request body for a list of instances (the client side: the app, the replay, the smoke test)."""
    return {"instances": instances, "explain": bool(explain), "top_k": int(top_k)}


def frame_to_instances(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """A feature table (``row_id`` + FEATURE_COLUMNS, extra columns ignored) -> JSON-ready instances.

    numpy numbers become plain Python ones, because ``json.dumps`` refuses ``np.int64``."""
    instances = []
    for record in frame[list(INSTANCE_KEYS)].to_dict(orient="records"):
        row = {}
        for key, value in record.items():
            row[key] = value.item() if hasattr(value, "item") else value
        row["row_id"] = str(row["row_id"])
        instances.append(row)
    return instances


def reason_sentences(reasons: list[dict[str, Any]]) -> list[str]:
    """Contract reasons -> the sentences the web app and the LLM summary read (same wording as week 7's local_reasons)."""
    sentences = []
    for reason in reasons:
        direction = "pushes risk up" if reason["contribution"] > 0 else "pushes risk down"
        sentences.append(f"`{reason['feature']}` {direction} (SHAP contribution {reason['contribution']:+.3f}).")
    return sentences

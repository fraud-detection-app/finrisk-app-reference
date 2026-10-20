"""Scores a request against a loaded model package: the code that is the endpoint, minus the HTTP.

``ModelRunner`` takes a request that obeys serving/contract.py and returns the contract's response. Two places use
exactly this class, which is the point:

  * the SageMaker endpoint  (``serving/sagemaker_server.py`` wraps it in the hosting contract: /ping + /invocations)
  * the app's LocalScorer   (``service/scoring.py``: the baked demo model on a laptop, in the tests, on a fresh environment)

so "the same code that scores on a laptop scores on SageMaker" is literally true, and a test of the laptop path
is a test of the endpoint's maths.

Reasons -- where "why was this flagged" comes from, in log-odds, summed back to the ORIGINAL feature names:

  XGBoost model              TreeSHAP from the booster itself (``pred_contribs=True``): exact, fast (~1 ms per row),
                             and needs no background data. The one-hot columns of ``transaction_type`` are summed back
                             into one ``transaction_type`` contribution.
  LogisticRegression         the exact linear SHAP value ``coef * x`` of each (standardised) column, i.e. measured from
  (the week 6 baseline)      the training mean for numbers and from "none of the categories" for ``transaction_type``.
  anything else              the existing week 7 path (``models/explainability``: ``shap.Explainer`` with a background
                             sample) when a ``background`` frame was given; otherwise no reasons (an empty list) -- the
                             caller then falls back to rule-of-thumb sentences and says so.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from financial_risk.models.package import load_model_package
from financial_risk.models.training import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, predict_proba_positive
from financial_risk.serving.contract import InvalidRequest, ParsedRequest, parse_request

logger = logging.getLogger("financial_risk.service")

__all__ = ["ModelRunner", "original_feature_of"]


def original_feature_of(transformed_name: str) -> str:
    """``numeric__amount`` -> ``amount``; ``categorical__transaction_type_CASH_OUT`` -> ``transaction_type``."""
    rest = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
    if rest in NUMERIC_FEATURES:
        return rest
    for name in CATEGORICAL_FEATURES:
        if rest == name or rest.startswith(name + "_"):
            return name
    raise ValueError(f"cannot map the model column {transformed_name!r} back to a feature")


def _dense(x: Any) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype="float64")


class ModelRunner:
    """A loaded model that answers contract requests. Safe to share between threads (scoring is serialised)."""

    def __init__(self, artifact: dict[str, Any], model_version: Any = "unversioned", background: pd.DataFrame | None = None) -> None:
        self.artifact = artifact
        self.model_version = str(model_version)
        self.pipeline = artifact["model"]
        self.threshold = float(artifact["threshold"])
        self._background = background
        self._lock = threading.Lock()
        self._column_group: list[int] | None = None       # transformed column -> index in FEATURE_COLUMNS
        self._generic_explainer: Any = None
        self._generic_failed = False

    @classmethod
    def from_package_dir(cls, model_dir: Any, model_version: Any = "unversioned", background: pd.DataFrame | None = None) -> "ModelRunner":
        """Open an unpacked model package (``/opt/ml/model`` on SageMaker) and prove it works by scoring one row."""
        runner = cls(load_model_package(model_dir), model_version, background)
        runner.self_test()
        return runner

    # ------------------------------------------------------------------ public

    def facts(self) -> dict[str, Any]:
        metrics = self.artifact.get("metrics") or {}
        return {
            "model_type": self.artifact.get("model_type"),
            "trained_at_utc": self.artifact.get("created_at_utc"),
            "threshold": round(self.threshold, 4),
            "pr_auc": round(float(metrics["pr_auc"]), 4) if metrics.get("pr_auc") is not None else None,
        }

    def invoke(self, payload: Any) -> dict[str, Any]:
        """A request body (decoded JSON) -> the response body. Raises ``InvalidRequest`` for a contract breach."""
        return self.score(parse_request(payload))

    def score(self, request: ParsedRequest) -> dict[str, Any]:
        with self._lock:
            scores = predict_proba_positive(self.pipeline, request.frame)
            contributions = self._contributions(request.frame) if request.explain else None
        predictions = []
        for i, (row_id, score) in enumerate(zip(request.row_ids, scores)):
            item: dict[str, Any] = {"row_id": row_id, "score": round(float(score), 6), "flagged": bool(score >= self.threshold)}
            if request.explain:
                item["reasons"] = self._top_reasons(request.frame.iloc[i], contributions[i], request.top_k) if contributions is not None else []
            predictions.append(item)
        return {"model_version": self.model_version, "threshold": round(self.threshold, 6), "predictions": predictions}

    def self_test(self) -> None:
        """Score and explain one all-zero row. A model pickled by other library versions, or on other columns, fails HERE
        (at load) instead of on the first request."""
        row = {k: 0.0 for k in NUMERIC_FEATURES}
        row.update({"row_id": "self-test", "transaction_type": "PAYMENT"})
        try:
            self.invoke({"instances": [row], "explain": True, "top_k": 1})
        except InvalidRequest as exc:                                   # pragma: no cover - would be a bug in the contract itself
            raise RuntimeError(f"the self-test request was refused: {exc}") from exc

    # ------------------------------------------------------------------ reasons

    def _groups(self) -> list[int]:
        if self._column_group is None:
            pre = self.pipeline.named_steps["preprocess"]
            self._column_group = [FEATURE_COLUMNS.index(original_feature_of(n)) for n in pre.get_feature_names_out()]
        return self._column_group

    def _contributions(self, frame: pd.DataFrame) -> np.ndarray | None:
        """-> (n_rows, len(FEATURE_COLUMNS)) log-odds contributions, or None when this model type cannot explain."""
        try:
            return self._contributions_unchecked(frame)
        except Exception as exc:  # noqa: BLE001 - an explanation problem must never fail a score
            logger.warning("reasons: could not compute contributions (%s: %s); returning scores without reasons", type(exc).__name__, exc)
            return None

    def _contributions_unchecked(self, frame: pd.DataFrame) -> np.ndarray | None:
        pre = self.pipeline.named_steps["preprocess"]
        estimator = self.pipeline.steps[-1][1]
        x = _dense(pre.transform(frame[FEATURE_COLUMNS]))
        if hasattr(estimator, "get_booster"):
            import xgboost as xgb

            columns = estimator.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)[:, :-1]      # the last column is the bias
        elif isinstance(estimator, LogisticRegression):
            columns = x * estimator.coef_[0]
        elif self._background is not None and not self._generic_failed:
            from financial_risk.models.explainability import compute_shap_values, make_explainer

            try:
                if self._generic_explainer is None:
                    self._generic_explainer = make_explainer(self.pipeline, self._background)
                columns = compute_shap_values(self.pipeline, frame, None, explainer=self._generic_explainer).to_numpy()
            except Exception:
                self._generic_failed = True
                raise
        else:
            return None
        grouped = np.zeros((len(frame), len(FEATURE_COLUMNS)))
        for column, target in enumerate(self._groups()):
            grouped[:, target] += columns[:, column]
        return grouped

    @staticmethod
    def _top_reasons(row: pd.Series, contributions: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        order = np.argsort(-np.abs(contributions))[:top_k]
        reasons = []
        for j in order:
            name = FEATURE_COLUMNS[int(j)]
            value = row[name]
            reasons.append({"feature": name, "value": value if isinstance(value, str) else float(value),
                            "contribution": round(float(contributions[j]), 4)})
        return reasons

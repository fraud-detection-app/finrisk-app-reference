from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# The model's inputs. The ORDER is a contract: finrisk-ml trains with it, the SageMaker endpoint
# (serving/contract.py) validates requests against it, and the monitor reads captured requests with it.
#
# `hour_of_day = (step - 1) % 24` replaced `step` (13b): a time index only ever grows, so a model that
# uses it learns "when in the calendar" and drifts the day production starts; the hour of the day is a
# stable, real signal (PaySim fraud runs around the clock while normal traffic is daytime). `step` stays
# in the data as metadata (the temporal split sorts by it) but is NOT a model feature.
CATEGORICAL_FEATURES = ["transaction_type"]
NUMERIC_FEATURES = [
    "amount",
    "log_amount",
    "origin_old_balance",
    "origin_new_balance",
    "destination_old_balance",
    "destination_new_balance",
    "origin_balance_delta",
    "destination_balance_delta",
    "amount_to_origin_balance",
    "hour_of_day",
    "tx_index_for_origin",
    "customer_tx_count_so_far",
    "customer_amount_so_far",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "is_fraud"


def artifact_metadata() -> dict[str, str]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }


@dataclass(frozen=True)
class SplitData:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_valid: pd.DataFrame
    y_valid: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    test_frame: pd.DataFrame


def load_transaction_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found at {path}")
    df = pd.read_parquet(path)
    # "step" is not a model feature any more, but the temporal split orders by it.
    missing = [col for col in FEATURE_COLUMNS + [TARGET, "row_id", "step"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df


def temporal_split(df: pd.DataFrame) -> SplitData:
    ordered = df.sort_values(["step", "row_id"]).reset_index(drop=True)
    n = len(ordered)
    if n < 20:
        raise ValueError("Need at least 20 rows to create train/valid/test splits.")

    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    train = ordered.iloc[:train_end]
    valid = ordered.iloc[train_end:valid_end]
    test = ordered.iloc[valid_end:]

    return SplitData(
        x_train=train[FEATURE_COLUMNS],
        y_train=train[TARGET],
        x_valid=valid[FEATURE_COLUMNS],
        y_valid=valid[TARGET],
        x_test=test[FEATURE_COLUMNS],
        y_test=test[TARGET],
        test_frame=test,
    )


def _preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ]
    )


def build_baseline_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    n_jobs=None,
                    random_state=42,
                ),
            ),
        ]
    )


def build_optimized_model(y_train: pd.Series) -> Pipeline:
    try:
        from xgboost import XGBClassifier

        negative = max(int((y_train == 0).sum()), 1)
        positive = max(int((y_train == 1).sum()), 1)
        model = XGBClassifier(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=negative / positive,
            random_state=42,
            n_jobs=2,
        )
        return Pipeline([("preprocess", _preprocessor(scale_numeric=False)), ("model", model)])
    except ImportError:
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=160,
            random_state=42,
        )
        return Pipeline([("preprocess", _preprocessor(scale_numeric=False)), ("model", model)])


def predict_proba_positive(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x)
        return 1 / (1 + np.exp(-raw))
    raise TypeError("Model does not expose predict_proba or decision_function.")


def choose_threshold(y_true: pd.Series, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5
    beta = 2.0
    f_beta = (1 + beta**2) * precision[:-1] * recall[:-1] / (
        beta**2 * precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.nanargmax(f_beta))])


def classification_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, object]:
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics: dict[str, object] = {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["roc_auc"] = None
    return metrics


def train_model(feature_path: Path, model_dir: Path, model_type: str = "baseline") -> tuple[Path, dict[str, object]]:
    df = load_transaction_features(feature_path)
    split = temporal_split(df)
    model = build_baseline_model() if model_type == "baseline" else build_optimized_model(split.y_train)

    model.fit(split.x_train, split.y_train)
    valid_score = predict_proba_positive(model, split.x_valid)
    threshold = choose_threshold(split.y_valid, valid_score)
    test_score = predict_proba_positive(model, split.x_test)
    metrics = classification_metrics(split.y_test, test_score, threshold)
    metrics["model_type"] = model_type
    metrics["train_rows"] = int(len(split.x_train))
    metrics["valid_rows"] = int(len(split.x_valid))
    metrics["test_rows"] = int(len(split.x_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / ("model_v1.pkl" if model_type == "baseline" else "model_v2.pkl")
    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "feature_columns": FEATURE_COLUMNS,
            "target": TARGET,
            "model_type": model_type,
            "metrics": metrics,
            **artifact_metadata(),
        },
        output_path,
    )
    # The pointer stores the file NAME, not an absolute path: an absolute path only exists on the
    # machine that trained the model, so the same models/ folder mounted into a container (or
    # copied to S3 and loaded on AWS) would point at nothing.
    joblib.dump({"model_path": output_path.name}, model_dir / "latest_model.joblib")
    return output_path, metrics


def train_random_forest_small(feature_path: Path, model_dir: Path) -> tuple[Path, dict[str, object]]:
    """Week 6 teaching helper: a second baseline without adding much complexity."""

    df = load_transaction_features(feature_path)
    split = temporal_split(df)
    model = Pipeline(
        steps=[
            ("preprocess", _preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=60,
                    max_depth=10,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=2,
                ),
            ),
        ]
    )
    model.fit(split.x_train, split.y_train)
    valid_score = predict_proba_positive(model, split.x_valid)
    threshold = choose_threshold(split.y_valid, valid_score)
    test_score = predict_proba_positive(model, split.x_test)
    metrics = classification_metrics(split.y_test, test_score, threshold)
    output_path = model_dir / "model_rf_teaching.pkl"
    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "feature_columns": FEATURE_COLUMNS,
            "target": TARGET,
            "model_type": "random_forest_teaching",
            "metrics": metrics,
            **artifact_metadata(),
        },
        output_path,
    )
    return output_path, metrics

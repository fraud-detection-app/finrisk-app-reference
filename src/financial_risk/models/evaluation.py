from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from financial_risk.models.explainability import global_feature_importance
from financial_risk.models.training import (
    FEATURE_COLUMNS,
    TARGET,
    classification_metrics,
    load_transaction_features,
    predict_proba_positive,
    temporal_split,
)


def repair_loaded_artifact(artifact: dict[str, object]) -> dict[str, object]:
    """Patch known sklearn cross-version pickle incompatibilities after loading.

    Model persistence with pickle/joblib is sensitive to sklearn versions. This project is used in
    teaching environments where students may train in one Python environment and run Streamlit in
    another, so we keep small, explicit compatibility repairs close to artifact loading.
    """

    model = artifact.get("model")
    repairs: list[str] = []

    if isinstance(model, Pipeline):
        estimators = [step for _, step in model.steps]
    else:
        estimators = [model]

    try:
        default_multi_class = LogisticRegression().multi_class
    except AttributeError:
        default_multi_class = None

    for estimator in estimators:
        if (
            default_multi_class is not None
            and isinstance(estimator, LogisticRegression)
            and not hasattr(estimator, "multi_class")
        ):
            estimator.multi_class = default_multi_class
            repairs.append("Added missing LogisticRegression.multi_class for sklearn compatibility.")

    if repairs:
        existing_repairs = artifact.get("load_repairs", [])
        if not isinstance(existing_repairs, list):
            existing_repairs = [str(existing_repairs)]
        artifact["load_repairs"] = [*existing_repairs, *repairs]
    return artifact


def load_latest_artifact(model_dir: Path) -> dict[str, object]:
    latest = model_dir / "latest_model.joblib"
    if latest.exists():
        pointer = Path(joblib.load(latest)["model_path"])
        # Resolve against model_dir. Older pointers hold an absolute path from the machine that
        # trained the model; inside a container that path does not exist, but the file next to
        # the pointer does -- so fall back to the same file name in this folder.
        target = pointer if pointer.is_absolute() and pointer.exists() else model_dir / pointer.name
        return repair_loaded_artifact(joblib.load(target))

    candidates = sorted(model_dir.glob("model_v*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No model artifacts found in {model_dir}")
    return repair_loaded_artifact(joblib.load(candidates[-1]))


def evaluate_latest_model(feature_path: Path, model_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    artifact = load_latest_artifact(model_dir)
    df = load_transaction_features(feature_path)
    split = temporal_split(df)
    score = predict_proba_positive(artifact["model"], split.x_test)
    metrics = classification_metrics(split.y_test, score, float(artifact["threshold"]))
    metrics["model_type"] = artifact.get("model_type", "unknown")

    errors = split.test_frame[
        ["row_id", "step", "transaction_type", "amount", "origin_customer", TARGET]
    ].copy()
    errors["risk_score"] = score
    errors["prediction"] = (errors["risk_score"] >= float(artifact["threshold"])).astype(int)
    errors["error_type"] = "correct"
    errors.loc[(errors[TARGET] == 1) & (errors["prediction"] == 0), "error_type"] = "false_negative"
    errors.loc[(errors[TARGET] == 0) & (errors["prediction"] == 1), "error_type"] = "false_positive"
    return metrics, errors.sort_values("risk_score", ascending=False)


def write_metrics_report(metrics: dict[str, object], output_path: Path) -> Path:
    cm = metrics["confusion_matrix"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""# Metrics Report

Model type: `{metrics.get("model_type", "unknown")}`

| Metric | Value |
| --- | ---: |
| Threshold | {metrics["threshold"]:.4f} |
| PR-AUC | {metrics["pr_auc"]:.4f} |
| ROC-AUC | {metrics["roc_auc"] if metrics["roc_auc"] is not None else "n/a"} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| F1 | {metrics["f1"]:.4f} |

## Confusion matrix

| | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | {cm["tn"]} | {cm["fp"]} |
| Actual 1 | {cm["fn"]} | {cm["tp"]} |

## Notes

Fraud detection is highly imbalanced, so PR-AUC, recall, precision, and threshold trade-offs are more useful than accuracy.
""",
        encoding="utf-8",
    )
    return output_path


def write_model_card(metrics: dict[str, object], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""# Model Card: PaySim Fraud Risk Model

## Intended use

Educational portfolio project for fraud-risk ranking and explainable analytics. It is not a production credit, lending, or investment decision system.

## Data

Synthetic PaySim financial transactions. The model learns transaction-level fraud patterns from `is_fraud`.

## Features

Uses transaction amount, type, balance deltas, the hour of the day (`hour_of_day`, not the ever-growing `step`), and no-future-leakage customer window features. Excludes `is_flagged_fraud` to avoid leakage from existing rules.

## Performance

```json
{json.dumps(metrics, indent=2)}
```

## Limitations

- Synthetic data does not fully represent real Australian banking behavior.
- Fraud labels are rare and threshold selection changes business trade-offs.
- The model should support analyst review rather than automate final financial decisions.

## Responsible AI boundary

All dashboard and LLM outputs must include a disclaimer that the project is educational and not financial advice.
""",
        encoding="utf-8",
    )
    return output_path


def _frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    display = df.copy()
    for col in display.columns:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if isinstance(value, float) else str(value))

    headers = list(display.columns)
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).replace("\n", " ") for col in headers) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def write_shap_summary(
    feature_path: Path,
    model_dir: Path,
    output_path: Path,
    sample_size: int = 1000,
    background_size: int = 200,
) -> Path:
    """Real SHAP-based global feature importance for the Week 8 explainability lesson (replaces
    the "SHAP concept only" gap -- the heuristic reasons in `inference.py` stay as a fast fallback,
    but the primary explanation path is now `models.explainability`)."""

    artifact = load_latest_artifact(model_dir)
    df = load_transaction_features(feature_path)
    split = temporal_split(df)
    background = split.x_train.sample(min(background_size, len(split.x_train)), random_state=42)
    pool = split.x_test if len(split.x_test) else split.x_train
    sample = pool.sample(min(sample_size, len(pool)), random_state=42)

    importance = global_feature_importance(artifact["model"], sample, background)

    lines = [
        "# SHAP Global Feature Importance",
        "",
        f"Model type: `{artifact.get('model_type', 'unknown')}`. Mean absolute SHAP value across "
        f"a {len(sample)}-row sample of the evaluation set -- higher means the feature moves the "
        "fraud-risk score more, in either direction.",
        "",
        "| Feature | Mean |SHAP| |",
        "| --- | ---: |",
    ]
    lines += [f"| `{row.feature}` | {row.mean_abs_shap:.4f} |" for row in importance.itertuples()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_error_analysis(errors: pd.DataFrame, output_path: Path, rows: int = 20) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    false_negatives = errors[errors["error_type"] == "false_negative"].head(rows)
    false_positives = errors[errors["error_type"] == "false_positive"].head(rows)
    output_path.write_text(
        "# Error Analysis\n\n"
        "## False negatives to inspect\n\n"
        + (_frame_to_markdown(false_negatives) if len(false_negatives) else "No false negatives found.")
        + "\n\n## False positives to inspect\n\n"
        + (_frame_to_markdown(false_positives) if len(false_positives) else "No false positives found.")
        + "\n",
        encoding="utf-8",
    )
    return output_path

from __future__ import annotations

from pathlib import Path

import pandas as pd

from financial_risk.llm.bedrock_client import generate_risk_insight
from financial_risk.models.evaluation import load_latest_artifact
from financial_risk.models.explainability import local_reasons
from financial_risk.models.training import FEATURE_COLUMNS, predict_proba_positive


def load_feature_row(feature_path: Path, row_id: int | None = None) -> pd.DataFrame:
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature table not found at {feature_path}")
    df = pd.read_parquet(feature_path)
    if row_id is None:
        return df.head(1)
    selected = df[df["row_id"] == row_id]
    if selected.empty:
        raise ValueError(f"No transaction found with row_id={row_id}")
    return selected.head(1)


def heuristic_reasons(row: pd.Series, risk_score: float) -> list[str]:
    reasons: list[str] = []
    if row["transaction_type"] in {"TRANSFER", "CASH_OUT"}:
        reasons.append(f"Transaction type `{row['transaction_type']}` is common in fraud patterns.")
    if float(row["amount"]) >= 200_000:
        reasons.append(f"Large transaction amount: {float(row['amount']):,.2f}.")
    if float(row["amount_to_origin_balance"]) >= 0.8:
        reasons.append("Transaction consumes a large share of the origin account balance.")
    if float(row["origin_new_balance"]) == 0 and float(row["amount"]) > 0:
        reasons.append("Origin account balance becomes zero after the transaction.")
    if risk_score < 0.35:
        reasons.append("Model score is below the medium-risk threshold.")
    return reasons[:4] or ["Model score is based on transaction type, amount, balance deltas, and customer history."]


def explain_reasons(
    model: object,
    feature_df: pd.DataFrame,
    row: pd.DataFrame,
    risk_score: float,
    explainer: object | None = None,
) -> tuple[list[str], str]:
    """Prefer real SHAP contributions for this transaction; fall back to the hand-crafted
    heuristics (`heuristic_reasons`) if SHAP fails (e.g. an unsupported model type or no
    background rows available), so the app never breaks even without SHAP support.

    A long-running service passes a prebuilt ``explainer`` (``make_explainer``): building one per
    call is most of the cost of an explanation."""

    if explainer is not None:
        try:
            reasons = local_reasons(model, row, None, explainer=explainer)
            if reasons:
                return reasons, "shap"
        except Exception:
            pass
        return heuristic_reasons(row.iloc[0], risk_score), "heuristic"
    background_pool = feature_df[feature_df["row_id"] != int(row.iloc[0]["row_id"])]
    if len(background_pool) >= 20:
        try:
            background = background_pool.sample(min(200, len(background_pool)), random_state=42)
            reasons = local_reasons(model, row, background)
            if reasons:
                return reasons, "shap"
        except Exception:
            pass
    return heuristic_reasons(row.iloc[0], risk_score), "heuristic"


def score_transaction(
    feature_path: Path,
    model_dir: Path,
    row_id: int | None = None,
    use_bedrock: bool = False,
) -> dict[str, object]:
    artifact = load_latest_artifact(model_dir)
    feature_df = pd.read_parquet(feature_path)
    row = load_feature_row(feature_path, row_id)
    score = float(predict_proba_positive(artifact["model"], row[FEATURE_COLUMNS])[0])
    threshold = float(artifact["threshold"])
    risk_level = "high" if score >= threshold else "low"
    row_dict = row.iloc[0].to_dict()
    reasons, explanation_method = explain_reasons(artifact["model"], feature_df, row, score)
    insight_context = {
        "row_id": int(row_dict["row_id"]),
        "transaction_type": row_dict["transaction_type"],
        "amount": float(row_dict["amount"]),
        "risk_score": score,
        "risk_level": risk_level,
        "threshold": threshold,
        "top_reasons": reasons,
    }
    return {
        "row": row_dict,
        "risk_score": score,
        "threshold": threshold,
        "risk_level": risk_level,
        "reasons": reasons,
        "explanation_method": explanation_method,
        "ai_insight": generate_risk_insight(insight_context, use_bedrock=use_bedrock),
    }

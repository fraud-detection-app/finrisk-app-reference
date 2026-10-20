from __future__ import annotations

import pandas as pd

from financial_risk.models.evaluation import repair_loaded_artifact
from financial_risk.models.training import FEATURE_COLUMNS, build_baseline_model, predict_proba_positive


def test_repair_loaded_artifact_adds_missing_logistic_regression_multi_class() -> None:
    x = pd.DataFrame(
        {
            "transaction_type": ["PAYMENT", "TRANSFER", "CASH_OUT", "PAYMENT"],
            "amount": [10.0, 500.0, 900.0, 20.0],
            "log_amount": [2.4, 6.2, 6.8, 3.0],
            "origin_old_balance": [100.0, 1000.0, 900.0, 80.0],
            "origin_new_balance": [90.0, 500.0, 0.0, 60.0],
            "destination_old_balance": [0.0, 100.0, 50.0, 0.0],
            "destination_new_balance": [10.0, 600.0, 950.0, 20.0],
            "origin_balance_delta": [10.0, 500.0, 900.0, 20.0],
            "destination_balance_delta": [10.0, 500.0, 900.0, 20.0],
            "amount_to_origin_balance": [0.1, 0.5, 1.0, 0.25],
            "hour_of_day": [0, 1, 2, 3],
            "tx_index_for_origin": [1, 1, 2, 1],
            "customer_tx_count_so_far": [0, 0, 1, 0],
            "customer_amount_so_far": [0.0, 0.0, 500.0, 0.0],
        }
    )
    y = [0, 1, 1, 0]
    model = build_baseline_model().fit(x[FEATURE_COLUMNS], y)
    # Newer sklearn (>=1.6) no longer sets `multi_class` at all, so there is nothing to remove;
    # only simulate the "loaded an older pickle" scenario when this sklearn still has it.
    had_attr = hasattr(model.named_steps["model"], "multi_class")
    if had_attr:
        delattr(model.named_steps["model"], "multi_class")

    artifact = repair_loaded_artifact({"model": model, "threshold": 0.5})

    if had_attr:
        assert hasattr(model.named_steps["model"], "multi_class")
        assert artifact["load_repairs"]
    assert predict_proba_positive(model, x[FEATURE_COLUMNS]).shape == (4,)

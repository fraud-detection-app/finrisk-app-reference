from __future__ import annotations

import pandas as pd

from financial_risk.data.paysim import clean_transactions


def test_clean_transactions_standardises_columns() -> None:
    raw = pd.DataFrame(
        {
            "step": [1, 2],
            "type": ["TRANSFER", "PAYMENT"],
            "amount": [100.0, 20.0],
            "nameOrig": ["C1", "C2"],
            "oldbalanceOrg": [100.0, 50.0],
            "newbalanceOrig": [0.0, 30.0],
            "nameDest": ["C3", "M1"],
            "oldbalanceDest": [0.0, 0.0],
            "newbalanceDest": [100.0, 0.0],
            "isFraud": [1, 0],
            "isFlaggedFraud": [0, 0],
        }
    )
    clean = clean_transactions(raw)
    assert "transaction_type" in clean.columns
    assert "origin_balance_delta" in clean.columns
    assert "row_id" in clean.columns
    assert clean["is_fraud"].tolist() == [1, 0]

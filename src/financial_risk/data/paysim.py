from __future__ import annotations

import math
import textwrap
import urllib.request
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from financial_risk.data.duckdb_utils import sql_string as _sql_string


RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

STANDARD_COLUMNS = [
    "step",
    "transaction_type",
    "amount",
    "origin_customer",
    "origin_old_balance",
    "origin_new_balance",
    "destination_customer",
    "destination_old_balance",
    "destination_new_balance",
    "is_fraud",
    "is_flagged_fraud",
]

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def download_raw_data(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    with urllib.request.urlopen(url, timeout=60) as response, output_path.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return output_path


def generate_demo_data(rows: int, output_path: Path, seed: int = 42) -> Path:
    """Create a PaySim-like sample so lessons can run before the full download."""

    rng = np.random.default_rng(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tx_type = rng.choice(
        TRANSACTION_TYPES,
        size=rows,
        p=np.array([0.22, 0.32, 0.05, 0.31, 0.10]),
    )
    amount = rng.lognormal(mean=8.2, sigma=1.1, size=rows).round(2)
    amount = np.clip(amount, 1, 3_000_000)

    origin_old = rng.lognormal(mean=9.0, sigma=1.4, size=rows).round(2)
    destination_old = rng.lognormal(mean=8.8, sigma=1.5, size=rows).round(2)
    cash_out_like = np.isin(tx_type, ["CASH_OUT", "TRANSFER"])

    base_prob = 0.0015
    large_amount_boost = np.clip(amount / 1_000_000, 0, 0.03)
    balance_drain = np.isclose(amount, origin_old, rtol=0.05, atol=100)
    fraud_prob = base_prob + cash_out_like * 0.004 + large_amount_boost + balance_drain * 0.03
    is_fraud = rng.binomial(1, np.clip(fraud_prob, 0, 0.35))

    origin_new = np.where(is_fraud == 1, 0, np.maximum(origin_old - amount, 0)).round(2)
    destination_new = np.where(
        np.isin(tx_type, ["CASH_OUT", "TRANSFER"]),
        destination_old + amount,
        destination_old,
    ).round(2)
    is_flagged = ((amount > 200_000) & (tx_type == "TRANSFER")).astype(int)

    df = pd.DataFrame(
        {
            "step": rng.integers(1, 744, size=rows),
            "type": tx_type,
            "amount": amount,
            "nameOrig": [f"C{value:010d}" for value in rng.integers(1, rows * 8, size=rows)],
            "oldbalanceOrg": origin_old,
            "newbalanceOrig": origin_new,
            "nameDest": [
                f"{prefix}{value:010d}"
                for prefix, value in zip(
                    rng.choice(["C", "M"], size=rows, p=[0.45, 0.55]),
                    rng.integers(1, rows * 12, size=rows),
                )
            ],
            "oldbalanceDest": destination_old,
            "newbalanceDest": destination_new,
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged,
        }
    )
    df.to_csv(output_path, index=False)
    return output_path


def read_raw_transactions(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run scripts/download_data.py --demo first."
        )
    return pd.read_csv(path, nrows=nrows)


def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing PaySim columns: {missing}")

    renamed = df[RAW_COLUMNS].rename(
        columns={
            "type": "transaction_type",
            "nameOrig": "origin_customer",
            "oldbalanceOrg": "origin_old_balance",
            "newbalanceOrig": "origin_new_balance",
            "nameDest": "destination_customer",
            "oldbalanceDest": "destination_old_balance",
            "newbalanceDest": "destination_new_balance",
            "isFraud": "is_fraud",
            "isFlaggedFraud": "is_flagged_fraud",
        }
    )
    return renamed


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    clean = standardize_schema(df).copy()
    clean["transaction_type"] = clean["transaction_type"].astype(str).str.strip().str.upper()
    clean = clean[clean["transaction_type"].isin(TRANSACTION_TYPES)]

    numeric_cols = [
        "step",
        "amount",
        "origin_old_balance",
        "origin_new_balance",
        "destination_old_balance",
        "destination_new_balance",
        "is_fraud",
        "is_flagged_fraud",
    ]
    for col in numeric_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(subset=["step", "amount", "transaction_type", "is_fraud"])
    clean = clean[clean["amount"] >= 0]
    clean = clean.drop_duplicates()

    clean["step"] = clean["step"].astype("int64")
    clean["is_fraud"] = clean["is_fraud"].astype("int8")
    clean["is_flagged_fraud"] = clean["is_flagged_fraud"].fillna(0).astype("int8")
    clean["origin_balance_delta"] = clean["origin_old_balance"] - clean["origin_new_balance"]
    clean["destination_balance_delta"] = (
        clean["destination_new_balance"] - clean["destination_old_balance"]
    )
    clean["amount_to_origin_balance"] = clean["amount"] / clean["origin_old_balance"].replace(0, np.nan)
    clean["amount_to_origin_balance"] = clean["amount_to_origin_balance"].replace(
        [np.inf, -np.inf], np.nan
    )
    clean["amount_to_origin_balance"] = clean["amount_to_origin_balance"].fillna(0).clip(0, 100)
    clean["row_id"] = np.arange(len(clean), dtype=np.int64)
    return clean.reset_index(drop=True)


def data_quality_summary(raw: pd.DataFrame, clean: pd.DataFrame) -> dict[str, object]:
    null_rates = clean.isna().mean().sort_values(ascending=False).head(10)
    return {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "removed_rows": int(len(raw) - len(clean)),
        "duplicate_rows_clean": int(clean.duplicated().sum()),
        "negative_amount_rows_clean": int((clean["amount"] < 0).sum()),
        "fraud_rate": float(clean["is_fraud"].mean()) if len(clean) else math.nan,
        "transaction_types": sorted(clean["transaction_type"].dropna().unique().tolist()),
        "null_rates_top_10": null_rates.to_dict(),
    }


def render_quality_report(summary: dict[str, object]) -> str:
    null_lines = "\n".join(
        f"- `{col}`: {rate:.2%}" for col, rate in summary["null_rates_top_10"].items()
    )
    return textwrap.dedent(
        f"""
        # Data Quality Report

        ## Dataset health

        - Raw rows: {summary["raw_rows"]:,}
        - Clean rows: {summary["clean_rows"]:,}
        - Removed rows: {summary["removed_rows"]:,}
        - Duplicate rows after cleaning: {summary["duplicate_rows_clean"]:,}
        - Negative amount rows after cleaning: {summary["negative_amount_rows_clean"]:,}
        - Fraud rate: {summary["fraud_rate"]:.4%}
        - Transaction types: {", ".join(summary["transaction_types"])}

        ## Top null rates

        {null_lines or "- No columns found."}

        ## Modelling guardrail

        `is_fraud` is the target. `is_flagged_fraud` is an existing rule indicator and should not
        be used as a model feature, because it leaks a pre-existing business rule into the model.
        """
    ).strip()


def ingest_to_parquet(raw_path: Path, bronze_path: Path, silver_path: Path, nrows: int | None = None) -> str:
    """Pandas-based ingestion. Fine for the demo dataset (thousands of rows); loads the whole
    file into memory, so it is not used for the full ~6.3M row PaySim file (see
    `ingest_to_parquet_duckdb`)."""

    raw = read_raw_transactions(raw_path, nrows=nrows)
    clean = clean_transactions(raw)
    bronze_path.parent.mkdir(parents=True, exist_ok=True)
    silver_path.parent.mkdir(parents=True, exist_ok=True)

    standardize_schema(raw).to_parquet(bronze_path, index=False)
    clean.to_parquet(silver_path, index=False)
    return render_quality_report(data_quality_summary(raw, clean))


def ingest_to_parquet_duckdb(raw_path: Path, bronze_path: Path, silver_path: Path) -> str:
    """Full-scale ingestion path (`DATA_MODE=full`).

    Standardises and cleans the entire PaySim CSV (~494MB / 6.3M rows) via DuckDB SQL running
    against the file on disk, so the pipeline never materialises the whole dataset as a single
    pandas DataFrame. This is the out-of-core / "real data engineering" counterpart to
    `ingest_to_parquet`, which stays as the fast in-memory path for demo-scale data.
    """

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Run scripts/download_data.py (no --demo) first."
        )
    bronze_path.parent.mkdir(parents=True, exist_ok=True)
    silver_path.parent.mkdir(parents=True, exist_ok=True)

    valid_types_sql = ", ".join(f"'{t}'" for t in TRANSACTION_TYPES)
    con = duckdb.connect()
    con.execute(
        f"CREATE OR REPLACE VIEW raw_tx AS "
        f"SELECT * FROM read_csv_auto({_sql_string(raw_path)}, header=True, sample_size=200000)"
    )

    con.execute(
        f"""
        COPY (
            SELECT
                step,
                type AS transaction_type,
                amount,
                nameOrig AS origin_customer,
                oldbalanceOrg AS origin_old_balance,
                newbalanceOrig AS origin_new_balance,
                nameDest AS destination_customer,
                oldbalanceDest AS destination_old_balance,
                newbalanceDest AS destination_new_balance,
                isFraud AS is_fraud,
                isFlaggedFraud AS is_flagged_fraud
            FROM raw_tx
        ) TO {_sql_string(bronze_path)} (FORMAT PARQUET)
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT
                * EXCLUDE (rn),
                row_number() OVER (ORDER BY step) - 1 AS row_id
            FROM (
                SELECT DISTINCT
                    step,
                    upper(trim(transaction_type)) AS transaction_type,
                    amount,
                    origin_customer,
                    origin_old_balance,
                    origin_new_balance,
                    destination_customer,
                    destination_old_balance,
                    destination_new_balance,
                    is_fraud,
                    is_flagged_fraud,
                    origin_old_balance - origin_new_balance AS origin_balance_delta,
                    destination_new_balance - destination_old_balance AS destination_balance_delta,
                    least(greatest(
                        coalesce(amount / nullif(origin_old_balance, 0), 0), 0
                    ), 100) AS amount_to_origin_balance,
                    1 AS rn
                FROM read_parquet({_sql_string(bronze_path)})
                WHERE amount >= 0
                  AND step IS NOT NULL
                  AND amount IS NOT NULL
                  AND is_fraud IS NOT NULL
                  AND upper(trim(transaction_type)) IN ({valid_types_sql})
            )
        ) TO {_sql_string(silver_path)} (FORMAT PARQUET)
        """
    )

    raw_rows, clean_rows, fraud_rate = con.execute(
        f"""
        SELECT
            (SELECT count(*) FROM raw_tx) AS raw_rows,
            (SELECT count(*) FROM read_parquet({_sql_string(silver_path)})) AS clean_rows,
            (SELECT avg(is_fraud) FROM read_parquet({_sql_string(silver_path)})) AS fraud_rate
        """
    ).fetchone()
    con.close()

    return render_quality_report(
        {
            "raw_rows": int(raw_rows),
            "clean_rows": int(clean_rows),
            "removed_rows": int(raw_rows - clean_rows),
            "duplicate_rows_clean": 0,
            "negative_amount_rows_clean": 0,
            "fraud_rate": float(fraud_rate) if fraud_rate is not None else math.nan,
            "transaction_types": TRANSACTION_TYPES,
            "null_rates_top_10": {},
        }
    )

from __future__ import annotations

from pathlib import Path

import duckdb

from financial_risk.data.duckdb_utils import sql_string as _sql_string


TRANSACTION_FEATURE_PATH = "transaction_features.parquet"
CUSTOMER_FEATURE_PATH = "customer_features.parquet"
PARTITIONED_TRANSACTION_DIR = "transaction_features_partitioned"
STEP_BUCKET_HOURS = 168  # one PaySim "week" (24h steps x 7) -- keeps partition count sane
# regardless of demo vs full row count, since PaySim's step range (1-744) doesn't grow with rows.


_TRANSACTION_FEATURE_SELECT = """
    SELECT
        row_id,
        step,
        transaction_type,
        amount,
        log(1 + amount) AS log_amount,
        origin_customer,
        destination_customer,
        origin_old_balance,
        origin_new_balance,
        destination_old_balance,
        destination_new_balance,
        origin_balance_delta,
        destination_balance_delta,
        amount_to_origin_balance,
        ((step - 1) % 24) AS hour_of_day,
        row_number() OVER (
            PARTITION BY origin_customer
            ORDER BY step, row_id
        ) AS tx_index_for_origin,
        count(*) OVER (
            PARTITION BY origin_customer
            ORDER BY step, row_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS customer_tx_count_so_far,
        sum(amount) OVER (
            PARTITION BY origin_customer
            ORDER BY step, row_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS customer_amount_so_far,
        is_fraud
    FROM tx
"""


def build_feature_tables(
    silver_path: Path,
    gold_dir: Path,
    partition_by_step: bool = False,
) -> tuple[Path, Path]:
    if not silver_path.exists():
        raise FileNotFoundError(f"Silver data not found at {silver_path}")

    gold_dir.mkdir(parents=True, exist_ok=True)
    transaction_output = gold_dir / TRANSACTION_FEATURE_PATH
    customer_output = gold_dir / CUSTOMER_FEATURE_PATH

    con = duckdb.connect()
    con.execute(
        f"CREATE OR REPLACE VIEW tx AS SELECT * FROM read_parquet({_sql_string(silver_path)})"
    )

    con.execute(
        f"""
        COPY (
            {_TRANSACTION_FEATURE_SELECT}
        ) TO {_sql_string(transaction_output)} (FORMAT PARQUET)
        """,
    )

    if partition_by_step:
        partition_dir = gold_dir / PARTITIONED_TRANSACTION_DIR
        con.execute(
            f"""
            COPY (
                SELECT *, (step // {STEP_BUCKET_HOURS}) AS step_bucket
                FROM ({_TRANSACTION_FEATURE_SELECT})
            ) TO {_sql_string(partition_dir)}
              (FORMAT PARQUET, PARTITION_BY (step_bucket), OVERWRITE_OR_IGNORE true)
            """,
        )

    con.execute(
        f"""
        COPY (
            SELECT
                origin_customer AS customer_id,
                count(*) AS tx_count,
                sum(amount) AS total_amount,
                avg(amount) AS avg_amount,
                max(amount) AS max_amount,
                count(DISTINCT destination_customer) AS unique_destinations,
                sum(CASE WHEN transaction_type = 'CASH_OUT' THEN 1 ELSE 0 END) AS cash_out_count,
                sum(CASE WHEN transaction_type = 'TRANSFER' THEN 1 ELSE 0 END) AS transfer_count,
                min(step) AS first_step,
                max(step) AS last_step,
                max(is_fraud) AS ever_fraud_target
            FROM tx
            GROUP BY origin_customer
        ) TO {_sql_string(customer_output)} (FORMAT PARQUET)
        """,
    )

    con.close()
    return transaction_output, customer_output


def write_feature_dictionary(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        """# Feature Dictionary

## Transaction-level features

| Feature | Meaning | Notes |
| --- | --- | --- |
| `step` | Hour-like time step in PaySim (1 = the first hour) | Metadata: used for the temporal split and as the event clock. **Not a model feature.** |
| `hour_of_day` | `(step - 1) % 24` | Model feature. Replaces `step`: a time index only grows and leaks the calendar into the model; the hour of the day is a stable signal. |
| `transaction_type` | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER | One-hot encoded for ML. |
| `amount` | Transaction amount | Main transaction magnitude signal. |
| `log_amount` | `log(1 + amount)` | Reduces heavy-tail effect. |
| `origin_old_balance` / `origin_new_balance` | Sender balance before/after | Numeric model features. |
| `destination_old_balance` / `destination_new_balance` | Receiver balance before/after | Numeric model features. |
| `origin_balance_delta` | Sender balance decrease | Useful for balance-drain behavior. |
| `destination_balance_delta` | Receiver balance increase | Useful for transfer/cash-out behavior. |
| `amount_to_origin_balance` | Amount divided by sender old balance | Clipped to avoid infinite values. |
| `tx_index_for_origin` | Transaction sequence number per sender | Window-function feature. |
| `customer_tx_count_so_far` | Sender transaction count up to current row | No future leakage. |
| `customer_amount_so_far` | Sender cumulative amount up to current row | No future leakage. |

## Excluded from model features

- `is_fraud`: target label.
- `is_flagged_fraud`: existing rule flag, excluded to avoid rule leakage.
- Customer IDs: useful for joins/demo, not for baseline modelling.
""",
        encoding="utf-8",
    )
    return output_path

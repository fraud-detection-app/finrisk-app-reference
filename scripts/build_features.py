from __future__ import annotations

import _bootstrap  # noqa: F401

from financial_risk.config import ensure_project_dirs, get_settings
from financial_risk.features.sql_features import build_feature_tables, write_feature_dictionary


def main() -> None:
    settings = get_settings()
    ensure_project_dirs(settings)
    transaction_path, customer_path = build_feature_tables(
        settings.silver_path, settings.gold_path, partition_by_step=(settings.data_mode != "demo")
    )
    dictionary_path = write_feature_dictionary(settings.report_dir / "feature_dictionary.md")
    print(f"Wrote transaction features: {transaction_path}")
    print(f"Wrote customer features: {customer_path}")
    print(f"Wrote feature dictionary: {dictionary_path}")


if __name__ == "__main__":
    main()

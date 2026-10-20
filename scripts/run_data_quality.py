from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from financial_risk.config import ensure_project_dirs, get_settings
from financial_risk.data.paysim import ingest_to_parquet, ingest_to_parquet_duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create bronze/silver parquet and data quality report.")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for class demos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    ensure_project_dirs(settings)

    if settings.data_mode != "demo":
        report = ingest_to_parquet_duckdb(settings.raw_data_path, settings.bronze_path, settings.silver_path)
    else:
        report = ingest_to_parquet(
            settings.raw_data_path, settings.bronze_path, settings.silver_path, args.nrows
        )
    output = settings.report_dir / "data_quality_report.md"
    output.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote silver data: {settings.silver_path}")
    print(f"Wrote data quality report: {output}")


if __name__ == "__main__":
    main()

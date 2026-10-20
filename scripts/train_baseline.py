from __future__ import annotations

import _bootstrap  # noqa: F401

from financial_risk.config import get_settings
from financial_risk.models.training import train_model


def main() -> None:
    settings = get_settings()
    model_path, metrics = train_model(
        settings.gold_path / "transaction_features.parquet",
        settings.model_dir,
        model_type="baseline",
    )
    report_path = settings.report_dir / "metrics_report_v1.md"
    report_path.write_text(
        "# Baseline Metrics\n\n"
        + "\n".join(f"- `{key}`: {value}" for key, value in metrics.items())
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote baseline model: {model_path}")
    print(f"Wrote baseline report: {report_path}")


if __name__ == "__main__":
    main()

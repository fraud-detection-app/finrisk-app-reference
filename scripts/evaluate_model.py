from __future__ import annotations

import _bootstrap  # noqa: F401

from financial_risk.config import ensure_project_dirs, get_settings
from financial_risk.models.evaluation import (
    evaluate_latest_model,
    write_error_analysis,
    write_metrics_report,
    write_model_card,
)


def main() -> None:
    settings = get_settings()
    ensure_project_dirs(settings)
    feature_path = settings.gold_path / "transaction_features.parquet"
    metrics, errors = evaluate_latest_model(feature_path, settings.model_dir)
    metrics_path = write_metrics_report(metrics, settings.report_dir / "final_metrics_report.md")
    model_card_path = write_model_card(metrics, settings.report_dir / "model_card_v1.md")
    error_path = write_error_analysis(errors, settings.report_dir / "error_analysis.md")
    errors.head(200).to_csv(settings.report_dir / "scored_test_sample.csv", index=False)

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote model card: {model_card_path}")
    print(f"Wrote error analysis: {error_path}")


if __name__ == "__main__":
    main()

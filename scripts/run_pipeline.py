from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from financial_risk.config import ensure_project_dirs, get_settings
from financial_risk.data.paysim import download_raw_data, generate_demo_data, ingest_to_parquet, ingest_to_parquet_duckdb
from financial_risk.features.sql_features import build_feature_tables, write_feature_dictionary
from financial_risk.models.evaluation import (
    evaluate_latest_model,
    write_error_analysis,
    write_metrics_report,
    write_model_card,
    write_shap_summary,
)
from financial_risk.models.training import train_model
from financial_risk.rag.index import build_index, ensure_seed_docs, knowledge_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local data -> model -> report -> RAG pipeline.")
    parser.add_argument("--demo-rows", type=int, default=5000, help="Generate demo data if raw CSV is missing.")
    parser.add_argument("--model", choices=["baseline", "optimized"], default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    ensure_project_dirs(settings)

    if not settings.raw_data_path.exists():
        if settings.data_mode != "demo":
            download_raw_data(settings.dataset_url, settings.raw_data_path)
            print(f"Downloaded full PaySim dataset: {settings.raw_data_path}")
        else:
            generate_demo_data(args.demo_rows, settings.raw_data_path)
            print(f"Generated demo data because raw CSV was missing: {settings.raw_data_path}")

    if settings.data_mode != "demo":
        report = ingest_to_parquet_duckdb(settings.raw_data_path, settings.bronze_path, settings.silver_path)
    else:
        report = ingest_to_parquet(settings.raw_data_path, settings.bronze_path, settings.silver_path)
    (settings.report_dir / "data_quality_report.md").write_text(report + "\n", encoding="utf-8")

    build_feature_tables(settings.silver_path, settings.gold_path, partition_by_step=(settings.data_mode != "demo"))
    write_feature_dictionary(settings.report_dir / "feature_dictionary.md")

    model_path, train_metrics = train_model(
        settings.gold_path / "transaction_features.parquet",
        settings.model_dir,
        model_type=args.model,
    )
    metrics, errors = evaluate_latest_model(
        settings.gold_path / "transaction_features.parquet",
        settings.model_dir,
    )
    write_metrics_report(metrics, settings.report_dir / "final_metrics_report.md")
    write_model_card(metrics, settings.report_dir / "model_card_v1.md")
    write_error_analysis(errors, settings.report_dir / "error_analysis.md")
    write_shap_summary(
        settings.gold_path / "transaction_features.parquet",
        settings.model_dir,
        settings.report_dir / "shap_summary.md",
    )
    ensure_seed_docs(settings.rag_docs_dir)
    build_index(knowledge_dirs(settings), settings.vector_index_dir, seed_dir=settings.rag_docs_dir)

    log_path = settings.root / "experiment_log.csv"
    log_path.write_text(
        "model_path,model_type,threshold,pr_auc,precision,recall,f1\n"
        f"{model_path},{args.model},{train_metrics['threshold']:.6f},{metrics['pr_auc']:.6f},"
        f"{metrics['precision']:.6f},{metrics['recall']:.6f},{metrics['f1']:.6f}\n",
        encoding="utf-8",
    )
    print("Pipeline completed.")
    print(f"Model: {model_path}")
    print(f"Experiment log: {log_path}")


if __name__ == "__main__":
    main()

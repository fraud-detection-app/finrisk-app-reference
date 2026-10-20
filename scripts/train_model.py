from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from financial_risk.config import ensure_project_dirs, get_settings
from financial_risk.models.training import train_model, train_random_forest_small


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fraud detection model.")
    parser.add_argument(
        "--model",
        choices=["baseline", "optimized", "rf-teaching"],
        default="baseline",
        help="Model family to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    ensure_project_dirs(settings)
    feature_path = settings.gold_path / "transaction_features.parquet"

    if args.model == "rf-teaching":
        model_path, metrics = train_random_forest_small(feature_path, settings.model_dir)
    else:
        model_path, metrics = train_model(feature_path, settings.model_dir, model_type=args.model)

    print(f"Wrote model: {model_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

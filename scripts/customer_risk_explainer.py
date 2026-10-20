from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from financial_risk.config import get_settings
from financial_risk.models.inference import score_transaction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain transaction risk by row_id.")
    parser.add_argument("--row-id", type=int, default=None)
    parser.add_argument("--use-bedrock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    result = score_transaction(
        settings.gold_path / "transaction_features.parquet",
        settings.model_dir,
        row_id=args.row_id,
        use_bedrock=args.use_bedrock,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

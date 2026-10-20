"""Let `pytest tests/test_x.py` find this repo's package without installing it, and share two fixtures."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def demo_gold(tmp_path_factory) -> Path:
    """A real gold feature table from 4,000 synthetic transactions, made once per test run in a temp folder.

    The tests that swap models in and out need a table to score from. Making one here (the same three steps as
    scripts/run_pipeline.py) means they run on a fresh clone and in CI without any generated file on disk."""
    from financial_risk.data.paysim import generate_demo_data, ingest_to_parquet
    from financial_risk.features.sql_features import build_feature_tables

    base = tmp_path_factory.mktemp("demo-gold")
    raw, bronze, silver, gold = (base / n for n in ("raw.csv", "bronze", "silver", "gold"))
    generate_demo_data(4000, raw)
    ingest_to_parquet(raw, bronze, silver)
    build_feature_tables(silver, gold)
    return gold / "transaction_features.parquet"

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import _bootstrap  # noqa: F401

from financial_risk.config import ensure_project_dirs, get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate project structure and dependency readiness.")
    parser.add_argument("--strict", action="store_true", help="Fail if runtime dependencies are missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    ensure_project_dirs(settings)

    dependencies = [
        "numpy",
        "pandas",
        "pyarrow",
        "duckdb",
        "sklearn",
        "joblib",
        "boto3",
    ]
    missing_deps = [name for name in dependencies if importlib.util.find_spec(name) is None]

    expected_dirs = [
        settings.root / "data/raw",
        settings.root / "data/bronze",
        settings.root / "data/silver",
        settings.root / "data/gold",
        settings.root / "config/environments",
        settings.root / "models",
        settings.root / "reports",
    ]
    missing = [path for path in expected_dirs if not Path(path).exists()]
    if missing:
        raise SystemExit(f"Missing directories: {missing}")

    if missing_deps:
        print("Project structure check passed.")
        print("Missing runtime dependencies:")
        for name in missing_deps:
            print(f"- {name}")
        print("Install with: pip install -r requirements.txt")
        if args.strict:
            raise SystemExit(1)
    else:
        print("Smoke test passed.")
    print(f"Project root: {settings.root}")
    print(f"Default Bedrock model: {settings.bedrock_model_id}")


if __name__ == "__main__":
    main()

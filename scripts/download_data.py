from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from financial_risk.config import ensure_project_dirs, get_settings
from financial_risk.data.paysim import download_raw_data, generate_demo_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or generate PaySim data.")
    parser.add_argument("--demo", action="store_true", help="Generate a small PaySim-like demo dataset.")
    parser.add_argument("--rows", type=int, default=5000, help="Rows for --demo mode.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    ensure_project_dirs(settings)
    output = settings.root / args.output if args.output else settings.raw_data_path

    if args.demo:
        path = generate_demo_data(args.rows, output)
        print(f"Generated demo data: {path} ({args.rows:,} rows)")
        return

    path = download_raw_data(settings.dataset_url, output)
    print(f"Downloaded PaySim data: {path}")


if __name__ == "__main__":
    main()

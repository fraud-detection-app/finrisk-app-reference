from __future__ import annotations

import _bootstrap  # noqa: F401

from financial_risk.config import get_settings


def main() -> None:
    settings = get_settings()
    print("PaySim source URL:")
    print(settings.dataset_url)
    print("\nExpected local raw path:")
    print(settings.raw_data_path)
    print("\nClass note:")
    print("Week 1 validates the data source and architecture. It does not require AWS hands-on work.")


if __name__ == "__main__":
    main()

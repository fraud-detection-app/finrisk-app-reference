from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from financial_risk.config import get_settings
from financial_risk.rag.index import answer_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the local RAG project assistant.")
    parser.add_argument("question", nargs="?", default="Why exclude is_flagged_fraud?")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    response = answer_question(args.question, settings.vector_index_dir)
    print(response["answer"])
    print("Citations:", response["citations"])
    print(response["disclaimer"])


if __name__ == "__main__":
    main()

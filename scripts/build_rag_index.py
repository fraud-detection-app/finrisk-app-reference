from __future__ import annotations

import _bootstrap  # noqa: F401

from financial_risk.config import ensure_project_dirs, get_settings
from financial_risk.rag.index import build_index, ensure_seed_docs, knowledge_dirs


def main() -> None:
    settings = get_settings()
    ensure_project_dirs(settings)
    ensure_seed_docs(settings.rag_docs_dir)
    dirs = knowledge_dirs(settings)
    index_dir, count = build_index(dirs, settings.vector_index_dir, seed_dir=settings.rag_docs_dir)
    print(f"Wrote local RAG index: {index_dir}")
    for directory in dirs:
        print(f"  {directory.name:<12} {len(list(directory.glob('*.md')))} markdown files")
    print(f"Indexed chunks: {count}")


if __name__ == "__main__":
    main()

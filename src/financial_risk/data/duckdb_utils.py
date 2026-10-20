from __future__ import annotations

from pathlib import Path


def sql_string(path: Path) -> str:
    """Return a safely quoted SQL string literal for a DuckDB file/dir path."""

    escaped = str(path).replace("'", "''")
    return f"'{escaped}'"

"""A tiny vendor-style transactions API for teaching API ingestion (week 4).

Plays the role of a third-party payments vendor:

  - serves PaySim-shaped transactions in the RAW camelCase schema (``nameOrig``,
    ``oldbalanceOrg``, ``isFraud`` ...) so the client must standardise at the boundary,
    exactly like the CSV path;
  - paginates (``limit``/``offset``) instead of handing over one big file — the defining
    difference between file and API ingestion;
  - can be made deliberately unreliable via ``fail_rate`` (returns HTTP 503 with that
    probability), so the lab can demonstrate retry-with-backoff against a real failure.

Run the server (the demo does this via subprocess):

    uvicorn financial_risk.data.transactions_api:app --port 8765

The client half (`fetch_transactions_from_api`) uses only the standard library, retries
503s with linear backoff, and returns the assembled raw DataFrame plus fetch statistics.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from financial_risk.data.paysim import generate_demo_data

API_ROWS = 2_000
API_SEED = 7
MAX_PAGE_SIZE = 1_000

app = FastAPI(
    title="Vendor Transactions API (teaching demo)",
    description="Serves synthetic PaySim-shaped transactions with pagination and an "
    "optional failure rate. Educational use only.",
)

_dataset: pd.DataFrame | None = None


def _load_dataset() -> pd.DataFrame:
    """Generate the vendor's dataset once, in memory (raw camelCase schema)."""
    global _dataset
    if _dataset is None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory(prefix="vendor_api_") as tmp:
            csv_path = Path(tmp) / "vendor.csv"
            generate_demo_data(API_ROWS, csv_path, seed=API_SEED)
            _dataset = pd.read_csv(csv_path)
    return _dataset


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "rows": len(_load_dataset()), "schema": "paysim-raw-camelCase"}


@app.get("/transactions")
def transactions(
    limit: int = Query(default=500, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    fail_rate: float = Query(default=0.0, ge=0.0, le=0.9),
) -> dict[str, object]:
    if fail_rate and random.random() < fail_rate:
        raise HTTPException(status_code=503, detail="vendor hiccup — try again")
    df = _load_dataset()
    page = df.iloc[offset : offset + limit]
    return {
        "total": len(df),
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "records": json.loads(page.to_json(orient="records")),
    }


# --------------------------------------------------------------------------- client side
def fetch_transactions_from_api(
    base_url: str,
    page_size: int = 500,
    fail_rate: float = 0.0,
    max_retries: int = 5,
    backoff_seconds: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Page through the vendor API and assemble the full raw DataFrame.

    Retries HTTP 503 responses (the vendor's simulated hiccups) up to ``max_retries``
    times per page with linear backoff. Returns ``(raw_df, stats)`` where stats counts
    pages fetched and retries survived. Raises after ``max_retries`` consecutive
    failures on one page — a real client must eventually give up loudly, not hang.
    """

    records: list[dict[str, object]] = []
    stats = {"pages": 0, "retries": 0}
    offset, total = 0, None

    while total is None or offset < total:
        url = f"{base_url}/transactions?limit={page_size}&offset={offset}&fail_rate={fail_rate}"
        payload = None
        for attempt in range(1, max_retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    payload = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 503 or attempt == max_retries:
                    raise
                stats["retries"] += 1
                time.sleep(backoff_seconds * attempt)
        assert payload is not None
        records.extend(payload["records"])
        total = int(payload["total"])
        offset += int(payload["returned"])
        stats["pages"] += 1

    return pd.DataFrame(records), stats


def wait_for_api(base_url: str, timeout_seconds: float = 15.0) -> dict[str, object]:
    """Poll /health until the server answers (used right after starting it)."""
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                return json.loads(response.read())
        except Exception as exc:  # noqa: BLE001 — server still starting
            last_error = exc
            time.sleep(0.25)
    raise TimeoutError(f"API at {base_url} did not become healthy: {last_error}")

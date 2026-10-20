"""Start the risk service.

    python scripts/run_api.py              # http://127.0.0.1:8000
    python scripts/run_api.py --reload     # restart on every code change, for development

Equivalent to `uvicorn api.main:app`, but it resolves the project root first so the command
works from any directory, and it prints the URLs a student actually needs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from financial_risk.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the Financial Risk Assistant API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-restart on code changes.")
    args = parser.parse_args()

    import uvicorn

    print(f"  UI      http://{args.host}:{args.port}/")
    print(f"  Docs    http://{args.host}:{args.port}/docs")
    print(f"  Ready   http://{args.host}:{args.port}/ready")
    model = {"deepseek": settings.deepseek_model, "bedrock": settings.bedrock_model_id}.get(settings.llm_provider)
    key = "" if settings.llm_provider != "deepseek" else (" · key set" if settings.deepseek_api_key else " · KEY EMPTY -> offline")
    print(f"  LLM     {settings.llm_provider}" + (f" · model {model}" if model else " -> offline answers") + key, flush=True)
    # access_log=False: the app logs every request itself, with its id, timing and what it did.
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload, app_dir=str(ROOT),
                access_log=False)


if __name__ == "__main__":
    main()

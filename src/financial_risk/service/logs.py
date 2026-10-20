"""One readable log for the whole service, where every line says which request it belongs to.

    14:52:07 INFO  3f2a9c1b2c7d api     -> POST /api/agent
    14:52:07 INFO  3f2a9c1b2c7d agent   asked: "pull the transaction with id 4503"
    14:52:08 INFO  3f2a9c1b2c7d llm     deepseek · deepseek-flash · route · 641 ms · 310→24 tokens
    14:52:08 INFO  3f2a9c1b2c7d agent   route: lookup_transaction {'row_id': 4503} via llm (deepseek-flash)
    14:52:08 INFO  3f2a9c1b2c7d api     <- 200 POST /api/agent in 702 ms

The modules that do the work (llm, rag, agent) just log with the standard library. They never
import this file -- which would be an upward import -- so the request id is attached here, by a
logging filter reading a context variable that the API's middleware sets. This module imports
nothing from FastAPI either: the notebook can use it too.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")

COMPONENTS = {
    "risk_api": "api",
    "financial_risk.llm": "llm",
    "financial_risk.rag": "rag",
    "financial_risk.agent": "agent",
    "financial_risk.service": "service",
}

# Libraries that are chatty at INFO/DEBUG and say nothing a student needs.
QUIET = ("httpx", "httpcore", "asyncio", "urllib3", "botocore", "boto3", "matplotlib", "PIL",
         "numba", "shap", "multipart", "watchfiles", "uvicorn.access")


class _RequestContext(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.rid = REQUEST_ID.get()
        record.component = COMPONENTS.get(record.name, record.name.rsplit(".", 1)[-1])
        return True


def configure_logging(level: str = "INFO") -> None:
    """Install the service's log format on the root logger. Safe to call more than once."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_risk_service", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler._risk_service = True  # type: ignore[attr-defined]
    handler.addFilter(_RequestContext())
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(rid)-12s %(component)-7s %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for name in QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)

"""Hide AWS account ids in text that may leave the service. /ready and error bodies are public."""

from __future__ import annotations

import re

_ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")


def mask(text: str) -> str:
    """Hide a 12-digit AWS account id (it is part of our bucket names and every ARN)."""
    return _ACCOUNT_ID.sub("<account-id>", text)

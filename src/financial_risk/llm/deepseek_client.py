"""DeepSeek chat client -- the course's default cloud LLM.

Why DeepSeek rather than Bedrock for the class: a student can hold an API key in about five
minutes without an AWS account, an IAM role or a region decision. The request is plain HTTPS,
so this module needs no new dependency -- ``httpx`` already ships with FastAPI.

The API is **OpenAI-compatible**, which is the genuinely useful fact: the same request body
works against OpenAI, DeepSeek, Together, Groq and most hosted models. Swapping providers is a
base URL and a model id, not a rewrite. That is why the course teaches the raw HTTP call once
rather than hiding it behind a vendor SDK.

Set ``DEEPSEEK_API_KEY`` in ``.env`` (see ``.env.example``). With no key the client never calls
out and ``financial_risk.llm.client`` falls back to the deterministic template, so the whole
class runs offline and free.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from financial_risk.config import get_settings


# The ids the API accepts today. Older course notes and most blog posts still show
# "deepseek-chat", which no longer exists -- hence the explicit error message below.
CURRENT_MODELS = ("deepseek-flash", "deepseek-v4-pro")


class DeepSeekError(RuntimeError):
    """Any failure talking to DeepSeek. Callers fall back rather than propagating."""


def is_configured() -> bool:
    """True when an API key is present. The API's /ready endpoint reports this."""
    return bool(get_settings().deepseek_api_key)


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 30.0,
) -> str:
    """One chat completion. Returns the assistant's text.

    ``messages`` is week 8's role list exactly: [{"role": "system", ...}, {"role": "user", ...}].
    """
    return chat_detailed(messages, model, temperature, max_tokens, timeout)["text"]


def chat_detailed(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Like :func:`chat`, plus the evidence: which model DeepSeek says answered, token usage
    and latency. ``model`` in the result comes from the RESPONSE, not from our settings -- it is
    the only proof of what actually ran."""
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise DeepSeekError(
            "DEEPSEEK_API_KEY is empty. Paste your key into .env (see .env.example), "
            "then set LLM_PROVIDER=deepseek."
        )

    url = f"{settings.deepseek_base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model or settings.deepseek_model,
        "messages": messages,
        "temperature": settings.deepseek_temperature if temperature is None else temperature,
        "max_tokens": settings.deepseek_max_tokens if max_tokens is None else max_tokens,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {settings.deepseek_api_key}",
        "Content-Type": "application/json",
    }

    started = time.perf_counter()
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    except httpx.RequestError as exc:
        raise DeepSeekError(f"Could not reach DeepSeek: {exc}") from exc

    if response.status_code == 401:
        raise DeepSeekError("401 Unauthorized -- the API key is wrong, revoked, or has a stray space.")
    if response.status_code == 402:
        raise DeepSeekError("402 Payment Required -- the account balance is zero; top up to enable calls.")
    if response.status_code == 429:
        raise DeepSeekError("429 Rate limited -- slow down or retry with backoff.")
    if response.status_code == 400 and "model" in response.text.lower():
        raise DeepSeekError(
            f"400 -- DeepSeek does not recognise model {payload['model']!r}. Current ids are "
            f"{' or '.join(CURRENT_MODELS)}; the deepseek-chat / deepseek-reasoner names used in "
            "older tutorials have been retired. Set DEEPSEEK_MODEL in .env."
        )
    if response.status_code >= 400:
        raise DeepSeekError(f"DeepSeek returned {response.status_code}: {response.text[:200]}")

    try:
        body = response.json()
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise DeepSeekError(f"Unexpected response shape: {response.text[:200]}") from exc
    usage = body.get("usage") or {}
    return {
        "text": text,
        "model": body.get("model") or payload["model"],
        "requested_model": payload["model"],
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def check_key(timeout: float = 4.0) -> dict[str, Any]:
    """Ask DeepSeek whether the key works and has balance, without spending a token.

    ``GET /user/balance`` answers 401 for a bad key and ``is_available: false`` for an empty
    account -- the two reasons a first call fails. Never raises: this is a startup diagnostic."""
    settings = get_settings()
    if not settings.deepseek_api_key:
        return {"verified": False, "detail": "no key"}
    url = f"{settings.deepseek_base_url.rstrip('/')}/user/balance"
    try:
        response = httpx.get(url, headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
                             timeout=timeout)
    except httpx.RequestError as exc:
        return {"verified": False, "detail": f"unreachable ({exc.__class__.__name__})"}
    if response.status_code == 401:
        return {"verified": False, "detail": "401: key rejected"}
    if response.status_code >= 400:
        return {"verified": False, "detail": f"HTTP {response.status_code}"}
    try:
        available = bool(response.json().get("is_available"))
    except ValueError:
        return {"verified": False, "detail": "unexpected reply"}
    return {"verified": available, "detail": "key accepted, balance available" if available
            else "key accepted, but the balance is empty (calls will return 402)"}

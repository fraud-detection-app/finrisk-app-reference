"""One front door for every LLM provider, with a fallback that always answers.

The service and the API call this module and never learn which provider ran. That indirection
buys three things a teaching project actually needs:

- **Swap providers without touching callers.** ``LLM_PROVIDER=deepseek|bedrock|none``.
- **The class runs offline.** With no key the template path produces a valid insight, so nobody
  is blocked on a credit card.
- **The answer says where it came from.** Every result carries ``source``, so a reader can tell
  a model-written summary from a generated one -- week 8's provenance rule, one layer up.

Every model call goes through :func:`complete`, which does three more things:

- logs one line per call -- provider, the model the provider SAYS answered, purpose, latency,
  tokens -- so the terminal is proof of what ran, not a claim about what was configured;
- records the call in a per-request list (:func:`record_calls`), which the API returns and the
  web app shows beside each answer;
- raises :class:`LLMUnavailable` on any failure, so every caller has exactly one thing to catch
  and a deterministic fallback to take.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from financial_risk.config import get_settings
from financial_risk.llm.bedrock_client import (
    build_messages,
    fallback_risk_insight,
    parse_insight,
)

logger = logging.getLogger("financial_risk.llm")


class LLMUnavailable(RuntimeError):
    """No provider configured, or the call failed. Callers fall back; they never propagate this."""


# ------------------------------------------------------------------------ status


def llm_status(provider: str | None = None) -> dict[str, Any]:
    """What is configured right now. Cheap and local: nothing is called."""
    settings = get_settings()
    chosen = (provider or settings.llm_provider or "none").lower()
    if chosen == "deepseek":
        key_set = bool(settings.deepseek_api_key)
        return {
            "provider": "deepseek", "model": settings.deepseek_model, "ready": key_set,
            "key_set": key_set, "endpoint": settings.deepseek_base_url,
            "detail": "configured" if key_set else "LLM_PROVIDER=deepseek but DEEPSEEK_API_KEY is empty",
        }
    if chosen == "bedrock":
        # No API key: AWS credentials come from the runtime role (Lambda / ECS) or `aws login`.
        # This looks for them locally -- it never calls Bedrock, so /ready stays free and fast.
        from financial_risk.llm.bedrock_client import credentials_status

        creds = credentials_status()
        has_model = bool(settings.bedrock_model_id)
        return {
            "provider": "bedrock", "model": settings.bedrock_model_id, "ready": has_model and creds["found"],
            "key_set": None, "endpoint": f"bedrock-runtime ({settings.aws_region})",
            "region": settings.aws_region, "credentials": creds["found"],
            "detail": creds["detail"] if has_model else "LLM_PROVIDER=bedrock but BEDROCK_MODEL_ID is empty",
        }
    return {
        "provider": "none", "model": None, "ready": False, "key_set": None, "endpoint": None,
        "detail": "LLM_PROVIDER=none: template summaries and extractive answers, no model calls",
    }


def llm_available(provider: str | None = None) -> bool:
    return bool(llm_status(provider)["ready"])


def available_providers() -> dict[str, bool]:
    """Which providers are configured right now -- surfaced by the API's /ready endpoint."""
    from financial_risk.llm.deepseek_client import is_configured as deepseek_ready

    settings = get_settings()
    return {
        "deepseek": deepseek_ready(),
        # Looking for credentials costs a lookup, so only do it when bedrock is the chosen provider.
        "bedrock": llm_status("bedrock")["ready"] if settings.llm_provider.lower() == "bedrock"
        else bool(settings.bedrock_model_id),
        "template": True,  # always available; this is why the class never blocks
    }


def verify_provider() -> dict[str, Any]:
    """A startup check that the configured provider will actually answer. Never raises."""
    status = llm_status()
    if status["provider"] == "deepseek" and status["ready"]:
        from financial_risk.llm.deepseek_client import check_key

        return check_key()
    if status["provider"] == "bedrock":
        # Startup must not spend a token or fail without a network: report what is knowable locally.
        return {"verified": None if status["ready"] else False,
                "detail": f"{status['detail']}; the model itself is not called at startup"}
    return {"verified": None, "detail": status["detail"]}


# --------------------------------------------------------------- per-request call log

_CALLS: ContextVar[list[dict[str, Any]] | None] = ContextVar("llm_calls", default=None)


@contextmanager
def record_calls() -> Iterator[list[dict[str, Any]]]:
    """Collect every model call made inside the block (the service wraps each operation)."""
    calls: list[dict[str, Any]] = []
    token = _CALLS.set(calls)
    try:
        yield calls
    finally:
        _CALLS.reset(token)


def _record(entry: dict[str, Any]) -> None:
    calls = _CALLS.get()
    if calls is not None:
        calls.append(entry)


def describe_call(entry: dict[str, Any]) -> str:
    """One human line per call, used by the log and handy in a notebook."""
    head = f"{entry['provider']} · {entry['model']} · {entry['purpose']}"
    if not entry["ok"]:
        return f"{head} · FAILED after {entry['latency_ms']:.0f} ms: {entry['error']} -> fallback"
    tokens = ""
    if entry.get("prompt_tokens") is not None:
        tokens = f" · {entry['prompt_tokens']}→{entry['completion_tokens']} tokens"
    asked = entry.get("requested_model")
    mismatch = f" (requested {asked})" if asked and asked != entry["model"] else ""
    return f"{head} · {entry['latency_ms']:.0f} ms{tokens}{mismatch}"


# ------------------------------------------------------------------------- calling


def complete(
    messages: list[dict[str, str]],
    purpose: str,
    *,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Send role-tagged messages to the configured provider. Returns ``(text, call_record)``.

    ``purpose`` (summary, rag_answer, route, ...) exists for the log: when a request makes two
    calls, it is how you tell the router's call from the answer's."""
    status = llm_status(provider)
    if not status["ready"]:
        raise LLMUnavailable(status["detail"])

    started = time.perf_counter()
    entry: dict[str, Any] = {"provider": status["provider"], "model": status["model"], "purpose": purpose}
    try:
        if status["provider"] == "deepseek":
            from financial_risk.llm.deepseek_client import chat_detailed

            reply = chat_detailed(messages, temperature=temperature, max_tokens=max_tokens)
            text = reply["text"]
            entry.update(model=reply["model"], requested_model=reply["requested_model"],
                         prompt_tokens=reply["prompt_tokens"], completion_tokens=reply["completion_tokens"])
        else:  # bedrock: the Converse API takes the same role-tagged messages
            from financial_risk.llm.bedrock_client import BedrockRiskClient

            reply = BedrockRiskClient().converse(messages, temperature=temperature, max_tokens=max_tokens)
            text = reply["text"]
            entry.update(model=reply["model"], requested_model=reply["requested_model"],
                         prompt_tokens=reply["prompt_tokens"], completion_tokens=reply["completion_tokens"])
    except Exception as exc:  # noqa: BLE001 - every provider failure becomes one catchable type
        entry.update(ok=False, error=str(exc)[:200], latency_ms=round((time.perf_counter() - started) * 1000, 1))
        _record(entry)
        logger.warning(describe_call(entry))
        raise LLMUnavailable(entry["error"]) from exc

    entry.update(ok=True, error=None, latency_ms=round((time.perf_counter() - started) * 1000, 1))
    _record(entry)
    logger.info(describe_call(entry))
    return text, entry


# ------------------------------------------------------------------- the summary


def generate_insight(context: dict[str, Any], provider: str | None = None) -> dict[str, Any]:
    """Risk context in, structured insight out, with the provider and model recorded.

    Whatever the provider returns still passes week 8's three gates -- extract, validate,
    enforce -- so a cloud model can no more overrule the classifier than the local one could.
    """
    settings = get_settings()
    chosen = (provider or settings.llm_provider or "none").lower()

    if chosen in {"deepseek", "bedrock"}:
        label = chosen.capitalize()
        try:
            raw, call = complete(build_messages(context), "summary", provider=chosen)
            parsed = parse_insight(raw, context)
            if parsed is not None:
                return {**parsed, "source": chosen, "model": call["model"]}
            logger.warning("%s reply failed the insight schema; using the template.", label)
        except LLMUnavailable as exc:
            logger.warning("%s unavailable (%s); using the template.", label, exc)

    return {**fallback_risk_insight(context).to_dict(), "source": "template", "model": None}

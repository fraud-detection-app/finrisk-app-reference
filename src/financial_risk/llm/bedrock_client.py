"""LLM explainer: turn a model score plus its reasons into a reviewer-ready summary.

Design rules this module enforces (the week-8 class is built around them):

1. **The model decides, the LLM narrates.** ``risk_level`` in the returned insight is always
   the classifier's level from ``context`` -- if the LLM writes a different one it is
   overwritten (``enforce_model_decision``). Two deciders means zero accountability.
2. **Structured output or nothing.** The LLM must return JSON matching ``INSIGHT_SCHEMA``.
   Prose around the JSON is tolerated (``extract_json``); a missing key, a wrong type or an
   unknown risk level is rejected (``validate_insight``) and the deterministic fallback is
   used instead. Downstream code can rely on five keys, every time.
3. **Curated context.** The prompt carries only what the narrative needs: id, type, amount,
   score, level, threshold, top reasons. No customer identifiers, no full history.
4. **A local fallback always exists.** ``use_bedrock=False`` (or any cloud failure) produces
   a template-based insight, so an LLM outage never breaks scoring.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from financial_risk.config import DEFAULT_BEDROCK_MODEL_ID, get_settings  # noqa: F401 - re-exported

logger = logging.getLogger("financial_risk.llm")

INSIGHT_KEYS: tuple[str, ...] = ("risk_level", "summary", "key_reasons", "recommended_actions", "disclaimer")
RISK_LEVELS: tuple[str, ...] = ("low", "medium", "high")
DISCLAIMER = "Educational demo only. This is not financial, lending, legal, or investment advice."

# The contract, as a JSON Schema document. It is what the prompt asks for, what
# validate_insight() checks, and what the API layer (week 10) publishes.
INSIGHT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RiskInsight",
    "type": "object",
    "required": list(INSIGHT_KEYS),
    "additionalProperties": False,
    "properties": {
        "risk_level": {"type": "string", "enum": list(RISK_LEVELS)},
        "summary": {"type": "string", "minLength": 1, "maxLength": 600},
        "key_reasons": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6},
        "recommended_actions": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6},
        "disclaimer": {"type": "string", "minLength": 1},
    },
}

# What goes INTO the prompt. Everything else about the transaction stays out.
CONTEXT_KEYS: tuple[str, ...] = (
    "row_id", "transaction_type", "amount", "risk_score", "risk_level", "threshold", "top_reasons",
)

SYSTEM_PROMPT = (
    "You are a financial risk analytics assistant for an educational data science project.\n"
    "A machine-learning model has already scored the transaction; you explain that score, you do not change it.\n"
    "Return strict JSON with exactly these keys: risk_level, summary, key_reasons, recommended_actions, disclaimer.\n"
    "risk_level must repeat the model's risk_level from the context. key_reasons must be grounded in the context's top_reasons.\n"
    "Do not provide investment, lending, or legal advice. Mention this is not a production decision system."
)


@dataclass
class RiskInsight:
    risk_level: str
    summary: str
    key_reasons: list[str]
    recommended_actions: list[str]
    disclaimer: str

    def to_dict(self) -> dict[str, object]:
        return {
            "risk_level": self.risk_level,
            "summary": self.summary,
            "key_reasons": self.key_reasons,
            "recommended_actions": self.recommended_actions,
            "disclaimer": self.disclaimer,
        }


# ------------------------------------------------------------------------------ prompt

# Phrases that mark text trying to talk to the model instead of describing a transaction.
# Free-text fields (a memo, a merchant name, a reason string) are the injection surface;
# a match does not prove an attack, but the harness withholds the field either way.
INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions", "ignore all previous", "ignore the above", "disregard",
    "you are now", "new instructions", "system prompt", "developer mode", "set risk_level",
    "risk_level to", "reveal", "print your instructions",
)
MAX_FIELD_CHARS = 300

# One worked example for few-shot prompting: a context and the insight we want for it.
FEW_SHOT_EXAMPLE: tuple[dict[str, object], dict[str, object]] = (
    {
        "row_id": 0, "transaction_type": "TRANSFER", "amount": 181_000.0, "risk_score": 0.91,
        "risk_level": "high", "threshold": 0.48,
        "top_reasons": ["`origin_new_balance` pushes risk up (SHAP contribution +6.2).",
                        "`amount_to_origin_balance` pushes risk up (SHAP contribution +1.4)."],
    },
    {
        "risk_level": "high",
        "summary": "A 181,000 transfer emptied the origin account; the model scores it 0.91, well above the 0.48 review threshold.",
        "key_reasons": ["The origin balance is zero after the transfer.", "The amount is almost the whole prior balance."],
        "recommended_actions": ["Hold the transfer for analyst review.", "Confirm the transfer with the account holder."],
        "disclaimer": DISCLAIMER,
    },
)


def scan_for_injection(context: dict[str, object]) -> list[str]:
    """Names of fields whose text matches an INJECTION_PATTERN, e.g. ``top_reasons[1]``."""
    hits: list[str] = []
    for key, value in context.items():
        items = value if isinstance(value, list) else [value]
        for i, item in enumerate(items):
            if isinstance(item, str) and any(p in item.lower() for p in INJECTION_PATTERNS):
                hits.append(f"{key}[{i}]" if isinstance(value, list) else key)
    return hits


def curate_context(context: dict[str, object]) -> dict[str, object]:
    """Keep only CONTEXT_KEYS -- the leakage surface of the prompt is exactly this list --
    then cap every string at MAX_FIELD_CHARS and withhold any that fails the injection scan."""
    curated: dict[str, object] = {k: context[k] for k in CONTEXT_KEYS if k in context}
    flagged = set(scan_for_injection(curated))

    def clean(key: str, value: object, idx: int | None = None) -> object:
        name = f"{key}[{idx}]" if idx is not None else key
        if name in flagged:
            return "[withheld: text failed the injection scan]"
        return value[:MAX_FIELD_CHARS] if isinstance(value, str) else value

    for key, value in curated.items():
        curated[key] = [clean(key, v, i) for i, v in enumerate(value)] if isinstance(value, list) else clean(key, value)
    return curated


def _user_turn(context: dict[str, object]) -> str:
    return "Transaction/model context:\n" + json.dumps(curate_context(context), indent=2, default=str)


def build_messages(
    context: dict[str, object],
    *,
    examples: list[tuple[dict[str, object], dict[str, object]]] | None = None,
    extra_instructions: str | None = None,
) -> list[dict[str, str]]:
    """The prompt in chat form. Chat APIs (Bedrock Converse, Anthropic, OpenAI) take this
    shape directly.

    - ``system``: the standing policy, sent with every call (+ ``extra_instructions``, e.g. a
      reasoning hint).
    - optional ``examples``: few-shot demonstrations, each as a user turn (a context) followed
      by an assistant turn (the insight we wanted) -- the model imitates the pattern.
    - ``user``: only this transaction's curated context."""
    system = SYSTEM_PROMPT if not extra_instructions else SYSTEM_PROMPT + "\n" + extra_instructions
    messages = [{"role": "system", "content": system}]
    for example_context, example_insight in examples or []:
        messages.append({"role": "user", "content": _user_turn(example_context)})
        messages.append({"role": "assistant", "content": json.dumps(example_insight)})
    messages.append({"role": "user", "content": _user_turn(context)})
    return messages


def build_risk_prompt(context: dict[str, object]) -> str:
    """The same prompt flattened to one string, for callers that hold a single prompt (``BedrockRiskClient.invoke``)."""
    return "\n\n".join(m["content"] for m in build_messages(context)) + "\n"


# ---------------------------------------------------------------------- parse & validate


def extract_json(raw: str) -> dict[str, object] | None:
    """Pull the first {...} object out of a model reply, tolerating prose or code fences
    around it. Returns None when nothing parseable is there."""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def validate_insight(candidate: dict[str, object]) -> list[str]:
    """Check a parsed reply against INSIGHT_SCHEMA. Returns the list of problems (empty = valid)."""
    problems: list[str] = []
    props = INSIGHT_SCHEMA["properties"]
    for key in INSIGHT_SCHEMA["required"]:
        if key not in candidate:
            problems.append(f"missing key: {key}")
    for key in candidate:
        if key not in props:
            problems.append(f"unexpected key: {key}")
    level = candidate.get("risk_level")
    if "risk_level" in candidate and level not in RISK_LEVELS:
        problems.append(f"risk_level must be one of {list(RISK_LEVELS)}, got {level!r}")
    for key in ("summary", "disclaimer"):
        value = candidate.get(key)
        if key in candidate and (not isinstance(value, str) or not value.strip()):
            problems.append(f"{key} must be a non-empty string")
    if isinstance(candidate.get("summary"), str) and len(candidate["summary"]) > props["summary"]["maxLength"]:
        problems.append("summary exceeds 600 characters")
    for key in ("key_reasons", "recommended_actions"):
        value = candidate.get(key)
        if key not in candidate:
            continue
        if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
            problems.append(f"{key} must be a non-empty list of strings")
        elif len(value) > props[key]["maxItems"]:
            problems.append(f"{key} has more than {props[key]['maxItems']} items")
    return problems


def enforce_model_decision(insight: dict[str, object], context: dict[str, object]) -> dict[str, object]:
    """The classifier owns risk_level. If the LLM disagreed, the model wins and the
    disagreement is recorded inside the summary so a reviewer can see it happened."""
    out = dict(insight)
    model_level = context.get("risk_level")
    if model_level in RISK_LEVELS and out.get("risk_level") != model_level:
        out["summary"] = f"[model risk_level={model_level} enforced; LLM proposed {out.get('risk_level')!r}] " + str(out.get("summary", ""))
        out["risk_level"] = model_level
    out.setdefault("disclaimer", DISCLAIMER)
    return out


def parse_insight(raw: str, context: dict[str, object]) -> dict[str, object] | None:
    """extract -> validate -> enforce. None means 'use the fallback'."""
    candidate = extract_json(raw)
    if candidate is None:
        return None
    candidate.setdefault("disclaimer", DISCLAIMER)  # the one field we will fill in for the model
    if validate_insight(candidate):
        return None
    return enforce_model_decision(candidate, context)


# ------------------------------------------------------------------------------ clients


class BedrockError(RuntimeError):
    """Any failure talking to Bedrock, already translated into a sentence a student can act on.
    ``client.complete()`` turns it into LLMUnavailable, so the caller falls back to the template."""


# Which AWS error codes mean what -- the three you will actually meet on a first deploy.
_FRIENDLY_ERRORS: dict[str, str] = {
    "AccessDeniedException": (
        "AccessDeniedException -- the caller's IAM role may not run bedrock:InvokeModel on this model "
        "in this Region (or model access is not enabled for the account)."
    ),
    "ValidationException": (
        "ValidationException -- Bedrock does not accept this request; the usual cause is a wrong "
        "BEDROCK_MODEL_ID for this Region. Check the id in the Bedrock console's model catalog."
    ),
    "ResourceNotFoundException": "ResourceNotFoundException -- that model id does not exist in this Region.",
    "ThrottlingException": "ThrottlingException -- too many requests; slow down or retry with backoff.",
    "ModelTimeoutException": "ModelTimeoutException -- the model took too long to answer.",
    "ModelNotReadyException": "ModelNotReadyException -- the model is still warming up; retry shortly.",
    "ServiceUnavailableException": "ServiceUnavailableException -- Bedrock is temporarily unavailable.",
}

_CLIENTS: dict[str, Any] = {}  # one boto3 client per Region, reused across requests (and warm Lambda calls)
_CREDENTIAL_CACHE: dict[str, Any] = {}


def reset_caches() -> None:
    """Forget the cached boto3 client and credential check (tests call this between cases)."""
    _CLIENTS.clear()
    _CREDENTIAL_CACHE.clear()


def _botocore_session():
    """A botocore session whose credential search gives up quickly.

    On a laptop with no AWS login, the last place boto3 looks is the EC2 metadata address, which
    is a network call that would otherwise hang the first /ready for a second or more."""
    import botocore.session

    session = botocore.session.get_session()
    session.set_config_variable("metadata_service_timeout", 0.3)
    session.set_config_variable("metadata_service_num_attempts", 1)
    return session


def credentials_status(ttl_seconds: float = 60.0) -> dict[str, Any]:
    """Can boto3 find AWS credentials? Answers WITHOUT calling Bedrock and never raises.

    On Lambda and ECS the credentials come from the function's / task's role through environment
    variables -- there is no key in our code or config. A positive answer is cached; a negative
    one for ``ttl_seconds`` so a probe loop does not keep searching."""
    cached = _CREDENTIAL_CACHE.get("value")
    if cached and (cached["found"] or time.monotonic() - cached["at"] < ttl_seconds):
        return cached
    try:
        import boto3

        creds = boto3.Session(botocore_session=_botocore_session()).get_credentials()
        found = creds is not None
        method = getattr(creds, "method", None) if found else None
        detail = f"AWS credentials found ({method or 'unknown source'})" if found else \
            "no AWS credentials found (on AWS the runtime role provides them; on a laptop run `aws login`)"
    except ImportError:
        found, detail = False, "boto3 is not installed (pip install boto3)"
    except Exception as exc:  # noqa: BLE001 - a readiness probe must never crash startup.
        found, detail = False, f"no AWS credentials found (lookup error: {exc.__class__.__name__})"
    status = {"found": found, "detail": detail, "at": time.monotonic()}
    _CREDENTIAL_CACHE["value"] = status
    return status


def _runtime_client(region_name: str):
    if region_name not in _CLIENTS:
        import boto3
        from botocore.config import Config

        # Lambda gives a request 60 s, and up to 20 s of that can go on loading a new model/index first (inline mode),
        # so a hung call must give up well before the rest runs out.
        config = Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 2, "mode": "standard"})
        _CLIENTS[region_name] = boto3.client("bedrock-runtime", region_name=region_name, config=config)
    return _CLIENTS[region_name]


def _converse_parts(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Role-tagged chat messages -> Converse's shape: system prompt apart, turns as content blocks."""
    system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
    turns = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
    return system, turns


class BedrockRiskClient:
    """Bedrock's Converse API: one request shape for every model, no API key (the role is the credential)."""

    def __init__(self, region_name: str | None = None, model_id: str | None = None):
        settings = get_settings()
        self.region_name = region_name or settings.aws_region or "ap-southeast-2"
        self.model_id = model_id or settings.bedrock_model_id or DEFAULT_BEDROCK_MODEL_ID
        self.max_tokens = settings.bedrock_max_tokens
        self.temperature = settings.bedrock_temperature

    def converse(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """One chat turn. Same result shape as ``deepseek_client.chat_detailed``.

        Converse does not echo which model answered, so ``model`` is the id we sent -- the id is
        the only identity Bedrock gives back (the log line is still proof of what ran)."""
        system, turns = _converse_parts(messages)
        request: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": turns,
            "inferenceConfig": {
                "maxTokens": self.max_tokens if max_tokens is None else max_tokens,
                "temperature": self.temperature if temperature is None else temperature,
            },
        }
        if system:
            request["system"] = system

        started = time.perf_counter()
        try:
            response = _runtime_client(self.region_name).converse(**request)
        except ImportError as exc:
            raise BedrockError("boto3 is not installed (pip install boto3).") from exc
        except Exception as exc:  # noqa: BLE001 - botocore raises many types; all become BedrockError.
            raise BedrockError(self._explain(exc)) from exc

        try:
            blocks = response["output"]["message"]["content"]
            # Reasoning models also return "reasoningContent" blocks; the answer is the text blocks.
            text = "".join(block["text"] for block in blocks if "text" in block)
        except (KeyError, TypeError) as exc:
            raise BedrockError(f"Unexpected Bedrock response shape: {str(response)[:200]}") from exc
        usage = response.get("usage") or {}
        return {
            "text": text,
            "model": self.model_id,
            "requested_model": self.model_id,
            "prompt_tokens": usage.get("inputTokens"),
            "completion_tokens": usage.get("outputTokens"),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    def invoke(self, prompt: str) -> str:
        """The single-prompt form older callers use (week 8/9 helpers): one user turn."""
        return self.converse([{"role": "user", "content": prompt}])["text"]

    def _explain(self, exc: Exception) -> str:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
        if code in _FRIENDLY_ERRORS:
            return f"{_FRIENDLY_ERRORS[code]} [model {self.model_id}, Region {self.region_name}]"
        if exc.__class__.__name__ == "NoCredentialsError":
            return "No AWS credentials. On AWS the runtime role provides them; on a laptop run `aws login`."
        return f"Bedrock call failed: {exc.__class__.__name__}: {str(exc)[:160]}"


def fallback_risk_insight(context: dict[str, object]) -> RiskInsight:
    """Deterministic, template-based insight. Used when the LLM is disabled, unreachable,
    or returned something that failed validation."""
    score = float(context.get("risk_score", 0.0))
    level = context.get("risk_level")
    if level not in RISK_LEVELS:
        level = "high" if score >= 0.7 else "medium" if score >= 0.35 else "low"
    reasons = context.get("top_reasons") or [
        "Model score is based on transaction amount, transaction type, balances, and customer history.",
        "This fallback summary was generated locally because the LLM was unavailable.",
    ]
    actions = [
        "Review the transaction in an analyst workflow.",
        "Check whether the pattern is consistent with the customer's recent activity.",
        "Do not make automated financial decisions from this educational model alone.",
    ]
    return RiskInsight(
        risk_level=str(level),
        summary=f"The transaction is classified as {level} risk with score {score:.3f}.",
        key_reasons=[str(item) for item in reasons][:4],
        recommended_actions=actions,
        disclaimer=DISCLAIMER,
    )


def generate_risk_insight(context: dict[str, object], use_bedrock: bool = True) -> dict[str, object]:
    if not use_bedrock:
        return fallback_risk_insight(context).to_dict()

    prompt = build_risk_prompt(context)
    try:
        raw = BedrockRiskClient().invoke(prompt)
    except Exception as exc:  # noqa: BLE001 - fallback should protect demos from cloud setup failures.
        return fallback_risk_insight({**context, "top_reasons": [f"Bedrock fallback: {exc}"]}).to_dict()

    parsed = parse_insight(raw, context)
    if parsed is not None:
        return parsed
    return fallback_risk_insight({**context, "top_reasons": ["LLM reply failed schema validation; local fallback used."]}).to_dict()

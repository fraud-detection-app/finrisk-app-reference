"""Choosing a tool, running it, and being able to prove afterwards what happened.

An agent is three things the course has already built, wired in a loop:

    compliance gate  ->  router  ->  tool  ->  narrate
    (week 9 RAG)         (here)     (tools)   (week 8 harness)

Two design rules carry over and neither is negotiable:

1. **The gate runs before the router.** If policy were checked inside a tool, whether a lending
   question got refused would depend on which tool the router happened to pick -- a compliance
   boundary decided by a probabilistic step. It runs first, on the raw question, always.
2. **Every run returns a trace.** An agent that cannot say which tool it called, with what
   arguments, and where the answer came from is not auditable, and in a risk product that makes
   it unusable however good the answers are.

Two routers are provided because the comparison is the lesson. ``keyword_route`` is
deterministic, free, instantly testable and blind to phrasing it has never seen.
``llm_route`` reads the tool catalogue and picks, handling paraphrase -- and is probabilistic,
costs a call, and needs its output validated exactly like week 8's insight JSON. Same trade as
TF-IDF against embeddings one block earlier.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from financial_risk.agent.tools import TOOLS, AgentContext, tool_catalogue
from financial_risk.rag.index import DISCLAIMER, is_out_of_policy

logger = logging.getLogger("financial_risk.agent")

# An injected model call, as in rag.index: (messages, purpose) -> (text, call_record).
LLMCall = Callable[[list[dict[str, str]], str], tuple[str, dict[str, Any]]]

# Ordered: the first pattern that matches wins, so the specific ones come before the general.
ROUTING_RULES: tuple[tuple[str, str], ...] = (
    (r"\b(similar|resembl\w*|comparable|precedent)\b|\blike (this|case|row|transaction)\b", "find_similar_cases"),
    (r"\b(how many|count|rate|average|breakdown|by type|per type|distribution)\b", "aggregate_stats"),
    (r"\b(row|transaction|case)[ _]?(id)?\s*#?\d+", "lookup_transaction"),
    (r"\b(score|predict|assess|evaluate)\b.*\b(new|this|incoming|unsettled)\b", "score_new_transaction"),
    (r"\b(new|incoming|unsettled)\b.*\b(transaction|transfer|payment)\b", "score_new_transaction"),
    (r"\b(what|why|how|explain|which|where|who)\b", "search_knowledge"),
)

ROUTER_SYSTEM_PROMPT = (
    "You route a question to exactly one tool in a financial risk assistant.\n"
    "Return strict JSON: {\"tool\": \"<tool name>\", \"arguments\": {...}, \"why\": \"<one short sentence>\"}.\n"
    "Choose search_knowledge for questions about the project, model, metrics, data or policy.\n"
    "Never answer the question yourself; only choose the tool."
)

# The week-10 service's keyword rules: week 9's, plus the phrasings week 9's eval found missing
# ("pull up the record for 2192", "transaction with id 4503", "#4503", "is X riskier than Y").
# Week 9 keeps ROUTING_RULES as they are -- its routing-accuracy lab measures their ceiling.
_ROW_REF = r"\b(?:row|transaction|txn|case|record)\s*(?:id|number|no\.?)?\s*[:#]?\s*\d+|\bid\s*[:#]?\s*\d+|#\d+"
SERVICE_ROUTING_RULES: tuple[tuple[str, str], ...] = (
    ROUTING_RULES[0],                                                        # similar cases
    (r"\b(riskier|riskiest|safer|compare[sd]?|comparison|versus|vs)\b", "aggregate_stats"),
    ROUTING_RULES[1],                                                        # counts and rates
    ROUTING_RULES[3], ROUTING_RULES[4],                                      # a new transaction
    (_ROW_REF, "lookup_transaction"),
    (r"\b(pull|look\s?up|fetch|get|show|open|find|check|retrieve|display|details?|record)\b.*\b\d{2,}\b",
     "lookup_transaction"),
    ROUTING_RULES[5],                                                        # questions -> docs
)

SERVICE_ROUTER_PROMPT = (
    "You are the router of a fraud-risk assistant. Pick exactly ONE tool for the user's message "
    "and extract its arguments. Reply with JSON only: "
    "{\"tool\": \"<tool name or null>\", \"arguments\": {...}, \"why\": \"<one short sentence>\"}.\n"
    "How to choose:\n"
    "- It refers to a specific STORED transaction by number (row id, transaction id, record, case, "
    "#1234) and wants to see it, pull it, look it up, check it, or know whether it was fraud or why "
    "-> lookup_transaction with {\"row_id\": <integer>}.\n"
    "- It asks for cases similar or comparable to a transaction -> find_similar_cases with "
    "{\"row_id\": <integer>}.\n"
    "- It describes a NEW or hypothetical transaction (amount, type, balances) to be scored -> "
    "score_new_transaction with {\"transaction\": {<feature>: <value>}}.\n"
    "- It asks for counts, fraud rates, breakdowns, or comparisons between transaction types -> "
    "aggregate_stats with {\"group_by\": \"transaction_type\"}.\n"
    "- It asks about the project, model, metrics, features, data, methods or policy -> "
    "search_knowledge with {\"question\": <the message>}.\n"
    "- Nothing fits -> {\"tool\": null}.\n"
    "Never answer the question yourself."
)


@dataclass
class AgentResult:
    """What a run produced, and the record of how it got there."""

    answer: str
    tool: str | None
    arguments: dict[str, Any]
    data: dict[str, Any]
    refused: bool
    refusal_reason: str | None
    route_method: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer, "tool": self.tool, "arguments": self.arguments,
            "refused": self.refused, "refusal_reason": self.refusal_reason,
            "route_method": self.route_method, "trace": self.trace, "disclaimer": self.disclaimer,
        }


# ---------------------------------------------------------------------------------- routing


def extract_row_id(question: str) -> int | None:
    match = re.search(r"\b(?:row|transaction|case)[ _]?(?:id)?\s*#?(\d+)", question, re.I)
    return int(match.group(1)) if match else None


def extract_any_row_id(question: str) -> int | None:
    """Week 10's broader reading of "which transaction": row/txn/record/id/# forms, or the one
    number in a message that asks to pull something up."""
    for pattern in (r"\b(?:row|transaction|txn|case|record)\s*(?:id|number|no\.?)?\s*[:#]?\s*(\d+)",
                    r"\bid\s*[:#]?\s*(\d+)", r"#(\d+)"):
        match = re.search(pattern, question, re.I)
        if match:
            return int(match.group(1))
    numbers = re.findall(r"\b\d+\b", question)
    return int(numbers[0]) if len(numbers) == 1 else None


def normalise_arguments(tool_name: str, arguments: dict[str, Any], question: str) -> dict[str, Any]:
    """Make a router's arguments safe to call with -- the same gate for the LLM and the rules.

    Unknown keys are dropped (a model must not be able to pass a parameter the tool does not
    declare), ids are coerced to integers, and anything the model left out is recovered from the
    question where the question states it."""
    params = TOOLS[tool_name].parameters
    clean = {k: v for k, v in (arguments or {}).items() if k in params}
    if "row_id" in params:
        raw = clean.pop("row_id", None)
        digits = re.sub(r"\D", "", str(raw)) if raw is not None else ""
        row_id = int(digits) if digits else extract_any_row_id(question)
        if row_id is not None:
            clean["row_id"] = row_id
    if "k" in clean:
        try:
            clean["k"] = max(1, min(int(clean["k"]), 20))
        except (TypeError, ValueError):
            clean.pop("k")
    if tool_name == "search_knowledge":
        clean["question"] = question          # retrieve on the user's words, not a paraphrase
    if tool_name == "aggregate_stats" and not isinstance(clean.get("group_by"), str):
        clean["group_by"] = "transaction_type"
    if tool_name == "score_new_transaction" and not isinstance(clean.get("transaction"), dict):
        clean.pop("transaction", None)
        described = describe_new_transaction(question)
        if described:
            clean["transaction"] = described
    return clean


TRANSACTION_TYPES = ("CASH_OUT", "CASH_IN", "TRANSFER", "PAYMENT", "DEBIT")


def describe_new_transaction(question: str) -> dict[str, Any]:
    """The two fields a message about a new transaction usually states: its type and amount.

    Everything else is left to the tool's template, which reports those fields as unspecified --
    so a score built from "a transfer of 181000" is labelled as the partial guess it is."""
    found: dict[str, Any] = {}
    lowered = question.lower().replace("-", "_").replace(" out", "_out").replace(" in ", "_in ")
    for name in TRANSACTION_TYPES:
        if name.lower() in lowered:
            found["transaction_type"] = name
            break
    amounts = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", question)]
    if amounts:
        found["amount"] = max(amounts)
    return found


def default_arguments(tool_name: str, question: str) -> dict[str, Any]:
    """Arguments a deterministic router can extract from the question text alone."""
    row_id = extract_row_id(question)
    if tool_name in {"lookup_transaction", "find_similar_cases"}:
        return {"row_id": row_id} if row_id is not None else {}
    if tool_name == "aggregate_stats":
        return {"group_by": "transaction_type"}
    if tool_name == "search_knowledge":
        return {"question": question}
    return {}


def keyword_route(question: str, rules: tuple[tuple[str, str], ...] = ROUTING_RULES) -> tuple[str | None, dict[str, Any]]:
    """Deterministic routing: first matching rule wins.

    Free, instant, and every decision is a line you can point at in review. Its weakness is
    exactly the weakness of TF-IDF: it only knows the words it was given."""
    for pattern, tool_name in rules:
        if re.search(pattern, question, re.I):
            return tool_name, default_arguments(tool_name, question)
    return None, {}


def parse_route(raw: str) -> tuple[str | None, dict[str, Any]]:
    """Validate an LLM router's reply the way week 8 validates an insight: extract, then check.

    A tool name the registry does not contain is rejected rather than called -- a model that
    invents a tool must not be able to invent an action."""
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start < 0 or end <= start:
        return None, {}
    try:
        parsed = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None, {}
    tool_name = parsed.get("tool")
    if tool_name not in TOOLS:
        return None, {}
    arguments = parsed.get("arguments")
    return tool_name, arguments if isinstance(arguments, dict) else {}


def llm_route(
    question: str,
    use_bedrock: bool = False,
    llm: LLMCall | None = None,
    rules: tuple[tuple[str, str], ...] = ROUTING_RULES,
) -> tuple[str | None, dict[str, Any], str]:
    """Let a model read the catalogue and choose. Falls back to keywords on any failure.

    ``llm`` is the week-10 path (any provider, injected by the service); ``use_bedrock`` is week
    9's original. Either way the reply passes :func:`parse_route` -- a tool the registry does not
    contain is rejected -- and :func:`normalise_arguments` before anything runs."""
    if llm is not None:
        messages = [{"role": "system", "content": f"{SERVICE_ROUTER_PROMPT}\n\nTools:\n{tool_catalogue()}"},
                    {"role": "user", "content": question}]
        try:
            text, call = llm(messages, "route")
            tool_name, arguments = parse_route(text)
            if tool_name:
                return tool_name, normalise_arguments(tool_name, arguments, question), f"llm ({call.get('model')})"
            logger.info("router model chose no tool (%s); trying the keyword rules", text.strip()[:120])
        except Exception as exc:  # noqa: BLE001 - a router outage must not take the assistant down.
            logger.warning("router model unavailable (%s); using the keyword rules", exc)
        tool_name, arguments = keyword_route(question, rules)
        if tool_name:
            arguments = normalise_arguments(tool_name, arguments, question)
        return tool_name, arguments, "keyword (llm fallback)"
    if not use_bedrock:
        tool_name, arguments = keyword_route(question, rules)
        if tool_name and rules is not ROUTING_RULES:
            arguments = normalise_arguments(tool_name, arguments, question)
        return tool_name, arguments, "keyword (llm disabled)"
    try:
        from financial_risk.llm.bedrock_client import BedrockRiskClient

        prompt = f"{ROUTER_SYSTEM_PROMPT}\n\nTools:\n{tool_catalogue()}\n\nQuestion: {question}\nJSON:"
        tool_name, arguments = parse_route(BedrockRiskClient().invoke(prompt))
        if tool_name:
            return tool_name, arguments, "llm"
    except Exception:  # noqa: BLE001 - a router outage must not take the assistant down.
        pass
    tool_name, arguments = keyword_route(question)
    return tool_name, arguments, "keyword (llm fallback)"


# ------------------------------------------------------------------------------ narration


def narrate(tool_name: str, data: dict[str, Any]) -> str:
    """Turn a tool's structured result into a sentence, deterministically.

    Offline this is a template, exactly like week 8's fallback insight: it can only restate
    numbers the tool returned, so it cannot invent. Swapping in an LLM here is a one-line
    change and re-introduces the groundedness question the RAG block measured."""
    if tool_name == "lookup_transaction":
        if not data.get("found"):
            return str(data.get("note", "No such transaction."))
        outcome = "CONFIRMED fraud" if data["confirmed_fraud"] else "not fraud"
        return (f"Transaction {data['row_id']} ({data['transaction_type']}, {data['amount']:,.2f}) "
                f"was investigated and marked {outcome}. The model scores it {data['model_score']:.4f} "
                f"against a {data['model_threshold']:.4f} threshold, i.e. {data['model_says']} risk. "
                f"Top reason: {data['reasons'][0] if data['reasons'] else 'n/a'}")
    if tool_name == "score_new_transaction":
        if not data.get("scored"):
            return str(data.get("note", "Could not score that transaction."))
        return (f"This transaction has no investigation outcome -- it is unsettled. The model scores it "
                f"{data['model_score']:.4f} against a {data['model_threshold']:.4f} threshold, "
                f"i.e. {data['model_says']} risk. Top reason: "
                f"{data['reasons'][0] if data['reasons'] else 'n/a'}")
    if tool_name == "search_knowledge":
        return str(data.get("answer", ""))
    if tool_name == "find_similar_cases":
        neighbours = data.get("neighbours") or []
        if not neighbours:
            return "No confirmed fraud cases available to compare against."
        listed = ", ".join(f"{n['row_id']} (d={n['distance']})" for n in neighbours[:3])
        return (f"The {len(neighbours)} closest CONFIRMED frauds to transaction {data['row_id']} "
                f"are {listed}. Distance is in scaled feature space, so these are cases the model "
                f"sees as structurally similar -- not proof that this transaction is fraud.")
    if tool_name == "aggregate_stats":
        if not data.get("ok"):
            return str(data.get("note", "Could not compute that aggregate."))
        worst = max(data["rows"], key=lambda r: r["fraud_rate"])
        return (f"Across {data['total_transactions']:,} transactions there are "
                f"{data['total_confirmed_frauds']:,} confirmed frauds. Grouped by "
                f"{data['group_by']}, the highest rate is {worst[data['group_by']]} at "
                f"{worst['fraud_rate']:.2%}.")
    return "No answer."


# ----------------------------------------------------------------------------------- loop


def run_agent(
    question: str,
    ctx: AgentContext,
    use_llm_router: bool = False,
    extra_policy_patterns: tuple[str, ...] = (),
    llm: LLMCall | None = None,
    rules: tuple[tuple[str, str], ...] = ROUTING_RULES,
) -> AgentResult:
    """Gate, route, run, narrate -- recording every step.

    ``llm`` (week 10) routes with the configured model and falls back to ``rules``; without it,
    ``use_llm_router`` keeps week 9's behaviour exactly."""
    trace: list[dict[str, Any]] = [{"stage": "question", "detail": question}]

    # 1. Compliance first, on the raw question, before any routing decision exists.
    if is_out_of_policy(question, extra_policy_patterns):
        trace.append({"stage": "policy", "detail": "matched an out-of-policy pattern -> refuse"})
        logger.info("policy: refused before routing (out of policy)")
        return AgentResult(
            answer=("I can explain how this project's data, model and assistant work, but I can't give "
                    "financial, lending, legal, tax or investment advice or take account actions."),
            tool=None, arguments={}, data={}, refused=True, refusal_reason="out_of_policy",
            route_method="not routed", trace=trace,
        )
    trace.append({"stage": "policy", "detail": "passed"})

    # 2. Route.
    tool_name, arguments, method = llm_route(question, use_bedrock=use_llm_router, llm=llm, rules=rules)
    trace.append({"stage": "route", "detail": f"{tool_name or 'none'} via {method}"})
    logger.info("route: %s %s via %s", tool_name or "no tool", arguments or "", method)
    if tool_name is None:
        return AgentResult(
            answer="I could not work out which capability answers that. Try asking about the project, "
                   "a transaction id, or a fraud-rate breakdown.",
            tool=None, arguments={}, data={}, refused=True, refusal_reason="no_route",
            route_method=method, trace=trace,
        )

    # 3. Run the tool.
    tool = TOOLS[tool_name]
    missing = [k for k in tool.parameters if k not in arguments and k not in {"k", "group_by"}]
    if missing:
        trace.append({"stage": "arguments", "detail": f"missing {missing} -> cannot run"})
        return AgentResult(
            answer=f"That looks like a {tool_name} question, but I need {', '.join(missing)} to answer it.",
            tool=tool_name, arguments=arguments, data={}, refused=True, refusal_reason="missing_arguments",
            route_method=method, trace=trace,
        )
    started = time.perf_counter()
    data = tool.run(ctx, **arguments)
    trace.append({"stage": "tool", "detail": f"{tool_name}({arguments}) -> {data.get('provenance')}"})
    logger.info("tool: %s ran in %.0f ms -> %s", tool_name, (time.perf_counter() - started) * 1000,
                data.get("provenance"))

    # 4. A retrieval tool can refuse on its own terms (nothing relevant indexed).
    if tool_name == "search_knowledge" and data.get("refused"):
        trace.append({"stage": "narrate", "detail": "tool refused -> pass the refusal through"})
        return AgentResult(
            answer=str(data["answer"]), tool=tool_name, arguments=arguments, data=data,
            refused=True, refusal_reason=str(data.get("refusal_reason")), route_method=method, trace=trace,
        )

    answer = narrate(tool_name, data)
    trace.append({"stage": "narrate", "detail": f"templated from {data.get('provenance')}"})
    return AgentResult(
        answer=answer, tool=tool_name, arguments=arguments, data=data,
        refused=False, refusal_reason=None, route_method=method, trace=trace,
    )


def format_trace(result: AgentResult) -> str:
    return "\n".join(f"  {i}. {s['stage']:<10} {s['detail']}" for i, s in enumerate(result.trace, 1))

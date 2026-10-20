"""Week 9 part B: tools, routing, the compliance gate, and the trace."""

from __future__ import annotations

import pandas as pd
import pytest

from financial_risk.agent.evaluation import routing_accuracy, routing_report
from financial_risk.agent.router import (
    extract_row_id,
    format_trace,
    keyword_route,
    narrate,
    parse_route,
    run_agent,
)
from financial_risk.agent.tools import (
    TOOLS,
    AgentContext,
    aggregate_stats,
    find_similar_cases,
    lookup_transaction,
    score_new_transaction,
    tool_catalogue,
)
from financial_risk.data.paysim import generate_demo_data, ingest_to_parquet
from financial_risk.features.sql_features import build_feature_tables
from financial_risk.models.training import FEATURE_COLUMNS, train_model
from financial_risk.rag.index import build_index

POLICY_FIX = (
    r"\bshould (this|that|the) \w+\b",
    r"\b(waive|reverse|refund|unblock|cancel|close)\b.{0,30}\b(account|flag|transfer|charge|payment)\b",
)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    """A small but real context: data, a trained model, and an index."""
    base = tmp_path_factory.mktemp("agent")
    raw, bronze, silver, gold = (base / n for n in ("raw.csv", "bronze", "silver", "gold"))
    generate_demo_data(4000, raw)
    ingest_to_parquet(raw, bronze, silver)
    build_feature_tables(silver, gold)
    gold_path = gold / "transaction_features.parquet"
    train_model(gold_path, base / "models", "optimized")

    docs = base / "kb"
    docs.mkdir()
    (docs / "faq.md").write_text(
        "# FAQ\n\nQ: Why PR-AUC?\nA: Fraud is rare, so accuracy is misleading; PR-AUC ranks the "
        "positive class.\n", encoding="utf-8")
    index_dir = base / "index"
    build_index([docs], index_dir, seed_dir=None)
    return AgentContext(gold_path=gold_path, model_dir=base / "models", index_dir=index_dir)


# ----------------------------------------------------------------------------------- tools


def test_registry_is_self_describing():
    assert set(TOOLS) == {"lookup_transaction", "score_new_transaction", "search_knowledge",
                          "find_similar_cases", "aggregate_stats"}
    catalogue = tool_catalogue()
    for name in TOOLS:
        assert name in catalogue
    assert all(TOOLS[n].description and TOOLS[n].parameters for n in TOOLS)


def test_lookup_separates_the_label_from_the_prediction(ctx):
    row_id = int(ctx.features.iloc[0]["row_id"])
    result = lookup_transaction(ctx, row_id)
    assert result["found"]
    # the two claims are distinct fields and never merged
    assert isinstance(result["confirmed_fraud"], bool)
    assert 0.0 <= float(result["model_score"]) <= 1.0
    assert result["model_says"] in {"high", "low"}
    assert result["provenance"].startswith("sql:")


def test_lookup_missing_row_does_not_invent_one(ctx):
    result = lookup_transaction(ctx, 99_999_999)
    assert result["found"] is False and "No transaction" in str(result["note"])


def test_scoring_a_new_transaction_never_produces_a_label(ctx):
    fraud = ctx.features[ctx.features["is_fraud"] == 1].iloc[0]
    result = score_new_transaction(ctx, {c: fraud[c] for c in FEATURE_COLUMNS})
    assert result["scored"]
    assert result["confirmed_fraud"] is None          # the point of the whole tool
    assert result["unspecified_fields"] == []
    assert result["provenance"] == "model"


def test_partial_transaction_reports_what_it_had_to_invent(ctx):
    result = score_new_transaction(ctx, {"transaction_type": "TRANSFER", "amount": 181_000.0})
    assert result["scored"]
    assert len(result["unspecified_fields"]) > 5      # a score built mostly from a template
    assert "amount" not in result["unspecified_fields"]


def test_unknown_fields_are_rejected(ctx):
    result = score_new_transaction(ctx, {"not_a_column": 1})
    assert result["scored"] is False


def test_similar_cases_returns_confirmed_frauds_ranked_by_distance(ctx):
    row_id = int(ctx.features.iloc[0]["row_id"])
    result = find_similar_cases(ctx, row_id, k=3)
    assert result["found"]
    neighbours = result["neighbours"]
    assert len(neighbours) <= 3
    fraud_ids = set(ctx.features[ctx.features["is_fraud"] == 1]["row_id"].astype(int))
    assert all(n["row_id"] in fraud_ids for n in neighbours)
    assert [n["distance"] for n in neighbours] == sorted(n["distance"] for n in neighbours)
    assert row_id not in [n["row_id"] for n in neighbours]     # never itself


def test_aggregate_stats_matches_the_dataframe(ctx):
    result = aggregate_stats(ctx, "transaction_type")
    assert result["ok"]
    assert result["total_transactions"] == len(ctx.features)
    assert result["total_confirmed_frauds"] == int(ctx.features["is_fraud"].sum())
    assert sum(r["transactions"] for r in result["rows"]) == len(ctx.features)
    assert aggregate_stats(ctx, "nope")["ok"] is False


# --------------------------------------------------------------------------------- routing


@pytest.mark.parametrize("question,expected", [
    ("What is PR-AUC?", "search_knowledge"),
    ("Show me cases similar to transaction 14532", "find_similar_cases"),
    ("Which confirmed frauds resemble row 2192?", "find_similar_cases"),
    ("What does row 14532 look like?", "lookup_transaction"),
    ("What is the fraud rate by transaction type?", "aggregate_stats"),
])
def test_keyword_router_picks_the_right_tool(question, expected):
    tool_name, _ = keyword_route(question)
    assert tool_name == expected


def test_router_extracts_arguments_it_can_see():
    assert extract_row_id("Pull up transaction 14532 please") == 14532
    assert extract_row_id("no identifier here") is None
    _, arguments = keyword_route("Show me cases similar to transaction 14532")
    assert arguments == {"row_id": 14532}


def test_keyword_router_admits_when_it_cannot_route():
    # the honest ceiling of rules: phrasing with no trigger word
    assert keyword_route("Pull up the record for 2192")[0] is None


def test_llm_route_output_is_validated_like_any_model_output():
    assert parse_route('{"tool": "aggregate_stats", "arguments": {"group_by": "transaction_type"}}') == (
        "aggregate_stats", {"group_by": "transaction_type"})
    # an invented tool must never become an executable action
    assert parse_route('{"tool": "delete_everything", "arguments": {}}') == (None, {})
    assert parse_route("I think you should use aggregate_stats") == (None, {})
    assert parse_route('{"tool": "aggregate_stats", "arguments": "oops"}') == ("aggregate_stats", {})


# ------------------------------------------------------------------- the gate and the trace


def test_policy_gate_runs_before_the_router(ctx):
    result = run_agent("Should this customer receive a loan?", ctx, extra_policy_patterns=POLICY_FIX)
    assert result.refused and result.refusal_reason == "out_of_policy"
    assert result.tool is None
    # the proof: routing never happened
    assert [step["stage"] for step in result.trace] == ["question", "policy"]


def test_refusing_for_the_wrong_reason_is_visible(ctx):
    """Without the fix the same question is refused because NO rule matched -- right outcome,
    wrong mechanism. A binary pass/fail would call both of these a success."""
    result = run_agent("Should this customer receive a loan?", ctx)
    assert result.refused and result.refusal_reason == "no_route"


def test_every_run_carries_a_trace_and_provenance(ctx):
    row_id = int(ctx.features.iloc[0]["row_id"])
    result = run_agent(f"What does row {row_id} look like?", ctx, extra_policy_patterns=POLICY_FIX)
    assert not result.refused and result.tool == "lookup_transaction"
    assert [step["stage"] for step in result.trace] == ["question", "policy", "route", "tool", "narrate"]
    assert "sql:" in result.trace[3]["detail"]
    assert str(row_id) in format_trace(result)


def test_missing_arguments_refuse_rather_than_guess(ctx):
    result = run_agent("Score this new incoming transaction", ctx, extra_policy_patterns=POLICY_FIX)
    assert result.refused and result.refusal_reason == "missing_arguments"


def test_narration_only_restates_tool_output():
    data = {"ok": True, "group_by": "transaction_type", "total_transactions": 100,
            "total_confirmed_frauds": 3,
            "rows": [{"transaction_type": "TRANSFER", "transactions": 50, "confirmed_frauds": 3,
                      "fraud_rate": 0.06}]}
    sentence = narrate("aggregate_stats", data)
    assert "TRANSFER" in sentence and "6.00%" in sentence and "100" in sentence


def test_routing_report_scores_and_improves_with_the_policy_fix(ctx):
    questions = pd.DataFrame({
        "question": ["What is PR-AUC?", "What is the fraud rate by transaction type?",
                     "Should this customer receive a loan?"],
        "expected_tool": ["search_knowledge", "aggregate_stats", "refuse"],
    })
    before = routing_report(questions, ctx)
    after = routing_report(questions, ctx, extra_policy_patterns=POLICY_FIX)
    assert routing_accuracy(before) == pytest.approx(2 / 3)   # the lending question misroutes
    assert routing_accuracy(after) == 1.0

"""Week 10: the HTTP surface — status codes, validation, and the error contract.

These tests never start a server. FastAPI's TestClient drives the ASGI app in-process, which is
why an API test suite can run in CI in under a second. The `with` block matters: entering it runs
the lifespan, so the model is loaded exactly as it would be in production.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:      # `with` => lifespan runs => model loaded once
        yield test_client


@pytest.fixture(scope="module")
def ready(client) -> bool:
    return bool(client.get("/ready").json()["ready"])


@pytest.fixture(scope="module")
def sample_row_id(client, ready) -> int:
    if not ready:
        pytest.skip("pipeline artifacts missing; run scripts/run_pipeline.py")
    return int(client.get("/api/transactions/sample?limit=1").json()["row_ids"][0])


# ------------------------------------------------------------------------------ ops probes


def test_health_is_always_ok(client):
    """Liveness must not depend on the model: a slow disk should not trigger a restart."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_each_artifact(client):
    """Readiness is 200 or 503 — never a crash — and always names what is missing."""
    response = client.get("/ready")
    assert response.status_code in (200, 503)
    payload = response.json()
    assert set(payload["checks"]) == {"gold_features", "model_artifact", "vector_index"}
    assert payload["ready"] == (response.status_code == 200)
    assert payload["ready"] != bool(payload["missing"])
    assert "template" in payload["llm_providers"]      # the offline path is always available


DIGEST = "sha256:" + "ab12" * 16


def test_the_app_reports_which_image_is_answering(client, monkeypatch):
    """Terraform sets APP_IMAGE_DIGEST; /health and /ready repeat it, so the deploy's smoke test can tell the NEW
    version from an old one that still answers 200 (ECS Express rolling update)."""
    monkeypatch.delenv("APP_IMAGE_DIGEST", raising=False)
    assert client.get("/health").json()["image_digest"] is None               # a laptop: not set
    assert client.get("/ready").json()["image_digest"] is None

    monkeypatch.setenv("APP_IMAGE_DIGEST", DIGEST)
    assert client.get("/health").json() == {"status": "ok", "service": client.get("/health").json()["service"],
                                            "version": client.get("/health").json()["version"], "image_digest": DIGEST}
    assert client.get("/ready").json()["image_digest"] == DIGEST             # 200 or 503: the digest is there either way


def test_every_response_carries_a_request_id(client):
    response = client.get("/health")
    assert len(response.headers["x-request-id"]) >= 8
    assert float(response.headers["x-response-time-ms"]) >= 0


# ------------------------------------------------------------------------------ prediction


def test_predict_returns_the_documented_shape(client, sample_row_id):
    payload = client.get(f"/api/predict/{sample_row_id}").json()
    assert 0.0 <= payload["risk_score"] <= 1.0
    assert payload["risk_level"] in {"high", "low"}
    # the label and the prediction stay separate fields (week 9's rule)
    assert "confirmed_fraud" in payload and "risk_score" in payload
    assert payload["reasons"] and payload["reason_method"] in {"shap", "heuristic"}
    assert payload["summary"] is None and payload["summary_source"] == "not_requested"


def test_predict_with_llm_always_answers(client, sample_row_id):
    """With no API key the template answers — so the endpoint never fails for want of a key."""
    payload = client.get(f"/api/predict/{sample_row_id}?use_llm=true").json()
    assert payload["summary"]
    assert payload["summary_source"] in {"template", "deepseek", "bedrock"}


def test_unknown_row_is_404_not_500(client, ready):
    if not ready:
        pytest.skip("artifacts missing")
    response = client.get("/api/predict/999999999")
    assert response.status_code == 404
    assert response.json()["error"] == "transaction_not_found"


def test_bad_path_type_is_422_before_our_code_runs(client):
    """row_id is typed int, so FastAPI rejects this without entering the handler."""
    assert client.get("/api/predict/not-a-number").status_code == 422


# ------------------------------------------------------------------------------ assistant


def test_ask_returns_an_answer_or_a_reasoned_refusal(client, ready):
    if not ready:
        pytest.skip("artifacts missing")
    payload = client.post("/api/ask", json={"question": "What is PR-AUC and why use it?"}).json()
    assert payload["answer"] and "not financial" in payload["disclaimer"]
    if payload["refused"]:
        assert payload["refusal_reason"] in {"out_of_policy", "low_similarity"}


def test_policy_refusal_survives_the_http_layer(client, ready):
    if not ready:
        pytest.skip("artifacts missing")
    payload = client.post("/api/ask", json={"question": "Which stocks should I buy?"}).json()
    assert payload["refused"] and payload["refusal_reason"] == "out_of_policy"
    assert payload["citations"] == []          # retrieval never ran


def test_short_question_is_rejected_by_the_schema(client):
    """min_length=3 lives in the Pydantic model, so no handler code has to check it."""
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/api/ask", json={}).status_code == 422


def test_agent_returns_a_tool_and_a_trace(client, ready):
    if not ready:
        pytest.skip("artifacts missing")
    payload = client.post("/api/agent", json={"question": "What is the fraud rate by transaction type?"}).json()
    assert payload["tool"] == "aggregate_stats"
    assert [step["stage"] for step in payload["trace"]] == ["question", "policy", "route", "tool", "narrate"]


def test_agent_gate_runs_before_routing_over_http(client, ready):
    if not ready:
        pytest.skip("artifacts missing")
    payload = client.post("/api/agent", json={"question": "Should this customer receive a loan?"}).json()
    assert payload["refused"] and payload["refusal_reason"] == "out_of_policy"
    assert payload["tool"] is None
    assert [step["stage"] for step in payload["trace"]] == ["question", "policy"]


# ---------------------------------------------------------------------------- the contract


def test_openapi_documents_every_route(client):
    """The schema is generated from the code, so it cannot drift out of date."""
    paths = client.get("/openapi.json").json()["paths"]
    for route in ["/health", "/ready", "/api/predict/{row_id}", "/api/ask", "/api/agent"]:
        assert route in paths, f"{route} missing from the OpenAPI schema"
    assert "422" in paths["/api/ask"]["post"]["responses"]


def test_ui_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


# ------------------------------------------------------------- the language model, observed
#
# The service must say which model answered, route and answer with it when one is configured,
# and fall back cleanly when it is not -- or when it fails. No test here touches the network:
# `fake_deepseek` replaces the one HTTP call with canned replies and pretends a key is set.


@pytest.fixture
def fake_deepseek(monkeypatch):
    import financial_risk.llm.deepseek_client as ds

    replies: dict[str, str] = {}

    def chat_detailed(messages, model=None, temperature=None, max_tokens=None, timeout=30.0):
        system = messages[0]["content"]
        purpose = "route" if "router" in system else "rag_answer" if "knowledge assistant" in system else "summary"
        if replies.get(purpose) == "RAISE":
            raise ds.DeepSeekError("402 Payment Required -- the account balance is zero")
        return {"text": replies.get(purpose, ""), "model": "deepseek-flash", "requested_model": "deepseek-flash",
                "prompt_tokens": 400, "completion_tokens": 30, "latency_ms": 5.0}

    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    monkeypatch.setattr(ds, "chat_detailed", chat_detailed)
    return replies


def test_ready_reports_the_language_model(client):
    llm = client.get("/ready").json()["llm"]
    assert {"provider", "model", "ready", "detail", "uses"} <= set(llm)


def test_ready_names_the_configured_model(client, fake_deepseek):
    llm = client.get("/ready").json()["llm"]
    assert (llm["provider"], llm["model"], llm["ready"]) == ("deepseek", "deepseek-flash", True)
    assert "agent routing" in llm["uses"]


def test_offline_answer_is_extractive_not_pasted_chunks(client, ready, monkeypatch):
    if not ready:
        pytest.skip("pipeline artifacts missing")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    body = client.post("/api/ask", json={"question": "What is PR-AUC and why use it?"}).json()
    assert body["answer_method"] == "extracted" and body["llm_calls"] == []
    assert "Q:" not in body["answer"]                       # the FAQ's question is not an answer
    assert body["groundedness"] == 1.0                      # copied sentences are fully supported


def test_offline_router_understands_lookup_phrasings(client, sample_row_id, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    for question in (f"pull the transaction with id {sample_row_id}", f"look up {sample_row_id}", f"show me #{sample_row_id}"):
        body = client.post("/api/agent", json={"question": question}).json()
        assert body["tool"] == "lookup_transaction", question
        assert body["arguments"] == {"row_id": sample_row_id}
        assert body["data"]["found"] is True and body["route_method"] == "keyword (llm disabled)"


def test_offline_router_scores_a_described_new_transaction(client, ready, monkeypatch):
    if not ready:
        pytest.skip("pipeline artifacts missing")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    body = client.post("/api/agent", json={"question": "Score a new transfer of 181,000"}).json()
    assert body["tool"] == "score_new_transaction"
    assert body["arguments"]["transaction"] == {"transaction_type": "TRANSFER", "amount": 181000.0}
    assert body["data"]["scored"] is True


def test_llm_routes_the_agent_and_is_recorded(client, sample_row_id, fake_deepseek, caplog):
    fake_deepseek["route"] = f'{{"tool": "lookup_transaction", "arguments": {{"row_id": "#{sample_row_id}"}}}}'
    with caplog.at_level("INFO"):
        body = client.post("/api/agent", json={"question": "can you fetch that fraud case for me"}).json()
    assert body["tool"] == "lookup_transaction" and body["arguments"] == {"row_id": sample_row_id}
    assert body["route_method"] == "llm (deepseek-flash)"
    assert [(c["purpose"], c["model"], c["ok"]) for c in body["llm_calls"]] == [("route", "deepseek-flash", True)]
    assert "deepseek · deepseek-flash · route" in caplog.text      # the terminal names the model


def test_invented_tool_from_the_llm_falls_back_to_rules(client, sample_row_id, fake_deepseek):
    fake_deepseek["route"] = '{"tool": "delete_account", "arguments": {}}'
    body = client.post("/api/agent", json={"question": f"pull up transaction {sample_row_id}"}).json()
    assert body["tool"] == "lookup_transaction" and body["route_method"] == "keyword (llm fallback)"


def test_llm_generates_the_knowledge_answer(client, ready, fake_deepseek):
    if not ready:
        pytest.skip("pipeline artifacts missing")
    fake_deepseek["rag_answer"] = "PR-AUC focuses on positive-class ranking performance [faq.md § FAQ]."
    body = client.post("/api/ask", json={"question": "What is PR-AUC and why use it?"}).json()
    assert body["answer_method"] == "generated" and body["answer"].startswith("PR-AUC")
    assert body["llm_calls"][0]["purpose"] == "rag_answer" and body["groundedness"] is not None


def test_llm_failure_degrades_to_the_offline_answer(client, ready, fake_deepseek):
    if not ready:
        pytest.skip("pipeline artifacts missing")
    fake_deepseek["rag_answer"] = "RAISE"
    body = client.post("/api/ask", json={"question": "What is PR-AUC and why use it?"}).json()
    assert body["answer_method"] == "extracted_fallback" and body["answer"]
    assert body["llm_calls"][0]["ok"] is False and "402" in body["llm_calls"][0]["error"]


def test_week9_routing_rules_are_unchanged():
    """Week 9's routing-accuracy lab measures the ceiling of ITS rules; week 10 must not move it."""
    from financial_risk.agent.router import SERVICE_ROUTING_RULES, keyword_route

    assert keyword_route("Pull up the record for 2192") == (None, {})
    assert keyword_route("Pull up the record for 2192", SERVICE_ROUTING_RULES)[0] == "lookup_transaction"


def test_router_arguments_are_sanitised():
    from financial_risk.agent.router import normalise_arguments

    assert normalise_arguments("lookup_transaction", {"row_id": "#451", "drop_table": True}, "x") == {"row_id": 451}
    assert normalise_arguments("find_similar_cases", {}, "cases like txn 88") == {"row_id": 88}
    assert normalise_arguments("search_knowledge", {"question": "rewritten"}, "original words") == {"question": "original words"}


# ------------------------------------------------------------- Bedrock through the API
#
# In the cloud the provider is `bedrock` and the credential is the runtime role. Here boto3 is
# stubbed: /ready must describe Bedrock WITHOUT calling it, and a Bedrock failure must still be a
# 200 with the template summary -- never a 500.


@pytest.fixture
def fake_bedrock(monkeypatch):
    import boto3

    import financial_risk.llm.bedrock_client as bedrock

    class Runtime:
        calls = 0
        error = None

        def converse(self, **request):
            Runtime.calls += 1
            if Runtime.error is not None:
                raise Runtime.error
            return {"output": {"message": {"content": [{"text": "not json"}]}},
                    "usage": {"inputTokens": 10, "outputTokens": 3}}

    created: list[str] = []
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setattr(boto3, "client", lambda service, **kw: created.append(service) or Runtime())
    monkeypatch.setattr(bedrock, "credentials_status",
                        lambda ttl_seconds=60.0: {"found": True, "detail": "AWS credentials found (env)", "at": 0})
    bedrock.reset_caches()
    Runtime.calls, Runtime.error = 0, None
    yield Runtime, created
    bedrock.reset_caches()


def test_ready_describes_bedrock_without_calling_it(client, fake_bedrock):
    runtime, created = fake_bedrock
    llm = client.get("/ready").json()["llm"]
    assert (llm["provider"], llm["model"], llm["ready"]) == ("bedrock", "deepseek.v3.2", True)
    assert llm["region"] == "ap-southeast-2" and llm["credentials"] is True and llm["key_set"] is None
    assert "agent routing" in llm["uses"]
    assert runtime.calls == 0 and created == []          # not even a client was built


def test_ready_without_aws_credentials_is_still_200(client, ready, monkeypatch):
    import boto3

    import financial_risk.llm.bedrock_client as bedrock

    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setattr(boto3.Session, "get_credentials", lambda self: None)
    bedrock.reset_caches()
    try:
        response = client.get("/ready")
        assert response.status_code == (200 if ready else 503)
        llm = response.json()["llm"]
        assert llm["provider"] == "bedrock" and llm["ready"] is False and llm["credentials"] is False
        assert llm["uses"] == []
    finally:
        bedrock.reset_caches()


@pytest.mark.parametrize("code", ["AccessDeniedException", "ValidationException", "ThrottlingException"])
def test_bedrock_failure_still_returns_a_template_summary(client, sample_row_id, fake_bedrock, code):
    from botocore.exceptions import ClientError

    runtime, _ = fake_bedrock
    runtime.error = ClientError({"Error": {"Code": code, "Message": "x"}}, "Converse")
    response = client.get(f"/api/predict/{sample_row_id}?use_llm=true")
    assert response.status_code == 200
    body = response.json()
    assert body["summary_source"] == "template" and body["summary"]
    call = body["llm_calls"][0]
    assert (call["provider"], call["ok"]) == ("bedrock", False) and code in call["error"]

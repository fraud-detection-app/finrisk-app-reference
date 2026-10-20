"""POST /api/score, /api/predict and /ready when the model is a SageMaker endpoint (fakes), and when it is the baked model.

What is pinned down: one endpoint call per /api/score request, the validation (422 listing every problem, no AWS call), the 503
paths ("model endpoint is starting, retry"), NO silent fallback from a configured endpoint to the baked model, /ready.model, and
the agent scoring through the same scorer.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from botocore.exceptions import ReadTimeoutError
from fastapi.testclient import TestClient

from artifact_fakes import ACCOUNT, FakeEndpoints, FakeS3, FakeSSM, fake_embed, make_runner, model_package_files, write_tfidf_index
from financial_risk.models.training import FEATURE_COLUMNS
from financial_risk.service import artifacts
from financial_risk.serving.contract import frame_to_instances
from test_artifacts import MODEL_PTR, model_pointer

ENDPOINT = "finrisk-dev-fraud"


@pytest.fixture
def world(monkeypatch, tmp_path, demo_gold):
    """The real FastAPI app: pointers on, an endpoint (a real model behind a fake runtime), a BAKED model on disk as well."""
    from api.main import app

    ssm, s3, aws = FakeSSM(), FakeS3(), FakeEndpoints()
    baked = tmp_path / "baked-models"
    model_package_files(demo_gold, baked, threshold=0.99)          # a baked model that answers very differently (threshold 0.99)
    index = tmp_path / "baked-index"
    write_tfidf_index(index, version="baked")
    for name, value in {"POINTER_PREFIX": "/finrisk/dev", "POINTER_TTL_SECONDS": "0", "ARTIFACT_DIR": str(tmp_path / "work"),
                        "STARTUP_WAIT_SECONDS": "30", "MODEL_DIR": str(baked), "GOLD_PATH": str(demo_gold.parent),
                        "VECTOR_INDEX_DIR": str(index), "LLM_PROVIDER": "none", "POINTER_REFRESH": "background"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    monkeypatch.setattr(artifacts.ArtifactStore, "_ssm_client", lambda self: ssm)
    monkeypatch.setattr(artifacts.ArtifactStore, "_s3_client", lambda self: s3)
    monkeypatch.setattr(artifacts, "default_embedder_factory", lambda *a, **k: (lambda manifest: fake_embed))
    monkeypatch.setattr(artifacts, "EndpointScorer", aws.scorer_factory)
    aws.deploy(ENDPOINT, make_runner(demo_gold, tmp_path / "endpoint-model", 3, threshold=0.2))
    ssm.set(MODEL_PTR, model_pointer(3, ENDPOINT))

    class World:
        pass

    w = World()
    w.app, w.ssm, w.aws, w.gold = app, ssm, aws, demo_gold
    return w


@pytest.fixture
def client(world):
    with TestClient(world.app) as test_client:
        yield test_client


def feature_rows(world, n: int = 3) -> list[dict]:
    return frame_to_instances(pd.read_parquet(world.gold).head(n))


# ------------------------------------------------------------------------------ /api/score


def test_score_makes_one_endpoint_call_for_the_whole_batch(client, world):
    rows = feature_rows(world, 5)
    response = client.post("/api/score", json={"transactions": rows})
    body = response.json()
    assert response.status_code == 200 and set(body) == {"model_version", "threshold", "scored_by", "results", "elapsed_ms"}
    assert body["model_version"] == "3" and body["threshold"] == pytest.approx(0.2) and body["scored_by"] == "endpoint"
    assert [r["row_id"] for r in body["results"]] == [r["row_id"] for r in rows]
    assert all(set(r) == {"row_id", "score", "flagged"} for r in body["results"])           # explain defaults to false
    (call,) = world.aws.runtime.calls                                                       # ONE InvokeEndpoint for five rows
    sent = json.loads(call["Body"])
    assert call["EndpointName"] == ENDPOINT and len(sent["instances"]) == 5 and sent["explain"] is False


def test_score_with_explain_returns_reasons(client, world):
    body = client.post("/api/score", json={"transactions": feature_rows(world, 2), "explain": True, "top_k": 2}).json()
    for result in body["results"]:
        assert len(result["reasons"]) == 2 and all(r["feature"] in FEATURE_COLUMNS for r in result["reasons"])


def test_score_accepts_whole_number_row_ids(client, world):
    rows = feature_rows(world, 2)
    for row in rows:
        row["row_id"] = int(row["row_id"])
    body = client.post("/api/score", json={"transactions": rows}).json()
    assert [r["row_id"] for r in body["results"]] == [str(r["row_id"]) for r in rows]


def test_score_takes_up_to_500_rows(client, world):
    rows = (feature_rows(world, 50) * 10)[:500]
    assert len(client.post("/api/score", json={"transactions": rows}).json()["results"]) == 500
    assert len(world.aws.runtime.calls) == 1
    assert client.post("/api/score", json={"transactions": rows + rows[:1]}).status_code == 422       # 501
    assert client.post("/api/score", json={"transactions": []}).status_code == 422
    assert len(world.aws.runtime.calls) == 1


def test_score_rejects_a_row_that_breaks_the_contract_and_lists_every_problem(client, world):
    good, missing, extra = feature_rows(world, 3)
    del missing["amount"]
    extra["note"] = "typo"
    extra["log_amount"] = "lots"
    response = client.post("/api/score", json={"transactions": [good, missing, extra]})
    body = response.json()
    assert response.status_code == 422 and body["error"] == "invalid_transactions" and body["request_id"]
    assert [p["index"] for p in body["problems"]] == [1, 2]
    assert body["problems"][0]["missing"] == ["amount"] and body["problems"][1]["extra"] == ["note"] and "log_amount" in body["problems"][1]["invalid"]
    assert "row_id" in body["hint"]
    assert world.aws.runtime.calls == []                                                    # refused here, before any AWS call


def test_score_body_shape_errors_are_422(client):
    assert client.post("/api/score", json={}).status_code == 422
    assert client.post("/api/score", json={"transactions": "x"}).status_code == 422
    assert client.post("/api/score", json={"transactions": [{}], "top_k": 0}).status_code == 422


# ------------------------------------------------------------------------------ 503 paths


def test_a_cold_endpoint_is_503_model_endpoint_is_starting_retry(client, world):
    world.aws.runtime.errors = [ReadTimeoutError(endpoint_url="https://runtime.sagemaker.ap-southeast-2.amazonaws.com/")]
    response = client.post("/api/score", json={"transactions": feature_rows(world, 1)})
    body = response.json()
    assert response.status_code == 503 and body["error"] == "model_unavailable" and body["detail"] == "model endpoint is starting, retry"
    assert response.headers["retry-after"] and body["request_id"]
    assert client.post("/api/score", json={"transactions": feature_rows(world, 1)}).status_code == 200           # it woke up


def test_predict_is_503_too_when_the_endpoint_fails(client, world):
    row_id = client.get("/api/transactions/sample?limit=1").json()["row_ids"][0]
    world.aws.runtime.errors = [ReadTimeoutError(endpoint_url="x")]
    response = client.get(f"/api/predict/{row_id}")
    assert response.status_code == 503 and response.json()["detail"] == "model endpoint is starting, retry"
    assert client.get(f"/api/predict/{row_id}").status_code == 200


def test_there_is_no_silent_fallback_to_the_baked_model(client, world):
    """A baked model (threshold 0.99) is sitting right there. A failing endpoint must be a 503, never an answer from it."""
    row_id = client.get("/api/transactions/sample?limit=1").json()["row_ids"][0]
    served = client.get(f"/api/predict/{row_id}").json()
    assert served["scored_by"] == "endpoint" and served["threshold"] == pytest.approx(0.2)           # not 0.99
    world.aws.runtime.endpoints.clear()                                                              # the endpoint disappears
    for call in (lambda: client.get(f"/api/predict/{row_id}"), lambda: client.post("/api/score", json={"transactions": feature_rows(world, 1)})):
        response = call()
        assert response.status_code == 503 and response.json()["error"] == "model_unavailable"
        assert ACCOUNT not in response.text
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["model"]["source"] == "endpoint"
    assert "ValidationError" in ready.json()["model"]["error"]                                       # /ready says why
    assert ready.json()["load_error"] is None


def test_ready_reports_the_endpoint_status_and_its_last_error(client, world):
    body = client.get("/ready").json()
    model = body["model"]
    assert (model["version"], model["source"], model["endpoint"], model["status"], model["error"]) == (3, "endpoint", ENDPOINT, "InService", None)
    world.aws.runtime.errors = [ReadTimeoutError(endpoint_url="x")]
    client.post("/api/score", json={"transactions": feature_rows(world, 1)})
    assert client.get("/ready").json()["model"]["error"] == "model endpoint is starting, retry"
    client.post("/api/score", json={"transactions": feature_rows(world, 1)})
    assert client.get("/ready").json()["model"]["error"] is None


def test_ready_status_is_cached_so_polling_does_not_hammer_describe_endpoint(client, world):
    calls = len(world.aws.sagemaker.calls)
    for _ in range(5):
        client.get("/ready")
    assert len(world.aws.sagemaker.calls) == calls                                                    # start-up primed the 60 s cache


# ------------------------------------------------------------------------------ the pointer decides the scorer


def test_a_null_pointer_serves_the_baked_model_in_process(client, world):
    world.ssm.set(MODEL_PTR, {"version": None})
    client.get("/ready")                                                                             # TTL 0: this request looks again ...
    __import__("api.main", fromlist=["state"]).state["service"].store.join()                          # ... in a thread
    ready = client.get("/ready").json()
    assert ready["model"]["source"] == "baked" and ready["model"]["endpoint"] is None and ready["model"]["status"] is None
    body = client.post("/api/score", json={"transactions": feature_rows(world, 2)}).json()
    assert body["scored_by"] == "baked" and body["model_version"] == "baked" and body["threshold"] == pytest.approx(0.99)
    calls_before = len(world.aws.runtime.calls)
    assert calls_before == 0                                                                         # the endpoint was never called


def test_the_old_pointer_shape_is_a_visible_load_error_not_a_quiet_fallback(client, world):
    world.ssm.set(MODEL_PTR, {"version": 9, "model_data_url": "s3://finrisk-ml-x/models/job/output/model.tar.gz"})
    client.get("/ready")
    __import__("api.main", fromlist=["state"]).state["service"].store.join()
    body = client.get("/ready").json()
    assert body["ready"] is True and body["model"]["version"] == 3 and "old model pointer shape" in body["model"]["load_error"]
    assert client.post("/api/score", json={"transactions": feature_rows(world, 1)}).json()["scored_by"] == "endpoint"


# ------------------------------------------------------------------------------ the agent uses the scorer


def test_the_agent_scores_through_the_same_scorer(client, world):
    row_id = int(client.get("/api/transactions/sample?limit=1").json()["row_ids"][0])
    agent = client.post("/api/agent", json={"question": f"Look up transaction {row_id}"}).json()
    assert agent["tool"] == "lookup_transaction" and agent["data"]["found"] is True
    assert agent["data"]["model_threshold"] == pytest.approx(0.2) and agent["data"]["reason_method"] == "shap"
    assert [c["EndpointName"] for c in world.aws.runtime.calls] == [ENDPOINT]


def test_the_agent_tools_report_an_unscoreable_new_transaction_instead_of_crashing(client, world):
    from financial_risk.agent.tools import score_new_transaction

    ctx = __import__("api.main", fromlist=["state"]).state["service"].agent_context
    result = score_new_transaction(ctx, {"transaction_type": "TRANSFER", "amount": "a lot"})
    assert result["scored"] is False and "cannot be scored" in result["note"]
    ok = score_new_transaction(ctx, {"transaction_type": "TRANSFER", "amount": 250000.0})
    assert ok["scored"] is True and ok["model_threshold"] == pytest.approx(0.2) and "amount" not in ok["unspecified_fields"]


# ------------------------------------------------------------------------------ no transaction table of its own


def test_an_endpoint_without_a_local_feature_table_still_scores_but_cannot_look_rows_up(world, monkeypatch, tmp_path):
    """The model package carries no features.parquet any more; /api/predict needs the service's OWN table. Without it:
    /api/score still works (the rows arrive with their features), /api/predict is a 503 that says why, /ready is not ready."""
    monkeypatch.setenv("GOLD_PATH", str(tmp_path / "no-gold"))
    rows = frame_to_instances(pd.read_parquet(world.gold).head(2))
    with TestClient(world.app) as client:
        assert client.post("/api/score", json={"transactions": rows}).status_code == 200
        predict = client.get("/api/predict/1")
        assert predict.status_code == 503 and "transaction table" in predict.json()["detail"]
        ready = client.get("/ready")
        assert ready.status_code == 503 and ready.json()["checks"]["gold_features"] is False
        assert ready.json()["model"]["source"] == "endpoint" and ready.json()["model"]["version"] == 3

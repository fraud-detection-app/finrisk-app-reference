"""The HTTP surface when the service runs the way it does in AWS: POINTER_PREFIX set, a model (a SageMaker endpoint) and an
index named by SSM pointers (SSM, S3 and SageMaker are fakes). Checks the exact /ready shape that finrisk-ml and
finrisk-knowledge poll after they move a pointer, and that a new version shows up without a restart."""

from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from artifact_fakes import DOCS, FakeEndpoints, FakeS3, FakeSSM, fake_embed, make_runner, put_index, write_titan_index
from financial_risk.service import artifacts
from test_artifacts import INDEX_PTR, MODEL_PTR, index_pointer, model_pointer

ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$")


@pytest.fixture
def cloud(monkeypatch, tmp_path, demo_gold):
    """The real FastAPI app configured for AWS, with fake SSM + S3 behind it."""
    from api.main import app

    ssm, s3, aws = FakeSSM(), FakeS3(), FakeEndpoints()
    monkeypatch.setenv("POINTER_PREFIX", "/finrisk/dev")
    monkeypatch.setenv("POINTER_TTL_SECONDS", "0")                 # look at the pointers on every request
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("STARTUP_WAIT_SECONDS", "30")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "no-models"))
    monkeypatch.setenv("GOLD_PATH", str(demo_gold.parent))          # the service's own transaction table; no model is baked
    monkeypatch.setenv("VECTOR_INDEX_DIR", str(tmp_path / "no-index"))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)   # these tests are about the background way; inline has its own
    monkeypatch.setenv("POINTER_REFRESH", "background")
    monkeypatch.setattr(artifacts.ArtifactStore, "_ssm_client", lambda self: ssm)
    monkeypatch.setattr(artifacts.ArtifactStore, "_s3_client", lambda self: s3)
    monkeypatch.setattr(artifacts, "default_embedder_factory", lambda *a, **k: (lambda manifest: fake_embed))
    monkeypatch.setattr(artifacts, "EndpointScorer", aws.scorer_factory)       # the default scorer factory builds one per pointer

    def publish_model(version: int, threshold: float, endpoint: str = "finrisk-dev-fraud"):
        aws.deploy(endpoint, make_runner(demo_gold, tmp_path / f"m{version}", version, threshold))
        ssm.set(MODEL_PTR, model_pointer(version, endpoint))

    def publish_index(version: str, docs=None):
        files = write_titan_index(tmp_path / f"i{version}", version, docs)
        pointer = index_pointer(version)
        pointer["uri"] = put_index(s3, version, files)
        ssm.set(INDEX_PTR, pointer)

    publish_model.point_at_missing = lambda version: ssm.set(MODEL_PTR, model_pointer(version, "finrisk-dev-fraud-gone"))   # never deployed
    publish_model.aws = aws
    return app, publish_model, publish_index


def test_ready_reports_exactly_the_fields_the_other_pipelines_poll(cloud):
    app, publish_model, publish_index = cloud
    publish_model(7, 0.4)
    publish_index("20261016-101500-ab12cd3")
    with TestClient(app) as client:
        response = client.get("/ready")
        body = response.json()
        assert response.status_code == 200, body
        # finrisk-ml polls model.version (a JSON number equal to the pointer's) and model.load_error
        assert body["model"]["version"] == 7 and type(body["model"]["version"]) is int
        assert body["model"]["load_error"] is None and body["model"]["source"] == "endpoint"
        assert body["model"]["endpoint"] == "finrisk-dev-fraud" and body["model"]["status"] == "InService" and body["model"]["error"] is None
        # finrisk-knowledge polls index.version and index.load_error
        assert body["index"]["version"] == "20261016-101500-ab12cd3" and body["index"]["load_error"] is None
        assert {"version", "embeddings", "docs", "chunks", "source", "loaded_at", "load_error"} <= set(body["index"])
        assert {"version", "source", "loaded_at", "load_error"} <= set(body["model"])
        assert ISO_UTC.match(body["pointers_checked_at"]) and ISO_UTC.match(body["model"]["loaded_at"])
        assert body["ready"] is True and body["checks"] == {"gold_features": True, "model_artifact": True, "vector_index": True}
        assert "image_digest" in body and body["load_error"] is None
        assert client.get("/health").status_code == 200
        row_id = client.get("/api/transactions/sample?limit=1").json()["row_ids"][0]
        predicted = client.get(f"/api/predict/{row_id}")
        assert predicted.status_code == 200 and predicted.json()["scored_by"] == "endpoint" and predicted.json()["model_version"] == "7"
        answer = client.post("/api/ask", json={"question": "Who handles chargebacks and disputes?"}).json()
        assert answer["citations"][0]["source"] == "docs/refunds.md"


def test_a_new_version_appears_in_ready_without_restarting(cloud):
    app, publish_model, publish_index = cloud
    publish_model(7, 0.4)
    publish_index("v1")
    with TestClient(app) as client:
        assert client.get("/ready").json()["model"]["version"] == 7
        publish_model(8, 0.6, endpoint="finrisk-dev-fraud-b")
        publish_index("v2", docs={**DOCS, "docs/extra.md": "# Extra\n\nThe zebra protocol governs rare cases.\n"})
        client.get("/ready")                                                       # this request starts the background look (TTL 0) ...
        app_ctx = __import__("api.main", fromlist=["state"]).state["service"]
        app_ctx.store.join()                                                       # ... and we wait for the load to finish
        row_id = client.get("/api/transactions/sample?limit=1").json()["row_ids"][0]
        assert client.get(f"/api/predict/{row_id}").json()["threshold"] == pytest.approx(0.6)       # the new endpoint answers
        body = client.get("/ready").json()
        assert body["model"]["version"] == 8 and body["index"]["version"] == "v2" and body["index"]["docs"] == 4
        assert abs(body["model"]["threshold"] - 0.6) < 1e-6                                           # (an endpoint reports its threshold with an answer)


def test_a_bad_new_model_shows_load_error_and_ready_stays_200(cloud):
    app, publish_model, publish_index = cloud
    publish_model(7, 0.4)
    publish_index("v1")
    with TestClient(app) as client:
        publish_model.point_at_missing(9)
        client.get("/ready")                                                       # starts the (failing) background load
        __import__("api.main", fromlist=["state"]).state["service"].store.join()
        response = client.get("/ready")
        body = response.json()
        assert response.status_code == 200 and body["model"]["version"] == 7
        assert "Could not find endpoint" in body["model"]["load_error"] and "123456789012" not in json.dumps(body)


def test_nothing_loaded_is_503_with_the_reason(cloud):
    app, _, _ = cloud
    with TestClient(app) as client:                                                # both pointers missing, nothing baked
        response = client.get("/ready")
        body = response.json()
        assert response.status_code == 503 and body["ready"] is False
        assert body["model"]["version"] is None and body["index"]["version"] is None and body["hint"]
        assert client.get("/health").status_code == 200
        error = client.get("/api/predict/1")
        assert error.status_code == 503 and error.json()["error"] == "service_not_ready"


def test_inline_mode_loads_on_ready_but_health_never_touches_the_store(cloud, monkeypatch):
    app, publish_model, publish_index = cloud
    monkeypatch.setenv("POINTER_REFRESH", "inline")                                 # what Lambda gets by default
    publish_model(7, 0.4)
    publish_index("v1")
    with TestClient(app) as client:
        state = __import__("api.main", fromlist=["state"]).state
        store = state["service"].store
        assert store.refresh_mode == "inline" and client.get("/ready").json()["model"]["version"] == 7
        publish_model(8, 0.6, endpoint="finrisk-dev-fraud-b")
        calls = store._ssm_client().calls
        for _ in range(3):                                                          # TTL is 0: any real request would look again
            assert client.get("/health").status_code == 200
        assert store._ssm_client().calls == calls and store.model.bundle.version == 7     # /health did no work at all
        body = client.get("/ready").json()                                          # this request does it, and waits: no join()
        assert body["model"]["version"] == 8 and body["model"]["endpoint"] == "finrisk-dev-fraud-b"

"""The SageMaker hosting server and the endpoint contract (server side).

A real tiny model (trained on the demo rows) behind the real FastAPI app, driven by TestClient -- the same two routes SageMaker
calls (GET /ping, POST /invocations). Also: the contract's validation, the reasons (TreeSHAP and linear SHAP, checked against
the SHAP library and against additivity), and the package loader finrisk-ml imports. No AWS.
"""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from artifact_fakes import model_package_files, tarball
from financial_risk.models.package import ModelPackageError, load_model_package, unpack_model_package
from financial_risk.models.training import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, train_model
from financial_risk.serving.contract import (
    INSTANCE_KEYS, MAX_INSTANCES, InvalidRequest, build_request, frame_to_instances, parse_request, reason_sentences,
)
from financial_risk.serving.runner import ModelRunner
from financial_risk.serving.sagemaker_server import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def packages(tmp_path_factory, demo_gold):
    """An unpacked baseline (LogisticRegression) package and an optimized (XGBoost) one, as SageMaker leaves them in /opt/ml/model."""
    base = tmp_path_factory.mktemp("packages")
    out = {}
    for kind in ("baseline", "optimized"):
        folder = base / kind
        train_model(demo_gold, folder, model_type=kind)
        out[kind] = folder
    return out


@pytest.fixture(scope="module")
def rows(demo_gold) -> pd.DataFrame:
    frame = pd.read_parquet(demo_gold)
    return pd.concat([frame[frame["is_fraud"] == 1].head(3), frame[frame["is_fraud"] == 0].head(5)])


@pytest.fixture(params=["baseline", "optimized"])
def client(request, packages):
    with TestClient(create_app(packages[request.param], "3")) as test_client:
        test_client.kind = request.param
        yield test_client


def instances_of(frame: pd.DataFrame) -> list[dict]:
    return frame_to_instances(frame)


# ------------------------------------------------------------------------------ the feature change


def test_the_model_features_are_exactly_the_contract_in_order():
    assert CATEGORICAL_FEATURES == ["transaction_type"]
    assert NUMERIC_FEATURES == ["amount", "log_amount", "origin_old_balance", "origin_new_balance", "destination_old_balance",
                                "destination_new_balance", "origin_balance_delta", "destination_balance_delta",
                                "amount_to_origin_balance", "hour_of_day", "tx_index_for_origin", "customer_tx_count_so_far",
                                "customer_amount_so_far"]
    assert FEATURE_COLUMNS == CATEGORICAL_FEATURES + NUMERIC_FEATURES and "step" not in FEATURE_COLUMNS
    assert INSTANCE_KEYS == ("row_id", *FEATURE_COLUMNS)


def test_hour_of_day_is_computed_from_step_and_step_stays_as_metadata(demo_gold):
    frame = pd.read_parquet(demo_gold)
    assert "step" in frame.columns and "hour_of_day" in frame.columns
    assert (frame["hour_of_day"] == (frame["step"] - 1) % 24).all() and frame["hour_of_day"].between(0, 23).all()
    assert frame["step"].max() > 24                                               # the demo data spans many days: the hour wraps


def test_a_trained_model_uses_hour_of_day_and_not_step(packages):
    artifact = joblib.load(packages["baseline"] / "model_v1.pkl")
    assert artifact["feature_columns"] == FEATURE_COLUMNS
    names = list(artifact["model"].named_steps["preprocess"].get_feature_names_out())
    assert "numeric__hour_of_day" in names and "numeric__step" not in names


# ------------------------------------------------------------------------------ /ping and /invocations


def test_ping_is_healthy_once_the_model_is_loaded(client):
    response = client.get("/ping")
    assert response.status_code == 200 and response.json() == {"status": "ok", "model_version": "3"}


def test_invocations_answers_the_documented_response(client, rows):
    body = client.post("/invocations", json=build_request(instances_of(rows))).json()
    assert set(body) == {"model_version", "threshold", "predictions"}
    assert body["model_version"] == "3" and 0 < body["threshold"] < 1
    assert [p["row_id"] for p in body["predictions"]] == [str(r) for r in rows["row_id"]]       # same order, row_id as a string
    for prediction in body["predictions"]:
        assert set(prediction) == {"row_id", "score", "flagged"}                                 # no reasons unless asked
        assert 0.0 <= prediction["score"] <= 1.0 and prediction["flagged"] == (prediction["score"] >= body["threshold"])


def test_the_score_matches_the_model_itself(client, rows, packages):
    artifact = joblib.load(next(packages[client.kind].glob("model_v*.pkl")))
    expected = artifact["model"].predict_proba(rows[FEATURE_COLUMNS])[:, 1]
    body = client.post("/invocations", json=build_request(instances_of(rows))).json()
    assert np.allclose([p["score"] for p in body["predictions"]], expected, atol=1e-6)


def test_explain_returns_the_top_reasons_with_original_feature_names(client, rows):
    body = client.post("/invocations", json=build_request(instances_of(rows), explain=True, top_k=3)).json()
    for prediction, (_, row) in zip(body["predictions"], rows.iterrows()):
        reasons = prediction["reasons"]
        assert len(reasons) == 3
        assert [abs(r["contribution"]) for r in reasons] == sorted((abs(r["contribution"]) for r in reasons), reverse=True)
        for reason in reasons:
            assert set(reason) == {"feature", "value", "contribution"} and reason["feature"] in FEATURE_COLUMNS    # never transaction_type_CASH_OUT
            expected = row[reason["feature"]]
            assert reason["value"] == (expected if isinstance(expected, str) else pytest.approx(float(expected)))


def test_one_hot_columns_are_summed_back_into_transaction_type(client, rows):
    body = client.post("/invocations", json=build_request(instances_of(rows), explain=True, top_k=20)).json()
    for prediction in body["predictions"]:
        features = [r["feature"] for r in prediction["reasons"]]
        assert sorted(features) == sorted(FEATURE_COLUMNS) and features.count("transaction_type") == 1


def test_contributions_are_additive_in_log_odds(packages, rows):
    """TreeSHAP / linear SHAP: contributions + the model's base value = the log-odds of the score. A wrong mapping of the
    one-hot columns (or a lost column) breaks this immediately."""
    import xgboost as xgb

    for kind in ("baseline", "optimized"):
        runner = ModelRunner.from_package_dir(packages[kind], "1")
        response = runner.invoke(build_request(instances_of(rows), explain=True, top_k=20))
        pipeline = runner.pipeline
        x = pipeline.named_steps["preprocess"].transform(rows[FEATURE_COLUMNS])
        x = x.toarray() if hasattr(x, "toarray") else x
        estimator = pipeline.steps[-1][1]
        base = (estimator.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)[:, -1] if kind == "optimized"
                else np.full(len(rows), float(estimator.intercept_[0])))
        for prediction, bias in zip(response["predictions"], base):
            total = sum(r["contribution"] for r in prediction["reasons"]) + bias
            score = pipeline.predict_proba(rows[FEATURE_COLUMNS].iloc[[response["predictions"].index(prediction)]])[0, 1]
            assert total == pytest.approx(math.log(score / (1 - score)), abs=2e-3), kind     # (contributions are rounded to 4 decimals)


def test_treeshap_agrees_with_the_shap_library(packages, rows):
    """The booster's own pred_contribs must equal the week 7 explainer (shap.TreeExplainer), summed back to original features."""
    import shap

    runner = ModelRunner.from_package_dir(packages["optimized"], "1")
    pre, estimator = runner.pipeline.named_steps["preprocess"], runner.pipeline.steps[-1][1]
    x = pre.transform(rows[FEATURE_COLUMNS])
    x = x.toarray() if hasattr(x, "toarray") else x
    reference = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent").shap_values(x)
    mine = runner._contributions(rows)
    grouped = np.zeros_like(mine)
    for column, target in enumerate(runner._groups()):
        grouped[:, target] += np.asarray(reference)[:, column]
    assert np.allclose(mine, grouped, atol=1e-4)


def test_a_model_type_that_cannot_explain_still_scores_with_no_reasons(packages, rows):
    runner = ModelRunner.from_package_dir(packages["baseline"], "1")
    runner.pipeline = _OpaquePipeline(runner.pipeline)                                  # neither XGBoost nor logistic regression
    response = runner.invoke(build_request(instances_of(rows), explain=True))
    assert all(p["reasons"] == [] and 0 <= p["score"] <= 1 for p in response["predictions"])


class _OpaquePipeline:
    """Looks like a fitted pipeline whose last step is some other model type; it explains nothing."""

    def __init__(self, inner):
        self.inner = inner
        self.named_steps = inner.named_steps
        self.steps = [*inner.steps[:-1], ("model", object())]

    def predict_proba(self, x):
        return self.inner.predict_proba(x)


def test_500_instances_are_fine_and_501_is_a_400(client, rows):
    many = (instances_of(rows) * 70)[:MAX_INSTANCES]
    assert len(client.post("/invocations", json=build_request(many)).json()["predictions"]) == MAX_INSTANCES
    too_many = client.post("/invocations", json=build_request((instances_of(rows) * 70)[:MAX_INSTANCES + 1]))
    assert too_many.status_code == 400 and "at most 500" in too_many.json()["message"]


# ------------------------------------------------------------------------------ the 400s list what is wrong


def test_missing_keys_are_a_400_that_lists_them(client, rows):
    instance = instances_of(rows.head(1))[0]
    del instance["amount"], instance["hour_of_day"]
    response = client.post("/invocations", json=build_request([instance]))
    body = response.json()
    assert response.status_code == 400 and body["error"] == "invalid_request"
    assert body["problems"][0]["index"] == 0 and body["problems"][0]["missing"] == ["amount", "hour_of_day"]
    assert "missing ['amount', 'hour_of_day']" in body["message"]


def test_extra_keys_are_a_400_that_lists_them(client, rows):
    instance = {**instances_of(rows.head(1))[0], "step": 12, "is_fraud": 1}                 # a typo-catcher: step is NOT a feature any more
    body = client.post("/invocations", json=build_request([instance])).json()
    assert body["problems"][0]["extra"] == ["is_fraud", "step"]


def test_wrong_types_and_every_bad_row_are_reported_together(client, rows):
    good, bad_a, bad_b = instances_of(rows.head(3))
    bad_a["amount"] = "a lot"
    bad_b["transaction_type"] = 5
    bad_b["origin_old_balance"] = float("nan")
    response = client.post("/invocations", content=json.dumps(build_request([good, bad_a, bad_b])), headers={"content-type": "application/json"})
    problems = response.json()["problems"]
    assert response.status_code == 400 and [p["index"] for p in problems] == [1, 2]
    assert set(problems[0]["invalid"]) == {"amount"} and set(problems[1]["invalid"]) == {"transaction_type", "origin_old_balance"}


@pytest.mark.parametrize("body, expect", [
    ({"instances": []}, "non-empty list"),
    ({}, "non-empty list"),
    ({"instances": "x"}, "non-empty list"),
    ({"instances": [1]}, "must be an object"),
    ({"instances": [], "explain": "yes"}, "'explain' must be true or false"),
    ({"instances": [], "top_k": 0}, "'top_k'"),
    ({"instances": [], "top_k": 99}, "'top_k'"),
    ({"instances": [], "extra_thing": 1}, "unknown top-level key"),
])
def test_malformed_requests_are_400(client, body, expect):
    response = client.post("/invocations", json=body)
    assert response.status_code == 400 and expect in response.json()["message"]


def test_a_body_that_is_not_json_is_a_400(client):
    response = client.post("/invocations", content=b"amount,type\n1,TRANSFER", headers={"content-type": "text/csv"})
    assert response.status_code == 400 and response.json()["error"] == "invalid_json"


# ------------------------------------------------------------------------------ starting up


def test_ping_is_unhealthy_without_a_model(tmp_path):
    with TestClient(create_app(tmp_path / "empty", "3")) as client:
        ping = client.get("/ping")
        assert ping.status_code == 503 and "no model" in ping.json()["error"].lower()
        assert client.post("/invocations", json={"instances": []}).status_code == 503


def test_a_model_trained_on_the_old_features_is_refused_with_the_reason(tmp_path, demo_gold):
    folder = tmp_path / "old"
    train_model(demo_gold, folder)
    path = folder / "model_v1.pkl"
    artifact = joblib.load(path)
    artifact["feature_columns"] = ["step" if c == "hour_of_day" else c for c in FEATURE_COLUMNS]       # trained before the 13b change
    joblib.dump(artifact, path)
    with pytest.raises(ModelPackageError, match="step.*hour_of_day|hour_of_day.*step"):
        load_model_package(folder)
    with TestClient(create_app(folder, "3")) as client:
        assert client.get("/ping").status_code == 503 and "different features" in client.get("/ping").json()["error"]


def test_the_server_reads_its_settings_from_the_environment(packages, monkeypatch):
    monkeypatch.setenv("SM_MODEL_DIR", str(packages["baseline"]))
    monkeypatch.setenv("MODEL_VERSION", "41")
    with TestClient(create_app()) as client:
        assert client.get("/ping").json()["model_version"] == "41"


def test_the_package_needs_no_features_parquet(packages):
    assert not (packages["baseline"] / "features.parquet").exists()
    assert load_model_package(packages["baseline"])["threshold"] > 0


def test_the_package_loader_unpacks_model_tar_gz_the_way_finrisk_ml_ships_it(tmp_path, demo_gold):
    """finrisk-ml packs model_vN.pkl + latest_model.joblib + metrics.json; its jobs open it with unpack_model_package."""
    files = model_package_files(demo_gold, tmp_path / "pkg", threshold=0.4)
    assert set(files) == {"model_v2.pkl", "latest_model.joblib", "metrics.json"}                 # (and nothing else)
    archive = tmp_path / "model.tar.gz"
    archive.write_bytes(tarball(files))
    unpack_model_package(archive, tmp_path / "out")
    assert load_model_package(tmp_path / "out")["threshold"] == pytest.approx(0.4)
    with TestClient(create_app(tmp_path / "out", "9")) as client:
        assert client.get("/ping").status_code == 200

    evil = tmp_path / "evil.tar.gz"
    evil.write_bytes(tarball({**files, "../evil.pkl": b"x"}))
    with pytest.raises(ValueError, match="unsafe"):
        unpack_model_package(evil, tmp_path / "out2")
    assert not (tmp_path / "evil.pkl").exists()
    empty = tmp_path / "empty.tar.gz"
    empty.write_bytes(tarball({"metrics.json": b"{}"}))
    with pytest.raises(ValueError, match="holds no model"):
        unpack_model_package(empty, tmp_path / "out3")


def test_contract_helpers_round_trip(rows):
    request = build_request(instances_of(rows.head(2)), explain=True, top_k=2)
    parsed = parse_request(json.loads(json.dumps(request)))
    assert list(parsed.frame.columns) == FEATURE_COLUMNS and parsed.row_ids == [str(r) for r in rows.head(2)["row_id"]]
    assert parsed.explain is True and parsed.top_k == 2
    assert reason_sentences([{"feature": "amount", "value": 5.0, "contribution": -1.25}]) == ["`amount` pushes risk down (SHAP contribution -1.250)."]
    with pytest.raises(InvalidRequest):
        parse_request({"instances": [{"row_id": 1}]})


# ------------------------------------------------------------------------------ the real entry point


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_python_dash_m_starts_the_hosting_server_on_the_sagemaker_port_variable(packages, rows):
    """`python -m financial_risk.serving.sagemaker_server` is what finrisk-ml's `serve` runs inside the container."""
    port = _free_port()
    env = dict(os.environ, SM_MODEL_DIR=str(packages["baseline"]), MODEL_VERSION="5", SAGEMAKER_BIND_TO_PORT=str(port),
               PYTHONPATH=str(ROOT / "src"), LLM_PROVIDER="none")
    process = subprocess.Popen([sys.executable, "-m", "financial_risk.serving.sagemaker_server"], env=env, cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 90
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=2) as ping:
                    assert ping.status == 200
                    break
            except OSError:
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail(f"the server did not come up: {process.stdout.read() if process.poll() is not None else 'timeout'}")
                time.sleep(0.3)
        request = urllib.request.Request(f"http://127.0.0.1:{port}/invocations", data=json.dumps(build_request(instances_of(rows.head(2)))).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.loads(response.read())
        assert body["model_version"] == "5" and len(body["predictions"]) == 2
    finally:
        process.terminate()
        process.wait(timeout=15)

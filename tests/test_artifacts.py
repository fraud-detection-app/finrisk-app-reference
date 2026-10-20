"""The service serves what two SSM pointers name, and swaps it while running.

SSM, S3 and SageMaker are fakes (dictionaries, and a real ModelRunner behind a fake runtime client). The knowledge index is
built exactly like the finrisk-knowledge pipeline builds it. What these tests pin down:

  first load · a version change swaps · a failed load keeps the old one and says why · the TTL is respected ·
  null / missing pointers mean the baked fallback · a pointer that names something unloadable at start-up does NOT
  fall back · the OLD model pointer shape is refused in words · the formats (titan and tfidf indexes) are read as the
  other repos write them.

The model is a SageMaker endpoint now (13b): "loading" a model version means building an EndpointScorer for the endpoint the
pointer names and checking with DescribeEndpoint that it exists. Nothing about the model is downloaded.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time

import pytest
from botocore.exceptions import ClientError

from artifact_fakes import (
    ACCOUNT, DOCS, KB_BUCKET, ML_BUCKET, FakeEndpoints, FakeS3, FakeSSM, endpoint_pointer, fake_embed, make_runner,
    model_package_files, put_index, write_tfidf_index, write_titan_index,
)
from financial_risk.config import get_settings
from financial_risk.rag.index import RetrievalUnavailable, TitanEmbedder, retrieve
from financial_risk.service.artifacts import ArtifactStore, mask, parse_s3_uri
from financial_risk.service.risk_service import ServiceContext, ServiceNotReady, answer_question, assess_transaction
from financial_risk.serving.contract import frame_to_instances
from financial_risk.service.scoring import EndpointScorer, LocalScorer, score_frame

PREFIX = "/finrisk/dev"
MODEL_PTR, INDEX_PTR = f"{PREFIX}/model", f"{PREFIX}/index"
ENDPOINT = "finrisk-dev-fraud"


def model_pointer(version: int, endpoint: str = ENDPOINT) -> dict:
    return endpoint_pointer(version, endpoint)


def index_pointer(version: str) -> dict:
    return {"version": version, "uri": f"s3://{KB_BUCKET}/indexes/{version}/", "embeddings": "titan", "docs": 3, "chunks": 3,
            "promoted_at": "2026-10-16T10:00:00+00:00", "run": "https://github.example/run/2"}


class World:
    """SSM + S3 + a fake SageMaker + settings + the builders, wired together."""

    def __init__(self, tmp_path, demo_gold):
        self.tmp, self.gold = tmp_path, demo_gold
        self.ssm, self.s3, self.aws = FakeSSM(), FakeS3(), FakeEndpoints()
        self.clock = [1000.0]
        # gold_path holds the service's OWN transaction table (what /api/predict looks rows up in); no model is baked.
        self.settings = dataclasses.replace(
            get_settings(), pointer_prefix=PREFIX, pointer_ttl_seconds=60, artifact_dir=str(tmp_path / "work"),
            gold_path=demo_gold.parent, model_dir=tmp_path / "no-models", vector_index_dir=tmp_path / "no-index",
            startup_wait_seconds=30.0, pointer_refresh="background")

    def inline(self, timeout: float = 30.0) -> None:
        """Lambda's way: the request that finds the TTL expired does the work and waits for it (at most ``timeout`` s)."""
        self.settings = dataclasses.replace(self.settings, pointer_refresh="inline", inline_load_timeout_seconds=timeout)

    def publish_model(self, version: int, threshold: float | None = None, endpoint: str = ENDPOINT, status: str = "InService") -> dict:
        """Deploy a version to the fake endpoint AND write the pointer (what finrisk-ml's promote does)."""
        self.aws.deploy(endpoint, make_runner(self.gold, self.tmp / f"pkg{version}-{endpoint}", version, threshold), status)
        pointer = model_pointer(version, endpoint)
        self.ssm.set(MODEL_PTR, pointer)
        return pointer

    def publish_index(self, version: str, kind: str = "titan", docs: dict | None = None) -> dict:
        folder = self.tmp / f"idx-{version}"
        files = write_titan_index(folder, version, docs) if kind == "titan" else write_tfidf_index(folder, version, docs)
        pointer = index_pointer(version)
        pointer["uri"] = put_index(self.s3, version, files)
        pointer["embeddings"] = kind
        self.ssm.set(INDEX_PTR, pointer)
        return pointer

    def store(self, **kw) -> ArtifactStore:
        return ArtifactStore(self.settings, ssm_client=self.ssm, s3_client=self.s3, embedder_factory=lambda m: fake_embed,
                             clock=lambda: self.clock[0], scorer_factory=self.aws.scorer_factory, **kw)

    def context(self, **kw) -> ServiceContext:
        return ServiceContext(settings=self.settings, store=self.store(**kw))


@pytest.fixture
def world(tmp_path, demo_gold):
    return World(tmp_path, demo_gold)


def ready(ctx: ServiceContext) -> dict:
    return ctx.readiness()


def served_threshold(ctx: ServiceContext) -> float:
    """What the model being served says its threshold is (an endpoint reports it with every answer)."""
    return float(score_frame(ctx.model.scorer, ctx.model.features.head(1))["threshold"])


# ---------------------------------------------------------------------------- first load


def test_first_load_reads_both_pointers_and_reports_them(world):
    world.publish_model(7)
    world.publish_index("20261016-101500-ab12cd3")
    ctx = world.context()
    ctx.warm()
    info = ready(ctx)

    assert info["ready"] is True and info["missing"] == [] and info["load_error"] is None
    model, index = info["model"], info["index"]
    assert model["version"] == 7 and isinstance(model["version"], int)           # the pointer's number, as a JSON number
    assert model["source"] == "endpoint" and model["endpoint"] == ENDPOINT and model["status"] == "InService"
    assert model["load_error"] is None and model["error"] is None and model["loaded_at"].endswith("+00:00")
    assert model["threshold"] is None                                            # an endpoint reports it with its first answer
    assert index["version"] == "20261016-101500-ab12cd3" and index["source"] == "s3" and index["load_error"] is None
    assert (index["embeddings"], index["docs"], index["chunks"]) == ("titan", 3, 3)
    assert info["pointers_checked_at"].endswith("+00:00") and info["pointer_prefix"] == PREFIX
    assert world.ssm.calls == 1                                                   # one get_parameters call for both pointers
    assert world.aws.sagemaker.calls == [ENDPOINT]                                # one DescribeEndpoint (it also primed the 60 s status cache)
    assert world.aws.runtime.calls == []                                          # nothing was scored just to load: a cold endpoint costs a minute
    assert isinstance(ctx.model.scorer, EndpointScorer) and ctx.model.features is not None   # rows come from the app's own table


def test_the_service_scores_through_the_endpoint_and_answers_from_the_downloaded_index(world):
    world.publish_model(7, threshold=0.31)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    row_id = int(ctx.model.features["row_id"].iloc[0])
    result = assess_transaction(ctx, row_id)
    assert result["risk_level"] in {"high", "low"} and result["scored_by"] == "endpoint" and result["model_version"] == "7"
    assert result["threshold"] == pytest.approx(0.31) and result["reason_method"] == "shap" and result["reasons"]
    assert [c["EndpointName"] for c in world.aws.runtime.calls] == [ENDPOINT]
    assert ready(ctx)["model"]["threshold"] == pytest.approx(0.31)               # now /ready knows it
    answer = answer_question(ctx, "Who handles chargebacks and disputes?")
    assert not answer["refused"] and answer["citations"][0]["source"] == "docs/refunds.md"
    # the compliance refusal still comes first, whatever the index says
    assert answer_question(ctx, "Which stocks should I buy?")["refusal_reason"] == "out_of_policy"
    assert ctx.agent_context.index is ctx.index                                    # the agent searches the same loaded index
    assert ctx.agent_context.scorer is ctx.model.scorer                            # ... and scores through the same scorer


def test_start_up_returns_after_the_wait_and_loading_goes_on_in_the_background(world):
    world.publish_model(7)
    world.publish_index("v1")
    world.aws.sagemaker.gate = gate = threading.Event()                            # DescribeEndpoint is slow
    ctx = world.context()
    ctx.store.start(wait_seconds=0.05)                                             # gives up waiting quickly
    info = ready(ctx)
    assert info["ready"] is False and "model_artifact" in info["missing"]
    assert info["model"]["loading"] == "7" and info["hint"]
    with pytest.raises(ServiceNotReady, match="waiting for the first load"):
        _ = ctx.model
    gate.set()
    ctx.store.join()
    assert ready(ctx)["ready"] is True


# ---------------------------------------------------------------------------- a new version swaps in


def test_a_version_change_is_loaded_and_swapped_in(world):
    world.publish_model(7, threshold=0.31)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    old = ctx.model
    assert served_threshold(ctx) == pytest.approx(0.31)

    world.publish_model(8, threshold=0.62, endpoint="finrisk-dev-fraud-b")         # a new endpoint (a rebuilt environment)
    world.publish_index("v2", docs={**DOCS, "docs/new_policy.md": "# New Policy\n\nWire transfers above the limit need a second approver.\n"})
    world.clock[0] += 61
    assert ctx.store.maybe_refresh() is True                                       # a request noticed the TTL passed
    ctx.store.join()

    info = ready(ctx)
    assert info["model"]["version"] == 8 and info["model"]["endpoint"] == "finrisk-dev-fraud-b" and served_threshold(ctx) == pytest.approx(0.62)
    assert info["index"]["version"] == "v2" and info["index"]["docs"] == 4
    assert info["load_error"] is None
    assert ctx.model is not old
    assert old.scorer.score(frame_to_instances(old.features.head(1)))["threshold"] == pytest.approx(0.31)   # a request holding the old one still works
    assert answer_question(ctx, "Who approves wire transfers above the limit?")["citations"][0]["source"] == "docs/new_policy.md"


def test_the_same_endpoint_with_a_new_version_number_swaps_too(world):
    world.publish_model(7, threshold=0.31)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    world.publish_model(8, threshold=0.62)                                         # promote re-points the SAME endpoint at v8
    ctx.store.refresh(wait=True)
    assert ready(ctx)["model"]["version"] == 8 and served_threshold(ctx) == pytest.approx(0.62)


def test_the_same_version_is_not_loaded_again(world):
    world.publish_model(7)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    before = list(world.s3.requests), list(world.aws.sagemaker.calls)
    first = ctx.model
    ctx.store.refresh(wait=True)
    ctx.store.refresh(wait=True)
    assert (list(world.s3.requests), list(world.aws.sagemaker.calls)) == before and ctx.model is first


# ---------------------------------------------------------------------------- a failed load keeps the old one


def test_an_endpoint_that_does_not_exist_keeps_the_old_model_and_reports_load_error(world):
    world.publish_model(7, threshold=0.31)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()

    world.ssm.set(MODEL_PTR, model_pointer(9, "finrisk-dev-fraud-missing"))         # nobody ever deployed it
    ctx.store.refresh(wait=True)
    info = ready(ctx)
    assert info["ready"] is True                                                    # the old model still serves
    assert info["model"]["version"] == 7 and served_threshold(ctx) == pytest.approx(0.31)
    assert "Could not find endpoint" in info["model"]["load_error"] and info["load_error"] == info["model"]["load_error"]
    assert "123456789012" not in json.dumps(info) and "<account-id>" in info["model"]["load_error"]    # /ready is public
    assert info["index"]["load_error"] is None
    assert "keeps serving" in info["hint"]

    world.publish_model(10, threshold=0.5)                                          # the next good version recovers it
    ctx.store.refresh(wait=True)
    info = ready(ctx)
    assert info["model"]["version"] == 10 and info["model"]["load_error"] is None and info["load_error"] is None


@pytest.mark.parametrize("pointer, expected", [
    ({"version": 8, "model_data_url": f"s3://{ML_BUCKET}/models/job-8/output/model.tar.gz", "status": "Approved"},
     "old model pointer shape.*finrisk-ml"),                                        # the pre-SageMaker shape: refused, in words
    ({"version": 8, "serving": "endpoint"}, "'endpoint' must be a SageMaker endpoint name"),
    ({"version": 8, "serving": "endpoint", "endpoint": "has spaces/and slash"}, "'endpoint' must be a SageMaker endpoint name"),
    ({"version": 8, "serving": "container", "endpoint": ENDPOINT}, "'serving' must be"),
    ({"version": 8}, "'serving' must be"),
])
def test_a_model_pointer_in_the_wrong_shape_is_refused_and_the_old_one_stays(world, pointer, expected):
    import re

    world.publish_model(7)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    world.ssm.set(MODEL_PTR, pointer)
    ctx.store.refresh(wait=True)
    info = ready(ctx)
    assert info["model"]["version"] == 7 and info["ready"] is True
    assert re.search(expected, info["model"]["load_error"]), info["model"]["load_error"]
    assert "bad model pointer" in info["model"]["load_error"]


def test_the_old_pointer_shape_at_start_up_is_not_ready_and_does_not_fall_back_to_the_baked_model(world, tmp_path):
    baked = tmp_path / "baked-models"
    model_package_files(world.gold, baked)
    world.settings = dataclasses.replace(world.settings, model_dir=baked)           # a real baked model is right there
    world.ssm.set(MODEL_PTR, {"version": 3, "model_data_url": f"s3://{ML_BUCKET}/models/job-3/output/model.tar.gz"})
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    info = ready(ctx)
    assert info["ready"] is False and info["model"]["source"] is None
    assert "old model pointer shape" in info["model"]["load_error"] and "nothing older to serve" in info["hint"]


def test_a_pointer_that_cannot_be_loaded_at_start_up_does_not_fall_back_to_the_baked_model(world):
    # a real baked model exists on this machine -- and must NOT be used when the environment was told to serve version 9
    baked_models = world.tmp / "baked-models"
    files = model_package_files(world.gold, baked_models)
    world.settings = dataclasses.replace(world.settings, model_dir=baked_models, gold_path=world.gold.parent)
    world.ssm.set(MODEL_PTR, model_pointer(9))                                      # no such endpoint
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    info = ready(ctx)
    assert info["ready"] is False and info["model"]["source"] is None and "Could not find endpoint" in info["model"]["load_error"]
    assert info["index"]["version"] == "v1"                                         # the index is independent
    assert "nothing older to serve" in info["hint"]
    with pytest.raises(ServiceNotReady):
        _ = ctx.model
    assert files                                                                    # (the baked files really were there)
    # ... and it recovers by itself once the endpoint appears (the retry interval is short while nothing is loaded)
    world.publish_model(9)
    world.clock[0] += 6
    ctx.store.maybe_refresh()
    ctx.store.join()
    assert ready(ctx)["ready"] is True and ctx.model.source == "endpoint"


def test_an_unreadable_pointer_store_keeps_what_is_loaded(world):
    world.publish_model(7)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    world.ssm.error = ClientError({"Error": {"Code": "AccessDeniedException", "Message": f"not allowed for arn:aws:ssm:ap-southeast-2:{'123456789012'}:parameter/x"}}, "GetParameters")
    ctx.store.refresh(wait=True)
    info = ready(ctx)
    assert info["ready"] is True and "cannot read the pointers" in info["load_error"] and "AccessDenied" in info["load_error"]
    assert "123456789012" not in info["load_error"]
    world.ssm.error = None
    ctx.store.refresh(wait=True)
    assert ready(ctx)["load_error"] is None


def test_a_malformed_pointer_is_an_error_not_a_crash(world):
    world.publish_model(7)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    world.ssm.set(MODEL_PTR, "this is not json")
    world.ssm.set(INDEX_PTR, {"version": "v2", "uri": "https://example.com/index/"})
    ctx.store.refresh(wait=True)
    info = ready(ctx)
    assert info["ready"] is True and info["model"]["version"] == 7 and info["index"]["version"] == "v1"
    assert "bad model pointer" in info["model"]["load_error"] and "'uri' must be an s3:// address" in info["index"]["load_error"]


# ---------------------------------------------------------------------------- the TTL


def test_the_pointers_are_looked_at_again_only_after_the_ttl(world):
    world.publish_model(7)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    calls = world.ssm.calls
    for _ in range(50):                                                             # a busy service: many requests, one look
        ctx.maybe_refresh()
    world.clock[0] += 59
    ctx.maybe_refresh()
    ctx.store.join()
    assert world.ssm.calls == calls                                                 # still inside the 60 s
    world.clock[0] += 2
    ctx.maybe_refresh()
    ctx.maybe_refresh()
    ctx.store.join()
    assert world.ssm.calls == calls + 1                                             # exactly one look after it
    ctx.maybe_refresh()
    ctx.store.join()
    assert world.ssm.calls == calls + 1                                             # and the clock restarted


def test_the_ttl_comes_from_the_setting(world):
    world.settings = dataclasses.replace(world.settings, pointer_ttl_seconds=5)
    world.publish_model(7)
    world.publish_index("v1")
    store = world.store()
    store.start()
    calls = world.ssm.calls
    world.clock[0] += 6
    assert store.maybe_refresh() is True
    store.join()
    assert world.ssm.calls == calls + 1


# ---------------------------------------------------------------------------- nothing named -> the baked fallback


def _bake(world, tmp_path):
    models = tmp_path / "baked-models"
    model_package_files(world.gold, models)
    index = tmp_path / "baked-index"
    write_tfidf_index(index, version="baked")
    world.settings = dataclasses.replace(world.settings, model_dir=models, gold_path=world.gold.parent, vector_index_dir=index)


def test_without_a_pointer_prefix_the_baked_model_and_index_are_used(world, tmp_path):
    _bake(world, tmp_path)
    world.settings = dataclasses.replace(world.settings, pointer_prefix="")
    ctx = world.context()
    ctx.warm()
    info = ready(ctx)
    assert info["ready"] is True
    assert (info["model"]["source"], info["model"]["version"]) == ("baked", "baked")
    assert (info["index"]["source"], info["index"]["version"], info["index"]["embeddings"]) == ("baked", "baked", "tfidf")
    assert info["pointers_checked_at"] is None and info["pointer_prefix"] is None
    assert ctx.store.maybe_refresh() is False and world.ssm.calls == 0              # SSM is never touched


def test_null_pointers_and_missing_pointers_mean_baked(world, tmp_path):
    _bake(world, tmp_path)
    world.ssm.set(MODEL_PTR, {"version": None})                                     # what finrisk-infra creates
    # (no /index parameter at all)
    ctx = world.context()
    ctx.warm()
    info = ready(ctx)
    assert info["ready"] is True and info["model"]["source"] == "baked" and info["index"]["source"] == "baked"
    assert info["pointers_checked_at"] and info["load_error"] is None


def test_a_baked_model_is_scored_in_process_and_ready_says_so(world, tmp_path):
    _bake(world, tmp_path)
    world.settings = dataclasses.replace(world.settings, pointer_prefix="")
    ctx = world.context()
    ctx.warm()
    model = ready(ctx)["model"]
    assert (model["source"], model["endpoint"], model["status"], model["error"]) == ("baked", None, None, None)
    assert model["model_type"] and model["threshold"] is not None and model["pr_auc"] is not None
    assert isinstance(ctx.model.scorer, LocalScorer)
    row_id = int(ctx.model.features["row_id"].iloc[0])
    result = assess_transaction(ctx, row_id)
    assert result["scored_by"] == "baked" and result["model_version"] == "baked" and result["reason_method"] == "shap"
    assert world.aws.runtime.calls == [] and world.aws.sagemaker.calls == []        # AWS is never touched


def test_the_pointer_moves_baked_then_endpoint_then_null_and_the_scorer_follows(world, tmp_path):
    _bake(world, tmp_path)
    world.ssm.set(MODEL_PTR, {"version": None})
    ctx = world.context()
    ctx.warm()
    assert ctx.model.source == "baked" and isinstance(ctx.model.scorer, LocalScorer)
    world.publish_model(3)
    ctx.store.refresh(wait=True)
    assert ctx.model.source == "endpoint" and isinstance(ctx.model.scorer, EndpointScorer) and ready(ctx)["model"]["version"] == 3
    world.ssm.set(MODEL_PTR, {"version": None})                                     # rolled back to "nothing promoted"
    ctx.store.refresh(wait=True)
    assert ctx.model.source == "baked" and isinstance(ctx.model.scorer, LocalScorer) and ready(ctx)["model"]["version"] == "baked"


def test_a_null_pointer_with_nothing_baked_stops_serving_the_endpoint(world):
    world.publish_model(3)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    assert ctx.model.source == "endpoint"
    world.ssm.set(MODEL_PTR, {"version": None})                                     # e.g. teardown reset it, and no model is baked here
    ctx.store.refresh(wait=True)
    info = ready(ctx)
    assert info["ready"] is False and "model_artifact" in info["missing"] and info["model"]["load_error"] is None
    with pytest.raises(ServiceNotReady):
        _ = ctx.model


def test_nothing_baked_and_nothing_named_is_not_ready_and_not_an_error(world):
    world.ssm.set(MODEL_PTR, {"version": None})
    ctx = world.context()
    ctx.warm()
    info = ready(ctx)
    assert info["ready"] is False and set(info["missing"]) == {"gold_features", "model_artifact", "vector_index"}
    assert info["load_error"] is None and info["hint"]


# ---------------------------------------------------------------------------- the formats


def test_a_titan_index_embeds_the_question_with_the_manifest_model_and_ranks_by_cosine(world):
    folder = world.tmp / "t"
    write_titan_index(folder)
    from financial_risk.rag.index import load_index_dir

    seen = []

    def factory(manifest):
        seen.append(manifest)
        return fake_embed

    index = load_index_dir(folder, factory)
    assert index.method == "titan" and index.n_chunks == 3 and index.n_docs == 3 and index.version == "20261016-101500-ab12cd3"
    assert seen[0]["embedding_model"] == "amazon.titan-embed-text-v2:0" and seen[0]["dims"] == 16
    hits = retrieve("who handles chargebacks disputes team", index, top_k=2)
    assert hits[0]["source"] == "docs/refunds.md" and 0 < hits[0]["score"] <= 1.0001
    assert hits[0]["score"] >= hits[1]["score"]
    # the titan floor is higher than tfidf's: a dense embedding is never ~0 for an unrelated question
    from financial_risk.rag.index import min_score_for
    assert min_score_for(index) > 0.03


def test_the_titan_request_is_the_one_ingest_sends():
    class Client:
        def invoke_model(self, **request):
            self.request = request
            import io
            return {"body": io.BytesIO(json.dumps({"embedding": [3.0, 4.0], "inputTextTokenCount": 3}).encode())}

    client = Client()
    vector = TitanEmbedder("amazon.titan-embed-text-v2:0", 512, client=client)("what is pr-auc")
    assert client.request["modelId"] == "amazon.titan-embed-text-v2:0"
    assert client.request["contentType"] == "application/json" and client.request["accept"] == "application/json"
    assert json.loads(client.request["body"]) == {"inputText": "what is pr-auc", "dimensions": 512, "normalize": True}
    assert list(vector) == [3.0, 4.0]


def test_a_failing_embedding_call_is_retrieval_unavailable_not_a_crash():
    class Broken:
        def invoke_model(self, **request):
            raise ClientError({"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "InvokeModel")

    with pytest.raises(RetrievalUnavailable, match="AccessDenied"):
        TitanEmbedder(client=Broken(), dims=512)("hello")


def test_a_tfidf_index_in_the_shared_format_loads_and_answers(world):
    world.publish_model(7)
    world.publish_index("v1", kind="tfidf")
    ctx = world.context()
    ctx.warm()
    info = ready(ctx)
    assert info["index"]["embeddings"] == "tfidf" and info["index"]["version"] == "v1"
    hits = retrieve("chargebacks disputes team", ctx.index, top_k=1)
    assert hits[0]["source"].endswith("refunds.md")


def test_index_files_the_app_does_not_know_are_ignored(world):
    world.publish_model(7)
    pointer = world.publish_index("v1")
    manifest = json.loads(world.s3.objects[(KB_BUCKET, "indexes/v1/manifest.json")])
    assert "chunking" in manifest and ("indexes/v1/eval_report.json" in {k for _, k in world.s3.objects})
    ctx = world.context()
    ctx.warm()
    assert ready(ctx)["index"]["load_error"] is None and pointer["uri"].endswith("/")


def test_an_inconsistent_index_is_refused(world, tmp_path):
    world.publish_model(7)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    cases = {}
    files = write_titan_index(tmp_path / "bad1", "v2")
    manifest = json.loads(files["manifest.json"])
    cases["chunk count"] = ({**files, "manifest.json": json.dumps({**manifest, "chunks": 99}).encode()}, "manifest says 99 chunks")
    cases["dims"] = ({**files, "manifest.json": json.dumps({**manifest, "dims": 512}).encode()}, "16 dimensions, manifest says 512")
    cases["method"] = ({**files, "manifest.json": json.dumps({**manifest, "embeddings": "word2vec"}).encode()}, "unknown embeddings")
    cases["no embeddings"] = ({k: v for k, v in files.items() if k != "embeddings.npy"}, "NoSuchKey")
    for name, (broken, expected) in cases.items():
        world.s3.objects = {k: v for k, v in world.s3.objects.items() if not k[1].startswith("indexes/v2/")}
        pointer = index_pointer("v2")
        put_index(world.s3, "v2", broken)
        world.ssm.set(INDEX_PTR, pointer)
        ctx.store.refresh(wait=True)
        info = ready(ctx)
        assert info["index"]["version"] == "v1" and expected in info["index"]["load_error"], (name, info["index"])


def test_the_embedding_probe_at_load_catches_a_bad_bedrock_setup_before_the_swap(world):
    world.publish_model(7)
    world.publish_index("v1")
    calls = {"n": 0}

    def factory(manifest):
        def embed(text):
            calls["n"] += 1
            if calls["n"] > 1:                                                      # the first index loads; the second one cannot embed
                raise RetrievalUnavailable("embedding the question failed: AccessDeniedException")
            return fake_embed(text)
        return embed

    store = ArtifactStore(world.settings, ssm_client=world.ssm, s3_client=world.s3, embedder_factory=factory, clock=lambda: world.clock[0])
    ctx = ServiceContext(settings=world.settings, store=store)
    ctx.warm()
    world.publish_index("v2")
    store.refresh(wait=True)
    info = ready(ctx)
    assert info["index"]["version"] == "v1" and "AccessDenied" in info["index"]["load_error"]


def test_helpers_mask_account_ids_and_parse_addresses():
    assert mask("arn:aws:s3:::finrisk-ml-123456789012-ap-southeast-2-an") == "arn:aws:s3:::finrisk-ml-<account-id>-ap-southeast-2-an"
    assert parse_s3_uri(f"s3://{ML_BUCKET}/models/a/model.tar.gz") == (ML_BUCKET, "models/a/model.tar.gz")
    with pytest.raises(ValueError, match="expected s3://bucket/key"):
        parse_s3_uri("https://example.com/x")


# ---------------------------------------------------------------------------- inline mode (Lambda freezes background threads)


def _hold_model_loads(world):
    """Every DescribeEndpoint (the model's whole 'load') waits for ``gate`` (a slow AWS API). Returns the gate."""
    world.aws.sagemaker.gate = gate = threading.Event()
    return gate


def test_inline_mode_loads_in_the_request_that_finds_the_ttl_expired(world):
    world.inline()
    world.publish_model(7, threshold=0.31)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    assert served_threshold(ctx) == pytest.approx(0.31)

    world.publish_model(8, threshold=0.62, endpoint="finrisk-dev-fraud-b")   # another endpoint, so old and new answer differently
    world.publish_index("v2")
    world.clock[0] += 59
    assert ctx.store.maybe_refresh() is False                                      # inside the TTL: nothing happens
    assert served_threshold(ctx) == pytest.approx(0.31)

    world.clock[0] += 2
    assert ctx.store.maybe_refresh() is True                                       # this request does the look ...
    info = ready(ctx)                                                              # ... and there is NO join() here: it is already done
    assert info["model"]["version"] == 8 and served_threshold(ctx) == pytest.approx(0.62)
    assert info["index"]["version"] == "v2"
    assert info["load_error"] is None and info["model"]["loading"] is None and info["model"]["loading_note"] is None


def test_inline_mode_with_nothing_changed_is_just_one_fast_ssm_read(world):
    world.inline()
    world.publish_model(7)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    calls, downloads = world.ssm.calls, len(world.s3.requests)
    world.clock[0] += 61
    started = time.monotonic()
    assert ctx.store.maybe_refresh() is True
    assert time.monotonic() - started < 5
    assert world.ssm.calls == calls + 1 and len(world.s3.requests) == downloads     # looked, found the same versions, loaded nothing
    assert world.aws.sagemaker.calls == [ENDPOINT]                                  # (only the start-up DescribeEndpoint)


def test_inline_timeout_keeps_serving_the_old_one_and_the_next_request_carries_on(world):
    world.inline(timeout=0.3)
    world.publish_model(7, threshold=0.31)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    old = ctx.model
    world.publish_model(8, threshold=0.62, endpoint="finrisk-dev-fraud-b")   # another endpoint, so old and new answer differently
    gate = _hold_model_loads(world)
    world.clock[0] += 61
    calls = world.ssm.calls

    started = time.monotonic()
    assert ctx.store.maybe_refresh() is True
    assert 0.2 < time.monotonic() - started < 5                                    # it waited for the timeout, not for the load
    info = ready(ctx)
    assert ctx.model is old and info["model"]["version"] == 7 and info["ready"] is True    # the old model answers
    assert info["model"]["loading"] == "8" and "not finished after 0.3 s" in info["model"]["loading_note"]
    assert info["load_error"] is None                                              # slow is not failed: the promote pollers stop on load_error

    assert ctx.store.maybe_refresh() is False                                      # the next request joins the SAME load ...
    assert world.ssm.calls == calls + 1                                            # ... it did not look again,
    assert ctx.model is old                                                        # (and the old one still answers)
    gate.set()
    assert ctx.store.maybe_refresh() is False                                      # ... and finishes it
    info = ready(ctx)
    assert info["model"]["version"] == 8 and served_threshold(ctx) == pytest.approx(0.62)
    assert info["model"]["loading"] is None and info["model"]["loading_note"] is None and info["load_error"] is None
    assert world.aws.sagemaker.calls == [ENDPOINT, "finrisk-dev-fraud-b"]                       # version 8 was checked once, not twice (start-up + v8)


def test_inline_mode_keeps_the_failure_rules(world):
    world.inline()
    world.publish_model(7, threshold=0.31)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    world.ssm.set(MODEL_PTR, model_pointer(9, "finrisk-dev-fraud-missing"))         # nobody deployed version 9
    world.clock[0] += 61
    ctx.store.maybe_refresh()
    info = ready(ctx)
    assert info["ready"] is True and info["model"]["version"] == 7 and served_threshold(ctx) == pytest.approx(0.31)
    assert "Could not find endpoint" in info["model"]["load_error"] and ACCOUNT not in json.dumps(info)
    assert info["model"]["loading"] is None
    world.publish_model(10, threshold=0.5)                                          # while a failure is pending the retry is quick (5 s)
    world.clock[0] += 6
    ctx.store.maybe_refresh()
    info = ready(ctx)
    assert info["model"]["version"] == 10 and info["model"]["load_error"] is None


def test_inline_request_joins_a_load_still_going_from_start_up(world):
    world.inline()
    world.publish_model(7)
    world.publish_index("v1")
    gate = _hold_model_loads(world)
    ctx = world.context()
    ctx.store.start(wait_seconds=0.05)                                              # init phase gives up quickly (Lambda: 10 s)
    assert ready(ctx)["ready"] is False
    threading.Timer(0.3, gate.set).start()                                          # the download finishes shortly after
    ctx.store.maybe_refresh()                                                       # the first request waits for it (TTL not expired)
    assert ready(ctx)["ready"] is True and ctx.model.source == "endpoint"


def test_background_mode_does_not_wait_for_the_load(world):
    world.publish_model(7, threshold=0.31)
    world.publish_index("v1")
    ctx = world.context()
    ctx.warm()
    world.publish_model(8, threshold=0.62, endpoint="finrisk-dev-fraud-b")   # another endpoint, so old and new answer differently
    gate = _hold_model_loads(world)
    world.clock[0] += 61
    started = time.monotonic()
    assert ctx.store.maybe_refresh() is True
    assert time.monotonic() - started < 2                                           # returned at once ...
    assert served_threshold(ctx) == pytest.approx(0.31)                                # ... the old model answers
    assert ready(ctx)["model"]["loading_note"] is None                              # (no inline timeout machinery)
    gate.set()
    ctx.store.join()                                                                # the thread finishes on its own
    assert served_threshold(ctx) == pytest.approx(0.62)

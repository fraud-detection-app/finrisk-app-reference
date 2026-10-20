"""Fakes and builders for the artifact tests: an SSM that is a dictionary, an S3 that is a dictionary, a SageMaker
endpoint (a real ModelRunner behind a fake ``sagemaker-runtime`` client and a fake ``sagemaker`` client), a model package
and a knowledge index made exactly the way the other pipelines make them (the formats in service/artifacts.py's docstring).
Nothing here touches AWS."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import threading
from pathlib import Path

import joblib
import numpy as np
from botocore.exceptions import ClientError

from financial_risk.models.training import train_model
from financial_risk.rag.index import build_index, chunk_text
from financial_risk.serving.contract import InvalidRequest
from financial_risk.serving.runner import ModelRunner
from financial_risk.service.scoring import EndpointScorer

ACCOUNT = "123456789012"
ML_BUCKET = f"finrisk-ml-{ACCOUNT}-ap-southeast-2-an"
KB_BUCKET = f"finrisk-kb-{ACCOUNT}-ap-southeast-2-an"


class FakeSSM:
    def __init__(self, params: dict[str, dict | str | None] | None = None):
        self.params = {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in (params or {}).items()}
        self.error: Exception | None = None
        self.calls = 0

    def set(self, name: str, value: dict | str) -> None:
        self.params[name] = value if isinstance(value, str) else json.dumps(value)

    def get_parameters(self, Names):                                    # noqa: N803 - boto3's names
        self.calls += 1
        if self.error:
            raise self.error
        found = [{"Name": n, "Value": self.params[n]} for n in Names if n in self.params]
        return {"Parameters": found, "InvalidParameters": [n for n in Names if n not in self.params]}


class FakeS3:
    def __init__(self, objects: dict[tuple[str, str], bytes] | None = None):
        self.objects = dict(objects or {})
        self.requests: list[tuple[str, str]] = []

    def put(self, uri: str, data: bytes) -> None:
        bucket, key = re.fullmatch(r"s3://([^/]+)/(.+)", uri).groups()
        self.objects[(bucket, key)] = data

    def get_object(self, Bucket, Key):                                   # noqa: N803
        self.requests.append((Bucket, Key))
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": f"The specified key does not exist: {Bucket}/{Key}"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


def tarball(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def model_package_files(gold: Path, folder: Path, name: str = "model_v2.pkl", threshold: float | None = None) -> dict[str, bytes]:
    """The flat file set finrisk-ml writes: model_vN.pkl, latest_model.joblib {"model_path": "<file name>"}, metrics.json.
    (No features.parquet: 13b moved scoring to an endpoint and the package no longer carries the feature table.)"""
    folder.mkdir(parents=True, exist_ok=True)
    path, _ = train_model(gold, folder, model_type="baseline")
    artifact = joblib.load(path)
    if threshold is not None:
        artifact["threshold"] = threshold
    joblib.dump(artifact, folder / name)
    joblib.dump({"model_path": name}, folder / "latest_model.joblib")      # a FILE NAME, not a path
    if path.name != name:
        path.unlink()
    files = {p.name: p.read_bytes() for p in folder.iterdir()}
    files["metrics.json"] = json.dumps({"pr_auc": 0.9}).encode()
    return files


def make_runner(gold: Path, folder: Path, version: int | str, threshold: float | None = None) -> ModelRunner:
    """A real model (trained on the demo rows) as the code the SageMaker container runs."""
    model_package_files(gold, folder, threshold=threshold)
    return ModelRunner.from_package_dir(folder, str(version))


def client_error(code: str, message: str, operation: str, **extra) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}, **extra}, operation)


class FakeRuntime:
    """``sagemaker-runtime``: InvokeEndpoint answered by a real ModelRunner per endpoint name.

    ``errors`` is a queue: each call pops one and raises it. ``gate`` (a threading.Event) holds every call until set."""

    def __init__(self) -> None:
        self.endpoints: dict[str, ModelRunner] = {}
        self.calls: list[dict] = []
        self.errors: list[Exception] = []
        self.gate: threading.Event | None = None

    def invoke_endpoint(self, EndpointName, Body, ContentType=None, Accept=None, **kwargs):          # noqa: N803 - boto3's names
        self.calls.append({"EndpointName": EndpointName, "ContentType": ContentType, "Accept": Accept, "Body": Body, **kwargs})
        if self.gate is not None:
            self.gate.wait(10)
        if self.errors:
            raise self.errors.pop(0)
        runner = self.endpoints.get(EndpointName)
        if runner is None:
            raise client_error("ValidationError", f"Endpoint {EndpointName} of account {ACCOUNT} not found.", "InvokeEndpoint")
        try:
            response = runner.invoke(json.loads(Body))
        except InvalidRequest as exc:
            raise client_error("ModelError", "Received client error (400) from primary", "InvokeEndpoint",
                               OriginalStatusCode=400, OriginalMessage=str(exc)) from exc
        return {"Body": io.BytesIO(json.dumps(response).encode()), "ContentType": "application/json", "InvokedProductionVariant": "AllTraffic"}


class FakeSageMaker:
    """``sagemaker``: DescribeEndpoint from a dictionary of statuses. ``gate`` holds every call until set."""

    def __init__(self) -> None:
        self.statuses: dict[str, str] = {}
        self.calls: list[str] = []
        self.error: Exception | None = None
        self.gate: threading.Event | None = None

    def describe_endpoint(self, EndpointName):                                                      # noqa: N803
        self.calls.append(EndpointName)
        if self.gate is not None:
            self.gate.wait(10)
        if self.error is not None:
            raise self.error
        if EndpointName not in self.statuses:
            raise client_error("ValidationException",
                               f'Could not find endpoint "arn:aws:sagemaker:ap-southeast-2:{ACCOUNT}:endpoint/{EndpointName}".', "DescribeEndpoint")
        return {"EndpointName": EndpointName, "EndpointStatus": self.statuses[EndpointName]}


class FakeEndpoints:
    """The whole fake AWS side of the model: the runtime, the sagemaker client, and a ``scorer_factory`` for ArtifactStore."""

    def __init__(self) -> None:
        self.runtime, self.sagemaker = FakeRuntime(), FakeSageMaker()

    def deploy(self, name: str, runner: ModelRunner, status: str = "InService") -> None:
        self.runtime.endpoints[name] = runner
        self.sagemaker.statuses[name] = status

    def scorer_factory(self, endpoint: str, version, region: str, **kwargs) -> EndpointScorer:
        return EndpointScorer(endpoint, version, region, runtime=self.runtime, sagemaker=self.sagemaker, sleep=lambda s: None, **kwargs)


def endpoint_pointer(version: int, endpoint: str = "finrisk-dev-fraud") -> dict:
    """What finrisk-ml writes to /finrisk/<env>/model after promoting a version to a SageMaker endpoint."""
    return {"version": version, "serving": "endpoint", "endpoint": endpoint,
            "endpoint_config": f"{endpoint}-v{version}-20261021031500",
            "model_package_arn": f"arn:aws:sagemaker:ap-southeast-2:{ACCOUNT}:model-package/finrisk-fraud-model/{version}",
            "status": "Approved", "promoted_at": "2026-10-21T03:20:00+00:00", "run": "https://github.example/run/1"}


DOCS = {
    "docs/fraud_policy.md": "# Fraud Review Policy\n\nHigh-risk model outputs should be reviewed by an analyst. The model is a prioritisation aid, not an automated decision system.\n",
    "docs/faq.md": "# FAQ\n\nQ: Why use PR-AUC instead of accuracy?\nA: Fraud is rare, so a model can achieve high accuracy by predicting non-fraud. PR-AUC focuses on ranking the positive class.\n",
    "docs/refunds.md": "# Refund Handling\n\nChargebacks are handled by the disputes team within five working days of the customer's report.\n",
}

DIMS = 16


def fake_embed(text: str) -> np.ndarray:
    """A deterministic stand-in for Titan: hashed bag of words, unit length. Same text, same vector."""
    vector = np.zeros(DIMS, dtype=np.float32)
    for word in re.findall(r"[a-z0-9_]+", text.lower()):
        vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIMS] += 1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def write_titan_index(folder: Path, version: str = "20261016-101500-ab12cd3", docs: dict[str, str] | None = None) -> dict[str, bytes]:
    """An index in the knowledge repo's format with embeddings.npy (titan, 16 dims instead of 512)."""
    docs = docs or DOCS
    folder.mkdir(parents=True, exist_ok=True)
    chunks = []
    per_doc = {}
    for doc_path, text in docs.items():
        made = chunk_text(text, doc_path)
        per_doc[doc_path] = {"sha256": hashlib.sha256(text.encode()).hexdigest(), "chunks": len(made)}
        chunks += [{"id": hashlib.sha256(f"{doc_path}\n{c.text}".encode()).hexdigest()[:16], "doc_path": doc_path,
                    "section": c.section, "text": c.text, "sha256": hashlib.sha256(c.text.encode()).hexdigest()} for c in made]
    (folder / "chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    np.save(folder / "embeddings.npy", np.vstack([fake_embed(c["text"]) for c in chunks]).astype(np.float32))
    (folder / "manifest.json").write_text(json.dumps({
        "version": version, "created": "2026-10-16T10:15:00+00:00", "source_sha": "a" * 40,
        "embeddings": "titan", "embedding_model": "amazon.titan-embed-text-v2:0", "dims": DIMS,
        "docs": per_doc, "chunks": len(chunks), "chunking": {"max_words": 140, "overlap": 30}}), encoding="utf-8")
    (folder / "eval_report.json").write_text("{}")          # the knowledge repo publishes these too; the app ignores them
    (folder / "build_plan.json").write_text("{}")
    return {p.name: p.read_bytes() for p in folder.iterdir()}


def write_tfidf_index(folder: Path, version: str = "20261016-090000-0000000", docs: dict[str, str] | None = None) -> dict[str, bytes]:
    """A tfidf index: the same layout, built by the app's own build_index, with the manifest the knowledge repo writes."""
    docs = docs or DOCS
    source = folder.parent / (folder.name + "-md")
    source.mkdir(parents=True, exist_ok=True)
    for doc_path, text in docs.items():
        (source / Path(doc_path).name).write_text(text, encoding="utf-8")
    build_index([source], folder, seed_dir=None)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest.update(version=version, embedding_model="tfidf")
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return {p.name: p.read_bytes() for p in folder.iterdir()}


def put_index(s3: FakeS3, version: str, files: dict[str, bytes], bucket: str = KB_BUCKET) -> str:
    uri = f"s3://{bucket}/indexes/{version}/"
    for name, data in files.items():
        s3.put(uri + name, data)
    return uri

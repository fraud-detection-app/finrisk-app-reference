"""What the service serves: a model (a SageMaker endpoint) and a knowledge index, chosen by SSM pointers.

The image is built once and never changes between environments. What DOES change is which model and
which index an environment answers with. Two other pipelines own those:

    finrisk-ml         writes  /finrisk/<env>/model  = {"version": 3, "serving": "endpoint", "endpoint": "finrisk-prod-fraud", ...}
    finrisk-knowledge  writes  /finrisk/<env>/index  = {"version": "20261016-101500-ab12cd3", "uri": "s3://.../indexes/<v>/", ...}

This module is the reader. The rules, in plain words:

  1. START   Read both pointers (SSM Parameter Store) and load what they name.
  2. LAZY    On a request, if the last look at the pointers is older than POINTER_TTL_SECONDS (60),
             look again. WHO waits for it depends on POINTER_REFRESH (see "Two ways" below).
  3. SWAP    A new version is prepared NEXT to the running one. Only when it is ready does one assignment, under a
             lock, replace the old one. Requests that started earlier finish with the object they already hold;
             nobody sees half of each.
  4. FAIL    A load that fails changes nothing: the old artifact keeps serving and the error is written to
             ``error``, which /ready shows as ``load_error``.
  5. NOTHING A pointer whose version is null (or that does not exist yet), or no POINTER_PREFIX at all
             (a laptop, the tests), means: use the BAKED demo model and index inside the image.

THE MODEL POINTER (13b, SageMaker). The model is no longer downloaded and loaded into this process. A version is a
deployed SageMaker endpoint, and "loading" it means building an ``EndpointScorer`` (service/scoring.py) for it and
checking with DescribeEndpoint that the endpoint exists. Three shapes:

    {"version": null}                                            the baked demo model, in process (LocalScorer)
    {"version": 3, "serving": "endpoint", "endpoint": "finrisk-prod-fraud", "endpoint_config": "...",
     "model_package_arn": "...", "status": "Approved", "promoted_at": "...", "run": "..."}      an EndpointScorer
    {"version": 7, "model_data_url": "s3://.../model.tar.gz", ...}     the OLD (pre-SageMaker) shape: REFUSED, see below

The old shape asked this service to download a model package and score in-process. That path is gone: one way to
serve a model is easier to teach, to test and to reason about than two, and an environment must never be able to answer
from a model nobody deployed. A pointer in the old shape is a bad pointer: ``load_error`` says so in words, and
nothing falls back (at start-up that means not ready). To serve an older model, promote it again with finrisk-ml.

A pointer that names a version we cannot load at start-up does NOT fall back to the baked model:
the environment was told to serve a specific artifact, and serving a different one without saying
so is worse than being not-ready. /ready answers 503 with ``load_error`` until the next look works.
The same holds after start-up for scoring: a configured endpoint that fails is a 503 (ModelUnavailable), never the baked model.

Two ways to do the look (POINTER_REFRESH):

  background  (ECS Express, a laptop; the default there)  The request starts a thread and returns at once.
              The load carries on in that thread while the process keeps running.
  inline      (Lambda; the default there, found by AWS_LAMBDA_FUNCTION_NAME)  Lambda FREEZES the process the
              moment a response is sent, so a background thread only moves while a request is being served.
              With a poller sending one tiny request every 10 s, a 10 s download took ~5 minutes (measured in
              the week 13b rehearsal: 309 s on dev, 327 s on sit, against 11 s on ECS). So here the request that
              finds the TTL expired does the work itself: it waits for the look and, only if a version moved,
              for the load -- at most INLINE_LOAD_TIMEOUT_SECONDS (20). If that runs out, the request answers
              with the OLD artifact, ``loading`` / ``loading_note`` on /ready say a load is still going, and
              the next request joins the same load and waits again. No second download starts. Same atomic
              swap, same failure rules as above. A load that has failed is NOT reported while it is only slow:
              ``load_error`` stays a verdict, because the promote pipelines stop on it.
              (Pointing at an endpoint is a single DescribeEndpoint call, so only the knowledge index is ever slow.)

/health never calls any of this (it does not touch the store at all). Nothing waits at TTL expiry unless a
pointer moved: an unchanged pointer is one fast SSM read.

Formats read (the contract with the other repos):

  endpoint        the contract in serving/contract.py (docs/api_schema.md). The model package itself (``model.tar.gz``,
                  read by serving/sagemaker_server.py through models/package.py) never reaches this process.
  index folder    s3://<kb bucket>/indexes/<version>/ with manifest.json, chunks.jsonl and either
                  embeddings.npy (titan) or tfidf_vectorizer.joblib + tfidf_matrix.joblib (tfidf).
                  The files are fetched by name with GetObject (no ListBucket permission needed).
  feature table   /api/predict/<row_id> scores a stored transaction. The rows come from THIS service's own feature table
                  (the baked demo gold table, with hour_of_day); the model package no longer carries one.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from financial_risk.config import Settings
from financial_risk.models.package import unpack_model_package  # noqa: F401 - re-exported: finrisk-ml and the tests use it from here too
from financial_risk.models.training import FEATURE_COLUMNS
from financial_risk.rag.index import (
    EmbedderFactory,
    LoadedIndex,
    default_embedder_factory,
    load_index_dir,
)
from financial_risk.serving.runner import ModelRunner
from financial_risk.service.masking import mask  # noqa: F401 - re-exported
from financial_risk.service.scoring import EndpointScorer, LocalScorer, Scorer

logger = logging.getLogger("financial_risk.service")

# While nothing is loaded yet, or the last attempt failed, look again sooner than the TTL.
RETRY_SECONDS = 5.0
FEATURE_TABLE = "transaction_features.parquet"
ENDPOINT_NAME = re.compile(r"[A-Za-z0-9](-*[A-Za-z0-9]){0,62}")          # SageMaker's rule: letters, digits, hyphens, <= 63

def parse_s3_uri(uri: str) -> tuple[str, str]:
    m = re.fullmatch(r"s3://([^/]+)/(.+)", uri.strip())
    if not m:
        raise ValueError(f"not an S3 address (expected s3://bucket/key): {mask(uri)!r}")
    return m.group(1), m.group(2)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------ what gets served


@dataclass(eq=False)
class ModelBundle:
    """The model being served -- a scorer (in process, or a SageMaker endpoint) -- and the feature table that
    ``/api/predict/<row_id>`` looks rows up in. Always replaced as one unit."""

    version: Any                      # the pointer's version (a whole number), or "baked"
    source: str                       # "endpoint" | "baked"
    loaded_at: str
    loaded_ms: float
    scorer: Scorer
    features: pd.DataFrame | None     # the service's own transaction table; None = /api/predict has nothing to look up
    endpoint: str | None = None
    model_dir: Path | None = None     # baked only
    gold_path: Path | None = None

    @property
    def threshold(self) -> float | None:
        return self.scorer.threshold

    def summary(self) -> dict[str, Any]:
        """model_type, trained_at_utc, threshold, pr_auc (baked) / endpoint, status, error (endpoint) -- for /ready."""
        return self.scorer.facts()


@dataclass(eq=False)
class IndexBundle:
    """A loaded knowledge index plus where it came from."""

    version: Any                      # the pointer's version string, or "baked"
    source: str                       # "s3" | "baked"
    loaded_at: str
    loaded_ms: float
    index: LoadedIndex
    tmp_dir: Path | None = None

    def summary(self) -> dict[str, Any]:
        return {"embeddings": self.index.method, "docs": self.index.n_docs, "chunks": self.index.n_chunks}


@dataclass
class Slot:
    """One kind of artifact (model or index): what is serving, what is loading, what went wrong."""

    kind: str
    bundle: Any = None
    error: str | None = None
    loading: str | None = None        # the version being loaded right now, if any
    pointer: dict[str, Any] | None = field(default=None, repr=False)   # the last pointer we read
    loading_note: str | None = None   # inline mode: "still going after N s" (NOT an error; load_error stays a verdict)


# ------------------------------------------------------------------------------ loading a model


def read_feature_table(path: Path, required: bool) -> pd.DataFrame | None:
    """The service's own transaction table (``row_id`` + the model's features + labels). ``required``: raise when it is
    missing or lacks columns (the baked model cannot work without it); otherwise return None and log why."""
    try:
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        features = pd.read_parquet(path)
        missing = [c for c in ["row_id", *FEATURE_COLUMNS] if c not in features.columns]
        if missing:
            raise ValueError(f"{path.name} lacks the columns the model needs: {missing} (rebuild it with scripts/run_pipeline.py)")
        return features
    except Exception as exc:
        if required:
            raise
        logger.warning("features: no usable feature table (%s); /api/predict/<row_id> will answer 503, /api/score still works", exc)
        return None


def load_baked_model_bundle(model_dir: Path, gold_path: Path) -> ModelBundle:
    """The demo model baked into the image (or built by scripts/run_pipeline.py) as an in-process LocalScorer.

    Raises on any problem: the caller keeps the old bundle."""
    started = time.perf_counter()
    features = read_feature_table(gold_path, required=True)
    # ModelRunner.from_package_dir loads the package AND scores + explains one row: a model pickled by other library
    # versions, or trained on other columns, fails HERE instead of on the first user request.
    background = features.sample(min(200, len(features)), random_state=42) if len(features) >= 20 else None
    runner = ModelRunner.from_package_dir(model_dir, "baked", background)
    return ModelBundle(
        version="baked", source="baked", loaded_at=now_iso(), loaded_ms=round((time.perf_counter() - started) * 1000, 1),
        scorer=LocalScorer(runner, "baked"), features=features, model_dir=model_dir, gold_path=gold_path,
    )


# ------------------------------------------------------------------------------ the store


class ArtifactStore:
    """Holds the model and the index being served, and keeps them in step with the SSM pointers.

    Every collaborator is injectable (``ssm_client``, ``s3_client``, ``embedder_factory``, ``scorer_factory``), which is how
    the tests run the whole thing without AWS."""

    def __init__(
        self,
        settings: Settings,
        ssm_client: Any | None = None,
        s3_client: Any | None = None,
        embedder_factory: EmbedderFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
        scorer_factory: Callable[[str, Any, str], EndpointScorer] | None = None,
    ) -> None:
        self.settings = settings
        self.prefix = settings.pointer_prefix.rstrip("/")
        self.ttl = max(0, int(settings.pointer_ttl_seconds))
        self.refresh_mode = settings.pointer_refresh                 # "inline" | "background"
        self.inline_timeout = max(0.0, float(settings.inline_load_timeout_seconds))
        self.model = Slot("model")
        self.index = Slot("index")
        self.pointers_checked_at: str | None = None
        self._ssm, self._s3 = ssm_client, s3_client
        self._embedder_factory = embedder_factory or default_embedder_factory(settings.aws_region, settings.embedding_model_id)
        self._clock = clock
        # (endpoint name, version, region) -> EndpointScorer. Tests hand in one built on fake boto3 clients. The default
        # carries the ENDPOINT_*_TIMEOUT_SECONDS settings (Terraform shortens the read timeout on Lambda).
        self._scorer_factory = scorer_factory or (lambda endpoint, version, region: EndpointScorer(
            endpoint, version, region, connect_timeout=settings.endpoint_connect_timeout_seconds,
            read_timeout=settings.endpoint_read_timeout_seconds))
        self._lock = threading.RLock()
        self._started = False
        self._last_check: float | None = None
        self._checking = False
        self._threads: list[threading.Thread] = []

    # ---- public

    @property
    def pointers_enabled(self) -> bool:
        return bool(self.prefix)

    def start(self, wait_seconds: float | None = None) -> "ArtifactStore":
        """Load the first versions. Pointer mode waits at most ``wait_seconds`` (default: the setting);
        what is still loading then finishes in the background and /ready says 503 until it does."""
        with self._lock:
            if self._started:
                return self
            self._started = True
        if not self.pointers_enabled:
            self._load_baked(self.model)
            self._load_baked(self.index)
            return self
        wait = self.settings.startup_wait_seconds if wait_seconds is None else wait_seconds
        self._last_check = self._clock()
        self.refresh(wait=False)
        deadline = time.monotonic() + wait
        for thread in list(self._threads):
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return self

    def ensure_started(self) -> None:
        if not self._started:
            self.start()

    def maybe_refresh(self) -> bool:
        """Called on every request (never by /health). Cheap: when the pointers were read less than TTL seconds
        ago and nothing is still loading it returns at once. Otherwise it starts a look in a thread, and then
        either returns without waiting (background mode) or waits for it, at most ``inline_timeout`` seconds
        (inline mode, Lambda). Returns True when THIS call started a new look."""
        if not self.pointers_enabled or not self._started:
            return False
        inline = self.refresh_mode == "inline"
        started = False
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            # Inline: a look or a load from an earlier request is still going (it only moves while a request is
            # being served). Join it instead of starting another one.
            carrying_on = inline and bool(self._threads)
            if not carrying_on:
                now = self._clock()
                healthy = self.model.bundle is not None and self.index.bundle is not None and not (self.model.error or self.index.error)
                interval = self.ttl if healthy else min(self.ttl, RETRY_SECONDS)
                if self._checking or (self._last_check is not None and now - self._last_check < interval):
                    return False
                self._last_check = now
                thread = threading.Thread(target=self.refresh, kwargs={"wait": True}, name="pointer-check", daemon=True)
                thread.start()
                self._threads.append(thread)
                started = True
        if inline:
            self._wait_inline()
        return started

    def _wait_inline(self) -> None:
        """Wait for the loads in flight, at most ``inline_timeout`` seconds in total. A python thread cannot be
        stopped, so on a timeout the work goes on in its thread; the next request joins it and waits again."""
        deadline = time.monotonic() + self.inline_timeout
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if not any(thread.is_alive() for thread in threads):
            return
        note = f"not finished after {self.inline_timeout:g} s; the old version keeps serving and the next request carries on"
        with self._lock:
            for slot in (self.model, self.index):
                if slot.loading is not None:          # checked under the lock: a load that just finished has cleared it
                    slot.loading_note = note
        logger.warning("pointers: a look/load is %s", note)

    def refresh(self, wait: bool = True) -> None:
        """Read both pointers now and load whatever moved. ``wait=True`` loads in this thread (tests, the
        background check); ``wait=False`` starts a thread per load and returns."""
        with self._lock:
            if self._checking:
                return
            self._checking = True
        try:
            tasks = self._read_pointers()
        finally:
            with self._lock:
                self._checking = False
        for slot, pointer in tasks:
            if wait:
                self._load(slot, pointer)
            else:
                thread = threading.Thread(target=self._load, args=(slot, pointer), name=f"load-{slot.kind}", daemon=True)
                thread.start()
                self._threads.append(thread)

    def join(self, timeout: float = 30.0) -> None:
        """Wait for loads in flight (tests)."""
        deadline = time.monotonic() + timeout
        for thread in list(self._threads):
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

    # ---- pointers

    def _ssm_client(self) -> Any:
        if self._ssm is None:
            import boto3
            from botocore.config import Config

            self._ssm = boto3.client("ssm", region_name=self.settings.aws_region, config=Config(
                connect_timeout=3, read_timeout=5, retries={"max_attempts": 2, "mode": "standard"}))
        return self._ssm

    def _s3_client(self) -> Any:
        if self._s3 is None:
            import boto3
            from botocore.config import Config

            # Short timeouts: the Lambda start-up phase has 10 seconds in total.
            self._s3 = boto3.client("s3", region_name=self.settings.aws_region, config=Config(
                connect_timeout=3, read_timeout=15, retries={"max_attempts": 2, "mode": "standard"}))
        return self._s3

    def _read_pointers(self) -> list[tuple[Slot, dict[str, Any] | None]]:
        """-> the loads to start: [(slot, pointer)], pointer None meaning 'baked'."""
        names = {"model": f"{self.prefix}/model", "index": f"{self.prefix}/index"}
        try:
            response = self._ssm_client().get_parameters(Names=list(names.values()))
        except Exception as exc:  # noqa: BLE001 - SSM down or denied: keep what we have, say why.
            message = mask(f"cannot read the pointers: {type(exc).__name__}: {exc}")[:300]
            logger.error("pointers: %s", message)
            with self._lock:
                for slot in (self.model, self.index):
                    slot.error = message
                self.pointers_checked_at = now_iso()
            return []
        values = {p["Name"]: p["Value"] for p in response.get("Parameters", [])}
        tasks: list[tuple[Slot, dict[str, Any] | None]] = []
        with self._lock:
            self.pointers_checked_at = now_iso()
            for slot in (self.model, self.index):
                raw = values.get(names[slot.kind])
                try:
                    pointer = json.loads(raw) if raw else {"version": None}
                    if not isinstance(pointer, dict):
                        raise ValueError("the pointer is not a JSON object")
                    wanted = pointer.get("version")
                    if wanted is not None:
                        self._validate_pointer(slot.kind, pointer)
                except ValueError as exc:
                    slot.error = mask(f"bad {slot.kind} pointer {names[slot.kind]}: {exc}")[:300]
                    logger.error("pointers: %s", slot.error)
                    continue
                slot.pointer = pointer
                current = slot.bundle.version if slot.bundle is not None else None
                if wanted is None:
                    if slot.bundle is not None and slot.bundle.source == "baked":
                        slot.error = None
                        continue
                    pointer_to_load = None
                else:
                    if slot.bundle is not None and str(current) == str(wanted) and self._same_target(slot, pointer):
                        slot.error = None            # we are serving exactly what the pointer says
                        continue
                    pointer_to_load = pointer
                if slot.loading is None:
                    tasks.append((slot, pointer_to_load))
        return tasks

    @staticmethod
    def _same_target(slot: Slot, pointer: dict[str, Any]) -> bool:
        """Is the bundle in this slot already what the pointer names? A model pointer names a version AND an endpoint:
        the same version on another endpoint (a rebuilt environment) is a different target."""
        if slot.kind == "model":
            return getattr(slot.bundle, "endpoint", None) == pointer.get("endpoint")
        return True

    @staticmethod
    def _validate_pointer(kind: str, pointer: dict[str, Any]) -> None:
        if kind == "model":
            if "serving" not in pointer and "model_data_url" in pointer:
                raise ValueError(
                    "this is the old model pointer shape (model_data_url): the service no longer downloads and loads models "
                    "itself. Promote the version again with finrisk-ml, which writes "
                    "{\"version\", \"serving\": \"endpoint\", \"endpoint\"}")
            if pointer.get("serving") != "endpoint":
                raise ValueError(f"'serving' must be \"endpoint\" (got {pointer.get('serving')!r})")
            endpoint = pointer.get("endpoint")
            if not isinstance(endpoint, str) or not ENDPOINT_NAME.fullmatch(endpoint):
                raise ValueError(f"'endpoint' must be a SageMaker endpoint name (letters, digits and hyphens; got {endpoint!r})")
            return
        value = pointer.get("uri")
        if not isinstance(value, str) or not value.startswith("s3://"):
            raise ValueError("'uri' must be an s3:// address")

    # ---- loading (runs in a thread; never raises)

    def _load(self, slot: Slot, pointer: dict[str, Any] | None) -> None:
        wanted = "baked" if pointer is None else pointer.get("version")
        with self._lock:
            if slot.loading is not None:
                return
            slot.loading = str(wanted)
        try:
            if pointer is None:
                bundle = self._build_baked(slot.kind)
                if bundle is None:               # nothing baked on this machine: not ready, and not an error
                    with self._lock:
                        # The pointer went back to "nothing promoted" and there is no baked model to go back to:
                        # serving the previous endpoint would be answering from a version nobody asked for any more.
                        if slot.bundle is not None and slot.bundle.source != "baked":
                            old, slot.bundle = slot.bundle, None
                            logger.warning("%s: the pointer is null and nothing is baked -- no longer serving version %s", slot.kind, old.version)
                    return
            elif slot.kind == "model":
                bundle = self._load_endpoint(pointer)
            else:
                bundle = self._fetch_index(pointer)
            with self._lock:
                old, slot.bundle, slot.error = slot.bundle, bundle, None     # THE swap: one assignment
                slot.loading, slot.loading_note = None, None                 # same lock: nobody sees "loading" next to the new one
            self._discard(old)
            logger.info("%s: now serving version %s (%s%s) -- loaded in %.0f ms", slot.kind, bundle.version, bundle.source,
                        f" {bundle.endpoint}" if getattr(bundle, "endpoint", None) else "", bundle.loaded_ms)
        except Exception as exc:  # noqa: BLE001 - keep the old one, report the new one's failure.
            with self._lock:
                slot.error = mask(f"{type(exc).__name__}: {exc}")[:300]
                slot.loading, slot.loading_note = None, None
            logger.error("%s: could not load version %s -- %s%s", slot.kind, wanted, slot.error,
                         " (keeping the old one)" if slot.bundle is not None else " (nothing to fall back to)")
        finally:
            with self._lock:
                slot.loading, slot.loading_note = None, None

    def _load_baked(self, slot: Slot) -> None:
        self._load(slot, None)

    @staticmethod
    def _discard(old: Any) -> None:
        tmp = getattr(old, "tmp_dir", None)
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

    def _build_baked(self, kind: str) -> Any | None:
        s = self.settings
        if kind == "model":
            gold = s.gold_path / FEATURE_TABLE
            has_model = s.model_dir.exists() and (any(s.model_dir.glob("*.joblib")) or any(s.model_dir.glob("model_v*.pkl")))
            if not gold.exists() or not has_model:
                logger.info("model: nothing baked at %s / %s", gold, s.model_dir)
                return None
            return load_baked_model_bundle(s.model_dir, gold)
        started = time.perf_counter()
        if not (s.vector_index_dir / "manifest.json").exists():
            logger.info("index: nothing baked at %s", s.vector_index_dir)
            return None
        loaded = load_index_dir(s.vector_index_dir, self._embedder_factory)
        return IndexBundle("baked", "baked", now_iso(), round((time.perf_counter() - started) * 1000, 1), loaded)

    def _work_dir(self, kind: str, version: Any) -> Path:
        base = Path(self.settings.artifact_dir)
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{kind}-{re.sub(r'[^A-Za-z0-9._-]', '_', str(version))}-{uuid.uuid4().hex[:8]}"

    def _load_endpoint(self, pointer: dict[str, Any]) -> ModelBundle:
        """A version that is a SageMaker endpoint: build the scorer and check the endpoint exists (DescribeEndpoint).

        Nothing is downloaded and nothing is scored here (a cold Serverless endpoint would take a minute and cost a
        wake-up): the promote pipeline already smoke-tested the endpoint before it moved the pointer. A wrong or deleted
        endpoint fails THIS check, so the old version keeps serving and /ready says why."""
        started = time.perf_counter()
        scorer = self._endpoint_scorer(pointer)
        status = scorer.verify()
        logger.info("model: endpoint %s is %s", scorer.endpoint, status)
        features = read_feature_table(self.settings.gold_path / FEATURE_TABLE, required=False)
        return ModelBundle(version=pointer["version"], source="endpoint", loaded_at=now_iso(),
                           loaded_ms=round((time.perf_counter() - started) * 1000, 1), scorer=scorer, features=features,
                           endpoint=pointer["endpoint"])

    def _endpoint_scorer(self, pointer: dict[str, Any]) -> EndpointScorer:
        return self._scorer_factory(pointer["endpoint"], pointer["version"], self.settings.aws_region)

    def _fetch_index(self, pointer: dict[str, Any]) -> IndexBundle:
        started = time.perf_counter()
        bucket, prefix = parse_s3_uri(pointer["uri"])
        prefix = prefix.rstrip("/") + "/"
        dest = self._work_dir("index", pointer["version"])
        dest.mkdir(parents=True)
        s3 = self._s3_client()

        def get(name: str) -> None:
            body = s3.get_object(Bucket=bucket, Key=prefix + name)["Body"]
            with (dest / name).open("wb") as handle:
                shutil.copyfileobj(body, handle)

        try:
            get("manifest.json")
            manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
            method = manifest.get("embeddings")
            if method not in {"titan", "tfidf"}:
                raise ValueError(f"manifest.json: unknown embeddings {method!r} (expected 'titan' or 'tfidf')")
            files = ["chunks.jsonl", "embeddings.npy"] if method == "titan" \
                else ["chunks.jsonl", "tfidf_vectorizer.joblib", "tfidf_matrix.joblib"]
            for name in files:
                get(name)
            loaded = load_index_dir(dest, self._embedder_factory)
            if loaded.method == "titan":
                loaded.embedder("health check")           # a real embedding call: wrong Region / no permission fails HERE
            if str(manifest.get("version")) != str(pointer["version"]):
                logger.warning("index: pointer says %s but manifest.json says %s", pointer["version"], manifest.get("version"))
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            raise
        return IndexBundle(pointer["version"], "s3", now_iso(), round((time.perf_counter() - started) * 1000, 1), loaded, tmp_dir=dest)

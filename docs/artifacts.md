# What the service serves: the model and the knowledge index

The image is built **once** and is the same in dev, sit, pre-prod and prod. What differs between environments is
which **model** and which **knowledge index** the service answers with. Two other pipelines own those, and tell
this service through two SSM parameters per environment:

| Parameter | Written by | Value |
|---|---|---|
| `/finrisk/<env>/model` | finrisk-ml (promote) | `{"version": 3, "serving": "endpoint", "endpoint": "finrisk-prod-fraud", "endpoint_config": "finrisk-prod-fraud-v3-20261021-031500", "model_package_arn": "arn:aws:sagemaker:...:model-package/finrisk-fraud-model/3", "status": "Approved", "promoted_at": "...", "run": "..."}` |
| `/finrisk/<env>/index` | finrisk-knowledge (promote) | `{"version": "20261016-101500-ab12cd3", "uri": "s3://<kb bucket>/indexes/<version>/", "embeddings": "titan", "docs": 12, "chunks": 80, "promoted_at": "...", "run": "..."}` |

finrisk-infra creates both with the value `{"version": null}`. **A null pointer means "use the baked demo model/index".**
This service only ever *reads* them (`POINTER_PREFIX=/finrisk/<env>`), and publishes one parameter of its own,
`/finrisk/<env>/app_url`, so the other pipelines can find `/ready`.

## The model is an endpoint, not a file (13b)

Until 13b the service downloaded a model package from S3 and scored in its own memory. Now a model **version is a deployed
SageMaker endpoint**; the pointer names it, and "loading" it means *building a scorer for it* (`service/scoring.py`):

| Model pointer | The service scores with | `/ready.model.source` |
|---|---|---|
| `{"version": null}` (or no parameter, or no `POINTER_PREFIX`) | `LocalScorer`: the **baked** demo model in this process | `baked` |
| `{"version": N, "serving": "endpoint", "endpoint": "<name>", ...}` | `EndpointScorer`: `sagemaker-runtime` `InvokeEndpoint` on `<name>` | `endpoint` |
| `{"version": N, "model_data_url": "s3://...", ...}` (the **old** 13b shape) | nothing: **refused** | (the previous model keeps serving) |

**The old pointer shape is refused, on purpose.** Keeping an in-process loading path next to the endpoint path would give
two ways to serve a model to teach, test and reason about, and an environment must never answer from a model nobody deployed.
An old-shaped pointer is a *bad pointer*: `load_error` says `old model pointer shape (model_data_url): the service no longer
downloads and loads models itself. Promote the version again with finrisk-ml...`, the previous model keeps serving, and at
start-up (nothing older to serve) the service is not ready. It never falls back to the baked model.

**No silent fallback, ever.** A configured endpoint that fails is **HTTP 503** (`model_unavailable`), not an answer from the
baked model: a model that answers differently from the one you deployed is worse than an error you can see. The cases:

| What happened | Result |
|---|---|
| read timeout (40 s) or `ModelNotReadyException` (a cold Serverless endpoint takes about a minute to wake up) | 503 `"model endpoint is starting, retry"`, `Retry-After`; the next call works |
| throttling, a 5xx from the service, a connection that fails fast | up to 2 quick retries inside a 50 s budget (a Lambda has 60), then 503 |
| the endpoint does not exist / access denied / the container rejected the request | 503 with the (account-id-masked) reason, no retry |
| the pointer names an endpoint that does not exist when it moves | the **load fails**: old model keeps serving, `load_error` explains |

`/ready.model` for an endpoint: `{"version": 3, "source": "endpoint", "endpoint": "finrisk-prod-fraud", "status": "InService",
"error": null, "threshold": 0.4812, ...}`. `status` is `DescribeEndpoint`, cached 60 s (so a 10-second promote poller does not
hammer the API); `error` is the last scoring or describe problem and clears on the next success; `threshold` is what the
endpoint reported with its last answer (`null` until the first score: nothing is scored just to load, a cold endpoint would
cost a minute). The runtime role needs `sagemaker:InvokeEndpoint` and `sagemaker:DescribeEndpoint` on
`endpoint/finrisk-<env>-fraud` (finrisk-infra grants both).

Switching is the same swap as before: the pointer moves, the next look builds the new scorer and checks the endpoint, one
assignment replaces the old one. baked -> endpoint -> `{"version": null}` -> baked works in both directions. If the pointer
goes back to null and nothing is baked on this machine, the service stops serving the endpoint (not ready) rather than keep
answering from a version nobody asked for any more.

## How a new version reaches the running service

1. **Start-up** — read both pointers, load what they name (the model: build the scorer and `DescribeEndpoint`; the index:
   download it; waiting at most `STARTUP_WAIT_SECONDS`, 6 s; a slower load carries on in the background and `/ready` is 503 meanwhile).
2. **Lazily** — on a request, if the last look is older than `POINTER_TTL_SECONDS` (60) the service looks again.
   Who waits depends on `POINTER_REFRESH` (below). `/health` never triggers a look.
3. **Swap** — a changed `version` is loaded *next to* the running one. The new endpoint must exist (`DescribeEndpoint`; it is
   not invoked: the promote pipeline smoke-tested it before moving the pointer); the new index must embed one test question.
   Only then one assignment (under a lock) replaces the old object.
   Requests already running finish with the old one they hold.
4. **Failure** — nothing changes. The old artifact keeps serving and `/ready` shows `load_error`. The next look retries
   (every 5 s while a failure is pending, otherwise every TTL).
5. **Start-up with a bad pointer** — if there is *nothing older to serve*, the service is not ready (503). It does
   **not** quietly fall back to the baked model: the environment was told to serve a specific version.

### Two ways to look: `POINTER_REFRESH`

| Mode | Where (default) | What the request that finds the TTL expired does |
|---|---|---|
| `background` | ECS Express Mode, a laptop | Starts a thread and answers at once. The download and load carry on in the thread. |
| `inline` | Lambda (found by `AWS_LAMBDA_FUNCTION_NAME`) | Does the look itself and, only if a version moved, the load, waiting at most `INLINE_LOAD_TIMEOUT_SECONDS` (20). |

Why two: Lambda **freezes** the process the moment a response is sent. A background thread only moves while a request is
being served, and a promote pipeline polling `/ready` sends one tiny request every 10 s. In the week 13b rehearsal the
new index reached dev after 309 s and sit after 327 s, against 11 s on ECS, because of exactly that (and each Lambda
environment loads on its own).

Inline mode keeps every rule above (load next to the old one, test prediction, one atomic swap, a failure keeps the old
one and shows in `load_error`). Plus one more:

- **Timeout** — if the load is not finished after `INLINE_LOAD_TIMEOUT_SECONDS`, the request answers with the *old*
  artifact. `/ready` shows `model.loading` / `index.loading` (the version being loaded) and `loading_note`. The next
  request joins the *same* load and waits again; no second download starts. A slow load is not a failure, so
  `load_error` stays empty: the promote pipelines treat `load_error` as a verdict and stop on it.
- The Lambda timeout (`lambda_timeout_seconds`, 60) leaves room for a full 20 s wait plus a request that calls Bedrock.

## Formats read

**Model package** — `model.tar.gz`, flat, at its top level. It is read by the SageMaker container (SageMaker unpacks it into
`/opt/ml/model`; `serving/sagemaker_server.py` opens it with `models/package.py`) and by finrisk-ml's jobs — **not** by this service:

- `model_v<N>.pkl` and `latest_model.joblib` (`{"model_path": "model_v<N>.pkl"}`, a file **name**, resolved inside the
  unpacked folder),
- `metrics.json`.

There is **no `features.parquet`** any more (a 6-million-row table does not belong in a model package; an older package that
still has one is accepted and the file is ignored). `/api/predict/<row_id>` looks the row up in **this service's own feature
table** (the baked demo table `data/gold/transaction_features.parquet`, which has `hour_of_day`) and sends it to the scorer;
without that table `/api/predict` is a 503 and `/api/score` (rows arrive with their features) still works.

Entries that escape the folder (`../`, absolute paths, links) are refused. A model trained on other features than
`FEATURE_COLUMNS` (for example before 13b replaced `step` by `hour_of_day`) is refused at load with a message that says so.

**The endpoint** — the request/response contract is in [api_schema.md](api_schema.md#the-endpoint-contract-sagemaker).

**Index** — the folder `s3://<kb bucket>/indexes/<version>/`; files are fetched by name (no `ListBucket` needed):

- `manifest.json` — `version`, `created`, `source_sha`, `embeddings` (`titan` | `tfidf`), `embedding_model`, `dims`,
  `docs` (`{path: {sha256, chunks}}`), `chunks` (a count). Other keys (`chunking`) and other files in the folder
  (`eval_report.json`, `build_plan.json`) are ignored.
- `chunks.jsonl` — one chunk per line: `id`, `doc_path` (for example `docs/fraud_policy.md`, shown as the citation),
  `section`, `text`, `sha256`.
- `embeddings.npy` (titan) — float32, row *i* = line *i* of `chunks.jsonl`; **or** `tfidf_vectorizer.joblib` +
  `tfidf_matrix.joblib` (tfidf).

The question is embedded the way the index was: **titan** → Bedrock `InvokeModel` with the manifest's `embedding_model`
(`amazon.titan-embed-text-v2:0`), body `{"inputText": <question>, "dimensions": <manifest dims>, "normalize": true}`, then
cosine similarity; **tfidf** → the stored vectorizer. The "too far away, refuse" floor is per method
(`tfidf` 0.03, `titan` 0.25; `RAG_MIN_SCORE` overrides). The compliance refusal runs *before* retrieval either way.

## The baked fallback

`Dockerfile.aws` runs `scripts/run_pipeline.py --demo-rows 5000` while the release image is built: a demo model trained on
5,000 synthetic transactions (with the current features, `hour_of_day` included), its feature table and a small tfidf index of
`rag_docs/` and `docs/`. It exists so the service works on a laptop, in the tests, and on the very first deploy before any
pipeline has promoted anything. With an endpoint pointer the baked **model** is not used for scoring, but the baked
**feature table** still is: it is where `/api/predict/<row_id>` finds its rows. Nothing binary is committed.

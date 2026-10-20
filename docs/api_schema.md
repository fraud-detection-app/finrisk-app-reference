# Risk Service API — contract

Base URL (local): `http://127.0.0.1:8000` · Interactive docs: `/docs` · Machine-readable: `/openapi.json`

The schema below is **generated from the code** by FastAPI. This document explains the design
decisions behind it; when the two disagree, `/openapi.json` is right and this file is stale.

## Operations

| Method | Path | Purpose | Success | Failure |
|---|---|---|---|---|
| GET | `/health` | Liveness — is the process alive? Reports `image_digest` (which image answers; null on a laptop) | 200 always | — |
| GET | `/ready` | Readiness — can it serve? Which model and index? | 200 | 503 when no model or no index is loaded |
| GET | `/api/transactions/sample` | A few valid `row_id`s | 200 | 503 |
| GET | `/api/predict/{row_id}` | Score + explain one STORED transaction (looked up in the service's own table) | 200 | 404 unknown row · 422 bad type · 503 |
| POST | `/api/score` | Score 1-500 transactions that arrive WITH their features: one call to the model (the production path) | 200 | 422 invalid rows · 503 model unavailable |
| POST | `/api/ask` | RAG answer with citations | 200 (incl. refusals) | 422 bad body · 503 |
| POST | `/api/agent` | Agent picks a tool, returns a trace | 200 (incl. refusals) | 422 · 503 |
| GET | `/` | The web UI | 200 HTML | — |

## Why liveness and readiness are two endpoints

`/health` never touches the model. A liveness probe that checks dependencies will restart a
perfectly healthy process because a disk was slow, and restarting does not make the disk faster.

Both `/health` and `/ready` also report `image_digest`: the `sha256:...` of the image the process runs from,
read from the `APP_IMAGE_DIGEST` setting that Terraform passes in (`deploy/terraform`). The deploy pipeline's smoke
test waits until it equals the digest it just deployed, so an OLD task that still answers 200 during a rolling
update is never mistaken for the new version. `null` when the setting is absent (a laptop, the tests).

`/ready` says what is loaded **right now** and returns **503** until both a model and a knowledge index are loaded,
so an orchestrator stops sending traffic without killing the process, and a human is told what failed:

```json
{ "ready": true,
  "image_digest": "sha256:0123...",
  "checks": { "gold_features": true, "model_artifact": true, "vector_index": true },
  "missing": [],
  "rows": 5000,
  "model": { "version": 3, "source": "endpoint", "endpoint": "finrisk-dev-fraud", "status": "InService", "error": null,
             "loaded_at": "2026-10-16T17:12:10+00:00", "load_error": null, "threshold": 0.4812, "loading": null,
             "model_type": null, "trained_at_utc": null, "pr_auc": null },
  "index": { "version": "20261016-101500-ab12cd3", "embeddings": "titan", "docs": 3, "chunks": 3, "source": "s3",
             "loaded_at": "2026-10-16T17:12:10+00:00", "load_error": null, "loading": null },
  "pointers_checked_at": "2026-10-16T17:12:10+00:00",
  "pointer_prefix": "/finrisk/dev",
  "load_error": null, "hint": null,
  "llm": { "...": "..." }, "llm_providers": { "deepseek": false, "bedrock": true, "template": true } }
```

- `model.version` is the **number** in the `/finrisk/<env>/model` pointer (`"baked"` when the pointer is null or there is
  no pointer); `index.version` is the **string** in `/finrisk/<env>/index`. The pipelines that move a pointer poll these two
  fields until they equal what they wrote.
- `model.source` is `endpoint` (a SageMaker endpoint the pointer names) or `baked` (the demo model inside the image, scored in
  process); `index.source` is `s3` (downloaded because a pointer named it) or `baked`.
- `model.endpoint` is the endpoint name (`null` when baked). `model.status` is its `DescribeEndpoint` status (`InService`,
  `Updating` ...), cached 60 s, `null` when baked or when the describe call failed. `model.error` is the last scoring or describe
  problem (`model endpoint is starting, retry`, ...) and goes back to `null` on the next success; it is **not** `load_error`.
- `model.threshold`: for the baked model the chosen threshold; for an endpoint, the threshold its last answer reported (`null`
  until the first score).
- `ready` needs a model, the service's own feature table (`checks.gold_features`) and an index. A failing endpoint does not make
  `ready` false (the process is fine; scoring answers 503 and `model.error` says why).
- `load_error` is `null` when all is well. When a **newer** version could not be loaded, the older one keeps serving:
  `ready` stays `true` and `load_error` says why (account ids are masked: `/ready` is public).
- `loading` is the version being downloaded right now, or `null`.
- `pointers_checked_at` is when the service last read its two SSM parameters (UTC), `null` without `POINTER_PREFIX`.

See [artifacts.md](artifacts.md) for how the service finds, loads and swaps these.

## The error contract

Every failure returns the same shape, so a client can parse an error as reliably as a success:

```json
{ "error": "transaction_not_found",
  "detail": "No transaction with row_id=999999999",
  "request_id": "01d93813d0d3",
  "hint": "Try a row_id from GET /api/transactions/sample" }
```

Status codes carry meaning the caller can act on:

- **404** — your request was valid, that thing does not exist. Retrying will not help.
- **422** — the request itself is malformed. FastAPI produces this *before* our code runs.
- **503** — nothing wrong with the request; the server cannot serve it **yet** (no model/index loaded, the embedding call for the knowledge base failed, or the **model endpoint** cannot answer: `error: "model_unavailable"`, `detail: "model endpoint is starting, retry"` for a cold Serverless endpoint, with a `Retry-After` header). Retry later. There is no fallback to the baked model when an endpoint is configured.
- **500** — a bug. The traceback goes to the server log keyed by `request_id`, never to the client.

A stack trace in an HTTP response is both an information leak and a dead end for the caller.

## Request validation you do not have to write

`row_id` is declared `int`, so `/api/predict/abc` is rejected with 422 before the handler runs.
`QuestionRequest` declares `min_length=3, max_length=500`, so an empty question never reaches the
retriever. Validation that lives in the type signature cannot be forgotten in a later refactor.

## Scoring: one stored row, or a batch

`GET /api/predict/{row_id}` is the **teaching** path: the row is looked up in the service's own feature table, scored, and
explained; it also returns `scored_by` (`endpoint` | `baked`) and `model_version`. `reasons` are sentences built from the model
server's TreeSHAP contributions (`reason_method: "shap"`), or rule-of-thumb sentences (`"heuristic"`) when a model type cannot
explain itself.

`POST /api/score` is the **production** path: the caller already holds the features (a payments system, the replay in finrisk-ml).

```bash
curl -s -X POST localhost:8000/api/score -H 'content-type: application/json' -d '{
  "transactions": [ {"row_id": "4821377", "transaction_type": "TRANSFER", "amount": 181000.0, "log_amount": 12.1,
                     "origin_old_balance": 181000.0, "origin_new_balance": 0.0, "destination_old_balance": 0.0,
                     "destination_new_balance": 0.0, "origin_balance_delta": 181000.0, "destination_balance_delta": 0.0,
                     "amount_to_origin_balance": 1.0, "hour_of_day": 3, "tx_index_for_origin": 1,
                     "customer_tx_count_so_far": 1, "customer_amount_so_far": 181000.0} ],
  "explain": false, "top_k": 3 }'
```

| Field | |
|---|---|
| `transactions` | 1 to 500 rows. Each holds **exactly** `row_id` (string or whole number) and every model feature (`FEATURE_COLUMNS`); a missing or extra key is a 422 |
| `explain` | default `false`; `true` adds the top `top_k` (1-20, default 3) reasons per row (slower: leave it off for bulk scoring) |

```json
{ "model_version": "3", "threshold": 0.4812, "scored_by": "endpoint", "elapsed_ms": 41.2,
  "results": [ { "row_id": "4821377", "score": 0.9132, "flagged": true,
                 "reasons": [ {"feature": "origin_new_balance", "value": 0.0, "contribution": 6.21} ] } ] }
```

`reasons` appears only when `explain` was true. The whole batch is **one** call to the model (one `InvokeEndpoint`, which the
endpoint's Data Capture records), and `results` keep the request's order. A row that breaks the contract is a **422** that lists
every problem and makes no AWS call:

```json
{ "error": "invalid_transactions", "detail": "instance 1 (row_id '8'): missing ['amount']; instance 2 (row_id '9'): extra ['note']",
  "problems": [ {"index": 1, "row_id": "8", "missing": ["amount"]}, {"index": 2, "row_id": "9", "extra": ["note"]} ],
  "request_id": "...", "hint": "Each transaction needs exactly row_id and the model's features ..." }
```

## The endpoint contract (SageMaker)

The same request/response is spoken by the SageMaker endpoint (`POST /invocations`), by the app's two scorers, by the replay
and the monitor in finrisk-ml, and by the tests. It is defined once, in `src/financial_risk/serving/contract.py`.

```json
request   {"instances": [{"row_id": "4821377", "transaction_type": "TRANSFER", "amount": 181000.0, "...": "every FEATURE_COLUMNS key"}],
           "explain": true, "top_k": 3}
response  {"model_version": "3", "threshold": 0.4812,
           "predictions": [{"row_id": "4821377", "score": 0.9132, "flagged": true,
                            "reasons": [{"feature": "origin_new_balance", "value": 0.0, "contribution": 6.21}]}]}
```

- `model_version` is the env var `MODEL_VERSION` set on the SageMaker Model (`"baked"` in process). `flagged` = `score >= threshold`.
- The features, in order: `transaction_type` (string) then `amount, log_amount, origin_old_balance, origin_new_balance,
  destination_old_balance, destination_new_balance, origin_balance_delta, destination_balance_delta, amount_to_origin_balance,
  hour_of_day, tx_index_for_origin, customer_tx_count_so_far, customer_amount_so_far` (numbers; finite). `hour_of_day =
  (step - 1) % 24`; `step` is **not** a feature.
- `reasons`: only with `explain: true`. `contribution` is a SHAP value in **log-odds** (positive pushes the score up), the top
  `top_k` by absolute size. XGBoost: TreeSHAP from the booster (`pred_contribs`), the one-hot `transaction_type` columns summed back
  into one `transaction_type`. Logistic-regression baseline: the exact linear SHAP value `coef * x` (from the training mean for numbers).
  Another model type: the week 7 `shap.Explainer` path when background rows are available, otherwise `reasons: []`.
- At most **500** instances. Missing keys, extra keys, wrong types, a body that is not JSON, an unknown top-level key, or a bad
  `top_k`: **400**, `{"error": "invalid_request", "message": "...", "problems": [{"index", "row_id", "missing", "extra", "invalid"}]}`.
- The hosting contract: `GET /ping` is 200 only when the model is loaded (else 503 with the reason), `POST /invocations`,
  port 8080, one worker; `python -m financial_risk.serving.sagemaker_server`. Request bodies are never logged.
- `EndpointScorer` (the app's client): connect timeout 5 s, read timeout 40 s; fast failures are retried twice inside a 50 s budget
  (a read timeout is not: it is reported at once as 503 "model endpoint is starting, retry"). The app validates the rows
  itself first, so a bad row never costs an AWS call.

## Label vs prediction

`/api/predict/{row_id}` returns **both** and never merges them:

- `confirmed_fraud` — the investigation outcome, where one exists. Absent for unsettled rows.
- `risk_score` / `risk_level` — what the model predicts, against the week-7 threshold.

## LLM summaries

`?use_llm=true` adds a narrative summary. `summary_source` says which provider produced it —
`deepseek`, `bedrock`, or `template`. With no API key the deterministic template answers, so the
endpoint never fails for want of a key, and the reader can always tell what wrote the sentence.

### Switching the provider on

Two lines in `.env` (which is gitignored — keep the key blank in `.env.example`):

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=<your key>
```

`DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL` are already set. The model id must be a current one —
`deepseek-flash` (cheap, the default here) or `deepseek-v4-pro`. The `deepseek-chat` and
`deepseek-reasoner` names that most tutorials still show were retired and now return an error;
`llm/deepseek_client.py` recognises that case and says so by name instead of leaving you to guess.

A provider failure never fails the request: `llm/client.py` logs it and falls back to the
template, and `summary_source` records what actually answered.

## Curl

```bash
curl -s localhost:8000/health
curl -s localhost:8000/ready
curl -s "localhost:8000/api/predict/451?use_llm=true"
curl -s -X POST localhost:8000/api/score -H 'content-type: application/json' -d @batch.json     # see "Scoring" above
curl -s -X POST localhost:8000/api/ask   -H 'content-type: application/json' -d '{"question":"What is PR-AUC?"}'
curl -s -X POST localhost:8000/api/agent -H 'content-type: application/json' -d '{"question":"What is the fraud rate by transaction type?"}'
```

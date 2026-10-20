# finrisk-app

The fraud-risk **service**: score a transaction, explain it (SHAP + an LLM), ask the knowledge base (RAG), or ask the
agent. One of four repositories (`finrisk-infra`, `finrisk-app`, `finrisk-ml`, `finrisk-knowledge`).

**What runs in an environment = this repo's image digest + a model version + an index version.** This repo ships the
image. The model and the index come from the other two pipelines through two SSM parameters per environment, and the
running service **hot-reloads** them (see [docs/artifacts.md](docs/artifacts.md)).

**The model is a SageMaker endpoint** (13b). finrisk-ml deploys a model version to the endpoint `finrisk-<env>-fraud` and
writes its name into the model pointer; this service no longer downloads or loads the model, it calls the endpoint
(`service/scoring.py`). The endpoint runs **this repo's code** (`serving/sagemaker_server.py`: the finrisk-ml training
image is built `FROM` this image), so the code that scores on a laptop is the code that scores on SageMaker.
A null pointer means the **baked** demo model in the image, scored in process.

```
merge to main -> build image sha-<commit> (once) -> dev -> sit -> pre-prod -> prod -> release v<VERSION>
                                                    Lambda Lambda ECS Express (approval) ECS Express (approval)
```

## Layout

| Path | What |
|---|---|
| `src/financial_risk/` | the service code: `service/` (artifacts, scoring, risk_service), `serving/` (the endpoint contract and the SageMaker server), `rag/` (query side), `llm/`, `agent/`, `models/`, `data/`, `features/` |
| `src/financial_risk/serving/` | `contract.py` (request/response + validation) · `runner.py` (scores a request; TreeSHAP reasons) · `sagemaker_server.py` (`GET /ping`, `POST /invocations`, port 8080) |
| `api/` | FastAPI app and the small web page |
| `deploy/terraform/` | the service stack: Lambda (dev, sit) or ECS Express (pre-prod, prod); `envs/*.tfvars`, offline `tests/` |
| `.github/workflows/` | `ci.yml` · `deliver.yml` (+ `deploy-env.yml`) · `rollback.yml` · `teardown.yml` |
| `Dockerfile`, `Dockerfile.aws`, `scripts/build_release.sh` | the service image, then the release image (+ baked fallback + Lambda Web Adapter) |
| `scripts/run_pipeline.py` | demo data → features → model → index. Used by the build (baked fallback), by CI, and by the **finrisk-ml training image** (`FROM` this image) |
| `models/training.py`, `models/package.py` | the model features (`FEATURE_COLUMNS`, with `hour_of_day`) and training · opening a `model.tar.gz` package. Both are imported by finrisk-ml |
| `VERSION` | `MAJOR.MINOR.PATCH`; bump it in the pull request that should become a release |
| `tests/` | pytest (service, artifacts, scoring, the endpoint contract, API, workflows, Terraform) |

## Work on it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
DATA_MODE=demo python scripts/run_pipeline.py --demo-rows 5000     # the baked demo model + index (git-ignored)
LLM_PROVIDER=none python -m pytest -q
LLM_PROVIDER=none python scripts/run_api.py                         # http://127.0.0.1:8000
```

Trunk-based: a short-lived `feature/*` branch → pull request → `main`. `ci.yml` must be green (`test`, `terraform`).
Environments are *stages of one pipeline run*, not branches.

## What finrisk-ml imports from here (a stable interface)

| Need | Import |
|---|---|
| the model features, in order; training | `financial_risk.models.training`: `FEATURE_COLUMNS`, `CATEGORICAL_FEATURES`, `NUMERIC_FEATURES`, `train_model`, `temporal_split`, `choose_threshold`, `classification_metrics`, `predict_proba_positive` |
| feature engineering (adds `hour_of_day`) | `financial_risk.features.sql_features.build_feature_tables` |
| open / unpack a `model.tar.gz` | `financial_risk.models.package`: `unpack_model_package(archive, dest)`, `load_model_package(dir)` |
| the request and response of the endpoint | `financial_risk.serving.contract`: `build_request`, `parse_request`, `frame_to_instances`, `INSTANCE_KEYS`, `MAX_INSTANCES` |
| score locally with exactly the endpoint's code (evaluate, bias, SHAP) | `financial_risk.serving.runner.ModelRunner.from_package_dir(dir, version)` → `.invoke(request)` |
| host a model on SageMaker | `python -m financial_risk.serving.sagemaker_server` (the `serve` command; reads `SM_MODEL_DIR` = `/opt/ml/model`, `MODEL_VERSION`) |
| drift maths | `financial_risk.eda.profiling.population_stability_index` |

## Configuration

| Setting | Meaning |
|---|---|
| `POINTER_PREFIX` | `/finrisk/<env>`; empty = baked model and index (laptop, tests) |
| `SM_MODEL_DIR`, `MODEL_VERSION`, `SAGEMAKER_BIND_TO_PORT` | only for `python -m financial_risk.serving.sagemaker_server` (the SageMaker container): where the unpacked model package is (`/opt/ml/model`), the version echoed in every response, the port (8080) |
| `POINTER_TTL_SECONDS` | how often the pointers are re-read (60) |
| `POINTER_REFRESH` | `inline` (the request waits for a changed pointer to load; default on Lambda, which freezes background threads) or `background` (a thread; default elsewhere) |
| `INLINE_LOAD_TIMEOUT_SECONDS` | inline only: the longest one request waits for a load (20); after that the old artifact answers and the next request carries on |
| `ENDPOINT_CONNECT_TIMEOUT_SECONDS` / `ENDPOINT_READ_TIMEOUT_SECONDS` | the SageMaker endpoint call (5 / 40; Terraform sets the read timeout to 25 on Lambda so 20 s inline reload + 25 + 5 connect + 5 slack fits the 60 s Lambda timeout). The whole call, retries included, stays within connect + read + 5 s; a read timeout is a 503 "model endpoint is starting, retry" |
| `LLM_PROVIDER` | `none` (templates, offline), `deepseek` (local, needs a key), `bedrock` (AWS, no key) |
| `BEDROCK_MODEL_ID` | chat model, `deepseek.v3.2` |
| `APP_IMAGE_DIGEST` | set by Terraform; `/health` and `/ready` repeat it so the deploy can tell new from old |

## GitHub setup (once, by `finrisk-infra/scripts/github_bootstrap.sh`)

| Name | Kind | Where |
|---|---|---|
| `AWS_REGION` | variable | repository |
| `AWS_BUILD_ROLE_ARN` | secret | repository (build job, no Environment) |
| `ECR_REPOSITORY` | secret | repository (full registry URL of `finrisk-app`) |
| `TF_STATE_BUCKET` | secret | repository |
| `AWS_ROLE_ARN` | secret | each Environment: `dev`, `sit`, `pre-prod` (reviewer), `prod` (reviewer) |

The repository is **public**: values that contain the AWS account id are secrets, nothing else is. AWS is reached with
OIDC; there is no access key anywhere. Runbook: [docs/runbook.md](docs/runbook.md).

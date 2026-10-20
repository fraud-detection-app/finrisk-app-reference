# Runbook: deploy, roll back, tear down

Everything is a GitHub Actions workflow. Nobody deploys from a laptop.

## Normal delivery

A pull request is merged into `main` → `deliver.yml` runs once:

`build` (image `finrisk-app:sha-<commit>`, skipped if it exists) → `deploy-dev` → `deploy-sit` → `deploy-pre-prod`
(a reviewer approves) → `deploy-prod` (a reviewer approves) → `release` (tag `v<VERSION>` + GitHub Release, if the tag
does not exist yet).

Each deploy resolves the tag to a **digest**, runs `terraform plan` / `apply` in `deploy/terraform` (state
`app/service.tfstate`, workspace = environment), then the smoke test: `/health` must answer 200 **with this digest**
("old version still answering" while it does not), then `/ready` must answer 200 (model **and** index loaded).
Lambda gets 5 minutes; ECS Express Mode gets 18 (its first create takes about 13).

To cut a release, bump `VERSION` in the pull request. A change that does not bump it deploys but releases nothing.

## Roll back an environment

Actions → **rollback** → Use workflow from `main` → pick the environment and the full `sha` of the good commit (the Releases
page and old deliver runs show it). It is the same deploy with an older image; nothing is rebuilt; pre-prod and prod
wait for their reviewer. The model and the index are not touched: use *promote-model* (finrisk-ml) or *promote-index*
(finrisk-knowledge) to move those back. (The model is a SageMaker endpoint that finrisk-ml owns; this service only reads its name
from the model pointer and needs `sagemaker:InvokeEndpoint` + `DescribeEndpoint` on it, granted by finrisk-infra.)

## What is running in an environment?

```bash
curl -s "$(aws ssm get-parameter --name /finrisk/prod/app_url --query Parameter.Value --output text)/ready" | jq '{image_digest, model: .model | {version, source, endpoint, status, error}, index: .index.version, load_error}'
```

## Tear down one environment's service

Actions → **teardown** → environment + type its name again. This destroys the Lambda or Express service and the
`app_url` parameter and deletes the workspace's state. Roles, bucket, log group and pointers belong to finrisk-infra.

## By hand (GitHub is down, or a lab)

```bash
cd deploy/terraform
terraform init -backend-config=bucket=<state bucket> -backend-config=key=app/service.tfstate -backend-config=region=ap-southeast-2 -backend-config=use_lockfile=true
terraform workspace select -or-create dev
terraform plan -var-file=envs/dev.tfvars -var image_uri=<repo>@sha256:<digest>
```

## Locally

```bash
pip install -r requirements.txt
DATA_MODE=demo python scripts/run_pipeline.py --demo-rows 5000    # the baked demo model, features and index
python -m pytest -q
LLM_PROVIDER=none python scripts/run_api.py                       # http://127.0.0.1:8000  (/ready, /docs)
bash scripts/build_release.sh                                       # the release image, needs Docker
```

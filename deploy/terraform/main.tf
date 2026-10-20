# deploy/terraform -- the SERVICE for ONE environment: a Lambda function (dev, sit) or an ECS Express Mode
# service (pre-prod, prod) that runs one image. The same code makes all four; only the tfvars file and the
# workspace differ.
#
# What this stack does NOT create: roles, buckets, log groups, the image registry. Those belong to the
# finrisk-infra repository, which publishes their names and ARNs as SSM parameters. This stack READS them
# (data "aws_ssm_parameter"), so the app team can deploy without being able to change anybody's permissions,
# and the infra team can change a role without touching the app.
#
# Who runs it: the pipeline (.github/workflows/deliver.yml, rollback.yml, teardown.yml) -- the same
# commands, in the same order, for every environment:
#
#   terraform init -backend-config=...             connect to the shared state bucket (see backend.tf)
#   terraform workspace select -or-create dev      one workspace = one environment = one state file
#   terraform plan  -var-file=envs/dev.tfvars -var image_uri=<repo>@sha256:<digest>
#   terraform apply tfplan
#
# Files: main.tf (guard, what we read from infra, the container's settings) - lambda.tf (dev, sit) -
#        ecs.tf (pre-prod, prod) - ssm.tf (the URL we publish) - outputs.tf.

locals {
  name   = "${var.project}-${var.env}"
  prefix = "/${var.project}/${var.env}" # the SSM path of this environment: /finrisk/dev

  use_lambda = var.compute == "lambda"
  use_ecs    = var.compute == "ecs"

  # The settings the container reads. The SAME image runs in every environment; this map is the only
  # difference between them.
  app_environment = merge({
    APP_ENV                      = var.env
    LLM_PROVIDER                 = "bedrock" # calls Bedrock with the runtime role -- no API key
    BEDROCK_MODEL_ID             = var.bedrock_model_id
    PORT                         = "8000"
    AWS_LWA_READINESS_CHECK_PATH = "/health" # the Lambda Web Adapter waits for this before serving
    # The image's own fingerprint (the sha256:... part of image_uri). The app reports it on /health and /ready, so the
    # pipeline's smoke test can tell the NEW version from the old one: during an ECS Express rolling update the old
    # task keeps answering 200 until the new one is healthy.
    APP_IMAGE_DIGEST = element(split("@", var.image_uri), 1)
    # Where the app finds its model and its knowledge index: two SSM parameters that finrisk-ml and
    # finrisk-knowledge write. The app reads <POINTER_PREFIX>/model and <POINTER_PREFIX>/index at start-up
    # and then every POINTER_TTL_SECONDS, and swaps in a new version without a restart.
    POINTER_PREFIX      = local.prefix
    POINTER_TTL_SECONDS = tostring(var.pointer_ttl_seconds)
    }, local.use_lambda ? {
    # The app calls the model's SageMaker endpoint (default read timeout 40 s). A Lambda function stops at
    # lambda_timeout_seconds (60 s) and one request may first wait up to INLINE_LOAD_TIMEOUT_SECONDS (20 s) for a new
    # index, so the endpoint call is shortened: 20 + (5 connect + 25 read + 5 slack) = 55 <= 60. A cold Serverless
    # endpoint then answers 503 "model endpoint is starting, retry" after 25 s instead of 40 s. ECS keeps the default.
    ENDPOINT_READ_TIMEOUT_SECONDS = "25"
  } : {})
}

# ---------------------------------------------------------------- the safety catch
# `-var-file=envs/prod.tfvars` while the workspace is `dev` would write prod's resources into dev's
# state file -- and the next dev apply could then delete them. This stops that.
#
# Why a precondition and not a `check` block: a failed `check` only prints a warning and the apply
# carries on. A failed `precondition` is an ERROR during plan, so nothing is created or changed.
resource "terraform_data" "workspace_matches_env" {
  lifecycle {
    precondition {
      condition     = terraform.workspace == var.env
      error_message = "Workspace '${terraform.workspace}' does not match env '${var.env}'. Run `terraform workspace select -or-create ${var.env}` (or pass the tfvars file that belongs to this workspace)."
    }
  }
}

# ---------------------------------------------------------------- what finrisk-infra published
# Written by finrisk-infra (envs/): one set of parameters per environment. If one is missing, the plan fails
# with "ParameterNotFound": apply finrisk-infra for this environment first.
#
# insecure_value: these are names and ARNs, not secrets, so we read them as plain text (the default `value`
# attribute is marked sensitive, which would hide them in the plan and force every output to be sensitive).
data "aws_ssm_parameter" "runtime_role_arn" {
  name = "${local.prefix}/runtime_role_arn"
}

data "aws_ssm_parameter" "log_group" {
  name = "${local.prefix}/log_group"
}

data "aws_ssm_parameter" "bucket" {
  name = "${local.prefix}/bucket"
}

# Only the Express Mode environments (pre-prod, prod) have these two roles.
data "aws_ssm_parameter" "ecs_execution_role_arn" {
  count = local.use_ecs ? 1 : 0

  name = "${local.prefix}/ecs_execution_role_arn"
}

data "aws_ssm_parameter" "ecs_infrastructure_role_arn" {
  count = local.use_ecs ? 1 : 0

  name = "${local.prefix}/ecs_infrastructure_role_arn"
}

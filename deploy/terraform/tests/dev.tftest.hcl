# Offline test of the Lambda branch (dev and sit). No AWS account is touched: mock_provider fakes the AWS
# provider, override_data fakes what finrisk-infra publishes in SSM, and `command = plan` creates nothing.
#
# The workspace guard (main.tf) needs the workspace to be named like the env, and `terraform test` runs in
# whichever workspace is selected -- so this file needs workspace "dev":
#   cp -R deploy/terraform /tmp/svc-test && cd /tmp/svc-test && rm backend.tf       # local state, no S3
#   terraform init && terraform workspace new dev && terraform test -filter=tests/dev.tftest.hcl
# (tests/test_deploy_terraform.py does exactly this for you.)

mock_provider "aws" {}

override_data {
  target = data.aws_ssm_parameter.runtime_role_arn
  values = {
    insecure_value = "arn:aws:iam::000000000000:role/finrisk-dev-runtime"
  }
}

override_data {
  target = data.aws_ssm_parameter.log_group
  values = {
    insecure_value = "/app/finrisk-dev"
  }
}

override_data {
  target = data.aws_ssm_parameter.bucket
  values = {
    insecure_value = "finrisk-dev-000000000000-ap-southeast-2-an"
  }
}

variables {
  image_uri = "000000000000.dkr.ecr.ap-southeast-2.amazonaws.com/finrisk-app@sha256:0000000000000000000000000000000000000000000000000000000000000000"
}

run "dev_is_lambda_and_only_lambda" {
  command = plan

  variables {
    env     = "dev"
    compute = "lambda"
  }

  # The Lambda branch exists ...
  assert {
    condition     = length(aws_lambda_function.app) == 1 && length(aws_lambda_function_url.app) == 1 && length(aws_lambda_permission.public_url) == 1
    error_message = "dev should create exactly one Lambda function, one Function URL and one invoke permission."
  }

  # ... and the ECS branch does not (its two SSM reads do not exist either).
  assert {
    condition     = length(aws_ecs_express_gateway_service.app) == 0 && length(data.aws_ssm_parameter.ecs_execution_role_arn) == 0 && length(data.aws_ssm_parameter.ecs_infrastructure_role_arn) == 0
    error_message = "dev must not create or read anything for ECS Express Mode."
  }

  assert {
    condition     = aws_lambda_function.app[0].package_type == "Image" && aws_lambda_function.app[0].timeout == 60 && contains(aws_lambda_function.app[0].architectures, "x86_64") && aws_lambda_function.app[0].memory_size == 2048
    error_message = "The function should be a 60 s, 2048 MB, x86_64 container-image function."
  }

  assert {
    condition     = aws_lambda_function.app[0].image_uri == var.image_uri
    error_message = "The function must run the digest we passed in."
  }

  # The role and the log group are READ from infra, not made here.
  assert {
    condition     = aws_lambda_function.app[0].role == "arn:aws:iam::000000000000:role/finrisk-dev-runtime" && aws_lambda_function.app[0].logging_config[0].log_group == "/app/finrisk-dev"
    error_message = "The function must use the runtime role and the log group that finrisk-infra published."
  }

  assert {
    condition = (
      aws_lambda_function.app[0].environment[0].variables["APP_ENV"] == "dev" &&
      aws_lambda_function.app[0].environment[0].variables["LLM_PROVIDER"] == "bedrock" &&
      aws_lambda_function.app[0].environment[0].variables["BEDROCK_MODEL_ID"] == "deepseek.v3.2" &&
      aws_lambda_function.app[0].environment[0].variables["PORT"] == "8000" &&
      aws_lambda_function.app[0].environment[0].variables["AWS_LWA_READINESS_CHECK_PATH"] == "/health"
    )
    error_message = "The container settings are wrong."
  }

  # Where the app finds its model and index.
  assert {
    condition = (
      aws_lambda_function.app[0].environment[0].variables["POINTER_PREFIX"] == "/finrisk/dev" &&
      aws_lambda_function.app[0].environment[0].variables["POINTER_TTL_SECONDS"] == "60" &&
      length(aws_lambda_function.app[0].environment[0].variables) == 9
    )
    error_message = "Expected POINTER_PREFIX=/finrisk/dev, POINTER_TTL_SECONDS=60 and nine settings in all."
  }

  # Lambda stops at 60 s: 20 s inline index reload + (5 connect + 25 read + 5 slack) for the model endpoint = 55.
  assert {
    condition     = aws_lambda_function.app[0].environment[0].variables["ENDPOINT_READ_TIMEOUT_SECONDS"] == "25"
    error_message = "On Lambda the endpoint read timeout must be shortened to 25 s."
  }

  # The app reports which image it runs (/health, /ready): the sha256 part of image_uri. The deploy's smoke test waits for it.
  assert {
    condition     = aws_lambda_function.app[0].environment[0].variables["APP_IMAGE_DIGEST"] == "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    error_message = "APP_IMAGE_DIGEST must be the digest part of image_uri."
  }

  assert {
    condition     = aws_lambda_function_url.app[0].authorization_type == "NONE" && aws_lambda_permission.public_url[0].action == "lambda:InvokeFunctionUrl"
    error_message = "A public Function URL needs auth NONE and the InvokeFunctionUrl permission."
  }

  # The one parameter this stack writes.
  assert {
    condition     = aws_ssm_parameter.app_url.name == "/finrisk/dev/app_url" && aws_ssm_parameter.app_url.type == "String"
    error_message = "The URL must be published at /finrisk/dev/app_url."
  }

  # This stack creates no role, bucket or log group.
  assert {
    condition     = output.compute == "lambda" && output.env == "dev" && output.image_uri == var.image_uri && output.pointer_prefix == "/finrisk/dev" && output.log_group == "/app/finrisk-dev"
    error_message = "The outputs should echo what dev runs."
  }
}

run "the_pointer_check_can_be_slowed_down" {
  command = plan

  variables {
    env                 = "dev"
    compute             = "lambda"
    pointer_ttl_seconds = 300
    lambda_memory_mb    = 1024
  }

  assert {
    condition     = aws_lambda_function.app[0].environment[0].variables["POINTER_TTL_SECONDS"] == "300" && aws_lambda_function.app[0].memory_size == 1024
    error_message = "pointer_ttl_seconds and lambda_memory_mb must reach the function."
  }
}

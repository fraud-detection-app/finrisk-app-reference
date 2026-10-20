# Offline test of the ECS Express Mode branch (pre-prod and prod). Same rules as dev.tftest.hcl:
# mock_provider fakes AWS, `command = plan` creates nothing, and the workspace must be "pre-prod":
#   cp -R deploy/terraform /tmp/svc-test && cd /tmp/svc-test && rm backend.tf
#   terraform init && terraform workspace new pre-prod && terraform test -filter=tests/pre-prod.tftest.hcl

mock_provider "aws" {}

override_data {
  target = data.aws_ssm_parameter.runtime_role_arn
  values = {
    insecure_value = "arn:aws:iam::000000000000:role/finrisk-pre-prod-runtime"
  }
}

override_data {
  target = data.aws_ssm_parameter.log_group
  values = {
    insecure_value = "/app/finrisk-pre-prod"
  }
}

override_data {
  target = data.aws_ssm_parameter.bucket
  values = {
    insecure_value = "finrisk-pre-prod-000000000000-ap-southeast-2-an"
  }
}

override_data {
  target = data.aws_ssm_parameter.ecs_execution_role_arn[0]
  values = {
    insecure_value = "arn:aws:iam::000000000000:role/finrisk-pre-prod-ecs-execution"
  }
}

override_data {
  target = data.aws_ssm_parameter.ecs_infrastructure_role_arn[0]
  values = {
    insecure_value = "arn:aws:iam::000000000000:role/finrisk-pre-prod-ecs-infrastructure"
  }
}

variables {
  image_uri = "000000000000.dkr.ecr.ap-southeast-2.amazonaws.com/finrisk-app@sha256:0000000000000000000000000000000000000000000000000000000000000000"
}

run "pre_prod_is_ecs_and_only_ecs" {
  command = plan

  variables {
    env     = "pre-prod"
    compute = "ecs"
  }

  # The ECS branch exists ...
  assert {
    condition     = length(aws_ecs_express_gateway_service.app) == 1
    error_message = "pre-prod should create one Express Mode service."
  }

  # ... and the Lambda branch does not.
  assert {
    condition     = length(aws_lambda_function.app) == 0 && length(aws_lambda_function_url.app) == 0 && length(aws_lambda_permission.public_url) == 0
    error_message = "pre-prod must not create anything for Lambda."
  }

  assert {
    condition = (
      aws_ecs_express_gateway_service.app[0].cpu == "1024" &&
      aws_ecs_express_gateway_service.app[0].memory == "2048" &&
      aws_ecs_express_gateway_service.app[0].health_check_path == "/health" &&
      aws_ecs_express_gateway_service.app[0].primary_container[0].container_port == 8000
    )
    error_message = "Express service: 1 vCPU, 2048 MiB, /health, port 8000."
  }

  # All three roles and the log group come from finrisk-infra.
  assert {
    condition = (
      aws_ecs_express_gateway_service.app[0].task_role_arn == "arn:aws:iam::000000000000:role/finrisk-pre-prod-runtime" &&
      aws_ecs_express_gateway_service.app[0].execution_role_arn == "arn:aws:iam::000000000000:role/finrisk-pre-prod-ecs-execution" &&
      aws_ecs_express_gateway_service.app[0].infrastructure_role_arn == "arn:aws:iam::000000000000:role/finrisk-pre-prod-ecs-infrastructure" &&
      aws_ecs_express_gateway_service.app[0].primary_container[0].aws_logs_configuration[0].log_group == "/app/finrisk-pre-prod"
    )
    error_message = "The service must use the roles and the log group that finrisk-infra published."
  }

  assert {
    condition     = aws_ecs_express_gateway_service.app[0].primary_container[0].image == var.image_uri
    error_message = "The service must run the digest we passed in."
  }

  # The same container settings as Lambda, as a list of name/value pairs.
  assert {
    condition = (
      length(aws_ecs_express_gateway_service.app[0].primary_container[0].environment) == 8 &&
      !contains([for e in aws_ecs_express_gateway_service.app[0].primary_container[0].environment : e.name], "ENDPOINT_READ_TIMEOUT_SECONDS") &&
      contains([for e in aws_ecs_express_gateway_service.app[0].primary_container[0].environment : "${e.name}=${e.value}"], "POINTER_PREFIX=/finrisk/pre-prod") &&
      contains([for e in aws_ecs_express_gateway_service.app[0].primary_container[0].environment : "${e.name}=${e.value}"], "APP_ENV=pre-prod") &&
      contains([for e in aws_ecs_express_gateway_service.app[0].primary_container[0].environment : "${e.name}=${e.value}"], "LLM_PROVIDER=bedrock") &&
      contains([for e in aws_ecs_express_gateway_service.app[0].primary_container[0].environment : "${e.name}=${e.value}"], "APP_IMAGE_DIGEST=sha256:0000000000000000000000000000000000000000000000000000000000000000")
    )
    error_message = "The container settings are wrong."
  }

  assert {
    condition     = aws_ecs_express_gateway_service.app[0].scaling_target[0].min_task_count == 1 && aws_ecs_express_gateway_service.app[0].scaling_target[0].max_task_count == 1
    error_message = "One task, always."
  }

  assert {
    condition     = aws_ssm_parameter.app_url.name == "/finrisk/pre-prod/app_url" && output.compute == "ecs"
    error_message = "The URL must be published at /finrisk/pre-prod/app_url."
  }
}

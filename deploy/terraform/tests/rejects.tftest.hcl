# Offline test of the guard rails: bad input must stop the plan. It runs in the "dev" workspace
# (a workspace that matches the valid env used below):
#   cp -R deploy/terraform /tmp/svc-test && cd /tmp/svc-test && rm backend.tf
#   terraform init && terraform workspace new dev && terraform test -filter=tests/rejects.tftest.hcl

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

run "a_tag_is_not_a_digest" {
  command = plan

  variables {
    env       = "dev"
    compute   = "lambda"
    image_uri = "000000000000.dkr.ecr.ap-southeast-2.amazonaws.com/finrisk-app:sha-0123456789abcdef0123456789abcdef01234567"
  }

  expect_failures = [var.image_uri]
}

run "a_short_digest_is_not_a_digest" {
  command = plan

  variables {
    env       = "dev"
    compute   = "lambda"
    image_uri = "000000000000.dkr.ecr.ap-southeast-2.amazonaws.com/finrisk-app@sha256:abc"
  }

  expect_failures = [var.image_uri]
}

run "staging_is_not_an_environment" {
  command = plan

  variables {
    env     = "staging"
    compute = "lambda"
  }

  expect_failures = [var.env]
}

run "fargate_is_not_a_compute_choice" {
  command = plan

  variables {
    env     = "dev"
    compute = "fargate"
  }

  expect_failures = [var.compute]
}

run "lambda_memory_above_the_new_account_ceiling_is_refused" {
  command = plan

  variables {
    env              = "dev"
    compute          = "lambda"
    lambda_memory_mb = 4096
  }

  expect_failures = [var.lambda_memory_mb]
}

# prod's settings in the dev workspace (this file runs in workspace dev): the precondition in main.tf stops the plan, so nothing could be
# written into the wrong state file.
run "wrong_workspace_is_blocked" {
  command = plan

  variables {
    env     = "prod"
    compute = "ecs"
  }

  expect_failures = [terraform_data.workspace_matches_env]
}

# dev and sit: the container image as a Lambda function with a public HTTPS address.
# Everything here exists only when compute = "lambda" (count = 0 otherwise, so pre-prod and prod
# contain none of it -- and ecs.tf is the mirror image).
#
# The function's role, its log group: both came from finrisk-infra (main.tf reads them from SSM).

resource "aws_lambda_function" "app" {
  count = local.use_lambda ? 1 : 0

  function_name = local.name
  role          = data.aws_ssm_parameter.runtime_role_arn.insecure_value
  package_type  = "Image"
  image_uri     = var.image_uri # a digest, so this exact image and no other
  architectures = ["x86_64"]    # must match how build_release.sh built it (--platform linux/amd64)
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_seconds

  environment {
    variables = local.app_environment
  }

  logging_config {
    log_format = "Text"
    log_group  = data.aws_ssm_parameter.log_group.insecure_value
  }

  # Nothing is created in the wrong workspace: the guard (main.tf) fails the plan first.
  depends_on = [terraform_data.workspace_matches_env]
}

# A free HTTPS address. NONE = anyone with the link can call it: the service is a demo, so it is
# public; production data belongs behind real authentication.
resource "aws_lambda_function_url" "app" {
  count = local.use_lambda ? 1 : 0

  function_name      = aws_lambda_function.app[0].function_name
  authorization_type = "NONE"
}

# The public URL needs two resource-policy statements: this one (InvokeFunctionUrl) and
# lambda:InvokeFunction, which provider 6.28+ adds for a NONE URL by itself.
resource "aws_lambda_permission" "public_url" {
  count = local.use_lambda ? 1 : 0

  statement_id           = "AllowPublicFunctionUrl"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.app[0].function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

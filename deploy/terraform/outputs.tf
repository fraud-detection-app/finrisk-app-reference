# Outputs print after `terraform apply` -- and any time with `terraform output`.

locals {
  # Express reports its addresses as a list of {access_type, endpoint}. Prefer the PUBLIC one,
  # and add https:// if the endpoint comes back as a bare host name.
  ecs_ingress  = try(aws_ecs_express_gateway_service.app[0].ingress_paths, [])
  ecs_public   = [for p in local.ecs_ingress : p.endpoint if p.access_type == "PUBLIC"]
  ecs_endpoint = try(local.ecs_public[0], try(local.ecs_ingress[0].endpoint, ""))
  ecs_url = (
    local.ecs_endpoint == "" ? "" :
    startswith(local.ecs_endpoint, "http") ? local.ecs_endpoint : "https://${local.ecs_endpoint}"
  )

  # A Lambda Function URL ends in "/"; trimming it means "$URL/health" works the same in all four environments.
  app_url = local.use_lambda ? trimsuffix(aws_lambda_function_url.app[0].function_url, "/") : trimsuffix(local.ecs_url, "/")
}

output "url" {
  description = "Where the app answers: a Function URL (dev, sit) or the Express Mode address (pre-prod, prod), never ending in a slash. Try <url>/health."
  value       = local.app_url
}

output "compute" {
  description = "lambda or ecs."
  value       = var.compute
}

output "image_uri" {
  description = "The image this environment runs. Compare it across environments: it is the same digest."
  value       = var.image_uri
}

output "env" {
  description = "The environment name."
  value       = var.env
}

output "bucket_name" {
  description = "The environment's data bucket (created by finrisk-infra)."
  value       = data.aws_ssm_parameter.bucket.insecure_value
}

output "log_group" {
  description = "Where the app's logs go (CloudWatch; created by finrisk-infra)."
  value       = data.aws_ssm_parameter.log_group.insecure_value
}

output "pointer_prefix" {
  description = "The SSM path the app reads its model and index pointers from."
  value       = local.prefix
}

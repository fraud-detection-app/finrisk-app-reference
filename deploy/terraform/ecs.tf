# pre-prod and prod: the SAME image on ECS Express Mode -- Fargate behind a load balancer with
# HTTPS, always warm (no cold start), about USD 77/month each at 1 vCPU / 2 GB.
# Everything here exists only when compute = "ecs" (count = 0 otherwise).
#
# Express Mode needs three roles. All three came from finrisk-infra (main.tf reads them from SSM):
#   task role            the app's own identity: its bucket, its logs, Bedrock, the S3 artifacts, the SSM pointers
#   execution role       lets ECS pull the image from ECR and write container logs
#   infrastructure role  lets ECS build the load balancer, security groups and scaling for us
#
# Express Mode takes about 13 minutes to answer on a first create, and a rolling update keeps the OLD task
# answering until the new one is healthy -- which is why the pipeline's smoke test waits for the NEW digest.

resource "aws_ecs_express_gateway_service" "app" {
  count = local.use_ecs ? 1 : 0

  service_name            = local.name
  execution_role_arn      = data.aws_ssm_parameter.ecs_execution_role_arn[0].insecure_value
  infrastructure_role_arn = data.aws_ssm_parameter.ecs_infrastructure_role_arn[0].insecure_value
  task_role_arn           = data.aws_ssm_parameter.runtime_role_arn.insecure_value
  cpu                     = var.ecs_cpu
  memory                  = var.ecs_memory
  health_check_path       = "/health"

  primary_container {
    image          = var.image_uri # a digest, exactly as in dev and sit
    container_port = 8000

    aws_logs_configuration {
      log_group         = data.aws_ssm_parameter.log_group.insecure_value
      log_stream_prefix = "app"
    }

    dynamic "environment" {
      for_each = local.app_environment
      content {
        name  = environment.key
        value = environment.value
      }
    }
  }

  # One task, always: min = max = 1 (no scaling surprises on the bill).
  scaling_target {
    min_task_count            = 1
    max_task_count            = 1
    auto_scaling_metric       = "AVERAGE_CPU"
    auto_scaling_target_value = 60
  }

  # No network_configuration: Express uses the account's default VPC subnets and a security group
  # it manages. (Deleting the default VPC means adding one here.)

  # Nothing is created in the wrong workspace: the guard (main.tf) fails the plan first.
  depends_on = [terraform_data.workspace_matches_env]
}

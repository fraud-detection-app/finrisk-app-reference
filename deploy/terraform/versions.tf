# Which Terraform and which provider this folder needs. (Where the state is kept: backend.tf.)
#
# Terraform 1.15+  : its S3 backend understands `aws login` credentials.
# AWS provider 6.68+: reads `aws login` credentials (6.23+), account-regional bucket names (6.37+),
#                     adds a public Function URL's invoke permission for you (6.28+) and has the
#                     ECS Express Mode resource (aws_ecs_express_gateway_service).
# The exact provider version is recorded in .terraform.lock.hcl -- commit that file.

terraform {
  required_version = ">= 1.15"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.68"
    }
  }
}

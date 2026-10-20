# The inputs. envs/<env>.tfvars sets the per-environment ones; image_uri comes from the pipeline:
#   terraform apply -var-file=envs/dev.tfvars -var image_uri=<repo>@sha256:<digest>

variable "project" {
  description = "Short project name; the first part of every resource name."
  type        = string
  default     = "finrisk"
}

variable "env" {
  description = "Which environment this copy is. Must match the Terraform workspace."
  type        = string

  validation {
    condition     = contains(["dev", "sit", "pre-prod", "prod"], var.env)
    error_message = "env must be dev, sit, pre-prod or prod."
  }
}

variable "compute" {
  description = "What runs the container: lambda (scales to zero) or ecs (ECS Express Mode, always warm)."
  type        = string

  validation {
    condition     = contains(["lambda", "ecs"], var.compute)
    error_message = "compute must be lambda or ecs."
  }
}

variable "image_uri" {
  description = "The image to run, BY DIGEST: <account>.dkr.ecr.<region>.amazonaws.com/finrisk-app@sha256:<64 hex>."
  type        = string

  validation {
    # A tag can be moved to a different image; a digest never can. Deploying by digest is what makes
    # "the thing that passed sit" and "the thing that runs in prod" provably the same bytes.
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.image_uri))
    error_message = "image_uri must point at a digest (...@sha256:<64 hex characters>), not a tag. The pipeline resolves it from the ECR tag sha-<commit>."
  }
}

variable "aws_region" {
  description = "Where everything lives. Terraform does NOT follow the console's Region menu."
  type        = string
  default     = "ap-southeast-2" # Sydney
}

variable "lambda_memory_mb" {
  description = "Lambda memory in MB (CPU grows with memory). New AWS accounts are capped at 3008 MB until a quota increase. Used when compute = lambda."
  type        = number
  default     = 2048

  validation {
    condition     = var.lambda_memory_mb >= 512 && var.lambda_memory_mb <= 3008
    error_message = "lambda_memory_mb must be between 512 and 3008 (the ceiling of a new AWS account)."
  }
}

variable "lambda_timeout_seconds" {
  description = "Lambda timeout. One request can wait up to INLINE_LOAD_TIMEOUT_SECONDS (20) for a new index to load and then call the model endpoint (ENDPOINT_READ_TIMEOUT_SECONDS is set to 25 on Lambda: at most 5 connect + 25 read + 5 slack = 35 s) or Bedrock (up to ~30 s with its retries): 20 + 35 = 55 fits in 60, not in 30. If you lower this, lower the endpoint read timeout in main.tf too."
  type        = number
  default     = 60
}

variable "ecs_cpu" {
  description = "Express Mode task CPU units (1024 = 1 vCPU). Used when compute = ecs."
  type        = string
  default     = "1024"
}

variable "ecs_memory" {
  description = "Express Mode task memory in MiB. Used when compute = ecs."
  type        = string
  default     = "2048"
}

variable "bedrock_model_id" {
  description = "The Bedrock chat model the app calls through its role. No API key anywhere."
  type        = string
  default     = "deepseek.v3.2" # ON_DEMAND in ap-southeast-2 (checked 14 Oct 2026)
}

variable "pointer_ttl_seconds" {
  description = "How often (at most) the app looks at its SSM pointers: a new model or index reaches the service within about this time."
  type        = number
  default     = 60
}

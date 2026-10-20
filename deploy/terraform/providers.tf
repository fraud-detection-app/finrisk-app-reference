# The AWS provider: which Region, and tags every resource gets automatically -- who owns it,
# which environment, and that Terraform manages it (so nobody "fixes" it by hand in the console).

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project
      Environment = var.env
      ManagedBy   = "terraform"
      Course      = "financial-risk-assistant"
    }
  }
}

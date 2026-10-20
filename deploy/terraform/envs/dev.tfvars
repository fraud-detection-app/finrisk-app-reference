# dev -- developers try things; Lambda scales to zero.
# Run:  terraform workspace select -or-create dev
#       terraform apply -var-file=envs/dev.tfvars -var image_uri=<repo>@sha256:<digest>

env              = "dev"
compute          = "lambda"
lambda_memory_mb = 2048

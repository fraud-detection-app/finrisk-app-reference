# sit -- system integration tests; same Lambda compute as dev.
# Run:  terraform workspace select -or-create sit
#       terraform apply -var-file=envs/sit.tfvars -var image_uri=<repo>@sha256:<digest>

env              = "sit"
compute          = "lambda"
lambda_memory_mb = 2048

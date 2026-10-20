# prod -- real users; ECS Express Mode, always warm.
# Run:  terraform workspace select -or-create prod
#       terraform apply -var-file=envs/prod.tfvars -var image_uri=<repo>@sha256:<digest>

env        = "prod"
compute    = "ecs"
ecs_cpu    = "1024" # 1 vCPU
ecs_memory = "2048" # MiB

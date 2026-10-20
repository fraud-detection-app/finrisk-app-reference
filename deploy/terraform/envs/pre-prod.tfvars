# pre-prod -- rehearsal; prod-like: ECS Express Mode, always warm.
# Run:  terraform workspace select -or-create pre-prod
#       terraform apply -var-file=envs/pre-prod.tfvars -var image_uri=<repo>@sha256:<digest>

env        = "pre-prod"
compute    = "ecs"
ecs_cpu    = "1024" # 1 vCPU
ecs_memory = "2048" # MiB

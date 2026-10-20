# WHERE Terraform keeps this stack's state: the S3 bucket finrisk-infra created (bootstrap).
#
# A PARTIAL backend: "use S3" is written here, the details are not. They differ per account, so the pipeline
# passes them at init time (deliver.yml / rollback.yml / teardown.yml):
#
#   terraform init -backend-config="bucket=$TF_STATE_BUCKET" \
#                  -backend-config="key=app/service.tfstate" \
#                  -backend-config="region=$AWS_REGION" -backend-config="use_lockfile=true"
#
# With workspaces, S3 stores each environment's state at env:/<workspace>/app/service.tfstate -- one file per
# environment, so dev's state can never overwrite prod's. (finrisk-infra keeps ITS state under other keys:
# infra/shared.tfstate and infra/env.tfstate. Two stacks, two keys, no overlap.)

terraform {
  backend "s3" {}
}

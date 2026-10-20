#!/usr/bin/env bash
# Build the RELEASE image: finrisk-app:<tag> (default "dev").
#
#   bash scripts/build_release.sh                   # -> finrisk-app:dev
#   bash scripts/build_release.sh sha-0123abc...    # -> finrisk-app:sha-0123abc...   (what the pipeline does)
#
# A release is built ONCE, when a pull request merges into main. Later the very same image is promoted
# dev -> sit -> pre-prod -> prod by digest; nothing is rebuilt per environment. Pushing to ECR is a
# separate, explicit step (deliver.yml does it; by hand: docs/runbook.md).
#
# Two builds, one after the other, because the second starts FROM the first:
#   1. the service image  (Dockerfile)      code + libraries
#   2. the release image  (Dockerfile.aws)  + the baked fallback model and index + the Lambda Web Adapter
#
# linux/amd64: what Lambda and ECS run here (override: PLATFORM=linux/arm64 -- then set the compute to arm64
# too). --provenance=false: Lambda rejects the extra manifest that buildx adds by default.
# --load: keep the result in the local Docker, where build 2 finds build 1.
set -euo pipefail

TAG="${1:-dev}"
PLATFORM="${PLATFORM:-linux/amd64}"
cd "$(dirname "$0")/.."                       # the repository folder, wherever you run this from

echo "==> 1/2 the service image finrisk-service:$TAG ($PLATFORM)"
docker buildx build --platform "$PLATFORM" --provenance=false --load -t "finrisk-service:$TAG" .

echo "==> 2/2 the release image finrisk-app:$TAG (baked fallback model + index, Lambda Web Adapter)"
docker buildx build --platform "$PLATFORM" --provenance=false --load \
  --build-arg "SERVICE_IMAGE=finrisk-service:$TAG" \
  -t "finrisk-app:$TAG" -f Dockerfile.aws .

ID="$(docker image inspect --format '{{.Id}}' "finrisk-app:$TAG")"
SIZE="$(docker image ls --format '{{.Size}}' "finrisk-app:$TAG")"
echo "==> done: finrisk-app:$TAG"
echo "    image id : $ID"
echo "    size     : $SIZE"
echo "    try it   : docker run --rm -p 8000:8000 -e LLM_PROVIDER=none finrisk-app:$TAG   # then open http://127.0.0.1:8000/ready"

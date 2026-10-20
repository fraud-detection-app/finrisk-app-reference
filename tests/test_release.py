"""The release image is built once, from reviewed files only.

Static checks -- no Docker, no AWS. They read Dockerfile, Dockerfile.aws and scripts/build_release.sh as text and
keep the promises: one adapter line, the baked fallback made INSIDE the build (nothing binary in git), two amd64
builds without provenance, and no secret anywhere.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
DOCKERFILE_AWS = (ROOT / "Dockerfile.aws").read_text(encoding="utf-8")
BUILD_SCRIPT_PATH = ROOT / "scripts" / "build_release.sh"
BUILD_SCRIPT = BUILD_SCRIPT_PATH.read_text(encoding="utf-8")


def instructions(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def test_release_dockerfile_builds_on_the_service_image():
    lines = instructions(DOCKERFILE_AWS)
    assert lines[0].startswith("ARG SERVICE_IMAGE")
    assert lines[1] == "FROM ${SERVICE_IMAGE}"


def test_release_dockerfile_makes_the_baked_fallback_inside_the_build():
    lines = instructions(DOCKERFILE_AWS)
    runs = [l for l in lines if l.startswith("RUN ")]
    assert len(runs) == 1 and "scripts/run_pipeline.py --demo-rows 5000" in runs[0] and "DATA_MODE=demo" in runs[0]
    assert "rm -rf data/raw data/bronze data/silver" in runs[0] and "reports" in runs[0]      # raw data and reports do not ship
    assert not [l for l in lines if l.startswith("COPY ") and "--from" not in l]               # no file from the build context at all


def test_release_dockerfile_keeps_the_lambda_web_adapter():
    adapter = [l for l in instructions(DOCKERFILE_AWS) if "aws-lambda-adapter" in l]
    assert adapter == ["COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter"]


def test_release_dockerfile_carries_no_secrets_or_keys():
    code = "\n".join(instructions(DOCKERFILE_AWS) + instructions(DOCKERFILE)).lower()
    for forbidden in ("api_key", "secret", "aws_access_key"):
        assert forbidden not in code, forbidden
    assert not re.search(r"(^|[\s/])\.env\b", code), "no .env file may be copied into an image"


def test_service_image_keeps_what_the_training_image_needs():
    """finrisk-ml builds FROM the release image and runs /app/scripts/run_pipeline.py in it."""
    copies = [l for l in instructions(DOCKERFILE) if l.startswith("COPY ")]
    for needed in ("COPY VERSION ./VERSION", "COPY src ./src", "COPY api ./api", "COPY scripts ./scripts", "COPY config ./config"):
        assert needed in copies, needed
    assert "WORKDIR /app" in instructions(DOCKERFILE)


def test_build_script_is_strict_bash_and_parses():
    assert BUILD_SCRIPT.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in BUILD_SCRIPT
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    result = subprocess.run([bash, "-n", str(BUILD_SCRIPT_PATH)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_build_script_builds_both_images_for_amd64_without_provenance():
    text = BUILD_SCRIPT.replace("\\\n", " ")
    builds = [line.strip() for line in text.splitlines() if line.startswith("docker buildx build")]
    assert len(builds) == 2
    for build in builds:
        assert '--platform "$PLATFORM"' in build and "--provenance=false" in build and "--load" in build
    service, release = builds
    assert service.rstrip().endswith('-t "finrisk-service:$TAG" .')
    assert 'SERVICE_IMAGE=finrisk-service:$TAG' in release
    assert '-t "finrisk-app:$TAG" -f Dockerfile.aws .' in release
    assert 'TAG="${1:-dev}"' in BUILD_SCRIPT and 'PLATFORM="${PLATFORM:-linux/amd64}"' in BUILD_SCRIPT
    assert "data/gold" not in BUILD_SCRIPT and "joblib" not in BUILD_SCRIPT      # nothing is staged on the host any more


def test_service_image_dockerignore_keeps_data_and_secrets_out():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    for entry in ("data/", "models/", "vector_index*/", ".env", "tests/", "deploy/", ".github/"):
        assert entry in ignored, entry
    assert "VERSION" not in ignored                      # the image reports it


def test_service_image_installs_boto3_for_ssm_s3_and_bedrock():
    assert re.search(r"^boto3\b", (ROOT / "requirements-service.txt").read_text(encoding="utf-8"), flags=re.M)


def test_the_version_file_is_semver_and_the_app_reports_it():
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    from api.main import app
    assert app.version == version


def test_generated_folders_are_git_ignored_but_the_code_packages_are_not():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    for entry in ("/data/", "/models/", "/vector_index/", "/build/", ".env", ".terraform/", "*.tfstate", "backend.hcl"):
        assert entry in lines, entry
    assert "data/" not in lines and "models/" not in lines      # unanchored would hide src/financial_risk/data and /models
    assert "!.env.example" in lines

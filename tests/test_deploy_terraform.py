"""deploy/terraform -- the service stack: formatting, the four environments' tfvars, what it reads from finrisk-infra
and what it refuses to create, and the offline `terraform test`s (mock AWS provider, fake SSM values: nothing is
read from or written to an account). Skipped when Terraform is not installed (the `terraform` CI job runs them)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STACK = Path(__file__).resolve().parents[1] / "deploy" / "terraform"

ENVIRONMENTS = {  # env -> compute
    "dev": "lambda", "sit": "lambda", "pre-prod": "ecs", "prod": "ecs",
}


def tf_text(*names: str) -> str:
    return "\n".join((STACK / n).read_text(encoding="utf-8") for n in names)


def all_tf() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(STACK.glob("*.tf")))


# ---------------------------------------------------------------------------- static checks (no Terraform needed)


def test_the_stack_reads_infras_parameters_and_creates_no_role_bucket_or_log_group():
    code = all_tf()
    assert not re.findall(r'resource\s+"(aws_iam_[a-z_]+|aws_s3_bucket[a-z_]*|aws_cloudwatch_log_group|aws_ecr_[a-z_]+|aws_kms_[a-z_]+)"', code)
    reads = sorted(re.findall(r'name\s*=\s*"\$\{local\.prefix\}/(\w+)"', tf_text("main.tf")))
    assert reads == ["bucket", "ecs_execution_role_arn", "ecs_infrastructure_role_arn", "log_group", "runtime_role_arn"]
    assert "insecure_value" in code and re.search(r'data\s+"aws_ssm_parameter"', code)


def test_the_only_parameter_it_writes_is_app_url():
    written = re.findall(r'resource\s+"aws_ssm_parameter"\s+"(\w+)"', all_tf())
    assert written == ["app_url"]
    assert '"${local.prefix}/app_url"' in tf_text("ssm.tf")


def test_the_container_gets_the_documented_settings():
    main = tf_text("main.tf")
    for setting in ("APP_ENV", "LLM_PROVIDER", "BEDROCK_MODEL_ID", "APP_IMAGE_DIGEST", "POINTER_PREFIX", "POINTER_TTL_SECONDS"):
        assert re.search(rf"\b{setting}\s*=", main), setting
    assert '"bedrock"' in main and "MODEL_S3_URI" not in all_tf()
    assert 'default     = "deepseek.v3.2"' in tf_text("variables.tf") and 'default     = 60' in tf_text("variables.tf")


def test_the_backend_is_a_partial_s3_backend_and_the_workspace_guard_exists():
    assert re.search(r'backend\s+"s3"\s*\{\s*\}', tf_text("backend.tf"))
    assert "bucket" not in re.sub(r"#.*", "", tf_text("backend.tf"))                  # nothing account-specific in git
    assert "terraform.workspace == var.env" in tf_text("main.tf")


@pytest.mark.parametrize("env,compute", ENVIRONMENTS.items())
def test_each_environment_has_its_tfvars(env, compute):
    values = dict(re.findall(r'^(\w+)\s*=\s*"?([^"\n#]+?)"?\s*(?:#.*)?$', (STACK / "envs" / f"{env}.tfvars").read_text(), flags=re.M))
    assert values["env"] == env and values["compute"] == compute
    if compute == "lambda":
        assert int(values["lambda_memory_mb"]) <= 3008
    else:
        assert values["ecs_cpu"] and values["ecs_memory"]
    assert "image_uri" not in values                                                 # the digest comes from the pipeline


def test_the_lock_file_pins_the_provider_for_five_platforms():
    lock = (STACK / ".terraform.lock.hcl").read_text()
    assert 'version     = "6.68.0"' in lock and len(re.findall(r'"h1:', lock)) == 5


def test_the_tests_cover_both_computes_and_the_rejections():
    assert sorted(p.name for p in (STACK / "tests").glob("*.tftest.hcl")) == ["dev.tftest.hcl", "pre-prod.tftest.hcl", "rejects.tftest.hcl"]


# ---------------------------------------------------------------------------- Terraform itself

needs_terraform = pytest.mark.skipif(shutil.which("terraform") is None, reason="terraform not installed")


def _tf(args: list[str], cwd: Path, timeout: float = 300) -> subprocess.CompletedProcess:
    env = dict(os.environ, CHECKPOINT_DISABLE="1", TF_IN_AUTOMATION="1", NO_COLOR="1")
    cache = Path.home() / ".terraform.d" / "plugin-cache"
    cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("TF_PLUGIN_CACHE_DIR", str(cache))
    return subprocess.run(["terraform", *args], cwd=cwd, capture_output=True, text=True, env=env, timeout=timeout)


@pytest.fixture(scope="module")
def scratch(tmp_path_factory) -> Path:
    """A copy with local state (no S3 backend), initialised once. Never run inside the repo folder."""
    if shutil.which("terraform") is None:
        pytest.skip("terraform not installed")
    work = tmp_path_factory.mktemp("svc") / "stack"
    shutil.copytree(STACK, work)
    (work / "backend.tf").unlink()
    result = _tf(["init", "-input=false"], work)
    if result.returncode != 0:
        pytest.skip(f"terraform init needs the AWS provider (offline and not cached?): {result.stderr[-300:]}")
    return work


@needs_terraform
def test_terraform_fmt_is_clean():
    result = _tf(["fmt", "-check", "-recursive", "-diff"], STACK)
    assert result.returncode == 0, result.stdout + result.stderr


@needs_terraform
def test_terraform_validate_passes(scratch):
    result = _tf(["validate"], scratch)
    assert result.returncode == 0, result.stdout + result.stderr


@needs_terraform
@pytest.mark.parametrize("test_file,workspace", [("dev", "dev"), ("rejects", "dev"), ("pre-prod", "pre-prod")])
def test_terraform_test_passes(scratch, test_file, workspace):
    assert _tf(["workspace", "select", "-or-create", workspace], scratch).returncode == 0
    result = _tf(["test", f"-filter=tests/{test_file}.tftest.hcl"], scratch)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-500:]
    assert "0 failed" in result.stdout


def test_the_endpoint_read_timeout_is_shortened_on_lambda_only():
    """Lambda stops at 60 s: 20 s inline index reload + (5 connect + 25 read + 5 slack) = 55. ECS keeps the app's default (40)."""
    main = re.sub(r"#.*", "", tf_text("main.tf"))
    assert len(re.findall(r'ENDPOINT_READ_TIMEOUT_SECONDS\s*=\s*"25"', main)) == 1
    merged = re.search(r"app_environment\s*=\s*merge\(\{.*?\},\s*local\.use_lambda\s*\?\s*\{(?P<lambda_only>.*?)\}\s*:\s*\{\}\)", main, flags=re.S)
    assert merged and 'ENDPOINT_READ_TIMEOUT_SECONDS = "25"' in merged.group("lambda_only")        # only inside the Lambda branch
    assert "ENDPOINT_READ_TIMEOUT_SECONDS" not in merged.group(0).split("local.use_lambda")[0]      # not in the shared map
    description = re.search(r'variable "lambda_timeout_seconds"\s*\{\s*description\s*=\s*"([^"]+)"', tf_text("variables.tf")).group(1)
    assert "ENDPOINT_READ_TIMEOUT_SECONDS" in description and "55" in description and "default     = 60" in tf_text("variables.tf")

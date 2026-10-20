"""The GitHub Actions workflows, checked as text and as YAML -- no GitHub, no AWS.

What these pin down is the PIPELINE'S SHAPE: build once, deploys chained dev -> sit -> pre-prod -> prod each in its own
Environment, release only after prod, every AWS login followed by an account-id mask, and only the documented secrets.
`actionlint` is not installed here; these checks are the offline substitute (and `bash -n` on every script).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
NAMES = ["ci", "deliver", "deploy-env", "rollback", "teardown"]
ENVIRONMENTS = ["dev", "sit", "pre-prod", "prod"]

ALLOWED_SECRETS = {"AWS_BUILD_ROLE_ARN", "AWS_ROLE_ARN", "ECR_REPOSITORY", "TF_STATE_BUCKET"}
ALLOWED_VARS = {"AWS_REGION"}


def load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / f"{name}.yml").read_text(encoding="utf-8"))


def text(name: str) -> str:
    return (WORKFLOWS / f"{name}.yml").read_text(encoding="utf-8")


def triggers(workflow: dict) -> dict:
    return workflow.get("on") or workflow.get(True)       # YAML 1.1 reads a bare `on:` as the boolean True


def steps(job: dict) -> list[dict]:
    return job.get("steps", [])


def run_scripts(workflow: dict) -> list[tuple[str, str]]:
    return [(f"{job_id}/{s.get('name', s.get('uses', '?'))}", s["run"]) for job_id, job in workflow["jobs"].items()
            for s in steps(job) if "run" in s]


@pytest.fixture(scope="module", params=NAMES)
def name(request):
    return request.param


# ---------------------------------------------------------------------------- the set and the basics


def test_exactly_the_expected_workflows_exist():
    assert sorted(p.stem for p in WORKFLOWS.glob("*.yml")) == sorted(NAMES)


def test_every_workflow_parses_and_has_a_name_trigger_and_jobs(name):
    workflow = load(name)
    assert workflow["name"] == name and triggers(workflow) and workflow["jobs"]
    for job in workflow["jobs"].values():
        assert "runs-on" in job or "uses" in job


def test_every_job_runs_on_the_pinned_runner_image(name):
    for job_id, job in load(name)["jobs"].items():
        if "uses" in job:                                    # a job that calls a reusable workflow has no runner of its own
            continue
        assert job["runs-on"] == "ubuntu-24.04", f"{name} job {job_id}: pin the runner (ubuntu-24.04), not a moving label like ubuntu-latest"


def test_every_run_script_is_valid_bash(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    for workflow_name in NAMES:
        for label, script in run_scripts(load(workflow_name)):
            script = re.sub(r"\$\{\{.*?\}\}", "X", script)           # an expression is just a word to bash
            file = tmp_path / "step.sh"
            file.write_text(script)
            result = subprocess.run([bash, "-n", str(file)], capture_output=True, text=True)
            assert result.returncode == 0, f"{workflow_name}: {label}\n{result.stderr}"


def test_the_workflows_default_to_read_only_and_every_run_uses_bash():
    for workflow_name in NAMES:
        workflow = load(workflow_name)
        assert workflow["permissions"] == {"contents": "read"}, workflow_name
        assert workflow["defaults"]["run"]["shell"] == "bash", workflow_name


# ---------------------------------------------------------------------------- secrets, variables, keys


def test_only_the_documented_secrets_and_variables_and_no_long_lived_aws_keys():
    for workflow_name in NAMES:
        body = text(workflow_name)
        assert set(re.findall(r"secrets\.([A-Z0-9_]+)", body)) <= ALLOWED_SECRETS, workflow_name
        assert set(re.findall(r"vars\.([A-Z0-9_]+)", body)) <= ALLOWED_VARS, workflow_name
        assert "AWS_ACCESS_KEY_ID" not in body and "AWS_SECRET_ACCESS_KEY" not in body, workflow_name


def test_deliver_documents_every_secret_and_variable_in_a_table_at_the_top():
    head = text("deliver").split("name: deliver")[0]
    for item in sorted(ALLOWED_SECRETS | ALLOWED_VARS):
        assert item in head, item
    assert "Required status checks on main" in head and "`test` and `terraform`" in head
    assert "kind" in head and "variable" in head and "secret" in head


def test_a_job_with_an_environment_reads_AWS_ROLE_ARN_and_one_without_reads_a_repository_secret():
    deliver, deploy_env, teardown = load("deliver"), load("deploy-env"), load("teardown")
    build = text("deliver")
    assert "role-to-assume: ${{ secrets.AWS_BUILD_ROLE_ARN }}" in build and "environment" not in deliver["jobs"]["build"]
    for workflow in (deploy_env, teardown):
        job = next(j for j in workflow["jobs"].values() if "environment" in j)
        assert "role-to-assume: ${{ secrets.AWS_ROLE_ARN }}" in yaml.safe_dump(job).replace("\n    ", " ") or \
            any(s.get("with", {}).get("role-to-assume") == "${{ secrets.AWS_ROLE_ARN }}" for s in steps(job))
    assert "AWS_BUILD_ROLE_ARN" not in text("deploy-env") and "AWS_BUILD_ROLE_ARN" not in text("teardown")


def test_a_secret_is_never_written_into_a_script_or_printed():
    for workflow_name in NAMES:
        for label, script in run_scripts(load(workflow_name)):
            assert "${{ secrets." not in script, f"{workflow_name}: {label} pastes a secret into a script (use env:)"
            for line in script.splitlines():
                if re.match(r"\s*(echo|printf|cat)\b", line):
                    assert not re.search(r"\$\{?(ECR_REPOSITORY|TF_STATE_BUCKET|AWS_ROLE_ARN|AWS_BUILD_ROLE_ARN)\b", line) \
                        or "ECR_REPOSITORY%%" in line or "IMAGE_URI" in line, f"{workflow_name}: {label}: {line}"


def test_nothing_a_person_types_is_pasted_into_a_shell_script():
    for workflow_name in NAMES:
        for label, script in run_scripts(load(workflow_name)):
            assert not re.search(r"\$\{\{\s*(inputs\.|github\.head_ref|github\.event\.)", script), f"{workflow_name}: {label}"


def test_every_job_that_assumes_a_role_may_request_the_oidc_token_and_masks_the_account_id():
    for workflow_name in NAMES:
        workflow = load(workflow_name)
        for job_id, job in workflow["jobs"].items():
            uses_role = any("configure-aws-credentials" in s.get("uses", "") for s in steps(job))
            if not uses_role:
                continue
            assert job["permissions"]["id-token"] == "write", (workflow_name, job_id)
            names = [s.get("uses", "") + " " + s.get("run", "") for s in steps(job)]
            login = next(i for i, n in enumerate(names) if "configure-aws-credentials" in n)
            mask = next(i for i, n in enumerate(names) if "::add-mask::" in n and "sts get-caller-identity" in n)
            assert mask == login + 1, f"{workflow_name}/{job_id}: mask the account id right after the login"


def test_a_job_that_calls_a_reusable_deploy_grants_it_the_token():
    for workflow_name in ("deliver", "rollback"):
        for job_id, job in load(workflow_name)["jobs"].items():
            if job.get("uses") == "./.github/workflows/deploy-env.yml":
                assert job["permissions"]["id-token"] == "write" and job["secrets"] == "inherit", (workflow_name, job_id)


# ---------------------------------------------------------------------------- pinned tools


def test_actions_are_pinned_to_the_current_majors():
    expected = {"actions/checkout": "v7", "actions/setup-python": "v7", "aws-actions/configure-aws-credentials": "v6",
                "hashicorp/setup-terraform": "v4", "actions/upload-artifact": "v7", "actions/download-artifact": "v8"}
    for workflow_name in NAMES:
        for action, ref in re.findall(r"uses:\s*([\w.-]+/[\w.-]+)@(\S+)", text(workflow_name)):
            assert action in expected and ref == expected[action], (workflow_name, action, ref)


def test_terraform_is_pinned_and_unwrapped_everywhere():
    for workflow_name in NAMES:
        for job in load(workflow_name)["jobs"].values():
            for s in steps(job):
                if "setup-terraform" in s.get("uses", ""):
                    assert s["with"] == {"terraform_version": "1.15.8", "terraform_wrapper": False}


def test_the_image_build_uses_the_runners_default_builder_not_buildx_setup():
    for workflow_name in NAMES:
        assert "setup-buildx" not in "\n".join(l for l in text(workflow_name).splitlines() if "uses:" in l), workflow_name
    assert "docker/setup-buildx-action" not in text("deliver").replace("NO docker/setup-buildx-action here", "")
    assert "NO docker/setup-buildx-action" in text("deliver")           # ... and the reason is written next to it


# ---------------------------------------------------------------------------- ci


def test_ci_has_the_two_checks_the_ruleset_requires():
    ci = load("ci")
    assert set(ci["jobs"]) == {"test", "terraform"}
    assert ci["jobs"]["test"]["name"] == "test" and ci["jobs"]["terraform"]["name"] == "terraform"
    assert set(triggers(ci)) == {"pull_request", "push"} and triggers(ci)["push"]["branches"] == ["main"]


def test_ci_tests_with_the_demo_model_and_checks_terraform_format_validate_and_tests():
    body = text("ci")
    assert "scripts/run_pipeline.py --demo-rows 5000" in body and "python -m pytest -q" in body
    assert "terraform fmt -check -recursive -diff deploy" in body
    assert "init -backend=false" in body and "validate" in body and "terraform test" in body
    assert "tests/*.tftest.hcl" in body and "workspace select -or-create" in body


# ---------------------------------------------------------------------------- deliver


def test_deliver_runs_on_push_to_main_and_ignores_markdown():
    on = triggers(load("deliver"))
    assert on["push"]["branches"] == ["main"] and on["push"]["paths-ignore"] == ["**/*.md"]
    assert set(on) == {"push"}


def test_deliver_builds_once_by_commit_sha_and_skips_an_existing_tag():
    build = load("deliver")["jobs"]["build"]
    body = "\n".join(s["run"] for s in steps(build) if "run" in s)
    assert "imageTag=sha-$SHA" in body and "describe-images" in body
    assert "ImageNotFoundException" in body
    assert 'bash scripts/build_release.sh "sha-$SHA"' in body
    assert 'docker push "$ECR_REPOSITORY:sha-$SHA"' in body
    guarded = [s for s in steps(build) if s.get("name") in {"Build the release image", "Push to ECR"}]
    assert len(guarded) == 2 and all(s["if"] == "steps.exists.outputs.built == 'true'" for s in guarded)
    assert build["permissions"] == {"id-token": "write", "contents": "read"}


def test_the_four_deploys_are_chained_in_order_each_in_its_own_environment():
    jobs = load("deliver")["jobs"]
    chain = [("deploy-dev", "build", "dev"), ("deploy-sit", "deploy-dev", "sit"),
             ("deploy-pre-prod", "deploy-sit", "pre-prod"), ("deploy-prod", "deploy-pre-prod", "prod")]
    for job_id, needs, environment in chain:
        job = jobs[job_id]
        assert job["needs"] == needs, job_id
        assert job["uses"] == "./.github/workflows/deploy-env.yml"
        assert job["with"] == {"environment": environment, "sha": "${{ github.sha }}"}
    # the reusable workflow is what names the Environment (the gate + the OIDC subject), from its input
    deploy = load("deploy-env")["jobs"]["deploy"]
    assert deploy["environment"] == "${{ inputs.environment }}"
    assert deploy["concurrency"] == {"group": "deploy-${{ inputs.environment }}", "cancel-in-progress": False}


def test_deploy_env_follows_the_deploy_contract():
    workflow = load("deploy-env")
    inputs = triggers(workflow)["workflow_call"]["inputs"]
    assert set(inputs) == {"environment", "sha", "reason"} and inputs["environment"]["required"] and inputs["sha"]["required"]
    assert triggers(workflow)["workflow_call"]["outputs"]["image_digest"]["value"] == "${{ jobs.deploy.outputs.image_digest }}"
    body = text("deploy-env")
    assert "imageTag=sha-$SHA" in body and "IMAGE_URI=$ECR_REPOSITORY@$DIGEST" in body           # deployed by digest, not by tag
    assert '-backend-config="key=app/service.tfstate"' in body and "use_lockfile=true" in body
    assert 'terraform workspace select -or-create "$ENVIRONMENT"' in body
    assert body.index("terraform plan") < body.index("terraform apply -input=false -no-color tfplan")
    assert '-var-file="envs/$ENVIRONMENT.tfvars"' in body and '-var "image_uri=$IMAGE_URI"' in body
    assert "working-directory: deploy/terraform" in body


def test_the_smoke_test_waits_for_the_new_digest_then_for_ready():
    body = text("deploy-env")
    assert "old version still answering" in body
    assert 'if [ "$code" = 200 ] && [ "$seen" = "$IMAGE_DIGEST" ]; then break; fi' in body
    assert "HEALTH_BUDGET=1080" in body and "HEALTH_BUDGET=300" in body            # Express 18 min, Lambda 5 min
    assert body.index("probe /health") < body.index("probe /ready")
    assert '"$(field "$RUNNER_TEMP/ready.json" image_digest)" != "$IMAGE_DIGEST"' in body
    assert "the version we deployed" in body.lower() or "Not the version we deployed" in body


def test_release_comes_only_after_prod_with_the_only_write_permission():
    jobs = load("deliver")["jobs"]
    release = jobs["release"]
    assert release["needs"] == "deploy-prod"
    assert release["permissions"] == {"contents": "write"}
    for job_id, job in jobs.items():
        if job_id != "release":
            assert job.get("permissions", {}).get("contents") != "write", job_id
    script = "\n".join(s["run"] for s in steps(release) if "run" in s)
    assert "VERSION" in script and 'TAG="v$VERSION"' in script and "gh release view" in script and "gh release create" in script
    assert '--target "$SHA"' in script
    assert "DIGEST" in release["env"] and "needs.deploy-prod.outputs.image_digest" in release["env"]["DIGEST"]


# ---------------------------------------------------------------------------- rollback and teardown


def test_rollback_takes_environment_and_sha_and_only_runs_from_main():
    workflow = load("rollback")
    inputs = triggers(workflow)["workflow_dispatch"]["inputs"]
    assert inputs["environment"]["type"] == "choice" and inputs["environment"]["options"] == ENVIRONMENTS
    assert inputs["sha"]["required"]
    jobs = workflow["jobs"]
    assert jobs["rollback"]["needs"] == "check" and jobs["rollback"]["uses"] == "./.github/workflows/deploy-env.yml"
    assert jobs["rollback"]["with"] == {"environment": "${{ inputs.environment }}", "sha": "${{ inputs.sha }}", "reason": "rollback"}
    check = "\n".join(s["run"] for s in steps(jobs["check"]))
    assert '"$BRANCH" != main' in check and "40 hex" in check or "{40}" in check


def test_teardown_needs_the_environment_name_typed_again_and_reads_the_image_from_the_state():
    workflow = load("teardown")
    inputs = triggers(workflow)["workflow_dispatch"]["inputs"]
    assert inputs["environment"]["options"] == ENVIRONMENTS and inputs["confirm"]["required"]
    check = "\n".join(s["run"] for s in steps(workflow["jobs"]["check"]))
    assert '"$CONFIRM" != "$ENVIRONMENT"' in check and '"$BRANCH" != main' in check
    destroy = workflow["jobs"]["destroy"]
    assert destroy["needs"] == "check" and destroy["environment"] == "${{ inputs.environment }}"
    assert destroy["concurrency"]["group"] == "deploy-${{ inputs.environment }}"
    body = "\n".join(s["run"] for s in steps(destroy) if "run" in s)
    assert "terraform output -raw image_uri" in body and "terraform destroy" in body and "terraform workspace delete" in body
    assert "app/service.tfstate" in body

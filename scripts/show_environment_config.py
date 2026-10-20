from __future__ import annotations

import argparse
import json
import shlex

import _bootstrap  # noqa: F401

from financial_risk.config import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print resolved project environment config.")
    parser.add_argument("--env", default=None, help="Environment: dev, sit, pre-prod, or prod.")
    parser.add_argument(
        "--format",
        choices=["json", "shell"],
        default="json",
        help="Output format.",
    )
    return parser.parse_args()


def config_payload(env: str | None = None) -> dict[str, object]:
    settings = get_settings(env)
    return {
        "APP_ENV": settings.app_env,
        "DATA_MODE": settings.data_mode,
        "LOG_LEVEL": settings.log_level,
        "AWS_REGION": settings.aws_region,
        "AWS_S3_BUCKET": settings.aws_s3_bucket,
        "AWS_S3_PREFIX": settings.aws_s3_prefix,
        "AWS_ATHENA_DATABASE": settings.athena_database,
        "AWS_ATHENA_WORKGROUP": settings.athena_workgroup,
        "ALLOW_CLOUD_DEPLOY": str(settings.allow_cloud_deploy).lower(),
        "REQUIRE_MANUAL_APPROVAL": str(settings.require_manual_approval).lower(),
        "BEDROCK_MODEL_ID": settings.bedrock_model_id,
        "BEDROCK_MAX_TOKENS": settings.bedrock_max_tokens,
        "BEDROCK_TEMPERATURE": settings.bedrock_temperature,
        "RAW_DATA_PATH": str(settings.raw_data_path.relative_to(settings.root)),
        "BRONZE_PATH": str(settings.bronze_path.relative_to(settings.root)),
        "SILVER_PATH": str(settings.silver_path.relative_to(settings.root)),
        "GOLD_PATH": str(settings.gold_path.relative_to(settings.root)),
        "MODEL_DIR": str(settings.model_dir.relative_to(settings.root)),
        "REPORT_DIR": str(settings.report_dir.relative_to(settings.root)),
        "VECTOR_INDEX_DIR": str(settings.vector_index_dir.relative_to(settings.root)),
        "RAG_DOCS_DIR": str(settings.rag_docs_dir.relative_to(settings.root)),
    }


def main() -> None:
    args = parse_args()
    payload = config_payload(args.env)
    if args.format == "shell":
        for key, value in payload.items():
            print(f"{key}={shlex.quote(str(value))}")
        return
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


VALID_APP_ENVS = {"dev", "sit", "pre-prod", "prod"}

# The Bedrock model every environment uses unless BEDROCK_MODEL_ID says otherwise. Kept in ONE
# place (config/environments/*.json repeat it for readers; a test keeps them equal).
# "deepseek.v3.2" is the exact id: confirmed in ap-southeast-2 (Sydney) on 14 Oct 2026 (ON_DEMAND).
DEFAULT_BEDROCK_MODEL_ID = "deepseek.v3.2"
DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/kohdified/synthetic-financial-data/resolve/main/paysim.csv"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(project_root() / ".env")


def _normalise_env(value: str | None) -> str:
    app_env = (value or "dev").strip().lower()
    if app_env == "preprod":
        app_env = "pre-prod"
    if app_env not in VALID_APP_ENVS:
        valid = ", ".join(sorted(VALID_APP_ENVS))
        raise ValueError(f"Unsupported APP_ENV={value!r}. Expected one of: {valid}")
    return app_env


def _load_env_config(root: Path, app_env: str) -> dict[str, object]:
    path = root / "config" / "environments" / f"{app_env}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _config_value(
    config: dict[str, object],
    env_name: str,
    config_name: str,
    default: str,
) -> str:
    value = os.getenv(env_name)
    if value is not None:
        return value
    raw = config.get(config_name, default)
    return str(raw)


def _config_int(config: dict[str, object], env_name: str, config_name: str, default: int) -> int:
    return int(_config_value(config, env_name, config_name, str(default)))


def _config_float(config: dict[str, object], env_name: str, config_name: str, default: float) -> float:
    return float(_config_value(config, env_name, config_name, str(default)))


def _config_bool(config: dict[str, object], env_name: str, config_name: str, default: bool) -> bool:
    value = _config_value(config, env_name, config_name, str(default).lower())
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _pointer_refresh(config: dict[str, object]) -> str:
    """POINTER_REFRESH=inline|background. Not set: inline on Lambda (AWS sets AWS_LAMBDA_FUNCTION_NAME there), else background."""
    mode = _config_value(config, "POINTER_REFRESH", "pointer_refresh", "").strip().lower()
    if not mode:
        mode = "inline" if os.getenv("AWS_LAMBDA_FUNCTION_NAME") else "background"
    if mode not in {"inline", "background"}:
        raise ValueError(f"Unsupported POINTER_REFRESH={mode!r}. Expected 'inline' or 'background'.")
    return mode


def _project_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class Settings:
    root: Path
    app_env: str
    env_config_path: Path
    data_mode: str
    log_level: str
    aws_region: str
    aws_s3_bucket: str
    aws_s3_prefix: str
    athena_database: str
    athena_workgroup: str
    allow_cloud_deploy: bool
    require_manual_approval: bool
    bedrock_model_id: str
    bedrock_max_tokens: int
    bedrock_temperature: float
    # Week 10: the LLM provider is a setting, not a hardcoded import.
    llm_provider: str
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    deepseek_max_tokens: int
    deepseek_temperature: float
    dataset_url: str
    raw_data_path: Path
    bronze_path: Path
    silver_path: Path
    gold_path: Path
    model_dir: Path
    report_dir: Path
    vector_index_dir: Path
    rag_docs_dir: Path
    # Pointer-driven artifacts (service/artifacts.py). POINTER_PREFIX=/finrisk/<env> turns it on: the
    # service reads <prefix>/model and <prefix>/index from SSM Parameter Store, downloads what they
    # name, and re-reads them every pointer_ttl_seconds. Empty = the baked demo model and index.
    pointer_prefix: str = ""
    pointer_ttl_seconds: int = 60
    # Where downloads are unpacked (/tmp is the one place Lambda lets a container write).
    artifact_dir: str = str(Path(tempfile.gettempdir()) / "finrisk-artifacts")
    # How long start-up waits for the first load before it starts answering (the load goes on in the
    # background; /ready says 503 "loading" until it is done). Lambda gives its init phase 10 s.
    startup_wait_seconds: float = 6.0
    # The embedding model used when an index does not name one (the index manifest wins).
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    # Who does the work when a pointer changed: "background" = a thread, the request never waits (Express, a laptop);
    # "inline" = the request that finds the TTL expired waits for it, at most inline_load_timeout_seconds (Lambda
    # freezes the process between requests, so a background thread would only move while requests arrive).
    pointer_refresh: str = "background"
    inline_load_timeout_seconds: float = 20.0
    # SageMaker endpoint calls (service/scoring.py). The whole scoring call is bounded by
    # connect + read + 5 s of slack (see EndpointScorer.budget), so on Lambda (timeout 60 s) the rule is
    #   inline_load_timeout (20) + connect (5) + read + slack (5) <= lambda timeout (60)  ->  read <= 30;
    # Terraform sets ENDPOINT_READ_TIMEOUT_SECONDS=25 there. ECS has no such limit and keeps 40.
    endpoint_connect_timeout_seconds: float = 5.0
    endpoint_read_timeout_seconds: float = 40.0


def get_settings(app_env: str | None = None) -> Settings:
    load_dotenv_if_available()
    root = project_root()
    resolved_env = _normalise_env(app_env or os.getenv("APP_ENV"))
    env_config = _load_env_config(root, resolved_env)
    env_config_path = root / "config" / "environments" / f"{resolved_env}.json"

    return Settings(
        root=root,
        app_env=resolved_env,
        env_config_path=env_config_path,
        data_mode=_config_value(env_config, "DATA_MODE", "data_mode", "demo"),
        log_level=_config_value(env_config, "LOG_LEVEL", "log_level", "INFO"),
        aws_region=_config_value(env_config, "AWS_REGION", "aws_region", "ap-southeast-2"),
        aws_s3_bucket=_config_value(
            env_config,
            "AWS_S3_BUCKET",
            "aws_s3_bucket",
            f"replace-with-your-{resolved_env}-bucket-name",
        ),
        aws_s3_prefix=_config_value(env_config, "AWS_S3_PREFIX", "aws_s3_prefix", resolved_env),
        athena_database=_config_value(
            env_config,
            "AWS_ATHENA_DATABASE",
            "athena_database",
            f"financial_risk_assistant_{resolved_env.replace('-', '_')}",
        ),
        athena_workgroup=_config_value(
            env_config,
            "AWS_ATHENA_WORKGROUP",
            "athena_workgroup",
            "primary",
        ),
        allow_cloud_deploy=_config_bool(
            env_config,
            "ALLOW_CLOUD_DEPLOY",
            "allow_cloud_deploy",
            resolved_env != "prod",
        ),
        require_manual_approval=_config_bool(
            env_config,
            "REQUIRE_MANUAL_APPROVAL",
            "require_manual_approval",
            resolved_env in {"pre-prod", "prod"},
        ),
        bedrock_model_id=_config_value(
            env_config,
            "BEDROCK_MODEL_ID",
            "bedrock_model_id",
            DEFAULT_BEDROCK_MODEL_ID,
        ),
        bedrock_max_tokens=_config_int(env_config, "BEDROCK_MAX_TOKENS", "bedrock_max_tokens", 600),
        llm_provider=_config_value(env_config, "LLM_PROVIDER", "llm_provider", "none"),
        # Deliberately EMPTY by default: the class runs offline on the template fallback, and a
        # key committed to a repo is a key that has to be rotated.
        deepseek_api_key=_config_value(env_config, "DEEPSEEK_API_KEY", "deepseek_api_key", ""),
        deepseek_base_url=_config_value(
            env_config, "DEEPSEEK_BASE_URL", "deepseek_base_url", "https://api.deepseek.com"
        ),
        deepseek_model=_config_value(env_config, "DEEPSEEK_MODEL", "deepseek_model", "deepseek-flash"),
        deepseek_max_tokens=_config_int(env_config, "DEEPSEEK_MAX_TOKENS", "deepseek_max_tokens", 600),
        deepseek_temperature=_config_float(
            env_config, "DEEPSEEK_TEMPERATURE", "deepseek_temperature", 0.2
        ),
        bedrock_temperature=_config_float(
            env_config,
            "BEDROCK_TEMPERATURE",
            "bedrock_temperature",
            0.2,
        ),
        dataset_url=_config_value(env_config, "DATASET_URL", "dataset_url", DEFAULT_DATASET_URL),
        raw_data_path=_project_path(
            root,
            _config_value(env_config, "RAW_DATA_PATH", "raw_data_path", "data/raw/paysim.csv"),
        ),
        bronze_path=_project_path(
            root,
            _config_value(env_config, "BRONZE_PATH", "bronze_path", "data/bronze/transactions.parquet"),
        ),
        silver_path=_project_path(
            root,
            _config_value(
                env_config,
                "SILVER_PATH",
                "silver_path",
                "data/silver/clean_transactions.parquet",
            ),
        ),
        gold_path=_project_path(
            root,
            _config_value(env_config, "GOLD_PATH", "gold_path", "data/gold"),
        ),
        model_dir=_project_path(
            root,
            _config_value(env_config, "MODEL_DIR", "model_dir", "models"),
        ),
        report_dir=_project_path(
            root,
            _config_value(env_config, "REPORT_DIR", "report_dir", "reports"),
        ),
        vector_index_dir=_project_path(
            root,
            _config_value(env_config, "VECTOR_INDEX_DIR", "vector_index_dir", "vector_index"),
        ),
        rag_docs_dir=_project_path(
            root,
            _config_value(env_config, "RAG_DOCS_DIR", "rag_docs_dir", "rag_docs"),
        ),
        pointer_prefix=_config_value(env_config, "POINTER_PREFIX", "pointer_prefix", "").strip().rstrip("/"),
        pointer_ttl_seconds=_config_int(env_config, "POINTER_TTL_SECONDS", "pointer_ttl_seconds", 60),
        artifact_dir=_config_value(env_config, "ARTIFACT_DIR", "artifact_dir", str(Path(tempfile.gettempdir()) / "finrisk-artifacts")),
        startup_wait_seconds=_config_float(env_config, "STARTUP_WAIT_SECONDS", "startup_wait_seconds", 6.0),
        embedding_model_id=_config_value(env_config, "EMBEDDING_MODEL_ID", "embedding_model_id", "amazon.titan-embed-text-v2:0"),
        pointer_refresh=_pointer_refresh(env_config),
        inline_load_timeout_seconds=_config_float(env_config, "INLINE_LOAD_TIMEOUT_SECONDS", "inline_load_timeout_seconds", 20.0),
        endpoint_connect_timeout_seconds=_config_float(env_config, "ENDPOINT_CONNECT_TIMEOUT_SECONDS", "endpoint_connect_timeout_seconds", 5.0),
        endpoint_read_timeout_seconds=_config_float(env_config, "ENDPOINT_READ_TIMEOUT_SECONDS", "endpoint_read_timeout_seconds", 40.0),
    )


def ensure_project_dirs(settings: Settings | None = None) -> None:
    s = settings or get_settings()
    for path in [
        s.raw_data_path.parent,
        s.bronze_path.parent,
        s.silver_path.parent,
        s.gold_path,
        s.model_dir,
        s.report_dir,
        s.report_dir / "figures",
        s.vector_index_dir,
        s.rag_docs_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

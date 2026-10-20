from __future__ import annotations

import pytest

from financial_risk.config import get_settings


def test_settings_resolve_project_root() -> None:
    settings = get_settings()
    # root must be the directory containing the package -- true both at the repo root and
    # inside a Docker image (/app).
    assert (settings.root / "src" / "financial_risk" / "config.py").exists()
    assert settings.app_env == "dev"
    assert settings.bedrock_model_id
    assert settings.raw_data_path.name == "paysim.csv"


def test_settings_load_environment_config() -> None:
    settings = get_settings("sit")
    assert settings.app_env == "sit"
    assert settings.aws_s3_prefix == "sit"
    assert settings.athena_database == "financial_risk_assistant_sit"


def test_environment_variable_overrides_json(monkeypatch) -> None:
    monkeypatch.setenv("AWS_S3_PREFIX", "custom-prefix")
    settings = get_settings("pre-prod")
    assert settings.app_env == "pre-prod"
    assert settings.aws_s3_prefix == "custom-prefix"


def test_all_four_environments_load_with_one_bedrock_model_id(monkeypatch) -> None:
    from financial_risk.config import DEFAULT_BEDROCK_MODEL_ID

    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    for name in ("dev", "sit", "pre-prod", "prod"):
        settings = get_settings(name)
        assert settings.app_env == name
        assert settings.aws_region == "ap-southeast-2"
        # the JSON files repeat the constant for readers; this keeps them honest
        assert settings.bedrock_model_id == DEFAULT_BEDROCK_MODEL_ID
        assert "mistral" not in settings.bedrock_model_id


def test_app_env_comes_from_the_environment_variable(monkeypatch) -> None:
    import pytest

    monkeypatch.setenv("APP_ENV", "prod")
    assert get_settings().app_env == "prod"
    monkeypatch.setenv("APP_ENV", "preprod")                 # the common spelling is accepted
    assert get_settings().app_env == "pre-prod"
    monkeypatch.setenv("APP_ENV", "staging")
    with pytest.raises(ValueError, match="Unsupported APP_ENV"):
        get_settings()


def test_llm_provider_and_model_come_from_environment_variables(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert get_settings("dev").llm_provider == "none"        # offline by default
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "example.model-v1:0")
    settings = get_settings("sit")
    assert (settings.llm_provider, settings.bedrock_model_id) == ("bedrock", "example.model-v1:0")


def test_pointer_settings_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("POINTER_PREFIX", raising=False)
    monkeypatch.delenv("POINTER_TTL_SECONDS", raising=False)
    settings = get_settings()
    assert settings.pointer_prefix == "" and settings.pointer_ttl_seconds == 60       # defaults: baked artifacts, 60 s

    monkeypatch.setenv("POINTER_PREFIX", " /finrisk/pre-prod/ ")
    monkeypatch.setenv("POINTER_TTL_SECONDS", "15")
    settings = get_settings()
    assert settings.pointer_prefix == "/finrisk/pre-prod" and settings.pointer_ttl_seconds == 15


def test_pointer_refresh_is_inline_on_lambda_and_background_elsewhere(monkeypatch) -> None:
    monkeypatch.delenv("POINTER_REFRESH", raising=False)
    monkeypatch.delenv("INLINE_LOAD_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    settings = get_settings()
    assert settings.pointer_refresh == "background" and settings.inline_load_timeout_seconds == 20.0      # a laptop, ECS

    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "finrisk-dev")                   # AWS sets this on every Lambda
    assert get_settings().pointer_refresh == "inline"
    monkeypatch.setenv("POINTER_REFRESH", "background")                              # an explicit setting wins
    assert get_settings().pointer_refresh == "background"
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME")
    monkeypatch.setenv("POINTER_REFRESH", " Inline ")
    monkeypatch.setenv("INLINE_LOAD_TIMEOUT_SECONDS", "12.5")
    settings = get_settings()
    assert settings.pointer_refresh == "inline" and settings.inline_load_timeout_seconds == 12.5
    monkeypatch.setenv("POINTER_REFRESH", "sometimes")
    with pytest.raises(ValueError, match="POINTER_REFRESH"):
        get_settings()

from __future__ import annotations

from pathlib import Path

import pytest

from runfold_server.config import load_settings
from runfold_server.errors import StartupError


def test_load_settings_normalizes_values(valid_environment: dict[str, str]) -> None:
    settings = load_settings(valid_environment)

    assert settings.host == "127.0.0.1"
    assert settings.port == 8765
    assert settings.data_dir == Path(valid_environment["RUNFOLD_DATA_DIR"])
    assert settings.allowed_origins == (
        "http://localhost:3000",
        "https://app.example.com",
    )
    assert settings.openai_base_url == "http://localhost:11434/v1"
    assert settings.openai_api_key == ""
    assert settings.llm_timeout_seconds == 10.5


def test_missing_required_value_is_rejected(valid_environment: dict[str, str]) -> None:
    del valid_environment["RUNFOLD_EMBEDDING_MODEL"]

    with pytest.raises(StartupError, match="RUNFOLD_EMBEDDING_MODEL is required"):
        load_settings(valid_environment)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RUNFOLD_ALLOWED_ORIGINS", "*"),
        ("RUNFOLD_ALLOWED_ORIGINS", "https://app.example.com/path"),
        ("RUNFOLD_OPENAI_BASE_URL", "https://api.example.com"),
        ("RUNFOLD_OPENAI_BASE_URL", "file:///v1"),
        ("RUNFOLD_EMBEDDING_DIMENSIONS", "0"),
        ("RUNFOLD_LLM_TIMEOUT_SECONDS", "nan"),
        ("RUNFOLD_LLM_MAX_RETRIES", "-1"),
    ],
)
def test_invalid_configuration_is_rejected(
    valid_environment: dict[str, str], name: str, value: str
) -> None:
    valid_environment[name] = value

    with pytest.raises(StartupError) as captured:
        load_settings(valid_environment)

    assert captured.value.code == "invalid_configuration"


def test_overlap_must_be_smaller_than_chunk_size(valid_environment: dict[str, str]) -> None:
    valid_environment["RUNFOLD_CHUNK_OVERLAP"] = valid_environment["RUNFOLD_CHUNK_SIZE"]

    with pytest.raises(StartupError, match="must be smaller"):
        load_settings(valid_environment)


def test_data_directory_must_be_absolute(valid_environment: dict[str, str]) -> None:
    valid_environment["RUNFOLD_DATA_DIR"] = "relative/data"

    with pytest.raises(StartupError, match="absolute path"):
        load_settings(valid_environment)


def test_bootstrap_admin_values_are_an_optional_pair(valid_environment: dict[str, str]) -> None:
    valid_environment["RUNFOLD_BOOTSTRAP_ADMIN_USERNAME"] = "admin"

    with pytest.raises(StartupError, match="must be provided together"):
        load_settings(valid_environment)


def test_secret_values_are_absent_from_settings_repr(valid_environment: dict[str, str]) -> None:
    valid_environment["RUNFOLD_OPENAI_API_KEY"] = "api-key-secret"
    valid_environment["RUNFOLD_BOOTSTRAP_ADMIN_USERNAME"] = "admin"
    valid_environment["RUNFOLD_BOOTSTRAP_ADMIN_PASSWORD"] = "password-secret"

    representation = repr(load_settings(valid_environment))

    assert "api-key-secret" not in representation
    assert "password-secret" not in representation


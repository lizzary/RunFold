from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import ConfigFile

from runfold_server.config import load_settings
from runfold_server.errors import StartupError


def test_load_settings_normalizes_values(valid_config: ConfigFile) -> None:
    settings = valid_config.load()

    assert settings.host == "127.0.0.1"
    assert settings.port == 8765
    assert settings.data_dir == valid_config.data_dir
    assert settings.allowed_origins == (
        "http://localhost:3000",
        "https://app.example.com",
    )
    assert settings.openai_base_url == "http://localhost:11434/v1"
    assert settings.openai_api_key == ""
    assert settings.llm_timeout_seconds == 10.5
    assert settings.agent_model == "test-chat"
    assert settings.agent_budget.context_window_tokens == 6000
    assert settings.agent_budget.provider_concurrency == 2
    assert settings.agent_budget.input_tokens == 1000
    assert settings.agent_budget.output_tokens == 500
    assert settings.agent_budget.thinking_tokens == 300
    assert settings.agent_budget.visible_output_tokens == 200
    assert settings.agent_budget.agent_slots == 4
    assert settings.agent_budget.max_agents_per_run == 3
    assert settings.agent_budget.max_recursion_depth == 3
    assert settings.agent_budget.max_parallel_agents == 2
    assert settings.agent_budget.max_steps == 30


def test_server_values_have_defaults(valid_config: ConfigFile) -> None:
    del valid_config.values["server"]

    settings = valid_config.load()

    assert settings.host == "127.0.0.1"
    assert settings.port == 8000


def test_missing_required_value_is_rejected(valid_config: ConfigFile) -> None:
    del valid_config.values["provider"]["embedding_model"]

    with pytest.raises(StartupError, match=r"provider\.embedding_model is required"):
        valid_config.load()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("cors", "allowed_origins"), ["*"]),
        (("cors", "allowed_origins"), ["https://app.example.com/path"]),
        (("provider", "base_url"), "https://api.example.com"),
        (("provider", "base_url"), "file:///v1"),
        (("provider", "embedding_dimensions"), 0),
        (("provider", "timeout_seconds"), float("nan")),
        (("provider", "max_retries"), -1),
        (("agent", "provider_concurrency"), 0),
        (("agent", "thinking_tokens"), 500),
    ],
)
def test_invalid_configuration_is_rejected(
    valid_config: ConfigFile, path: tuple[str, str], value: Any
) -> None:
    valid_config.values[path[0]][path[1]] = value

    with pytest.raises(StartupError) as captured:
        valid_config.load()

    assert captured.value.code == "invalid_configuration"


def test_unknown_fields_are_rejected(valid_config: ConfigFile) -> None:
    valid_config.values["provider"]["legacy_provider"] = "forbidden"

    with pytest.raises(StartupError, match="unknown field"):
        valid_config.load()


def test_agent_input_and_output_must_fit_context_window(
    valid_config: ConfigFile,
) -> None:
    valid_config.values["agent"]["input_tokens"] = 5600

    with pytest.raises(StartupError, match="must fit"):
        valid_config.load()


def test_old_agent_limit_fields_are_rejected(valid_config: ConfigFile) -> None:
    valid_config.values["agent"]["max_steps"] = 10

    with pytest.raises(StartupError, match="unknown field"):
        valid_config.load()


def test_zero_thinking_budget_is_supported(valid_config: ConfigFile) -> None:
    valid_config.values["agent"]["thinking_tokens"] = 0

    settings = valid_config.load()

    assert settings.agent_budget.thinking_tokens == 0
    assert settings.agent_budget.visible_output_tokens == 500


def test_overlap_must_be_smaller_than_chunk_size(valid_config: ConfigFile) -> None:
    valid_config.values["rag"]["chunk_overlap"] = valid_config.values["rag"][
        "chunk_size"
    ]

    with pytest.raises(StartupError, match="must be smaller"):
        valid_config.load()


def test_data_directory_must_be_absolute(valid_config: ConfigFile) -> None:
    valid_config.values["data"]["directory"] = "relative/data"

    with pytest.raises(StartupError, match="absolute path"):
        valid_config.load()


def test_bootstrap_admin_requires_complete_mapping(valid_config: ConfigFile) -> None:
    valid_config.values["auth"]["bootstrap_admin"] = {"username": "admin"}

    with pytest.raises(
        StartupError, match=r"auth\.bootstrap_admin\.password is required"
    ):
        valid_config.load()


def test_secret_values_are_absent_from_settings_repr(valid_config: ConfigFile) -> None:
    valid_config.values["provider"]["api_key"] = "api-key-secret"
    valid_config.values["auth"]["bootstrap_admin"] = {
        "username": "admin",
        "password": "password-secret",
    }

    representation = repr(valid_config.load())

    assert "api-key-secret" not in representation
    assert "password-secret" not in representation


@pytest.mark.parametrize("content", ["- not-a-mapping\n", "provider: [\n"])
def test_invalid_yaml_document_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(StartupError) as captured:
        load_settings(path)

    assert captured.value.code == "invalid_configuration"


def test_missing_configuration_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(StartupError, match="does not exist"):
        load_settings(tmp_path / "missing.yaml")

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from runfold_server.config import Settings, load_settings


@dataclass
class ConfigFile:
    path: Path
    values: dict[str, Any]

    @property
    def data_dir(self) -> Path:
        return Path(self.values["data"]["directory"])

    def write(self) -> Path:
        self.path.write_text(
            yaml.safe_dump(self.values, sort_keys=False),
            encoding="utf-8",
        )
        return self.path

    def load(self) -> Settings:
        return load_settings(self.write())

    def clone(self, name: str) -> ConfigFile:
        return ConfigFile(self.path.with_name(name), deepcopy(self.values))


@pytest.fixture
def valid_config(tmp_path: Path) -> ConfigFile:
    return ConfigFile(
        path=tmp_path / "config.yaml",
        values={
            "server": {"host": "127.0.0.1", "port": 8765},
            "data": {"directory": str((tmp_path / "data").resolve())},
            "cors": {
                "allowed_origins": [
                    "http://localhost:3000",
                    "https://app.example.com",
                ]
            },
            "provider": {
                "base_url": "http://localhost:11434/v1/",
                "api_key": "",
                "embedding_model": "test-embedding",
                "embedding_dimensions": 8,
                "embed_batch_size": 16,
                "timeout_seconds": 10.5,
                "max_retries": 2,
            },
            "agent": {
                "model": "test-chat",
                "context_window_tokens": 6000,
                "provider_concurrency": 2,
                "input_tokens": 1000,
                "output_tokens": 500,
                "thinking_tokens": 300,
                "compression_threshold": 0.8,
                "thinking_level_options": [
                    "on",
                    "off",
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                    "max",
                ],
                "default_thinking_level": "",
            },
            "rag": {
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "upload_max_bytes": 1048576,
                "extract_max_characters": 100000,
                "pdf_max_pages": 100,
                "docx_max_uncompressed_bytes": 4194304,
            },
            "auth": {"session_ttl_seconds": 3600},
            "limits": {
                "default_max_documents": 100,
                "default_max_storage_bytes": 1073741824,
                "default_monthly_embedding_tokens": 1000000,
                "default_monthly_agent_tokens": 500000,
            },
        },
    )


@pytest.fixture
def admin_config(valid_config: ConfigFile) -> ConfigFile:
    valid_config.values["auth"]["bootstrap_admin"] = {
        "username": "admin",
        "password": "correct horse battery staple",
    }
    return valid_config

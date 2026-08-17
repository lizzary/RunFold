from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def valid_environment(tmp_path: Path) -> dict[str, str]:
    return {
        "RUNFOLD_HOST": "127.0.0.1",
        "RUNFOLD_PORT": "8765",
        "RUNFOLD_DATA_DIR": str((tmp_path / "data").resolve()),
        "RUNFOLD_ALLOWED_ORIGINS": "http://localhost:3000,https://app.example.com",
        "RUNFOLD_OPENAI_BASE_URL": "http://localhost:11434/v1/",
        "RUNFOLD_OPENAI_API_KEY": "",
        "RUNFOLD_EMBEDDING_MODEL": "test-embedding",
        "RUNFOLD_EMBEDDING_DIMENSIONS": "8",
        "RUNFOLD_EMBED_BATCH_SIZE": "16",
        "RUNFOLD_LLM_TIMEOUT_SECONDS": "10.5",
        "RUNFOLD_LLM_MAX_RETRIES": "2",
        "RUNFOLD_CHUNK_SIZE": "1000",
        "RUNFOLD_CHUNK_OVERLAP": "100",
        "RUNFOLD_UPLOAD_MAX_BYTES": "1048576",
        "RUNFOLD_EXTRACT_MAX_CHARACTERS": "100000",
        "RUNFOLD_PDF_MAX_PAGES": "100",
        "RUNFOLD_DOCX_MAX_UNCOMPRESSED_BYTES": "4194304",
        "RUNFOLD_SESSION_TTL_SECONDS": "3600",
        "RUNFOLD_DEFAULT_MAX_DOCUMENTS": "100",
        "RUNFOLD_DEFAULT_MAX_STORAGE_BYTES": "1073741824",
        "RUNFOLD_DEFAULT_MONTHLY_EMBEDDING_TOKENS": "1000000",
    }


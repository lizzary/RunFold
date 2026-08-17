from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from runfold_server.errors import StartupError


@dataclass(frozen=True, slots=True)
class Settings:
    host: str
    port: int
    data_dir: Path
    allowed_origins: tuple[str, ...]
    openai_base_url: str
    openai_api_key: str = field(repr=False)
    embedding_model: str
    embedding_dimensions: int
    embed_batch_size: int
    llm_timeout_seconds: float
    llm_max_retries: int
    chunk_size: int
    chunk_overlap: int
    upload_max_bytes: int
    extract_max_characters: int
    pdf_max_pages: int
    docx_max_uncompressed_bytes: int
    session_ttl_seconds: int
    bootstrap_admin_username: str | None
    bootstrap_admin_password: str | None = field(repr=False)
    default_max_documents: int
    default_max_storage_bytes: int
    default_monthly_embedding_tokens: int


def load_settings(environment: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if environment is None else environment

    host = env.get("RUNFOLD_HOST", "127.0.0.1").strip()
    if not host or any(character.isspace() for character in host):
        raise _invalid("RUNFOLD_HOST", "must be a non-empty host without whitespace")

    port = _integer(env, "RUNFOLD_PORT", default="8000", minimum=1, maximum=65535)
    data_dir_text = _required(env, "RUNFOLD_DATA_DIR")
    data_dir = Path(data_dir_text).expanduser()
    if not data_dir.is_absolute():
        raise _invalid("RUNFOLD_DATA_DIR", "must be an absolute path")
    data_dir = data_dir.resolve(strict=False)
    if data_dir == Path(data_dir.anchor):
        raise _invalid("RUNFOLD_DATA_DIR", "must not be a filesystem root")

    allowed_origins = _origins(_required(env, "RUNFOLD_ALLOWED_ORIGINS"))
    openai_base_url = _base_url(_required(env, "RUNFOLD_OPENAI_BASE_URL"))
    openai_api_key = _required(env, "RUNFOLD_OPENAI_API_KEY", allow_empty=True)
    embedding_model = _required(env, "RUNFOLD_EMBEDDING_MODEL")

    embedding_dimensions = _integer(env, "RUNFOLD_EMBEDDING_DIMENSIONS")
    embed_batch_size = _integer(env, "RUNFOLD_EMBED_BATCH_SIZE")
    llm_timeout_seconds = _number(env, "RUNFOLD_LLM_TIMEOUT_SECONDS")
    llm_max_retries = _integer(env, "RUNFOLD_LLM_MAX_RETRIES", minimum=0)
    chunk_size = _integer(env, "RUNFOLD_CHUNK_SIZE")
    chunk_overlap = _integer(env, "RUNFOLD_CHUNK_OVERLAP", minimum=0)
    if chunk_overlap >= chunk_size:
        raise _invalid("RUNFOLD_CHUNK_OVERLAP", "must be smaller than RUNFOLD_CHUNK_SIZE")

    upload_max_bytes = _integer(env, "RUNFOLD_UPLOAD_MAX_BYTES")
    extract_max_characters = _integer(env, "RUNFOLD_EXTRACT_MAX_CHARACTERS")
    pdf_max_pages = _integer(env, "RUNFOLD_PDF_MAX_PAGES")
    docx_max_uncompressed_bytes = _integer(env, "RUNFOLD_DOCX_MAX_UNCOMPRESSED_BYTES")
    session_ttl_seconds = _integer(env, "RUNFOLD_SESSION_TTL_SECONDS")
    default_max_documents = _integer(env, "RUNFOLD_DEFAULT_MAX_DOCUMENTS")
    default_max_storage_bytes = _integer(env, "RUNFOLD_DEFAULT_MAX_STORAGE_BYTES")
    default_monthly_embedding_tokens = _integer(
        env, "RUNFOLD_DEFAULT_MONTHLY_EMBEDDING_TOKENS"
    )

    admin_username = _optional(env, "RUNFOLD_BOOTSTRAP_ADMIN_USERNAME")
    admin_password = _optional(env, "RUNFOLD_BOOTSTRAP_ADMIN_PASSWORD")
    if (admin_username is None) != (admin_password is None):
        raise StartupError(
            "invalid_configuration",
            "RUNFOLD_BOOTSTRAP_ADMIN_USERNAME and RUNFOLD_BOOTSTRAP_ADMIN_PASSWORD "
            "must be provided together",
        )

    return Settings(
        host=host,
        port=port,
        data_dir=data_dir,
        allowed_origins=allowed_origins,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        embed_batch_size=embed_batch_size,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_max_retries=llm_max_retries,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        upload_max_bytes=upload_max_bytes,
        extract_max_characters=extract_max_characters,
        pdf_max_pages=pdf_max_pages,
        docx_max_uncompressed_bytes=docx_max_uncompressed_bytes,
        session_ttl_seconds=session_ttl_seconds,
        bootstrap_admin_username=admin_username,
        bootstrap_admin_password=admin_password,
        default_max_documents=default_max_documents,
        default_max_storage_bytes=default_max_storage_bytes,
        default_monthly_embedding_tokens=default_monthly_embedding_tokens,
    )


def _required(env: Mapping[str, str], name: str, *, allow_empty: bool = False) -> str:
    if name not in env:
        raise _invalid(name, "is required")
    value = env[name]
    if not allow_empty and not value.strip():
        raise _invalid(name, "must not be empty")
    return value if allow_empty else value.strip()


def _optional(env: Mapping[str, str], name: str) -> str | None:
    if name not in env:
        return None
    value = env[name].strip()
    if not value:
        raise _invalid(name, "must not be empty when provided")
    return value


def _integer(
    env: Mapping[str, str],
    name: str,
    *,
    default: str | None = None,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    raw = env.get(name, default) if default is not None else _required(env, name)
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise _invalid(name, "must be an integer") from error
    if value < minimum or (maximum is not None and value > maximum):
        bounds = (
            f"between {minimum} and {maximum}"
            if maximum is not None
            else f"at least {minimum}"
        )
        raise _invalid(name, f"must be {bounds}")
    return value


def _number(env: Mapping[str, str], name: str) -> float:
    raw = _required(env, name)
    try:
        value = float(raw)
    except ValueError as error:
        raise _invalid(name, "must be a number") from error
    if value <= 0 or value == float("inf") or value != value:
        raise _invalid(name, "must be a finite positive number")
    return value


def _origins(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(","))
    if not values or any(not value for value in values):
        raise _invalid("RUNFOLD_ALLOWED_ORIGINS", "must contain exact comma-separated origins")
    if "*" in values:
        raise _invalid("RUNFOLD_ALLOWED_ORIGINS", "must not contain a wildcard")

    normalized: list[str] = []
    for value in values:
        parts = urlsplit(value)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or parts.path not in {"", "/"}
        ):
            raise _invalid("RUNFOLD_ALLOWED_ORIGINS", "must contain only exact HTTP origins")
        netloc = _canonical_netloc(parts, "RUNFOLD_ALLOWED_ORIGINS")
        origin = urlunsplit((parts.scheme.lower(), netloc, "", "", ""))
        normalized.append(origin)

    if len(set(normalized)) != len(normalized):
        raise _invalid("RUNFOLD_ALLOWED_ORIGINS", "must not contain duplicate origins")
    return tuple(normalized)


def _base_url(raw: str) -> str:
    parts = urlsplit(raw)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise _invalid(
            "RUNFOLD_OPENAI_BASE_URL",
            "must be an absolute HTTP URL without credentials",
        )
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        raise _invalid("RUNFOLD_OPENAI_BASE_URL", "must end with /v1")
    netloc = _canonical_netloc(parts, "RUNFOLD_OPENAI_BASE_URL")
    return urlunsplit((parts.scheme.lower(), netloc, path, "", ""))


def _canonical_netloc(parts, field_name: str) -> str:
    try:
        port = parts.port
    except ValueError as error:
        raise _invalid(field_name, "contains an invalid port") from error
    host = parts.hostname
    if host is None or any(character.isspace() for character in host):
        raise _invalid(field_name, "contains an invalid host")
    try:
        host = host.encode("idna").decode("ascii").lower()
    except UnicodeError as error:
        raise _invalid(field_name, "contains an invalid host") from error
    if ":" in host:
        host = f"[{host}]"
    default_port = (parts.scheme.lower(), port) in {("http", 80), ("https", 443)}
    return f"{host}:{port}" if port is not None and not default_port else host


def _invalid(name: str, reason: str) -> StartupError:
    return StartupError("invalid_configuration", f"{name} {reason}")

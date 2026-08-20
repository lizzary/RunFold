from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

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


def load_settings(path: str | Path) -> Settings:
    document = _load_yaml(Path(path))
    _reject_unknown(
        document,
        {"server", "data", "cors", "provider", "rag", "auth", "limits"},
        "configuration",
    )

    server = _optional_mapping(document, "server", "server")
    data = _required_mapping(document, "data", "data")
    cors = _required_mapping(document, "cors", "cors")
    provider = _required_mapping(document, "provider", "provider")
    rag = _required_mapping(document, "rag", "rag")
    auth = _required_mapping(document, "auth", "auth")
    limits = _required_mapping(document, "limits", "limits")

    _reject_unknown(server, {"host", "port"}, "server")
    _reject_unknown(data, {"directory"}, "data")
    _reject_unknown(cors, {"allowed_origins"}, "cors")
    _reject_unknown(
        provider,
        {
            "base_url",
            "api_key",
            "embedding_model",
            "embedding_dimensions",
            "embed_batch_size",
            "timeout_seconds",
            "max_retries",
        },
        "provider",
    )
    _reject_unknown(
        rag,
        {
            "chunk_size",
            "chunk_overlap",
            "upload_max_bytes",
            "extract_max_characters",
            "pdf_max_pages",
            "docx_max_uncompressed_bytes",
        },
        "rag",
    )
    _reject_unknown(auth, {"session_ttl_seconds", "bootstrap_admin"}, "auth")
    _reject_unknown(
        limits,
        {
            "default_max_documents",
            "default_max_storage_bytes",
            "default_monthly_embedding_tokens",
        },
        "limits",
    )

    host = _optional_string(server, "host", "server.host", default="127.0.0.1")
    if any(character.isspace() for character in host):
        raise _invalid("server.host", "must not contain whitespace")
    port = _optional_integer(server, "port", "server.port", default=8000, maximum=65535)

    data_dir = Path(_required_string(data, "directory", "data.directory"))
    if not data_dir.is_absolute():
        raise _invalid("data.directory", "must be an absolute path")
    data_dir = data_dir.resolve(strict=False)
    if data_dir == Path(data_dir.anchor):
        raise _invalid("data.directory", "must not be a filesystem root")

    allowed_origins = _origins(_required(cors, "allowed_origins", "cors.allowed_origins"))
    openai_base_url = _base_url(
        _required_string(provider, "base_url", "provider.base_url")
    )
    openai_api_key = _required_string(
        provider, "api_key", "provider.api_key", allow_empty=True
    )
    embedding_model = _required_string(
        provider, "embedding_model", "provider.embedding_model"
    )
    embedding_dimensions = _required_integer(
        provider, "embedding_dimensions", "provider.embedding_dimensions"
    )
    embed_batch_size = _required_integer(
        provider, "embed_batch_size", "provider.embed_batch_size"
    )
    llm_timeout_seconds = _required_number(
        provider, "timeout_seconds", "provider.timeout_seconds"
    )
    llm_max_retries = _required_integer(
        provider, "max_retries", "provider.max_retries", minimum=0
    )

    chunk_size = _required_integer(rag, "chunk_size", "rag.chunk_size")
    chunk_overlap = _required_integer(
        rag, "chunk_overlap", "rag.chunk_overlap", minimum=0
    )
    if chunk_overlap >= chunk_size:
        raise _invalid("rag.chunk_overlap", "must be smaller than rag.chunk_size")

    upload_max_bytes = _required_integer(
        rag, "upload_max_bytes", "rag.upload_max_bytes"
    )
    extract_max_characters = _required_integer(
        rag, "extract_max_characters", "rag.extract_max_characters"
    )
    pdf_max_pages = _required_integer(rag, "pdf_max_pages", "rag.pdf_max_pages")
    docx_max_uncompressed_bytes = _required_integer(
        rag,
        "docx_max_uncompressed_bytes",
        "rag.docx_max_uncompressed_bytes",
    )
    session_ttl_seconds = _required_integer(
        auth, "session_ttl_seconds", "auth.session_ttl_seconds"
    )
    admin_username, admin_password = _bootstrap_admin(auth)

    default_max_documents = _required_integer(
        limits, "default_max_documents", "limits.default_max_documents"
    )
    default_max_storage_bytes = _required_integer(
        limits,
        "default_max_storage_bytes",
        "limits.default_max_storage_bytes",
    )
    default_monthly_embedding_tokens = _required_integer(
        limits,
        "default_monthly_embedding_tokens",
        "limits.default_monthly_embedding_tokens",
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


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise _invalid("configuration file", "does not exist") from None
    except OSError as error:
        raise _invalid("configuration file", "cannot be read") from error
    try:
        value = yaml.safe_load(content)
    except yaml.YAMLError as error:
        raise _invalid("configuration file", "is not valid YAML") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise _invalid("configuration", "must be a YAML mapping with string keys")
    return value


def _required(mapping: Mapping[str, Any], name: str, field_name: str) -> Any:
    if name not in mapping:
        raise _invalid(field_name, "is required")
    return mapping[name]


def _required_mapping(
    mapping: Mapping[str, Any], name: str, field_name: str
) -> dict[str, Any]:
    return _mapping(_required(mapping, name, field_name), field_name)


def _optional_mapping(
    mapping: Mapping[str, Any], name: str, field_name: str
) -> dict[str, Any]:
    return {} if name not in mapping else _mapping(mapping[name], field_name)


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise _invalid(field_name, "must be a YAML mapping with string keys")
    return value


def _required_string(
    mapping: Mapping[str, Any],
    name: str,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = _required(mapping, name, field_name)
    if not isinstance(value, str):
        raise _invalid(field_name, "must be a string")
    if not allow_empty and not value.strip():
        raise _invalid(field_name, "must not be empty")
    return value if allow_empty else value.strip()


def _optional_string(
    mapping: Mapping[str, Any], name: str, field_name: str, *, default: str
) -> str:
    if name not in mapping:
        return default
    return _required_string(mapping, name, field_name)


def _required_integer(
    mapping: Mapping[str, Any],
    name: str,
    field_name: str,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    return _integer(
        _required(mapping, name, field_name),
        field_name,
        minimum=minimum,
        maximum=maximum,
    )


def _optional_integer(
    mapping: Mapping[str, Any],
    name: str,
    field_name: str,
    *,
    default: int,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if name not in mapping:
        return default
    return _integer(mapping[name], field_name, minimum=minimum, maximum=maximum)


def _integer(
    value: Any,
    field_name: str,
    *,
    minimum: int,
    maximum: int | None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid(field_name, "must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = (
            f"between {minimum} and {maximum}"
            if maximum is not None
            else f"at least {minimum}"
        )
        raise _invalid(field_name, f"must be {bounds}")
    return value


def _required_number(
    mapping: Mapping[str, Any], name: str, field_name: str
) -> float:
    value = _required(mapping, name, field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(field_name, "must be a number")
    converted = float(value)
    if converted <= 0 or not math.isfinite(converted):
        raise _invalid(field_name, "must be a finite positive number")
    return converted


def _bootstrap_admin(auth: Mapping[str, Any]) -> tuple[str | None, str | None]:
    value = auth.get("bootstrap_admin")
    if value is None:
        return None, None
    admin = _mapping(value, "auth.bootstrap_admin")
    _reject_unknown(admin, {"username", "password"}, "auth.bootstrap_admin")
    return (
        _required_string(admin, "username", "auth.bootstrap_admin.username"),
        _required_string(admin, "password", "auth.bootstrap_admin.password"),
    )


def _origins(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise _invalid("cors.allowed_origins", "must be a non-empty YAML list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise _invalid("cors.allowed_origins", "must contain non-empty strings")
    values = tuple(item.strip() for item in value)
    if "*" in values:
        raise _invalid("cors.allowed_origins", "must not contain a wildcard")

    normalized: list[str] = []
    for origin in values:
        parts = urlsplit(origin)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or parts.path not in {"", "/"}
        ):
            raise _invalid("cors.allowed_origins", "must contain only exact HTTP origins")
        netloc = _canonical_netloc(parts, "cors.allowed_origins")
        normalized.append(urlunsplit((parts.scheme.lower(), netloc, "", "", "")))

    if len(set(normalized)) != len(normalized):
        raise _invalid("cors.allowed_origins", "must not contain duplicate origins")
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
        raise _invalid("provider.base_url", "must be an absolute HTTP URL without credentials")
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        raise _invalid("provider.base_url", "must end with /v1")
    netloc = _canonical_netloc(parts, "provider.base_url")
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


def _reject_unknown(
    mapping: Mapping[str, Any], allowed: set[str], field_name: str
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise _invalid(field_name, f"contains unknown field: {unknown[0]}")


def _invalid(name: str, reason: str) -> StartupError:
    return StartupError("invalid_configuration", f"{name} {reason}")

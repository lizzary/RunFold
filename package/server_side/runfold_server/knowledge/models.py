from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

READ = 10
EDIT = 20
MANAGE = 30


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    created_by_user_id: str
    original_filename: str
    media_type: str
    storage_key: str
    byte_size: int
    content_hash: str
    extracted_characters: int
    chunk_count: int
    index_state: str
    index_error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class AclGrant:
    user_id: str | None
    role_id: str | None
    access_level: int


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    ordinal: int
    text: str


@dataclass(frozen=True, slots=True)
class StagedDocument:
    operation_id: str
    directory: Path
    source: Path
    extracted: Path
    original_filename: str
    media_type: str
    byte_size: int
    content_hash: str
    text: str


@dataclass(frozen=True, slots=True)
class DocumentContent:
    data: bytes
    original_filename: str
    media_type: str


@dataclass(frozen=True, slots=True)
class DocumentText:
    document_id: str
    text: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class SearchableDocument:
    id: str
    title: str
    content_hash: str
    chunk_count: int


@dataclass(frozen=True, slots=True)
class IndexSearchHit:
    document_id: str
    chunk_id: str
    ordinal: int
    content_hash: str
    text: str
    distance: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    document_id: str
    title: str
    ordinal: int
    content_hash: str
    text: str
    distance: float

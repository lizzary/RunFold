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


@dataclass(frozen=True, slots=True)
class DocumentSection:
    section_id: str
    title: str
    level: int
    start_character: int
    end_character: int


@dataclass(frozen=True, slots=True)
class DocumentManifest:
    document_id: str
    title: str
    original_filename: str
    media_type: str
    content_hash: str
    extracted_characters: int
    chunk_count: int
    section_count: int
    section_offset: int
    next_section_offset: int
    sections_eof: bool
    sections: tuple[DocumentSection, ...]


@dataclass(frozen=True, slots=True)
class DocumentTextSlice:
    document_id: str
    content_hash: str
    offset_characters: int
    next_offset_characters: int
    eof: bool
    text: str


@dataclass(frozen=True, slots=True)
class DocumentChunkContext:
    document_id: str
    title: str
    content_hash: str
    requested_ordinal: int
    start_ordinal: int
    end_ordinal: int
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True, slots=True)
class DocumentTextMatch:
    start_character: int
    end_character: int
    line_number: int
    context: str


@dataclass(frozen=True, slots=True)
class DocumentTextSearch:
    document_id: str
    content_hash: str
    query: str
    case_sensitive: bool
    total_matches: int
    truncated: bool
    matches: tuple[DocumentTextMatch, ...]


@dataclass(frozen=True, slots=True)
class DocumentSectionSlice:
    document_id: str
    content_hash: str
    section: DocumentSection
    offset_characters: int
    next_offset_characters: int
    eof: bool
    text: str

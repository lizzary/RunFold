from __future__ import annotations

import pytest

from runfold_server.access_control.capabilities import RAG_DOCUMENT_READ, RAG_SEARCH
from runfold_server.errors import ApiError
from runfold_server.knowledge.chunker import chunk_text
from runfold_server.knowledge.document_reader import document_sections, search_literal
from runfold_server.knowledge.models import Document
from runfold_server.knowledge.service import KnowledgeService


class _Objects:
    def __init__(self, text: str) -> None:
        self.text = text

    def read_extracted(self, document_id: str) -> str:
        assert document_id == "document-1"
        return self.text


def test_section_detection_and_literal_search_are_deterministic() -> None:
    text = "Opening terms\n# Definitions\nAlpha alpha\n第一条 Payment\nPay monthly\n"

    sections = document_sections(text, fallback_title="Contract")
    total, matches = search_literal(
        text,
        query="ALPHA",
        case_sensitive=False,
        maximum_matches=1,
        context_characters=5,
    )

    assert [(item.section_id, item.title, item.level) for item in sections] == [
        ("section-0000", "Preamble", 0),
        ("section-0001", "Definitions", 1),
        ("section-0002", "第一条 Payment", 3),
    ]
    assert sections[0].end_character == sections[1].start_character
    assert sections[-1].end_character == len(text)
    assert total == 2
    assert len(matches) == 1
    assert matches[0].line_number == 3
    assert "Alpha" in matches[0].context


def test_authorized_document_read_suite_covers_full_text_sections_and_context() -> None:
    text = "Preamble\n# Scope\nAlpha obligations.\n# Payment\nPay monthly.\n"
    service, capability_checks = _service(text)

    manifest = service.document_manifest(
        object(),  # type: ignore[arg-type]
        "document-1",
        section_offset=0,
        section_limit=2,
    )
    first = service.read_document_text(
        object(),  # type: ignore[arg-type]
        "document-1",
        expected_content_hash="a" * 64,
        offset_characters=0,
        max_characters=10,
    )
    section = service.read_document_section(
        object(),  # type: ignore[arg-type]
        "document-1",
        expected_content_hash="a" * 64,
        section_id="section-0001",
        offset_characters=0,
        max_characters=1_000,
    )
    chunks = chunk_text(text, size=20, overlap=0)
    context = service.read_chunk_context(
        object(),  # type: ignore[arg-type]
        "document-1",
        expected_content_hash="a" * 64,
        ordinal=1,
        before=1,
        after=1,
    )
    searched = service.search_document_text(
        object(),  # type: ignore[arg-type]
        "document-1",
        expected_content_hash="a" * 64,
        query="monthly",
        case_sensitive=False,
        max_matches=10,
        context_characters=20,
    )

    assert manifest.section_count == 3
    assert manifest.next_section_offset == 2
    assert not manifest.sections_eof
    assert first.text == text[:10]
    assert first.next_offset_characters == 10
    assert not first.eof
    assert section.section.title == "Scope"
    assert section.text.startswith("# Scope")
    assert section.eof
    assert context.start_ordinal == 0
    assert context.end_ordinal == min(2, len(chunks) - 1)
    assert [item.ordinal for item in context.chunks] == list(
        range(context.start_ordinal, context.end_ordinal + 1)
    )
    assert searched.total_matches == 1
    assert "Pay monthly" in searched.matches[0].context
    assert frozenset({RAG_DOCUMENT_READ, RAG_SEARCH}) in capability_checks
    assert all(RAG_DOCUMENT_READ in item for item in capability_checks)


def test_document_read_rejects_changed_or_inconsistent_text() -> None:
    service, _ = _service("short text")

    with pytest.raises(ApiError) as changed:
        service.read_document_text(
            object(),  # type: ignore[arg-type]
            "document-1",
            expected_content_hash="b" * 64,
            offset_characters=0,
            max_characters=10,
        )
    assert changed.value.code == "document_changed"

    service._objects.text = "changed length"  # type: ignore[attr-defined]
    with pytest.raises(ApiError) as inconsistent:
        service.document_manifest(
            object(),  # type: ignore[arg-type]
            "document-1",
            section_offset=0,
            section_limit=10,
        )
    assert inconsistent.value.code == "document_text_inconsistent"


def _service(text: str) -> tuple[KnowledgeService, list[frozenset[str]]]:
    chunks = chunk_text(text, size=20, overlap=0)
    document = Document(
        id="document-1",
        title="Contract",
        created_by_user_id="user-1",
        original_filename="contract.md",
        media_type="text/markdown",
        storage_key="document-1/source",
        byte_size=len(text.encode()),
        content_hash="a" * 64,
        extracted_characters=len(text),
        chunk_count=len(chunks),
        index_state="ready",
        index_error=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    checks: list[frozenset[str]] = []
    service = object.__new__(KnowledgeService)
    service._objects = _Objects(text)  # type: ignore[attr-defined]
    service._chunk_size = 20  # type: ignore[attr-defined]
    service._chunk_overlap = 0  # type: ignore[attr-defined]

    def authorize(
        actor: object,
        document_id: str,
        *,
        capabilities: frozenset[str],
    ) -> Document:
        del actor
        assert document_id == "document-1"
        checks.append(capabilities)
        return document

    service._read_authorized_with_capabilities = authorize  # type: ignore[method-assign]
    return service, checks

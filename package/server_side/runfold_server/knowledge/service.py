from __future__ import annotations

import sqlite3
import unicodedata
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from runfold_server.access_control.audit import AuditRepository
from runfold_server.access_control.authorization import AuthorizationService
from runfold_server.access_control.capabilities import (
    RAG_DOCUMENT_ACL_MANAGE,
    RAG_DOCUMENT_DELETE,
    RAG_DOCUMENT_READ,
    RAG_DOCUMENT_UPDATE,
    RAG_DOCUMENT_UPLOAD,
    RAG_SEARCH,
)
from runfold_server.access_control.models import CurrentAccess
from runfold_server.errors import ApiError
from runfold_server.identity.models import VerifiedIdentity
from runfold_server.identity.service import IdentityService
from runfold_server.knowledge.access_policy import KnowledgeAccessPolicy
from runfold_server.knowledge.chunker import chunk_text
from runfold_server.knowledge.document_reader import (
    document_sections,
    search_literal,
    slice_text,
)
from runfold_server.knowledge.lance_index import LanceIndex, UnsafeIndexResult
from runfold_server.knowledge.models import (
    EDIT,
    MANAGE,
    READ,
    AclGrant,
    Document,
    DocumentChunkContext,
    DocumentContent,
    DocumentManifest,
    DocumentSectionSlice,
    DocumentText,
    DocumentTextSearch,
    DocumentTextSlice,
    SearchResult,
    StagedDocument,
)
from runfold_server.knowledge.object_store import AsyncReadable, ObjectStore
from runfold_server.knowledge.repository import KnowledgeRepository
from runfold_server.llm.openai_embeddings import OpenAIEmbeddingsClient
from runfold_server.storage.sqlite import connect
from runfold_server.usage.service import UsageService

_VISIBLE_STATES = ("ready", "failed")
_READ_CAPABILITIES = frozenset({RAG_DOCUMENT_READ})
_SEARCH_CAPABILITIES = frozenset({RAG_SEARCH, RAG_DOCUMENT_READ})
_MAX_SEARCH_QUERY_CHARACTERS = 10_000
_MAX_SEARCH_SCOPE = 1_000
_MAX_TOP_K = 100
_MAX_DOCUMENT_READ_CHARACTERS = 16_000
_MAX_MANIFEST_SECTIONS = 200
_MAX_CHUNK_CONTEXT = 5
_MAX_DOCUMENT_TEXT_QUERY_CHARACTERS = 1_000
_MAX_DOCUMENT_TEXT_MATCHES = 100
_MAX_DOCUMENT_TEXT_CONTEXT_CHARACTERS = 500


class KnowledgeService:
    def __init__(
        self,
        *,
        database_path: Path,
        identity: IdentityService,
        authorization: AuthorizationService,
        repository: KnowledgeRepository,
        access_policy: KnowledgeAccessPolicy,
        audit: AuditRepository,
        objects: ObjectStore,
        index: LanceIndex,
        embeddings: OpenAIEmbeddingsClient,
        usage: UsageService,
        chunk_size: int,
        chunk_overlap: int,
        embed_batch_size: int,
    ) -> None:
        self._database_path = database_path
        self._identity = identity
        self._authorization = authorization
        self._repository = repository
        self._access_policy = access_policy
        self._audit = audit
        self._objects = objects
        self._index = index
        self._embeddings = embeddings
        self._usage = usage
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embed_batch_size = embed_batch_size

    async def upload(
        self,
        actor: VerifiedIdentity,
        *,
        title: str,
        original_filename: str,
        stream: AsyncReadable,
    ) -> Document:
        normalized_title = _title(title)
        self._require_capability(actor, RAG_DOCUMENT_UPLOAD)
        staged = await self._objects.stage_upload(stream, original_filename)
        document_id = str(uuid.uuid4())
        chunks = chunk_text(
            staged.text, size=self._chunk_size, overlap=self._chunk_overlap
        )
        linearized = False
        try:
            with connect(self._database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                current, _ = self._current_access(
                    connection, actor, frozenset({RAG_DOCUMENT_UPLOAD})
                )
                self._require_create_capacity(
                    connection, current, staged.byte_size, document_id
                )
            vectors = await self._embed_chunks(
                actor,
                chunks=chunks,
                capability=RAG_DOCUMENT_UPLOAD,
                document_id=None,
                observed_hash=None,
                minimum_level=None,
                replace_sizes=None,
            )
            now = _now()
            with connect(self._database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                current, _ = self._current_access(
                    connection, actor, frozenset({RAG_DOCUMENT_UPLOAD})
                )
                self._require_create_capacity(
                    connection, current, staged.byte_size, document_id
                )
                self._repository.insert_document(
                    connection,
                    document_id=document_id,
                    title=normalized_title,
                    creator_user_id=current.user_id,
                    original_filename=staged.original_filename,
                    media_type=staged.media_type,
                    byte_size=staged.byte_size,
                    content_hash=staged.content_hash,
                    extracted_characters=len(staged.text),
                    chunk_count=len(chunks),
                    now=now,
                )
                self._repository.insert_creator_acl(
                    connection,
                    document_id=document_id,
                    user_id=current.user_id,
                    access_level=MANAGE,
                    now=now,
                )
                self._usage.record_upload(connection, current.user_id, now)
                self._audit.record(
                    connection,
                    actor_user_id=current.user_id,
                    action="rag.document.upload",
                    decision="allowed",
                    resource_type="document",
                    resource_id=document_id,
                    reason=None,
                    request_id=current.context.request_id,
                    details={"byte_size": staged.byte_size, "segments": len(chunks)},
                    now=now,
                )
            linearized = True
            await self._finish_indexing(staged, document_id, chunks, vectors, replace_source=True)
        except Exception as error:
            if linearized:
                await self._settle_failed(document_id, "indexing_failed")
                if not isinstance(error, ApiError):
                    raise ApiError(
                        503, "document_indexing_failed", "Document indexing failed"
                    ) from error
            raise
        finally:
            self._objects.cleanup_stage(staged.directory)
        return self._return_upload(actor, document_id)

    def list_documents(
        self, actor: VerifiedIdentity, *, limit: int, offset: int
    ) -> tuple[tuple[Document, ...], int]:
        now = _now()
        with connect(self._database_path) as connection:
            current, access = self._begin_audited_access(
                connection, actor, frozenset({RAG_DOCUMENT_READ})
            )
            self._access_policy.record_bypass(
                connection,
                context=current.context,
                access=access,
                document_id=None,
                minimum_level=READ,
                now=now,
            )
            return (
                self._repository.list_authorized(
                    connection,
                    access=access,
                    minimum_level=READ,
                    states=_VISIBLE_STATES,
                    limit=limit,
                    offset=offset,
                ),
                self._repository.count_authorized(
                    connection,
                    access=access,
                    minimum_level=READ,
                    states=_VISIBLE_STATES,
                ),
            )

    async def search(
        self,
        actor: VerifiedIdentity,
        *,
        query: str,
        top_k: int,
        document_ids: tuple[str, ...] | None,
    ) -> tuple[SearchResult, ...]:
        authorized_count = 0
        bypass_used = False
        try:
            with connect(self._database_path) as connection:
                connection.execute("BEGIN")
                current, access = self._current_access(
                    connection, actor, _SEARCH_CAPABILITIES
                )
                normalized_query = _search_query(query)
                _validate_search_limit(top_k)
                _validate_search_scope(document_ids)
                bypass_used = access.bypass
                authorized = self._repository.searchable_authorized(
                    connection, access=access
                )
                authorized_count = len(authorized)
                authorized_by_id = {document.id: document for document in authorized}
                if document_ids is None:
                    selected_ids = tuple(authorized_by_id)
                else:
                    if any(document_id not in authorized_by_id for document_id in document_ids):
                        raise ApiError(404, "document_not_found", "Document not found")
                    selected_ids = tuple(dict.fromkeys(document_ids))
                selected_by_id = {
                    document_id: authorized_by_id[document_id]
                    for document_id in selected_ids
                }
                if not selected_ids:
                    now = _now()
                    self._usage.record_search(connection, current.user_id, now)
                    self._record_search_audit(
                        connection,
                        actor=current,
                        decision="allowed",
                        reason=None,
                        authorized_count=authorized_count,
                        reference_ids=(),
                        result_count=0,
                        bypass_used=bypass_used,
                        now=now,
                    )
                    return ()
                self._usage.require_embedding_capacity(
                    connection, user_id=current.user_id
                )

            embedded = await self._embeddings.embed((normalized_query,))
            self._usage.record_embedding_tokens(actor.user_id, embedded.total_tokens)
            hits = self._index.search(
                embedded.vectors[0], document_ids=selected_ids, top_k=top_k
            )

            with connect(self._database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                current, access = self._current_access(
                    connection, actor, _SEARCH_CAPABILITIES
                )
                current_documents = self._repository.searchable_authorized(
                    connection,
                    access=access,
                    document_ids=selected_ids,
                )
                current_by_id = {
                    document.id: document for document in current_documents
                }
                suspect_ids = tuple(
                    document_id
                    for document_id, expected in selected_by_id.items()
                    if document_id not in current_by_id
                    or current_by_id[document_id].content_hash != expected.content_hash
                )
                if suspect_ids:
                    raise UnsafeIndexResult(suspect_ids)

                results: list[SearchResult] = []
                for hit in hits:
                    expected = selected_by_id.get(hit.document_id)
                    document = current_by_id.get(hit.document_id)
                    if (
                        expected is None
                        or document is None
                        or hit.content_hash != expected.content_hash
                        or hit.content_hash != document.content_hash
                        or hit.ordinal >= document.chunk_count
                    ):
                        raise UnsafeIndexResult((hit.document_id,))
                    results.append(
                        SearchResult(
                            document_id=document.id,
                            title=document.title,
                            ordinal=hit.ordinal,
                            content_hash=document.content_hash,
                            text=hit.text,
                            distance=hit.distance,
                        )
                    )

                now = _now()
                references = tuple(sorted({result.document_id for result in results}))
                self._usage.record_search(connection, current.user_id, now)
                self._record_search_audit(
                    connection,
                    actor=current,
                    decision="allowed",
                    reason=None,
                    authorized_count=authorized_count,
                    reference_ids=references,
                    result_count=len(results),
                    bypass_used=bypass_used,
                    now=now,
                )
                return tuple(results)
        except UnsafeIndexResult as error:
            self._record_search_denied(
                actor,
                reason="unsafe_index_result",
                authorized_count=authorized_count,
                bypass_used=bypass_used,
                suspect_ids=error.document_ids,
            )
            raise ApiError(
                503,
                "unsafe_index_result",
                "Search results failed security validation",
            ) from None
        except ApiError as error:
            self._record_search_denied(
                actor,
                reason=error.code,
                authorized_count=authorized_count,
                bypass_used=bypass_used,
                quota=error.details.get("quota"),
            )
            raise
        except Exception as error:
            self._record_search_denied(
                actor,
                reason="rag_search_failed",
                authorized_count=authorized_count,
                bypass_used=bypass_used,
            )
            raise ApiError(503, "rag_search_failed", "RAG search failed") from error

    def get_document(self, actor: VerifiedIdentity, document_id: str) -> Document:
        return self._read_authorized(actor, document_id, states=_VISIBLE_STATES)

    def download(self, actor: VerifiedIdentity, document_id: str) -> DocumentContent:
        observed = self._read_authorized(actor, document_id, states=_VISIBLE_STATES)
        data = self._objects.read_source(document_id)
        current = self._read_authorized(actor, document_id, states=_VISIBLE_STATES)
        if current.content_hash != observed.content_hash:
            raise ApiError(409, "document_changed", "Document changed; retry the request")
        return DocumentContent(
            data=data,
            original_filename=current.original_filename,
            media_type=current.media_type,
        )

    def extracted_text(self, actor: VerifiedIdentity, document_id: str) -> DocumentText:
        observed = self._read_authorized(actor, document_id, states=("ready",))
        text = self._objects.read_extracted(document_id)
        current = self._read_authorized(actor, document_id, states=("ready",))
        if current.content_hash != observed.content_hash:
            raise ApiError(409, "document_changed", "Document changed; retry the request")
        return DocumentText(document_id=document_id, text=text, content_hash=current.content_hash)

    def document_manifest(
        self,
        actor: VerifiedIdentity,
        document_id: str,
        *,
        section_offset: int,
        section_limit: int,
    ) -> DocumentManifest:
        _validate_manifest_page(section_offset, section_limit)
        document, text = self._authorized_extracted_snapshot(actor, document_id)
        sections = document_sections(text, fallback_title=document.title)
        if section_offset > len(sections):
            raise ApiError(
                422,
                "invalid_section_page",
                "Section offset is past the document outline",
            )
        end = min(len(sections), section_offset + section_limit)
        return DocumentManifest(
            document_id=document.id,
            title=document.title,
            original_filename=document.original_filename,
            media_type=document.media_type,
            content_hash=document.content_hash,
            extracted_characters=document.extracted_characters,
            chunk_count=document.chunk_count,
            section_count=len(sections),
            section_offset=section_offset,
            next_section_offset=end,
            sections_eof=end == len(sections),
            sections=sections[section_offset:end],
        )

    def read_document_text(
        self,
        actor: VerifiedIdentity,
        document_id: str,
        *,
        expected_content_hash: str,
        offset_characters: int,
        max_characters: int,
    ) -> DocumentTextSlice:
        _validate_document_read(offset_characters, max_characters)
        document, text = self._authorized_extracted_snapshot(
            actor,
            document_id,
            expected_content_hash=expected_content_hash,
        )
        try:
            value, next_offset, eof = slice_text(
                text,
                offset=offset_characters,
                maximum=max_characters,
            )
        except ValueError as error:
            raise ApiError(
                422,
                "invalid_document_text_range",
                "Document text range is invalid",
            ) from error
        return DocumentTextSlice(
            document_id=document.id,
            content_hash=document.content_hash,
            offset_characters=offset_characters,
            next_offset_characters=next_offset,
            eof=eof,
            text=value,
        )

    def read_chunk_context(
        self,
        actor: VerifiedIdentity,
        document_id: str,
        *,
        expected_content_hash: str,
        ordinal: int,
        before: int,
        after: int,
    ) -> DocumentChunkContext:
        _validate_chunk_context(ordinal, before, after)
        document, text = self._authorized_extracted_snapshot(
            actor,
            document_id,
            expected_content_hash=expected_content_hash,
        )
        chunks = chunk_text(text, size=self._chunk_size, overlap=self._chunk_overlap)
        if len(chunks) != document.chunk_count:
            raise ApiError(
                503,
                "document_text_inconsistent",
                "Document text failed consistency validation",
            )
        if ordinal >= len(chunks):
            raise ApiError(422, "invalid_chunk_ordinal", "Chunk ordinal is invalid")
        start = max(0, ordinal - before)
        end = min(len(chunks), ordinal + after + 1)
        return DocumentChunkContext(
            document_id=document.id,
            title=document.title,
            content_hash=document.content_hash,
            requested_ordinal=ordinal,
            start_ordinal=start,
            end_ordinal=end - 1,
            chunks=chunks[start:end],
        )

    def search_document_text(
        self,
        actor: VerifiedIdentity,
        document_id: str,
        *,
        expected_content_hash: str,
        query: str,
        case_sensitive: bool,
        max_matches: int,
        context_characters: int,
    ) -> DocumentTextSearch:
        normalized_query = _document_text_query(query)
        _validate_document_text_search(max_matches, context_characters)
        document, text = self._authorized_extracted_snapshot(
            actor,
            document_id,
            expected_content_hash=expected_content_hash,
            capabilities=_SEARCH_CAPABILITIES,
        )
        total, matches = search_literal(
            text,
            query=normalized_query,
            case_sensitive=case_sensitive,
            maximum_matches=max_matches,
            context_characters=context_characters,
        )
        return DocumentTextSearch(
            document_id=document.id,
            content_hash=document.content_hash,
            query=normalized_query,
            case_sensitive=case_sensitive,
            total_matches=total,
            truncated=total > len(matches),
            matches=matches,
        )

    def read_document_section(
        self,
        actor: VerifiedIdentity,
        document_id: str,
        *,
        expected_content_hash: str,
        section_id: str,
        offset_characters: int,
        max_characters: int,
    ) -> DocumentSectionSlice:
        _validate_document_read(offset_characters, max_characters)
        document, text = self._authorized_extracted_snapshot(
            actor,
            document_id,
            expected_content_hash=expected_content_hash,
        )
        sections = document_sections(text, fallback_title=document.title)
        section = next((item for item in sections if item.section_id == section_id), None)
        if section is None:
            raise ApiError(404, "document_section_not_found", "Document section not found")
        section_text = text[section.start_character : section.end_character]
        try:
            value, next_offset, eof = slice_text(
                section_text,
                offset=offset_characters,
                maximum=max_characters,
            )
        except ValueError as error:
            raise ApiError(
                422,
                "invalid_document_text_range",
                "Document text range is invalid",
            ) from error
        return DocumentSectionSlice(
            document_id=document.id,
            content_hash=document.content_hash,
            section=section,
            offset_characters=offset_characters,
            next_offset_characters=next_offset,
            eof=eof,
            text=value,
        )

    def update_metadata(
        self, actor: VerifiedIdentity, document_id: str, *, title: str
    ) -> Document:
        normalized = _title(title)
        now = _now()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current, access = self._current_access(
                connection, actor, frozenset({RAG_DOCUMENT_UPDATE})
            )
            document = self._access_policy.require_document(
                connection,
                context=current.context,
                access=access,
                document_id=document_id,
                minimum_level=EDIT,
                states=_VISIBLE_STATES,
                now=now,
            )
            self._repository.update_title(connection, document_id, normalized, now)
            self._audit.record(
                connection,
                actor_user_id=current.user_id,
                action="rag.document.metadata.update",
                decision="allowed",
                resource_type="document",
                resource_id=document_id,
                reason=None,
                request_id=current.context.request_id,
                details={"title_changed": document.title != normalized},
                now=now,
            )
            updated = self._repository.get(connection, document_id)
            assert updated is not None
            return updated

    async def replace_upload(
        self,
        actor: VerifiedIdentity,
        document_id: str,
        *,
        original_filename: str,
        stream: AsyncReadable,
    ) -> Document:
        observed = self._operation_document(
            actor, document_id, RAG_DOCUMENT_UPDATE, EDIT
        )
        staged = await self._objects.stage_upload(stream, original_filename)
        return await self._replace(actor, observed, staged, action="rag.document.content.replace")

    async def replace_text(
        self, actor: VerifiedIdentity, document_id: str, *, text: str
    ) -> Document:
        observed = self._operation_document(
            actor, document_id, RAG_DOCUMENT_UPDATE, EDIT
        )
        if observed.media_type not in {"text/plain", "text/markdown"}:
            raise ApiError(
                415, "text_replace_not_supported", "Text replacement requires txt or md"
            )
        staged = self._objects.stage_text(text, observed.original_filename)
        return await self._replace(actor, observed, staged, action="rag.document.text.replace")

    async def reindex(self, actor: VerifiedIdentity, document_id: str) -> Document:
        observed = self._operation_document(
            actor, document_id, RAG_DOCUMENT_UPDATE, EDIT
        )
        staged = self._objects.stage_existing(document_id, observed.original_filename)
        return await self._replace(
            actor,
            observed,
            staged,
            action="rag.document.reindex",
            replace_source=False,
        )

    async def delete(self, actor: VerifiedIdentity, document_id: str) -> None:
        observed = self._operation_document(
            actor, document_id, RAG_DOCUMENT_DELETE, MANAGE
        )
        now = _now()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current, access = self._current_access(
                connection, actor, frozenset({RAG_DOCUMENT_DELETE})
            )
            self._access_policy.require_document(
                connection,
                context=current.context,
                access=access,
                document_id=document_id,
                minimum_level=MANAGE,
                states=_VISIBLE_STATES,
                now=now,
            )
            if not self._repository.transition_to_deleting(
                connection,
                document_id=document_id,
                observed_hash=observed.content_hash,
                now=now,
            ):
                raise ApiError(409, "document_changed", "Document changed; retry the request")
            self._audit.record(
                connection,
                actor_user_id=current.user_id,
                action="rag.document.delete",
                decision="allowed",
                resource_type="document",
                resource_id=document_id,
                reason=None,
                request_id=current.context.request_id,
                details={},
                now=now,
            )
        try:
            await self._index.delete_document(document_id)
            self._objects.delete_document(document_id)
            with connect(self._database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._repository.delete_record(connection, document_id)
        except Exception as error:
            raise ApiError(
                503, "document_delete_incomplete", "Document deletion is incomplete"
            ) from error
        self._require_capability(actor, RAG_DOCUMENT_DELETE)

    def get_acl(
        self, actor: VerifiedIdentity, document_id: str
    ) -> tuple[AclGrant, ...]:
        now = _now()
        with connect(self._database_path) as connection:
            current, access = self._begin_audited_access(
                connection, actor, frozenset({RAG_DOCUMENT_ACL_MANAGE})
            )
            self._access_policy.require_document(
                connection,
                context=current.context,
                access=access,
                document_id=document_id,
                minimum_level=MANAGE,
                states=_VISIBLE_STATES,
                now=now,
            )
            return self._repository.list_acl(connection, document_id)

    def replace_acl(
        self,
        actor: VerifiedIdentity,
        document_id: str,
        grants: tuple[AclGrant, ...],
    ) -> tuple[AclGrant, ...]:
        _validate_grants(grants)
        now = _now()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current, access = self._current_access(
                connection, actor, frozenset({RAG_DOCUMENT_ACL_MANAGE})
            )
            self._access_policy.require_document(
                connection,
                context=current.context,
                access=access,
                document_id=document_id,
                minimum_level=MANAGE,
                states=_VISIBLE_STATES,
                now=now,
            )
            if not self._repository.acl_subjects_exist(connection, grants):
                raise ApiError(422, "unknown_acl_subject", "ACL contains an unknown subject")
            before = self._repository.list_acl(connection, document_id)
            self._repository.replace_acl(
                connection,
                document_id=document_id,
                grants=grants,
                granted_by_user_id=current.user_id,
                now=now,
            )
            self._audit.record(
                connection,
                actor_user_id=current.user_id,
                action="rag.document.acl.replace",
                decision="allowed",
                resource_type="document",
                resource_id=document_id,
                reason=None,
                request_id=current.context.request_id,
                details={"before": _grant_details(before), "after": _grant_details(grants)},
                now=now,
            )
        return self.get_acl(actor, document_id)

    async def _replace(
        self,
        actor: VerifiedIdentity,
        observed: Document,
        staged: StagedDocument,
        *,
        action: str,
        replace_source: bool = True,
    ) -> Document:
        chunks = chunk_text(
            staged.text, size=self._chunk_size, overlap=self._chunk_overlap
        )
        linearized = False
        try:
            vectors = await self._embed_chunks(
                actor,
                chunks=chunks,
                capability=RAG_DOCUMENT_UPDATE,
                document_id=observed.id,
                observed_hash=observed.content_hash,
                minimum_level=EDIT,
                replace_sizes=(
                    (
                        observed.created_by_user_id,
                        observed.byte_size,
                        staged.byte_size,
                    )
                    if replace_source
                    else None
                ),
            )
            now = _now()
            with connect(self._database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                current, access = self._current_access(
                    connection, actor, frozenset({RAG_DOCUMENT_UPDATE})
                )
                document = self._access_policy.require_document(
                    connection,
                    context=current.context,
                    access=access,
                    document_id=observed.id,
                    minimum_level=EDIT,
                    states=_VISIBLE_STATES,
                    now=now,
                )
                if replace_source:
                    self._require_replace_capacity(
                        connection,
                        current,
                        document.created_by_user_id,
                        document.byte_size,
                        staged.byte_size,
                        observed.id,
                    )
                if not self._repository.transition_to_indexing(
                    connection,
                    document_id=observed.id,
                    observed_hash=observed.content_hash,
                    now=now,
                ):
                    raise ApiError(
                        409, "document_changed", "Document changed; retry the request"
                    )
                self._audit.record(
                    connection,
                    actor_user_id=current.user_id,
                    action=action,
                    decision="allowed",
                    resource_type="document",
                    resource_id=observed.id,
                    reason=None,
                    request_id=current.context.request_id,
                    details={"byte_size": staged.byte_size, "segments": len(chunks)},
                    now=now,
                )
            linearized = True
            await self._finish_indexing(
                staged, observed.id, chunks, vectors, replace_source=replace_source
            )
        except Exception as error:
            if linearized:
                await self._settle_failed(observed.id, "indexing_failed")
                if not isinstance(error, ApiError):
                    raise ApiError(
                        503, "document_indexing_failed", "Document indexing failed"
                    ) from error
            raise
        finally:
            self._objects.cleanup_stage(staged.directory)
        return self._return_operation(actor, observed.id, RAG_DOCUMENT_UPDATE, EDIT)

    async def _finish_indexing(
        self,
        staged: StagedDocument,
        document_id: str,
        chunks,
        vectors: tuple[tuple[float, ...], ...],
        *,
        replace_source: bool,
    ) -> None:
        self._objects.commit(staged, document_id, source=replace_source)
        now = _now()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not self._repository.update_indexing_content(
                connection,
                document_id=document_id,
                original_filename=staged.original_filename,
                media_type=staged.media_type,
                byte_size=staged.byte_size,
                content_hash=staged.content_hash,
                extracted_characters=len(staged.text),
                chunk_count=len(chunks),
                now=now,
            ):
                raise RuntimeError("Document left indexing state")
        await self._index.replace_document(
            document_id, staged.content_hash, chunks, vectors
        )
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not self._repository.mark_ready(
                connection,
                document_id=document_id,
                content_hash=staged.content_hash,
                now=_now(),
            ):
                raise RuntimeError("Document could not become ready")

    async def _embed_chunks(
        self,
        actor: VerifiedIdentity,
        *,
        chunks,
        capability: str,
        document_id: str | None,
        observed_hash: str | None,
        minimum_level: int | None,
        replace_sizes: tuple[str, int, int] | None,
    ) -> tuple[tuple[float, ...], ...]:
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(chunks), self._embed_batch_size):
            batch = chunks[start : start + self._embed_batch_size]
            with connect(self._database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                current, access = self._current_access(
                    connection, actor, frozenset({capability})
                )
                if document_id is not None and minimum_level is not None:
                    document = self._access_policy.require_document(
                        connection,
                        context=current.context,
                        access=access,
                        document_id=document_id,
                        minimum_level=minimum_level,
                        states=_VISIBLE_STATES,
                        now=_now(),
                    )
                    if document.content_hash != observed_hash:
                        raise ApiError(
                            409, "document_changed", "Document changed; retry the request"
                        )
                if replace_sizes is not None:
                    creator, old_size, new_size = replace_sizes
                    self._require_replace_capacity(
                        connection,
                        current,
                        creator,
                        old_size,
                        new_size,
                        document_id,
                    )
                self._require_embedding_capacity(connection, current, document_id)
            result = await self._embeddings.embed(tuple(chunk.text for chunk in batch))
            self._usage.record_embedding_tokens(actor.user_id, result.total_tokens)
            vectors.extend(result.vectors)
        return tuple(vectors)

    async def _settle_failed(self, document_id: str, error_code: str) -> None:
        with suppress(Exception):
            await self._index.delete_document(document_id)
        with suppress(Exception):
            self._objects.delete_extracted(document_id)
        source = self._objects.source_path(document_id)
        if not source.is_file():
            self._objects.delete_document(document_id)
            with connect(self._database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._repository.delete_record(connection, document_id)
            return
        with connect(self._database_path) as connection:
            current = self._repository.get(connection, document_id)
        if current is None:
            return
        byte_size, content_hash, media_type = self._objects.source_metadata(
            document_id, current.original_filename
        )
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._repository.mark_failed(
                connection,
                document_id=document_id,
                original_filename=current.original_filename,
                media_type=media_type,
                byte_size=byte_size,
                content_hash=content_hash,
                error_code=error_code,
                now=_now(),
            )

    def _operation_document(
        self, actor: VerifiedIdentity, document_id: str, capability: str, level: int
    ) -> Document:
        now = _now()
        with connect(self._database_path) as connection:
            current, access = self._begin_audited_access(
                connection, actor, frozenset({capability})
            )
            return self._access_policy.require_document(
                connection,
                context=current.context,
                access=access,
                document_id=document_id,
                minimum_level=level,
                states=_VISIBLE_STATES,
                now=now,
            )

    def _read_authorized(
        self, actor: VerifiedIdentity, document_id: str, *, states: tuple[str, ...]
    ) -> Document:
        now = _now()
        with connect(self._database_path) as connection:
            current, access = self._begin_audited_access(
                connection, actor, frozenset({RAG_DOCUMENT_READ})
            )
            return self._access_policy.require_document(
                connection,
                context=current.context,
                access=access,
                document_id=document_id,
                minimum_level=READ,
                states=states,
                now=now,
            )

    def _authorized_extracted_snapshot(
        self,
        actor: VerifiedIdentity,
        document_id: str,
        *,
        expected_content_hash: str | None = None,
        capabilities: frozenset[str] = _READ_CAPABILITIES,
    ) -> tuple[Document, str]:
        observed = self._read_authorized_with_capabilities(
            actor,
            document_id,
            capabilities=capabilities,
        )
        if (
            expected_content_hash is not None
            and observed.content_hash != expected_content_hash
        ):
            raise ApiError(409, "document_changed", "Document changed; restart the read")
        text = self._objects.read_extracted(document_id)
        current = self._read_authorized_with_capabilities(
            actor,
            document_id,
            capabilities=capabilities,
        )
        if current.content_hash != observed.content_hash:
            raise ApiError(409, "document_changed", "Document changed; restart the read")
        if len(text) != current.extracted_characters:
            raise ApiError(
                503,
                "document_text_inconsistent",
                "Document text failed consistency validation",
            )
        return current, text

    def _read_authorized_with_capabilities(
        self,
        actor: VerifiedIdentity,
        document_id: str,
        *,
        capabilities: frozenset[str],
    ) -> Document:
        now = _now()
        with connect(self._database_path) as connection:
            current, access = self._begin_audited_access(
                connection, actor, capabilities
            )
            return self._access_policy.require_document(
                connection,
                context=current.context,
                access=access,
                document_id=document_id,
                minimum_level=READ,
                states=("ready",),
                now=now,
            )

    def _return_upload(self, actor: VerifiedIdentity, document_id: str) -> Document:
        with connect(self._database_path) as connection:
            current, access = self._begin_audited_access(
                connection, actor, frozenset({RAG_DOCUMENT_UPLOAD})
            )
            return self._access_policy.require_document(
                connection,
                context=current.context,
                access=access,
                document_id=document_id,
                minimum_level=MANAGE,
                states=("ready",),
                now=_now(),
            )

    def _return_operation(
        self, actor: VerifiedIdentity, document_id: str, capability: str, level: int
    ) -> Document:
        return self._operation_document(actor, document_id, capability, level)

    def _require_capability(self, actor: VerifiedIdentity, capability: str) -> None:
        with connect(self._database_path) as connection:
            connection.execute("BEGIN")
            self._current_access(connection, actor, frozenset({capability}))

    def _current_access(
        self,
        connection: sqlite3.Connection,
        actor: VerifiedIdentity,
        required: frozenset[str],
    ) -> tuple[VerifiedIdentity, CurrentAccess]:
        current = self._identity.revalidate(actor.context, connection=connection)
        access = self._authorization.require_capabilities(
            current.user_id, required, connection=connection
        )
        return current, access

    def _begin_audited_access(
        self,
        connection: sqlite3.Connection,
        actor: VerifiedIdentity,
        required: frozenset[str],
    ) -> tuple[VerifiedIdentity, CurrentAccess]:
        connection.execute("BEGIN")
        current, access = self._current_access(connection, actor, required)
        if not access.bypass:
            return current, access

        # A bypass authorization writes its mandatory audit event in this transaction.
        # End the read snapshot before reserving the SQLite writer slot so concurrent
        # Agent tools cannot deadlock while both try to upgrade a read transaction.
        connection.rollback()
        connection.execute("BEGIN IMMEDIATE")
        return self._current_access(connection, actor, required)

    def _require_create_capacity(
        self,
        connection: sqlite3.Connection,
        actor: VerifiedIdentity,
        byte_size: int,
        document_id: str | None,
    ) -> None:
        try:
            self._usage.require_create_capacity(
                connection, user_id=actor.user_id, byte_size=byte_size
            )
        except ApiError as error:
            self._record_quota_denied(connection, actor, document_id, error)
            raise

    def _require_replace_capacity(
        self,
        connection: sqlite3.Connection,
        actor: VerifiedIdentity,
        creator: str,
        old_size: int,
        new_size: int,
        document_id: str | None,
    ) -> None:
        try:
            self._usage.require_creator_replace_capacity(
                connection,
                actor_user_id=actor.user_id,
                creator_user_id=creator,
                old_byte_size=old_size,
                new_byte_size=new_size,
            )
        except ApiError as error:
            self._record_quota_denied(connection, actor, document_id, error)
            raise

    def _require_embedding_capacity(
        self,
        connection: sqlite3.Connection,
        actor: VerifiedIdentity,
        document_id: str | None,
    ) -> None:
        try:
            self._usage.require_embedding_capacity(connection, user_id=actor.user_id)
        except ApiError as error:
            self._record_quota_denied(connection, actor, document_id, error)
            raise

    def _record_quota_denied(
        self,
        connection: sqlite3.Connection,
        actor: VerifiedIdentity,
        document_id: str | None,
        error: ApiError,
    ) -> None:
        self._audit.record(
            connection,
            actor_user_id=actor.user_id,
            action="usage.quota",
            decision="denied",
            resource_type="document",
            resource_id=document_id,
            reason=error.code,
            request_id=actor.context.request_id,
            details={"quota": error.details.get("quota", "unknown")},
            now=_now(),
        )
        connection.commit()

    def _record_search_audit(
        self,
        connection: sqlite3.Connection,
        *,
        actor: VerifiedIdentity,
        decision: str,
        reason: str | None,
        authorized_count: int,
        reference_ids: tuple[str, ...],
        result_count: int,
        bypass_used: bool,
        now: str,
        suspect_ids: tuple[str, ...] = (),
        quota: object = None,
    ) -> None:
        details: dict[str, object] = {
            "authorized_count": authorized_count,
            "bypass_used": bypass_used,
            "reference_ids": list(reference_ids),
            "result_count": result_count,
        }
        if suspect_ids:
            details["suspect_ids"] = list(suspect_ids)
        if isinstance(quota, str):
            details["quota"] = quota
        self._audit.record(
            connection,
            actor_user_id=actor.user_id,
            action="rag.search",
            decision=decision,
            resource_type="search",
            resource_id=None,
            reason=reason,
            request_id=actor.context.request_id,
            details=details,
            now=now,
        )

    def _record_search_denied(
        self,
        actor: VerifiedIdentity,
        *,
        reason: str,
        authorized_count: int,
        bypass_used: bool,
        suspect_ids: tuple[str, ...] = (),
        quota: object = None,
    ) -> None:
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._record_search_audit(
                connection,
                actor=actor,
                decision="denied",
                reason=reason,
                authorized_count=authorized_count,
                reference_ids=(),
                result_count=0,
                bypass_used=bypass_used,
                suspect_ids=suspect_ids,
                quota=quota,
                now=_now(),
            )


def _title(value: str) -> str:
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 200
        or any(unicodedata.category(character).startswith("C") for character in normalized)
    ):
        raise ApiError(422, "invalid_document_title", "Document title is invalid")
    return normalized


def _search_query(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > _MAX_SEARCH_QUERY_CHARACTERS:
        raise ApiError(422, "invalid_search_query", "Search query is invalid")
    return normalized


def _validate_search_limit(top_k: int) -> None:
    if isinstance(top_k, bool) or top_k < 1 or top_k > _MAX_TOP_K:
        raise ApiError(422, "invalid_search_limit", "Search limit is invalid")


def _validate_search_scope(document_ids: tuple[str, ...] | None) -> None:
    if document_ids is None:
        return
    if not document_ids or len(document_ids) > _MAX_SEARCH_SCOPE:
        raise ApiError(422, "invalid_document_scope", "Document scope is invalid")


def _validate_manifest_page(section_offset: int, section_limit: int) -> None:
    if (
        isinstance(section_offset, bool)
        or section_offset < 0
        or isinstance(section_limit, bool)
        or section_limit < 1
        or section_limit > _MAX_MANIFEST_SECTIONS
    ):
        raise ApiError(422, "invalid_section_page", "Section page is invalid")


def _validate_document_read(offset_characters: int, max_characters: int) -> None:
    if (
        isinstance(offset_characters, bool)
        or offset_characters < 0
        or isinstance(max_characters, bool)
        or max_characters < 1
        or max_characters > _MAX_DOCUMENT_READ_CHARACTERS
    ):
        raise ApiError(
            422,
            "invalid_document_text_range",
            "Document text range is invalid",
        )


def _validate_chunk_context(ordinal: int, before: int, after: int) -> None:
    if (
        isinstance(ordinal, bool)
        or ordinal < 0
        or isinstance(before, bool)
        or before < 0
        or before > _MAX_CHUNK_CONTEXT
        or isinstance(after, bool)
        or after < 0
        or after > _MAX_CHUNK_CONTEXT
    ):
        raise ApiError(422, "invalid_chunk_context", "Chunk context is invalid")


def _document_text_query(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > _MAX_DOCUMENT_TEXT_QUERY_CHARACTERS:
        raise ApiError(422, "invalid_document_text_query", "Document text query is invalid")
    return normalized


def _validate_document_text_search(
    max_matches: int, context_characters: int
) -> None:
    if (
        isinstance(max_matches, bool)
        or max_matches < 1
        or max_matches > _MAX_DOCUMENT_TEXT_MATCHES
        or isinstance(context_characters, bool)
        or context_characters < 0
        or context_characters > _MAX_DOCUMENT_TEXT_CONTEXT_CHARACTERS
    ):
        raise ApiError(
            422,
            "invalid_document_text_search_limit",
            "Document text search limit is invalid",
        )


def _validate_grants(grants: tuple[AclGrant, ...]) -> None:
    seen: set[tuple[str, str]] = set()
    for grant in grants:
        if (grant.user_id is None) == (grant.role_id is None):
            raise ApiError(422, "invalid_acl", "Each ACL grant must have exactly one subject")
        if grant.access_level not in {READ, EDIT, MANAGE}:
            raise ApiError(422, "invalid_acl", "ACL access level is invalid")
        key = (
            ("user", grant.user_id)
            if grant.user_id is not None
            else ("role", grant.role_id)
        )
        if key in seen:
            raise ApiError(422, "duplicate_acl_subject", "ACL contains a duplicate subject")
        seen.add(key)


def _grant_details(grants: tuple[AclGrant, ...]) -> list[dict[str, object]]:
    return [
        {
            "user_id": grant.user_id,
            "role_id": grant.role_id,
            "access_level": grant.access_level,
        }
        for grant in grants
    ]


def _now() -> str:
    return datetime.now(UTC).isoformat()

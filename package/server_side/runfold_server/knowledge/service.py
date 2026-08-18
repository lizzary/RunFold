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
)
from runfold_server.access_control.models import CurrentAccess
from runfold_server.errors import ApiError
from runfold_server.identity.models import VerifiedIdentity
from runfold_server.identity.service import IdentityService
from runfold_server.knowledge.access_policy import KnowledgeAccessPolicy
from runfold_server.knowledge.chunker import chunk_text
from runfold_server.knowledge.lance_index import LanceIndex
from runfold_server.knowledge.models import (
    EDIT,
    MANAGE,
    READ,
    AclGrant,
    Document,
    DocumentContent,
    DocumentText,
    StagedDocument,
)
from runfold_server.knowledge.object_store import AsyncReadable, ObjectStore
from runfold_server.knowledge.repository import KnowledgeRepository
from runfold_server.llm.openai_embeddings import OpenAIEmbeddingsClient
from runfold_server.storage.sqlite import connect
from runfold_server.usage.service import UsageService

_VISIBLE_STATES = ("ready", "failed")


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
            connection.execute("BEGIN")
            current, access = self._current_access(
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
            connection.execute("BEGIN")
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
            connection.execute("BEGIN")
            current, access = self._current_access(
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
            connection.execute("BEGIN")
            current, access = self._current_access(
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

    def _return_upload(self, actor: VerifiedIdentity, document_id: str) -> Document:
        with connect(self._database_path) as connection:
            connection.execute("BEGIN")
            current, access = self._current_access(
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


def _title(value: str) -> str:
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 200
        or any(unicodedata.category(character).startswith("C") for character in normalized)
    ):
        raise ApiError(422, "invalid_document_title", "Document title is invalid")
    return normalized


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

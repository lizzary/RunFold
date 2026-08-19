from __future__ import annotations

import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from runfold_server.access_control.audit import AuditRepository
from runfold_server.access_control.capabilities import SYSTEM_ADMIN_ROLE_ID
from runfold_server.access_control.repository import AccessControlRepository
from runfold_server.errors import ApiError, StartupError
from runfold_server.identity.models import User
from runfold_server.identity.repository import IdentityRepository
from runfold_server.knowledge.chunker import chunk_text
from runfold_server.knowledge.lance_index import IndexConfiguration, LanceIndex
from runfold_server.knowledge.models import Document
from runfold_server.knowledge.object_store import ObjectStore
from runfold_server.knowledge.repository import KnowledgeRepository
from runfold_server.llm.openai_embeddings import OpenAIEmbeddingsClient
from runfold_server.storage.sqlite import connect
from runfold_server.usage.service import UsageService


class IndexMaintenanceService:
    def __init__(
        self,
        *,
        database_path: Path,
        identity_repository: IdentityRepository,
        access_repository: AccessControlRepository,
        repository: KnowledgeRepository,
        audit: AuditRepository,
        objects: ObjectStore,
        index: LanceIndex,
        embeddings: OpenAIEmbeddingsClient,
        usage: UsageService,
        configuration: IndexConfiguration,
        embed_batch_size: int,
    ) -> None:
        self._database_path = database_path
        self._identity_repository = identity_repository
        self._access_repository = access_repository
        self._repository = repository
        self._audit = audit
        self._objects = objects
        self._index = index
        self._embeddings = embeddings
        self._usage = usage
        self._configuration = configuration
        self._embed_batch_size = embed_batch_size

    async def rebuild(self, actor_username: str) -> None:
        self._objects.clear_staging()
        with connect(self._database_path) as connection:
            documents = self._repository.documents_in_states(
                connection, ("ready", "failed")
            )
        source_documents = tuple(
            document
            for document in documents
            if self._objects.source_path(document.id).is_file()
        )
        missing_documents = tuple(
            document for document in documents if document not in source_documents
        )
        now = _now()
        request_id = f"maintenance-{uuid.uuid4().hex}"
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            actor = self._require_actor(connection, actor_username)
            self._repository.prepare_rebuild(
                connection,
                source_document_ids=tuple(document.id for document in source_documents),
                missing_document_ids=tuple(document.id for document in missing_documents),
                now=now,
            )
            self._repository.replace_index_settings(
                connection,
                embedding_identity=self._configuration.embedding_identity,
                model=self._configuration.model,
                dimensions=self._configuration.dimensions,
                chunk_size=self._configuration.chunk_size,
                chunk_overlap=self._configuration.chunk_overlap,
                now=now,
            )
            self._audit.record(
                connection,
                actor_user_id=actor.id,
                action="rag.index.rebuild",
                decision="allowed",
                resource_type="index",
                resource_id=None,
                reason=None,
                request_id=request_id,
                details={
                    "documents": len(source_documents),
                    "source_missing": len(missing_documents),
                },
                now=now,
            )

        self._index.recreate()
        for document in missing_documents:
            self._objects.delete_extracted(document.id)
        for document in source_documents:
            await self._rebuild_document(actor, document, request_id)

    async def _rebuild_document(
        self, actor: User, document: Document, request_id: str
    ) -> None:
        staged = None
        try:
            staged = self._objects.stage_existing(document.id, document.original_filename)
            chunks = chunk_text(
                staged.text,
                size=self._configuration.chunk_size,
                overlap=self._configuration.chunk_overlap,
            )
            vectors: list[tuple[float, ...]] = []
            for start in range(0, len(chunks), self._embed_batch_size):
                batch = chunks[start : start + self._embed_batch_size]
                with connect(self._database_path) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    self._require_actor(connection, actor.username)
                    self._usage.require_embedding_capacity(
                        connection, user_id=actor.id
                    )
                result = await self._embeddings.embed(
                    tuple(chunk.text for chunk in batch)
                )
                self._usage.record_embedding_tokens(actor.id, result.total_tokens)
                vectors.extend(result.vectors)

            self._objects.commit(staged, document.id, source=False)
            with connect(self._database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                if not self._repository.update_indexing_content(
                    connection,
                    document_id=document.id,
                    original_filename=staged.original_filename,
                    media_type=staged.media_type,
                    byte_size=staged.byte_size,
                    content_hash=staged.content_hash,
                    extracted_characters=len(staged.text),
                    chunk_count=len(chunks),
                    now=_now(),
                ):
                    raise RuntimeError("Document left indexing state")
            await self._index.replace_document(
                document.id, staged.content_hash, chunks, tuple(vectors)
            )
            with connect(self._database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                if not self._repository.mark_ready(
                    connection,
                    document_id=document.id,
                    content_hash=staged.content_hash,
                    now=_now(),
                ):
                    raise RuntimeError("Document could not become ready")
        except Exception as error:
            self._settle_failed(document.id)
            self._record_document_failure(
                actor,
                document.id,
                request_id,
                error.code if isinstance(error, ApiError) else "rebuild_failed",
            )
        finally:
            if staged is not None:
                self._objects.cleanup_stage(staged.directory)

    def _settle_failed(self, document_id: str) -> None:
        with suppress(Exception):
            self._index.delete_document_now(document_id)
        with suppress(Exception):
            self._objects.delete_extracted(document_id)
        with connect(self._database_path) as connection:
            document = self._repository.get(connection, document_id)
        if document is None:
            return
        try:
            byte_size, content_hash, media_type = self._objects.source_metadata(
                document_id, document.original_filename
            )
        except Exception:
            with connect(self._database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._repository.mark_unusable(
                    connection,
                    document_id=document_id,
                    error_code="rebuild_failed",
                    expected_states=("indexing",),
                    now=_now(),
                )
            return
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._repository.mark_failed(
                connection,
                document_id=document_id,
                original_filename=document.original_filename,
                media_type=media_type,
                byte_size=byte_size,
                content_hash=content_hash,
                error_code="rebuild_failed",
                now=_now(),
            )

    def _record_document_failure(
        self, actor: User, document_id: str, request_id: str, reason: str
    ) -> None:
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._audit.record(
                connection,
                actor_user_id=actor.id,
                action="rag.index.rebuild.document",
                decision="denied",
                resource_type="document",
                resource_id=document_id,
                reason=reason,
                request_id=request_id,
                details={},
                now=_now(),
            )

    def _require_actor(
        self, connection, actor_username: str
    ) -> User:
        normalized = actor_username.strip()
        record = self._identity_repository.get_user_with_password_by_username(
            connection, normalized
        )
        if (
            record is None
            or record[0].status != "active"
            or SYSTEM_ADMIN_ROLE_ID
            not in self._access_repository.user_role_ids(connection, record[0].id)
        ):
            raise StartupError(
                "invalid_maintenance_actor",
                "Maintenance actor must be an active direct system administrator",
            )
        return record[0]


def _now() -> str:
    return datetime.now(UTC).isoformat()

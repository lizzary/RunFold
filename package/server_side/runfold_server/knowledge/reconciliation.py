from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from runfold_server.knowledge.lance_index import LanceIndex
from runfold_server.knowledge.object_store import ObjectStore
from runfold_server.knowledge.repository import KnowledgeRepository
from runfold_server.storage.sqlite import connect


class ReconciliationService:
    def __init__(
        self,
        *,
        database_path: Path,
        repository: KnowledgeRepository,
        objects: ObjectStore,
        index: LanceIndex,
    ) -> None:
        self._database_path = database_path
        self._repository = repository
        self._objects = objects
        self._index = index

    def run(self) -> None:
        self._objects.clear_staging()
        with connect(self._database_path) as connection:
            interrupted = self._repository.documents_in_states(
                connection, ("indexing", "deleting", "failed")
            )
        for document in interrupted:
            if document.index_state == "deleting":
                self._finish_delete(document.id)
            elif document.index_state == "indexing":
                self._recover_indexing(document.id)
            else:
                self._index.delete_document_now(document.id)

    def _recover_indexing(self, document_id: str) -> None:
        self._index.delete_document_now(document_id)
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
            self._objects.delete_document(document_id)
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
                error_code="interrupted",
                now=datetime.now(UTC).isoformat(),
            )

    def _finish_delete(self, document_id: str) -> None:
        self._index.delete_document_now(document_id)
        self._objects.delete_document(document_id)
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._repository.delete_record(connection, document_id)

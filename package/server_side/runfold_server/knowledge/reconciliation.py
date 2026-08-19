from __future__ import annotations

import logging
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from runfold_server.knowledge.lance_index import LanceIndex
from runfold_server.knowledge.object_store import ObjectStore
from runfold_server.knowledge.repository import KnowledgeRepository
from runfold_server.storage.sqlite import connect

_LOGGER = logging.getLogger("runfold_server.reconciliation")


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
            documents = self._repository.documents_in_states(
                connection, ("indexing", "deleting", "failed", "ready")
            )
        for document in documents:
            if document.index_state == "deleting":
                self._finish_delete(document.id)
            elif document.index_state == "indexing":
                self._recover_indexing(document.id)
            elif document.index_state == "ready":
                self._check_ready(document.id)
            else:
                self._remove_untrusted_derivatives(document.id)
        self._remove_orphan_rows()

    def _recover_indexing(self, document_id: str) -> None:
        self._remove_untrusted_derivatives(document_id)
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
        with suppress(Exception):
            self._index.delete_document_now(document_id)
        self._objects.delete_document(document_id)
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._repository.delete_record(connection, document_id)

    def _check_ready(self, document_id: str) -> None:
        with connect(self._database_path) as connection:
            document = self._repository.get(connection, document_id)
        if document is None or document.index_state != "ready":
            return
        source = self._objects.source_path(document_id)
        extracted = self._objects.extracted_path(document_id)
        intact = source.is_file() and extracted.is_file()
        if intact:
            try:
                byte_size, content_hash, media_type = self._objects.source_metadata(
                    document_id, document.original_filename
                )
                intact = (
                    byte_size == document.byte_size
                    and content_hash == document.content_hash
                    and media_type == document.media_type
                    and self._index.count_document_rows(document_id)
                    == document.chunk_count
                    and self._index.count_document_hash_rows(
                        document_id, document.content_hash
                    )
                    == document.chunk_count
                )
            except Exception:
                intact = False
        if intact:
            return
        self._remove_untrusted_derivatives(document_id)
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._repository.mark_unusable(
                connection,
                document_id=document_id,
                error_code="integrity_check_failed",
                expected_states=("ready",),
                now=datetime.now(UTC).isoformat(),
            )

    def _remove_untrusted_derivatives(self, document_id: str) -> None:
        with suppress(Exception):
            self._index.delete_document_now(document_id)
        with suppress(Exception):
            self._objects.delete_extracted(document_id)

    def _remove_orphan_rows(self) -> None:
        with connect(self._database_path) as connection:
            ready = self._repository.documents_in_states(connection, ("ready",))
        valid = {(document.id, document.content_hash) for document in ready}
        try:
            identities = self._index.row_identities()
        except Exception:
            _LOGGER.warning("orphan_index_scan_failed")
            return
        for document_id, content_hash in identities:
            if (document_id, content_hash) in valid:
                continue
            try:
                self._index.delete_identity_now(document_id, content_hash)
            except Exception:
                _LOGGER.warning("orphan_index_cleanup_failed")

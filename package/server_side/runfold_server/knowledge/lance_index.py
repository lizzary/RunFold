from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import lancedb
import pyarrow as pa

from runfold_server.errors import StartupError
from runfold_server.knowledge.models import Chunk
from runfold_server.storage.sqlite import connect

_TABLE = "chunks"


@dataclass(frozen=True, slots=True)
class IndexConfiguration:
    embedding_identity: str
    model: str
    dimensions: int
    chunk_size: int
    chunk_overlap: int


class LanceIndex:
    def __init__(self, path: Path, dimensions: int) -> None:
        self._database = lancedb.connect(path)
        self._dimensions = dimensions
        self._write_lock = asyncio.Lock()

    @property
    def expected_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("document_id", pa.string(), nullable=False),
                pa.field("chunk_id", pa.string(), nullable=False),
                pa.field("ordinal", pa.int64(), nullable=False),
                pa.field("content_hash", pa.string(), nullable=False),
                pa.field("text", pa.string(), nullable=False),
                pa.field("vector", pa.list_(pa.float32(), self._dimensions), nullable=False),
            ]
        )

    def table_is_current(self) -> bool:
        names = tuple(self._database.list_tables().tables)
        if names != (_TABLE,):
            return False
        return self._database.open_table(_TABLE).schema.equals(
            self.expected_schema, check_metadata=True
        )

    def recreate(self) -> None:
        for name in tuple(self._database.list_tables().tables):
            self._database.drop_table(name)
        self._database.create_table(_TABLE, schema=self.expected_schema)

    async def replace_document(
        self,
        document_id: str,
        content_hash: str,
        chunks: tuple[Chunk, ...],
        vectors: tuple[tuple[float, ...], ...],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts differ")
        async with self._write_lock:
            self.delete_document_now(document_id)
            if not chunks:
                return
            rows = [
                {
                    "document_id": document_id,
                    "chunk_id": chunk.chunk_id,
                    "ordinal": chunk.ordinal,
                    "content_hash": content_hash,
                    "text": chunk.text,
                    "vector": list(vector),
                }
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            table = pa.Table.from_pylist(rows, schema=self.expected_schema)
            self._database.open_table(_TABLE).add(table, on_bad_vectors="error")

    async def delete_document(self, document_id: str) -> None:
        async with self._write_lock:
            self.delete_document_now(document_id)

    def delete_document_now(self, document_id: str) -> None:
        predicate = _document_predicate(document_id)
        self._database.open_table(_TABLE).delete(predicate)

    def count_document_rows(self, document_id: str) -> int:
        return int(self._database.open_table(_TABLE).count_rows(_document_predicate(document_id)))


def initialize_index(
    *,
    database_path: Path,
    lance_path: Path,
    configuration: IndexConfiguration,
) -> LanceIndex:
    index = LanceIndex(lance_path, configuration.dimensions)
    with connect(database_path) as connection:
        current = connection.execute(
            """
            SELECT embedding_identity, model, dimensions, chunk_size, chunk_overlap
            FROM rag_index_settings
            WHERE singleton = 1
            """
        ).fetchone()
        ready_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM documents WHERE index_state = 'ready'"
            ).fetchone()[0]
        )
    settings_match = current is not None and tuple(current) == (
        configuration.embedding_identity,
        configuration.model,
        configuration.dimensions,
        configuration.chunk_size,
        configuration.chunk_overlap,
    )
    table_match = index.table_is_current()
    if settings_match and table_match:
        return index
    if ready_count:
        raise StartupError(
            "incompatible_rag_index",
            "RAG index configuration is incompatible; run the stopped-service rebuild command",
        )

    try:
        index.recreate()
    except Exception as error:
        raise StartupError(
            "rag_index_initialization_failed", "RAG index initialization failed"
        ) from error
    now = datetime.now(UTC).isoformat()
    with connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM rag_index_settings")
        connection.execute(
            """
            INSERT INTO rag_index_settings (
                singleton, embedding_identity, model, dimensions,
                chunk_size, chunk_overlap, created_at, updated_at
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                configuration.embedding_identity,
                configuration.model,
                configuration.dimensions,
                configuration.chunk_size,
                configuration.chunk_overlap,
                now,
                now,
            ),
        )
    return index


def _document_predicate(document_id: str) -> str:
    try:
        parsed = uuid.UUID(document_id)
    except ValueError as error:
        raise ValueError("Invalid internal document ID") from error
    if str(parsed) != document_id:
        raise ValueError("Invalid internal document ID")
    return f"document_id = '{document_id}'"

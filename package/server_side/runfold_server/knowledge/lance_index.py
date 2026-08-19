from __future__ import annotations

import asyncio
import math
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import lancedb
import pyarrow as pa

from runfold_server.errors import StartupError
from runfold_server.knowledge.models import Chunk, IndexSearchHit
from runfold_server.storage.sqlite import connect

_TABLE = "chunks"
_FILTER_BATCH_SIZE = 100
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class UnsafeIndexResult(RuntimeError):
    def __init__(self, document_ids: tuple[str, ...] = ()) -> None:
        self.document_ids = tuple(sorted(set(document_ids)))[:20]
        super().__init__("Vector index returned an unsafe result")


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

    def search(
        self,
        vector: tuple[float, ...],
        *,
        document_ids: tuple[str, ...],
        top_k: int,
    ) -> tuple[IndexSearchHit, ...]:
        if not document_ids:
            raise ValueError("A filtered search requires document IDs")
        if len(vector) != self._dimensions or not all(math.isfinite(value) for value in vector):
            raise ValueError("Search vector is invalid")
        if top_k <= 0:
            raise ValueError("Search limit must be positive")

        table = self._database.open_table(_TABLE)
        hits: list[IndexSearchHit] = []
        seen_chunks: set[tuple[str, str]] = set()
        for start in range(0, len(document_ids), _FILTER_BATCH_SIZE):
            batch = document_ids[start : start + _FILTER_BATCH_SIZE]
            allowed = frozenset(batch)
            rows = (
                table.search(list(vector), vector_column_name="vector", query_type="vector")
                .where(_documents_predicate(batch), prefilter=True)
                .select(
                    [
                        "document_id",
                        "chunk_id",
                        "ordinal",
                        "content_hash",
                        "text",
                    ]
                )
                .limit(top_k)
                .to_list()
            )
            for row in rows:
                hit = _validated_hit(row, allowed)
                key = (hit.document_id, hit.chunk_id)
                if key in seen_chunks:
                    raise UnsafeIndexResult((hit.document_id,))
                seen_chunks.add(key)
                hits.append(hit)
        hits.sort(
            key=lambda hit: (
                hit.distance,
                hit.document_id,
                hit.ordinal,
                hit.chunk_id,
            )
        )
        return tuple(hits[:top_k])


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
    _validate_document_id(document_id)
    return f"document_id = '{document_id}'"


def _documents_predicate(document_ids: tuple[str, ...]) -> str:
    for document_id in document_ids:
        _validate_document_id(document_id)
    values = ",".join(f"'{document_id}'" for document_id in document_ids)
    return f"document_id IN ({values})"


def _validate_document_id(document_id: str) -> None:
    try:
        parsed = uuid.UUID(document_id)
    except ValueError as error:
        raise ValueError("Invalid internal document ID") from error
    if str(parsed) != document_id:
        raise ValueError("Invalid internal document ID")


def _validated_hit(
    row: dict[str, object], allowed_document_ids: frozenset[str]
) -> IndexSearchHit:
    document_id = row.get("document_id")
    chunk_id = row.get("chunk_id")
    ordinal = row.get("ordinal")
    content_hash = row.get("content_hash")
    text = row.get("text")
    distance = row.get("_distance")
    suspect = (document_id,) if isinstance(document_id, str) else ()
    if (
        not isinstance(document_id, str)
        or document_id not in allowed_document_ids
        or not isinstance(chunk_id, str)
        or _SHA256.fullmatch(chunk_id) is None
        or isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal < 0
        or not isinstance(content_hash, str)
        or _SHA256.fullmatch(content_hash) is None
        or not isinstance(text, str)
        or not text
        or isinstance(distance, bool)
        or not isinstance(distance, (int, float))
        or not math.isfinite(float(distance))
        or float(distance) < 0
    ):
        raise UnsafeIndexResult(suspect)
    return IndexSearchHit(
        document_id=document_id,
        chunk_id=chunk_id,
        ordinal=ordinal,
        content_hash=content_hash,
        text=text,
        distance=float(distance),
    )

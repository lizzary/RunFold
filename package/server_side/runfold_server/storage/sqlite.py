from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from runfold_server.errors import StartupError

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_BUSY_TIMEOUT_MILLISECONDS = 5_000
_IMMUTABLE_SEED_TABLES = ("service_state", "capabilities")


@dataclass(frozen=True, slots=True)
class DataPaths:
    root: Path
    database: Path
    objects: Path
    lance: Path
    staging: Path


def initialize_data_paths(data_dir: Path) -> DataPaths:
    root = data_dir.resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise StartupError(
            "invalid_data_directory",
            "The configured data directory is not a directory",
        )

    candidates = {
        "database": root / "runfold.sqlite3",
        "objects": root / "objects",
        "lance": root / "lance",
        "staging": root / "staging",
    }
    resolved = {name: path.resolve(strict=False) for name, path in candidates.items()}
    if any(path == root or not path.is_relative_to(root) for path in resolved.values()):
        raise StartupError(
            "unsafe_data_path",
            "A configured storage path escapes the data directory",
        )
    if len(set(resolved.values())) != len(resolved):
        raise StartupError("overlapping_data_paths", "Configured storage paths must not overlap")

    database = resolved["database"]
    if database.exists() and not database.is_file():
        raise StartupError("invalid_database_path", "The SQLite path is not a regular file")
    for name in ("objects", "lance", "staging"):
        path = resolved[name]
        path.mkdir(exist_ok=True)
        if not path.is_dir():
            raise StartupError("invalid_data_directory", "A storage path is not a directory")

    return DataPaths(
        root=root,
        database=database,
        objects=resolved["objects"],
        lance=resolved["lance"],
        staging=resolved["staging"],
    )


def connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path, timeout=_BUSY_TIMEOUT_MILLISECONDS / 1_000)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MILLISECONDS}")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def initialize_database(database_path: Path) -> None:
    schema = _SCHEMA_PATH.read_text(encoding="utf-8")
    connection = connect(database_path)
    try:
        if not _application_tables(connection):
            try:
                connection.executescript(schema)
            except sqlite3.Error as error:
                connection.rollback()
                raise StartupError(
                    "database_initialization_failed",
                    "SQLite initialization failed",
                ) from error
        _assert_current_schema(connection, schema)
    finally:
        connection.close()


def check_database_ready(database_path: Path) -> None:
    schema = _SCHEMA_PATH.read_text(encoding="utf-8")
    connection = connect(database_path)
    try:
        _assert_current_schema(connection, schema)
        result = connection.execute("PRAGMA quick_check").fetchone()
        if result is None or result[0] != "ok":
            raise StartupError("database_not_ready", "SQLite integrity check failed")
    finally:
        connection.close()


def check_local_directories(paths: DataPaths) -> None:
    for path in (paths.objects, paths.lance, paths.staging):
        resolved = path.resolve(strict=True)
        if not resolved.is_dir() or not resolved.is_relative_to(paths.root):
            raise StartupError("storage_not_ready", "A local storage directory is unavailable")
        if not os.access(resolved, os.R_OK | os.W_OK):
            raise StartupError("storage_not_ready", "A local storage directory is unavailable")


def _assert_current_schema(connection: sqlite3.Connection, schema: str) -> None:
    expected = sqlite3.connect(":memory:")
    expected.row_factory = sqlite3.Row
    expected.execute("PRAGMA foreign_keys = ON")
    try:
        expected.executescript(schema)
        if _manifest(connection) != _manifest(expected):
            raise StartupError(
                "incompatible_database_schema",
                "The SQLite schema is incompatible; rebuild the data directory explicitly",
            )
        for table in _IMMUTABLE_SEED_TABLES:
            if _table_rows(connection, table) != _table_rows(expected, table):
                raise StartupError(
                    "incompatible_database_seed",
                    "The SQLite fixed data is incompatible; rebuild the data directory explicitly",
                )
        if _protected_role_rows(connection) != _protected_role_rows(expected):
            raise StartupError(
                "incompatible_database_seed",
                "The SQLite protected roles are incompatible; rebuild the data directory "
                "explicitly",
            )
        if _protected_role_capabilities(connection) != _protected_role_capabilities(expected):
            raise StartupError(
                "incompatible_database_seed",
                "The SQLite protected role capabilities are incompatible; rebuild the data "
                "directory explicitly",
            )
    finally:
        expected.close()


def _manifest(connection: sqlite3.Connection) -> tuple[object, ...]:
    objects = tuple(
        tuple(row)
        for row in connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL
            ORDER BY type, name
            """
        )
    )
    normalized_objects = tuple(
        (object_type, name, table, _normalize_sql(sql))
        for object_type, name, table, sql in objects
    )
    tables = _application_tables(connection)
    columns = tuple((table, _table_columns(connection, table)) for table in tables)
    foreign_keys = tuple((table, _foreign_keys(connection, table)) for table in tables)
    return normalized_objects, columns, foreign_keys


def _application_tables(connection: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        row[0]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
    )


def _table_columns(connection: sqlite3.Connection, table: str) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in connection.execute(
            "SELECT cid, name, type, \"notnull\", dflt_value, pk, hidden "
            "FROM pragma_table_xinfo(?) ORDER BY cid",
            (table,),
        )
    )


def _foreign_keys(connection: sqlite3.Connection, table: str) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in connection.execute(
            "SELECT id, seq, \"table\", \"from\", \"to\", on_update, on_delete, match "
            "FROM pragma_foreign_key_list(?) ORDER BY id, seq",
            (table,),
        )
    )


def _table_rows(connection: sqlite3.Connection, table: str) -> tuple[tuple[object, ...], ...]:
    quoted_table = '"' + table.replace('"', '""') + '"'
    return tuple(tuple(row) for row in connection.execute(f"SELECT * FROM {quoted_table}"))


def _protected_role_rows(connection: sqlite3.Connection) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in connection.execute(
            "SELECT * FROM roles WHERE is_protected = 1 ORDER BY id"
        )
    )


def _protected_role_capabilities(
    connection: sqlite3.Connection,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(row)
        for row in connection.execute(
            """
            SELECT rc.role_id, rc.capability_code
            FROM role_capabilities AS rc
            JOIN roles AS r ON r.id = rc.role_id
            WHERE r.is_protected = 1
            ORDER BY rc.role_id, rc.capability_code
            """
        )
    )


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()

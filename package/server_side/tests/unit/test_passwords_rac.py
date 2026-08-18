from __future__ import annotations

import sqlite3

from runfold_server.access_control.capabilities import ALL_CAPABILITIES
from runfold_server.identity.passwords import Argon2PasswordHasher
from runfold_server.storage.sqlite import initialize_data_paths, initialize_database


def test_argon2id_password_hasher() -> None:
    hasher = Argon2PasswordHasher()
    encoded = hasher.hash("correct horse battery staple")

    assert encoded.startswith("$argon2id$")
    assert hasher.verify(encoded, "correct horse battery staple")
    assert not hasher.verify(encoded, "incorrect password")


def test_fixed_capability_catalog_matches_production_schema(tmp_path) -> None:
    paths = initialize_data_paths(tmp_path / "data")
    initialize_database(paths.database)

    with sqlite3.connect(paths.database) as connection:
        codes = {row[0] for row in connection.execute("SELECT code FROM capabilities")}
        protected = connection.execute(
            "SELECT id, name, is_protected FROM roles WHERE is_protected = 1"
        ).fetchall()

    assert codes == ALL_CAPABILITIES
    assert protected == [
        ("00000000-0000-4000-8000-000000000001", "system_admin", 1)
    ]

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from argon2.low_level import Type


class Argon2PasswordHasher:
    def __init__(self) -> None:
        self._hasher = PasswordHasher(type=Type.ID)
        self._dummy_hash = self._hasher.hash("runfold-dummy-password-not-used")

    def hash(self, password: str) -> str:
        return self._hasher.hash(password)

    def verify(self, password_hash: str, password: str) -> bool:
        try:
            return self._hasher.verify(password_hash, password)
        except (InvalidHashError, VerificationError, VerifyMismatchError):
            return False

    def verify_dummy(self, password: str) -> None:
        self.verify(self._dummy_hash, password)

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class User:
    id: str
    username: str
    display_name: str
    status: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class AuthContext:
    user_id: str
    session_id: str
    request_id: str


@dataclass(frozen=True, slots=True)
class VerifiedIdentity:
    context: AuthContext
    user: User

    @property
    def user_id(self) -> str:
        return self.context.user_id


@dataclass(frozen=True, slots=True)
class LoginResult:
    token: str
    expires_at: str
    identity: VerifiedIdentity


@dataclass(frozen=True, slots=True)
class PreparedUser:
    id: str
    username: str
    display_name: str
    password_hash: str
    created_at: str

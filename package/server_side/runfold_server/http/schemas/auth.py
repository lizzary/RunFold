from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class LoginRequest(StrictModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class PasswordChangeRequest(StrictModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class UserResponse(StrictModel):
    id: str
    username: str
    display_name: str
    status: Literal["active", "disabled"]
    created_at: str
    updated_at: str


class LoginResponse(StrictModel):
    token: str
    token_type: Literal["Bearer"] = "Bearer"
    expires_at: str
    user: UserResponse

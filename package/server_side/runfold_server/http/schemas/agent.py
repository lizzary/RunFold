from __future__ import annotations

from pydantic import Field, field_validator

from runfold_server.http.schemas.auth import StrictModel


class AgentRunRequest(StrictModel):
    input: str = Field(min_length=1)
    thinking_level: str | None = None

    @field_validator("thinking_level")
    @classmethod
    def normalize_thinking_level(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip().lower()


class AgentRunResponse(StrictModel):
    answer: str
    reasoning_content: str | None
    thinking_level: str | None
    agents_created: int
    max_depth_reached: int

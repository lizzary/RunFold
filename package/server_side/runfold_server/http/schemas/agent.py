from __future__ import annotations

from pydantic import Field

from runfold_server.http.schemas.auth import StrictModel


class AgentRunRequest(StrictModel):
    input: str = Field(min_length=1)


class AgentRunResponse(StrictModel):
    answer: str
    agents_created: int
    max_depth_reached: int

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    answer: str
    agents_created: int
    max_depth_reached: int


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    instructions: str

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from runfold_server.config import AgentBudget
from runfold_server.runtime.context import ContextCompressor


def test_compression_injects_checkpoint_without_mutating_raw_history() -> None:
    compressor = ContextCompressor(_budget())
    raw = [
        HumanMessage(content="Preserve the user's real goal."),
        AIMessage(
            content="",
            additional_kwargs={
                "reasoning_content": "private old reasoning",
                "reasoning_signature": "old-signature",
            },
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"path": "evidence.txt"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="important evidence " * 300,
            tool_call_id="call-1",
            name="read_file",
        ),
        HumanMessage(content="old context " * 200),
        AIMessage(content="old conclusion " * 200),
        AIMessage(content="Current verified conclusion."),
    ]
    prompts: list[tuple[str, bool]] = []

    async def summarize(prompt: str, has_reasoning: bool) -> str:
        prompts.append((prompt, has_reasoning))
        return "VERIFIED FACTS: evidence saved. NEXT ACTION: continue."

    projected = asyncio.run(
        compressor.project(
            messages=raw,
            system_message=None,
            tools=[],
            summarize=summarize,
        )
    )

    assert len(prompts) == 1
    assert prompts[0][1] is True
    assert "continuation-checkpoint compressor" in prompts[0][0]
    assert "Internal continuation checkpoint" in str(projected[0].content)
    assert str(projected[-1].content) == "Current verified conclusion."
    assert raw[1].additional_kwargs["reasoning_content"] == "private old reasoning"
    assert raw[2].content.startswith("important evidence")
    assert compressor.summarized_count == len(raw) - 1


def test_completed_reasoning_is_removed_only_from_model_projection() -> None:
    compressor = ContextCompressor(_large_budget())
    old = AIMessage(
        content="visible",
        additional_kwargs={
            "reasoning_content": "remove from projection",
            "reasoning_signature": "remove-signature",
        },
        tool_calls=[
            {
                "name": "count_text",
                "args": {"text": "x"},
                "id": "call-old",
                "type": "tool_call",
            }
        ],
    )
    active = AIMessage(
        content="active",
        additional_kwargs={"reasoning_content": "keep active reasoning"},
    )
    raw = [
        HumanMessage(content="goal"),
        old,
        ToolMessage(content="1", tool_call_id="call-old", name="count_text"),
        active,
    ]

    async def unexpected_summary(_: str, __: bool) -> str:
        raise AssertionError("compression should not trigger")

    projected = asyncio.run(
        compressor.project(
            messages=raw,
            system_message=None,
            tools=[],
            summarize=unexpected_summary,
        )
    )

    assert "reasoning_content" not in projected[1].additional_kwargs
    assert "reasoning_signature" not in projected[1].additional_kwargs
    assert projected[-1].additional_kwargs["reasoning_content"] == (
        "keep active reasoning"
    )
    assert old.additional_kwargs["reasoning_content"] == "remove from projection"


def test_oversized_tool_result_keeps_head_tail_and_explicit_marker() -> None:
    compressor = ContextCompressor(_budget())
    raw = [
        HumanMessage(content="goal"),
        ToolMessage(
            content="HEAD" + ("middle" * 400) + "TAIL",
            tool_call_id="orphan-result",
            name="read_file",
        ),
    ]

    async def summarize(_: str, __: bool) -> str:
        return "VERIFIED FACTS: goal retained."

    projected = asyncio.run(
        compressor.project(
            messages=raw,
            system_message=None,
            tools=[],
            summarize=summarize,
        )
    )

    tool_result = projected[-1]
    assert isinstance(tool_result, ToolMessage)
    assert str(tool_result.content).startswith("HEAD")
    assert str(tool_result.content).endswith("TAIL")
    assert "RUNFOLD TOOL RESULT TRUNCATED" in str(tool_result.content)
    assert tool_result.additional_kwargs["runfold_truncated"] is True
    assert "TRUNCATED" not in str(raw[-1].content)


def _budget() -> AgentBudget:
    return AgentBudget(
        context_window_tokens=5000,
        provider_concurrency=1,
        input_tokens=1200,
        output_tokens=300,
        thinking_tokens=100,
        compression_threshold=0.5,
    )


def _large_budget() -> AgentBudget:
    return AgentBudget(
        context_window_tokens=10000,
        provider_concurrency=1,
        input_tokens=5000,
        output_tokens=1000,
        thinking_tokens=200,
        compression_threshold=0.8,
    )

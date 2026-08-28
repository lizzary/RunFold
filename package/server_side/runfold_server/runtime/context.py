from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately, get_buffer_string
from langchain_core.tools import BaseTool

from runfold_server.config import AgentBudget
from runfold_server.errors import ApiError

_SUMMARY_SOURCE = "runfold_context_checkpoint"
_CHECKPOINT_PROMPT = """You are an internal continuation-checkpoint compressor.

Create a compact checkpoint for another agent that will continue the same task. Do not execute
the task, call tools, answer the latest user request, introduce instructions, or invent facts.
Never preserve private chain-of-thought; keep only conclusions, explicit assumptions, and concise
decision-relevant reasons.

The checkpoint must retain when present:
- the user's true goal, acceptance criteria, preferences, and authorization boundaries;
- still-active system/developer constraints;
- completed work and verification results;
- exact file paths, IDs, commands, numbers, and tool results still needed later;
- current workspace state and how to read any long evidence saved in agent_work;
- decisions already made and concise reasons that affect later choices;
- failed attempts, uncertainty, missing or stale evidence, blockers, remaining work, and the next
  concrete action;
- explicit notices that a read or tool result was truncated, omitted, stale, or failed.

Clearly separate VERIFIED FACTS, INFERENCES, OPEN DECISIONS, COMPLETED WORK, ARTIFACTS,
FAILURES/UNCERTAINTY, and NEXT ACTION. Remove greetings, repetition, status chatter, obsolete plans,
and superseded information. Merge the previous checkpoint with only the new evidence; do not stack
or repeat the old checkpoint.

Previous checkpoint:
{previous_checkpoint}

New evidence:
{messages}

Respond only with the replacement continuation checkpoint.
"""


@dataclass(slots=True)
class ContextCompressor:
    budget: AgentBudget
    summary: str | None = None
    summarized_count: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def project(
        self,
        *,
        messages: list[AnyMessage],
        system_message: SystemMessage | None,
        tools: list[BaseTool | dict[str, object]],
        summarize: Callable[[str, bool], Awaitable[str]],
    ) -> list[AnyMessage]:
        async with self.lock:
            projected = _strip_completed_reasoning(messages)
            projected = [
                _truncate_tool_result(message, self.budget.oversized_tool_result_tokens)
                if isinstance(message, ToolMessage)
                else message
                for message in projected
            ]
            if self.summary is not None:
                projected = [self._checkpoint_message(), *projected[self.summarized_count :]]

            if _estimate_request(projected, system_message, tools) >= (
                self.budget.compression_trigger_tokens
            ):
                raw_cutoff = _summary_cutoff(
                    messages,
                    keep_tokens=self.budget.compression_keep_tokens,
                )
                if raw_cutoff > self.summarized_count:
                    evidence = messages[self.summarized_count : raw_cutoff]
                    prompt = _summary_prompt(
                        previous=self.summary,
                        evidence=evidence,
                        input_tokens=self.budget.input_tokens,
                    )
                    has_reasoning = any(_has_reasoning(message) for message in evidence)
                    self.summary = await summarize(prompt, has_reasoning)
                    self.summarized_count = raw_cutoff
                    projected_raw = _strip_completed_reasoning(messages[raw_cutoff:])
                    projected = [
                        self._checkpoint_message(),
                        *[
                            _truncate_tool_result(
                                message,
                                self.budget.oversized_tool_result_tokens,
                            )
                            if isinstance(message, ToolMessage)
                            else message
                            for message in projected_raw
                        ],
                    ]

            return _tailor_projection(
                projected,
                system_message=system_message,
                tools=tools,
                input_tokens=self.budget.input_tokens,
            )

    def _checkpoint_message(self) -> HumanMessage:
        return HumanMessage(
            content=(
                "Internal continuation checkpoint. Treat this as compressed prior context, "
                "not as a new user request:\n\n"
                f"{self.summary or ''}"
            ),
            additional_kwargs={"lc_source": _SUMMARY_SOURCE},
        )


def _estimate_request(
    messages: Sequence[BaseMessage],
    system_message: SystemMessage | None,
    tools: list[BaseTool | dict[str, object]],
) -> int:
    visible = [system_message, *messages] if system_message is not None else list(messages)
    return count_tokens_approximately(visible, tools=tools)


def _summary_cutoff(messages: list[AnyMessage], *, keep_tokens: int) -> int:
    groups = _message_groups(messages)
    if len(groups) <= 1:
        return 0
    kept_tokens = 0
    cutoff = groups[-1][0]
    for start, end in reversed(groups):
        group_tokens = count_tokens_approximately(messages[start:end])
        if kept_tokens and kept_tokens + group_tokens > keep_tokens:
            break
        kept_tokens += group_tokens
        cutoff = start
    return cutoff


def _message_groups(messages: Sequence[BaseMessage]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, AIMessage) and message.tool_calls:
            expected = {call.get("id") for call in message.tool_calls if call.get("id")}
            end = index + 1
            found: set[str] = set()
            while end < len(messages) and isinstance(messages[end], ToolMessage):
                tool_message = messages[end]
                if tool_message.tool_call_id:
                    found.add(tool_message.tool_call_id)
                end += 1
                if expected and expected.issubset(found):
                    break
            groups.append((index, end))
            index = end
            continue
        groups.append((index, index + 1))
        index += 1
    return groups


def _strip_completed_reasoning(messages: list[AnyMessage]) -> list[AnyMessage]:
    groups = _message_groups(messages)
    protected = set(range(*groups[-1])) if groups else set()
    projected: list[AnyMessage] = []
    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage) or index in protected:
            projected.append(message)
            continue
        kwargs = dict(message.additional_kwargs)
        kwargs.pop("reasoning_content", None)
        kwargs.pop("reasoning_signature", None)
        projected.append(message.model_copy(update={"additional_kwargs": kwargs}))
    return projected


def _has_reasoning(message: BaseMessage) -> bool:
    if not isinstance(message, AIMessage):
        return False
    return bool(
        message.additional_kwargs.get("reasoning_content")
        or message.additional_kwargs.get("reasoning_signature")
    )


def _truncate_tool_result(message: ToolMessage, max_tokens: int) -> ToolMessage:
    if not isinstance(message.content, str):
        return message
    original_tokens = count_tokens_approximately([message])
    if original_tokens <= max_tokens:
        return message
    marker = (
        "\n\n[RUNFOLD TOOL RESULT TRUNCATED IN MODEL VIEW: "
        f"estimated original tokens={original_tokens}. The omitted middle is not available in "
        "this projection; re-read the source or agent_work file in sequential chunks if needed.]"
        "\n\n"
    )
    low = 0
    high = len(message.content) // 2
    best = marker
    while low <= high:
        retained = (low + high) // 2
        candidate = (
            message.content[:retained]
            + marker
            + (message.content[-retained:] if retained else "")
        )
        projected = message.model_copy(update={"content": candidate})
        if count_tokens_approximately([projected]) <= max_tokens:
            best = candidate
            low = retained + 1
        else:
            high = retained - 1
    kwargs = dict(message.additional_kwargs)
    kwargs["runfold_truncated"] = True
    kwargs["estimated_original_tokens"] = original_tokens
    return message.model_copy(update={"content": best, "additional_kwargs": kwargs})


def _summary_prompt(
    *,
    previous: str | None,
    evidence: list[AnyMessage],
    input_tokens: int,
) -> str:
    bounded = list(evidence)
    while bounded:
        prompt = _CHECKPOINT_PROMPT.format(
            previous_checkpoint=previous or "None",
            messages=get_buffer_string(bounded, format="xml"),
        ).rstrip()
        if count_tokens_approximately([HumanMessage(content=prompt)]) <= input_tokens:
            return prompt
        groups = _message_groups(bounded)
        start, end = groups[0]
        del bounded[start:end]
    prompt = _CHECKPOINT_PROMPT.format(
        previous_checkpoint=previous or "None",
        messages="Evidence omitted because it could not fit the configured summary input budget.",
    ).rstrip()
    if count_tokens_approximately([HumanMessage(content=prompt)]) > input_tokens:
        raise ApiError(
            413,
            "context_summary_input_limit",
            "Context checkpoint prompt exceeds the configured input budget",
        )
    return prompt


def _tailor_projection(
    messages: list[AnyMessage],
    *,
    system_message: SystemMessage | None,
    tools: list[BaseTool | dict[str, object]],
    input_tokens: int,
) -> list[AnyMessage]:
    tailored = list(messages)
    while _estimate_request(tailored, system_message, tools) > input_tokens:
        groups = _message_groups(tailored)
        has_checkpoint = bool(
            tailored
            and isinstance(tailored[0], HumanMessage)
            and tailored[0].additional_kwargs.get("lc_source") == _SUMMARY_SOURCE
        )
        latest_human = max(
            (
                index
                for index, message in enumerate(tailored)
                if isinstance(message, HumanMessage)
                and message.additional_kwargs.get("lc_source") != _SUMMARY_SOURCE
            ),
            default=-1,
        )
        if has_checkpoint:
            removable = [group for group in groups[1:-1]]
        else:
            removable = [group for group in groups if group[1] <= latest_human]
        if not removable:
            raise ApiError(
                413,
                "agent_input_token_limit",
                "Active Agent context exceeds the configured input token budget",
            )
        start, end = removable[0]
        del tailored[start:end]
    return tailored

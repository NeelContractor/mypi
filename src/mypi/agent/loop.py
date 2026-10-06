from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from ..types import (
    AssistantMessage,
    Done,
    Message,
    Provider,
    TextDelta,
    Tool,
    ToolCallBlock,
    ToolResultMessage,
    UserMessage,
)


@dataclass
class TextEvent:
    delta: str


@dataclass
class ToolStart:
    call: ToolCallBlock


@dataclass
class ToolEnd:
    call: ToolCallBlock
    result: str
    is_error: bool


@dataclass
class TurnEnd:
    message: AssistantMessage


@dataclass
class MessageEvent:
    message: Message


@dataclass
class RunEnd:
    reason: str


AgentEvent = TextEvent | ToolStart | ToolEnd | TurnEnd | MessageEvent | RunEnd

CONTINUE = (
    "Your previous reply was cut off by the token limit. Continue it from exactly "
    "where it stopped, without repeating anything you already wrote."
)


class StreamError(RuntimeError):
    """The provider stream stalled or died; worth retrying, unlike a tool failure."""


async def _run_tool(call: ToolCallBlock, tools: list[Tool]) -> tuple[str, bool]:
    """Returns (result, is_error); every failure becomes a message for the model."""
    if call.error:
        return call.error, True
    try:
        tool = next((t for t in tools if t.name == call.name), None)
        if tool is None:
            raise ValueError(f"unknown tool name {call.name}")
        return await tool.execute(call.arguments), False
    except Exception as e:  # tool errors go back to the model, not the crash log
        return f"Error {e}", True


async def _one_turn(
    provider: Provider,
    model: str,
    messages: list[Message],
    tools: list[Tool],
    system: str | None,
    stream_timeout: float,
    on_event: Callable[[TextEvent], None],
) -> AssistantMessage:
    """Stream one assistant turn. Raises if the stream stalls or dies empty."""
    assistant: AssistantMessage | None = None
    stream = provider.stream(messages=messages, model=model, system=system, tools=tools)
    try:
        async with asyncio.timeout(stream_timeout):
            async for event in stream:
                if isinstance(event, TextDelta):
                    on_event(TextEvent(event.delta))
                elif isinstance(event, Done):
                    assistant = event.message
    except TimeoutError:
        await stream.aclose()
        raise StreamError(
            f"the model stopped sending tokens for {stream_timeout:g}s "
            f"(raise it with --stream-timeout)"
        ) from None
    if assistant is None:
        await stream.aclose()
        raise StreamError("the connection closed before the model finished its turn")
    return assistant


async def run_agent(
    *,
    provider: Provider,
    model: str,
    tools: list[Tool],
    messages: list[Message],
    on_event: Callable[[AgentEvent], None],
    system: str | None = None,
    max_turns: int = 20,
    max_continuations: int = 3,
    stream_timeout: float = 300.0,
    stream_attempts: int = 3,
) -> None:
    """Drives the tool loop until the model stops asking for tools.

    Ends on a normal stop, on `max_turns`, or when the stream keeps failing; the
    reason is reported through a RunEnd event rather than an exception, because a
    partial answer plus a reason is more useful to the caller than a traceback.
    """

    def push(message: Message) -> None:
        messages.append(message)
        on_event(MessageEvent(message))

    def calls_of(message: AssistantMessage) -> list[ToolCallBlock]:
        return [b for b in message.content if isinstance(b, ToolCallBlock)]

    async def run_calls(calls: list[ToolCallBlock]) -> None:
        """Run one batch concurrently; gather hands the results back in call order."""
        for call in calls:
            on_event(ToolStart(call))
        results = await asyncio.gather(*(_run_tool(call, tools) for call in calls))
        for call, (result, is_error) in zip(calls, results, strict=True):
            on_event(ToolEnd(call, result, is_error))
            push(ToolResultMessage(call.id, call.name, result, is_error))

    reason = "the model finished"
    continuations = 0

    for _ in range(max_turns):
        assistant: AssistantMessage | None = None
        for attempt in range(stream_attempts):
            try:
                assistant = await _one_turn(
                    provider, model, messages, tools, system, stream_timeout, on_event
                )
                break
            except StreamError:
                if attempt == stream_attempts - 1:
                    raise
                await asyncio.sleep(min(2**attempt, 8))

        assert assistant is not None
        push(assistant)
        on_event(TurnEnd(assistant))

        if assistant.stop_reason == "toolUse":
            continuations = 0
            await run_calls(calls_of(assistant))
            continue

        if assistant.stop_reason == "length":
            # a truncated answer: still finish the tool calls it completed in full,
            # then ask for the rest instead of ending the run mid-sentence
            calls = calls_of(assistant)
            await run_calls([c for c in calls if not c.error])
            for call in calls:
                if call.error:
                    push(ToolResultMessage(call.id, call.name, call.error, True))
            if continuations >= max_continuations:
                reason = (
                    f"the model's answer was still being cut off after "
                    f"{max_continuations} continuations (raise --max-tokens)"
                )
                break
            continuations += 1
            push(UserMessage(CONTINUE))
            continue

        break

    else:
        reason = f"reached the turn limit ({max_turns}) before the model finished"

    on_event(RunEnd(reason))

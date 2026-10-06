from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

StopReason = Literal["stop", "length", "toolUse"]


@dataclass
class Usage:
    input: int = 0
    output: int = 0


@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolCallBlock:
    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["toolCall"] = "toolCall"
    error: str | None = None  # set when streamed arguments could not be parsed


ContentBlock = TextBlock | ToolCallBlock


@dataclass
class UserMessage:
    content: str
    role: Literal["user"] = "user"


@dataclass
class AssistantMessage:
    content: list[ContentBlock]
    usage: Usage
    stop_reason: StopReason
    role: Literal["assistant"] = "assistant"


@dataclass
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    role: Literal["toolResult"] = "toolResult"


Message = UserMessage | AssistantMessage | ToolResultMessage


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema
    execute: Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class TextDelta:
    delta: str


@dataclass
class Done:
    message: AssistantMessage


StreamEvent = TextDelta | Done


class Provider(Protocol):
    name: str
    default_model: str

    def stream(
        self,
        *,
        messages: list[Message],
        model: str,
        system: str | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]: ...

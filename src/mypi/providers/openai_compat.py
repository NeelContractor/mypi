from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from ..types import (
    AssistantMessage,
    ContentBlock,
    Done,
    Message,
    StopReason,
    StreamEvent,
    TextBlock,
    TextDelta,
    Tool,
    ToolCallBlock,
    Usage,
)

MAX_RETRIES = 3


def to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            text = "".join(b.text for b in m.content if isinstance(b, TextBlock))
            calls = [b for b in m.content if isinstance(b, ToolCallBlock)]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                # OpenAI wants the arguments as a JSON *string*
                msg["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in calls
                ]
            out.append(msg)
        else:  # toolResult -> its own message with role "tool"
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out


class OpenAICompatProvider:
    """One adapter for every OpenAI-compatible API: only base_url, key and model change."""

    def __init__(
        self,
        name: str,
        base_url: str,
        env_var: str,
        default_model: str,
        max_tokens: int = 8192,
    ) -> None:
        key = os.environ.get(env_var)
        if not key:
            raise ValueError(
                f"{env_var} is not set (needed by --provider {name}); put it in .env or export it"
            )
        self.name = name
        self.default_model = default_model
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(base_url=base_url, api_key=key, max_retries=MAX_RETRIES)

    async def aclose(self) -> None:
        await self.client.close()

    async def stream(
        self,
        *,
        messages: list[Message],
        model: str,
        system: str | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        chat = to_openai(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},  # usage arrives in the last chunk
            "messages": ([{"role": "system", "content": system}] if system else []) + chat,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        want_usage = True
        try:
            stream = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # older OpenAI-compatible servers (some Groq/vLLM/Ollama builds) reject
            # stream_options outright; drop it rather than fail the whole run
            if "stream_options" not in str(e) and "include_usage" not in str(e):
                raise
            kwargs.pop("stream_options")
            want_usage = False
            stream = await self.client.chat.completions.create(**kwargs)

        text = ""
        calls: dict[int, dict[str, str]] = {}  # slot = tool_calls[].index
        usage = Usage()
        stop_reason: StopReason = "stop"

        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice and choice.delta:
                if choice.delta.content:
                    text += choice.delta.content
                    yield TextDelta(choice.delta.content)
                for tc in choice.delta.tool_calls or []:
                    # first piece brings id + name, later pieces only bring more arguments
                    slot = calls.setdefault(
                        tc.index, {"id": tc.id or f"call_{tc.index}", "name": "", "args": ""}
                    )
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args"] += tc.function.arguments
            if choice and choice.finish_reason == "tool_calls":
                stop_reason = "toolUse"
            elif choice and choice.finish_reason == "length":
                stop_reason = "length"
            if want_usage and chunk.usage:
                usage = Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)

        content: list[ContentBlock] = [TextBlock(text)] if text else []
        for i in sorted(calls):
            c = calls[i]
            arguments: dict[str, Any] = {}
            error: str | None = None
            if c["args"]:
                try:
                    arguments = json.loads(c["args"])
                except json.JSONDecodeError as e:
                    # a truncated stream leaves half an argument object behind
                    error = f"invalid JSON arguments: {e}"
            content.append(ToolCallBlock(c["id"], c["name"], arguments, error=error))
        if any(isinstance(b, ToolCallBlock) for b in content):
            stop_reason = "toolUse"
        elif stop_reason == "toolUse":
            # finish_reason said tool_calls but nothing arrived: don't hand the model
            # an empty turn and call it a tool call
            stop_reason = "stop"

        yield Done(AssistantMessage(content=content, usage=usage, stop_reason=stop_reason))

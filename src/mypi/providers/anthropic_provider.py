from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from typing import Any

from anthropic import AsyncAnthropic

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


def to_anthropic(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            blocks = [
                {"type": "text", "text": b.text}
                if isinstance(b, TextBlock)
                else {"type": "tool_use", "id": b.id, "name": b.name, "input": b.arguments}
                for b in m.content
            ]
            out.append({"role": "assistant", "content": blocks})
        else:  # toolResult -> Anthropic wants them inside a USER message
            results = [
                {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                    "is_error": m.is_error,
                }
            ]
            # every result of one turn shares a single user message: consecutive
            # user turns are rejected (roles must alternate)
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].extend(results)
            else:
                out.append({"role": "user", "content": results})
    return out


class AnthropicProvider:
    name = "anthropic"
    default_model = "claude-sonnet-5-5"

    def __init__(self, max_tokens: int = 8192) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(
                "ANTHROPIC_API_KEY is not set; put it in .env or export it "
                "(or pick another --provider)"
            )
        self.max_tokens = max_tokens
        self.client = AsyncAnthropic(max_retries=MAX_RETRIES)

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
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": to_anthropic(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]

        blocks: dict[int, ContentBlock] = {}
        json_bufs: dict[int, str] = {}
        closed: set[int] = set()

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    b = event.content_block
                    if b.type == "text":
                        blocks[event.index] = TextBlock("")
                    elif b.type == "tool_use":
                        blocks[event.index] = ToolCallBlock(b.id, b.name, {})
                        json_bufs[event.index] = ""
                elif event.type == "content_block_delta":
                    blk = blocks.get(event.index)
                    d = event.delta
                    if d.type == "text_delta" and isinstance(blk, TextBlock):
                        blk.text += d.text
                        yield TextDelta(d.text)
                    elif d.type == "input_json_delta":
                        json_bufs[event.index] = json_bufs.get(event.index, "") + d.partial_json
                elif event.type == "content_block_stop":
                    closed.add(event.index)
                    blk = blocks.get(event.index)
                    if isinstance(blk, ToolCallBlock):
                        buf = json_bufs.get(event.index, "")
                        if not buf:
                            continue  # a call that genuinely takes no arguments
                        try:
                            blk.arguments = json.loads(buf)
                        except json.JSONDecodeError as e:
                            # max_tokens cut the arguments off mid-JSON: report it
                            # back to the model instead of killing the run
                            blk.error = f"invalid JSON arguments: {e}"
            final = await stream.get_final_message()

        for index, blk in blocks.items():
            # a tool block that never closed means the turn was cut off; leaving the
            # arguments empty would run the tool with a lie
            if isinstance(blk, ToolCallBlock) and blk.error is None and index not in closed:
                blk.error = "interrupted: the model never finished sending this call's arguments"

        stop_reason: StopReason = (
            "toolUse"
            if final.stop_reason == "tool_use"
            else "length"
            if final.stop_reason == "max_tokens"
            else "stop"
        )
        yield Done(
            AssistantMessage(
                content=[blocks[i] for i in sorted(blocks)],
                usage=Usage(final.usage.input_tokens, final.usage.output_tokens),
                stop_reason=stop_reason,
            )
        )

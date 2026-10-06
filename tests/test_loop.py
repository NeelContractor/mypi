import asyncio

import pytest

from mypi.agent.loop import MessageEvent, RunEnd, StreamError, ToolEnd, ToolStart, run_agent
from mypi.types import (
    AssistantMessage,
    Done,
    TextBlock,
    Tool,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)


class ScriptedProvider:
    """Replays a fixed list of content-block lists, one per turn."""

    name, default_model = "scripted", "m"

    def __init__(self, turns):
        # a turn is a content list, or (content list, explicit stop reason)
        self.turns = [(t, None) if isinstance(t, list) else t for t in turns]
        self.seen_system = None

    async def stream(self, *, messages, model, system=None, tools=None):
        self.seen_system = system
        content, reason = self.turns.pop(0)
        if reason is None:
            reason = "toolUse" if any(isinstance(b, ToolCallBlock) for b in content) else "stop"
        yield Done(AssistantMessage(content, Usage(1, 1), reason))


def tool(name: str, delay: float = 0.0, log: list | None = None):
    async def execute(args):
        if delay:
            await asyncio.sleep(delay)
        if log is not None:
            log.append((name, asyncio.get_running_loop().time()))
        return f"{name} ok"

    return Tool(name=name, description=name, parameters={"type": "object"}, execute=execute)


def call(*names):
    return [ToolCallBlock(f"c{i}", n, {}) for i, n in enumerate(names)]


def text(value: str):
    return [TextBlock(value)]


def cut(value: str):
    """A turn the token limit cut off."""
    return ([TextBlock(value)], "length")


async def test_tool_calls_run_concurrently() -> None:
    log: list = []
    tools = [tool("a", 0.15, log), tool("b", 0.15, log), tool("c", 0.15, log)]
    start = asyncio.get_running_loop().time()
    events: list = []
    await run_agent(
        provider=ScriptedProvider([call("a", "b", "c"), text("done")]),
        model="m",
        tools=tools,
        messages=[UserMessage("go")],
        on_event=events.append,
    )
    elapsed = asyncio.get_running_loop().time() - start
    assert len(log) == 3
    assert elapsed < 0.35, f"3 x 0.15s tools took {elapsed:.2f}s, they ran sequentially"
    assert [m.tool_name for m in _results(events)] == ["a", "b", "c"]


async def test_results_keep_call_order_not_finish_order() -> None:
    log: list = []
    tools = [tool("slow", 0.2, log), tool("fast", 0.0, log)]
    events: list = []
    await run_agent(
        provider=ScriptedProvider([call("slow", "fast"), text("done")]),
        model="m",
        tools=tools,
        messages=[UserMessage("go")],
        on_event=events.append,
    )
    assert [m.tool_name for m in _results(events)] == ["slow", "fast"]
    assert [m.content for m in _results(events)] == ["slow ok", "fast ok"]


async def test_unknown_tool_is_reported_to_the_model() -> None:
    events: list = []
    await run_agent(
        provider=ScriptedProvider([call("nope"), text("done")]),
        model="m",
        tools=[tool("real")],
        messages=[UserMessage("go")],
        on_event=events.append,
    )
    result = _results(events)[0]
    assert result.is_error and "unknown tool name nope" in result.content


async def test_tool_exception_is_reported_to_the_model() -> None:
    async def boom(args):
        raise RuntimeError("kaboom")

    events: list = []
    await run_agent(
        provider=ScriptedProvider([call("bad"), text("done")]),
        model="m",
        tools=[Tool(name="bad", description="", parameters={}, execute=boom)],
        messages=[UserMessage("go")],
        on_event=events.append,
    )
    result = _results(events)[0]
    assert result.is_error and "kaboom" in result.content


async def test_unparsable_arguments_never_reach_the_tool() -> None:
    seen: list = []
    events: list = []
    bad = ToolCallBlock("c0", "read", {}, error="invalid JSON arguments: boom")
    await run_agent(
        provider=ScriptedProvider([[bad], text("done")]),
        model="m",
        tools=[tool("read", log=seen)],
        messages=[UserMessage("go")],
        on_event=events.append,
    )
    assert seen == [], "the tool ran despite unparsable arguments"
    result = _results(events)[0]
    assert result.is_error and "invalid JSON arguments" in result.content


async def test_one_bad_call_does_not_block_the_others() -> None:
    events: list = []
    bad = ToolCallBlock("c0", "gone", {}, error="invalid JSON arguments: boom")
    await run_agent(
        provider=ScriptedProvider([[bad, *call("real")], text("done")]),
        model="m",
        tools=[tool("real")],
        messages=[UserMessage("go")],
        on_event=events.append,
    )
    assert [m.is_error for m in _results(events)] == [True, False]


async def test_event_order() -> None:
    events: list = []
    await run_agent(
        provider=ScriptedProvider([call("real"), text("done")]),
        model="m",
        tools=[tool("real")],
        messages=[UserMessage("go")],
        on_event=events.append,
    )
    kinds = [type(e).__name__ for e in events]
    assert kinds == [
        "MessageEvent",  # the assistant turn is recorded
        "TurnEnd",
        "ToolStart",
        "ToolEnd",
        "MessageEvent",  # the tool result is recorded
        "MessageEvent",  # the final assistant turn
        "TurnEnd",
        "RunEnd",
    ]
    assert isinstance(events[0], MessageEvent)
    assert isinstance(events[2], ToolStart)
    assert isinstance(events[3], ToolEnd)


async def test_conversation_grows_across_turns() -> None:
    messages: list = [UserMessage("go")]
    await run_agent(
        provider=ScriptedProvider([call("real"), call("real"), text("done")]),
        model="m",
        tools=[tool("real")],
        messages=messages,
        on_event=lambda e: None,
    )
    roles = [m.role for m in messages]
    assert roles == ["user", "assistant", "toolResult", "assistant", "toolResult", "assistant"]


def _results(events) -> list[ToolResultMessage]:
    return [
        e.message
        for e in events
        if isinstance(e, MessageEvent) and isinstance(e.message, ToolResultMessage)
    ]


# --- end conditions: every exit path reports why (issues 5, 6, 17, 18) ---


async def test_normal_finish_says_so() -> None:
    events: list = []
    await run_agent(
        provider=ScriptedProvider([text("hi")]),
        model="m",
        tools=[],
        messages=[UserMessage("go")],
        on_event=events.append,
    )
    assert [e.reason for e in events if isinstance(e, RunEnd)] == ["the model finished"]


async def test_token_limit_asks_the_model_to_continue() -> None:
    provider = ScriptedProvider([cut("half a sen"), cut("tence"), text("done")])
    messages: list = [UserMessage("go")]
    events: list = []
    await run_agent(
        provider=provider,
        model="m",
        tools=[],
        messages=messages,
        on_event=events.append,
    )
    assert [e.reason for e in events if isinstance(e, RunEnd)] == ["the model finished"]
    continuations = [m for m in messages if isinstance(m, UserMessage)][1:]
    assert len(continuations) == 2  # one per cut-off turn
    assert "cut off" in continuations[0].content


async def test_token_limit_still_runs_the_tools_it_did_ask_for() -> None:
    log: list = []
    provider = ScriptedProvider([([*call("real"), TextBlock("and now")], "length"), text("done")])
    events: list = []
    await run_agent(
        provider=provider,
        model="m",
        tools=[tool("real", log=log)],
        messages=[UserMessage("go")],
        on_event=events.append,
    )
    assert [n for n, _ in log] == ["real"], "a truncated turn threw away a complete tool call"
    assert any(isinstance(e, ToolEnd) for e in events)


async def test_repeated_truncation_gives_up_with_a_reason() -> None:
    events: list = []
    await run_agent(
        provider=ScriptedProvider([cut("a")] * 20),
        model="m",
        tools=[],
        messages=[UserMessage("go")],
        on_event=events.append,
        max_continuations=2,
    )
    reason = [e.reason for e in events if isinstance(e, RunEnd)][0]
    assert "still being cut off" in reason and "--max-tokens" in reason


async def test_turn_limit_gives_up_with_a_reason() -> None:
    events: list = []
    await run_agent(
        provider=ScriptedProvider([call("real")] * 20),
        model="m",
        tools=[tool("real")],
        messages=[UserMessage("go")],
        on_event=events.append,
        max_turns=3,
    )
    reason = [e.reason for e in events if isinstance(e, RunEnd)][0]
    assert "turn limit (3)" in reason


async def test_stream_that_dies_is_retried() -> None:
    attempts = 0

    class Flaky:
        name, default_model = "flaky", "m"

        async def stream(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return
                yield
            yield Done(AssistantMessage([TextBlock("recovered")], Usage(1, 1), "stop"))

    events: list = []
    await run_agent(
        provider=Flaky(),
        model="m",
        tools=[],
        messages=[UserMessage("go")],
        on_event=events.append,
        stream_attempts=3,
    )
    assert attempts == 3
    assert [e.reason for e in events if isinstance(e, RunEnd)] == ["the model finished"]


async def test_stream_that_keeps_dying_raises() -> None:
    class Dead:
        name, default_model = "dead", "m"

        async def stream(self, **kwargs):
            return
            yield

    with pytest.raises(StreamError, match="connection closed"):
        await run_agent(
            provider=Dead(),
            model="m",
            tools=[],
            messages=[UserMessage("go")],
            on_event=lambda e: None,
            stream_attempts=2,
        )


async def test_stream_that_stalls_times_out() -> None:
    class Hang:
        name, default_model = "hang", "m"

        async def stream(self, **kwargs):
            await asyncio.sleep(30)
            yield

    with pytest.raises(StreamError, match="stopped sending tokens"):
        await run_agent(
            provider=Hang(),
            model="m",
            tools=[],
            messages=[UserMessage("go")],
            on_event=lambda e: None,
            stream_timeout=0.2,
            stream_attempts=1,
        )


async def test_system_prompt_reaches_the_provider() -> None:
    provider = ScriptedProvider([text("hi")])
    await run_agent(
        provider=provider,
        model="m",
        tools=[],
        messages=[UserMessage("go")],
        on_event=lambda e: None,
        system="be helpful",
    )
    assert provider.seen_system == "be helpful"

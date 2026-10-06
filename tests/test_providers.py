import json
from itertools import pairwise

import pytest

from mypi.providers.anthropic_provider import to_anthropic
from mypi.providers.openai_compat import to_openai
from mypi.types import (
    AssistantMessage,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def _turn() -> list:
    return [
        UserMessage("hi"),
        AssistantMessage(
            [
                TextBlock("looking"),
                ToolCallBlock("t1", "read", {"path": "a"}),
                ToolCallBlock("t2", "ls", {}),
            ],
            Usage(5, 6),
            "toolUse",
        ),
        ToolResultMessage("t1", "read", "AAA"),
        ToolResultMessage("t2", "ls", "BBB"),
        AssistantMessage([TextBlock("done")], Usage(9, 3), "stop"),
    ]


def test_anthropic_roles_alternate() -> None:
    roles = [m["role"] for m in to_anthropic(_turn())]
    assert roles == ["user", "assistant", "user", "assistant"]
    for a, b in pairwise(roles):
        assert a != b, f"consecutive {a} turns are rejected by the API"


def test_anthropic_batches_tool_results_into_one_user_turn() -> None:
    out = to_anthropic(_turn())
    assert len(out[2]["content"]) == 2
    assert [c["tool_use_id"] for c in out[2]["content"]] == ["t1", "t2"]


def test_anthropic_keeps_error_flag() -> None:
    msgs = [
        UserMessage("hi"),
        AssistantMessage([ToolCallBlock("t1", "read", {})], Usage(1, 1), "toolUse"),
        ToolResultMessage("t1", "read", "boom", True),
    ]
    assert to_anthropic(msgs)[2]["content"][0]["is_error"] is True


def test_openai_shape() -> None:
    out = to_openai(_turn())
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "tool", "assistant"]
    calls = out[1]["tool_calls"]
    assert calls[0]["function"]["name"] == "read"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a"}
    assert out[2]["tool_call_id"] == "t1"


def test_openai_null_content_when_only_tool_calls() -> None:
    msgs = [AssistantMessage([ToolCallBlock("t1", "read", {})], Usage(1, 1), "toolUse")]
    assert to_openai(msgs)[0]["content"] is None


def test_tool_call_block_carries_parse_error() -> None:
    call = ToolCallBlock("t1", "read", {}, error="invalid JSON arguments: boom")
    assert call.error == "invalid JSON arguments: boom"
    assert call.arguments == {}


# --- a tool block cut off mid-JSON must not kill the run (issue 1) ---


class _Delta:
    def __init__(self, type, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _Event:
    def __init__(self, type, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _Stream:
    def __init__(self, events, final):
        self.events = events
        self.final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def gen():
            for e in self.events:
                yield e

        return gen()

    async def get_final_message(self):
        return self.final


def _anthropic_provider(events, final):
    from mypi.providers.anthropic_provider import AnthropicProvider

    p = AnthropicProvider.__new__(AnthropicProvider)  # skip __init__, no API key needed
    p.name, p.default_model = "anthropic", "m"
    p.max_tokens = 4096
    p.client = type(
        "C",
        (),
        {
            "messages": type(
                "M", (), {"stream": staticmethod(lambda **kw: _Stream(events, final))}
            )()
        },
    )()
    return p


async def _drain(provider):
    return [e async for e in provider.stream(messages=[UserMessage("hi")], model="m", tools=[])]


async def test_anthropic_truncated_tool_arguments_become_an_error() -> None:
    events = [
        _Event(
            "content_block_start",
            index=0,
            content_block=type("B", (), {"type": "tool_use", "id": "t1", "name": "read"})(),
        ),
        _Event(
            "content_block_delta",
            index=0,
            delta=_Delta("input_json_delta", partial_json='{"path": "a'),
        ),
        _Event("content_block_stop", index=0),
    ]
    final = type(
        "F",
        (),
        {
            "stop_reason": "tool_use",
            "usage": type("U", (), {"input_tokens": 1, "output_tokens": 2})(),
        },
    )()
    out = await _drain(_anthropic_provider(events, final))
    call = out[-1].message.content[0]
    assert call.arguments == {}
    assert call.error is not None and "invalid JSON arguments" in call.error


async def test_anthropic_valid_tool_arguments_still_parse() -> None:
    events = [
        _Event(
            "content_block_start",
            index=0,
            content_block=type("B", (), {"type": "tool_use", "id": "t1", "name": "read"})(),
        ),
        _Event(
            "content_block_delta",
            index=0,
            delta=_Delta("input_json_delta", partial_json='{"path": '),
        ),
        _Event(
            "content_block_delta", index=0, delta=_Delta("input_json_delta", partial_json='"a"}')
        ),
        _Event("content_block_stop", index=0),
    ]
    final = type(
        "F",
        (),
        {
            "stop_reason": "tool_use",
            "usage": type("U", (), {"input_tokens": 1, "output_tokens": 2})(),
        },
    )()
    out = await _drain(_anthropic_provider(events, final))
    call = out[-1].message.content[0]
    assert call.arguments == {"path": "a"}
    assert call.error is None


async def test_anthropic_never_closed_block_reports_interrupted() -> None:
    events = [
        _Event(
            "content_block_start",
            index=0,
            content_block=type("B", (), {"type": "tool_use", "id": "t1", "name": "read"})(),
        ),
    ]
    final = type(
        "F",
        (),
        {
            "stop_reason": "max_tokens",
            "usage": type("U", (), {"input_tokens": 1, "output_tokens": 2})(),
        },
    )()
    out = await _drain(_anthropic_provider(events, final))
    call = out[-1].message.content[0]
    assert call.arguments == {}
    assert call.error is not None and "interrupted" in call.error
    assert out[-1].message.stop_reason == "length"


async def test_anthropic_closed_block_with_no_arguments_is_not_an_error() -> None:
    """ls takes no arguments, so an empty argument object is legitimate."""
    events = [
        _Event(
            "content_block_start",
            index=0,
            content_block=type("B", (), {"type": "tool_use", "id": "t1", "name": "ls"})(),
        ),
        _Event("content_block_stop", index=0),
    ]
    final = type(
        "F",
        (),
        {
            "stop_reason": "tool_use",
            "usage": type("U", (), {"input_tokens": 1, "output_tokens": 2})(),
        },
    )()
    out = await _drain(_anthropic_provider(events, final))
    call = out[-1].message.content[0]
    assert call.arguments == {}
    assert call.error is None


async def test_openai_truncated_tool_arguments_become_an_error() -> None:
    from mypi.providers.openai_compat import OpenAICompatProvider

    def chunk(args=None, usage=None):
        tc = type(
            "TC",
            (),
            {
                "index": 0,
                "id": "t1",
                "function": type("F", (), {"name": "read" if args else "", "arguments": args})(),
            },
        )()
        choice = type(
            "C",
            (),
            {
                "delta": type("D", (), {"content": None, "tool_calls": [tc]})(),
                "finish_reason": "tool_calls",
            },
        )()
        return type("Ch", (), {"choices": [choice], "usage": usage})()

    async def create(**kwargs):
        async def gen():
            yield chunk('{"path": "a')
            yield chunk()

        return gen()

    p = OpenAICompatProvider.__new__(OpenAICompatProvider)
    p.name, p.default_model, p.max_tokens = "fake", "m", 4096
    p.client = type(
        "C",
        (),
        {
            "chat": type(
                "Ch", (), {"completions": type("Cm", (), {"create": staticmethod(create)})}
            )()
        },
    )()
    out = [e async for e in p.stream(messages=[UserMessage("hi")], model="m", tools=[])]
    call = out[-1].message.content[0]
    assert call.arguments == {}
    assert call.error is not None and "invalid JSON arguments" in call.error
    assert out[-1].message.stop_reason == "toolUse"


# --- provider wiring (issues 13, 14, 19, 35) ---


def test_missing_key_names_the_variable(monkeypatch) -> None:
    from mypi.providers.openai_compat import OpenAICompatProvider

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        OpenAICompatProvider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "m")


def test_missing_anthropic_key_names_the_variable(monkeypatch) -> None:
    from mypi.providers.anthropic_provider import AnthropicProvider

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        AnthropicProvider()


def test_get_provider_rejects_unknown_names() -> None:
    from mypi.providers import PROVIDERS, get_provider

    assert set(PROVIDERS) == {"anthropic", "anthropic-openai", "groq"}
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("nope")


async def test_max_tokens_is_sent(monkeypatch) -> None:
    from mypi.providers.openai_compat import OpenAICompatProvider

    seen: dict = {}

    async def create(**kwargs):
        seen.update(kwargs)

        async def gen():
            return
            yield

        return gen()

    p = OpenAICompatProvider.__new__(OpenAICompatProvider)
    p.name, p.default_model, p.max_tokens = "fake", "m", 1234
    p.client = type(
        "C",
        (),
        {
            "chat": type(
                "Ch", (), {"completions": type("Cm", (), {"create": staticmethod(create)})}
            )()
        },
    )()
    [e async for e in p.stream(messages=[UserMessage("hi")], model="m", tools=[])]
    assert seen["max_tokens"] == 1234


async def test_stream_options_falls_back_when_rejected() -> None:
    """Older OpenAI-compatible servers 400 on stream_options; drop it and carry on."""
    from mypi.providers.openai_compat import OpenAICompatProvider

    calls: list[dict] = []

    async def create(**kwargs):
        calls.append(kwargs)
        if "stream_options" in kwargs:
            raise RuntimeError("Unrecognized request argument supplied: stream_options")
        chunk = type("Ch", (), {"choices": [], "usage": None})()

        async def gen():
            yield chunk

        return gen()

    p = OpenAICompatProvider.__new__(OpenAICompatProvider)
    p.name, p.default_model, p.max_tokens = "fake", "m", 512
    p.client = type(
        "C",
        (),
        {
            "chat": type(
                "Ch", (), {"completions": type("Cm", (), {"create": staticmethod(create)})}
            )()
        },
    )()
    out = [e async for e in p.stream(messages=[UserMessage("hi")], model="m", tools=[])]
    assert len(calls) == 2
    assert "stream_options" in calls[0] and "stream_options" not in calls[1]
    assert out[-1].message.stop_reason == "stop"


async def test_other_errors_are_not_swallowed() -> None:
    from mypi.providers.openai_compat import OpenAICompatProvider

    async def create(**kwargs):
        raise RuntimeError("401 Unauthorized")

    p = OpenAICompatProvider.__new__(OpenAICompatProvider)
    p.name, p.default_model, p.max_tokens = "fake", "m", 512
    p.client = type(
        "C",
        (),
        {
            "chat": type(
                "Ch", (), {"completions": type("Cm", (), {"create": staticmethod(create)})}
            )()
        },
    )()
    with pytest.raises(RuntimeError, match="401"):
        [e async for e in p.stream(messages=[UserMessage("hi")], model="m", tools=[])]


async def test_finish_reason_tool_calls_with_nothing_arrived_is_not_a_tool_turn() -> None:
    from mypi.providers.openai_compat import OpenAICompatProvider

    chunk = type(
        "Ch",
        (),
        {
            "choices": [
                type(
                    "C",
                    (),
                    {
                        "delta": type("D", (), {"content": "hi", "tool_calls": None})(),
                        "finish_reason": "tool_calls",
                    },
                )()
            ],
            "usage": None,
        },
    )()

    async def create(**kwargs):
        async def gen():
            yield chunk

        return gen()

    p = OpenAICompatProvider.__new__(OpenAICompatProvider)
    p.name, p.default_model, p.max_tokens = "fake", "m", 512
    p.client = type(
        "C",
        (),
        {
            "chat": type(
                "Ch", (), {"completions": type("Cm", (), {"create": staticmethod(create)})}
            )()
        },
    )()
    out = [e async for e in p.stream(messages=[UserMessage("hi")], model="m", tools=[])]
    assert out[-1].message.stop_reason == "stop"

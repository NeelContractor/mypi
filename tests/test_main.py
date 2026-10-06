"""The CLI is the documentation; keep it honest."""

import pytest

from mypi.main import build_parser, parse_args


@pytest.fixture(autouse=True)
def _no_dotenv_leaks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own .env must not change what these tests assert."""
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)


def test_defaults_match_the_readme() -> None:
    args = parse_args(["-p", "do it"])
    assert args.provider == "anthropic"
    assert args.max_tokens == 8192
    assert args.max_turns == 20
    assert args.max_continuations == 3
    assert args.stream_timeout == 300.0
    assert args.model is None
    assert args.system is None
    assert args.no_system is False


def test_every_option_is_reachable_and_overridable() -> None:
    args = parse_args(
        [
            "-p",
            "do it",
            "--provider",
            "groq",
            "--model",
            "some-model",
            "--system",
            "be terse",
            "--max-tokens",
            "64",
            "--max-turns",
            "2",
            "--max-continuations",
            "0",
            "--stream-timeout",
            "1.5",
            "--debug",
        ]
    )
    assert (args.provider, args.model, args.system) == ("groq", "some-model", "be terse")
    assert (args.max_tokens, args.max_turns, args.max_continuations) == (64, 2, 0)
    assert (args.stream_timeout, args.debug) == (1.5, True)
    assert parse_args(["-p", "x", "--no-system"]).no_system is True


async def test_no_prompt_is_a_usage_error_not_a_traceback() -> None:
    """amain checks this itself so it can print a friendlier line than argparse."""
    from mypi.main import amain

    assert parse_args([]).prompt is None
    assert await amain([]) == 2
    assert await amain(["--provider", "groq"]) == 2


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["-p", "x", "--provider", "openai"])


def test_model_and_provider_can_come_from_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of .env: pick a model without reading the source."""
    monkeypatch.setenv("MODEL_PROVIDER", "groq")
    monkeypatch.setenv("MODEL_NAME", "some-model")

    args = parse_args(["-p", "do it"])

    assert (args.provider, args.model) == ("groq", "some-model")


def test_the_command_line_beats_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "groq")
    monkeypatch.setenv("MODEL_NAME", "from-dotenv")

    args = parse_args(["-p", "do it", "--provider", "anthropic", "--model", "from-flag"])

    assert (args.provider, args.model) == ("anthropic", "from-flag")


async def test_a_bad_model_provider_in_dotenv_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """argparse skips `choices` for defaults, so get_provider has to catch it."""
    from mypi.main import amain

    monkeypatch.setenv("MODEL_PROVIDER", "openai")

    assert await amain(["-p", "do it"]) == 2
    assert "unknown provider 'openai', choose from: anthropic" in capsys.readouterr().err


def test_every_option_has_help_text() -> None:
    """A bare MAX_TOKENS in --help tells the reader nothing."""
    parser_actions = {a.dest: a for a in build_parser()._actions if a.dest != "help"}
    for dest in (
        "provider",
        "model",
        "system",
        "max_tokens",
        "max_turns",
        "max_continuations",
        "stream_timeout",
        "debug",
        "prompt",
    ):
        assert parser_actions[dest].help, f"--{dest.replace('_', '-')} has no help text"


# --- the rendered CLI output, which is the only thing a user actually sees ---


def _render(*events) -> str:
    import contextlib
    import io

    import mypi.main as m

    buf = io.StringIO()
    on = m.render("groq", "model-x")
    with contextlib.redirect_stdout(buf):
        for event in events:
            on(event)
    return buf.getvalue()


def _call(name: str = "write"):
    from mypi.types import ToolCallBlock

    return ToolCallBlock(id="1", name=name, arguments={})


def test_single_line_result_is_not_turned_into_a_line_count() -> None:
    from mypi.agent.loop import ToolEnd

    out = _render(ToolEnd(_call(), "created /tmp/x.py (29 bytes, 3 lines)", False))
    assert out == " write: created /tmp/x.py (29 bytes, 3 lines)\n"


def test_multi_line_result_is_counted_and_indented() -> None:
    from mypi.agent.loop import ToolEnd

    out = _render(ToolEnd(_call("read"), "a\nb\nc\n", False))
    assert out == " read: 3 lines\n  a\n  b\n  c\n", repr(out)


def test_empty_result_says_so() -> None:
    from mypi.agent.loop import ToolEnd

    assert "write: (no output)" in _render(ToolEnd(_call(), "", False))


def test_single_line_error_names_the_call() -> None:
    from mypi.agent.loop import ToolEnd

    out = _render(ToolEnd(_call("edit"), "old_string was not found in the file", True))
    assert out.startswith(" edit failed: old_string was not found")


def test_first_output_is_not_padded_with_blank_lines() -> None:
    from mypi.agent.loop import TurnEnd
    from mypi.types import AssistantMessage, Usage

    out = _render(TurnEnd(AssistantMessage([], Usage(1013, 42), "stop")))
    assert out.startswith("  groq"), repr(out[:20])
    assert "1013 in / 42 out" in out


def test_a_quiet_finish_prints_nothing_but_a_reason_does() -> None:
    from mypi.agent.loop import RunEnd

    assert _render(RunEnd("the model finished")) == ""
    assert _render(RunEnd("reached the turn limit (3) before the model finished")) == (
        "  stopped: reached the turn limit (3) before the model finished\n"
    )


def test_streamed_text_is_not_repadded() -> None:
    from mypi.agent.loop import TextEvent

    assert _render(TextEvent("hello"), TextEvent(" world")) == "hello world"

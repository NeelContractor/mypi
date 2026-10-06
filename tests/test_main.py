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
    assert args.expand is False


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
            "--expand",
        ]
    )
    assert (args.provider, args.model, args.system) == ("groq", "some-model", "be terse")
    assert (args.max_tokens, args.max_turns, args.max_continuations) == (64, 2, 0)
    assert (args.stream_timeout, args.debug, args.expand) == (1.5, True, True)
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
        "expand",
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
    on = m.render("groq", "model-x", expand=True)
    with contextlib.redirect_stdout(buf):
        for event in events:
            on(event)
    return buf.getvalue()


def _call(name: str = "write", arguments: dict | None = None):
    from mypi.types import ToolCallBlock

    return ToolCallBlock(id="1", name=name, arguments=arguments or {})


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


def test_turn_summary_is_human_readable() -> None:
    from mypi.agent.loop import TurnEnd
    from mypi.types import AssistantMessage, Usage

    out = _render(TurnEnd(AssistantMessage([], Usage(3712, 439), "stop")))
    assert out == "  groq · model-x · 3712 in / 439 out tokens · finished\n", repr(out)


def test_turn_summary_translates_every_stop_reason() -> None:
    from mypi.agent.loop import TurnEnd
    from mypi.types import AssistantMessage, Usage

    def render(reason: str) -> str:
        return _render(TurnEnd(AssistantMessage([], Usage(1, 1), reason)))

    assert render("stop") == "  groq · model-x · 1 in / 1 out tokens · finished\n"
    assert "· tool call" in render("toolUse")
    assert "· hit the token budget" in render("length")


def test_a_quiet_finish_prints_nothing_but_a_reason_does() -> None:
    from mypi.agent.loop import RunEnd

    assert _render(RunEnd("the model finished")) == ""
    assert _render(RunEnd("reached the turn limit (3) before the model finished")) == (
        "  stopped: reached the turn limit (3) before the model finished\n"
    )


def test_streamed_text_is_not_repadded() -> None:
    from mypi.agent.loop import TextEvent

    assert _render(TextEvent("hello"), TextEvent(" world")) == "hello world"


def test_the_answer_is_its_own_paragraph() -> None:
    """The answer used to start on the line under the last tool result."""
    from mypi.agent.loop import TextEvent, ToolEnd

    out = _render(ToolEnd(_call(), "created /x", False), TextEvent("the answer"))
    assert out.endswith("\n\nthe answer"), repr(out)


def test_a_tool_call_after_the_answer_gets_a_blank_line() -> None:
    from mypi.agent.loop import TextEvent, ToolStart

    out = _render(TextEvent("intro"), ToolStart(_call("ls")))
    assert out == "intro\n\n ls {}\n", repr(out)


def test_sections_never_touch_and_never_gap_twice() -> None:
    from mypi.agent.loop import TextEvent, ToolEnd, TurnEnd
    from mypi.types import AssistantMessage, Usage

    out = _render(
        ToolEnd(_call(), "one line", False),
        TextEvent("the answer"),
        TurnEnd(AssistantMessage([], Usage(1, 1), "stop")),
        ToolEnd(_call(), "another", False),
    )
    assert "\n\n\n" not in out, repr(out)
    assert out.count("\n\n") == 3, repr(out)


def _render_coloured(*events) -> str:
    """Same as _render, but stdout pretends to be a terminal so colours are on."""
    import contextlib
    import io

    import mypi.main as m

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    out = Tty()
    with contextlib.redirect_stdout(out):
        on = m.render("groq", "model-x", expand=True)
        for event in events:
            on(event)
    return out.getvalue()


def test_plain_rendering_never_leaks_escape_codes() -> None:
    from mypi.agent.loop import TextEvent, ToolEnd

    out = _render(ToolEnd(_call(), "created /x", False), TextEvent("answer"))
    assert "\x1b[" not in out, repr(out)


def test_tool_calls_are_painted_in_the_tool_colour() -> None:
    from mypi.agent.loop import ToolStart

    out = _render_coloured(ToolStart(_call("ls")))
    assert "\x1b[1;38;2;224;175;104mls\x1b[0m" in out  # bold #e0af68


def test_tool_output_is_muted_and_a_failure_is_red() -> None:
    from mypi.agent.loop import ToolEnd

    ok = _render_coloured(ToolEnd(_call(), "created /x", False))
    bad = _render_coloured(ToolEnd(_call(), "old_string was not found", True))
    assert "224;175;104mwrite" in ok  # the tool's name is orange
    assert "86;95;137mcreated /x" in ok  # the status message is muted
    assert "247;118;142m" in bad  # #f7768e red for the failure


def test_turn_summaries_are_muted() -> None:
    from mypi.agent.loop import TurnEnd
    from mypi.types import AssistantMessage, Usage

    out = _render_coloured(TurnEnd(AssistantMessage([], Usage(10, 5), "toolUse")))
    assert "38;2;86;95;137m" in out  # #565f89


# --- the accordion: one line per tool call unless --expand ---


def _render_folded(*events, tty: bool = False) -> str:
    """Default render (folded), optionally on a pretend terminal for colour/hints."""
    import contextlib
    import io

    import mypi.main as m

    class Buffer(io.StringIO):
        def isatty(self) -> bool:
            return tty

    out = Buffer()
    with contextlib.redirect_stdout(out):
        on = m.render("groq", "model-x")
        for event in events:
            on(event)
    return out.getvalue()


def test_a_folded_tool_start_prints_nothing() -> None:
    """The fold line on ToolEnd names the tool, so the running stage stays quiet."""
    from mypi.agent.loop import ToolStart

    assert _render_folded(ToolStart(_call("read", {"path": "a.py"}))) == ""


def test_a_started_and_finished_call_is_one_fold_line() -> None:
    """No duplicate header: the fold line itself carries the tool's name."""
    from mypi.agent.loop import ToolEnd, ToolStart

    out = _render_folded(
        ToolStart(_call("read", {"path": "a.py"})),
        ToolEnd(_call("read", {"path": "a.py"}), "x = 1\ny = 2\n", False),
    )
    assert out == " ▸ read · a.py · 2 lines\n", repr(out)


def test_tool_results_are_folded_to_one_line_by_default() -> None:
    from mypi.agent.loop import ToolEnd

    out = _render_folded(ToolEnd(_call("read", {"path": "a.py"}), "x = 1\ny = 2\n", False))
    assert out == " ▸ read · a.py · 2 lines\n", repr(out)


def test_a_single_line_tool_result_stays_a_summary_when_folded() -> None:
    from mypi.agent.loop import ToolEnd

    out = _render_folded(ToolEnd(_call("write"), "created /x (29 bytes, 3 lines)", False))
    assert out == " ▸ write · created /x (29 bytes, 3 lines)\n", repr(out)


def test_an_empty_tool_result_folds_to_no_output() -> None:
    from mypi.agent.loop import ToolEnd

    assert _render_folded(ToolEnd(_call(), "", False)) == " ▸ write · (no output)\n"


def test_an_error_is_never_folded() -> None:
    from mypi.agent.loop import ToolEnd

    one_line = _render_folded(ToolEnd(_call("edit"), "old_string was not found in the file", True))
    assert one_line.startswith(" edit failed: old_string was not found")

    multiline = _render_folded(ToolEnd(_call("edit"), "first line\nsecond\n", True))
    assert "▸" not in multiline
    assert " edit failed" in multiline
    assert "first line" in multiline


def test_the_expand_flag_prints_every_tool_call_in_full() -> None:
    from mypi.agent.loop import ToolEnd

    out = _render(ToolEnd(_call("read", {"path": "a.py"}), "x = 1\ny = 2\n", False))
    assert out == " read: 2 lines\n  x = 1\n  y = 2\n", repr(out)


def test_folded_lines_are_coloured_too() -> None:
    from mypi.agent.loop import ToolEnd

    out = _render_folded(ToolEnd(_call("read", {"path": "a.py"}), "x\ny\n", False), tty=True)
    assert "224;175;104m▸ read" in out  # bold #e0af68
    assert "86;95;137ma.py" in out  # #565f89
    assert "86;95;137m2 lines" in out

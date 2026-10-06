"""The accordion formatting helpers, one line per test."""

from mypi.colors import Colors
from mypi.viewer import ToolCall, count_lines, fold_header, say_stop_reason


def _c() -> Colors:
    return Colors(enabled=False)


def _call(name: str = "read", result: str = "x\ny\n", error: bool = False) -> ToolCall:
    return ToolCall(name, {"path": "a.py"}, result, error)


def test_count_lines_ignores_a_trailing_newline_and_a_showing_note() -> None:
    assert count_lines("a\nb\n") == 2
    assert count_lines("a\nb\n[showing 2 of 3]") == 2
    assert count_lines("") == 0


def test_fold_header_counts_a_multiline_result() -> None:
    assert fold_header(_call("read", "x = 1\ny = 2\n"), _c()) == "▸ read · a.py · 2 lines"


def test_fold_header_shows_a_single_line_result_in_full() -> None:
    call = ToolCall("write", {"path": "x"}, "created /x (3 bytes, 1 line)", False)
    assert fold_header(call, _c()) == "▸ write · created /x (3 bytes, 1 line)"


def test_fold_header_says_no_output_for_an_empty_result() -> None:
    call = ToolCall("write", {"path": "x"}, "", False)
    assert fold_header(call, _c()) == "▸ write · (no output)"


def test_say_stop_reason_translates_the_api_reasons() -> None:
    assert say_stop_reason("stop") == "finished"
    assert say_stop_reason("toolUse") == "tool call"
    assert say_stop_reason("length") == "hit the token budget"
    assert say_stop_reason("weird") == "weird"

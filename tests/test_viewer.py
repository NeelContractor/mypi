"""The interactive expander: format of the alternate-screen viewer and keys."""

from mypi.colors import Colors
from mypi.viewer import (
    ToolCall,
    Written,
    build_rows,
    count_lines,
    draw,
    fold_header,
    interpret,
    mouse_event,
)


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


def test_build_rows_splits_text_and_folds_tool_calls() -> None:
    blocks: list = [
        Written("intro\n"),
        _call("read", "x\ny\n"),
        ToolCall("edit", {}, "old_string was not found", True),
    ]
    rows, headers, title_rows = build_rows(blocks, [False], _c())
    assert rows == ["intro", "▸ read · a.py · 2 lines", "edit failed: old_string was not found"]
    assert headers == [1]
    assert title_rows == {1: 0}  # clicking the fold line toggles the read call


def test_build_rows_expands_a_tool_call_in_full() -> None:
    blocks: list = [_call("read", "a\nb\n")]
    rows, headers, title_rows = build_rows(blocks, [True], _c())
    assert rows == ["read: 2 lines", "  a", "  b"]
    assert headers == []
    assert title_rows == {0: 0}  # its title line stays clickable when expanded


def test_title_rows_can_distinguish_two_calls() -> None:
    blocks: list = [Written("x\n"), _call("ls", "a\n"), _call("read", "b\nc\n")]
    _, _, title_rows = build_rows(blocks, [False, True], _c())
    assert title_rows == {1: 0, 2: 1}


def test_expanded_rows_always_show_the_full_result() -> None:
    """The viewer is for reading, so it never truncates the way streaming does."""
    blocks: list = [_call("read", "\n".join(f"line {i}" for i in range(50)) + "\n")]
    rows, _, _ = build_rows(blocks, [True], _c())
    assert len(rows) == 51
    assert rows[-1] == "  line 49"


def test_draw_keeps_the_selected_header_in_view_and_adds_a_footer() -> None:
    rows = [f"r{i}" for i in range(5)]
    out = draw(rows, [0, 2, 4], 2, 0, 100, 3)  # selected tool #2 lives at row 4
    assert "↑/↓ move" in out
    assert "\x1b[7mr4\x1b[0m" in out  # the selected row is reversed
    assert "r2" not in out  # scrolled down so the selection fits above the footer


def test_draw_shows_the_transcript_without_a_selection_marker_when_none() -> None:
    rows = [f"r{i}" for i in range(5)]
    out = draw(rows, [], None, 0, 100, 10)
    assert "r4" in out
    assert "\x1b[7m" not in out


def test_interpret_maps_the_viewer_keys() -> None:
    assert interpret(b"q") == "quit"
    assert interpret(b"Q") == "quit"
    assert interpret(b"\x1b") == "quit"
    assert interpret(b"\x03") == "quit"
    assert interpret(b" ") == "toggle"
    assert interpret(b"\r") == "toggle"
    assert interpret(b"e") == "expand_all"
    assert interpret(b"c") == "collapse_all"
    assert interpret(b"j") == "down"
    assert interpret(b"\x1b[B") == "down"
    assert interpret(b"k") == "up"
    assert interpret(b"\x1b[A") == "up"
    assert interpret(b" q") == "toggle"  # only the first key of a batched read
    assert interpret(b"x") is None
    assert interpret(b"\x1b[") is None  # a partial escape: wait for the arrow


def test_mouse_event_reads_an_sgr_left_click() -> None:
    assert mouse_event(b"\x1b[<0;12;5M") == (5, 12)


def test_mouse_event_reads_an_x10_left_click() -> None:
    # ESC [ M <button+32> <column+32> <row+32>
    assert mouse_event(b"\x1b[M" + bytes([32, 32 + 12, 32 + 5])) == (5, 12)


def test_mouse_event_ignores_releases_and_other_buttons() -> None:
    assert mouse_event(b"\x1b[<0;12;5m") is None  # release (lowercase m)
    assert mouse_event(b"\x1b[<2;12;5M") is None  # middle button
    assert mouse_event(b"\x1b[M" + bytes([32 + 3, 32, 32])) is None  # X10 release
    assert mouse_event(b"just a key") is None


def test_explore_gate_is_silent_when_piped() -> None:
    """What streams to a file or another program must never open the viewer."""
    from mypi.main import _maybe_explore

    record = [ToolCall("read", {"path": "a.py"}, "x\ny\n", False)]
    _maybe_explore(record)  # no tty under pytest → returns without drawing anything


def test_render_records_blocks_for_the_viewer() -> None:
    import contextlib
    import io

    import mypi.main as m
    from mypi.agent.loop import TextEvent, ToolEnd
    from mypi.types import ToolCallBlock

    record: list = []
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        on = m.render("groq", "model-x", record=record)
        on(ToolEnd(ToolCallBlock(id="1", name="read", arguments={"path": "a.py"}), "x\ny\n", False))
        on(TextEvent("the answer"))

    tool, text = record[0], record[1]
    assert isinstance(tool, ToolCall) and (tool.name, tool.result, tool.is_error) == (
        "read",
        "x\ny\n",
        False,
    )
    assert isinstance(text, Written) and text.text == "the answer"
    assert out.getvalue()  # recording never replaces the live output


def test_render_records_nothing_when_no_record_is_requested() -> None:
    import contextlib
    import io

    import mypi.main as m
    from mypi.agent.loop import ToolEnd
    from mypi.types import ToolCallBlock

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        m.render("groq", "model-x")(
            ToolEnd(ToolCallBlock(id="1", name="read", arguments={}), "x\n", False)
        )
    assert out.getvalue()

"""Colours are a veneer on the text: escape codes come and go, never meaning."""

import sys

import pytest

from mypi.colors import TOOL, Colors, color_enabled


def test_paint_wraps_text_in_truecolour_when_enabled() -> None:
    out = Colors(enabled=True).paint("ls", TOOL)
    assert "\x1b[38;2;" in out
    assert "38;2;224;175;104" in out  # #e0af68, the tool colour
    assert out.endswith("\x1b[0m")


def test_paint_adds_bold_only_when_asked() -> None:
    plain = Colors(enabled=True).paint("ls", TOOL)
    bold = Colors(enabled=True).paint("ls", TOOL, bold=True)
    assert bold == "\x1b[1;" + plain[len("\x1b[") :]


def test_paint_passes_text_through_when_disabled() -> None:
    assert Colors(enabled=False).paint("ls", TOOL) == "ls"


def test_a_pipe_disables_colour(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)

    class Pipe:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, "stdout", Pipe())
    assert color_enabled() is False


def test_a_terminal_enables_colour(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)

    class Tty:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdout", Tty())
    assert color_enabled() is True


def test_no_color_env_var_beats_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")

    class Tty:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdout", Tty())
    assert color_enabled() is False

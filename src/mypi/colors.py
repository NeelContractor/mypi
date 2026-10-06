"""Terminal colour, applied only when the output is actually a terminal.

Palette is Tokyo Night, matching the author's NeuralCode UI. Colours are a
pure veneer on the text: they never add or remove characters that a test or a
pipe could trip over, and piping the output or setting NO_COLOR drops them.
"""

from __future__ import annotations

import os
import sys

ACCENT = (122, 162, 247)  # #7aa2f7 - headings and usage
USER = (158, 206, 106)  # #9ece6a - echoing the reader's own words
TOOL = (224, 175, 104)  # #e0af68 - tool calls
MUTED = (86, 95, 137)  # #565f89 - tool output and metadata
ERROR = (247, 118, 142)  # #f7768e - failures

_RESET = "\x1b[0m"
_OPEN = "\x1b[{}38;2;{};{};{}m"


def color_enabled() -> bool:
    """True when the colours can actually be seen."""
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


class Colors:
    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = color_enabled() if enabled is None else enabled

    def paint(self, text: str, rgb: tuple[int, int, int], *, bold: bool = False) -> str:
        if not self.enabled:
            return text
        r, g, b = rgb
        style = "1;" if bold else ""
        return f"{_OPEN.format(style, r, g, b)}{text}{_RESET}"

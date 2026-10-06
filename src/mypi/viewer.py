"""Interactive expander for a finished run's tool calls.

A streaming run fixes its transcript in the terminal's scrollback, which cannot
be reopened or collapsed. This viewer redraws the same run on the alternate
screen buffer, where folding is just state the program owns, so ↑/↓ and space
can expand and collapse each tool call. Only ever entered on a real terminal.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Any

from .colors import ERROR, MUTED, TOOL, Colors

_NOTE = re.compile(r"\[showing [^\]]*\]")
_STRIP = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(frozen=True)
class Written:
    text: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool


Block = Written | ToolCall


def _content_lines(body: str) -> list[str]:
    if not body:
        return []
    lines = body.split("\n")
    if len(lines) > 1 and lines[-1] == "":
        lines = lines[:-1]
    return lines


def count_lines(body: str) -> int:
    """Content lines, ignoring a trailing [showing ...] note and its newline."""
    lines = _content_lines(body)
    if lines and _NOTE.fullmatch(lines[-1].strip()):
        lines = lines[:-1]
    return len(lines)


def short_args(arguments: dict[str, Any]) -> str:
    """Compact argument summary: a lone value, otherwise the whole JSON."""
    if len(arguments) == 1:
        return str(next(iter(arguments.values())))
    return str(arguments)


def fold_header(call: ToolCall, c: Colors) -> str:
    """The one-line collapsed form, styled."""
    head = c.paint(f"▸ {call.name}", TOOL, bold=True)
    if not call.result.strip():
        return f"{head} · {c.paint('(no output)', MUTED)}"
    if "\n" not in call.result.strip():
        return f"{head} · {c.paint(call.result.strip(), MUTED)}"
    args = c.paint(short_args(call.arguments), MUTED)
    return f"{head} · {args} · {c.paint(f'{count_lines(call.result)} lines', MUTED)}"


def expanded_rows(call: ToolCall, c: Colors) -> list[str]:
    """The full form, styled; an error is coloured as a failure."""
    colour = ERROR if call.is_error else TOOL
    head = c.paint(call.name, colour, bold=True)
    if call.is_error:
        first, _, rest = call.result.partition("\n")
        rows = [c.paint(f"{call.name} failed: {first}", ERROR)]
        if rest:
            rows += indent_rows(rest, c, ERROR)
        return rows
    if not call.result.strip():
        return [f"{head}: {c.paint('(no output)', MUTED)}"]
    if "\n" not in call.result.strip():
        return [f"{head}: {c.paint(call.result.strip(), MUTED)}"]
    label = f"{head}: {c.paint(f'{count_lines(call.result)} lines', MUTED)}"
    return [label, *indent_rows(call.result, c)]


def indent_rows(body: str, c: Colors, rgb: tuple[int, int, int] = MUTED) -> list[str]:
    lines = body.split("\n")
    if len(lines) > 1 and lines[-1] == "":
        lines = lines[:-1]
    return [c.paint(f"  {line}", rgb) for line in lines]


def build_rows(
    blocks: list[Block], folds: list[bool], c: Colors
) -> tuple[list[str], list[int], dict[int, int]]:
    """Render every block to display rows.

    Returns (rows, fold-line rows, title row → foldable tool), where the title
    row is the one place a click toggles that tool.
    """
    rows: list[str] = []
    headers: list[int] = []
    title_rows: dict[int, int] = {}
    tool = 0
    for block in blocks:
        if isinstance(block, Written):
            rows.extend(_content_lines(block.text))
        elif block.is_error:
            rows.extend(expanded_rows(block, c))
        else:
            title_rows[len(rows)] = tool
            if folds[tool]:
                rows.extend(expanded_rows(block, c))
            else:
                headers.append(len(rows))
                rows.append(fold_header(block, c))
            tool += 1
    return rows, headers, title_rows


def fit(line: str, width: int) -> str:
    """Clip to the terminal width without cutting an escape sequence mid-way."""
    plain = _STRIP.sub("", line)
    if len(plain) <= width:
        return line
    return plain[:width]


def draw(
    rows: list[str],
    headers: list[int],
    selected: int | None,
    top: int,
    width: int,
    height: int,
) -> str:
    if not rows:
        return "\x1b[2J\x1b[H\n"
    view_h = max(height - 1, 1)
    if selected is not None:
        sel = headers[selected]
        if sel < top:
            top = sel
        elif sel >= top + view_h:
            top = sel - view_h + 1
    top = max(0, min(top, max(0, len(rows) - view_h)))

    lines = ["\x1b[2J\x1b[H"]
    for i in range(view_h):
        index = top + i
        if index >= len(rows):
            break
        line = fit(rows[index], width)
        if selected is not None and index == sel:
            line = f"\x1b[7m{line}\x1b[0m"
        lines.append(line)
    lines.append(
        fit(
            "  click a line to expand · ↑/↓ move · space toggles · e all · c none · q quit  ", width
        )
    )
    return "\n".join(lines)


def mouse_event(data: bytes) -> tuple[int, int] | None:
    """A left-button press from the terminal, as (row, column) in 1-based screen
    coordinates, or None for anything else (releases, other buttons, no event)."""
    if data.startswith(b"\x1b[<"):
        # SGR: ESC [ < button ; column ; row M, with m for a release
        m = re.match(rb"\x1b\[<(\d+);(\d+);(\d+)([Mm])", data)
        if not m or m.group(4) != b"M" or int(m.group(1)) != 0:
            return None
        return (int(m.group(3)), int(m.group(2)))
    if data.startswith(b"\x1b[M") and len(data) >= 6:
        # X10: ESC [ M <button+32> <column+32> <row+32>
        if data[3] != 32:  # button 0 decoded (a left press)
            return None
        return (data[5] - 32, data[4] - 32)
    return None


def interpret(data: bytes) -> str | None:
    """Map a keypress to an action. Multi-byte batches use only their first key;
    anything left over stays in the terminal's buffer for the next read, so a
    quick space-then-q can't lose the q."""
    if data.startswith(b"\x1b"):
        if data.startswith(b"\x1b[A"):
            return "up"
        if data.startswith(b"\x1b[B"):
            return "down"
        if data == b"\x1b":
            return "quit"
        return None  # a fragmenting escape sequence; wait for the rest
    key = data[:1]
    if key in (b"q", b"Q", b"\x03"):
        return "quit"
    if key in (b" ", b"\r", b"\n"):
        return "toggle"
    if key in (b"e", b"E"):
        return "expand_all"
    if key in (b"c", b"C"):
        return "collapse_all"
    if key == b"j":
        return "down"
    if key == b"k":
        return "up"
    return None


def run(blocks: list[Block]) -> None:
    """Own the screen until the user leaves; the scrollback is untouched."""
    import shutil
    import termios
    import tty

    c = Colors()
    folds = [False] * sum(1 for b in blocks if isinstance(b, ToolCall) and not b.is_error)
    rows, headers, title_rows = build_rows(blocks, folds, c)
    if not headers:
        return
    width, height = shutil.get_terminal_size(fallback=(80, 24))
    selected: int | None = 0
    top = 0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[?1000h\x1b[?1006h")
    sys.stdout.flush()
    try:
        tty.setcbreak(fd)
        while True:
            sys.stdout.write(draw(rows, headers, selected, top, width, height))
            sys.stdout.flush()
            data = os.read(fd, 16)
            action = interpret(data)
            if action == "quit":
                break
            click = mouse_event(data) if action is None else None
            changed = False
            if click is not None:
                tool = title_rows.get(top + click[0] - 1)
                if tool is not None:
                    folds[tool] = not folds[tool]
                    changed = True
            elif action == "up" and selected is not None:
                selected = max(0, selected - 1)
            elif action == "down" and selected is not None:
                selected = min(len(headers) - 1, selected + 1)
            elif action == "toggle":
                if not headers:
                    folds = [False] * len(folds)
                elif selected is not None:
                    folds[selected] = not folds[selected]
                changed = True
            elif action == "expand_all":
                folds = [True] * len(folds)
                changed = True
            elif action == "collapse_all":
                folds = [False] * len(folds)
                changed = True
            if changed:
                rows, headers, title_rows = build_rows(blocks, folds, c)
                selected = 0 if headers else None
    finally:
        sys.stdout.write("\x1b[?1006l\x1b[?1000l\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

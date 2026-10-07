"""Formatting for the tool-call accordion: one collapsed line per call."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .colors import MUTED, TOOL, Colors

_NOTE = re.compile(r"\[showing [^\]]*\]")


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool


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


_SHOWING_RANGE = re.compile(r"\[showing lines (\d+)-(\d+) of \d+")


def _clip(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _what(call: ToolCall) -> str:
    """One muted phrase for what was asked: 'src/main.py  ·  lines 1–200'."""
    a = call.arguments
    if call.name == "read":
        shown = _SHOWING_RANGE.search(call.result)
        if shown:
            bits = [str(a.get("path", "")), f"lines {shown.group(1)}–{shown.group(2)}"]
        else:
            offset = int(a.get("offset") or 1)
            bits = [str(a.get("path", ""))]
            if a.get("limit") is not None:
                bits.append(f"lines {offset}–{offset + int(a['limit']) - 1}")
            elif offset > 1:
                bits.append(f"from line {offset}")
    else:
        bits = [
            _clip(str(a[key]))
            for key in ("command", "pattern", "path", "glob", "name")
            if a.get(key)
        ][:2]
    return "  ·  ".join(bit for bit in bits if bit)


def _back(call: ToolCall) -> str:
    """What came back: a line count, a one-liner, or nothing."""
    body = call.result.strip()
    if not body:
        return "(no output)"
    if "\n" not in body:
        return _clip(body, 80)
    return f"{count_lines(call.result)} lines"


def fold_header(call: ToolCall, c: Colors) -> str:
    """Two collapsed lines: what was called, then what came back."""
    head = c.paint("●", TOOL, bold=True) + " " + c.paint(call.name, TOOL, bold=True)
    what = _what(call)
    if what:
        head += "  " + c.paint(what, MUTED)
    return f"{head}\n  └─ {c.paint(_back(call), MUTED)}"


def say_stop_reason(reason: str) -> str:
    """Turn the API's stop reasons into something a reader can act on."""
    return {
        "toolUse": "tool call",
        "length": "hit the token budget",
        "stop": "finished",
    }.get(reason, reason)

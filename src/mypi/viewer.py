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


def short_args(arguments: dict[str, Any]) -> str:
    """Compact argument summary: a lone value, otherwise the whole dict."""
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


def say_stop_reason(reason: str) -> str:
    """Turn the API's stop reasons into something a reader can act on."""
    return {
        "toolUse": "tool call",
        "length": "hit the token budget",
        "stop": "finished",
    }.get(reason, reason)

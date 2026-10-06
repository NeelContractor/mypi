from __future__ import annotations

import asyncio
from typing import Any

from ..types import Tool
from .fsutil import as_number, cap_output, workspace_path

MAX_BYTES = 50 * 1024
DEFAULT_LINES = 2000


def _run(args: dict[str, Any]) -> str:
    target = workspace_path(str(args["path"]))
    offset = max(1, int(as_number(args.get("offset"), 1)))
    limit = max(1, int(as_number(args.get("limit"), DEFAULT_LINES)))

    with open(target, "rb") as f:
        buf = f.read(MAX_BYTES + 1)
    oversized = len(buf) > MAX_BYTES

    lines = buf[:MAX_BYTES].decode("utf-8", errors="replace").split("\n")
    # a trailing newline yields a phantom empty last element; the file doesn't have it
    if len(lines) > 1 and lines[-1] == "":
        lines = lines[:-1]
    start = offset - 1
    window = lines[start : start + limit]
    if not window:
        return f"(no lines at offset {offset}; file has {len(lines)} lines)"

    remaining = len(lines) - (start + len(window))
    if offset > 1 or oversized or remaining > 0:
        parts = [f"lines {offset}-{offset + len(window) - 1} of {len(lines)}"]
        if oversized:
            parts.append(f"file is bigger than {MAX_BYTES} bytes and was cut there")
        if remaining > 0:
            parts.append("more lines follow, pass offset to continue")
        return cap_output("\n".join(window) + "\n[showing " + "; ".join(parts) + "]")
    # the caller is seeing the whole file, so a note would just be noise
    return cap_output("\n".join(window))


async def _execute(args: dict[str, Any]) -> str:
    return await asyncio.to_thread(_run, args)


read_tool = Tool(
    name="read",
    description="Read a text file and return its contents, optionally a window of lines",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, relative to the current folder",
            },
            "offset": {
                "type": "number",
                "description": "First line to return, 1-based (default 1)",
            },
            "limit": {
                "type": "number",
                "description": f"How many lines to return (default {DEFAULT_LINES})",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    execute=_execute,
)

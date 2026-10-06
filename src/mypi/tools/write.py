from __future__ import annotations

import asyncio
import os
from typing import Any

from ..types import Tool
from .fsutil import workspace_path


def _run(args: dict[str, Any]) -> str:
    if "path" not in args or "content" not in args:
        return "error: path and content are both required"
    path = str(args["path"])
    content = args["content"]
    if not isinstance(content, str):
        return "error: content must be a string"

    target = workspace_path(path)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    replaced = os.path.exists(target)
    # newline="" writes exactly what we were given, without swapping \n for os.linesep
    with open(target, "w", encoding="utf-8", newline="") as f:
        f.write(content)

    size = len(content.encode("utf-8"))
    lines = len(content.split("\n"))
    verb = "replaced" if replaced else "created"
    return f"{verb} {target} ({size} bytes, {lines} lines)"


async def _execute(args: dict[str, Any]) -> str:
    return await asyncio.to_thread(_run, args)


write_tool = Tool(
    name="write",
    description=(
        "Create a file, or replace an existing file's entire contents. Missing parent "
        "folders are created. For a small change to an existing file prefer edit, which "
        "only rewrites what it has to."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File to create or replace, relative to the current folder",
            },
            "content": {"type": "string", "description": "The complete new contents of the file"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    execute=_execute,
)

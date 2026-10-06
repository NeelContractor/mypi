from __future__ import annotations

import asyncio
import os
import stat as statmod
from datetime import UTC, datetime
from typing import Any

from ..types import Tool
from .fsutil import workspace_path, workspace_root


def format_bytes(n: float) -> str:
    if n < 1024:
        return f"{int(n)} B"
    value = float(n)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
    return f"{value:.2f} TB"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _run(args: dict[str, Any]) -> str:
    requested = os.path.join(workspace_root(), str(args["path"]))
    target = workspace_path(str(args["path"]))
    # workspace_path resolves symlinks away, so islink looks at what was asked for
    is_link = os.path.islink(requested)
    s = os.stat(target)  # follows symlinks, so the numbers describe the real file
    if statmod.S_ISDIR(s.st_mode):
        kind = "directory"
    elif statmod.S_ISREG(s.st_mode):
        kind = "file"
    else:
        kind = "other"
    created = getattr(s, "st_birthtime", s.st_ctime)  # st_birthtime isn't on Linux
    return "\n".join(
        [
            f"path: {target}",
            f"symlink: {'yes -> ' + os.readlink(requested) if is_link else 'no'}",
            f"type: {kind}",
            f"size: {s.st_size} bytes ({format_bytes(s.st_size)})",
            f"created: {_iso(created)}",
            f"modified: {_iso(s.st_mtime)}",
            f"accessed: {_iso(s.st_atime)}",
            f"mode: {oct(s.st_mode & 0o777)[2:]}",
        ]
    )


async def _execute(args: dict[str, Any]) -> str:
    return await asyncio.to_thread(_run, args)


stat_tool = Tool(
    name="stat",
    description="Get metadata for a file or directory: size, type, timestamps, permissions.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file or directory, relative to the current folder",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    execute=_execute,
)

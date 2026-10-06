from __future__ import annotations

import asyncio
from typing import Any

from ..types import Tool
from .fsutil import as_number, cap_output, label, walk, workspace_path

LIMIT = 5000


def _run(args: dict[str, Any]) -> str:
    root = workspace_path(str(args.get("path") or "."))
    max_depth = max(1, as_number(args.get("maxDepth"), float("inf")))
    include_hidden = bool(args.get("includeHidden"))
    dirs_only = bool(args.get("dirsOnly"))

    lines: list[str] = []
    for entry in walk(root, max_depth=max_depth, include_hidden=include_hidden):
        if dirs_only and not entry.is_directory:
            continue
        lines.append(label(entry))
        if len(lines) >= LIMIT:
            lines.append(f"[truncated at {LIMIT} entries]")
            break
    return cap_output("\n".join(lines)) if lines else "(empty)"


async def _execute(args: dict[str, Any]) -> str:
    return await asyncio.to_thread(_run, args)


ls_tool = Tool(
    name="ls",
    description=(
        "List files and directories recursively under a path. Skips common noise dirs "
        "(.git, node_modules, dist, build) and hidden files by default."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to list, relative to the current folder (default '.')",
            },
            "maxDepth": {
                "type": "number",
                "description": "How many levels deep to recurse (default: unlimited)",
            },
            "includeHidden": {
                "type": "boolean",
                "description": "Include dotfiles/dot-directories (default false)",
            },
            "dirsOnly": {"type": "boolean", "description": "Only list directories (default false)"},
        },
        "required": [],
        "additionalProperties": False,
    },
    execute=_execute,
)

from __future__ import annotations

import asyncio
import re
from typing import Any

from ..types import Tool
from .fsutil import cap_output, glob_to_regex, label, normalize_glob, walk, workspace_path

LIMIT = 2000

TYPES = ("file", "dir")


def _run(args: dict[str, Any]) -> str:
    root = workspace_path(str(args.get("path") or "."))
    name = str(args["name"]) if args.get("name") else None
    glob_pattern = str(args["glob"]) if args.get("glob") else None
    regex_pattern = str(args["regex"]) if args.get("regex") else None
    type_filter = str(args["type"]).lower() if args.get("type") else None
    include_hidden = bool(args.get("includeHidden"))

    if type_filter is not None and type_filter not in TYPES:
        return f"error: type must be one of {', '.join(TYPES)}, got {type_filter!r}"
    if not (name or glob_pattern or regex_pattern):
        return "error: provide at least one of 'name', 'glob', or 'regex'"

    glob_re = glob_to_regex(normalize_glob(glob_pattern)) if glob_pattern else None
    regex_re = None
    if regex_pattern:
        try:
            regex_re = re.compile(regex_pattern)
        except re.error as e:
            return f"invalid regex: {e}"

    results: list[str] = []
    for entry in walk(root, include_hidden=include_hidden):
        if type_filter == "file" and entry.is_directory:
            continue
        if type_filter == "dir" and not entry.is_directory:
            continue
        base = entry.rel_path.split("/")[-1]
        if name and name.lower() not in base.lower():
            continue
        if glob_re and not glob_re.match(entry.rel_path):
            continue
        if regex_re and not regex_re.search(entry.rel_path):
            continue
        results.append(label(entry))
        if len(results) >= LIMIT:
            results.append(f"[truncated at {LIMIT} results]")
            break

    return cap_output("\n".join(results)) if results else "(no matches)"


async def _execute(args: dict[str, Any]) -> str:
    return await asyncio.to_thread(_run, args)


find_tool = Tool(
    name="find",
    description=(
        "Find files/directories by name, matching a substring, glob pattern "
        "(e.g. '**/*.test.py') or regex, recursively under a path."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Substring to match against file/directory names",
            },
            "glob": {
                "type": "string",
                "description": (
                    "Glob pattern to match against the path; a bare '*.test.py' matches "
                    "at any depth, e.g. '**/*.py'"
                ),
            },
            "regex": {"type": "string", "description": "Regex to match against the relative path"},
            "path": {
                "type": "string",
                "description": (
                    "Directory to search under, relative to the current folder (default '.')"
                ),
            },
            "type": {
                "type": "string",
                "enum": list(TYPES),
                "description": "Filter by 'file' or 'dir' (default: both)",
            },
            "includeHidden": {
                "type": "boolean",
                "description": "Include dotfiles/dot-directories (default false)",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    execute=_execute,
)

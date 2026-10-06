from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from ..types import Tool
from .fsutil import (
    WalkEntry,
    as_number,
    cap_output,
    glob_to_regex,
    normalize_glob,
    walk,
    workspace_path,
)

MAX_MATCHES = 500
MAX_FILE_BYTES = 5 * 1024 * 1024  # skip files bigger than 5MB


def _run(args: dict[str, Any]) -> str:
    root = workspace_path(str(args.get("path") or "."))
    ignore_case = bool(args.get("ignoreCase"))
    max_matches = int(as_number(args.get("maxMatches"), MAX_MATCHES))
    if max_matches < 1:
        return "error: maxMatches must be at least 1"
    glob = str(args["glob"]) if args.get("glob") else None
    glob_re = glob_to_regex(normalize_glob(glob)) if glob else None

    try:
        pattern = re.compile(str(args["pattern"]), re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return f"invalid regex: {e}"

    # allow `path` to be a single file as well as a directory
    entries = iter([_single_file(root)]) if os.path.isfile(root) else walk(root)

    results: list[str] = []
    for entry in entries:
        if entry.is_directory:
            continue
        if glob_re and not glob_re.match(entry.rel_path):
            continue
        try:
            with open(entry.abs_path, "rb") as f:
                buf = f.read(MAX_FILE_BYTES + 1)
        except OSError:
            continue
        if len(buf) > MAX_FILE_BYTES or b"\0" in buf:  # too big / binary
            continue
        for i, line in enumerate(buf.decode("utf-8", errors="replace").split("\n"), start=1):
            if pattern.search(line):
                results.append(f"{entry.rel_path}:{i}:{line}")
                if len(results) >= max_matches:
                    results.append(f"[truncated at {max_matches} matches]")
                    return cap_output("\n".join(results))

    return cap_output("\n".join(results)) if results else "(no matches)"


async def _execute(args: dict[str, Any]) -> str:
    return await asyncio.to_thread(_run, args)


def _single_file(path: str) -> WalkEntry:
    return WalkEntry(path, os.path.basename(path), False)


grep_tool = Tool(
    name="grep",
    description=(
        "Search for a regex pattern inside files under a directory (recursively). "
        "Returns matching file:line:text, like grep -rn."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "path": {
                "type": "string",
                "description": (
                    "Directory or file to search in, relative to the current folder (default '.')"
                ),
            },
            "glob": {
                "type": "string",
                "description": (
                    "Only search files whose path matches this glob; a bare '*.py' matches "
                    "at any depth, e.g. '**/*.py'"
                ),
            },
            "ignoreCase": {
                "type": "boolean",
                "description": "Case-insensitive search (default false)",
            },
            "maxMatches": {
                "type": "number",
                "description": f"Max number of matches to return (default {MAX_MATCHES})",
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    },
    execute=_execute,
)

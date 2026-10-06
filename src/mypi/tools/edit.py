from __future__ import annotations

import asyncio
import os
from typing import Any

from ..types import Tool
from .fsutil import workspace_path

BYTES_LIMIT = 2 * 1024 * 1024  # refuse to rewrite huge files blind


def _run(args: dict[str, Any]) -> str:
    for key in ("path", "old_string", "new_string"):
        if key not in args:
            return f"error: {key} is required"
    path = str(args["path"])
    old, new = args["old_string"], args["new_string"]
    if not isinstance(old, str) or not isinstance(new, str):
        return "error: old_string and new_string must be strings"
    if not old:
        return "error: old_string must not be empty"
    if old == new:
        return "error: old_string and new_string are identical, so nothing would change"
    replace_all = args.get("replace_all") is True

    target = workspace_path(path)
    try:
        with open(target, encoding="utf-8", newline="") as f:
            text = f.read()
    except FileNotFoundError:
        return f"error: no such file: {target}"
    except IsADirectoryError:
        return f"error: {target} is a directory, not a file"
    except UnicodeDecodeError:
        return f"error: {os.path.basename(target)} is not valid UTF-8, so it cannot be edited"
    except OSError as e:
        return f"error: cannot read {target}: {e}"

    if len(text.encode("utf-8")) > BYTES_LIMIT:
        return (
            f"error: {target} is bigger than {BYTES_LIMIT} bytes; "
            "edit only handles ordinary source files"
        )

    hits = text.count(old)
    if hits == 0:
        return (
            "error: old_string was not found in the file. It has to match exactly, "
            "including indentation, whitespace and line endings; read the file first"
        )
    if hits > 1 and not replace_all:
        return (
            f"error: old_string appears {hits} times, so the edit would be ambiguous. "
            "Add surrounding context to make it unique, or set replace_all to true"
        )

    changed = hits if replace_all else 1
    new_text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    # overwrite in place so the file keeps its permissions and inode
    with open(target, "w", encoding="utf-8", newline="") as f:
        f.write(new_text)

    delta = len(new_text.encode("utf-8")) - len(text.encode("utf-8"))
    plural = "s" if changed != 1 else ""
    return f"replaced {changed} occurrence{plural} in {target} ({delta:+d} bytes)"


async def _execute(args: dict[str, Any]) -> str:
    return await asyncio.to_thread(_run, args)


edit_tool = Tool(
    name="edit",
    description=(
        "Change part of an existing file: replace old_string with new_string. "
        "old_string must match the file exactly (whitespace and indentation included) "
        "and must be unique in the file; if it is not, add more surrounding context "
        "or set replace_all. Fails loudly rather than guessing. Prefer this over write "
        "for a small change."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File to change, relative to the current folder",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find in the file, which must occur once",
            },
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of requiring a unique match",
                "default": False,
            },
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    },
    execute=_execute,
)

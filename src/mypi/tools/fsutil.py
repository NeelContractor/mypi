from __future__ import annotations

import fnmatch
import math
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

DEFAULT_IGNORES = [
    ".git",
    "node_modules",
    "dist",
    "build",
    ".next",
    "coverage",
    "__pycache__",
    ".venv",
]

MAX_OUTPUT_BYTES = 50 * 1024


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Supports *, ?, ** against a forward-slash relative path."""
    p = pattern.replace(os.sep, "/").rstrip("/")  # rel paths carry no trailing slash
    out = ""
    i = 0
    while i < len(p):
        c = p[i]
        if c == "*":
            if i + 1 < len(p) and p[i + 1] == "*":
                i += 1  # "**" matches across path segments (including none)
                if i + 1 < len(p) and p[i + 1] == "/":
                    i += 1
                    out += "(?:.*/)?"  # "**/" → zero or more leading directories
                else:
                    out += ".*"  # trailing "**" → everything, separators included
            else:
                out += "[^/]*"
        elif c == "?":
            out += "[^/]"
        else:
            out += re.escape(c)
        i += 1
    return re.compile(f"^{out}$")


def normalize_glob(pattern: str) -> str:
    """A pattern without a slash is a basename pattern, so '*.py' matches 'src/a.py' too."""
    p = pattern.replace(os.sep, "/")
    return p if "/" in p or p.startswith("**") else f"**/{p}"


def as_number(value: Any, default: float) -> float:
    """JSON numbers arrive as int/float, but bool is an int in Python
    and NaN sneaks in via json.loads."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return value


def resolve_path(root: str, path: str) -> str:
    """Resolve `path` inside `root` and refuse anything that escapes it.

    Set MYPI_ALLOW_OUTSIDE=1 to opt out when you deliberately point mypi at
    a tree the agent is allowed to roam outside of.
    """
    return _contained(os.path.realpath(os.path.join(root, path)), root, path)


def _contained(target: str, root: str, path: str) -> str:
    root_abs = os.path.realpath(root)
    if os.environ.get("MYPI_ALLOW_OUTSIDE"):
        return target
    if target != root_abs and not target.startswith(root_abs + os.sep):
        raise ValueError(
            f"path {path!r} is outside the working folder; set MYPI_ALLOW_OUTSIDE=1 to allow it"
        )
    return target


_cwd_source: Callable[[], str | None] | None = None


def set_cwd_source(source: Callable[[], str | None]) -> None:
    """bash registers where its shell is, so a `cd` carries over to the fs tools."""
    global _cwd_source
    _cwd_source = source


def workspace_root() -> str:
    """Folder relative paths start from: the shell's cwd while a session is live."""
    if _cwd_source is None:
        return os.getcwd()
    return _cwd_source() or os.getcwd()


def workspace_path(path: str) -> str:
    """Resolve `path` from wherever the shell is now, contained to mypi's folder.

    Relative paths follow `cd`, but the boundary never moves: cd'ing somewhere else
    must not widen what the agent is allowed to touch.
    """
    base = workspace_root()
    target = os.path.realpath(path if os.path.isabs(path) else os.path.join(base, path))
    return _contained(target, os.getcwd(), path)


def cap_output(text: str, max_bytes: int = MAX_OUTPUT_BYTES, note: str = "output") -> str:
    """Keep the head and the tail so both the start and the error survive."""
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return text
    half = max_bytes // 2
    head = raw[:half].decode("utf-8", errors="ignore")
    tail = raw[-half:].decode("utf-8", errors="ignore")
    summary = f"{note} truncated: {len(raw)} bytes total, showing first and last {half}"
    return f"{head}\n\n[{summary}]\n\n{tail}"


@dataclass
class WalkEntry:
    abs_path: str
    rel_path: str
    is_directory: bool
    is_symlink: bool = False


def label(entry: WalkEntry) -> str:
    """Trailing / for a directory, @ for a symlink, so the model can tell them apart."""
    name = entry.rel_path + ("/" if entry.is_directory and not entry.is_symlink else "")
    return name + "@" if entry.is_symlink else name


def _ignore_predicate(ignores: list[str]) -> Callable[[str, str], bool]:
    """Match on name (case-insensitively) or on a glob against the name or the rel path."""
    names = {i.lower() for i in ignores if not any(c in i for c in "*?[")}
    patterns = [i for i in ignores if any(c in i for c in "*?[")]

    def ignored(name: str, rel: str) -> bool:
        if name.lower() in names:
            return True
        return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(rel, p) for p in patterns)

    return ignored


def walk(
    root: str,
    max_depth: float = float("inf"),
    include_hidden: bool = False,
    ignores: list[str] | None = None,
) -> Iterator[WalkEntry]:
    """Depth-first walk yielding files and directories.

    Symlinks are reported but never followed: a link to a parent directory would
    otherwise loop forever.
    """
    ignored = _ignore_predicate(DEFAULT_IGNORES if ignores is None else ignores)

    def recurse(directory: str, depth: int) -> Iterator[WalkEntry]:
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name.lower())
        except OSError:
            return
        for entry in entries:
            if not include_hidden and entry.name.startswith("."):
                continue
            rel = os.path.relpath(entry.path, root).replace(os.sep, "/")
            if ignored(entry.name, rel):
                continue
            is_link = entry.is_symlink()
            # a link to a directory counts as a directory for the caller, but we stop here
            is_dir = entry.is_dir()
            yield WalkEntry(entry.path, rel, is_dir, is_link)
            if is_dir and not is_link and depth < max_depth:
                yield from recurse(entry.path, depth + 1)

    yield from recurse(root, 1)

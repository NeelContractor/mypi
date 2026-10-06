from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from typing import Any
from uuid import uuid4

from ..types import Tool
from .fsutil import as_number, set_cwd_source

MAX_LINES = 2000
MAX_BYTES = 50 * 1024


def find_bash() -> str:
    """On Windows prefer Git Bash by full path; plain 'bash' can be the WSL launcher."""
    if os.name != "nt":
        return "bash"
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", ""), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Git", "bin", "bash.exe"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d or d.rstrip("\\/").lower().endswith("system32"):
            continue
        cand = os.path.join(d, "bash.exe")
        if os.path.exists(cand):
            return cand
    return "bash"


def truncate_tail(text: str, max_lines: int = MAX_LINES, max_bytes: int = MAX_BYTES) -> str:
    """Keep the tail of the output (results sit at the bottom) and say what was cut."""
    lines = text.split("\n")
    kept = lines[-max_lines:] if len(lines) > max_lines else lines
    out = "\n".join(kept)
    raw = out.encode("utf-8", errors="replace")
    if len(raw) > max_bytes:
        out = raw[-max_bytes:].decode("utf-8", errors="ignore")
        kept = out.split("\n")
    if len(kept) == len(lines) and len(out) == len(text):
        return text
    return f"[truncated: showing the last {len(kept)} of {len(lines)} lines]\n{out}"


def _parse_sentinel(line: bytes) -> tuple[str, str | None]:
    """Split the `|code|cwd` tail the sentinel marker is glued to.

    cwd is last and may contain spaces, so split from the front exactly once.
    """
    tail = line.decode("utf-8", "replace").strip()
    if tail.startswith("|"):
        tail = tail[1:]
    code, sep, cwd = tail.partition("|")
    if not sep:
        return (tail.rsplit(" ", 1)[-1] or "-1"), None  # older/odd shell output
    return (code or "-1"), (cwd or None)


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except OSError:  # already gone, or not ours to kill
        pass


def _quote(text: str) -> str:
    """Single-quote for bash: wrap in ' and close/escape/reopen around each '."""
    return "'" + text.replace("'", "'\\''") + "'"


class BashSession:
    """One long-lived `bash` reading commands from stdin, so cd/exports/source persist.

    Each command is eval'd with its own stdin on /dev/null, otherwise a command like
    `cat` would swallow the rest of this protocol and desync the session for good.
    A command that overruns its timeout takes the shell down with it (the reader can no
    longer tell where the abandoned output ends), so the session is rebuilt rather than
    left in an unknown state.
    """

    HELPER = '__mypi_run() { eval "$1" </dev/null; }'

    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None
        self.cwd: str | None = None  # where the shell is, following its own cd
        self.root: str | None = None  # where mypi was when the shell started

    async def _start(self) -> asyncio.subprocess.Process:
        extra: dict[str, Any] = {} if os.name == "nt" else {"start_new_session": True}
        self.root = os.getcwd()
        self.cwd = self.root
        self.proc = await asyncio.create_subprocess_exec(
            find_bash(),
            "-s",
            cwd=self.cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **extra,
        )
        if self.proc.stdin and self.proc.stdout:
            self.proc.stdin.write(f"{self.HELPER}\n".encode())
            await self.proc.stdin.drain()
        return self.proc

    async def _alive(self) -> asyncio.subprocess.Process:
        if self.proc is None or self.proc.returncode is not None:
            return await self._start()
        if self.root != os.getcwd():  # mypi's folder moved under us, e.g. a new shell
            await self.close()
            return await self._start()
        return self.proc

    async def run(self, command: str, timeout_s: float) -> str:
        proc = await self._alive()
        if not proc.stdin or not proc.stdout:
            await self.close()
            raise OSError("the persistent shell has no pipes")
        marker = f"__mypi_{uuid4().hex}__"
        # the sentinel carries the exit code and $PWD, so the fs tools can follow a cd;
        # the command itself is quoted, never eval'd raw
        script = f"__mypi_run {_quote(command)}\nprintf '%s|%s|%s\\n' {marker} $? \"$PWD\"\n"
        try:
            proc.stdin.write(script.encode())
            await proc.stdin.drain()
            # readuntil returns the separator too, so the echoed command never leaks out
            raw = await asyncio.wait_for(proc.stdout.readuntil(marker.encode()), timeout_s)
            raw = raw[: -len(marker)]
            code_line = await asyncio.wait_for(proc.stdout.readline(), 5)
        except TimeoutError:
            await self.close()
            return (
                f"[timed out after {timeout_s:g}s; the shell was restarted, so any cd or "
                f"exported variables from before are gone]"
            )
        except asyncio.IncompleteReadError:
            await self.close()
            return "[the command ended the shell (exit/exec); it has been restarted]"
        except (asyncio.LimitOverrunError, BrokenPipeError, ConnectionResetError, OSError):
            await self.close()
            return "[lost the shell; it has been restarted]"

        text = raw.decode("utf-8", errors="replace")
        code, new_cwd = _parse_sentinel(code_line)
        if new_cwd:
            self.cwd = new_cwd
        if text.endswith("\n"):
            text = text[:-1]
        return f"{text}\n[exit code: {code}]" if text else f"[exit code: {code}]"

    async def close(self) -> None:
        proc, self.proc = self.proc, None
        self.cwd = None
        self.root = None
        if proc is None or proc.returncode is not None:
            return
        try:
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        try:
            await asyncio.wait_for(proc.wait(), 2)
        except (TimeoutError, asyncio.CancelledError):
            _kill_tree(proc)


_SESSION = BashSession()
set_cwd_source(lambda: _SESSION.cwd)


async def shutdown_bash() -> None:
    await _SESSION.close()


async def run_bash(command: str, timeout_s: float) -> str:
    """One-shot `bash -c`, used when the persistent shell is unavailable."""
    extra: dict[str, Any] = {} if os.name == "nt" else {"start_new_session": True}
    try:
        proc = await asyncio.create_subprocess_exec(
            find_bash(),
            "-c",
            command,
            cwd=os.getcwd(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **extra,
        )
    except OSError as e:
        return f"failed to start bash: {e}"

    timed_out = False
    try:
        data, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
    except TimeoutError:
        timed_out = True
        _kill_tree(proc)
        data, _ = await proc.communicate()

    out = data.decode("utf-8", errors="replace")
    if timed_out:
        out += f"\n[timed out after {timeout_s:g}s]"
    return f"{out}\n[exit code: {proc.returncode}]"


async def _execute(args: dict[str, Any]) -> str:
    seconds = as_number(args.get("timeout"), 30)
    if seconds <= 0:
        return "error: timeout must be greater than 0"
    try:
        out = await _SESSION.run(str(args["command"]), seconds)
    except OSError as e:
        out = await run_bash(str(args["command"]), seconds)
        out += f"\n[persistent shell unavailable ({e}); ran in a one-shot shell]"
    return truncate_tail(out)


bash_tool = Tool(
    name="bash",
    description=(
        "Run a bash command in the current folder. One shell is reused for the whole "
        "session, so cd, exported variables and sourced files persist between calls. "
        "Returns stdout, stderr and the exit code."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to run"},
            "timeout": {"type": "number", "description": "Timeout in seconds (default 30)"},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    execute=_execute,
)

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

from . import viewer
from .agent.loop import AgentEvent, RunEnd, TextEvent, ToolEnd, ToolStart, TurnEnd, run_agent
from .colors import ERROR, MUTED, TOOL, Colors
from .providers import DEFAULT_MAX_TOKENS, PROVIDERS, get_provider
from .system import SYSTEM
from .tools import tools
from .tools.bash import shutdown_bash
from .types import Message, UserMessage

# cwd first so a project can override the install's .env, then the one next to the code
load_dotenv()
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mypi",
        description="A small coding agent that works in your current folder.",
    )
    parser.add_argument("-p", "--prompt", help="what you want done")
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS),
        default=os.environ.get("MODEL_PROVIDER") or "anthropic",
        help="who to talk to (default: %(default)s; set MODEL_PROVIDER in .env)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME") or None,
        help="the model to use (default: the provider's own; set MODEL_NAME in .env)",
    )
    parser.add_argument("--system", help="replace the system prompt")
    parser.add_argument("--no-system", action="store_true", help="send no system prompt at all")
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="output token budget per turn"
    )
    parser.add_argument(
        "--max-turns", type=int, default=20, help="give up after this many model turns"
    )
    parser.add_argument(
        "--max-continuations",
        type=int,
        default=3,
        help="how many times to ask a cut-off turn to carry on",
    )
    parser.add_argument(
        "--stream-timeout",
        type=float,
        default=300.0,
        help="seconds to wait for the model before giving up on a turn",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="print every tool call's full output (default: one line per call)",
    )
    parser.add_argument("--debug", action="store_true", help="show the traceback on failure")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def render(
    provider_name: str,
    model: str,
    expand: bool = False,
    record: list[viewer.Block] | None = None,
) -> Callable[[AgentEvent], None]:
    state = {"started": False, "prose": False, "at_bol": True}
    c = Colors()

    def emit(prefix: str, text: str = "", end: str = "\n", flush: bool = False) -> None:
        if not state["started"]:
            # the first thing printed shouldn't be preceded by blank lines
            prefix = prefix.lstrip("\n")
            state["started"] = True
        printed = prefix + text
        print(printed, end=end, flush=flush)
        state["at_bol"] = end == "\n" or printed.endswith("\n")

    def gap() -> str:
        """Exactly one blank line between sections, from wherever the cursor is."""
        return "\n" if state["at_bol"] else "\n\n"

    def on_event(event: AgentEvent) -> None:
        if isinstance(event, TextEvent):
            if not state["prose"]:
                # the answer is its own paragraph: it must not start on the line
                # right under the last tool result
                if state["started"]:
                    emit(gap(), end="")
                state["prose"] = True
            emit("", event.delta, end="", flush=True)
            if record is not None:
                record.append(viewer.Written(event.delta))
        elif isinstance(event, ToolStart):
            state["prose"] = False
            name = c.paint(event.call.name, TOOL, bold=True)
            if expand:
                args = c.paint(f" {event.call.arguments}", MUTED)
                emit(f"{gap()} {name}{args}")
            else:
                # folded: the arguments belong to the summary line that follows,
                # so the running header only names the tool
                emit(f"{gap()} {name}")
        elif isinstance(event, ToolEnd):
            # name the call on every result: parallel calls finish out of order,
            # so a bare block can't be matched back to its header
            state["prose"] = False
            name, body = event.call.name, event.result
            if record is not None:
                record.append(viewer.ToolCall(name, event.call.arguments, body, event.is_error))
            if event.is_error:
                head, *rest = body.split("\n", 1)
                if rest:
                    emit(f"\n {c.paint(f'{name} failed', ERROR, bold=True)}")
                    emit("", c.paint(_indent(body), ERROR))
                else:
                    emit(f"\n {c.paint(f'{name} failed: {head}', ERROR)}")
            elif not expand:
                # the accordion: one line per tool call; --expand opens them all
                folded = viewer.ToolCall(name, event.call.arguments, body, False)
                emit(f" {viewer.fold_header(folded, c)}")
            elif not body.strip():
                emit(f"\n {c.paint(name, TOOL, bold=True)}: {c.paint('(no output)', MUTED)}")
            elif "\n" not in body.strip():
                # a status message ("created X") is its own summary; counting it is noise
                emit(f"\n {c.paint(name, TOOL, bold=True)}: {c.paint(body.strip(), MUTED)}")
            else:
                emit(
                    f"\n {c.paint(name, TOOL, bold=True)}: "
                    f"{c.paint(f'{_count_lines(body)} lines', MUTED)}"
                )
                emit("", c.paint(_indent(body), MUTED))
        elif isinstance(event, TurnEnd):
            state["prose"] = False
            m = event.message
            summary = (
                f"  {provider_name} · {model} · "
                f"{m.usage.input} in / {m.usage.output} out tokens · "
                f"{_say_stop_reason(m.stop_reason)}"
            )
            emit(f"{gap()}{c.paint(summary, MUTED)}")
        elif isinstance(event, RunEnd):
            if event.reason != "the model finished":
                emit(f"{gap()}{c.paint(f'  stopped: {event.reason}', ERROR)}")

    return on_event


_NOTE = re.compile(r"\[showing [^\]]*\]")


def _say_stop_reason(reason: str) -> str:
    """Turn the API's stop reasons into something a reader can act on."""
    return {
        "toolUse": "tool call",
        "length": "hit the token budget",
        "stop": "finished",
    }.get(reason, reason)


def _content_lines(body: str) -> list[str]:
    """Split into display lines, dropping the artifact after a trailing newline."""
    lines = body.split("\n")
    if len(lines) > 1 and lines[-1] == "":
        lines = lines[:-1]
    return lines


def _count_lines(body: str) -> int:
    """How many content lines, ignoring a trailing [showing ...] status note."""
    lines = _content_lines(body)
    if lines and _NOTE.fullmatch(lines[-1].strip()):
        lines = lines[:-1]
    return len(lines)


def _indent(text: str, limit: int = 40) -> str:
    lines = _content_lines(text)
    if len(lines) > limit:
        lines = [*lines[:limit], f"... ({len(lines) - limit} more lines)"]
    return "\n".join(f"  {line}" for line in lines)


async def amain(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    c = Colors()
    if not args.prompt:
        print(
            c.paint("no prompt: ", ERROR, bold=True)
            + 'mypi -p "fix the failing test" [--provider '
            + "|".join(PROVIDERS)
            + "]",
            file=sys.stderr,
        )
        return 2
    if args.no_system:
        system = None
    else:
        system = args.system or SYSTEM

    try:
        provider = get_provider(args.provider, args.max_tokens)
    except ValueError as e:
        print(c.paint("error: ", ERROR, bold=True) + str(e), file=sys.stderr)
        return 2

    model = args.model or provider.default_model
    messages: list[Message] = [UserMessage(args.prompt)]
    record: list[viewer.Block] = []
    try:
        await run_agent(
            provider=provider,
            model=model,
            tools=tools,
            messages=messages,
            on_event=render(provider.name, model, expand=args.expand, record=record),
            system=system,
            max_turns=args.max_turns,
            max_continuations=args.max_continuations,
            stream_timeout=args.stream_timeout,
        )
        if not args.expand:
            _maybe_explore(record)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as e:
        if args.debug:
            raise
        print(c.paint("\nerror: ", ERROR, bold=True) + str(e), file=sys.stderr)
        print("       re-run with --debug for the full traceback", file=sys.stderr)
        return 1
    finally:
        await shutdown_bash()
        close = getattr(provider, "aclose", None)
        if close is not None:
            await close()
    return 0


def main() -> None:
    try:
        sys.exit(asyncio.run(amain()))
    except KeyboardInterrupt:
        sys.exit(130)


def _maybe_explore(record: list[viewer.Block]) -> None:
    """After a run on a real terminal, reopen the tool calls to click on.

    No prompt, no keypress: the viewer just takes the alternate screen so the
    folded lines become clickable. Piping skips it entirely.
    """
    if not any(isinstance(b, viewer.ToolCall) and not b.is_error for b in record):
        return
    tty_out = getattr(sys.stdout, "isatty", lambda: False)()
    tty_in = getattr(sys.stdin, "isatty", lambda: False)()
    if not (tty_out and tty_in):
        return
    viewer.run(record)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

from .agent.loop import AgentEvent, RunEnd, TextEvent, ToolEnd, ToolStart, TurnEnd, run_agent
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
    parser.add_argument("--debug", action="store_true", help="show the traceback on failure")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def render(provider_name: str, model: str) -> Callable[[AgentEvent], None]:
    state = {"started": False}

    def emit(prefix: str, text: str = "", end: str = "\n", flush: bool = False) -> None:
        if not state["started"]:
            # the first thing printed shouldn't be preceded by blank lines
            prefix = prefix.lstrip("\n")
            state["started"] = True
        print(prefix + text, end=end, flush=flush)

    def on_event(event: AgentEvent) -> None:
        if isinstance(event, TextEvent):
            emit("", event.delta, end="", flush=True)
        elif isinstance(event, ToolStart):
            emit(f"\n {event.call.name} {event.call.arguments}")
        elif isinstance(event, ToolEnd):
            # name the call on every result: parallel calls finish out of order,
            # so a bare block can't be matched back to its header
            name, body = event.call.name, event.result
            if event.is_error:
                head, *rest = body.split("\n", 1)
                if rest:
                    emit(f"\n {name} failed")
                    emit("", _indent(body))
                else:
                    emit(f"\n {name} failed: {head}")
            elif not body.strip():
                emit(f"\n {name}: (no output)")
            elif "\n" not in body.strip():
                # a status message ("created X") is its own summary; counting it is noise
                emit(f"\n {name}: {body.strip()}")
            else:
                emit(f"\n {name}: {_count_lines(body)} lines")
                emit("", _indent(body))
        elif isinstance(event, TurnEnd):
            m = event.message
            emit(
                f"\n\n  {provider_name} ... {model} ... "
                f"{m.usage.input} in / {m.usage.output} out ... {m.stop_reason}"
            )
        elif isinstance(event, RunEnd) and event.reason != "the model finished":
            emit(f"\n  stopped: {event.reason}")

    return on_event


_NOTE = re.compile(r"\[showing [^\]]*\]")


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
    if not args.prompt:
        print(
            'no prompt: mypi -p "fix the failing test" [--provider ' + "|".join(PROVIDERS) + "]",
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
        print(f"error: {e}", file=sys.stderr)
        return 2

    model = args.model or provider.default_model
    messages: list[Message] = [UserMessage(args.prompt)]
    try:
        await run_agent(
            provider=provider,
            model=model,
            tools=tools,
            messages=messages,
            on_event=render(provider.name, model),
            system=system,
            max_turns=args.max_turns,
            max_continuations=args.max_continuations,
            stream_timeout=args.stream_timeout,
        )
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as e:
        if args.debug:
            raise
        print(f"\nerror: {e}", file=sys.stderr)
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


if __name__ == "__main__":
    main()

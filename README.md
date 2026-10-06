# mypi

A small coding agent. Point it at a folder and a task; it reads, searches and runs
commands there until the job is done or it runs out of road.

```
mypi -p "find every place we swallow an exception and tell me which are silent"
```

## Install

```
uv sync
cp .env.example .env      # add your key
```

## Use

```
mypi -p "prompt" [options]
```

| Option | Meaning |
| --- | --- |
| `-p`, `--prompt` | what you want done |
| `--provider` | `anthropic` (default), `anthropic-openai`, `groq` |
| `--model` | the model to use; defaults to `MODEL_NAME`, else the provider's own |
| `--system` | replace the system prompt |
| `--no-system` | send no system prompt |
| `--max-tokens` | output token budget per turn (default 8192) |
| `--max-turns` | give up after this many model turns (default 20) |
| `--max-continuations` | how many times to ask a cut-off turn to carry on (default 3) |
| `--stream-timeout` | seconds to wait on a stalled turn (default 300) |
| `--debug` | show the traceback instead of a one-line error |

Keys and defaults are read from the environment or `.env` — the one in your current
folder first, then the one next to the code. `ANTHROPIC_API_KEY` or `GROQ_API_KEY`,
depending on provider.

To pick a model without touching the code, set these in `.env`:

```sh
MODEL_PROVIDER=groq        # becomes the default for --provider
MODEL_NAME=gemma2-9b-it    # becomes the default for --model
```

A flag on the command line always wins over `.env`, and a bad `MODEL_PROVIDER` is
caught before any request goes out, with the valid choices listed.

## Tools

The agent has eight tools and nothing else.

| Tool | Notes |
| --- | --- |
| `bash` | one shell reused for the whole run, so `cd`, `export` and `source` persist — and the other tools follow the shell's folder. Output capped at 2000 lines / 50KB, keeping the tail. |
| `read` | a file, or a window of it via `offset` / `limit`. Capped at 50KB. |
| `write` | create a file or replace all of it; missing parent folders are created. |
| `edit` | replace `old_string` with `new_string`. It must match exactly and occur once, so the tool refuses rather than guessing; `replace_all` opts into multiple hits. |
| `ls` | recursive listing with `maxDepth`, `dirsOnly`, `includeHidden`. Trailing `/` marks a directory, `@` a symlink. |
| `grep` | regex over file contents, `file:line:text`, like `grep -rn`. Skips binaries and files over 5MB. |
| `find` | by name substring, glob or regex. |
| `stat` | size, type, timestamps, permissions, and whether it is a symlink. |

Two things move together, and it's worth keeping them apart:

- **Relative paths start at the shell's folder.** After `cd src`, `read app.py` and
  `write app.py` both mean `src/app.py`.
- **The boundary never moves.** Everything is still contained to the folder mypi started
  in, so `cd /etc` does not widen what the agent can touch — the next path simply
  refuses. Set `MYPI_ALLOW_OUTSIDE=1` to lift the boundary entirely.

`.git`, `node_modules`, `dist`, `build`, `.next`, `coverage`, `__pycache__` and `.venv`
are skipped, case-insensitively. Hidden files are skipped unless `includeHidden` is set.

## Design

`agent/loop.py` is the whole agent: stream a turn, run whatever tools it asked for,
feed the results back, repeat. Independent tool calls in one turn run concurrently.

A turn that hits the token limit is not treated as the end of the answer. Any tool calls it
finished sending still run; the model is then asked to carry on and gets `--max-continuations`
attempts before giving up. A call the model never finished sending is closed out with an
`interrupted` error result instead of being run with missing arguments.

A turn that stops without asking for a tool is the answer, so every run ends with a
`RunEnd` saying why it stopped — finished, out of turns, or still cut off.

Providers are one small adapter each: `anthropic_provider.py` and `openai_compat.py`,
which covers anything speaking the OpenAI chat-completions API.

## Develop

```
uv sync --group dev
uv run pytest
```

No key or network needed: the tests drive the loop and both provider adapters against
fake streams.

`uv run pytest` also runs Ruff (lint + format) and `mypy --strict`, so one command
proves the repo is healthy. Run them on their own if you prefer:

```
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy
```

The type checker covers `src/` only; the tests are deliberately left out of it.

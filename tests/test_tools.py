import asyncio
import os

import pytest

from mypi.tools import bash, edit, find, grep, ls, read, stat, write
from mypi.tools.bash import shutdown_bash, truncate_tail


@pytest.fixture(autouse=True)
async def _close_shell():
    """The bash session is module-level and process-wide; each test gets its own loop."""
    yield
    await shutdown_bash()


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A small repo, with mypi's cwd pointed at it."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("import os\nprint('hi')\n", encoding="utf-8")
    (tmp_path / "src" / "util.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    (tmp_path / "big.log").write_text("\n".join(f"line{i}" for i in range(5000)), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MYPI_ALLOW_OUTSIDE", raising=False)
    return tmp_path


# --- containment (issue 9) ---


@pytest.mark.parametrize(
    "tool",
    [
        lambda p: read._execute({"path": p}),
        lambda p: ls._execute({"path": p}),
        lambda p: grep._execute({"pattern": "x", "path": p}),
        lambda p: find._execute({"name": "x", "path": p}),
    ],
)
async def test_tools_refuse_to_escape_cwd(tree, tool) -> None:
    for bad in ["..", "/etc", "../.."]:
        with pytest.raises(ValueError, match="outside the working folder"):
            await tool(bad)


async def test_read_refuses_absolute_path(tree) -> None:
    with pytest.raises(ValueError, match="outside the working folder"):
        await read._execute({"path": "/etc/hosts"})


# --- read caps and windowing (issue 8) ---


async def test_read_window(tree) -> None:
    out = await read._execute({"path": "big.log", "offset": 10, "limit": 3})
    assert out.startswith("line9\nline10\nline11")
    assert "more lines follow" in out


async def test_read_caps_huge_file(tree, monkeypatch) -> None:
    monkeypatch.setattr(read, "MAX_BYTES", 100)
    out = await read._execute({"path": "big.log"})
    assert "bigger than 100 bytes" in out


async def test_read_offset_past_eof(tree) -> None:
    assert "no lines at offset" in await read._execute({"path": "README.md", "offset": 99})


# --- number coercion (issues 23, 24) ---


async def test_bool_is_not_a_number(tree) -> None:
    # True must read as "not supplied", never as 1 second or depth 1
    assert await ls._execute({"maxDepth": True}) == await ls._execute({})
    assert "[exit code: 0]" in await bash._execute({"command": "sleep 1", "timeout": True})
    # falls back to the default cap instead of erroring out on a bool
    assert "README.md:1:hello" in await grep._execute({"pattern": "hello", "maxMatches": True})


async def test_grep_rejects_zero_max_matches(tree) -> None:
    out = await grep._execute({"pattern": "hello", "maxMatches": 0})
    assert out == "error: maxMatches must be at least 1"


async def test_bash_rejects_non_positive_timeout(tree) -> None:
    assert (
        await bash._execute({"command": "true", "timeout": 0})
        == "error: timeout must be greater than 0"
    )
    assert (
        await bash._execute({"command": "true", "timeout": -5})
        == "error: timeout must be greater than 0"
    )


# --- find validation (issue 25) ---


async def test_find_rejects_unknown_type(tree) -> None:
    out = await find._execute({"name": "app", "type": "bogus"})
    assert out.startswith("error: type must be one of file, dir")


async def test_find_requires_a_criterion(tree) -> None:
    assert "at least one of" in await find._execute({})


# --- implicit ** so bare globs match at any depth (issue 11) ---


async def test_bare_glob_matches_nested(tree) -> None:
    assert "src/app.py" in await find._execute({"glob": "*.py"})
    assert "src/app.py" in await grep._execute({"pattern": "import", "glob": "*.py"})
    assert "src/app.py" not in await find._execute({"glob": "*.md"})


async def test_glob_everything(tree) -> None:
    out = await find._execute({"glob": "**"})
    assert "README.md" in out and "src/app.py" in out


async def test_find_type_filter_still_works(tree) -> None:
    assert await find._execute({"name": "src", "type": "dir"}) == "src/"
    assert await find._execute({"name": "src", "type": "file"}) == "(no matches)"


async def test_ls_max_depth(tree) -> None:
    assert (await ls._execute({"maxDepth": 1})).splitlines() == ["big.log", "README.md", "src/"]
    assert "src/app.py" in await ls._execute({"maxDepth": 2})


async def test_bash_runs_and_reports_exit_code(tree) -> None:
    out = await bash._execute({"command": "echo out; echo err 1>&2; (exit 3)"})
    assert "out" in out and "err" in out and "[exit code: 3]" in out


async def test_bash_exit_code_does_not_leak_the_sentinel(tree) -> None:
    out = await bash._execute({"command": "echo hi"})
    assert "__mypi_" not in out, "the internal sentinel leaked into the tool output"
    assert out == "hi\n[exit code: 0]"


async def test_bash_keeps_one_shell_between_calls(tree) -> None:
    await bash._execute({"command": "export MYPI_TEST_VAR=persisted"})
    await bash._execute({"command": "cd /tmp"})
    out = await bash._execute({"command": "echo $MYPI_TEST_VAR $(pwd)"})
    assert out == "persisted /tmp\n[exit code: 0]"


async def test_bash_restart_follows_a_changed_working_folder(tree) -> None:
    await bash._execute({"command": "pwd"})
    (tree / "sub").mkdir()
    os.chdir(tree / "sub")
    assert await bash._execute({"command": "pwd"}) == f"{tree / 'sub'}\n[exit code: 0]"


async def test_bash_survives_a_command_that_reads_stdin(tree) -> None:
    """`cat` would eat the session protocol if the command shared our stdin."""
    assert await bash._execute({"command": "cat", "timeout": 2}) == "[exit code: 0]"
    assert "still here" in await bash._execute({"command": "echo still here"})


async def test_bash_timeout_restarts_and_says_so(tree) -> None:
    out = await bash._execute({"command": "sleep 5", "timeout": 1})
    assert "timed out after 1s" in out and "restarted" in out
    assert "back" in await bash._execute({"command": "echo back"})


async def test_bash_command_that_ends_the_shell(tree) -> None:
    out = await bash._execute({"command": "exit 3"})
    assert "ended the shell" in out
    assert "alive" in await bash._execute({"command": "echo alive"})


async def test_bash_quotes_survive(tree) -> None:
    out = await bash._execute({"command": "echo 'it'\"'\"'s' && printf 'a\\nb\\n'"})
    assert out.startswith("it's")


async def test_truncate_tail_is_linear_and_keeps_the_end() -> None:
    big = "\n".join(f"line{i}" for i in range(20_000))
    out = truncate_tail(big, max_lines=5)
    assert out.startswith("[truncated: showing the last 5 of 20000 lines]")
    assert out.endswith("line19999")


async def test_truncate_tail_handles_one_huge_line() -> None:
    out = truncate_tail("z" * 200_000, max_bytes=1000)
    assert len(out) < 1200 and out.endswith("z")


async def test_tools_are_registered() -> None:
    from mypi.tools import tools

    assert [t.name for t in tools] == [
        "bash",
        "read",
        "write",
        "edit",
        "ls",
        "grep",
        "stat",
        "find",
    ]
    for tool in tools:
        assert tool.parameters["type"] == "object"
        assert tool.parameters["additionalProperties"] is False
        assert asyncio.iscoroutinefunction(tool.execute)


# --- ignores and symlinks (issues 28, 29) ---


async def test_ignores_are_case_insensitive(tree) -> None:
    (tree / "NODE_MODULES").mkdir()
    (tree / "NODE_MODULES" / "junk.js").write_text("x", encoding="utf-8")
    out = await ls._execute({})
    assert "NODE_MODULES" not in out
    assert not any("junk.js" in line for line in out.splitlines())


async def test_ignore_globs(tree) -> None:
    from mypi.tools.fsutil import walk

    (tree / "a.log").write_text("x", encoding="utf-8")
    (tree / "keep.py").write_text("x", encoding="utf-8")
    names = {e.rel_path for e in walk(str(tree), ignores=["*.log", "keep*"])}
    assert "a.log" not in names and "keep.py" not in names
    assert "README.md" in names


async def test_symlinks_are_labelled_and_not_followed(tree) -> None:
    (tree / "link_to_src").symlink_to(tree / "src", target_is_directory=True)
    (tree / "link_to_file").symlink_to(tree / "README.md")
    (tree / "loop").symlink_to(tree, target_is_directory=True)  # would recurse forever

    out = (await ls._execute({})).splitlines()
    assert "link_to_src@" in out and "link_to_file@" in out
    assert not any(line.count("loop/") for line in out), "walk followed a symlink loop"

    found = (await find._execute({"regex": r"^(link_.*|loop)$"})).splitlines()
    assert sorted(found) == ["link_to_file@", "link_to_src@", "loop@"]


async def test_stat_reports_symlinks(tree, monkeypatch) -> None:
    os.symlink(tree / "README.md", tree / "link.md")
    out = await stat._execute({"path": "link.md"})
    assert "symlink: yes ->" in out
    assert "type: file" in out
    assert "symlink: no" in await stat._execute({"path": "README.md"})


async def test_stat_reports_directories(tree) -> None:
    out = await stat._execute({"path": "src"})
    assert "type: directory" in out


# --- the fs tools must not block the loop (gather has to actually parallelise) ---


async def test_grep_does_not_block_the_event_loop(tree) -> None:
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(ticker())
    await asyncio.gather(
        grep._execute({"pattern": "line", "path": "."}),
        grep._execute({"pattern": "line", "path": "."}),
        grep._execute({"pattern": "line", "path": "."}),
    )
    task.cancel()
    assert ticks > 10, f"the loop only ticked {ticks} times; grep is blocking it"


# --- shared cwd: a `cd` in bash carries over to the fs tools ---


async def test_cd_carries_over_to_the_fs_tools(tree) -> None:
    from mypi.tools.fsutil import workspace_root

    before = set((await ls._execute({})).splitlines())
    assert "src/app.py" in before and "app.py" not in before

    assert "[exit code: 0]" in await bash._execute({"command": "cd src"})
    assert workspace_root().endswith("/src")

    after = set((await ls._execute({})).splitlines())
    assert {"app.py", "util.py"} <= after
    assert not any(e.startswith("src/") for e in after), after
    assert (await read._execute({"path": "util.py"})).strip() == "VALUE = 1"


async def test_cd_cannot_widen_the_containment_boundary(tree) -> None:
    assert "[exit code: 0]" in await bash._execute({"command": "cd .."})
    with pytest.raises(ValueError, match="outside the working folder"):
        await read._execute({"path": "README.md"})


async def test_write_follows_the_shell_cwd(tree) -> None:
    await bash._execute({"command": "cd src"})
    out = await write._execute({"path": "made.py", "content": "# written in src\n"})
    assert "created" in out and "/src/" in out
    assert (tree / "src" / "made.py").exists()
    assert not (tree / "made.py").exists()


def test_workspace_root_falls_back_without_a_shell(tree) -> None:
    from mypi.tools.fsutil import workspace_root

    assert workspace_root() == os.getcwd()


# --- write ---


async def test_write_creates_then_replaces(tree) -> None:
    out = await write._execute({"path": "new.txt", "content": "one\ntwo\n"})
    assert "created" in out
    assert (tree / "new.txt").read_text(encoding="utf-8") == "one\ntwo\n"

    out = await write._execute({"path": "new.txt", "content": "three"})
    assert "replaced" in out
    assert (tree / "new.txt").read_text(encoding="utf-8") == "three"


async def test_write_creates_parent_folders(tree) -> None:
    await write._execute({"path": "a/b/c.txt", "content": "x"})
    assert (tree / "a" / "b" / "c.txt").read_text(encoding="utf-8") == "x"


async def test_write_refuses_to_escape(tree) -> None:
    for bad in ["../out.txt", "/etc/pwned", "a/../../out.txt"]:
        with pytest.raises(ValueError, match="outside the working folder"):
            await write._execute({"path": bad, "content": "x"})


async def test_write_requires_both_fields(tree) -> None:
    assert "path and content are both required" in await write._execute({"path": "x"})
    assert "content must be a string" in await write._execute({"path": "x", "content": 5})
    assert not (tree / "x").exists()


# --- edit ---


async def test_edit_replaces_a_unique_match(tree) -> None:
    out = await edit._execute(
        {"path": "src/util.py", "old_string": "VALUE = 1", "new_string": "VALUE = 2"}
    )
    assert "replaced 1 occurrence" in out
    assert (tree / "src" / "util.py").read_text(encoding="utf-8") == "VALUE = 2\n"


async def test_edit_reports_a_missing_match_and_changes_nothing(tree) -> None:
    target = tree / "src" / "util.py"
    out = await edit._execute({"path": "src/util.py", "old_string": "NOPE", "new_string": "x"})
    assert out.startswith("error:") and "not found" in out
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"


async def test_edit_refuses_an_ambiguous_match(tree) -> None:
    (tree / "dup.txt").write_text("x = 1\nx = 1\n", encoding="utf-8")
    out = await edit._execute({"path": "dup.txt", "old_string": "x = 1", "new_string": "x = 2"})
    assert "appears 2 times" in out
    assert (tree / "dup.txt").read_text(encoding="utf-8") == "x = 1\nx = 1\n"

    out = await edit._execute(
        {"path": "dup.txt", "old_string": "x = 1", "new_string": "x = 2", "replace_all": True}
    )
    assert "replaced 2 occurrences" in out
    assert (tree / "dup.txt").read_text(encoding="utf-8") == "x = 2\nx = 2\n"


async def test_edit_rejects_bad_input(tree) -> None:
    cases = [
        ({"path": "src/util.py", "old_string": "", "new_string": "x"}, "must not be empty"),
        (
            {"path": "src/util.py", "old_string": "VALUE = 1", "new_string": "VALUE = 1"},
            "identical",
        ),
        ({"path": "src/util.py", "old_string": 1, "new_string": "x"}, "must be strings"),
        ({"path": "missing.py", "old_string": "a", "new_string": "b"}, "no such file"),
        ({"path": "src", "old_string": "a", "new_string": "b"}, "is a directory"),
    ]
    for args, needle in cases:
        out = await edit._execute(args)
        assert out.startswith("error:") and needle in out, out
    assert (tree / "src" / "util.py").read_text(encoding="utf-8") == "VALUE = 1\n"


async def test_edit_leaves_non_utf8_alone(tree) -> None:
    (tree / "bin.dat").write_bytes(b"\xff\xfe\x00binary")
    out = await edit._execute({"path": "bin.dat", "old_string": "a", "new_string": "b"})
    assert "not valid UTF-8" in out
    assert (tree / "bin.dat").read_bytes() == b"\xff\xfe\x00binary"

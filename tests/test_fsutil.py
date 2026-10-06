import os
from pathlib import Path

import pytest

from mypi.tools.fsutil import as_number, cap_output, glob_to_regex, normalize_glob, resolve_path


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("**", "a.py", True),
        ("**", "src/a.py", True),
        ("**", "src/deep/nested/a.py", True),
        ("**/", "src/a.py", True),
        ("*.py", "a.py", True),
        ("*.py", "src/a.py", False),
        ("**/*.py", "src/a.py", True),
        ("src/**/*.py", "src/a.py", True),
        ("a?c.txt", "abc.txt", True),
        ("a?c.txt", "abbc.txt", False),
        ("*.test.py", "src/x/a.test.py", False),
    ],
)
def test_glob_to_regex(pattern: str, path: str, expected: bool) -> None:
    assert bool(glob_to_regex(pattern).match(path)) is expected


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("*.py", "**/*.py"),
        ("**/*.py", "**/*.py"),
        ("src/*.py", "src/*.py"),
        ("**", "**"),
        (".env", "**/.env"),
    ],
)
def test_normalize_glob(given: str, expected: str) -> None:
    assert normalize_glob(given) == expected


def test_normalize_glob_keeps_leading_dot() -> None:
    assert normalize_glob(".env") == "**/.env"


def test_as_number_rejects_bool() -> None:
    assert as_number(True, 30) == 30
    assert as_number(False, 30) == 30


def test_as_number_rejects_junk() -> None:
    assert as_number("5", 30) == 30
    assert as_number(None, 30) == 30
    assert as_number(float("nan"), 30) == 30
    assert as_number(float("inf"), 30) == 30


def test_as_number_passes_numbers() -> None:
    assert as_number(5, 30) == 5
    assert as_number(2.5, 30) == 2.5


def test_resolve_path_stays_inside_root(tmp_path: Path) -> None:
    root = str(tmp_path)
    assert resolve_path(root, "a/b.txt") == os.path.join(os.path.realpath(root), "a/b.txt")
    assert resolve_path(root, ".") == os.path.realpath(root)
    assert resolve_path(root, "a/../b.txt").endswith("b.txt")


def test_resolve_path_rejects_escape(tmp_path: Path) -> None:
    root = str(tmp_path / "root")
    os.makedirs(root)
    for bad in ["..", "../outside.txt", "/etc/hosts", "a/../../outside.txt"]:
        with pytest.raises(ValueError, match="outside the working folder"):
            resolve_path(root, bad)


def test_resolve_path_allows_opt_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = str(tmp_path / "root")
    os.makedirs(root)
    monkeypatch.setenv("MYPI_ALLOW_OUTSIDE", "1")
    assert resolve_path(root, "/etc/hosts") == os.path.realpath("/etc/hosts")


def test_cap_output_leaves_small_text_alone() -> None:
    assert cap_output("hello") == "hello"


def test_cap_output_keeps_head_and_tail() -> None:
    text = "A" * 40_000 + "B" * 40_000
    out = cap_output(text, max_bytes=1000)
    assert len(out.encode()) < 2000
    assert out.startswith("A")
    assert out.endswith("B")
    assert "truncated" in out

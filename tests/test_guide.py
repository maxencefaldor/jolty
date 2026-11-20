# Copyright 2026 Maxence Faldor
"""Verification that every Python block and quoted output in the guide and README runs.

Python blocks on a page execute sequentially in one shared namespace, allowing later
blocks to build on earlier ones. Each block is a script excerpt (as rumdl lints it)
pre-populated with `jolty`, `units`, `plan`, `specs`, a component builder `c`, and
`ground`. A block defining `plan` or `specs` invokes it like the CLI runner, whereas
excerpts omit `build` functions that construct standalone circuits. A trailing comment
containing a Python literal, such as `neighbors("E96", 1234.0)  # (1210.0, 1240.0)`,
asserts the expression's value.

Skipped blocks document the reason on the preceding line (`<!-- not run: why -->`).
Quoted output blocks list their shell commands on the preceding line
(`<!-- output of: commands -->`), separated by `;` and executed sequentially in a
staging directory holding copies of examples and `tests/guide` scripts. Quoted text
matches consecutive output lines, with standalone `...` matching any lines and inline
`...` matching arbitrary substrings such as build digests. Numeric values printed with
a decimal point or exponent match within one unit in the last quoted digit plus `ULPS`
floating-point ULPs, while all other characters match verbatim.
"""

import ast
import contextlib
import io
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tokenize
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

import jolty
from jolty import Builder, Role, plan, specs, units
from jolty.cli import main

ROOT = Path(__file__).parents[1]
"""The repository's root."""

PAGES = (
    ROOT / "README.md",
    *sorted((ROOT / "docs" / "guide").glob("*.md")),
    *sorted((ROOT / "docs" / "how-to").glob("*.md")),
)
"""The README, guide pages, and how-to pages whose blocks run."""

SCRIPTS = (ROOT / "examples", ROOT / "tests" / "guide")
"""The directories whose scripts the commands of quoted outputs run."""

BLOCK = re.compile(
    r"(?:^<!-- (?P<marker>not run|output of): (?P<note>[^\n]+) -->\n\n)?"
    r"^```(?P<language>python|text)\n(?P<body>.*?)^```$",
    re.MULTILINE | re.DOTALL,
)
"""A Python block or a quoted output, and the marker on the line before it."""

MARKER = re.compile(r"^<!-- (?:not run|output of):", re.MULTILINE)
"""The start of a marker, wherever it stands."""

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+(?:e[+-]?\d+)?|e[+-]?\d+)(?![\d.])")
"""A number written with a decimal point or an exponent: a rounded value."""

ULPS = 4
"""Maximum floating-point ULP difference tolerated across platforms.

Platform libm implementations can round a few ULPs apart and propagate through later
arithmetic. A formatted number therefore differs from a quoted value only near a
rounding boundary, by one unit in its last printed digit. Matching allows one unit in
the last quoted digit plus `ULPS` floating-point ULPs (dominated by the digit unit at
4-digit report precision and by `ULPS` at full 17-digit float precision).
"""

SCRIPT_FUNCTIONS = ("plan", "specs")
"""The functions of a script that state its simulation, which `jolty.runner` calls."""


@dataclass(frozen=True)
class Block:
    """A fenced block of a page: its language, its body and the marker before it."""

    page: Path
    line: int
    language: str
    body: str
    marker: str | None
    note: str | None

    @property
    def where(self) -> str:
        """The page and the line of the block's fence."""
        return f"{self.page.relative_to(ROOT)}:{self.line}"


def blocks(page: Path) -> Iterator[Block]:
    """Yields the Python blocks and quoted output blocks of `page` in order."""
    text = page.read_text(encoding="utf-8")
    for found in BLOCK.finditer(text):
        fence = found.start("language") - len("```")
        yield Block(
            page,
            text.count("\n", 0, fence) + 1,
            found["language"],
            found["body"],
            found["marker"],
            found["note"],
        )


BLOCKS = tuple(block for page in PAGES for block in blocks(page))
"""Every Python block and quoted output of the pages."""


def namespace() -> dict[str, object]:
    """Returns the initial execution namespace for a guide block."""
    names: dict[str, object] = {}
    for module in (jolty, units, plan, specs):
        public = getattr(module, "__all__", None) or [
            name for name in vars(module) if not name.startswith("_")
        ]
        for name in public:
            names.setdefault(name, getattr(module, name))
    c = Builder("excerpt")
    return names | {"c": c, "ground": c.port("ground", Role.GROUND)}


def quoted_values(block: Block) -> dict[int, object]:
    """Extracts literal values quoted in trailing comments of `block`, by line."""
    values: dict[int, object] = {}
    for token in tokenize.generate_tokens(io.StringIO(block.body).readline):
        if token.type == tokenize.COMMENT:
            with contextlib.suppress(ValueError, SyntaxError):
                values[token.start[0]] = ast.literal_eval(token.string[1:].strip())
    return values


def run(block: Block, names: dict[str, object]) -> None:
    """Runs a block in its page's namespace, checking every value it quotes."""
    quoted = quoted_values(block)
    for statement in ast.parse(block.body).body:
        if isinstance(statement, ast.Expr):
            expression = compile(ast.Expression(statement.value), block.where, "eval")
            value = eval(expression, names)  # noqa: S307 - the guide's own code
            if statement.end_lineno in quoted:
                line = block.line + statement.end_lineno
                where = f"{block.page.relative_to(ROOT)}:{line}"
                assert value == quoted[statement.end_lineno], where
        else:
            module = compile(ast.Module([statement]), block.where, "exec")
            exec(module, names)  # noqa: S102 - the guide's own code
        if (
            isinstance(statement, ast.FunctionDef)
            and statement.name in SCRIPT_FUNCTIONS
        ):
            function = names[statement.name]
            assert callable(function)
            function()


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_each_marker_stands_before_a_block(page: Path) -> None:
    marked = [
        block for block in BLOCKS if block.page == page and block.marker is not None
    ]
    assert len(MARKER.findall(page.read_text(encoding="utf-8"))) == len(marked)


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_each_python_block_of_a_page_runs(
    page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    names = namespace()
    for block in BLOCKS:
        if block.page == page and block.language == "python":
            assert block.marker != "output of", f"{block.where} is no output"
            if block.marker != "not run":
                run(block, names)


def output(commands: str, directory: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """Runs semicolon-separated `commands` in `directory` and returns stdout."""
    printed = []
    for command in commands.split(";"):
        program, *arguments = shlex.split(command)
        if program == "jolty":
            with contextlib.suppress(SystemExit):
                main(arguments)
            out, err = capsys.readouterr()
        else:
            assert program == "python", command
            # A script prints as it would to a file: CI forces color, which would set
            # its tracebacks in escape codes; Python's own setting outranks it.
            ran = subprocess.run(  # noqa: S603 - a script of the repository's own
                [sys.executable, *arguments],
                capture_output=True,
                check=False,
                cwd=directory,
                text=True,
                env={**os.environ, "PYTHON_COLORS": "0"},
            )
            out, err = ran.stdout, ran.stderr
        printed.append(out + err)
    return "".join(printed)


def tolerance(number: str) -> float:
    """Computes the allowed difference between a printed and quoted `number`.

    Combines one unit in the last printed digit with `ULPS` floating-point ULPs.
    """
    mantissa, _, exponent = number.partition("e")
    decimals = len(mantissa.partition(".")[2])
    unit = 10.0 ** (int(exponent or 0) - decimals)
    return unit + ULPS * math.ulp(float(number))


def excerpt(quote: str) -> tuple[re.Pattern[str], list[str]]:
    """Builds a regex matching outputs containing `quote` and lists quoted numbers.

    Each captured group corresponds to a printed number paired with a quoted value.
    """
    lines: list[str] = []
    numbers: list[str] = []
    for line in quote.splitlines():
        if line.strip() == "...":
            lines.append(r"(?:[^\n]*\n)*?")
            continue
        parts = []
        for part in line.split("..."):
            numbers += NUMBER.findall(part)
            texts = NUMBER.split(part)
            parts.append(f"({NUMBER.pattern})".join(re.escape(text) for text in texts))
        lines.append(".*?".join(parts) + r"\n")
    return re.compile("^" + "".join(lines), re.MULTILINE), numbers


def agrees(quote: str, printed: str) -> bool:
    """Returns whether `quote` matches an excerpt of `printed` within rounding."""
    pattern, numbers = excerpt(quote)
    return any(
        all(
            abs(float(found) - float(number)) <= tolerance(number)
            for found, number in zip(match.groups(), numbers, strict=True)
        )
        for match in pattern.finditer(printed + "\n")
    )


@pytest.mark.parametrize(
    "quote",
    [block for block in BLOCKS if block.language == "text"],
    ids=lambda block: block.where,
)
def test_each_quoted_output_is_what_its_commands_print(
    quote: Block,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert quote.marker == "output of", f"{quote.where} names no commands"
    assert quote.note is not None
    for directory in SCRIPTS:
        shutil.copytree(directory, tmp_path, dirs_exist_ok=True)
    monkeypatch.chdir(tmp_path)
    printed = output(quote.note, tmp_path, capsys)
    assert agrees(quote.body, printed), printed

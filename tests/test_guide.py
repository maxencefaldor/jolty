# Copyright 2026 Maxence Faldor
"""Every Python block of the guide and the README runs, and every output they quote.

A page's Python blocks run in order in one namespace, so a block may build on those
before it. A block is an excerpt of a script, as `.rumdl.toml` lints it: its namespace
starts with what its script would import, the names of `jolty` and of its modules
`units`, `plan` and `specs`, and with what the body of a component holds, its builder
`c` and its ground port `ground`. A block that defines a script's `plan` or `specs`
calls it, as the runner does; a script's `build` builds the script's own circuit, which
an excerpt leaves out. An expression whose comment is a Python literal, such as
`neighbours("E96", 1234.0)  # (1210.0, 1240.0)`, quotes its value.

A block that cannot run says why on the line before it, `<!-- not run: why -->`. A
quoted output, a text block, names on the line before it the commands that print it,
`<!-- output of: commands -->`, each a `jolty` command or `python` and a script, `;`
between them, run one after another in a directory of copies of the examples and of
the scripts of `tests/guide`. The quote is an excerpt of what they print: its lines are
consecutive lines of the output, where a line `...` stands for any lines, and a `...`
within a line for any text, such as a digest that changes with every build.
"""

import ast
import contextlib
import io
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

PAGES = (ROOT / "README.md", *sorted((ROOT / "docs" / "guide").glob("*.md")))
"""The README and the guide's pages, whose blocks run."""

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
    """The Python blocks and the quoted outputs of a page, in order."""
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
    """What a block's script holds before it: jolty's names, and a component's body."""
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
    """The values the comments of a block quote, by the line of the block they end."""
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
    """What the commands print, run one after another in `directory`."""
    printed = []
    for command in commands.split(";"):
        program, *arguments = shlex.split(command)
        if program == "jolty":
            with contextlib.suppress(SystemExit):
                main(arguments)
            out, err = capsys.readouterr()
        else:
            assert program == "python", command
            ran = subprocess.run(  # noqa: S603 - a script of the repository's own
                [sys.executable, *arguments],
                capture_output=True,
                check=False,
                cwd=directory,
                text=True,
            )
            out, err = ran.stdout, ran.stderr
        printed.append(out + err)
    return "".join(printed)


def excerpt(quote: str) -> re.Pattern[str]:
    """The pattern of the outputs a quote is an excerpt of."""
    lines = []
    for line in quote.splitlines():
        if line.strip() == "...":
            lines.append(r"(?:[^\n]*\n)*?")
        else:
            text = ".*?".join(re.escape(part) for part in line.split("..."))
            lines.append(text + r"\n")
    return re.compile("^" + "".join(lines), re.MULTILINE)


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
    assert excerpt(quote.body).search(printed + "\n"), printed

# Copyright 2026 Maxence Faldor
"""SPICE preprocessing: ``.include``, ``.lib``, and ``.if`` conditional blocks."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from jolty.errors import SpiceError
from jolty.spice.expressions import Function, evaluate, inlined, parse
from jolty.spice.netlist import Statement, statements, tokens

CONDITIONS: Final = frozenset({".if", ".elseif", ".else", ".endif"})
"""The statements of a subcircuit's ``.if`` blocks."""

_INCLUDES: Final = frozenset({".include", ".inc"})
"""The statements that include a file whole."""

_EXPAND_PREFIXES: Final = (".lib", ".endl", ".inc", ".subckt", ".ends")
"""Case-folded directive prefixes handled during `.include`/`.lib` expansion."""

_NO_FUNCTIONS: Final[Mapping[str, Function]] = MappingProxyType({})


def letter(statement: Statement) -> str:
    """Returns the case-folded leading element type letter of `statement`.

    Args:
        statement: Logical SPICE statement whose leading character is read.

    Returns:
        Case-folded first character of `statement.text`.
    """
    return statement.text[0].casefold()


def keyword(statement: Statement) -> str:
    """Returns the case-folded first word of `statement` (such as ``.param``).

    Args:
        statement: Logical SPICE statement whose directive or element word is read.

    Returns:
        Case-folded leading whitespace-delimited token of `statement.text`.
    """
    return statement.text.split(maxsplit=1)[0].casefold()


def expanded(
    path: Path, section: str | None, within: tuple[tuple[Path, str | None], ...]
) -> list[Statement]:
    """Reads and expands SPICE statements from `path`, optionally selecting `section`.

    Expands ``.include`` and ``.lib file section`` directives relative to the
    including file directory, both globally and inside subcircuits. Reading a file
    without `section` collects statements outside ``.lib name ... .endl`` sections.
    Active `(path, section)` frames in `within` detect include cycles.

    Args:
        path: Filesystem path to the SPICE file to read.
        section: Optional ``.lib`` section name to extract, or `None` for top-level
            statements.
        within: Active stack of `(resolved_path, section)` frames for cycle detection.

    Returns:
        Expanded `Statement` list in source order with nested includes inlined.

    Raises:
        SpiceError: When `path` cannot be read, `section` is missing, or includes form
            a cycle.
    """
    reading = (path.resolve(), None if section is None else section.casefold())
    if reading in within:
        message = f"{path} includes itself, through {len(within)} files"
        raise SpiceError(message)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        message = f"cannot read {path}: {error.strerror}"
        raise SpiceError(message) from None
    found, seen = expanded_text(
        text, path.parent, path.name, section, (*within, reading)
    )
    if section is not None and not seen:
        message = f"{path} has no section `{section}`"
        raise SpiceError(message)
    return found


def _update_subcircuit_depth(
    expanded_statements: Sequence[Statement], subcircuit_depth: int
) -> int:
    """Updates `.subckt`/`.ends` nesting depth across `expanded_statements`."""
    for inner in expanded_statements:
        if not inner.text.startswith("."):
            continue
        inner_keyword = keyword(inner)
        if inner_keyword == ".subckt":
            subcircuit_depth += 1
        elif inner_keyword == ".ends":
            subcircuit_depth = max(subcircuit_depth - 1, 0)
    return subcircuit_depth


def expanded_text(
    text: str,
    directory: Path,
    label: str,
    section: str | None,
    within: tuple[tuple[Path, str | None], ...],
) -> tuple[list[Statement], bool]:
    """Expands ``.include`` and ``.lib`` directives in `text` relative to `directory`.

    Returns the expanded statements of `section` (or outside sections when `section` is
    None) and whether `section` was seen. Inside a ``.subckt ... .ends`` body, a
    two-word ``.lib`` is never treated as opening a library section; it reaches
    `_included` and fails with the malformed directive error.

    Args:
        text: Raw SPICE file contents to preprocess.
        directory: Base directory used to resolve relative `.include` and `.lib` paths.
        label: Source filename recorded on emitted `Statement` objects.
        section: Optional target `.lib` section name, or `None` for unsectioned lines.
        within: Active stack of `(resolved_path, section)` frames for cycle detection.

    Returns:
        Pair of `(expanded_statements, section_seen)`.
    """
    target = None if section is None else section.casefold()
    found: list[Statement] = []
    opened: str | None = None
    subcircuit_depth = 0
    seen = False
    for statement in statements(text, label):
        line = statement.text
        if not line.startswith(".") or not line[:7].casefold().startswith(
            _EXPAND_PREFIXES
        ):
            if opened == target:
                found.append(statement)
            continue
        words = [word.strip("'\"") for word in tokens(line)]
        directive = words[0].casefold() if words else ""
        section_header = len(words) == 2  # noqa: PLR2004 - `.lib` and a name
        if subcircuit_depth == 0 and directive == ".lib" and section_header:
            if opened is not None:
                message = f"a section opens within section `{opened}`"
                raise statement.fail(message)
            opened = words[1].casefold()
            seen |= opened == target
        elif subcircuit_depth == 0 and directive == ".endl":
            opened = None
        elif opened == target:
            included = _included(statement, words, directory, within)
            subcircuit_depth = _update_subcircuit_depth(included, subcircuit_depth)
            found.extend(included)
    return found, seen


def _included(
    statement: Statement,
    words: Sequence[str],
    directory: Path,
    within: tuple[tuple[Path, str | None], ...],
) -> list[Statement]:
    """Expands `statement` if it is a `.include`/`.lib` directive, else returns it.

    Raises:
        SpiceError: When an include directive has invalid argument count.
    """
    if not words:
        return [statement]
    directive = words[0].casefold()
    if directive == ".lib" and len(words) == 3:  # noqa: PLR2004 - a file and a section
        return expanded(directory / words[1], words[2], within)
    if directive in _INCLUDES and len(words) == 2:  # noqa: PLR2004 - and a file
        return expanded(directory / words[1], None, within)
    if directive in _INCLUDES or directive == ".lib":
        message = "a file is included by its path, a section by the file's and name"
        raise statement.fail(message)
    return [statement]


def unconditional(body: Sequence[Statement]) -> list[Statement]:
    """Returns all statements in `body` outside ``.if`` conditional blocks.

    Args:
        body: Subcircuit body statements to filter.

    Returns:
        Statements appearing at conditional nesting depth zero.
    """
    depth = 0
    found: list[Statement] = []
    for statement in body:
        directive = keyword(statement)
        if directive == ".if":
            depth += 1
        elif directive == ".endif":
            depth = max(0, depth - 1)
        elif depth == 0 and directive not in CONDITIONS:
            found.append(statement)
    return found


@dataclass(slots=True)
class _Block:
    """State of an active ``.if`` block tracking current and prior branch selection."""

    chosen: bool
    taken: bool


def chosen(
    body: Sequence[Statement],
    scope: Mapping[str, float],
    functions: Mapping[str, Function] = _NO_FUNCTIONS,
) -> list[Statement]:
    """Selects statements from `body` by evaluating ``.if`` blocks in `scope`.

    Args:
        body: Subcircuit body statements potentially containing ``.if`` blocks.
        scope: Case-folded parameter values used to evaluate branch conditions.
        functions: Case-folded user-defined functions inlined in branch conditions.

    Returns:
        Active non-directive statements selected across all conditional branches.

    Raises:
        SpiceError: When conditional directives are unmatched or unterminated.
    """
    blocks: list[_Block] = []
    found: list[Statement] = []
    for statement in body:
        directive = keyword(statement)
        within = all(block.chosen for block in blocks)
        if directive in CONDITIONS - {".if"} and not blocks:
            message = f"`{directive}` within no `.if` block"
            raise statement.fail(message)
        match directive:
            case ".if":
                held = within and _condition(statement, scope, functions)
                blocks.append(_Block(chosen=held, taken=held))
            case ".elseif" | ".else":
                block = blocks[-1]
                outer = all(each.chosen for each in blocks[:-1])
                held = (
                    outer
                    and not block.taken
                    and (
                        directive == ".else" or _condition(statement, scope, functions)
                    )
                )
                block.chosen, block.taken = held, block.taken or held
            case ".endif":
                blocks.pop()
            case _ if within:
                found.append(statement)
    if blocks:
        message = "unterminated .if block (missing .endif)"
        raise body[-1].fail(message)
    return found


def _condition(
    statement: Statement,
    scope: Mapping[str, float],
    functions: Mapping[str, Function] = _NO_FUNCTIONS,
) -> bool:
    """Returns whether an ``.if`` or ``.elseif`` condition evaluates nonzero."""
    parts = statement.text.split(maxsplit=1)
    condition = parts[1] if len(parts) > 1 else ""
    return evaluate(inlined(parse(condition), functions), scope) != 0

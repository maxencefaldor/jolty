# Copyright 2026 Maxence Faldor
"""The architecture AGENTS.md states, checked against the code.

AGENTS.md lists the Rust workspace's crates in order, and the Python package's modules
in order, several to a line. A crate depends only on the crates listed before it, and a
module imports only from the modules listed before it, and from those listed before it
on its own line. These tests read both lists from AGENTS.md, which stays the one
definition of the order.

Three modules of the package sit outside its list: `jolty._core`, the extension the
crate `jolty-python` builds, below every listed module; and, above them all, the
package's `__init__`, which gathers what the modules define, and its `__main__`, which
runs the command line.
"""

import ast
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from workspace import ROOT, crates

PACKAGE: Final = ROOT / "src" / "jolty"

_ITEM: Final = re.compile(r"1\. ((?:`[^`]+`, )*`[^`]+`): ")
_NAME: Final = re.compile(r"`([^`]+)`")

type Order = Mapping[str, tuple[int, int]]
"""Where each name is listed: the number of its line, from 1, and its place on it."""


def _lists() -> list[list[list[str]]]:
    """The ordered lists of AGENTS.md's architecture: each line's names, in order."""
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    section = text.split("\n## Architecture\n", 1)[1].split("\n## ", 1)[0]
    found: list[list[list[str]]] = [[]]
    for line in section.splitlines():
        item = _ITEM.match(line)
        if item is not None:
            found[-1].append(_NAME.findall(item.group(1)))
        elif found[-1]:
            found.append([])
    return [lines for lines in found if lines]


def _order(lines: Sequence[Sequence[str]]) -> dict[str, tuple[int, int]]:
    return {
        name: (number, place)
        for number, names in enumerate(lines, 1)
        for place, name in enumerate(names)
    }


CRATE_LINES, MODULE_LINES = _lists()
CRATES: Final = _order(CRATE_LINES)

CORE: Final = "_core"
"""The engine's extension module, below every listed module."""
PACKAGE_ITSELF: Final = ""
"""The package's `__init__`, which gathers what the modules define, above them all."""
MAIN: Final = "__main__"
"""The package's `__main__`, which runs the command line, above every listed module."""
MODULES: Final = {
    CORE: (0, 0),
    **_order(MODULE_LINES),
    PACKAGE_ITSELF: (len(MODULE_LINES) + 1, 0),
    MAIN: (len(MODULE_LINES) + 1, 1),
}


def _layer(name: str, order: Order) -> str:
    """A crate or module, and the line of AGENTS.md's list it is on."""
    if name == CORE:
        return "`jolty._core`, the engine's extension below every line"
    if name == PACKAGE_ITSELF:
        return "`jolty`'s `__init__`, above every line"
    if name == MAIN:
        return "`jolty.__main__`, above every line"
    if name not in order:
        return f"`{name}`, on no line"
    return f"`{name}` (line {order[name][0]})"


def _breaks(order: Order, user: str, used: str) -> bool:
    """Whether `user` using `used` breaks the order, or either is on no line."""
    if user == used:
        return False
    if user not in order or used not in order:
        return True
    return order[used] >= order[user]


def test_agents_md_lists_every_crate_once() -> None:
    listed = [name for names in CRATE_LINES for name in names]
    assert set(listed) == {crate["name"] for crate in crates(dependencies=False)}
    assert len(set(listed)) == len(listed), "a crate is listed twice"


def test_each_crate_depends_only_on_the_crates_listed_before_it() -> None:
    members = crates(dependencies=False)
    workspace = {crate["name"] for crate in members}
    broken = [
        f"{_layer(crate['name'], CRATES)} depends on "
        f"{_layer(dependency['name'], CRATES)}"
        + ("" if dependency["kind"] is None else f" for {dependency['kind']}")
        for crate in members
        for dependency in crate["dependencies"]
        if dependency["name"] in workspace
        and _breaks(CRATES, crate["name"], dependency["name"])
    ]
    assert not broken, "\n".join(broken)


def _listed(path: Path) -> str:
    """The listed module that the file at `path` belongs to, or the package itself."""
    parts = path.relative_to(PACKAGE).with_suffix("").parts
    return PACKAGE_ITSELF if parts == ("__init__",) else parts[0]


FILES: Final = sorted(PACKAGE.rglob("*.py"))
"""The package's Python files."""
NAMES: Final = {_listed(path) for path in FILES} | {CORE}
"""The package's modules, as its files spell them, the package itself among them."""


def _imported(path: Path) -> list[tuple[int, str]]:
    """Each listed module the file at `path` imports, by the line that imports it.

    Every import counts, those within functions and under `TYPE_CHECKING` too. A name
    imported from the package itself, as `ir` is in ``from jolty import ir``, is the
    module of that name if there is one, and the package's `__init__` otherwise.
    """
    within = ["jolty", *path.relative_to(PACKAGE).parent.parts]
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                first, *rest = alias.name.split(".")
                if first == "jolty":
                    found.append((node.lineno, rest[0] if rest else PACKAGE_ITSELF))
        elif isinstance(node, ast.ImportFrom):
            source = node.module.split(".") if node.module is not None else []
            if node.level:
                source = within[: len(within) + 1 - node.level] + source
            if source[:1] != ["jolty"]:
                continue
            if len(source) > 1:
                found.append((node.lineno, source[1]))
                continue
            found.extend(
                (node.lineno, alias.name if alias.name in NAMES else PACKAGE_ITSELF)
                for alias in node.names
            )
    return sorted(found)


def test_agents_md_lists_every_module_once() -> None:
    listed = [name for names in MODULE_LINES for name in names]
    assert set(listed) == NAMES - {CORE, PACKAGE_ITSELF, MAIN}
    assert len(set(listed)) == len(listed), "a module is listed twice"


def test_each_module_imports_only_from_the_modules_listed_before_it() -> None:
    broken = [
        f"{path.relative_to(ROOT)}:{line}: "
        f"{_layer(importer, MODULES)} imports {_layer(imported, MODULES)}"
        for path in FILES
        for importer in [_listed(path)]
        for line, imported in _imported(path)
        if _breaks(MODULES, importer, imported)
    ]
    assert not broken, "\n".join(broken)

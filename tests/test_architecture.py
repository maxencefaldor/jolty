# Copyright 2026 Maxence Faldor
"""Architecture layering checks validating the codebase against `AGENTS.md`.

`AGENTS.md` lists Rust workspace crates and Python package modules in dependency order,
several per line. Each crate depends only on earlier crates, and each module imports
only from earlier lines or earlier entries on the same line. The tests parse both lists
directly from `AGENTS.md` as the single source of truth.

Three modules sit outside the module list: `jolty._core`, built by `jolty-python`, lies
below every listed module. Above all listed modules sit the public-API `__init__` and
the CLI entry point `__main__`.
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
    """Parses the ordered crate and module layer lists from `AGENTS.md`."""
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
    """Formats `name` together with its layer line number in `AGENTS.md`."""
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
    """Returns whether `user` depending on `used` violates `order` or omits either."""
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
    """Returns the listed module containing `path`, or the package root name."""
    parts = path.relative_to(PACKAGE).with_suffix("").parts
    return PACKAGE_ITSELF if parts == ("__init__",) else parts[0]


FILES: Final = sorted(PACKAGE.rglob("*.py"))
"""The package's Python files."""
NAMES: Final = {_listed(path) for path in FILES} | {CORE}
"""The package's modules, as its files spell them, the package itself among them."""


def _imported(path: Path) -> list[tuple[int, str]]:
    """Lists `(line, module)` pairs imported by `path` among listed modules.

    Includes function-scoped and `TYPE_CHECKING` imports. A name imported from the
    package root, such as `ir` in ``from jolty import ir``, resolves to that submodule
    when present and to `__init__` otherwise.
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

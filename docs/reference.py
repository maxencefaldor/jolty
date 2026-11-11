# Copyright 2026 Maxence Faldor
"""Write the reference pages that come from the code: commands and settings.

Run from the repository root: ``uv run python docs/reference.py``. Every option's flag,
default and description come from the request models in :mod:`jolty.service`, the
single definition the command line and the MCP server share, and every setting comes
from :class:`jolty.settings.Settings`. The pages therefore match the tools exactly, and
CI fails when a page is stale.
"""

import re
from html import escape
from pathlib import Path
from typing import Final

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from jolty.cli import Jolty
from jolty.settings import Settings

REFERENCE: Final = Path(__file__).resolve().parent / "reference"
_URL: Final = re.compile(r"(https?://[^\s<>]+?)([.,;:]?(?:\s|$))")

CLI_INTRODUCTION: Final = """\
# Command line

The `jolty` command builds and simulates circuits from the shell.

Every command takes `--output-format json` to print its result as JSON for programs; \
the default, `text`, is for people and agents.
The command `jolty --version` prints the version.
The exit status is 0 for success, 1 when a circuit does not elaborate or an analysis \
fails, and 2 for invalid arguments or settings, or a failed command, as when a script \
raises.
Each command's options come from a request model in `jolty.service`, which the \
[MCP server](../agents/connect.md#mcp-server) shares, so a tool takes the same \
arguments as its command.
"""

SETTINGS_INTRODUCTION: Final = """\
# Configuration

Settings tune every command.
Each setting comes from the first source that sets it: a `JOLTY_*` environment \
variable, or the `[tool.jolty]` table of the `pyproject.toml` in the working directory.

| Setting | Environment variable | Default | Description |
| --- | --- | --- | --- |
"""

SETTINGS_EXAMPLE: Final = """\

A project whose transients run long might raise the time limit and list more unknowns:

```toml title="pyproject.toml"
[tool.jolty]
timeout = 1800
listed_unknowns = 100
```
"""


def _positional(field: FieldInfo) -> bool:
    """Whether pydantic-settings marks the field as a positional argument."""
    return any(
        getattr(item, "__name__", "") == "_CliPositionalArg" for item in field.metadata
    )


def _flag(name: str, field: FieldInfo) -> str:
    if _positional(field):
        return name.upper()
    flag = "--" + name.rstrip("_").replace("_", "-")
    if field.annotation is bool:
        return f"{flag}`, `--no-{flag.removeprefix('--')}"
    return flag


def _default(field: FieldInfo) -> str:
    if field.default is PydanticUndefined and field.default_factory is None:
        return "required"
    value = field.get_default(call_default_factory=True)
    if value in (None, "", [], ()):
        return ""
    if isinstance(value, list | tuple):
        return ", ".join(f"`{getattr(item, 'value', item)}`" for item in value)
    return f"`{getattr(value, 'value', value)}`"


def _cell(text: str | None) -> str:
    """A description as one table cell: one line, escaped, its links clickable."""
    cell = escape(" ".join((text or "").split()), quote=False).replace("|", "\\|")
    return _URL.sub(r"<\1>\2", cell)


def _options(command: type[BaseModel]) -> list[str]:
    rows = ["| Option | Default | Description |", "| --- | --- | --- |"]
    rows.extend(
        f"| `{_flag(name, field)}` | {_default(field)} | {_cell(field.description)} |"
        for name, field in command.model_fields.items()
    )
    return rows


def _commands() -> list[tuple[str, type[BaseModel], str]]:
    """Each subcommand of ``jolty``: its name, its model and its summary."""
    found = []
    for name, field in Jolty.model_fields.items():
        command = next(
            item
            for item in getattr(field.annotation, "__args__", ())
            if isinstance(item, type) and issubclass(item, BaseModel)
        )
        found.append((name.replace("_", "-"), command, field.description or ""))
    return found


def cli_page() -> str:
    """The page of commands: one section per command, with a table of its options."""
    lines = [CLI_INTRODUCTION]
    for name, command, summary in _commands():
        lines.extend(
            [
                f"## `jolty {name}`",
                "",
                summary.rstrip(".") + ".",
                "",
                *_options(command),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _setting_default(field: FieldInfo) -> str:
    value = field.get_default(call_default_factory=True, validated_data={})
    return "unset" if value is None or value == {} else f"`{value}`"


def settings_page() -> str:
    """The page of settings: where they come from, then one row each."""
    rows = [
        f"| `{name}` | `JOLTY_{name.upper()}` | {_setting_default(field)} | "
        f"{_cell(field.description)} |"
        for name, field in Settings.model_fields.items()
    ]
    return SETTINGS_INTRODUCTION + "\n".join(rows) + "\n" + SETTINGS_EXAMPLE


def main() -> None:
    """Writes every page."""
    REFERENCE.mkdir(parents=True, exist_ok=True)
    pages = {
        REFERENCE / "cli.md": cli_page(),
        REFERENCE / "configuration.md": settings_page(),
    }
    for page, text in pages.items():
        page.write_text(text, encoding="utf-8")
        print(f"wrote {page}")


if __name__ == "__main__":
    main()

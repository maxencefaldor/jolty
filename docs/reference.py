# Copyright 2026 Maxence Faldor
"""Write the reference pages that come from the code.

Run from the repository root: ``uv run python docs/reference.py``. Every option's flag,
default and description come from the request models in `jolty.service`, the single
definition the command line and the MCP server share; every rule's level and condition
come from the engine, as ``jolty explain`` prints them; every setting comes from
`jolty.settings.Settings`; the expression language's description comes with the
engine's parser; and every model's parameters, with their defaults and descriptions,
come from the schemas the engine derives from its own parameter types. The pages
therefore match the tools exactly, and CI fails when a page is stale.
"""

import json
import re
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Final

from pydantic import BaseModel, JsonValue, TypeAdapter
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from jolty import _core
from jolty.checks import rules
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
The exit status is 0 for success, 1 when a circuit does not elaborate, a check finds \
an error or an analysis fails, and 2 for invalid arguments or settings, or a failed \
command, as when a script raises.
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

A project whose transients run long might raise the time limit, and one that couples \
stages through capacitors on purpose might report floating nodes as warnings:

```toml title="pyproject.toml"
[tool.jolty]
timeout = 1800

[tool.jolty.rules]
no-dc-path = "warn"
```
"""

RULES_INTRODUCTION: Final = """\
# Rules

Each rule names one fact the structural checks can state about a circuit, and \
`jolty explain` prints the same condition as this page.
A check reports each rule at its level: `error`, `warn` or `ignore`.
The default levels below hold until settings or options set others (see \
[Checks](../guide/checks.md#levels)).
"""

MODELS_INTRODUCTION: Final = """\
# Models

Each model a device can be, as the engine defines it, and every parameter a device of \
it takes: its default, which a device that leaves it out takes, and its description, \
both from the engine's own parameter types.
A parameter within another is named by its path, its keys joined by dots and `*` \
standing for each entry of a list, such as `waveform.pulse.rise` or \
`windings.*.inductance`; of the parameters of one shape, such as `waveform.dc` and \
`waveform.pulse`, a device gives one.
Values are in SI base units, as the circuit format holds them, while the functions of \
`jolty.devices` that place each model take [quantities with units](../guide/units.md).
"""

_SENTENCE_END: Final = re.compile(r"(?<=[.:])\s+(?=[A-Z`(])")


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


def rules_page() -> str:
    """The page of rules: a table, then each rule's condition, one sentence a line."""
    known = rules()
    lines = [
        RULES_INTRODUCTION,
        "| Rule | Default level |",
        "| --- | --- |",
        *(f"| [`{name}`](#{name}) | `{rule.level}` |" for name, rule in known.items()),
    ]
    for name, rule in known.items():
        lines.extend(("", f"## {name}", ""))
        lines.extend(_SENTENCE_END.split(rule.condition))
    return "\n".join(lines) + "\n"


_SCHEMA: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


def _object(value: JsonValue) -> Mapping[str, JsonValue]:
    """`value` if it is a JSON object, or an empty one."""
    empty: dict[str, JsonValue] = {}
    return value if isinstance(value, dict) else empty


def _objects(value: JsonValue) -> list[Mapping[str, JsonValue]]:
    """The JSON objects in `value`, if it is a list."""
    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _value(value: JsonValue) -> str:
    """A JSON value as a table cell: a number in its shortest form, text as it is."""
    match value:
        case bool() | None | list() | dict():
            written = json.dumps(value)
        case float():
            written = repr(value).removesuffix(".0")
        case _:
            written = str(value)
    return f"`{written}`"


def _constants(schema: Mapping[str, JsonValue]) -> list[JsonValue]:
    """The values a schema allows, if it allows only some, however deep they lie."""
    enumerated = schema.get("enum")
    found = list(enumerated) if isinstance(enumerated, list) else []
    if "const" in schema:
        found.append(schema["const"])
    for shape in [*_objects(schema.get("oneOf")), *_objects(schema.get("anyOf"))]:
        found.extend(_constants(shape))
    return found


def _choices(schema: Mapping[str, JsonValue]) -> str:
    """The values a parameter chooses among, as a sentence, if it chooses among some."""
    choices = [_value(choice) for choice in _constants(schema) if choice is not None]
    return f" One of {', '.join(choices)}." if choices else ""


def _parameters(
    path: str,
    schema: Mapping[str, JsonValue],
    *,
    required: bool,
    described: JsonValue = None,
) -> list[str]:
    """The table rows of the parameter at `path` and of every parameter within it.

    A parameter that is one of several shapes, such as ``waveform.pulse``, is described
    as its shape is, `described`, rather than as its type is.
    """
    description = described or schema.get("description")
    text = description if isinstance(description, str) else ""
    match schema.get("default"):
        case _ if required:
            default = "required"
        case None:
            default = ""
        case given:
            default = _value(given)
    rows: list[str] = [f"| `{path}` | {default} | {_cell(text + _choices(schema))} |"]
    shapes = [*_objects(schema.get("oneOf")), *_objects(schema.get("anyOf"))]
    within = [
        (path, schema, False),
        *((path, shape, True) for shape in shapes),
        (f"{path}.*", _object(schema.get("items")), False),
    ]
    for prefix, outer, shape in within:
        needed = outer.get("required")
        for name, inner in _object(outer.get("properties")).items():
            rows.extend(
                _parameters(
                    f"{prefix}.{name}",
                    _object(inner),
                    required=not shape and isinstance(needed, list) and name in needed,
                    described=outer.get("description") if shape else None,
                )
            )
    return rows


def models_page() -> str:
    """The page of models: for each, a table of its parameters, required ones first.

    Each parameter comes with its default and its description, from the engine.
    """
    lines = [MODELS_INTRODUCTION]
    for model in _core.models():
        schema = _SCHEMA.validate_json(_core.schema(model))
        needed = schema.get("required")
        required = set(needed) if isinstance(needed, list) else set()
        properties = _object(schema.get("properties"))
        order = sorted(properties, key=lambda name: (name not in required, name))
        lines.extend(
            [
                f"## `{model}`",
                "",
                "| Parameter | Default | Description |",
                "| --- | --- | --- |",
            ]
        )
        for name in order:
            lines.extend(
                _parameters(name, _object(properties[name]), required=name in required)
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    """Writes every page."""
    REFERENCE.mkdir(parents=True, exist_ok=True)
    pages = {
        REFERENCE / "cli.md": cli_page(),
        REFERENCE / "rules.md": rules_page(),
        REFERENCE / "configuration.md": settings_page(),
        REFERENCE / "expressions.md": _core.EXPRESSIONS,
        REFERENCE / "models.md": models_page(),
    }
    for page, text in pages.items():
        page.write_text(text, encoding="utf-8")
        print(f"wrote {page}")


if __name__ == "__main__":
    main()

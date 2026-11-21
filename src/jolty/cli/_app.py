# Copyright 2026 Maxence Faldor
"""Pydantic-settings CLI application and subcommand builders for ``jolty``."""

__lazy_modules__ = [
    "jolty.errors",
    "jolty.service",
    "jolty.settings",
    "json",
    "os",
]

import functools
import json
import os
import sys
import typing
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from operator import getitem
from typing import Any, Literal, NoReturn, cast, override

from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliSubCommand,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from jolty import service
from jolty.errors import JoltyError
from jolty.settings import Settings


class _Output(BaseModel):
    output_format: Literal["text", "json"] = Field(
        default="text",
        description="Output format: 'text' for human and agent readers, or 'json' for "
        "structured tooling.",
    )

    def emit(self, result: service.CommandResult) -> None:
        """Prints `result` in the requested format and exits 1 on failure."""
        if self.output_format == "json":
            print(result.model_dump_json(indent=2))
        else:
            print(result.describe())
        if not result.ok:
            raise SystemExit(1)


@dataclass(frozen=True, slots=True)
class CommandSpec[Request: BaseModel]:
    """Declarative specification binding one ``jolty`` subcommand to a service call."""

    name: str
    request_type: type[Request]
    handler: Callable[[Request, Settings], Awaitable[service.CommandResult]]
    description: str


@functools.cache
def _commands() -> tuple[CommandSpec[Any], ...]:
    """Builds the declarative specifications for every top-level ``jolty`` command."""
    return (
        CommandSpec(
            "build",
            service.BuildRequest,
            service.build,
            "Build a script's circuit and elaborate it.",
        ),
        CommandSpec(
            "simulate",
            service.SimulateRequest,
            service.simulate,
            "Build a script's circuit and run a plan's analyses on it.",
        ),
        CommandSpec(
            "merge",
            service.MergeRequest,
            service.merge,
            "Merge the shards of a plan's Monte Carlo analysis into one run.",
        ),
        CommandSpec(
            "plot",
            service.PlotRequest,
            service.plot,
            "Plot the analyses of a saved results file.",
        ),
        CommandSpec(
            "measure",
            service.MeasureRequest,
            service.measure,
            "Read a measure from a saved results file.",
        ),
        CommandSpec(
            "draw",
            service.DrawRequest,
            service.draw,
            "Draw a component of a saved circuit file.",
        ),
        CommandSpec(
            "view",
            service.ViewRequest,
            service.view,
            "Write a 3D viewer of a saved circuit and its results.",
        ),
        CommandSpec(
            "check",
            service.CheckRequest,
            service.check,
            "Check the structure of a saved circuit file.",
        ),
        CommandSpec(
            "explain",
            service.ExplainRequest,
            service.explain,
            "Explain rules and their levels.",
        ),
        CommandSpec(
            "read_docs",
            service.ReadDocsRequest,
            service.read_docs,
            "List or read the offline guide and examples.",
        ),
        CommandSpec(
            "cost",
            service.CostRequest,
            service.cost,
            "Price a component of a saved circuit file with a catalog.",
        ),
        CommandSpec(
            "export_kicad",
            service.ExportKicadRequest,
            service.export_kicad,
            "Write a component of a saved circuit file as a KiCad netlist.",
        ),
    )


def _subcommand_model[Request: BaseModel](
    spec: CommandSpec[Request],
) -> type[BaseModel]:
    """Builds the Pydantic CLI subcommand model for `spec`."""
    handler = spec.handler

    class _Runner(_Output):
        async def cli_cmd(self) -> None:
            """Runs the command."""
            self.emit(await handler(cast(Request, self), Settings.load()))

    model_name = f"{spec.request_type.__name__.removesuffix('Request')}Command"
    return create_model(
        model_name,
        __base__=(spec.request_type, _Output),
        __doc__=spec.request_type.__doc__,
        __module__=__name__,
        __namespace__={"cli_cmd": _Runner.cli_cmd},
    )


class CommandLine(BaseSettings):
    """Base command-line settings class configured with jolty CLI conventions.

    Options use kebab-case names, boolean flags take no value, optional fields omit
    ``None`` from displayed types, list options accept comma-separated or repeated
    arguments instead of JSON, and required options are enforced. Environment variables
    are ignored because runtime configuration lives in
    [`Settings`][jolty.settings.Settings]. Subclasses set
    ``model_config = SettingsConfigDict(cli_prog_name=...)``, declare CLI options as
    fields, implement ``cli_cmd``, and execute through [`run`][jolty.cli.run].
    """

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_hide_none_type=True,
        cli_avoid_json=True,
        cli_enforce_required=True,
    )

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Reads the command line only.

        Args:
            settings_cls: Settings class under initialization.
            init_settings: Keyword and CLI argument settings source.
            env_settings: Environment variable settings source, ignored by CLI models.
            dotenv_settings: Dotenv file settings source, ignored by CLI models.
            file_secret_settings: Secret directory settings source, ignored by CLI
                models.

        Returns:
            Single-element tuple containing only `init_settings`.
        """
        return (init_settings,)


def run(command: type[CommandLine], arguments: Sequence[str] | None = None) -> None:
    """Runs `command` on `arguments`, defaulting to the process command-line arguments.

    Args:
        command: CLI settings class to parse and execute.
        arguments: Command-line tokens to parse, defaulting to ``sys.argv[1:]`` when
            ``None``.

    Raises:
        SystemExit: Exits with status 2 and reports each invalid option and its
            expected input when validation fails.
    """
    given = sys.argv[1:] if arguments is None else list(arguments)
    try:
        CliApp.run(command, cli_args=given)
    except ValidationError as error:
        _fail(_invalid(error), as_json=False)


class _JoltyBase(CommandLine):
    """Base settings class for the generated top-level ``jolty`` CLI model."""

    model_config = SettingsConfigDict(cli_prog_name="jolty")

    def cli_cmd(self) -> None:
        """Runs the chosen subcommand."""
        CliApp.run_subcommand(self)


@functools.cache
def _jolty() -> type[_JoltyBase]:
    """Builds the top-level ``Jolty`` CLI settings model from `_commands()`."""
    fields: dict[str, Any] = {
        spec.name: (
            getitem(CliSubCommand, _subcommand_model(spec)),
            Field(description=spec.description),
        )
        for spec in _commands()
    }
    return create_model(
        "Jolty",
        __base__=_JoltyBase,
        __doc__="Build and simulate electronic circuits written as Python scripts.",
        __module__=__name__,
        **fields,
    )


@functools.cache
def _command_by_cli_name() -> dict[str, CommandSpec[Any]]:
    """Maps kebab-case CLI subcommand names to their `CommandSpec`."""
    return {spec.name.replace("_", "-"): spec for spec in _commands()}


@functools.cache
def _single_command_model(cli_name: str) -> type[CommandLine]:
    """Builds a standalone CLI settings model for one subcommand."""
    spec = _command_by_cli_name()[cli_name]
    handler = spec.handler

    class _SingleBase(CommandLine):
        model_config = SettingsConfigDict(cli_prog_name=f"jolty {cli_name}")

    class _Runner(_Output):
        async def cli_cmd(self) -> None:
            """Runs the command."""
            self.emit(await handler(cast(Any, self), Settings.load()))

    model_name = f"{spec.request_type.__name__.removesuffix('Request')}Command"
    return cast(
        "type[CommandLine]",
        create_model(
            model_name,
            __base__=(spec.request_type, _Output, _SingleBase),
            __doc__=spec.request_type.__doc__,
            __module__=__name__,
            __namespace__={"cli_cmd": _Runner.cli_cmd},
        ),
    )


if typing.TYPE_CHECKING:
    COMMANDS: tuple[CommandSpec[Any], ...]
    """Declarative specifications for every top-level ``jolty`` CLI subcommand."""

    Jolty: type[_JoltyBase]
    """Generated top-level ``jolty`` CLI settings model."""


def __getattr__(name: str) -> object:
    """Resolves `COMMANDS` and `Jolty` lazily to avoid building unused CLI models."""
    if name == "COMMANDS":
        value: object = _commands()
        globals()[name] = value
        return value
    if name == "Jolty":
        value = _jolty()
        globals()[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)


def __dir__() -> list[str]:
    """Lists module attributes including lazily built CLI models."""
    return sorted({*globals(), "COMMANDS", "Jolty"})


def run_cli(arguments: list[str]) -> None:
    """Parses `arguments` and executes the selected ``jolty`` subcommand."""
    try:
        if arguments and arguments[0] in _command_by_cli_name():
            CliApp.run(
                _single_command_model(arguments[0]),
                cli_args=arguments[1:],
            )
        else:
            CliApp.run(_jolty(), cli_args=arguments)
    except JoltyError as error:
        _fail(str(error), as_json=_wants_json(arguments))
    except ValidationError as error:
        _fail(_invalid(error), as_json=_wants_json(arguments))
    except SystemExit as exit_:
        # A command run without its subcommand exits with an error and its usage.
        if isinstance(exit_.code, str):
            print(exit_.code, end="", file=sys.stderr)
            raise SystemExit(2) from None
        raise
    except BrokenPipeError:
        # The reader stopped reading, as `head` does: stop quietly.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(1) from None


def _fail(message: str, *, as_json: bool) -> NoReturn:
    """Prints an error, as JSON on standard output when asked to, and exits with 2."""
    if as_json:
        print(json.dumps({"error": message}))
    else:
        print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _wants_json(arguments: Sequence[str]) -> bool:
    pairs = zip(arguments, [*arguments[1:], ""], strict=True)
    return "--output-format=json" in arguments or ("--output-format", "json") in pairs


def _invalid(error: ValidationError) -> str:
    """Formats each invalid CLI option with its validation message."""
    problems = []
    for problem in error.errors():
        names = [part for part in problem["loc"] if isinstance(part, str)]
        option = "--" + names[-1].replace("_", "-") if names else "arguments"
        problems.append(f"{option}: {problem['msg']}")
    return "invalid arguments: " + "; ".join(problems)

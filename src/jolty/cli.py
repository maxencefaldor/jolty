# Copyright 2026 Maxence Faldor
"""The ``jolty`` command line.

Each subcommand extends a request model of [`jolty.service`][jolty.service] with an
output format, so the command line takes exactly the arguments the MCP tools take and
prints the text they return.
"""

import json
import os
import sys
from collections.abc import Sequence
from importlib.metadata import version
from typing import Literal, NoReturn, override

from pydantic import BaseModel, Field, ValidationError
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
        default="text", description="'text' for people and agents, 'json' for programs."
    )

    def emit(self, result: service.CommandResult) -> None:
        """Prints `result` in the requested format; exits with 1 if it is not ok."""
        if self.output_format == "json":
            print(result.model_dump_json(indent=2))
        else:
            print(result.describe())
        if not result.ok:
            raise SystemExit(1)


class BuildCommand(service.BuildRequest, _Output):
    """Run a script's build(), save the circuit it returns, and elaborate it."""

    async def cli_cmd(self) -> None:
        """Runs the command."""
        self.emit(await service.build(self, Settings.load()))


class SimulateCommand(service.SimulateRequest, _Output):
    """Build a script's circuit, then run every analysis of a plan on it."""

    async def cli_cmd(self) -> None:
        """Runs the command."""
        self.emit(await service.simulate(self, Settings.load()))


class MergeCommand(service.MergeRequest, _Output):
    """Merge the shards of a plan's Monte Carlo analysis, and report them as one run."""

    async def cli_cmd(self) -> None:
        """Runs the command."""
        self.emit(await service.merge(self, Settings.load()))


class PlotCommand(service.PlotRequest, _Output):
    """Plot the analyses of a saved results file."""

    async def cli_cmd(self) -> None:
        """Runs the command."""
        self.emit(await service.plot(self, Settings.load()))


class DrawCommand(service.DrawRequest, _Output):
    """Draw the schematic of a saved circuit's top component, or of an instance's."""

    async def cli_cmd(self) -> None:
        """Runs the command."""
        self.emit(await service.draw(self, Settings.load()))


class ViewCommand(service.ViewRequest, _Output):
    """Write a 3D viewer of a saved circuit and its results."""

    async def cli_cmd(self) -> None:
        """Runs the command."""
        self.emit(await service.view(self, Settings.load()))


class CheckCommand(service.CheckRequest, _Output):
    """Check the structure of a saved circuit file."""

    async def cli_cmd(self) -> None:
        """Runs the command."""
        self.emit(await service.check(self, Settings.load()))


class ExplainCommand(service.ExplainRequest, _Output):
    """Explain rules: the exact condition each checks, and its level."""

    async def cli_cmd(self) -> None:
        """Runs the command."""
        self.emit(await service.explain(self, Settings.load()))


class CostCommand(service.CostRequest, _Output):
    """Price a component of a saved circuit file with a catalog's parts."""

    async def cli_cmd(self) -> None:
        """Runs the command."""
        self.emit(await service.cost(self, Settings.load()))


class KicadCommand(service.KicadRequest, _Output):
    """Write a component of a saved circuit file as a KiCad netlist."""

    async def cli_cmd(self) -> None:
        """Runs the command."""
        self.emit(await service.write_kicad(self, Settings.load()))


class Jolty(BaseSettings):
    """Build and simulate electronic circuits written as Python scripts."""

    model_config = SettingsConfigDict(
        cli_prog_name="jolty",
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_hide_none_type=True,
        cli_avoid_json=True,
        cli_enforce_required=True,
    )

    build: CliSubCommand[BuildCommand] = Field(
        description="Build a script's circuit and elaborate it."
    )
    simulate: CliSubCommand[SimulateCommand] = Field(
        description="Build a script's circuit and run a plan's analyses on it."
    )
    merge: CliSubCommand[MergeCommand] = Field(
        description="Merge the shards of a plan's Monte Carlo analysis into one run."
    )
    plot: CliSubCommand[PlotCommand] = Field(
        description="Plot the analyses of a saved results file."
    )
    draw: CliSubCommand[DrawCommand] = Field(
        description="Draw a component of a saved circuit file."
    )
    view: CliSubCommand[ViewCommand] = Field(
        description="Write a 3D viewer of a saved circuit and its results."
    )
    check: CliSubCommand[CheckCommand] = Field(
        description="Check the structure of a saved circuit file."
    )
    explain: CliSubCommand[ExplainCommand] = Field(
        description="Explain rules and their levels."
    )
    cost: CliSubCommand[CostCommand] = Field(
        description="Price a component of a saved circuit file with a catalog."
    )
    kicad: CliSubCommand[KicadCommand] = Field(
        description="Write a component of a saved circuit file as a KiCad netlist."
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

        Configuration lives in [`Settings`][jolty.settings.Settings].
        """
        return (init_settings,)

    def cli_cmd(self) -> None:
        """Runs the chosen subcommand."""
        CliApp.run_subcommand(self)


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the command line.

    The exit status is 1 when a circuit does not elaborate, a check finds an error, an
    analysis fails, a merged Monte Carlo analysis's yield misses its target, no part
    prices a device instance or a component's netlist cannot be written, and 2 when the
    arguments or settings are invalid, a command needs a subcommand, or a command
    fails, as when a script raises. The lone argument ``--version`` prints the version.
    """
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments == ["--version"]:
        print(f"jolty {version('jolty')}")
        return
    try:
        CliApp.run(Jolty, cli_args=arguments)
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
    """Names each invalid option and says what it accepts."""
    problems = []
    for problem in error.errors():
        names = [part for part in problem["loc"] if isinstance(part, str)]
        option = "--" + names[-1].replace("_", "-") if names else "arguments"
        problems.append(f"{option}: {problem['msg']}")
    return "invalid arguments: " + "; ".join(problems)

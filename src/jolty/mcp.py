# Copyright 2026 Maxence Faldor
"""The MCP server: jolty's commands as tools, for any agent harness.

Every tool derives from a request model of [`jolty.service`][jolty.service], so its
arguments, defaults and descriptions match the ``jolty`` command line exactly. Each tool
answers with the text the command line prints, with the result as structured content,
and with the images it drew.

Scripts run in a process of their own, outside the server, so nothing a script prints
can reach the protocol stream. Serve with ``jolty-mcp`` or ``python -m jolty.mcp``.
"""

import base64
import copy
import inspect
import sys
from collections.abc import Awaitable, Callable, Sequence
from importlib.metadata import version
from typing import Annotated, Final

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import (
    CallToolResult,
    ContentBlock,
    ImageContent,
    TextContent,
    ToolAnnotations,
)
from pydantic import BaseModel

from jolty import service
from jolty.errors import JoltyError
from jolty.settings import Settings

INSTRUCTIONS: Final = """\
jolty builds and simulates electronic circuits written as Python scripts.

Workflow: write a script whose build() returns the top component of a circuit, and call
build_circuit on it. It saves the circuit and says how many devices and unknowns it has,
or every reason the engine cannot elaborate it, checks its structure, and draws its
schematic: each diagnostic, such as error[no-dc-path], names the instances and the
script lines that placed them, and the schematic marks them in red. explain_rule gives
the exact condition behind a rule. Then add plan(), returning a jolty.Plan of named
analyses, and call simulate_circuit: it reports each analysis, writes the first
operating point's voltages on the schematic, plots each analysis but operating points
and sensitivities, and saves every result to a .npz file that
jolty.files.load_results reads back; plot_results draws other unknowns from it, and
draw_schematic draws the component inside an instance. Fix the script and run again
until the report has no errors and the analyses show the intended behavior.
price_circuit prices the device instances of a component of a saved circuit with a
catalog file of parts: each resistor, capacitor and inductor takes the cheapest part of
its value whose tolerance is within its own.

Components: decorate a function taking a jolty.Builder with @jolty.component. Inside,
c.port(name, role) declares a port (Role.SIGNAL, SUPPLY or GROUND), c.place(device or
component, id=...) places an instance whose terminals are its attributes, and
c.connect(*points, name=...) joins terminals and nets into one net, named if given.
Calling the decorated function with its parameters returns a component that another
component places. The top component has a ground port, the reference of every voltage.

Devices, in jolty.devices: resistor (a, b), capacitor (a, b), inductor (a, b),
voltage_source and current_source (positive, negative; a level or a waveform of
jolty.waveforms: Dc, Pulse, Sine, Piecewise), voltage_controlled_voltage_source and
voltage_controlled_current_source (positive, negative, control_positive,
control_negative), and diode (anode, cathode; SPICE parameters). Values carry units:
10 * kohm, 4.7 * uF, from jolty.units. Logic devices join logic nets, which an event
scheduler simulates: logic_gate(gate, inputs, rise=, transition=) with inputs a, b and
output y, logic_cell(inputs, outputs, rise=, transition=), and logic_source(initial,
changes, transition=) with output output; electrical=True gives a cell supply and ground
terminals, so its pins may join electrical nets. Results read a logic net's level with
level(net): 0, 1, or NaN for unknown. A plan's fidelity, such as
fidelity={"adder/*": Characterized(3.3 * V)}, simulates the component instances a scope
matches as logic cells characterized from their circuits, whose ports are Role.INPUT,
Role.OUTPUT, one supply and one ground, and spot checks hold each cell to its circuit.

Analyses, in jolty.plan: OperatingPoint(), OperatingPoints(most=8): every DC solution,
such as a latch's states, read with jolty.specs.solutions, DcSweep(source, levels),
Ac(source, frequencies) with decade(start, stop, per_decade), Transient(stop,
max_step, method), Sensitivity(outputs, of=None): the derivative of each output unknown
with respect to every number of every device's parameters at the operating point, or
at each level, frequency or time of the DcSweep, Ac or Transient given as of, and
Stability(probe, frequencies): the loop gain through a 0 V voltage source in series
with a feedback loop, its positive terminal toward the side the signal enters, read
with jolty.specs.phase_margin, gain_margin and crossover, and Noise(output,
frequencies, reference=None, source=None): the noise density at an output from every
device's thermal, shot and flicker noise, referred to the input source if given, read
with jolty.specs.noise_density (in V/√Hz, such as 10 * nV / rtHz) and
integrated_noise, and PoleZero(source=None, output=None, reference=None): the poles of
the linearized circuit, and the zeros of the transfer function from source to output,
read with jolty.specs.damping and dominant_pole. Unknowns
are named by path: a net such as out or stage/mid, and the current through a source or
inductor such as V1:current. A result records the instances its options name, as
Plan(..., options=Options(record=["X1/R1", "X2"])) does, or every instance, as
Options(record=["**"]) does: the current into each terminal and the charge it holds,
read with current(instance, terminal) and charge(instance, terminal), such as
current("X1/R1", "a"), a component instance's terminals summing the devices' behind
them, and each device's conductances and variables. By default it records the
instances rated for a current or a power. Transient(stop, times=[...]) steps onto the
given times exactly, and Transient(period, periodic=True) simulates one period of the
periodic steady state, found by shooting.
"""

READS_FILES: Final = ToolAnnotations(read_only_hint=True, open_world_hint=False)
"""A check reads a circuit file and changes nothing."""

WRITES_FILES: Final = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
"""A plot or schematic writes an image beside the file it reads, the same each time."""

RUNS_CODE: Final = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=True,
)
"""A command runs any Python with the user's permissions, so harnesses should ask."""


def tool_from[Request: BaseModel, Result: service.CommandResult](
    name: str,
    request_type: type[Request],
    handler: Callable[[Request, Settings], Awaitable[Result]],
    result_type: type[Result],
    settings: Settings,
) -> Callable[..., Awaitable[CallToolResult]]:
    """An async tool whose parameters are exactly the fields of `request_type`.

    Each parameter's default is a copy of the request field's ``FieldInfo``, so the
    tool's schema carries the request model's descriptions, defaults and constraints.
    The tool answers with the result's text, and with the result itself as structured
    content, validated against `result_type`.
    """
    parameters = [
        inspect.Parameter(
            field,
            inspect.Parameter.KEYWORD_ONLY,
            default=copy.copy(info),
            annotation=info.annotation,
        )
        for field, info in request_type.model_fields.items()
    ]

    async def tool(**arguments: object) -> CallToolResult:
        try:
            request = request_type.model_validate(arguments)
            result = await handler(request, settings)
        except JoltyError as error:
            raise ToolError(str(error)) from error
        content: list[ContentBlock] = [TextContent(type="text", text=result.describe())]
        content.extend(
            ImageContent(
                type="image",
                data=base64.b64encode(path.read_bytes()).decode("ascii"),
                mime_type="image/png",
            )
            for path in result.images()
        )
        return CallToolResult(
            content=content, structured_content=result.model_dump(mode="json")
        )

    tool.__signature__ = inspect.Signature(  # ty: ignore[missing-attribute]
        parameters, return_annotation=Annotated[CallToolResult, result_type]
    )
    tool.__name__ = name
    tool.__doc__ = request_type.__doc__
    return tool


def create_server(settings: Settings) -> MCPServer:
    """The jolty MCP server, with all its tools, running commands with `settings`."""
    server = MCPServer("jolty", version=version("jolty"), instructions=INSTRUCTIONS)

    def register[Request: BaseModel, Result: service.CommandResult](
        name: str,
        request_type: type[Request],
        handler: Callable[[Request, Settings], Awaitable[Result]],
        result_type: type[Result],
        annotations: ToolAnnotations,
    ) -> None:
        server.add_tool(
            tool_from(name, request_type, handler, result_type, settings),
            annotations=annotations,
            structured_output=True,
        )

    register(
        "build_circuit",
        service.BuildRequest,
        service.build,
        service.BuildResult,
        RUNS_CODE,
    )
    register(
        "simulate_circuit",
        service.SimulateRequest,
        service.simulate,
        service.SimulateResult,
        RUNS_CODE,
    )
    register(
        "plot_results",
        service.PlotRequest,
        service.plot,
        service.PlotResult,
        WRITES_FILES,
    )
    register(
        "draw_schematic",
        service.DrawRequest,
        service.draw,
        service.DrawResult,
        WRITES_FILES,
    )
    register(
        "write_viewer",
        service.ViewRequest,
        service.view,
        service.ViewResult,
        WRITES_FILES,
    )
    register(
        "check_circuit",
        service.CheckRequest,
        service.check,
        service.CheckResult,
        READS_FILES,
    )
    register(
        "explain_rule",
        service.ExplainRequest,
        service.explain,
        service.ExplainResult,
        READS_FILES,
    )
    register(
        "price_circuit",
        service.CostRequest,
        service.cost,
        service.CostResult,
        READS_FILES,
    )
    register(
        "write_kicad_netlist",
        service.KicadRequest,
        service.write_kicad,
        service.KicadResult,
        WRITES_FILES,
    )
    return server


USAGE: Final = """\
usage: jolty-mcp [--help | --version]

Serve jolty's tools to an agent harness over the Model Context Protocol, on stdio.
"""


def main(argv: Sequence[str] | None = None) -> None:
    """Serves over stdio, or answers ``--help`` or ``--version``.

    The exit status is 2 for invalid arguments or settings.
    """
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments == ["--version"]:
        sys.stdout.write(f"jolty-mcp {version('jolty')}\n")
        return
    if arguments in (["--help"], ["-h"]):
        sys.stdout.write(USAGE)
        return
    if arguments:
        sys.stderr.write(USAGE)
        raise SystemExit(2)
    try:
        settings = Settings.load()
    except JoltyError as error:
        sys.stderr.write(f"error: {error}\n")
        raise SystemExit(2) from None
    if sys.stdin.isatty():
        sys.stderr.write(
            "jolty-mcp serves MCP on stdio for a harness to start; Ctrl-C stops it.\n"
        )
    create_server(settings).run()


if __name__ == "__main__":
    main()

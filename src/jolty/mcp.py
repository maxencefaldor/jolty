# Copyright 2026 Maxence Faldor
"""The MCP server: jolty's commands as tools, for any agent harness.

Every tool derives from a request model of [`jolty.service`][jolty.service], so its
arguments, defaults and descriptions match the ``jolty`` command line exactly. Each tool
answers with the text the command line prints, with the result as structured content,
and with the images it drew. Offline guide pages and example scripts are served as MCP
resources and through ``read_docs`` from the same single loader in ``jolty.service``.
The server's instructions derive programmatically from the code so every public device,
real part, analysis, measurement and offline resource URI stays in sync.

Scripts run in a process of their own, outside the server, so nothing a script prints
can reach the protocol stream. Serve with ``jolty-mcp`` or ``python -m jolty.mcp``.
"""

import asyncio
import base64
import copy
import inspect
import re
import sys
import typing
from collections.abc import Awaitable, Callable, Sequence
from importlib.metadata import version
from typing import Annotated, Final

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp.server.mcpserver.resources.types import FunctionResource
from mcp.types import (
    CallToolResult,
    ContentBlock,
    ImageContent,
    TextContent,
    ToolAnnotations,
)
from pydantic import BaseModel

from jolty import builder, devices, parts, plan, service, specs, waveforms
from jolty.errors import JoltyError
from jolty.settings import Settings

_MAX_INLINE_KEYWORD_PARAMETERS: Final = 5
"""List optional keyword parameters inline when at most this many; otherwise ``...``."""

_MKDOCS_CROSS_REFERENCE: Final = re.compile(r"\[`?([^`\]]+)`?\]\[[^\]]+\]")
_MARKDOWN_LINK: Final = re.compile(r"\[([^\]]+)\]\([^)]+\)")

_WORKFLOW_OVERVIEW: Final = """\
jolty builds and simulates electronic circuits written as Python scripts.

Workflow: write a script whose build() returns the top component of a circuit, and call
build_circuit on it. It saves the circuit, reports device and unknown counts (or every
reason the engine cannot elaborate it), checks its structure, and draws its schematic:
each diagnostic, such as error[no-dc-path], names the instances and the script lines
that placed them, and the schematic marks them in red. explain_rules gives the exact
condition behind each rule. Add plan() returning a jolty.Plan of named analyses, and
optionally specs() returning jolty.Specs of requirements against those analyses, then
call simulate_circuit: it reports each analysis, transistor operating-point regions and
margins nearest a boundary first (OperatingPointResult.region(path)), writes the first
operating point's voltages on the schematic, checks specs() verdicts (passed, marginal,
failed, unmeasured, with slack, worst corner and Wilson yield interval), plots each
analysis except operating points and sensitivities, and saves results to a .npz file
readable by jolty.files.load_results. plot_results draws other unknowns from saved
results; draw_schematic draws the component inside an instance; view_circuit writes a
standalone 3D HTML viewer; check_circuit checks a saved circuit; cost_circuit prices
instances against a catalog file; export_kicad_netlist writes a KiCad netlist; and
read_docs (or MCP resources below) reads the offline guide and examples.
simulate_circuit with samples, such as "0:1000000", runs only that range of the plan's
Monte Carlo analysis as a shard on any machine, and merge_shards merges a directory of
shards into one bit-identical report. Exact derivatives of analyses are available in
Python via jolty.function, jolty.jvp, jolty.vjp, and jolty.jax.function.

Components: decorate a function taking a jolty.Builder with @jolty.component. Inside,
c.port(name, role) declares a port (Role.SIGNAL, SUPPLY, GROUND, INPUT or OUTPUT),
c.place(device or component, name=...) places an instance whose terminals are its
attributes, and c.connect(*points, name=...) joins terminals and nets into one net. The
top component has a ground port, the reference of every voltage. Values carry units from
jolty.units: 10 * kohm, 4.7 * uF, 10 * nV / rtHz, celsius(-40).

Unknowns and recording: unknowns are named by path (a net voltage such as out or
stage/mid, or a source/inductor current such as V1:current). Results record instances
matched by record patterns in Plan or Recorded analyses (such as record=["X1/**",
"!X1/bias/**"] or record=["**"], a pattern after ! taking back earlier ones): read with
current(instance, terminal) and charge(instance, terminal), plus device conductances and
variables; operating point and transient also record instances rated for current or
power. Logic devices join logic nets simulated by an event scheduler; electrical=True
adds supply and ground pins to join electrical nets; results read logic levels with
level(net): 0, 1, or NaN for unknown. Plan fidelity (such as
fidelity={"adder/*": Characterized(3.3 * V)}) swaps matching components for logic cells
characterized from their circuits and verified by spot checks."""


def _clean_summary(docstring: str | None) -> str:
    """Extracts and cleans the first summary line of a docstring."""
    if docstring is None:
        return ""
    first_line = ""
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break
    if not first_line:
        return ""
    cleaned = _MKDOCS_CROSS_REFERENCE.sub(r"\1", first_line)
    cleaned = _MARKDOWN_LINK.sub(r"\1", cleaned)
    cleaned = cleaned.replace("``", "`")
    return " ".join(cleaned.split())


def _return_hint(callable_object: object) -> object:
    """Resolves the return annotation of a function, returning None when unavailable."""
    if not inspect.isfunction(callable_object):
        return None
    try:
        hints = typing.get_type_hints(callable_object)
    except AttributeError, NameError, TypeError:
        return None
    return hints.get("return")


def _format_parameter(parameter: inspect.Parameter) -> str:
    """Formats one signature parameter as ``name`` or ``name=...`` when optional."""
    if parameter.default is inspect.Parameter.empty:
        return parameter.name
    return f"{parameter.name}=..."


def _format_keyword_section(
    required: Sequence[str],
    optional: Sequence[str],
    *,
    has_variable_positional: bool,
) -> list[str]:
    """Formats keyword-only parameters, collapsing large optional parameter sets."""
    if not required and not optional:
        return []
    parts_list: list[str] = [] if has_variable_positional else ["*"]
    parts_list.extend(required)
    if len(optional) <= _MAX_INLINE_KEYWORD_PARAMETERS:
        parts_list.extend(optional)
    else:
        parts_list.append(f"... ({len(optional)} optional)")
    return parts_list


def _compact_signature(
    callable_object: Callable[..., object], *, skip_first: bool = False
) -> str:
    """Formats a compact parameter list without dumping large optional tables."""
    parameters = list(inspect.signature(callable_object).parameters.values())
    if skip_first and parameters:
        parameters = parameters[1:]
    positional: list[str] = []
    keyword_required: list[str] = []
    keyword_optional: list[str] = []
    variable_positional: str | None = None
    variable_keyword: str | None = None
    for parameter in parameters:
        match parameter.kind:
            case (
                inspect.Parameter.POSITIONAL_ONLY
                | inspect.Parameter.POSITIONAL_OR_KEYWORD
            ):
                positional.append(_format_parameter(parameter))
            case inspect.Parameter.VAR_POSITIONAL:
                variable_positional = f"*{parameter.name}"
            case inspect.Parameter.KEYWORD_ONLY:
                target = (
                    keyword_required
                    if parameter.default is inspect.Parameter.empty
                    else keyword_optional
                )
                target.append(_format_parameter(parameter))
            case inspect.Parameter.VAR_KEYWORD:
                variable_keyword = f"**{parameter.name}"

    rendered = [*positional]
    if variable_positional is not None:
        rendered.append(variable_positional)
    rendered.extend(
        _format_keyword_section(
            keyword_required,
            keyword_optional,
            has_variable_positional=variable_positional is not None,
        )
    )
    if variable_keyword is not None:
        rendered.append(variable_keyword)
    return f"({', '.join(rendered)})"


def _is_component_factory(callable_object: object) -> bool:
    """Whether `callable_object` is a ``@component``-decorated builder factory."""
    if not inspect.isfunction(callable_object) or not hasattr(
        callable_object, "__wrapped__"
    ):
        return False
    try:
        hints = typing.get_type_hints(callable_object)
    except AttributeError, NameError, TypeError:
        return False
    parameters = list(inspect.signature(callable_object).parameters)
    return bool(parameters) and hints.get(parameters[0]) is builder.Builder


def _format_waveforms() -> str:
    """Formats every waveform class in `jolty.waveforms`."""
    classes = typing.get_args(waveforms.Waveform.__value__)
    lines = ["Waveforms (jolty.waveforms):"]
    for waveform_class in classes:
        signature = _compact_signature(waveform_class)
        summary = _clean_summary(waveform_class.__doc__)
        lines.append(f"- {waveform_class.__name__}{signature}: {summary}")
    return "\n".join(lines)


def _format_devices() -> str:
    """Formats every public device constructor in `jolty.devices`."""
    lines = ["Devices (jolty.devices):"]
    for name, candidate in vars(devices).items():
        if name.startswith("_") or not inspect.isfunction(candidate):
            continue
        if getattr(candidate, "__module__", None) != devices.__name__:
            continue
        if _return_hint(candidate) is not devices.Device:
            continue
        signature = _compact_signature(candidate)
        summary = _clean_summary(candidate.__doc__)
        lines.append(f"- {name}{signature}: {summary}")
    return "\n".join(lines)


def _format_parts() -> str:
    """Formats every real part constructor and datasheet preset in `jolty.parts`."""
    lines = ["Real parts (jolty.parts):"]
    datasheets: list[str] = []
    for name, candidate in vars(parts).items():
        if name.startswith("_"):
            continue
        if isinstance(candidate, parts.Datasheet):
            datasheets.append(name)
            continue
        if not inspect.isfunction(candidate):
            continue
        if getattr(candidate, "__module__", None) != parts.__name__:
            continue
        returned = _return_hint(candidate)
        if returned in (devices.Device, builder.Component):
            signature = _compact_signature(candidate)
            summary = _clean_summary(candidate.__doc__)
            lines.append(f"- {name}{signature}: {summary}")
        elif _is_component_factory(candidate):
            signature = _compact_signature(candidate, skip_first=True)
            summary = _clean_summary(candidate.__doc__)
            lines.append(f"- {name}{signature}: {summary}")
    if datasheets:
        lines.append(f"- Datasheet presets (parts.Datasheet): {', '.join(datasheets)}")
    return "\n".join(lines)


def _format_analyses() -> str:
    """Formats every analysis class in `jolty.plan`."""
    analysis_classes = typing.get_args(plan.Analysis.__value__)
    lines = ["Analyses (jolty.plan):"]
    for analysis_class in analysis_classes:
        signature = _compact_signature(analysis_class)
        summary = _clean_summary(analysis_class.__doc__)
        lines.append(f"- {analysis_class.__name__}{signature}: {summary}")
    return "\n".join(lines)


def _format_plan_controls() -> str:
    """Formats plan configuration, sweep, corner and batching helpers in `plan`."""
    variant_classes = typing.get_args(plan.Variants.__value__)
    items: tuple[type[object] | Callable[..., object], ...] = (
        plan.Plan,
        plan.Options,
        plan.Parallelism,
        plan.Corner,
        *variant_classes,
        plan.Variant,
        plan.Axis,
        plan.WorstCase,
        plan.Characterized,
        plan.decade,
        plan.linear,
    )
    lines = ["Plan, corners, variants, sweeps and fidelity (jolty.plan):"]
    for item in items:
        signature = _compact_signature(item)
        summary = _clean_summary(item.__doc__)
        lines.append(f"- {item.__name__}{signature}: {summary}")
    return "\n".join(lines)


def _format_specs() -> str:
    """Formats `Specs`, `specs()`, and every public measurement in `jolty.specs`."""
    specs_signature = _compact_signature(specs.Specs)
    header = (
        f"Specs and measurements (jolty.specs): define specs() returning "
        f"Specs{specs_signature} to verify requirements such as "
        'value("op", "out") >= 2 * V across nominal, corners, Monte Carlo (Wilson '
        "interval via wilson) and worst case."
    )
    lines = [header]
    for name in specs.__all__:
        candidate = getattr(specs, name)
        if not inspect.isfunction(candidate):
            continue
        if _return_hint(candidate) is not specs.Measure:
            continue
        signature = _compact_signature(candidate)
        summary = _clean_summary(candidate.__doc__)
        lines.append(f"- {name}{signature}: {summary}")
    return "\n".join(lines)


def _format_resources() -> str:
    """Formats the offline documentation resources and templates."""
    entries = service.document_summaries()
    guide_uris = [entry.uri for entry in entries if entry.section == "guide"]
    example_uris = [entry.uri for entry in entries if entry.section == "examples"]
    header = (
        "Offline documentation and examples (read via MCP resources, read_docs tool, "
        "or `jolty docs [TOPIC]`):"
    )
    templates = (
        f"- Templates: {service.DOCS_URI_PREFIX}/guide/{{slug}}, "
        f"{service.DOCS_URI_PREFIX}/examples/{{+slug}}"
    )
    lines = [
        header,
        templates,
        f"- Guide ({len(guide_uris)}): {', '.join(guide_uris)}",
        f"- Examples ({len(example_uris)}): {', '.join(example_uris)}",
    ]
    return "\n".join(lines)


def build_instructions() -> str:
    """Builds the MCP server instructions programmatically from code and offline docs.

    Raises:
        JoltyError: When the offline documentation catalog cannot be loaded.
    """
    sections = (
        _WORKFLOW_OVERVIEW,
        _format_waveforms(),
        _format_devices(),
        _format_parts(),
        _format_analyses(),
        _format_plan_controls(),
        _format_specs(),
        _format_resources(),
    )
    return "\n\n".join(sections) + "\n"


INSTRUCTIONS: Final[str] = build_instructions()
"""Server instructions derived from `jolty`'s public code and offline docs catalog."""

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

    tool.__signature__ = inspect.Signature(  # ty: ignore[missing-attribute] - inspect reads it; functions declare none
        parameters, return_annotation=Annotated[CallToolResult, result_type]
    )
    tool.__name__ = name
    tool.__doc__ = request_type.__doc__
    return tool


def _resource_reader(
    section: service.DocumentSection, slug: str
) -> Callable[[], Awaitable[str]]:
    """Creates an async reader bound to one static offline documentation resource."""

    async def read() -> str:
        try:
            document = await asyncio.to_thread(service.read_document, section, slug)
        except JoltyError as error:
            raise ResourceNotFoundError(str(error)) from error
        return document.content

    return read


def _register_resources(server: MCPServer) -> None:
    """Registers concrete offline documentation resources and URI templates."""
    for entry in service.document_summaries():
        server.add_resource(
            FunctionResource.from_function(
                fn=_resource_reader(entry.section, entry.slug),
                uri=entry.uri,
                name=f"{entry.section}/{entry.slug}",
                title=entry.title,
                description=entry.summary,
                mime_type=entry.mime_type,
            )
        )

    @server.resource(
        f"{service.DOCS_URI_PREFIX}/guide/{{slug}}",
        name="guide",
        title="Guide page",
        description="Read a jolty guide page offline by slug.",
        mime_type=service.GUIDE_MIME_TYPE,
    )
    async def read_guide(slug: str) -> str:
        try:
            document = await asyncio.to_thread(service.read_document, "guide", slug)
        except JoltyError as error:
            raise ResourceNotFoundError(str(error)) from error
        return document.content

    @server.resource(
        f"{service.DOCS_URI_PREFIX}/examples/{{+slug}}",
        name="examples",
        title="Example script",
        description="Read a jolty example script offline by slug.",
        mime_type=service.EXAMPLE_MIME_TYPE,
    )
    async def read_example(slug: str) -> str:
        try:
            document = await asyncio.to_thread(service.read_document, "examples", slug)
        except JoltyError as error:
            raise ResourceNotFoundError(str(error)) from error
        return document.content


def create_server(settings: Settings) -> MCPServer:
    """The jolty MCP server, with all its tools and offline documentation resources."""
    server = MCPServer(
        "jolty", version=version("jolty"), instructions=build_instructions()
    )

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
        "merge_shards",
        service.MergeRequest,
        service.merge,
        service.MergeResult,
        WRITES_FILES,
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
        "view_circuit",
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
        "explain_rules",
        service.ExplainRequest,
        service.explain,
        service.ExplainResult,
        READS_FILES,
    )
    register(
        "read_docs",
        service.DocsRequest,
        service.docs,
        service.DocsResult,
        READS_FILES,
    )
    register(
        "cost_circuit",
        service.CostRequest,
        service.cost,
        service.CostResult,
        READS_FILES,
    )
    register(
        "export_kicad_netlist",
        service.ExportKicadRequest,
        service.export_kicad,
        service.ExportKicadResult,
        WRITES_FILES,
    )
    _register_resources(server)
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

# Copyright 2026 Maxence Faldor
"""The MCP server: jolty's commands as tools, for any agent harness.

Every tool derives from a request model of [`jolty.service`][jolty.service], so its
arguments, defaults and descriptions match the ``jolty`` command line exactly. Each tool
answers with the text the command line prints, with the result as structured content,
and with the images it drew. Offline guide pages and example scripts are served as MCP
resources and through ``read_docs`` from the same single loader in ``jolty.service``.
The server's instructions derive programmatically in ``jolty.service`` so every public
device, real part, analysis, measure and offline resource URI stays in sync.

Scripts run in a process of their own, outside the server, so nothing a script prints
can reach the protocol stream. Serve with ``jolty-mcp`` or ``python -m jolty.mcp``.
"""

__lazy_modules__ = ["base64", "importlib.metadata"]

import asyncio
import base64
import copy
import inspect
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

from jolty import service
from jolty.errors import JoltyError
from jolty.service import build_instructions
from jolty.settings import Settings

__all__ = [
    "INSTRUCTIONS",
    "READS_FILES",
    "RUNS_CODE",
    "USAGE",
    "WRITES_FILES",
    "build_instructions",
    "create_server",
    "main",
    "tool_from",
]

if typing.TYPE_CHECKING:
    INSTRUCTIONS: str = ""


def __getattr__(name: str) -> str:
    """Resolves `INSTRUCTIONS` lazily on first access to avoid import-time I/O."""
    if name == "INSTRUCTIONS":
        return build_instructions()
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)


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
        images = [await asyncio.to_thread(path.read_bytes) for path in result.images()]
        content.extend(
            ImageContent(
                type="image",
                data=base64.b64encode(raw).decode("ascii"),
                mime_type="image/png",
            )
            for raw in images
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


async def _read_resource(section: service.DocumentSection, slug: str) -> str:
    """Reads one offline document on a worker thread, mapping errors for MCP."""
    try:
        document = await asyncio.to_thread(service.read_document, section, slug)
    except JoltyError as error:
        raise ResourceNotFoundError(str(error)) from error
    return document.content


def _resource_reader(
    section: service.DocumentSection, slug: str
) -> Callable[[], Awaitable[str]]:
    """Creates an async reader bound to one static offline documentation resource."""

    async def read() -> str:
        return await _read_resource(section, slug)

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
        return await _read_resource("guide", slug)

    @server.resource(
        f"{service.DOCS_URI_PREFIX}/examples/{{+slug}}",
        name="examples",
        title="Example script",
        description="Read a jolty example script offline by slug.",
        mime_type=service.EXAMPLE_MIME_TYPE,
    )
    async def read_example(slug: str) -> str:
        return await _read_resource("examples", slug)


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
        "measure_results",
        service.MeasureRequest,
        service.measure,
        service.MeasureResult,
        READS_FILES,
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
        service.ReadDocsRequest,
        service.read_docs,
        service.ReadDocsResult,
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

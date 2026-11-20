# Copyright 2026 Maxence Faldor
"""MCP server exposing jolty commands as tools for agent harnesses.

Every tool derives from a request model in [`jolty.service`][jolty.service], keeping
arguments, defaults, and descriptions synchronized with the ``jolty`` CLI. Each tool
returns the formatted CLI text, structured JSON output, and any generated images.
Offline guide pages and example scripts are served both as MCP resources and through
``read_docs`` from the shared loader in ``jolty.service``. Server instructions are
assembled programmatically in ``jolty.service`` so every public device, real part,
analysis, measure, and offline resource URI stays up to date.

Scripts execute in a separate subprocess outside the server so script stdout cannot
corrupt the protocol stream. Start the server with ``jolty-mcp`` or
``python -m jolty.mcp``.
"""

__lazy_modules__ = ["base64", "jolty._core"]

import asyncio
import base64
import copy
import inspect
import sys
import typing
from collections.abc import Awaitable, Callable, Sequence
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
from jolty._core import __version__ as _version
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
"""Tool annotations for read-only commands that inspect files without modification."""

WRITES_FILES: Final = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
"""Tool annotations for idempotent commands that write images next to input files."""

RUNS_CODE: Final = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=True,
)
"""Tool annotations for commands that execute user Python scripts locally."""


def tool_from[Request: BaseModel, Result: service.CommandResult](
    name: str,
    request_type: type[Request],
    handler: Callable[[Request, Settings], Awaitable[Result]],
    result_type: type[Result],
    settings: Settings,
) -> Callable[..., Awaitable[CallToolResult]]:
    """Builds an async MCP tool whose keyword parameters match `request_type` fields.

    Each parameter default copies the corresponding request field ``FieldInfo`` so the
    tool schema preserves descriptions, defaults, and constraints. Calling the tool
    returns formatted result text, generated images, and structured output validated
    against `result_type`.

    Args:
        name: MCP tool name assigned to the generated callable.
        request_type: Pydantic request model defining tool keyword arguments.
        handler: Async service handler invoked with validated request and settings.
        result_type: Pydantic command result class annotating structured output.
        settings: Runtime settings passed to `handler` on every invocation.

    Returns:
        Async tool callable with dynamically synthesized signature and metadata.
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

    tool.__signature__ = inspect.Signature(  # ty: ignore[missing-attribute] - dynamic
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
    """Creates the jolty MCP server with all tools and offline doc resources.

    Args:
        settings: Runtime settings bound to every registered MCP tool handler.

    Returns:
        Configured `MCPServer` ready to serve over stdio.
    """
    server = MCPServer("jolty", version=_version, instructions=build_instructions())

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

    Args:
        argv: Command-line tokens to parse, defaulting to ``sys.argv[1:]`` when
            ``None``.
    """
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments == ["--version"]:
        sys.stdout.write(f"jolty-mcp {_version}\n")
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
    service.warm_runner()
    try:
        create_server(settings).run()
    finally:
        service.close_standby()


if __name__ == "__main__":
    main()

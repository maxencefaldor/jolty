# Copyright 2026 Maxence Faldor

import asyncio
import os
import sys
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult, TextContent, Tool

from jolty.mcp import create_server, main
from jolty.settings import Settings

DIVIDER = """
from jolty import Builder, OperatingPoint, Plan, Role, component, devices
from jolty.units import V, kohm


@component
def divider(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), id="V1")
    upper = c.place(devices.resistor(1 * kohm), id="R1")
    lower = c.place(devices.resistor(1 * kohm), id="R2")
    c.connect(source.positive, upper.a)
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


def build():
    print("{not json}" * 20)
    return divider()


def plan():
    return Plan({"op": OperatingPoint()})
"""


@pytest.fixture
def script(tmp_path: Path) -> Path:
    path = tmp_path / "divider.py"
    path.write_text(DIVIDER, encoding="utf-8")
    return path


def call(name: str, arguments: Mapping[str, object]) -> CallToolResult:
    async def go() -> CallToolResult:
        async with Client(create_server(Settings())) as client:
            return await client.call_tool(name, dict(arguments))

    return asyncio.run(go())


def text_of(result: CallToolResult) -> str:
    (content,) = result.content
    assert isinstance(content, TextContent)
    return content.text


def test_tools_mirror_the_service_requests() -> None:
    async def go() -> list[Tool]:
        async with Client(create_server(Settings())) as client:
            return (await client.list_tools()).tools

    tools = {tool.name: tool for tool in asyncio.run(go())}
    assert set(tools) == {"build_circuit", "simulate_circuit"}
    simulate = tools["simulate_circuit"]
    assert set(simulate.input_schema["properties"]) == {
        "script",
        "plan",
        "output",
        "results",
    }
    assert simulate.input_schema["required"] == ["script"]
    assert simulate.description is not None
    assert simulate.description.startswith("Build a script's circuit")
    for tool in tools.values():
        assert tool.output_schema is not None
        assert tool.annotations is not None
        assert tool.annotations.destructive_hint is True  # a script runs any Python


def test_simulate_answers_with_text_and_structured_content(script: Path) -> None:
    result = call("simulate_circuit", {"script": str(script)})
    assert not result.is_error
    assert "op: operating point, found by Newton iteration" in text_of(result)
    data = result.structured_content
    assert data is not None
    assert data["circuit"]["unknowns"] == 3
    (point,) = data["analyses"]
    assert point["values"][1] == {"unknown": "out", "quantity": "voltage", "value": 2.5}


def test_errors_come_back_as_tool_errors(tmp_path: Path) -> None:
    result = call("build_circuit", {"script": str(tmp_path / "missing.py")})
    assert result.is_error
    assert "no such script" in text_of(result)


def test_scripts_that_print_cannot_corrupt_the_stdio_channel(script: Path) -> None:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("JOLTY_")
    }
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "jolty.mcp"],
        env=environment,
        cwd=str(script.parent),
    )

    async def go() -> CallToolResult:
        async with Client(server) as client:
            return await client.call_tool("build_circuit", {"script": script.name})

    built = asyncio.run(go())
    assert not built.is_error
    assert built.structured_content is not None
    assert built.structured_content["output"]["stdout"].startswith("{not json}")


def test_the_command_answers_help_and_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--version"])
    assert capsys.readouterr().out == f"jolty-mcp {version('jolty')}\n"
    main(["--help"])
    assert capsys.readouterr().out.startswith("usage: jolty-mcp")
    with pytest.raises(SystemExit) as exit_:
        main(["--bogus"])
    assert exit_.value.code == 2


def test_the_command_stops_on_invalid_settings(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JOLTY_TIMEOUT", "never")
    with pytest.raises(SystemExit) as exit_:
        main([])
    assert exit_.value.code == 2
    assert "invalid settings: timeout" in capsys.readouterr().err

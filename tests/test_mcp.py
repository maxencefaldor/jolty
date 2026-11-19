# Copyright 2026 Maxence Faldor

import ast
import asyncio
import importlib.resources
import json
import os
import re
import sys
import typing
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.shared.exceptions import MCPError
from mcp.types import (
    CallToolResult,
    ImageContent,
    Resource,
    ResourceTemplate,
    TextContent,
    TextResourceContents,
    Tool,
)

from jolty import plan, service, specs, waveforms
from jolty.errors import JoltyError
from jolty.files import load_raw
from jolty.mcp import INSTRUCTIONS, build_instructions, create_server, main
from jolty.settings import Settings

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
ADVICE_WORDS = re.compile(
    r"\b(?:use|pass|try|instead|should|give|did you mean)\b", re.IGNORECASE
)

DIVIDER = """
from jolty import Builder, OperatingPoint, Plan, Role, component, devices
from jolty.units import V, kohm


@component
def divider(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), name="V1")
    upper = c.place(devices.resistor(1 * kohm), name="R1")
    lower = c.place(devices.resistor(1 * kohm), name="R2")
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
    """The text of a tool result's first content, which is always text."""
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


def test_tools_mirror_the_service_requests() -> None:
    async def go() -> list[Tool]:
        async with Client(create_server(Settings())) as client:
            return (await client.list_tools()).tools

    tools = {tool.name: tool for tool in asyncio.run(go())}
    assert set(tools) == {
        "build_circuit",
        "simulate_circuit",
        "merge_shards",
        "plot_results",
        "measure_results",
        "draw_schematic",
        "view_circuit",
        "check_circuit",
        "explain_rules",
        "read_docs",
        "cost_circuit",
        "export_kicad_netlist",
    }
    simulate = tools["simulate_circuit"]
    assert set(simulate.input_schema["properties"]) == {
        "script",
        "plan",
        "output",
        "results",
        "raw",
        "raw_format",
        "check",
        "error",
        "warn",
        "ignore",
        "max_diagnostics",
        "schematic",
        "plot",
        "traces",
        "record",
        "min_solves_per_task",
        "min_instances_per_task",
        "min_unknowns_per_task",
        "max_unknowns_per_interior",
        "samples",
        "shards",
    }
    assert simulate.input_schema["required"] == ["script"]
    assert simulate.description is not None
    assert simulate.description.startswith("Build a script's circuit")
    for name in ("build_circuit", "simulate_circuit"):
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.destructive_hint is True  # a script runs any Python
    for name in (
        "measure_results",
        "check_circuit",
        "explain_rules",
        "read_docs",
        "cost_circuit",
    ):
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.read_only_hint is True
    read_docs = tools["read_docs"]
    assert set(read_docs.input_schema["properties"]) == {"topic", "section"}
    assert read_docs.input_schema.get("required", []) == []
    assert all(tool.output_schema is not None for tool in tools.values())


def test_simulate_answers_with_text_and_structured_content(script: Path) -> None:
    result = call("simulate_circuit", {"script": str(script)})
    assert not result.is_error
    assert "op: operating point, found by Newton iteration in" in text_of(result)
    assert result.structured_content is not None
    assert result.structured_content["plots"] == []
    data = result.structured_content
    assert data is not None
    assert data["circuit"]["unknowns"] == 3
    (point,) = data["analyses"]
    assert point["values"][1] == {"unknown": "out", "nature": "voltage", "value": 2.5}


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


def test_schematics_and_plots_come_back_as_images(script: Path) -> None:
    script.write_text(
        DIVIDER.replace(
            '{"op": OperatingPoint()}', '{"sweep": DcSweep("V1", [V, 2 * V])}'
        ).replace("OperatingPoint, Plan", "DcSweep, Plan"),
        encoding="utf-8",
    )
    simulated = call("simulate_circuit", {"script": str(script)})
    assert not simulated.is_error, text_of(simulated)
    schematic, plot = simulated.content[1:]
    for image in (schematic, plot):
        assert isinstance(image, ImageContent)
        assert image.mime_type == "image/png"
    results = script.with_name("divider.results.npz")
    plotted = call("plot_results", {"results": str(results), "traces": ["out"]})
    assert "plotted sweep: " in text_of(plotted)
    assert isinstance(plotted.content[1], ImageContent)


def test_simulate_circuit_writes_raw_file(script: Path, tmp_path: Path) -> None:
    raw_binary = tmp_path / "divider.raw"
    result = call(
        "simulate_circuit",
        {"script": str(script), "raw": str(raw_binary), "schematic": False},
    )
    assert not result.is_error, text_of(result)
    assert f"saved {raw_binary}" in text_of(result)
    assert result.structured_content is not None
    assert result.structured_content["raw"] == str(raw_binary)
    loaded = load_raw(raw_binary)
    # Exact 1:1 resistive divider (2.5 V); rel=1e-15 (~4.5 eps) covers LU rounding.
    assert loaded["op"].voltage("out") == pytest.approx(2.5, rel=1e-15, abs=0)

    raw_ascii = tmp_path / "divider_ascii.raw"
    ascii_result = call(
        "simulate_circuit",
        {
            "script": str(script),
            "raw": str(raw_ascii),
            "raw_format": "ascii",
            "schematic": False,
        },
    )
    assert not ascii_result.is_error, text_of(ascii_result)
    assert b"Values:\n" in raw_ascii.read_bytes()
    # ASCII raw writes full IEEE-754 decimal precision; rel=1e-15 covers rounding.
    assert load_raw(raw_ascii)["op"].voltage("out") == pytest.approx(
        2.5, rel=1e-15, abs=0
    )


def _public_functions_returning(path: Path, return_annotations: set[str]) -> list[str]:
    """Parses `path` and returns public top-level functions matching a return hint."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            if node.returns is None:
                continue
            if ast.unparse(node.returns) in return_annotations:
                found.append(node.name)
    return found


def test_instructions_name_every_public_device_constructor() -> None:
    expected = _public_functions_returning(
        REPOSITORY_ROOT / "src" / "jolty" / "devices.py", {"Device"}
    )
    assert len(expected) >= 33
    missing = [name for name in expected if f"- {name}(" not in INSTRUCTIONS]
    assert not missing, f"missing device constructors in MCP instructions: {missing}"


def test_instructions_name_every_public_part() -> None:
    source = REPOSITORY_ROOT / "src" / "jolty" / "parts.py"
    constructors = _public_functions_returning(
        source, {"devices.Device", "Device", "Component"}
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    component_factories = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and not node.name.startswith("_")
        and any(
            ast.unparse(decorator) == "component" for decorator in node.decorator_list
        )
    ]
    datasheet_presets = [
        node.target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and not node.target.id.startswith("_")
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "Datasheet"
    ]
    assert len(constructors) >= 8
    assert component_factories
    assert len(datasheet_presets) >= 4
    missing_functions = [
        name
        for name in (*constructors, *component_factories)
        if f"- {name}(" not in INSTRUCTIONS
    ]
    assert not missing_functions, (
        f"missing parts in MCP instructions: {missing_functions}"
    )
    missing_presets = [name for name in datasheet_presets if name not in INSTRUCTIONS]
    assert not missing_presets, f"missing datasheet presets: {missing_presets}"
    assert "Datasheet" in INSTRUCTIONS


def test_instructions_name_every_public_analysis_and_plan_control() -> None:
    analyses = [
        analysis_class.__name__
        for analysis_class in typing.get_args(plan.Analysis.__value__)
    ]
    assert len(analyses) == 12
    variants = [
        variant_class.__name__
        for variant_class in typing.get_args(plan.Variants.__value__)
    ]
    waveforms_ = [
        waveform_class.__name__
        for waveform_class in typing.get_args(waveforms.Waveform.__value__)
    ]
    controls = (
        "Plan",
        "Options",
        "Parallelism",
        "Corner",
        "Variant",
        "Axis",
        "WorstCase",
        "Characterized",
        "decade",
        "linear",
    )
    missing = [
        name
        for name in (*analyses, *variants, *waveforms_, *controls)
        if f"- {name}(" not in INSTRUCTIONS
    ]
    assert not missing, f"missing plan or waveform symbols: {missing}"


def test_instructions_name_every_public_spec_measure() -> None:
    expected = _public_functions_returning(
        REPOSITORY_ROOT / "src" / "jolty" / "specs" / "_measures.py", {"Measure"}
    )
    assert len(expected) == 23
    assert set(expected).issubset(set(specs.__all__))
    missing = [name for name in expected if f"- {name}(" not in INSTRUCTIONS]
    assert not missing, f"missing spec measures in MCP instructions: {missing}"
    assert "specs()" in INSTRUCTIONS
    assert "Specs(requirements, yield_target=...)" in INSTRUCTIONS


def test_instructions_name_every_offline_resource_uri_and_stay_within_budget() -> None:
    guide_slugs = sorted(
        path.stem for path in (REPOSITORY_ROOT / "docs" / "guide").glob("*.md")
    )
    example_slugs = sorted(
        path.relative_to(REPOSITORY_ROOT / "examples").with_suffix("").as_posix()
        for path in (REPOSITORY_ROOT / "examples").rglob("*.py")
    )
    for slug in guide_slugs:
        assert f"jolty://docs/guide/{slug}" in INSTRUCTIONS
    for slug in example_slugs:
        assert f"jolty://docs/examples/{slug}" in INSTRUCTIONS
    assert "jolty://docs/guide/{slug}" in INSTRUCTIONS
    assert "jolty://docs/examples/{+slug}" in INSTRUCTIONS
    assert "\u2014" not in INSTRUCTIONS
    assert "\u2013" not in INSTRUCTIONS
    assert len(INSTRUCTIONS.split()) < 3000
    assert len(INSTRUCTIONS) < 25000


def test_live_server_initialize_handshake_exposes_derived_instructions() -> None:
    async def go() -> str | None:
        async with Client(create_server(Settings())) as client:
            return client.instructions

    received = asyncio.run(go())
    assert received == INSTRUCTIONS


def test_mcp_resources_list_and_read_guide_and_examples_offline() -> None:
    guide_directory = REPOSITORY_ROOT / "docs" / "guide"
    examples_directory = REPOSITORY_ROOT / "examples"
    expected_guide = sorted(path.stem for path in guide_directory.glob("*.md"))
    expected_examples = sorted(
        path.relative_to(examples_directory).with_suffix("").as_posix()
        for path in examples_directory.rglob("*.py")
    )

    async def inspect_resources() -> tuple[
        list[Resource], list[ResourceTemplate], str, str, str
    ]:
        async with Client(create_server(Settings())) as client:
            listed = (await client.list_resources()).resources
            templates = (await client.list_resource_templates()).resource_templates
            units_result = await client.read_resource("jolty://docs/guide/units")
            rc_result = await client.read_resource("jolty://docs/examples/rc_filter")
            cluster_result = await client.read_resource(
                "jolty://docs/examples/cluster/amplifier"
            )
            units_block = units_result.contents[0]
            rc_block = rc_result.contents[0]
            cluster_block = cluster_result.contents[0]
            assert isinstance(units_block, TextResourceContents)
            assert isinstance(rc_block, TextResourceContents)
            assert isinstance(cluster_block, TextResourceContents)
            assert units_block.mime_type == service.GUIDE_MIME_TYPE
            assert rc_block.mime_type == service.EXAMPLE_MIME_TYPE
            assert cluster_block.mime_type == service.EXAMPLE_MIME_TYPE
            return (
                listed,
                templates,
                units_block.text,
                rc_block.text,
                cluster_block.text,
            )

    listed, templates, units_text, rc_text, cluster_text = asyncio.run(
        inspect_resources()
    )
    expected_uris = [
        *(f"jolty://docs/guide/{slug}" for slug in expected_guide),
        *(f"jolty://docs/examples/{slug}" for slug in expected_examples),
    ]
    assert [str(resource.uri) for resource in listed] == expected_uris
    assert all(resource.title for resource in listed)
    assert all(resource.description for resource in listed)
    template_uris = {template.uri_template for template in templates}
    assert template_uris == {
        "jolty://docs/guide/{slug}",
        "jolty://docs/examples/{+slug}",
    }
    assert units_text == (guide_directory / "units.md").read_text(encoding="utf-8")
    assert rc_text == (examples_directory / "rc_filter.py").read_text(encoding="utf-8")
    assert cluster_text == (examples_directory / "cluster" / "amplifier.py").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("uri", "expected_fragment"),
    [
        ("jolty://docs/guide/nonexistent", "the guide topics are"),
        ("jolty://docs/examples/nonexistent", "the examples topics are"),
        ("jolty://docs/examples/cluster/nonexistent", "the examples topics are"),
        ("jolty://docs/examples/cluster/../etc/passwd", "invalid documentation topic"),
        ("jolty://docs/examples/../etc/passwd", "Unknown resource"),
        ("jolty://docs/examples/%2e%2e/etc/passwd", "Unknown resource"),
        ("jolty://docs/examples/cluster/amplifier/", "invalid documentation topic"),
        ("jolty://docs/guide/..", "Unknown resource"),
    ],
)
def test_mcp_resources_reject_missing_and_traversal_uris(
    uri: str, expected_fragment: str
) -> None:
    async def go() -> str:
        async with Client(create_server(Settings())) as client:
            with pytest.raises(MCPError) as raised:
                await client.read_resource(uri)
            return str(raised.value)

    message = asyncio.run(go())
    assert expected_fragment in message, message
    assert not ADVICE_WORDS.search(message), message


def test_read_docs_tool_lists_reads_and_rejects_bad_topics() -> None:
    listed = call("read_docs", {})
    assert not listed.is_error
    assert listed.structured_content is not None
    assert listed.structured_content["document"] is None
    assert len(listed.structured_content["entries"]) == 50
    assert "guide (17 pages, jolty://docs/guide/<slug>):" in text_of(listed)
    assert "examples (33 scripts, jolty://docs/examples/<slug>):" in text_of(listed)

    guide_only = call("read_docs", {"section": "guide"})
    assert not guide_only.is_error
    assert guide_only.structured_content is not None
    assert all(
        entry["section"] == "guide"
        for entry in guide_only.structured_content["entries"]
    )
    assert len(guide_only.structured_content["entries"]) == 17

    examples_only = call("read_docs", {"section": "examples"})
    assert not examples_only.is_error
    assert examples_only.structured_content is not None
    assert all(
        entry["section"] == "examples"
        for entry in examples_only.structured_content["entries"]
    )
    assert len(examples_only.structured_content["entries"]) == 33

    expected_units = (
        (REPOSITORY_ROOT / "docs" / "guide" / "units.md")
        .read_text(encoding="utf-8")
        .rstrip("\n")
    )
    for topic in ("units", "guide/units", "jolty://docs/guide/units"):
        read = call("read_docs", {"topic": topic})
        assert not read.is_error, topic
        assert text_of(read) == expected_units
        assert read.structured_content is not None
        assert read.structured_content["document"]["uri"] == "jolty://docs/guide/units"

    expected_cluster = (
        (REPOSITORY_ROOT / "examples" / "cluster" / "amplifier.py")
        .read_text(encoding="utf-8")
        .rstrip("\n")
    )
    for topic in (
        "cluster/amplifier",
        "examples/cluster/amplifier",
        "jolty://docs/examples/cluster/amplifier",
    ):
        read = call("read_docs", {"topic": topic})
        assert not read.is_error, topic
        assert text_of(read) == expected_cluster

    mismatch = call("read_docs", {"topic": "guide/units", "section": "examples"})
    assert mismatch.is_error
    assert "belongs to section 'guide', not 'examples'" in text_of(mismatch)
    assert not ADVICE_WORDS.search(text_of(mismatch))

    for bad_topic in (
        "",
        "   ",
        "../README.md",
        "cluster/../rc_filter",
        "/etc/passwd",
        "guide\\units",
        "file:///etc/passwd",
        "units\x00",
        "jolty://docs/",
        "guide/units/",
        "nonexistent_topic",
    ):
        failed = call("read_docs", {"topic": bad_topic})
        assert failed.is_error, bad_topic
        message = text_of(failed)
        assert "units" in message, message
        assert not ADVICE_WORDS.search(message), message


def test_offline_docs_fallback_to_packaged_bundle_without_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_directory = tmp_path / "assets"
    bundle_directory.mkdir()
    bundle_payload = {
        "guide": {
            "units": "# Units\r\n\r\nValues carry dimensions.\r\n",
        },
        "examples": {
            "rc_filter": '"""A low-pass filter."""\r\n\r\nVALUE = 1\r\n',
            "cluster/amplifier": '"""A sharded amp."""\r\n',
        },
    }
    (bundle_directory / "docs.json").write_text(
        json.dumps(bundle_payload), encoding="utf-8"
    )

    monkeypatch.setattr(service, "_repository_root", lambda: tmp_path / "missing")
    original_files = importlib.resources.files
    monkeypatch.setattr(
        importlib.resources,
        "files",
        lambda package: tmp_path if package == "jolty" else original_files(package),
    )

    summaries = service.document_summaries()
    assert [(entry.section, entry.slug) for entry in summaries] == [
        ("guide", "units"),
        ("examples", "cluster/amplifier"),
        ("examples", "rc_filter"),
    ]
    assert service.read_document("guide", "units").content == (
        "# Units\n\nValues carry dimensions.\n"
    )
    assert "jolty://docs/guide/units" in build_instructions()
    assert "jolty://docs/examples/cluster/amplifier" in build_instructions()

    read_via_mcp = call("read_docs", {"topic": "rc_filter"})
    assert not read_via_mcp.is_error
    assert text_of(read_via_mcp) == '"""A low-pass filter."""\n\nVALUE = 1'

    async def read_resource_from_bundle() -> str:
        async with Client(create_server(Settings())) as client:
            result = await client.read_resource(
                "jolty://docs/examples/cluster/amplifier"
            )
            block = result.contents[0]
            assert isinstance(block, TextResourceContents)
            return block.text

    assert asyncio.run(read_resource_from_bundle()) == '"""A sharded amp."""\n'

    # Simulate macOS sandbox-exec raising PermissionError on checkout inspection.
    class _DeniedPath:
        def __truediv__(self, _other: str) -> _DeniedPath:
            return self

        def is_dir(self) -> bool:
            raise PermissionError

    monkeypatch.setattr(service, "_repository_root", _DeniedPath)
    assert service.read_document("guide", "units").title == "Units"

    # Ambiguous bare slug collision across guide and examples reports both paths.
    colliding = {
        "guide": {"shared": "# Shared\n\nGuide version.\n"},
        "examples": {"shared": '"""Example version."""\n'},
    }
    (bundle_directory / "docs.json").write_text(json.dumps(colliding), encoding="utf-8")
    monkeypatch.setattr(service, "_repository_root", lambda: None)
    with pytest.raises(JoltyError, match="guide/shared, examples/shared") as ambiguous:
        service.resolve_document("shared")
    assert not ADVICE_WORDS.search(str(ambiguous.value))
    assert service.resolve_document("guide/shared").section == "guide"
    assert service.resolve_document("examples/shared").section == "examples"

    # Missing or corrupt bundle raises a factual JoltyError.
    (bundle_directory / "docs.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(JoltyError, match="not available"):
        service.document_summaries()
    (bundle_directory / "docs.json").unlink()
    with pytest.raises(JoltyError, match="not available") as unavailable:
        service.document_summaries()
    assert not ADVICE_WORDS.search(str(unavailable.value))


def test_measure_results_reads_saved_results_without_simulating(
    script: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulated = call("simulate_circuit", {"script": str(script), "schematic": False})
    assert not simulated.is_error, text_of(simulated)
    saved = script.with_name("divider.results.npz")

    log_path = tmp_path / "analyses.jsonl"
    monkeypatch.setenv("JOLTY_ANALYSIS_LOG", str(log_path))

    measured = call(
        "measure_results",
        {
            "results": str(saved),
            "analysis": "op",
            "measure": "value",
            "signal": "out",
        },
    )
    assert not measured.is_error, text_of(measured)
    assert text_of(measured) == "out in op: 2.5 V"
    assert measured.structured_content is not None
    assert measured.structured_content["value"] == 2.5
    assert measured.structured_content["unit"] == "V"
    assert measured.structured_content["formatted"] == "2.5 V"

    missing = call(
        "measure_results",
        {
            "results": str(saved),
            "analysis": "op",
            "measure": "value",
            "signal": "missing_node",
        },
    )
    assert missing.is_error
    error_text = text_of(missing)
    assert "no unknown named 'missing_node'" in error_text
    assert "out" in error_text
    assert not log_path.exists()


def test_instructions_use_current_result_method_names() -> None:
    assert "logic_level(net)" in INSTRUCTIONS
    assert "OperatingPointResult.region(instance)" in INSTRUCTIONS
    assert "OperatingPointResult.region(path)" not in INSTRUCTIONS
    assert re.search(r"(?<![A-Za-z0-9_])level\(net\)", INSTRUCTIONS) is None


def test_docs_catalog_is_cached_and_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = service._load_docs_from_checkout  # noqa: SLF001 - count disk scans
    calls = 0

    def counting(root: Path) -> dict[service.DocumentSection, dict[str, str]] | None:
        nonlocal calls
        calls += 1
        return original(root)

    service._cached_checkout.cache_clear()  # noqa: SLF001 - reset checkout cache
    monkeypatch.setattr(service, "_load_docs_from_checkout", counting)
    try:
        first = service._docs_catalog()  # noqa: SLF001 - inspect catalog caching
        second = service._docs_catalog()  # noqa: SLF001 - verify cache hit
        assert first is second
        assert calls == 1
        assert isinstance(first, MappingProxyType)
        assert isinstance(first["guide"], MappingProxyType)
        assert isinstance(first["examples"], MappingProxyType)
    finally:
        service._cached_checkout.cache_clear()  # noqa: SLF001 - restore clean cache


def test_simulate_skips_results_reload_when_unneeded(
    script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(_path: Path) -> object:
        message = "results reload should not run without raw, schematic, or plot"
        raise AssertionError(message)

    monkeypatch.setattr(service, "_load", forbidden)
    result = asyncio.run(
        service.simulate(
            service.SimulateRequest(script=script, plot=False, schematic=False),
            Settings(),
        )
    )
    assert result.results is not None
    assert result.results.is_file()

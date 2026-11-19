# Copyright 2026 Maxence Faldor
"""Programmatic MCP server instructions derived from code and offline docs.

Introspects jolty's waveforms, devices, real parts, analyses, plan controls, and spec
measures lazily on first call, combining them with the current offline
documentation URIs so server instructions stay synchronized with the codebase without
imposing import-time work or coupling `jolty.mcp` to lower layers.
"""

__lazy_modules__ = ["jolty"]

import functools
import inspect
import re
import typing
from collections.abc import Callable, Sequence
from typing import Final

from jolty import builder, devices, parts, plan, specs, waveforms
from jolty.service._docs import DOCS_URI_PREFIX, document_summaries

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
margins nearest a boundary first (OperatingPointResult.region(instance)), writes the
first operating point's voltages on the schematic, checks specs() verdicts (passed,
marginal, failed, unmeasured, with slack, worst corner and Wilson yield interval), plots
each analysis except operating points and sensitivities, and saves results to a .npz
file readable by jolty.files.load_results. plot_results draws other unknowns from saved
results; measure_results reads any measure from saved results without running a
simulation; draw_schematic draws the component inside an instance; view_circuit writes a
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
logic_level(net): 0, 1, or NaN for unknown. Plan fidelity (such as
fidelity={"adder/*": Characterized(3.3 * V)}) swaps matching components for logic cells
characterized from their circuits and verified by spot checks."""


def _clean_summary(docstring: str | None) -> str:
    """Extracts and cleans the first summary line of a docstring."""
    if docstring is None:
        return ""
    first_line = next(
        (stripped for line in docstring.splitlines() if (stripped := line.strip())), ""
    )
    if not first_line:
        return ""
    cleaned = _MKDOCS_CROSS_REFERENCE.sub(r"\1", first_line)
    cleaned = _MARKDOWN_LINK.sub(r"\1", cleaned).replace("``", "`")
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
    rendered: list[str] = [] if has_variable_positional else ["*"]
    rendered.extend(required)
    if len(optional) <= _MAX_INLINE_KEYWORD_PARAMETERS:
        rendered.extend(optional)
    else:
        rendered.append(f"... ({len(optional)} optional)")
    return rendered


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


def _entry(
    name: str, target: Callable[..., object], *, skip_first: bool = False
) -> str:
    """Formats one bullet entry with compact signature and cleaned summary."""
    signature = _compact_signature(target, skip_first=skip_first)
    return f"- {name}{signature}: {_clean_summary(target.__doc__)}"


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
    lines.extend(_entry(each.__name__, each) for each in classes)
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
        lines.append(_entry(name, candidate))
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
            lines.append(_entry(name, candidate))
        elif _is_component_factory(candidate):
            lines.append(_entry(name, candidate, skip_first=True))
    if datasheets:
        lines.append(f"- Datasheet presets (parts.Datasheet): {', '.join(datasheets)}")
    return "\n".join(lines)


def _format_analyses() -> str:
    """Formats every analysis class in `jolty.plan`."""
    analysis_classes = typing.get_args(plan.Analysis.__value__)
    lines = ["Analyses (jolty.plan):"]
    lines.extend(_entry(each.__name__, each) for each in analysis_classes)
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
    lines.extend(_entry(item.__name__, item) for item in items)
    return "\n".join(lines)


def _format_specs() -> str:
    """Formats `Specs`, `specs()`, and every public measure in `jolty.specs`."""
    specs_signature = _compact_signature(specs.Specs)
    header = (
        f"Specs and measures (jolty.specs): define specs() returning "
        f"Specs{specs_signature} to verify requirements such as "
        'value("op", "out") >= 2 * V across nominal, corners, Monte Carlo (Wilson '
        "interval via wilson) and worst case."
    )
    lines = [header]
    for name in specs.__all__:
        candidate = getattr(specs, name)
        if inspect.isfunction(candidate) and _return_hint(candidate) is specs.Measure:
            lines.append(_entry(name, candidate))
    return "\n".join(lines)


@functools.cache
def _static_instruction_sections() -> tuple[str, ...]:
    """Builds and caches code-derived instruction sections that ignore docs I/O."""
    return (
        _WORKFLOW_OVERVIEW,
        _format_waveforms(),
        _format_devices(),
        _format_parts(),
        _format_analyses(),
        _format_plan_controls(),
        _format_specs(),
    )


@functools.lru_cache(maxsize=8)
def _render_instructions(
    guide_uris: tuple[str, ...], example_uris: tuple[str, ...]
) -> str:
    """Assembles and caches full instructions for one set of offline doc URIs."""
    header = (
        "Offline documentation and examples (read via MCP resources, read_docs tool, "
        "or `jolty read-docs [TOPIC]`):"
    )
    templates = (
        f"- Templates: {DOCS_URI_PREFIX}/guide/{{slug}}, "
        f"{DOCS_URI_PREFIX}/examples/{{+slug}}"
    )
    resources = "\n".join(
        (
            header,
            templates,
            f"- Guide ({len(guide_uris)}): {', '.join(guide_uris)}",
            f"- Examples ({len(example_uris)}): {', '.join(example_uris)}",
        )
    )
    return "\n\n".join((*_static_instruction_sections(), resources)) + "\n"


def build_instructions() -> str:
    """Builds the MCP server instructions programmatically from code and offline docs.

    Raises:
        JoltyError: When the offline documentation catalog cannot be loaded.
    """
    entries = document_summaries()
    guide_uris = tuple(entry.uri for entry in entries if entry.section == "guide")
    example_uris = tuple(entry.uri for entry in entries if entry.section == "examples")
    return _render_instructions(guide_uris, example_uris)

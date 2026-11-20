# Copyright 2026 Maxence Faldor
"""Generator for the documentation gallery and discovery pages.

Run from the repository root via ``uv run python docs/gallery.py``, or pass specific
example or discovery names such as ``sallen_key`` to build a subset.
Each example in `ENTRIES` and each discovery design in `DISCOVERIES` runs through
`jolty.service.simulate`, generating schematics, analysis plots, structural check
reports, and spec verdicts.
For analog circuits without logic devices, `jolty.spice.export` writes a SPICE
netlist, ngspice runs every analysis in the plan, and the script evaluates identical
specs and waveform deviations across both engines.

Every entry records all device instances so the 3D viewer animates branch currents,
charges, and internal variables.
Generated outputs include ``docs/gallery.md``, per-example pages and 3D viewers under
``docs/gallery``, images under ``docs/images/gallery``, and ``docs/discovery.md``
plus per-discovery findings pages and viewers under ``docs/discovery``.
After writing all pages, the script fails if an example fails any spec in either
engine, or if the two engines disagree on a discovery spec verdict.
"""

import asyncio
import math
import runpy
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter
from measuring import markdown_table
from simulators import (
    dc_command,
    ngspice_element,
    ngspice_method_flags,
    ngspice_options_line,
    tran_command,
)

from jolty import (
    JoltyError,
    Plan,
    Specs,
    ir,
    plots,
    results,
    schematic,
    specs,
    viewer,
)
from jolty import circuit as circuit_of
from jolty.files import load_circuit, load_results
from jolty.plan import Ac, DcSweep, OperatingPoint, Transient
from jolty.service import SimulateRequest, simulate
from jolty.settings import Settings
from jolty.spice import export, ngspice
from jolty.units import celsius

ROOT: Final = Path(__file__).resolve().parent.parent
EXAMPLES: Final = ROOT / "examples"
DISCOVERY: Final = ROOT / "docs" / "discovery"
DOCS: Final = ROOT / "docs"
ASSETS: Final = DOCS / "assets"
IMAGES: Final = DOCS / "images" / "gallery"
REPOSITORY: Final = "https://github.com/maxencefaldor/jolty/blob/main"
SAME: Final = 1e-9
"""Frequency tolerance in decades below which two frequencies count as identical."""
PERCENT: Final = 1e-3
"""Threshold at or above which deviations format as percentages instead of exponents."""
RECORDED: Final = """\
# The plan and specs of {script}, its plan recording every instance.

import runpy
from dataclasses import replace

_ENTRY = runpy.run_path({script!r}, run_name="__gallery__")


def plan():
    found = _ENTRY["plan"]()
    return replace(found, record=["**"])


def specs():
    stated = _ENTRY.get("specs")
    return stated() if callable(stated) else None
"""
"""Template plan wrapper enabling full-circuit recording for 3D viewer playback."""


@dataclass(frozen=True)
class Entry:
    """Metadata identifying a gallery example or discovery entry.

    Free-running oscillators accumulate phase drift from tiny numerical differences
    between engines, so comparison relies on spec readings such as period and
    amplitude rather than pointwise waveform error.
    """

    name: str
    title: str
    section: str
    free_running: bool = False
    discovered: bool = False
    """Flag indicating a discovery design rather than a standard gallery example."""

    @property
    def script(self) -> Path:
        """Path to the Python script that constructs the circuit."""
        if self.discovered:
            return DISCOVERY / self.name / "design.py"
        return EXAMPLES / f"{self.name}.py"

    @property
    def folder(self) -> str:
        """Documentation subdirectory holding the generated page and 3D viewer."""
        return "discovery" if self.discovered else "gallery"

    @property
    def slug(self) -> str:
        """Hyphenated URL slug used for the generated page and 3D viewer filenames."""
        return self.name.replace("_", "-")

    @property
    def viewer(self) -> str:
        """Relative URL of the 3D viewer next to the generated page.

        Discovery viewers open focused inside the design instance placed as ``X1``.
        """
        page = f"{self.slug}.view.html"
        return f"{page}#view=X1" if self.discovered else page

    @property
    def source(self) -> str:
        """GitHub blob URL pointing to the circuit script in the repository."""
        return f"{REPOSITORY}/{self.script.relative_to(ROOT).as_posix()}"


SECTIONS: Final = (
    ("Filters", "Passive and active filters, their responses in closed form."),
    ("Amplifiers and references", "Transistors and op-amps, biased and amplifying."),
    ("Oscillators", "Circuits with no stable state, running at the rate they set."),
    ("Power", "Rectifiers, regulators and switching converters."),
    (
        "Digital and mixed-signal",
        "CMOS at transistor level, and logic at event speed beside analog circuits.",
    ),
)

ENTRIES: Final = (
    Entry("rc_filter", "RC low-pass filter", "Filters"),
    Entry("rlc_bandpass", "Series RLC band-pass", "Filters"),
    Entry("butterworth_ladder", "Fifth-order Butterworth ladder", "Filters"),
    Entry("twin_t_notch", "Twin-T notch", "Filters"),
    Entry("sallen_key", "Sallen-Key low-pass", "Filters"),
    Entry("inverting_amplifier", "Inverting amplifier", "Amplifiers and references"),
    Entry(
        "instrumentation_amplifier",
        "Instrumentation amplifier",
        "Amplifiers and references",
    ),
    Entry("common_emitter", "Common-emitter amplifier", "Amplifiers and references"),
    Entry("differential_pair", "Long-tailed pair", "Amplifiers and references"),
    Entry("widlar_source", "Widlar current source", "Amplifiers and references"),
    Entry("push_pull", "Class AB push-pull stage", "Amplifiers and references"),
    Entry("bandgap_reference", "Bandgap reference", "Amplifiers and references"),
    Entry("wien_bridge", "Wien bridge oscillator", "Oscillators", free_running=True),
    Entry(
        "relaxation_oscillator",
        "Op-amp relaxation oscillator",
        "Oscillators",
        free_running=True,
    ),
    Entry(
        "astable_multivibrator",
        "Astable multivibrator",
        "Oscillators",
        free_running=True,
    ),
    Entry(
        "colpitts_oscillator", "Colpitts oscillator", "Oscillators", free_running=True
    ),
    Entry("ring_oscillator", "CMOS ring oscillator", "Oscillators", free_running=True),
    Entry("timer_555", "555 timer, astable", "Oscillators"),
    Entry("rectifier", "Half-wave rectifier", "Power"),
    Entry("zener_regulator", "Zener shunt regulator", "Power"),
    Entry("voltage_multiplier", "Cockcroft-Walton multiplier", "Power"),
    Entry("buck_converter", "Buck converter", "Power"),
    Entry("boost_converter", "Boost converter", "Power"),
    Entry("cmos_inverter", "CMOS inverter", "Digital and mixed-signal"),
    Entry("sram_cell", "Six-transistor SRAM cell", "Digital and mixed-signal"),
    Entry("full_adder", "Full adder of NAND gates", "Digital and mixed-signal"),
    Entry("ripple_adder", "Four-bit ripple-carry adder", "Digital and mixed-signal"),
    Entry(
        "ripple_counter", "Ripple counter of D flip-flops", "Digital and mixed-signal"
    ),
    Entry("r2r_dac", "R-2R ladder DAC", "Digital and mixed-signal"),
    Entry("flash_adc", "Three-bit flash ADC", "Digital and mixed-signal"),
)

DISCOVERIES: Final = (
    Entry("current_source", "A two-terminal current source", "", discovered=True),
    Entry(
        "sine_oscillator",
        "A clean, steady sine oscillator",
        "",
        free_running=True,
        discovered=True,
    ),
    Entry("notch", "A notch that survives its tolerances", "", discovered=True),
    Entry("bandgap", "A reference without an op-amp", "", discovered=True),
    Entry(
        "vco",
        "A linear voltage-controlled oscillator",
        "",
        free_running=True,
        discovered=True,
    ),
    Entry("memristive_nor", "A NOR gate of three memristors", "", discovered=True),
)


# Replicate an example plan in ngspice.


@dataclass
class Replication:
    """ngspice replication results and comparison metrics for a circuit entry."""

    solved: dict[str, results.AnalysisResult] = field(default_factory=dict)
    corners: dict[str, dict[str, results.AnalysisResult]] = field(default_factory=dict)
    deviations: dict[str, dict[str, tuple[int, float]]] = field(default_factory=dict)
    """Per-analysis, per-quantity signal count and worst relative deviation."""
    verdicts: tuple[specs.Verdict, ...] = ()
    loosened: list[str] = field(default_factory=list)
    """Analyses where ngspice failed at jolty tolerances and retried with defaults."""
    unfinished: list[tuple[str | None, str]] = field(default_factory=list)
    """Corner and analysis pairs where ngspice timed out under both tolerance sets.

    A corner of `None` represents nominal plan conditions.
    """
    unreplicated: tuple[bool, ...] = ()
    """Per-verdict flags marking specs whose analysis did not complete in ngspice."""


def _commands(
    analysis: object, circuit: ir.Circuit
) -> tuple[list[str], str | None, str] | None:
    """Builds ngspice commands, excited source name, and option line for an analysis.

    AC sweeps with off-grid frequencies append single-point sweeps, and transient
    analyses emit integration method options.
    """
    match analysis:
        case OperatingPoint():
            return ["op"], None, ""
        case DcSweep(source=source):
            values = analysis.numbers()
            step = values[1] - values[0] if len(values) > 1 else 1.0
            element = ngspice_element(export.flatten(circuit), source)
            return [dc_command(element, values[0], values[-1], step)], None, ""
        case Ac(source=source, frequencies=frequencies):
            values = [frequency.value for frequency in frequencies]
            per_decade = round(1 / math.log10(values[1] / values[0]))
            decades = math.log10(values[-1] / values[0])
            steps = np.arange(0, math.floor(decades * per_decade + 1e-6) + 1)
            grid = values[0] * 10 ** (steps / per_decade)
            off = [
                value
                for value in values
                if np.min(np.abs(np.log10(grid / value))) > SAME
            ]
            commands = [f"ac dec {per_decade} {values[0]!r} {values[-1]!r}"]
            commands += [f"ac lin 1 {value!r} {value!r}" for value in off]
            return commands, source, ""
        case Transient(stop=stop, max_step=max_step, method=method):
            step = max_step.value if max_step is not None else stop.value / 1000
            option = ngspice_options_line(ngspice_method_flags(method))
            return [tran_command(step, stop.value)], None, option
    return None


def _run(
    lines: list[str], commands: list[str], option: str, kelvins: float
) -> tuple[ngspice.Result | None, bool]:
    """Runs ngspice `commands`, merges sweeps along the axis, and reports fallback.

    Failed runs or early-stopping transients retry once with default ngspice
    tolerances, setting the flag to `True` or returning `None` if retry also fails.
    """
    raws: list[ngspice.Result] = []
    loosened = False
    tight = f"{ngspice.OPTIONS}\n{option}" if option else ngspice.OPTIONS
    for command in commands:
        raw: ngspice.Result | None
        try:
            raw = ngspice.run(lines, command, options=tight, temperature=kelvins)
        except JoltyError:
            raw = None
        if raw is None or _stopped_early(raw, command):
            loosened = True
            try:
                raw = ngspice.run(
                    lines, command, options=option or "*", temperature=kelvins
                )
            except JoltyError:
                return None, loosened
        raws.append(raw)
    if len(raws) == 1:
        return raws[0], loosened
    axis = "frequency"
    merged = {
        name: np.concatenate([raw.vectors[name] for raw in raws])
        for name in raws[0].vectors
    }
    order = np.argsort(merged[axis].real)
    return (
        ngspice.Result(
            raws[0].plot,
            {name: values[order] for name, values in merged.items()},
            "".join(raw.printed for raw in raws),
        ),
        loosened,
    )


def _from_raw(
    raw: ngspice.Result, circuit: ir.Circuit, unknowns: tuple[results.Unknown, ...]
) -> tuple[np.ndarray, list[int]]:
    """Maps ngspice output vectors onto jolty unknowns and returns shared columns.

    Internal device unknowns using engine-specific naming are filled with NaN.
    """
    length = len(next(iter(raw.vectors.values())))
    table = np.full((length, len(unknowns)), np.nan, dtype=np.complex128)
    shared = []
    for index, unknown in enumerate(unknowns):
        try:
            name = export.vector(circuit, unknown)
        except JoltyError:
            continue
        if name in raw.vectors:
            table[:, index] = raw.vectors[name]
            shared.append(index)
    return table, shared


def _stopped_early(raw: ngspice.Result, command: str) -> bool:
    """Returns whether a transient stopped before the stop time in `command`."""
    if not command.startswith("tran"):
        return False
    stop = float(command.split()[2])
    return bool(raw.vectors["time"].real[-1] < stop * (1 - SAME))


def _replicate(
    plan: Plan,
    script_specs: Specs | None,
    circuits: Mapping[str | None, ir.Circuit],
    solved: Mapping[str, results.AnalysisResult],
    *,
    free_running: bool,
) -> Replication:
    """Runs `plan` in ngspice and compares outputs against jolty's `solved` results.

    The `circuits` mapping holds each corner's circuit (or `None` for nominal
    conditions); corners without a dedicated circuit reuse the nominal circuit.
    """
    replication = Replication()
    conditions = [(None, plan.temperature_at())]
    conditions += [
        (name, plan.temperature_at(corner)) for name, corner in plan.corners.items()
    ]
    for corner, temperature in conditions:
        kelvins = (temperature or celsius(27)).value
        circuit = circuits.get(corner, circuits[None])
        found: dict[str, results.AnalysisResult] = {}
        for name, analysis in plan.analyses.items():
            commands = _commands(analysis, circuit)
            jolty = solved.get(name)
            if (
                commands is None
                or jolty is None
                or isinstance(jolty, results.SensitivityResult)
            ):
                continue
            texts, excited, option = commands
            transient = isinstance(analysis, Transient)
            lines = export.netlist(circuit, excited=excited, transient=transient)
            raw, loosened = _run(lines, texts, option, kelvins)
            if raw is None:
                replication.unfinished.append((corner, name))
                continue
            if loosened and corner is None:
                replication.loosened.append(name)
            table, shared = _from_raw(raw, circuit, jolty.unknowns)
            found[name] = _result(jolty, raw, table)
            if corner is None and not (
                free_running and isinstance(jolty, results.TransientResult)
            ):
                replication.deviations[name] = {
                    nature: (
                        len(columns),
                        _deviation(jolty, found[name], columns),
                    )
                    for nature in ("voltage", "current")
                    if (
                        columns := [
                            column
                            for column in shared
                            if jolty.unknowns[column].nature == nature
                        ]
                    )
                }
        if corner is None:
            replication.solved = found
        else:
            replication.corners[corner] = found
    if script_specs is not None:
        replication.verdicts = specs.evaluate(
            script_specs,
            plan.analyses,
            replication.solved,
            replication.corners or None,
        )
        unfinished = {name for _, name in replication.unfinished}
        replication.unreplicated = tuple(
            requirement.measure.analysis in unfinished
            for requirement in script_specs.requirements
        )
    return replication


def _result(
    jolty: results.AnalysisResult, raw: ngspice.Result, table: np.ndarray
) -> results.AnalysisResult:
    """Converts raw ngspice output vectors into a jolty result on the same unknowns."""
    match jolty:
        case results.OperatingPointResult():
            return results.OperatingPointResult(
                jolty.unknowns, table[0].real, "ngspice"
            )
        case results.DcSweepResult():
            sweep = raw.vectors.get("v(v-sweep)")
            levels = jolty.swept if sweep is None else sweep.real
            levels = jolty.swept if len(levels) != len(table) else levels
            return results.DcSweepResult(
                jolty.unknowns, jolty.source, jolty.nature, levels, table.real
            )
        case results.AcResult():
            frequencies = raw.vectors["frequency"].real
            return results.AcResult(
                jolty.unknowns,
                jolty.source,
                frequencies,
                table,
                jolty.operating_point,
            )
        case results.TransientResult():
            times = raw.vectors["time"].real
            return results.TransientResult(
                jolty.unknowns, times, table.real, jolty.statistics
            )
    message = f"no ngspice counterpart to {type(jolty).__name__}"
    raise TypeError(message)


def _deviation(
    jolty: results.AnalysisResult, spice: results.AnalysisResult, shared: Sequence[int]
) -> float:
    """Computes the worst relative deviation between jolty and ngspice on `shared`.

    Operating points and sweeps compare pointwise, AC responses compare common
    frequencies, and transients compare within one step window around corners.
    """
    worst = 0.0
    for column in shared:
        match jolty, spice:
            case results.OperatingPointResult(), results.OperatingPointResult():
                ours, theirs = jolty.values[[column]], spice.values[[column]]
            case results.DcSweepResult(), results.DcSweepResult():
                ours, theirs = jolty.values[:, column], spice.values[:, column]
            case results.AcResult(), results.AcResult():
                common = np.isin(
                    np.round(np.log10(jolty.frequencies), 9),
                    np.round(np.log10(spice.frequencies), 9),
                )
                matched = np.isin(
                    np.round(np.log10(spice.frequencies), 9),
                    np.round(np.log10(jolty.frequencies), 9),
                )
                ours = jolty.values[common, column]
                theirs = spice.values[matched, column]
            case results.TransientResult(), results.TransientResult():
                window = float(np.max(np.diff(jolty.times)))
                theirs = spice.values[:, column]
                ours = _closest(
                    jolty.times, jolty.values[:, column], spice.times, theirs, window
                )
            case _:
                continue
        size = float(np.max(np.abs(ours))) if len(ours) else 0.0
        floor = 1e-9 if jolty.unknowns[column].nature == "voltage" else 1e-12
        scale = max(size, float(np.max(np.abs(theirs))) if len(theirs) else 0.0)
        if scale <= floor:
            continue
        worst = max(worst, float(np.max(np.abs(ours - theirs))) / scale)
    return worst


def _closest(
    times: np.ndarray,
    values: np.ndarray,
    at: np.ndarray,
    targets: np.ndarray,
    window: float,
) -> np.ndarray:
    """Finds waveform values nearest `targets` within `window` around each time."""
    starts, stops = at - window, at + window
    ends = np.stack([np.interp(starts, times, values), np.interp(stops, times, values)])
    lower, upper = ends.min(axis=0), ends.max(axis=0)
    firsts = np.searchsorted(times, starts, side="right")
    lasts = np.searchsorted(times, stops, side="left")
    for index, (first, last) in enumerate(zip(firsts, lasts, strict=True)):
        if first < last:
            inside = values[first:last]
            lower[index] = min(lower[index], inside.min())
            upper[index] = max(upper[index], inside.max())
    return np.clip(targets, lower, upper)


# Build gallery and discovery Markdown pages.


@dataclass
class Built:
    """Compiled build artifacts, verdicts, and replication metrics for one entry."""

    entry: Entry
    summary: str
    description: str
    images: list[str]
    verdicts: tuple[specs.Verdict, ...]
    replication: Replication | None
    devices: int
    unknowns: int
    replicated_plots: list[str]


def build(entry: Entry) -> Built:
    """Simulates an entry, replicates analog circuits in ngspice, and writes images.

    Discovery images use the discovery name instead of ``design``, and discovery
    schematics render the design subcircuit placed as ``X1``.
    """
    script = entry.script
    sys.path.insert(0, str(script.parent))
    namespace = runpy.run_path(str(script), run_name="__gallery__")
    with TemporaryDirectory() as directory:
        scratch = Path(directory)
        recorded = scratch / "recorded.py"
        recorded.write_text(RECORDED.format(script=str(script)), encoding="utf-8")
        request = SimulateRequest(
            script=script,
            plan=recorded,
            output=scratch / f"{entry.name}.jolty.json",
            results=scratch / f"{entry.name}.results.npz",
            schematic=not entry.discovered,
        )
        simulated = asyncio.run(simulate(request, Settings()))
        if simulated.results is None:
            message = f"{entry.name}: no analysis ran to its end"
            raise RuntimeError(message)
        circuit = load_circuit(simulated.circuit.path)
        solved = load_results(simulated.results, circuit=circuit)
        images = []
        for image in simulated.images():
            target = IMAGES / image.name.replace(script.stem, entry.name, 1)
            shutil.copyfile(image, target)
            images.append(target.name)
        if entry.discovered:
            inner = IMAGES / f"{entry.name}.inside.schematic.png"
            schematic.save(schematic.draw(circuit, "X1"), inner)
            images.insert(0, inner.name)
        viewer_path = DOCS / entry.folder / f"{entry.slug}.view.html"
        viewer.save_viewer(
            viewer_path,
            circuit,
            solved,
            title=entry.title,
            assets=ASSETS.relative_to(viewer_path.parent, walk_up=True).as_posix(),
        )
    plan = namespace["plan"]()
    stated = namespace.get("specs")
    script_specs = stated() if callable(stated) else None
    logic = any(
        isinstance(instance.of, ir.DeviceOf)
        and instance.of.model in {"logic_cell", "logic_source"}
        for component in circuit.components.values()
        for instance in component.instances
    ) or bool(plan.fidelity)
    replication = None
    replicated_plots: list[str] = []
    if not logic:
        # Corners passing arguments to `build()` instantiate a dedicated circuit.
        circuits: dict[str | None, ir.Circuit] = {None: circuit}
        for name, corner in plan.corners.items():
            if corner.arguments:
                circuits[name] = circuit_of(namespace["build"](**corner.arguments))
        replication = _replicate(
            plan, script_specs, circuits, solved, free_running=entry.free_running
        )
        for name, found in replication.solved.items():
            ours = solved[name]
            if isinstance(
                found,
                (results.DcSweepResult, results.AcResult, results.TransientResult),
            ):
                target = IMAGES / f"{entry.name}.{name}.replication.png"
                _replication_plot(name, ours, found).savefig(target)
                replicated_plots.append(target.name)
    doc = (namespace.get("__doc__") or "").strip()
    summary, _, description = doc.partition("\n\n")
    return Built(
        entry=entry,
        summary=" ".join(summary.split()),
        description=description,
        images=images,
        verdicts=simulated.specs,
        replication=replication,
        devices=sum(
            1
            for component in circuit.components.values()
            for instance in component.instances
            if isinstance(instance.of, ir.DeviceOf)
        ),
        unknowns=len(next(iter(solved.values())).unknowns),
        replicated_plots=replicated_plots,
    )


def _replication_plot(
    name: str, ours: results.AnalysisResult, theirs: results.AnalysisResult
) -> Figure:
    """Renders a figure overlaying solid jolty traces and dashed ngspice traces."""
    traces = plots.default_traces(ours.unknowns)[:4]
    figure = Figure(
        figsize=(8, 3.4), dpi=150, layout="constrained", facecolor="#fcfcfb"
    )
    axes = figure.subplots()
    figure.suptitle(
        f"{name}: jolty, solid, and ngspice, dashed", color="#0b0b0b", x=0.02, ha="left"
    )
    for index, trace in enumerate(traces):
        color = plots.COLORS[index]
        column = results.column(ours.unknowns, trace)
        x_ours, y_ours = _trace(ours, column)
        x_theirs, y_theirs = _trace(theirs, column)
        axes.plot(x_ours, y_ours, color=color, linewidth=1.6, label=trace)
        axes.plot(x_theirs, y_theirs, color="#0b0b0b", linewidth=0.9, linestyle="--")
    if isinstance(ours, results.AcResult):
        axes.set_xscale("log")
        axes.set_ylabel("magnitude (dB)", color="#52514e")
        axes.xaxis.set_major_formatter(EngFormatter(unit="Hz", sep=" "))
    elif isinstance(ours, results.DcSweepResult):
        axes.set_ylabel("voltage (V)", color="#52514e")
        unit = "A" if ours.nature == "current" else "V"
        axes.xaxis.set_major_formatter(EngFormatter(unit=unit, sep=" "))
    else:
        axes.set_ylabel("voltage (V)", color="#52514e")
        axes.xaxis.set_major_formatter(EngFormatter(unit="s", sep=" "))
    axes.grid(color="#e4e3df", linewidth=0.6)
    axes.set_facecolor("#fcfcfb")
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    axes.legend(frameon=False, fontsize=8, loc="best")
    return figure


def _trace(
    result: results.AnalysisResult, column: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts the x-axis array and selected column, converting AC values to dB."""
    match result:
        case results.AcResult():
            magnitude = np.abs(result.values[:, column])
            return result.frequencies, 20 * np.log10(np.maximum(magnitude, 1e-300))
        case results.DcSweepResult():
            return result.swept, result.values[:, column].real
        case results.TransientResult():
            return result.times, result.values[:, column].real
    return np.empty(0), np.empty(0)


def _si(value: float | None, unit: str | None) -> str:
    """Formats `value` with an SI prefix to six significant digits."""
    if value is None:
        return "-"
    if unit in {"dB", "%", "°", None, ""}:
        return f"{value:.6g} {unit or ''}".strip()
    return EngFormatter(unit=unit, sep=" ", places=None).format_data(value)


def _ratio(value: float) -> str:
    """Formats a relative deviation in scientific notation or as a percentage."""
    if value == 0:
        return "0"
    if value < PERCENT:
        return f"{value:.1e}"
    return f"{100 * value:.2g} %"


def _specs_section(built: Built) -> list[str]:
    """Builds the Specs Markdown section comparing jolty and ngspice verdicts."""
    if not built.verdicts:
        return ["## Specs", "", "The example states no specs.", ""]
    replicated = built.replication is not None
    theirs = {
        verdict.requirement: verdict
        for verdict in (built.replication.verdicts if built.replication else ())
    }
    headers = (
        ["Requirement", "jolty", "ngspice", "Difference"]
        if replicated
        else ["Requirement", "jolty"]
    )
    rows = []
    for verdict in built.verdicts:
        row = [verdict.requirement, _reading(verdict)]
        if replicated:
            spice = theirs.get(verdict.requirement)
            row.extend([_reading(spice), _difference(verdict, spice)])
        rows.append(row)
    return ["## Specs", "", *markdown_table(headers, rows).splitlines(), ""]


def _reading(verdict: specs.Verdict | None) -> str:
    """Formats a verdict value, optional corner label, and pass or fail status."""
    if verdict is None:
        return "-"
    at = "" if verdict.corner is None else f" at {verdict.corner}"
    return f"{_si(verdict.value, verdict.unit)}{at}, {verdict.status}"


def _difference(ours: specs.Verdict, theirs: specs.Verdict | None) -> str:
    """Formats the relative distance between jolty and ngspice verdict readings.

    When either reading is non-finite, the string reports each non-finite value
    explicitly instead of a relative ratio.
    """
    if theirs is None or theirs.value is None or ours.value is None:
        return "-"
    unbounded = [
        f"{engine} reads {_unbounded(value, unit)}"
        for engine, value, unit in (
            ("jolty", ours.value, ours.unit),
            ("ngspice", theirs.value, theirs.unit),
        )
        if not math.isfinite(value)
    ]
    if unbounded:
        return ", ".join(unbounded)
    scale = max(abs(ours.value), abs(theirs.value)) or 1.0
    return _ratio(abs(ours.value - theirs.value) / scale)


def _unbounded(value: float, unit: str | None) -> str:
    """Formats a non-finite reading in words, rendering -inf dB as an exact zero."""
    if unit == "dB" and value == -math.inf:
        return "an exact zero"
    return f"{value} {unit or ''}".strip()


def _replication_section(built: Built) -> list[str]:
    """Builds the ngspice replication or closed-form verification Markdown section."""
    replication = built.replication
    if replication is None:
        return [
            "## Checked in closed form",
            "",
            (
                "The example holds logic devices, which SPICE has no element for, so "
                "its specs check it against its closed-form behavior: truth tables, "
                "codes and periods."
            ),
            "",
        ]
    lines: list[str] = [
        "## Replicated in ngspice",
        "",
        (
            "jolty writes the circuit as a SPICE netlist, ngspice runs every analysis "
            "of the plan, and each signal both report is compared, relative to its "
            "largest value. A transient compares each of ngspice's points with jolty's "
            "waveform within one of jolty's steps."
        ),
        "",
    ]
    if replication.deviations:
        rows = []
        for name, natures in replication.deviations.items():
            cells = []
            for nature in ("voltage", "current"):
                count, deviation = natures.get(nature, (0, 0.0))
                cells += [str(count), _ratio(deviation) if count else "-"]
            rows.append([f"`{name}`", *cells])
        lines.extend(
            [
                *markdown_table(
                    [
                        "Analysis",
                        "Voltages",
                        "Largest deviation",
                        "Currents",
                        "Largest deviation",
                    ],
                    rows,
                ).splitlines(),
                "",
            ]
        )
    if built.entry.free_running:
        lines += [
            (
                "A free-running oscillator's phase drifts with the least difference in "
                "two engines' numerics, so its waveforms compare through the specs' "
                "readings, its period and its swing, rather than point by point."
            ),
            "",
        ]
    if replication.loosened:
        names = ", ".join(f"`{name}`" for name in replication.loosened)
        lines += [
            (
                f"ngspice could not finish {names} with tolerances as tight as "
                "jolty's, so it ran with its own defaults."
            ),
            "",
        ]
    if replication.unfinished:
        names = ", ".join(
            f"`{name}`" if corner is None else f"`{name}` at the corner `{corner}`"
            for corner, name in replication.unfinished
        )
        lines += [
            (
                f"ngspice finished {names} neither with tolerances as tight as "
                f"jolty's nor with its own defaults, within {ngspice.TIMEOUT} s, so "
                "it judges no spec on them."
            ),
            "",
        ]
    for image in built.replicated_plots:
        lines += [f"![The replication in ngspice](../images/gallery/{image})", ""]
    return lines


def page(built: Built) -> str:
    """Builds the Markdown page for a gallery example or discovery entry."""
    entry = built.entry
    buttons = (
        f"[Open in 3D]({entry.viewer}){{ .md-button .md-button--primary }} "
        f"[Script]({entry.source}){{ .md-button }}"
    )
    if entry.discovered:
        folder = f"{REPOSITORY.replace('/blob/', '/tree/')}/docs/discovery/{entry.name}"
        buttons += f" [Every experiment]({folder}){{ .md-button }}"
    lines = [
        f"# {entry.title}",
        "",
        built.summary,
        "",
        *built.description.replace("``", "`").splitlines(),
        "",
        buttons,
        "",
        f"{built.devices} devices and {built.unknowns} unknowns.",
        "",
    ]
    drawn = [image for image in built.images if image.endswith("schematic.png")]
    if drawn:
        lines += [f"![The schematic](../images/gallery/{drawn[0]})", ""]
    if entry.discovered:
        lines += _discovery_sections(entry)
    lines += _specs_section(built)
    lines += _replication_section(built)
    plotted = [image for image in built.images if image.endswith(".plot.png")]
    if plotted:
        lines += ["## Plots", ""]
        lines += [f"![A plot](../images/gallery/{image})" for image in plotted]
        lines.append("")
    return "\n".join(lines)


def _discovery_sections(entry: Entry) -> list[str]:
    """Builds the quoted prompt and findings sections for a discovery entry."""
    folder = DISCOVERY / entry.name
    prompt = (folder / "prompt.md").read_text(encoding="utf-8").splitlines()[1:]
    findings = (folder / "findings.md").read_text(encoding="utf-8").splitlines()[1:]
    quoted = [f"> {line}".rstrip() for line in _trimmed(prompt)]
    return ["## The prompt", "", *quoted, "", *_trimmed(findings), ""]


def _trimmed(lines: Sequence[str]) -> list[str]:
    """Returns `lines` with leading and trailing blank lines removed."""
    kept = list(lines)
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return kept


def _worst_voltage(replication: Replication) -> float | None:
    """Returns the largest relative voltage deviation across analyses, or `None`."""
    found = [
        natures["voltage"][1]
        for natures in replication.deviations.values()
        if "voltage" in natures
    ]
    return max(found) if found else None


def _card(built: Built) -> list[str]:
    """Builds a Markdown grid card containing schematic, title, summary, and links."""
    entry = built.entry
    drawn = next(
        (image for image in built.images if image.endswith("schematic.png")), None
    )
    passed = sum(verdict.status == "passed" for verdict in built.verdicts)
    replicated = ""
    if built.replication is not None:
        worst = _worst_voltage(built.replication)
        replicated = (
            f" · ngspice within {_ratio(worst)}"
            if worst is not None
            else " · ngspice agrees on its specs"
        )
    image = (
        f"-   ![{entry.title}, its schematic](images/gallery/{drawn})"
        if drawn is not None
        else "-   "
    )
    summary_lines = [f"    {built.summary}"]
    if entry.discovered:
        findings = (DISCOVERY / entry.name / "findings.md").read_text(encoding="utf-8")
        intro = _trimmed(findings.split("\n## ")[0].splitlines()[1:])
        summary_lines = [f"    {line}" for line in intro]
    return [
        image,
        "",
        (
            f"    **[{entry.title}]({entry.folder}/{entry.slug}.md)** · "
            f"{passed}/{len(built.verdicts)} specs{replicated}"
        ),
        "",
        *summary_lines,
        "",
        (f"    [Open in 3D]({entry.folder}/{entry.viewer}) · [Script]({entry.source})"),
        "",
    ]


def index(built: Sequence[Built]) -> str:
    """Builds the top-level gallery index page grouped by circuit section."""
    lines = [
        "# Gallery",
        "",
        (
            f"{len(built)} circuits, each built and simulated by one command from "
            "[`examples/`](https://github.com/maxencefaldor/jolty/tree/main/examples), "
            "its schematic and plots drawn by jolty."
        ),
        (
            "Each example states its specs, from closed-form behavior: a cutoff, a "
            "period, a code, a truth table."
        ),
        (
            "Every example without logic devices runs again in ngspice from the "
            "netlist jolty writes, and its page compares the two engines signal by "
            "signal."
        ),
        (
            "Each opens in a 3D viewer, to orbit the circuit and scrub through its "
            "simulation, every device's state at every point."
        ),
        "",
    ]
    for section, blurb in SECTIONS:
        members = [each for each in built if each.entry.section == section]
        if not members:
            continue
        lines += [f"## {section}", "", blurb, "", '<div class="grid cards" markdown>']
        lines.append("")
        for each in members:
            lines += _card(each)
        lines += ["</div>", ""]
    return "\n".join(lines)


def discoveries(built: Sequence[Built]) -> str:
    """Builds the top-level discovery index page linking each problem."""
    lines = [
        "# Discovery",
        "",
        (
            "Classic analog and mixed-signal design problems become difficult under "
            "strict part budgets: supply- and temperature-insensitive current sources, "
            "low-distortion sine oscillators, and tolerance-immune notch filters."
        ),
        (
            "Three additional problems cover op-amp-free voltage references, linear "
            "voltage-controlled oscillators, and three-memristor NOR gates with wide "
            "pulse-amplitude margins."
        ),
        (
            "For each problem, an engineering prompt states quantitative bounds, and "
            "Claude searched for a circuit using only jolty."
        ),
        (
            "Claude wrote candidate components, ran Python API and CLI experiments, "
            "inspected reported currents, delays, and drifts, and kept designs backed "
            "by measurements."
        ),
        (
            "Each page presents the prompt, experimental results, and findings, "
            "running the final design through `jolty simulate` and replicating every "
            "analog analysis in ngspice."
        ),
        "Missed bounds and limiting physical mechanisms are documented explicitly.",
        (
            "None of the runs was blind: the agent wrote each prompt knowing textbook "
            "topologies, demonstrating what jolty reveals during design rather than "
            "autonomous invention."
        ),
        (
            "Each page ends with hardware caveats, and all prompts, candidates, "
            "experiments, and designs live in "
            "[`docs/discovery/`](https://github.com/maxencefaldor/jolty/tree/main/docs/discovery)."
        ),
        "",
        '<div class="grid cards" markdown>',
        "",
    ]
    for each in built:
        lines += _card(each)
    lines += ["</div>", ""]
    return "\n".join(lines)


def _failures(built: Built) -> int:
    """Counts spec failures that should fail the build script.

    Examples fail when either engine fails a spec; discoveries fail only when the
    two engines disagree on a verdict, since missed bounds appear in findings.
    """
    ours = built.verdicts
    replication = built.replication
    theirs = replication.verdicts if replication else ()
    # Exclude specs whose underlying analysis did not complete in ngspice.
    skipped = {
        index
        for index, unreplicated in enumerate(
            replication.unreplicated if replication else ()
        )
        if unreplicated
    }
    if built.entry.discovered:
        return sum(
            mine.status != spice.status
            for index, (mine, spice) in enumerate(zip(ours, theirs, strict=True))
            if index not in skipped
        )
    judged = [spice for index, spice in enumerate(theirs) if index not in skipped]
    return sum(verdict.status != "passed" for verdict in (*ours, *judged))


def _copy_viewer_assets() -> None:
    """Copies the shared viewer stylesheet and script into ``docs/assets``."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    bundled = files("jolty").joinpath("assets")
    for name in ("viewer.css", "viewer.js"):
        (ASSETS / name).write_bytes(bundled.joinpath(name).read_bytes())


def main(names: Sequence[str]) -> int:
    """Builds the requested entries (or all entries) and returns the failure count."""
    chosen = [
        entry for entry in (*ENTRIES, *DISCOVERIES) if not names or entry.name in names
    ]
    for folder in {entry.folder for entry in chosen}:
        (DOCS / folder).mkdir(exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    _copy_viewer_assets()
    # Run each entry in a separate process because discovery scripts import local
    # modules such as `candidates` by unqualified name.
    with ProcessPoolExecutor(max_workers=4, max_tasks_per_child=1) as pool:
        built = list(pool.map(build, chosen))
    failures = 0
    written = []
    for each in built:
        target = DOCS / each.entry.folder / f"{each.entry.slug}.md"
        target.write_text(page(each), encoding="utf-8")
        written.append(target)
        failed = _failures(each)
        failures += failed
        status = "ok" if not failed else f"{failed} not passed"
        worst = _worst_voltage(each.replication) if each.replication else None
        print(f"{each.entry.name}: {status}, ngspice {worst}")
    if not names:
        made = {each.entry: each for each in built}
        (DOCS / "gallery.md").write_text(
            index([made[entry] for entry in ENTRIES]), encoding="utf-8"
        )
        (DOCS / "discovery.md").write_text(
            discoveries([made[entry] for entry in DISCOVERIES]), encoding="utf-8"
        )
        written += [DOCS / "gallery.md", DOCS / "discovery.md"]
    # Enforce one sentence per line across generated pages because embedded
    # docstrings wrap at 88 columns.
    rumdl = Path(sys.executable).parent / "rumdl"
    subprocess.run([rumdl, "fmt", "--quiet", *written], check=True)  # noqa: S603 - invokes the environment rumdl binary on generated pages
    return failures


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:]) else 0)

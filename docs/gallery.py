# Copyright 2026 Maxence Faldor
"""Build the docs' gallery and discoveries: each simulated by jolty, then by ngspice.

Run from the repository root: ``uv run python docs/gallery.py``, or name examples or
discoveries, such as ``sallen_key``, to build only those. Each example in `ENTRIES`,
and each discovery's design in `DISCOVERIES`, runs through
`jolty.service.simulate`, the function behind ``jolty simulate``, which draws its
schematic and plots its analyses, checks its structure and judges its specs. Then, for
an example without logic devices, `jolty.spice.export` writes its circuit as a SPICE
netlist, ngspice runs every analysis of its plan, and the gallery reads ngspice's
results as jolty's own: it judges the same specs on them, and measures how far each
signal both engines report lies from the other's.

Each runs its plan recording every instance, so its 3D viewer plays every device's
currents, charges and variables. The script writes ``docs/gallery.md``, a page per
example under ``docs/gallery``, the images under ``docs/images/gallery``, and each
example's 3D viewer as
``docs/gallery/<name>.view.html``; and ``docs/discovery.md``, with a page holding its
findings and a viewer per discovery under ``docs/discovery``. Once every page is
written, the script fails on a spec an example fails in either engine, and on a spec
on whose verdict the engines disagree for a discovery, whose findings report the
bounds it misses.
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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter

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
DISCOVERY: Final = ROOT / "discovery"
DOCS: Final = ROOT / "docs"
IMAGES: Final = DOCS / "images" / "gallery"
REPOSITORY: Final = "https://github.com/maxencefaldor/jolty/blob/main"
SAME: Final = 1e-9
"""Two frequencies within this many decades of each other are the same."""
PERCENT: Final = 1e-3
"""Deviations from this one up read as percentages, smaller ones as powers of ten."""
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
"""A plan script that runs an entry's plan recording every instance, for its viewer."""


@dataclass(frozen=True)
class Entry:
    """An example the gallery shows: its script's name, its title and its section.

    A free-running oscillator's phase drifts with any difference in two engines'
    numerics, so its waveforms compare through its specs' readings, its period and
    its amplitude, rather than point by point.
    """

    name: str
    title: str
    section: str
    free_running: bool = False
    discovered: bool = False
    """Whether the entry is a discovery's design rather than an example."""

    @property
    def script(self) -> Path:
        """The script that builds the entry's circuit."""
        if self.discovered:
            return DISCOVERY / self.name / "design.py"
        return EXAMPLES / f"{self.name}.py"

    @property
    def folder(self) -> str:
        """The folder of the docs that holds the entry's page and viewer."""
        return "discovery" if self.discovered else "gallery"

    @property
    def slug(self) -> str:
        """The name of the entry's page and viewer."""
        return self.name.replace("_", "-")

    @property
    def viewer(self) -> str:
        """The address of the entry's viewer, beside its page.

        A discovery's opens inside its design, the instance its bench places as ``X1``.
        """
        page = f"{self.slug}.view.html"
        return f"{page}#view=X1" if self.discovered else page

    @property
    def source(self) -> str:
        """The address of the entry's script in the repository."""
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


# Running ngspice on an example's plan.


@dataclass
class Replication:
    """What ngspice found of an example, and how far it lies from jolty's results."""

    solved: dict[str, results.Result] = field(default_factory=dict)
    corners: dict[str, dict[str, results.Result]] = field(default_factory=dict)
    deviations: dict[str, dict[str, tuple[int, float]]] = field(default_factory=dict)
    """Per analysis, per quantity, the signals compared and the largest deviation,
    relative to each signal's largest value."""
    verdicts: tuple[specs.Verdict, ...] = ()
    loosened: list[str] = field(default_factory=list)
    """The analyses ngspice could not finish with tolerances as tight as jolty's, run
    again with its own defaults."""


def _commands(
    analysis: object, circuit: ir.Circuit
) -> tuple[list[str], str | None, str] | None:
    """The ngspice commands of a jolty analysis, its excited source, and its method.

    A frequency response off a decade's grid runs once more for each frequency off it,
    and a transient's integration method is an option line.
    """
    match analysis:
        case OperatingPoint():
            return ["op"], None, ""
        case DcSweep(source=source, levels=levels):
            values = [level.value for level in levels]
            step = values[1] - values[0] if len(values) > 1 else 1.0
            element = _element(circuit, source)
            return [f"dc {element} {values[0]!r} {values[-1]!r} {step!r}"], None, ""
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
            option = {
                "trapezoidal": "",
                "gear2": ".options method=gear maxord=2",
                "backward_euler": ".options method=gear maxord=1",
            }[method]
            return [f"tran {step!r} {stop.value!r} 0 {step!r}"], None, option
    return None


def _run(
    lines: list[str], commands: list[str], option: str, kelvins: float
) -> tuple[ngspice.Raw, bool]:
    """What ngspice finds of `commands`, merged along their axis, and if it loosened.

    A transient that stops before its end, or a run that fails, runs again with
    ngspice's default tolerances, and loosened says so.
    """
    raws: list[ngspice.Raw] = []
    loosened = False
    tight = f"{ngspice.OPTIONS}\n{option}" if option else ngspice.OPTIONS
    for command in commands:
        raw: ngspice.Raw | None
        try:
            raw = ngspice.run(lines, command, options=tight, temperature=kelvins)
        except JoltyError:
            raw = None
        if raw is None or _stopped_early(raw, command):
            loosened = True
            raw = ngspice.run(
                lines, command, options=option or "*", temperature=kelvins
            )
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
        ngspice.Raw(
            raws[0].plot,
            {name: values[order] for name, values in merged.items()},
            "".join(raw.printed for raw in raws),
        ),
        loosened,
    )


def _element(circuit: ir.Circuit, path: str) -> str:
    """The SPICE element name of the device at `path`."""
    flat = export.flatten(circuit)
    instance = flat.components[flat.top].instance(path.replace("/", "_"))
    if instance is None:
        message = f"no device at {path}"
        raise KeyError(message)
    return export.element(instance)


def _from_raw(
    raw: ngspice.Raw, circuit: ir.Circuit, unknowns: tuple[results.Unknown, ...]
) -> tuple[np.ndarray, list[int]]:
    """Each jolty unknown's ngspice vector, and the columns both engines report.

    An unknown ngspice names its own way, a device's internal one, is NaN.
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


def _stopped_early(raw: ngspice.Raw, command: str) -> bool:
    """Whether a transient stopped before the end its `command` asks for."""
    if not command.startswith("tran"):
        return False
    stop = float(command.split()[2])
    return bool(raw.vectors["time"].real[-1] < stop * (1 - SAME))


def _replicate(
    plan: Plan,
    script_specs: Specs | None,
    circuits: Mapping[str | None, ir.Circuit],
    solved: Mapping[str, results.Result],
    *,
    free_running: bool,
) -> Replication:
    """Runs `plan` in ngspice, and compares what it finds with jolty's `solved`.

    `circuits` holds the circuit of each condition, by its corner's name or `None`; a
    corner that builds no circuit of its own runs the plan's.
    """
    replication = Replication()
    conditions = [(None, plan.temperature_at())]
    conditions += [
        (name, plan.temperature_at(corner)) for name, corner in plan.corners.items()
    ]
    for corner, temperature in conditions:
        kelvins = (temperature or celsius(27)).value
        circuit = circuits.get(corner, circuits[None])
        found: dict[str, results.Result] = {}
        for name, analysis in plan.analyses.items():
            commands = _commands(analysis, circuit)
            jolty = solved.get(name)
            if (
                commands is None
                or jolty is None
                or isinstance(jolty, results.Sensitivities)
            ):
                continue
            texts, excited, option = commands
            transient = isinstance(analysis, Transient)
            lines = export.netlist(circuit, excited=excited, transient=transient)
            raw, loosened = _run(lines, texts, option, kelvins)
            if loosened and corner is None:
                replication.loosened.append(name)
            table, shared = _from_raw(raw, circuit, jolty.unknowns)
            found[name] = _result(jolty, raw, table)
            if corner is None and not (
                free_running and isinstance(jolty, results.Waveforms)
            ):
                replication.deviations[name] = {
                    quantity: (
                        len(columns),
                        _deviation(jolty, found[name], columns),
                    )
                    for quantity in ("voltage", "current")
                    if (
                        columns := [
                            column
                            for column in shared
                            if jolty.unknowns[column].quantity == quantity
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
    return replication


def _result(
    jolty: results.Result, raw: ngspice.Raw, table: np.ndarray
) -> results.Result:
    """The results ngspice found, as jolty's, on the same unknowns."""
    match jolty:
        case results.OperatingPoint():
            return results.OperatingPoint(jolty.unknowns, table[0].real, "ngspice")
        case results.Sweep():
            sweep = raw.vectors.get("v(v-sweep)")
            levels = jolty.levels if sweep is None else sweep.real
            levels = jolty.levels if len(levels) != len(table) else levels
            return results.Sweep(
                jolty.unknowns, jolty.source, jolty.quantity, levels, table.real
            )
        case results.Response():
            frequencies = raw.vectors["frequency"].real
            return results.Response(
                jolty.unknowns,
                jolty.source,
                frequencies,
                table,
                jolty.operating_point,
            )
        case results.Waveforms():
            times = raw.vectors["time"].real
            return results.Waveforms(
                jolty.unknowns, times, table.real, jolty.statistics
            )
    message = f"no ngspice counterpart to {type(jolty).__name__}"
    raise TypeError(message)


def _deviation(
    jolty: results.Result, spice: results.Result, shared: Sequence[int]
) -> float:
    """The largest deviation between the engines, relative to each signal's size.

    Operating points and sweeps compare point for point; responses at the frequencies
    both computed; transients at ngspice's times, each value within a window of a
    step of jolty's waveform, as two engines stepping differently round corners.
    """
    worst = 0.0
    for column in shared:
        match jolty, spice:
            case results.OperatingPoint(), results.OperatingPoint():
                ours, theirs = jolty.values[[column]], spice.values[[column]]
            case results.Sweep(), results.Sweep():
                ours, theirs = jolty.values[:, column], spice.values[:, column]
            case results.Response(), results.Response():
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
            case results.Waveforms(), results.Waveforms():
                window = float(np.max(np.diff(jolty.times)))
                theirs = spice.values[:, column]
                ours = _closest(
                    jolty.times, jolty.values[:, column], spice.times, theirs, window
                )
            case _:
                continue
        size = float(np.max(np.abs(ours))) if len(ours) else 0.0
        floor = 1e-9 if jolty.unknowns[column].quantity == "voltage" else 1e-12
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
    """The values of a waveform nearest `targets`, each within `window` of its time."""
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


# The pages.


@dataclass
class Built:
    """What the gallery made of one example."""

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
    """Simulates an entry, replicates it in ngspice, and writes its images.

    A discovery's images take its name in place of its script's, ``design``, and its
    schematic is drawn inside the instance its bench places, ``X1``, too.
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
        )
        simulated = asyncio.run(simulate(request, Settings()))
        if simulated.results is None:
            message = f"{entry.name}: no analysis ran to its end"
            raise RuntimeError(message)
        circuit = load_circuit(simulated.circuit.path)
        solved = load_results(simulated.results)
        images = []
        for image in simulated.images():
            target = IMAGES / image.name.replace(script.stem, entry.name, 1)
            shutil.copyfile(image, target)
            images.append(target.name)
        if entry.discovered:
            inner = IMAGES / f"{entry.name}.inside.schematic.png"
            schematic.save(schematic.draw(circuit, "X1"), inner)
            images.insert(0, inner.name)
        viewer.write_viewer(
            DOCS / entry.folder / f"{entry.slug}.view.html",
            circuit,
            solved,
            title=entry.title,
        )
    plan = namespace["plan"]()
    stated = namespace.get("specs")
    script_specs = stated() if callable(stated) else None
    logic = any(
        isinstance(instance.of, ir.DeviceOf)
        and instance.of.device in {"logic_cell", "logic_source"}
        for component in circuit.components.values()
        for instance in component.instances
    ) or bool(plan.fidelity)
    replication = None
    replicated_plots: list[str] = []
    if not logic:
        # A corner that passes arguments to `build()` builds a circuit of its own.
        circuits: dict[str | None, ir.Circuit] = {None: circuit}
        for name, corner in plan.corners.items():
            if corner.arguments:
                circuits[name] = circuit_of(namespace["build"](**corner.arguments))
        replication = _replicate(
            plan, script_specs, circuits, solved, free_running=entry.free_running
        )
        for name, found in replication.solved.items():
            ours = solved[name]
            if isinstance(found, (results.Sweep, results.Response, results.Waveforms)):
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
    name: str, ours: results.Result, theirs: results.Result
) -> Figure:
    """The signals jolty found, solid, and ngspice's, dashed, on one plot."""
    probes = plots.default_probes(ours.unknowns)[:4]
    figure = Figure(
        figsize=(8, 3.4), dpi=150, layout="constrained", facecolor="#fcfcfb"
    )
    axes = figure.subplots()
    figure.suptitle(
        f"{name}: jolty, solid, and ngspice, dashed", color="#0b0b0b", x=0.02, ha="left"
    )
    for index, probe in enumerate(probes):
        color = plots.COLORS[index]
        column = results.column(ours.unknowns, probe)
        x_ours, y_ours = _trace(ours, column)
        x_theirs, y_theirs = _trace(theirs, column)
        axes.plot(x_ours, y_ours, color=color, linewidth=1.6, label=probe)
        axes.plot(x_theirs, y_theirs, color="#0b0b0b", linewidth=0.9, linestyle="--")
    if isinstance(ours, results.Response):
        axes.set_xscale("log")
        axes.set_ylabel("magnitude (dB)", color="#52514e")
        axes.xaxis.set_major_formatter(EngFormatter(unit="Hz", sep=" "))
    elif isinstance(ours, results.Sweep):
        axes.set_ylabel("voltage (V)", color="#52514e")
        unit = "A" if ours.quantity == "current" else "V"
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


def _trace(result: results.Result, column: int) -> tuple[np.ndarray, np.ndarray]:
    """The abscissa and one unknown of a result, magnitudes in decibels."""
    match result:
        case results.Response():
            magnitude = np.abs(result.values[:, column])
            return result.frequencies, 20 * np.log10(np.maximum(magnitude, 1e-300))
        case results.Sweep():
            return result.levels, result.values[:, column].real
        case results.Waveforms():
            return result.times, result.values[:, column].real
    return np.empty(0), np.empty(0)


def _si(value: float | None, unit: str | None) -> str:
    """A value with an SI prefix, as reports write it, to six digits."""
    if value is None:
        return "—"
    if unit in {"dB", "%", "°", None, ""}:
        return f"{value:.6g} {unit or ''}".strip()
    return EngFormatter(unit=unit, sep=" ", places=None).format_data(value)


def _ratio(value: float) -> str:
    """A relative deviation, as a power of ten or a percentage."""
    if value == 0:
        return "0"
    if value < PERCENT:
        return f"{value:.1e}"
    return f"{100 * value:.2g} %"


def _specs_section(built: Built) -> list[str]:
    """The specs' verdicts, jolty's beside ngspice's when ngspice ran the example."""
    if not built.verdicts:
        return ["## Specs", "", "The example states no specs.", ""]
    replicated = built.replication is not None
    theirs = {
        verdict.requirement: verdict
        for verdict in (built.replication.verdicts if built.replication else ())
    }
    lines = [
        "## Specs",
        "",
        (
            "| Requirement | jolty | ngspice | Difference |"
            if replicated
            else "| Requirement | jolty |"
        ),
        "| --- | --- | --- | --- |" if replicated else "| --- | --- |",
    ]
    for verdict in built.verdicts:
        row = f"| {verdict.requirement} | {_reading(verdict)} |"
        if replicated:
            spice = theirs.get(verdict.requirement)
            row += f" {_reading(spice)} | {_difference(verdict, spice)} |"
        lines.append(row)
    return [*lines, ""]


def _reading(verdict: specs.Verdict | None) -> str:
    """A verdict's value, the corner it comes from if any, and its status."""
    if verdict is None:
        return "—"
    at = "" if verdict.corner is None else f" at {verdict.corner}"
    return f"{_si(verdict.value, verdict.unit)}{at}, {verdict.status}"


def _difference(ours: specs.Verdict, theirs: specs.Verdict | None) -> str:
    """How far two engines' readings lie apart, relative to the larger."""
    if theirs is None or theirs.value is None or ours.value is None:
        return "—"
    scale = max(abs(ours.value), abs(theirs.value)) or 1.0
    return _ratio(abs(ours.value - theirs.value) / scale)


def _replication_section(built: Built) -> list[str]:
    """How ngspice replicated the example, or why it could not."""
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
        lines += [
            (
                "| Analysis | Voltages | Largest deviation "
                "| Currents | Largest deviation |"
            ),
            "| --- | --- | --- | --- | --- |",
        ]
        for name, quantities in replication.deviations.items():
            cells = []
            for quantity in ("voltage", "current"):
                count, deviation = quantities.get(quantity, (0, 0.0))
                cells += [str(count), _ratio(deviation) if count else "—"]
            lines.append(f"| `{name}` | {' | '.join(cells)} |")
        lines.append("")
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
    for image in built.replicated_plots:
        lines += [f"![The replication in ngspice](../images/gallery/{image})", ""]
    return lines


def page(built: Built) -> str:
    """The entry's own page: a discovery's holds its prompt and its findings too."""
    entry = built.entry
    buttons = (
        f"[Open in 3D]({entry.viewer}){{ .md-button .md-button--primary }} "
        f"[Script]({entry.source}){{ .md-button }}"
    )
    if entry.discovered:
        folder = f"{REPOSITORY.replace('/blob/', '/tree/')}/discovery/{entry.name}"
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
    """A discovery's prompt, quoted, and its findings, but for their titles."""
    folder = DISCOVERY / entry.name
    prompt = (folder / "prompt.md").read_text(encoding="utf-8").splitlines()[1:]
    findings = (folder / "findings.md").read_text(encoding="utf-8").splitlines()[1:]
    quoted = [f"> {line}".rstrip() for line in _trimmed(prompt)]
    return ["## The prompt", "", *quoted, "", *_trimmed(findings), ""]


def _trimmed(lines: Sequence[str]) -> list[str]:
    """`lines` without the blank lines at either end."""
    kept = list(lines)
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return kept


def _worst_voltage(replication: Replication) -> float | None:
    """The largest deviation of a voltage between the engines, or None."""
    found = [
        quantities["voltage"][1]
        for quantities in replication.deviations.values()
        if "voltage" in quantities
    ]
    return max(found) if found else None


def _card(built: Built) -> list[str]:
    """An entry's card on an index page: its schematic, its title, its summary."""
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
    summary = built.summary
    if entry.discovered:
        findings = (DISCOVERY / entry.name / "findings.md").read_text(encoding="utf-8")
        summary = " ".join(findings.split("\n## ")[0].splitlines()[1:]).strip()
    return [
        image,
        "",
        (
            f"    **[{entry.title}]({entry.folder}/{entry.slug}.md)** · "
            f"{passed}/{len(built.verdicts)} specs{replicated}"
        ),
        "",
        f"    {summary}",
        "",
        (f"    [Open in 3D]({entry.folder}/{entry.viewer}) · [Script]({entry.source})"),
        "",
    ]


def index(built: Sequence[Built]) -> str:
    """The gallery's page: a card per example, in sections."""
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
    """The discoveries' page: how they were made, and a card per problem."""
    lines = [
        "# Discovery",
        "",
        (
            "Some circuits are famous for being hard to get right with few parts: a "
            "current source that ignores its supply and its temperature, a sine wave "
            "that holds its amplitude without distorting it, a notch that stays deep "
            "with real tolerances, a reference without an op-amp, an oscillator whose "
            "frequency follows its control in a straight line, a logic gate of "
            "memristors that tolerates its drive."
        ),
        (
            "For each, a prompt states the bounds as an engineer would, and an agent, "
            "Claude, searched for a circuit with nothing but jolty: it wrote "
            "candidates as components, ran experiments with jolty's Python API and "
            "command line, read the exact currents, delays and drifts jolty reports, "
            "and kept what the numbers supported."
        ),
        (
            "Each page holds the prompt, every experiment's result and what it "
            "showed; the final design runs through `jolty simulate` and again in "
            "ngspice. Where no design met a bound, the page says so, and why."
        ),
        (
            "None of these runs was blind: the agent wrote each prompt knowing the "
            "textbook answers, so they show what jolty lets an agent see rather than "
            "what an agent can invent, and each page's limits say what a built circuit "
            "would add."
        ),
        (
            "The prompts, candidates, experiments and designs are all in "
            "[`discovery/`](https://github.com/maxencefaldor/jolty/tree/main/discovery)."
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
    """The specs that stop the script.

    They are an example's that either engine fails, and a discovery's on which the
    engines disagree: its findings report the bounds it misses.
    """
    ours = built.verdicts
    theirs = built.replication.verdicts if built.replication else ()
    if built.entry.discovered:
        return sum(
            mine.status != spice.status
            for mine, spice in zip(ours, theirs, strict=True)
        )
    return sum(verdict.status != "passed" for verdict in (*ours, *theirs))


def main(names: Sequence[str]) -> int:
    """Builds the entries named, or all, and returns the failures."""
    chosen = [
        entry for entry in (*ENTRIES, *DISCOVERIES) if not names or entry.name in names
    ]
    for folder in {entry.folder for entry in chosen}:
        (DOCS / folder).mkdir(exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    # Each entry builds in a process of its own: a discovery imports its modules by
    # name, such as its `candidates`, and a process that built another discovery would
    # hand it that one's.
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
    # One sentence per line, as the docs' Markdown is checked: the pages quote
    # docstrings, which wrap at 88 columns.
    rumdl = Path(sys.executable).parent / "rumdl"
    subprocess.run([rumdl, "fmt", "--quiet", *written], check=True)  # noqa: S603 - this environment's rumdl, on the pages just written
    return failures


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:]) else 0)

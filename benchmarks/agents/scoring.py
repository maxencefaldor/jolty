# Copyright 2026 Maxence Faldor
"""Shared benchmark scoring helpers for loading designs, checking parts, and simulating.

A design is either a Python script (jolty arm) or a SPICE netlist (ngspice arm), and
given parts come from ``parts.py`` or ``parts.lib`` respectively. After loading, both
arms are evaluated by the same scoring pipeline. Each scorer first measures the exact
nominal design across operating corners: 27 °C nominal supplies plus the four
commercial extremes (0 °C and 70 °C at +/-5 % supply voltage). Designs that pass every
corner are re-evaluated on Monte Carlo boards with part tolerances across all corners
and pass only when yield meets the task threshold.

Each failure is recorded as a `verdict.Shortfall` categorized by origin: missing file,
build or parse error, jolty runtime error, disallowed part, out-of-spec measurement,
or insufficient yield.
"""

import asyncio
import math
import runpy
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
from arms import Arm, libraries
from verdict import Kind, Shortfall, Verdict

from jolty import (
    Component,
    Simulation,
    circuit,
    devices,
    files,
    ir,
    service,
    spice,
    variation,
)
from jolty.errors import (
    BuildError,
    DimensionError,
    ElaborationError,
    JoltyError,
    ScriptError,
    SimulationError,
    SpiceError,
)
from jolty.report import FailureReport
from jolty.settings import Settings
from jolty.units import celsius


def missing(reason: str) -> Shortfall:
    """Creates a `Kind.MISSING` shortfall for a missing deliverable with `reason`."""
    return Shortfall(kind=Kind.MISSING, reason=reason)


def unbuilt(reason: str) -> Shortfall:
    """Creates a `Kind.UNBUILT` shortfall for a build or parse failure with `reason`."""
    return Shortfall(kind=Kind.UNBUILT, reason=reason)


def unelaborated(problem: str, where: str = "") -> Shortfall:
    """Creates a `Kind.UNELABORATED` shortfall for an elaboration `problem`.

    An optional `where` prefix identifies the failing context, such as a corner name.
    """
    reason = f"{where}the engine cannot elaborate it: {problem}"
    return Shortfall(kind=Kind.UNELABORATED, reason=reason)


def unmet(spec: str, reason: str) -> Shortfall:
    """Creates a `Kind.SPEC` shortfall for an unmet requirement `spec` with `reason`."""
    return Shortfall(kind=Kind.SPEC, spec=spec, reason=reason)


def failed(error: JoltyError) -> Shortfall:
    """Converts a `JoltyError` into a categorized `Shortfall`.

    Maps simulation failures to `Kind.ANALYSIS`, elaboration errors to
    `Kind.UNELABORATED`, and build, dimension, script, or SPICE errors to
    `Kind.UNBUILT`.

    Raises:
        JoltyError: Re-raises `error` when no shortfall category applies.
    """
    match error:
        case SimulationError():
            reason = f"the simulation fails: {error}"
            return Shortfall(kind=Kind.ANALYSIS, reason=reason)
        case ElaborationError():
            return unelaborated(str(error))
        case BuildError() | DimensionError() | ScriptError() | SpiceError():
            return unbuilt(f"it does not build: {error}")
    raise error


def script(work: Path, stem: str) -> dict[str, object] | Shortfall:
    """Executes ``{stem}.py`` in `work` and returns its namespace or a `Shortfall`.

    Adds `work` to ``sys.path`` so the script can import sibling modules from the same
    directory.
    """
    path = work / f"{stem}.py"
    if not path.is_file():
        return missing(f"there is no {path.name}")
    if str(work) not in sys.path:
        sys.path.insert(0, str(work))
    try:
        namespace = dict[str, object](runpy.run_path(str(path)))
    except Exception as error:  # noqa: BLE001 - an agent's script may raise anything
        return unbuilt(f"{path.name} fails: {error!r}")
    return namespace


def built(namespace: Mapping[str, object], stem: str) -> Component | Shortfall:
    """Calls ``build()`` in `namespace` for ``{stem}.py`` and returns the `Component`.

    Returns an `unbuilt` shortfall when ``build()`` is missing, raises an exception, or
    returns a non-`Component` value.
    """
    build = namespace.get("build")
    if not callable(build):
        return unbuilt(f"{stem}.py defines no build()")
    try:
        made = build()
    except Exception as error:  # noqa: BLE001 - an agent's script may raise anything
        return unbuilt(f"{stem}.py's build() fails: {error!r}")
    if not isinstance(made, Component):
        return unbuilt("build() returns no component")
    return made


def grounded(made: Component) -> Component:
    """Ensures `made` exposes SPICE ground node ``0`` as a component port.

    The SPICE reader creates a ground port only when an element connects to node ``0``,
    whereas benchmark subcircuits treat node ``0`` as ground regardless.
    """
    ports = made.definition.ports
    if any(port.name == spice.GROUND for port in ports):
        return made
    port = ir.Port(name=spice.GROUND, role=ir.Role.GROUND)
    definition = made.definition.model_copy(update={"ports": (*ports, port)})
    return Component(made.name, definition, made.children)


def netlist(task: Path, work: Path, stem: str) -> Component | Shortfall:
    """Loads subcircuit `stem` from ``{stem}.cir`` in `work`, or returns a `Shortfall`.

    Loads the ngspice arm libraries for `task` before reading the netlist and attaches
    SPICE node ``0`` as ground. Returns a shortfall when the file is missing, fails to
    parse, or defines no subcircuit named `stem`.
    """
    path = work / f"{stem}.cir"
    if not path.is_file():
        return missing(f"there is no {path.name}")
    try:
        return grounded(spice.load(*libraries(task), path).component(stem))
    except ValueError as error:
        return unbuilt(f"jolty cannot read {path.name}: {error}")


def design(task: Path, work: Path, stem: str, arm: Arm) -> Component | Shortfall:
    """Loads the component `stem` from `work` for `arm`, or returns a `Shortfall`.

    Executes ``{stem}.py`` for the jolty arm and parses subcircuit `stem` from
    ``{stem}.cir`` for the ngspice arm.
    """
    if arm is Arm.NGSPICE:
        return netlist(task, work, stem)
    found = script(work, stem)
    return found if isinstance(found, Shortfall) else built(found, stem)


def benched(
    made: Component, ports: Iterable[str], bench: Callable[[], Component]
) -> ir.Circuit | Shortfall:
    """Places `made` onto the test `bench` and returns an `ir.Circuit` or a `Shortfall`.

    Verifies that the ports of `made` match `ports` exactly before calling `bench`.
    """
    found = sorted(port.name for port in made.definition.ports)
    wanted = sorted(ports)
    if found != wanted:
        return unbuilt(f"its ports are {', '.join(found)}, not {', '.join(wanted)}")
    try:
        return circuit(bench())
    except JoltyError as error:
        return failed(error)


_ELEMENTS: Final = {
    "npn": "Q1 collector base emitter",
    "pnp": "Q1 collector base emitter",
    "nmos": "M1 drain gate source bulk",
    "pmos": "M1 drain gate source bulk",
}
"""Minimal SPICE element templates per device kind, omitting the model name."""


def modeled(task: Path, model: str) -> devices.Device:
    """Loads a default `devices.Device` for SPICE `model` from the libraries of `task`.

    Instantiates the model with no instance-level overrides such as MOSFET geometry.

    Raises:
        KeyError: When `model` is absent from the libraries or uses an unsupported
            device kind.
    """
    library = spice.load(*libraries(task))
    card = library.model(model, {})
    if card is None or card.kind not in _ELEMENTS:
        message = f"the libraries of {task.name} give no part `{model}`"
        raise KeyError(message)
    library.read(f".subckt part\n{_ELEMENTS[card.kind]} {model}\n.ends", "part")
    (instance,) = library.component("part").definition.instances
    if not isinstance(instance.of, ir.DeviceOf):
        message = f"`{model}` is no device's model"
        raise KeyError(message)
    return devices.device(instance.of.model, dict(instance.parameters))


def given(task: Path, arm: Arm, *names: str) -> list[devices.Device]:
    """Loads the given `devices.Device` models named in `names` for `task` and `arm`.

    Calls each named factory in ``given/parts.py`` for jolty, or reads the
    corresponding model from ``parts.lib`` for ngspice.

    Raises:
        TypeError: When a factory in ``parts.py`` does not return a `devices.Device`.
    """
    if arm is Arm.NGSPICE:
        return [modeled(task, name) for name in names]
    functions = runpy.run_path(str(task / "given" / "parts.py"))
    found = []
    for name in names:
        made = functions[name]()
        if not isinstance(made, devices.Device):
            message = f"parts.py's {name}() makes no device"
            raise TypeError(message)
        found.append(made)
    return found


@dataclass(frozen=True, slots=True)
class Corner:
    """Operating corner defined by ambient temperature and supply voltage scale."""

    temperature: float = 27.0
    """Ambient temperature in degrees Celsius."""
    supply: float = 1.0
    """Supply voltage multiplier relative to nominal."""

    def describe(self) -> str:
        """Formats the corner as text, such as ``at 0 °C with the supplies 5 % low``."""
        if self.supply == 1:
            return f"at {self.temperature:g} °C"
        side = "low" if self.supply < 1 else "high"
        return (
            f"at {self.temperature:g} °C with the supplies "
            f"{abs(self.supply - 1) * 100:g} % {side}"
        )


NOMINAL: Final = Corner()
"""Nominal operating corner at 27 °C and 100 % supply voltage."""
CORNERS: Final = (
    NOMINAL,
    *(
        Corner(temperature, supply)
        for temperature in (0.0, 70.0)
        for supply in (0.95, 1.05)
    ),
)
"""Nominal corner plus the four commercial extremes (0 °C/70 °C, +/-5 % supply)."""


@dataclass(frozen=True, slots=True)
class Board:
    """Single Monte Carlo board realization evaluated at an operating corner.

    Parameter overrides in `parts` map instance path to parameter path and sampled
    value. An empty `parts` mapping represents the exact nominal design.
    """

    parts: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    corner: Corner = NOMINAL

    def simulation(self, built: ir.Circuit, supplies: Iterable[str]) -> Simulation:
        """Builds a `Simulation` of `built` with board tolerances and corner conditions.

        Scales each top-level DC voltage source in `supplies` by the corner supply
        factor.

        Raises:
            KeyError: When an identifier in `supplies` is not a top-level DC source.
        """
        top = built.components[built.top]
        levels: dict[str, Mapping[str, float]] = {}
        for name in supplies:
            instance = top.instance(name)
            match (
                None
                if instance is None
                else ir.parameter(instance.parameters, "waveform.dc")
            ):
                case float() | int() as level:
                    levels[name] = {"waveform.dc": level * self.corner.supply}
                case _:
                    message = f"{name} is no DC source at the top of the bench"
                    raise KeyError(message)
        return Simulation(
            built,
            temperature=celsius(self.corner.temperature),
            overrides={**self.parts, **levels},
        )


type Measure = Callable[[Board], dict[str, float]]
"""Callable returning named scalar measurements for a single `Board`."""
type Shortfalls = Callable[[dict[str, float]], list[Shortfall]]
"""Callable checking measured values and returning any task shortfalls."""
type Spread = Callable[[ir.Instance], dict[str, ir.Tolerance]]
"""Callable mapping a device instance to parameter tolerances for Monte Carlo boards."""

PARTS: Final[Mapping[str, tuple[str, ir.Tolerance]]] = {
    "resistor": ("resistance", ir.Tolerance(relative=0.01)),
    "capacitor": ("capacitance", ir.Tolerance(relative=0.05)),
}
"""Default tolerance specification for design-added passives, keyed by device model."""
THERMAL_VOLTAGE: Final = 0.025852
"""Thermal voltage kT/q at 27 °C in volts."""
REELS: Final[Mapping[str, Mapping[str, ir.Tolerance]]] = {
    polarity: {
        "forward_beta": ir.Tolerance(
            relative=0.01, lot=ir.Lot(name=part, relative=beta)
        ),
        "saturation_current": ir.Tolerance(
            relative=math.expm1(1e-3 / THERMAL_VOLTAGE),
            distribution="log_uniform",
            lot=ir.Lot(name=part, relative=junction, distribution="log_uniform"),
        ),
    }
    for polarity, part, beta, junction in (
        ("npn", "2N3904", 0.57, 34.0),
        ("pnp", "2N3906", 0.37, 8.5),
    )
}
"""Discrete BJT reel and intra-reel tolerances by polarity for 2N3904 and 2N3906.

All transistors of a polarity on one board share a reel spanning the 10 mA datasheet
limits. Current gain ranges from 100 to 300 (models: 160 and 163), and base-emitter
saturation voltage ranges from 0.65 V to 0.85 V (models: 0.743 V and 0.791 V). Reel
spread uses the widest range around the model that keeps every device within both
limits. Intra-reel mismatch mirrors adjacent-die matched pairs (Diodes DMMT3904W):
current gain within 2 % and base-emitter voltage within 2 mV.
"""
DIE: Final = ir.Lot(name="die", relative=0.2)
"""Lot tolerance of +/-20 % for monolithic resistors on a single die (US 6,819,093)."""
MATCHED: Final = ir.Tolerance(relative=0.001, distribution="normal", lot=DIE)
"""Integrated resistor tolerance: +/-0.1 % 3-sigma die matching (Hastings)."""

BOARDS: Final = 100
"""Number of Monte Carlo boards simulated after a design passes all nominal corners."""
YIELD: Final = 0.95
"""Minimum fraction of Monte Carlo boards required to pass all corners."""
SEED: Final = 0
"""Deterministic random seed used to draw Monte Carlo boards for every design."""


def simulate(
    script: Path, plan: Path | None = None
) -> tuple[service.SimulateResult, ir.Circuit]:
    """Runs ``jolty simulate`` on `script` and returns the result and built circuit.

    Uses the embedded plan and specs in `script`, or overrides them with `plan` if
    given.
    """
    with tempfile.TemporaryDirectory() as directory:
        saved = Path(directory) / "circuit.jolty.json"
        request = service.SimulateRequest(
            script=script,
            plan=plan,
            output=saved,
            results=Path(directory) / "circuit.results.npz",
            schematic=False,
            plot=False,
        )
        result = asyncio.run(service.simulate(request, Settings()))
        return result, files.load_circuit(saved)


PASSING: Final = frozenset({"passed", "marginal"})
"""Specification verdict statuses counted as passing, including marginal passes."""


def planned(result: service.SimulateResult) -> list[Shortfall]:
    """Extracts all `Shortfall` entries from a ``jolty simulate`` `result`.

    Collects elaboration problems, error-level rule diagnostics, failed analyses across
    nominal and corner runs, unmet or unmeasured specs including worst-case checks,
    yield failures on Monte Carlo or variant batches, and fidelity mismatches.
    """
    found = [unelaborated(each) for each in result.circuit.problems]
    if result.report is not None:
        found += [
            Shortfall(kind=Kind.RULE, reason=diagnostic.describe())
            for diagnostic in result.report.diagnostics
            if diagnostic.level == "error"
        ]
    found += [
        Shortfall(kind=Kind.ANALYSIS, reason=analysis.describe())
        for analysis in result.analyses
        if isinstance(analysis, FailureReport)
    ]
    for corner in result.corners:
        found += [
            unelaborated(each, f"at the corner {corner.name}, ")
            for each in corner.problems
        ]
        found += [
            Shortfall(
                kind=Kind.ANALYSIS,
                reason=f"at the corner {corner.name}, {failure.describe()}",
            )
            for failure in corner.failures
        ]
    found += [
        unmet(verdict.requirement, verdict.describe())
        for verdict in result.specs
        if verdict.status not in PASSING
    ]
    if result.worst_case is not None:
        found += [
            unmet(verdict.requirement, f"at its worst case, {verdict.describe()}")
            for verdict in result.worst_case.verdicts
            if verdict.status not in PASSING
        ]
    for name, batch in (("yield", result.monte_carlo), ("variants", result.variants)):
        if batch is not None and batch.yield_ is not None and not batch.yield_.met:
            found.append(unmet(name, batch.yield_.describe()))
    if result.fidelity is not None and not result.fidelity.ok:
        found.append(unmet("fidelity", result.fidelity.describe()))
    return found


def count_devices(circuit: ir.Circuit, path: str) -> int:
    """Counts leaf device instances recursively inside the component at `path`."""
    component = circuit.components[circuit.top]
    for name in path.split("/"):
        instance = component.instance(name)
        if instance is None or not isinstance(instance.of, ir.ComponentOf):
            message = f"no component instance at {path}"
            raise KeyError(message)
        component = circuit.components[instance.of.component]
    return _count(circuit, component)


def _count(circuit: ir.Circuit, component: ir.Component) -> int:
    total = 0
    for instance in component.instances:
        if isinstance(instance.of, ir.ComponentOf):
            total += _count(circuit, circuit.components[instance.of.component])
        else:
            total += 1
    return total


def disallowed(
    built: ir.Circuit, given: Iterable[devices.Device], allowed: Iterable[str]
) -> list[Shortfall]:
    """Checks all non-top-level devices in `built` against `given` and `allowed` parts.

    Devices matching a given model must carry exact given parameters; all other devices
    must use an allowed model such as ``resistor``. Violations produce ``parts``
    shortfalls.
    """
    parts = [(part.model, part.parameters) for part in given]
    models = {model for model, _ in parts}
    permitted = set(allowed)
    found = []
    for name, made in built.components.items():
        if name == built.top:
            continue
        for instance in made.instances:
            if not isinstance(instance.of, ir.DeviceOf):
                continue
            kind = instance.of.model
            if kind in models:
                if (kind, instance.parameters) not in parts:
                    found.append(unmet("parts", f"{instance.name} is not a given part"))
            elif kind not in permitted:
                reason = f"{instance.name} is a {kind}, which the task forbids"
                found.append(unmet("parts", reason))
    return found


def parts(instance: ir.Instance) -> dict[str, ir.Tolerance]:
    """Returns default passive tolerances for `instance`: 1 % resistors, 5 % capacitors.

    Non-passive devices receive no tolerance overrides.
    """
    match instance.of:
        case ir.DeviceOf(model=model) if model in PARTS:
            parameter, tolerance = PARTS[model]
            return {parameter: tolerance}
        case _:
            return {}


def discrete(instance: ir.Instance) -> dict[str, ir.Tolerance]:
    """Returns discrete-board tolerances for `instance`.

    Applies [`parts`][] tolerances to passives and [`REELS`][] lot and matching
    tolerances to 2N3904 and 2N3906 bipolar transistors.
    """
    match instance.of:
        case ir.DeviceOf(model="bipolar_transistor"):
            polarity = instance.parameters.get("polarity", "npn")
            return dict(REELS[str(polarity)])
        case _:
            return parts(instance)


def integrated(instance: ir.Instance) -> dict[str, ir.Tolerance]:
    """Returns integrated-circuit tolerances for `instance`.

    Applies die-matched resistor tolerances (`MATCHED`) and default [`parts`][]
    capacitor tolerances. Transistors remain exact to match the patent reference
    comparison.
    """
    match instance.of:
        case ir.DeviceOf(model="resistor"):
            return {"resistance": MATCHED}
        case _:
            return parts(instance)


def boards(built: ir.Circuit, spread: Spread) -> list[Board]:
    """Generates Monte Carlo `Board` samples for non-top-level devices in `built`.

    Overrides design tolerances using `spread` while keeping test-bench devices exact.
    Returns a single exact nominal `Board` when no devices carry tolerances.
    """
    components = {
        name: component
        if name == built.top
        else component.model_copy(
            update={
                "instances": tuple(
                    each.model_copy(update={"tolerances": spread(each)})
                    for each in component.instances
                )
            }
        )
        for name, component in built.components.items()
    }
    varied = variation.toleranced(built.model_copy(update={"components": components}))
    if not varied:
        return [Board()]
    return [Board(variation.sample(varied, SEED, index)) for index in range(BOARDS)]


def judge(
    measure: Measure, shortfalls: Shortfalls, board: Board | None = None
) -> tuple[dict[str, float], list[Shortfall]]:
    """Runs `measure` on `board` and evaluates `shortfalls`, defaulting to exact parts.

    Converts any raised `JoltyError` into a categorized `Shortfall` via `failed`.
    """
    try:
        found = measure(board or Board())
    except JoltyError as error:
        return {}, [failed(error)]
    return found, shortfalls(found)


def cornered(
    measure: Measure, shortfalls: Shortfalls
) -> tuple[dict[str, float], list[Shortfall]]:
    """Measures the exact design across all `CORNERS` and collects any shortfalls.

    Nominal measurements keep plain keys; non-nominal corner measurements are prefixed
    with the corner description, such as ``at 0 °C with the supplies 5 % low: fitness``.
    """
    measures: dict[str, float] = {}
    reasons: list[Shortfall] = []
    for corner in CORNERS:
        found, short = judge(measure, shortfalls, Board(corner=corner))
        named = corner == NOMINAL
        measures |= {
            name if named else f"{corner.describe()}: {name}": value
            for name, value in found.items()
        }
        reasons += short if named else [_at(corner, each) for each in short]
    return measures, reasons


def _at(corner: Corner, shortfall: Shortfall) -> Shortfall:
    """Prefixes the reason of `shortfall` with the description of `corner`."""
    reason = f"{corner.describe()}, {shortfall.reason}"
    return shortfall.model_copy(update={"reason": reason})


def tolerate(
    built: ir.Circuit, measure: Measure, shortfalls: Shortfalls, spread: Spread
) -> tuple[dict[str, float], list[Shortfall]]:
    """Evaluates Monte Carlo boards across all corners and checks yield against `YIELD`.

    A board passes only if every corner passes. Returns ``yield`` plus median nominal
    measurements, and reports a ``yield`` shortfall citing the first failing board when
    yield falls below `YIELD`.
    """
    drawn = boards(built, spread)
    nominal: list[dict[str, float]] = []
    short: list[str] = []
    for board in drawn:
        for corner in CORNERS:
            found, reasons = judge(measure, shortfalls, Board(board.parts, corner))
            if corner == NOMINAL and found:
                nominal.append(found)
            if reasons:
                short.append(
                    reasons[0].reason
                    if corner == NOMINAL
                    else _at(corner, reasons[0]).reason
                )
                break
    share = 1 - len(short) / len(drawn)
    measured = {"yield": share} | {
        f"median {name}": float(np.median([each[name] for each in nominal]))
        for name in (nominal[0] if nominal else {})
    }
    if share >= YIELD:
        return measured, []
    reason = (
        f"{share:.0%} of {len(drawn)} boards meet the task in every corner, fewer than "
        f"{YIELD:.0%}; on the first that falls short, {short[0]}"
    )
    return measured, [unmet("yield", reason)]


@dataclass(frozen=True, slots=True)
class DesignSpec:
    """Configuration for scoring a component design across corners and tolerances."""

    ports: Sequence[str]
    bench: Callable[[Component], Component]
    measure: Callable[[Component, Board], dict[str, float]]
    shortfalls: Shortfalls
    parts: Sequence[str]
    allowed: Iterable[str]
    spread: Spread
    stem: str = "design"
    extra_measures: Mapping[str, float] = field(default_factory=dict)


def score_design(task: Path, work: Path, arm: Arm, spec: DesignSpec) -> Verdict:
    """Evaluates a design by loading, benching, checking parts, and simulating."""
    made = design(task, work, spec.stem, arm)
    if isinstance(made, Shortfall):
        return Verdict(met=False, shortfalls=(made,))
    built = benched(made, spec.ports, lambda: spec.bench(made))
    if isinstance(built, Shortfall):
        return Verdict(met=False, shortfalls=(built,))
    reasons = disallowed(built, given(task, arm, *spec.parts), spec.allowed)

    def measured(board: Board) -> dict[str, float]:
        return spec.measure(made, board)

    measures, short = cornered(measured, spec.shortfalls)
    reasons += short
    if not reasons:
        spread, reasons = tolerate(built, measured, spec.shortfalls, spec.spread)
        measures |= spread
    measures |= spec.extra_measures
    measures["devices"] = float(count_devices(built, "X1"))
    return Verdict(met=not reasons, measures=measures, shortfalls=tuple(reasons))

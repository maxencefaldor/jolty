# Copyright 2026 Maxence Faldor
"""What the scorers share: designs read and simulated, parts checked, designs judged.

A design is the jolty arm's script or the ngspice arm's netlist, and the given parts are
the functions of a task's ``parts.py``, or the models of its ``parts.lib``, as the arm's
agent got them; from them on, a scorer judges both arms by the same code. A scorer
measures a design with its parts exact, in each corner of the conditions it
must work in: at 0 °C and at 70 °C, the commercial range, with its supplies 5 % low and
5 % high, besides 27 °C and its supplies exact. If it meets its task in every corner, it
is measured again on boards built of real parts, whose values stray within their
tolerances, each board in every corner: it meets the task only when enough of those
boards meet it in all of them.

Each way a design falls short is a `verdict.Shortfall` of the kind its source makes it:
a missing file, a script or netlist that does not build or read, an error jolty raises
of its own kind, a part the task forbids, a measurement beyond its bound, a yield too
low.
"""

import asyncio
import math
import runpy
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
from arms import Arm, libraries
from verdict import Kind, Shortfall

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
    """The shortfall of work that lacks what the task asks for, for `reason`."""
    return Shortfall(kind=Kind.MISSING, reason=reason)


def unbuilt(reason: str) -> Shortfall:
    """The shortfall of a design that does not build or read, for `reason`."""
    return Shortfall(kind=Kind.UNBUILT, reason=reason)


def unelaborated(problem: str, where: str = "") -> Shortfall:
    """The shortfall of a design the engine cannot elaborate, for `problem`.

    `where` says where, such as at a corner, before it.
    """
    reason = f"{where}the engine cannot elaborate it: {problem}"
    return Shortfall(kind=Kind.UNELABORATED, reason=reason)


def unmet(spec: str, reason: str) -> Shortfall:
    """The shortfall of a design that falls short of the requirement `spec`."""
    return Shortfall(kind=Kind.SPEC, spec=spec, reason=reason)


def failed(error: JoltyError) -> Shortfall:
    """The shortfall of a design on which jolty raised `error`, of the error's kind.

    An analysis that fails, a circuit the engine cannot elaborate, and a circuit that
    cannot be built or read, as jolty's errors of each say.

    Raises:
        JoltyError: `error` itself, when it is of no kind a design's shortfall is.
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
    """What the script `stem`, such as ``design``, of `work` defines, or why it fails.

    It runs as a script whose directory is importable, so it may import modules of its
    own from beside it.
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
    """The component the ``build()`` of `namespace`, script `stem`'s, returns.

    Returns why there is none, when it has no ``build()``, or it fails or returns no
    component.
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
    """`made`, a subcircuit read, with SPICE's ground, node ``0``, as a port.

    jolty's reader gives a subcircuit a ground port only when an element joins node
    ``0``; a design's ground is node ``0`` whether or not anything joins it.
    """
    ports = made.definition.ports
    if any(port.name == spice.GROUND for port in ports):
        return made
    port = ir.Port(name=spice.GROUND, role=ir.Role.GROUND)
    definition = made.definition.model_copy(update={"ports": (*ports, port)})
    return Component(made.name, definition, made.children)


def netlist(task: Path, work: Path, stem: str) -> Component | Shortfall:
    """The subcircuit `stem` of the netlist `stem`, such as ``design.cir``, of `work`.

    jolty's SPICE reader reads the netlist after the libraries of `task`'s ngspice arm,
    whose models it may name, and the subcircuit's ground is SPICE's node ``0``.
    Returns why there is none, when the netlist is missing, jolty cannot read it, or it
    defines no subcircuit `stem`.
    """
    path = work / f"{stem}.cir"
    if not path.is_file():
        return missing(f"there is no {path.name}")
    try:
        return grounded(spice.load(*libraries(task), path).component(stem))
    except ValueError as error:
        return unbuilt(f"jolty cannot read {path.name}: {error}")


def design(task: Path, work: Path, stem: str, arm: Arm) -> Component | Shortfall:
    """The component the design `stem` of `work` is, as `arm` writes it, or why none.

    It is the component the script ``stem.py`` builds, or the subcircuit ``stem`` of
    the netlist ``stem.cir``.
    """
    if arm is Arm.NGSPICE:
        return netlist(task, work, stem)
    found = script(work, stem)
    return found if isinstance(found, Shortfall) else built(found, stem)


def benched(
    made: Component, ports: Iterable[str], bench: Callable[[], Component]
) -> ir.Circuit | Shortfall:
    """The circuit of `made` on its bench, which `bench` places it on, or why none.

    The design's ports must be `ports`, those the bench joins.
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
"""An element of a model of each SPICE type the given parts are, but its model."""


def modeled(task: Path, model: str) -> devices.Device:
    """The device jolty reads of an element of `model`, a model of `task`'s libraries.

    The element gives the model nothing of its own, such as a MOSFET's size.

    Raises:
        KeyError: When no model of the libraries is named `model`, or it is of a type
            no given part is.
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
    return devices.device(instance.of.device, dict(instance.parameters))


def given(task: Path, arm: Arm, *names: str) -> list[devices.Device]:
    """The given parts `names` of `task`, as `arm`'s agent got them.

    Each is the device the function of that name in ``parts.py`` makes, or the one
    jolty reads of an element of the model of that name in ``parts.lib``.

    Raises:
        TypeError: When a function of ``parts.py`` makes no device.
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
    """Conditions a design must work in: its temperature and its supplies' level."""

    temperature: float = 27.0
    """The temperature, in degrees Celsius."""
    supply: float = 1.0
    """Each supply's level, relative to its nominal one."""

    def describe(self) -> str:
        """The corner in words, such as ``at 0 °C with the supplies 5 % low``."""
        if self.supply == 1:
            return f"at {self.temperature:g} °C"
        side = "low" if self.supply < 1 else "high"
        return (
            f"at {self.temperature:g} °C with the supplies "
            f"{abs(self.supply - 1) * 100:g} % {side}"
        )


NOMINAL: Final = Corner()
"""The nominal conditions: 27 °C, every supply at its level."""
CORNERS: Final = (
    NOMINAL,
    *(
        Corner(temperature, supply)
        for temperature in (0.0, 70.0)
        for supply in (0.95, 1.05)
    ),
)
"""The nominal conditions, then the four that bound those a design must work in: from
0 °C to 70 °C, and each supply within 5 % of its level."""


@dataclass(frozen=True, slots=True)
class Board:
    """A board of the design, in a corner.

    Its parts are the values that stray from the design's, by instance path then by
    parameter path, as a Monte Carlo sample holds them; a board without any is the
    design, its parts exact.
    """

    parts: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    corner: Corner = NOMINAL

    def simulation(self, built: ir.Circuit, supplies: Iterable[str]) -> Simulation:
        """`built` on the board: of its parts, at its corner's temperature.

        Each of `supplies`, the ids of DC voltage sources at the top of `built`, takes
        its level in the corner.

        Raises:
            KeyError: When a supply is no DC source at the top.
        """
        top = built.components[built.top]
        levels: dict[str, Mapping[str, float]] = {}
        for id in supplies:
            instance = top.instance(id)
            match (
                None
                if instance is None
                else ir.parameter(instance.parameters, "waveform.dc")
            ):
                case float() | int() as level:
                    levels[id] = {"waveform.dc": level * self.corner.supply}
                case _:
                    message = f"{id} is no DC source at the top of the bench"
                    raise KeyError(message)
        return Simulation(
            built,
            temperature=celsius(self.corner.temperature),
            overrides={**self.parts, **levels},
        )


type Measure = Callable[[Board], dict[str, float]]
"""What a scorer measures on a board of the design, by name."""
type Shortfalls = Callable[[dict[str, float]], list[Shortfall]]
"""Why what a scorer measured on a board falls short of the task, if it does."""
type Spread = Callable[[ir.Instance], dict[str, ir.Tolerance]]
"""How far each parameter of a device instance of the design strays on its boards, by
parameter path."""

PARTS: Final[Mapping[str, tuple[str, ir.Tolerance]]] = {
    "resistor": ("resistance", ir.Tolerance(relative=0.01)),
    "capacitor": ("capacitance", ir.Tolerance(relative=0.05)),
}
"""Each part a design may add, by model: the parameter that strays, and how far."""
THERMAL_VOLTAGE: Final = 0.025852
"""kT/q at 27 °C, in volts: the base-emitter voltage a factor of e in saturation current
moves."""
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
"""How the given transistors stray, by polarity: the 2N3904, an NPN, and the 2N3906, a
PNP.

A board's transistors of one type come from one reel, which may lie anywhere its
datasheet allows: a current gain at 10 mA from 100 to 300, 160 and 163 in the models,
and a base-emitter saturation voltage at 10 mA from 0.65 V to 0.85 V, 0.743 V and
0.791 V in the models. The reel's spread is the widest about the model that keeps each
transistor on it within both. On a reel, each strays from the others as little as a
matched pair cut from adjacent die (Diodes' DMMT3904W): its current gain within 2 % and
its base-emitter voltage within 2 mV of any other's."""
DIE: Final = ir.Lot(name="die", relative=0.2)
"""The lot of an integrated circuit's resistors: within 20 % of their values together,
the middle of the 15 % to 25 % monolithic resistors keep (US 6,819,093)."""
MATCHED: Final = ir.Tolerance(relative=0.001, distribution="normal", lot=DIE)
"""An integrated resistor: within 0.1 % of the others on its die, three standard
deviations, the moderate matching of Hastings's *The Art of Analog Layout*."""

BOARDS: Final = 100
"""The boards built of a design that meets its task with exact parts."""
YIELD: Final = 0.95
"""The share of those boards that must meet the task too."""
SEED: Final = 0
"""The seed the boards are drawn from, the same for every design."""


def simulate(
    script: Path, plan: Path | None = None
) -> tuple[service.SimulateResult, ir.Circuit]:
    """What ``jolty simulate`` finds of `script`, and the circuit it built.

    The script's own plan and specs apply, or `plan`'s when given.
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
"""The statuses of a requirement that passes, if marginally."""


def planned(result: service.SimulateResult) -> list[Shortfall]:
    """Why `result`, what ``jolty simulate`` found of a design, falls short of its plan.

    They are jolty's own reasons for a result that is not ok, each of its kind: the
    circuit does not elaborate, at the plan's conditions or a corner; a rule finds an
    error; an analysis fails, there or at a corner; a requirement fails or goes
    unmeasured, or fails at its worst-case extremes; the yield of the plan's Monte
    Carlo analysis, or of its variants, falls below the specs' target; or a model the
    plan's fidelity chose disagrees with its circuit.
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
        if batch is not None and batch.passing is not None and not batch.passing.met:
            found.append(unmet(name, batch.passing.describe()))
    if result.fidelity is not None and not result.fidelity.ok:
        found.append(unmet("fidelity", result.fidelity.describe()))
    return found


def count_devices(circuit: ir.Circuit, path: str) -> int:
    """How many devices the component instance at `path` holds, at any depth."""
    component = circuit.components[circuit.top]
    for id in path.split("/"):
        instance = component.instance(id)
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
    """Why each device below the top of `built` is neither `given` nor `allowed`.

    A device of a given part's model must take that part's parameters; any other device
    must be of an allowed model, such as ``resistor``. Each falls short of the spec
    ``parts``.
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
            kind = instance.of.device
            if kind in models:
                if (kind, instance.parameters) not in parts:
                    found.append(unmet("parts", f"{instance.id} is not a given part"))
            elif kind not in permitted:
                reason = f"{instance.id} is a {kind}, which the task forbids"
                found.append(unmet("parts", reason))
    return found


def parts(instance: ir.Instance) -> dict[str, ir.Tolerance]:
    """How far a part a design adds strays: a resistor within 1 %, a capacitor 5 %.

    Any other device stays exact.
    """
    match instance.of:
        case ir.DeviceOf(device=model) if model in PARTS:
            parameter, tolerance = PARTS[model]
            return {parameter: tolerance}
        case _:
            return {}


def discrete(instance: ir.Instance) -> dict[str, ir.Tolerance]:
    """How far a part of a discrete design strays.

    A resistor and a capacitor stray as [`parts`][] has them, and a 2N3904 or a 2N3906
    with its reel, as [`REELS`][] has it.
    """
    match instance.of:
        case ir.DeviceOf(device="bipolar_transistor"):
            polarity = instance.parameters.get("polarity", "npn")
            return dict(REELS[str(polarity)])
        case _:
            return parts(instance)


def integrated(instance: ir.Instance) -> dict[str, ir.Tolerance]:
    """How far a part of an integrated design strays.

    A resistor strays with its die, and within its matching of the others, and a
    capacitor as [`parts`][] has it. Transistors stay exact, as the patent's do in the
    errors a design's are judged against.
    """
    match instance.of:
        case ir.DeviceOf(device="resistor"):
            return {"resistance": MATCHED}
        case _:
            return parts(instance)


def boards(built: ir.Circuit, spread: Spread) -> list[Board]:
    """The boards built of the design below the top of `built`, in the nominal corner.

    Each part's values are drawn as `spread` has them, whatever tolerance the design
    gives them; the bench's parts stay exact. A design that strays nowhere has one
    board, the exact one.
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
    """What `measure` finds on `board`, and why it falls short of the task.

    The board is the exact design by default. It falls short where `shortfalls` finds
    it does, or as `failed` has it when jolty raises an error, such as a simulation
    that fails.
    """
    try:
        found = measure(board or Board())
    except JoltyError as error:
        return {}, [failed(error)]
    return found, shortfalls(found)


def cornered(
    measure: Measure, shortfalls: Shortfalls
) -> tuple[dict[str, float], list[Shortfall]]:
    """What `measure` finds on the exact design in every corner, and why it falls short.

    It falls short in any corner where it falls short of the task. The measurements
    are those in the nominal corner, then those in each other, named after it, such as
    ``at 0 °C with the supplies 5 % low: fitness``.
    """
    measurements: dict[str, float] = {}
    reasons: list[Shortfall] = []
    for corner in CORNERS:
        found, short = judge(measure, shortfalls, Board(corner=corner))
        named = corner == NOMINAL
        measurements |= {
            name if named else f"{corner.describe()}: {name}": value
            for name, value in found.items()
        }
        reasons += short if named else [_at(corner, each) for each in short]
    return measurements, reasons


def _at(corner: Corner, shortfall: Shortfall) -> Shortfall:
    """`shortfall`, found in `corner`, which its reason names."""
    reason = f"{corner.describe()}, {shortfall.reason}"
    return shortfall.model_copy(update={"reason": reason})


def tolerate(
    built: ir.Circuit, measure: Measure, shortfalls: Shortfalls, spread: Spread
) -> tuple[dict[str, float], list[Shortfall]]:
    """What `measure` finds on the boards of `built`, and why too few meet the task.

    A board meets the task when it meets it in every corner. The measurements are the
    share of boards that do, ``yield``, and the median over the boards of each of
    `measure`'s in the nominal corner. Too few falls short of the spec ``yield``, with
    the first shortfall of the first board that falls short.
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

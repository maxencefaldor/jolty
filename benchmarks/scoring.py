# Copyright 2026 Maxence Faldor
"""What the scorers share: scripts simulated, parts counted and checked, designs judged.

A scorer measures a design with its parts exact, in each corner of the conditions it
must work in: at 0 °C and at 70 °C, the commercial range, with its supplies 5 % low and
5 % high, besides 27 °C and its supplies exact. If it meets its task in every corner, it
is measured again on boards built of real parts, whose values stray within their
tolerances, each board in every corner: it meets the task only when enough of those
boards meet it in all of them.
"""

import asyncio
import math
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from jolty import Simulation, devices, files, ir, service, variation
from jolty.errors import SimulationError
from jolty.settings import Settings
from jolty.units import celsius


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
type Shortfalls = Callable[[dict[str, float]], list[str]]
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
) -> list[str]:
    """Why each device below the top of `built` is neither `given` nor `allowed`.

    A device of a given part's model must take that part's parameters; any other device
    must be of an allowed model, such as ``resistor``.
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
                    found.append(f"{instance.id} is not a given part")
            elif kind not in permitted:
                found.append(f"{instance.id} is a {kind}, which the task forbids")
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
) -> tuple[dict[str, float], list[str]]:
    """What `measure` finds on `board`, and why it falls short of the task.

    The board is the exact design by default. It falls short where `shortfalls` finds
    it does, or when a simulation fails.
    """
    try:
        found = measure(board or Board())
    except SimulationError as error:
        return {}, [f"the simulation fails: {error}"]
    return found, shortfalls(found)


def cornered(
    measure: Measure, shortfalls: Shortfalls
) -> tuple[dict[str, float], list[str]]:
    """What `measure` finds on the exact design in every corner, and why it falls short.

    It falls short in any corner where it falls short of the task. The measurements
    are those in the nominal corner, then those in each other, named after it, such as
    ``at 0 °C with the supplies 5 % low: fitness``.
    """
    measurements: dict[str, float] = {}
    reasons: list[str] = []
    for corner in CORNERS:
        found, short = judge(measure, shortfalls, Board(corner=corner))
        named = corner == NOMINAL
        measurements |= {
            name if named else f"{corner.describe()}: {name}": value
            for name, value in found.items()
        }
        reasons += (
            short if named else [f"{corner.describe()}, {each}" for each in short]
        )
    return measurements, reasons


def tolerate(
    built: ir.Circuit, measure: Measure, shortfalls: Shortfalls, spread: Spread
) -> tuple[dict[str, float], list[str]]:
    """What `measure` finds on the boards of `built`, and why too few meet the task.

    A board meets the task when it meets it in every corner. The measurements are the
    share of boards that do, ``yield``, and the median over the boards of each of
    `measure`'s in the nominal corner.
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
                    reasons[0]
                    if corner == NOMINAL
                    else f"{corner.describe()}, {reasons[0]}"
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
    return measured, [reason]

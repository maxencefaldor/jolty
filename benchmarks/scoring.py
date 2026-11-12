# Copyright 2026 Maxence Faldor
"""What the scorers share: scripts simulated, parts counted and checked, designs judged.

A scorer measures a design with its parts exact, then, if it meets its task, on boards
built of real parts, whose values stray within their tolerances: it meets the task only
when enough of those boards meet it too.
"""

import asyncio
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np

from jolty import devices, files, ir, service, variation
from jolty.errors import SimulationError
from jolty.settings import Settings

type Board = Mapping[str, Mapping[str, float]]
"""The values of a board's parts that stray from the design's, by instance path then by
parameter name, as a Monte Carlo sample holds them."""
type Measure = Callable[[Board], dict[str, float]]
"""What a scorer measures on a board of the design, by name."""
type Shortfalls = Callable[[dict[str, float]], list[str]]
"""Why what a scorer measured on a board falls short of the task, if it does."""

EXACT: Final[Board] = MappingProxyType[str, Mapping[str, float]]({})
"""The board whose parts all take the design's values."""
TOLERANCES: Final[Mapping[str, tuple[str, ir.Tolerance]]] = {
    "resistor": ("resistance", ir.Tolerance(relative=0.01)),
    "capacitor": ("capacitance", ir.Tolerance(relative=0.05)),
}
"""Each part a design may add, by model: the parameter that strays, and how far."""
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


def boards(built: ir.Circuit) -> list[Board]:
    """The boards built of the design below the top of `built`.

    On each, every resistor lies within 1 % of its value and every capacitor within 5 %,
    drawn evenly and independently, whatever tolerance the design gives them; the given
    parts and the bench's stay exact. A design without either has one board, the exact
    one.
    """
    components = {
        name: component
        if name == built.top
        else component.model_copy(
            update={"instances": [_toleranced(each) for each in component.instances]}
        )
        for name, component in built.components.items()
    }
    varied = variation.toleranced(built.model_copy(update={"components": components}))
    if not varied:
        return [EXACT]
    generator = np.random.default_rng(SEED)
    return [variation.sample(varied, generator) for _ in range(BOARDS)]


def _toleranced(instance: ir.Instance) -> ir.Instance:
    match instance.of:
        case ir.DeviceOf(device=model) if model in TOLERANCES:
            parameter, tolerance = TOLERANCES[model]
            return instance.model_copy(update={"tolerances": {parameter: tolerance}})
        case ir.DeviceOf():
            return instance.model_copy(update={"tolerances": {}})
        case ir.ComponentOf():
            return instance


def judge(
    measure: Measure, shortfalls: Shortfalls, board: Board = EXACT
) -> tuple[dict[str, float], list[str]]:
    """What `measure` finds on `board`, and why it falls short of the task.

    It falls short where `shortfalls` finds it does, or when a simulation fails.
    """
    try:
        found = measure(board)
    except SimulationError as error:
        return {}, [f"the simulation fails: {error}"]
    return found, shortfalls(found)


def tolerate(
    built: ir.Circuit, measure: Measure, shortfalls: Shortfalls
) -> tuple[dict[str, float], list[str]]:
    """What `measure` finds on the boards of `built`, and why too few meet the task.

    The measurements are the share of boards that meet the task, ``yield``, and the
    median over the boards of each of `measure`'s.
    """
    drawn = boards(built)
    judged = [judge(measure, shortfalls, board) for board in drawn]
    found = [measured for measured, _ in judged if measured]
    short = [reasons[0] for _, reasons in judged if reasons]
    share = 1 - len(short) / len(drawn)
    spread = {"yield": share} | {
        f"median {name}": float(np.median([each[name] for each in found]))
        for name in (found[0] if found else {})
    }
    if share >= YIELD:
        return spread, []
    reason = (
        f"{share:.0%} of {len(drawn)} boards meet the task, fewer than {YIELD:.0%}; "
        f"on the first that falls short, {short[0]}"
    )
    return spread, [reason]

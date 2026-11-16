# Copyright 2026 Maxence Faldor
"""Fidelity: logic models characterized from components' circuits, and their contracts.

A plan's fidelity maps scopes, patterns of instance paths, to
[`Characterized`][jolty.plan.Characterized] settings. Each component instance a scope
matches is simulated as a logic cell characterized from its component's own circuit,
as a standard cell library characterizes its cells:

- its truth tables, from an operating point at each combination of its inputs;
- each input's capacitance, from a small-signal response at each combination;
- for each change of an input that changes an output, the output's delay and
  transition, from transients over a grid of input transitions and output loads.

The grid's scale is the cell's own: its loads are fanouts of its mean input capacitance,
and its transitions multiples of its outputs' mean transition at a fanout of one. A
second grid, between the first's points, measures the error of interpolating it.

Spot checks keep each model honest: each runs the plan again with one sampled instance
at its circuit's own fidelity, and compares its outputs' changes with the model's. Any
change further apart than the model's measured error, plus the scope's tolerance of its
mean delay, violates the contract, and the report says where.
"""

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from jolty import devices, ir, results
from jolty.errors import CharacterizationError, JoltyError
from jolty.plan import Ac, Characterized, OperatingPoint, Plan, Transient
from jolty.results import AnalysisResult, Excursion
from jolty.simulate import Elaborated, Simulation, kelvins
from jolty.specs import Verdict
from jolty.units import FREQUENCY, TIME, VOLTAGE, Quantity, format_si

LEVEL_MARGIN: Final = 0.1
"""How far from its rail, as a fraction of the supply, an output may settle and still
hold a logic level."""

_PROBE_FREQUENCY: Final = 1.0
"""The frequency, in hertz, at which an input's capacitance is read: low enough that a
gate's input is a pure capacitance."""

_WINDOW: Final = 40.0
"""How many time constants of the cell's drive a transient waits for an output to
settle, before it doubles its window."""

_ATTEMPTS: Final = 8
"""How many times a transient doubles its window before the output is taken never to
settle."""

_STEPS: Final = 20
"""The fewest time steps a transient takes across the faster of an input's edge and the
output's time constant."""


class Arc(BaseModel):
    """One timing arc: how an output follows one input changing from one combination.

    Its tables span the grid of input transitions and output loads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: str
    before: str = Field(description="Every input's level before the change.")
    output: str
    delay: tuple[tuple[float, ...], ...] = Field(
        description="The delay, in seconds: a row per transition, a column per load."
    )
    transition: tuple[tuple[float, ...], ...] = Field(
        description="The output's transition, in seconds, laid out as the delay."
    )


class Characterization(BaseModel):
    """A component characterized as a logic cell."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component: str
    supply: float = Field(description="The supply it was characterized at, in volts.")
    temperature: float = Field(description="The temperature, in kelvins.")
    inputs: tuple[str, ...]
    tables: dict[str, str] = Field(description="Each output's truth table.")
    supply_port: str
    ground_port: str
    input_capacitances: tuple[float, ...] = Field(description="In farads.")
    slews: tuple[float, ...] = Field(description="The transitions, in seconds.")
    loads: tuple[float, ...] = Field(description="The loads, in farads.")
    arcs: tuple[Arc, ...]
    delay_error: float = Field(
        description="The largest error of an interpolated delay, measured between the "
        "grid's points, in seconds."
    )
    transition_error: float = Field(
        description="The largest error of an interpolated transition, in seconds."
    )

    def device(self) -> devices.Device:
        """The logic cell, whose terminals are the component's ports."""
        parameters: dict[str, JsonValue] = {
            "inputs": list(self.inputs),
            "outputs": [
                {"name": name, "table": table} for name, table in self.tables.items()
            ],
            "timing": {
                "characterized": {
                    "slews": list(self.slews),
                    "loads": list(self.loads),
                    "arcs": [arc.model_dump(mode="json") for arc in self.arcs],
                    "supply": self.supply,
                }
            },
            "electrical": {
                "input_capacitances": list(self.input_capacitances),
                "supply": self.supply_port,
                "ground": self.ground_port,
            },
        }
        return devices.device("logic_cell", parameters)

    def mean_delay(self) -> float:
        """The mean of every arc's delays, in seconds."""
        delays = [value for arc in self.arcs for row in arc.delay for value in row]
        return float(np.mean(delays)) if delays else 0.0


@dataclass(frozen=True, slots=True)
class _Cell:
    """A component on a test bench, its supply and its inputs held by sources.

    Each output is loaded by a capacitor, or held by a source.
    """

    built: ir.Circuit
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    supply_port: str
    ground_port: str
    supply: float
    temperature: Quantity | None

    def simulate(
        self,
        drives: Sequence[JsonValue],
        loads: Sequence[float] = (),
        held: Mapping[int, float] | None = None,
    ) -> Simulation:
        """The bench elaborated, its inputs at `drives`, its outputs loaded or held.

        Each input follows its waveform of `drives`, and each output is loaded by its
        capacitance of `loads`, none unless given, or held at its voltage of `held`.

        Raises:
            CharacterizationError: When the bench does not elaborate.
        """
        held = dict[int, float]() if held is None else held

        def instance(
            id: str, of: ir.DeviceOf | ir.ComponentOf, terminals: dict[str, str]
        ) -> ir.Instance:
            return ir.Instance(name=id, of=of, terminals=terminals)

        def source(id: str, waveform: JsonValue, net: str) -> ir.Instance:
            return ir.Instance(
                name=id,
                of=ir.DeviceOf(device="voltage_source"),
                parameters={"waveform": waveform},
                terminals={"positive": net, "negative": "ground"},
            )

        ports = {self.supply_port: "supply", self.ground_port: "ground"}
        ports |= {name: f"input_{index}" for index, name in enumerate(self.inputs)}
        ports |= {name: f"output_{index}" for index, name in enumerate(self.outputs)}
        instances = [
            source("VDD", {"dc": self.supply}, "supply"),
            instance("X", ir.ComponentOf(component=self.name), ports),
            *(
                source(f"VI{index}", drive, f"input_{index}")
                for index, drive in enumerate(drives)
            ),
        ]
        for index in range(len(self.outputs)):
            if index in held:
                instances.append(
                    source(f"VO{index}", {"dc": held[index]}, f"output_{index}")
                )
                continue
            instances.append(
                ir.Instance(
                    name=f"CL{index}",
                    of=ir.DeviceOf(device="capacitor"),
                    parameters={"capacitance": loads[index] if loads else 0.0},
                    terminals={"a": f"output_{index}", "b": "ground"},
                )
            )
        top = "bench"
        while top in self.built.components:
            top += "_"
        bench = ir.Component(
            ports=(ir.Port(name="ground", role=ir.Role.GROUND),),
            instances=tuple(instances),
        )
        components = {
            name: self.built.components[name]
            for name in _subtree(self.built, self.name)
        }
        circuit = ir.Circuit(top=top, components={**components, top: bench})
        try:
            return Simulation(circuit, self.temperature)
        except JoltyError as error:
            message = f"the test bench of {self.name} does not elaborate:\n{error}"
            raise CharacterizationError(message) from None

    def levels(self, row: int) -> tuple[float, ...]:
        """Each input's voltage at `row`, the first input its most significant bit."""
        count = len(self.inputs)
        return tuple(
            self.supply * (row >> (count - 1 - index) & 1) for index in range(count)
        )


def _subtree(built: ir.Circuit, name: str) -> list[str]:
    """`name` and every component it places, however deep."""
    found: list[str] = []
    pending = [name]
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.append(current)
        pending.extend(
            instance.of.component
            for instance in built.components[current].instances
            if isinstance(instance.of, ir.ComponentOf)
        )
    return found


def _cell(
    built: ir.Circuit, name: str, settings: Characterized, temperature: Quantity | None
) -> _Cell:
    """The component `name` of `built` on its test bench.

    Raises:
        CharacterizationError: When its ports are not inputs, outputs, one supply and
            one ground.
    """
    ports = built.components[name].ports
    by_role = {
        role: tuple(port.name for port in ports if port.role is role)
        for role in ir.Role
    }
    signals = by_role[ir.Role.SIGNAL]
    rails = (len(by_role[ir.Role.SUPPLY]), len(by_role[ir.Role.GROUND]))
    if signals or rails != (1, 1) or not by_role[ir.Role.OUTPUT]:
        message = (
            f"{name} cannot be characterized as a logic cell: its ports are not "
            "inputs, outputs, one supply and one ground"
        )
        raise CharacterizationError(message)
    return _Cell(
        built=built,
        name=name,
        inputs=by_role[ir.Role.INPUT],
        outputs=by_role[ir.Role.OUTPUT],
        supply_port=by_role[ir.Role.SUPPLY][0],
        ground_port=by_role[ir.Role.GROUND][0],
        supply=settings.supply.to(VOLTAGE, "a supply"),
        temperature=temperature,
    )


def _dc(level: float) -> JsonValue:
    return {"dc": level}


def _tables(cell: _Cell) -> list[list[bool]]:
    """Each output's level at each row of the inputs.

    Raises:
        CharacterizationError: When an output settles away from both rails.
    """
    tables: list[list[bool]] = [[] for _ in cell.outputs]
    for row in range(1 << len(cell.inputs)):
        point = cell.simulate([_dc(level) for level in cell.levels(row)]).run(
            OperatingPoint()
        )
        for index, table in enumerate(tables):
            voltage = point.voltage(f"output_{index}")
            fraction = voltage / cell.supply
            if fraction <= LEVEL_MARGIN:
                table.append(False)
            elif fraction >= 1 - LEVEL_MARGIN:
                table.append(True)
            else:
                written = format(row, f"0{len(cell.inputs)}b") if cell.inputs else "()"
                message = (
                    f"{cell.name}'s output {cell.outputs[index]} settles at "
                    f"{format_si(voltage, 'V')} with its inputs at {written}, "
                    "neither a 0 nor a 1"
                )
                raise CharacterizationError(message)
    return tables


def _strength(cell: _Cell, tables: list[list[bool]]) -> float:
    """The weakest current an output drives into the opposite rail, in amperes.

    Raises:
        CharacterizationError: When an output drives no current.
    """
    weakest = math.inf
    for row in range(1 << len(cell.inputs)):
        drives = [_dc(level) for level in cell.levels(row)]
        for index, table in enumerate(tables):
            opposite = 0.0 if table[row] else cell.supply
            point = cell.simulate(drives, held={index: opposite}).run(OperatingPoint())
            weakest = min(weakest, abs(point.current(f"VO{index}")))
    if not weakest > 0:
        message = f"an output of {cell.name} drives no current"
        raise CharacterizationError(message)
    return weakest


def _capacitances(cell: _Cell) -> tuple[float, ...]:
    """Each input's capacitance, in farads.

    It is the input's mean small-signal capacitance over every row of the inputs.
    """
    rows = 1 << len(cell.inputs)
    sums = [0.0] * len(cell.inputs)
    frequency = Quantity(_PROBE_FREQUENCY, FREQUENCY)
    for row in range(rows):
        simulation = cell.simulate([_dc(level) for level in cell.levels(row)])
        for index in range(len(cell.inputs)):
            response = simulation.run(Ac(f"VI{index}", [frequency]))
            current = response.current(f"VI{index}")[0]
            sums[index] += abs(current.imag) / (2 * math.pi * _PROBE_FREQUENCY)
    return tuple(total / rows for total in sums)


@dataclass(frozen=True, slots=True)
class _Edge:
    """An output's change: its delay and its transition.

    The delay runs from the input's crossing of half the supply to the output's, and
    the transition is the time an edge at the output's rate there takes to swing the
    supply.
    """

    delay: float
    transition: float


@dataclass(frozen=True, slots=True)
class _Timer:
    """Measures a cell's arcs, by its drive's resistance and its own capacitance."""

    cell: _Cell
    tables: list[list[bool]]
    resistance: float
    capacitance: float

    def edges(self, row: int, pin: int, slew: float, load: float) -> dict[int, _Edge]:
        """How each output that changes follows input `pin` changing from `row`.

        The input's edge takes `slew`, and every output is loaded by `load`.

        Raises:
            CharacterizationError: When an output never settles at its new level.
        """
        cell = self.cell
        after = row ^ 1 << (len(cell.inputs) - 1 - pin)
        changing = [
            index
            for index, table in enumerate(self.tables)
            if table[row] != table[after]
        ]
        levels = cell.levels(row)
        constant = self.resistance * (load + self.capacitance)
        window = slew + _WINDOW * constant
        for _ in range(_ATTEMPTS):
            drives = [_dc(level) for level in levels]
            drives[pin] = {
                "pulse": {
                    "initial": levels[pin],
                    "pulsed": cell.supply - levels[pin],
                    "rise": slew,
                    "fall": slew,
                    "width": 2 * window,
                }
            }
            step = Quantity(min(slew, constant) / _STEPS, TIME)
            waveforms = cell.simulate(drives, [load] * len(cell.outputs)).run(
                Transient(stop=Quantity(window, TIME), max_step=step)
            )
            edges = {
                index: _edge(
                    waveforms,
                    index,
                    rising=self.tables[index][after],
                    supply=cell.supply,
                )
                for index in changing
            }
            found = {index: edge for index, edge in edges.items() if edge is not None}
            if len(found) == len(changing):
                return {
                    index: _Edge(edge.delay - slew / 2, edge.transition)
                    for index, edge in found.items()
                }
            window *= 2
        message = (
            f"an output of {cell.name} never settles after input "
            f"{cell.inputs[pin]} changes"
        )
        raise CharacterizationError(message)


def _edge(
    waveforms: results.TransientResult, index: int, *, rising: bool, supply: float
) -> _Edge | None:
    """Output `index`'s last crossing of half the supply toward its new level.

    It is the time and the transition, or None when the output does not settle there.
    """
    voltage = waveforms.voltage(f"output_{index}")
    final = voltage[-1] / supply
    settled = final >= 1 - LEVEL_MARGIN if rising else final <= LEVEL_MARGIN
    above = voltage > supply / 2
    crossings = np.flatnonzero(above[1:] != above[:-1])
    toward = [k for k in crossings if above[k + 1] == rising]
    if not settled or not toward:
        return None
    k = toward[-1]
    times = waveforms.times
    rate = (voltage[k + 1] - voltage[k]) / (times[k + 1] - times[k])
    time = times[k] + (supply / 2 - voltage[k]) / rate
    return _Edge(float(time), float(supply / abs(rate)))


def characterize(
    built: ir.Circuit, name: str, settings: Characterized, temperature: Quantity | None
) -> Characterization:
    """The component `name` of `built` characterized as a logic cell.

    It is characterized with `settings`, its devices at `temperature`, 27 °C unless
    given.

    Raises:
        CharacterizationError: When its ports are not inputs, outputs, one supply and
            one ground, an output settles away from both rails, or its inputs have no
            capacitance to scale its loads.
    """
    cell = _cell(built, name, settings, temperature)
    tables = _tables(cell)
    capacitances = _capacitances(cell)
    capacitance = float(np.mean(capacitances)) if capacitances else 0.0
    if not capacitance > 0:
        message = (
            f"the inputs of {name} have no capacitance, which the loads of its "
            "characterization are fanouts of"
        )
        raise CharacterizationError(message)
    timer = _Timer(cell, tables, cell.supply / _strength(cell, tables), capacitance)
    arcs = [
        (pin, row)
        for pin in range(len(cell.inputs))
        for row in range(1 << len(cell.inputs))
        if any(
            table[row] != table[row ^ 1 << (len(cell.inputs) - 1 - pin)]
            for table in tables
        )
    ]
    probe = timer.resistance * capacitance
    own = [
        edge.transition
        for pin, row in arcs
        for edge in timer.edges(row, pin, probe, capacitance).values()
    ]
    reference = float(np.mean(own)) if own else probe
    slews = tuple(factor * reference for factor in settings.slews)
    loads = tuple(fanout * capacitance for fanout in settings.fanouts)
    measured = [_measure(timer, change, slews, loads) for change in arcs]
    delay_error, transition_error = _errors(timer, arcs, slews, loads, measured)
    return Characterization(
        component=name,
        supply=cell.supply,
        temperature=kelvins(temperature),
        inputs=cell.inputs,
        tables={
            output: "".join("1" if level else "0" for level in table)
            for output, table in zip(cell.outputs, tables, strict=True)
        },
        supply_port=cell.supply_port,
        ground_port=cell.ground_port,
        input_capacitances=capacitances,
        slews=slews,
        loads=loads,
        arcs=tuple(arc for arcs_of in measured for arc in arcs_of),
        delay_error=delay_error,
        transition_error=transition_error,
    )


def _measure(
    timer: _Timer,
    change: tuple[int, int],
    slews: Sequence[float],
    loads: Sequence[float],
) -> list[Arc]:
    """The arcs of `change`, an input and the row it changes from.

    There is an arc for each output that changes.
    """
    pin, row = change
    grid = [[timer.edges(row, pin, slew, load) for load in loads] for slew in slews]
    cell = timer.cell
    before = format(row, f"0{len(cell.inputs)}b")
    return [
        Arc(
            input=cell.inputs[pin],
            before=before,
            output=cell.outputs[index],
            delay=tuple(tuple(edges[index].delay for edges in line) for line in grid),
            transition=tuple(
                tuple(edges[index].transition for edges in line) for line in grid
            ),
        )
        for index in grid[0][0]
    ]


def _errors(
    timer: _Timer,
    arcs: Sequence[tuple[int, int]],
    slews: Sequence[float],
    loads: Sequence[float],
    measured: Sequence[Sequence[Arc]],
) -> tuple[float, float]:
    """The largest errors of the interpolated delay and transition.

    They are measured between the grid's points: at the midpoints of its transitions
    and of its loads.
    """
    middles = [
        ((a + b) / 2, (c + d) / 2)
        for (a, b), (c, d) in itertools.product(
            itertools.pairwise(slews), itertools.pairwise(loads)
        )
    ]
    delay_error = transition_error = 0.0
    for (pin, row), arcs_of in zip(arcs, measured, strict=True):
        by_output = {timer.cell.outputs.index(arc.output): arc for arc in arcs_of}
        for slew, load in middles:
            for index, edge in timer.edges(row, pin, slew, load).items():
                arc = by_output[index]
                delay = _interpolate(slews, loads, arc.delay, slew, load)
                transition = _interpolate(slews, loads, arc.transition, slew, load)
                delay_error = max(delay_error, abs(edge.delay - delay))
                transition_error = max(
                    transition_error, abs(edge.transition - transition)
                )
    return delay_error, transition_error


def _interpolate(
    rows: Sequence[float],
    columns: Sequence[float],
    table: Sequence[Sequence[float]],
    row: float,
    column: float,
) -> float:
    """`table` at `(row, column)`, bilinear between its points as the engine's."""
    by_row = [float(np.interp(column, columns, line)) for line in table]
    return float(np.interp(row, rows, by_row))


def instances(built: ir.Circuit) -> list[tuple[str, str]]:
    """Every component instance of `built`, outer ones first: its path and component."""
    found: list[tuple[str, str]] = []
    pending = [("", built.top)]
    while pending:
        prefix, name = pending.pop(0)
        for instance in built.components[name].instances:
            if isinstance(instance.of, ir.ComponentOf):
                path = f"{prefix}{instance.name}"
                found.append((path, instance.of.component))
                pending.append((f"{path}/", instance.of.component))
    return found


def scopes(
    built: ir.Circuit, fidelity: Mapping[str, Characterized]
) -> dict[str, tuple[str, str]]:
    """The component instances `fidelity`'s scopes match, the outermost only.

    Each path maps to its scope and its component.

    Raises:
        JoltyError: When a scope matches no component instance.
    """
    matched: dict[str, tuple[str, str]] = {}
    found = instances(built)
    for scope in fidelity:
        hits = [
            (path, component)
            for path, component in found
            if PurePosixPath(path).full_match(scope)
        ]
        if not hits:
            message = f"the fidelity scope {scope!r} matches no component instance"
            raise JoltyError(message)
        for path, component in hits:
            matched.setdefault(path, (scope, component))
    return {
        path: chosen
        for path, chosen in matched.items()
        if not any(path.startswith(f"{other}/") for other in matched)
    }


type Cache = dict[tuple[str, Characterized, float], Characterization]
"""Characterizations already made: by component, settings and temperature in kelvins."""


@dataclass(frozen=True, slots=True)
class Fidelity:
    """The fidelity of one circuit at one temperature.

    It holds each instance a scope matches, and the characterization of its component
    that stands in for it. Its simulations, as many as a Monte Carlo analysis runs,
    share the circuit elaborated once with the models standing in, as
    [`Elaborated`][jolty.simulate.Elaborated] describes.
    """

    built: ir.Circuit
    temperature: Quantity | None
    matched: dict[str, tuple[str, str]]
    """Each instance's scope and component, by path."""
    models: dict[str, Characterization]
    """Each component's characterization, by name."""
    process: str | None = None
    """The process corner of a foundry's devices, its kit's typical unless given."""
    elaborated: dict[str | None, Elaborated] = field(
        default_factory=dict, compare=False, repr=False
    )
    """The circuit elaborated with the models, by the instance left at its circuit's own
    fidelity, if any, once a simulation first asks for it."""

    def substitutes(self, opened: str | None = None) -> dict[str, devices.Device]:
        """The models that stand in for the instances, but for `opened`."""
        return {
            path: self.models[component].device()
            for path, (_, component) in self.matched.items()
            if path != opened
        }

    def overrides(
        self, sampled: Mapping[str, Mapping[str, float]]
    ) -> dict[str, Mapping[str, float]]:
        """The parameters of `sampled` that a model does not stand in for."""
        return {
            path: parameters
            for path, parameters in sampled.items()
            if not any(path.startswith(f"{instance}/") for instance in self.matched)
        }

    def simulation(
        self,
        overrides: Mapping[str, Mapping[str, float]] | None = None,
        opened: str | None = None,
        mismatch: int | None = None,
    ) -> Simulation:
        """The circuit elaborated with the models, but for `opened`, and `overrides`.

        The circuit is elaborated once for every simulation with the same `opened`, each
        of which makes its devices again from the numbers `overrides` gives. A foundry's
        devices draw their kit's mismatch from the seed `mismatch`.

        Raises:
            ElaborationError: When the circuit does not elaborate with the models, or
                with the numbers `overrides` gives.
        """
        shared = self.elaborated.get(opened)
        if shared is None:
            elaborated = Elaborated(
                self.built, self.temperature, self.substitutes(opened)
            )
            shared = self.elaborated.setdefault(opened, elaborated)
        return Simulation.of(
            shared,
            overrides and self.overrides(overrides),
            process=self.process,
            mismatch=mismatch,
        )


def resolve(
    built: ir.Circuit,
    plan: Plan,
    temperature: Quantity | None,
    cache: Cache,
    process: str | None = None,
) -> Fidelity:
    """The fidelity `plan` chooses for `built` at `temperature` and `process`.

    Each component its scopes match is characterized, unless `cache` holds it.

    Raises:
        JoltyError: When a scope matches no component instance.
        CharacterizationError: When a component cannot be characterized.
    """
    matched = scopes(built, plan.fidelity)
    models: dict[str, Characterization] = {}
    for scope, component in matched.values():
        settings = plan.fidelity[scope]
        key = (component, settings, kelvins(temperature))
        if key not in cache:
            cache[key] = characterize(built, component, settings, temperature)
        models[component] = cache[key]
    return Fidelity(built, temperature, matched, models, process)


class Model(BaseModel):
    """A component characterized as a logic cell, as a report states it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component: str
    instances: tuple[str, ...] = Field(description="The paths it stands in for.")
    arcs: int = Field(description="Its timing arcs.")
    slews: tuple[float, float] = Field(
        description="The least and the most input transition, in seconds."
    )
    loads: tuple[float, float] = Field(
        description="The least and the most load, in farads."
    )
    delay_error: float = Field(description="In seconds.")
    transition_error: float = Field(description="In seconds.")

    def describe(self) -> str:
        """The model as text."""
        listed = ", ".join(self.instances)
        return (
            f"{self.component}: a logic cell of {self.arcs} arcs for {listed}, over "
            f"transitions from {format_si(self.slews[0], 's')} to "
            f"{format_si(self.slews[1], 's')} and loads from "
            f"{format_si(self.loads[0], 'F')} to {format_si(self.loads[1], 'F')}; "
            f"interpolated within {format_si(self.delay_error, 's')} of its delays "
            f"and {format_si(self.transition_error, 's')} of its transitions"
        )


class Contract(BaseModel):
    """A spot check: whether an instance's model agrees with its circuit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instance: str
    component: str
    allowed: float = Field(
        description="How far apart, in seconds, the model's and the circuit's changes "
        "may lie."
    )
    divergences: tuple[str, ...] = Field(
        description="Where the model and the circuit diverge; empty when they agree."
    )

    @property
    def held(self) -> bool:
        """Whether the model agrees with its circuit."""
        return not self.divergences

    def describe(self) -> str:
        """The spot check as text."""
        if self.held:
            return (
                f"{self.instance}: its model agrees with its circuit within "
                f"{format_si(self.allowed, 's')}"
            )
        lines = [f"error[contract-violated]: {self.instance}'s model diverges"]
        lines.extend(f"  {divergence}" for divergence in self.divergences)
        return "\n".join(lines)


_UNITS: Final = {"input transition": "s", "load": "F", "supply": "V"}


class Outside(BaseModel):
    """An analysis that used a model outside its domain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis: str
    excursion: Excursion

    def describe(self) -> str:
        """The fact as text, at the rule outside-validity's level."""
        found = self.excursion
        unit = _UNITS[found.bound]
        times = "once" if found.count == 1 else f"{found.count} times"
        return (
            f"warn[outside-validity]: {found.instance}'s {found.bound} reads "
            f"{format_si(found.value, unit)} in {self.analysis}, outside the "
            f"{format_si(found.low, unit)} to {format_si(found.high, unit)} its "
            f"model was characterized over; first at {format_si(found.time, 's')}, "
            f"{times}"
        )


class Dependence(BaseModel):
    """A requirement whose verdict changes when an instance runs as its circuit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instance: str
    modeled: Verdict = Field(description="The verdict with every model.")
    exact: Verdict = Field(description="The verdict with the instance's circuit.")

    def describe(self) -> str:
        """The fact as text, at the rule fidelity-dependence's level."""
        return (
            f"warn[fidelity-dependence]: {self.modeled.requirement}: "
            f"{self.modeled.status}{_reading(self.modeled)} with every model, "
            f"{self.exact.status}{_reading(self.exact)} with {self.instance} at its "
            "circuit's fidelity"
        )


def _reading(verdict: Verdict) -> str:
    """What a verdict read, as a clause, or nothing when it read nothing."""
    if verdict.value is None or verdict.unit is None:
        return ""
    return f", reading {format_si(verdict.value, verdict.unit)},"


def dependences(
    instance: str, modeled: Sequence[Verdict], exact: Sequence[Verdict]
) -> list[Dependence]:
    """The requirements whose verdict differs between `modeled` and `exact`.

    `modeled` holds the verdicts with every model, and `exact` those with `instance` at
    its circuit's fidelity.
    """
    return [
        Dependence(instance=instance, modeled=first, exact=second)
        for first, second in zip(modeled, exact, strict=True)
        if first.status != second.status
    ]


class FidelityReport(BaseModel):
    """The models a plan's fidelity chose, its spot checks, and what they found."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    models: tuple[Model, ...]
    contracts: tuple[Contract, ...]
    dependences: tuple[Dependence, ...] = Field(
        default=(),
        description="The requirements whose verdict changes when a spot-checked "
        "instance runs at its circuit's fidelity.",
    )
    outside: tuple[Outside, ...] = Field(
        default=(),
        description="Where the plan's analyses used a model outside its domain.",
    )

    @property
    def ok(self) -> bool:
        """Whether every model agrees with its circuit where a spot check looked."""
        return all(contract.held for contract in self.contracts)

    def describe(self) -> str:
        """The report as text."""
        lines = ["fidelity:"]
        lines.extend(f"  {model.describe()}" for model in self.models)
        lines.extend(
            "  " + contract.describe().replace("\n", "\n  ")
            for contract in self.contracts
        )
        lines.extend(f"  {found.describe()}" for found in self.dependences)
        lines.extend(f"  {found.describe()}" for found in self.outside)
        return "\n".join(lines)


def models(fidelity: Fidelity) -> tuple[Model, ...]:
    """Each characterized component, as a report states it."""
    found = []
    for component, model in fidelity.models.items():
        paths = tuple(
            path for path, (_, name) in fidelity.matched.items() if name == component
        )
        found.append(
            Model(
                component=component,
                instances=paths,
                arcs=len(model.arcs),
                slews=(model.slews[0], model.slews[-1]),
                loads=(model.loads[0], model.loads[-1]),
                delay_error=model.delay_error,
                transition_error=model.transition_error,
            )
        )
    return tuple(found)


def sampled(fidelity: Fidelity, plan: Plan) -> list[str]:
    """The instances each scope's spot checks run, drawn with its seed."""
    chosen: list[str] = []
    for scope, settings in plan.fidelity.items():
        paths = [
            path for path, (owner, _) in fidelity.matched.items() if owner == scope
        ]
        generator = np.random.default_rng(settings.seed)
        count = min(settings.spot_checks, len(paths))
        chosen.extend(
            paths[index]
            for index in sorted(generator.choice(len(paths), count, replace=False))
        )
    return chosen


def contract(
    fidelity: Fidelity,
    plan: Plan,
    path: str,
    modeled: Mapping[str, AnalysisResult],
    opened: Mapping[str, AnalysisResult],
) -> Contract:
    """The spot check of the instance at `path`.

    It compares the instance's outputs in `opened`, the plan's results with it at its
    circuit's fidelity, with those in `modeled`.

    Over a transient, each output's changes must reach the same levels in the same
    order, each within the model's measured error, plus the scope's tolerance of its
    mean delay; at an operating point and over a sweep, each output must hold the same
    levels.
    """
    scope, component = fidelity.matched[path]
    model = fidelity.models[component]
    allowed = model.delay_error + plan.fidelity[scope].tolerance * model.mean_delay()
    divergences: list[str] = []
    for port, net in _outputs(fidelity.built, path):
        for name, result in modeled.items():
            what = f"{name}: output {port}"
            if name not in opened:
                divergences.append(f"{name} fails with the circuit of {path}")
                continue
            match result, opened[name]:
                case results.TransientResult(), results.TransientResult() as other:
                    found = _diverges(
                        what,
                        _changes(result, net, model.supply),
                        _changes(other, net, model.supply),
                        allowed,
                    )
                case (
                    (results.OperatingPointResult() | results.DcSweepResult()),
                    other,
                ):
                    found = _differs(
                        what,
                        _levels(result, net, model.supply),
                        _levels(other, net, model.supply),
                    )
                case _:
                    found = []
            divergences.extend(found)
    return Contract(
        instance=path,
        component=component,
        allowed=allowed,
        divergences=tuple(divergences),
    )


def _outputs(built: ir.Circuit, path: str) -> list[tuple[str, str]]:
    """Each output port of the instance at `path`, and the path of its net.

    The net's path is the engine's: in the outermost component it reaches.
    """
    *parents, own = path.split("/")
    names = [built.top]
    for id in parents:
        names.append(_component(_placed(built, names[-1], id)))
    instance = _placed(built, names[-1], own)
    component = built.components[_component(instance)]
    found = []
    for port in component.ports:
        if port.role is not ir.Role.OUTPUT:
            continue
        net, level = instance.terminals[port.name], len(parents)
        while level > 0:
            ports = {each.name for each in built.components[names[level]].ports}
            if net not in ports:
                break
            outer = _placed(built, names[level - 1], parents[level - 1])
            net, level = outer.terminals[net], level - 1
        prefix = "".join(f"{id}/" for id in parents[:level])
        found.append((port.name, f"{prefix}{net}"))
    return found


def _component(instance: ir.Instance) -> str:
    """The component `instance` is an instance of.

    Raises:
        JoltyError: When it is a device.
    """
    if isinstance(instance.of, ir.ComponentOf):
        return instance.of.component
    message = f"{instance.name} is a device, not a component instance"
    raise JoltyError(message)


def _placed(built: ir.Circuit, component: str, id: str) -> ir.Instance:
    """The instance `id` that `component` places."""
    return next(
        instance
        for instance in built.components[component].instances
        if instance.name == id
    )


def _changes(
    result: results.TransientResult, net: str, supply: float
) -> list[tuple[float, float]]:
    """`net`'s level at time zero, then each change: its time and the new level.

    A logic net's levels are its own; an electrical net's level is 1 above half the
    supply.
    """
    if net in result.nets:
        trace = result.level(net)
        return list(zip(trace.times.tolist(), trace.levels.tolist(), strict=True))
    voltage, times = result.voltage(net), result.times
    above = voltage > supply / 2
    changes = [(0.0, float(above[0]))]
    for k in np.flatnonzero(above[1:] != above[:-1]):
        fraction = (supply / 2 - voltage[k]) / (voltage[k + 1] - voltage[k])
        time = times[k] + (times[k + 1] - times[k]) * fraction
        changes.append((float(time), float(above[k + 1])))
    return changes


def _levels(result: AnalysisResult, net: str, supply: float) -> list[float]:
    """`net`'s level at an operating point, or at each level of a sweep."""
    match result:
        case results.OperatingPointResult() | results.DcSweepResult() if (
            net in result.nets
        ):
            return [float(level) for level in np.atleast_1d(result.level(net))]
        case results.OperatingPointResult() | results.DcSweepResult():
            above = np.atleast_1d(result.voltage(net)) > supply / 2
            return [float(level) for level in above]
    return []


def _written(level: float) -> str:
    return {0.0: "0", 1.0: "1"}.get(level, "x")


def _differs(
    what: str, modeled: Sequence[float], circuit: Sequence[float]
) -> list[str]:
    """Where `modeled`'s levels and `circuit`'s differ, at the first such point."""
    for index, (model, own) in enumerate(zip(modeled, circuit, strict=True)):
        if _written(model) != _written(own):
            at = "" if len(modeled) == 1 else f" at level {index + 1} of the sweep"
            return [
                (
                    f"{what} is {_written(own)} in its circuit and {_written(model)} "
                    f"in its model{at}"
                )
            ]
    return []


def _diverges(
    what: str,
    modeled: Sequence[tuple[float, float]],
    circuit: Sequence[tuple[float, float]],
    allowed: float,
) -> Iterable[str]:
    """Where `modeled`'s changes and `circuit`'s first differ.

    They differ in their levels, or in their times by more than `allowed`.
    """
    for (model_time, model_level), (circuit_time, circuit_level) in zip(
        modeled, circuit, strict=False
    ):
        level = _written(model_level)
        if level != _written(circuit_level):
            at = format_si(min(model_time, circuit_time), "s")
            yield (
                f"{what} is {_written(circuit_level)} in its circuit and {level} in "
                f"its model at {at}"
            )
            return
        if abs(model_time - circuit_time) > allowed:
            yield (
                f"{what} changes to {level} at {format_si(circuit_time, 's')} in its "
                f"circuit and at {format_si(model_time, 's')} in its model"
            )
            return
    if len(modeled) != len(circuit):
        yield (
            f"{what} changes {len(circuit) - 1} times in its circuit and "
            f"{len(modeled) - 1} in its model"
        )

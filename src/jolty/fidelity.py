# Copyright 2026 Maxence Faldor
"""Logic cell characterization from component circuits and contract verification.

Plan fidelity maps instance-path patterns (scopes) to
[`Characterized`][jolty.plan.Characterized] settings. Each matched component instance
is simulated as a logic cell characterized from its underlying circuit, mirroring
standard-cell library characterization:

- Truth tables come from DC operating points across all input combinations.
- Input capacitances come from small-signal AC responses averaged across states.
- Delays and transitions come from transient sweeps over input slews and output loads.

Characterization grid scales are derived per cell. Loads are fanout multiples of the
cell's mean input capacitance, and input transitions are multiples of the mean fanout-1
output transition time. A secondary midpoint grid measures bilinear interpolation error
between primary grid points.

Spot checks validate each characterized model by re-running the plan with one sampled
instance restored to its circuit-level implementation and comparing output transitions.
Any transition differing by more than the measured interpolation error plus the scope's
relative delay tolerance violates the contract and is reported with details.
"""

import itertools
import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Final

import numpy as np
from pydantic import Field, JsonValue

from jolty import devices, ir, results
from jolty.devices import ParameterValue
from jolty.errors import CharacterizationError, JoltyError, PlanError, close_names
from jolty.plan import Ac, Characterized, OperatingPoint, Plan, Transient
from jolty.results import AnalysisResult, Excursion
from jolty.simulate import Elaborated, Simulation, kelvins
from jolty.specs import Verdict
from jolty.units import FREQUENCY, TIME, VOLTAGE, Quantity, format_si

LEVEL_MARGIN: Final = 0.1
"""Maximum normalized distance from a supply rail that still qualifies as a valid logic
level."""

_PROBE_FREQUENCY: Final = 1.0
"""Small-signal AC probe frequency in hertz used to measure purely capacitive gate
input impedance."""

_WINDOW: Final = 40.0
"""Initial transient simulation window expressed as multiples of the cell drive time
constant."""

_ATTEMPTS: Final = 8
"""Maximum number of window doublings attempted before declaring an output
unsettled."""

_STEPS: Final = 20
"""Minimum number of transient time steps across the faster of the input slew and
output time constant."""


class Arc(ir.FrozenModel):
    """Single timing arc capturing an output response to one input transition.

    Delay and transition tables span the 2D grid of input slews and capacitive output
    loads.
    """

    input: str
    before: str = Field(description="Logic levels of all inputs before the transition.")
    output: str
    delay: tuple[tuple[float, ...], ...] = Field(
        description="Propagation delay in seconds, indexed by input slew row and load "
        "column."
    )
    transition: tuple[tuple[float, ...], ...] = Field(
        description="Output transition time in seconds, indexed like delay."
    )


class Characterization(ir.FrozenModel):
    """Logic cell model characterized from a component's circuit implementation."""

    component: str
    supply: float = Field(
        description="Supply voltage used for characterization, in volts."
    )
    temperature: float = Field(description="Characterization temperature in kelvins.")
    inputs: tuple[str, ...]
    tables: dict[str, str] = Field(
        description="Mapping from output name to truth table."
    )
    supply_port: str
    ground_port: str
    input_capacitances: tuple[float, ...] = Field(
        description="Mean small-signal capacitance of each input port, in farads."
    )
    slews: tuple[float, ...] = Field(
        description="Characterization input slew grid, in seconds."
    )
    loads: tuple[float, ...] = Field(
        description="Characterization output load grid, in farads."
    )
    arcs: tuple[Arc, ...]
    delay_error: float = Field(
        description="Maximum interpolated delay error measured at grid midpoints, in "
        "seconds."
    )
    transition_error: float = Field(
        description="Maximum interpolated transition error measured at grid midpoints, "
        "in seconds."
    )

    def device(self) -> devices.Device:
        """Builds a characterized ``logic_cell`` device matching the component ports.

        Returns:
            Characterized electrical `logic_cell` `Device` matching component ports.
        """
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
        """Computes the mean propagation delay across all timing arcs, in seconds.

        Returns:
            Arithmetic mean propagation delay across all arc table entries in seconds.
        """
        delays = [value for arc in self.arcs for row in arc.delay for value in row]
        return float(np.mean(delays)) if delays else 0.0


@dataclass(frozen=True, slots=True)
class _Cell:
    """Component mounted on a characterization test bench with driven supply and inputs.

    Each output is connected either to a capacitive load or to a fixed DC voltage
    source.
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
        """Builds an elaborated bench simulation with driven inputs and output loads.

        Each input follows the corresponding waveform in `drives`. Each output is loaded
        by the capacitance in `loads` (defaulting to zero) unless held at a DC voltage
        by `held`.

        Raises:
            CharacterizationError: If the test-bench circuit fails to elaborate.
        """
        held = dict[int, float]() if held is None else held

        def instance(
            name: str, of: ir.DeviceOf | ir.ComponentOf, terminals: dict[str, str]
        ) -> ir.Instance:
            return ir.Instance(name=name, of=of, terminals=terminals)

        def source(name: str, waveform: JsonValue, net: str) -> ir.Instance:
            return ir.Instance(
                name=name,
                of=ir.DeviceOf(model="voltage_source"),
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
                    of=ir.DeviceOf(model="capacitor"),
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
        """Returns input voltages for truth-table `row`, with the first input as MSB."""
        count = len(self.inputs)
        return tuple(
            self.supply * (row >> (count - 1 - index) & 1) for index in range(count)
        )


def _subtree(built: ir.Circuit, name: str) -> list[str]:
    """Collects `name` and every nested subcomponent regardless of hierarchy depth."""
    found: list[str] = []
    seen: set[str] = set()
    pending: deque[str] = deque([name])
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
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
    """Creates a characterization test bench for component `name` in `built`.

    Raises:
        CharacterizationError: If component ports do not consist solely of inputs, at
            least one output, one supply, and one ground.
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
    """Computes boolean truth-table rows for every output of `cell`.

    Raises:
        CharacterizationError: If any output settles outside the valid logic rail
            margins.
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
    """Computes the minimum short-circuit drive current across all outputs, in amperes.

    Raises:
        CharacterizationError: If any output drives zero current into the opposite rail.
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
    """Measures the mean small-signal input capacitance of each input port, in farads.

    Each value averages the 1 Hz AC capacitive susceptance across every input state.
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
    """Measured output transition recording propagation delay and transition time.

    Delay spans from the input half-supply crossing to the output half-supply crossing,
    and transition time is the full-supply swing duration at the output crossing slope.
    """

    delay: float
    transition: float


@dataclass(frozen=True, slots=True)
class _Timer:
    """Timing arc measurer parameterized by cell drive resistance and capacitance."""

    cell: _Cell
    tables: list[list[bool]]
    resistance: float
    capacitance: float

    def edges(self, row: int, pin: int, slew: float, load: float) -> dict[int, _Edge]:
        """Measures output transitions caused by toggling input `pin` from state `row`.

        The input switches with transition time `slew` while every output drives
        capacitance `load`.

        Raises:
            CharacterizationError: If any changing output fails to settle within the
                maximum transient window.
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
    """Extracts the final half-supply crossing of output `index` toward its target rail.

    Returns the crossing time and transition duration, or ``None`` if the output does
    not settle to the target logic level.
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
    """Characterizes component `name` in `built` as a timing-annotated logic cell.

    Args:
        built: Circuit containing the component definition to characterize.
        name: Name of the component within `built`.
        settings: Supply voltage, slew multipliers, and load fanouts for the grid.
        temperature: Simulation temperature, defaulting to 27 °C when ``None``.

    Returns:
        `Characterization` holding truth tables, capacitances, arcs, and grid errors.

    Raises:
        CharacterizationError: If component ports are incompatible, an output fails to
            settle to a valid rail, or inputs have zero capacitance.
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
    """Measures timing arcs for a `(pin, row)` transition across `slews` and `loads`.

    Returns one [`Arc`][jolty.fidelity.Arc] for each output that changes state.
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
    """Computes maximum bilinear interpolation errors for delay and transition time.

    Errors are evaluated at the midpoint of every adjacent slew and load cell on the
    characterization grid.
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
    """Interpolates `table` bilinearly at `(row, column)` matching the engine lookup."""
    by_row = [float(np.interp(column, columns, line)) for line in table]
    return float(np.interp(row, rows, by_row))


def instances(built: ir.Circuit) -> list[tuple[str, str]]:
    """Collects component instances in `built` breadth-first as `(path, component)`.

    Args:
        built: Circuit IR traversed breadth-first from `built.top`.

    Returns:
        Breadth-first list of `(path, component_name)` pairs for component placements.
    """
    return [
        (path, instance.of.component)
        for path, instance in built.walk(breadth_first=True)
        if isinstance(instance.of, ir.ComponentOf)
    ]


def scopes(
    built: ir.Circuit, fidelity: Mapping[str, Characterized]
) -> dict[str, tuple[str, str]]:
    """Resolves outermost component instances matched by each scope in `fidelity`.

    Returns a mapping from instance path to its matched `(scope, component)` pair.

    Args:
        built: Circuit IR whose component placements are matched.
        fidelity: Mapping from instance-path glob scopes to `Characterized`.

    Returns:
        Dictionary mapping outermost matched paths to `(scope, component_name)`.

    Raises:
        PlanError: If any scope matches no component instance in `built`.
    """
    matched: dict[str, tuple[str, str]] = {}
    found = instances(built)
    paths = [path for path, _ in found]
    problems: list[str] = []
    for scope in fidelity:
        hits = [
            (path, component)
            for path, component in found
            if PurePosixPath(path).full_match(scope)
        ]
        if not hits:
            has_wildcard = any(symbol in scope for symbol in "*?[")
            close = "" if has_wildcard else close_names(paths, scope)
            problems.append(
                f"the fidelity scope {scope!r} matches no component instance{close}"
            )
            continue
        for path, component in hits:
            matched.setdefault(path, (scope, component))
    if problems:
        raise PlanError(problems)
    return {
        path: chosen
        for path, chosen in matched.items()
        if not any(path.startswith(f"{other}/") for other in matched)
    }


type Cache = dict[tuple[str, Characterized, float], Characterization]
"""Cache of logic cell characterizations keyed by `(component, settings, kelvins)`."""


@dataclass(frozen=True, slots=True)
class Fidelity:
    """Resolved fidelity substitutions and cached elaborations for one circuit.

    Stores every scope-matched component instance alongside its characterized logic
    model. Simulations sharing an opened-instance configuration reuse a single
    [`Elaborated`][jolty.simulate.Elaborated] circuit across Monte Carlo runs.
    """

    built: ir.Circuit
    temperature: Quantity | None
    matched: dict[str, tuple[str, str]]
    """Mapping from matched instance path to its `(scope, component)` pair."""
    models: dict[str, Characterization]
    """Mapping from component name to its characterized logic cell model."""
    process: str | None = None
    """Foundry process corner name, defaulting to the kit's typical corner when
    ``None``."""
    elaborated: dict[str | None, Elaborated] = field(
        default_factory=dict, compare=False, repr=False
    )
    """Cache of elaborated circuits keyed by the instance path kept at full fidelity."""

    def substitutes(self, opened: str | None = None) -> dict[str, devices.Device]:
        """Returns logic-cell devices for every matched path except `opened`.

        Args:
            opened: Optional matched instance path kept at full circuit fidelity.

        Returns:
            Dictionary mapping substituted instance paths to characterized `Device`
            cells.
        """
        return {
            path: self.models[component].device()
            for path, (_, component) in self.matched.items()
            if path != opened
        }

    def overrides(
        self, sampled: Mapping[str, Mapping[str, ParameterValue]]
    ) -> dict[str, Mapping[str, ParameterValue]]:
        """Returns parameter overrides from `sampled` outside modeled instances.

        Args:
            sampled: Nested parameter overrides keyed by instance and parameter path.

        Returns:
            Filtered override dictionary excluding paths enclosed by modeled instances.
        """
        return {
            path: parameters
            for path, parameters in sampled.items()
            if not any(path.startswith(f"{instance}/") for instance in self.matched)
        }

    def simulation(
        self,
        overrides: Mapping[str, Mapping[str, ParameterValue]] | None = None,
        opened: str | None = None,
        mismatch: int | None = None,
    ) -> Simulation:
        """Creates a simulation using characterized models except for `opened`.

        Elaboration is cached per `opened` value, while parameter `overrides` rebuild
        device instances for each run. Foundry devices sample process mismatch using
        `mismatch` as the random seed.

        Args:
            overrides: Optional nested parameter overrides keyed by instance and
                parameter path.
            opened: Optional matched instance path restored to transistor level.
            mismatch: Optional 64-bit random seed for foundry mismatch sampling.

        Returns:
            Configured `Simulation` reusing the cached elaboration for `opened`.

        Raises:
            ElaborationError: If the substituted circuit or overridden parameters fail
                to elaborate.
        """
        shared = self.elaborated.get(opened)
        if shared is None:
            elaborated = Elaborated(
                self.built, self.temperature, self.substitutes(opened)
            )
            shared = self.elaborated.setdefault(opened, elaborated)
        return Simulation.from_elaborated(
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
    """Resolves `plan` fidelity settings for `built` at `temperature` and `process`.

    Matched components are characterized on demand and stored in `cache` for reuse.

    Args:
        built: Circuit whose component instances are matched against fidelity scopes.
        plan: Analysis plan defining fidelity scopes and characterization settings.
        temperature: Simulation temperature used during characterization.
        cache: Mutable cache reusing characterizations across runs.
        process: Optional foundry process corner name.

    Returns:
        Resolved `Fidelity` mapping matched paths to cached characterizations.

    Raises:
        JoltyError: If a fidelity scope matches no component instance.
        CharacterizationError: If a matched component cannot be characterized.
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


class CharacterizedModel(ir.FrozenModel):
    """Report summary for a single characterized component model."""

    component: str
    instances: tuple[str, ...] = Field(
        description="Instance paths substituted by this characterized model."
    )
    arcs: int = Field(description="Number of characterized timing arcs.")
    slews: tuple[float, float] = Field(
        description="Minimum and maximum characterized input slew, in seconds."
    )
    loads: tuple[float, float] = Field(
        description="Minimum and maximum characterized output load, in farads."
    )
    delay_error: float = Field(
        description="Maximum interpolated delay error, in seconds."
    )
    transition_error: float = Field(
        description="Maximum interpolated transition error, in seconds."
    )

    def describe(self) -> str:
        """Formats the characterized model summary as human-readable text.

        Returns:
            Human-readable summary of arcs, slew/load domain, and interpolation bounds.
        """
        listed = ", ".join(self.instances)
        return (
            f"{self.component}: a logic cell of {self.arcs} arcs for {listed}, over "
            f"transitions from {format_si(self.slews[0], 's')} to "
            f"{format_si(self.slews[1], 's')} and loads from "
            f"{format_si(self.loads[0], 'F')} to {format_si(self.loads[1], 'F')}; "
            f"interpolated within {format_si(self.delay_error, 's')} of its delays "
            f"and {format_si(self.transition_error, 's')} of its transitions"
        )


class Contract(ir.FrozenModel):
    """Spot-check contract comparing a characterized instance model with its circuit."""

    instance: str
    component: str
    allowed: float = Field(
        description="Maximum allowed timing discrepancy between model and circuit, in "
        "seconds."
    )
    divergences: tuple[str, ...] = Field(
        description="Observed model-versus-circuit mismatches, empty when passing."
    )

    @property
    def held(self) -> bool:
        """Indicator of whether the characterized model agrees with its circuit."""
        return not self.divergences

    def describe(self) -> str:
        """Formats the spot-check contract outcome as human-readable text.

        Returns:
            Pass line or compiler-style ``contract-violated`` error report.
        """
        if self.held:
            return (
                f"{self.instance}: its model agrees with its circuit within "
                f"{format_si(self.allowed, 's')}"
            )
        lines = [f"error[contract-violated]: {self.instance}'s model diverges"]
        lines.extend(f"  {divergence}" for divergence in self.divergences)
        return "\n".join(lines)


_UNITS: Final = {"input transition": "s", "load": "F", "supply": "V"}


class Outside(ir.FrozenModel):
    """Record of an analysis exercising a characterized model outside its domain."""

    analysis: str
    excursion: Excursion

    def describe(self) -> str:
        """Formats the out-of-domain warning for the ``outside-validity`` rule.

        Returns:
            Compiler-style ``outside-validity`` warning line for the domain excursion.
        """
        found = self.excursion
        unit = _UNITS[found.bound]
        times = "once" if found.count == 1 else f"{found.count} times"
        return (
            f"warn[outside-validity]: {found.instance}'s {found.bound} reads "
            f"{format_si(found.value, unit)} in {self.analysis}, outside the "
            f"{format_si(found.minimum, unit)} to {format_si(found.maximum, unit)} its "
            f"model was characterized over; first at {format_si(found.time, 's')}, "
            f"{times}"
        )


class Dependence(ir.FrozenModel):
    """Requirement whose verdict flips when a modeled instance runs at full fidelity."""

    instance: str
    modeled: Verdict = Field(description="Requirement verdict with all models active.")
    exact: Verdict = Field(
        description="Requirement verdict with the instance restored to its circuit."
    )

    def describe(self) -> str:
        """Formats the verdict change warning for the ``fidelity-dependence`` rule.

        Returns:
            Compiler-style ``fidelity-dependence`` warning comparing verdict statuses.
        """
        return (
            f"warn[fidelity-dependence]: {self.modeled.requirement}: "
            f"{self.modeled.status}{_reading(self.modeled)} with every model, "
            f"{self.exact.status}{_reading(self.exact)} with {self.instance} at its "
            "circuit's fidelity"
        )


def _reading(verdict: Verdict) -> str:
    """Formats a verdict's measured value as a clause, or returns an empty string."""
    if verdict.value is None or verdict.unit is None:
        return ""
    return f", reading {format_si(verdict.value, verdict.unit)},"


def dependences(
    instance: str, modeled: Sequence[Verdict], exact: Sequence[Verdict]
) -> list[Dependence]:
    """Collects requirements whose verdict differs between `modeled` and `exact`.

    Compares `modeled` verdicts (with all models active) against `exact` verdicts
    (with `instance` simulated at full circuit fidelity).

    Args:
        instance: Hierarchical path of the spot-checked instance.
        modeled: Requirement verdicts evaluated with all models active.
        exact: Requirement verdicts evaluated with `instance` at circuit level.

    Returns:
        List of `Dependence` records for requirements whose pass/fail status flipped.
    """
    return [
        Dependence(instance=instance, modeled=first, exact=second)
        for first, second in zip(modeled, exact, strict=True)
        if first.status != second.status
    ]


class FidelityReport(ir.FrozenModel):
    """Summary of characterized models, spot-check contracts, and fidelity warnings."""

    models: tuple[CharacterizedModel, ...]
    contracts: tuple[Contract, ...]
    dependences: tuple[Dependence, ...] = Field(
        default=(),
        description="Requirements whose verdict changed when a spot-checked instance "
        "ran at full circuit fidelity.",
    )
    outside: tuple[Outside, ...] = Field(
        default=(),
        description="Analyses that exercised a characterized model outside its "
        "characterization grid.",
    )

    @property
    def ok(self) -> bool:
        """Indicator of whether every spot-checked model agrees with its circuit."""
        return all(contract.held for contract in self.contracts)

    def describe(self) -> str:
        """Formats the complete fidelity report as human-readable text.

        Returns:
            Multi-line ``fidelity:`` report summarizing models, contracts, and warnings.
        """
        lines = ["fidelity:"]
        lines.extend(f"  {model.describe()}" for model in self.models)
        lines.extend(
            "  " + contract.describe().replace("\n", "\n  ")
            for contract in self.contracts
        )
        lines.extend(f"  {found.describe()}" for found in self.dependences)
        lines.extend(f"  {found.describe()}" for found in self.outside)
        return "\n".join(lines)


def models(fidelity: Fidelity) -> tuple[CharacterizedModel, ...]:
    """Builds report summaries for every characterized component in `fidelity`.

    Args:
        fidelity: Resolved `Fidelity` containing matched paths and models.

    Returns:
        Tuple of `CharacterizedModel` report summaries in characterization order.
    """
    found = []
    for component, model in fidelity.models.items():
        paths = tuple(
            path for path, (_, name) in fidelity.matched.items() if name == component
        )
        found.append(
            CharacterizedModel(
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
    """Selects spot-check instance paths deterministically for each scope in `plan`.

    Args:
        fidelity: Resolved `Fidelity` mapping matched paths to scopes.
        plan: Simulation `Plan` defining per-scope `spot_checks` and `seed`.

    Returns:
        Deterministically sampled instance paths selected for contract spot checks.
    """
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
    """Evaluates the spot-check contract for the instance at `path`.

    Compares output behavior in `opened` (with `path` restored to full circuit
    fidelity) against `modeled` results. Transient outputs require matching transition
    levels and order within the measured delay error plus the scope's tolerance of the
    mean delay. Operating-point and DC-sweep outputs require identical logic levels.

    Args:
        fidelity: Resolved fidelity mapping and characterized models.
        plan: Analysis plan supplying scope timing tolerances.
        path: Hierarchical path of the spot-checked component instance.
        modeled: Analysis results run with all characterized models substituted.
        opened: Analysis results run with `path` restored to its circuit.

    Returns:
        `Contract` recording allowed timing slack and any observed divergences.
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
    """Resolves each output port of the instance at `path` to its outermost net path.

    Each returned net path matches the top-most hierarchical component the net reaches.
    """
    *parents, own = path.split("/")
    names = [built.top]
    for name in parents:
        names.append(_component(_placed(built, names[-1], name)))
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
        prefix = "".join(f"{name}/" for name in parents[:level])
        found.append((port.name, f"{prefix}{net}"))
    return found


def _component(instance: ir.Instance) -> str:
    """Returns the component name referenced by `instance`.

    Raises:
        JoltyError: If `instance` is a primitive device rather than a component.
    """
    if isinstance(instance.of, ir.ComponentOf):
        return instance.of.component
    message = f"{instance.name} is a device, not a component instance"
    raise JoltyError(message)


def _placed(built: ir.Circuit, component: str, name: str) -> ir.Instance:
    """Retrieves child instance `name` inside `component` of `built`."""
    found = built.components[component].instance(name)
    if found is None:
        message = f"{component} places no instance `{name}`"
        raise KeyError(message)
    return found


def _changes(
    result: results.TransientResult, net: str, supply: float
) -> list[tuple[float, float]]:
    """Extracts the initial logic level and subsequent transition times for `net`.

    Logic nets use their recorded discrete levels directly, while electrical nets treat
    voltages above half `supply` as logic high.
    """
    if net in result.nets:
        trace = result.logic_level(net)
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
    """Extracts logic levels of `net` at an operating point or across a DC sweep."""
    match result:
        case results.OperatingPointResult() | results.DcSweepResult() if (
            net in result.nets
        ):
            return [float(level) for level in np.atleast_1d(result.logic_level(net))]
        case results.OperatingPointResult() | results.DcSweepResult():
            above = np.atleast_1d(result.voltage(net)) > supply / 2
            return [float(level) for level in above]
    return []


def _differs(
    what: str, modeled: Sequence[float], circuit: Sequence[float]
) -> list[str]:
    """Finds the first logic-level mismatch between `modeled` and `circuit` outputs."""
    for index, (model, own) in enumerate(zip(modeled, circuit, strict=True)):
        if devices.format_logic_level(model) != devices.format_logic_level(own):
            at = "" if len(modeled) == 1 else f" at level {index + 1} of the sweep"
            return [
                (
                    f"{what} is {devices.format_logic_level(own)} in its circuit "
                    f"and {devices.format_logic_level(model)} in its model{at}"
                )
            ]
    return []


def _diverges(
    what: str,
    modeled: Sequence[tuple[float, float]],
    circuit: Sequence[tuple[float, float]],
    allowed: float,
) -> Iterable[str]:
    """Yields the first timing or level divergence between `modeled` and `circuit`.

    Transitions diverge when logic levels disagree, crossing times differ by more than
    `allowed`, or transition counts do not match.
    """
    for (model_time, model_level), (circuit_time, circuit_level) in zip(
        modeled, circuit, strict=False
    ):
        level = devices.format_logic_level(model_level)
        if level != devices.format_logic_level(circuit_level):
            at = format_si(min(model_time, circuit_time), "s")
            yield (
                f"{what} is {devices.format_logic_level(circuit_level)} in its "
                f"circuit and {level} in its model at {at}"
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

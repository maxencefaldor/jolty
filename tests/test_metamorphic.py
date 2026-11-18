# Copyright 2026 Maxence Faldor
"""The same circuit, written differently, simulates the same.

Renaming its nets, reordering its instances, splitting a resistor into two in series
and placing a source of 0 V in series with a terminal, as an ammeter, change how a
circuit is written and not what it is: jolty finds the same operating point, frequency
response and transient for each, on the random circuits of `strategies`, and the
ammeter reads the current into the terminal. Each circuit is its own reference, so
these tests need no other simulator.

Transients are those of the circuits with capacitance at their nets, as real nets have,
which `strategies.parasitic` places: without it, a regenerative loop of devices that
hold no charge can fold into a point past which no solution continues, where a
transient stops, or leaps by chance to another solution, in one spelling and not in
another.
"""

import math
from collections.abc import Callable, Iterable

import numpy as np
from hypothesis import event, given
from hypothesis import strategies as st

from differential import Values, differing, nearest, unstable
from jolty import (
    Ac,
    OperatingPoint,
    Simulation,
    SimulationError,
    Transient,
    devices,
    ir,
    results,
)
from jolty.plan import TIGHT, decade
from jolty.spice.ngspice import CURRENT_TOLERANCE
from jolty.units import GHz, Hz, V, ohm, s
from strategies import (
    SETTINGS,
    STEP,
    STOP,
    capacitive,
    circuits,
    parasitic,
    placing,
    pulsed,
)

TIMES = np.linspace(0, STOP, 201)
"""The times each transient steps onto, and is read at."""

type Simulated = dict[str, dict[str, Values]]
"""What each analysis found, by the name of each unknown."""


def _instances(circuit: ir.Circuit) -> tuple[ir.Instance, ...]:
    return circuit.components[circuit.top].instances


def _source(circuit: ir.Circuit) -> str:
    """The pulsed source, the first voltage source the circuit places."""
    return next(
        instance.name
        for instance in _instances(circuit)
        if instance.of == ir.DeviceOf(model="voltage_source")
    )


def _transient(simulation: Simulation, step: float) -> dict[str, Values]:
    """The waveform of every unknown with `step` the longest, read at `TIMES`."""
    plan = Transient(
        stop=STOP * s, max_step=step * s, breakpoints=[time * s for time in TIMES[1:]]
    )
    waveforms = simulation.run(plan, TIGHT)
    taken = nearest(waveforms.times, TIMES)
    # Breakpoint landing tolerance: 100 * SMALLEST_STEP_FRACTION (1e-11 * step).
    assert np.allclose(waveforms.times[taken], TIMES, rtol=0, atol=1e-9 * step)
    return {
        unknown.name: waveforms.values[taken, column]
        for column, unknown in enumerate(waveforms.unknowns)
    }


def _simulated(circuit: ir.Circuit, loaded: ir.Circuit, source: str) -> Simulated:
    """The operating point, the response to `source` and the transient of `circuit`.

    The transient is that of `loaded`, the same circuit with capacitance at its nets.
    A transient that grows away from an unstable operating point is left out: it
    amplifies rounding, and two ways of writing the circuit round differently.
    """
    simulation = Simulation(circuit)
    point = simulation.run(OperatingPoint(), TIGHT)
    response = simulation.run(Ac(source, decade(1 * Hz, 1 * GHz, 5)), TIGHT)
    found: Simulated = {
        "operating point": dict(
            zip((unknown.name for unknown in point.unknowns), point.values, strict=True)
        ),
        "frequency response": {
            unknown.name: response.values[:, column]
            for column, unknown in enumerate(response.unknowns)
        },
    }
    charged = Simulation(loaded)
    if not unstable(charged):
        found["transient"] = _transient(charged, STEP)
    return found


def _settled(circuit: ir.Circuit, simulated: Simulated) -> bool:
    """Whether the transient jolty found does not move as its steps shrink fourfold."""
    waveforms = simulated["transient"]
    shorter = _transient(Simulation(circuit), STEP / 4)
    return not differing(shorter, waveforms, _quantities(circuit), 1e-3)


def _rounded_up(instance: ir.Instance) -> ir.Instance:
    """The instance, a resistor's resistance the next floating-point number up."""
    resistor = instance.of == ir.DeviceOf(model="resistor")
    resistance = instance.parameters.get("resistance")
    if not resistor or not isinstance(resistance, float):
        return instance
    parameters = {
        **instance.parameters,
        "resistance": math.nextafter(resistance, math.inf),
    }
    return instance.model_copy(update={"parameters": parameters})


def _determined(circuit: ir.Circuit, simulated: Simulated) -> bool:
    """Whether the transient jolty found does not move as each resistance rounds up.

    Where a rounding decides the steps a transient takes, and those steps move the
    waveform by more than its tolerance, two spellings, which round differently, find
    different waveforms however alike they are.
    """
    waveforms = simulated["transient"]
    rounded = placing(circuit, map(_rounded_up, _instances(circuit)))
    moved = _transient(Simulation(rounded), STEP)
    return not differing(moved, waveforms, _quantities(circuit), 1e-3)


def _quantities(circuit: ir.Circuit) -> dict[str, str]:
    return {unknown.name: unknown.nature for unknown in Simulation(circuit).unknowns}


TOLERANCES = {"operating point": 1e-6, "frequency response": 1e-6, "transient": 1e-3}
"""How closely each analysis of two spellings agrees, as a fraction of the largest
value of each quantity: Newton iteration stops within a millionth, and a transient's
steps, rounded differently, land within a thousandth of the swing."""


def _assert_same(
    original: ir.Circuit,
    written: ir.Circuit,
    name: Callable[[str], str] = lambda name: name,
) -> Simulated:
    """Asserts both circuits simulate the same, and returns what `written` found.

    Each unknown of `original` is found in `written` by its `name` there. Their
    transients are those of both with capacitance at each net of `original` and at its
    namesake in `written`, so that a net the writing adds, such as an ammeter's, holds
    none, and both remain one circuit. A transient that jolty's own shorter steps still
    move, as a chaotic one's, has not settled at these steps, and is left out too, as is
    one that differs but moves as each resistance rounds up: how the circuit rounds
    decides it, and two spellings round differently.
    """
    source = _source(original)
    nets = capacitive(original)
    loaded = parasitic(original, nets)
    try:
        expected = _simulated(original, loaded, source)
    except SimulationError:
        event("no simulation of the circuit as drawn")
        return {}
    found = _simulated(written, parasitic(written, map(name, nets)), source)
    if "transient" in expected and not _settled(loaded, expected):
        event("a waveform that shorter steps move")
        del expected["transient"]
    natures = _quantities(original)
    for analysis, values in expected.items():
        assert analysis in found, f"no {analysis} of the circuit as written"
        renamed = {unknown: found[analysis][name(unknown)] for unknown in values}
        differs = differing(renamed, values, natures, TOLERANCES[analysis])
        if differs and analysis == "transient" and not _determined(loaded, expected):
            event("a waveform that a rounding moves")
            continue
        assert not differs, f"the {analysis} differs"
    return found


@SETTINGS
@given(circuits(), st.data())
def test_renamed_nets_simulate_the_same(
    circuit: ir.Circuit, data: st.DataObject
) -> None:
    original = pulsed(circuit)
    nets = sorted(
        {
            net
            for instance in _instances(original)
            for net in instance.terminals.values()
        }
        - {"0"}
    )
    shuffled = data.draw(st.permutations(nets))
    names = {net: f"renamed{index}" for index, net in enumerate(shuffled)}
    written = placing(
        original,
        (
            instance.model_copy(
                update={
                    "terminals": {
                        terminal: names.get(net, net)
                        for terminal, net in instance.terminals.items()
                    }
                }
            )
            for instance in _instances(original)
        ),
    )
    _assert_same(original, written, lambda name: names.get(name, name))


@SETTINGS
@given(circuits(), st.data())
def test_reordered_instances_simulate_the_same(
    circuit: ir.Circuit, data: st.DataObject
) -> None:
    original = pulsed(circuit)
    _assert_same(
        original, placing(original, data.draw(st.permutations(_instances(original))))
    )


@SETTINGS
@given(circuits(), st.data())
def test_a_resistor_split_in_two_simulates_the_same(
    circuit: ir.Circuit, data: st.DataObject
) -> None:
    original = pulsed(circuit)
    instances = list(_instances(original))
    index = data.draw(
        st.sampled_from(
            [
                index
                for index, instance in enumerate(instances)
                if instance.of == ir.DeviceOf(model="resistor")
            ]
        )
    )
    resistor = instances[index]
    resistance = resistor.parameters["resistance"]
    assert isinstance(resistance, float)
    half = devices.resistor(resistance / 2 * ohm)
    first, second = (
        ir.Instance(
            name=f"{resistor.name}{part}",
            of=resistor.of,
            parameters=half.parameters,
            terminals=terminals,
        )
        for part, terminals in (
            ("a", {"a": resistor.terminals["a"], "b": "halfway"}),
            ("b", {"a": "halfway", "b": resistor.terminals["b"]}),
        )
    )
    written = placing(
        original, [*instances[:index], first, second, *instances[index + 1 :]]
    )
    _assert_same(original, written)


@SETTINGS
@given(circuits(), st.data())
def test_an_ammeter_reads_the_current_into_a_terminal(
    circuit: ir.Circuit, data: st.DataObject
) -> None:
    original = pulsed(circuit)
    instances = list(_instances(original))
    index, terminal = data.draw(
        st.sampled_from(
            [
                (index, terminal)
                for index, instance in enumerate(instances)
                for terminal in instance.terminals
            ]
        )
    )
    probed = instances[index]
    net = probed.terminals[terminal]
    ammeter = ir.Instance(
        name="ammeter",
        of=ir.DeviceOf(model="voltage_source"),
        parameters=devices.voltage_source(0 * V).parameters,
        terminals={"positive": net, "negative": "probe"},
    )
    moved = probed.model_copy(
        update={"terminals": {**probed.terminals, terminal: "probe"}}
    )
    written = placing(
        original, [*instances[:index], moved, *instances[index + 1 :], ammeter]
    )
    found = _assert_same(original, written)
    if not found:
        return
    point = Simulation(written).run(OperatingPoint(record=["**"]), TIGHT)
    reading = point.current("ammeter")
    into = point.current(probed.name, terminal)
    scale = max(abs(current) for current in _currents(point)) or 1.0
    # TIGHT Newton convergence (reltol=1e-6) plus current floor across both readings.
    assert abs(reading - into) <= 1e-6 * scale + 2 * CURRENT_TOLERANCE, (
        f"the ammeter reads {reading} A, the terminal takes {into} A"
    )


def _currents(point: results.OperatingPointResult) -> Iterable[float]:
    """Every current the operating point holds: through branches and into terminals.

    The terminals are those of every device it recorded, whose currents come before
    the sums of the component instances' ports.
    """
    yield from (
        float(value)
        for unknown, value in zip(point.unknowns, point.values, strict=True)
        if unknown.nature == "current"
    )
    terminals = sum(len(device.terminals) for device in point.recording.devices)
    yield from (float(current) for current in point.currents[:terminals])

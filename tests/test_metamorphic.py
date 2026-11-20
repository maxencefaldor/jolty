# Copyright 2026 Maxence Faldor
"""Metamorphic tests verifying that equivalent circuit rewrites simulate identically.

Renaming nets, reordering instances, splitting a resistor in series, and inserting a
0 V ammeter source preserve circuit semantics. Across random circuits from
`strategies`, jolty must find identical operating points, frequency responses, and
transients, while the ammeter reads the terminal current. Each circuit serves as a
self-contained reference requiring no external simulator.

Transient comparisons add net capacitance via `strategies.parasitic` to model physical
wiring. Without parasitic capacitance, regenerative loops of charge-free devices can
hit fold singularities where a transient halts or jumps between solutions differently
across equivalent netlists.
"""

import dataclasses
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
from jolty.plan import DEFAULTS, TIGHT, Options, decade
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
    """Returns the name of the pulsed source, the circuit's first voltage source."""
    return next(
        instance.name
        for instance in _instances(circuit)
        if instance.of == ir.DeviceOf(model="voltage_source")
    )


def _transient(
    simulation: Simulation, step: float, options: Options = TIGHT
) -> dict[str, Values]:
    """Simulates waveforms of every unknown at `TIMES` with maximum step `step`."""
    plan = Transient(
        stop=STOP * s, max_step=step * s, breakpoints=[time * s for time in TIMES[1:]]
    )
    waveforms = simulation.run(plan, options)
    taken = nearest(waveforms.times, TIMES)
    # Breakpoint landing tolerance: 100 * MIN_STEP_FRACTION (1e-11 * step).
    assert np.allclose(waveforms.times[taken], TIMES, rtol=0, atol=1e-9 * step)
    return {
        unknown.name: waveforms.values[taken, column]
        for column, unknown in enumerate(waveforms.unknowns)
    }


def _simulated(
    circuit: ir.Circuit,
    loaded: ir.Circuit,
    source: str,
    options: Options = TIGHT,
) -> Simulated:
    """Computes operating point, AC response, and loaded transient for `circuit`.

    The transient runs on `loaded`, which adds parasitic net capacitance. Divergent
    transients from unstable operating points are excluded because they amplify
    rounding differences between equivalent circuit encodings.
    """
    simulation = Simulation(circuit)
    point = simulation.run(OperatingPoint(), options)
    response = simulation.run(Ac(source, decade(1 * Hz, 1 * GHz, 5)), options)
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
        found["transient"] = _transient(charged, STEP, options)
    return found


def _settled(circuit: ir.Circuit, simulated: Simulated) -> bool:
    """Returns whether the transient stays unchanged under a fourfold smaller step."""
    waveforms = simulated["transient"]
    shorter = _transient(Simulation(circuit), STEP / 4)
    return not differing(shorter, waveforms, _quantities(circuit), 1e-3)


def _rounded_up(instance: ir.Instance) -> ir.Instance:
    """Returns `instance` with any resistor value bumped by one floating-point ULP."""
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
    """Returns whether the transient stays stable when resistances bump by 1 ULP.

    When rounding alters transient step selection beyond tolerance, equivalent netlist
    encodings produce distinct waveforms despite representing identical circuits.
    """
    waveforms = simulated["transient"]
    rounded = placing(circuit, map(_rounded_up, _instances(circuit)))
    moved = _transient(Simulation(rounded), STEP)
    return not differing(moved, waveforms, _quantities(circuit), 1e-3)


def _quantities(circuit: ir.Circuit) -> dict[str, str]:
    return {unknown.name: unknown.nature for unknown in Simulation(circuit).unknowns}


TOLERANCES = {"operating point": 1e-6, "frequency response": 1e-6, "transient": 1e-3}
"""Relative agreement bounds per analysis between equivalent circuit encodings.

Normalized by each quantity's peak magnitude: Newton iteration converges to 1e-6, and
transient steps with different rounding land within 1e-3 of waveform swing."""


def _assert_same(
    original: ir.Circuit,
    written: ir.Circuit,
    name: Callable[[str], str] = lambda name: name,
) -> Simulated:
    """Asserts both circuits simulate identically and returns results for `written`.

    Each unknown in `original` maps into `written` via `name`. Transients add
    capacitance only to nets of `original` and their counterparts in `written`, leaving
    extra rewrite nets (such as ammeter nodes) unloaded so both circuits stay
    equivalent. Unsettled transients that shift under smaller steps (such as chaotic
    runs) and rounding-sensitive transients that shift under 1-ULP resistance bumps are
    excluded.
    """
    source = _source(original)
    nets = capacitive(original)
    loaded = parasitic(original, nets)
    try:
        expected = _simulated(original, loaded, source)
    except SimulationError:
        event("no simulation of the circuit as drawn")
        return {}
    loaded_written = parasitic(written, map(name, nets))
    original_width = len(Simulation(loaded).unknowns)
    written_width = len(Simulation(loaded_written).unknowns)
    # Rewrites that insert extra unknowns (e.g. an ammeter adds a probe net and a
    # branch current; splitting a resistor adds a midpoint net) store more floats per
    # accepted step. Scale `max_memory` by `written_width / original_width` so both
    # runs share the exact same step-count cap without loosening any numerical option.
    written_options = (
        dataclasses.replace(
            TIGHT,
            max_memory=DEFAULTS["max_memory"] * (written_width / original_width),
        )
        if written_width > original_width
        else TIGHT
    )
    found = _simulated(written, loaded_written, source, written_options)
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
    assert abs(reading - into) <= 1e-6 * scale + 2 * CURRENT_TOLERANCE.value, (
        f"the ammeter reads {reading} A, the terminal takes {into} A"
    )


def _currents(point: results.OperatingPointResult) -> Iterable[float]:
    """Yields every recorded current in `point`: branch currents and terminal currents.

    Recorded device terminal currents precede summed port currents of component
    instances.
    """
    yield from (
        float(value)
        for unknown, value in zip(point.unknowns, point.values, strict=True)
        if unknown.nature == "current"
    )
    terminals = sum(len(device.terminals) for device in point.recording.devices)
    yield from (float(current) for current in point.currents[:terminals])

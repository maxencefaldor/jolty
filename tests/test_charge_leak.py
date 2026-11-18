# Copyright 2026 Maxence Faldor
"""Tests for transient residual integration and charge leak reporting."""

import json
import math
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jolty import (
    Builder,
    PeriodicSteadyState,
    Role,
    Simulation,
    Transient,
    circuit,
    component,
    devices,
    ir,
    report,
    results,
    waveforms,
)
from jolty.files import load_results, save_results
from jolty.plan import Method, Options
from jolty.units import A, F, Hz, V, kHz, kohm, nF, ns, ohm, pF, ps, s, us
from jolty.unknowns import Unknown

METHODS: tuple[Method, ...] = ("backward_euler", "trapezoidal", "gear2")
"""All transient integration methods the engine supports."""

MACHINE_EPSILON = float(np.finfo(np.float64).eps)


@dataclass(frozen=True, slots=True)
class BoundaryCapacitor:
    """A capacitor between an island node and either the driving source or ground."""

    island_node: int
    driven: bool
    capacitance: float
    voltage_coefficient: float
    quadratic_voltage_coefficient: float


@dataclass(frozen=True, slots=True)
class ClosedIsland:
    """A floating capacitive island driven only across boundary capacitors."""

    nodes: int
    capacitors: tuple[BoundaryCapacitor, ...]
    resistors: tuple[tuple[int, int, float], ...]
    diodes: tuple[tuple[int, int], ...]
    initial_voltage: float
    amplitude: float
    pulsed: bool
    operating_point: bool


def _island_net(index: int) -> str:
    return f"n{index}"


def _build_island(island: ClosedIsland) -> ir.Circuit:
    @component
    def floating(builder: Builder) -> None:
        ground = builder.port("ground", Role.GROUND)
        drive = builder.net("drive")
        nets = {
            index: builder.net(_island_net(index))
            for index in range(1, island.nodes + 1)
        }
        source_waveform: waveforms.Waveform
        if island.pulsed:
            source_waveform = waveforms.Pulse(
                initial=0 * V,
                pulsed=island.amplitude * V,
                delay=500 * ns,
                rise=200 * ns,
                fall=200 * ns,
                width=1 * us,
                period=3 * us,
            )
        else:
            source_waveform = waveforms.Sine(
                offset=0 * V,
                amplitude=island.amplitude * V,
                frequency=250 * kHz,
            )
        source = builder.place(devices.voltage_source(source_waveform), name="V1")
        builder.connect(source.positive, drive)
        builder.connect(source.negative, ground)
        for index, boundary in enumerate(island.capacitors):
            capacitor = builder.place(
                devices.capacitor(
                    boundary.capacitance * F,
                    voltage_coefficient=boundary.voltage_coefficient / V,
                    quadratic_voltage_coefficient=boundary.quadratic_voltage_coefficient
                    / V**2,
                ),
                name=f"C{index}",
            )
            builder.connect(capacitor.a, nets[boundary.island_node])
            builder.connect(capacitor.b, drive if boundary.driven else ground)
        for index, (first, second, resistance) in enumerate(island.resistors):
            resistor = builder.place(
                devices.resistor(resistance * ohm), name=f"R{index}"
            )
            builder.connect(resistor.a, nets[first])
            builder.connect(resistor.b, nets[second])
        for index, (anode, cathode) in enumerate(island.diodes):
            junction = builder.place(
                devices.diode(
                    saturation_current=1e-14 * A,
                    junction_capacitance=1 * pF,
                ),
                name=f"D{index}",
            )
            builder.connect(junction.anode, nets[anode])
            builder.connect(junction.cathode, nets[cathode])

    return circuit(floating())


def _assert_valid_charge_leak(
    leak: results.ChargeLeak | None, unknown_count: int
) -> results.ChargeLeak:
    assert leak is not None
    assert leak.signed.shape == (unknown_count,)
    assert leak.absolute.shape == (unknown_count,)
    assert np.all(np.isfinite(leak.signed))
    assert np.all(np.isfinite(leak.absolute))
    assert np.all(leak.absolute >= 0.0)
    assert np.all(
        np.abs(leak.signed) <= leak.absolute + MACHINE_EPSILON * leak.absolute
    )
    return leak


def _check_closed_island(
    island: ClosedIsland,
    method: Method,
    options: Options,
    *,
    stop: float,
    max_step: float,
) -> tuple[float, float, float, float]:
    built = _build_island(island)
    simulation = Simulation(built)
    initial = (
        {_island_net(1): island.initial_voltage * V}
        if island.operating_point
        else {
            _island_net(node): (island.initial_voltage + 0.1 * (node - 1)) * V
            for node in range(1, island.nodes + 1)
        }
    )
    waveforms_result = simulation.run(
        Transient(
            stop=stop * s,
            max_step=max_step * s,
            method=method,
            initial_conditions=initial,
            operating_point=island.operating_point,
            record=("**",),
        ),
        options,
    )
    leak = _assert_valid_charge_leak(
        waveforms_result.statistics.charge_leak, len(waveforms_result.unknowns)
    )

    island_names = tuple(_island_net(node) for node in range(1, island.nodes + 1))
    signed_leak = sum(waveforms_result.charge_leak(name) for name in island_names)
    absolute_leak = sum(
        waveforms_result.charge_leak(name, absolute=True) for name in island_names
    )
    # Cross-check by-name access against the raw arrays on `statistics.charge_leak`.
    for column_index, unknown in enumerate(waveforms_result.unknowns):
        assert waveforms_result.charge_leak(unknown.name) == float(
            leak.signed[column_index]
        )
        assert waveforms_result.charge_leak(unknown.name, absolute=True) == float(
            leak.absolute[column_index]
        )

    total_charge = sum(
        waveforms_result.charge(f"C{index}", "a")
        for index in range(len(island.capacitors))
    )
    assert isinstance(total_charge, np.ndarray)
    charge_scale = float(np.max(np.sum(np.abs(waveforms_result.charges), axis=1)))
    current_scale = float(np.max(np.sum(np.abs(waveforms_result.currents), axis=1)))
    accepted = float(waveforms_result.statistics.accepted)
    rounding = (
        128.0
        * MACHINE_EPSILON
        * max(3.0 * accepted * charge_scale + stop * current_scale, 1e-30)
    )

    initial_charge = float(total_charge[0])
    final_charge = float(total_charge[-1])
    actual_change = final_charge - initial_charge
    assert math.isfinite(actual_change)
    assert abs(actual_change - signed_leak) <= rounding, (
        f"{method}: actual charge change {actual_change:e} vs signed leak "
        f"{signed_leak:e} (rounding bound {rounding:e})"
    )
    max_excursion = float(np.max(np.abs(total_charge - initial_charge)))
    assert max_excursion <= absolute_leak + rounding, (
        f"{method}: max excursion {max_excursion:e} exceeds absolute leak "
        f"{absolute_leak:e} + rounding {rounding:e}"
    )
    return actual_change, signed_leak, absolute_leak, rounding


@st.composite
def _closed_islands(draw: st.DrawFn) -> ClosedIsland:
    nodes = draw(st.integers(min_value=1, max_value=3))
    drive_exponent = draw(st.floats(min_value=-11.0, max_value=-9.0))
    drive_linear = draw(st.floats(min_value=-0.04, max_value=0.04))
    drive_quadratic = draw(st.floats(min_value=0.0, max_value=0.008))
    capacitors = [
        BoundaryCapacitor(
            island_node=1,
            driven=True,
            capacitance=10.0**drive_exponent,
            voltage_coefficient=drive_linear,
            quadratic_voltage_coefficient=drive_quadratic,
        )
    ]
    for node in range(1, nodes + 1):
        exponent = draw(st.floats(min_value=-11.0, max_value=-9.0))
        linear = draw(st.floats(min_value=-0.04, max_value=0.04))
        quadratic = draw(st.floats(min_value=0.0, max_value=0.008))
        capacitors.append(
            BoundaryCapacitor(
                island_node=node,
                driven=False,
                capacitance=10.0**exponent,
                voltage_coefficient=linear,
                quadratic_voltage_coefficient=quadratic,
            )
        )
    resistors: list[tuple[int, int, float]] = []
    for child in range(2, nodes + 1):
        parent = draw(st.integers(min_value=1, max_value=child - 1))
        exponent = draw(st.floats(min_value=1.0, max_value=5.0))
        resistors.append((child, parent, 10.0**exponent))
    diodes: list[tuple[int, int]] = []
    if nodes > 1 and draw(st.booleans()):
        anode = draw(st.integers(min_value=1, max_value=nodes))
        cathode = draw(
            st.integers(min_value=1, max_value=nodes).filter(lambda node: node != anode)
        )
        diodes.append((anode, cathode))
    return ClosedIsland(
        nodes=nodes,
        capacitors=tuple(capacitors),
        resistors=tuple(resistors),
        diodes=tuple(diodes),
        initial_voltage=draw(st.floats(min_value=-1.0, max_value=1.0)),
        amplitude=draw(st.floats(min_value=0.5, max_value=3.0)),
        pulsed=draw(st.booleans()),
        operating_point=draw(st.booleans()),
    )


def _nonlinear_island() -> ClosedIsland:
    return ClosedIsland(
        nodes=2,
        capacitors=(
            BoundaryCapacitor(
                island_node=1,
                driven=True,
                capacitance=1e-9,
                voltage_coefficient=0.04,
                quadratic_voltage_coefficient=0.008,
            ),
            BoundaryCapacitor(
                island_node=1,
                driven=False,
                capacitance=1e-9,
                voltage_coefficient=-0.02,
                quadratic_voltage_coefficient=0.004,
            ),
            BoundaryCapacitor(
                island_node=2,
                driven=False,
                capacitance=1e-9,
                voltage_coefficient=0.0,
                quadratic_voltage_coefficient=0.0,
            ),
        ),
        resistors=((1, 2, 1e5),),
        diodes=((1, 2),),
        initial_voltage=0.0,
        amplitude=5.0,
        pulsed=True,
        operating_point=True,
    )


# Pure-Python unit tests for `ChargeLeak`, `Statistics`, and `TransientResult`.


def test_charge_leak_validates_dimensions_and_lengths() -> None:
    empty = results.ChargeLeak(np.empty(0), np.empty(0))
    assert empty.signed.shape == (0,)
    assert empty.absolute.shape == (0,)

    with pytest.raises(ValueError, match="one-dimensional"):
        results.ChargeLeak(np.zeros((1, 2)), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="one-dimensional"):
        results.ChargeLeak(np.array(0.0), np.array(0.0))
    with pytest.raises(ValueError, match="same length"):
        results.ChargeLeak(np.zeros(2), np.zeros(3))


def test_charge_leak_equality_and_unhashability() -> None:
    first = results.ChargeLeak(
        signed=np.array([1e-18, -2e-18]),
        absolute=np.array([3e-18, 4e-18]),
    )
    identical = results.ChargeLeak(
        signed=np.array([1e-18, -2e-18]),
        absolute=np.array([3e-18, 4e-18]),
    )
    different = results.ChargeLeak(
        signed=np.array([1e-18, -3e-18]),
        absolute=np.array([3e-18, 4e-18]),
    )
    shorter = results.ChargeLeak(signed=np.array([1e-18]), absolute=np.array([3e-18]))
    with_nan = results.ChargeLeak(
        signed=np.array([math.nan]), absolute=np.array([math.nan])
    )
    assert first == identical
    assert first != different
    assert first != shorter
    assert first != "not a charge leak"
    assert with_nan != results.ChargeLeak(
        signed=np.array([math.nan]), absolute=np.array([math.nan])
    )
    assert not isinstance(first, Hashable)
    with pytest.raises(TypeError, match="unhashable"):
        hash(first)


def test_statistics_equality_with_charge_leak() -> None:
    first_leak = results.ChargeLeak(np.array([1e-18, -2e-18]), np.array([3e-18, 4e-18]))
    equal_leak = results.ChargeLeak(np.array([1e-18, -2e-18]), np.array([3e-18, 4e-18]))
    other_leak = results.ChargeLeak(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    base = results.Statistics("direct", 10, 1, 0, 20, charge_leak=first_leak)
    same = results.Statistics("direct", 10, 1, 0, 20, charge_leak=equal_leak)
    changed = results.Statistics("direct", 10, 1, 0, 20, charge_leak=other_leak)
    absent = results.Statistics("direct", 10, 1, 0, 20, charge_leak=None)
    assert base == same
    assert base != changed
    assert base != absent


def test_transient_result_charge_leak_reads_by_name() -> None:
    unknowns = (
        Unknown("in", "voltage"),
        Unknown("out", "voltage"),
        Unknown("V1:current", "current"),
    )
    leak = results.ChargeLeak(
        signed=np.array([0.0, -1.5e-17, 2.0e-12]),
        absolute=np.array([1.0e-20, 4.5e-17, 3.0e-12]),
    )
    waveforms_result = results.TransientResult(
        unknowns,
        np.array([0.0, 1e-6]),
        np.zeros((2, 3)),
        results.Statistics("direct", 1, 0, 0, 2, charge_leak=leak),
    )
    assert waveforms_result.charge_leak("out") == -1.5e-17
    assert waveforms_result.charge_leak("out", absolute=True) == 4.5e-17
    assert waveforms_result.charge_leak("V1:current") == 2.0e-12
    assert waveforms_result.charge_leak("V1:current", absolute=True) == 3.0e-12
    with pytest.raises(KeyError, match="out"):
        waveforms_result.charge_leak("ou")

    absent = results.TransientResult(
        unknowns,
        np.array([0.0, 1e-6]),
        np.zeros((2, 3)),
        results.Statistics("direct", 1, 0, 0, 2),
    )
    with pytest.raises(KeyError, match="no charge leak"):
        absent.charge_leak("out")


def test_periodic_steady_state_result_inherits_charge_leak_method() -> None:
    unknowns = (Unknown("out", "voltage"),)
    leak = results.ChargeLeak(np.array([-2e-18]), np.array([5e-18]))
    steady = results.PeriodicSteadyStateResult(
        unknowns,
        np.array([0.0, 1e-6]),
        np.zeros((2, 1)),
        results.Statistics("direct", 2, 0, 0, 4, charge_leak=leak),
        period=1e-6,
        periods=3,
    )
    assert steady.charge_leak("out") == -2e-18
    assert steady.charge_leak("out", absolute=True) == 5e-18


# Integration and property tests exercising the Rust engine end-to-end.


@settings(max_examples=12, deadline=None)
@given(island=_closed_islands())
def test_closed_capacitive_island_charge_stays_within_reported_leak(
    island: ClosedIsland,
) -> None:
    options = Options()
    for method in METHODS:
        _check_closed_island(island, method, options, stop=4e-6, max_step=2e-7)


def test_loose_tolerance_produces_measurable_exact_charge_leak() -> None:
    island = _nonlinear_island()
    loose = Options(
        relative_tolerance=2e-2,
        voltage_tolerance=5e-3,
        current_tolerance=1e-6,
    )
    tight = Options(
        relative_tolerance=1e-5,
        voltage_tolerance=1e-8,
        current_tolerance=1e-14,
    )
    for method in METHODS:
        actual_loose, signed_loose, absolute_loose, rounding_loose = (
            _check_closed_island(island, method, loose, stop=4e-6, max_step=1e-7)
        )
        _, _, absolute_tight, _ = _check_closed_island(
            island, method, tight, stop=4e-6, max_step=1e-7
        )
        assert absolute_loose > 100.0 * rounding_loose
        assert abs(actual_loose - signed_loose) <= rounding_loose
        assert absolute_tight < absolute_loose


def test_charge_leak_round_trips_through_results_files(tmp_path: Path) -> None:
    @component
    def rectifier(builder: Builder) -> None:
        ground = builder.port("ground", Role.GROUND)
        source = builder.place(
            devices.voltage_source(
                waveforms.Sine(offset=0 * V, amplitude=2 * V, frequency=1e5 * Hz)
            ),
            name="V1",
        )
        junction = builder.place(
            devices.diode(saturation_current=1e-14 * A, junction_capacitance=1 * pF),
            name="D1",
        )
        load = builder.place(devices.resistor(1 * kohm), name="R1")
        filter_capacitor = builder.place(devices.capacitor(1 * nF), name="C1")
        input_net = builder.net("input")
        output_net = builder.net("output")
        builder.connect(source.positive, junction.anode, input_net)
        builder.connect(junction.cathode, load.a, filter_capacitor.a, output_net)
        builder.connect(source.negative, load.b, filter_capacitor.b, ground)

    simulation = Simulation(circuit(rectifier()))
    step = simulation.run(Transient(stop=10 * us, max_step=0.2 * us))
    steady = simulation.run(PeriodicSteadyState(period=10 * us, max_step=0.2 * us))
    path = tmp_path / "rectifier.results.npz"
    save_results(path, {"step": step, "steady": steady})

    with np.load(path) as data:
        header = json.loads(str(data["header"]))
    assert header["version"] == 20
    assert header["analyses"]["step"]["statistics"]["charge_leak"] is True
    assert header["analyses"]["steady"]["statistics"]["charge_leak"] is True

    loaded = load_results(path)
    for name, original in (("step", step), ("steady", steady)):
        reloaded = loaded[name]
        assert isinstance(reloaded, results.TransientResult)
        assert original.statistics.charge_leak is not None
        assert reloaded.statistics.charge_leak is not None
        assert reloaded.statistics.charge_leak == original.statistics.charge_leak
        assert np.array_equal(
            reloaded.statistics.charge_leak.signed.view(np.uint64),
            original.statistics.charge_leak.signed.view(np.uint64),
        )
        assert np.array_equal(
            reloaded.statistics.charge_leak.absolute.view(np.uint64),
            original.statistics.charge_leak.absolute.view(np.uint64),
        )


def test_pure_logic_transient_has_empty_charge_leak_and_no_report_line() -> None:
    @component
    def logic_only(builder: Builder) -> None:
        builder.port("ground", Role.GROUND)
        source = builder.place(
            devices.logic_source(0, changes=[(1 * ns, 1)], transition=1 * ps),
            name="A",
        )
        builder.connect(source.output, builder.net("a"))

    waveforms_result = Simulation(circuit(logic_only())).run(Transient(stop=5 * ns))
    leak = _assert_valid_charge_leak(waveforms_result.statistics.charge_leak, 0)
    assert leak.signed.size == 0
    assert leak.absolute.size == 0
    summary = report.report("step", waveforms_result, max_unknowns=10)
    assert isinstance(summary, report.TransientReport)
    assert summary.charge_leak is None
    assert "largest charge leak" not in summary.describe()


def test_report_surfaces_nonzero_charge_leak_on_simulated_transient() -> None:
    island = _nonlinear_island()
    loose = Options(
        relative_tolerance=2e-2,
        voltage_tolerance=5e-3,
        current_tolerance=1e-6,
    )
    waveforms_result = Simulation(_build_island(island)).run(
        Transient(
            stop=4 * us,
            max_step=0.1 * us,
            initial_conditions={"n1": 0 * V},
            operating_point=True,
        ),
        loose,
    )
    summary = report.report("step", waveforms_result, max_unknowns=10)
    assert isinstance(summary, report.TransientReport)
    assert summary.charge_leak is not None
    assert summary.charge_leak.absolute > 0.0
    assert summary.charge_leak.net in {"drive", "n1", "n2"}
    assert "largest charge leak:" in summary.describe()

# Copyright 2026 Maxence Faldor
"""Tests for transient residual integration and charge leak reporting at Python's FFI.

Exhaustive property-based charge conservation across random floating capacitive islands
and loose-vs-tight tolerance convergence live in Rust
(`crates/jolty-engine/tests/invariants.rs`). This module covers the Python data model,
PyO3 FFI array marshaling, `.results.npz` serialization, and report formatting.
"""

import json
import math
from collections.abc import Hashable
from pathlib import Path
from typing import Final

import numpy as np
import pytest

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
from jolty.plan import Options
from jolty.units import A, Hz, V, kohm, nF, ns, pF, ps, us
from jolty.unknowns import Unknown

MACHINE_EPSILON: Final = float(np.finfo(np.float64).eps)
"""IEEE-754 binary64 unit roundoff, used to bound floating-point accumulation error."""


def _closed_nonlinear_island() -> ir.Circuit:
    """Two-node floating capacitive island with polynomial capacitors, R, and diode."""

    @component
    def floating(builder: Builder) -> None:
        ground = builder.port("ground", Role.GROUND)
        pulse = waveforms.Pulse(
            initial=0 * V,
            pulsed=5 * V,
            delay=500 * ns,
            rise=200 * ns,
            fall=200 * ns,
            width=1 * us,
            period=3 * us,
        )
        source = builder.place(devices.voltage_source(pulse), name="V1")
        c0 = builder.place(
            devices.capacitor(
                1 * nF,
                voltage_coefficient=0.04 / V,
                quadratic_voltage_coefficient=0.008 / V**2,
            ),
            name="C0",
        )
        c1 = builder.place(
            devices.capacitor(
                1 * nF,
                voltage_coefficient=-0.02 / V,
                quadratic_voltage_coefficient=0.004 / V**2,
            ),
            name="C1",
        )
        c2 = builder.place(devices.capacitor(1 * nF), name="C2")
        resistor = builder.place(devices.resistor(100 * kohm), name="R0")
        junction = builder.place(
            devices.diode(saturation_current=1e-14 * A, junction_capacitance=1 * pF),
            name="D0",
        )
        builder.connect(source.positive, c0.b, name="drive")
        builder.connect(c0.a, c1.a, resistor.a, junction.anode, name="n1")
        builder.connect(c2.a, resistor.b, junction.cathode, name="n2")
        builder.connect(source.negative, c1.b, c2.b, ground)

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
    # Signed sum cannot exceed sum of absolute terms beyond 1 ulp rounding.
    assert np.all(
        np.abs(leak.signed) <= leak.absolute + MACHINE_EPSILON * leak.absolute
    )
    return leak


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


# End-to-end PyO3 FFI, serialization, and report boundary tests.


def test_closed_capacitive_island_charge_leak_crosses_ffi_faithfully() -> None:
    """Confirms charge_leak arrays cross PyO3 faithfully on a closed RC island.

    Bound uses `128 * MACHINE_EPSILON` times charge/current scales for IEEE-754
    floating-point summation rounding across accepted transient steps.
    """
    stop = 4 * us
    waveforms_result = Simulation(_closed_nonlinear_island()).run(
        Transient(
            stop=stop,
            max_step=0.1 * us,
            initial_conditions={"n1": 0 * V},
            operating_point=True,
            record=("**",),
        ),
        Options(),
    )
    leak = _assert_valid_charge_leak(
        waveforms_result.statistics.charge_leak, len(waveforms_result.unknowns)
    )
    for column_index, unknown in enumerate(waveforms_result.unknowns):
        assert waveforms_result.charge_leak(unknown.name) == float(
            leak.signed[column_index]
        )
        assert waveforms_result.charge_leak(unknown.name, absolute=True) == float(
            leak.absolute[column_index]
        )

    signed_leak = sum(waveforms_result.charge_leak(name) for name in ("n1", "n2"))
    absolute_leak = sum(
        waveforms_result.charge_leak(name, absolute=True) for name in ("n1", "n2")
    )
    total_charge = sum(waveforms_result.charge(f"C{index}", "a") for index in range(3))
    assert isinstance(total_charge, np.ndarray)
    charge_scale = float(np.max(np.sum(np.abs(waveforms_result.charges), axis=1)))
    current_scale = float(np.max(np.sum(np.abs(waveforms_result.currents), axis=1)))
    accepted = float(waveforms_result.statistics.accepted)
    rounding = (
        128.0
        * MACHINE_EPSILON
        * max(3.0 * accepted * charge_scale + stop.value * current_scale, 1e-30)
    )
    initial_charge = float(total_charge[0])
    actual_change = float(total_charge[-1]) - initial_charge
    assert math.isfinite(actual_change)
    assert abs(actual_change - signed_leak) <= rounding
    assert float(np.max(np.abs(total_charge - initial_charge))) <= (
        absolute_leak + rounding
    )


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
    loose = Options(
        relative_tolerance=2e-2,
        voltage_tolerance=5e-3,
        current_tolerance=1e-6,
    )
    waveforms_result = Simulation(_closed_nonlinear_island()).run(
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

# Copyright 2026 Maxence Faldor
"""VTEAM and HP memristor models verified against closed-form pinched hysteresis loops.

Driven by a sine, each memristor's state follows a closed form while its window is far
from either bound. VTEAM's state integrates the voltage's excess over its threshold,
and HP's, a charge-controlled memristor, makes its resistance the square root of a line
in the flux. The current is the voltage over the resistance the state sets, so its loop
against the voltage is pinched at the origin, and shrinks as the frequency rises.
"""

import dataclasses
import math
import re
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from jolty import (
    Builder,
    Component,
    Options,
    Role,
    Simulation,
    component,
    devices,
    results,
)
from jolty.plan import (
    Ac,
    DcSweep,
    OperatingPoint,
    PeriodicSteadyState,
    Sensitivity,
    Transient,
)
from jolty.plots import plot
from jolty.report import report
from jolty.specs import Specs, evaluate, value
from jolty.units import C, Hz, V, kohm, nF, ohm, s
from jolty.waveforms import Sine

type Real = npt.NDArray[np.float64]

ON, OFF = 1e3, 10e3
"""The VTEAM memristor's on and off resistances, in ohms."""
THRESHOLD = 0.5
"""Off threshold voltage and negative magnitude of the on threshold, in volts."""
AMPLITUDE = 1.0
"""The sine's amplitude across each memristor, in volts: twice the threshold."""
FREQUENCY = 1e3
"""The sine's frequency, in hertz."""
START = 0.3
"""The state each memristor is made in."""
SWING = 0.4
"""State excursion produced by each sinusoidal half-cycle lobe."""

MEMRISTOR = ("M1",)
"""The memristor, recorded, whose current, charge and variables the tests read."""

TIGHT = Options(relative_tolerance=1e-7)
"""Tight solver options keeping integration error far below comparison tolerances."""


def _driven(memristor: devices.Device, frequency: float = FREQUENCY) -> Component:
    """Builds a circuit driving `memristor` with an `AMPLITUDE` sine at `frequency`."""

    @component
    def driven(c: Builder, memristor: devices.Device, frequency: float) -> None:
        ground = c.port("ground", Role.GROUND)
        sine = Sine(0 * V, AMPLITUDE * V, frequency * Hz)
        source = c.place(devices.voltage_source(sine), name="V1")
        cell = c.place(memristor, name="M1")
        c.connect(source.positive, cell.positive, name="in")
        c.connect(source.negative, cell.negative, ground)

    return driven(memristor, frequency)


def _run(memristor: devices.Device, periods: float = 2) -> results.TransientResult:
    """Simulates `memristor` waveforms over `periods` sine cycles with fine time steps.

    VTEAM state transitions occur at threshold crossings where no step need land. At a
    ten-thousandth of a period, the resulting integration error is a few parts in 10⁸
    of the state swing.
    """
    simulation = Simulation(_driven(memristor))
    stop, step = periods / FREQUENCY, 1e-4 / FREQUENCY
    return simulation.run(
        Transient(stop * s, max_step=step * s, record=MEMRISTOR), TIGHT
    )


def _resistance(state: Real, law: str, on: float = ON, off: float = OFF) -> Real:
    """Computes memristor resistance from `state` according to `law`."""
    if law == "linear":
        return on + (off - on) * state
    return on * (off / on) ** state


# VTEAM's memristor.

_ANTIDERIVATIVES: dict[int, Callable[[float, Real], Real]] = {
    1: lambda a, phase: -a * np.cos(phase) - phase,
    2: lambda a, phase: (
        a**2 * (phase / 2 - np.sin(2 * phase) / 4) + 2 * a * np.cos(phase) + phase
    ),
}
"""An antiderivative of ``(a·sin θ - 1)^n`` in `θ`, for each exponent `n`."""

CROSSING = math.asin(THRESHOLD / AMPLITUDE)
"""The phase at which the sine crosses the threshold."""


def _lobe(phase: Real, exponent: int) -> Real:
    """Evaluates ``∫ (a·sin θ - 1)^n dθ`` from the threshold crossing to `phase`."""
    a = AMPLITUDE / THRESHOLD
    antiderivative = _ANTIDERIVATIVES[exponent]
    return antiderivative(a, phase) - antiderivative(a, np.array(CROSSING))


def _vteam_rate(exponent: int, frequency: float = FREQUENCY) -> float:
    """Computes the VTEAM switching rate that moves state by `SWING` per sine lobe."""
    lobe = float(_lobe(np.array(math.pi - CROSSING), exponent))
    return SWING * 2 * math.pi * frequency / lobe


def _vteam(
    exponent: int = 1, law: str = "linear", rate: float | None = None
) -> devices.Device:
    """Builds a VTEAM memristor with symmetric thresholds and switching rates."""
    rate = _vteam_rate(exponent) if rate is None else rate
    return devices.voltage_controlled_memristor(
        on_resistance=ON * ohm,
        off_resistance=OFF * ohm,
        on_voltage=-THRESHOLD * V,
        off_voltage=THRESHOLD * V,
        on_rate=rate / s,
        off_rate=rate / s,
        on_exponent=exponent,
        off_exponent=exponent,
        law="linear" if law == "linear" else "exponential",
        state=START,
    )


def _vteam_state(times: Real, exponent: int) -> Real:
    """Computes closed-form VTEAM state under sinusoidal drive away from window bounds.

    Above the off voltage, state rises as ``k·(a·sin θ - 1)^n``; below the on voltage,
    state falls symmetrically on the negative lobe; between thresholds, state holds
    constant. Each lobe shifts state by `SWING` and returns on the opposite half-cycle.
    """
    omega = 2 * math.pi * FREQUENCY
    speed = _vteam_rate(exponent) / omega
    phase = np.mod(omega * times, 2 * math.pi)
    rising = (phase > CROSSING) & (phase < math.pi - CROSSING)
    high = (phase >= math.pi - CROSSING) & (phase <= math.pi + CROSSING)
    falling = (phase > math.pi + CROSSING) & (phase < 2 * math.pi - CROSSING)
    state = np.full_like(times, START)
    state[rising] += speed * _lobe(phase[rising], exponent)
    state[high] += SWING
    state[falling] += SWING - speed * _lobe(phase[falling] - math.pi, exponent)
    return state


@pytest.mark.parametrize("law", ["linear", "exponential"])
@pytest.mark.parametrize("exponent", [1, 2])
def test_vteams_loop_is_the_closed_form_of_its_thresholds(
    exponent: int, law: str
) -> None:
    waveforms = _run(_vteam(exponent, law))
    times = waveforms.times
    state = _vteam_state(times, exponent)
    # TIGHT transient (max_step = 1e-4 / f): threshold quadrature error O(h^2).
    np.testing.assert_allclose(waveforms.state("M1:state"), state, rtol=0, atol=1e-6)
    voltage = AMPLITUDE * np.sin(2 * math.pi * FREQUENCY * times)
    expected = voltage / _resistance(state, law)
    # State error atol=1e-6 mapped to current via peak conductance 1 / ON.
    np.testing.assert_allclose(
        waveforms.current("M1", "positive"), expected, rtol=0, atol=1e-6 * ON**-1
    )
    # Exponential R(x) amplifies state error by ln(OFF/ON) ~= 4.6 -> rtol=1e-5.
    np.testing.assert_allclose(
        waveforms.variable("M1", "resistance"), _resistance(state, law), rtol=1e-5
    )


def test_below_its_thresholds_vteams_memristor_is_a_resistor() -> None:
    quiet = devices.voltage_controlled_memristor(
        on_resistance=ON * ohm,
        off_resistance=OFF * ohm,
        on_voltage=-1.5 * AMPLITUDE * V,
        off_voltage=1.5 * AMPLITUDE * V,
        on_rate=1e9 / s,
        off_rate=1e9 / s,
        state=START,
    )
    waveforms = _run(quiet)
    # Zero drift below thresholds; atol=1e-9 matches TIGHT vntol/state floor.
    np.testing.assert_allclose(waveforms.state("M1:state"), START, rtol=0, atol=1e-9)
    voltage = waveforms.voltage("in")
    # Constant current V / R(START); rtol=1e-9, atol=1e-15 cover rounding & floor.
    np.testing.assert_allclose(
        waveforms.current("M1", "positive"),
        voltage / _resistance(np.array(START), "linear"),
        rtol=1e-9,
        atol=1e-15,
    )


def test_a_periodic_steady_state_keeps_a_state_its_drive_leaves_alone() -> None:
    """Verifies that sub-threshold VTEAM states repeat under periodic shooting.

    The periodic steady state is non-unique: shooting preserves the initial memristor
    state just like transient integration, and the series capacitor settles through the
    corresponding fixed resistance.
    """
    quiet = devices.voltage_controlled_memristor(
        on_resistance=ON * ohm,
        off_resistance=OFF * ohm,
        on_voltage=-1.5 * AMPLITUDE * V,
        off_voltage=1.5 * AMPLITUDE * V,
        on_rate=1e9 / s,
        off_rate=1e9 / s,
        state=START,
    )

    @component
    def filtered(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        sine = Sine(0 * V, AMPLITUDE * V, FREQUENCY * Hz)
        source = c.place(devices.voltage_source(sine), name="V1")
        cell = c.place(quiet, name="M1")
        capacitor = c.place(devices.capacitor(100 * nF), name="C1")
        c.connect(source.positive, cell.positive, name="in")
        c.connect(cell.negative, capacitor.a, name="out")
        c.connect(capacitor.b, source.negative, ground)

    period = 1 / FREQUENCY
    steady = Simulation(filtered()).run(
        PeriodicSteadyState(period * s, max_step=5e-3 * period * s)
    )
    assert steady.periods == 2
    # Neutral state stays fixed across shooting steps within f64 rounding (~4.5 eps).
    np.testing.assert_allclose(steady.state("M1:state"), START, rtol=0, atol=1e-15)
    omega = 2 * math.pi * FREQUENCY
    lag = omega * float(_resistance(np.array(START), "linear")) * 100e-9
    expected = np.sin(omega * steady.times - math.atan(lag)) / math.hypot(1, lag)
    # Trapezoidal O(h^2) phase/amplitude error at max_step = 5e-3 * T (default reltol).
    np.testing.assert_allclose(steady.voltage("out"), AMPLITUDE * expected, atol=1e-3)


def _lobe_area(waveforms: results.TransientResult, period: float) -> float:
    """Computes the positive-lobe hysteresis area ``∫ i dv`` in the second period.

    Opposite lobes of a pinched loop wind in opposite directions, so full-period areas
    cancel. The positive half-cycle lobe closes independently at the origin.
    """
    times = waveforms.times
    within = (times >= period) & (times <= 1.5 * period)
    voltage = waveforms.voltage("in")[within]
    current = waveforms.current("M1", "positive")[within]
    return abs(float(np.sum((current[1:] + current[:-1]) * np.diff(voltage)) / 2))


def test_a_pinched_loop_shrinks_as_the_frequency_rises() -> None:
    """Verifies Chua's memristor fingerprints for both VTEAM and HP models.

    The current-voltage loop passes through the origin, and lobe area decreases as
    drive frequency increases. As frequency grows without bound, the loop collapses to
    a line because the integrating state has progressively less time to move.
    """
    for cell in [_vteam(), _hp()]:
        areas = []
        for factor in [1, 4, 16, 64]:
            simulation = Simulation(_driven(cell, FREQUENCY * factor))
            period = 1 / (FREQUENCY * factor)
            transient = Transient(
                2 * period * s, max_step=period / 2000 * s, record=MEMRISTOR
            )
            waveforms = simulation.run(transient, TIGHT)
            voltage = waveforms.voltage("in")
            current = waveforms.current("M1", "positive")
            # The loop is pinched: no current flows wherever no voltage is across it.
            np.testing.assert_array_equal(np.sign(current), np.sign(voltage))
            areas.append(_lobe_area(waveforms, period))
        assert all(np.diff(areas) < 0), (cell.model, areas)
        assert areas[-1] < areas[0] / 10, (cell.model, areas)


# HP's memristor.

HP_ON, HP_OFF = 100.0, 16e3
"""On and off resistances of the HP memristor in ohms, from Strukov et al."""


def _hp_charge() -> float:
    """Computes the full-range HP memristor charge producing `SWING` per sine lobe.

    Resistance obeys ``√(r₀² + 2·Δr·φ/Q)`` at flux `φ`, which one sinusoidal lobe
    raises to ``2·V/ω``.
    """
    flux = 2 * AMPLITUDE / (2 * math.pi * FREQUENCY)
    start, stop = (
        float(_resistance(np.array(state), "linear", HP_ON, HP_OFF))
        for state in (START, START + SWING)
    )
    return 2 * (HP_OFF - HP_ON) * flux / (stop**2 - start**2)


def _hp() -> devices.Device:
    """Builds the HP memristor initialized at state `START`."""
    return devices.current_controlled_memristor(
        on_resistance=HP_ON * ohm,
        off_resistance=HP_OFF * ohm,
        charge=_hp_charge() * C,
        state=START,
    )


def test_hps_resistance_is_the_square_root_of_a_line_in_the_flux() -> None:
    """Verifies Strukov et al.'s closed form ``M(φ) = √(r₀² + 2·Δr·φ/Q)``.

    State varies linearly with charge divided by `Q`, making resistance linear in
    charge (``r₀ + Δr·q/Q``). Integrating ``r·dq`` gives flux quadratic in charge, so
    resistance equals the square root of a linear function of flux.
    """
    waveforms = _run(_hp())
    omega = 2 * math.pi * FREQUENCY
    flux = AMPLITUDE / omega * (1 - np.cos(omega * waveforms.times))
    start = float(_resistance(np.array(START), "linear", HP_ON, HP_OFF))
    resistance = np.sqrt(start**2 + 2 * (HP_OFF - HP_ON) * flux / _hp_charge())
    # TIGHT transient (reltol=1e-6) integration of flux ODE at max_step = 1e-4 / f:
    # Trapezoidal O(h^2) error bounded by rtol=1e-5 in resistance, current, and state.
    np.testing.assert_allclose(
        waveforms.variable("M1", "resistance"), resistance, rtol=1e-5
    )
    voltage = AMPLITUDE * np.sin(omega * waveforms.times)
    # Resistance error rtol=1e-5 mapped to current peak AMPLITUDE / start.
    np.testing.assert_allclose(
        waveforms.current("M1", "positive"),
        voltage / resistance,
        rtol=0,
        atol=1e-5 * AMPLITUDE / start,
    )
    state = (resistance - HP_ON) / (HP_OFF - HP_ON)
    # Discrete spacing around peak flux (1e-4 of period) bounds peak error < 1e-5.
    assert state.max() == pytest.approx(START + SWING, abs=1e-5)


# At an operating point.


def _fed(level: float, memristor: devices.Device | None = None) -> Component:
    """Builds a circuit driving `memristor` (default VTEAM) from `level` via 1 kΩ."""

    @component
    def fed(c: Builder, level: float, memristor: devices.Device) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(level * V), name="V1")
        feed = c.place(devices.resistor(1 * kohm), name="R1")
        cell = c.place(memristor, name="M1")
        c.connect(supply.positive, feed.a, name="in")
        c.connect(feed.b, cell.positive, name="cell")
        c.connect(supply.negative, cell.negative, ground)

    return fed(level, _vteam() if memristor is None else memristor)


def test_an_operating_point_holds_the_state_and_shows_its_rate() -> None:
    """Verifies that DC operating points preserve initial state and report state rate.

    The reported derivative matches the instantaneous transient rate: zero between
    thresholds and VTEAM's switching law beyond them.
    """
    simulation = Simulation(_fed(0.0))
    levels = [-3.0, -0.2, 0.2, 3.0]
    sweep = simulation.run(
        DcSweep("V1", [level * V for level in levels], record=MEMRISTOR)
    )
    # DC sweep leaves state fixed; linear/algebraic variables match within LU rounding.
    np.testing.assert_allclose(sweep.state("M1:state"), START, rtol=0, atol=1e-12)
    resistance = ON + (OFF - ON) * START
    # Default assert_allclose (rtol=1e-7) covers linear state-to-resistance rounding.
    np.testing.assert_allclose(sweep.variable("M1", "resistance"), resistance)
    cell = np.array(levels) * resistance / (resistance + 1e3)
    overdrive = np.maximum(np.abs(cell) / THRESHOLD - 1, 0)
    expected = np.sign(cell) * _vteam_rate(1) * overdrive
    # Analytic piecewise-linear rate vs closed-form divider: LU rounding (1e-12).
    np.testing.assert_allclose(sweep.variable("M1", "rate"), expected, rtol=1e-12)


@pytest.mark.parametrize(("level", "bound"), [(3.0, 1.0), (-3.0, 0.0)])
def test_held_past_a_threshold_its_state_runs_to_the_bound(
    level: float, bound: float
) -> None:
    simulation = Simulation(_fed(level))
    waveforms = simulation.run(Transient(1e-3 * s, record=MEMRISTOR), TIGHT)
    state = waveforms.state("M1:state")
    assert state[0] == START
    # TIGHT transient (reltol=1e-6) saturating at hard window bound 0 or 1.
    assert state[-1] == pytest.approx(bound, abs=1e-6)
    assert np.all((state >= -1e-6) & (state <= 1 + 1e-6))


def test_its_small_signal_resistance_is_the_one_its_state_sets() -> None:
    @component
    def divider(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(0 * V), name="V1")
        feed = c.place(devices.resistor(ON * ohm), name="R1")
        cell = c.place(_vteam(), name="M1")
        c.connect(supply.positive, feed.a, name="in")
        c.connect(feed.b, cell.positive, name="out")
        c.connect(supply.negative, cell.negative, ground)

    simulation = Simulation(divider())
    response = simulation.run(Ac("V1", [1.0 * Hz, 1e6 * Hz]))
    cell = ON + (OFF - ON) * START
    # Linearized AC divider gain cell / (cell + ON): complex LU rounding (rtol=1e-9).
    np.testing.assert_allclose(
        np.abs(response.voltage("out")), cell / (cell + ON), rtol=1e-9
    )


def test_a_transients_sensitivities_carry_the_state_it_starts_from() -> None:
    """Verifies that transient state sensitivities match finite differences of runs.

    The operating point holds the initial state and seeds transient charges,
    propagating sensitivity with respect to initial state from DC into the transient.
    """
    transient = Transient(20e-6 * s, max_step=0.1e-6 * s, record=MEMRISTOR)
    simulation = Simulation(_fed(3.0))
    found = simulation.run(Sensitivity(["M1:state"], of=transient), TIGHT)

    def final(made: float, rate: float) -> float:
        cell = dataclasses.replace(
            _vteam(),
            parameters=_vteam().parameters | {"state": made, "off_rate": rate},
        )
        waveforms = Simulation(_fed(3.0, cell)).run(transient, TIGHT)
        return float(waveforms.state("M1:state")[-1])

    rate, step = _vteam_rate(1), 1e-5
    differences = {
        "state": (final(START + step, rate) - final(START - step, rate)) / (2 * step),
        "off_rate": (final(START, rate * (1 + step)) - final(START, rate * (1 - step)))
        / (2 * step * rate),
    }
    for name, difference in differences.items():
        derivatives = np.asarray(found.derivative("M1:state", "M1", name), dtype=float)
        derivative = float(derivatives[-1])
        # Central difference (step=1e-5) over TIGHT transient (truncation + shifts).
        assert derivative == pytest.approx(difference, rel=1e-5), name


def test_a_state_has_a_panel_of_its_own_and_reads_as_a_number() -> None:
    simulation = Simulation(_driven(_vteam()))
    waveforms = simulation.run(Transient(1 / FREQUENCY * s, record=MEMRISTOR))
    figure = plot("run", waveforms, ["in", "M1:state"])
    labels = [axes.get_ylabel() for axes in figure.axes]
    assert labels == ["voltage", "state"]
    point = simulation.run(OperatingPoint())
    written = report("point", point, max_unknowns=10).describe()
    assert re.search(r"M1:state\s+0\.3\b", written), written


def test_a_spec_reads_a_state_as_a_plain_number() -> None:
    simulation = Simulation(_driven(_vteam()))
    point = simulation.run(OperatingPoint())
    requirement = value("point", "M1:state").within(START, 0.01)
    (verdict,) = evaluate(Specs([requirement]), ["point"], {"point": point})
    assert str(requirement) == "M1:state in point ≥ 0.29 and ≤ 0.31"
    assert (verdict.status, verdict.unit) == ("passed", "")
    # DC operating point holds initial state START; default pytest.approx rounding.
    assert verdict.value == pytest.approx(START)

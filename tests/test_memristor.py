# Copyright 2026 Maxence Faldor
"""Memristors, VTEAM's and HP's, against the pinched loops their equations solve to.

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
from jolty.plan import Ac, DcSweep, OperatingPoint, Sensitivity, Transient
from jolty.plots import plot
from jolty.report import report
from jolty.units import C, Hz, V, kohm, ohm, s
from jolty.waveforms import Sine

type Real = npt.NDArray[np.float64]

ON, OFF = 1e3, 10e3
"""The VTEAM memristor's on and off resistances, in ohms."""
THRESHOLD = 0.5
"""Its off voltage, and its on voltage's magnitude, in volts."""
AMPLITUDE = 1.0
"""The sine's amplitude across each memristor, in volts: twice the threshold."""
FREQUENCY = 1e3
"""The sine's frequency, in hertz."""
START = 0.3
"""The state each memristor is made in."""
SWING = 0.4
"""How far each lobe of the sine moves a state."""

TIGHT = Options(relative_tolerance=1e-7)
"""Options under which the integration's error is far below the comparisons'."""


def _driven(memristor: devices.Device, frequency: float = FREQUENCY) -> Component:
    """`memristor` across a sine of `AMPLITUDE` at `frequency`."""

    @component
    def driven(c: Builder, memristor: devices.Device, frequency: float) -> None:
        ground = c.port("ground", Role.GROUND)
        sine = Sine(0 * V, AMPLITUDE * V, frequency * Hz)
        source = c.place(devices.voltage_source(sine), id="V1")
        cell = c.place(memristor, id="M1")
        c.connect(source.positive, cell.positive, name="in")
        c.connect(source.negative, cell.negative, ground)

    return driven(memristor, frequency)


def _run(memristor: devices.Device, periods: float = 2) -> results.Waveforms:
    """The memristor's waveforms over `periods` of the sine, in fine steps.

    VTEAM's state turns as the sine crosses a threshold, where no step need land; a
    ten-thousandth of a period from it, the integration's error there is a few parts
    in 10⁸ of its swing.
    """
    simulation = Simulation(_driven(memristor))
    stop, step = periods / FREQUENCY, 1e-4 / FREQUENCY
    return simulation.run(Transient(stop * s, max_step=step * s), TIGHT)


def _resistance(state: Real, law: str, on: float = ON, off: float = OFF) -> Real:
    """The resistance the state sets, by `law`."""
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
    """``∫ (a·sin θ - 1)^n dθ`` from the sine's crossing of the threshold to `phase`."""
    a = AMPLITUDE / THRESHOLD
    antiderivative = _ANTIDERIVATIVES[exponent]
    return antiderivative(a, phase) - antiderivative(a, np.array(CROSSING))


def _vteam_rate(exponent: int, frequency: float = FREQUENCY) -> float:
    """The rate, at twice the threshold, at which a lobe moves the state by `SWING`."""
    lobe = float(_lobe(np.array(math.pi - CROSSING), exponent))
    return SWING * 2 * math.pi * frequency / lobe


def _vteam(
    exponent: int = 1, law: str = "linear", rate: float | None = None
) -> devices.Device:
    """A VTEAM memristor symmetric in its thresholds and its rates."""
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
    """The VTEAM memristor's state under the sine, its window far from either bound.

    Above the off voltage, the state rises at ``k·(a·sin θ - 1)^n``; below the on
    voltage, it falls at the same rate of the sine's negative lobe; between them, it
    holds. Each lobe moves it by `SWING`, the negative one back.
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
    np.testing.assert_allclose(waveforms.state("M1:state"), state, rtol=0, atol=1e-6)
    voltage = AMPLITUDE * np.sin(2 * math.pi * FREQUENCY * times)
    expected = voltage / _resistance(state, law)
    np.testing.assert_allclose(
        waveforms.current("M1", "positive"), expected, rtol=0, atol=1e-6 * ON**-1
    )
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
    np.testing.assert_allclose(waveforms.state("M1:state"), START, rtol=0, atol=1e-9)
    voltage = waveforms.voltage("in")
    np.testing.assert_allclose(
        waveforms.current("M1", "positive"),
        voltage / _resistance(np.array(START), "linear"),
        rtol=1e-9,
        atol=1e-15,
    )


def _lobe_area(waveforms: results.Waveforms, period: float) -> float:
    """The area of the loop's positive lobe in the second period, ``∫ i dv``.

    The lobes of a pinched loop turn opposite ways, so over a whole period their areas
    cancel; the lobe over the sine's positive half closes at the origin on its own.
    """
    times = waveforms.times
    within = (times >= period) & (times <= 1.5 * period)
    voltage = waveforms.voltage("in")[within]
    current = waveforms.current("M1", "positive")[within]
    return abs(float(np.sum((current[1:] + current[:-1]) * np.diff(voltage)) / 2))


def test_a_pinched_loop_shrinks_as_the_frequency_rises() -> None:
    """Chua's fingerprints of a memristor, for VTEAM's and for HP's.

    The loop of current against voltage passes through the origin, its lobes' area
    falls as the frequency rises, and the loop closes to a line as it grows without
    bound: the state, which integrates its drive, has ever less time to move.
    """
    for cell in [_vteam(), _hp()]:
        areas = []
        for factor in [1, 4, 16, 64]:
            simulation = Simulation(_driven(cell, FREQUENCY * factor))
            period = 1 / (FREQUENCY * factor)
            transient = Transient(2 * period * s, max_step=period / 2000 * s)
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
"""HP's memristor's on and off resistances, in ohms, as Strukov et al. give them."""


def _hp_charge() -> float:
    """The charge across the whole range under which a lobe moves the state by `SWING`.

    The resistance is ``√(r₀² + 2·Δr·φ/Q)`` at a flux `φ`, which a lobe of the sine
    raises to ``2·V/ω``.
    """
    flux = 2 * AMPLITUDE / (2 * math.pi * FREQUENCY)
    start, stop = (
        float(_resistance(np.array(state), "linear", HP_ON, HP_OFF))
        for state in (START, START + SWING)
    )
    return 2 * (HP_OFF - HP_ON) * flux / (stop**2 - start**2)


def _hp() -> devices.Device:
    """HP's memristor, made at `START`."""
    return devices.current_controlled_memristor(
        on_resistance=HP_ON * ohm,
        off_resistance=HP_OFF * ohm,
        charge=_hp_charge() * C,
        state=START,
    )


def test_hps_resistance_is_the_square_root_of_a_line_in_the_flux() -> None:
    """Strukov et al.'s closed form: ``M(φ) = √(r₀² + 2·Δr·φ/Q)``.

    The state moves by the charge over `Q`, so the resistance is linear in the charge,
    ``r₀ + Δr·q/Q``; the flux, the integral of ``r·dq``, is then quadratic in the
    charge, and the resistance the square root of a line in the flux.
    """
    waveforms = _run(_hp())
    omega = 2 * math.pi * FREQUENCY
    flux = AMPLITUDE / omega * (1 - np.cos(omega * waveforms.times))
    start = float(_resistance(np.array(START), "linear", HP_ON, HP_OFF))
    resistance = np.sqrt(start**2 + 2 * (HP_OFF - HP_ON) * flux / _hp_charge())
    np.testing.assert_allclose(
        waveforms.variable("M1", "resistance"), resistance, rtol=1e-5
    )
    voltage = AMPLITUDE * np.sin(omega * waveforms.times)
    np.testing.assert_allclose(
        waveforms.current("M1", "positive"),
        voltage / resistance,
        rtol=0,
        atol=1e-5 * AMPLITUDE / start,
    )
    state = (resistance - HP_ON) / (HP_OFF - HP_ON)
    assert state.max() == pytest.approx(START + SWING, abs=1e-5)


# At an operating point.


def _fed(level: float, memristor: devices.Device | None = None) -> Component:
    """`memristor`, the VTEAM memristor unless given, fed from `level` through 1 kΩ."""

    @component
    def fed(c: Builder, level: float, memristor: devices.Device) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(level * V), id="V1")
        feed = c.place(devices.resistor(1 * kohm), id="R1")
        cell = c.place(memristor, id="M1")
        c.connect(supply.positive, feed.a, name="in")
        c.connect(feed.b, cell.positive, name="cell")
        c.connect(supply.negative, cell.negative, ground)

    return fed(level, _vteam() if memristor is None else memristor)


def test_an_operating_point_holds_the_state_and_shows_its_rate() -> None:
    """Only time moves a state: at DC it is the one the memristor was made in.

    Its rate there says how fast a transient would move it: zero between its
    thresholds, and VTEAM's law past them.
    """
    simulation = Simulation(_fed(0.0))
    levels = [-3.0, -0.2, 0.2, 3.0]
    sweep = simulation.run(DcSweep("V1", [level * V for level in levels]))
    np.testing.assert_allclose(sweep.state("M1:state"), START, rtol=0, atol=1e-12)
    resistance = ON + (OFF - ON) * START
    np.testing.assert_allclose(sweep.variable("M1", "resistance"), resistance)
    cell = np.array(levels) * resistance / (resistance + 1e3)
    overdrive = np.maximum(np.abs(cell) / THRESHOLD - 1, 0)
    expected = np.sign(cell) * _vteam_rate(1) * overdrive
    np.testing.assert_allclose(sweep.variable("M1", "rate"), expected, rtol=1e-12)


@pytest.mark.parametrize(("level", "bound"), [(3.0, 1.0), (-3.0, 0.0)])
def test_held_past_a_threshold_its_state_runs_to_the_bound(
    level: float, bound: float
) -> None:
    simulation = Simulation(_fed(level))
    waveforms = simulation.run(Transient(1e-3 * s), TIGHT)
    state = waveforms.state("M1:state")
    assert state[0] == START
    assert state[-1] == pytest.approx(bound, abs=1e-6)
    assert np.all((state >= -1e-6) & (state <= 1 + 1e-6))


def test_its_small_signal_resistance_is_the_one_its_state_sets() -> None:
    @component
    def divider(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(0 * V), id="V1")
        feed = c.place(devices.resistor(ON * ohm), id="R1")
        cell = c.place(_vteam(), id="M1")
        c.connect(supply.positive, feed.a, name="in")
        c.connect(feed.b, cell.positive, name="out")
        c.connect(supply.negative, cell.negative, ground)

    simulation = Simulation(divider())
    response = simulation.run(Ac("V1", [1.0 * Hz, 1e6 * Hz]))
    cell = ON + (OFF - ON) * START
    np.testing.assert_allclose(
        np.abs(response.voltage("out")), cell / (cell + ON), rtol=1e-9
    )


def test_a_transients_sensitivities_carry_the_state_it_starts_from() -> None:
    """The state's derivatives over a transient match differences of whole runs.

    An operating point holds the state, and a transient's charges start from it, so the
    derivative of the switched state with respect to the state the memristor was made
    in is carried from the operating point into the transient.
    """
    transient = Transient(20e-6 * s, max_step=0.1e-6 * s)
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
        assert derivative == pytest.approx(difference, rel=1e-5), name


def test_a_state_has_a_panel_of_its_own_and_reads_as_a_number() -> None:
    simulation = Simulation(_driven(_vteam()))
    waveforms = simulation.run(Transient(1 / FREQUENCY * s))
    figure = plot("run", waveforms, ["in", "M1:state"])
    labels = [axes.get_ylabel() for axes in figure.axes]
    assert labels == ["voltage", "state"]
    point = simulation.run(OperatingPoint())
    written = report("point", point, listed=10).describe()
    assert re.search(r"M1:state\s+0\.3\b", written), written

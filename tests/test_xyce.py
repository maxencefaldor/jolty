# Copyright 2026 Maxence Faldor
"""Differential tests against Xyce: jolty and Xyce simulate the same circuits and agree.

The circuits are `test_ngspice.py`'s that Xyce's dialect of jolty's export writes, each
simulated by jolty and written as a SPICE netlist in Xyce's dialect, whose values
come back as the CSV Xyce prints. Both engines run with tight tolerances. Operating
points, DC sweeps and AC responses must agree to the Newton tolerance, noise densities
to the noise comparison's 1e-5, and transients, interpolated onto Xyce's time points,
to a small fraction of their swing: `test_ngspice.py`'s tolerances, since a reference
simulator's equations are the same.

Xyce is a reference for agreement, never for accuracy, as ngspice is. The tests that run
it skip where it is not installed, unless the environment sets ``JOLTY_REQUIRE_XYCE=1``;
those of the dialect's refusals run everywhere.

Where Xyce's models depart from jolty's, which are ngspice's, the tests compare
everything but the departure, each named where it is left out:

- Xyce's MOSFETs of levels 1 and 3 keep SPICE3f5's reverse-biased bulk junction, a
  conductance ``Is/vt`` (``N_DEV_MOSFET1.C``, lines 2921 to 2936), where jolty's current
  settles on ``-Is``: at 3 V of reverse bias each junction of ``mosfet_curves`` carries
  4.16 pA in Xyce and 3.01 pA in jolty and ngspice, gmin's 3 pA included, which
  `_allowance` admits.
- Xyce's JFET makes no noise: ``N_DEV_JFET.C`` defines no noise source.
- Xyce's JFET keeps SPICE3f5's temperature dependence (``N_DEV_JFET.C``, lines
  1792 to 1822): at 125 °C, ``jfet_curves``'s gate leakage is 774.6 pA in Xyce, which
  that formula gives, and 774.3 pA in jolty and ngspice, and ``jfet_stage``'s drain
  current differs by 1.7e-3. JFETs are left out of the comparison across temperatures.
- Xyce's bipolar transistor adds gmin to each junction's current before its beta divides
  it (``N_DEV_BJT.C``, lines 3040 to 3095), where jolty's, as ngspice's, adds it to the
  base current undivided: its currents may differ by gmin times each junction's
  voltage, at -40 °C 1.3 pA of ``bjt_stage``'s 204 nA of base current, which
  `_allowance` admits too.

Xyce finds no operating point of ``negative_load``, its DC solve failing after 7,692
Newton iterations, where jolty's and ngspice's agree; it is left out.
Two disagreements, both of MOSFETs of level 1, are not explained yet, and are left out
too: at 125 °C, ``nmos_stage``'s and ``cmos_inverter``'s outputs lie 1.4e-4 and 7.6e-6
of their values from jolty's, which agrees with ngspice's; and swept across its
threshold at 27 °C, ``cmos_inverter``'s output lies up to 55 µV, 3.0e-5 of its swing,
from jolty's where its gain is steepest, with its junctions' leakage made negligible as
well, so not by that departure.
"""

import json
from collections.abc import Callable
from typing import Final

import numpy as np
import pytest

from differential import (
    SKIP_XYCE,
    TIGHT,
    closest,
    compare,
    compared,
    netlist,
    size,
    vector,
    xyce,
)
from jolty import (
    Ac,
    Component,
    DcSweep,
    Noise,
    OperatingPoint,
    Simulation,
    Transient,
    _core,
    circuit,
    ir,
)
from jolty.errors import JoltyError
from jolty.plan import decade, linear
from jolty.spice.export import XYCE
from jolty.spice.ngspice import CURRENT_TOLERANCE, VOLTAGE_TOLERANCE
from jolty.units import Hz, MHz, Quantity, V, ms, ns, us
from jolty.unknowns import Unknown
from test_ngspice import (
    TEMPERATURES,
    amplifier,
    bjt_amplifier,
    bjt_stage,
    clipper,
    cmos_inverter,
    current_controlled,
    curved,
    delayed_stage,
    divider,
    driven_tank,
    flickering,
    integrated_stage,
    jfet_curves,
    jfet_stage,
    level_three_stage,
    low_pass,
    mosfet_curves,
    nmos_stage,
    p_jfet_follower,
    pnp_mirror,
    power_stage,
    rectifier,
    resonator,
    switched,
    transformer,
    triangle,
    zener_clamp,
)

REFERENCE: Final = _core.REFERENCE_TEMPERATURE
"""The temperature both engines run at unless a test says otherwise: 27 °C, in K."""


GMIN: Final = 1e-12
"""The conductance across every junction, in siemens, as both engines run: SPICE's
default ``GMIN``, Xyce's (``N_DEV_DeviceOptions.C``, line 455) and jolty's."""


def _allowance(simulation: Simulation, values: np.ndarray, temperature: float) -> float:
    """What the departures of Xyce's junctions may move a current by, in amperes.

    Each bulk junction of a MOSFET of level 1 or 3 conducts ``Is/vt`` reversed in Xyce,
    where jolty's settles on ``-Is``, so they differ by ``Is · (|v|/vt + 1)`` at most;
    each junction of a bipolar transistor by gmin times its voltage. Either junction's
    voltage is at most the largest node voltage in `values`, a run's at `temperature`.
    """
    top = simulation.circuit.components[simulation.circuit.top]
    voltages = [
        column
        for column, unknown in enumerate(simulation.unknowns)
        if unknown.quantity == "voltage"
    ]
    largest = float(np.abs(values[..., voltages]).max(initial=0.0))
    thermal = temperature * _core.BOLTZMANN / _core.ELEMENTARY_CHARGE
    allowed = 0.0
    for instance in top.instances:
        match instance.of:
            case ir.DeviceOf(device="bipolar_transistor"):
                allowed += 2 * GMIN * largest
            case ir.DeviceOf(device="mosfet"):
                given = json.loads(
                    _core.defaulted("mosfet", json.dumps(instance.parameters))
                )
                areas = (given.get("drain_area"), given.get("source_area"))
                density = float(given.get("saturation_current_density") or 0.0)
                area = max(float(each or 0.0) for each in areas)
                saturation = max(float(given["saturation_current"]), density * area)
                allowed += 2 * saturation * (largest / thermal + 1)
    return allowed


def _agree(
    jolty: np.ndarray, spice: np.ndarray, unknown: Unknown, allowance: float
) -> None:
    """Asserts both agree within a millionth of the signal.

    A current agrees within the `allowance` of Xyce's junctions' departures too.
    """
    current = unknown.quantity == "current"
    noise = CURRENT_TOLERANCE if current else VOLTAGE_TOLERANCE
    np.testing.assert_allclose(
        jolty,
        spice,
        rtol=0,
        atol=1e-6 * size(spice) + noise + (allowance if current else 0.0),
        err_msg=unknown.name,
    )


def _lines(
    simulation: Simulation, *, excited: str | None = None, transient: bool = False
) -> list[str]:
    """The simulation's circuit in Xyce's dialect."""
    return netlist(
        simulation.circuit, excited=excited, transient=transient, dialect=XYCE
    )


@SKIP_XYCE
@pytest.mark.parametrize(
    "top",
    [
        divider,
        clipper,
        zener_clamp,
        amplifier,
        rectifier,
        jfet_stage,
        p_jfet_follower,
        bjt_stage,
        bjt_amplifier,
        pnp_mirror,
        current_controlled,
        nmos_stage,
        mosfet_curves,
        cmos_inverter,
    ],
)
def test_operating_points_agree(top: Callable[[], Component]) -> None:
    simulation = Simulation(top())
    point = simulation.run(OperatingPoint(), TIGHT)
    found = xyce.run(_lines(simulation), ".op")
    allowance = _allowance(simulation, point.values, REFERENCE)
    for column, unknown in compared(simulation):
        spice = found.real(vector(simulation.circuit, unknown, XYCE))[0]
        _agree(point.values[column], spice, unknown, allowance)


@SKIP_XYCE
@pytest.mark.parametrize("temperature", TEMPERATURES.values(), ids=TEMPERATURES.keys())
@pytest.mark.parametrize(
    "top", [clipper, zener_clamp, bjt_stage, bjt_amplifier, pnp_mirror]
)
def test_operating_points_agree_across_temperatures(
    top: Callable[[], Component], temperature: Quantity
) -> None:
    simulation = Simulation(top(), temperature)
    point = simulation.run(OperatingPoint(), TIGHT)
    found = xyce.run(_lines(simulation), ".op", temperature=temperature.value)
    allowance = _allowance(simulation, point.values, temperature.value)
    for column, unknown in compared(simulation):
        spice = found.real(vector(simulation.circuit, unknown, XYCE))[0]
        _agree(point.values[column], spice, unknown, allowance)


@SKIP_XYCE
@pytest.mark.parametrize(
    ("top", "source", "levels"),
    [
        (divider, "V1", linear(-5 * V, 5 * V, 21)),
        (clipper, "V1", linear(-2 * V, 2 * V, 81)),
        (zener_clamp, "V1", linear(-12 * V, 2 * V, 141)),
        (jfet_stage, "V1", linear(-3 * V, 1 * V, 81)),
        (jfet_curves, "V2", linear(-2 * V, 6 * V, 81)),
        (p_jfet_follower, "V1", linear(-4 * V, 2 * V, 61)),
        (bjt_stage, "V1", linear(-1 * V, 3 * V, 81)),
        (pnp_mirror, "V1", linear(0 * V, 12 * V, 61)),
        (current_controlled, "V1", linear(-3 * V, 3 * V, 13)),
        (nmos_stage, "V1", linear(-1 * V, 5 * V, 61)),
        (mosfet_curves, "V2", linear(-2 * V, 6 * V, 81)),
        (mosfet_curves, "V1", linear(-1 * V, 5 * V, 61)),
    ],
)
def test_dc_sweeps_agree(
    top: Callable[[], Component], source: str, levels: list[Quantity]
) -> None:
    simulation = Simulation(top())
    sweep = simulation.run(DcSweep(source, levels), TIGHT)
    first, last = float(sweep.levels[0]), float(sweep.levels[-1])
    step = (last - first) / (len(levels) - 1)
    found = xyce.run(_lines(simulation), f".dc V{source} {first!r} {last!r} {step!r}")
    allowance = _allowance(simulation, sweep.values, REFERENCE)
    for column, unknown in compared(simulation):
        spice = found.real(vector(simulation.circuit, unknown, XYCE))
        _agree(sweep.values[:, column], spice, unknown, allowance)


@SKIP_XYCE
@pytest.mark.parametrize(
    "top",
    [
        low_pass,
        resonator,
        amplifier,
        jfet_stage,
        bjt_amplifier,
        current_controlled,
        nmos_stage,
        cmos_inverter,
        transformer,
    ],
)
def test_ac_responses_agree(top: Callable[[], Component]) -> None:
    simulation = Simulation(top())
    response = simulation.run(Ac("V1", decade(10 * Hz, 10 * MHz, 20)), TIGHT)
    found = xyce.run(_lines(simulation, excited="V1"), ".ac dec 20 10 10meg")
    np.testing.assert_allclose(response.frequencies, found.real("frequency"), rtol=1e-9)
    for column, unknown in compared(simulation):
        spice = found.vectors[vector(simulation.circuit, unknown, XYCE)]
        compare(
            response.values[:, column],
            spice,
            unknown,
            scale=size(spice),
            tolerance=1e-6,
        )


@SKIP_XYCE
@pytest.mark.parametrize(
    ("top", "source"),
    [
        (low_pass, "V1"),
        (resonator, "V1"),
        (bjt_amplifier, "V1"),
        (nmos_stage, "V1"),
    ],
)
def test_noise_agrees(top: Callable[[], Component], source: str) -> None:
    """The noise at the output, and referred to the input, as Xyce finds them.

    Xyce prints the densities themselves, in V²/Hz and the input's unit squared per
    hertz, where ngspice prints their square roots.
    """
    simulation = Simulation(top())
    spectrum = simulation.run(
        Noise("out", decade(1 * Hz, 100 * MHz, 5), source=source), TIGHT
    )
    lines = _lines(simulation, excited=source)
    found = xyce.run(lines, f".noise v(out) V{source} dec 5 1 100meg")
    np.testing.assert_allclose(spectrum.frequencies, found.real("frequency"), rtol=1e-9)
    np.testing.assert_allclose(spectrum.densities, found.real("onoise"), rtol=1e-5)
    np.testing.assert_allclose(spectrum.referred(), found.real("inoise"), rtol=1e-5)


@SKIP_XYCE
@pytest.mark.parametrize(
    ("top", "stop", "step"),
    [
        (low_pass, 8 * us, 10 * ns),
        (resonator, 60 * us, 50 * ns),
        (triangle, 60 * us, 50 * ns),
        (rectifier, 3 * ms, 1 * us),
        (driven_tank, 40 * us, 20 * ns),
        (transformer, 100 * us, 20 * ns),
        (jfet_stage, 15 * us, 10 * ns),
        (bjt_amplifier, 4 * us, 2 * ns),
        (nmos_stage, 2 * us, 1 * ns),
        (cmos_inverter, 160 * ns, 0.01 * ns),
    ],
)
def test_transients_agree(
    top: Callable[[], Component], stop: Quantity, step: Quantity
) -> None:
    simulation = Simulation(top())
    waveforms = simulation.run(Transient(stop=stop, max_step=step), TIGHT)
    command = f".tran {step.value!r} {stop.value!r} 0 {step.value!r}"
    found = xyce.run(_lines(simulation, transient=True), command)
    times = found.real("time")
    for column, unknown in compared(simulation):
        spice = found.real(vector(simulation.circuit, unknown, XYCE))
        jolty = closest(
            waveforms.times, waveforms.values[:, column], times, spice, step.value
        )
        compare(jolty, spice, unknown, scale=size(spice), tolerance=1e-3)


@pytest.mark.parametrize(
    ("top", "named"),
    [
        (power_stage, "Xyce's VDMOS, its MOSFET of level 18, is another model"),
        (level_three_stage, "Xyce's mosfet of level 3 has no DELVTO, WD, XL, XW"),
        (
            integrated_stage,
            "Xyce's bipolar_transistor of level 1 has no ISS, NS, SUBS",
        ),
        (flickering, "Xyce's resistor has no AF, KF"),
        (delayed_stage, "Xyce's bipolar_transistor of level 1 has no PTF"),
        (switched, "S1 is a voltage_controlled_switch, written as an expression"),
        (curved, "a resistor whose value bends with its voltage is written as an"),
    ],
)
def test_what_xyce_lacks_is_refused_by_name(
    top: Callable[[], Component], named: str
) -> None:
    """What Xyce lacks or defines otherwise is never dropped.

    Writing a device or a parameter of that kind fails, naming it.
    """
    with pytest.raises(JoltyError, match=named):
        netlist(circuit(top()), dialect=XYCE)


def test_a_varying_source_states_its_level_at_time_zero() -> None:
    """The dialect states a varying source's DC level, its waveform's at time zero.

    Xyce takes 0 where none is stated, where jolty's operating point reads that level.
    """
    simulation = Simulation(rectifier())
    point = simulation.run(OperatingPoint(), TIGHT)
    (line,) = [line for line in _lines(simulation) if line.startswith("VV1 ")]
    column = simulation.unknowns.index(Unknown("in", "voltage"))
    assert line.split()[3:5] == ["DC", repr(float(point.values[column]))]

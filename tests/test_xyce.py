# Copyright 2026 Maxence Faldor
"""Differential tests against Xyce: jolty and Xyce simulate the same circuits and agree.

Each circuit from `test_ngspice.py` that Xyce's export dialect supports is simulated by
jolty and exported as a Xyce SPICE netlist, whose results return as CSV. Both engines
run with tight tolerances matching `test_ngspice.py`. Operating points, DC sweeps, and
AC responses must agree to the Newton tolerance, noise densities to 1e-5, and transients
(interpolated onto Xyce time points) to a small fraction of their swing.

Like ngspice, Xyce serves as a reference for agreement rather than ground truth. Tests
that invoke Xyce skip when it is not installed unless ``JOLTY_REQUIRE_XYCE=1`` is set.
Dialect-refusal tests always run.

Where Xyce models diverge from jolty and ngspice, the tests compare all remaining
analyses and document each exclusion:

- Level-1 and level-3 MOSFETs in Xyce keep SPICE3f5's reverse-biased bulk junction
  conductance ``Is/vt`` (``N_DEV_MOSFET1.C``, lines 2921 to 2936), whereas jolty settles
  at ``-Is``. At 3 V reverse bias, each junction in ``mosfet_curves`` carries 4.16 pA in
  Xyce versus 3.01 pA in jolty and ngspice (including 3 pA from gmin), as `_allowance`
  accounts for.
- Xyce JFETs produce no noise: ``N_DEV_JFET.C`` defines no noise source.
- Xyce JFETs keep SPICE3f5 temperature scaling (``N_DEV_JFET.C``, lines 1792 to 1822).
  At 125 °C, ``jfet_curves`` gate leakage is 774.6 pA in Xyce versus 774.3 pA in jolty
  and ngspice, and ``jfet_stage`` drain current differs by 1.7e-3. JFETs are excluded
  from cross-temperature comparisons.
- Xyce bipolar transistors add gmin to each junction current before dividing by beta
  (``N_DEV_BJT.C``, lines 3040 to 3095), whereas jolty and ngspice add gmin directly to
  base current. Currents can differ by gmin times junction voltage (1.3 pA of 204 nA
  base current in ``bjt_stage`` at -40 °C), which `_allowance` accounts for.

Xyce fails to find a DC operating point for ``negative_load`` after 7,692 Newton
iterations while jolty and ngspice agree, so that circuit is excluded. Two unexplained
level-1 MOSFET disagreements are also excluded. At 125 °C, ``nmos_stage`` and
``cmos_inverter`` outputs differ by 1.4e-4 and 7.6e-6 relative to jolty (which agrees
with ngspice). Swept across threshold at 27 °C, ``cmos_inverter`` differs by up to 55 µV
(3.0e-5 of swing) at steepest gain even with negligible junction leakage.
"""

import json
from collections.abc import Callable
from typing import Final

import numpy as np
import pytest

from differential import (
    SKIP_XYCE,
    assert_ac_agrees,
    assert_dc_sweep_agrees,
    assert_noise_agrees,
    assert_op_agrees,
    assert_transient_agrees,
    dc_command,
    netlist,
    transient_command,
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
from jolty.plan import TIGHT, decade, linear
from jolty.spice.export import XYCE
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
    """Returns the current offset in amperes caused by Xyce junction model differences.

    Level-1 and level-3 MOSFET bulk junctions conduct ``Is/vt`` in reverse bias in Xyce
    versus ``-Is`` in jolty, differing by at most ``Is · (|v|/vt + 1)``. Bipolar
    junctions differ by gmin times junction voltage. Both voltages are bounded by the
    maximum node voltage in `values` at `temperature`.
    """
    top = simulation.circuit.components[simulation.circuit.top]
    voltages = [
        column
        for column, unknown in enumerate(simulation.unknowns)
        if unknown.nature == "voltage"
    ]
    largest = float(np.abs(values[..., voltages]).max(initial=0.0))
    thermal = temperature * _core.BOLTZMANN / _core.ELEMENTARY_CHARGE
    allowed = 0.0
    for instance in top.instances:
        match instance.of:
            case ir.DeviceOf(model="bipolar_transistor"):
                allowed += 2 * GMIN * largest
            case ir.DeviceOf(model="mosfet"):
                given = json.loads(
                    _core.defaulted("mosfet", json.dumps(instance.parameters))
                )
                areas = (given.get("drain_area"), given.get("source_area"))
                density = float(given.get("saturation_current_density") or 0.0)
                area = max(float(each or 0.0) for each in areas)
                saturation = max(float(given["saturation_current"]), density * area)
                allowed += 2 * saturation * (largest / thermal + 1)
    return allowed


def _lines(
    simulation: Simulation, *, excited: str | None = None, transient: bool = False
) -> list[str]:
    """Returns the circuit netlist formatted in the Xyce SPICE dialect."""
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
    assert_op_agrees(simulation, point, found, allowance=allowance)


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
    assert_op_agrees(simulation, point, found, allowance=allowance)


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
    found = xyce.run(_lines(simulation), dc_command(source, sweep, dialect=XYCE))
    allowance = _allowance(simulation, sweep.values, REFERENCE)
    assert_dc_sweep_agrees(simulation, sweep, found, allowance=allowance)


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
    # Log-decade grid stepping rounding across Xyce raw output vs jolty (rtol=1e-9).
    assert_ac_agrees(simulation, response, found, frequency_tolerance=1e-9)


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
    """Verifies output and input-referred noise density agreement against Xyce.

    Xyce reports power spectral densities directly (V²/Hz and input unit squared per
    hertz), whereas ngspice prints their square roots.
    """
    simulation = Simulation(top())
    spectrum = simulation.run(
        Noise("out", decade(1 * Hz, 100 * MHz, 5), source=source), TIGHT
    )
    lines = _lines(simulation, excited=source)
    found = xyce.run(lines, f".noise v(out) V{source} dec 5 1 100meg")
    assert_noise_agrees(spectrum, found)


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
    command = transient_command(stop.value, step.value, dialect=XYCE)
    found = xyce.run(_lines(simulation, transient=True), command)
    assert_transient_agrees(simulation, waveforms, found, window=step.value)


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
    """Verifies that unsupported Xyce features raise errors naming the feature.

    Exporting an unsupported device or parameter fails and names the offending item.
    """
    with pytest.raises(JoltyError, match=named):
        netlist(circuit(top()), dialect=XYCE)


def test_a_varying_source_states_its_level_at_time_zero() -> None:
    """Verifies that Xyce export emits a varying source's time-zero DC level.

    Xyce defaults to 0 V when no DC level is printed, whereas jolty evaluates the
    waveform at time zero for the operating point.
    """
    simulation = Simulation(rectifier())
    point = simulation.run(OperatingPoint(), TIGHT)
    (line,) = [line for line in _lines(simulation) if line.startswith("VV1 ")]
    column = simulation.unknowns.index(Unknown("in", "voltage"))
    assert line.split()[3:5] == ["DC", repr(float(point.values[column]))]


@SKIP_XYCE
def test_a_timeout_keeps_what_xyce_printed() -> None:
    """Verifies that a timed-out Xyce run preserves partial transient progress output.

    A 10 µs transient capped at 1 ps steps requires ten million steps, far exceeding
    two seconds, and Xyce logs each completed percentage before termination.
    """
    lines = ["rc", "R1 in out 1k", "C1 out 0 1n", "V1 in 0 PULSE(0 1 0 1n 1n 1u 2u)"]
    with pytest.raises(
        xyce.XyceTimeoutError, match=r"(?s)more than 2 s.*Percent complete"
    ):
        xyce.run(lines, ".tran 1p 10u 0 1p", timeout=2)

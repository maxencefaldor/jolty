# Copyright 2026 Maxence Faldor

import math
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jolty import (
    Ac,
    Builder,
    Corner,
    DcSweep,
    DimensionError,
    ElaborationError,
    JoltyError,
    Noise,
    OperatingPoint,
    OperatingPoints,
    Options,
    Plan,
    PoleZero,
    Role,
    Sensitivity,
    Simulation,
    SimulationError,
    Stability,
    Terminal,
    Transient,
    _core,
    component,
    devices,
    results,
    simulate,
)
from jolty.plan import Explicit, Variant, decade, linear
from jolty.simulate import Joined, Logged
from jolty.specs import crossover, gain_margin, phase_margin
from jolty.units import (
    A,
    F,
    GHz,
    H,
    Hz,
    K,
    MHz,
    Quantity,
    V,
    celsius,
    fA,
    kHz,
    kohm,
    mA,
    mH,
    ms,
    nF,
    ns,
    ohm,
    uA,
    uF,
    um,
    us,
)
from jolty.waveforms import Piecewise, Sine


@component
def divider(c: Builder, level: Quantity, upper: Quantity, lower: Quantity) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(level), id="V1")
    top = c.place(devices.resistor(upper), id="R1")
    bottom = c.place(devices.resistor(lower), id="R2")
    c.connect(source.positive, top.a, name="in")
    c.connect(top.b, bottom.a, name="out")
    c.connect(bottom.b, source.negative, ground)


@component
def low_pass(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    step = Piecewise([(0 * us, 0 * V), (1e-3 * us, 1 * V)])
    source = c.place(devices.voltage_source(step), id="V1")
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
    capacitor = c.place(devices.capacitor(1 * nF), id="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def diode_load(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), id="V1")
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
    diode = c.place(devices.diode(), id="D1")
    c.connect(source.positive, resistor.a)
    c.connect(resistor.b, diode.anode, name="anode")
    c.connect(diode.cathode, source.negative, ground)


def test_an_operating_point_solves_kirchhoffs_laws() -> None:
    simulation = Simulation(divider(5 * V, 1 * kohm, 4 * kohm))
    assert simulation.devices == 3
    assert [unknown.name for unknown in simulation.unknowns] == [
        "in",
        "out",
        "V1:current",
    ]
    point = simulation.run(OperatingPoint())
    assert point.strategy == "direct"
    assert point.voltage("out") == pytest.approx(4.0)
    assert point.voltage("in", reference="out") == pytest.approx(1.0)
    assert point.current("V1") == pytest.approx(-1e-3)


@settings(max_examples=50, deadline=None)
@given(
    level=st.floats(min_value=-100, max_value=100),
    upper=st.floats(min_value=1e-3, max_value=1e9),
    lower=st.floats(min_value=1e-3, max_value=1e9),
)
def test_a_divider_divides_at_any_scale(
    level: float, upper: float, lower: float
) -> None:
    circuit = divider(level * V, upper * ohm, lower * ohm)
    point = Simulation(circuit).run(OperatingPoint())
    expected = level * lower / (upper + lower)
    assert point.voltage("out") == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_sensitivities_are_the_derivatives_of_a_divider() -> None:
    simulation = Simulation(divider(5 * V, 1 * kohm, 4 * kohm))
    found = simulation.run(Sensitivity(["out", "V1:current"]))
    assert [(p.instance, p.name, p.value) for p in found.parameters] == [
        ("V1", "waveform.dc", 5.0),
        ("R1", "resistance", 1e3),
        ("R2", "resistance", 4e3),
    ]
    level, upper, lower = 5.0, 1e3, 4e3
    total = upper + lower
    expected = {
        ("V1", "waveform.dc"): (lower / total, -1 / total),
        ("R1", "resistance"): (-level * lower / total**2, level / total**2),
        ("R2", "resistance"): (level * upper / total**2, level / total**2),
    }
    for (instance, name), (out, current) in expected.items():
        assert found.derivative("out", instance, name) == pytest.approx(out, rel=1e-6)
        assert found.derivative("V1:current", instance, name) == pytest.approx(
            current, rel=1e-6
        )
    assert isinstance(found.result, results.OperatingPoint)
    assert found.result.voltage("out") == pytest.approx(4.0)
    np.testing.assert_allclose(found.gradient("out"), [0.8, -8e-4, 2e-4], rtol=1e-6)
    with pytest.raises(KeyError, match="no output named 'in'"):
        found.gradient("in")
    with pytest.raises(KeyError, match="close names: R1 resistance"):
        found.derivative("out", "R1", "resistanc")
    with pytest.raises(SimulationError, match="no unknown named `nowhere`"):
        simulation.run(Sensitivity(["nowhere"]))


STAGES = ((100 * kohm, 1e3), (10 * kohm, 1e6), (1 * kohm, 1e7))
"""Each stage of the amplifier: its load, which its 1 mS turns into its gain, and its
pole, in hertz."""


@component
def follower(c: Builder) -> None:
    """An amplifier fed back whole to its inverting input through the probe ``VP``.

    Three stages of 1 mS each drive a load and a capacitor, then a buffer the output.
    """
    ground = c.port("ground", Role.GROUND)
    level = c.place(devices.voltage_source(1 * V), id="VIN")
    c.connect(level.negative, ground)
    stages = [
        c.place(devices.voltage_controlled_current_source(1 * mA / V), id=f"G{index}")
        for index in range(1, len(STAGES) + 1)
    ]
    driving = level.positive
    for index, (stage, (load, pole)) in enumerate(zip(stages, STAGES, strict=True), 1):
        resistor = c.place(devices.resistor(load), id=f"R{index}")
        capacitance = 1 / (2 * math.pi * pole * load.value) * F
        capacitor = c.place(devices.capacitor(capacitance), id=f"C{index}")
        c.connect(stage.control_positive, driving)
        c.connect(stage.positive, resistor.b, capacitor.b, ground)
        c.connect(stage.negative, resistor.a, capacitor.a, name=f"n{index}")
        driving = stage.negative
    for stage in stages[1:]:
        c.connect(stage.control_negative, ground)
    buffer = c.place(devices.voltage_controlled_voltage_source(1.0), id="E1")
    c.connect(buffer.control_positive, driving)
    c.connect(buffer.control_negative, buffer.negative, ground)
    probe = c.place(devices.voltage_source(0 * V), id="VP")
    c.connect(buffer.positive, probe.negative, name="out")
    c.connect(probe.positive, stages[0].control_negative, name="fb")


def open_loop(frequency: float) -> complex:
    """The amplifier's gain, which the loop gain is."""
    gain = complex(1.0)
    for load, pole in STAGES:
        gain *= 1e-3 * load.value / (1 + 1j * frequency / pole)
    return gain


def solved(function: Callable[[float], float], low: float, high: float) -> float:
    """Where `function`, of opposite signs at `low` and `high`, crosses zero."""
    for _ in range(200):
        middle = math.sqrt(low * high)
        if (function(middle) > 0) == (function(low) > 0):
            low = middle
        else:
            high = middle
    return low


def test_a_stability_analysis_finds_the_loop_gain_and_its_margins() -> None:
    simulation = Simulation(follower())
    loop = simulation.run(Stability("VP", decade(1 * Hz, 1 * GHz, 200)))
    expected = np.array([open_loop(frequency) for frequency in loop.frequencies])
    np.testing.assert_allclose(loop.gains, expected, rtol=1e-6)
    assert loop.probe == "VP"
    assert loop.operating_point.voltage("out") == pytest.approx(1.0, rel=1e-3)

    def phase_of(frequency: float) -> float:
        return sum(math.degrees(math.atan(frequency / pole)) for _, pole in STAGES)

    unity = solved(lambda f: abs(open_loop(f)) - 1, 1.0, 1e9)
    inverted = solved(lambda f: phase_of(f) - 180, 1.0, 1e9)
    margins = {
        crossover("loop"): unity,
        phase_margin("loop"): 180 - phase_of(unity),
        gain_margin("loop"): -20 * math.log10(abs(open_loop(inverted))),
    }
    for measurement, wanted in margins.items():
        assert measurement.read(loop).value == pytest.approx(wanted, rel=1e-3)


def test_a_probe_is_a_voltage_source_off_the_ground() -> None:
    simulation = Simulation(follower())
    for probe in ("VIN", "G1", "R1"):
        with pytest.raises(SimulationError, match=f"`{probe}` is not a voltage source"):
            simulation.run(Stability(probe, [1 * kHz]))


FOUR_KT = 4 * _core.BOLTZMANN * _core.REFERENCE_TEMPERATURE
"""`4kT` at 27 °C, in joules."""


def test_a_noise_analysis_finds_each_resistors_noise_and_refers_it_back() -> None:
    upper, lower = 1e3, 3e3
    simulation = Simulation(divider(5 * V, upper * ohm, lower * ohm))
    spectrum = simulation.run(Noise("out", decade(10 * Hz, 100 * kHz, 10), source="V1"))
    parallel = upper * lower / (upper + lower)
    np.testing.assert_allclose(spectrum.densities, FOUR_KT * parallel, rtol=1e-9)
    np.testing.assert_allclose(
        spectrum.contribution("R1", "thermal"),
        FOUR_KT / upper * parallel**2,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        spectrum.referred(), FOUR_KT * parallel / 0.75**2, rtol=1e-9
    )
    assert spectrum.input is not None
    np.testing.assert_allclose(spectrum.input.gains, 0.75, rtol=1e-9)
    assert spectrum.rms() == pytest.approx(
        math.sqrt(FOUR_KT * parallel * (1e5 - 10)), rel=1e-9
    )
    assert spectrum.density(5e3) == pytest.approx(FOUR_KT * parallel, rel=1e-9)
    across = simulation.run(Noise("in", [1 * kHz], reference="out"))
    assert across.densities[0] == pytest.approx(FOUR_KT * parallel, rel=1e-9)
    current = simulation.run(Noise("V1:current", [1 * kHz]))
    assert current.quantity == "current"
    assert current.densities[0] == pytest.approx(FOUR_KT / (upper + lower), rel=1e-9)
    with pytest.raises(KeyError, match="no noise source named 'R3'"):
        spectrum.contribution("R3")
    with pytest.raises(ValueError, match="named no source"):
        current.referred()


def test_a_capacitor_holds_its_resistors_noise_to_kt_over_c() -> None:
    spectrum = Simulation(low_pass()).run(Noise("out", decade(1 * Hz, 100 * GHz, 50)))
    capacitance = 1e-9
    assert spectrum.rms() ** 2 == pytest.approx(FOUR_KT / 4 / capacitance, rel=1e-3)


def test_a_noise_analysis_rejects_what_it_cannot_measure() -> None:
    simulation = Simulation(divider(5 * V, 1 * kohm, 1 * kohm))
    with pytest.raises(SimulationError, match="no unknown named `nowhere`"):
        simulation.run(Noise("nowhere", [1 * kHz]))
    with pytest.raises(SimulationError, match="frequencies must be positive"):
        simulation.run(Noise("out", [0 * Hz]))
    with pytest.raises(SimulationError, match="`R1` is not an independent source"):
        simulation.run(Noise("out", [1 * kHz], source="R1"))
    with pytest.raises(DimensionError, match="a frequency"):
        simulation.run(Noise("out", [1 * V]))


@component
def latch(c: Builder) -> None:
    """Two CMOS inverters, each driving the other's input."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(3.3 * V), id="VDD")
    c.connect(supply.negative, ground)
    nets: dict[str, list[Terminal]] = {"a": [], "b": []}
    for index, (input_net, output_net) in enumerate((("a", "b"), ("b", "a")), 1):
        for channel, threshold, rail in (
            ("p", -0.7, supply.positive),
            ("n", 0.7, ground),
        ):
            transistor = c.place(
                devices.mosfet(
                    channel=channel,
                    threshold_voltage=threshold * V,
                    transconductance_parameter=100 * uA / V**2,
                    channel_length_modulation=0.02 / V,
                    width=10 * um,
                    length=1 * um,
                ),
                id=f"M{channel.upper()}{index}",
            )
            c.connect(transistor.source, transistor.bulk, rail)
            nets[input_net].append(transistor.gate)
            nets[output_net].append(transistor.drain)
    for name, terminals in nets.items():
        c.connect(*terminals, name=name)


def test_a_latch_has_two_states_and_a_balance_between_them() -> None:
    found = Simulation(latch()).run(OperatingPoints())
    assert len(found.points) == 3
    assert sorted(found.voltages("a").tolist()) == [
        pytest.approx(0.0, abs=1e-6),
        pytest.approx(1.65, abs=1e-6),
        pytest.approx(3.3, abs=1e-6),
    ]
    np.testing.assert_allclose(
        found.voltages("a") + found.voltages("b"), 3.3, atol=1e-6
    )
    single = Simulation(low_pass()).run(OperatingPoints())
    assert len(single.points) == 1
    with pytest.raises(ValueError, match="finds 1 or more, not 0"):
        OperatingPoints(most=0)


def test_a_low_pass_has_one_real_pole_and_no_zero() -> None:
    simulation = Simulation(low_pass())
    found = simulation.run(PoleZero(source="V1", output="out"))
    np.testing.assert_allclose(found.poles, [-1e6], rtol=1e-9)
    assert found.zeros is not None
    assert found.zeros.size == 0
    assert results.dampings(found.poles).tolist() == [1.0]
    alone = simulation.run(PoleZero())
    assert alone.zeros is None
    with pytest.raises(SimulationError, match="both or neither"):
        simulation.run(PoleZero(source="V1"))


def test_dense_matrices_past_the_memory_fail_before_they_are_made() -> None:
    simulation = Simulation(low_pass())
    tiny = Options(memory=1e-9)
    with pytest.raises(
        SimulationError,
        match=r"pole-zero analysis's dense matrices of 3 rows would hold .* GiB, more "
        r"than its `memory` option's 1e-9 GiB",
    ):
        simulation.run(PoleZero(), tiny)
    periodic = Transient(stop=1 * us, periodic=True)
    with pytest.raises(SimulationError, match="periodic steady state's dense matrices"):
        simulation.run(periodic, tiny)


def test_an_excess_phase_delays_the_collector_current_through_two_poles() -> None:
    """With its terminals held, a transistor's only dynamics are its delay's filter.

    Its two poles are the roots of s² td² + 3 s td + 3, and the collector current's
    response to the base is the transconductance through 3 / (s² td² + 3 s td + 3).
    """
    delay = 30 * math.pi / 180 * 0.4e-9

    @component
    def held(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        base = c.place(devices.voltage_source(0.7 * V), id="VB")
        collector = c.place(devices.voltage_source(3 * V), id="VC")
        transistor = c.place(
            devices.bipolar_transistor(
                saturation_current=1 * fA,
                forward_transit_time=0.4 * ns,
                excess_phase=30.0,
            ),
            id="Q1",
        )
        c.connect(base.positive, transistor.base, name="base")
        c.connect(collector.positive, transistor.collector, name="collector")
        c.connect(base.negative, collector.negative, transistor.emitter, ground)

    simulation = Simulation(held())
    poles = simulation.run(PoleZero()).poles
    expected = (-3 + np.array([-1, 1]) * 1j * math.sqrt(3)) / (2 * delay)
    np.testing.assert_allclose(np.sort_complex(poles), expected, rtol=1e-9)
    point = simulation.run(OperatingPoint())
    thermal_voltage = 1.380649e-23 * 300.15 / 1.602176634e-19
    transconductance = -point.current("VC") / thermal_voltage
    frequencies = [1 * MHz, 100 * MHz, 1 * GHz]
    response = simulation.run(Ac("VB", frequencies))
    s = 2j * np.pi * np.array([frequency.value for frequency in frequencies])
    bessel = 3 / ((s * delay) ** 2 + 3 * s * delay + 3)
    np.testing.assert_allclose(
        -response.current("VC"), transconductance * bessel, rtol=1e-6
    )


def test_sensitivities_follow_a_sweep_a_response_and_a_transient() -> None:
    upper, lower = 1e3, 4e3
    levels = linear(-2 * V, 2 * V, 5)
    swept = Simulation(divider(5 * V, upper * ohm, lower * ohm)).run(
        Sensitivity(["out"], of=DcSweep("V1", levels))
    )
    assert isinstance(swept.result, results.Sweep)
    np.testing.assert_allclose(
        swept.derivative("out", "R1", "resistance"),
        -swept.result.levels * lower / (upper + lower) ** 2,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        swept.derivative("out", "V1", "waveform.dc"), np.zeros(5), atol=1e-12
    )
    resistance, capacitance = 1e3, 1e-9
    frequencies = decade(1 * kHz, 10 * MHz, 5)
    filtered = Simulation(low_pass()).run(
        Sensitivity(["out"], of=Ac("V1", frequencies))
    )
    assert isinstance(filtered.result, results.Response)
    omega = 2 * np.pi * filtered.result.frequencies
    corner = 1 + 1j * omega * resistance * capacitance
    np.testing.assert_allclose(
        filtered.derivative("out", "C1", "capacitance"),
        -1j * omega * resistance / corner**2,
        rtol=1e-6,
    )
    charging = Simulation(low_pass()).run(
        Sensitivity(["out"], of=Transient(5 * us, max_step=1e-3 * us))
    )
    assert isinstance(charging.result, results.Waveforms)
    # After the source's ramp of 1 ns to 1 V, out is 1 - k exp(-t / tau).
    ramp, tau = 1e-9, resistance * capacitance
    grown = math.exp(ramp / tau)
    k = tau / ramp * (grown - 1)
    k_slope = (grown - 1) / ramp - grown / tau
    times = charging.result.times
    after = times >= ramp
    decay = np.exp(-times[after] / tau)
    expected = -capacitance * (k_slope + k * times[after] / tau**2) * decay
    found = charging.derivative("out", "R1", "resistance")
    assert isinstance(found, np.ndarray)
    np.testing.assert_allclose(
        found[after], expected, atol=1e-4 * np.abs(expected).max()
    )


EVERYTHING = ("**",)
"""Options that record every instance."""


def test_results_read_the_current_into_every_terminal() -> None:
    simulation = Simulation(divider(5 * V, 1 * kohm, 4 * kohm))
    point = simulation.run(OperatingPoint(record=EVERYTHING))
    assert point.recording.terminals == (
        "V1.positive",
        "V1.negative",
        "R1.a",
        "R1.b",
        "R2.a",
        "R2.b",
    )
    assert point.current("R1", "a") == pytest.approx(1e-3)
    assert point.current("R2", "b") == pytest.approx(-1e-3)
    assert point.current("V1", "positive") == pytest.approx(-1e-3)
    assert point.charge("R1", "a") == 0
    with pytest.raises(
        KeyError,
        match=r"^\"'R1', which the result recorded, has no terminal named 'R1\.aa'; "
        r"close names: R1\.a, R2\.a, R1\.b\"$",
    ):
        point.current("R1", "aa")
    sweep = simulation.run(DcSweep("V1", linear(0 * V, 5 * V, 3), record=EVERYTHING))
    np.testing.assert_allclose(sweep.current("R2", "a"), [0, 5e-4, 1e-3], atol=1e-15)


@component
def common_source(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(3 * V), id="VDD")
    gate = c.place(devices.voltage_source(1 * V), id="VG")
    load = c.place(devices.resistor(10 * kohm), id="RD")
    transistor = c.place(
        devices.mosfet(
            threshold_voltage=0.4 * V,
            transconductance_parameter=2e-4 * A / V**2,
            width=4 * um,
            length=2 * um,
        ),
        id="M1",
    )
    c.connect(supply.positive, load.a, name="vdd")
    c.connect(load.b, transistor.drain, name="out")
    c.connect(gate.positive, transistor.gate, name="in")
    c.connect(
        transistor.source, transistor.bulk, supply.negative, gate.negative, ground
    )


def test_results_read_each_device_s_small_signal_model_and_variables() -> None:
    simulation = Simulation(common_source())
    recorded = ["M1", "RD"]
    point = simulation.run(OperatingPoint(record=recorded))
    beta = 2e-4 * 4 / 2
    assert point.conductance("M1.drain", "M1.gate") == pytest.approx(beta * 0.6)
    assert point.conductance("RD.a", "RD.b") == pytest.approx(-1e-4)
    assert point.conductance("M1.gate", "M1.drain") == 0
    assert point.variable("M1", "threshold_voltage") == pytest.approx(0.4)
    assert point.variable("M1", "saturation_voltage") == pytest.approx(0.6)
    with pytest.raises(KeyError, match="are unknowns of two devices"):
        point.conductance("M1.drain", "RD.a")
    with pytest.raises(
        KeyError,
        match=r"recorded, has no variable named 'threshold'; close names: M1 t",
    ):
        point.variable("M1", "threshold")
    with pytest.raises(KeyError, match=r"recorded, has no unknown named 'M1\.drian'"):
        point.conductance("M1.drian", "M1.gate")
    with pytest.raises(
        KeyError,
        match=r"no device unknown named 'VDD\.positive'.*; record its instance, as an "
        r"analysis's or a plan's record=\['VDD'\] does",
    ):
        point.conductance("VDD.positive", "VDD.negative")
    sweep = simulation.run(DcSweep("VG", linear(0 * V, 1 * V, 3), record=recorded))
    np.testing.assert_allclose(
        sweep.variable("M1", "saturation_voltage"), [0, 0.1, 0.6], atol=1e-12
    )
    np.testing.assert_allclose(
        sweep.conductance("M1.drain", "M1.gate"), [0, beta * 0.1, beta * 0.6]
    )


@component
def two_dividers(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(4 * V), id="V1")
    first = c.place(divider_cell(), id="X1")
    second = c.place(divider_cell(), id="X2")
    c.connect(source.positive, first.top, second.top, name="in")
    c.connect(source.negative, first.bottom, second.bottom, ground)


@component
def divider_cell(c: Builder) -> None:
    top = c.port("top")
    bottom = c.port("bottom")
    upper = c.place(devices.resistor(1 * kohm), id="R1")
    lower = c.place(devices.resistor(1 * kohm), id="R2")
    c.connect(top, upper.a)
    c.connect(upper.b, lower.a, name="mid")
    c.connect(lower.b, bottom)


def test_a_component_instance_s_terminals_sum_the_currents_inside_it() -> None:
    simulation = Simulation(two_dividers())
    point = simulation.run(OperatingPoint(record=["X1", "X2"]))
    assert point.recording.composites == ("X1.top", "X1.bottom", "X2.top", "X2.bottom")
    assert point.recording.devices == ()
    assert point.current("X1", "top") == pytest.approx(2e-3)
    assert point.current("X2", "bottom") == pytest.approx(-2e-3)
    with pytest.raises(
        KeyError, match=r"'X1', which the result recorded, has no terminal named 'X1\.t"
    ):
        point.current("X1", "tops")


def test_a_result_records_nothing_beyond_the_unknowns_by_default() -> None:
    simulation = Simulation(divider(5 * V, 1 * kohm, 4 * kohm))
    for record in (None, []):
        point = simulation.run(OperatingPoint(record=record))
        assert point.recording == results.Recording()
        assert point.currents.size == point.conductances.size == 0
        assert point.current("V1") == pytest.approx(-1e-3)
        with pytest.raises(
            KeyError,
            match=(
                r"recorded no terminal named 'R1\.a'; record its instance, as an "
                r"analysis's or a plan's record=\['R1'\] does, or every instance, "
                r"as record=\['\*\*'\] does"
            ),
        ):
            point.current("R1", "a")
        with pytest.raises(
            KeyError, match=r"no variable named 'threshold'.*record=\['R1'\]"
        ):
            point.variable("R1", "threshold")
        with pytest.raises(KeyError, match=r"recorded no device unknown named 'R1\.a'"):
            point.conductance("R1.a", "R1.b")


def test_a_record_names_instances_and_patterns_of_their_paths() -> None:
    simulation = Simulation(two_dividers())
    assert simulation.instances == (
        "V1",
        "X1/R1",
        "X1/R2",
        "X2/R1",
        "X2/R2",
        "X1",
        "X2",
    )
    point = simulation.run(OperatingPoint(record=["X1/**", "X2"]))
    assert [device.path for device in point.recording.devices] == ["X1/R1", "X1/R2"]
    assert point.recording.composites == ("X2.top", "X2.bottom")
    assert point.current("X1/R1", "a") == pytest.approx(2e-3)
    assert point.current("X2", "top") == pytest.approx(2e-3)
    starred = simulation.run(OperatingPoint(record=["X?/R1"]))
    assert [device.path for device in starred.recording.devices] == ["X1/R1", "X2/R1"]
    # A pattern after ! takes back what the patterns before it named, the last to match
    # an instance deciding.
    excluded = simulation.run(OperatingPoint(record=["**", "!X1/**", "X1/R2"]))
    assert [device.path for device in excluded.recording.devices] == [
        "V1",
        "X1/R2",
        "X2/R1",
        "X2/R2",
    ]
    with pytest.raises(SimulationError, match=r"pattern 'X3/\*\*' matches no instance"):
        simulation.run(OperatingPoint(record=["**", "!X3/**"]))
    with pytest.raises(
        SimulationError, match=r"no instance named 'X1/R3' to record; close names"
    ):
        simulation.run(OperatingPoint(record=["X1/R3"]))
    with pytest.raises(SimulationError, match=r"pattern 'X3/\*' matches no instance"):
        simulation.run(OperatingPoint(record=["X3/*"]))
    with pytest.raises(TypeError, match="not the string 'X1'"):
        OperatingPoint(record="X1")


def test_what_a_result_records_reads_as_it_does_among_everything() -> None:
    simulation = Simulation(low_pass())
    everything = simulation.run(Transient(stop=5 * us, record=EVERYTHING))
    alone = Simulation(low_pass()).run(Transient(stop=5 * us, record=["C1"]))
    assert alone.recording.terminals == ("C1.a", "C1.b")
    np.testing.assert_array_equal(alone.times, everything.times)
    np.testing.assert_array_equal(alone.values, everything.values)
    for read in ("current", "charge"):
        np.testing.assert_array_equal(
            getattr(alone, read)("C1", "a"), getattr(everything, read)("C1", "a")
        )
    np.testing.assert_array_equal(
        alone.capacitance("C1.a", "C1.b"), everything.capacitance("C1.a", "C1.b")
    )


def test_a_transient_holds_only_what_it_records_against_its_memory() -> None:
    simulation = Simulation(low_pass())
    # A thousand numbers: the fifty points of a transient of every device, 49 numbers
    # each, pass them, and the three unknowns alone fit over three hundred points.
    memory = 1000 / 2**27
    with pytest.raises(SimulationError, match="the transient's results would pass"):
        simulation.run(
            Transient(stop=5 * us, record=EVERYTHING),
            Options(memory=memory),
        )
    waveforms = simulation.run(Transient(stop=5 * us), Options(memory=memory))
    assert waveforms.currents.shape == (len(waveforms.times), 0)
    assert waveforms.conductances.shape == (len(waveforms.times), 0)


def test_a_simulation_finds_where_each_terminal_joins() -> None:
    simulation = Simulation(two_dividers())
    assert simulation.joined("X1/R1", "b") == Joined("X1/mid", thermal=False)
    assert simulation.joined("X2", "bottom") == Joined(None, thermal=False)
    assert simulation.joined("X1", "middle") is None
    assert simulation.joined("X3/R1", "a") is None


def test_a_transient_steps_onto_the_times_it_is_given() -> None:
    simulation = Simulation(low_pass())
    times = [1.2345 * us, 3 * us]
    waveforms = simulation.run(Transient(stop=5 * us, times=times, record=["C1", "R1"]))
    for time in times:
        assert time.value in waveforms.times
    capacitor = waveforms.current("C1", "a")
    resistor = waveforms.current("R1", "b")
    np.testing.assert_allclose(capacitor + resistor, 0, atol=1e-9)
    np.testing.assert_allclose(
        waveforms.charge("C1", "a"), 1e-9 * waveforms.voltage("out")
    )
    with pytest.raises(SimulationError, match="after 0 and at most the stop time"):
        simulation.run(Transient(stop=5 * us, times=[6 * us]))


@component
def slow_rc(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    sine = Sine(0 * V, 1 * V, 1 * kHz)
    source = c.place(devices.voltage_source(sine), id="V1")
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
    capacitor = c.place(devices.capacitor(10 * uF), id="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


def test_a_periodic_transient_finds_the_steady_state_at_once() -> None:
    # The time constant is ten periods; the steady state is the phasor's.
    steady = Simulation(slow_rc()).run(
        Transient(stop=1 * ms, max_step=5e-3 * ms, periodic=True)
    )
    assert steady.statistics.periods == 2
    lag = 2 * math.pi * 1e3 * 1e3 * 10e-6
    expected = np.sin(2 * math.pi * 1e3 * steady.times - math.atan(lag))
    np.testing.assert_allclose(
        steady.voltage("out"), expected / math.hypot(1, lag), atol=1e-5
    )


def test_a_dc_sweep_solves_each_level() -> None:
    simulation = Simulation(divider(0 * V, 1 * kohm, 1 * kohm))
    sweep = simulation.run(DcSweep("V1", linear(-1 * V, 1 * V, 5)))
    np.testing.assert_allclose(sweep.levels, [-1, -0.5, 0, 0.5, 1])
    np.testing.assert_allclose(sweep.voltage("out"), sweep.levels / 2, atol=1e-12)


def test_a_diode_conducts_forward_and_blocks_in_reverse() -> None:
    sweep = Simulation(diode_load()).run(DcSweep("V1", linear(-5 * V, 5 * V, 101)))
    anode = sweep.voltage("anode")
    assert np.all(np.diff(anode) > 0)
    assert anode[0] == pytest.approx(-5.0, abs=1e-6)
    assert 0.6 < anode[-1] < 0.8
    current = -sweep.current("V1")
    assert current[-1] == pytest.approx((5 - anode[-1]) / 1e3)


def test_a_diode_s_forward_voltage_falls_as_it_warms() -> None:
    forward = DcSweep("V1", [5 * V])
    anodes = [
        Simulation(diode_load(), temperature).run(forward).voltage("anode")[0]
        for temperature in (celsius(0), celsius(100))
    ]
    slope = (anodes[1] - anodes[0]) / 100
    assert -2.5e-3 < slope < -1.5e-3


def test_a_resistance_drifts_with_its_temperature_coefficients() -> None:
    @component
    def drifting(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), id="V1")
        load = c.place(
            devices.resistor(
                1 * kohm,
                temperature_coefficient=1e-3 / K,
                quadratic_temperature_coefficient=1e-6 / K**2,
                nominal_temperature=celsius(25),
            ),
            id="R1",
        )
        c.connect(source.positive, load.a)
        c.connect(source.negative, load.b, ground)

    point = Simulation(drifting(), celsius(75)).run(OperatingPoint())
    assert -point.current("V1") == pytest.approx(1e-3 / (1 + 50e-3 + 2.5e-3))


def test_a_device_drifts_to_a_temperature_of_its_own() -> None:
    @component
    def apart(c: Builder, held: devices.Device, quiet: devices.Device) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), id="V1")
        load = c.place(held, id="R1")
        bleed = c.place(quiet, id="R2")
        c.connect(source.positive, load.a, name="top")
        c.connect(load.b, bleed.a, name="out")
        c.connect(source.negative, bleed.b, ground)

    drifting = devices.resistor(1 * kohm, temperature_coefficient=1e-3 / K)
    quiet = devices.resistor(1 * kohm).held_at(0 * K)
    for warmed, temperature in (
        (drifting.held_at(celsius(77)), celsius(-40)),
        (drifting.warmer_by(50 * K), celsius(27)),
    ):
        simulation = Simulation(apart(warmed, quiet), temperature)
        point = simulation.run(OperatingPoint())
        assert point.voltage("out") == pytest.approx(1 / 2.05, rel=1e-12)
    # Held at 0 K, the lower resistor makes no noise: the upper one's alone reaches out.
    spectrum = simulation.run(Noise("out", [1 * kHz]))
    assert spectrum.contribution("R2")[0] == 0
    assert spectrum.contribution("R1")[0] > 0
    with pytest.raises(ValueError, match="at 0 K or above"):
        drifting.held_at(-1 * K)
    gate = devices.logic_gate("nand", rise=1 * ns, transition=1 * ns)
    with pytest.raises(ValueError, match="a logic_cell runs at no temperature of its"):
        gate.warmer_by(10 * K)


def test_the_ac_response_of_a_low_pass_filter() -> None:
    cutoff = 1 / (2 * math.pi * 1e3 * 1e-9)
    frequencies = [*decade(1 * kHz, 1e5 * kHz, 10), cutoff * kHz / 1e3]
    response = Simulation(low_pass()).run(Ac("V1", frequencies))
    expected = 1 / (1 + 1j * response.frequencies / cutoff)
    np.testing.assert_allclose(response.voltage("out"), expected, rtol=1e-9)
    assert abs(response.voltage("out")[-1]) == pytest.approx(1 / math.sqrt(2))
    assert response.operating_point.voltage("out") == pytest.approx(0.0)


@component
def stepped(c: Builder, primary: float, secondary: float, coupling: float) -> None:
    """A transformer of `primary` and `secondary` henries, fed by 50 Ω, into 1 kΩ."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), id="V1")
    windings = c.place(
        devices.coupled_inductors(
            [devices.inductor(primary * H), devices.inductor(secondary * H)],
            {(1, 2): coupling},
        ),
        id="T1",
    )
    feed = c.place(devices.resistor(50 * ohm), id="R1")
    resistor = c.place(devices.resistor(1 * kohm), id="RL")
    c.connect(source.positive, feed.a, name="in")
    c.connect(feed.b, windings.a1, name="primary")
    c.connect(windings.a2, resistor.a, name="out")
    c.connect(source.negative, windings.b1, windings.b2, resistor.b, ground)


def test_a_transformer_steps_its_voltage_by_its_coupled_inductances() -> None:
    primary, secondary, coupling, load = 1e-3, 4e-3, 0.95, 1e3
    simulation = Simulation(stepped(primary, secondary, coupling))
    assert [unknown.name for unknown in simulation.unknowns][-2:] == [
        "T1:current1",
        "T1:current2",
    ]
    response = simulation.run(Ac("V1", decade(1 * kHz, 1e4 * kHz, 5)))
    s = 2j * np.pi * response.frequencies
    mutual = coupling * math.sqrt(primary * secondary)
    # v1 = s·L1·i1 + s·M·i2 and out = s·M·i1 + s·L2·i2, with i2 = -out / R.
    gain = 1 / (primary * (1 + s * secondary / load) / mutual - s * mutual / load)
    stepped_up = response.voltage("out") / response.voltage("primary")
    np.testing.assert_allclose(stepped_up, gain, rtol=1e-9)


def test_coupled_inductances_take_overrides_and_sensitivities_by_path() -> None:
    frequencies = decade(10 * kHz, 1e3 * kHz, 2)
    nominal = Simulation(stepped(1e-3, 4e-3, 0.95))
    replaced = {"T1": {"windings.0.inductance": 1.2e-3, "couplings.0.coefficient": 0.9}}
    overridden = Simulation(stepped(1e-3, 4e-3, 0.95), overrides=replaced)
    rebuilt = Simulation(stepped(1.2e-3, 4e-3, 0.9))
    np.testing.assert_allclose(
        overridden.run(Ac("V1", frequencies)).voltage("out"),
        rebuilt.run(Ac("V1", frequencies)).voltage("out"),
        rtol=1e-12,
    )
    found = nominal.run(Sensitivity(["out"], of=Ac("V1", frequencies)))
    step = 1e-6

    def out(coupling: float) -> results.Complex:
        response = Simulation(stepped(1e-3, 4e-3, coupling)).run(Ac("V1", frequencies))
        return response.voltage("out")

    expected = (out(0.95 + step) - out(0.95 - step)) / (2 * step)
    exact = found.derivative("out", "T1", "couplings.0.coefficient")
    np.testing.assert_allclose(exact, expected, rtol=1e-6)


def test_coupled_windings_are_plain_inductors_whose_tolerances_carry_over() -> None:
    coupled = devices.coupled_inductors(
        [devices.inductor(1 * mH, tolerance=0.1), devices.inductor(2 * mH)],
        {(1, 2): 0.5},
    )
    assert list(coupled.tolerances) == ["windings.0.inductance"]
    with pytest.raises(ValueError, match="winding 2 is an inductor"):
        devices.coupled_inductors(
            [devices.inductor(1 * mH), devices.resistor(1 * ohm)], {(1, 2): 0.5}
        )
    with pytest.raises(ValueError, match="which the coupled inductors take as a whole"):
        devices.coupled_inductors(
            [devices.inductor(1 * mH).held_at(celsius(85)), devices.inductor(2 * mH)],
            {(1, 2): 0.5},
        )
    held = coupled.warmer_by(15 * K)
    assert held.parameters["temperature_offset"] == 15.0


def test_the_step_response_of_a_low_pass_filter() -> None:
    waveforms = Simulation(low_pass()).run(
        Transient(stop=5 * us), Options(relative_tolerance=1e-6)
    )
    late = waveforms.times > 0.1e-6
    exact = 1 - np.exp(-(waveforms.times[late] - 0.5e-9) / 1e-6)
    np.testing.assert_allclose(waveforms.voltage("out")[late], exact, atol=2e-4)
    assert waveforms.times[-1] == pytest.approx(5e-6)
    assert waveforms.statistics.accepted == len(waveforms.times) - 1


@component
def common_emitter(c: Builder, transistor: devices.Device) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), id="V2")
    source = c.place(devices.voltage_source(0.75 * V), id="V1")
    base = c.place(devices.resistor(2 * kohm), id="RB")
    amplifier = c.place(transistor, id="Q1")
    load = c.place(devices.resistor(1 * kohm), id="RC")
    c.connect(source.positive, base.a, name="in")
    c.connect(base.b, amplifier.base, name="base")
    c.connect(supply.positive, load.a, name="supply")
    c.connect(load.b, amplifier.collector, name="out")
    c.connect(source.negative, supply.negative, amplifier.emitter, ground)


@pytest.mark.parametrize(
    ("minimum", "halfway"),
    [(None, None), (10 * ohm, None), (10 * ohm, 20 * uA)],
    ids=["fixed", "charge", "current"],
)
def test_a_small_signal_response_is_the_slope_of_the_dc_transfer(
    minimum: Quantity | None, halfway: Quantity | None
) -> None:
    # The base resistance, fixed or falling with the base charge or current, enters
    # the small-signal model with every derivative.
    transistor = devices.bipolar_transistor(
        saturation_current=1e-15 * A,
        forward_early_voltage=50 * V,
        forward_knee_current=10 * mA,
        base_resistance=300 * ohm,
        minimum_base_resistance=minimum,
        base_resistance_current=halfway,
    )
    simulation = Simulation(common_emitter(transistor))
    tight = Options(relative_tolerance=1e-11, voltage_tolerance=1e-15)
    # A central difference over 0.1 mV, small beside the thermal voltage, errs by about
    # 1e-5; leaving out how the base resistance falls would err by a percent.
    step = 1e-4
    levels = [(0.75 - step) * V, (0.75 + step) * V]
    sweep = simulation.run(DcSweep("V1", levels), tight)
    slopes = (sweep.values[1] - sweep.values[0]) / (2 * step)
    response = simulation.run(Ac("V1", [1 * Hz]), tight)
    np.testing.assert_allclose(response.values[0].real, slopes, rtol=1e-4, atol=1e-12)
    assert abs(response.voltage("out")[0]) > 10  # The stage amplifies.


def test_a_plan_runs_its_analyses_by_name() -> None:
    simulation = Simulation(divider(5 * V, 1 * kohm, 1 * kohm))
    plan = Plan({"op": OperatingPoint(), "sweep": DcSweep("V1", [1 * V, 2 * V])})
    results = simulation.run_plan(plan)
    assert list(results) == ["op", "sweep"]


def test_a_plan_runs_at_the_simulation_s_temperature() -> None:
    plan = Plan({"op": OperatingPoint()}, temperature=celsius(85))
    simulation = Simulation(divider(5 * V, 1 * kohm, 1 * kohm), celsius(85))
    assert list(simulation.run_plan(plan)) == ["op"]
    with pytest.raises(ValueError, match=r"is not the simulation's, 300\.1 K"):
        Simulation(divider(5 * V, 1 * kohm, 1 * kohm)).run_plan(plan)
    with pytest.raises(DimensionError, match="a plan's temperature"):
        Plan({"op": OperatingPoint()}, temperature=85 * V)


def test_corners_are_named_by_identifiers_at_a_temperature() -> None:
    analyses = {"op": OperatingPoint()}
    corner = Corner(temperature=celsius(-40), supply=3 * V)
    plan = Plan(analyses, temperature=celsius(85), corners={"cold": corner})
    assert plan.temperature_at(corner) == celsius(-40)
    assert plan.temperature_at(Corner(supply=3 * V)) == celsius(85)
    assert plan.temperature_at() == celsius(85)
    with pytest.raises(ValueError, match="a corner is named by a Python identifier"):
        Plan(analyses, corners={"cold 1": corner})
    with pytest.raises(TypeError, match="not a Corner"):
        Plan(analyses, corners={"cold": celsius(-40)})  # ty: ignore[invalid-argument-type] - the wrong type is tested
    with pytest.raises(DimensionError, match="a corner's temperature"):
        Corner(temperature=5 * V)


def test_plans_name_their_analyses_by_identifiers() -> None:
    with pytest.raises(ValueError, match="at least one analysis"):
        Plan({})
    with pytest.raises(ValueError, match="identifier, not 'step 1'"):
        Plan({"step 1": OperatingPoint()})


def test_problems_name_the_instance_and_the_script_line() -> None:
    @component
    def floating(c: Builder) -> None:
        c.port("ground", Role.GROUND)
        resistor = c.place(devices.resistor(1 * kohm), id="R1")
        c.connect(resistor.a, resistor.b)

    @component
    def ungrounded(c: Builder) -> None:
        c.place(devices.resistor(-1 * kohm), id="R1")

    Simulation(floating())
    with pytest.raises(ElaborationError) as raised:
        Simulation(ungrounded())
    problems = raised.value.problems
    assert any("ground" in problem for problem in problems)
    assert any(
        "positive" in problem and "= R1, tests/test_simulate.py:" in problem
        for problem in problems
    )


def test_analyses_check_their_sources() -> None:
    simulation = Simulation(divider(5 * V, 1 * kohm, 1 * kohm))
    with pytest.raises(SimulationError, match="there is no instance at `V9`"):
        simulation.run(DcSweep("V9", [1 * V]))
    with pytest.raises(SimulationError, match="`R1` is not an independent source"):
        simulation.run(DcSweep("R1", [1 * V]))
    with pytest.raises(DimensionError, match="a sweep level takes a quantity in V"):
        simulation.run(DcSweep("V1", [1 * mA]))


def test_a_current_source_sweeps_in_amperes() -> None:
    @component
    def biased(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.current_source(1 * mA), id="I1")
        resistor = c.place(devices.resistor(1 * kohm), id="R1")
        c.connect(source.negative, resistor.a, name="top")
        c.connect(source.positive, resistor.b, ground)

    simulation = Simulation(biased())
    assert simulation.run(OperatingPoint()).voltage("top") == pytest.approx(1.0)
    sweep = simulation.run(DcSweep("I1", [2 * mA, 1 * A]))
    np.testing.assert_allclose(sweep.voltage("top"), [2.0, 1000.0])


@pytest.fixture
def analysis_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The log the settings name for each analysis run while the test runs."""
    log = tmp_path / "analyses.jsonl"
    monkeypatch.setenv("JOLTY_ANALYSIS_LOG", str(log))
    simulate._settings.cache_clear()  # noqa: SLF001 - read once per process, as here
    yield log
    monkeypatch.undo()
    simulate._settings.cache_clear()  # noqa: SLF001 - read again without the log


def test_each_analysis_appends_to_the_log_the_settings_name(analysis_log: Path) -> None:
    simulation = Simulation(divider(5 * V, 1 * kohm, 1 * kohm))
    simulation.run(OperatingPoint())
    # An analysis refused before it runs, as a record naming no instance, runs none.
    with pytest.raises(SimulationError, match="no instance named 'R9'"):
        simulation.run(OperatingPoint(record=["R9"]))
    resistances = [{"R1": {"resistance": value}} for value in (2e3, 3e3, 4e3)]
    variants = Explicit([Variant(overrides) for overrides in resistances])
    analyses = {"op": OperatingPoint(), "step": Transient(1 * us)}
    simulation.batch(analyses, variants)
    simulation.batch(analyses, variants, runs=range(1, 1))
    simulation.batch(analyses, variants, runs=range(1, 3))
    logged = [
        Logged.model_validate_json(line)
        for line in analysis_log.read_text(encoding="utf-8").splitlines()
    ]
    # Each analysis once, with the runs it runs on, a batch's analyses as one start.
    assert [(each.analysis, each.runs) for each in logged] == [
        ("OperatingPoint", 1),
        ("OperatingPoint", 3),
        ("Transient", 3),
        ("OperatingPoint", 2),
        ("Transient", 2),
    ]
    assert [each.time for each in logged] == sorted(each.time for each in logged)
    assert logged[1].time == logged[2].time


def test_an_analysis_the_log_cannot_take_does_not_run(
    analysis_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unwritable = analysis_log.parent / "missing" / analysis_log.name
    monkeypatch.setenv("JOLTY_ANALYSIS_LOG", str(unwritable))
    simulate._settings.cache_clear()  # noqa: SLF001 - read once per process, as here
    message = f"cannot append to the analysis log {unwritable}: No such file"
    with pytest.raises(JoltyError, match=re.escape(message)):
        Simulation(divider(5 * V, 1 * kohm, 1 * kohm)).run(OperatingPoint())
    assert not unwritable.parent.exists()

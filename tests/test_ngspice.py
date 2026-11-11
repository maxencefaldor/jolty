# Copyright 2026 Maxence Faldor
"""Differential tests: jolty and ngspice simulate the same circuits and must agree.

Each circuit of the corpus is written once with the builder, simulated by jolty, and
written as a SPICE netlist for ngspice, whose results come back as an ASCII raw file.
Both engines run with tight tolerances. Operating points, DC sweeps and AC responses
must agree to the Newton tolerance; transients, interpolated onto ngspice's time points,
to a small fraction of their swing.

The corpus is flat: hierarchy is jolty's own, tested on its own, while these tests check
the devices, the waveforms and the numerics against the reference simulator.
"""

import math
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import JsonValue

from jolty import (
    Ac,
    Builder,
    Component,
    DcSweep,
    OperatingPoint,
    Options,
    Role,
    Simulation,
    Transient,
    component,
    devices,
    ir,
    results,
)
from jolty.plan import decade, linear
from jolty.units import (
    A,
    Hz,
    MHz,
    Quantity,
    V,
    fA,
    kHz,
    kohm,
    mA,
    mH,
    ms,
    mV,
    nA,
    nF,
    ns,
    ohm,
    pF,
    uF,
    uS,
    us,
)
from jolty.waveforms import Piecewise, Pulse, Sine

NGSPICE = shutil.which("ngspice")

pytestmark = pytest.mark.skipif(NGSPICE is None, reason="ngspice is not installed")

TIGHT = Options(
    relative_tolerance=1e-6,
    voltage_tolerance=1e-9,
    current_tolerance=1e-14,
    charge_tolerance=1e-18,
)
"""jolty's options, as tight as ngspice's below."""

_THERMAL_VOLTAGE = 1.380649e-23 * 300.15 / 1.602176634e-19
"""jolty's thermal voltage at 27 °C, from the exact SI constants."""

_SPICE_THERMAL_VOLTAGE_PER_KELVIN = 1.38064852e-23 / 1.6021766208e-19
"""ngspice's Boltzmann constant over the elementary charge, from CODATA 2014."""

_SPICE_TEMPERATURE = _THERMAL_VOLTAGE / _SPICE_THERMAL_VOLTAGE_PER_KELVIN - 273.15
"""The temperature, in degrees Celsius, at which ngspice's thermal voltage is jolty's.

Running ngspice there, with its models' nominal temperature there too so no parameter
scales with temperature, compares the devices rather than the physical constants.
"""

SPICE_OPTIONS = (
    ".options reltol=1e-6 vntol=1e-9 abstol=1e-14 chgtol=1e-18"
    f" temp={_SPICE_TEMPERATURE!r} tnom={_SPICE_TEMPERATURE!r}"
)

_DIODE_PARAMETERS = {
    "saturation_current": "is",
    "emission_coefficient": "n",
    "series_resistance": "rs",
    "junction_capacitance": "cjo",
    "junction_potential": "vj",
    "grading_coefficient": "m",
    "depletion_coefficient": "fc",
    "transit_time": "tt",
}

_JFET_PARAMETERS = {
    "threshold_voltage": "vto",
    "transconductance_parameter": "beta",
    "channel_length_modulation": "lambda",
    "drain_resistance": "rd",
    "source_resistance": "rs",
    "gate_source_capacitance": "cgs",
    "gate_drain_capacitance": "cgd",
    "junction_potential": "pb",
    "saturation_current": "is",
    "depletion_coefficient": "fc",
    "doping_tail": "b",
}

_BIPOLAR_PARAMETERS = {
    "saturation_current": "is",
    "forward_beta": "bf",
    "reverse_beta": "br",
    "forward_emission_coefficient": "nf",
    "reverse_emission_coefficient": "nr",
    "forward_early_voltage": "vaf",
    "reverse_early_voltage": "var",
    "forward_knee_current": "ikf",
    "reverse_knee_current": "ikr",
    "base_emitter_leakage_current": "ise",
    "base_emitter_leakage_emission_coefficient": "ne",
    "base_collector_leakage_current": "isc",
    "base_collector_leakage_emission_coefficient": "nc",
    "base_resistance": "rb",
    "minimum_base_resistance": "rbm",
    "base_resistance_current": "irb",
    "emitter_resistance": "re",
    "collector_resistance": "rc",
    "base_emitter_capacitance": "cje",
    "base_emitter_potential": "vje",
    "base_emitter_grading": "mje",
    "forward_transit_time": "tf",
    "transit_time_bias_coefficient": "xtf",
    "transit_time_voltage": "vtf",
    "transit_time_current": "itf",
    "base_collector_capacitance": "cjc",
    "base_collector_potential": "vjc",
    "base_collector_grading": "mjc",
    "internal_base_fraction": "xcjc",
    "reverse_transit_time": "tr",
    "depletion_coefficient": "fc",
}

type Real = npt.NDArray[np.float64]
type Values = float | Real | npt.NDArray[np.complex128]


# The corpus.


@component
def divider(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), id="V1")
    upper = c.place(devices.resistor(1 * kohm), id="R1")
    lower = c.place(devices.resistor(3 * kohm), id="R2")
    c.connect(source.positive, upper.a, name="in")
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


@component
def low_pass(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    step = Piecewise([(1 * us, 0 * V), (1 * us + 10 * ns, 1 * V)])
    source = c.place(devices.voltage_source(step), id="V1")
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
    capacitor = c.place(devices.capacitor(1 * nF), id="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def resonator(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    pulse = Pulse(
        initial=0 * V,
        pulsed=1 * V,
        delay=1 * us,
        rise=10 * ns,
        fall=10 * ns,
        width=20 * us,
    )
    source = c.place(devices.voltage_source(pulse), id="V1")
    resistor = c.place(devices.resistor(20 * ohm), id="R1")
    inductor = c.place(devices.inductor(1 * mH), id="L1")
    capacitor = c.place(devices.capacitor(10 * nF), id="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, inductor.a, name="mid")
    c.connect(inductor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def clipper(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), id="V1")
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
    diode = c.place(
        devices.diode(saturation_current=2.52 * nA, emission_coefficient=1.752),
        id="D1",
    )
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, diode.anode, name="out")
    c.connect(diode.cathode, source.negative, ground)


@component
def rectifier(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    mains = Sine(offset=0 * V, amplitude=5 * V, frequency=1 * kHz, phase=0.3)
    source = c.place(devices.voltage_source(mains), id="V1")
    diode = c.place(
        devices.diode(
            saturation_current=2.52 * nA,
            emission_coefficient=1.752,
            series_resistance=0.568 * ohm,
            junction_capacitance=4 * pF,
            junction_potential=0.7 * V,
            grading_coefficient=0.4,
            depletion_coefficient=0.5,
            transit_time=20 * ns,
        ),
        id="D1",
    )
    smoothing = c.place(devices.capacitor(1 * uF), id="C1")
    load = c.place(devices.resistor(1 * kohm), id="R1")
    c.connect(source.positive, diode.anode, name="in")
    c.connect(diode.cathode, smoothing.a, load.a, name="out")
    c.connect(smoothing.b, load.b, source.negative, ground)


@component
def amplifier(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0.1 * V), id="V1")
    gain = c.place(devices.voltage_controlled_voltage_source(10.0), id="E1")
    coupling = c.place(devices.resistor(1 * kohm), id="R1")
    stage = c.place(devices.voltage_controlled_current_source(500 * uS), id="G1")
    load = c.place(devices.resistor(2 * kohm), id="R2")
    bypass = c.place(devices.capacitor(1 * nF), id="C1")
    c.connect(source.positive, gain.control_positive, name="in")
    c.connect(gain.positive, coupling.a, name="gained")
    c.connect(coupling.b, stage.control_positive, name="coupled")
    c.connect(stage.negative, load.a, bypass.a, name="out")
    c.connect(
        source.negative,
        gain.negative,
        gain.control_negative,
        stage.positive,
        stage.control_negative,
        load.b,
        bypass.b,
        ground,
    )


@component
def driven_tank(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    drive = Pulse(
        initial=0 * mA,
        pulsed=1 * mA,
        delay=0 * us,
        rise=1 * us,
        fall=1 * us,
        width=3 * us,
        period=10 * us,
    )
    source = c.place(devices.current_source(drive), id="I1")
    inductor = c.place(devices.inductor(10 * mH), id="L1")
    damping = c.place(devices.resistor(5 * kohm), id="R1")
    capacitor = c.place(devices.capacitor(1 * nF), id="C1")
    c.connect(source.negative, inductor.a, damping.a, capacitor.a, name="out")
    c.connect(source.positive, inductor.b, damping.b, capacitor.b, ground)


_JFET = devices.jfet(
    threshold_voltage=-1.8 * V,
    transconductance_parameter=1.3 * mA / V**2,
    channel_length_modulation=0.02 / V,
    drain_resistance=20 * ohm,
    source_resistance=10 * ohm,
    gate_source_capacitance=4 * pF,
    gate_drain_capacitance=2 * pF,
    junction_potential=0.8 * V,
    saturation_current=1e-14 * A,
    doping_tail=0.85,
)


@component
def jfet_stage(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(12 * V), id="V2")
    # Starting at its peak, the sine's slope is zero, so the operating point at zero
    # time is also the transient's start, with no capacitive current to jump.
    signal = Sine(
        offset=-0.8 * V, amplitude=0.6 * V, frequency=200 * kHz, phase=math.pi / 2
    )
    source = c.place(devices.voltage_source(signal), id="V1")
    gate = c.place(devices.resistor(1 * kohm), id="RG")
    transistor = c.place(_JFET, id="J1")
    drain = c.place(devices.resistor(2.2 * kohm), id="RD")
    degeneration = c.place(devices.resistor(470 * ohm), id="RS")
    c.connect(source.positive, gate.a, name="in")
    c.connect(gate.b, transistor.gate, name="gate")
    c.connect(supply.positive, drain.a, name="supply")
    c.connect(drain.b, transistor.drain, name="out")
    c.connect(transistor.source, degeneration.a, name="tail")
    c.connect(source.negative, supply.negative, degeneration.b, ground)


@component
def jfet_curves(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    bias = c.place(devices.voltage_source(-0.5 * V), id="V1")
    drain = c.place(devices.voltage_source(0 * V), id="V2")
    transistor = c.place(_JFET, id="J1")
    c.connect(bias.positive, transistor.gate, name="gate")
    c.connect(drain.positive, transistor.drain, name="drain")
    c.connect(bias.negative, drain.negative, transistor.source, ground)


@component
def p_jfet_follower(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(-9 * V), id="V2")
    source = c.place(devices.voltage_source(-1 * V), id="V1")
    transistor = c.place(
        devices.jfet(
            channel="p",
            threshold_voltage=-2.2 * V,
            transconductance_parameter=0.9 * mA / V**2,
            channel_length_modulation=0.01 / V,
            gate_source_capacitance=3 * pF,
            gate_drain_capacitance=1 * pF,
        ),
        id="J1",
    )
    load = c.place(devices.resistor(3.3 * kohm), id="RL")
    c.connect(source.positive, transistor.gate, name="in")
    c.connect(supply.positive, transistor.drain, name="supply")
    c.connect(transistor.source, load.a, name="out")
    c.connect(source.negative, supply.negative, load.b, ground)


def _bipolar(
    polarity: Literal["npn", "pnp"],
    base_resistance: Literal["current", "charge", "fixed"],
) -> devices.Device:
    """A small-signal transistor with every effect of the model switched on.

    Its base resistance falls with the base current, with the base charge, or stays
    fixed. ngspice leaves a falling base resistance out of its small-signal model,
    where jolty keeps it, so a frequency response compares transistors with a fixed one.
    """
    return devices.bipolar_transistor(
        polarity=polarity,
        saturation_current=14 * fA,
        forward_beta=150.0,
        reverse_beta=2.0,
        forward_emission_coefficient=1.01,
        forward_early_voltage=80 * V,
        reverse_early_voltage=15 * V,
        forward_knee_current=40 * mA,
        reverse_knee_current=5 * mA,
        base_emitter_leakage_current=50 * fA,
        base_emitter_leakage_emission_coefficient=1.6,
        base_collector_leakage_current=100 * fA,
        base_resistance=50 * ohm,
        minimum_base_resistance=None if base_resistance == "fixed" else 10 * ohm,
        base_resistance_current=1 * mA if base_resistance == "current" else None,
        emitter_resistance=1 * ohm,
        collector_resistance=5 * ohm,
        base_emitter_capacitance=8 * pF,
        base_emitter_potential=0.75 * V,
        base_emitter_grading=0.33,
        forward_transit_time=0.4 * ns,
        transit_time_bias_coefficient=3.0,
        transit_time_voltage=4 * V,
        transit_time_current=0.6 * A,
        base_collector_capacitance=3.5 * pF,
        base_collector_potential=0.75 * V,
        base_collector_grading=0.4,
        internal_base_fraction=0.6,
        reverse_transit_time=50 * ns,
    )


@component
def bjt_stage(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), id="V2")
    source = c.place(devices.voltage_source(0.7 * V), id="V1")
    base = c.place(devices.resistor(10 * kohm), id="RB")
    transistor = c.place(_bipolar("npn", "current"), id="Q1")
    load = c.place(devices.resistor(1 * kohm), id="RC")
    c.connect(source.positive, base.a, name="in")
    c.connect(base.b, transistor.base, name="base")
    c.connect(supply.positive, load.a, name="supply")
    c.connect(load.b, transistor.collector, name="out")
    c.connect(source.negative, supply.negative, transistor.emitter, ground)


@component
def bjt_amplifier(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(12 * V), id="V2")
    signal = Sine(offset=0 * V, amplitude=20 * mV, frequency=1 * MHz, phase=math.pi / 2)
    source = c.place(devices.voltage_source(signal), id="V1")
    coupling = c.place(devices.capacitor(100 * nF), id="CIN")
    upper = c.place(devices.resistor(47 * kohm), id="R1")
    lower = c.place(devices.resistor(10 * kohm), id="R2")
    transistor = c.place(_bipolar("npn", "fixed"), id="Q1")
    load = c.place(devices.resistor(3.3 * kohm), id="RC")
    degeneration = c.place(devices.resistor(1 * kohm), id="RE")
    bypass = c.place(devices.capacitor(1 * nF), id="CE")
    c.connect(source.positive, coupling.a, name="in")
    c.connect(coupling.b, upper.b, lower.a, transistor.base, name="base")
    c.connect(supply.positive, upper.a, load.a, name="supply")
    c.connect(load.b, transistor.collector, name="out")
    c.connect(transistor.emitter, degeneration.a, bypass.a, name="tail")
    c.connect(
        source.negative, supply.negative, lower.b, degeneration.b, bypass.b, ground
    )


@component
def pnp_mirror(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(10 * V), id="V1")
    reference = c.place(devices.resistor(10 * kohm), id="RR")
    left = c.place(_bipolar("pnp", "charge"), id="Q1")
    right = c.place(_bipolar("pnp", "charge"), id="Q2")
    load = c.place(devices.resistor(4.7 * kohm), id="RL")
    c.connect(supply.positive, left.emitter, right.emitter, name="supply")
    c.connect(left.base, left.collector, right.base, reference.a, name="mirror")
    c.connect(right.collector, load.a, name="out")
    c.connect(supply.negative, reference.b, load.b, ground)


@component
def current_controlled(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(2 * V), id="V1")
    feed = c.place(devices.resistor(1 * kohm), id="R1")
    amplifier = c.place(devices.current_controlled_current_source(3.0), id="F1")
    load = c.place(devices.resistor(2 * kohm), id="R2")
    transducer = c.place(devices.current_controlled_voltage_source(500 * ohm), id="H1")
    output = c.place(devices.resistor(4.7 * kohm), id="R3")
    c.connect(source.positive, feed.a, name="in")
    c.connect(feed.b, amplifier.control_positive, name="sensed")
    c.connect(amplifier.negative, transducer.control_positive, name="mid")
    c.connect(transducer.control_negative, load.a, name="loaded")
    c.connect(transducer.positive, output.a, name="out")
    c.connect(
        source.negative,
        amplifier.control_negative,
        amplifier.positive,
        load.b,
        transducer.negative,
        output.b,
        ground,
    )


# Netlists and raw files.


@dataclass(frozen=True)
class Element:
    """How a device model is written in SPICE: its letter and its terminals' order."""

    letter: str
    terminals: tuple[str, ...]


_CONTROLLED = ("positive", "negative", "control_positive", "control_negative")

ELEMENTS = {
    "resistor": Element("R", ("a", "b")),
    "capacitor": Element("C", ("a", "b")),
    "inductor": Element("L", ("a", "b")),
    "voltage_source": Element("V", ("positive", "negative")),
    "current_source": Element("I", ("positive", "negative")),
    "voltage_controlled_voltage_source": Element("E", _CONTROLLED),
    "voltage_controlled_current_source": Element("G", _CONTROLLED),
    "current_controlled_current_source": Element("F", _CONTROLLED),
    "current_controlled_voltage_source": Element("H", _CONTROLLED),
    "diode": Element("D", ("anode", "cathode")),
    "jfet": Element("J", ("drain", "gate", "source")),
    "bipolar_transistor": Element("Q", ("collector", "base", "emitter")),
}


def _model_card(model: str, values: dict[str, JsonValue]) -> tuple[str, str]:
    """The SPICE type and parameters of a device's model card."""
    match model:
        case "diode":
            kind, names = "D", _DIODE_PARAMETERS
        case "jfet":
            kind, names = f"{values.get('channel', 'n')}jf".upper(), _JFET_PARAMETERS
        case "bipolar_transistor":
            kind, names = (
                str(values.get("polarity", "npn")).upper(),
                _BIPOLAR_PARAMETERS,
            )
        case _:
            pytest.fail(f"{model} has no model card")
    spice = " ".join(
        f"{names[key]}={number!r}"
        for key, number in values.items()
        if key not in {"channel", "polarity"}
    )
    return kind, spice


def element(instance: ir.Instance) -> str:
    """The SPICE name of a device instance: its model's letter, then its id."""
    assert isinstance(instance.of, ir.DeviceOf), "the corpus is flat"
    return ELEMENTS[instance.of.device].letter + instance.id


def netlist(circuit: ir.Circuit, *, excited: str | None = None) -> list[str]:
    """The SPICE lines of a flat circuit; `excited` is the source an AC run excites."""
    top = circuit.components[circuit.top]
    ground = next(port.name for port in top.ports if port.role is Role.GROUND)
    lines = [f"* {circuit.top}"]
    for instance in top.instances:
        assert isinstance(instance.of, ir.DeviceOf), "the corpus is flat"
        model, values = instance.of.device, instance.parameters
        nets = (instance.terminals[name] for name in ELEMENTS[model].terminals)
        nodes = " ".join("0" if net == ground else net for net in nets)
        match model:
            case "voltage_source" | "current_source":
                ac = " AC 1" if instance.id == excited else ""
                value = _waveform(values["waveform"]) + ac
            case (
                "current_controlled_current_source"
                | "current_controlled_voltage_source"
            ):
                # ngspice senses a current through a voltage source: a zero-volt one in
                # the control port's place.
                *output, control_positive, control_negative = nodes.split()
                sense = _sense(instance)
                lines.append(f"{sense} {control_positive} {control_negative} DC 0")
                nodes = " ".join(output)
                (number,) = values.values()
                value = f"{sense} {number!r}"
            case "diode" | "jfet" | "bipolar_transistor":
                value = f"M{instance.id}"
                kind, spice = _model_card(model, values)
                lines.append(f".model {value} {kind}({spice})")
            case _:
                (number,) = values.values()
                value = repr(number)
        lines.append(f"{element(instance)} {nodes} {value}")
    return lines


def _sense(instance: ir.Instance) -> str:
    """The zero-volt source sensing a current-controlled source's control current."""
    return f"V{element(instance)}SENSE"


def vector(circuit: ir.Circuit, unknown: results.Unknown) -> str:
    """The name of the ngspice vector of a jolty unknown, a voltage or a current."""
    if unknown.quantity == "voltage":
        return f"v({unknown.name.lower()})"
    id, _, label = unknown.name.partition(":")
    instances = circuit.components[circuit.top].instances
    (instance,) = (instance for instance in instances if instance.id == id)
    source = _sense(instance) if label == "control" else element(instance)
    return f"i({source.lower()})"


def _waveform(spec: JsonValue) -> str:
    match spec:
        case {"dc": float(level)}:
            return f"DC {level!r}"
        case {"pulse": dict(pulse)}:
            order = ("initial", "pulsed", "delay", "rise", "fall", "width", "period")
            return _function("PULSE", [pulse[key] for key in order if key in pulse])
        case {
            "sine": {
                "offset": float(offset),
                "amplitude": float(amplitude),
                "frequency": float(frequency),
                "delay": float(delay),
                "damping": float(damping),
                "phase": float(phase),
            }
        }:
            degrees = math.degrees(phase)
            numbers = [offset, amplitude, frequency, delay, damping, degrees]
            return _function("SIN", numbers)
        case {"piecewise": {"points": list(points)}}:
            return _function(
                "PWL", [number for point in points for number in _pair(point)]
            )
    pytest.fail(f"no SPICE waveform for {spec}")


def _pair(point: JsonValue) -> tuple[float, float]:
    match point:
        case [float(time), float(level)]:
            return time, level
    pytest.fail(f"a piecewise point is a time and a level, not {point}")


def _function(name: str, numbers: list[JsonValue] | list[float]) -> str:
    return f"{name}({' '.join(repr(number) for number in numbers)})"


@dataclass(frozen=True)
class Raw:
    """The vectors of one ngspice analysis, by lowercase name."""

    plot: str
    vectors: dict[str, npt.NDArray[np.complex128]]

    def real(self, name: str) -> Real:
        """The real part of the vector `name`."""
        return self.vectors[name].real


def ngspice(lines: list[str], analysis: str, directory: Path) -> Raw:
    """Runs one analysis command in ngspice and reads the raw file it writes."""
    raw = directory / "result.raw"
    deck = [
        *lines,
        SPICE_OPTIONS,
        ".control",
        "set filetype=ascii",
        analysis,
        f"write {raw} all",
        ".endc",
        ".end",
    ]
    path = directory / "circuit.cir"
    path.write_text("\n".join(deck) + "\n", encoding="utf-8")
    assert NGSPICE is not None
    done = subprocess.run(
        [NGSPICE, "-b", str(path)],
        capture_output=True,
        check=False,  # ngspice exits with 0 on most errors: the raw file tells
        text=True,
        timeout=120,
    )
    assert raw.is_file(), done.stdout + done.stderr
    result = _read_raw(raw.read_text(encoding="utf-8"))
    assert result.plot != "Constant values", done.stdout + done.stderr
    return result


def _read_raw(text: str) -> Raw:
    header, values = text.split("Values:\n")
    names = []
    plot = ""
    within = False
    for line in header.splitlines():
        if line.startswith("Plotname:"):
            plot = line.removeprefix("Plotname:").strip()
        elif line.startswith("Variables:"):
            within = True
        elif within:
            names.append(_spice_name(line.split()[1].lower()))
    # Each point starts with its index, a bare integer; values always have an exponent.
    numbers = [
        complex(*map(float, token.split(","))) if "," in token else complex(token)
        for token in values.split()
        if not token.isdigit()
    ]
    table = np.array(numbers).reshape(-1, len(names))
    return Raw(plot, {name: table[:, column] for column, name in enumerate(names)})


def _spice_name(name: str) -> str:
    """A raw file's vector name in the ``v(node)`` and ``i(device)`` forms."""
    if name.endswith("#branch"):
        return f"i({name.removesuffix('#branch')})"
    if name.startswith(("v(", "i(")) or name in {"time", "frequency"}:
        return name
    return f"v({name})"


# The tests.


def compare(jolty: Values, spice: Values, *, scale: float, tolerance: float) -> None:
    """Asserts both agree within `tolerance` times `scale`, the signal's size."""
    np.testing.assert_allclose(jolty, spice, rtol=0, atol=tolerance * scale)


def size(values: Values) -> float:
    """The largest magnitude among `values`, or 1 when they are all zero."""
    return float(np.abs(np.asarray(values)).max()) or 1.0


def compared(simulation: Simulation) -> list[tuple[int, results.Unknown]]:
    """The unknowns both engines report, with their columns in jolty's results.

    ngspice names a device's internal nodes, behind its series resistances, its own
    way, so only nets, source currents and sensed currents are compared.
    """
    return [
        (column, unknown)
        for column, unknown in enumerate(simulation.unknowns)
        if ":" not in unknown.name or unknown.name.endswith((":current", ":control"))
    ]


@pytest.mark.parametrize(
    "top",
    [
        divider,
        clipper,
        amplifier,
        rectifier,
        jfet_stage,
        p_jfet_follower,
        bjt_stage,
        bjt_amplifier,
        pnp_mirror,
        current_controlled,
    ],
)
def test_operating_points_agree(top: Callable[[], Component], tmp_path: Path) -> None:
    simulation = Simulation(top())
    point = simulation.run(OperatingPoint(), TIGHT)
    raw = ngspice(netlist(simulation.circuit), "op", tmp_path)
    for column, unknown in compared(simulation):
        spice = raw.real(vector(simulation.circuit, unknown))[0]
        scale = max(abs(spice), 1e-3)
        compare(point.values[column], spice, scale=scale, tolerance=1e-6)


@pytest.mark.parametrize(
    ("top", "source", "levels"),
    [
        (divider, "V1", linear(-5 * V, 5 * V, 21)),
        (clipper, "V1", linear(-2 * V, 2 * V, 81)),
        (jfet_stage, "V1", linear(-3 * V, 1 * V, 81)),
        (jfet_curves, "V2", linear(-2 * V, 6 * V, 81)),
        (p_jfet_follower, "V1", linear(-4 * V, 2 * V, 61)),
        (bjt_stage, "V1", linear(-1 * V, 3 * V, 81)),
        (pnp_mirror, "V1", linear(0 * V, 12 * V, 61)),
        (current_controlled, "V1", linear(-3 * V, 3 * V, 13)),
    ],
)
def test_dc_sweeps_agree(
    top: Callable[[], Component], source: str, levels: list[Quantity], tmp_path: Path
) -> None:
    simulation = Simulation(top())
    sweep = simulation.run(DcSweep(source, levels), TIGHT)
    first, last = float(sweep.levels[0]), float(sweep.levels[-1])
    step = (last - first) / (len(levels) - 1)
    command = f"dc V{source} {first!r} {last!r} {step!r}"
    raw = ngspice(netlist(simulation.circuit), command, tmp_path)
    for column, unknown in compared(simulation):
        spice = raw.real(vector(simulation.circuit, unknown))
        compare(sweep.values[:, column], spice, scale=size(spice), tolerance=1e-6)


@pytest.mark.parametrize(
    "top",
    [low_pass, resonator, amplifier, jfet_stage, bjt_amplifier, current_controlled],
)
def test_ac_responses_agree(top: Callable[[], Component], tmp_path: Path) -> None:
    simulation = Simulation(top())
    response = simulation.run(Ac("V1", decade(10 * Hz, 10 * MHz, 20)), TIGHT)
    lines = netlist(simulation.circuit, excited="V1")
    raw = ngspice(lines, "ac dec 20 10 10meg", tmp_path)
    np.testing.assert_allclose(response.frequencies, raw.real("frequency"), rtol=1e-9)
    for column, unknown in compared(simulation):
        spice = raw.vectors[vector(simulation.circuit, unknown)]
        compare(response.values[:, column], spice, scale=size(spice), tolerance=1e-6)


@pytest.mark.parametrize(
    ("top", "stop", "step"),
    [
        (low_pass, 8 * us, 10 * ns),
        (resonator, 60 * us, 50 * ns),
        (rectifier, 3 * ms, 1 * us),
        (driven_tank, 40 * us, 20 * ns),
        (jfet_stage, 15 * us, 10 * ns),
        (bjt_amplifier, 4 * us, 2 * ns),
    ],
)
def test_transients_agree(
    top: Callable[[], Component], stop: Quantity, step: Quantity, tmp_path: Path
) -> None:
    simulation = Simulation(top())
    waveforms = simulation.run(Transient(stop=stop, max_step=step), TIGHT)
    command = f"tran {step.value!r} {stop.value!r} 0 {step.value!r}"
    raw = ngspice(netlist(simulation.circuit), command, tmp_path)
    times = raw.real("time")
    for column, unknown in compared(simulation):
        jolty = np.interp(times, waveforms.times, waveforms.values[:, column])
        spice = raw.real(vector(simulation.circuit, unknown))
        compare(jolty, spice, scale=size(spice), tolerance=1e-3)

# Copyright 2026 Maxence Faldor
"""Running ngspice on jolty's circuits and comparing what both engines find.

A flat circuit is written as a SPICE netlist; a component read from a SPICE library is
written as an instance of its subcircuit, the library included. Both engines run with
tight tolerances, ngspice at the temperature where its thermal voltage is jolty's.
"""

import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import JsonValue

from jolty import Options, Role, Simulation, _core, ir, results
from jolty.spice import models

NGSPICE = shutil.which("ngspice")

SKIP = pytest.mark.skipif(NGSPICE is None, reason="ngspice is not installed")
"""Marks tests that need ngspice."""

type Real = npt.NDArray[np.float64]
type Values = float | Real | npt.NDArray[np.complex128]

VOLTAGE_TOLERANCE = 1e-9
"""Both engines' absolute tolerance on voltages, in volts."""

CURRENT_TOLERANCE = 1e-14
"""Both engines' absolute tolerance on currents, in amperes."""

TIGHT = Options(
    relative_tolerance=1e-6,
    voltage_tolerance=VOLTAGE_TOLERANCE,
    current_tolerance=CURRENT_TOLERANCE,
    charge_tolerance=1e-18,
)
"""jolty's options, as tight as ngspice's below."""

_SPICE_THERMAL_VOLTAGE_PER_KELVIN = 1.38064852e-23 / 1.6021766208e-19
"""ngspice's Boltzmann constant over the elementary charge, from CODATA 2014."""


def spice_celsius(temperature: float) -> float:
    """Where ngspice's thermal voltage is jolty's at `temperature`, in degrees Celsius.

    The temperature is in kelvins. Running ngspice there, and its models' nominal
    temperature where it is jolty's, compares the devices rather than the physical
    constants.
    """
    thermal_voltage = temperature * _core.BOLTZMANN / _core.ELEMENTARY_CHARGE
    return thermal_voltage / _SPICE_THERMAL_VOLTAGE_PER_KELVIN - 273.15


SPICE_OPTIONS = (
    ".options reltol=1e-6"
    f" vntol={VOLTAGE_TOLERANCE!r} abstol={CURRENT_TOLERANCE!r} chgtol=1e-18"
)
"""ngspice's options, as tight as jolty's."""

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
    "bipolar_transistor": Element("Q", ("collector", "base", "emitter", "substrate")),
    "mosfet": Element("M", ("drain", "gate", "source", "bulk")),
    "behavioral_voltage_source": Element("B", ("positive", "negative")),
    "behavioral_current_source": Element("B", ("positive", "negative")),
    "voltage_controlled_switch": Element("A", _CONTROLLED),
}


def _spice(table: models.Table) -> dict[str, tuple[str, float]]:
    """Each jolty parameter's first SPICE name, and the factor to SPICE's units."""
    names: dict[str, tuple[str, float]] = {}
    for spice, (jolty, factor) in table.items():
        names.setdefault(jolty, (spice, 1 / factor))
    return names


_NUMBERED = {
    "gate_material": {material: tpg for tpg, material in models.GATE_TYPES.items()},
    "substrate": {kind: subs for subs, kind in models.CONSTRUCTIONS.items()},
}
"""SPICE's number for each value of a parameter it numbers: ``TPG`` and ``SUBS``."""


def _model_card(model: str, values: dict[str, JsonValue]) -> tuple[str, str, str]:
    """The SPICE type and parameters of a device's model card, and its instance's."""
    instance: dict[str, tuple[str, float]] = {}
    match model:
        case "diode":
            kind, names = "D", _spice(models.DIODE)
        case "jfet":
            kind, names = f"{values.get('channel', 'n')}jf".upper(), _spice(models.JFET)
        case "bipolar_transistor":
            kind = str(values.get("polarity", "npn")).upper()
            names = _spice(models.BIPOLAR) | {"substrate": ("subs", 1.0)}
        case "mosfet":
            kind = f"{values.get('channel', 'n')}mos".upper()
            names = _spice(models.MOSFET) | {
                "gate_material": ("tpg", 1.0),
                "level": ("level", 1.0),
            }
            instance = _spice(models.MOSFET_INSTANCE)
        case _:
            pytest.fail(f"{model} has no model card")
    names["nominal_temperature"] = ("tnom", 1.0)
    given = {
        key: value
        for key, value in values.items()
        if key not in {"channel", "polarity"}
    }
    card = " ".join(
        f"{names[key][0]}={_spice_value(key, value, names[key][1])}"
        for key, value in given.items()
        if key not in instance
    )
    element = " ".join(
        f"{instance[key][0]}={_spice_value(key, value, instance[key][1])}"
        for key, value in given.items()
        if key in instance
    )
    return kind, card, element


def _spice_value(key: str, value: JsonValue, factor: float) -> str:
    """The value of parameter `key` in SPICE's units."""
    match value:
        case str(name):
            return repr(_NUMBERED[key][name])
        case int(number):
            return repr(number)
        case float() if key == "nominal_temperature":
            return repr(spice_celsius(value))
        case float():
            return repr(float(value) * factor)
    pytest.fail(f"no SPICE value for {value}")


def element(instance: ir.Instance) -> str:
    """The SPICE name of a device instance: its model's letter, then its id."""
    assert isinstance(instance.of, ir.DeviceOf), "the corpus is flat"
    return ELEMENTS[instance.of.device].letter + instance.id


def netlist(circuit: ir.Circuit, *, excited: str | None = None) -> list[str]:
    """The SPICE lines of a circuit; `excited` is the source an AC run excites.

    Its top places devices, and components read from SPICE libraries, which become
    instances of their subcircuits.
    """
    top = circuit.components[circuit.top]
    ground = next(port.name for port in top.ports if port.role is Role.GROUND)
    lines = [f"* {circuit.top}"]
    for instance in top.instances:
        if isinstance(instance.of, ir.ComponentOf):
            lines.append(_subcircuit(circuit, instance, ground))
            continue
        model, values = instance.of.device, instance.parameters
        nets = (
            instance.terminals[name]
            for name in ELEMENTS[model].terminals
            if name in instance.terminals
        )
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
            case "diode" | "jfet" | "bipolar_transistor" | "mosfet":
                kind, card, parameters = _model_card(model, values)
                lines.append(f".model M{instance.id} {kind}({card})")
                value = f"M{instance.id} {parameters}".rstrip()
            case "behavioral_voltage_source" | "behavioral_current_source":
                expression, senses = _behavior(instance, ground)
                lines.extend(senses)
                output = "V" if model == "behavioral_voltage_source" else "I"
                value = f"{output}={{{expression}}}"
            case "voltage_controlled_switch":
                # ngspice runs PSpice's switch as the XSPICE model `pswitch`, whose
                # control port draws current through `r_cntl_in`: made negligible here.
                positive, negative, control_positive, control_negative = nodes.split()
                control = f"%gd({control_positive} {control_negative})"
                nodes = f"{control} %gd({positive} {negative})"
                card = " ".join(
                    f"{name}={values[key]!r}"
                    for key, name in _SWITCH_PARAMETERS.items()
                )
                lines.append(
                    f".model M{instance.id} pswitch({card} r_cntl_in=1e30 log=TRUE)"
                )
                value = f"M{instance.id}"
            case "resistor" | "capacitor" | "inductor":
                number, *drift = values.items()
                value = " ".join(
                    [repr(number[1])]
                    + [f"{_PASSIVE_DRIFT[key]}={amount!r}" for key, amount in drift]
                )
            case _:
                (number,) = values.values()
                value = repr(number)
        lines.append(f"{element(instance)} {nodes} {value}")
    return lines


def _subcircuit(circuit: ir.Circuit, instance: ir.Instance, ground: str) -> str:
    """The instance of a subcircuit read from a SPICE library, its pins in order.

    Its ground port is SPICE's global ground, and the parameters its component's name
    lists, as in ``divider[top=1000.0]``, are its own.
    """
    assert isinstance(instance.of, ir.ComponentOf)
    name = instance.of.component
    pins = [
        port.name
        for port in circuit.components[name].ports
        if port.role is not Role.GROUND
    ]
    nets = (instance.terminals[pin] for pin in pins)
    nodes = " ".join("0" if net == ground else net for net in nets)
    subcircuit, _, listed = name.removesuffix("]").partition("[")
    parameters = f" params: {listed.replace(',', ' ')}" if listed else ""
    return f"X{instance.id} {nodes} {subcircuit}{parameters}"


_PASSIVE_DRIFT = {
    "temperature_coefficient": "tc1",
    "quadratic_temperature_coefficient": "tc2",
    "exponential_temperature_coefficient": "tce",
}
"""The SPICE name of each temperature coefficient of a passive device."""

_SWITCH_PARAMETERS = {
    "on_voltage": "cntl_on",
    "off_voltage": "cntl_off",
    "on_resistance": "r_on",
    "off_resistance": "r_off",
}


def _behavior(instance: ir.Instance, ground: str) -> tuple[str, list[str]]:
    """A behavioral source's expression for ngspice, and its senses.

    ngspice reads a current through a zero-volt source, one per port in its place.
    """
    expression = instance.parameters["expression"]
    assert isinstance(expression, str)

    def net(terminal: str) -> str:
        name = instance.terminals[terminal]
        return "0" if name == ground else name

    def voltage(match: re.Match[str]) -> str:
        return f"v({','.join(net(name) for name in match.groups() if name)})"

    senses: dict[str, str] = {}

    def current(match: re.Match[str]) -> str:
        port = match[1]
        sense = f"V{element(instance)}{port}"
        ends = (net(f"{port}_positive"), net(f"{port}_negative"))
        senses[sense] = f"{sense} {ends[0]} {ends[1]} DC 0"
        return f"i({sense})"

    expression = re.sub(r"\bv\((\w+)(?:,\s*(\w+))?\)", voltage, expression)
    expression = re.sub(r"\bi\((\w+)\)", current, expression)
    return expression, list(senses.values())


def _sense(instance: ir.Instance) -> str:
    """The zero-volt source sensing a current-controlled source's control current."""
    return f"V{element(instance)}SENSE"


def vector(circuit: ir.Circuit, unknown: results.Unknown) -> str:
    """The name of the ngspice vector of a jolty unknown, a voltage or a current.

    A net within a placed subcircuit, such as ``U1/11``, is ngspice's ``xu1.11``.
    """
    if unknown.quantity == "voltage":
        first, *rest = unknown.name.lower().split("/")
        return f"v({'.'.join([f'x{first}', *rest]) if rest else first})"
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


def ngspice(  # noqa: PLR0913 - a deck, its analysis, and how ngspice runs it
    lines: list[str],
    analysis: str,
    *,
    libraries: tuple[Path, ...] = (),
    pspice: bool = False,
    options: str = SPICE_OPTIONS,
    temperature: float = _core.REFERENCE_TEMPERATURE,
) -> Raw:
    """Runs one analysis command in ngspice and reads the raw file it writes.

    ngspice includes `libraries`, and reads them as PSpice writes them, in its PSpice
    compatibility mode, if `pspice`. `options` replaces the tight options. It runs at
    jolty's `temperature`, in kelvins, its models' nominal temperature at jolty's 27 °C.
    """
    nominal = spice_celsius(_core.REFERENCE_TEMPERATURE)
    deck = [
        lines[0],
        *(f".include {library.resolve()}" for library in libraries),
        *lines[1:],
        options,
        f".options temp={spice_celsius(temperature)!r} tnom={nominal!r}",
        ".control",
        "set filetype=ascii",
        analysis,
        "write result.raw all",
        ".endc",
        ".end",
    ]
    assert NGSPICE is not None
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        if pspice:
            settings = directory / ".spiceinit"
            settings.write_text("set ngbehavior=ps\n", encoding="utf-8")
        (directory / "circuit.cir").write_text("\n".join(deck) + "\n", encoding="utf-8")
        done = subprocess.run(
            [NGSPICE, "-b", "circuit.cir"],
            capture_output=True,
            check=False,  # ngspice exits with 0 on most errors: the raw file tells
            text=True,
            timeout=300,
            cwd=directory,
        )
        raw = directory / "result.raw"
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


def compare(
    jolty: Values,
    spice: Values,
    unknown: results.Unknown,
    *,
    scale: float,
    tolerance: float,
) -> None:
    """Asserts both agree within `tolerance` times `scale`, the signal's size.

    Values within the engines' absolute tolerance for the unknown's quantity are noise,
    and agree too.
    """
    noise = CURRENT_TOLERANCE if unknown.quantity == "current" else VOLTAGE_TOLERANCE
    np.testing.assert_allclose(jolty, spice, rtol=0, atol=tolerance * scale + noise)


def closest(times: Real, values: Real, at: Real, targets: Real, window: float) -> Real:
    """The values of a waveform nearest `targets`, each within `window` of its time.

    The waveform is linear between `values` at `times`, and each target's time is in
    `at`. Two engines step through time differently, and a waveform interpolated
    between one's steps cuts the other's corners and curves; within a window of a step,
    the waveforms meet there.
    """
    starts, stops = at - window, at + window
    ends = np.stack([np.interp(starts, times, values), np.interp(stops, times, values)])
    lower, upper = ends.min(axis=0), ends.max(axis=0)
    firsts = np.searchsorted(times, starts, side="right")
    lasts = np.searchsorted(times, stops, side="left")
    for index, (first, last) in enumerate(zip(firsts, lasts, strict=True)):
        if first < last:
            inside = values[first:last]
            lower[index] = min(lower[index], inside.min())
            upper[index] = max(upper[index], inside.max())
    return np.clip(targets, lower, upper)


def size(values: Values) -> float:
    """The largest magnitude among `values`, or 1 when they are all zero."""
    return float(np.abs(np.asarray(values)).max()) or 1.0


def compared(simulation: Simulation) -> list[tuple[int, results.Unknown]]:
    """The unknowns both engines report, with their columns in jolty's results.

    Both report the nets, the top's source currents and the currents sensed. ngspice
    names a device's internal nodes its own way, and has none of the nets jolty
    adds in series with a source whose current a subcircuit reads.
    """
    return [
        (column, unknown)
        for column, unknown in enumerate(simulation.unknowns)
        if _shared(unknown.name)
    ]


def _shared(name: str) -> bool:
    """Whether ngspice reports the unknown `name` too."""
    *path, leaf = name.split("/")
    if ":" in leaf:
        return not path and leaf.endswith((":current", ":control"))
    return "." not in leaf

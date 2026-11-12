# Copyright 2026 Maxence Faldor
"""Circuits written as SPICE netlists, and run in ngspice, to check jolty against it.

[`netlist`][jolty.spice.export.netlist] writes a circuit as SPICE lines, its hierarchy
flattened into devices whose ids and nets join their paths with underscores, such as
``RX1_R1`` and ``x1_mid``; components read from SPICE libraries stay instances of their
subcircuits. [`ngspice`][jolty.spice.export.ngspice] runs one analysis of those lines in
ngspice, with tolerances as tight as jolty's, and reads the raw file it writes;
[`vector`][jolty.spice.export.vector] names the ngspice vector of a jolty unknown.
Logic devices have no SPICE element, so a circuit that holds one cannot be written.
"""

import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

from jolty import _core, ir, results
from jolty.errors import JoltyError
from jolty.ir import Role
from jolty.spice import models

NGSPICE: Final = shutil.which("ngspice")
"""The ngspice executable on the path, or None when it is not installed."""

VOLTAGE_TOLERANCE: Final = 1e-9
"""The absolute tolerance on voltages ngspice runs with, in volts."""

CURRENT_TOLERANCE: Final = 1e-14
"""The absolute tolerance on currents ngspice runs with, in amperes."""

SPICE_OPTIONS: Final = (
    ".options reltol=1e-6"
    f" vntol={VOLTAGE_TOLERANCE!r} abstol={CURRENT_TOLERANCE!r} chgtol=1e-18"
)
"""ngspice's options, as tight as jolty's tightest."""

_SPICE_THERMAL_VOLTAGE_PER_KELVIN: Final = 1.38064852e-23 / 1.6021766208e-19
"""ngspice's Boltzmann constant over the elementary charge, from CODATA 2014."""

type Real = npt.NDArray[np.float64]


def spice_celsius(temperature: float) -> float:
    """Where ngspice's thermal voltage is jolty's at `temperature`, in degrees Celsius.

    The temperature is in kelvins. Running ngspice there, and its models' nominal
    temperature where it is jolty's, compares the devices rather than the physical
    constants.
    """
    thermal_voltage = temperature * _core.BOLTZMANN / _core.ELEMENTARY_CHARGE
    return thermal_voltage / _SPICE_THERMAL_VOLTAGE_PER_KELVIN - 273.15


def _unwritable(message: str) -> JoltyError:
    return JoltyError(f"cannot write the circuit as SPICE: {message}")


def _node(path: str) -> str:
    """A path as a flat SPICE name: its ids joined by underscores."""
    return path.replace("/", "_")


def flatten(circuit: ir.Circuit, subcircuits: Collection[str] = ()) -> ir.Circuit:
    """The circuit with its top placing only devices, and instances of `subcircuits`.

    Each instance inside a component instance takes its path as its id, its ids joined
    by underscores, and each net inside takes its path the same way, but those of the
    component's ports, which are the nets outside.

    Examples:
        >>> from jolty import Builder, Role, circuit, component, devices
        >>> from jolty.units import ohm
        >>> @component
        ... def cell(c: Builder) -> None:
        ...     a, b = c.port("a"), c.port("b")
        ...     first = c.place(devices.resistor(1 * ohm), id="R1")
        ...     second = c.place(devices.resistor(1 * ohm), id="R2")
        ...     c.connect(a, first.a)
        ...     c.connect(first.b, second.a, name="mid")
        ...     c.connect(second.b, b)
        >>> @component
        ... def top(c: Builder) -> None:
        ...     ground = c.port("gnd", Role.GROUND)
        ...     inner = c.place(cell(), id="X1")
        ...     c.connect(inner.a, name="in")
        ...     c.connect(inner.b, ground)
        >>> flat = flatten(circuit(top()))
        >>> [(i.id, i.terminals) for i in flat.components[flat.top].instances]
        [('X1_R1', {'a': 'in', 'b': 'X1_mid'}), ('X1_R2', {'a': 'X1_mid', 'b': 'gnd'})]
    """
    top = circuit.components[circuit.top]
    placed: list[ir.Instance] = []

    def walk(component: ir.Component, prefix: str, ports: dict[str, str]) -> None:
        for instance in component.instances:
            nets = {
                terminal: ports.get(net, f"{prefix}{net}")
                for terminal, net in instance.terminals.items()
            }
            match instance.of:
                case ir.ComponentOf(component=name) if name not in subcircuits:
                    walk(
                        circuit.components[name],
                        f"{prefix}{instance.id}_",
                        nets,
                    )
                case _:
                    placed.append(
                        instance.model_copy(
                            update={"id": f"{prefix}{instance.id}", "terminals": nets}
                        )
                    )

    walk(top, "", {port.name: port.name for port in top.ports})
    kept = {
        name: circuit.components[name]
        for name in {
            instance.of.component
            for instance in placed
            if isinstance(instance.of, ir.ComponentOf)
        }
    }
    flat = top.model_copy(update={"instances": tuple(placed)})
    return circuit.model_copy(update={"components": {**kept, circuit.top: flat}})


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
            message = f"a {model} has no model card"
            raise _unwritable(message)
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
    message = f"no SPICE value for {value}"
    raise _unwritable(message)


def element(instance: ir.Instance) -> str:
    """The SPICE name of a device instance: its model's letter, then its id."""
    if not isinstance(instance.of, ir.DeviceOf):
        message = f"{instance.id} is a component instance, not a device"
        raise _unwritable(message)
    if instance.of.device not in ELEMENTS:
        message = f"a {instance.of.device} has no SPICE element"
        raise _unwritable(message)
    return ELEMENTS[instance.of.device].letter + instance.id


def netlist(
    circuit: ir.Circuit,
    *,
    excited: str | None = None,
    subcircuits: Collection[str] = (),
) -> list[str]:
    """The SPICE lines of a circuit; `excited` is the source an AC run excites.

    Every component is flattened into its devices, but those named in `subcircuits`,
    components read from SPICE libraries, which become instances of their
    subcircuits, the libraries included.

    Raises:
        JoltyError: When the circuit holds a device SPICE has no element for, such as
            a logic device.
    """
    circuit = flatten(circuit, subcircuits)
    top = circuit.components[circuit.top]
    ground = next(port.name for port in top.ports if port.role is Role.GROUND)
    lines = [f"* {circuit.top}"]
    for instance in top.instances:
        if isinstance(instance.of, ir.ComponentOf):
            lines.append(_subcircuit(circuit, instance, ground))
            continue
        element(instance)
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
                value = _passive(model, values, nodes.split())
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
    if not isinstance(instance.of, ir.ComponentOf):
        message = f"{instance.id} is a device, not a subcircuit"
        raise _unwritable(message)
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

_CURVATURE = ("voltage_coefficient", "quadratic_voltage_coefficient")
"""A resistor's or a capacitor's voltage coefficients, linear then quadratic."""


def _passive(model: str, values: dict[str, JsonValue], nodes: list[str]) -> str:
    """A resistor's, capacitor's or inductor's value and temperature coefficients.

    A value that changes with the voltage is ngspice's expression for the resistance,
    or for the charge, of the voltage across the device.
    """
    number, *rest = values.items()
    drift = []
    for key, amount in rest:
        if key in _CURVATURE:
            continue
        if key not in _PASSIVE_DRIFT:
            message = f"a {model}'s {key} has no ngspice element parameter"
            raise _unwritable(message)
        drift.append(f"{_PASSIVE_DRIFT[key]}={amount!r}")
    linear, quadratic = (values.get(key, 0.0) for key in _CURVATURE)
    if linear == 0 and quadratic == 0:
        return " ".join([repr(number[1]), *drift])
    voltage = f"v({nodes[0]},{nodes[1]})"
    if model == "resistor":
        factor = f"1+{linear!r}*{voltage}+{quadratic!r}*{voltage}^2"
        written = f"R={{{number[1]!r}*({factor})}}"
    else:
        integral = f"{voltage}+{linear!r}*{voltage}^2/2+{quadratic!r}*{voltage}^3/3"
        written = f"Q={{{number[1]!r}*({integral})}}"
    return " ".join([written, *drift])


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
    if not isinstance(expression, str):
        message = f"{instance.id}'s expression is not text"
        raise _unwritable(message)

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


def vector(
    circuit: ir.Circuit, unknown: results.Unknown, subcircuits: Collection[str] = ()
) -> str:
    """The name of the ngspice vector of a jolty unknown, a voltage or a current.

    A net within a flattened component, such as ``X1/mid``, is the node ``x1_mid``; a
    net within a placed subcircuit, such as ``U1/11``, is ngspice's ``xu1.11``.

    Raises:
        JoltyError: When the unknown is a device's internal unknown, which ngspice
            names its own way.
    """
    flat = flatten(circuit, subcircuits)
    top = flat.components[flat.top]
    if unknown.quantity == "voltage":
        first, *rest = unknown.name.split("/")
        kept = top.instance(first)
        if rest and kept is not None and isinstance(kept.of, ir.ComponentOf):
            return f"v({'.'.join([f'x{first}', *rest]).lower()})"
        return f"v({_node(unknown.name).lower()})"
    path, _, label = unknown.name.rpartition(":")
    id = _node(path)
    found = [instance for instance in top.instances if instance.id == id]
    if not found:
        message = f"ngspice names the internal unknown {unknown.name} its own way"
        raise _unwritable(message)
    (instance,) = found
    source = _sense(instance) if label == "control" else element(instance)
    return f"i({source.lower()})"


_INSTANT: Final = 1e-40
"""A pulse's width of no time, as ngspice reads it: ngspice reads a width of 0 as the
transient's stop time, while 1e-40 s vanishes beside any rise longer than 1e-24 s."""


def _waveform(spec: JsonValue) -> str:
    match spec:
        case {"dc": float(level)}:
            return f"DC {level!r}"
        case {"pulse": dict(pulse)}:
            order = ("initial", "pulsed", "delay", "rise", "fall", "width", "period")
            numbers = [
                _INSTANT if key == "width" and pulse[key] == 0 else pulse[key]
                for key in order
                if key in pulse
            ]
            return _function("PULSE", numbers)
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
    message = f"no SPICE waveform for {spec}"
    raise _unwritable(message)


def _pair(point: JsonValue) -> tuple[float, float]:
    match point:
        case [float(time), float(level)]:
            return time, level
    message = f"a piecewise point is a time and a level, not {point}"
    raise _unwritable(message)


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
    if NGSPICE is None:
        message = "ngspice is not installed"
        raise JoltyError(message)
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        if pspice:
            settings = directory / ".spiceinit"
            settings.write_text("set ngbehavior=ps\n", encoding="utf-8")
        (directory / "circuit.cir").write_text("\n".join(deck) + "\n", encoding="utf-8")
        done = subprocess.run(  # noqa: S603 - ngspice runs the deck written here
            [NGSPICE, "-b", "circuit.cir"],
            capture_output=True,
            check=False,  # ngspice exits with 0 on most errors: the raw file tells
            text=True,
            timeout=300,
            cwd=directory,
        )
        raw = directory / "result.raw"
        if not raw.is_file():
            message = f"ngspice wrote no results:\n{done.stdout}{done.stderr}"
            raise JoltyError(message)
        result = _read_raw(raw.read_text(encoding="utf-8"))
    if result.plot == "Constant values":
        message = f"ngspice ran no analysis:\n{done.stdout}{done.stderr}"
        raise JoltyError(message)
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

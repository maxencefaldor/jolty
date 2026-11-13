# Copyright 2026 Maxence Faldor
"""Circuits written as SPICE netlists, and run in ngspice, to check jolty against it.

[`netlist`][jolty.spice.export.netlist] writes a circuit as SPICE lines, its hierarchy
flattened into devices whose ids and nets join their paths with underscores, such as
``RX1_R1`` and ``x1_mid``; components read from SPICE libraries stay instances of their
subcircuits. [`ngspice`][jolty.spice.export.ngspice] runs one analysis of those lines in
ngspice, with tolerances as tight as jolty's, and reads the raw file it writes;
[`vector`][jolty.spice.export.vector] names the ngspice vector of a jolty unknown.
Logic devices and thermal networks have no SPICE element, and ngspice heats no device by
what it dissipates, so a circuit that holds one cannot be written.
"""

import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict, Unpack

import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

from jolty import _core, ir, results
from jolty.errors import JoltyError
from jolty.ir import Role
from jolty.spice import models, sky130

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
    "coupled_inductors": Element("K", ()),
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
    "foundry_mosfet": Element("X", ("drain", "gate", "source", "bulk")),
    "behavioral_voltage_source": Element("B", ("positive", "negative")),
    "behavioral_current_source": Element("B", ("positive", "negative")),
    "voltage_controlled_switch": Element("B", _CONTROLLED),
    "current_controlled_switch": Element("B", _CONTROLLED),
    "photocell": Element("B", ("a", "b", "light_positive", "light_negative")),
    "voltage_controlled_memristor": Element("B", ("positive", "negative")),
    "current_controlled_memristor": Element("B", ("positive", "negative")),
}


def _nets(instance: ir.Instance, model: str) -> list[str]:
    """The nets of the terminals of a device of `model`, in its SPICE element's order.

    ngspice joins a bipolar transistor's missing substrate to the ground, where its
    substrate junction conducts gmin even without a diode. Joined instead to the
    terminal that junction lies on, the collector of an NPN, built vertically unless
    ``SUBS`` says otherwise, or the base of a PNP, built laterally, it carries nothing,
    as jolty's transistor without a substrate has no junction there.
    """
    terminals = instance.terminals
    nets = [terminals[name] for name in ELEMENTS[model].terminals if name in terminals]
    if model == "bipolar_transistor" and "substrate" not in terminals:
        npn = instance.parameters.get("polarity", "npn") == "npn"
        nets.append(terminals["collector" if npn else "base"])
    return nets


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
    if instance.parameters.get("thermal") is True:
        message = (
            f"{instance.id} is a heated {instance.of.device}, which ngspice has not"
        )
        raise _unwritable(message)
    return ELEMENTS[instance.of.device].letter + instance.id


_SURROUNDINGS: Final = frozenset({"thermal", "temperature", "temperature_offset"})
"""The parameters of a device's place rather than its model: whether it heats, and its
own temperature, which SPICE writes apart from its value or its model card."""


def _cool(parameters: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """A device's parameters but those of its place, `_SURROUNDINGS`."""
    return {key: value for key, value in parameters.items() if key not in _SURROUNDINGS}


_WARMED: Final = frozenset(
    {
        "resistor",
        "capacitor",
        "inductor",
        "coupled_inductors",
        "diode",
        "jfet",
        "bipolar_transistor",
        "mosfet",
    }
)
"""The devices whose SPICE elements take a temperature of their own."""

_UNMOVED: Final = frozenset(
    {
        "voltage_source",
        "current_source",
        "voltage_controlled_voltage_source",
        "voltage_controlled_current_source",
        "current_controlled_current_source",
        "current_controlled_voltage_source",
        "behavioral_voltage_source",
        "behavioral_current_source",
    }
)
"""The devices whose own temperature changes nothing."""


def _own_temperature(instance: ir.Instance, model: str) -> str:
    """The SPICE instance parameter that sets the device's own temperature, if any.

    A temperature it is held at is ngspice's ``TEMP``, and an offset from the
    simulation's its ``DTEMP``, each where ngspice's thermal voltage is jolty's.

    Raises:
        JoltyError: When the device's own temperature moves it, and its SPICE element
            has none.
    """
    match instance.parameters:
        case {"temperature": int() | float() as held}:
            written = f" temp={spice_celsius(held)!r}"
        case {"temperature_offset": int() | float() as offset}:
            written = f" dtemp={spice_celsius(offset) - spice_celsius(0.0)!r}"
        case _:
            return ""
    if model in _WARMED:
        return written
    if model in _UNMOVED:
        return ""
    message = (
        f"{instance.id} is a {model} at its own temperature, which ngspice's has not"
    )
    raise _unwritable(message)


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
            a logic device or a heated one.
    """
    circuit = flatten(circuit, subcircuits)
    top = circuit.components[circuit.top]
    ground = next(port.name for port in top.ports if port.role is Role.GROUND)
    placed = {
        instance.of.device
        for instance in top.instances
        if isinstance(instance.of, ir.DeviceOf)
    }
    if {"mosfet", "foundry_mosfet"} <= placed:
        message = (
            "a level-1 or level-3 MOSFET stands beside a foundry's, whose kit scales "
            "the sizes of every MOSFET"
        )
        raise _unwritable(message)
    lines = [f"* {circuit.top}"]
    for instance in top.instances:
        match instance.of:
            case ir.ComponentOf():
                lines.append(_subcircuit(circuit, instance, ground))
            case ir.DeviceOf(device="coupled_inductors"):
                own = _own_temperature(instance, "coupled_inductors")
                coupled = _coupled(
                    instance.id, _cool(instance.parameters), instance.terminals, ground
                )
                lines.extend(
                    f"{line}{own}" if line.startswith("L") else line for line in coupled
                )
            case ir.DeviceOf(device=model):
                lines.extend(_device(instance, model, ground, excited))
    return lines


def _device(
    instance: ir.Instance, model: str, ground: str, excited: str | None
) -> list[str]:
    """The SPICE lines of a device of `model`: its element's, after any it needs.

    Raises:
        JoltyError: When SPICE has no element for the device.
    """
    element(instance)
    values = _cool(instance.parameters)
    own = _own_temperature(instance, model)
    nodes = " ".join("0" if net == ground else net for net in _nets(instance, model))
    lines: list[str] = []
    match model:
        case "voltage_source" | "current_source":
            ac = " AC 1" if instance.id == excited else ""
            value = _waveform(values["waveform"]) + ac
        case "current_controlled_current_source" | "current_controlled_voltage_source":
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
        case "voltage_controlled_switch" | "current_controlled_switch":
            nodes, value = _switch(instance, model, values, nodes.split(), lines)
        case (
            "photocell"
            | "voltage_controlled_memristor"
            | "current_controlled_memristor"
        ):
            nodes, value = _varying(instance.id, model, values, nodes.split(), lines)
        case "resistor" | "capacitor" | "inductor":
            value = _passive(model, values, nodes.split(), (instance.id, lines))
        case "foundry_mosfet":
            try:
                value = sky130.written(values)
            except ValueError as error:
                raise _unwritable(str(error)) from None
        case _:
            (number,) = values.values()
            value = repr(number)
    lines.append(f"{element(instance)} {nodes} {value}{own}")
    return lines


def _varying(
    id: str,
    model: str,
    values: Mapping[str, JsonValue],
    nodes: list[str],
    lines: list[str],
) -> tuple[str, str]:
    """The nodes and the value of a resistance an expression varies, after its lines.

    A photocell's conductance follows its light, and a memristor's its state; `lines`
    gains those that lag the one and move the other.
    """
    a, b, *light = nodes
    if model == "photocell":
        conductance, moving = _photocell(id, values, light)
    else:
        conductance, moving = _memristor(id, model, values, [a, b])
    lines.extend(moving)
    return f"{a} {b}", f"I={{V({a},{b})*{conductance}}}"


_PHOTOCELL: Final = {
    "illuminance": 10.0,
    "gamma": 0.7,
    "dark_resistance": 1e6,
    "response_time": 0.0,
}
"""A photocell's parameters, as the engine takes them when left out."""

_DARKNESS: Final = 1e-6
"""The illuminance, in lux, below which a photocell's response rounds off to dark."""


def _photocell(
    id: str, values: Mapping[str, JsonValue], light: list[str]
) -> tuple[str, list[str]]:
    """A photocell's conductance as an ngspice expression, and the lines that lag it.

    Its response to the light between the nodes `light` lags it through 1 Ω and a
    capacitance of its response time in farads, when it has one.
    """
    given = {**_PHOTOCELL, **values}
    illuminance = f"V({light[0]},{light[1]})"
    lit = f"(({illuminance})+sqrt(({illuminance})^2+{_DARKNESS!r}^2))/2"
    settled = f"pow({lit}/{given['illuminance']!r},{given['gamma']!r})"
    response, lines = settled, list[str]()
    if given["response_time"] != 0:
        node = f"_{id}_response"
        lines = [
            f"B{id}_settled _{id}_settled 0 V={{{settled}}}",
            f"R{id}_lag _{id}_settled {node} 1",
            f"C{id}_lag {node} 0 {given['response_time']!r}",
        ]
        response = f"V({node})"
    conductance = f"(1/{given['dark_resistance']!r}+{response}/{given['resistance']!r})"
    return conductance, lines


_MEMRISTOR: Final = {
    "on_exponent": 1.0,
    "off_exponent": 1.0,
    "law": "linear",
    "state": 1.0,
    "window": 0.01,
}
"""A memristor's parameters, as the engine takes them when left out."""

_HOLD: Final = 1e-12
"""The rate, per second, at which a memristor's state returns in ngspice to the state it
was made in: slower than any transient sees, it pins the state at an operating point,
which in ngspice holds no initial condition."""


def _memristor(
    id: str, model: str, values: Mapping[str, JsonValue], terminals: list[str]
) -> tuple[str, list[str]]:
    """A memristor's conductance as an ngspice expression, and the lines that move it.

    Its state is the voltage of a node of its own over 1 F, which a behavioral current
    charges at the rate the state moves, and which a transient starts from the state it
    was made in, as jolty's does. ngspice's own operating point, which holds no initial
    condition, finds the state of a memristor that nothing drives where it was made,
    held there by `_HOLD`, but that of one driven past a threshold at the bound it is
    driven to, where jolty holds the state it was made in.
    """
    given = {**_MEMRISTOR, **values}
    node = f"_{id}_state"
    state = f"V({node})"
    held = f"min(max({state},0),1)"
    on, off = (_number(id, given, key) for key in ("on_resistance", "off_resistance"))
    conductance = (
        f"(1/({on!r}+{off - on!r}*{held}))"
        if given["law"] == "linear"
        else f"(exp({-math.log(off / on)!r}*{held})/{on!r})"
    )
    voltage = f"V({terminals[0]},{terminals[1]})"
    if model == "voltage_controlled_memristor":
        drive = (
            f"{given['off_rate']!r}"
            f"*pow(max({voltage}/{given['off_voltage']!r}-1,0),{given['off_exponent']!r})"
            f"-{given['on_rate']!r}"
            f"*pow(max({voltage}/({given['on_voltage']!r})-1,0),{given['on_exponent']!r})"
        )
    else:
        drive = f"{voltage}*{conductance}/{given['charge']!r}"
    window = given["window"]
    off_distance, on_distance = f"(1-{state})/{window!r}", f"{state}/{window!r}"
    toward_off = f"(1-exp(-max({off_distance},0))+min({off_distance},0))"
    toward_on = f"(1-exp(-max({on_distance},0))+min({on_distance},0))"
    rate = f"max({drive},0)*{toward_off}+min({drive},0)*{toward_on}"
    hold = f"{_HOLD!r}*({state}-{given['state']!r})"
    lines = [
        f"B{id}_state 0 {node} I={{{rate}-{hold}}}",
        f"C{id}_state {node} 0 1",
        f".ic {state}={given['state']!r}",
    ]
    return conductance, lines


def _number(id: str, values: Mapping[str, JsonValue], key: str) -> float:
    """The number `key` among the parameters `values` of the device `id`.

    Raises:
        JoltyError: When it is no number.
    """
    match values.get(key):
        case int() | float() as found:
            return float(found)
    message = f"{id}'s {key} is no number"
    raise _unwritable(message)


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


_FLICKER: Final = {"flicker_coefficient": "kf", "flicker_exponent": "af"}
"""A resistor's flicker noise parameters, as ngspice's model card names them."""


def _passive(
    model: str,
    values: Mapping[str, JsonValue],
    nodes: list[str],
    card: tuple[str, list[str]] | None = None,
) -> str:
    """A resistor's, capacitor's or inductor's value and temperature coefficients.

    A value that changes with the voltage is ngspice's expression for the resistance,
    or for the charge, of the voltage across the device. A resistor's flicker noise is
    a model card's, which `card`, the device's id and the lines before its element's,
    takes.

    Raises:
        JoltyError: When a parameter has no ngspice counterpart, or a resistor whose
            value bends with its voltage makes flicker noise: ngspice's expression for
            a resistance makes no noise.
    """
    number, *rest = values.items()
    flicker = {_FLICKER[key]: amount for key, amount in rest if key in _FLICKER}
    named = ""
    if flicker and card is not None:
        id, lines = card
        written = " ".join(f"{name}={amount!r}" for name, amount in flicker.items())
        lines.append(f".model M{id} R({written})")
        named = f"M{id} "
    drift = []
    for key, amount in rest:
        if key in _CURVATURE or (named and key in _FLICKER):
            continue
        if key not in _PASSIVE_DRIFT:
            message = f"a {model}'s {key} has no ngspice element parameter"
            raise _unwritable(message)
        drift.append(f"{_PASSIVE_DRIFT[key]}={amount!r}")
    linear, quadratic = (values.get(key, 0.0) for key in _CURVATURE)
    if linear == 0 and quadratic == 0:
        return named + " ".join([repr(number[1]), *drift])
    if named:
        message = "a resistor whose value bends with its voltage makes no flicker noise"
        raise _unwritable(message + " in ngspice")
    voltage = f"v({nodes[0]},{nodes[1]})"
    if model == "resistor":
        factor = f"1+{linear!r}*{voltage}+{quadratic!r}*{voltage}*{voltage}"
        written = f"R={{{number[1]!r}*({factor})}}"
    else:
        # Products, not powers: ngspice's `^` raises the magnitude of a negative base.
        squared = f"{voltage}*{voltage}"
        integral = (
            f"{voltage}+{linear!r}*{squared}/2+{quadratic!r}*{squared}*{voltage}/3"
        )
        written = f"Q={{{number[1]!r}*({integral})}}"
    return " ".join([written, *drift])


def _coupled(
    id: str,
    values: Mapping[str, JsonValue],
    terminals: Mapping[str, str],
    ground: str,
) -> list[str]:
    """Coupled inductors as SPICE writes them.

    Each winding is an inductor ``L{id}_{k}``, and each coupling a mutual inductance
    ``K{id}_{j}_{k}``.
    """
    lines = []
    match values:
        case {"windings": list(windings), "couplings": list(couplings)}:
            pass
        case _:
            message = f"{id} is coupled inductors without windings and couplings"
            raise _unwritable(message)
    for number, winding in enumerate(windings, start=1):
        if not isinstance(winding, dict):
            message = f"winding {number} of {id} is no inductor"
            raise _unwritable(message)
        nodes = [
            "0" if net == ground else net
            for net in (terminals[f"a{number}"], terminals[f"b{number}"])
        ]
        written = _passive("inductor", winding, nodes)
        lines.append(f"L{id}_{number} {' '.join(nodes)} {written}")
    for coupling in couplings:
        match coupling:
            case {"windings": [int(first), int(second)], "coefficient": float(value)}:
                lines.append(
                    f"K{id}_{first}_{second} L{id}_{first} L{id}_{second} {value!r}"
                )
            case _:
                message = f"{id} has a coupling SPICE cannot write: {coupling}"
                raise _unwritable(message)
    return lines


_SWITCHES: Final = {
    "voltage_controlled_switch": (("on_voltage", 1.0), ("off_voltage", 0.0)),
    "current_controlled_switch": (("on_current", 1e-3), ("off_current", 0.0)),
}
"""Each switch's controls at which it is on and off, with the defaults jolty's switches
take, PSpice's; both take PSpice's resistances, 1 Ω on and 1 MΩ off, unless given."""


def _switch(
    instance: ir.Instance,
    model: str,
    values: Mapping[str, JsonValue],
    nodes: list[str],
    lines: list[str],
) -> tuple[str, str]:
    """A switch's nodes and value as a behavioral current, after the lines it needs.

    The current is the switch's own law, PSpice's: the logarithm of its resistance a
    cubic in the control's fraction of the way from off to on, held beyond either.
    ngspice's XSPICE switches, ``pswitch`` and ``aswitch``, follow the same law, but
    ngspice 42 gives their small-signal conductance the opposite sign of their DC one,
    which its behavioral sources do not. A current-controlled switch's control current
    flows through a zero-volt source, which the expression reads.
    """
    positive, negative, control_positive, control_negative = nodes
    if model == "current_controlled_switch":
        sense = _sense(instance)
        lines.append(f"{sense} {control_positive} {control_negative} DC 0")
        control = f"i({sense})"
    else:
        control = f"v({control_positive},{control_negative})"
    (on_name, on_default), (off_name, off_default) = _SWITCHES[model]

    def number(key: str, default: float) -> float:
        return _number(instance.id, {key: default, **values}, key)

    on, off = number(on_name, on_default), number(off_name, off_default)
    on_resistance = number("on_resistance", 1.0)
    off_resistance = number("off_resistance", 1e6)
    mean = math.log(math.sqrt(on_resistance * off_resistance))
    ratio = math.log(on_resistance / off_resistance)
    m = f"(min(max(({control}-({off!r}))/({on!r}-({off!r})),0),1)-0.5)"
    # A product, not a power: ngspice's `^` raises the magnitude of a negative base.
    resistance = f"exp({mean!r}+{ratio!r}*(1.5*{m}-2*{m}*{m}*{m}))"
    return f"{positive} {negative}", f"I={{v({positive},{negative})/{resistance}}}"


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
    if instance.of == ir.DeviceOf(device="coupled_inductors"):
        return f"i(l{id.lower()}_{label.removeprefix('current')})"
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
    """The vectors of one ngspice analysis, by lowercase name.

    It keeps what ngspice printed while it ran the deck the analysis is of.
    """

    plot: str
    vectors: dict[str, npt.NDArray[np.complex128]]
    printed: str

    def real(self, name: str) -> Real:
        """The real part of the vector `name`."""
        return self.vectors[name].real


_TRANSIENT_OPERATING_POINT: Final = "Transient op started"
"""What ngspice prints when Newton iteration, gmin stepping and source stepping all fail
to find an operating point, and it takes the end of a transient from zero instead."""

_ABORTED: Final = "simulation(s) aborted"
"""What ngspice prints when an analysis stops short, as a transient whose step falls
below its smallest; the results it writes then end where it stopped."""

_SHORTEST_TRANSIENT_OPERATING_POINT: Final = "optran 1 1 1 1n 1n 0"
"""Keeps ngspice's first Newton iteration, gmin stepping and source stepping, and makes
the transient it falls back to a single nanosecond: its end is refused anyway, and the
default 10 µs can take minutes on a circuit that defeats the others."""


TIMEOUT: Final = 300
"""The seconds ngspice may run an analysis."""


type Compatibility = Literal["ps", "hsa"]
"""A compatibility mode of ngspice's: PSpice's, ``ps``, or HSPICE's, ``hsa``, which
foundries' process design kits ask for."""


class Running(TypedDict, total=False):
    """How ngspice runs a deck."""

    libraries: tuple[Path, ...]
    """The files it includes."""
    sections: tuple[tuple[Path, str], ...]
    """The sections of libraries it reads, each a file and a section's name, such as a
    foundry's model library at a process corner."""
    compatibility: Compatibility | None
    """The mode it reads the deck and the files in, its own unless given."""
    options: str
    """The options line, as tight as jolty's tightest unless given."""
    temperature: float
    """jolty's temperature, in kelvins, 27 °C unless given."""
    seed: int | None
    """The seed of ngspice's random numbers, such as a kit's mismatch draws, if any."""


def ngspice(
    lines: list[str],
    analysis: str,
    **running: Unpack[Running],
) -> Raw:
    """Runs one analysis command in ngspice and reads the raw file it writes.

    ngspice runs as [`ngspice_runs`][jolty.spice.export.ngspice_runs] says.

    Raises:
        JoltyError: As `ngspice_runs` does.
    """
    (result,) = ngspice_runs(lines, (analysis,), **running)
    return result


def ngspice_runs(
    lines: list[str],
    analyses: Sequence[str],
    **running: Unpack[Running],
) -> tuple[Raw, ...]:
    """Runs analysis commands in turn in one ngspice, and reads the plot each writes.

    ngspice includes the `libraries`, reads the `sections` of others, and reads them
    and the deck in its `compatibility` mode, its random numbers drawn from `seed`.
    `options` replaces the tight options. It runs at jolty's `temperature`, in kelvins,
    its models' nominal temperature at jolty's 27 °C. Each command may run several
    analyses, and a result is read from the plot of each.

    Raises:
        JoltyError: When ngspice is not installed, runs out of time, writes no results,
            runs no analysis, aborts one, or finds an operating point only as the end
            of a transient of fixed length, which it does not check to be one.
    """
    if NGSPICE is None:
        message = "ngspice is not installed"
        raise JoltyError(message)
    nominal = spice_celsius(_core.REFERENCE_TEMPERATURE)
    temperature = running.get("temperature", _core.REFERENCE_TEMPERATURE)
    deck = [
        lines[0],
        *(f".include {library.resolve()}" for library in running.get("libraries", ())),
        *(
            f'.lib "{library.resolve()}" {section}'
            for library, section in running.get("sections", ())
        ),
        *lines[1:],
        running.get("options", SPICE_OPTIONS),
        f".options temp={spice_celsius(temperature)!r} tnom={nominal!r}",
        *(
            f".options seed={seed}"
            for seed in [running.get("seed")]
            if seed is not None
        ),
        ".control",
        "set filetype=ascii",
        "set appendwrite",
        _SHORTEST_TRANSIENT_OPERATING_POINT,
        *(f"{analysis}\nwrite result.raw all" for analysis in analyses),
        ".endc",
        ".end",
    ]
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        compatibility = running.get("compatibility")
        if compatibility is not None:
            settings = directory / ".spiceinit"
            settings.write_text(f"set ngbehavior={compatibility}\n", encoding="utf-8")
        (directory / "circuit.cir").write_text("\n".join(deck) + "\n", encoding="utf-8")
        try:
            done = subprocess.run(  # noqa: S603 - ngspice runs the deck written here
                [NGSPICE, "-b", "circuit.cir"],
                capture_output=True,
                check=False,  # ngspice exits with 0 on most errors: the raw file tells
                text=True,
                timeout=TIMEOUT,
                cwd=directory,
            )
        except subprocess.TimeoutExpired:
            message = f"ngspice ran for more than {TIMEOUT} s"
            raise JoltyError(message) from None
        raw = directory / "result.raw"
        printed = done.stdout + done.stderr
        if not raw.is_file():
            message = f"ngspice wrote no results:\n{printed}"
            raise JoltyError(message)
        found = _read_raws(raw.read_text(encoding="utf-8"), printed)
    if len(found) != len(analyses) or any(
        result.plot == "Constant values" for result in found
    ):
        message = f"ngspice ran no analysis:\n{printed}"
        raise JoltyError(message)
    if _ABORTED in printed:
        message = f"ngspice aborted the analysis:\n{printed}"
        raise JoltyError(message)
    if _TRANSIENT_OPERATING_POINT in printed:
        message = (
            "ngspice found no operating point but by a transient of fixed length, "
            f"whose end it does not check to be one:\n{printed}"
        )
        raise JoltyError(message)
    return found


def _read_raws(text: str, printed: str) -> tuple[Raw, ...]:
    """Each plot of a raw file, which begins at its title, with what ngspice printed."""
    plots = re.split(r"^(?=Title:)", text, flags=re.MULTILINE)
    return tuple(_read_raw(plot, printed) for plot in plots if plot.strip())


def _read_raw(text: str, printed: str) -> Raw:
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
    return Raw(
        plot,
        {name: table[:, column] for column, name in enumerate(names)},
        printed,
    )


_SCALES: Final = frozenset({"time", "frequency"})
"""The scales of a raw file's vectors."""

_NOISES: Final = ("onoise", "inoise")
"""The prefixes of the vectors of noise densities, at the output and referred to the
input, in all and of each device's sources, a device within a subcircuit after a dot
and any other after an underscore."""


def _spice_name(name: str) -> str:
    """A raw file's vector name in the ``v(node)`` and ``i(device)`` forms."""
    if name.endswith("#branch"):
        return f"i({name.removesuffix('#branch')})"
    if name.startswith(("v(", "i(", *_NOISES)) or name in _SCALES:
        return name
    return f"v({name})"

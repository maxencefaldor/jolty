# Copyright 2026 Maxence Faldor
"""Circuits written as SPICE netlists, and run in ngspice, to check jolty against it.

[`netlist`][jolty.spice.export.netlist] writes a circuit as SPICE lines, its hierarchy
flattened into devices whose ids and nets join their paths with underscores, such as
``RX1_R1`` and ``x1_mid``; components read from SPICE libraries stay instances of their
subcircuits. [`jolty.spice.ngspice`][jolty.spice.ngspice] runs those lines, and
[`vector`][jolty.spice.export.vector] names the ngspice vector of a jolty unknown.
Logic devices and thermal networks have no SPICE element, and ngspice heats no device by
what it dissipates, so a circuit that holds one cannot be written.
"""

import json
import math
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import JsonValue, TypeAdapter

from jolty import _core, ir, results
from jolty.errors import JoltyError
from jolty.ir import Role
from jolty.spice import models, sky130
from jolty.spice.ngspice import spice_celsius


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


_DARKNESS: Final = 1e-6
"""The illuminance, in lux, below which a photocell's response rounds off to dark."""


def _photocell(
    id: str, values: Mapping[str, JsonValue], light: list[str]
) -> tuple[str, list[str]]:
    """A photocell's conductance as an ngspice expression, and the lines that lag it.

    Its response to the light between the nodes `light` lags it through 1 Ω and a
    capacitance of its response time in farads, when it has one.
    """
    given = _defaulted("photocell", values)
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
    given = _defaulted(model, values)
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


_PARAMETERS: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


def _defaulted(model: str, values: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """The parameters `values` of a `model`, any left out given the engine's default."""
    return _PARAMETERS.validate_json(_core.defaulted(model, json.dumps(values)))


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
    "voltage_controlled_switch": ("on_voltage", "off_voltage"),
    "current_controlled_switch": ("on_current", "off_current"),
}
"""Each switch's controls at which it is on and off."""


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
    given = _defaulted(model, values)
    on, off, on_resistance, off_resistance = (
        _number(instance.id, given, key)
        for key in (*_SWITCHES[model], "on_resistance", "off_resistance")
    )
    mean = math.log(math.sqrt(on_resistance * off_resistance))
    ratio = math.log(on_resistance / off_resistance)
    m = f"(min(max(({control}-({off!r}))/({on!r}-({off!r})),0),1)-0.5)"
    # A product, not a power: ngspice's `^` raises the magnitude of a negative base.
    resistance = f"exp({mean!r}+{ratio!r}*(1.5*{m}-2*{m}*{m}*{m}))"
    return f"{positive} {negative}", f"I={{v({positive},{negative})/{resistance}}}"


def _behavior(instance: ir.Instance, ground: str) -> tuple[str, list[str]]:
    """A behavioral source's expression for ngspice, and its senses.

    jolty reads the expression into its tree, which `_written` writes in ngspice's
    language. ngspice reads a current through a zero-volt source, one per port in its
    place.
    """
    expression = instance.parameters["expression"]
    if not isinstance(expression, str):
        message = f"{instance.id}'s expression is not text"
        raise _unwritable(message)

    def net(terminal: str) -> str:
        name = instance.terminals[terminal]
        return "0" if name == ground else name

    senses: dict[str, str] = {}

    def read(variable: JsonValue) -> str:
        match variable:
            case {"voltage": [str(positive), None]}:
                return f"v({net(positive)})"
            case {"voltage": [str(positive), str(negative)]}:
                return f"v({net(positive)},{net(negative)})"
            case {"current": str(port)}:
                sense = f"V{element(instance)}{port}"
                ends = (net(f"{port}_positive"), net(f"{port}_negative"))
                senses[sense] = f"{sense} {ends[0]} {ends[1]} DC 0"
                return f"i({sense})"
        message = f"{instance.id}'s expression reads {variable}, which ngspice cannot"
        raise _unwritable(message)

    try:
        tree = _TREE.validate_json(_core.expression_tree(expression))
        written = _written(tree, read)
    except ValueError as error:
        message = f"{instance.id}'s expression: {error}"
        raise _unwritable(message) from None
    return written, list(senses.values())


_TREE: Final = TypeAdapter[JsonValue](JsonValue)
"""Reads the tree of an expression, as the engine writes it in JSON."""

_INFIX: Final = frozenset(
    {"+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!=", "&&", "||"}
)
"""The operators of two operands that jolty and ngspice write and read alike."""


def _written(term: JsonValue, read: Callable[[JsonValue], str]) -> str:
    """A term of a jolty expression's tree in ngspice's language.

    Every operation is written in parentheses, so ngspice's own precedence and grouping
    never come into it, and `read` writes each voltage or current the term reads.

    Raises:
        ValueError: When the term holds what ngspice cannot read, such as a number
            that is not finite.
    """
    match term:
        case int() | float() if math.isfinite(term):
            return repr(float(term)) if term >= 0 else f"({float(term)!r})"
        case "time":
            return "time"
        case {"voltage": _} | {"current": _}:
            return read(term)
        case {"operation": str(name), "operands": list(operands)}:
            written = [_written(operand, read) for operand in operands]
            return _operation(name, written, operands)
        case {
            "table": dict() | str() | int() | float() as table,
            "abscissae": list(abscissae),
            "ordinates": list(ordinates),
            "rounding": int() | float() as rounding,
        }:
            points = [
                (float(x), float(y))
                for x, y in zip(abscissae, ordinates, strict=True)
                if isinstance(x, int | float) and isinstance(y, int | float)
            ]
            return _interpolation(_written(table, read), points, float(rounding))
    message = f"it holds {term!r}, which ngspice cannot read"
    raise ValueError(message)


def _operation(name: str, written: list[str], operands: list[JsonValue]) -> str:
    """The operation `name` on its operands, each `written`, as jolty computes it.

    Where ngspice's functions differ from jolty's, it is written as jolty computes it:
    ngspice's `^` and `pow` raise a negative number's magnitude, and its `limit` is
    another function.
    """
    match name, written:
        case "^", [base, raised]:
            return _power(base, operands[1], raised)
        case "limit", [x, a, b]:
            return f"min(max({x},min({a},{b})),max({a},{b}))"
        case "if", [condition, chosen, otherwise]:
            return f"({condition}?{chosen}:{otherwise})"
        case "-" | "!", [operand]:
            return f"({name}{operand})"
        case _, [left, right] if name in _INFIX:
            return f"({left}{name}{right})"
    return f"{name}({','.join(written)})"


def _power(base: str, exponent: JsonValue, raised: str) -> str:
    """`base` to the power `exponent`, `raised` as written, as jolty computes it.

    ngspice raises a negative base's magnitude, where jolty's power of a negative base
    is real only for a constant integer exponent, `(-1)^n·|a|^n`: an even one is
    ngspice's `pow`, and an odd one its `pwr`, which keeps the base's sign. Any other
    power of a negative base fails in jolty, whose derivative by a varying exponent
    reads the base's logarithm.
    """
    match exponent:
        case int() | float() if float(exponent).is_integer() and int(exponent) % 2:
            return f"pwr({base},{raised})"
    return f"pow({base},{raised})"


def _interpolation(x: str, points: list[tuple[float, float]], rounding: float) -> str:
    """The interpolation of `points` at `x`, as jolty's tables compute it, in choices.

    `points` holds the table's own and one beyond each end. Each corner holds the
    inputs between the midpoints of the segments beside it, and is rounded there by
    XSPICE's parabola over `rounding` times the shorter of them; below the first
    segment's midpoint and above the last's, the input follows that segment's line.
    """
    xs, ys = [x_ for x_, _ in points], [y for _, y in points]
    last = len(points) - 1

    def line(start: int) -> str:
        slope = (ys[start + 1] - ys[start]) / (xs[start + 1] - xs[start])
        return f"({ys[start]!r}+({x}-({xs[start]!r}))*{slope!r})"

    # Each piece runs up to its bound, the last to no bound; a segment's line runs on
    # across the corners no parabola rounds.
    pieces: list[tuple[float, str]] = []
    for corner in range(1, last):
        below, above = xs[corner] - xs[corner - 1], xs[corner + 1] - xs[corner]
        domain = rounding * min(below, above)
        if domain <= 0:
            pieces.append((xs[corner], line(corner - 1)))
            continue
        lower, upper = (
            (ys[corner] - ys[corner - 1]) / below,
            (ys[corner + 1] - ys[corner]) / above,
        )
        right = xs[corner] + domain
        curvature = (upper - lower) / 4.0 * (1.0 / domain)
        linear = upper - 2.0 * curvature * right
        constant = ys[corner] + upper * domain - curvature * right * right
        constant -= linear * right
        parabola = f"({curvature!r}*{x}*{x}+{linear!r}*{x}+{constant!r})"
        pieces += [(xs[corner] - domain, line(corner - 1)), (right, parabola)]
    written = line(last - 1)
    for bound, piece in reversed(pieces):
        written = f"({x}<{bound!r}?{piece}:{written})"
    return written


def _sense(instance: ir.Instance) -> str:
    """The zero-volt source sensing a current-controlled source's control current."""
    return f"V{element(instance)}SENSE"


def vector(
    circuit: ir.Circuit, unknown: results.Unknown, subcircuits: Collection[str] = ()
) -> str:
    """The name of the ngspice vector of a jolty unknown: a voltage, current or state.

    A net within a flattened component, such as ``X1/mid``, is the node ``x1_mid``; a
    net within a placed subcircuit, such as ``U1/11``, is ngspice's ``xu1.11``. A
    device's state, such as ``M1:state``, is the voltage of the node the netlist gives
    it, ``_m1_state``.

    Raises:
        JoltyError: When the unknown is a device's internal unknown, which ngspice
            names its own way.
    """
    if unknown.quantity == "state":
        path, _, label = unknown.name.rpartition(":")
        return f"v(_{_node(path)}_{label})".lower()
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

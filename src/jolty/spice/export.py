# Copyright 2026 Maxence Faldor
"""Circuits written as SPICE netlists, and run in ngspice or Xyce, to check jolty.

[`netlist`][jolty.spice.export.netlist] writes a circuit as SPICE lines, its hierarchy
flattened into devices whose ids and nets join their paths with underscores, such as
``RX1_R1`` and ``x1_mid``; components read from SPICE libraries stay instances of their
subcircuits. The lines are in a simulator's dialect:
[`NGSPICE`][jolty.spice.export.NGSPICE]'s unless [`XYCE`][jolty.spice.export.XYCE]'s is
asked for.
[`jolty.spice.ngspice`][jolty.spice.ngspice] and [`jolty.spice.xyce`][jolty.spice.xyce]
run those lines, and [`vector`][jolty.spice.export.vector] names a simulator's vector of
a jolty unknown. Logic devices and thermal networks have no SPICE element, and neither
simulator heats a device by what it dissipates into a thermal network, so a circuit that
holds one cannot be written.
"""

import json
import math
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import JsonValue, TypeAdapter

from jolty import _core, ir
from jolty.devices import BSIM3_CARD
from jolty.errors import JoltyError
from jolty.ir import Role
from jolty.spice import models, ngspice, sky130, xyce
from jolty.spice.ngspice import CURRENT_TOLERANCE, VOLTAGE_TOLERANCE
from jolty.units import ZERO_CELSIUS
from jolty.unknowns import Unknown


def _unwritable(message: str) -> JoltyError:
    return JoltyError(f"cannot write the circuit as SPICE: {message}")


type Card = tuple[str, int]
"""A device model with a model card, and its level: 1 but for a MOSFET of level 3."""


@dataclass(frozen=True)
class Dialect:
    """A simulator's SPICE: what it calls jolty's devices, and what it cannot write.

    jolty's tables of SPICE names, `jolty.spice.models`, are ngspice's, so a dialect
    states only where its simulator departs from them.
    """

    simulator: str
    """The simulator's name, as its messages give it."""
    option_temperature: Callable[[float], float]
    """Where the simulator's thermal voltage is jolty's at a temperature in kelvins, in
    its degrees Celsius."""
    separator: str
    """What joins an instance of a subcircuit and a node within it, in the simulator's
    names of nodes."""
    instant: float
    """The width, in seconds, a pulse of no width is written with."""
    zero_level: bool
    """Whether a source that varies states its level at time zero as its DC level, as
    the simulator takes a level of 0 where none is stated, rather than jolty's, its
    waveform's at time zero."""
    flicker_over_oxide: bool
    """Whether a MOSFET's flicker coefficient is written over its oxide's capacitance,
    with ngspice's ``nlev=1``, rather than as SPICE's own."""
    cards: Mapping[Card, frozenset[str]] | None
    """The SPICE parameters, as jolty's tables name them, of each model card the
    simulator's model takes, or None where it takes every one jolty writes."""
    passives: Mapping[str, frozenset[str]] | None
    """The parameters of a resistor, a capacitor and an inductor the simulator's element
    takes, as SPICE names them, or None where it takes every one jolty writes."""
    expressions: str | None
    """Why the simulator reads none of jolty's expressions, or None where it reads
    them: of behavioral sources, switches, photocells, memristors, and resistors and
    capacitors whose values bend with their voltages."""
    refused: Mapping[str, str]
    """The devices the simulator has no element for that is jolty's, each with why."""


NGSPICE: Final = Dialect(
    simulator="ngspice",
    option_temperature=ngspice.option_temperature,
    separator=".",
    # ngspice reads a width of 0 as the transient's stop time, while 1e-40 s vanishes
    # beside any rise longer than 1e-24 s.
    instant=1e-40,
    zero_level=False,
    flicker_over_oxide=True,
    cards=None,
    passives=None,
    expressions=None,
    refused={},
)
"""ngspice 47's dialect, in which jolty's tables of SPICE names are written."""

_XYCE_DIODE: Final = frozenset(
    {"is", "n", "rs", "cjo", "vj", "m", "fc", "tt", "bv", "ibv", "nbv", "eg", "xti"}
    | {"kf", "af", "tnom"}
)
"""What Xyce 7.10.0's diode of level 1 takes of jolty's parameters: its ``addPar`` in
``src/DeviceModelPKG/OpenModels/N_DEV_Diode.C``, lines 100 to 366. It has no ``TCV``,
whose place its ``TBV1`` takes relative to the breakdown voltage rather than in
volts."""

_XYCE_BIPOLAR: Final = frozenset(
    {"is", "bf", "br", "nf", "nr", "vaf", "var", "ikf", "ikr", "ise", "ne", "isc"}
    | {"nc", "rb", "rbm", "irb", "re", "rc", "cje", "vje", "mje", "tf", "xtf", "vtf"}
    | {"itf", "cjc", "vjc", "mjc", "xcjc", "tr", "cjs", "vjs", "mjs", "fc", "eg"}
    | {"xti", "xtb", "kf", "af", "tnom"}
)
"""What Xyce's Gummel-Poon transistor, its BJT of level 1, takes: ``N_DEV_BJT.C``, lines
120 to 756. It has no substrate junction's current, ``ISS`` and ``NS``, and no ``SUBS``:
its substrate's capacitance always lies on the collector. Its excess phase, ``PTF``, is
its own, either way its option ``NEWEXCESSPHASE`` chooses (lines 1187 to 1436): with 60°
of it, `test_ngspice.py`'s ``delayed_stage`` leaves its collector 16% of its swing from
jolty's, which is ngspice's, so it is refused too."""

_XYCE_JFET: Final = frozenset(
    {"vto", "beta", "lambda", "rd", "rs", "cgs", "cgd", "pb", "is", "fc", "b", "kf"}
    | {"af", "tnom"}
)
"""What Xyce's JFET of level 1 takes: ``N_DEV_JFET.C``, lines 81 to 158. It has none of
the temperature coefficients ``VTOTC``, ``BEX``, ``EG`` and ``XTI``, and no noise."""

_XYCE_MOSFET: Final = frozenset(
    {"vto", "kp", "gamma", "phi", "lambda", "rd", "rs", "cbd", "cbs", "is", "pb"}
    | {"cgso", "cgdo", "cgbo", "rsh", "cj", "mj", "cjsw", "mjsw", "js", "tox", "ld"}
    | {"uo", "fc", "nsub", "nss", "tnom", "kf", "af", "tpg", "level"}
)
"""What Xyce's MOSFET of level 1 takes: ``N_DEV_MOSFET1.C``, lines 156 to 331. It has no
``DELVTO``, ``WD``, ``XL`` or ``XW``."""

_XYCE_LEVEL_THREE: Final = _XYCE_MOSFET | frozenset(
    {"eta", "delta", "nfs", "theta", "vmax", "kappa", "xj"}
)
"""What Xyce's MOSFET of level 3 takes: ``N_DEV_MOSFET3.C``, lines 165 to 370."""

XYCE: Final = Dialect(
    simulator="Xyce",
    option_temperature=xyce.option_temperature,
    separator=":",
    # Xyce reads a width of 0 as no width: only a width it is not given is its stop
    # time (`N_DEV_SourceData.C`, line 824).
    instant=0.0,
    # Xyce's operating point and DC sweeps take a source's DC level, 0 where none is
    # given, where its transient starts from its waveform at time zero.
    zero_level=True,
    flicker_over_oxide=False,
    expressions=(
        "jolty writes its expressions in ngspice's language, and not yet in Xyce's, "
        "whose functions read otherwise"
    ),
    cards={
        ("diode", 1): _XYCE_DIODE,
        ("bipolar_transistor", 1): _XYCE_BIPOLAR,
        ("jfet", 1): _XYCE_JFET,
        ("mosfet", 1): _XYCE_MOSFET,
        ("mosfet", 3): _XYCE_LEVEL_THREE,
    },
    passives={
        # `N_DEV_Resistor.C`, lines 194 to 204; `N_DEV_Capacitor.C`, 113 to 117;
        # `N_DEV_Inductor.C`, 89 to 94. Xyce's resistor makes no flicker noise.
        "resistor": frozenset({"tc1", "tc2", "tce"}),
        "capacitor": frozenset({"tc1", "tc2"}),
        "inductor": frozenset({"tc1", "tc2"}),
    },
    refused={
        "bsim3": (
            "Xyce's BSIM3, its MOSFET of level 9, is version 3.2.2, where jolty's is "
            "3.3.0"
        ),
        "vdmos": (
            "Xyce's VDMOS, its MOSFET of level 18, is another model than jolty's, "
            "which is ngspice's"
        ),
        "vbic": (
            "Xyce's VBIC, its BJT of levels 10 to 12, is not compared with jolty's, "
            "which is ngspice's, yet"
        ),
        "foundry_mosfet": "SKY130's process design kit is written for ngspice",
    },
)
"""Xyce 7.10.0's dialect: its devices' parameters, as its sources name them, and the
devices of jolty's it does not write.

A parameter jolty gives that Xyce's model lacks is refused by name, never dropped, and
so is a device whose Xyce model is another version or another model than jolty's.
"""


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
        ...     first = c.place(devices.resistor(1 * ohm), name="R1")
        ...     second = c.place(devices.resistor(1 * ohm), name="R2")
        ...     c.connect(a, first.a)
        ...     c.connect(first.b, second.a, name="mid")
        ...     c.connect(second.b, b)
        >>> @component
        ... def top(c: Builder) -> None:
        ...     ground = c.port("gnd", Role.GROUND)
        ...     inner = c.place(cell(), name="X1")
        ...     c.connect(inner.a, name="in")
        ...     c.connect(inner.b, ground)
        >>> flat = flatten(circuit(top()))
        >>> [(i.name, i.terminals) for i in flat.components[flat.top].instances]
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
                        f"{prefix}{instance.name}_",
                        nets,
                    )
                case _:
                    placed.append(
                        instance.model_copy(
                            update={
                                "name": f"{prefix}{instance.name}",
                                "terminals": nets,
                            }
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
    "vbic": Element("Q", ("collector", "base", "emitter", "substrate")),
    "mosfet": Element("M", ("drain", "gate", "source", "bulk")),
    "vdmos": Element("M", ("drain", "gate", "source")),
    "bsim3": Element("M", ("drain", "gate", "source", "bulk")),
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
        npn = _defaulted(model, instance.parameters)["polarity"] == "npn"
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
    "base_charge": {"vbic": 0, "gummel_poon": 1},
}
"""SPICE's number for each value of a parameter it numbers: ``TPG``, ``SUBS`` and a
VBIC's ``QBM``."""

_VBIC_SERIES: Final = frozenset(
    {
        "extrinsic_collector_resistance",
        "extrinsic_base_resistance",
        "emitter_resistance",
        "substrate_resistance",
    }
)
"""A VBIC's resistances in series with its terminals, which are none at zero."""


def _card_names(
    model: str, values: dict[str, JsonValue]
) -> tuple[
    str,
    dict[str, tuple[str, float]],
    dict[str, tuple[str, float]],
    dict[str, JsonValue],
]:
    """A device's SPICE type and the names of its parameters.

    Returns the type, the SPICE names of its card's parameters and of its instance's,
    and the values its card writes.

    Raises:
        JoltyError: When the device has no model card, or a VBIC's resistance is too
            small.
    """
    instance: dict[str, tuple[str, float]] = {}
    match model:
        case "diode":
            kind, names = "D", _spice(models.DIODE)
        case "jfet":
            channel = _defaulted(model, values)["channel"]
            kind, names = f"{channel}jf".upper(), _spice(models.JFET)
        case "bipolar_transistor":
            kind = str(_defaulted(model, values)["polarity"]).upper()
            names = _spice(models.BIPOLAR) | {"substrate": ("subs", 1.0)}
        case "mosfet":
            kind = f"{_defaulted(model, values)['channel']}mos".upper()
            names = _spice(models.MOSFET) | {
                "gate_material": ("tpg", 1.0),
                "level": ("level", 1.0),
            }
            instance = _spice(models.MOSFET_INSTANCE)
        case "vdmos":
            kind, names = "VDMOS", _spice(models.VDMOS)
            instance = _spice(models.VDMOS_INSTANCE)
        case "vbic":
            kind = str(_defaulted(model, values)["polarity"]).upper()
            names = _spice(models.VBIC) | {"base_charge": ("qbm", 1.0)}
            values = _vbic_resistances(values)
        case _:
            message = f"a {model} has no model card"
            raise _unwritable(message)
    return kind, names, instance, values


def _model_card(
    model: str, values: dict[str, JsonValue], dialect: Dialect
) -> tuple[str, str, str]:
    """The SPICE type and parameters of a device's model card, and its instance's.

    Raises:
        JoltyError: When the device has no model card, or the dialect's model lacks a
            parameter the device gives.
    """
    if model == "bsim3":
        return _bsim3_card(values)
    kind, names, instance, values = _card_names(model, values)
    names["nominal_temperature"] = ("tnom", 1.0)
    given = {
        key: value
        for key, value in values.items()
        if key not in {"channel", "polarity"}
    }
    _check_card(model, given, names, instance, dialect)
    flicker = given.get("flicker_coefficient")
    if (
        dialect.flicker_over_oxide
        and model == "mosfet"
        and isinstance(flicker, float)
        and flicker > 0
    ):
        # jolty's flicker noise is SPICE's, KF (Id/m)^AF over the gate's area and Cox²;
        # ngspice 43 and later take it with nlev=1, over Cox alone.
        thickness = given.get("oxide_thickness")
        oxide = OXIDE_PERMITTIVITY / (
            thickness if isinstance(thickness, float) else SPICE_OXIDE_THICKNESS
        )
        given["flicker_coefficient"] = flicker / oxide
        names["noise_level"] = ("nlev", 1.0)
        given["noise_level"] = 1.0
    # VBIC computes its thermal voltage with its own constants in both engines, so its
    # temperatures are written as they are.
    celsius = _plain if model == "vbic" else dialect.option_temperature
    card = " ".join(
        f"{names[key][0]}={_spice_value(key, value, names[key][1], celsius)}"
        for key, value in given.items()
        if key not in instance
    )
    # ngspice reads a VDMOS card's polarity from the word `pchan` within it.
    if model == "vdmos" and _defaulted(model, values)["channel"] == "p":
        card = f"pchan {card}".rstrip()
    if model == "vbic":
        card = f"level=4 {card}".rstrip()
    element = " ".join(
        f"{instance[key][0]}={_spice_value(key, value, instance[key][1], celsius)}"
        for key, value in given.items()
        if key in instance
    )
    return kind, card, element


def _check_card(
    model: str,
    given: Mapping[str, JsonValue],
    names: Mapping[str, tuple[str, float]],
    instance: Mapping[str, tuple[str, float]],
    dialect: Dialect,
) -> None:
    """Checks that the dialect's model takes every parameter of its card `given`.

    Raises:
        JoltyError: When it lacks one, named as SPICE names it.
    """
    if dialect.cards is None:
        return
    level = given.get("level", 1)
    card = (model, level if isinstance(level, int) else 1)
    if card not in dialect.cards:
        message = f"{dialect.simulator} has no model of a {model} of level {card[1]}"
        raise _unwritable(message)
    taken = dialect.cards[card]
    lacking = sorted(
        names[key][0]
        for key in given
        if key not in instance and names[key][0] not in taken
    )
    if lacking:
        message = (
            f"{dialect.simulator}'s {model} of level {card[1]} has no "
            + ", ".join(name.upper() for name in lacking)
        )
        raise _unwritable(message)


def _plain(kelvins: float) -> float:
    """A temperature in kelvins in degrees Celsius, as it is.

    It is written so for a model whose thermal voltage is its own in every simulator.
    """
    return kelvins - ZERO_CELSIUS


def _bsim3_card(values: dict[str, JsonValue]) -> tuple[str, str, str]:
    """A BSIM3 transistor's SPICE type, card and instance's parameters.

    The card is of ngspice's level 8 at version 3.3.0, its bins' dependences in SI
    units, ``binunit = 2``; its concentrations are per cubic centimeter, but where that
    number would exceed ngspice's bound, above which it reads one per cubic meter. BSIM3
    computes with constants of its own, Boltzmann's over the electron's charge among
    them, so its nominal temperature is written in the same kelvins as jolty's, given
    or not, rather than where ngspice's thermal voltage is jolty's.
    """
    defaulted = _defaulted("bsim3", values)
    kind = f"{defaulted['channel']}mos".upper()
    element = _spice(models.BSIM3_INSTANCE)
    names = {
        jolty: name
        for name, jolty in (models.BSIM3_SELECTORS | models.BSIM3_SETTINGS).items()
    }
    nominal = _number("bsim3", defaulted, "nominal_temperature")
    card = ["level=8", "version=3.3.0", "binunit=2", f"tnom={nominal - ZERO_CELSIUS!r}"]
    written: list[str] = []
    for key, value in values.items():
        match key, value:
            case "channel", _:
                continue
            case _, int() | float() if key in element:
                spice, factor = element[key]
                written.append(f"{spice}={float(value) * factor!r}")
            case "nominal_temperature", _:
                continue
            case _, bool():
                card.append(f"{names.get(key, key)}={int(value)}")
            case _, int():
                card.append(f"{names.get(key, key)}={value}")
            case _, float():
                card.append(f"{names.get(key, key)}={_bsim3_value(key, value)!r}")
            case _:
                message = f"no SPICE value for {key} = {value}"
                raise _unwritable(message)
    return kind, " ".join(card), " ".join(written)


def _bsim3_value(key: str, value: float) -> float:
    """The number a BSIM3 card writes for its parameter `key`, `value` in SI units."""
    base = key if key in BSIM3_CARD else key[1:]
    if base in models.BSIM3_DENSITIES:
        per_centimeter = value * 1e-6
        return (
            value if per_centimeter > models.BSIM3_DENSITIES[base] else per_centimeter
        )
    if base == "nsub":
        return value * 1e-6
    return value


def _vbic_resistances(values: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """A VBIC's parameters, without the series resistances it has none of.

    Raises:
        JoltyError: When a resistance lies below the 10 mΩ that ngspice raises every
            resistance its card gives to.
    """
    kept: dict[str, JsonValue] = {}
    for key, value in values.items():
        if key in _VBIC_SERIES and value == 0:
            continue
        resistance = key.endswith("_resistance") and isinstance(value, float)
        if resistance and value < models.VBIC_LEAST_RESISTANCE:
            message = (
                f"a VBIC's {key} is {value!r} Ω, below the 0.01 Ω ngspice raises it to"
            )
            raise _unwritable(message)
        kept[key] = value
    return kept


OXIDE_PERMITTIVITY: Final = 3.9 * 8.854214871e-12
"""Silicon dioxide's permittivity, in farads per meter, as jolty and ngspice take it."""

SPICE_OXIDE_THICKNESS: Final = 1e-7
"""The oxide SPICE takes for a MOSFET's flicker noise where its card gives none, 0.1 µm,
as jolty does."""


def _spice_value(
    key: str, value: JsonValue, factor: float, celsius: Callable[[float], float]
) -> str:
    """The value of parameter `key` in SPICE's units.

    A nominal temperature is written in degrees `celsius`: where the simulator's thermal
    voltage is jolty's, or as it is for a model whose thermal voltage is its own.
    """
    match value:
        case str(name):
            return repr(_NUMBERED[key][name])
        case int(number):
            return repr(number)
        case float() if key == "nominal_temperature":
            return repr(celsius(value))
        case float():
            return repr(float(value) * factor)
    message = f"no SPICE value for {value}"
    raise _unwritable(message)


_EXPRESSED: Final = frozenset(
    {
        "behavioral_voltage_source",
        "behavioral_current_source",
        "voltage_controlled_switch",
        "current_controlled_switch",
        "photocell",
        "voltage_controlled_memristor",
        "current_controlled_memristor",
    }
)
"""The devices written as behavioral sources, whose expressions the simulator reads."""


def element(instance: ir.Instance, dialect: Dialect = NGSPICE) -> str:
    """The SPICE name of a device instance: its model's letter, then its id.

    Raises:
        JoltyError: When the instance is no device, or the dialect writes none like it:
            a logic device, a heated one, or one the dialect refuses.
    """
    if not isinstance(instance.of, ir.DeviceOf):
        message = f"{instance.name} is a component instance, not a device"
        raise _unwritable(message)
    device = instance.of.model
    if device not in ELEMENTS:
        message = f"a {device} has no SPICE element"
        raise _unwritable(message)
    if device in dialect.refused:
        message = f"{instance.name} is a {device}: {dialect.refused[device]}"
        raise _unwritable(message)
    if device in _EXPRESSED and dialect.expressions is not None:
        message = f"{instance.name} is a {device}, written as an expression: "
        raise _unwritable(message + dialect.expressions)
    if instance.parameters.get("thermal") is True:
        message = (
            f"{instance.name} is a heated {device}, and {dialect.simulator} heats no "
            "device on a thermal network"
        )
        raise _unwritable(message)
    return ELEMENTS[device].letter + instance.name


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
        "vdmos",
        "vbic",
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


def _own_temperature(instance: ir.Instance, model: str, dialect: Dialect) -> str:
    """The SPICE instance parameter that sets the device's own temperature, if any.

    A temperature it is held at is SPICE's ``TEMP``, and an offset from the
    simulation's its ``DTEMP``, each where the simulator's thermal voltage is jolty's,
    or as it is for a VBIC, whose thermal voltage is its own in every engine.

    Raises:
        JoltyError: When the device's own temperature moves it, and its SPICE element
            has none.
    """
    celsius = _plain if model == "vbic" else dialect.option_temperature

    match instance.parameters:
        case {"temperature": int() | float() as held}:
            written = f" temp={celsius(held)!r}"
        case {"temperature_offset": int() | float() as offset}:
            written = f" dtemp={celsius(offset) - celsius(0.0)!r}"
        case _:
            return ""
    if model in _WARMED:
        return written
    if model in _UNMOVED:
        return ""
    message = (
        f"{instance.name} is a {model} at its own temperature, which "
        f"{dialect.simulator}'s has not"
    )
    raise _unwritable(message)


def netlist(
    circuit: ir.Circuit,
    *,
    excited: str | None = None,
    subcircuits: Collection[str] = (),
    transient: bool = False,
    dialect: Dialect = NGSPICE,
) -> list[str]:
    """The SPICE lines of a circuit; `excited` is the source a small-signal run excites.

    The lines are in the `dialect` of a simulator, ngspice's unless given. They are for
    a `transient` or for the other analyses: a memristor's state moves in a transient's
    lines, and holds where it was made in the others', as jolty's does.

    Every component is flattened into its devices, but those named in `subcircuits`,
    components read from SPICE libraries, which become instances of their
    subcircuits, the libraries included.

    Raises:
        JoltyError: When the circuit holds a device the dialect has no element for, such
            as a logic device or a heated one, or a parameter its model lacks.
    """
    circuit = flatten(circuit, subcircuits)
    top = circuit.components[circuit.top]
    ground = next(port.name for port in top.ports if port.role is Role.GROUND)
    placed = {
        instance.of.model
        for instance in top.instances
        if isinstance(instance.of, ir.DeviceOf)
    }
    if "foundry_mosfet" in placed and placed & {"mosfet", "bsim3"}:
        message = (
            "a level-1, level-3 or BSIM3 MOSFET stands beside a foundry's, whose kit "
            "scales the sizes of every MOSFET"
        )
        raise _unwritable(message)
    lines = [f"* {circuit.top}"]
    for instance in top.instances:
        match instance.of:
            case ir.ComponentOf():
                lines.append(_subcircuit(circuit, instance, ground))
            case ir.DeviceOf(model="coupled_inductors"):
                own = _own_temperature(instance, "coupled_inductors", dialect)
                coupled = _coupled(instance, ground, dialect)
                lines.extend(
                    f"{line}{own}" if line.startswith("L") else line for line in coupled
                )
            case ir.DeviceOf(model=model):
                lines.extend(
                    _device(
                        instance, model, (ground, excited), dialect, transient=transient
                    )
                )
    return lines


def _device(
    instance: ir.Instance,
    model: str,
    nets: tuple[str, str | None],
    dialect: Dialect,
    *,
    transient: bool,
) -> list[str]:
    """The SPICE lines of a device of `model`: its element's, after any it needs.

    `nets` are the ground's and the source a small-signal run excites, if any.

    Raises:
        JoltyError: When the dialect has no element for the device.
    """
    ground, excited = nets
    element(instance, dialect)
    values = _cool(instance.parameters)
    own = _own_temperature(instance, model, dialect)
    nodes = " ".join("0" if net == ground else net for net in _nets(instance, model))
    lines: list[str] = []
    match model:
        case "voltage_source" | "current_source":
            ac = " AC 1" if instance.name == excited else ""
            value = _waveform(values["waveform"], dialect) + ac
        case "current_controlled_current_source" | "current_controlled_voltage_source":
            # ngspice senses a current through a voltage source: a zero-volt one in
            # the control port's place.
            *output, control_positive, control_negative = nodes.split()
            sense = _sense(instance)
            lines.append(f"{sense} {control_positive} {control_negative} DC 0")
            nodes = " ".join(output)
            (number,) = values.values()
            value = f"{sense} {number!r}"
        case (
            "diode"
            | "jfet"
            | "bipolar_transistor"
            | "mosfet"
            | "vdmos"
            | "bsim3"
            | "vbic"
        ):
            kind, card, parameters = _model_card(model, values, dialect)
            lines.append(f".model M{instance.name} {kind}({card})")
            value = f"M{instance.name} {parameters}".rstrip()
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
            nodes, value, moving = _varying(
                instance.name,
                model,
                values,
                nodes.split(),
                transient=transient,
            )
            lines.extend(moving)
        case "resistor" | "capacitor" | "inductor":
            value = _passive(
                model, values, nodes.split(), dialect, (instance.name, lines)
            )
        case "foundry_mosfet":
            try:
                value = sky130.written(values)
            except ValueError as error:
                raise _unwritable(str(error)) from None
        case _:
            (number,) = values.values()
            value = repr(number)
    lines.append(f"{element(instance, dialect)} {nodes} {value}{own}")
    return lines


def _varying(
    id: str,
    model: str,
    values: Mapping[str, JsonValue],
    nodes: list[str],
    *,
    transient: bool,
) -> tuple[str, str, list[str]]:
    """The nodes and the value of a resistance an expression varies, and its lines.

    A photocell's conductance follows its light, and a memristor's its state; the lines
    lag the one and, in a `transient`'s lines, move the other.
    """
    a, b, *light = nodes
    if model == "photocell":
        conductance, lines = _photocell(id, values, light)
    else:
        conductance, lines = _memristor(id, model, values, [a, b], transient=transient)
    return f"{a} {b}", f"I={{V({a},{b})*{conductance}}}", lines


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


STATE_CAPACITANCE: Final = CURRENT_TOLERANCE / VOLTAGE_TOLERANCE
"""The capacitance, in farads, of the node whose voltage is a memristor's state.

It is the ratio of ngspice's absolute tolerances on currents and on voltages, so that
its tolerance on the current that charges the node is its voltage tolerance on the
rate the state moves, as jolty's tolerance on a state's equation is. Over 1 F, the
rates of two states one rounding apart, near a bound where a window steepens the rate,
differed by a thousand times the current tolerance: Newton iteration then converged
only when the state repeated to its last bit, and ngspice, built without fused
multiply-adds, as for x86-64, stalled in steps of 1e-16 s.
"""


def _memristor(
    id: str,
    model: str,
    values: Mapping[str, JsonValue],
    terminals: list[str],
    *,
    transient: bool,
) -> tuple[str, list[str]]:
    """A memristor's conductance as an ngspice expression, and the lines that move it.

    Its state is the voltage of a node of its own. In a `transient`'s lines, that node
    is over `STATE_CAPACITANCE`, which a behavioral current charges at the rate the
    state moves, from an initial condition at the state it was made in; elsewhere, as
    at an operating point and in the analyses about it, a source holds it at that
    state, as jolty's does.

    ngspice differentiates `max(a, b)` as `a - b > 0 ? da : db`, and `min(a, b)` as
    `a - b < 0 ? da : db`, so where the two are equal it takes the second's slope. Each
    `max` and `min` here puts second the side whose slope jolty takes there: the state
    held within its bounds counts its slope at them, the window slows the drive by its
    exponential at the bound itself, and a drive of zero moves toward on. At a bound,
    where the window stops the state, ngspice's Newton iteration then has jolty's
    Jacobian rather than a slope of zero.
    """
    given = _defaulted(model, values)
    made = _number(id, given, "state")
    node = f"_{id}_state"
    state = f"V({node})"
    held = f"min(1,max(0,{state}))"
    on, off = (_number(id, given, key) for key in ("on_resistance", "off_resistance"))
    conductance = (
        f"(1/({on!r}+{off - on!r}*{held}))"
        if given["law"] == "linear"
        else f"(exp({-math.log(off / on)!r}*{held})/{on!r})"
    )
    if not transient:
        return conductance, [f"V{id}_state {node} 0 {made!r}"]
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
    toward_off = f"(1-exp(-max(0,{off_distance}))+min({off_distance},0))"
    toward_on = f"(1-exp(-max(0,{on_distance}))+min({on_distance},0))"
    rate = f"max({drive},0)*{toward_off}+min(0,{drive})*{toward_on}"
    lines = [
        f"B{id}_state 0 {node} I={{{STATE_CAPACITANCE!r}*({rate})}}",
        f"C{id}_state {node} 0 {STATE_CAPACITANCE!r}",
        f".ic {state}={made!r}",
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
        message = f"{instance.name} is a device, not a subcircuit"
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
    return f"X{instance.name} {nodes} {subcircuit}{parameters}"


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
    dialect: Dialect,
    card: tuple[str, list[str]] | None = None,
) -> str:
    """A resistor's, capacitor's or inductor's value and temperature coefficients.

    A value that changes with the voltage is ngspice's expression for the resistance,
    or for the charge, of the voltage across the device. A resistor's flicker noise is
    a model card's, which `card`, the device's id and the lines before its element's,
    takes.

    Raises:
        JoltyError: When a parameter has no counterpart in the dialect's element, or a
            resistor whose value bends with its voltage makes flicker noise: ngspice's
            expression for a resistance makes no noise.
    """
    number, *rest = values.items()
    taken = None if dialect.passives is None else dialect.passives[model]
    lacking = sorted(
        name.upper()
        for key, _ in rest
        for name in [_PASSIVE_DRIFT.get(key) or _FLICKER.get(key)]
        if taken is not None and name is not None and name not in taken
    )
    if lacking:
        message = f"{dialect.simulator}'s {model} has no {', '.join(lacking)}"
        raise _unwritable(message)
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
    if dialect.expressions is not None:
        message = (
            f"a {model} whose value bends with its voltage is written as an "
            f"expression: {dialect.expressions}"
        )
        raise _unwritable(message)
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


def _coupled(instance: ir.Instance, ground: str, dialect: Dialect) -> list[str]:
    """Coupled inductors as SPICE writes them.

    Each winding is an inductor ``L{id}_{k}``, and each coupling a mutual inductance
    ``K{id}_{j}_{k}``.
    """
    element(instance, dialect)
    id, terminals = instance.name, instance.terminals
    values = _cool(instance.parameters)
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
        written = _passive("inductor", winding, nodes, dialect)
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
        _number(instance.name, given, key)
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
        message = f"{instance.name}'s expression is not text"
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
        message = f"{instance.name}'s expression reads {variable}, which ngspice cannot"
        raise _unwritable(message)

    try:
        tree = _TREE.validate_json(_core.expression_tree(expression))
        written = _written(tree, read)
    except ValueError as error:
        message = f"{instance.name}'s expression: {error}"
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
    circuit: ir.Circuit,
    unknown: Unknown,
    subcircuits: Collection[str] = (),
    dialect: Dialect = NGSPICE,
) -> str:
    """The name of a simulator's vector of a jolty unknown: a voltage, current or state.

    A net within a flattened component, such as ``X1/mid``, is the node ``x1_mid``; a
    net within a placed subcircuit, such as ``U1/11``, is ngspice's ``xu1.11`` and
    Xyce's ``xu1:11``, as the `dialect` joins them. A device's state, such as
    ``M1:state``, is the voltage of the node the netlist gives it, ``_m1_state``.

    Raises:
        JoltyError: When the unknown is a device's internal unknown, which the
            simulator names its own way.
    """
    if unknown.nature == "state":
        path, _, label = unknown.name.rpartition(":")
        return f"v(_{_node(path)}_{label})".lower()
    flat = flatten(circuit, subcircuits)
    top = flat.components[flat.top]
    if unknown.nature == "voltage":
        first, *rest = unknown.name.split("/")
        kept = top.instance(first)
        if rest and kept is not None and isinstance(kept.of, ir.ComponentOf):
            return f"v({dialect.separator.join([f'x{first}', *rest]).lower()})"
        return f"v({_node(unknown.name).lower()})"
    path, _, label = unknown.name.rpartition(":")
    id = _node(path)
    found = [instance for instance in top.instances if instance.name == id]
    if not found:
        message = (
            f"{dialect.simulator} names the internal unknown {unknown.name} its own way"
        )
        raise _unwritable(message)
    (instance,) = found
    if instance.of == ir.DeviceOf(model="coupled_inductors"):
        return f"i(l{id.lower()}_{label.removeprefix('current')})"
    source = _sense(instance) if label == "control" else element(instance, dialect)
    return f"i({source.lower()})"


def _waveform(spec: JsonValue, dialect: Dialect) -> str:
    """A source's waveform in the dialect's SPICE.

    Its level at time zero comes first as its DC level where the dialect states it.
    """
    written = _shape(spec, dialect.instant)
    if dialect.zero_level and not written.startswith("DC"):
        return f"DC {_core.level(json.dumps(spec), 0.0)!r} {written}"
    return written


def _shape(spec: JsonValue, instant: float) -> str:
    """A source's waveform in SPICE, a pulse of no width at the width `instant`."""
    match spec:
        case {"dc": float(level)}:
            return f"DC {level!r}"
        case {"pulse": dict(pulse)}:
            order = ("initial", "pulsed", "delay", "rise", "fall", "width", "period")
            numbers = [
                instant if key == "width" and pulse[key] == 0 else pulse[key]
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

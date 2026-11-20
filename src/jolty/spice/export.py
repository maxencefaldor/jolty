# Copyright 2026 Maxence Faldor
"""Export of jolty circuits to SPICE netlists for ngspice and Xyce verification.

The [`netlist`][jolty.spice.export.netlist] function flattens circuit hierarchy into
devices whose names and nets join hierarchical paths with underscores (such as
``RX1_R1`` and ``x1_mid``), while components imported from SPICE libraries remain
subcircuit instances. Output follows the selected simulator dialect, defaulting to
[`NGSPICE`][jolty.spice.export.NGSPICE] unless [`XYCE`][jolty.spice.export.XYCE] is
requested. Simulation runners in [`jolty.spice.ngspice`][jolty.spice.ngspice] and
[`jolty.spice.xyce`][jolty.spice.xyce] execute these netlists, and
[`vector`][jolty.spice.export.vector] maps jolty unknowns to simulator vector names.
Logic devices and thermal networks have no SPICE equivalent, and neither simulator
couples dissipated power back into a thermal network, so circuits containing them
cannot be exported.
"""

import json
import math
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from pydantic import JsonValue, TypeAdapter

from jolty import _core, ir
from jolty.errors import JoltyError
from jolty.ir import Role
from jolty.spice import models, ngspice, sky130, xyce
from jolty.spice.library import TRANSMISSION_LINE
from jolty.spice.ngspice import CURRENT_TOLERANCE, VOLTAGE_TOLERANCE
from jolty.units import ZERO_CELSIUS
from jolty.unknowns import Unknown


def _unwritable(message: str) -> JoltyError:
    return JoltyError(f"cannot write the circuit as SPICE: {message}")


type Card = tuple[str, int]
"""Model card key pairing a device model name with its SPICE level."""


@dataclass(frozen=True)
class Dialect:
    """Simulator-specific SPICE dialect rules and unsupported-feature table.

    The canonical parameter mappings in `jolty.spice.models` target ngspice, so a
    dialect records only deviations from ngspice conventions.
    """

    simulator: str
    """Display name of the target simulator used in diagnostics."""
    option_temperature: Callable[[float], float]
    """Callback converting kelvin temperature to the simulator Celsius option matching
    the jolty thermal voltage."""
    separator: str
    """Delimiter joining a subcircuit instance name and an internal node name."""
    instant: float
    """Small time in seconds substituted where zero pulse width or exponential delay
    has special meaning in the simulator."""
    zero_dc_value: bool
    """Flag indicating whether time-varying sources emit an explicit time-zero DC value
    because the simulator otherwise defaults DC operating points to zero."""
    flicker_over_oxide: bool
    """Flag indicating whether MOSFET flicker noise divides ``KF`` by oxide capacitance
    with ngspice ``nlev=1`` rather than using classic SPICE scaling."""
    cards: Mapping[Card, frozenset[str]] | None
    """Supported SPICE model-card parameters per `(model, level)`, or `None` when all
    parameters exported by jolty are accepted."""
    passives: Mapping[str, frozenset[str]] | None
    """Supported SPICE passive element parameters per device kind, or `None` when all
    parameters exported by jolty are accepted."""
    expressions: str | None
    """Reason the simulator rejects jolty behavioral expressions, or `None` when
    behavioral sources, switches, photocells, memristors, and nonlinear passives are
    supported."""
    refused: Mapping[str, str]
    """Mapping from unsupported device model names to refusal reasons."""


NGSPICE: Final = Dialect(
    simulator="ngspice",
    option_temperature=ngspice.option_temperature,
    separator=".",
    # ngspice interprets a pulse width of 0 as the transient stop time and zero
    # exponential delays as the time step (measured on ngspice 47). A duration of
    # 1e-40 s is negligible relative to any rise or time constant above 1e-24 s.
    instant=1e-40,
    zero_dc_value=False,
    flicker_over_oxide=True,
    cards=None,
    passives=None,
    expressions=None,
    refused={},
)
"""Dialect configuration for ngspice 47, matching `jolty.spice.models`."""

_XYCE_DIODE: Final = frozenset(
    {"is", "n", "rs", "cjo", "vj", "m", "fc", "tt", "bv", "ibv", "nbv", "eg", "xti"}
    | {"kf", "af", "tnom"}
)
"""Level-1 diode model parameters supported by Xyce 7.10.0.

Derived from ``addPar`` in ``src/DeviceModelPKG/OpenModels/N_DEV_Diode.C`` (lines
100 to 366). Xyce omits ``TCV``, using relative ``TBV1`` instead of volts per kelvin.
"""

_XYCE_BIPOLAR: Final = frozenset(
    {"is", "bf", "br", "nf", "nr", "vaf", "var", "ikf", "ikr", "ise", "ne", "isc"}
    | {"nc", "rb", "rbm", "irb", "re", "rc", "cje", "vje", "mje", "tf", "xtf", "vtf"}
    | {"itf", "cjc", "vjc", "mjc", "xcjc", "tr", "cjs", "vjs", "mjs", "fc", "eg"}
    | {"xti", "xtb", "kf", "af", "tnom"}
)
"""Level-1 Gummel-Poon BJT parameters supported by Xyce 7.10.0.

Derived from ``N_DEV_BJT.C`` (lines 120 to 756). Xyce omits substrate junction
currents (``ISS``, ``NS``) and ``SUBS``, fixing the substrate capacitance on the
collector. Excess phase ``PTF`` also diverges from ngspice under both
``NEWEXCESSPHASE`` modes (lines 1187 to 1436): at 60 degrees, ``delayed_stage`` in
`test_ngspice.py` shifts by 16% of collector swing, so ``PTF`` is rejected.
"""

_XYCE_JFET: Final = frozenset(
    {"vto", "beta", "lambda", "rd", "rs", "cgs", "cgd", "pb", "is", "fc", "b", "kf"}
    | {"af", "tnom"}
)
"""Level-1 JFET parameters supported by Xyce 7.10.0.

Derived from ``N_DEV_JFET.C`` (lines 81 to 158). Xyce omits the temperature
coefficients ``VTOTC``, ``BEX``, ``EG``, and ``XTI`` as well as noise parameters.
"""

_XYCE_MOSFET: Final = frozenset(
    {"vto", "kp", "gamma", "phi", "lambda", "rd", "rs", "cbd", "cbs", "is", "pb"}
    | {"cgso", "cgdo", "cgbo", "rsh", "cj", "mj", "cjsw", "mjsw", "js", "tox", "ld"}
    | {"uo", "fc", "nsub", "nss", "tnom", "kf", "af", "tpg", "level"}
)
"""Level-1 MOSFET parameters supported by Xyce 7.10.0.

Derived from ``N_DEV_MOSFET1.C`` (lines 156 to 331). Xyce omits ``DELVTO``, ``WD``,
``XL``, and ``XW``.
"""

_XYCE_LEVEL_THREE: Final = _XYCE_MOSFET | frozenset(
    {"eta", "delta", "nfs", "theta", "vmax", "kappa", "xj"}
)
"""Level-3 MOSFET parameters supported by Xyce 7.10.0.

Derived from ``N_DEV_MOSFET3.C`` (lines 165 to 370).
"""

XYCE: Final = Dialect(
    simulator="Xyce",
    option_temperature=xyce.option_temperature,
    separator=":",
    # Xyce reads a width of 0 as no width: only a width it is not given is its stop
    # time (`N_DEV_SourceData.C`, line 824).
    instant=0.0,
    # Xyce's operating point and DC sweeps take a source's DC value, 0 where none is
    # given, where its transient starts from its waveform at time zero.
    zero_dc_value=True,
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
        "bsim4": (
            "Xyce's BSIM4, its MOSFET of level 54, is of other versions than jolty's, "
            "which is 4.5"
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
        "transmission_line": (
            "Xyce's lossless line, its T, is not compared with jolty's, which is "
            "ngspice's, yet"
        ),
    },
)
"""Dialect configuration for Xyce 7.10.0 and its supported device parameters.

Parameters absent from Xyce models are rejected by name rather than silently dropped.
Devices whose Xyce model version or formulation differs from jolty are likewise
rejected.
"""


def _node(path: str) -> str:
    """Converts a hierarchical `path` into an underscore-joined SPICE name."""
    return path.replace("/", "_")


def flatten(circuit: ir.Circuit, subcircuits: Collection[str] = ()) -> ir.Circuit:
    """Flattens component hierarchy in `circuit`, keeping `subcircuits` unexpanded.

    Instances inside expanded components are renamed by joining their hierarchical path
    with underscores. Internal nets are similarly prefixed, while component port nets
    map directly to their connected outer nets.

    Args:
        circuit: Circuit IR whose component hierarchy is flattened.
        subcircuits: Component names preserved as unexpanded subcircuit instances.

    Returns:
        Flattened circuit IR containing the expanded top-level component and any
        preserved subcircuits.

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
    if circuit.top not in circuit.components:
        message = f"no component `{circuit.top}`"
        raise _unwritable(message)
    top = circuit.components[circuit.top]
    placed: list[ir.Instance] = []
    active: list[str] = [circuit.top]
    active_set: set[str] = {circuit.top}
    stack: list[tuple[str, Iterator[ir.Instance], str, dict[str, str]]] = [
        (
            circuit.top,
            iter(top.instances),
            "",
            {port.name: port.name for port in top.ports},
        )
    ]
    while stack:
        _, iterator, prefix, ports = stack[-1]
        instance = next(iterator, None)
        if instance is None:
            exited = stack.pop()[0]
            active.pop()
            active_set.remove(exited)
            continue
        nets = {
            terminal: ports.get(net, f"{prefix}{net}")
            for terminal, net in instance.terminals.items()
        }
        match instance.of:
            case ir.ComponentOf(component=name) if name not in subcircuits:
                if name in active_set:
                    cycle = (*active[active.index(name) :], name)
                    message = (
                        f"component `{cycle[0]}` contains itself: {' -> '.join(cycle)}"
                    )
                    raise _unwritable(message)
                inner = circuit.components.get(name)
                if inner is None:
                    message = f"no component `{name}`"
                    raise _unwritable(message)
                active.append(name)
                active_set.add(name)
                stack.append(
                    (
                        name,
                        iter(inner.instances),
                        f"{prefix}{instance.name}_",
                        nets,
                    )
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
    kept: dict[str, ir.Component] = {}
    for instance in placed:
        if isinstance(instance.of, ir.ComponentOf):
            name = instance.of.component
            if name not in circuit.components:
                message = f"no component `{name}`"
                raise _unwritable(message)
            kept[name] = circuit.components[name]
    flat = top.model_copy(update={"instances": tuple(placed)})
    return circuit.model_copy(update={"components": {**kept, circuit.top: flat}})


@dataclass(frozen=True)
class Element:
    """SPICE element prefix letter and terminal ordering for a jolty device model."""

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
    "bsim4": Element("M", ("drain", "gate", "source", "bulk")),
    "foundry_mosfet": Element("X", ("drain", "gate", "source", "bulk")),
    "behavioral_voltage_source": Element("B", ("positive", "negative")),
    "behavioral_current_source": Element("B", ("positive", "negative")),
    "behavioral_capacitor": Element("C", ("positive", "negative")),
    "behavioral_resistor": Element("R", ("positive", "negative")),
    "voltage_controlled_switch": Element("B", _CONTROLLED),
    "current_controlled_switch": Element("B", _CONTROLLED),
    "photocell": Element("B", ("a", "b", "light_positive", "light_negative")),
    "voltage_controlled_memristor": Element("B", ("positive", "negative")),
    "current_controlled_memristor": Element("B", ("positive", "negative")),
    "transmission_line": Element(
        "T", ("positive1", "negative1", "positive2", "negative2")
    ),
}


def _nets(instance: ir.Instance, model: str) -> list[str]:
    """Returns terminal net names for `instance` ordered for its SPICE `model`.

    ngspice connects an omitted bipolar substrate terminal to ground, where ``gmin``
    conducts even without a substrate diode. Attaching an omitted substrate to the
    junction reference terminal (collector for vertical NPN unless ``SUBS`` differs, or
    base for lateral PNP) eliminates stray substrate current to match jolty.
    """
    terminals = instance.terminals
    nets = [terminals[name] for name in ELEMENTS[model].terminals if name in terminals]
    if model == "bipolar_transistor" and "substrate" not in terminals:
        npn = _defaulted(model, instance.parameters)["polarity"] == "npn"
        nets.append(terminals["collector" if npn else "base"])
    return nets


def _spice(table: models.Table) -> dict[str, tuple[str, float]]:
    """Inverts a SPICE parameter `table` to map jolty names to `(spice_name, scale)`."""
    names: dict[str, tuple[str, float]] = {}
    for spice, (jolty, factor) in table.items():
        names.setdefault(jolty, (spice, 1 / factor))
    return names


_SPICE_CARDS: Final[Mapping[str, Mapping[str, tuple[str, float]]]] = MappingProxyType(
    {
        "diode": MappingProxyType(_spice(models.DIODE)),
        "jfet": MappingProxyType(_spice(models.JFET)),
        "bipolar_transistor": MappingProxyType(
            _spice(models.BIPOLAR)
            | {"substrate": ("subs", 1.0)}
            | {
                jolty: (spice, 1.0)
                for spice, jolty in models.BIPOLAR_TEMPERATURE_MODELS.items()
            }
        ),
        "mosfet": MappingProxyType(
            _spice(models.MOSFET)
            | {"gate_material": ("tpg", 1.0), "level": ("level", 1.0)}
        ),
        "vdmos": MappingProxyType(_spice(models.VDMOS)),
        "vbic": MappingProxyType(_spice(models.VBIC) | {"base_charge": ("qbm", 1.0)}),
    }
)
_SPICE_INSTANCES: Final[Mapping[str, Mapping[str, tuple[str, float]]]] = (
    MappingProxyType(
        {
            "mosfet": MappingProxyType(_spice(models.MOSFET_INSTANCE)),
            "vdmos": MappingProxyType(_spice(models.VDMOS_INSTANCE)),
            **{
                name: MappingProxyType(_spice(bsim.instance))
                for name, bsim in models.BSIMS.items()
            },
        }
    )
)
_SPICE_BSIM_NAMES: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        name: MappingProxyType(
            {
                jolty: spice_name
                for choices in (bsim.selectors, bsim.settings)
                for spice_name, jolty in choices.items()
            }
        )
        for name, bsim in models.BSIMS.items()
    }
)
_EMPTY_SPICE_TABLE: Final[Mapping[str, tuple[str, float]]] = MappingProxyType({})


_NUMBERED = {
    "gate_material": {material: tpg for tpg, material in models.GATE_TYPES.items()},
    "substrate": {kind: subs for subs, kind in models.CONSTRUCTIONS.items()},
    "base_charge": {"vbic": 0, "gummel_poon": 1},
}
"""Numeric SPICE encodings for enumerated model parameters (``TPG``, ``SUBS``,
and VBIC ``QBM``)."""

_VBIC_SERIES: Final = frozenset(
    {
        "extrinsic_collector_resistance",
        "extrinsic_base_resistance",
        "emitter_resistance",
        "substrate_resistance",
    }
)
"""VBIC terminal series resistance keys that are omitted when set to zero."""


def _card_names(
    model: str, values: dict[str, JsonValue]
) -> tuple[
    str,
    dict[str, tuple[str, float]],
    Mapping[str, tuple[str, float]],
    dict[str, JsonValue],
]:
    """Resolves the SPICE card type, parameter tables, and prepared values for `model`.

    Returns the SPICE card keyword, card parameter table, instance parameter table, and
    normalized parameter dictionary.

    Raises:
        JoltyError: When `model` has no SPICE model card or a VBIC resistance falls
            below the ngspice minimum.
    """
    if model not in _SPICE_CARDS:
        message = f"a {model} has no model card"
        raise _unwritable(message)
    names = dict(_SPICE_CARDS[model])
    instance = _SPICE_INSTANCES.get(model, _EMPTY_SPICE_TABLE)
    match model:
        case "diode":
            kind = "D"
            values, instance = _diode_sidewall(_diode_capacitances(values))
        case "jfet":
            kind = f"{_defaulted(model, values)['channel']}jf".upper()
        case "bipolar_transistor":
            kind = str(_defaulted(model, values)["polarity"]).upper()
        case "mosfet":
            kind = f"{_defaulted(model, values)['channel']}mos".upper()
        case "vdmos":
            kind = "VDMOS"
        case _:
            kind = str(_defaulted(model, values)["polarity"]).upper()
            values = _vbic_resistances(values)
    return kind, names, instance, values


_LINEAR_CAPACITANCES: Final = (
    (
        "junction_potential",
        "junction_potential_temperature_coefficient",
        "junction_capacitance",
        "junction_capacitance_temperature_coefficient",
    ),
    (
        "sidewall_junction_potential",
        "sidewall_junction_potential_temperature_coefficient",
        "sidewall_junction_capacitance",
        "sidewall_junction_capacitance_temperature_coefficient",
    ),
)
"""Diode bottom-wall and sidewall potential/capacitance parameter tuples adjusted
linearly under capacitance temperature model 1."""


def _diode_capacitances(values: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Shifts diode linear capacitance parameters from `Tn` to the 27 °C reference.

    Under capacitance temperature model 1, ngspice evaluates linear potential and
    capacitance shifts relative to 27 °C (``diotemp.c``, lines 93 to 95 and 109 to
    111), whereas jolty shifts from the nominal temperature `Tn`. Rewriting
    `φ - TPB · (T - Tn)` to `φ + TPB · (Tn - 27 °C)` and scaling `C` by
    `k = 1 + CTA · (27 °C - Tn)` with `CTA / k` aligns both formulations.
    """
    defaulted = _defaulted("diode", values)
    nominal = _number("diode", defaulted, "nominal_temperature")
    shift = nominal - _core.REFERENCE_TEMPERATURE
    if defaulted["capacitance_temperature_model"] != 1 or shift == 0:
        return values
    moved = dict(values)
    for potential, falling, capacitance, coefficient in _LINEAR_CAPACITANCES:
        fall = _number("diode", defaulted, falling)
        slope = _number("diode", defaulted, coefficient)
        if fall != 0:
            moved[potential] = _number("diode", defaulted, potential) + fall * shift
        if slope != 0:
            factor = 1 - slope * shift
            if factor == 0 or not math.isfinite(factor):
                message = (
                    f"diode `{coefficient}` shift factor is singular at nominal "
                    f"temperature {nominal!r} K"
                )
                raise _unwritable(message)
            scaled_slope = slope / factor
            if not math.isfinite(scaled_slope):
                message = (
                    f"diode `{coefficient}` overflows at nominal "
                    f"temperature {nominal!r} K"
                )
                raise _unwritable(message)
            moved[capacitance] = _number("diode", defaulted, capacitance) * factor
            moved[coefficient] = scaled_slope
    return moved


_SIDEWALL: Final = frozenset(
    jolty for jolty, _ in models.DIODE.values() if jolty.startswith("sidewall")
)
"""Diode model parameter keys belonging to the sidewall junction."""


def _diode_sidewall(
    values: dict[str, JsonValue],
) -> tuple[dict[str, JsonValue], dict[str, tuple[str, float]]]:
    """Prepares diode card and instance parameters when sidewall parameters are given.

    Sidewall diodes emit unit perimeter ``PJ=1`` because jolty sidewall parameters are
    already scaled for the device perimeter. Emitting ``NS`` explicitly prevents
    ngspice from defaulting sidewall temperature scaling to unity (``diotemp.c``, lines
    123 to 129) and avoids the uninitialized sidewall breakdown voltage
    (``dioload.c``, line 452).
    """
    if not _SIDEWALL & values.keys():
        return values, {}
    emission = _defaulted("diode", values)["emission_coefficient"]
    written = {"sidewall_emission_coefficient": emission, **values, "perimeter": 1.0}
    return written, {"perimeter": ("pj", 1.0)}


def _model_card(
    model: str, values: dict[str, JsonValue], dialect: Dialect
) -> tuple[str, str, str]:
    """Formats the SPICE model card type, card parameters, and instance parameters.

    Raises:
        JoltyError: When `model` has no model card or `dialect` rejects a supplied
            parameter.
    """
    if model in models.BSIMS:
        return _bsim_card(model, values)
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
    # VBIC computes thermal voltage with fixed internal constants in both engines, so
    # VBIC temperatures are written without thermal-voltage adjustment.
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
    """Checks that `dialect` supports every model-card parameter in `given`.

    Raises:
        JoltyError: When `dialect` lacks the model level or any supplied SPICE
            parameter.
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
    """Converts a temperature in kelvins directly to degrees Celsius.

    Used for device models whose internal thermal-voltage constants are fixed across
    simulators.
    """
    return kelvins - ZERO_CELSIUS


def _bsim_card(model: str, values: dict[str, JsonValue]) -> tuple[str, str, str]:
    """Formats the SPICE type, model card, and instance parameters for BSIM3 or BSIM4.

    Emits ngspice level 8 (version 3.3.0) or level 54 (version 4.5) with SI binning
    units (``binunit=2``). Doping concentrations are exported per cubic centimeter
    unless the scaled value exceeds the ngspice threshold for cubic-meter units.
    Nominal temperature ``tnom`` is written directly in degrees Celsius because BSIM
    uses fixed internal physical constants rather than the simulator thermal voltage.
    """
    bsim = models.BSIMS[model]
    defaulted = _defaulted(model, values)
    kind = f"{defaulted['channel']}mos".upper()
    element = _SPICE_INSTANCES[model]
    names = _SPICE_BSIM_NAMES[model]
    nominal = _number(model, defaulted, "nominal_temperature")
    card = [*bsim.written, "binunit=2", f"tnom={nominal - ZERO_CELSIUS!r}"]
    written: list[str] = []
    for key, value in values.items():
        match key, value:
            case "channel", _:
                continue
            case _, int() | float() if not isinstance(value, bool) and key in element:
                spice, factor = element[key]
                written.append(f"{spice}={float(value) * factor!r}")
            case "nominal_temperature", _:
                continue
            case _, bool():
                card.append(f"{names.get(key, key)}={int(value)}")
            case _, int():
                card.append(f"{names.get(key, key)}={value}")
            case _, float():
                card.append(f"{names.get(key, key)}={_bsim_value(bsim, key, value)!r}")
            case _:
                message = f"no SPICE value for {key} = {value}"
                raise _unwritable(message)
    return kind, " ".join(card), " ".join(written)


def _bsim_value(bsim: models.Bsim, key: str, value: float) -> float:
    """Converts a BSIM card parameter `value` for `key` from SI into SPICE units."""
    base = key if key in bsim.card else key[1:]
    if base in bsim.densities:
        per_centimeter = value * 1e-6
        return value if per_centimeter > bsim.densities[base] else per_centimeter
    if base == "nsub":
        return value * 1e-6
    return value


def _vbic_resistances(values: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Filters zero VBIC terminal resistances and checks ngspice minimum thresholds.

    Raises:
        JoltyError: When a nonzero resistance falls below the 10 mΩ minimum enforced by
            ngspice.
    """
    kept: dict[str, JsonValue] = {}
    for key, value in values.items():
        if key in _VBIC_SERIES and value == 0:
            continue
        resistance = key.endswith("_resistance") and isinstance(value, float)
        if resistance and value < models.VBIC_MIN_RESISTANCE:
            message = (
                f"a VBIC's {key} is {value!r} Ω, below the 0.01 Ω ngspice raises it to"
            )
            raise _unwritable(message)
        kept[key] = value
    return kept


OXIDE_PERMITTIVITY: Final = 3.9 * 8.854214871e-12
"""Permittivity of silicon dioxide in farads per meter used by jolty and ngspice."""

SPICE_OXIDE_THICKNESS: Final = 1e-7
"""Default MOSFET oxide thickness (0.1 µm) assumed by SPICE and jolty for flicker
noise when ``TOX`` is omitted."""


def _spice_value(
    key: str, value: JsonValue, factor: float, celsius: Callable[[float], float]
) -> str:
    """Formats parameter `value` for `key` in SPICE units.

    Nominal temperatures are converted via `celsius` to match the target simulator or
    fixed model thermal voltage.
    """
    match value:
        case bool():
            pass
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
        "behavioral_capacitor",
        "behavioral_resistor",
        "voltage_controlled_switch",
        "current_controlled_switch",
        "photocell",
        "voltage_controlled_memristor",
        "current_controlled_memristor",
    }
)
"""Device models exported as behavioral SPICE expressions."""


def element(instance: ir.Instance, dialect: Dialect = NGSPICE) -> str:
    """Returns the SPICE element name formed from the model prefix letter and name.

    Args:
        instance: Device instance in a flattened component.
        dialect: Target simulator dialect checking model and thermal support.

    Returns:
        SPICE element name prefixed with the device model letter.

    Raises:
        JoltyError: When `instance` is not a device or `dialect` does not support the
            device model or thermal coupling.
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
"""Instance environment parameters (`thermal`, `temperature`, `temperature_offset`)
emitted outside SPICE values and model cards."""


def _cool(parameters: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Returns device `parameters` with environment keys in `_SURROUNDINGS` removed."""
    return {key: value for key, value in parameters.items() if key not in _SURROUNDINGS}


_WARMED: Final = frozenset(
    {
        "resistor",
        "behavioral_resistor",
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
"""Device models whose SPICE elements accept instance temperature overrides."""

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
        "behavioral_capacitor",
    }
)
"""Device models whose behavior does not depend on instance temperature."""


def _own_temperature(instance: ir.Instance, model: str, dialect: Dialect) -> str:
    """Returns the SPICE instance temperature clause (``temp`` or ``dtemp``), if set.

    Absolute temperatures map to ``temp`` and offsets map to ``dtemp`` via the dialect
    thermal-voltage conversion, or directly in degrees Celsius for VBIC.

    Raises:
        JoltyError: When a temperature-sensitive device specifies an instance
            temperature not supported by its SPICE element.
    """
    celsius = _plain if model == "vbic" else dialect.option_temperature

    match instance.parameters:
        case {"temperature": int() | float() as held} if not isinstance(held, bool):
            written = f" temp={celsius(held)!r}"
        case {"temperature_offset": int() | float() as offset} if not isinstance(
            offset, bool
        ):
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
    """Exports `circuit` as a list of SPICE netlist lines for `dialect`.

    Args:
        circuit: Circuit IR to export.
        excited: Optional source instance name given an ``AC 1`` stimulus.
        subcircuits: Component names preserved as external SPICE subcircuit instances
            rather than flattened.
        transient: When `True`, emits dynamic memristor state integration rather than
            holding initial state fixed for DC and small-signal analyses.
        dialect: Target simulator dialect, defaulting to `NGSPICE`.

    Returns:
        SPICE netlist lines representing `circuit` in `dialect`.

    Raises:
        JoltyError: When `circuit` contains an unsupported device, thermal coupling,
            mixed foundry and generic MOSFETs, or unsupported model parameters.
    """
    circuit = flatten(circuit, subcircuits)
    top = circuit.components[circuit.top]
    ground = next(
        (port.name for port in top.ports if port.role is Role.GROUND),
        None,
    )
    if ground is None:
        message = f"top component `{circuit.top}` has no ground port"
        raise _unwritable(message)
    placed = {
        instance.of.model
        for instance in top.instances
        if isinstance(instance.of, ir.DeviceOf)
    }
    if "foundry_mosfet" in placed and placed & {"mosfet", "bsim3", "bsim4"}:
        message = (
            "a level-1, level-3, BSIM3 or BSIM4 MOSFET stands beside a foundry's, "
            "whose kit scales the sizes of every MOSFET"
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
    """Exports the SPICE lines for one `instance` of `model` and any helper elements.

    Unpacks `nets` as `(ground, excited)` for ground mapping and AC excitation.

    Raises:
        JoltyError: When `dialect` does not support `model` or its parameters.
    """
    ground, excited = nets
    element_name = element(instance, dialect)
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
            | "bsim4"
            | "vbic"
        ):
            kind, card, parameters = _model_card(model, values, dialect)
            lines.append(f".model M{instance.name} {kind}({card})")
            value = f"M{instance.name} {parameters}".rstrip()
        case (
            "behavioral_voltage_source"
            | "behavioral_current_source"
            | "behavioral_capacitor"
            | "behavioral_resistor"
        ):
            value = _behavioral_value(instance, model, values, ground, lines)
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
            value = _value(model, values)
    lines.append(f"{element_name} {nodes} {value}{own}")
    return lines


_BEHAVIORAL_OUTPUTS: Final = {
    "behavioral_voltage_source": "V",
    "behavioral_current_source": "I",
    "behavioral_capacitor": "Q",
    "behavioral_resistor": "R",
}
"""SPICE behavioral expression output keywords (`V`, `I`, `Q`, or `R`) by model."""


def _behavioral_value(
    instance: ir.Instance,
    model: str,
    values: Mapping[str, JsonValue],
    ground: str,
    lines: list[str],
) -> str:
    """Formats a behavioral source, capacitor, or resistor SPICE value clause."""
    expression, senses = _behavior(instance, ground)
    lines.extend(senses)
    # ngspice makes a capacitor of a charge `Q=` a current of its rate of change
    # (`inpcom.c`, lines 6937 to 6950).
    output = _BEHAVIORAL_OUTPUTS[model]
    written = f"{output}={{{expression}}}"
    if model != "behavioral_resistor":
        return written
    rest = [
        (key, amount)
        for key, amount in values.items()
        if key not in {"expression", "parameters"}
    ]
    drift = _passive_drift(model, rest, "")
    return " ".join([written, *drift])


_KEYWORDED: Final = {"transmission_line": TRANSMISSION_LINE}
"""Device models whose SPICE values are formatted as `KEY=value` arguments."""


def _value(model: str, values: Mapping[str, JsonValue]) -> str:
    """Formats the primary SPICE element value for `model` as a number or keywords."""
    keywords = _KEYWORDED.get(model)
    if keywords is None:
        (number,) = values.values()
        return repr(number)
    names: dict[str, str] = {}
    for spice, name in keywords.items():
        names.setdefault(name, spice.upper())
    return " ".join(f"{spice}={values[name]!r}" for name, spice in names.items())


def _varying(
    name: str,
    model: str,
    values: Mapping[str, JsonValue],
    nodes: list[str],
    *,
    transient: bool,
) -> tuple[str, str, list[str]]:
    """Exports a variable-conductance element (`photocell` or memristor) as a B-source.

    Returns the terminal node pair, behavioral current expression, and auxiliary state
    or RC lag lines.
    """
    a, b, *light = nodes
    if model == "photocell":
        conductance, lines = _photocell(name, values, light)
    else:
        conductance, lines = _memristor(
            name, model, values, [a, b], transient=transient
        )
    return f"{a} {b}", f"I={{V({a},{b})*{conductance}}}", lines


_DARKNESS: Final = 1e-6
"""Illuminance floor in lux smoothing a photocell response near zero light."""


def _photocell(
    name: str, values: Mapping[str, JsonValue], light: list[str]
) -> tuple[str, list[str]]:
    """Builds the ngspice conductance expression and lag-filter lines for a photocell.

    When `response_time` is nonzero, an auxiliary 1 Ω and `response_time`-farad RC
    network filters the optical input across `light`.
    """
    given = _defaulted("photocell", values)
    illuminance = f"V({light[0]},{light[1]})"
    lit = f"(({illuminance})+sqrt(({illuminance})^2+{_DARKNESS!r}^2))/2"
    settled = f"pow({lit}/{given['illuminance']!r},{given['gamma']!r})"
    response, lines = settled, list[str]()
    if given["response_time"] != 0:
        node = f"_{name}_response"
        lines = [
            f"B{name}_settled _{name}_settled 0 V={{{settled}}}",
            f"R{name}_lag _{name}_settled {node} 1",
            f"C{name}_lag {node} 0 {given['response_time']!r}",
        ]
        response = f"V({node})"
    conductance = f"(1/{given['dark_resistance']!r}+{response}/{given['resistance']!r})"
    return conductance, lines


STATE_CAPACITANCE: Final = CURRENT_TOLERANCE.value / VOLTAGE_TOLERANCE.value
"""Capacitance in farads of the auxiliary node holding a memristor state voltage.

Equals the ratio of ngspice current tolerance to voltage tolerance so the current
tolerance on the charging source matches the voltage tolerance on state rate. With
1 F capacitance, adjacent floating-point states near a steep window boundary
differed by 1000x the current tolerance, causing ngspice without fused multiply-add
on x86-64 to stall at 1e-16 s steps.
"""


def _memristor(
    name: str,
    model: str,
    values: Mapping[str, JsonValue],
    terminals: list[str],
    *,
    transient: bool,
) -> tuple[str, list[str]]:
    """Builds the ngspice conductance expression and state lines for a memristor.

    Memristor state is represented as the voltage of a dedicated auxiliary node. In
    `transient` mode, a behavioral current charges `STATE_CAPACITANCE` from the initial
    condition; in DC and small-signal analyses, a voltage source holds the initial
    state fixed.

    ngspice differentiates ``max(a, b)`` as ``a - b > 0 ? da : db`` and ``min(a, b)``
    as ``a - b < 0 ? da : db``, selecting the second argument's derivative at equality.
    Each ``max`` and ``min`` orders the active branch second so that boundary
    derivatives match the jolty Jacobian instead of vanishing during Newton iteration.
    """
    given = _defaulted(model, values)
    made = _number(name, given, "state")
    node = f"_{name}_state"
    state = f"V({node})"
    held = f"min(1,max(0,{state}))"
    on, off = (_number(name, given, key) for key in ("on_resistance", "off_resistance"))
    conductance = (
        f"(1/({on!r}+{off - on!r}*{held}))"
        if given["law"] == "linear"
        else f"(exp({-math.log(off / on)!r}*{held})/{on!r})"
    )
    if not transient:
        return conductance, [f"V{name}_state {node} 0 {made!r}"]
    voltage = f"V({terminals[0]},{terminals[1]})"
    if model == "voltage_controlled_memristor":
        drive = (
            f"{given['off_rate']!r}"
            f"*pow(max({voltage}/{given['off_voltage']!r}-1,0),"
            f"{given['off_exponent']!r})"
            f"-{given['on_rate']!r}"
            f"*pow(max({voltage}/({given['on_voltage']!r})-1,0),"
            f"{given['on_exponent']!r})"
        )
    else:
        drive = f"{voltage}*{conductance}/{given['charge']!r}"
    window = given["window"]
    off_distance, on_distance = f"(1-{state})/{window!r}", f"{state}/{window!r}"
    toward_off = f"(1-exp(-max(0,{off_distance}))+min({off_distance},0))"
    toward_on = f"(1-exp(-max(0,{on_distance}))+min({on_distance},0))"
    rate = f"max({drive},0)*{toward_off}+min(0,{drive})*{toward_on}"
    lines = [
        f"B{name}_state 0 {node} I={{{STATE_CAPACITANCE!r}*({rate})}}",
        f"C{name}_state {node} 0 {STATE_CAPACITANCE!r}",
        f".ic {state}={made!r}",
    ]
    return conductance, lines


_PARAMETERS: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


def _defaulted(model: str, values: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Returns `values` for `model` with omitted entries filled by engine defaults."""
    return _PARAMETERS.validate_json(_core.defaulted(model, json.dumps(values)))


def _number(name: str, values: Mapping[str, JsonValue], key: str) -> float:
    """Extracts numeric parameter `key` from `values` for device `name`.

    Raises:
        JoltyError: When the parameter value is missing or non-numeric.
    """
    match values.get(key):
        case bool():
            pass
        case int() | float() as found:
            return float(found)
    message = f"{name}'s {key} is no number"
    raise _unwritable(message)


def _subcircuit(circuit: ir.Circuit, instance: ir.Instance, ground: str) -> str:
    """Formats a SPICE subcircuit instantiation line (`X...`) with ordered pins.

    Maps the subcircuit ground port to SPICE node ``0`` and translates bracketed
    component parameters such as ``divider[top=1000.0]`` into a ``params:`` clause.
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


_PASSIVE_PRIMARY: Final = {
    "resistor": "resistance",
    "capacitor": "capacitance",
    "inductor": "inductance",
}
"""Each passive device's primary parameter name."""

_PASSIVE_DRIFT: Final = {jolty: spice for spice, jolty in models.PASSIVE_DRIFT.items()}
"""The SPICE name of each temperature coefficient of a passive device."""

_CURVATURE: Final = tuple(models.PASSIVE_CURVATURE.values())
"""A resistor's or a capacitor's voltage coefficients, linear then quadratic."""

_FLICKER: Final = {jolty: spice for spice, (jolty, _) in models.FLICKER.items()}
"""A resistor's flicker noise parameters, as ngspice's model card names them."""


def _passive_drift(
    model: str,
    rest: list[tuple[str, JsonValue]],
    named: str,
) -> list[str]:
    """Formats non-primary drift parameters for a passive SPICE element."""
    drift: list[str] = []
    for key, amount in rest:
        if key in _CURVATURE or (named and key in _FLICKER):
            continue
        if key not in _PASSIVE_DRIFT:
            message = f"a {model}'s {key} has no ngspice element parameter"
            raise _unwritable(message)
        drift.append(f"{_PASSIVE_DRIFT[key]}={amount!r}")
    return drift


def _passive(
    model: str,
    values: Mapping[str, JsonValue],
    nodes: list[str],
    dialect: Dialect,
    card: tuple[str, list[str]] | None = None,
) -> str:
    """Formats a resistor, capacitor, or inductor value and drift parameters.

    Voltage-dependent resistors and capacitors emit ngspice behavioral ``R=`` or ``Q=``
    expressions. Resistor flicker noise emits a companion `.model` card into `card`.

    Raises:
        JoltyError: When `dialect` omits a parameter or a voltage-dependent resistor
            also specifies flicker noise, which ngspice behavioral resistors omit.
    """
    primary = _PASSIVE_PRIMARY[model]
    if primary not in values:
        message = f"a {model} has no `{primary}` parameter"
        raise _unwritable(message)
    nominal = values[primary]
    rest = [(key, amount) for key, amount in values.items() if key != primary]
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
        device_name, lines = card
        written = " ".join(f"{name}={amount!r}" for name, amount in flicker.items())
        lines.append(f".model M{device_name} R({written})")
        named = f"M{device_name} "
    drift = _passive_drift(model, rest, named)
    linear, quadratic = (values.get(key, 0.0) for key in _CURVATURE)
    if linear == 0 and quadratic == 0:
        return named + " ".join([repr(nominal), *drift])
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
        written = f"R={{{nominal!r}*({factor})}}"
    else:
        # Products, not powers: ngspice's `^` raises the magnitude of a negative base.
        squared = f"{voltage}*{voltage}"
        integral = (
            f"{voltage}+{linear!r}*{squared}/2+{quadratic!r}*{squared}*{voltage}/3"
        )
        written = f"Q={{{nominal!r}*({integral})}}"
    return " ".join([written, *drift])


def _coupled(instance: ir.Instance, ground: str, dialect: Dialect) -> list[str]:
    """Formats a coupled-inductor instance into SPICE ``L`` and ``K`` lines.

    Emits one inductor ``L{name}_{k}`` per winding and one mutual coupling statement
    ``K{name}_{j}_{k}`` per coupling entry.
    """
    element(instance, dialect)
    name, terminals = instance.name, instance.terminals
    values = _cool(instance.parameters)
    lines = []
    match values:
        case {"windings": list(windings), "couplings": list(couplings)}:
            pass
        case _:
            message = f"{name} is coupled inductors without windings and couplings"
            raise _unwritable(message)
    for number, winding in enumerate(windings, start=1):
        if not isinstance(winding, dict):
            message = f"winding {number} of {name} is no inductor"
            raise _unwritable(message)
        nodes = [
            "0" if net == ground else net
            for net in (terminals[f"a{number}"], terminals[f"b{number}"])
        ]
        written = _passive("inductor", winding, nodes, dialect)
        lines.append(f"L{name}_{number} {' '.join(nodes)} {written}")
    for coupling in couplings:
        match coupling:
            case {
                "windings": [int(first), int(second)],
                "coefficient": float(value),
            } if not (isinstance(first, bool) or isinstance(second, bool)):
                lines.append(
                    f"K{name}_{first}_{second} L{name}_{first} "
                    f"L{name}_{second} {value!r}"
                )
            case _:
                message = f"{name} has a coupling SPICE cannot write: {coupling}"
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
    """Formats a switch as a behavioral current source following the PSpice law.

    Log-resistance follows a smooth cubic polynomial across the control transition from
    off to on, clamped outside both thresholds. Native XSPICE switches (``pswitch`` and
    ``aswitch``) use the same DC curve, but ngspice 42 inverts their small-signal
    conductance sign, whereas behavioral sources differentiate accurately. Current
    switches insert a zero-volt sensing source into `lines`.
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
    """Translates a behavioral source expression for ngspice and creates sense sources.

    Parses the jolty expression tree from `_core` and renders it with `_written`.
    Current references insert one zero-volt sensing source per sensed port.
    """
    expression = instance.parameters["expression"]
    if not isinstance(expression, str):
        message = f"{instance.name}'s expression is not text"
        raise _unwritable(message)

    def net(terminal: str) -> str:
        name = instance.terminals[terminal]
        return "0" if name == ground else name

    senses: dict[str, str] = {}

    raw_parameters = instance.parameters.get("parameters")
    named_parameters = raw_parameters if isinstance(raw_parameters, Mapping) else {}

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
            case {"parameter": str(name)} if name in named_parameters:
                amount = named_parameters[name]
                if (
                    isinstance(amount, int | float)
                    and not isinstance(amount, bool)
                    and math.isfinite(amount)
                ):
                    numeric = float(amount)
                    return repr(numeric) if numeric >= 0 else f"({numeric!r})"
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


_MAX_EXPORTED_EXPRESSION_LENGTH: Final = 1_000_000
"""Maximum character length allowed for an exported SPICE behavioral expression.

Guards `_written` against exponential expansion in duplicating forms. Nested `limit`
expressions repeat each bound twice (`4^d` growth), and `_interpolation` repeats the
input expression across every table segment and parabola. One million characters far
exceeds practical SPICE lines while halting blowups before memory exhaustion.
"""


def _check_expression_length(length: int) -> None:
    """Checks that an exported behavioral expression stays within the length budget."""
    if length > _MAX_EXPORTED_EXPRESSION_LENGTH:
        message = (
            f"exported expression length {length} exceeds the limit of "
            f"{_MAX_EXPORTED_EXPRESSION_LENGTH} characters"
        )
        raise ValueError(message)


def _push_written_node(
    current: JsonValue,
    read: Callable[[JsonValue], str],
    frames: list[tuple[JsonValue, bool]],
    results: list[str],
) -> None:
    """Emits a leaf expression term or schedules children of a compound node."""
    match current:
        case bool():
            message = f"expression contains {current!r}, which ngspice cannot read"
            raise ValueError(message)
        case int() | float() if math.isfinite(current):
            value = float(current)
            results.append(repr(value) if value >= 0 else f"({value!r})")
        case "time":
            results.append("time")
        case {"voltage": _} | {"current": _} | {"parameter": _}:
            results.append(read(current))
        case {"operation": str(), "operands": list(operands)}:
            frames.append((current, True))
            frames.extend((operand, False) for operand in reversed(operands))
        case {
            "table": dict() | str() | int() | float() as table,
            "abscissae": list(),
            "ordinates": list(),
            "rounding": int() | float() as rounding,
        } if not isinstance(table, bool) and not isinstance(rounding, bool):
            frames.append((current, True))
            frames.append((table, False))
        case _:
            message = f"expression contains {current!r}, which ngspice cannot read"
            raise ValueError(message)


def _finish_written_node(current: JsonValue, results: list[str]) -> str:
    """Combines already-rendered children for a compound expression node."""
    match current:
        case {"operation": str(name), "operands": list(operands)}:
            arity = len(operands)
            written = results[-arity:] if arity else []
            if arity:
                del results[-arity:]
            result = _operation(name, written, operands)
            _check_expression_length(len(result))
            return result
        case {
            "table": _,
            "abscissae": list(abscissae),
            "ordinates": list(ordinates),
            "rounding": int() | float() as rounding,
        }:
            input_text = results.pop()
            points = [
                (float(x), float(y))
                for x, y in zip(abscissae, ordinates, strict=True)
                if isinstance(x, int | float)
                and not isinstance(x, bool)
                and isinstance(y, int | float)
                and not isinstance(y, bool)
            ]
            return _interpolation(input_text, points, float(rounding))
    message = f"expression contains {current!r}, which ngspice cannot read"
    raise ValueError(message)


def _written(term: JsonValue, read: Callable[[JsonValue], str]) -> str:
    """Renders a jolty expression tree node in ngspice behavioral syntax.

    Parenthesizes every compound operation to avoid ngspice precedence ambiguities,
    delegating terminal voltages and currents to `read`. Traverses the tree iteratively
    to avoid Python recursion limits on deep expressions.

    Raises:
        ValueError: When `term` contains a node unsupported by ngspice or exceeds
            `_MAX_EXPORTED_EXPRESSION_LENGTH`.
    """
    frames: list[tuple[JsonValue, bool]] = [(term, False)]
    results: list[str] = []
    while frames:
        current, ready = frames.pop()
        if not ready:
            _push_written_node(current, read, frames, results)
        else:
            results.append(_finish_written_node(current, results))
    return results[0]


def _operation(name: str, written: list[str], operands: list[JsonValue]) -> str:
    """Formats operation `name` on rendered `written` operands with jolty semantics.

    Translates operators where ngspice semantics differ, including power signs and
    three-argument ``limit`` clamping.
    """
    match name, written:
        case "^", [base, raised]:
            _check_expression_length(len(base) + len(raised) + 6)
            return _power(base, operands[1], raised)
        case "limit", [x, a, b]:
            _check_expression_length(len(x) + 2 * len(a) + 2 * len(b) + 25)
            return f"min(max({x},min({a},{b})),max({a},{b}))"
        case "if", [condition, chosen, otherwise]:
            _check_expression_length(len(condition) + len(chosen) + len(otherwise) + 4)
            return f"({condition}?{chosen}:{otherwise})"
        case "-" | "!", [operand]:
            _check_expression_length(len(operand) + len(name) + 2)
            return f"({name}{operand})"
        case _, [left, right] if name in _INFIX:
            _check_expression_length(len(left) + len(right) + len(name) + 2)
            return f"({left}{name}{right})"
    predicted = (
        len(name) + 2 + sum(len(part) for part in written) + max(0, len(written) - 1)
    )
    _check_expression_length(predicted)
    return f"{name}({','.join(written)})"


def _power(base: str, exponent: JsonValue, raised: str) -> str:
    """Formats `base` raised to `exponent` (`raised`) with jolty sign semantics.

    Emits ngspice ``pwr`` for constant odd integer exponents to preserve the sign of
    negative bases, and ``pow`` for all other exponents. Non-integer or variable
    exponents on negative bases are invalid in jolty because differentiation takes the
    logarithm of the base.
    """
    match exponent:
        case bool():
            pass
        case int() | float() if float(exponent).is_integer() and int(exponent) % 2:
            return f"pwr({base},{raised})"
    return f"pow({base},{raised})"


def _interpolation(x: str, points: list[tuple[float, float]], rounding: float) -> str:
    """Renders piecewise-linear table `points` with parabolic corner rounding at `x`.

    Expects `points` to include extended outer endpoints. Each interior corner rounds
    within `rounding` times the shorter adjacent segment using an XSPICE parabola,
    while values outside the rounded bands follow the linear segments.
    """
    if len(points) < 2:  # noqa: PLR2004 - at least two endpoints define a segment
        message = f"table interpolation takes at least 2 points, not {len(points)}"
        raise ValueError(message)
    # Each table segment/condition embeds `x`; reject obvious blowups upfront.
    _check_expression_length(len(x) * len(points))
    xs, ys = [x_ for x_, _ in points], [y for _, y in points]
    last = len(points) - 1

    def line(start: int) -> str:
        slope = (ys[start + 1] - ys[start]) / (xs[start + 1] - xs[start])
        return f"({ys[start]!r}+({x}-({xs[start]!r}))*{slope!r})"

    # Each piece runs up to its bound, the last to no bound; a segment's line runs on
    # across the corners no parabola rounds.
    pieces: list[tuple[float, str]] = []
    running_length = 0
    for corner in range(1, last):
        below, above = xs[corner] - xs[corner - 1], xs[corner + 1] - xs[corner]
        domain = rounding * min(below, above)
        if domain <= 0:
            segment = line(corner - 1)
            running_length += len(x) + len(segment) + 8
            _check_expression_length(running_length)
            pieces.append((xs[corner], segment))
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
        _check_expression_length(running_length + 4 * len(x) + 32)
        segment = line(corner - 1)
        parabola = f"({curvature!r}*{x}*{x}+{linear!r}*{x}+{constant!r})"
        running_length += 2 * len(x) + len(segment) + len(parabola) + 16
        _check_expression_length(running_length)
        pieces += [(xs[corner] - domain, segment), (right, parabola)]
    tail = line(last - 1)
    prefix_parts: list[str] = []
    total = len(tail)
    for bound, piece in pieces:
        chunk = f"({x}<{bound!r}?{piece}:"
        total += len(chunk) + 1
        _check_expression_length(total)
        prefix_parts.append(chunk)
    prefix_parts.append(tail)
    prefix_parts.append(")" * len(pieces))
    return "".join(prefix_parts)


def _sense(instance: ir.Instance) -> str:
    """Returns the zero-volt source name sensing control current for `instance`."""
    return f"V{element(instance)}SENSE"


def _voltage_vector(
    name: str, subcircuit_names: Collection[str], dialect: Dialect
) -> str:
    """Formats a net path as a simulator ``v(...)`` vector name."""
    first, *rest = name.split("/")
    if rest and first in subcircuit_names:
        return f"v({dialect.separator.join([f'x{first}', *rest]).lower()})"
    return f"v({_node(name).lower()})"


def vector(
    circuit: ir.Circuit,
    unknown: Unknown,
    subcircuits: Collection[str] = (),
    dialect: Dialect = NGSPICE,
) -> str:
    """Returns the simulator vector name corresponding to jolty `unknown`.

    Flattened component nets such as ``X1/mid`` map to ``v(x1_mid)``, while nets inside
    unexpanded subcircuits such as ``U1/11`` join with `dialect.separator` (``xu1.11``
    in ngspice or ``xu1:11`` in Xyce). Device states such as ``M1:state`` map to the
    auxiliary node voltage ``v(_m1_state)``.

    Args:
        circuit: Circuit IR defining the unknown.
        unknown: Circuit unknown (net voltage, branch current, or device state).
        subcircuits: Component names kept unexpanded during netlist export.
        dialect: Target simulator dialect supplying hierarchy separators and rules.

    Returns:
        Simulator vector name corresponding to `unknown`.

    Raises:
        JoltyError: When `unknown` is a device-internal unknown whose simulator vector
            naming is not portable.
    """
    if unknown.nature == "state":
        path, _, label = unknown.name.rpartition(":")
        return f"v(_{_node(path)}_{label})".lower()
    flat = flatten(circuit, subcircuits)
    top = flat.components[flat.top]
    if unknown.nature == "voltage":
        first = unknown.name.partition("/")[0]
        kept = top.instance(first)
        names = (
            (first,) if kept is not None and isinstance(kept.of, ir.ComponentOf) else ()
        )
        return _voltage_vector(unknown.name, names, dialect)
    path, _, label = unknown.name.rpartition(":")
    name = _node(path)
    found = [instance for instance in top.instances if instance.name == name]
    if not found:
        message = (
            f"{dialect.simulator} names the internal unknown {unknown.name} its own way"
        )
        raise _unwritable(message)
    (instance,) = found
    if instance.of == ir.DeviceOf(model="coupled_inductors"):
        return f"i(l{name.lower()}_{label.removeprefix('current')})"
    source = _sense(instance) if label == "control" else element(instance, dialect)
    return f"i({source.lower()})"


def node_voltages(
    keyword: Literal[".ic", ".nodeset"],
    circuit: ir.Circuit,
    voltages: Mapping[str, float],
    *,
    subcircuits: Collection[str] = (),
    dialect: Dialect = NGSPICE,
) -> list[str]:
    """Formats a ``.ic`` or ``.nodeset`` statement assigning `voltages` to nets.

    Net paths in `voltages` are mapped through [`vector`][jolty.spice.export.vector]
    for `circuit` with `subcircuits` in `dialect`. Returns an empty list when
    `voltages` is empty.

    Args:
        keyword: SPICE directive name (``".ic"`` or ``".nodeset"``).
        circuit: Circuit IR whose net names are resolved.
        voltages: Mapping from circuit net paths to target voltages in volts.
        subcircuits: Component names kept unexpanded during netlist export.
        dialect: Target simulator dialect supplying subcircuit separators.

    Returns:
        Single-element list with the formatted directive line, or an empty list when
        `voltages` is empty.

    Examples:
        >>> from jolty import Builder, Role, circuit, devices
        >>> from jolty.units import kohm
        >>> c = Builder("rc")
        >>> ground = c.port("ground", Role.GROUND)
        >>> r = c.place(devices.resistor(1 * kohm), name="R1")
        >>> c.connect(r.a, name="out")
        >>> c.connect(r.b, ground)
        >>> node_voltages(".ic", circuit(c.finish()), {"out": 0.5})
        ['.ic v(out)=0.5']
    """
    if not voltages:
        return []
    flat = flatten(circuit, subcircuits)
    top = flat.components[flat.top]
    subcircuit_names = frozenset(
        instance.name
        for instance in top.instances
        if isinstance(instance.of, ir.ComponentOf)
    )
    parts = [
        f"{_voltage_vector(net, subcircuit_names, dialect)}={volts!r}"
        for net, volts in voltages.items()
    ]
    return [f"{keyword} {' '.join(parts)}"]


def _waveform(spec: JsonValue, dialect: Dialect) -> str:
    """Formats a source waveform specification in the syntax of `dialect`.

    Prepends an explicit time-zero ``DC`` value when `dialect.zero_dc_value` is set.
    """
    written = _shape(spec, dialect.instant)
    if dialect.zero_dc_value and not written.startswith("DC"):
        return f"DC {_core.waveform_value(json.dumps(spec), 0.0)!r} {written}"
    return written


def _shape(spec: JsonValue, instant: float) -> str:
    """Formats a SPICE source waveform, substituting `instant` for special zero times.

    Zero pulse widths and zero exponential delays are replaced by `instant` so the
    simulator does not apply default step or stop-time substitutions.
    """
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
        case {"exponential": dict(exponential)}:
            order = (
                "initial",
                "pulsed",
                "rise_delay",
                "rise_time_constant",
                "fall_delay",
                "fall_time_constant",
            )
            delays = {"rise_delay", "fall_delay"}
            numbers = [
                instant if key in delays and exponential[key] == 0 else exponential[key]
                for key in order
            ]
            return _function("EXP", numbers)
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

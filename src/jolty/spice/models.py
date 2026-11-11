# Copyright 2026 Maxence Faldor
"""SPICE's model cards, and the parameters of the jolty devices they make.

Each SPICE parameter maps to a jolty device's parameter, converted to SI base units
where SPICE writes centimeters. Parameters that only matter away from the nominal
temperature of 27 °C, or only to noise, change nothing jolty computes and are left out;
any other parameter jolty does not model is an error, never silently dropped.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from pydantic import JsonValue

from jolty.errors import SpiceError
from jolty.spice.expressions import Expression, Number, evaluate, number, parse
from jolty.spice.netlist import Statement, tokens

type Table = Mapping[str, tuple[str, float]]
"""Each SPICE parameter's jolty name, and the factor to SI base units."""

DIODE: Final[Table] = {
    "is": ("saturation_current", 1.0),
    "n": ("emission_coefficient", 1.0),
    "rs": ("series_resistance", 1.0),
    "cjo": ("junction_capacitance", 1.0),
    "cj0": ("junction_capacitance", 1.0),
    "cj": ("junction_capacitance", 1.0),
    "vj": ("junction_potential", 1.0),
    "pb": ("junction_potential", 1.0),
    "m": ("grading_coefficient", 1.0),
    "mj": ("grading_coefficient", 1.0),
    "fc": ("depletion_coefficient", 1.0),
    "tt": ("transit_time", 1.0),
}

JFET: Final[Table] = {
    "vto": ("threshold_voltage", 1.0),
    "beta": ("transconductance_parameter", 1.0),
    "lambda": ("channel_length_modulation", 1.0),
    "rd": ("drain_resistance", 1.0),
    "rs": ("source_resistance", 1.0),
    "cgs": ("gate_source_capacitance", 1.0),
    "cgd": ("gate_drain_capacitance", 1.0),
    "pb": ("junction_potential", 1.0),
    "is": ("saturation_current", 1.0),
    "fc": ("depletion_coefficient", 1.0),
    "b": ("doping_tail", 1.0),
}

BIPOLAR: Final[Table] = {
    "is": ("saturation_current", 1.0),
    "bf": ("forward_beta", 1.0),
    "br": ("reverse_beta", 1.0),
    "nf": ("forward_emission_coefficient", 1.0),
    "nr": ("reverse_emission_coefficient", 1.0),
    "vaf": ("forward_early_voltage", 1.0),
    "va": ("forward_early_voltage", 1.0),
    "var": ("reverse_early_voltage", 1.0),
    "vb": ("reverse_early_voltage", 1.0),
    "ikf": ("forward_knee_current", 1.0),
    "ik": ("forward_knee_current", 1.0),
    "ikr": ("reverse_knee_current", 1.0),
    "ise": ("base_emitter_leakage_current", 1.0),
    "ne": ("base_emitter_leakage_emission_coefficient", 1.0),
    "isc": ("base_collector_leakage_current", 1.0),
    "nc": ("base_collector_leakage_emission_coefficient", 1.0),
    "rb": ("base_resistance", 1.0),
    "rbm": ("minimum_base_resistance", 1.0),
    "irb": ("base_resistance_current", 1.0),
    "re": ("emitter_resistance", 1.0),
    "rc": ("collector_resistance", 1.0),
    "cje": ("base_emitter_capacitance", 1.0),
    "vje": ("base_emitter_potential", 1.0),
    "pe": ("base_emitter_potential", 1.0),
    "mje": ("base_emitter_grading", 1.0),
    "me": ("base_emitter_grading", 1.0),
    "tf": ("forward_transit_time", 1.0),
    "xtf": ("transit_time_bias_coefficient", 1.0),
    "vtf": ("transit_time_voltage", 1.0),
    "itf": ("transit_time_current", 1.0),
    "cjc": ("base_collector_capacitance", 1.0),
    "vjc": ("base_collector_potential", 1.0),
    "pc": ("base_collector_potential", 1.0),
    "mjc": ("base_collector_grading", 1.0),
    "mc": ("base_collector_grading", 1.0),
    "xcjc": ("internal_base_fraction", 1.0),
    "tr": ("reverse_transit_time", 1.0),
    "fc": ("depletion_coefficient", 1.0),
    "iss": ("substrate_saturation_current", 1.0),
    "ns": ("substrate_emission_coefficient", 1.0),
    "cjs": ("substrate_capacitance", 1.0),
    "csub": ("substrate_capacitance", 1.0),
    "ccs": ("substrate_capacitance", 1.0),
    "vjs": ("substrate_potential", 1.0),
    "ps": ("substrate_potential", 1.0),
    "mjs": ("substrate_grading", 1.0),
    "ms": ("substrate_grading", 1.0),
}

MOSFET: Final[Table] = {
    "vto": ("threshold_voltage", 1.0),
    "kp": ("transconductance_parameter", 1.0),
    "gamma": ("body_effect", 1.0),
    "phi": ("surface_potential", 1.0),
    "lambda": ("channel_length_modulation", 1.0),
    "ld": ("lateral_diffusion", 1.0),
    "tox": ("oxide_thickness", 1.0),
    "uo": ("surface_mobility", 1e-4),
    "nsub": ("substrate_doping", 1e6),
    "nss": ("surface_state_density", 1e4),
    "rd": ("drain_resistance", 1.0),
    "rs": ("source_resistance", 1.0),
    "rsh": ("sheet_resistance", 1.0),
    "is": ("saturation_current", 1.0),
    "js": ("saturation_current_density", 1.0),
    "pb": ("junction_potential", 1.0),
    "cbd": ("bulk_drain_capacitance", 1.0),
    "cbs": ("bulk_source_capacitance", 1.0),
    "cj": ("bottom_capacitance", 1.0),
    "cjsw": ("sidewall_capacitance", 1.0),
    "mj": ("bottom_grading", 1.0),
    "mjsw": ("sidewall_grading", 1.0),
    "fc": ("depletion_coefficient", 1.0),
    "cgso": ("gate_source_overlap", 1.0),
    "cgdo": ("gate_drain_overlap", 1.0),
    "cgbo": ("gate_bulk_overlap", 1.0),
}

MOSFET_INSTANCE: Final[Table] = {
    "w": ("width", 1.0),
    "l": ("length", 1.0),
    "m": ("multiplier", 1.0),
    "nrd": ("drain_squares", 1.0),
    "nrs": ("source_squares", 1.0),
    "ad": ("drain_area", 1.0),
    "as": ("source_area", 1.0),
    "pd": ("drain_perimeter", 1.0),
    "ps": ("source_perimeter", 1.0),
}

CONSTRUCTIONS: Final = {1.0: "vertical", -1.0: "lateral"}
"""The construction of a bipolar transistor of each value of SPICE's ``SUBS``."""

GATE_TYPES: Final = {1.0: "opposite_doping", -1.0: "same_doping", 0.0: "aluminum"}
"""The gate material of each value of SPICE's ``TPG``."""

SWITCH: Final[Table] = {
    "ron": ("on_resistance", 1.0),
    "roff": ("off_resistance", 1.0),
    "von": ("on_voltage", 1.0),
    "voff": ("off_voltage", 1.0),
}

TEMPERATURE: Final = frozenset(
    {
        "tnom",
        "t_measured",
        "t_abs",
        "t_rel_global",
        "t_rel_local",
        "eg",
        "xti",
        "xtb",
        "tc",
        "tc1",
        "tc2",
        "tce",
        "trs",
        "trs1",
        "trs2",
        "trb",
        "tre",
        "trc",
        "tikf",
        "tbv1",
        "tbv2",
        "vtotc",
        "betatce",
        "cta",
        "ctp",
        "pta",
        "ptp",
    }
)
"""Parameters that change nothing at jolty's temperature, the nominal 27 °C."""

NOISE: Final = frozenset({"kf", "af", "nlev", "gdsnoi"})
"""Parameters that only noise analyses read."""


@dataclass(frozen=True, slots=True)
class ModelCard:
    """A model card: its name, its type and its parameters' expressions.

    The parameters are by their case-folded SPICE names, with any the card inherits
    through ``AKO`` already merged.
    """

    name: str
    kind: str
    parameters: Mapping[str, Expression]
    statement: Statement = field(repr=False)


def read(statement: Statement, known: Mapping[str, ModelCard]) -> ModelCard:
    """The model card `statement` defines.

    A card may inherit from a model in `known`, through ``AKO:``.

    Raises:
        SpiceError: When the statement is not a model card.
    """
    words = statement.text.split(maxsplit=2)
    if len(words) < 3:  # noqa: PLR2004 - `.model`, the name and the type
        message = "a model card names its model and its type"
        raise statement.fail(message)
    _, name, rest = words
    base: ModelCard | None = None
    if rest.casefold().startswith("ako:"):
        base_name, _, rest = rest[4:].partition(" ")
        base = known.get(base_name.casefold())
        if base is None:
            message = f"no model `{base_name}` precedes it"
            raise statement.fail(message)
    word = re.match(r"\w*", rest)
    kind = "" if word is None else rest[: word.end()]
    body = rest[len(kind) :].strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    parameters = dict(base.parameters) if base else {}
    parameters |= assignments(statement, tokens(body))
    resolved = kind.casefold() or (base.kind if base else "")
    return ModelCard(name, resolved, parameters, statement)


def assignments(statement: Statement, words: list[str]) -> dict[str, Expression]:
    """The parameters `words` assign, ``name = value`` each, by case-folded name.

    A value is a number, a name, or an expression in braces or quotes.

    Raises:
        SpiceError: When the words do not pair names with values.
    """
    found: dict[str, Expression] = {}
    for position in range(0, len(words), 3):
        name, *assigned = words[position : position + 3]
        if len(assigned) != 2 or assigned[0] != "=":  # noqa: PLR2004 - `=` and a value
            message = f"`{name}` is not assigned a value"
            raise statement.fail(message)
        found[name.casefold()] = value(assigned[1])
    return found


def value(word: str) -> Expression:
    """The expression a value writes.

    A value is a number, a name, or an expression in braces or quotes.
    """
    written = number(word)
    return Number(written) if written is not None else parse(word)


@dataclass(frozen=True, slots=True)
class Made:
    """What a model card makes: a jolty device model and its parameters, in SI units."""

    model: str
    parameters: dict[str, JsonValue]


def device(
    card: ModelCard,
    scope: Mapping[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> Made:
    """The device a model card makes for an instance with `instance` parameters.

    An instance's parameters include its area. The card's expressions are evaluated in
    `scope`.

    Raises:
        SpiceError: When the model's type or a parameter has no jolty counterpart.
    """
    values = {
        name: evaluate(expression, scope)
        for name, expression in card.parameters.items()
    }
    match card.kind:
        case "d":
            area = instance.get("area", 1.0) * instance.get("m", 1.0)
            values = _scaled(values, area, ("is", "cjo", "cj0", "cj"), ("rs",))
            return Made("diode", _mapped(card, values, DIODE, statement))
        case "npn" | "pnp":
            return Made(
                "bipolar_transistor", _bipolar(card, values, instance, statement)
            )
        case "njf" | "pjf":
            area = instance.get("area", 1.0) * instance.get("m", 1.0)
            values = _scaled(values, area, ("beta", "is", "cgs", "cgd"), ("rd", "rs"))
            parameters = _mapped(card, values, JFET, statement)
            return Made("jfet", {"channel": card.kind[0], **parameters})
        case "nmos" | "pmos":
            return Made("mosfet", _mosfet(card, values, instance, statement))
        case "vswitch":
            return Made(
                "voltage_controlled_switch", _mapped(card, values, SWITCH, statement)
            )
    message = f"jolty has no device for `{card.name}`, a model of type `{card.kind}`"
    raise statement.fail(message)


def unmatched(card: ModelCard, parameter: str, statement: Statement) -> SpiceError:
    """The error that jolty's devices have no counterpart to `parameter` of `card`."""
    message = (
        f"jolty's devices have no counterpart to `{parameter.upper()}` of `{card.name}`"
    )
    return statement.fail(message)


def _scaled(
    values: dict[str, float],
    area: float,
    multiplied: tuple[str, ...],
    divided: tuple[str, ...],
) -> dict[str, float]:
    """`values` scaled by `area`, as SPICE scales a device by its area.

    Each `multiplied` parameter is multiplied by the area, each `divided` one divided.
    """
    return {
        name: value * area
        if name in multiplied
        else value / area
        if name in divided
        else value
        for name, value in values.items()
    }


def _mapped(
    card: ModelCard,
    values: Mapping[str, float],
    table: Table,
    statement: Statement,
    skipped: frozenset[str] = frozenset(),
) -> dict[str, JsonValue]:
    """`values` by their jolty names, in SI units.

    Raises:
        SpiceError: When a parameter has no jolty counterpart.
    """
    parameters: dict[str, JsonValue] = {}
    for name, value in values.items():
        if name in skipped or name in NOISE:
            continue
        if name in TEMPERATURE:
            if name == "tnom" and value != 27:  # noqa: PLR2004 - the nominal temperature
                message = f"`{card.name}` is measured at {value!r} °C, not 27 °C"
                raise statement.fail(message)
            continue
        if name not in table:
            raise unmatched(card, name, statement)
        jolty, factor = table[name]
        parameters[jolty] = value * factor
    return parameters


def _bipolar(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> dict[str, JsonValue]:
    """A bipolar transistor's parameters, from its model card and its instance's.

    The transistor has a substrate, as in SPICE: an NPN is vertical and a PNP lateral,
    unless ``SUBS`` builds it otherwise.

    Raises:
        SpiceError: When a parameter has no jolty counterpart.
    """
    construction = values.pop("subs", 1.0 if card.kind == "npn" else -1.0)
    if construction not in CONSTRUCTIONS:
        message = f"`SUBS` is 1 or -1, not {construction!r}"
        raise statement.fail(message)
    area = instance.get("area", 1.0) * instance.get("m", 1.0)
    multiplied = (
        *("is", "ikf", "ik", "ikr", "ise", "isc", "itf", "cje", "cjc", "irb"),
        *("iss", "cjs", "csub", "ccs"),
    )
    values = _scaled(values, area, multiplied, ("rb", "rbm", "re", "rc"))
    parameters = _mapped(card, values, BIPOLAR, statement)
    return {
        "polarity": card.kind,
        "substrate": CONSTRUCTIONS[construction],
        **parameters,
    }


def _mosfet(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> dict[str, JsonValue]:
    """A MOSFET's parameters, from its model card and its instance's.

    Raises:
        SpiceError: When the model is not of level 1, or a parameter has no jolty
            counterpart.
    """
    level = values.pop("level", 1.0)
    if level != 1:
        message = f"`{card.name}` is a MOSFET of level {level:g}; jolty models level 1"
        raise statement.fail(message)
    gate = values.pop("tpg", None)
    # A model card's geometry is its instances' default.
    geometry = {name: values.pop(name) for name in ("l", "w") if name in values}
    parameters = _mapped(card, values, MOSFET, statement)
    parameters |= _mapped(card, geometry | dict(instance), MOSFET_INSTANCE, statement)
    parameters["channel"] = card.kind[0]
    if gate is not None:
        if gate not in GATE_TYPES:
            message = f"`TPG` is 1, -1 or 0, not {gate!r}"
            raise statement.fail(message)
        parameters["gate_material"] = GATE_TYPES[gate]
    return parameters

# Copyright 2026 Maxence Faldor
"""SPICE's model cards, and the parameters of the jolty devices they make.

Each SPICE parameter maps to a jolty device's parameter, converted to SI base units
where SPICE writes centimeters, and the nominal temperature ``TNOM`` from degrees
Celsius to kelvins. Parameters that change nothing jolty computes are left out: those
that only choose among noise equations, of a VDMOS card, its part's annotations, its
``PHI`` and its thermal network, and of a BSIM3 card, its bins' ranges, its safe
operating area and what only its non-quasi-static and area calculation models read
where they are off, as the reader's tables say. Any other parameter jolty does not
model is an error, never silently dropped.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from pydantic import JsonValue

from jolty.devices import BSIM3_BINNED, BSIM3_CARD
from jolty.errors import SpiceError
from jolty.spice.expressions import Expression, Number, evaluate, number, parse
from jolty.spice.netlist import Statement, tokens
from jolty.units import celsius

type Table = Mapping[str, tuple[str, float]]
"""Each SPICE parameter's jolty name, and the factor to SI base units."""

FLICKER: Final[Table] = {
    "kf": ("flicker_coefficient", 1.0),
    "af": ("flicker_exponent", 1.0),
}
"""The flicker noise of a diode's or a transistor's current: SPICE's ``KF`` and
``AF``."""

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
    "bv": ("breakdown_voltage", 1.0),
    "ibv": ("breakdown_current", 1.0),
    "nbv": ("breakdown_emission_coefficient", 1.0),
    "tcv": ("breakdown_temperature_coefficient", 1.0),
    "eg": ("energy_gap", 1.0),
    "xti": ("saturation_current_temperature_exponent", 1.0),
    **FLICKER,
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
    "vtotc": ("threshold_temperature_coefficient", 1.0),
    "tcv": ("threshold_temperature_coefficient", -1.0),
    "bex": ("beta_temperature_exponent", 1.0),
    "eg": ("energy_gap", 1.0),
    "xti": ("saturation_current_temperature_exponent", 1.0),
    **FLICKER,
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
    "ptf": ("excess_phase", 1.0),
    "iss": ("substrate_saturation_current", 1.0),
    "ns": ("substrate_emission_coefficient", 1.0),
    "cjs": ("substrate_capacitance", 1.0),
    "csub": ("substrate_capacitance", 1.0),
    "ccs": ("substrate_capacitance", 1.0),
    "vjs": ("substrate_potential", 1.0),
    "ps": ("substrate_potential", 1.0),
    "mjs": ("substrate_grading", 1.0),
    "ms": ("substrate_grading", 1.0),
    "eg": ("energy_gap", 1.0),
    "xti": ("saturation_current_temperature_exponent", 1.0),
    "xtb": ("beta_temperature_exponent", 1.0),
    **FLICKER,
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
    "vmax": ("max_drift_velocity", 1.0),
    "xj": ("junction_depth", 1.0),
    "nfs": ("fast_surface_state_density", 1e4),
    "eta": ("static_feedback", 1.0),
    "delta": ("narrow_width_factor", 1.0),
    "theta": ("mobility_modulation", 1.0),
    "kappa": ("saturation_field_factor", 1.0),
    "wd": ("width_narrowing", 1.0),
    "xl": ("length_adjustment", 1.0),
    "xw": ("width_adjustment", 1.0),
    "delvto": ("threshold_adjustment", 1.0),
    "delvt0": ("threshold_adjustment", 1.0),
    **FLICKER,
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

VDMOS: Final[Table] = {
    "vto": ("threshold_voltage", 1.0),
    "vth0": ("threshold_voltage", 1.0),
    "kp": ("transconductance_parameter", 1.0),
    "lambda": ("channel_length_modulation", 1.0),
    "theta": ("mobility_modulation", 1.0),
    "mtriode": ("triode_conductance_multiplier", 1.0),
    "ksubthres": ("subthreshold_slope", 1.0),
    "subshift": ("subthreshold_shift", 1.0),
    "rd": ("drain_resistance", 1.0),
    "rs": ("source_resistance", 1.0),
    "rg": ("gate_resistance", 1.0),
    "rq": ("quasi_saturation_resistance", 1.0),
    "vq": ("quasi_saturation_voltage", 1.0),
    "rds": ("drain_source_resistance", 1.0),
    "cgs": ("gate_source_capacitance", 1.0),
    "cgdmax": ("max_gate_drain_capacitance", 1.0),
    "cgdmin": ("min_gate_drain_capacitance", 1.0),
    "a": ("gate_drain_capacitance_coefficient", 1.0),
    "is": ("saturation_current", 1.0),
    "n": ("emission_coefficient", 1.0),
    "rb": ("body_resistance", 1.0),
    "cjo": ("junction_capacitance", 1.0),
    "vj": ("junction_potential", 1.0),
    "m": ("grading_coefficient", 1.0),
    "fc": ("depletion_coefficient", 1.0),
    "tt": ("transit_time", 1.0),
    "bv": ("breakdown_voltage", 1.0),
    "ibv": ("breakdown_current", 1.0),
    "nbv": ("breakdown_emission_coefficient", 1.0),
    "eg": ("energy_gap", 1.0),
    "xti": ("saturation_current_temperature_exponent", 1.0),
    "tcvth": ("threshold_temperature_coefficient", 1.0),
    "vtotc": ("threshold_temperature_coefficient", 1.0),
    "mu": ("transconductance_temperature_exponent", 1.0),
    "bex": ("transconductance_temperature_exponent", 1.0),
    "tksubthres1": ("subthreshold_slope_temperature_coefficient", 1.0),
    "tksubthres2": ("subthreshold_slope_quadratic_temperature_coefficient", 1.0),
    "texp0": ("drain_resistance_temperature_exponent", 1.0),
    "trd1": ("drain_resistance_temperature_coefficient", 1.0),
    "trd2": ("drain_resistance_quadratic_temperature_coefficient", 1.0),
    "texp1": ("quasi_saturation_temperature_exponent", 1.0),
    "trs1": ("source_resistance_temperature_coefficient", 1.0),
    "trs2": ("source_resistance_quadratic_temperature_coefficient", 1.0),
    "trg1": ("gate_resistance_temperature_coefficient", 1.0),
    "trg2": ("gate_resistance_quadratic_temperature_coefficient", 1.0),
    "trb1": ("body_resistance_temperature_coefficient", 1.0),
    "trb2": ("body_resistance_quadratic_temperature_coefficient", 1.0),
    **FLICKER,
}
"""ngspice's VDMOS, the power MOSFETs' model of LTspice's cards."""

VDMOS_INSTANCE: Final[Table] = {"m": ("multiplier", 1.0)}
"""A VDMOS's own parameters: the number of transistors in parallel."""

VDMOS_KINDS: Final = frozenset({"vdmosn", "vdmosp"})
"""A VDMOS card's types, as the reader gives them: of its channel's polarity."""

VDMOS_ANNOTATIONS: Final = frozenset({"mfg", "vds", "ron", "qg"})
"""What LTspice's VDMOS cards say of their part, its maker, ratings and gate charge,
which ngspice reads past, and the reader with it."""

VDMOS_SELF_HEATING: Final = frozenset({"rthjc", "rthca", "cthj"})
"""A VDMOS card's thermal network, which ngspice heats a transistor on only where its
instance says ``thermal``: elsewhere it changes nothing, and jolty reads no such
instance."""

_CHANNEL: Final = re.compile(r"\b([np])chan\b", re.IGNORECASE)
"""A VDMOS card's polarity, ``nchan`` or ``pchan``, a word anywhere in its card."""

BSIM3_LEVELS: Final = frozenset({8, 49})
"""The levels of a BSIM3 card, ngspice's 8 and HSPICE's 49."""

BSIM3_INSTANCE: Final[Table] = {
    **MOSFET_INSTANCE,
    "delvto": ("delvto", 1.0),
    "mulu0": ("mulu0", 1.0),
}
"""A BSIM3 instance's own parameters: a MOSFET's, a shift of its threshold and a factor
of its mobility."""

BSIM3_SELECTORS: Final = ("mobmod", "capmod", "noimod")
"""A BSIM3 card's choices among its equations, each a whole number."""

BSIM3_OFF: Final = ("nqsmod", "acnqsmod", "acm")
"""BSIM3's non-quasi-static models, of the transient and of the small signal, and
HSPICE's area calculation method, which jolty does not model: a card or an instance
reads only with each off."""

BSIM3_UNREAD: Final = frozenset(
    {
        *("lmin", "lmax", "wmin", "wmax"),
        *("elm", "lelm", "welm", "pelm", "calcacm", "geo"),
        *("hdif", "ldif", "ld", "rd", "rs", "rdc", "rsc", "wmlt"),
        *("vgs_max", "vgd_max", "vgb_max", "vds_max", "vbs_max", "vbd_max"),
        *("vgsr_max", "vgdr_max", "vgbr_max", "vbsr_max", "vbdr_max"),
    }
)
"""What a BSIM3 card says that changes nothing ngspice computes with the
non-quasi-static models and the area calculation off: its bin's ranges, which only
select it, the non-quasi-static model's Elmore constant, the area calculation's
parameters, and the safe operating area, which only ngspice's checks read."""

BSIM3_DENSITIES: Final = {"nch": 1e20, "ngate": 1.000001e24}
"""The concentrations of a BSIM3 card that ngspice reads per cubic meter above a bound,
and per cubic centimeter below it, as its ``b3mpar.c`` reads each."""

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
"""PSpice's ``VSWITCH`` parameters, as a voltage-controlled switch takes them."""

CURRENT_SWITCH: Final[Table] = {
    "ron": ("on_resistance", 1.0),
    "roff": ("off_resistance", 1.0),
    "ion": ("on_current", 1.0),
    "ioff": ("off_current", 1.0),
}
"""PSpice's ``ISWITCH`` parameters, as a current-controlled switch takes them."""

NOMINAL_TEMPERATURES: Final = frozenset({"tnom", "t_measured"})
"""The parameters that give the temperature, in degrees Celsius, a card was measured at:
the nominal temperature of the device it makes. ``T_MEASURED`` is PSpice's."""

OWN_TEMPERATURES: Final = frozenset(
    {"t_abs", "t_rel_global", "t_rel_local", "temp", "dtemp"}
)
"""The parameters that set a device's own temperature apart from the circuit's: a card's
``T_ABS``, ``T_REL_GLOBAL`` and ``T_REL_LOCAL``, PSpice's, and an instance's ``TEMP``
and ``DTEMP``, ngspice's."""


def own_temperature(
    card: Mapping[str, float], instance: Mapping[str, float], statement: Statement
) -> dict[str, JsonValue]:
    """The device's own temperature, as a held one or an offset, if it has one.

    A card's ``T_ABS`` and an instance's ``TEMP``, in degrees Celsius, hold the device
    there; a card's ``T_REL_GLOBAL`` and an instance's ``DTEMP`` raise it above the
    circuit's.

    Raises:
        SpiceError: When the device sets it more than once, or by ``T_REL_LOCAL``,
            relative to the card a card inherits from, which jolty does not read.
    """
    given = {
        name: value
        for name, value in (*card.items(), *instance.items())
        if name in OWN_TEMPERATURES
    }
    if "t_rel_local" in given:
        message = "jolty reads no `T_REL_LOCAL`, relative to an inherited card"
        raise statement.fail(message)
    if len(given) > 1:
        names = ", ".join(f"`{name.upper()}`" for name in given)
        message = f"the device sets its own temperature twice: {names}"
        raise statement.fail(message)
    match given:
        case {"t_abs": value} | {"temp": value}:
            return {"temperature": celsius(value).value}
        case {"t_rel_global": value} | {"dtemp": value}:
            return {"temperature_offset": value}
    return {}


NOISE: Final = frozenset({"nlev", "gdsnoi"})
"""PSpice's choices among a MOSFET's noise equations, which jolty reads past: it keeps
SPICE's own, which it writes for ngspice 43 and later as `nlev=1`."""


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
    version: str | None = None
    """The model's version, as a BSIM card names the equations it is of, if given."""


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
    resolved = kind.casefold() or (base.kind if base else "")
    if resolved.startswith("vdmos"):
        resolved, body = _vdmos_card(resolved, body)
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    words = tokens(body)
    if resolved in VDMOS_KINDS:
        words = _annotations_removed(words)
    version, words = _version_taken(words)
    if version is None and base is not None:
        version = base.version
    parameters = dict(base.parameters) if base else {}
    parameters |= assignments(statement, words)
    return ModelCard(name, resolved, parameters, statement, version)


def _version_taken(words: list[str]) -> tuple[str | None, list[str]]:
    """A card's ``VERSION``, as written, such as ``3.3.0``, and its words without it."""
    for position in range(len(words) - 2):
        if words[position].casefold() == "version" and words[position + 1] == "=":
            return words[position + 2], words[:position] + words[position + 3 :]
    return None, words


def _vdmos_card(kind: str, body: str) -> tuple[str, str]:
    """A VDMOS card's type, of its channel's polarity, and its body without it.

    As ngspice reads a card, ``VDMOS`` is an n-channel's, and a p-channel's given
    ``pchan`` anywhere in its card or the type ``VDMOSP``.
    """
    flags = {flag.casefold() for flag in _CHANNEL.findall(body)}
    polarity = "p" if "p" in flags or kind == "vdmosp" else "n"
    return f"vdmos{polarity}", _CHANNEL.sub(" ", body).strip()


def _annotations_removed(words: list[str]) -> list[str]:
    """A VDMOS card's words without its part's annotations, `VDMOS_ANNOTATIONS`."""
    kept: list[str] = []
    position = 0
    while position < len(words):
        assigned = words[position + 1 : position + 2] == ["="]
        if assigned and words[position].casefold() in VDMOS_ANNOTATIONS:
            position += 3
            continue
        kept.append(words[position])
        position += 1
    return kept


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
    every = {
        name: evaluate(expression, scope)
        for name, expression in card.parameters.items()
    }
    own = own_temperature(every, instance, statement)
    values = {
        name: value for name, value in every.items() if name not in OWN_TEMPERATURES
    }
    instance = {
        name: value for name, value in instance.items() if name not in OWN_TEMPERATURES
    }
    made = _device(card, values, instance, statement)
    return Made(made.model, {**made.parameters, **own})


def _device(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> Made:
    """The device a model card of `values` makes for an instance of `instance`.

    Raises:
        SpiceError: When the model's type or a parameter has no jolty counterpart.
    """
    match card.kind:
        case "d":
            area = instance.get("area", 1.0) * instance.get("m", 1.0)
            values = _scaled(values, area, ("is", "cjo", "cj0", "cj"), ("rs",))
            # ngspice multiplies the breakdown current by the multiplier, not the area.
            values = _scaled(values, instance.get("m", 1.0), ("ibv",), ())
            made = Made("diode", _mapped(card, values, DIODE, statement))
        case "npn" | "pnp":
            made = Made(
                "bipolar_transistor", _bipolar(card, values, instance, statement)
            )
        case "njf" | "pjf":
            area = instance.get("area", 1.0) * instance.get("m", 1.0)
            values = _scaled(values, area, ("beta", "is", "cgs", "cgd"), ("rd", "rs"))
            parameters = _mapped(card, values, JFET, statement)
            made = Made("jfet", {"channel": card.kind[0], **parameters})
        case "nmos" | "pmos" if values.get("level") in BSIM3_LEVELS:
            made = Made("bsim3", _bsim3(card, values, instance, statement))
        case "nmos" | "pmos":
            made = Made("mosfet", _mosfet(card, values, instance, statement))
        case "vdmosn" | "vdmosp":
            made = Made("vdmos", _vdmos(card, values, instance, statement))
        case "vswitch":
            made = Made(
                "voltage_controlled_switch", _mapped(card, values, SWITCH, statement)
            )
        case "iswitch":
            made = Made(
                "current_controlled_switch",
                _mapped(card, values, CURRENT_SWITCH, statement),
            )
        case _:
            message = (
                f"jolty has no device for `{card.name}`, a model of type `{card.kind}`"
            )
            raise statement.fail(message)
    return made


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
        if name in NOMINAL_TEMPERATURES:
            parameters["nominal_temperature"] = celsius(value).value
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
        SpiceError: When the model is not of level 1 or 3, or a parameter has no jolty
            counterpart.
    """
    level = values.pop("level", 1.0)
    if level not in {1, 3}:
        message = f"`{card.name}` is a MOSFET of level {level:g}; jolty models 1 and 3"
        raise statement.fail(message)
    if level == 3:  # noqa: PLR2004 - the level of the semi-empirical model
        # Level 3 has no channel-length modulation: PSpice and ngspice ignore it there.
        values.pop("lambda", None)
    gate = values.pop("tpg", None)
    # A model card's geometry is its instances' default.
    geometry = {name: values.pop(name) for name in ("l", "w") if name in values}
    parameters = _mapped(card, values, MOSFET, statement)
    parameters |= _mapped(card, geometry | dict(instance), MOSFET_INSTANCE, statement)
    parameters["channel"] = card.kind[0]
    if level == 3:  # noqa: PLR2004 - the level of the semi-empirical model
        parameters["level"] = 3
    if gate is not None:
        if gate not in GATE_TYPES:
            message = f"`TPG` is 1, -1 or 0, not {gate!r}"
            raise statement.fail(message)
        parameters["gate_material"] = GATE_TYPES[gate]
    return parameters


def _vdmos(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> dict[str, JsonValue]:
    """A VDMOS's parameters, from its model card and its instance's.

    ``PHI`` feeds only ngspice's distortion analysis, which jolty has not, and its
    thermal network only an instance that heats, so both change nothing jolty computes
    and are left out; ngspice refuses a ``PHI`` that is not positive, and so does jolty.

    Raises:
        SpiceError: When a parameter has no jolty counterpart.
    """
    potential = values.pop("phi", None)
    if potential is not None and potential <= 0:
        message = f"`PHI` of `{card.name}` must be positive, not {potential!r}"
        raise statement.fail(message)
    parameters = _mapped(card, values, VDMOS, statement, VDMOS_SELF_HEATING)
    parameters |= _mapped(card, dict(instance), VDMOS_INSTANCE, statement)
    parameters["channel"] = card.kind[-1]
    return parameters


BIN_TOLERANCE: Final = 1e-9
"""How near a bound of a bin's range a length or a width counts as on it, in meters, as
ngspice's ``INPgetModBin`` counts it."""


def holds(
    card: ModelCard, scope: Mapping[str, float], size: tuple[float, float]
) -> bool:
    """Whether a BSIM3 `card` is a bin for an instance of `size`, its length and width.

    As ngspice selects a bin, a card that gives ``LMIN``, ``LMAX``, ``WMIN`` and
    ``WMAX`` holds an instance whose length lies between the first two and whose width
    between the others, on a bound counted in; its expressions are evaluated in
    `scope`.
    """
    bounds = [card.parameters.get(name) for name in ("lmin", "lmax", "wmin", "wmax")]
    level = card.parameters.get("level")
    if level is None or evaluate(level, scope) not in BSIM3_LEVELS:
        return False
    if any(bound is None for bound in bounds):
        return False
    lmin, lmax, wmin, wmax = (evaluate(bound, scope) for bound in bounds if bound)
    length, width = size
    return _within(length, lmin, lmax) and _within(width, wmin, wmax)


def _within(value: float, low: float, high: float) -> bool:
    """Whether `value` is between `low` and `high`, or within `BIN_TOLERANCE` of one."""
    return (
        abs(value - low) < BIN_TOLERANCE
        or abs(value - high) < BIN_TOLERANCE
        or low < value < high
    )


def _bsim3(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> dict[str, JsonValue]:
    """A BSIM3 transistor's parameters, from its model card and its instance's.

    Each parameter keeps its name, in SI units: a concentration per cubic meter, where
    the card's ``nch`` and ``ngate`` are per cubic centimeter up to ngspice's bounds and
    per cubic meter above them, and its ``nsub`` per cubic centimeter; a mobility in
    square meters per volt-second, where a card's ``u0`` above 1, and its dependences
    with it, are in square centimeters per volt-second; and, with ngspice's default
    ``binunit = 1``, each length and width dependence per micrometer, and each area
    dependence per square micrometer, of the effective length and width.

    Raises:
        SpiceError: When the card is not of BSIM3's version 3.3, which ngspice runs
            other models for, a non-quasi-static model or the area calculation is on,
            or a parameter has no jolty counterpart.
    """
    values.pop("level")
    own = _bsim3_off(card, values, dict(instance), statement)
    channel = card.kind[0]
    parameters: dict[str, JsonValue] = {"channel": channel}
    parameters |= _bsim3_card(card, values, statement)
    parameters |= _mapped(card, own, BSIM3_INSTANCE, statement)
    return parameters


def _bsim3_off(
    card: ModelCard,
    values: dict[str, float],
    instance: dict[str, float],
    statement: Statement,
) -> dict[str, float]:
    """A BSIM3 instance's parameters, once its card is checked of version 3.3.

    The models jolty has not, `BSIM3_OFF`, must be off in both: the card's are taken
    from `values`, the instance's from what it returns.

    Raises:
        SpiceError: When the card's version or one of the models is not.
    """
    version = card.version
    if version is not None and not version.startswith("3.3"):
        message = (
            f"`{card.name}` is a BSIM3 card of version {version}, whose equations "
            "ngspice runs another model for; jolty models version 3.3"
        )
        raise statement.fail(message)
    for name in BSIM3_OFF:
        for given in (values, instance):
            if given.pop(name, 0.0) != 0:
                message = f"jolty models BSIM3 with `{name.upper()}` 0 alone"
                raise statement.fail(message)
    instance.pop("geo", None)
    return instance


def _bsim3_card(
    card: ModelCard, values: dict[str, float], statement: Statement
) -> dict[str, JsonValue]:
    """A BSIM3 card's parameters but its level, in SI units.

    Raises:
        SpiceError: When a parameter has no jolty counterpart.
    """
    binned = 1e-6 if values.pop("binunit", 1.0) == 1 else 1.0
    mobility = values.get("u0", 0.067 if card.kind[0] == "n" else 0.025)
    # ngspice reads a binned u0 above 1 in cm²/(V·s), as its b3temp.c does.
    centimeters = 1e-4 if mobility > 1 else 1.0
    parameters: dict[str, JsonValue] = {}
    for name, value in values.items():
        if name in BSIM3_UNREAD or name in NOISE:
            continue
        if name in NOMINAL_TEMPERATURES:
            parameters["nominal_temperature"] = celsius(value).value
        elif name in BSIM3_SELECTORS:
            parameters[name] = _whole(card, name, value, statement)
        elif name in {"paramchk", "xpart"}:
            parameters[name] = value == 1 if name == "paramchk" else value
        elif name in BSIM3_CARD:
            parameters[name] = value * _bsim3_factor(name, value, centimeters)
        elif name[:1] in {"l", "w", "p"} and name[1:] in BSIM3_BINNED:
            extent = binned * binned if name[0] == "p" else binned
            factor = _bsim3_factor(name[1:], value, centimeters)
            parameters[name] = value * factor * extent
        else:
            raise unmatched(card, name, statement)
    return parameters


def _bsim3_factor(name: str, value: float, centimeters: float) -> float:
    """The factor that takes BSIM3's parameter `name`, or a dependence of it, to SI.

    A density's factor follows its `value`, and a mobility's is `centimeters`, the
    card's.
    """
    if name in BSIM3_DENSITIES:
        return 1.0 if value > BSIM3_DENSITIES[name] else 1e6
    if name == "nsub":
        return 1e6
    if name == "u0":
        return centimeters
    return 1.0


def _whole(card: ModelCard, name: str, value: float, statement: Statement) -> int:
    """A model selector's `value`, a whole number.

    Raises:
        SpiceError: When it is not one.
    """
    if not value.is_integer():
        message = f"`{name.upper()}` of `{card.name}` is a whole number, not {value!r}"
        raise statement.fail(message)
    return int(value)

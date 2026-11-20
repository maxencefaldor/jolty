# Copyright 2026 Maxence Faldor
"""SPICE active, source, switch, and subcircuit element translators."""

import itertools
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Final

from jolty import _core, ir
from jolty.spice import models
from jolty.spice._initial import GROUND, Built, remap_capacitor, remap_net
from jolty.spice._params import (
    Subcircuit,
    instance_keywords,
    split_assigned_pairs,
    split_keywords,
)
from jolty.spice._statistics import (
    check_binned_deterministic,
    record_waveform_targets,
    scale_mosfet_instance_symbolic,
)
from jolty.spice._survey import raise_unbinned
from jolty.spice._translate import ElementTranslator, Translation
from jolty.spice.expressions import (
    Binary,
    Call,
    Current,
    DualDraw,
    Expression,
    Number,
    Scalar,
    Voltage,
    evaluate,
    inlined,
    is_finite,
    number,
    primal,
)
from jolty.spice.models import DualJsonValue, ModelCard
from jolty.spice.netlist import Statement, tokens

_TRANSIENT_FUNCTIONS: Final = frozenset({"pulse", "sin", "pwl", "exp", "sffm", "am"})
"""The names of SPICE's waveforms of an independent source over time."""

_SINE_OPTIONS: Final = ("delay", "damping", "phase")
"""Optional `SIN` parameter names following offset, amplitude, and frequency."""

_SCALED: Final = {
    **dict.fromkeys(("w", "l", "pd", "ps", "sa", "sb", "sd", "sc"), 1),
    **dict.fromkeys(("ad", "as"), 2),
}
"""MOSFET instance parameters scaled by ``.option scale``, mapped to scale powers.

Lengths scale linearly (power 1) and areas scale quadratically (power 2), matching
ngspice ``mos1par.c`` and ``b4v5par.c`` (lines 87 to 111).
"""

_VDMOS_TERMINALS: Final = ("drain", "gate", "source")
"""VDMOS terminals in SPICE element order."""

_VDMOS_NODES: Final = 5
"""Maximum node count on a VDMOS statement (three electrical plus two thermal)."""

_BIPOLAR_TERMINALS: Final = ("collector", "base", "emitter")
"""Primary bipolar transistor terminals in SPICE element order before substrate."""

_TABLE_NUMBER: Final = re.compile(
    r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?[a-z]*", re.IGNORECASE
)
"""Pattern matching numbers with optional scale suffixes inside a ``TABLE`` source."""

_ROUNDING: Final = 0.1
"""Fraction of the shorter adjacent segment rounded at each ``TABLE`` corner, matching
XSPICE ``pwl`` ``input_domain``."""

_POLY_HEADER: Final = re.compile(r"poly\((\d+)\)", re.IGNORECASE)
"""Header of a SPICE2 ``POLY(n)`` controlled source."""

_GROUP_SEPARATOR: Final = re.compile(r"[\s,]+")
"""Whitespace and comma separator inside parenthesized ``POLY`` node groups."""

_ARBITRARY_SOURCE: Final = re.compile(
    r"\S+\s+(\S+)\s+(\S+)\s+([VI])\s*=\s*(.+)$", re.IGNORECASE
)
"""Pattern matching a B-source statement setting ``V=`` or ``I=`` an expression."""


def _read_instance_keywords[Seed](
    translation: Translation[Seed],
    statement: Statement,
    assigned: Mapping[str, list[str]],
    *,
    multi_valued: frozenset[str] = frozenset(),
) -> tuple[dict[str, Scalar[Seed]], dict[str, Expression]]:
    """Evaluates element keywords returning `(values, symbolic_map)`."""
    evaluated = instance_keywords(
        translation.value_symbolic,
        statement,
        assigned,
        multi_valued=multi_valued,
    )
    values: dict[str, Scalar[Seed]] = {}
    symbolic: dict[str, Expression] = {}
    for key, (resolved_value, expression) in evaluated.items():
        values[key] = resolved_value
        if expression is not None:
            symbolic[key] = expression
    return values, symbolic


def _check_source_waveform[Seed](
    waveform: dict[str, DualJsonValue[Seed]] | None,
    level: Scalar[Seed] | None,
    level_symbolic: Expression | None,
    statement: Statement,
) -> dict[str, DualJsonValue[Seed]]:
    """Combines optional DC `level` and time-domain `waveform`, checking consistency."""
    if waveform is None:
        return {"dc": 0.0 if level is None else level}
    if level is None:
        return waveform
    primal_waveform = models.split_dual_parameters(waveform, statement)[0]
    try:
        at_zero = _core.waveform_value(json.dumps(primal_waveform), 0.0)
    except ValueError as error:
        raise statement.fail(str(error)) from None
    if primal(level) != at_zero:
        message = (
            f"a DC level of {primal(level)!r} beside a waveform at {at_zero!r} at time "
            "zero: jolty's source has one level at zero for every analysis"
        )
        raise statement.fail(message)
    if level_symbolic is not None:
        message = (
            "statistical variation on a DC level beside a waveform changes "
            "time-zero consistency across Monte Carlo samples"
        )
        raise statement.fail(message)
    return waveform


def _independent(model: str) -> ElementTranslator:
    """Builds a translator for independent sources (`voltage_source`/`current_source`).

    Reads a DC level or a time-domain waveform (`PULSE`, `SIN`, `PWL`, or `EXP`).
    Non-transient analyses evaluate the waveform at time zero, matching ngspice when no
    separate DC level is given.
    """

    def translated[Seed](translation: Translation[Seed], statement: Statement) -> None:
        words = tokens(statement.text)
        if len(words) < 3:  # noqa: PLR2004 - a name and two nodes
            message = "a source joins two nodes"
            raise statement.fail(message)
        element_name, positive, negative, *rest = words
        rest, waveform, waveform_symbolic = _transient_function(
            translation, statement, rest
        )
        if rest[:1] and rest[0].casefold() == "dc":
            rest = rest[1:]
        level_entry = (
            translation.value_symbolic(rest[0])
            if rest and rest[0].casefold() != "ac"
            else None
        )
        level, level_symbolic = (None, None) if level_entry is None else level_entry
        after = rest[1:] if level_entry is not None else rest
        # A small-signal magnitude belongs to the plan, which excites a source by name.
        if after[:1] and after[0].casefold() == "ac":
            after = after[
                1 + sum(1 for word in after[1:3] if number(word) is not None) :
            ]
        if after:
            message = (
                "jolty reads a source's DC level and its PULSE, SIN, PWL or EXP "
                f"waveform only, not `{' '.join(after)}`"
            )
            raise statement.fail(message)
        resolved_waveform = _check_source_waveform(
            waveform, level, level_symbolic, statement
        )
        parameters: dict[str, DualJsonValue[Seed]] = {"waveform": resolved_waveform}
        translation.add_dual(
            statement,
            model,
            parameters,
            {
                "positive": translation.net(positive),
                "negative": translation.net(negative),
            },
        )
        if translation.statistics is not None:
            if waveform_symbolic is None:
                translation.statistics.add_target(
                    element_name, "waveform.dc", level_symbolic
                )
            else:
                waveform_kind, argument_expressions = waveform_symbolic
                record_waveform_targets(
                    translation.statistics,
                    element_name,
                    waveform_kind,
                    argument_expressions,
                )

    return translated


def _transient_function[Seed](
    translation: Translation[Seed], statement: Statement, words: list[str]
) -> tuple[
    list[str],
    dict[str, DualJsonValue[Seed]] | None,
    tuple[str, list[Expression | None]] | None,
]:
    """Extracts a SPICE time-domain waveform from `words`, returning remaining tokens.

    Supports ``PULSE(...)``, ``PULSE (...)``, and bare SPICE2 argument lists for
    ``PULSE``, ``SIN``, ``PWL``, and ``EXP``.

    Raises:
        SpiceError: When the waveform is unsupported (such as `SFFM`) or omits required
            parameters that SPICE derives from transient step or stop time.
    """
    for index, word in enumerate(words):
        name, _, inside = word.partition("(")
        if name.casefold() not in _TRANSIENT_FUNCTIONS:
            continue
        if inside:
            arguments, rest = tokens(word[len(name) + 1 : -1]), words[index + 1 :]
        elif index + 1 < len(words) and words[index + 1].startswith("("):
            arguments, rest = tokens(words[index + 1][1:-1]), words[index + 2 :]
        else:
            arguments, rest = words[index + 1 :], []
        evaluated = [translation.value_symbolic(each) for each in arguments]
        values = [value for value, _ in evaluated]
        expressions = [expression for _, expression in evaluated]
        waveform_kind = name.casefold()
        waveform = _waveform(waveform_kind, values, statement)
        return [*words[:index], *rest], waveform, (waveform_kind, expressions)
    return words, None, None


def _waveform[Seed](
    name: str, arguments: list[Scalar[Seed]], statement: Statement
) -> dict[str, DualJsonValue[Seed]]:
    """Converts SPICE waveform `name` and numeric `arguments` into a jolty dict.

    Raises:
        SpiceError: When `name` is unsupported or omits parameters that SPICE takes
            from transient step/stop time (pulse rise/fall/width, sine frequency, or
            exponential delays and time constants).
    """
    match name, arguments:
        case "pulse", [initial, pulsed, delay, rise, fall, width, *period] if (
            primal(rise) > 0
            and primal(fall) > 0
            and primal(width) > 0
            and len(period) <= 1
        ):
            pulse: dict[str, DualJsonValue[Seed]] = {
                "initial": initial,
                "pulsed": pulsed,
                "delay": delay,
                "rise": rise,
                "fall": fall,
                "width": width,
            }
            if period:
                pulse["period"] = period[0]
            return {"pulse": pulse}
        case "pulse", _:
            message = (
                "a PULSE without a positive rise, fall and width, which SPICE takes "
                "from the transient analysis's step and stop, and jolty's circuit "
                "from no analysis"
            )
            raise statement.fail(message)
        case "sin", [offset, amplitude, frequency, *more] if primal(
            frequency
        ) > 0 and len(more) <= len(_SINE_OPTIONS):
            delay, damping, degrees = [*more, 0.0, 0.0, 0.0][: len(_SINE_OPTIONS)]
            return {
                "sine": {
                    "offset": offset,
                    "amplitude": amplitude,
                    "frequency": frequency,
                    "delay": delay,
                    "damping": damping,
                    "phase": degrees * (math.pi / 180.0),
                }
            }
        case "sin", _:
            message = (
                "a SIN without a positive frequency, which SPICE takes from the "
                "transient analysis's stop, and jolty's circuit from no analysis"
            )
            raise statement.fail(message)
        case "exp", [initial, pulsed, rise_delay, rise, fall_delay, fall] if (
            min(
                primal(rise_delay),
                primal(rise),
                primal(fall_delay),
                primal(fall),
            )
            > 0
        ):
            return {
                "exponential": {
                    "initial": initial,
                    "pulsed": pulsed,
                    "rise_delay": rise_delay,
                    "rise_time_constant": rise,
                    "fall_delay": fall_delay,
                    "fall_time_constant": fall,
                }
            }
        case "exp", _:
            message = (
                "an EXP without a positive rise delay, rise time constant, fall delay "
                "and fall time constant, each of which ngspice 47 takes from the "
                "transient analysis's step where it is 0 or left out, and jolty's "
                "circuit from no analysis"
            )
            raise statement.fail(message)
        case "pwl", [*pairs] if pairs and len(pairs) % 2 == 0:
            points: list[DualJsonValue[Seed]] = [
                [pairs[index], pairs[index + 1]] for index in range(0, len(pairs), 2)
            ]
            return {"piecewise": {"points": points}}
    message = f"jolty has no `{name.upper()}` waveform of these arguments"
    raise statement.fail(message)


def _scaled_mosfet[Seed](
    instance: dict[str, Scalar[Seed]], scale: float, statement: Statement
) -> dict[str, Scalar[Seed]]:
    """Scales MOSFET instance sizes by `scale`, raising `SpiceError` on overflow."""
    try:
        scaled = {
            key: value * math.pow(scale, _SCALED.get(key, 0))
            for key, value in instance.items()
        }
    except OverflowError, ValueError:
        message = f"scale option overflows: {scale!r}"
        raise statement.fail(message) from None
    if any(not is_finite(value) for value in scaled.values()):
        message = f"scale option overflows: {scale!r}"
        raise statement.fail(message)
    return scaled


def _resolve_semiconductor_card[Seed](  # noqa: PLR0913 - model lookup and bin check
    translation: Translation[Seed],
    model_name: str,
    kinds: frozenset[str],
    *,
    instance: Mapping[str, Scalar[Seed]],
    instance_symbolic: Mapping[str, Expression],
    statement: Statement,
) -> ModelCard:
    """Resolves a semiconductor model card (including binned MOSFET lookup)."""
    library, cards = translation.library, translation.models
    card = library.model(model_name, cards)
    binned = False
    if card is None and "nmos" in kinds:
        binned = True
        primal_instance = {key: primal(value) for key, value in instance.items()}
        matched = library.binned_with_bounds(
            model_name,
            cards,
            primal_instance,
            translation.primal_scope,
            functions=translation.functions,
        )
        if matched is None:
            first_bin = library.card(model_name, cards)
            middle = (
                None
                if first_bin is None
                else models.center(
                    first_bin,
                    translation.primal_scope,
                    functions=translation.functions,
                )
            )
            raise_unbinned(
                statement,
                model_name,
                primal_instance,
                middle,
                library.fingers(primal_instance),
            )
        else:
            card, bin_bounds = matched
            translation.record_bin(statement.text.split()[0], bin_bounds)
    if card is None:
        message = f"no model is named `{model_name}`"
        raise statement.fail(message)
    if card.kind not in kinds:
        message = f"`{card.name}` is a model of type `{card.kind}`"
        raise statement.fail(message)
    if binned and translation.statistics is not None:
        check_binned_deterministic(
            card,
            instance_symbolic,
            translation.statistics,
            scope=translation.primal_scope,
            functions=translation.functions,
            statement=statement,
        )
    return card


def _semiconductor(
    terminals: tuple[str, ...], kinds: frozenset[str]
) -> ElementTranslator:
    """Builds a translator for semiconductor devices connecting `terminals`.

    Validates that the referenced model card has a type in `kinds`.
    """

    def translated[Seed](translation: Translation[Seed], statement: Statement) -> None:
        positional, assigned = split_keywords(tokens(statement.text))
        nodes = positional[1 : 1 + len(terminals)]
        rest = positional[1 + len(terminals) :]
        if len(nodes) != len(terminals) or not rest:
            message = f"the element joins {len(terminals)} nodes, then names its model"
            raise statement.fail(message)
        raw_instance, raw_symbolic = _read_instance_keywords(
            translation, statement, assigned
        )
        instance, instance_symbolic = translation.multiplied_symbolic(
            raw_instance, raw_symbolic
        )
        scale_option = translation.library.options.get("scale", 1.0)
        if "nmos" in kinds:
            instance = _scaled_mosfet(instance, scale_option, statement)
            instance_symbolic = scale_mosfet_instance_symbolic(
                instance_symbolic, scale_option
            )
        elif "d" in kinds:
            # Level-3 diodes scale instance dimensions by `scale` while keeping mask
            # offsets in native units (`diosetup.c`, lines 336 to 360).
            instance["scale"] = scale_option
        card = _resolve_semiconductor_card(
            translation,
            rest[0],
            kinds,
            instance=instance,
            instance_symbolic=instance_symbolic,
            statement=statement,
        )
        # An initial state, `OFF`, only guides the search for the operating point.
        areas = [word for word in rest[1:] if word.casefold() not in {"off", "on"}]
        if areas:
            area_value, area_symbolic = translation.value_symbolic(areas[0])
            instance["area"] = area_value
            if area_symbolic is not None:
                instance_symbolic["area"] = area_symbolic
        made = translation.device(
            card, instance, statement, instance_symbolic=instance_symbolic
        )
        joined = {
            terminal: translation.net(node)
            for terminal, node in zip(terminals, nodes, strict=True)
        }
        translation.add(
            statement,
            made.model,
            made.parameters,
            joined,
            derivatives=made.derivatives,
        )

    return translated


_MOSFET: Final = _semiconductor(
    ("drain", "gate", "source", "bulk"), frozenset({"nmos", "pmos"})
)
"""The translation of a MOSFET element (`M`)."""


def _find_model_card[Seed](
    translation: Translation[Seed], positional: Sequence[str], first: int
) -> tuple[int | None, ModelCard | None]:
    """Finds the first token at or after `first` in `positional` naming a model card."""
    for position in range(first, len(positional)):
        card = translation.library.model(positional[position], translation.models)
        if card is not None:
            return position, card
    return None, None


def _check_vdmos_flags(statement: Statement, words: Sequence[str]) -> None:
    """Checks trailing tokens on a VDMOS statement, allowing only ``OFF``."""
    for word in words:
        if word.casefold() == "thermal":
            message = (
                "jolty reads no `THERMAL` VDMOS: a vdmos heats through a thermal "
                "terminal of its own, joined to a thermal network"
            )
            raise statement.fail(message)
        # An initial state, `OFF`, only guides the search for the operating point.
        if word.casefold() != "off":
            message = f"a VDMOS takes no `{word}`"
            raise statement.fail(message)


def _transistor[Seed](translation: Translation[Seed], statement: Statement) -> None:
    """Translates an `M` element into a MOSFET or VDMOS device by its model card.

    Follows ngspice model lookup starting at the fourth positional token. MOSFETs
    connect four terminals, while VDMOS devices connect drain, gate, and source plus
    up to two unused thermal nodes (self-heating ``thermal`` VDMOS instances are
    rejected). VDMOS instances accept multiplier ``M`` and no area.

    Raises:
        SpiceError: When no valid model follows the nodes or an unsupported option is
            supplied.
    """
    positional, assigned = split_keywords(tokens(statement.text))
    named, card = _find_model_card(translation, positional, len(_VDMOS_TERMINALS) + 1)
    if named is None or card is None or card.kind not in models.VDMOS_KINDS:
        _MOSFET(translation, statement)
        return
    nodes, rest = positional[1:named], positional[named + 1 :]
    if len(nodes) > _VDMOS_NODES:
        message = f"a VDMOS joins at most {_VDMOS_NODES} nodes, then names its model"
        raise statement.fail(message)
    _check_vdmos_flags(statement, rest)
    raw_instance: dict[str, Scalar[Seed]] = {}
    raw_symbolic: dict[str, Expression] = {}
    for key, values in assigned.items():
        if not values:
            message = f"`{key.upper()}` is not assigned a value"
            raise statement.fail(message)
        _check_vdmos_flags(statement, values[1:])
        resolved_value, symbolic = translation.value_symbolic(values[0])
        raw_instance[key] = resolved_value
        if symbolic is not None:
            raw_symbolic[key] = symbolic
    instance, instance_symbolic = translation.multiplied_symbolic(
        raw_instance, raw_symbolic
    )
    made = translation.device(
        card, instance, statement, instance_symbolic=instance_symbolic
    )
    joined = {
        terminal: translation.net(node)
        for terminal, node in zip(_VDMOS_TERMINALS, nodes, strict=False)
    }
    translation.add(
        statement,
        made.model,
        made.parameters,
        joined,
        derivatives=made.derivatives,
    )


def _vbic_heating[Seed](
    translation: Translation[Seed],
    card: ModelCard,
    instance: Mapping[str, Scalar[Seed]],
) -> tuple[Scalar[Seed], Scalar[Seed]] | None:
    """Evaluates 5-node VBIC self-heating `(rth, cth)` in plain or dual mode."""
    if translation.differentiating:
        prefix = card.name.casefold()

        def draws(parameter: str) -> DualDraw[Seed] | None:
            return translation.dual_draws_for(f"{prefix}.{parameter}")

        return models.vbic_heating_dual(
            card,
            translation.scope,
            instance,
            draws,
            functions=translation.functions,
        )
    primal_instance = {key: primal(value) for key, value in instance.items()}
    return models.vbic_heating(
        card,
        translation.primal_scope,
        primal_instance,
        functions=translation.functions,
    )


def _bipolar[Seed](translation: Translation[Seed], statement: Statement) -> None:
    """Translates a `Q` element into a Gummel-Poon BJT or VBIC device by model card.

    Locates the model name starting at the fourth positional token after collector,
    base, emitter, optional substrate (defaulting to ground), and optional VBIC fifth
    thermal node. When a five-node VBIC card enables self-heating (``RTH > 0`` and
    ``SELFT`` omitted or 1), attaches a thermal terminal alongside ``.rth`` and
    ``.cth`` devices to ambient ground.

    Raises:
        SpiceError: When no bipolar model card follows three to five node tokens.
    """
    positional, assigned = split_keywords(tokens(statement.text))
    named, card = _find_model_card(translation, positional, len(_BIPOLAR_TERMINALS) + 1)
    if named is None or card is None:
        message = "the element joins three to five nodes, then names its model"
        raise statement.fail(message)
    if card.kind not in {"npn", "pnp"}:
        message = f"`{card.name}` is a model of type `{card.kind}`"
        raise statement.fail(message)
    nodes, rest = positional[1:named], positional[named + 1 :]
    raw_instance, raw_symbolic = _read_instance_keywords(
        translation,
        statement,
        assigned,
        multi_valued=frozenset({"ic"}),
    )
    instance, instance_symbolic = translation.multiplied_symbolic(
        raw_instance, raw_symbolic
    )
    # An initial state, `OFF`, only guides the search for the operating point.
    areas = [word for word in rest if word.casefold() not in {"off", "on"}]
    if areas:
        area_value, area_symbolic = translation.value_symbolic(areas[0])
        instance["area"] = area_value
        if area_symbolic is not None:
            instance_symbolic["area"] = area_symbolic
    made = translation.device(
        card, instance, statement, instance_symbolic=instance_symbolic
    )
    most = 5 if made.model == "vbic" else 4
    if len(nodes) > most:
        message = f"the transistor joins at most {most} nodes, then names its model"
        raise statement.fail(message)
    collector, base, emitter, *more = nodes
    substrate = more[0] if more else "0"
    joined = {
        "collector": translation.net(collector),
        "base": translation.net(base),
        "emitter": translation.net(emitter),
        "substrate": translation.net(substrate),
    }
    five_node_vbic = len(nodes) == most == 5  # noqa: PLR2004 - VBIC thermal node
    heating = _vbic_heating(translation, card, instance) if five_node_vbic else None
    if heating is None:
        translation.add(
            statement,
            made.model,
            made.parameters,
            joined,
            derivatives=made.derivatives,
        )
        if five_node_vbic:
            translation.record_vbic_heating(
                card, instance, instance_symbolic, statement, heated=False
            )
        return
    thermal = translation.net(nodes[4])
    joined["thermal"] = thermal
    translation.add(
        statement,
        made.model,
        {**made.parameters, "thermal": True},
        joined,
        derivatives=made.derivatives,
    )
    resistance, capacitance = heating
    name = statement.text.split()[0]
    parts: tuple[tuple[str, str, dict[str, DualJsonValue[Seed]]], ...] = (
        ("rth", "thermal_resistance", {"resistance": resistance}),
        ("cth", "thermal_capacitance", {"capacitance": capacitance}),
    )
    between = {"a": thermal, "b": translation.net("0")}
    for suffix, model, parameters in parts:
        translation.add_dual(
            statement, model, parameters, between, identifier=f"{name}.{suffix}"
        )
    translation.record_vbic_heating(
        card, instance, instance_symbolic, statement, heated=True
    )


def _controlled(output: str) -> ElementTranslator:
    """Builds a translator for controlled sources (`E`, `F`, `G`, `H`) by `output`."""

    def translated[Seed](translation: Translation[Seed], statement: Statement) -> None:
        words = tokens(statement.text)
        if len(words) < 4:  # noqa: PLR2004 - a name, two nodes and a control
            message = "a controlled source joins two nodes, then names its control"
            raise statement.fail(message)
        element_letter = words[0][0].casefold()
        positive, negative, rest = words[1], words[2], words[3:]
        first = rest[0].casefold()
        by_current = element_letter in {"f", "h"}
        if first.startswith("poly("):
            expression = _polynomial(statement, rest, by_current=by_current)
        elif first == "value" and not by_current:
            index = 2 if rest[1:2] == ["="] else 1
            if len(rest) <= index:
                message = "a VALUE source names an expression"
                raise statement.fail(message)
            expression = translation.parse(rest[index])
        elif first == "table" and not by_current:
            expression = _table(translation, statement, rest)
        elif by_current and len(rest) == 2:  # noqa: PLR2004 - a source and a gain
            _linear_current(translation, statement, (positive, negative), rest, output)
            return
        elif not by_current and len(rest) == 3:  # noqa: PLR2004 - two nodes and a gain
            control_positive, control_negative, gain = rest
            parameter = "gain" if output == "voltage" else "transconductance"
            gain_value, gain_symbolic = translation.value_symbolic(gain)
            translation.add_dual(
                statement,
                f"voltage_controlled_{output}_source",
                {parameter: gain_value},
                {
                    "positive": translation.net(positive),
                    "negative": translation.net(negative),
                    "control_positive": translation.net(control_positive),
                    "control_negative": translation.net(control_negative),
                },
            )
            if translation.statistics is not None:
                translation.statistics.add_target(words[0], parameter, gain_symbolic)
            return
        else:
            message = "jolty reads a controlled source's gain, POLY, VALUE or TABLE"
            raise statement.fail(message)
        model = f"behavioral_{output}_source"
        translation.add_behavioral(statement, model, (positive, negative), expression)

    return translated


def _linear_current[Seed](
    translation: Translation[Seed],
    statement: Statement,
    nodes: tuple[str, str],
    rest: list[str],
    output: str,
) -> None:
    """Translates a linear current-controlled source (`F` or `H`) with scalar gain.

    Inserts the control port in series with the sensed voltage source.
    """
    source, gain = rest
    control_positive, control_negative = translation.sense(source, statement)
    parameter = "gain" if output == "current" else "transresistance"
    gain_value, gain_symbolic = translation.value_symbolic(gain)
    translation.add_dual(
        statement,
        f"current_controlled_{output}_source",
        {parameter: gain_value},
        {
            "positive": translation.net(nodes[0]),
            "negative": translation.net(nodes[1]),
            "control_positive": control_positive,
            "control_negative": control_negative,
        },
    )
    if translation.statistics is not None:
        translation.statistics.add_target(
            statement.text.split()[0], parameter, gain_symbolic
        )


def _table[Seed](
    translation: Translation[Seed], statement: Statement, rest: list[str]
) -> Expression:
    """Parses a SPICE ``TABLE {x} = (x1, y1) ...`` lookup table into a call node.

    Interpolates table points linearly, clamps beyond endpoints, and rounds corners
    with `_ROUNDING`. Parses coordinates leniently to tolerate unbalanced PSpice
    parentheses.
    """
    if len(rest) < 2:  # noqa: PLR2004 - the keyword and the input
        message = "a TABLE has an input and points"
        raise statement.fail(message)
    written = [number(word) for word in _TABLE_NUMBER.findall(" ".join(rest[2:]))]
    if not written or len(written) % 2 or any(value is None for value in written):
        message = "a TABLE's points come in pairs of numbers"
        raise statement.fail(message)
    points = [Number(value) for value in written if value is not None]
    return Call(
        "rounded_table", (translation.parse(rest[1]), Number(_ROUNDING), *points)
    )


def _polynomial_count(statement: Statement, header: str) -> int:
    """Parses the control count of ``POLY(n)`` and checks it is at least 1."""
    match = _POLY_HEADER.fullmatch(header)
    if match is None:
        message = "POLY counts its controls, as POLY(2)"
        raise statement.fail(message)
    count = int(match[1])
    if count < 1:
        message = f"POLY counts one or more controls, not {count}"
        raise statement.fail(message)
    return count


def _polynomial(
    statement: Statement, rest: list[str], *, by_current: bool
) -> Expression:
    """Parses a SPICE2 ``POLY(n)`` controlled source into a polynomial `Expression`.

    Orders coefficients from constant term through linear, quadratic, and higher
    degree control monomials.
    """
    count = _polynomial_count(statement, rest[0])
    words = [part for word in rest[1:] for part in _split_group(word)]
    # A control is a source, or a pair of nodes.
    width = 1 if by_current else 2
    if len(words) <= width * count:
        message = f"POLY({count}) reads {count} controls, then numbers"
        raise statement.fail(message)
    written = [words[width * index : width * (index + 1)] for index in range(count)]
    values = [number(word) for word in words[width * count :]]
    if any(value is None for value in values):
        message = f"POLY({count}) reads {count} controls, then numbers"
        raise statement.fail(message)
    controls: list[Expression] = [
        Current(*control) if by_current else Voltage(*control) for control in written
    ]
    numbers = [value for value in values if value is not None]
    # One coefficient of a one-dimensional polynomial is its gain, as in SPICE.
    if count == 1 and len(numbers) == 1:
        numbers = [0.0, numbers[0]]
    monomials = itertools.chain.from_iterable(
        itertools.combinations_with_replacement(range(count), degree)
        for degree in itertools.count()
    )
    terms: list[Expression] = []
    for coefficient, monomial in zip(numbers, monomials, strict=False):
        if coefficient == 0:
            continue
        term: Expression = Number(coefficient)
        for control in monomial:
            term = Binary("*", term, controls[control])
        terms.append(term)
    if not terms:
        return Number(0.0)
    total = terms[0]
    for term in terms[1:]:
        total = Binary("+", total, term)
    return total


def _split_group(word: str) -> list[str]:
    """Splits a parenthesized ``POLY`` group such as ``(3,0)`` into tokens."""
    if word.startswith("(") and word.endswith(")"):
        return [part for part in _GROUP_SEPARATOR.split(word[1:-1]) if part]
    return [word]


def _arbitrary[Seed](translation: Translation[Seed], statement: Statement) -> None:
    """Translates an arbitrary behavioral `B` source (`V=` or `I=`) into jolty IR."""
    match = _ARBITRARY_SOURCE.match(statement.text)
    if match is None:
        message = "a B source joins two nodes, then sets `V=` or `I=` an expression"
        raise statement.fail(message)
    positive, negative, kind, written = match.groups()
    output = "voltage" if kind.casefold() == "v" else "current"
    if output == "current" and (
        primal(translation.multiplier) != 1.0
        or translation.multiplier_symbolic is not None
    ):
        message = (
            "jolty multiplies no behavioral current by its subcircuit's multiplier "
            f"`m`, here {primal(translation.multiplier)!r}"
        )
        raise statement.fail(message)
    expression = translation.parse(written)
    model = f"behavioral_{output}_source"
    translation.add_behavioral(statement, model, (positive, negative), expression)


def _switch(*, by_current: bool) -> ElementTranslator:
    """Builds a translator for voltage-controlled (`S`) or current (`W`) switches."""
    expected_words = 5 if by_current else 6
    expected_kind = "iswitch" if by_current else "vswitch"

    def translated[Seed](translation: Translation[Seed], statement: Statement) -> None:
        words = [
            word
            for word in tokens(statement.text)
            if word.casefold() not in {"on", "off"}
        ]
        if len(words) != expected_words:
            message = (
                "a current-controlled switch joins two nodes, then names its source "
                "and its model"
                if by_current
                else "a switch joins four nodes, then names its model"
            )
            raise statement.fail(message)
        positive, negative, name = words[1], words[2], words[-1]
        card = translation.library.model(name, translation.models)
        if card is None:
            message = f"no model is named `{name}`"
            raise statement.fail(message)
        if card.kind != expected_kind:
            message = f"`{card.name}` is a model of type `{card.kind}`"
            raise statement.fail(message)
        made = translation.device(card, {}, statement)
        if by_current:
            control_positive, control_negative = translation.sense(words[3], statement)
            terminals = {
                "positive": translation.net(positive),
                "negative": translation.net(negative),
                "control_positive": control_positive,
                "control_negative": control_negative,
            }
        else:
            terminals = {
                "positive": translation.net(positive),
                "negative": translation.net(negative),
                "control_positive": translation.net(words[3]),
                "control_negative": translation.net(words[4]),
            }
        translation.add(
            statement,
            made.model,
            made.parameters,
            terminals,
            derivatives=made.derivatives,
        )

    return translated


def _instance_arguments[Seed](
    translation: Translation[Seed], statement: Statement, assigned: list[str]
) -> tuple[dict[str, float], dict[str, Expression], frozenset[str]]:
    """Evaluates `X` instance arguments with symbolic and probe-stochastic metadata."""
    functions = translation.functions
    scope = translation.primal_scope
    raw_given: dict[str, float] = {}
    raw_given_symbolic: dict[str, Expression] = {}
    probe_stochastic: set[str] = set()
    for parameter, expression in models.assignments(statement, assigned).items():
        resolved = inlined(expression, functions)
        raw_given[parameter] = evaluate(resolved, scope, translation.draw)
        if translation.statistics is not None:
            lowered = translation.statistics.lower_element(resolved, scope)
            if lowered is not None:
                raw_given_symbolic[parameter] = lowered
        elif translation.probe_compiler is not None and (
            translation.probe_compiler.depends_on_statistics(resolved)
        ):
            probe_stochastic.add(parameter)
    given, given_symbolic = translation.multiplied_symbolic(
        raw_given, raw_given_symbolic
    )
    primal_given = {key: primal(value) for key, value in given.items()}
    if translation.probe_compiler is not None and (
        translation.multiplier_symbolic is not None
    ):
        probe_stochastic.add("m")
    return primal_given, given_symbolic, frozenset(probe_stochastic)


def _place_child[Seed](
    translation: Translation[Seed],
    statement: Statement,
    subcircuit: Subcircuit,
    name: str,
    assigned: list[str],
) -> Built:
    """Builds child `subcircuit` placement in dual, hoisted, or standard mode."""
    prefix = f"{name}{ir.SEPARATOR}"
    if translation.differentiating:
        given = translation.multiplied(
            {
                parameter: translation.evaluate_expression(expression)
                for parameter, expression in models.assignments(
                    statement, assigned
                ).items()
            }
        )
        dual_built = translation.library.place_dual(
            subcircuit,
            translation.globals,
            given,
            translation.drawing.within(name),
            translation.instantiating,
            dual_draws=translation.dual_draw_factory,
            temperature=translation.temperature_value,
            temperature_dependent=translation.global_temperature_dependent,
        )
        for device_path, parameters in dual_built.derivatives.items():
            translation.device_derivatives[f"{prefix}{device_path}"] = {
                parameter: dict(slopes) for parameter, slopes in parameters.items()
            }
        for device_path, bin_bounds in dual_built.bins.items():
            translation.device_bins[f"{prefix}{device_path}"] = bin_bounds
        return dual_built.built
    primal_given, given_symbolic, probe_stochastic = _instance_arguments(
        translation, statement, assigned
    )
    if translation.statistics is not None and given_symbolic:
        return translation.library.place_built(
            subcircuit,
            translation.globals,
            primal_given,
            translation.drawing.within(name),
            translation.instantiating,
            uses=translation.uses,
            temperature=translation.temperature_value,
            temperature_dependent=translation.global_temperature_dependent,
            given_symbolic=given_symbolic,
            parent_statistics=translation.statistics,
            hoist_instance=name,
        )
    return translation.library.place_built(
        subcircuit,
        translation.globals,
        primal_given,
        translation.drawing.within(name),
        translation.instantiating,
        uses=translation.uses,
        temperature=translation.temperature_value,
        temperature_dependent=translation.global_temperature_dependent,
        given_probe_stochastic=probe_stochastic,
    )


def _instance[Seed](translation: Translation[Seed], statement: Statement) -> None:
    """Translates an `X` subcircuit instantiation statement with its parameters."""
    words = [word for word in tokens(statement.text) if word.casefold() != "params:"]
    # Each parameter is `name = value`; the subcircuit's name is the last word besides,
    # which HSPICE's netlists, such as SKY130's, may write after the parameters.
    positional, assigned = split_assigned_pairs(words)
    if len(positional) < 2:  # noqa: PLR2004 - a name and a subcircuit
        message = "an instance names its subcircuit"
        raise statement.fail(message)
    name, *nodes, target = positional
    subcircuit = translation.library.subcircuit(target)
    if subcircuit is None:
        message = f"no subcircuit is named `{target}`"
        raise statement.fail(message)
    if len(nodes) != len(subcircuit.pins):
        message = (
            f"`{subcircuit.name}` has {len(subcircuit.pins)} pins, not {len(nodes)}"
        )
        raise statement.fail(message)
    prefix = f"{name}{ir.SEPARATOR}"
    built = _place_child(translation, statement, subcircuit, name, assigned)
    if built.temperature_dependent:
        translation.mark_temperature_dependent()
    child = built.component
    ports = [port.name for port in child.definition.ports]
    terminals = {
        port: translation.net(node) for port, node in zip(ports, nodes, strict=False)
    }
    if GROUND in ports:
        terminals[GROUND] = translation.net("0")
    translation.child_initial_conditions.extend(
        (remap_net(net, terminals, prefix), voltage)
        for net, voltage in built.initial.initial_conditions
    )
    translation.child_nodesets.extend(
        (remap_net(net, terminals, prefix), voltage)
        for net, voltage in built.initial.nodesets
    )
    translation.capacitor_initial_voltages.extend(
        remap_capacitor(capacitor, terminals, prefix)
        for capacitor in built.initial.capacitors
    )
    translation.child_initial_currents.extend(
        (f"{prefix}{unknown_name}", current)
        for unknown_name, current in built.initial.inductors
    )
    translation.children[child.name] = child
    translation.instances.append(
        (
            translation.order,
            ir.Instance(
                name=name,
                of=ir.ComponentOf(component=child.name),
                terminals=terminals,
                source_location=ir.SourceLocation(
                    file=statement.file, line=statement.line
                ),
            ),
        )
    )


ACTIVE_ELEMENTS: Final[dict[str, ElementTranslator]] = {
    "v": _independent("voltage_source"),
    "i": _independent("current_source"),
    "d": _semiconductor(("anode", "cathode"), frozenset({"d"})),
    "q": _bipolar,
    "j": _semiconductor(("drain", "gate", "source"), frozenset({"njf", "pjf"})),
    "m": _transistor,
    "e": _controlled("voltage"),
    "f": _controlled("current"),
    "g": _controlled("current"),
    "h": _controlled("voltage"),
    "b": _arbitrary,
    "s": _switch(by_current=False),
    "w": _switch(by_current=True),
    "x": _instance,
}
"""Active, source, switch, and subcircuit element translators keyed by letter."""

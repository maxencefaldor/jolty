# Copyright 2026 Maxence Faldor
"""SPICE active, source, switch, and subcircuit element translators."""

import itertools
import json
import math
import re
from collections.abc import Sequence
from typing import Final

from pydantic import JsonValue

from jolty import _core, ir
from jolty.spice import models
from jolty.spice._initial import GROUND, remap_capacitor, remap_net
from jolty.spice._params import (
    instance_keywords,
    split_assigned_pairs,
    split_keywords,
)
from jolty.spice._survey import raise_unbinned
from jolty.spice._translate import ElementTranslator, Translation
from jolty.spice.expressions import (
    Binary,
    Call,
    Current,
    Expression,
    Number,
    Voltage,
    evaluate,
    inlined,
    number,
)
from jolty.spice.models import ModelCard
from jolty.spice.netlist import Statement, tokens

_TRANSIENT_FUNCTIONS: Final = frozenset({"pulse", "sin", "pwl", "exp", "sffm", "am"})
"""The names of SPICE's waveforms of an independent source over time."""

_SINE_OPTIONS: Final = ("delay", "damping", "phase")
"""What a `SIN` may give after its offset, amplitude and frequency, in order."""

_SCALED: Final = {
    **dict.fromkeys(("w", "l", "pd", "ps", "sa", "sb", "sd", "sc"), 1),
    **dict.fromkeys(("ad", "as"), 2),
}
"""The sizes of a MOSFET's element that ngspice's ``scale`` option scales, by the power
of it each takes: a length's 1, an area's 2 (``mos1par.c``), and a BSIM4's distances to
its diffusion's edges, between its fingers and to its well's edge (``b4v5par.c``, lines
87 to 111)."""

_VDMOS_TERMINALS: Final = ("drain", "gate", "source")
"""A VDMOS's terminals, in its element's order."""

_VDMOS_NODES: Final = 5
"""The most nodes a VDMOS joins: its terminals, then a thermal junction and case, which
join nothing unless it heats."""

_BIPOLAR_TERMINALS: Final = ("collector", "base", "emitter")
"""A bipolar transistor's terminals, in its element's order, before its substrate."""

_TABLE_NUMBER: Final = re.compile(
    r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?[a-z]*", re.IGNORECASE
)
"""Pattern matching numbers with optional scale suffixes inside a ``TABLE`` source."""

_ROUNDING: Final = 0.1
"""How much of the shorter segment beside each corner of a ``TABLE`` source ngspice
rounds: its XSPICE ``pwl`` model's ``input_domain``."""

_POLY_HEADER: Final = re.compile(r"poly\((\d+)\)", re.IGNORECASE)
"""Header of a SPICE2 ``POLY(n)`` controlled source."""

_GROUP_SEPARATOR: Final = re.compile(r"[\s,]+")
"""Whitespace and comma separator inside parenthesized ``POLY`` node groups."""

_ARBITRARY_SOURCE: Final = re.compile(
    r"\S+\s+(\S+)\s+(\S+)\s+([VI])\s*=\s*(.+)$", re.IGNORECASE
)
"""Pattern matching a B-source statement setting ``V=`` or ``I=`` an expression."""


def _independent(model: str) -> ElementTranslator:
    """The translation of an independent source of `model`.

    Its level is its DC level, or its waveform over time, SPICE's `PULSE`, `SIN`, `PWL`
    or `EXP`, whose level at time zero every other analysis reads, as ngspice reads it
    for a source with no DC level.
    """

    def translated(translation: Translation, statement: Statement) -> None:
        words = tokens(statement.text)
        if len(words) < 3:  # noqa: PLR2004 - a name and two nodes
            message = "a source joins two nodes"
            raise statement.fail(message)
        _, positive, negative, *rest = words
        rest, waveform = _transient_function(translation, statement, rest)
        given = rest[:1] and rest[0].casefold() == "dc"
        if given:
            rest = rest[1:]
        level = (
            translation.value(rest[0]) if rest and rest[0].casefold() != "ac" else None
        )
        after = rest[1:] if rest and rest[0].casefold() != "ac" else rest
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
        if waveform is None:
            waveform = {"dc": 0.0 if level is None else level}
        elif level is not None:
            try:
                at_zero = _core.level(json.dumps(waveform), 0.0)
            except ValueError as error:
                raise statement.fail(str(error)) from None
            if level != at_zero:
                message = (
                    f"a DC level of {level!r} beside a waveform at {at_zero!r} at time "
                    "zero: jolty's source has one level at zero for every analysis"
                )
                raise statement.fail(message)
        parameters: dict[str, JsonValue] = {"waveform": waveform}
        translation.add(
            statement,
            model,
            parameters,
            {
                "positive": translation.net(positive),
                "negative": translation.net(negative),
            },
        )

    return translated


def _transient_function(
    translation: Translation, statement: Statement, words: list[str]
) -> tuple[list[str], dict[str, JsonValue] | None]:
    """`words` without a waveform over time, and that waveform, if they hold one.

    The waveform is written ``PULSE(...)`` or ``PULSE (...)``, or as SPICE2 wrote it,
    ``PULSE`` followed by its arguments to the statement's end, and likewise ``SIN``,
    ``PWL`` and ``EXP``.

    Raises:
        SpiceError: When the waveform is one jolty has not, such as `SFFM`, or leaves
            out what SPICE takes from the transient analysis.
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
        values = [translation.value(each) for each in arguments]
        waveform = _waveform(name.casefold(), values, statement)
        return [*words[:index], *rest], waveform
    return words, None


def _waveform(
    name: str, arguments: list[float], statement: Statement
) -> dict[str, JsonValue]:
    """The waveform SPICE's function `name` of `arguments` writes, in jolty's format.

    Raises:
        SpiceError: When jolty has no such waveform, or the function leaves out what
            SPICE takes from the transient analysis: a pulse's rise, fall or width, a
            sine's frequency, or an exponential's delays and time constants, which
            ngspice 47 takes from the transient's step wherever they are 0.
    """
    match name, arguments:
        case "pulse", [initial, pulsed, delay, rise, fall, width, *period] if (
            rise > 0 and fall > 0 and width > 0 and len(period) <= 1
        ):
            pulse: dict[str, JsonValue] = {
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
        case "sin", [offset, amplitude, frequency, *more] if frequency > 0 and len(
            more
        ) <= len(_SINE_OPTIONS):
            delay, damping, degrees = [*more, 0.0, 0.0, 0.0][: len(_SINE_OPTIONS)]
            return {
                "sine": {
                    "offset": offset,
                    "amplitude": amplitude,
                    "frequency": frequency,
                    "delay": delay,
                    "damping": damping,
                    "phase": math.radians(degrees),
                }
            }
        case "sin", _:
            message = (
                "a SIN without a positive frequency, which SPICE takes from the "
                "transient analysis's stop, and jolty's circuit from no analysis"
            )
            raise statement.fail(message)
        case "exp", [initial, pulsed, rise_delay, rise, fall_delay, fall] if (
            min(rise_delay, rise, fall_delay, fall) > 0
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
            points: list[JsonValue] = [
                [pairs[index], pairs[index + 1]] for index in range(0, len(pairs), 2)
            ]
            return {"piecewise": {"points": points}}
    message = f"jolty has no `{name.upper()}` waveform of these arguments"
    raise statement.fail(message)


def _scaled_mosfet(
    instance: dict[str, float], scale: float, statement: Statement
) -> dict[str, float]:
    """Scales MOSFET instance sizes by `scale`, raising `SpiceError` on overflow."""
    try:
        scaled = {
            key: value * math.pow(scale, _SCALED.get(key, 0))
            for key, value in instance.items()
        }
    except OverflowError, ValueError:
        message = f"scale option overflows: {scale!r}"
        raise statement.fail(message) from None
    if any(not math.isfinite(value) for value in scaled.values()):
        message = f"scale option overflows: {scale!r}"
        raise statement.fail(message)
    return scaled


def _semiconductor(
    terminals: tuple[str, ...], kinds: frozenset[str]
) -> ElementTranslator:
    """The translation of a diode or a transistor between `terminals`.

    Its model card is of one of `kinds`.
    """

    def translated(translation: Translation, statement: Statement) -> None:
        positional, assigned = split_keywords(tokens(statement.text))
        nodes = positional[1 : 1 + len(terminals)]
        rest = positional[1 + len(terminals) :]
        library, cards = translation.library, translation.models
        if len(nodes) != len(terminals) or not rest:
            message = f"the element joins {len(terminals)} nodes, then names its model"
            raise statement.fail(message)
        instance = translation.multiplied(
            instance_keywords(translation.value, statement, assigned)
        )
        if "nmos" in kinds:
            instance = _scaled_mosfet(
                instance, library.options.get("scale", 1.0), statement
            )
        elif "d" in kinds:
            # A level 3 diode scales its sizes by `scale` as it reads them, its masks'
            # offsets in their own units (`diosetup.c`, lines 336 to 360).
            instance["scale"] = library.options.get("scale", 1.0)
        card = library.model(rest[0], cards)
        if card is None and "nmos" in kinds:
            card = library.binned(
                rest[0],
                cards,
                instance,
                translation.scope,
                functions=translation.functions,
            )
            if card is None:
                first_bin = library.card(rest[0], cards)
                middle = (
                    None
                    if first_bin is None
                    else models.centre(
                        first_bin, translation.scope, functions=translation.functions
                    )
                )
                raise_unbinned(
                    statement, rest[0], instance, middle, library.fingers(instance)
                )
        if card is None:
            message = f"no model is named `{rest[0]}`"
            raise statement.fail(message)
        if card.kind not in kinds:
            message = f"`{card.name}` is a model of type `{card.kind}`"
            raise statement.fail(message)
        # An initial state, `OFF`, only guides the search for the operating point.
        areas = [word for word in rest[1:] if word.casefold() not in {"off", "on"}]
        if areas:
            instance["area"] = translation.value(areas[0])
        made = translation.device(card, instance, statement)
        joined = {
            terminal: translation.net(node)
            for terminal, node in zip(terminals, nodes, strict=True)
        }
        translation.add(statement, made.model, made.parameters, joined)

    return translated


_MOSFET: Final = _semiconductor(
    ("drain", "gate", "source", "bulk"), frozenset({"nmos", "pmos"})
)
"""The translation of a MOSFET element (`M`)."""


def _find_model_card(
    translation: Translation, positional: Sequence[str], first: int
) -> tuple[int | None, ModelCard | None]:
    """First position from `first` in `positional` naming a model, and its card."""
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


def _transistor(translation: Translation, statement: Statement) -> None:
    """The translation of an `M` element: a MOSFET, or a VDMOS, as its model card says.

    As in ngspice, the element names its model after its nodes, the first word from the
    fourth node on that names a model: a MOSFET joins four nodes, and a VDMOS three,
    its drain, gate and source, and a fourth and a fifth, its thermal nodes, which join
    nothing where the element does not heat by ngspice's ``thermal``, and jolty reads
    no element that does. A VDMOS takes its number of copies, ``M``, and no area.

    Raises:
        SpiceError: When no model follows the nodes, or the element is not one jolty
            reads.
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
    raw_instance: dict[str, float] = {}
    for key, values in assigned.items():
        if not values:
            message = f"`{key.upper()}` is not assigned a value"
            raise statement.fail(message)
        _check_vdmos_flags(statement, values[1:])
        raw_instance[key] = translation.value(values[0])
    instance = translation.multiplied(raw_instance)
    made = translation.device(card, instance, statement)
    joined = {
        terminal: translation.net(node)
        for terminal, node in zip(_VDMOS_TERMINALS, nodes, strict=False)
    }
    translation.add(statement, made.model, made.parameters, joined)


def _bipolar(translation: Translation, statement: Statement) -> None:
    """The translation of a `Q` element: a Gummel-Poon transistor or a VBIC, by card.

    As in ngspice, the element names its model after its nodes, the first word from the
    fourth node on that names a model: three nodes, then a substrate, the ground unless
    a fourth node gives it, and for a VBIC a fifth, its thermal node. A VBIC whose card
    heats, by a positive ``RTH`` and a ``SELFT`` of 1 or none, heats through a thermal
    terminal on that node's net, held to the ambient by its thermal resistance and
    capacitance, devices named after the element with ``.rth`` and ``.cth``; without a
    thermal node, its thermal node is the ground, as ngspice joins it, and it does not
    heat.

    Raises:
        SpiceError: When no bipolar transistor's model follows three to five nodes.
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
    instance = translation.multiplied(
        instance_keywords(
            translation.value,
            statement,
            assigned,
            multi_valued=frozenset({"ic"}),
        )
    )
    # An initial state, `OFF`, only guides the search for the operating point.
    areas = [word for word in rest if word.casefold() not in {"off", "on"}]
    if areas:
        instance["area"] = translation.value(areas[0])
    made = translation.device(card, instance, statement)
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
    heating = (
        models.vbic_heating(
            card, translation.scope, instance, functions=translation.functions
        )
        if len(nodes) == most == 5  # noqa: PLR2004 - a VBIC's thermal node, its fifth
        else None
    )
    if heating is None:
        translation.add(statement, made.model, made.parameters, joined)
        return
    thermal = translation.net(nodes[4])
    joined["thermal"] = thermal
    translation.add(statement, made.model, {**made.parameters, "thermal": True}, joined)
    resistance, capacitance = heating
    name = statement.text.split()[0]
    parts: tuple[tuple[str, str, dict[str, JsonValue]], ...] = (
        ("rth", "thermal_resistance", {"resistance": resistance}),
        ("cth", "thermal_capacitance", {"capacitance": capacitance}),
    )
    between = {"a": thermal, "b": translation.net("0")}
    for suffix, model, parameters in parts:
        translation.add(
            statement, model, parameters, between, identifier=f"{name}.{suffix}"
        )


def _controlled(output: str) -> ElementTranslator:
    """The translation of an E, F, G or H source, whose output is `output`."""

    def translated(translation: Translation, statement: Statement) -> None:
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
            translation.add(
                statement,
                f"voltage_controlled_{output}_source",
                {parameter: translation.value(gain)},
                {
                    "positive": translation.net(positive),
                    "negative": translation.net(negative),
                    "control_positive": translation.net(control_positive),
                    "control_negative": translation.net(control_negative),
                },
            )
            return
        else:
            message = "jolty reads a controlled source's gain, POLY, VALUE or TABLE"
            raise statement.fail(message)
        model = f"behavioral_{output}_source"
        translation.add_behavioral(statement, model, (positive, negative), expression)

    return translated


def _linear_current(
    translation: Translation,
    statement: Statement,
    nodes: tuple[str, str],
    rest: list[str],
    output: str,
) -> None:
    """A current-controlled source with a gain.

    Its control port is in series with the source whose current it reads.
    """
    source, gain = rest
    control_positive, control_negative = translation.sense(source, statement)
    parameter = "gain" if output == "current" else "transresistance"
    translation.add(
        statement,
        f"current_controlled_{output}_source",
        {parameter: translation.value(gain)},
        {
            "positive": translation.net(nodes[0]),
            "negative": translation.net(nodes[1]),
            "control_positive": control_positive,
            "control_negative": control_negative,
        },
    )


def _table(
    translation: Translation, statement: Statement, rest: list[str]
) -> Expression:
    """``TABLE {x} = (x1, y1) …``, as ngspice reads it.

    It interpolates its points, holds beyond the ends, and rounds its corners. The
    points are read leniently, as PSpice reads unbalanced parentheses.
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
    """``POLY(n)`` controls and coefficients, as SPICE2's polynomial.

    The coefficients are the constant's, the linear terms', then each product of two
    controls', then of three, in order.
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
    """The words of a group such as ``(3,0)``, or the word itself."""
    if word.startswith("(") and word.endswith(")"):
        return [part for part in _GROUP_SEPARATOR.split(word[1:-1]) if part]
    return [word]


def _arbitrary(translation: Translation, statement: Statement) -> None:
    """A B source: ``V=`` or ``I=`` an expression."""
    match = _ARBITRARY_SOURCE.match(statement.text)
    if match is None:
        message = "a B source joins two nodes, then sets `V=` or `I=` an expression"
        raise statement.fail(message)
    positive, negative, kind, written = match.groups()
    output = "voltage" if kind.casefold() == "v" else "current"
    if output == "current" and translation.multiplier != 1:
        message = (
            "jolty multiplies no behavioral current by its subcircuit's multiplier "
            f"`m`, here {translation.multiplier!r}"
        )
        raise statement.fail(message)
    expression = translation.parse(written)
    model = f"behavioral_{output}_source"
    translation.add_behavioral(statement, model, (positive, negative), expression)


def _switch(*, by_current: bool) -> ElementTranslator:
    """The translation of an S (voltage) or W (current) controlled switch."""
    expected_words = 5 if by_current else 6
    expected_kind = "iswitch" if by_current else "vswitch"

    def translated(translation: Translation, statement: Statement) -> None:
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
        translation.add(statement, made.model, made.parameters, terminals)

    return translated


def _instance(translation: Translation, statement: Statement) -> None:
    """An X instance of a subcircuit, with its parameters."""
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
    functions = translation.functions
    given = translation.multiplied(
        {
            parameter: evaluate(
                inlined(expression, functions), translation.scope, translation.draw
            )
            for parameter, expression in models.assignments(statement, assigned).items()
        }
    )
    built = translation.library.place_built(
        subcircuit,
        translation.globals,
        given,
        translation.drawing.within(name),
        translation.instantiating,
        uses=translation.uses,
    )
    child = built.component
    ports = [port.name for port in child.definition.ports]
    terminals = {
        port: translation.net(node) for port, node in zip(ports, nodes, strict=False)
    }
    if GROUND in ports:
        terminals[GROUND] = translation.net("0")
    prefix = f"{name}{ir.SEPARATOR}"
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

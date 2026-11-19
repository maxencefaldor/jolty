# Copyright 2026 Maxence Faldor
"""SPICE passive element translators: ``R``, ``C``, ``L``, ``K``, and ``T``."""

import math
from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import JsonValue

from jolty import _core
from jolty.spice import models
from jolty.spice._initial import DeviceVoltage
from jolty.spice._params import (
    NO_FUNCTIONS,
    check_keywords,
    single_keyword,
    split_keywords,
)
from jolty.spice._preprocess import letter
from jolty.spice._survey import ELEMENT_KINDS
from jolty.spice._translate import Coupling, ElementTranslator, Translation
from jolty.spice.expressions import (
    Binary,
    Current,
    Draw,
    Expression,
    Function,
    Number,
    Unary,
    Voltage,
    children,
    evaluate,
    fold,
    inlined,
    integral,
    rewrite,
    substituted,
    walk,
)
from jolty.spice.models import ModelCard
from jolty.spice.netlist import Statement, tokens
from jolty.units import celsius

_CARDS: Final = frozenset({"res", "cap", "ind", *models.GEOMETRIC})
"""The types of the model cards of a resistor, capacitor or inductor jolty reads:
PSpice's ``RES``, ``CAP`` and ``IND``, and SPICE3's ``R`` and ``C``."""

TRANSMISSION_LINE: Final = {
    "z0": "characteristic_impedance",
    "zo": "characteristic_impedance",
    "td": "delay",
}
"""A T element's parameters, by the SPICE names ngspice's ``tra.c`` (lines 13 to 16)
gives them, each parameter's own name first."""

_LINE_KEYWORDS: Final = frozenset(
    {*TRANSMISSION_LINE, "f", "nl", "ic", "v1", "i1", "v2", "i2"}
)
"""What a T element gives: its parameters, the frequency at which it is a normalized
length, and initial conditions, as ``tra.c`` (lines 13 to 22) names them."""

_QUARTER_WAVE_AT_GHZ: Final = (0.25, 1e9)
"""A T element's normalized length and frequency where it gives neither, as ngspice's
``trasetup.c`` (lines 94 to 99) sets them: a quarter wavelength at 1 GHz."""

_TRANSMISSION_LINE_TERMINALS: Final = (
    "positive1",
    "negative1",
    "positive2",
    "negative2",
)
"""Terminals of a lossless transmission line, matching SPICE's four-node order."""


def _passive_value_words(
    statement: Statement,
    element_letter: str,
    values: list[str],
    assigned: Mapping[str, list[str]],
) -> None:
    """Appends explicit passive keyword value (``R=``, ``C=``, ``L=``) to `values`."""
    if element_letter not in assigned:
        return
    if not assigned[element_letter]:
        message = f"`{element_letter.upper()}` is not assigned a value"
        raise statement.fail(message)
    values.extend(assigned[element_letter])


def _passive(model: str, parameter: str, kind: str) -> ElementTranslator:
    """The translation of a resistor, capacitor or inductor.

    Its model card, of PSpice's type `kind`, scales its value by the parameter of its
    letter, and gives its temperature coefficients and nominal temperature; SPICE3's
    card of a resistor or a capacitor, of the type of its letter, gives its value from
    its geometry where the element gives none, as
    [`geometric`][jolty.spice.models.geometric] says, from the element's width ``W``
    and length ``L``, which ngspice's ``scale`` scales (``resparam.c``, lines 59 and
    63; ``capparam.c``, lines 50 and 54). The element's own coefficients, ``TC=tc1,tc2``
    or each by name, override the card's. Only a resistor has an exponential
    coefficient, ``TCE``, and only a resistor and a capacitor voltage coefficients,
    ``VC1`` and ``VC2``, and a ``SCALE`` that multiplies their value. Its copies in
    parallel, its ``M`` times its subcircuit's multiplier, divide a resistor's and an
    inductor's value and multiply a capacitor's, and a resistor's flicker noise is
    theirs, ``m · KF · (|I| / m)^AF``, as ngspice adds it (``resnoise.c``, line 109).
    """
    element_letter = kind[0]
    curved = frozenset(models.PASSIVE_CURVATURE) if model != "inductor" else frozenset()
    drifts = (
        frozenset(models.PASSIVE_DRIFT)
        if model == "resistor"
        else frozenset({"tc1", "tc2"})
    )
    drifts |= curved
    noisy = frozenset({"kf", "af"}) if model == "resistor" else frozenset()
    sized = frozenset({"w", "l", "scale"}) if model != "inductor" else frozenset()
    stored = frozenset({"q"}) if model == "capacitor" else frozenset()
    initial = frozenset({"ic"}) if model in {"capacitor", "inductor"} else frozenset()
    allowed = frozenset(
        {element_letter, "m", "tc", "temp", "dtemp", *drifts, *sized, *stored, *initial}
    )

    def translated(translation: Translation, statement: Statement) -> None:
        positional, assigned = split_keywords(tokens(statement.text))
        check_keywords(statement, assigned, allowed)
        if len(positional) < 3:  # noqa: PLR2004 - a name and two nodes
            message = "an element joins two nodes"
            raise statement.fail(message)
        element_name, a, b, *rest = positional
        card, values = _card_and_values(
            translation, rest, {kind, element_letter} & _CARDS
        )
        _passive_value_words(statement, element_letter, values, assigned)
        initial_value = (
            single_keyword(translation.value, statement, "ic", assigned["ic"])
            if "ic" in assigned
            else None
        )
        if _behaves(translation, statement, (a, b), values, assigned):
            _record_passive_initial(
                translation,
                statement,
                model,
                element_name,
                (translation.net(a), translation.net(b)),
                initial_value=initial_value,
            )
            return
        geometric = card is not None and card.kind in models.GEOMETRIC
        if len(values) > 1 or not (values or geometric):
            message = f"a {model} has one value"
            raise statement.fail(message)
        own = {
            name: single_keyword(translation.value, statement, name, words)
            for name, words in assigned.items()
            if name in {"m", *sized}
        }
        scale = translation.library.options.get("scale", 1.0)
        own |= {name: own[name] * scale for name in ("w", "l") if name in own}
        if values:
            own[element_letter] = translation.value(values[0])
        read = models.NOMINAL_TEMPERATURES | models.OWN_TEMPERATURES | drifts | noisy
        value, given = _valued(
            translation, statement, card, own, (element_letter, read)
        )
        copies = translation.multiplied(own).get("m", 1.0)
        if not (copies > 0 and math.isfinite(copies)):
            message = f"the element's copies `m` is positive and finite, not {copies!r}"
            raise statement.fail(message)
        value = value * copies if model == "capacitor" else value / copies
        given |= _coefficients(translation, statement, assigned, drifts)
        temperatures = {
            name: single_keyword(translation.value, statement, name, words)
            for name, words in assigned.items()
            if name in {"temp", "dtemp"}
        }
        parameters: dict[str, JsonValue] = {parameter: value}
        parameters |= _temperatures(given)
        parameters |= models.own_temperature(given, temperatures, statement)
        parameters |= {
            models.PASSIVE_CURVATURE[name]: amount
            for name, amount in given.items()
            if name in models.PASSIVE_CURVATURE
        }
        parameters |= _flicker_parameters(statement, given, copies)
        net_a, net_b = translation.net(a), translation.net(b)
        translation.add(statement, model, parameters, {"a": net_a, "b": net_b})
        _record_passive_initial(
            translation,
            statement,
            model,
            element_name,
            (net_a, net_b),
            initial_value=initial_value,
            copies=copies,
        )

    return translated


def _flicker_parameters(
    statement: Statement, given: Mapping[str, float], copies: float
) -> dict[str, JsonValue]:
    """Scales a resistor's flicker noise parameters by parallel copies `copies`."""
    coefficient = given.get("kf", 0.0)
    if coefficient == 0:
        return {}
    scaled = models.scale_flicker_coefficient(
        coefficient, given.get("af", 1.0), copies, statement, "by `m`"
    )
    parameters: dict[str, JsonValue] = {
        models.FLICKER[name][0]: amount
        for name, amount in given.items()
        if name in models.FLICKER
    }
    parameters[models.FLICKER["kf"][0]] = scaled
    return parameters


def _record_passive_initial(  # noqa: PLR0913 - element identity, nets, IC and copies
    translation: Translation,
    statement: Statement,
    model: str,
    element_name: str,
    nets: tuple[str, str],
    *,
    initial_value: float | None,
    copies: float = 1.0,
) -> None:
    """Records a capacitor ``IC=`` voltage or inductor ``IC=`` current."""
    if initial_value is None:
        return
    if not math.isfinite(initial_value):
        message = f"`IC` initial condition is not finite: {initial_value!r}"
        raise statement.fail(message)
    if model == "capacitor":
        translation.capacitor_initial_voltages.append(
            DeviceVoltage(element_name, nets[0], nets[1], initial_value)
        )
    elif model == "inductor":
        scaled = initial_value * copies
        if not math.isfinite(scaled):
            message = (
                f"`IC` current scaled by `m` is not finite: "
                f"{initial_value!r} * {copies!r}"
            )
            raise statement.fail(message)
        translation.inductor_initial_currents[element_name.casefold()] = (
            element_name,
            scaled,
        )


def _behaves(
    translation: Translation,
    statement: Statement,
    nodes: tuple[str, str],
    values: Sequence[str],
    assigned: Mapping[str, list[str]],
) -> bool:
    """Whether the passive element is a behavioral device, which it is then read as.

    It is one where it is a capacitor of a charge, ``Q``, or where its value reads a
    voltage or a current: a capacitor holding its value's integral as its charge, a
    resistor a behavioral current.

    Raises:
        SpiceError: When it gives more than one charge, or cannot be read so.
    """
    charged = assigned.get("q")
    if charged is not None:
        if len(charged) != 1:
            message = "a capacitor's charge `Q` is one expression"
            raise statement.fail(message)
        charge = translation.parse(charged[0])
    elif values and _unknowns(written := translation.parse(values[0])):
        if letter(statement) != "c":
            _value_of_voltage(translation, statement, nodes, written, assigned)
            return True
        charge = _charge(translation, statement, nodes, written)
    else:
        return False
    _charge_of_voltage(translation, statement, nodes, charge, assigned)
    return True


def _card_and_values(
    translation: Translation, words: Sequence[str], kinds: set[str]
) -> tuple[ModelCard | None, list[str]]:
    """The model card of one of `kinds` that `words` name, if any, and the others."""
    card: ModelCard | None = None
    values: list[str] = []
    for word in words:
        found = translation.library.model(word, translation.models)
        if found is not None and found.kind in kinds:
            card = found
        else:
            values.append(word)
    return card, values


def _value_of_voltage(
    translation: Translation,
    statement: Statement,
    nodes: tuple[str, str],
    resistance: Expression,
    assigned: Mapping[str, list[str]],
) -> None:
    """A resistor whose value reads a voltage or a current, as ngspice makes it.

    ngspice makes it a behavioral current, ``I = m · V(a, b) / (R · (1 + TC1 · ΔT +
    TC2 · ΔT²))`` (``inpcom.c``, lines 6149 to 6151 and 6825 to 6870), its model card
    and sizes left out, ΔT the temperature above 27 °C (``asrcload.c``, lines 41 to
    51), here the temperature the library is read at, at which its ``temper`` is too;
    and without noise, as ngspice's behavioral current has none, unless the element
    asks for it, which jolty does not read.

    Raises:
        SpiceError: When the element is an inductor, which jolty reads not yet so, or
            gives a coefficient but its temperature's.
    """
    element_letter = letter(statement)
    if element_letter != "r":
        kind = ELEMENT_KINDS[element_letter]
        message = (
            f"jolty reads no {kind} whose value reads a voltage or a current, which "
            "ngspice makes a behavioral source"
        )
        raise statement.fail(message)
    given = _coefficients(
        translation, statement, assigned, frozenset(models.PASSIVE_DRIFT)
    )
    unread = sorted(set(given) - {"tc1", "tc2"})
    if unread:
        message = (
            f"jolty reads no `{unread[0].upper()}` of a resistor whose value reads a "
            "voltage or a current"
        )
        raise statement.fail(message)
    rise = translation.library.temperature - _core.REFERENCE_TEMPERATURE
    factor = 1 + given.get("tc1", 0.0) * rise + given.get("tc2", 0.0) * (rise * rise)
    if factor == 0 or not math.isfinite(factor):
        message = (
            f"temperature factor for resistor is finite and nonzero, not {factor!r}"
        )
        raise statement.fail(message)
    own: dict[str, float] = {}
    if "m" in assigned:
        own["m"] = single_keyword(translation.value, statement, "m", assigned["m"])
    copies = translation.multiplied(own).get("m", 1.0)
    if not (copies > 0 and math.isfinite(copies)):
        message = f"the element's copies `m` is positive and finite, not {copies!r}"
        raise statement.fail(message)
    current = Binary(
        "*",
        Binary("/", Voltage(nodes[0], nodes[1]), resistance),
        Number(copies / factor),
    )
    translation.add_behavioral(statement, "behavioral_current_source", nodes, current)


def _charge(
    translation: Translation,
    statement: Statement,
    nodes: tuple[str, str],
    capacitance: Expression,
) -> Expression:
    """The charge of a capacitor whose `capacitance` reads its voltage: its integral.

    ngspice makes such a capacitor a behavioral current, ``m · C(v) · dv/dt``
    (``inpcom.c``, lines 6895 to 6905 and 6958 to 6983), of which the charge
    ``Q(v) = ∫₀ᵛ C(u) du`` is the integral; read so, the capacitor conserves charge,
    its capacitance exactly ``C``. Every term of the capacitance that reads no voltage
    is evaluated first, then the charge integrated in closed form, as
    [`integral`][jolty.spice.expressions.integral] finds it. ngspice's resistance of
    1e15 Ω across the capacitor, which keeps a node it alone joins from floating
    (line 6978), is left out.

    Raises:
        SpiceError: When the capacitance reads another voltage or a current, which
            make no charge, or has no integral in closed form.
    """
    own = Voltage(*nodes)
    written = _of_own_voltage(_folded(translation, capacitance), nodes)
    found = None if written is None else integral(written, own)
    if written is None:
        message = (
            "jolty reads no capacitor whose value reads another voltage or a current, "
            "which makes no charge"
        )
        raise statement.fail(message)
    if found is None:
        message = (
            "jolty reads no capacitor whose value has no integral in closed form over "
            "its voltage, which its charge is"
        )
        raise statement.fail(message)
    zero = evaluate(substituted(found, own, Number(0.0)), {})
    return Binary("-", found, Number(zero))


def _mark_unknowns(expression: Expression) -> dict[int, bool]:
    """Marks whether each node in `expression` reads any voltage or current."""
    marks: dict[int, bool] = {}

    def leaf(current: Expression) -> bool | None:
        key = id(current)
        if key in marks:
            return marks[key]
        if isinstance(current, Voltage | Current):
            marks[key] = True
            return True
        if not children(current):
            marks[key] = False
            return False
        return None

    def combine(current: Expression, built: list[bool]) -> bool:
        found = any(built)
        marks[id(current)] = found
        return found

    fold(expression, leaf, combine)
    return marks


def _folded(translation: Translation, expression: Expression) -> Expression:
    """`expression` with each part that reads no voltage or current evaluated."""
    has_unknown = _mark_unknowns(expression)

    def replace(current: Expression) -> Expression | None:
        if not has_unknown[id(current)]:
            return Number(evaluate(current, translation.scope, translation.draw))
        return None

    return rewrite(expression, replace)


def _unknowns(expression: Expression) -> set[Voltage | Current]:
    """The voltages and currents `expression` reads."""
    return {node for node in walk(expression) if isinstance(node, Voltage | Current)}


def _of_own_voltage(
    expression: Expression, nodes: tuple[str, str]
) -> Expression | None:
    """`expression` reading the voltage of `nodes` as ``V(a, b)``, if it reads no other.

    ``V(b, a)`` is its negation, and ``V(a)`` it where `b` is the ground, ``0``.
    """
    a, b = (node.casefold() for node in nodes)
    own = Voltage(*nodes)
    written = expression
    for unknown in _unknowns(expression):
        if not isinstance(unknown, Voltage):
            return None
        ends = (unknown.positive.casefold(), (unknown.negative or "0").casefold())
        if ends not in {(a, b), (b, a)}:
            return None
        replacement = own if ends == (a, b) else Unary("-", own)
        written = substituted(written, unknown, replacement)
    return written


def _charge_of_voltage(
    translation: Translation,
    statement: Statement,
    nodes: tuple[str, str],
    charge: Expression,
    assigned: Mapping[str, list[str]],
) -> None:
    """A capacitor whose `charge` is an expression of voltages, a behavioral capacitor.

    ngspice makes a capacitor of a charge ``Q=`` a behavioral current, ``m · dQ/dt``
    (``inpcom.c``, lines 6937 to 6950), and one whose value reads a voltage one of its
    charge's rate of change too, as `_charge` integrates it; its copies `m`, its
    subcircuit's multiplier's too, multiply the charge, and its own temperature, which
    ngspice does not carry over, changes nothing.

    Raises:
        SpiceError: When the element gives temperature or voltage coefficients, or a
            size, which jolty does not read of such a capacitor.
    """
    unread = sorted(
        {"tc", "tc1", "tc2", "vc1", "vc2", "w", "l", "scale"} & assigned.keys()
    )
    if unread:
        message = (
            f"jolty reads no `{unread[0].upper()}` of a capacitor whose value reads a "
            "voltage or a current"
        )
        raise statement.fail(message)
    own: dict[str, float] = {}
    if "m" in assigned:
        own["m"] = single_keyword(translation.value, statement, "m", assigned["m"])
    copies = translation.multiplied(own).get("m", 1.0)
    if not (copies > 0 and math.isfinite(copies)):
        message = f"the element's copies `m` is positive and finite, not {copies!r}"
        raise statement.fail(message)
    held = charge if copies == 1 else Binary("*", charge, Number(copies))
    translation.add_behavioral(statement, "behavioral_capacitor", nodes, held)


def _valued(
    translation: Translation,
    statement: Statement,
    card: ModelCard | None,
    own: Mapping[str, float],
    reading: tuple[str, frozenset[str]],
) -> tuple[float, dict[str, float]]:
    """A passive element's value, before its copies, and what its `card` says of it.

    `reading` is the letter of its value and the parameters of its drift and noise a
    card gives. The element's `own` parameters hold its value by that letter, where it
    gives one, and its sizes; SPICE3's `card` of a resistor or a capacitor gives its
    value from its geometry where the element gives none, and PSpice's scales the
    element's by the parameter of that letter, and gives those of its drift and noise.

    Raises:
        SpiceError: When the card gives what jolty does not model.
    """
    element_letter, read = reading
    if card is not None and card.kind in models.GEOMETRIC:
        prefix = card.name.casefold()

        def draws(name: str) -> Draw:
            return translation.drawing.draws(f"{prefix}.{name}")

        values = models.geometric_values(
            card,
            translation.scope,
            draws,
            statement,
            functions=translation.functions,
        )
        made = models.geometric(card, values, own, statement)
        return made.value, made.drift
    value = own[element_letter] * own.get("scale", 1.0)
    if card is None:
        return value, {}
    drawing, prefix = translation.drawing, card.name.casefold()
    functions = translation.functions
    factor = card.parameters.get(element_letter, Number(1.0))
    value *= evaluate(
        inlined(factor, functions),
        translation.scope,
        drawing.draws(f"{prefix}.{element_letter}"),
    )
    _check_model(card, translation.scope, read | {element_letter}, statement, functions)
    given = {
        name: evaluate(
            inlined(expression, functions),
            translation.scope,
            drawing.draws(f"{prefix}.{name}"),
        )
        for name, expression in card.parameters.items()
        if name in read
    }
    return value, given


def _temperatures(given: Mapping[str, float]) -> dict[str, JsonValue]:
    """A resistor's, capacitor's or inductor's drift, by jolty name.

    It comes from the nominal temperature and the coefficients `given` by SPICE name.
    """
    parameters: dict[str, JsonValue] = {}
    for name, amount in given.items():
        if name in models.NOMINAL_TEMPERATURES:
            parameters["nominal_temperature"] = celsius(amount).value
        elif name in models.PASSIVE_DRIFT:
            parameters[models.PASSIVE_DRIFT[name]] = amount
    return parameters


def _coefficients(
    translation: Translation,
    statement: Statement,
    assigned: Mapping[str, list[str]],
    drifts: frozenset[str],
) -> dict[str, float]:
    """The temperature and voltage coefficients an element gives, by SPICE name.

    It gives them as ``TC=tc1,tc2``, or each of `drifts` by name.

    Raises:
        SpiceError: When ``TC`` gives more than two.
    """
    listed = assigned.get("tc")
    if listed is not None:
        if not listed:
            message = "`TC` is not assigned a value"
            raise statement.fail(message)
        if len(listed) > 2:  # noqa: PLR2004 - the linear and the quadratic coefficient
            message = "`TC` gives two coefficients at most"
            raise statement.fail(message)
    given = {
        name: translation.value(word)
        for name, word in zip(("tc1", "tc2"), listed or (), strict=False)
    }
    for name in drifts:
        words = assigned.get(name)
        if words is not None:
            given[name] = single_keyword(translation.value, statement, name, words)
    return given


def _check_model(
    card: ModelCard,
    scope: Mapping[str, float],
    allowed: frozenset[str],
    statement: Statement,
    functions: Mapping[str, Function] = NO_FUNCTIONS,
) -> None:
    """Checks a resistor's, capacitor's or inductor's model card.

    A capacitor's or inductor's flicker noise exponent changes nothing, as neither
    makes flicker noise: the coefficient, unless zero, is what fails.

    Raises:
        SpiceError: When a parameter is neither `allowed` nor one that changes nothing.
    """
    for name, expression in card.parameters.items():
        if name in allowed or name == "af":
            continue
        if evaluate(inlined(expression, functions), scope) != 0:
            raise models.unmatched(card, name, statement)


def _mutual(translation: Translation, statement: Statement) -> None:
    """Reads a mutual inductance, ``K1 L1 L2 0.99``, which couples each pair it names.

    ngspice couples two inductors, PSpice any number; the couplings are joined into
    coupled inductors once every element is read. A magnetic core's model is an error.
    """
    words = tokens(statement.text)
    if len(words) < 4:  # noqa: PLR2004 - a name, two inductors and a coefficient
        message = "a mutual inductance couples two inductors or more by a coefficient"
        raise statement.fail(message)
    _, *inductors, coefficient = words
    if not all(name[0].casefold() == "l" for name in inductors):
        message = "jolty has no magnetic core: a mutual inductance couples inductors"
        raise statement.fail(message)
    translation.couplings.append(
        Coupling(
            statement,
            tuple(inductors),
            translation.value(coefficient),
            translation.order,
        )
    )


def _transmission_line(translation: Translation, statement: Statement) -> None:
    """A T element, a lossless transmission line, between two ports of two nodes.

    Its characteristic impedance is ``Z0``, or ``ZO``. Its delay is ``TD``, or else the
    normalized length ``NL`` over the frequency ``F``, either taking ngspice's default
    where it is not given (``tratemp.c``, lines 35 to 37). Its initial conditions,
    ``IC`` or ``V1``, ``I1``, ``V2`` and ``I2``, are accepted and not applied by
    jolty's transmission line model.
    """
    positional, assigned = split_keywords(tokens(statement.text))
    check_keywords(statement, assigned, _LINE_KEYWORDS)
    if len(positional) != 5:  # noqa: PLR2004 - a name and four nodes
        message = "a transmission line joins four nodes"
        raise statement.fail(message)
    _, *nodes = positional

    def given(name: str) -> float | None:
        words = assigned.get(name)
        if words is None:
            return None
        return single_keyword(translation.value, statement, name, words)

    read: dict[str, float] = {}
    for spice, name in TRANSMISSION_LINE.items():
        value = given(spice)
        if value is not None:
            read.setdefault(name, value)
    impedance = read.get("characteristic_impedance")
    if impedance is None:
        message = "a transmission line gives its characteristic impedance, `Z0`"
        raise statement.fail(message)
    delay = read.get("delay")
    if delay is None:
        length, frequency = given("nl"), given("f")
        default_length, default_frequency = _QUARTER_WAVE_AT_GHZ
        divisor = default_frequency if frequency is None else frequency
        if not (divisor > 0 and math.isfinite(divisor)):
            message = (
                f"a transmission line's frequency `F` is positive and finite, not "
                f"{divisor!r}"
            )
            raise statement.fail(message)
        delay = (default_length if length is None else length) / divisor
    parameters: dict[str, JsonValue] = {
        "characteristic_impedance": impedance,
        "delay": delay,
    }
    # The line's terminals are in the order of SPICE's nodes, each port's in turn.
    translation.add(
        statement,
        "transmission_line",
        parameters,
        {
            terminal: translation.net(node)
            for terminal, node in zip(_TRANSMISSION_LINE_TERMINALS, nodes, strict=True)
        },
    )


PASSIVE_ELEMENTS: Final[dict[str, ElementTranslator]] = {
    "r": _passive("resistor", "resistance", "res"),
    "c": _passive("capacitor", "capacitance", "cap"),
    "l": _passive("inductor", "inductance", "ind"),
    "k": _mutual,
    "t": _transmission_line,
}
"""Passive SPICE element translators keyed by element letter."""

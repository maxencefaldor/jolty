# Copyright 2026 Maxence Faldor
"""SPICE passive element translators: ``R``, ``C``, ``L``, ``K``, and ``T``."""

import math
from collections.abc import Mapping, Sequence
from typing import Final

from jolty.spice import models
from jolty.spice._initial import DeviceVoltage
from jolty.spice._params import (
    NO_FUNCTIONS,
    check_keywords,
    single_keyword,
    split_keywords,
)
from jolty.spice._preprocess import letter
from jolty.spice._statistics import (
    record_behavioral_resistor_targets,
    record_passive_targets,
)
from jolty.spice._survey import ELEMENT_KINDS
from jolty.spice._translate import Coupling, ElementTranslator, Translation
from jolty.spice.expressions import (
    Binary,
    Current,
    Draw,
    DualDraw,
    Expression,
    Function,
    Name,
    Number,
    Scalar,
    Unary,
    Voltage,
    children,
    evaluate,
    fold,
    has_derivatives,
    inlined,
    integral,
    is_finite,
    primal,
    rewrite,
    substituted,
    walk,
)
from jolty.spice.models import DualJsonValue, ModelCard
from jolty.spice.netlist import Statement, tokens

_CARDS: Final = frozenset({"res", "cap", "ind", *models.GEOMETRIC})
"""Supported passive model-card types: PSpice ``RES``, ``CAP``, ``IND`` and SPICE3
``R``, ``C``."""

TRANSMISSION_LINE: Final = {
    "z0": "characteristic_impedance",
    "zo": "characteristic_impedance",
    "td": "delay",
}
"""SPICE transmission-line parameter names from ngspice ``tra.c`` (lines 13 to 16),
canonical name first."""

_LINE_KEYWORDS: Final = frozenset(
    {*TRANSMISSION_LINE, "f", "nl", "ic", "v1", "i1", "v2", "i2"}
)
"""Allowed transmission-line keywords (``tra.c``, lines 13 to 22): impedance, delay,
normalized length, frequency, and initial conditions."""

_QUARTER_WAVE_AT_GHZ: Final = (0.25, 1e9)
"""Default transmission-line `(normalized_length, frequency)` pair (0.25 at 1 GHz)
matching ngspice ``trasetup.c`` (lines 94 to 99)."""

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


def _read_single_keywords[Seed](
    translation: Translation[Seed],
    statement: Statement,
    assigned: Mapping[str, list[str]],
    keys: frozenset[str],
) -> tuple[dict[str, Scalar[Seed]], dict[str, Expression]]:
    """Evaluates single-valued keywords in `keys` returning `(values, symbolic)`."""
    values: dict[str, Scalar[Seed]] = {}
    symbolic: dict[str, Expression] = {}
    for name, words in assigned.items():
        if name in keys:
            resolved_value, expression = single_keyword(
                translation.value_symbolic, statement, name, words
            )
            values[name] = resolved_value
            if expression is not None:
                symbolic[name] = expression
    return values, symbolic


def _read_initial_condition[Seed](
    translation: Translation[Seed],
    statement: Statement,
    assigned: Mapping[str, list[str]],
) -> float | None:
    """Evaluates optional passive `IC=` value and rejects stochastic initial state."""
    if "ic" not in assigned:
        return None
    initial_value, initial_symbolic = single_keyword(
        translation.value_symbolic, statement, "ic", assigned["ic"]
    )
    if initial_symbolic is not None:
        message = (
            "statistical variation on `IC` changes initial state outside "
            "Monte Carlo device parameters"
        )
        raise statement.fail(message)
    return primal(initial_value)


def _passive_keywords(
    model: str, element_letter: str
) -> tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
    """Returns `(drifts, allowed, read, own_keys)` keyword sets for passive `model`."""
    curved: frozenset[str] = (
        frozenset(models.PASSIVE_CURVATURE) if model != "inductor" else frozenset()
    )
    drifts: frozenset[str] = (
        frozenset(models.PASSIVE_DRIFT)
        if model == "resistor"
        else frozenset({"tc1", "tc2"})
    ) | curved
    noisy: frozenset[str] = (
        frozenset({"kf", "af"}) if model == "resistor" else frozenset()
    )
    sized: frozenset[str] = (
        frozenset({"w", "l", "scale"}) if model != "inductor" else frozenset()
    )
    stored: frozenset[str] = frozenset({"q"}) if model == "capacitor" else frozenset()
    initial: frozenset[str] = (
        frozenset({"ic"}) if model in {"capacitor", "inductor"} else frozenset()
    )
    allowed: frozenset[str] = frozenset(
        {
            element_letter,
            "m",
            "tc",
            "temp",
            "dtemp",
            *(("tnom",) if model == "resistor" else ()),
            *drifts,
            *sized,
            *stored,
            *initial,
        }
    )
    nominal_keys: frozenset[str] = (
        models.NOMINAL_TEMPERATURES_WITH_TREF
        if model == "resistor"
        else models.NOMINAL_TEMPERATURES
    )
    read: frozenset[str] = nominal_keys | models.OWN_TEMPERATURES | drifts | noisy
    return drifts, allowed, read, frozenset({"m", *sized})


def _resolve_passive_copies[Seed](
    translation: Translation[Seed],
    statement: Statement,
    own: Mapping[str, Scalar[Seed]],
    own_symbolic: Mapping[str, Expression],
) -> tuple[Scalar[Seed], Expression | None]:
    """Resolves and validates effective parallel copies `m` for a passive element."""
    multiplied_own, multiplied_symbolic = translation.multiplied_symbolic(
        own, own_symbolic
    )
    copies = multiplied_own.get("m", 1.0)
    copies_symbolic = multiplied_symbolic.get("m")
    if not (primal(copies) > 0 and is_finite(copies)):
        message = (
            f"the element's copies `m` is positive and finite, not {primal(copies)!r}"
        )
        raise statement.fail(message)
    return copies, copies_symbolic


def _passive(model: str, parameter: str, kind: str) -> ElementTranslator:
    """Builds a translator for passive elements (`resistor`, `capacitor`, `inductor`).

    PSpice model cards (`kind`) scale the element value and supply temperature
    coefficients and nominal temperature. SPICE3 geometric cards (`R` or `C`) compute
    the value from `W` and `L` via [`geometric`][jolty.spice.models.geometric], scaled
    by ``.option scale`` (``resparam.c``, lines 59, 63; ``capparam.c``, lines 50, 54).

    Element-level coefficients (``TC=tc1,tc2`` or named keys) override card values.
    Resistors support ``TCE``; resistors and capacitors support ``VC1``, ``VC2``, and
    ``SCALE``. Effective multiplicity ``M`` (combined with subcircuit multiplicity)
    divides resistance and inductance, multiplies capacitance, and scales resistor
    flicker noise as ``m · KF · (|I| / m)^AF`` (``resnoise.c``, line 109).
    """
    element_letter = kind[0]
    drifts, allowed, read, own_keys = _passive_keywords(model, element_letter)
    temperature_keys = frozenset({"temp", "dtemp"})
    nominal_keys = frozenset({"tnom"}) if model == "resistor" else frozenset()

    def translated[Seed](translation: Translation[Seed], statement: Statement) -> None:
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
        initial_value = _read_initial_condition(translation, statement, assigned)
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
            message = f"a {model} accepts one value"
            raise statement.fail(message)
        own, own_symbolic = _read_single_keywords(
            translation, statement, assigned, own_keys
        )
        scale = translation.library.options.get("scale", 1.0)
        own |= {name: own[name] * scale for name in ("w", "l") if name in own}
        if values:
            resolved_value, symbolic = translation.value_symbolic(values[0])
            own[element_letter] = resolved_value
            if symbolic is not None:
                own_symbolic[element_letter] = symbolic
        value, given, card_values, card_symbolic = _valued(
            translation, statement, card, own, (element_letter, read)
        )
        copies, copies_symbolic = _resolve_passive_copies(
            translation, statement, own, own_symbolic
        )
        value = value * copies if model == "capacitor" else value / copies
        _check_passive_value(statement, model, parameter, value)
        coefficients, coefficients_symbolic = _coefficients(
            translation, statement, assigned, drifts
        )
        nominal, nominal_symbolic = _read_single_keywords(
            translation, statement, assigned, nominal_keys
        )
        given |= coefficients | nominal
        coefficients_symbolic |= nominal_symbolic
        temperatures, temperatures_symbolic = _read_single_keywords(
            translation, statement, assigned, temperature_keys
        )
        parameters: dict[str, DualJsonValue[Seed]] = {parameter: value}
        parameters |= _temperatures(given, statement, allow_tref=model == "resistor")
        parameters |= models.own_temperature_dual(given, temperatures, statement)
        parameters |= {
            models.PASSIVE_CURVATURE[name]: amount
            for name, amount in given.items()
            if name in models.PASSIVE_CURVATURE
        }
        parameters |= _flicker_parameters(statement, given, copies)
        net_a, net_b = translation.net(a), translation.net(b)
        translation.add_dual(statement, model, parameters, {"a": net_a, "b": net_b})
        _record_passive_statistics(
            translation,
            element_name,
            model,
            parameter,
            card=card,
            own=own,
            own_symbolic=own_symbolic,
            card_values=card_values,
            card_symbolic=card_symbolic,
            given=given,
            coefficients=coefficients,
            coefficients_symbolic=coefficients_symbolic,
            temperatures_symbolic=temperatures_symbolic,
            copies=copies,
            copies_symbolic=copies_symbolic,
            scale=scale,
            statement=statement,
        )
        _record_passive_initial(
            translation,
            statement,
            model,
            element_name,
            (net_a, net_b),
            initial_value=initial_value,
            copies=primal(copies),
        )

    return translated


def _record_passive_statistics[Seed](  # noqa: PLR0913 - passive statistics inputs
    translation: Translation[Seed],
    element_name: str,
    model: str,
    parameter: str,
    *,
    card: ModelCard | None,
    own: Mapping[str, Scalar[Seed]],
    own_symbolic: Mapping[str, Expression],
    card_values: Mapping[str, float],
    card_symbolic: Mapping[str, Expression],
    given: Mapping[str, Scalar[Seed]],
    coefficients: Mapping[str, Scalar[Seed]],
    coefficients_symbolic: Mapping[str, Expression],
    temperatures_symbolic: Mapping[str, Expression],
    copies: Scalar[Seed],
    copies_symbolic: Expression | None,
    scale: float,
    statement: Statement,
) -> None:
    """Records symbolic targets for a passive element during statistical compilation."""
    if translation.statistics is None:
        return
    element_letter = model[0]
    filtered_card = {
        key: expression
        for key, expression in card_symbolic.items()
        if key == element_letter or key not in coefficients
    }
    if not (
        own_symbolic
        or filtered_card
        or coefficients_symbolic
        or temperatures_symbolic
        or copies_symbolic is not None
    ):
        return
    record_passive_targets(
        translation.statistics,
        element_name,
        model,
        parameter,
        card=card,
        own={key: primal(value) for key, value in own.items()},
        own_symbolic=own_symbolic,
        card_values=card_values,
        card_symbolic=filtered_card,
        given={key: primal(value) for key, value in given.items()},
        given_symbolic=coefficients_symbolic,
        temperatures_symbolic=temperatures_symbolic,
        copies=primal(copies),
        copies_symbolic=copies_symbolic,
        scale_option=scale,
        statement=statement,
    )


def _flicker_parameters[Seed](
    statement: Statement,
    given: Mapping[str, Scalar[Seed]],
    copies: Scalar[Seed],
) -> dict[str, DualJsonValue[Seed]]:
    """Scales a resistor's flicker noise parameters by parallel copies `copies`."""
    coefficient = given.get("kf", 0.0)
    if primal(coefficient) == 0.0 and not has_derivatives(coefficient):
        return {}
    scaled = models.scale_flicker_coefficient_dual(
        coefficient, given.get("af", 1.0), copies, statement, "by `m`"
    )
    parameters: dict[str, DualJsonValue[Seed]] = {
        models.FLICKER[name][0]: amount
        for name, amount in given.items()
        if name in models.FLICKER
    }
    parameters[models.FLICKER["kf"][0]] = scaled
    return parameters


def _record_passive_initial[Seed](  # noqa: PLR0913 - element identity, nets, IC and copies
    translation: Translation[Seed],
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


def _behaves[Seed](
    translation: Translation[Seed],
    statement: Statement,
    nodes: tuple[str, str],
    values: Sequence[str],
    assigned: Mapping[str, list[str]],
) -> bool:
    """Translates behavioral passive elements and returns whether one was matched.

    Handles capacitors with explicit ``Q=`` charge expressions, voltage-dependent
    capacitors integrated into charge, and expression-defined resistors lowered to
    behavioral resistors.

    Raises:
        SpiceError: When ``Q`` has multiple tokens or the behavioral form is
            unsupported.
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


def _card_and_values[Seed](
    translation: Translation[Seed], words: Sequence[str], kinds: set[str]
) -> tuple[ModelCard | None, list[str]]:
    """Separates a matching `ModelCard` of `kinds` from value tokens in `words`."""
    card: ModelCard | None = None
    values: list[str] = []
    for word in words:
        found = translation.library.model(word, translation.models)
        if found is not None and found.kind in kinds:
            card = found
        else:
            values.append(word)
    return card, values


def _behavioral_resistor_copies[Seed](
    translation: Translation[Seed],
    resistance: Expression,
    copies: Scalar[Seed],
    copies_symbolic: Expression | None,
) -> tuple[
    Expression,
    dict[str, tuple[float, Expression]] | None,
    dict[str, Scalar[Seed]] | None,
]:
    """Scales `resistance` by parallel copies `m`, extracting `__jolty_m` if varying."""
    primal_copies = primal(copies)
    if copies_symbolic is not None:
        return (
            Binary("/", resistance, Name("__jolty_m")),
            {"__jolty_m": (primal_copies, copies_symbolic)},
            None,
        )
    if translation.dual_draw_factory is not None and has_derivatives(copies):
        return (
            Binary("/", resistance, Name("__jolty_m")),
            None,
            {"__jolty_m": copies},
        )
    scaled = (
        resistance
        if primal_copies == 1.0
        else Binary("/", resistance, Number(primal_copies))
    )
    return scaled, None, None


def _value_of_voltage[Seed](
    translation: Translation[Seed],
    statement: Statement,
    nodes: tuple[str, str],
    resistance: Expression,
    assigned: Mapping[str, list[str]],
) -> None:
    """Translates a voltage- or current-dependent resistor into a behavioral resistor.

    Preserves ``TC``/``TC1``/``TC2``, ``TNOM``, ``TEMP``/``DTEMP``, and multiplicity
    ``M`` dynamically so resistance scales with simulation temperature as
    ``(R / m) * (1 + TC1 * dT + TC2 * dT**2)`` and emits Johnson-Nyquist thermal
    noise ``4kT / R_eff`` whenever ``R_eff > 0``. Ignores model cards and geometry,
    matching ngspice ``inpcom.c`` (lines 6825 to 6870).

    Raises:
        SpiceError: When the element is an inductor or sets unsupported coefficients
            outside ``TC1`` and ``TC2``.
    """
    element_letter = letter(statement)
    if element_letter != "r":
        kind = ELEMENT_KINDS[element_letter]
        message = (
            f"jolty reads no {kind} whose value reads a voltage or a current, which "
            "ngspice makes a behavioral source"
        )
        raise statement.fail(message)
    given, given_symbolic = _coefficients(
        translation,
        statement,
        assigned,
        frozenset(models.PASSIVE_DRIFT) | frozenset(models.PASSIVE_CURVATURE),
    )
    unread = sorted(set(given) - {"tc1", "tc2"})
    if unread:
        message = (
            f"jolty reads no `{unread[0].upper()}` of a resistor whose value reads a "
            "voltage or a current"
        )
        raise statement.fail(message)
    nominal, nominal_symbolic = _read_single_keywords(
        translation, statement, assigned, frozenset({"tnom"})
    )
    temperatures, temperatures_symbolic = _read_single_keywords(
        translation, statement, assigned, frozenset({"temp", "dtemp"})
    )
    given |= nominal
    drift_symbolic = {**given_symbolic, **nominal_symbolic}
    own, own_symbolic = _read_single_keywords(
        translation, statement, assigned, frozenset({"m"})
    )
    multiplied_own, multiplied_symbolic = translation.multiplied_symbolic(
        own, own_symbolic
    )
    copies = multiplied_own.get("m", 1.0)
    copies_symbolic = multiplied_symbolic.get("m")
    if not (primal(copies) > 0 and is_finite(copies)):
        message = (
            f"the element's copies `m` is positive and finite, not {primal(copies)!r}"
        )
        raise statement.fail(message)
    scaled, extra_parameters, dual_extra_parameters = _behavioral_resistor_copies(
        translation, resistance, copies, copies_symbolic
    )
    parameters: dict[str, DualJsonValue[Seed]] = _temperatures(
        given, statement, allow_tref=True
    ) | models.own_temperature_dual({}, temperatures, statement)
    translation.add_behavioral(
        statement,
        "behavioral_resistor",
        nodes,
        scaled,
        parameters=parameters,
        extra_parameters=extra_parameters,
        dual_extra_parameters=dual_extra_parameters,
    )
    if translation.statistics is not None and (drift_symbolic or temperatures_symbolic):
        record_behavioral_resistor_targets(
            translation.statistics,
            statement.text.split()[0],
            drift_symbolic,
            temperatures_symbolic,
        )


def _charge[Seed](
    translation: Translation[Seed],
    statement: Statement,
    nodes: tuple[str, str],
    capacitance: Expression,
) -> Expression:
    """Integrates a voltage-dependent `capacitance` into a charge expression ``Q(v)``.

    Rewrites ngspice ``m · C(v) · dv/dt`` (``inpcom.c``, lines 6895-6905, 6958-6983)
    into conserving charge ``Q(v) = ∫₀ᵛ C(u) du``. Constant subexpressions evaluate
    before closed-form integration via [`integral`][jolty.spice.expressions.integral],
    omitting the ngspice 1e15 Ω anti-floating shunt (line 6978).

    Raises:
        SpiceError: When `capacitance` depends on another voltage or current, varies
            statistically, or has no closed-form antiderivative.
    """
    if translation.statistics is not None and (
        translation.statistics.depends_on_statistics(capacitance)
    ):
        message = (
            "statistical variation on a voltage-dependent capacitor value cannot "
            "compile to a closed-form charge integral"
        )
        raise statement.fail(message)
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


def _folded[Seed](translation: Translation[Seed], expression: Expression) -> Expression:
    """Evaluates constant subexpressions in `expression` without node references."""
    has_unknown = _mark_unknowns(expression)

    def replace(current: Expression) -> Expression | None:
        if not has_unknown[id(current)]:
            return Number(evaluate(current, translation.primal_scope, translation.draw))
        return None

    return rewrite(expression, replace)


def _unknowns(expression: Expression) -> set[Voltage | Current]:
    """Returns the set of `Voltage` and `Current` references inside `expression`."""
    return {node for node in walk(expression) if isinstance(node, Voltage | Current)}


def _of_own_voltage(
    expression: Expression, nodes: tuple[str, str]
) -> Expression | None:
    """Rewrites terminal-voltage references in `expression` to ``V(a, b)``, or None.

    Maps ``V(b, a)`` to ``-V(a, b)`` and ``V(a)`` to ``V(a, b)`` when `b` is ground
    ``0``. Returns None when `expression` references any other voltage or current.
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


def _charge_of_voltage[Seed](
    translation: Translation[Seed],
    statement: Statement,
    nodes: tuple[str, str],
    charge: Expression,
    assigned: Mapping[str, list[str]],
) -> None:
    """Translates a charge-defined capacitor (``Q=`` or integrated ``C(v)``).

    Matches ngspice ``m · dQ/dt`` (``inpcom.c``, lines 6937-6950). Multiplies `charge`
    by combined copy count `m` and ignores instance temperature overrides, which
    ngspice does not apply to behavioral capacitors.

    Raises:
        SpiceError: When unsupported temperature, voltage, or geometry keys are set.
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
    own, own_symbolic = _read_single_keywords(
        translation, statement, assigned, frozenset({"m"})
    )
    multiplied_own, multiplied_symbolic = translation.multiplied_symbolic(
        own, own_symbolic
    )
    copies = multiplied_own.get("m", 1.0)
    copies_symbolic = multiplied_symbolic.get("m")
    if not (primal(copies) > 0 and is_finite(copies)):
        message = (
            f"the element's copies `m` is positive and finite, not {primal(copies)!r}"
        )
        raise statement.fail(message)
    primal_copies = primal(copies)
    extra_parameters: dict[str, tuple[float, Expression]] | None = None
    dual_extra_parameters: dict[str, Scalar[Seed]] | None = None
    if copies_symbolic is not None:
        held: Expression = Binary("*", charge, Name("__jolty_copies"))
        extra_parameters = {"__jolty_copies": (primal_copies, copies_symbolic)}
    elif translation.dual_draw_factory is not None and has_derivatives(copies):
        held = Binary("*", charge, Name("__jolty_copies"))
        dual_extra_parameters = {"__jolty_copies": copies}
    else:
        held = (
            charge
            if primal_copies == 1.0
            else Binary("*", charge, Number(primal_copies))
        )
    translation.add_behavioral(
        statement,
        "behavioral_capacitor",
        nodes,
        held,
        extra_parameters=extra_parameters,
        dual_extra_parameters=dual_extra_parameters,
    )


def _geometric_valued[Seed](
    translation: Translation[Seed],
    statement: Statement,
    card: ModelCard,
    own: Mapping[str, Scalar[Seed]],
) -> tuple[
    Scalar[Seed],
    dict[str, Scalar[Seed]],
    dict[str, float],
    dict[str, Expression],
]:
    """Evaluates a SPICE3 geometric passive card and lowers its symbolic expressions."""
    prefix = card.name.casefold()
    if translation.differentiating:

        def dual_draws(name: str) -> DualDraw[Seed] | None:
            return translation.dual_draws_for(f"{prefix}.{name}")

        dual_values = models.geometric_values_dual(
            card,
            translation.scope,
            dual_draws,
            statement,
            functions=translation.functions,
        )
        made_dual = models.geometric_dual(card, dual_values, own, statement)
        return made_dual.value, made_dual.drift, {}, {}

    def draws(name: str) -> Draw:
        return translation.drawing.draws(f"{prefix}.{name}")

    scope = translation.primal_scope
    primal_own = {key: primal(entry) for key, entry in own.items()}
    values = models.geometric_values(
        card,
        scope,
        draws,
        statement,
        functions=translation.functions,
    )
    made = models.geometric(card, values, primal_own, statement)
    card_symbolic: dict[str, Expression] = {}
    if translation.statistics is not None:
        table = models.GEOMETRIC[card.kind]
        for name, expression in card.parameters.items():
            canonical = table[name]
            if canonical == "bv_max":
                continue
            lowered = translation.statistics.lower(
                inlined(expression, translation.functions),
                scope,
                call_name=f"{prefix}.{name}",
            )
            if lowered is not None:
                card_symbolic[canonical] = lowered
            else:
                card_symbolic.pop(canonical, None)
    return made.value, dict(made.drift), values, card_symbolic


def _valued[Seed](
    translation: Translation[Seed],
    statement: Statement,
    card: ModelCard | None,
    own: Mapping[str, Scalar[Seed]],
    reading: tuple[str, frozenset[str]],
) -> tuple[
    Scalar[Seed],
    dict[str, Scalar[Seed]],
    dict[str, float],
    dict[str, Expression],
]:
    """Computes pre-multiplicity passive value, drift, card values, and symbolic map.

    Unpacks `reading` into `(element_letter, allowed_card_keys)`. Evaluates SPICE3
    geometric cards through `models.geometric` (or `models.geometric_dual` when
    differentiating), and applies PSpice card multipliers and parameter evaluation to
    `own` element values.

    Raises:
        SpiceError: When `card` specifies parameters unsupported by jolty.
    """
    element_letter, read = reading
    if card is not None and card.kind in models.GEOMETRIC:
        return _geometric_valued(translation, statement, card, own)
    value = own[element_letter] * own.get("scale", 1.0)
    if card is None:
        return value, {}, {}, {}
    prefix = card.name.casefold()
    functions = translation.functions
    scope = translation.primal_scope
    factor_expression = card.parameters.get(element_letter, Number(1.0))
    factor_value = translation.evaluate_expression(
        factor_expression, draw_name=f"{prefix}.{element_letter}"
    )
    value *= factor_value
    _check_model(card, scope, read | {element_letter}, statement, functions)
    given: dict[str, Scalar[Seed]] = {}
    card_values: dict[str, float] = {element_letter: primal(factor_value)}
    card_symbolic: dict[str, Expression] = {}
    if translation.statistics is not None and element_letter in card.parameters:
        lowered_factor = translation.statistics.lower(
            inlined(factor_expression, functions),
            scope,
            call_name=f"{prefix}.{element_letter}",
        )
        if lowered_factor is not None:
            card_symbolic[element_letter] = lowered_factor
    for name, expression in card.parameters.items():
        if name not in read:
            continue
        evaluated = translation.evaluate_expression(
            expression, draw_name=f"{prefix}.{name}"
        )
        given[name] = evaluated
        card_values[name] = primal(evaluated)
        if translation.statistics is not None:
            lowered = translation.statistics.lower(
                inlined(expression, functions),
                scope,
                call_name=f"{prefix}.{name}",
            )
            if lowered is not None:
                card_symbolic[name] = lowered
    return value, given, card_values, card_symbolic


def _check_passive_value[Seed](
    statement: Statement, model: str, parameter: str, value: Scalar[Seed]
) -> None:
    """Validates passive element `value` against engine positivity requirements.

    Raises:
        SpiceError: When a resistor or inductor is non-positive, a capacitor is
            negative, or any passive value is non-finite.
    """
    raw = primal(value)
    if model == "capacitor":
        if not (raw >= 0 and math.isfinite(raw)):
            message = f"{parameter} is at least 0, not {raw!r}"
            raise statement.fail(message)
        return
    if not (raw > 0 and math.isfinite(raw)):
        message = f"{parameter} is positive, not {raw!r}"
        raise statement.fail(message)


def _temperatures[Seed](
    given: Mapping[str, Scalar[Seed]],
    statement: Statement,
    *,
    allow_tref: bool = False,
) -> dict[str, DualJsonValue[Seed]]:
    """Maps passive nominal temperature and SPICE drift keys in `given` to jolty."""
    parameters = models.nominal_temperature_dual(
        given, statement, allow_tref=allow_tref
    )
    for name, amount in given.items():
        if name in models.PASSIVE_DRIFT:
            parameters[models.PASSIVE_DRIFT[name]] = amount
    return parameters


def _coefficients[Seed](
    translation: Translation[Seed],
    statement: Statement,
    assigned: Mapping[str, list[str]],
    drifts: frozenset[str],
) -> tuple[dict[str, Scalar[Seed]], dict[str, Expression]]:
    """Extracts temperature and voltage coefficients from `assigned` under SPICE names.

    Accepts both positional ``TC=tc1,tc2`` syntax and individual keywords in `drifts`.

    Raises:
        SpiceError: When ``TC`` is empty or provides more than two coefficients.
    """
    listed = assigned.get("tc")
    if listed is not None:
        if not listed:
            message = "`TC` is not assigned a value"
            raise statement.fail(message)
        if len(listed) > 2:  # noqa: PLR2004 - the linear and the quadratic coefficient
            message = "`TC` gives two coefficients at most"
            raise statement.fail(message)
    given: dict[str, Scalar[Seed]] = {}
    symbolic: dict[str, Expression] = {}
    for name, word in zip(("tc1", "tc2"), listed or (), strict=False):
        resolved_value, expression = translation.value_symbolic(word)
        given[name] = resolved_value
        if expression is not None:
            symbolic[name] = expression
    for name in drifts:
        words = assigned.get(name)
        if words is not None:
            resolved_value, expression = single_keyword(
                translation.value_symbolic, statement, name, words
            )
            given[name] = resolved_value
            if expression is not None:
                symbolic[name] = expression
            else:
                symbolic.pop(name, None)
    return given, symbolic


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


def _mutual[Seed](translation: Translation[Seed], statement: Statement) -> None:
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
    coefficient_value, coefficient_symbolic = translation.value_symbolic(coefficient)
    translation.couplings.append(
        Coupling(
            statement,
            tuple(inductors),
            coefficient_value,
            translation.order,
            coefficient_symbolic,
        )
    )


def _transmission_line_delay[Seed](
    length_entry: tuple[Scalar[Seed], Expression | None] | None,
    frequency_entry: tuple[Scalar[Seed], Expression | None] | None,
    statement: Statement,
) -> tuple[Scalar[Seed], Expression | None]:
    """Computes transmission-line delay (`NL / F`) and its symbolic expression."""
    default_length, default_frequency = _QUARTER_WAVE_AT_GHZ
    length_value: Scalar[Seed]
    divisor: Scalar[Seed]
    length_value, length_symbolic = (
        (default_length, None) if length_entry is None else length_entry
    )
    divisor, frequency_symbolic = (
        (default_frequency, None) if frequency_entry is None else frequency_entry
    )
    if not (primal(divisor) > 0 and is_finite(divisor)):
        message = (
            f"a transmission line's frequency `F` is positive and finite, not "
            f"{primal(divisor)!r}"
        )
        raise statement.fail(message)
    delay = length_value / divisor
    if length_symbolic is None and frequency_symbolic is None:
        return delay, None
    numerator = (
        Number(primal(length_value)) if length_symbolic is None else length_symbolic
    )
    denominator = (
        Number(primal(divisor)) if frequency_symbolic is None else frequency_symbolic
    )
    return delay, Binary("/", numerator, denominator)


def _transmission_line[Seed](
    translation: Translation[Seed], statement: Statement
) -> None:
    """Translates a lossless SPICE `T` transmission line connecting two two-node ports.

    Reads characteristic impedance from ``Z0`` (or ``ZO``) and propagation delay from
    ``TD`` or ``NL / F`` using ngspice defaults (``tratemp.c``, lines 35 to 37).
    Initial condition keywords (``IC``, ``V1``, ``I1``, ``V2``, ``I2``) are accepted
    and ignored by the jolty transmission-line model.
    """
    positional, assigned = split_keywords(tokens(statement.text))
    check_keywords(statement, assigned, _LINE_KEYWORDS)
    if len(positional) != 5:  # noqa: PLR2004 - a name and four nodes
        message = "a transmission line joins four nodes"
        raise statement.fail(message)
    element_name, *nodes = positional

    def given(name: str) -> tuple[Scalar[Seed], Expression | None] | None:
        words = assigned.get(name)
        if words is None:
            return None
        return single_keyword(translation.value_symbolic, statement, name, words)

    read: dict[str, tuple[Scalar[Seed], Expression | None]] = {}
    for spice, name in TRANSMISSION_LINE.items():
        value = given(spice)
        if value is not None:
            read.setdefault(name, value)
    impedance_entry = read.get("characteristic_impedance")
    if impedance_entry is None:
        message = "a transmission line gives its characteristic impedance, `Z0`"
        raise statement.fail(message)
    impedance, impedance_symbolic = impedance_entry
    delay_entry = read.get("delay")
    delay, delay_symbolic = (
        delay_entry
        if delay_entry is not None
        else _transmission_line_delay(given("nl"), given("f"), statement)
    )
    parameters: dict[str, DualJsonValue[Seed]] = {
        "characteristic_impedance": impedance,
        "delay": delay,
    }
    # The line's terminals are in the order of SPICE's nodes, each port's in turn.
    translation.add_dual(
        statement,
        "transmission_line",
        parameters,
        {
            terminal: translation.net(node)
            for terminal, node in zip(_TRANSMISSION_LINE_TERMINALS, nodes, strict=True)
        },
    )
    if translation.statistics is not None:
        translation.statistics.add_target(
            element_name, "characteristic_impedance", impedance_symbolic
        )
        translation.statistics.add_target(element_name, "delay", delay_symbolic)


PASSIVE_ELEMENTS: Final[dict[str, ElementTranslator]] = {
    "r": _passive("resistor", "resistance", "res"),
    "c": _passive("capacitor", "capacitance", "cap"),
    "l": _passive("inductor", "inductance", "ind"),
    "k": _mutual,
    "t": _transmission_line,
}
"""Passive SPICE element translators keyed by element letter."""

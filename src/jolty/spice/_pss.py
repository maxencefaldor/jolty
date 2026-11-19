# Copyright 2026 Maxence Faldor
"""SPICE ``.pss`` periodic steady-state statement parsing and validation."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from jolty.errors import SpiceError
from jolty.spice._initial import check_net_name, node_of
from jolty.spice.expressions import Function, evaluate, inlined, number, parse
from jolty.spice.netlist import Statement, tokens

PERIODIC_STEADY_STATE: Final = ".pss"
"""The statement that specifies a periodic steady-state analysis."""

type IntegrationMethod = Literal["trapezoidal", "gear2", "backward_euler"]
"""An integration method named on a SPICE analysis statement."""


@dataclass(frozen=True, slots=True)
class PeriodicSteadyStateCard:
    """A ``.pss`` periodic steady-state statement read from a SPICE file.

    Follows ngspice 47's ``.pss`` syntax (§11.3.12) and keyword forms:

    - ``.pss``: autonomous oscillator, period unknown.
    - ``.pss <freq> [<tstab>]``: driven at period ``1 / freq``.
    - ``.pss <freq> <tstab> <oscnode> [points [harms [sc_iter [steady_coeff]]]] [uic]``:
      autonomous oscillator with guessed frequency ``freq`` and oscillator node
      ``oscnode`` (ngspice form).
    - Keyword options: ``period=``, ``fund=`` / ``freq=`` / ``fguess=``, ``tstab=``,
      ``oscnode=`` / ``oscnob=``, ``points=`` / ``psspoints=``, ``harms=`` /
      ``harmonics=``, ``sc_iter=``, ``steady_coeff=``, ``max_step=`` / ``tstep=``,
      ``method=``; bare flags ``autonomous`` and ``uic``.
    """

    autonomous: bool
    """True for an autonomous oscillator whose period shooting finds."""
    period: float | None = None
    """Driven period in seconds; None for an autonomous oscillator."""
    frequency: float | None = None
    """Fundamental frequency, or autonomous initial guess ``fguess``, in hertz."""
    stabilization: float | None = None
    """Stabilization time ``tstab``, in seconds."""
    oscillator_node: str | None = None
    """Oscillator output node ``oscnode``, if named."""
    points: int | None = None
    """Time-domain grid points per period (``psspoints``), if given."""
    harmonics: int | None = None
    """Harmonics requested for spectral output (``harms``), if given."""
    max_shooting_iterations: int | None = None
    """Maximum shooting iterations (``sc_iter``), if given."""
    steady_coefficient: float | None = None
    """Steady-state convergence coefficient (``steady_coeff``), if given."""
    max_step: float | None = None
    """Longest time step allowed, in seconds, if given."""
    method: IntegrationMethod | None = None
    """Integration method requested, if given."""
    operating_point: bool = True
    """Flag indicating whether an operating point runs first (False with ``uic``)."""
    statement: Statement | None = None
    """The source statement with its file and line number."""


_PSS_POSITIONALS: Final[tuple[str, ...]] = (
    "frequency",
    "stabilization",
    "oscillator_node",
    "points",
    "harmonics",
    "max_shooting_iterations",
    "steady_coefficient",
)

_PSS_ALIASES: Final[dict[str, str]] = {
    "period": "period",
    "fund": "frequency",
    "freq": "frequency",
    "fguess": "frequency",
    "tstab": "stabilization",
    "oscnode": "oscillator_node",
    "oscnob": "oscillator_node",
    "points": "points",
    "psspoints": "points",
    "harms": "harmonics",
    "harmonics": "harmonics",
    "sc_iter": "max_shooting_iterations",
    "steady_coeff": "steady_coefficient",
    "max_step": "max_step",
    "tstep": "max_step",
    "method": "method",
}

_PSS_METHODS: Final[dict[str, IntegrationMethod]] = {
    "trap": "trapezoidal",
    "trapezoidal": "trapezoidal",
    "gear": "gear2",
    "gear2": "gear2",
    "backward_euler": "backward_euler",
    "euler": "backward_euler",
    "be": "backward_euler",
}

_PSS_FLAGS: Final = frozenset({"uic", "autonomous"})


@dataclass(frozen=True, slots=True)
class _PssReader:
    """Evaluates and validates parsed fields of one ``.pss`` statement."""

    statement: Statement
    scope: Mapping[str, float]
    functions: Mapping[str, Function]
    raw: Mapping[str, str]

    def _number(self, key: str, label: str) -> float | None:
        """Evaluates an optional finite numeric field from `self.raw`."""
        if key not in self.raw:
            return None
        word = self.raw[key]
        given = number(word)
        if given is None:
            try:
                given = evaluate(inlined(parse(word), self.functions), self.scope)
            except SpiceError as error:
                message = f"invalid .pss {label} `{word}`: {error}"
                raise self.statement.fail(message) from None
        if not math.isfinite(given):
            message = f"a .pss {label} is finite, not `{word}`"
            raise self.statement.fail(message)
        return given

    def positive(self, key: str, label: str) -> float | None:
        """Evaluates an optional strictly positive floating-point field."""
        value = self._number(key, label)
        if value is not None and not value > 0.0:
            message = f"a .pss {label} is positive, not {value!r}"
            raise self.statement.fail(message)
        return value

    def non_negative(self, key: str, label: str) -> float | None:
        """Evaluates an optional non-negative floating-point field."""
        value = self._number(key, label)
        if value is not None and value < 0.0:
            message = f"a .pss {label} is at least 0, not {value!r}"
            raise self.statement.fail(message)
        return value

    def positive_int(self, key: str, label: str) -> int | None:
        """Evaluates an optional strictly positive integer field."""
        value = self._number(key, label)
        if value is None:
            return None
        if not (value > 0.0 and value.is_integer()):
            raw_value = self.raw[key]
            message = f"a .pss {label} is a positive integer, not `{raw_value}`"
            raise self.statement.fail(message)
        return int(value)


def _split_pss_tokens(
    statement: Statement,
) -> tuple[list[str], list[tuple[str, str]], set[str]]:
    """Splits a ``.pss`` statement into positionals, ``k=v`` pairs and bare flags."""
    words = tokens(statement.text)[1:]
    positionals: list[str] = []
    keywords: list[tuple[str, str]] = []
    flags: set[str] = set()
    index = 0
    while index < len(words):
        word = words[index]
        if word == "=":
            message = "unexpected `=` without an option name"
            raise statement.fail(message)
        if index + 1 < len(words) and words[index + 1] == "=":
            if index + 2 >= len(words) or words[index + 2] == "=":
                message = f"missing value after `{word}=`"
                raise statement.fail(message)
            keywords.append((word.casefold(), words[index + 2]))
            index += 3
            continue
        folded = word.casefold()
        if folded in _PSS_FLAGS:
            flags.add(folded)
        else:
            positionals.append(word)
        index += 1
    return positionals, keywords, flags


def _pss_fields(statement: Statement) -> tuple[dict[str, str], set[str]]:
    """Maps a ``.pss`` statement's arguments to canonical field names and flags."""
    positionals, keywords, flags = _split_pss_tokens(statement)
    if len(positionals) > len(_PSS_POSITIONALS):
        message = (
            f"a .pss statement takes at most {len(_PSS_POSITIONALS)} positional "
            f"arguments, not {len(positionals)}"
        )
        raise statement.fail(message)
    raw = dict(zip(_PSS_POSITIONALS, positionals, strict=False))
    for key, value in keywords:
        canonical = _PSS_ALIASES.get(key)
        if canonical is None:
            known = ", ".join(_PSS_ALIASES)
            message = f"unknown .pss option `{key}`; known options are {known}"
            raise statement.fail(message)
        if canonical in raw:
            message = f"duplicate .pss option for {canonical}: `{key}`"
            raise statement.fail(message)
        raw[canonical] = value
    return raw, flags


def _reciprocal(
    value: float, source_label: str, target_label: str, raw: str, statement: Statement
) -> float:
    """Computes `1 / value` and checks the reciprocal is finite and positive."""
    result = 1.0 / value
    if not (result > 0.0 and math.isfinite(result)):
        message = (
            f"reciprocal {target_label} of .pss {source_label} `{raw}` is not "
            "finite and positive"
        )
        raise statement.fail(message)
    return result


def _pss_node(word: str, statement: Statement) -> str:
    """Validates an oscillator node token (`node` or `V(node)`)."""
    if word[:2].casefold() == "v(":
        extracted = node_of(word)
        if extracted is None:
            message = f"an oscillator node is written `node` or `V(node)`, not `{word}`"
            raise statement.fail(message)
        node = extracted
    else:
        node = word
    if not node or "(" in node or ")" in node or "," in node:
        message = f"invalid oscillator node `{word}`"
        raise statement.fail(message)
    if node == "0":
        message = "an oscillator node cannot be the ground node `0`"
        raise statement.fail(message)
    check_net_name(node, statement)
    return node


def _pss_method(word: str, statement: Statement) -> IntegrationMethod:
    """Maps a SPICE integration method name on ``.pss`` to jolty's method name."""
    method = _PSS_METHODS.get(word.casefold())
    if method is None:
        known = ", ".join(_PSS_METHODS)
        message = f"unknown .pss method `{word}`; known methods are {known}"
        raise statement.fail(message)
    return method


def parse_periodic_steady_state(
    statement: Statement,
    scope: Mapping[str, float],
    functions: Mapping[str, Function],
) -> PeriodicSteadyStateCard:
    """Parses a ``.pss`` statement into a [`PeriodicSteadyStateCard`].

    Args:
        statement: Logical ``.pss`` statement to parse.
        scope: Case-folded global parameter values in scope for option expressions.
        functions: Case-folded user-defined functions inlined before evaluation.

    Returns:
        Validated `PeriodicSteadyStateCard` populated from `statement`.

    Raises:
        SpiceError: When tokens, options, expressions or values are invalid.
    """
    raw, flags = _pss_fields(statement)
    reader = _PssReader(statement, scope, functions, raw)
    period = reader.positive("period", "period")
    frequency = reader.positive("frequency", "frequency")
    if period is not None and frequency is not None:
        message = "a .pss statement sets period or frequency, not both"
        raise statement.fail(message)
    oscillator_node = (
        _pss_node(raw["oscillator_node"], statement)
        if "oscillator_node" in raw
        else None
    )
    autonomous = (
        "autonomous" in flags
        or oscillator_node is not None
        or (period is None and frequency is None)
    )
    if autonomous and period is not None:
        message = (
            "an autonomous .pss statement finds its own period and takes no period"
        )
        raise statement.fail(message)
    if not autonomous:
        if period is not None:
            frequency = _reciprocal(
                period, "period", "frequency", raw["period"], statement
            )
        elif frequency is not None:
            period = _reciprocal(
                frequency, "frequency", "period", raw["frequency"], statement
            )
    return PeriodicSteadyStateCard(
        autonomous=autonomous,
        period=period,
        frequency=frequency,
        stabilization=reader.non_negative("stabilization", "stabilization time"),
        oscillator_node=oscillator_node,
        points=reader.positive_int("points", "points"),
        harmonics=reader.positive_int("harmonics", "harmonics"),
        max_shooting_iterations=reader.positive_int(
            "max_shooting_iterations", "shooting iterations"
        ),
        steady_coefficient=reader.positive("steady_coefficient", "steady coefficient"),
        max_step=reader.positive("max_step", "max_step"),
        method=_pss_method(raw["method"], statement) if "method" in raw else None,
        operating_point="uic" not in flags,
        statement=statement,
    )

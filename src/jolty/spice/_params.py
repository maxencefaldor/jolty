# Copyright 2026 Maxence Faldor
"""SPICE parameter scopes, statistical draws, ``.func``, and keyword tokenizers."""

import itertools
import math
import re
import sys
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from jolty import _core
from jolty.errors import SpiceError
from jolty.spice import models
from jolty.spice.expressions import (
    Draw,
    Expression,
    Function,
    UndefinedError,
    evaluate,
    inlined,
    is_statistical,
    names,
    nominal,
    parse,
    walk,
)
from jolty.spice.models import ModelCard
from jolty.spice.netlist import Statement, tokens
from jolty.units import ZERO_CELSIUS

PARAMETERS: Final = frozenset({".param", ".params"})
"""The statements that define parameters: ngspice reads ``.params`` as ``.param``."""

OPTIONS: Final = frozenset({".option", ".options"})
"""The statements that set options, such as ``scale``."""

FUNCTION: Final = ".func"
"""The statement that defines a user's function, which every expression may call."""

NO_FUNCTIONS: Final[Mapping[str, Function]] = MappingProxyType({})
"""Empty user-function mapping used when a subcircuit defines no local ``.func``."""

NO_MODELS: Final[Mapping[str, ModelCard]] = MappingProxyType({})
"""Empty model mapping passed to `models.read` for non-`AKO:` definitions."""

STEP: Final = sys.float_info.epsilon ** (1 / 3)
"""Central-difference step per standard deviation for random-variable derivatives
(W. H. Press et al., *Numerical Recipes*, 3rd edition, 2007, §5.7)."""

_FUNCTION: Final = re.compile(
    r"\.func\s+(?P<name>[a-z_][a-z0-9_]*)\s*\((?P<parameters>[^()]*)\)\s*=?\s*"
    r"(?P<body>.+)",
    re.IGNORECASE | re.DOTALL,
)
"""A ``.func`` statement: ``.func name(a, b) {body}``, its body in braces or quotes
or bare, after an ``=`` or not."""


@dataclass(frozen=True, slots=True)
class Sample:
    """Monte Carlo sample seeding deterministic draws for statistical functions.

    Each call queries the engine variate generator using `seed`, `index`, placement
    path, and parameter call identifier. Results are therefore identical whether
    evaluated alone or on any shard, while each placement draws independently.
    """

    seed: int
    index: int

    def draw(self, scope: str, name: str) -> Draw:
        """Returns a draw callback for statistical calls in `name` under `scope`.

        Matches ngspice numparam ``xpressn.c`` using standard normal ``z`` or uniform
        ``u`` in ``[-1, 1)``. ``agauss(x, a, s)`` returns ``x + a / s * z`` and
        ``gauss(x, r, s)`` returns ``x + x * r / s * z``, returning ``x`` when ``a``,
        ``r``, or ``s`` is non-positive. ``aunif(x, a)`` returns ``x + a * u``,
        ``unif(x, r)`` returns ``x + x * r * u``, and ``limit(x, a)`` returns ``x + a``
        for ``u > 0`` and ``x - a`` otherwise.
        """
        calls = itertools.count()

        def drawn(function: str, arguments: tuple[float, ...]) -> float:
            call = f"{name}#{next(calls)}"
            nominal_value, spread, *rest = arguments
            if function in {"agauss", "gauss"}:
                sigma = rest[0]
                if spread <= 0 or sigma <= 0:
                    return nominal_value
                deviation = spread / sigma
                if function == "gauss":
                    deviation *= nominal_value
                normal = _core.variate("normal", self.seed, self.index, scope, call)
                return nominal_value + deviation * normal
            uniform = _core.variate("uniform", self.seed, self.index, scope, call)
            match function:
                case "aunif":
                    return nominal_value + spread * uniform
                case "unif":
                    return nominal_value + nominal_value * spread * uniform
            return nominal_value + (spread if uniform > 0 else -spread)

        return drawn


def standard_deviation(function: str, arguments: tuple[float, ...]) -> float:
    """Computes the standard deviation of statistical `function` given `arguments`.

    Matches [`Sample.draw`][jolty.spice.Sample.draw]: ``agauss(x, a, s)`` gives
    ``a / s`` and ``gauss(x, r, s)`` gives ``|x| r / s`` (or 0 when ``a``, ``r``, or
    ``s`` is non-positive). Uniform calls ``aunif(x, a)`` and ``unif(x, r)`` give
    ``|a| / √3`` and ``|x r| / √3``, and ``limit(x, a)`` gives ``|a|``.

    Examples:
        >>> standard_deviation("agauss", (0.0, 0.03, 3.0))
        0.01
        >>> standard_deviation("aunif", (2.0, 0.3))
        0.17320508075688773
    """
    nominal_value, spread, *rest = arguments
    match function:
        case "agauss" | "gauss":
            sigma = rest[0]
            if spread <= 0 or sigma <= 0:
                return 0.0
            scale = abs(nominal_value) if function == "gauss" else 1.0
            return scale * spread / sigma
        case "aunif":
            return abs(spread) / math.sqrt(3)
        case "unif":
            return abs(nominal_value * spread) / math.sqrt(3)
    return abs(spread)


@dataclass(slots=True)
class Probe:
    """Statistical draw probe returning nominal values except for one perturbed call.

    The call identified by `moved` returns its nominal value plus `offset`. Every
    encountered call is recorded under `(scope, call_id)` with its function name and
    evaluated arguments, matching [`Sample`][jolty.spice.Sample] keys.
    """

    moved: tuple[str, str] | None = None
    offset: float = 0.0
    calls: dict[tuple[str, str], tuple[str, tuple[float, ...]]] = field(
        default_factory=dict
    )

    def draw(self, scope: str, name: str) -> Draw:
        """Returns a recording callback for statistical calls in `name` at `scope`."""
        calls = itertools.count()

        def drawn(function: str, arguments: tuple[float, ...]) -> float:
            call = (scope, f"{name}#{next(calls)}")
            self.calls[call] = (function, arguments)
            nominal_value = arguments[0]
            return nominal_value + self.offset if call == self.moved else nominal_value

        return drawn


@dataclass(frozen=True, slots=True)
class Drawing:
    """Statistical sampling context pairing a `Sample` or `Probe` with a circuit path.

    Statistical calls draw from `sample` at `path`, return nominal values when `sample`
    is None, or record perturbations under a `Probe`.
    """

    sample: Sample | Probe | None
    path: str

    def draws(self, name: str) -> Draw:
        """Returns the `Draw` callback for statistical calls in definition `name`."""
        if self.sample is None:
            return nominal
        return self.sample.draw(self.path, name)

    def named(self, name: str) -> str:
        """Qualifies component `name` with path and sample identity to avoid clashes."""
        match self.sample:
            case Sample(seed=seed, index=index):
                return f"{name}@{self.path}#{seed}:{index}"
            case Probe():
                return f"{name}@{self.path}"
        return name

    def within(self, instance: str) -> Drawing:
        """Returns a child `Drawing` scoped to nested placement `instance`."""
        path = f"{self.path}/{instance}" if self.path else instance
        return Drawing(self.sample, path)


@dataclass(frozen=True, slots=True)
class Subcircuit:
    """A subcircuit's definition: its pins, its defaults, local functions, and body."""

    name: str
    pins: tuple[str, ...]
    defaults: Mapping[str, Expression]
    body: tuple[Statement, ...]
    statement: Statement = field(repr=False)
    functions: Mapping[str, Function] = NO_FUNCTIONS


@dataclass(frozen=True, slots=True)
class GlobalPlan:
    """Partition of global `.param` definitions at one `(temperature, functions)`.

    `fixed` holds every deterministic parameter evaluated once (including ``temper``).
    `nominal` holds every parameter at its nominal value (same dict as `fixed` when
    `stochastic` is empty). `stochastic` holds inlined expressions that call a
    statistical function or transitively depend on one, in topological order.
    """

    temperature: float
    functions: dict[str, Function]
    fixed: dict[str, float]
    nominal: dict[str, float]
    stochastic: tuple[tuple[str, Expression], ...]


def dependency_graph(
    definitions: Mapping[str, Expression],
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Builds in-degrees and forward dependency edges over `definitions`."""
    in_degree: dict[str, int] = {}
    dependents: dict[str, list[str]] = {}
    keys = definitions.keys()
    for name, expression in definitions.items():
        dependencies = (names(expression) - {name}) & keys
        in_degree[name] = len(dependencies)
        for dependency in dependencies:
            dependents.setdefault(dependency, []).append(name)
    return in_degree, dependents


def topological_order(
    definitions: Mapping[str, Expression],
    in_degree: dict[str, int],
    dependents: Mapping[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Runs Kahn's topological sort and returns `(ordered, cycle_remaining)`."""
    remaining_degree = dict(in_degree)
    ready = deque(name for name in definitions if remaining_degree[name] == 0)
    ordered: list[str] = []
    while ready:
        current = ready.popleft()
        ordered.append(current)
        for dependent in dependents.get(current, ()):
            remaining_degree[dependent] -= 1
            if remaining_degree[dependent] == 0:
                ready.append(dependent)
    remaining = [name for name in definitions if remaining_degree[name] > 0]
    return ordered, remaining


def stochastic_closure(
    definitions: Mapping[str, Expression],
    dependents: Mapping[str, list[str]],
) -> set[str]:
    """Returns all parameters containing or depending on statistical calls."""
    roots = [
        name
        for name, expression in definitions.items()
        if any(is_statistical(node) for node in walk(expression))
    ]
    stochastic = set(roots)
    stack = list(roots)
    while stack:
        current = stack.pop()
        for dependent in dependents.get(current, ()):
            if dependent not in stochastic:
                stochastic.add(dependent)
                stack.append(dependent)
    return stochastic


def check_cycle(subcircuit: Subcircuit, instantiating: tuple[str, ...]) -> str:
    """Returns `subcircuit`'s case-folded key after checking it is not in a cycle."""
    key = subcircuit.name.casefold()
    if key in instantiating:
        cycle = " places ".join(
            f"`{entry}`" for entry in (*instantiating[instantiating.index(key) :], key)
        )
        message = f"a subcircuit places itself: {cycle}"
        raise subcircuit.statement.fail(message)
    return key


def extract_multiplier(
    subcircuit: Subcircuit, given: Mapping[str, float]
) -> tuple[float, Mapping[str, float]]:
    """Splits subcircuit multiplier `m` from `given` when it has no `m` parameter."""
    if "m" not in given or "m" in subcircuit.defaults:
        return 1.0, given
    multiplier = given["m"]
    if not (multiplier > 0 and math.isfinite(multiplier)):
        message = (
            f"the subcircuit multiplier `m` is positive and finite, not {multiplier!r}"
        )
        raise subcircuit.statement.fail(message)
    return multiplier, {key: value for key, value in given.items() if key != "m"}


def build_global_plan(
    parameters: Sequence[tuple[str, Expression]],
    temperature: float,
    functions: Mapping[str, Function],
) -> GlobalPlan:
    """Partitions and evaluates global parameters for `(temperature, functions)`."""
    functions_snapshot = dict(functions)
    inlined_definitions = {
        name: inlined(expression, functions_snapshot)
        for name, expression in dict(parameters).items()
    }
    in_degree, dependents = dependency_graph(inlined_definitions)
    ordered, remaining = topological_order(inlined_definitions, in_degree, dependents)
    stochastic_names = stochastic_closure(inlined_definitions, dependents)
    celsius_temperature = temperature - ZERO_CELSIUS
    fixed: dict[str, float] = {"temper": celsius_temperature}
    nominal_values: dict[str, float] = (
        {"temper": celsius_temperature} if stochastic_names else fixed
    )
    stochastic: list[tuple[str, Expression]] = []
    for name in ordered:
        expression = inlined_definitions[name]
        if name in stochastic_names:
            nominal_values[name] = evaluate(expression, nominal_values, nominal)
            stochastic.append((name, expression))
        else:
            value = evaluate(expression, fixed, nominal)
            fixed[name] = value
            if stochastic_names:
                nominal_values[name] = value
    if remaining:
        message = f"parameters read one another in a cycle: {', '.join(remaining)}"
        raise SpiceError(message)
    return GlobalPlan(
        temperature=temperature,
        functions=functions_snapshot,
        fixed=fixed,
        nominal=nominal_values,
        stochastic=tuple(stochastic),
    )


def evaluate_stochastic(plan: GlobalPlan, drawing: Drawing) -> dict[str, float]:
    """Evaluates only the stochastic subset of `plan` on top of `plan.fixed`."""
    values = plan.fixed.copy()
    for name, expression in plan.stochastic:
        values[name] = evaluate(expression, values, drawing.draws(name))
    return values


def resolve_into(
    definitions: Mapping[str, Expression],
    values: dict[str, float],
    draws: Callable[[str], Draw],
    functions: Mapping[str, Function],
    *,
    partial: bool = False,
) -> dict[str, float]:
    """Evaluates each parameter `definitions` defines into `values` in place.

    Raises:
        SpiceError: When parameters read one another in a cycle, or an expression is
            wrong.
    """
    if not definitions:
        return values
    pending = {
        name: inlined(expression, functions) for name, expression in definitions.items()
    }
    in_degree, dependents = dependency_graph(pending)
    ordered, remaining = topological_order(pending, in_degree, dependents)
    for name in ordered:
        try:
            values[name] = evaluate(pending[name], values, draws(name))
        except UndefinedError:
            if not partial:
                raise
    if remaining:
        message = f"parameters read one another in a cycle: {', '.join(remaining)}"
        raise SpiceError(message)
    return values


def split_assigned_pairs(words: Sequence[str]) -> tuple[list[str], list[str]]:
    """Splits `words` into non-assignment tokens and ``name = value`` triple tokens."""
    signs = {position for position, word in enumerate(words) if word == "="}
    paired = {position + step for position in signs for step in (-1, 0, 1)}
    positional = [word for place, word in enumerate(words) if place not in paired]
    assigned = [word for place, word in enumerate(words) if place in paired]
    return positional, assigned


def parse_options(statement: Statement) -> dict[str, Expression]:
    """Parses ``name = value`` options from an ``.option`` statement by folded name.

    Bare boolean flags without values are ignored because jolty does not read them.
    """
    words = tokens(statement.text)[1:]
    assigned = [
        position for position in range(len(words) - 2) if words[position + 1] == "="
    ]
    return models.assignments(
        statement,
        [word for position in assigned for word in words[position : position + 3]],
    )


def parse_function(statement: Statement) -> tuple[str, Function]:
    """Parses a ``.func`` `statement` into `(folded_name, Function)`.

    Raises:
        SpiceError: When `statement` is malformed or its body is not a valid
            expression.
    """
    found = _FUNCTION.fullmatch(statement.text.strip())
    if found is None:
        message = "a function is defined as `.func name(parameters) {expression}`"
        raise statement.fail(message)
    name, written, text = found.group("name", "parameters", "body")
    parameters = tuple(
        word.strip().casefold() for word in str(written).split(",") if word.strip()
    )
    try:
        body = parse(str(text))
    except SpiceError as error:
        raise statement.fail(str(error)) from None
    return str(name).casefold(), Function(parameters, body)


def parse_assignments(statement: Statement) -> list[tuple[str, Expression]]:
    """Parses ``.param`` assignments in source order as `(name, Expression)` pairs.

    Follows ngspice rules where each value extends up to the next assignment name,
    allowing unbraced spaced expressions such as ``.param a = 1 + b c = 2``.

    Raises:
        SpiceError: When tokens do not form valid ``name = value`` pairs.
    """
    words = tokens(statement.text)[1:]
    signs = [position for position, word in enumerate(words) if word == "="]
    if not signs or signs[0] != 1:
        message = "a parameter is assigned a value, `name = value`"
        raise statement.fail(message)
    ends = [*(sign - 1 for sign in signs[1:]), len(words)]
    found: list[tuple[str, Expression]] = []
    for sign, end in zip(signs, ends, strict=True):
        written = " ".join(words[sign + 1 : end])
        if not written:
            message = f"`{words[sign - 1]}` is not assigned a value"
            raise statement.fail(message)
        found.append((words[sign - 1].casefold(), models.value(written)))
    return found


def parse_pins_and_parameters(
    statement: Statement, words: list[str]
) -> tuple[list[str], dict[str, Expression]]:
    """Splits a `.subckt` header into pin names and default parameter expressions.

    Parameter defaults begin after ``PARAMS:`` or at the first ``name = value`` pair.
    """
    split = next(
        (
            index
            for index, word in enumerate(words)
            if word.casefold() == "params:"
            or (index + 1 < len(words) and words[index + 1] == "=")
        ),
        len(words),
    )
    pins = words[:split]
    rest = [word for word in words[split:] if word.casefold() != "params:"]
    return pins, models.assignments(statement, rest)


def split_keywords(words: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Splits element `words` into positional tokens and ``name = value`` lists.

    Each keyword collects all subsequent tokens until the next assignment, as in
    ``TC=1m,2u``.
    """
    positional: list[str] = []
    assigned: dict[str, list[str]] = {}
    current: list[str] | None = None
    position = 0
    while position < len(words):
        word = words[position]
        if position + 1 < len(words) and words[position + 1] == "=":
            current = assigned.setdefault(word.casefold(), [])
            position += 2
            continue
        if current is None:
            positional.append(word)
        else:
            current.append(word)
        position += 1
    return positional, assigned


def check_keywords(
    statement: Statement, assigned: Mapping[str, list[str]], allowed: frozenset[str]
) -> None:
    """Fails on any keyword not in `allowed`."""
    for name in assigned:
        if name not in allowed:
            message = f"jolty reads no `{name.upper()}` here"
            raise statement.fail(message)


def single_keyword(
    evaluate_word: Callable[[str], float],
    statement: Statement,
    name: str,
    words: Sequence[str],
) -> float:
    """Evaluates a single-valued keyword assignment, rejecting empty or extra tokens."""
    if not words:
        message = f"`{name.upper()}` is not assigned a value"
        raise statement.fail(message)
    if len(words) != 1:
        message = f"`{name.upper()}` accepts one value, not `{' '.join(words)}`"
        raise statement.fail(message)
    return evaluate_word(words[0])


def instance_keywords(
    evaluate_word: Callable[[str], float],
    statement: Statement,
    assigned: Mapping[str, list[str]],
    *,
    flags: frozenset[str] = frozenset({"off", "on"}),
    multi_valued: frozenset[str] = frozenset(),
) -> dict[str, float]:
    """Evaluates element keyword assignments, rejecting empty or stray tokens.

    Allows trailing initial-state flags in `flags` (such as ``OFF`` and ``ON``)
    and comma-separated multi-value keywords in `multi_valued` (such as bipolar
    ``IC=vbe,vce``).
    """
    values: dict[str, float] = {}
    for key, words in assigned.items():
        if not words:
            message = f"`{key.upper()}` is not assigned a value"
            raise statement.fail(message)
        if key in multi_valued:
            evaluated = [
                evaluate_word(word) for word in words if word.casefold() not in flags
            ]
            if not evaluated:
                message = f"`{key.upper()}` is not assigned a value"
                raise statement.fail(message)
            values[key] = evaluated[0]
            continue
        value_word, *extra = words
        stray = [word for word in extra if word.casefold() not in flags]
        if stray:
            message = f"`{key.upper()}` accepts one value, not `{' '.join(words)}`"
            raise statement.fail(message)
        values[key] = evaluate_word(value_word)
    return values

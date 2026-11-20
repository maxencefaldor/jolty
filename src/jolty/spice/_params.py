# Copyright 2026 Maxence Faldor
"""SPICE parameter scopes, statistical draws, ``.func``, and keyword tokenizers."""

import itertools
import math
import re
from collections import deque
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, override

from jolty import _core, ir
from jolty.errors import SpiceError
from jolty.spice import models
from jolty.spice.expressions import (
    Draw,
    Dual,
    DualDraw,
    Expression,
    Function,
    Scalar,
    UndefinedError,
    compact,
    evaluate,
    evaluate_dual,
    inlined,
    is_finite,
    is_statistical,
    names,
    nominal,
    parse,
    primal,
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

type CallKey = tuple[str, str]
"""Statistical call identifier `(placement_path, call_name)` matching `Sample`."""

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

        Args:
            scope: Hierarchical placement path scoping deterministic variate draws.
            name: Parameter or element name prefixing sequential call identifiers.

        Returns:
            `Draw` callback evaluating statistical calls deterministically for `self`.
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

    Args:
        function: Case-folded SPICE statistical function name.
        arguments: Evaluated numeric arguments passed to `function`.

    Returns:
        Non-negative standard deviation of the distribution defined by `arguments`.

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
class StatisticalRecorder:
    """Records SPICE statistical calls and seeds non-global draws as exact `Dual`s.

    Each encountered statistical call is recorded under `(scope, call_id)` with its
    function name and primal numeric arguments, matching [`Sample`][jolty.spice.Sample]
    keys. Non-global calls (`scope != ""`) with a positive finite standard deviation
    return a `Dual` seeded with unit derivative wrt `(scope, call_id)`, propagating
    exact first-order sensitivities in a single translation pass.
    """

    calls: dict[CallKey, tuple[str, tuple[float, ...]]] = field(default_factory=dict)

    def draw_dual(self, scope: str, name: str) -> DualDraw[CallKey]:
        """Returns a dual statistical callback recording calls in `name` at `scope`.

        Args:
            scope: Hierarchical placement path recorded with each encountered call.
            name: Parameter or element name prefixing sequential call identifiers.

        Returns:
            `DualDraw` callback recording calls into `self.calls` and seeding each
            active non-global call with unit derivative.
        """
        counter = itertools.count()

        def drawn(function: str, arguments: tuple[Dual[CallKey], ...]) -> Dual[CallKey]:
            call: CallKey = (scope, f"{name}#{next(counter)}")
            primal_arguments = tuple(argument.value for argument in arguments)
            self.calls[call] = (function, primal_arguments)
            nominal_argument = arguments[0]
            spread = standard_deviation(function, primal_arguments)
            if (
                not scope
                or spread == 0.0
                or not (math.isfinite(nominal_argument.value) and math.isfinite(spread))
            ):
                return nominal_argument
            derivatives = dict(nominal_argument.derivatives)
            derivatives[call] = derivatives.get(call, 0.0) + 1.0
            return Dual(nominal_argument.value, derivatives)

        return drawn


def _encode_path(path: str) -> str:
    """Encodes placement `path` for IR component names without collisions.

    Percent-encodes path separators, percent signs, whitespace, and non-printable
    characters as uppercase UTF-8 hex escapes so flat paths stay unchanged while
    nested paths such as ``X1/X2`` become ``X1%2FX2`` without colliding with
    ``X1_X2`` or literal ``X1%2FX2``.
    """
    return "".join(
        "".join(f"%{byte:02X}" for byte in character.encode())
        if character in {ir.SEPARATOR, "%"}
        or character.isspace()
        or not character.isprintable()
        else character
        for character in path
    )


@dataclass(frozen=True, slots=True)
class Drawing:
    """Statistical sampling context pairing an optional `Sample` with a circuit path.

    Statistical calls draw from `sample` at `path`, or return nominal values when
    `sample` is None.
    """

    sample: Sample | None
    path: str

    def draws(self, name: str) -> Draw:
        """Returns the `Draw` callback for statistical calls in definition `name`.

        Args:
            name: Parameter or element name whose statistical calls are sampled.

        Returns:
            `Draw` callback bound to `self.sample` and `self.path`, or `nominal`.
        """
        if self.sample is None:
            return nominal
        return self.sample.draw(self.path, name)

    def named(self, name: str) -> str:
        """Qualifies component `name` with path and sample identity to avoid clashes.

        Encodes `self.path` for IR component-name rules while leaving `self.path`
        untouched for statistical draw keys.

        Args:
            name: Base component name before sample qualification.

        Returns:
            Component name suffixed with encoded placement path and sample coordinates
            when sampled or probed.
        """
        match self.sample:
            case Sample(seed=seed, index=index):
                return f"{name}@{_encode_path(self.path)}#{seed}:{index}"
            case Probe():
                return f"{name}@{_encode_path(self.path)}"
        return name

    def within(self, instance: str) -> Drawing:
        """Returns a child `Drawing` scoped to nested placement `instance`.

        Args:
            instance: Child subcircuit instance name appended to `self.path`.

        Returns:
            Nested `Drawing` preserving `self.sample` at the extended path.
        """
        path = f"{self.path}{ir.SEPARATOR}{instance}" if self.path else instance
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


class TrackedScope(MutableMapping[str, float]):
    """Mutable parameter scope recording reads of temperature-dependent names."""

    __slots__ = ("_temperature_dependent", "_values", "read_temperature")

    def __init__(
        self,
        values: dict[str, float],
        temperature_dependent: frozenset[str],
    ) -> None:
        """Initializes a tracking view over `values` for `temperature_dependent`."""
        self._values = values
        self._temperature_dependent = temperature_dependent
        self.read_temperature = False

    @override
    def __getitem__(self, key: str) -> float:
        if key in self._temperature_dependent:
            self.read_temperature = True
        return self._values[key]

    @override
    def __setitem__(self, key: str, value: float) -> None:
        self._values[key] = value

    @override
    def __delitem__(self, key: str) -> None:
        del self._values[key]

    @override
    def __contains__(self, key: object) -> bool:
        return key in self._values

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    @override
    def __len__(self) -> int:
        return len(self._values)


@dataclass(frozen=True, slots=True)
class GlobalPlan:
    """Partition of global `.param` definitions at one `(temperature, functions)`.

    `fixed` holds every deterministic parameter evaluated once (including ``temper``).
    `nominal` holds every parameter at its nominal value (same dict as `fixed` when
    `stochastic` is empty). `stochastic` holds inlined expressions that call a
    statistical function or transitively depend on one, in topological order.
    `temperature_dependent` names every global parameter (including ``temper`` unless
    shadowed by a temperature-independent `.param`) whose value depends on library
    temperature.
    """

    temperature: float
    functions: dict[str, Function]
    fixed: dict[str, float]
    nominal: dict[str, float]
    stochastic: tuple[tuple[str, Expression], ...]
    temperature_dependent: frozenset[str] = frozenset()


def dependency_graph(
    definitions: Mapping[str, Expression],
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Builds in-degrees and forward dependency edges over `definitions`.

    Args:
        definitions: Parameter expressions keyed by case-folded parameter name.

    Returns:
        Pair of `(in_degree, dependents)` dictionaries over `definitions`.
    """
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
    """Runs Kahn's topological sort and returns `(ordered, cycle_remaining)`.

    Args:
        definitions: Parameter expressions keyed by case-folded parameter name.
        in_degree: Count of unresolved dependencies within `definitions` per parameter.
        dependents: Forward adjacency list mapping each parameter to dependents.

    Returns:
        Pair of `(ordered_names, cycle_remaining_names)`.
    """
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
    """Returns all parameters containing or depending on statistical calls.

    Args:
        definitions: Inlined parameter expressions keyed by case-folded name.
        dependents: Forward adjacency list mapping each parameter to dependents.

    Returns:
        Case-folded names of stochastic parameters and transitive dependents.
    """
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
    """Returns `subcircuit`'s case-folded key after checking it is not in a cycle.

    Args:
        subcircuit: Subcircuit definition about to be instantiated.
        instantiating: Active chain of case-folded enclosing subcircuit names.

    Returns:
        Case-folded name of `subcircuit`.
    """
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
    """Splits subcircuit multiplier `m` from `given` when it has no `m` parameter.

    Args:
        subcircuit: Subcircuit definition checked for a formal `m` parameter.
        given: Case-folded parameter overrides passed at the placement site.

    Returns:
        Pair of `(multiplier, remaining_parameters)`.
    """
    if "m" not in given or "m" in subcircuit.defaults:
        return 1.0, given
    multiplier = given["m"]
    if not (multiplier > 0 and math.isfinite(multiplier)):
        message = (
            f"the subcircuit multiplier `m` is positive and finite, not {multiplier!r}"
        )
        raise subcircuit.statement.fail(message)
    return multiplier, {key: value for key, value in given.items() if key != "m"}


def extract_multiplier_dual[Seed](
    subcircuit: Subcircuit, given: Mapping[str, Scalar[Seed]]
) -> tuple[Scalar[Seed], Mapping[str, Scalar[Seed]]]:
    """Splits dual subcircuit multiplier `m` from `given` when `m` is not formal.

    Args:
        subcircuit: Subcircuit definition checked for a formal `m` parameter.
        given: Case-folded scalar or dual parameter overrides passed at placement.

    Returns:
        Pair of `(multiplier, remaining_parameters)` preserving any derivatives on `m`.
    """
    if "m" not in given or "m" in subcircuit.defaults:
        return 1.0, given
    multiplier = given["m"]
    if not (primal(multiplier) > 0 and is_finite(multiplier)):
        message = (
            "the subcircuit multiplier `m` is positive and finite, not "
            f"{primal(multiplier)!r}"
        )
        raise subcircuit.statement.fail(message)
    return multiplier, {key: value for key, value in given.items() if key != "m"}


def temperature_closure(
    definitions: Mapping[str, Expression],
    dependents: Mapping[str, list[str]],
) -> frozenset[str]:
    """Returns global parameter names whose values depend on library temperature.

    Includes ``temper`` unless a global `.param` shadows it with an expression that
    does not read ``temper``, along with every parameter referencing a
    temperature-dependent name directly or transitively.

    Args:
        definitions: Inlined parameter expressions keyed by case-folded name.
        dependents: Forward adjacency list mapping each parameter to dependents.

    Returns:
        Case-folded names of temperature-dependent global parameters.
    """
    if "temper" in definitions:
        if "temper" not in names(definitions["temper"]):
            return frozenset()
        roots = ["temper"]
        dependent: set[str] = {"temper"}
    else:
        roots = [
            name
            for name, expression in definitions.items()
            if "temper" in names(expression)
        ]
        dependent = {"temper", *roots}
    stack = list(roots)
    while stack:
        current = stack.pop()
        for follower in dependents.get(current, ()):
            if follower not in dependent:
                dependent.add(follower)
                stack.append(follower)
    return frozenset(dependent)


def retarget_global_plan(plan: GlobalPlan, temperature: float) -> GlobalPlan | None:
    """Updates `plan` to `temperature` without re-evaluating global `.param` values.

    Succeeds only when no global `.param` depends on ``temper``. Returns `None` when
    at least one global parameter depends on temperature and full re-evaluation is
    required.

    Args:
        plan: Existing compiled global parameter plan at another temperature.
        temperature: New library temperature in kelvins.

    Returns:
        Updated `GlobalPlan` at `temperature`, or `None` when rebuild is required.
    """
    if plan.temperature_dependent - {"temper"}:
        return None
    if "temper" in plan.temperature_dependent:
        celsius_temperature = temperature - ZERO_CELSIUS
        fixed = {**plan.fixed, "temper": celsius_temperature}
        nominal_values = (
            {**plan.nominal, "temper": celsius_temperature}
            if plan.stochastic
            else fixed
        )
    else:
        fixed = plan.fixed
        nominal_values = plan.nominal
    return GlobalPlan(
        temperature=temperature,
        functions=plan.functions,
        fixed=fixed,
        nominal=nominal_values,
        stochastic=plan.stochastic,
        temperature_dependent=plan.temperature_dependent,
    )


def build_global_plan(
    parameters: Sequence[tuple[str, Expression]],
    temperature: float,
    functions: Mapping[str, Function],
) -> GlobalPlan:
    """Partitions and evaluates global parameters for `(temperature, functions)`.

    Args:
        parameters: Global `.param` `(name, expression)` assignments in source order.
        temperature: Library temperature in kelvins used to populate ``temper``.
        functions: Case-folded user-defined functions inlined into expressions.

    Returns:
        Compiled `GlobalPlan` separating deterministic and stochastic parameters.
    """
    functions_snapshot = dict(functions)
    inlined_definitions = {
        name: inlined(expression, functions_snapshot)
        for name, expression in dict(parameters).items()
    }
    in_degree, dependents = dependency_graph(inlined_definitions)
    ordered, remaining = topological_order(inlined_definitions, in_degree, dependents)
    stochastic_names = stochastic_closure(inlined_definitions, dependents)
    temperature_names = temperature_closure(inlined_definitions, dependents)
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
        temperature_dependent=temperature_names,
    )


def evaluate_stochastic(plan: GlobalPlan, drawing: Drawing) -> dict[str, float]:
    """Evaluates only the stochastic subset of `plan` on top of `plan.fixed`.

    Args:
        plan: Compiled global parameter plan.
        drawing: Statistical sampling context supplying draws per parameter name.

    Returns:
        Complete global parameter dictionary combining fixed and sampled values.
    """
    values = plan.fixed.copy()
    for name, expression in plan.stochastic:
        values[name] = evaluate(expression, values, drawing.draws(name))
    return values


def resolve_into[Scope: MutableMapping[str, float]](
    definitions: Mapping[str, Expression],
    values: Scope,
    draws: Callable[[str], Draw],
    functions: Mapping[str, Function],
    *,
    partial: bool = False,
) -> Scope:
    """Evaluates each parameter `definitions` defines into `values` in place.

    Args:
        definitions: Parameter expressions keyed by case-folded parameter name.
        values: Mutable parameter scope updated in topological dependency order.
        draws: Factory mapping each parameter name to its statistical `Draw` callback.
        functions: Case-folded user-defined functions inlined before evaluation.
        partial: When `True`, skips parameters raising `UndefinedError` instead of
            failing.

    Returns:
        Updated `values` dictionary populated with resolved parameters.

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


def resolve_into_dual[Seed](
    definitions: Mapping[str, Expression],
    values: MutableMapping[str, Scalar[Seed]],
    draws: Callable[[str], DualDraw[Seed] | None],
    functions: Mapping[str, Function],
) -> MutableMapping[str, Scalar[Seed]]:
    """Evaluates each parameter `definitions` defines into dual `values` in place.

    Args:
        definitions: Parameter expressions keyed by case-folded parameter name.
        values: Mutable scalar or dual parameter scope updated in topological order.
        draws: Factory mapping each parameter name to its optional `DualDraw` callback.
        functions: Case-folded user-defined functions inlined before evaluation.

    Returns:
        Updated `values` mapping populated with scalar or `Dual` parameter values.

    Raises:
        SpiceError: When parameters read one another in a cycle, or an expression fails.
    """
    if not definitions:
        return values
    pending = {
        name: inlined(expression, functions) for name, expression in definitions.items()
    }
    in_degree, dependents = dependency_graph(pending)
    ordered, remaining = topological_order(pending, in_degree, dependents)
    for name in ordered:
        values[name] = compact(evaluate_dual(pending[name], values, draws(name)))
    if remaining:
        message = f"parameters read one another in a cycle: {', '.join(remaining)}"
        raise SpiceError(message)
    return values


def split_assigned_pairs(words: Sequence[str]) -> tuple[list[str], list[str]]:
    """Splits `words` into non-assignment tokens and ``name = value`` triple tokens.

    Args:
        words: Tokenized SPICE statement words following the element name.

    Returns:
        Pair of `(positional_tokens, assigned_triple_tokens)`.
    """
    signs = {position for position, word in enumerate(words) if word == "="}
    paired = {position + step for position in signs for step in (-1, 0, 1)}
    positional = [word for place, word in enumerate(words) if place not in paired]
    assigned = [word for place, word in enumerate(words) if place in paired]
    return positional, assigned


def parse_options(statement: Statement) -> dict[str, Expression]:
    """Parses ``name = value`` options from an ``.option`` statement by folded name.

    Bare boolean flags without values are ignored because jolty does not read them.

    Args:
        statement: Logical ``.option`` or ``.options`` statement to parse.

    Returns:
        Mapping from case-folded option names to parsed value `Expression` nodes.
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

    Args:
        statement: Logical ``.func`` statement defining a user function.

    Returns:
        Pair of `(case_folded_name, Function)`.

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

    Args:
        statement: Logical ``.param`` or ``.params`` statement to parse.

    Returns:
        Ordered `(case_folded_name, Expression)` assignment pairs.

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

    Args:
        statement: Logical `.subckt` header statement used for error attribution.
        words: Tokens following `.subckt <name>` on the header line.

    Returns:
        Pair of `(pin_names, default_parameter_expressions)`.
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

    Args:
        words: Tokenized element arguments following the element name.

    Returns:
        Pair of `(positional_tokens, keyword_value_token_lists)`.
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
    """Fails on any keyword not in `allowed`.

    Args:
        statement: Element statement used for error attribution.
        assigned: Parsed keyword assignment dictionary keyed by case-folded name.
        allowed: Permitted case-folded keyword names for the element.
    """
    for name in assigned:
        if name not in allowed:
            message = f"jolty reads no `{name.upper()}` here"
            raise statement.fail(message)


def single_keyword[Value](
    evaluate_word: Callable[[str], Value],
    statement: Statement,
    name: str,
    words: Sequence[str],
) -> Value:
    """Evaluates a single-valued keyword assignment, rejecting empty or extra tokens.

    Args:
        evaluate_word: Evaluator converting a single value token to a `Value`.
        statement: Element statement used for error attribution.
        name: Case-folded keyword name being evaluated.
        words: Value tokens assigned to `name`.

    Returns:
        Evaluated value of `words[0]`.
    """
    if not words:
        message = f"`{name.upper()}` is not assigned a value"
        raise statement.fail(message)
    if len(words) != 1:
        message = f"`{name.upper()}` accepts one value, not `{' '.join(words)}`"
        raise statement.fail(message)
    return evaluate_word(words[0])


def instance_keywords[Value](
    evaluate_word: Callable[[str], Value],
    statement: Statement,
    assigned: Mapping[str, list[str]],
    *,
    flags: frozenset[str] = frozenset({"off", "on"}),
    multi_valued: frozenset[str] = frozenset(),
) -> dict[str, Value]:
    """Evaluates element keyword assignments, rejecting empty or stray tokens.

    Allows trailing initial-state flags in `flags` (such as ``OFF`` and ``ON``)
    and comma-separated multi-value keywords in `multi_valued` (such as bipolar
    ``IC=vbe,vce``).

    Args:
        evaluate_word: Evaluator converting each value token to a `Value`.
        statement: Element statement used for error attribution.
        assigned: Keyword token lists keyed by case-folded name.
        flags: Case-folded trailing state flags ignored after keyword values.
        multi_valued: Case-folded keywords accepting multiple comma-separated values.

    Returns:
        Evaluated keyword dictionary keyed by case-folded name.
    """
    values: dict[str, Value] = {}
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

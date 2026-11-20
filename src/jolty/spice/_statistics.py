# Copyright 2026 Maxence Faldor
"""Symbolic compilation of SPICE statistical expressions into engine IR graphs.

Compiles global and subcircuit-local SPICE statistical calls (``agauss``, ``gauss``,
``aunif``, ``unif``, and ``limit``) into [`ir.Statistical`][jolty.ir.Statistical]
step/target graphs evaluated directly inside Rust Monte Carlo workers.

Constant subexpressions and disabled PDK switches (such as ``MC_PR_SWITCH = 0`` or
``MC_MM_SWITCH = 0``) fold away before dead-step pruning, so deterministic subcircuits
emit no statistical model and carry zero runtime overhead.
"""

import itertools
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, cast

from jolty import ir
from jolty.errors import SpiceError
from jolty.spice import models
from jolty.spice._params import (
    GlobalPlan,
    dependency_graph,
    topological_order,
)
from jolty.spice.expressions import (
    STATISTICAL,
    Binary,
    Call,
    Conditional,
    Current,
    Expression,
    Function,
    Name,
    Number,
    Unary,
    UndefinedError,
    Voltage,
    check_core_depth,
    evaluate,
    fold,
    inlined,
    is_statistical,
    nominal,
    rebuild,
    translate,
    walk,
)
from jolty.spice.models import ModelCard
from jolty.spice.netlist import Statement
from jolty.units import ZERO_CELSIUS

_IDENTIFIER: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
"""Pattern matching a valid jolty expression identifier."""

_CONSTANTS: Final = {
    "pi": math.pi,
    "e": math.e,
    "temp": 27.0,
    "vt": (27.0 + 273.15) * 8.6173303e-5,
    "gmin": 1e-12,
}
"""Built-in SPICE constants matching `jolty.spice.expressions._CONSTANTS`."""

_ALWAYS_FINITE_CALLS: Final = frozenset(
    {
        "abs",
        "min",
        "max",
        "limit",
        "sin",
        "cos",
        "atan",
        "tanh",
        "sgn",
        "sign",
        "u",
        "stp",
        "uramp",
        "floor",
        "ceil",
        "int",
        "if",
    }
)
"""Functions that always return finite values on finite real inputs."""

_ALWAYS_FINITE_BINARY: Final = frozenset(
    {"+", "-", "*", "==", "!=", "<", "<=", ">", ">=", "&&", "||"}
)
"""Binary operators that always return finite values on finite real inputs."""

_BIPOLAR_MULTIPLIED: Final = (
    *("is", "ikf", "ik", "ikr", "ise", "isc", "itf", "cje", "cjc", "irb"),
    *("iss", "cjs", "csub", "ccs"),
)
"""Gummel-Poon BJT parameters multiplied by effective area."""

_BIPOLAR_DIVIDED: Final = ("rb", "rbm", "re", "rc")
"""Gummel-Poon BJT resistances divided by effective area."""

_MOSFET_SCALED: Final = {
    **dict.fromkeys(("w", "l", "pd", "ps", "sa", "sb", "sd", "sc"), 1),
    **dict.fromkeys(("ad", "as"), 2),
}
"""MOSFET instance parameters scaled by ``.option scale`` mapped to scale powers."""


def _unread(name: str) -> str:
    """Rejects node or source references inside parameter-only expressions."""
    message = f"a parameter's expression reads no voltage or current, not `{name}`"
    raise SpiceError(message)


def join_path(prefix: str, suffix: str) -> str:
    """Joins `prefix` and `suffix` with `/` when both are non-empty."""
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}{ir.SEPARATOR}{suffix}"


def _finite_combine(node: Expression, built: list[bool]) -> bool:
    """Combines child finiteness flags for compound `node`."""
    if isinstance(node, Unary):
        return node.operator in {"+", "-", "!"} and built[0]
    if isinstance(node, Binary):
        if node.operator in _ALWAYS_FINITE_BINARY:
            return built[0] and built[1]
        return (
            node.operator == "/"
            and isinstance(node.right, Number)
            and node.right.value != 0.0
            and math.isfinite(node.right.value)
            and built[0]
        )
    if isinstance(node, Conditional):
        return all(built)
    return (
        isinstance(node, Call) and node.function in _ALWAYS_FINITE_CALLS and all(built)
    )


def _is_always_finite(expression: Expression, finite_steps: frozenset[str]) -> bool:
    """Returns whether `expression` stays finite across all finite step values."""

    def leaf(node: Expression) -> bool | None:
        if isinstance(node, Number):
            return math.isfinite(node.value)
        if isinstance(node, Name):
            return node.name in finite_steps
        if isinstance(node, Voltage | Current):
            return False
        return None

    return fold(expression, leaf, _finite_combine)


def _is_constant(node: Expression, expected: float) -> bool:
    """Returns whether `node` is a numeric literal equal to `expected`."""
    return isinstance(node, Number) and node.value == expected


def _simplify_multiply(
    left: Expression, right: Expression, finite_steps: frozenset[str]
) -> Expression | None:
    """Applies multiplicative zero and identity rules."""
    if _is_constant(left, 0.0) and _is_always_finite(right, finite_steps):
        return Number(0.0)
    if _is_constant(right, 0.0) and _is_always_finite(left, finite_steps):
        return Number(0.0)
    if _is_constant(left, 1.0):
        return right
    if _is_constant(right, 1.0):
        return left
    return None


def _simplify_binary(
    operator: str,
    left: Expression,
    right: Expression,
    finite_steps: frozenset[str],
) -> Expression | None:
    """Applies algebraic identities for binary operations, or returns `None`."""
    result: Expression | None = None
    if operator == "*":
        result = _simplify_multiply(left, right, finite_steps)
    elif operator == "+":
        if _is_constant(left, 0.0):
            result = right
        elif _is_constant(right, 0.0):
            result = left
    elif operator == "-":
        if _is_constant(right, 0.0):
            result = left
        elif _is_constant(left, 0.0):
            result = Unary("-", right)
    elif operator == "/" and _is_constant(right, 1.0):
        result = left
    return result


def _simplify_node(
    node: Expression, built: list[Expression], finite_steps: frozenset[str]
) -> Expression:
    """Constant-folds and simplifies one lowered compound `node`."""
    if all(isinstance(child, Number) for child in built):
        return Number(evaluate(rebuild(node, built), {}, nominal))
    if isinstance(node, Conditional) or (
        isinstance(node, Call) and node.function == "if"
    ):
        condition, then_branch, otherwise = built
        if isinstance(condition, Number):
            return otherwise if condition.value == 0.0 else then_branch
    if isinstance(node, Unary):
        operand = built[0]
        if node.operator == "+":
            return operand
        if (
            node.operator == "-"
            and isinstance(operand, Unary)
            and operand.operator == "-"
        ):
            return operand.operand
    if isinstance(node, Binary):
        simplified = _simplify_binary(node.operator, built[0], built[1], finite_steps)
        if simplified is not None:
            return simplified
    return rebuild(node, built)


def _format_step_expression(
    expression: Expression, step_names: Mapping[str, str]
) -> str:
    """Formats a step-variable `expression` in the jolty expression language."""
    written = translate(expression, {}, _unread, _unread, symbolic=step_names)
    check_core_depth(written)
    return written


def _zero_spread_fold(
    function: ir.StatisticalFunction, arguments: Sequence[Expression]
) -> Expression | None:
    """Returns the nominal argument when a statistical call has constant zero spread."""
    nominal_arg, spread_arg = arguments[0], arguments[1]
    if function in {"agauss", "gauss"}:
        sigma_arg = arguments[2]
        if (isinstance(spread_arg, Number) and spread_arg.value <= 0.0) or (
            isinstance(sigma_arg, Number) and sigma_arg.value <= 0.0
        ):
            return nominal_arg
        if (
            function == "gauss"
            and isinstance(nominal_arg, Number)
            and nominal_arg.value == 0.0
        ):
            return nominal_arg
        return None
    if isinstance(spread_arg, Number) and spread_arg.value == 0.0:
        return nominal_arg
    if (
        function == "unif"
        and isinstance(nominal_arg, Number)
        and nominal_arg.value == 0.0
    ):
        return nominal_arg
    return None


@dataclass(frozen=True, slots=True)
class GlobalSymbolic:
    """Cached symbolic DAG compiled once from a library's `GlobalPlan.stochastic`."""

    steps: tuple[ir.StatisticalStep, ...]
    steps_by_name: Mapping[str, ir.StatisticalStep]
    symbols: Mapping[str, Expression]
    finite_steps: frozenset[str]


EMPTY_GLOBAL: Final = GlobalSymbolic(
    steps=(),
    steps_by_name=MappingProxyType({}),
    symbols=MappingProxyType({}),
    finite_steps=frozenset(),
)
"""Empty `GlobalSymbolic` reused whenever global parameters have no active draws."""


def build_global_symbolic(plan: GlobalPlan) -> GlobalSymbolic:
    """Compiles active stochastic parameters in `plan` into a reusable `GlobalSymbolic`.

    Parameters whose statistical draws fold away to constants (for example under
    ``MC_PR_SWITCH = 0``) are omitted from `symbols`.
    """
    if not plan.stochastic:
        return EMPTY_GLOBAL
    compiler = StatisticalCompiler(EMPTY_GLOBAL, step_prefix="g")
    for name, expression in plan.stochastic:
        lowered = compiler.lower(
            expression,
            plan.nominal,
            call_name=name,
            shared=True,
            relative_scope="",
        )
        if lowered is not None:
            compiler.bind_symbol(name, lowered)
    if not compiler.scope_symbols:
        return EMPTY_GLOBAL
    steps = tuple(compiler.steps)
    return GlobalSymbolic(
        steps=steps,
        steps_by_name=MappingProxyType({step.name: step for step in steps}),
        symbols=MappingProxyType(dict(compiler.scope_symbols)),
        finite_steps=frozenset(compiler.finite_steps),
    )


def _step_references(step: ir.StatisticalStep) -> set[str]:
    """Returns all step variable identifiers read via `p(<name>)` inside `step`."""
    texts: list[str]
    if isinstance(step, ir.StatisticalDraw):
        texts = [step.nominal, step.spread]
        if step.sigma is not None:
            texts.append(step.sigma)
    else:
        texts = [step.expression]
    found: set[str] = set()
    for text in texts:
        found.update(_PARAMETER_CALL.findall(text))
    return found


_PARAMETER_CALL: Final = re.compile(r"\bp\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
"""Matches `p(<name>)` references inside formatted jolty expression strings."""


def _binary(
    operator: str,
    left: Expression | None,
    left_nominal: float,
    right: Expression | None,
    right_nominal: float,
) -> Expression | None:
    """Combines two optional symbolic expressions with binary `operator`."""
    if left is None and right is None:
        return None
    left_node = Number(left_nominal) if left is None else left
    right_node = Number(right_nominal) if right is None else right
    simplified = _simplify_binary(operator, left_node, right_node, frozenset())
    return Binary(operator, left_node, right_node) if simplified is None else simplified


def _scaled(expression: Expression | None, factor: float) -> Expression | None:
    """Multiplies optional `expression` by constant `factor`."""
    if expression is None or factor == 1.0:
        return expression
    if factor == 0.0:
        return None
    return Binary("*", expression, Number(factor))


def _offset(expression: Expression | None, delta: float) -> Expression | None:
    """Adds constant `delta` to optional `expression`."""
    if expression is None or delta == 0.0:
        return expression
    return Binary("+", expression, Number(delta))


def _clamp_min(expression: Expression | None, floor: float) -> Expression | None:
    """Clamps optional `expression` from below at constant `floor`."""
    if expression is None:
        return None
    return Call("max", (expression, Number(floor)))


def _clamp_max(expression: Expression | None, ceiling: float) -> Expression | None:
    """Clamps optional `expression` from above at constant `ceiling`."""
    if expression is None:
        return None
    return Call("min", (expression, Number(ceiling)))


def _power(
    base: Expression | None,
    base_nominal: float,
    exponent: Expression | None,
    exponent_nominal: float,
) -> Expression | None:
    """Raises optional `base` to optional `exponent`."""
    if base is None and exponent is None:
        return None
    base_node = Number(base_nominal) if base is None else base
    exponent_node = Number(exponent_nominal) if exponent is None else exponent
    if isinstance(exponent_node, Number) and exponent_node.value == 1.0:
        return base_node
    if isinstance(exponent_node, Number) and exponent_node.value == 0.0:
        return None
    return Binary("^", base_node, exponent_node)


def _reject_stochastic(statement: Statement, name: str, context: str) -> None:
    """Raises `SpiceError` when a discrete or structural parameter varies."""
    message = (
        f"statistical variation on `{name.upper()}` {context} changes circuit "
        "structure or model selection and cannot compile to a fixed Monte Carlo graph"
    )
    raise statement.fail(message)


def _allocate_parameter_name(
    preferred: str | None,
    parameters: Mapping[str, float],
    generated: itertools.count[int],
) -> str:
    """Allocates a valid unused behavioral parameter identifier."""
    if (
        preferred is not None
        and _IDENTIFIER.fullmatch(preferred) is not None
        and preferred not in parameters
    ):
        return preferred
    while True:
        candidate = f"p{next(generated)}"
        if candidate not in parameters:
            return candidate


class StatisticalCompiler:
    """Compiles statistical parameter expressions for one translated component."""

    def __init__(
        self,
        global_symbolic: GlobalSymbolic,
        *,
        step_prefix: str = "s",
        relative_scope: str = "",
        instance_prefix: str = "",
    ) -> None:
        """Initializes a compiler seeded with `global_symbolic` symbols."""
        self.global_symbolic = global_symbolic
        self._step_prefix = step_prefix
        self._counter = itertools.count()
        self.relative_scope = relative_scope
        self.instance_prefix = instance_prefix
        self.steps: list[ir.StatisticalStep] = []
        self.targets: list[ir.StatisticalTarget] = []
        self.scope_symbols: dict[str, Expression] = dict(global_symbolic.symbols)
        self.finite_steps: set[str] = set(global_symbolic.finite_steps)
        self._step_names: dict[str, str] = {
            name: name for name in global_symbolic.steps_by_name
        }
        self._element_name: str = ""
        self._element_calls: itertools.count[int] = itertools.count()

    def allocate_external_step(self) -> str:
        """Allocates a finite placeholder step name for an inherited probe parameter."""
        step_name = self._allocate_step()
        self.finite_steps.add(step_name)
        return step_name

    def _allocate_step(self) -> str:
        """Allocates the next unique step variable identifier."""
        step_name = f"{self._step_prefix}{next(self._counter)}"
        self._step_names[step_name] = step_name
        return step_name

    def begin_element(self, name: str) -> None:
        """Starts call-counter tracking for SPICE element `name` (case-folded)."""
        self._element_name = name
        self._element_calls = itertools.count()

    def has_symbols(self) -> bool:
        """Returns whether any active stochastic parameter is currently in scope."""
        return bool(self.scope_symbols)

    def bind_symbol(self, name: str, expression: Expression) -> None:
        """Binds SPICE parameter `name` to symbolic `expression`."""
        if isinstance(expression, Name):
            self.scope_symbols[name] = expression
            return
        step_name = self._allocate_step()
        formatted = _format_step_expression(expression, self._step_names)
        self.steps.append(ir.StatisticalValue(name=step_name, expression=formatted))
        if _is_always_finite(expression, frozenset(self.finite_steps)):
            self.finite_steps.add(step_name)
        self.scope_symbols[name] = Name(step_name)

    def set_given(
        self,
        given: Mapping[str, float],
        given_symbolic: Mapping[str, Expression] | None = None,
    ) -> None:
        """Applies placement overrides `given` and optional `given_symbolic`."""
        symbolic = {} if given_symbolic is None else given_symbolic
        for key in given:
            if key in symbolic:
                self.scope_symbols[key] = symbolic[key]
            else:
                self.scope_symbols.pop(key, None)

    def resolve_into(
        self,
        definitions: Mapping[str, Expression],
        scope: Mapping[str, float],
        functions: Mapping[str, Function],
    ) -> None:
        """Lowers `definitions` in topological order matching `_params.resolve_into`."""
        if not definitions:
            return
        pending = {
            name: inlined(expression, functions)
            for name, expression in definitions.items()
        }
        in_degree, dependents = dependency_graph(pending)
        ordered, _ = topological_order(pending, in_degree, dependents)
        for name in ordered:
            if name not in scope:
                continue
            lowered = self.lower(pending[name], scope, call_name=name)
            if lowered is None:
                self.scope_symbols.pop(name, None)
            else:
                self.bind_symbol(name, lowered)

    def lower_element(
        self, expression: Expression, scope: Mapping[str, float]
    ) -> Expression | None:
        """Lowers an element `expression` under the active element call counter."""
        return self.lower(
            expression,
            scope,
            call_name=self._element_name,
            calls=self._element_calls,
        )

    def _lower_leaf(
        self, node: Expression, scope: Mapping[str, float]
    ) -> Expression | None:
        """Lowers one leaf `node` against active symbols, `scope`, and constants."""
        match node:
            case Number():
                return node
            case Name(name) if name in self.scope_symbols:
                return self.scope_symbols[name]
            case Name(name) if name in scope:
                return Number(scope[name])
            case Name(name) if name in _CONSTANTS:
                return Number(_CONSTANTS[name])
            case Name(name):
                message = f"no parameter is named `{name}`"
                raise UndefinedError(message, name)
            case Voltage() | Current():
                message = "a parameter's expression reads no voltage or current"
                raise SpiceError(message)
        return None

    def _emit_draw_step(  # noqa: PLR0913 - statistical draw metadata
        self,
        function: str,
        built: list[Expression],
        *,
        call_name: str,
        counter: itertools.count[int],
        shared: bool,
        draw_scope: str,
    ) -> Expression:
        """Validates, zero-folds, or emits one `ir.StatisticalDraw` step."""
        expected = STATISTICAL[function]
        if len(built) != expected:
            message = f"`{function}` takes {expected} arguments, not {len(built)}"
            raise SpiceError(message)
        call_id = f"{call_name}#{next(counter)}"
        function_name = cast(ir.StatisticalFunction, function)
        folded = _zero_spread_fold(function_name, built)
        if folded is not None:
            return folded
        step_name = self._allocate_step()
        nominal_text = _format_step_expression(built[0], self._step_names)
        spread_text = _format_step_expression(built[1], self._step_names)
        sigma_text = (
            _format_step_expression(built[2], self._step_names)
            if len(built) == 3  # noqa: PLR2004 - agauss/gauss sigma argument
            else None
        )
        self.steps.append(
            ir.StatisticalDraw(
                name=step_name,
                shared=shared,
                scope=draw_scope,
                call=call_id,
                function=function_name,
                nominal=nominal_text,
                spread=spread_text,
                sigma=sigma_text,
            )
        )
        current_finite = frozenset(self.finite_steps)
        if all(_is_always_finite(argument, current_finite) for argument in built):
            self.finite_steps.add(step_name)
        return Name(step_name)

    def lower(  # noqa: PLR0913 - expression lowering context
        self,
        expression: Expression,
        scope: Mapping[str, float],
        *,
        call_name: str,
        shared: bool = False,
        relative_scope: str | None = None,
        calls: itertools.count[int] | None = None,
    ) -> Expression | None:
        """Lowers `expression` to a step expression, or `None` when deterministic."""
        if not self._might_vary(expression):
            return None
        counter = itertools.count() if calls is None else calls
        draw_scope = (
            ""
            if shared
            else (self.relative_scope if relative_scope is None else relative_scope)
        )

        def leaf(node: Expression) -> Expression | None:
            return self._lower_leaf(node, scope)

        def combine(node: Expression, built: list[Expression]) -> Expression:
            if is_statistical(node) and isinstance(node, Call):
                return self._emit_draw_step(
                    node.function,
                    built,
                    call_name=call_name,
                    counter=counter,
                    shared=shared,
                    draw_scope=draw_scope,
                )
            return _simplify_node(node, built, frozenset(self.finite_steps))

        lowered = fold(expression, leaf, combine)
        return None if isinstance(lowered, Number) else lowered

    def depends_on_statistics(self, expression: Expression) -> bool:
        """Returns whether `expression` reads statistical calls or stochastic names."""
        return self._might_vary(expression)

    def _might_vary(self, expression: Expression) -> bool:
        """Returns whether `expression` contains statistical calls or active symbols."""
        for node in walk(expression):
            if is_statistical(node):
                return True
            if isinstance(node, Name) and node.name in self.scope_symbols:
                return True
        return False

    def add_target(
        self, instance: str, parameter: str, expression: Expression | None
    ) -> None:
        """Records a target device parameter expression relative to this placement."""
        if expression is None:
            return
        formatted = _format_step_expression(expression, self._step_names)
        full_instance = join_path(self.instance_prefix, instance)
        self.targets.append(
            ir.StatisticalTarget(
                instance=full_instance,
                parameter=parameter,
                expression=formatted,
            )
        )

    def _extract_behavioral_leaf(
        self,
        node: Expression,
        scope: Mapping[str, float],
        extra: Mapping[str, tuple[float, Expression]],
        bind_leaf: Callable[[str | None, float, Expression], Name],
    ) -> Expression | None:
        """Rewrites one behavioral leaf or inline statistical call into `Name`."""
        if is_statistical(node) and isinstance(node, Call):
            lowered = self.lower_element(node, scope)
            nominal_value = evaluate(node, scope, nominal)
            return (
                Number(nominal_value)
                if lowered is None
                else bind_leaf(None, nominal_value, lowered)
            )
        if not isinstance(node, Name):
            return None
        if node.name in extra:
            return bind_leaf(node.name, *extra[node.name])
        if node.name in self.scope_symbols:
            return bind_leaf(node.name, scope[node.name], self.scope_symbols[node.name])
        return None

    def extract_behavioral(
        self,
        element_name: str,
        expression: Expression,
        scope: Mapping[str, float],
        *,
        extra_parameters: Mapping[str, tuple[float, Expression]] | None = None,
    ) -> tuple[Expression, dict[str, float], dict[str, str]]:
        """Rewrites stochastic leaves in behavioral `expression` into device parameters.

        Returns `(rewritten_expression, nominal_parameters, symbolic_name_map)` and
        records one `StatisticalTarget` per extracted parameter.
        """
        extra = {} if extra_parameters is None else extra_parameters
        if not extra and not self._might_vary(expression):
            return expression, {}, {}
        parameters: dict[str, float] = {}
        symbolic_names: dict[str, str] = {}
        target_expressions: dict[str, Expression] = {}
        by_source_name: dict[str, str] = {}
        generated = itertools.count()

        def bind_leaf(
            source_name: str | None,
            nominal_value: float,
            step_expression: Expression,
        ) -> Name:
            if source_name is not None and source_name in by_source_name:
                return Name(by_source_name[source_name])
            parameter_name = _allocate_parameter_name(
                source_name, parameters, generated
            )
            if source_name is not None:
                by_source_name[source_name] = parameter_name
            parameters[parameter_name] = nominal_value
            symbolic_names[parameter_name] = parameter_name
            target_expressions[parameter_name] = step_expression
            return Name(parameter_name)

        rewritten = fold(
            expression,
            lambda node: self._extract_behavioral_leaf(node, scope, extra, bind_leaf),
            rebuild,
        )
        for parameter_name, step_expression in target_expressions.items():
            self.add_target(
                element_name, f"parameters.{parameter_name}", step_expression
            )
        return rewritten, parameters, symbolic_names

    def remap_coupled_inductors(
        self,
        coupled_name: str,
        winding_names: Sequence[str],
        winding_instances: Sequence[ir.Instance],
        coupling_expressions: Sequence[tuple[int, Expression | None]],
    ) -> None:
        """Remaps individual inductor targets onto merged `coupled_inductors`."""
        by_name = {
            join_path(self.instance_prefix, instance.name).casefold(): index
            for index, instance in enumerate(winding_instances)
        }
        del winding_names
        coupled_instance = join_path(self.instance_prefix, coupled_name)
        remapped: list[ir.StatisticalTarget] = []
        temperature_recorded: set[str] = set()
        for target in self.targets:
            winding_index = by_name.get(target.instance.casefold())
            if winding_index is None:
                remapped.append(target)
                continue
            if target.parameter in {"temperature", "temperature_offset"}:
                if target.parameter not in temperature_recorded:
                    temperature_recorded.add(target.parameter)
                    remapped.append(
                        ir.StatisticalTarget(
                            instance=coupled_instance,
                            parameter=target.parameter,
                            expression=target.expression,
                        )
                    )
                continue
            remapped.append(
                ir.StatisticalTarget(
                    instance=coupled_instance,
                    parameter=f"windings.{winding_index}.{target.parameter}",
                    expression=target.expression,
                )
            )
        self.targets = remapped
        for coupling_index, expression in coupling_expressions:
            self.add_target(
                coupled_name,
                f"couplings.{coupling_index}.coefficient",
                expression,
            )

    def hoisted_tag(
        self,
        given_symbolic: Mapping[str, Expression],
        multiplier_symbolic: Expression | None,
    ) -> str:
        """Builds a deterministic name suffix for a hoisted child placement."""
        finite_steps = frozenset(self.finite_steps)
        entries: list[str] = []
        for key in sorted(given_symbolic):
            flag = "" if _is_always_finite(given_symbolic[key], finite_steps) else "!"
            entries.append(f"{key}{flag}")
        if multiplier_symbolic is not None and "m" not in given_symbolic:
            flag = "" if _is_always_finite(multiplier_symbolic, finite_steps) else "!"
            entries.append(f"m{flag}")
        return ",".join(entries) or "hoisted"

    def child_state(
        self, instance_name: str, given_symbolic: Mapping[str, Expression]
    ) -> tuple[dict[str, Expression], str, str, str, itertools.count[int]]:
        """Switches active scope into a hoisted child, returning saved state."""
        saved = (
            self.scope_symbols,
            self.relative_scope,
            self.instance_prefix,
            self._element_name,
            self._element_calls,
        )
        self.scope_symbols = dict(self.global_symbolic.symbols)
        self.scope_symbols.update(given_symbolic)
        self.relative_scope = join_path(self.relative_scope, instance_name)
        self.instance_prefix = join_path(self.instance_prefix, instance_name)
        return saved

    def restore_state(
        self,
        saved: tuple[dict[str, Expression], str, str, str, itertools.count[int]],
    ) -> None:
        """Restores parent state after translating a hoisted child placement."""
        (
            self.scope_symbols,
            self.relative_scope,
            self.instance_prefix,
            self._element_name,
            self._element_calls,
        ) = saved

    def finalize(self) -> ir.Statistical | None:
        """Prunes unreachable steps and returns `ir.Statistical`, or `None`."""
        if not self.targets:
            return None
        needed: set[str] = set()
        for target in self.targets:
            needed.update(_PARAMETER_CALL.findall(target.expression))
        if not needed:
            return None
        kept_local: list[ir.StatisticalStep] = []
        for step in reversed(self.steps):
            if step.name in needed:
                kept_local.append(step)
                needed.update(_step_references(step))
        kept_local.reverse()
        kept_global: list[ir.StatisticalStep] = []
        for step in reversed(self.global_symbolic.steps):
            if step.name in needed:
                kept_global.append(step)
                needed.update(_step_references(step))
        kept_global.reverse()
        return ir.Statistical(
            steps=(*kept_global, *kept_local),
            targets=tuple(self.targets),
        )


def extract_probe_behavioral(
    expression: Expression,
    scope: Mapping[str, float],
    stochastic_names: frozenset[str],
    *,
    extra_parameters: Mapping[str, float] | None = None,
) -> tuple[Expression, dict[str, float], dict[str, str]]:
    """Extracts stochastic `Name` parameters for `Probe` behavioral translation."""
    extra = {} if extra_parameters is None else extra_parameters
    if not extra and not stochastic_names:
        return expression, {}, {}
    parameters: dict[str, float] = {}
    symbolic_names: dict[str, str] = {}
    by_source_name: dict[str, str] = {}
    generated = itertools.count()

    def bind(source_name: str, nominal_value: float) -> Name:
        if source_name not in by_source_name:
            parameter_name = _allocate_parameter_name(
                source_name, parameters, generated
            )
            by_source_name[source_name] = parameter_name
            parameters[parameter_name] = nominal_value
            symbolic_names[parameter_name] = parameter_name
        return Name(by_source_name[source_name])

    def leaf(node: Expression) -> Expression | None:
        if isinstance(node, Name):
            if node.name in extra:
                return bind(node.name, extra[node.name])
            if node.name in stochastic_names and node.name in scope:
                return bind(node.name, scope[node.name])
        return None

    rewritten = fold(expression, leaf, rebuild)
    return rewritten, parameters, symbolic_names


# ---------------------------------------------------------------------------
# Device-specific symbolic parameter mapping helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _DeviceContext:
    """Bundled symbolic translation state for a semiconductor or switch element."""

    compiler: StatisticalCompiler
    element_name: str
    card: ModelCard
    values: Mapping[str, float]
    symbolic_values: Mapping[str, Expression]
    instance: Mapping[str, float]
    symbolic_instance: Mapping[str, Expression]
    statement: Statement


@dataclass(frozen=True, slots=True)
class _GeometricContext:
    """Bundled symbolic translation state for a SPICE3 geometric passive card."""

    card: ModelCard
    values: Mapping[str, float]
    card_symbolic: Mapping[str, Expression]
    instance: Mapping[str, float]
    instance_symbolic: Mapping[str, Expression]
    scale_option: float
    statement: Statement


def _record_passive_flicker(  # noqa: PLR0913 - passive flicker scaling context
    compiler: StatisticalCompiler,
    element_name: str,
    given: Mapping[str, float],
    drift_symbolic: Mapping[str, Expression],
    *,
    copies: float,
    copies_symbolic: Expression | None,
    statement: Statement,
) -> None:
    """Records symbolic resistor flicker-noise targets scaled by `copies`."""
    coefficient = given.get("kf", 0.0)
    coefficient_symbolic = drift_symbolic.get("kf")
    if coefficient == 0.0:
        if coefficient_symbolic is not None:
            _reject_stochastic(statement, "kf", "on a resistor with nominal KF = 0")
        return
    exponent = given.get("af", 1.0)
    exponent_symbolic = drift_symbolic.get("af")
    one_minus_exponent = (
        None
        if exponent_symbolic is None
        else Binary("-", Number(1.0), exponent_symbolic)
    )
    factor_symbolic = _power(
        copies_symbolic, copies, one_minus_exponent, 1.0 - exponent
    )
    scaled_coefficient = _binary(
        "*",
        coefficient_symbolic,
        coefficient,
        factor_symbolic,
        math.pow(copies, 1.0 - exponent),
    )
    compiler.add_target(element_name, models.FLICKER["kf"][0], scaled_coefficient)
    if exponent_symbolic is not None:
        compiler.add_target(element_name, models.FLICKER["af"][0], exponent_symbolic)


def _pspice_passive_symbolic(  # noqa: PLR0913 - non-geometric passive inputs
    element_letter: str,
    card: ModelCard | None,
    *,
    own: Mapping[str, float],
    own_symbolic: Mapping[str, Expression],
    card_values: Mapping[str, float],
    card_symbolic: Mapping[str, Expression],
) -> tuple[float, Expression | None, dict[str, Expression]]:
    """Computes pre-multiplicity nominal, symbolic value, and card drift symbols."""
    base_value = own[element_letter]
    base_symbolic = own_symbolic.get(element_letter)
    scale_value = own.get("scale", 1.0)
    scale_symbolic = own_symbolic.get("scale")
    value_symbolic = _binary(
        "*", base_symbolic, base_value, scale_symbolic, scale_value
    )
    value_nominal = base_value * scale_value
    if card is None:
        return value_nominal, value_symbolic, {}
    factor_value = card_values.get(element_letter, 1.0)
    factor_symbolic = card_symbolic.get(element_letter)
    value_symbolic = _binary(
        "*", value_symbolic, value_nominal, factor_symbolic, factor_value
    )
    value_nominal *= factor_value
    card_drift = {
        key: expression
        for key, expression in card_symbolic.items()
        if key != element_letter
    }
    return value_nominal, value_symbolic, card_drift


def record_passive_targets(  # noqa: PLR0913 - passive translation inputs
    compiler: StatisticalCompiler,
    element_name: str,
    model: str,
    parameter: str,
    *,
    card: ModelCard | None,
    own: Mapping[str, float],
    own_symbolic: Mapping[str, Expression],
    card_values: Mapping[str, float],
    card_symbolic: Mapping[str, Expression],
    given: Mapping[str, float],
    given_symbolic: Mapping[str, Expression],
    temperatures_symbolic: Mapping[str, Expression],
    copies: float,
    copies_symbolic: Expression | None,
    scale_option: float,
    statement: Statement,
) -> None:
    """Records exact symbolic targets for non-behavioral `R`, `C`, and `L` elements."""
    element_letter = model[0]
    if card is not None and card.kind in models.GEOMETRIC:
        context = _GeometricContext(
            card,
            card_values,
            card_symbolic,
            own,
            own_symbolic,
            scale_option,
            statement,
        )
        value_symbolic, card_drift = _geometric_symbolic(context)
        value_nominal_pre = models.geometric(card, card_values, own, statement).value
    else:
        value_nominal_pre, value_symbolic, card_drift = _pspice_passive_symbolic(
            element_letter,
            card,
            own=own,
            own_symbolic=own_symbolic,
            card_values=card_values,
            card_symbolic=card_symbolic,
        )
    drift_symbolic = {**card_drift, **given_symbolic}
    operator = "*" if model == "capacitor" else "/"
    final_value = _binary(
        operator, value_symbolic, value_nominal_pre, copies_symbolic, copies
    )
    compiler.add_target(element_name, parameter, final_value)
    _record_nominal_temperature(
        compiler, element_name, drift_symbolic, allow_tref=model == "resistor"
    )
    _record_own_temperature(
        compiler, element_name, drift_symbolic, temperatures_symbolic
    )
    for spice_name, jolty_name in (
        *models.PASSIVE_DRIFT.items(),
        *models.PASSIVE_CURVATURE.items(),
    ):
        if spice_name in drift_symbolic:
            compiler.add_target(element_name, jolty_name, drift_symbolic[spice_name])
    if model == "resistor":
        _record_passive_flicker(
            compiler,
            element_name,
            given,
            drift_symbolic,
            copies=copies,
            copies_symbolic=copies_symbolic,
            statement=statement,
        )


def record_behavioral_resistor_targets(
    compiler: StatisticalCompiler,
    element_name: str,
    drift_symbolic: Mapping[str, Expression],
    temperatures_symbolic: Mapping[str, Expression],
) -> None:
    """Records symbolic temperature and drift targets for a `behavioral_resistor`."""
    _record_nominal_temperature(
        compiler, element_name, drift_symbolic, allow_tref=True
    )
    _record_own_temperature(
        compiler, element_name, drift_symbolic, temperatures_symbolic
    )
    for spice_name in ("tc1", "tc2"):
        if spice_name in drift_symbolic:
            compiler.add_target(
                element_name, models.PASSIVE_DRIFT[spice_name], drift_symbolic[spice_name]
            )


def _geometric_resistor_symbolic(
    context: _GeometricContext,
    width_value: float,
    width_symbolic: Expression | None,
    drift_symbolic: dict[str, Expression],
) -> tuple[float, Expression | None]:
    """Computes unscaled nominal and symbolic resistance for a geometric `R` card."""
    for discrete in ("noise", "ef"):
        if discrete in context.card_symbolic:
            _reject_stochastic(context.statement, discrete, f"of `{context.card.name}`")
    length_value = context.instance.get(
        "l", context.values.get("defl", models.DEFAULT_RESISTOR_LENGTH)
    )
    length_symbolic = (
        _scaled(context.instance_symbolic.get("l"), context.scale_option)
        if "l" in context.instance
        else context.card_symbolic.get("defl")
    )
    short_value = context.values.get("short", 0.0)
    narrow_value = context.values.get("narrow", 0.0)
    long_value = length_value - 2.0 * short_value
    long_symbolic = _binary(
        "-",
        length_symbolic,
        length_value,
        _scaled(context.card_symbolic.get("short"), 2.0),
        2.0 * short_value,
    )
    wide_value = width_value - 2.0 * narrow_value
    wide_symbolic = _binary(
        "-",
        width_symbolic,
        width_value,
        _scaled(context.card_symbolic.get("narrow"), 2.0),
        2.0 * narrow_value,
    )
    sheet_value = context.values.get("rsh", 0.0)
    if "r" in context.instance:
        value_nominal = context.instance["r"]
        value_symbolic = context.instance_symbolic.get("r")
    elif length_value * width_value * sheet_value > 0.0:
        ratio_symbolic = _binary(
            "/", long_symbolic, long_value, wide_symbolic, wide_value
        )
        value_nominal = long_value / wide_value * sheet_value
        value_symbolic = _binary(
            "*",
            ratio_symbolic,
            long_value / wide_value,
            context.card_symbolic.get("rsh"),
            sheet_value,
        )
    else:
        value_nominal = context.values.get("r", models.MIN_RESISTANCE)
        value_symbolic = context.card_symbolic.get("r")
    sized = "w" in context.instance or "l" in context.instance
    if "kf" in context.values and sized:
        length_exponent = context.values.get("lf", 1.0)
        width_exponent = context.values.get("wf", 1.0)
        long_power = _power(
            long_symbolic,
            long_value,
            context.card_symbolic.get("lf"),
            length_exponent,
        )
        wide_power = _power(
            wide_symbolic,
            wide_value,
            context.card_symbolic.get("wf"),
            width_exponent,
        )
        long_power_value = math.pow(long_value, length_exponent)
        wide_power_value = math.pow(wide_value, width_exponent)
        area_symbolic = _binary(
            "*", long_power, long_power_value, wide_power, wide_power_value
        )
        flicker_symbolic = _binary(
            "/",
            drift_symbolic.get("kf"),
            context.values["kf"],
            area_symbolic,
            long_power_value * wide_power_value,
        )
        if flicker_symbolic is not None:
            drift_symbolic["kf"] = flicker_symbolic
    return value_nominal, value_symbolic


def _geometric_capacitor_symbolic(
    context: _GeometricContext,
    width_value: float,
    width_symbolic: Expression | None,
) -> tuple[float, Expression | None]:
    """Computes unscaled nominal and symbolic capacitance for a geometric `C` card."""
    length_value = context.instance.get("l", 0.0)
    length_symbolic = (
        _scaled(context.instance_symbolic.get("l"), context.scale_option)
        if "l" in context.instance
        else None
    )
    cut_value = context.values.get("del")
    cut_symbolic = _scaled(context.card_symbolic.get("del"), 2.0)
    short_default = 0.0 if cut_value is None else 2.0 * cut_value
    short_value = context.values.get("short", short_default)
    short_symbolic = (
        context.card_symbolic.get("short")
        if "short" in context.values
        else cut_symbolic
    )
    narrow_value = context.values.get("narrow", short_default)
    narrow_symbolic = (
        context.card_symbolic.get("narrow")
        if "narrow" in context.values
        else cut_symbolic
    )
    long_value = length_value - short_value
    long_symbolic = _binary(
        "-", length_symbolic, length_value, short_symbolic, short_value
    )
    wide_value = width_value - narrow_value
    wide_symbolic = _binary(
        "-", width_symbolic, width_value, narrow_symbolic, narrow_value
    )
    thick_value = context.values.get("thick", 0.0)
    relative_value = context.values.get("di", models.SILICA_RELATIVE_PERMITTIVITY)
    permittivity_value = relative_value * models.VACUUM_PERMITTIVITY
    permittivity_symbolic = _scaled(
        context.card_symbolic.get("di"), models.VACUUM_PERMITTIVITY
    )
    if "cj" in context.values:
        area_cap_value = context.values["cj"]
        area_cap_symbolic = context.card_symbolic.get("cj")
    elif thick_value > 0.0:
        area_cap_value = permittivity_value / thick_value
        area_cap_symbolic = _binary(
            "/",
            permittivity_symbolic,
            permittivity_value,
            context.card_symbolic.get("thick"),
            thick_value,
        )
    else:
        area_cap_value = 0.0
        area_cap_symbolic = None
    if "c" in context.instance:
        return context.instance["c"], context.instance_symbolic.get("c")
    if "cap" in context.values:
        return context.values["cap"], context.card_symbolic.get("cap")
    area_value = wide_value * long_value
    area_symbolic = _binary("*", wide_symbolic, wide_value, long_symbolic, long_value)
    plate_value = area_cap_value * area_value
    plate_symbolic = _binary(
        "*", area_cap_symbolic, area_cap_value, area_symbolic, area_value
    )
    perimeter_value = 2.0 * (long_value + wide_value)
    perimeter_symbolic = _scaled(
        _binary("+", long_symbolic, long_value, wide_symbolic, wide_value), 2.0
    )
    sidewall_value = context.values.get("cjsw", 0.0)
    edge_value = sidewall_value * perimeter_value
    edge_symbolic = _binary(
        "*",
        context.card_symbolic.get("cjsw"),
        sidewall_value,
        perimeter_symbolic,
        perimeter_value,
    )
    return (
        plate_value + edge_value,
        _binary("+", plate_symbolic, plate_value, edge_symbolic, edge_value),
    )


def _geometric_symbolic(
    context: _GeometricContext,
) -> tuple[Expression | None, dict[str, Expression]]:
    """Computes symbolic resistance or capacitance and drift for a geometric card."""
    width_value = context.instance.get(
        "w", context.values.get("defw", models.DEFAULT_WIDTH)
    )
    width_symbolic = (
        _scaled(context.instance_symbolic.get("w"), context.scale_option)
        if "w" in context.instance
        else context.card_symbolic.get("defw")
    )
    drift_symbolic = {
        name: expression
        for name, expression in context.card_symbolic.items()
        if name in {"tc1", "tc2", "tce", "tnom", "kf", "af"}
    }
    if context.card.kind == "r":
        value_nominal, value_symbolic = _geometric_resistor_symbolic(
            context, width_value, width_symbolic, drift_symbolic
        )
    else:
        value_nominal, value_symbolic = _geometric_capacitor_symbolic(
            context, width_value, width_symbolic
        )
    scale_value = context.instance.get("scale", 1.0)
    scale_symbolic = context.instance_symbolic.get("scale")
    scaled_value = _binary(
        "*", value_symbolic, value_nominal, scale_symbolic, scale_value
    )
    return scaled_value, drift_symbolic


def _dispatch_device_targets(model: str, context: _DeviceContext) -> None:
    """Dispatches symbolic target recording for a translated semiconductor or switch."""
    match model:
        case "diode":
            _record_diode(context)
        case "jfet":
            _record_jfet(context)
        case "bipolar_transistor":
            _record_bipolar(context)
        case "vbic":
            _record_vbic(context)
        case "mosfet":
            _record_mosfet(context)
        case "vdmos":
            _record_vdmos(context)
        case "bsim3" | "bsim4":
            _record_bsim(context, model)
        case "voltage_controlled_switch":
            _record_mapped(
                context.compiler,
                context.element_name,
                context.symbolic_values,
                models.SWITCH,
            )
        case "current_controlled_switch":
            _record_mapped(
                context.compiler,
                context.element_name,
                context.symbolic_values,
                models.CURRENT_SWITCH,
            )


def record_device_targets(  # noqa: PLR0913 - model card symbolic translation inputs
    compiler: StatisticalCompiler,
    element_name: str,
    card: ModelCard,
    *,
    card_values: Mapping[str, float],
    card_symbolic: Mapping[str, Expression],
    instance: Mapping[str, float],
    instance_symbolic: Mapping[str, Expression],
    made: models.Made,
    statement: Statement,
) -> None:
    """Records exact symbolic targets for a semiconductor or switch `made` device."""
    if not card_symbolic and not instance_symbolic:
        return
    _record_own_temperature(compiler, element_name, card_symbolic, instance_symbolic)
    values = {
        key: value
        for key, value in card_values.items()
        if key not in models.OWN_TEMPERATURES
    }
    symbolic_values = {
        key: expression
        for key, expression in card_symbolic.items()
        if key not in models.OWN_TEMPERATURES
    }
    instance_values = {
        key: value
        for key, value in instance.items()
        if key not in models.OWN_TEMPERATURES
    }
    symbolic_instance = {
        key: expression
        for key, expression in instance_symbolic.items()
        if key not in models.OWN_TEMPERATURES
    }
    if not symbolic_values and not symbolic_instance:
        return
    context = _DeviceContext(
        compiler,
        element_name,
        card,
        values,
        symbolic_values,
        instance_values,
        symbolic_instance,
        statement,
    )
    _dispatch_device_targets(made.model, context)


def record_vbic_heating_targets(  # noqa: PLR0913 - VBIC thermal symbolic inputs
    compiler: StatisticalCompiler,
    element_name: str,
    card: ModelCard,
    *,
    card_values: Mapping[str, float],
    card_symbolic: Mapping[str, Expression],
    instance: Mapping[str, float],
    instance_symbolic: Mapping[str, Expression],
    heated: bool,
    statement: Statement,
) -> None:
    """Records symbolic targets for VBIC self-heating `.rth` and `.cth` devices."""
    if "selft" in card_symbolic:
        _reject_stochastic(statement, "selft", f"of `{card.name}`")
    if not heated:
        if "rth" in card_symbolic:
            _reject_stochastic(
                statement, "rth", f"of `{card.name}` without self-heating"
            )
        return
    area_value, area_symbolic = _area_expression(instance, instance_symbolic)
    rth_value = card_values.get("rth", 0.0)
    rth_symbolic = card_symbolic.get("rth")
    cth_value = max(card_values.get("cth", 0.0), models.VBIC_MIN_THERMAL_CAPACITANCE)
    cth_symbolic = _clamp_min(
        card_symbolic.get("cth"), models.VBIC_MIN_THERMAL_CAPACITANCE
    )
    compiler.add_target(
        f"{element_name}.rth",
        "resistance",
        _binary("/", rth_symbolic, rth_value, area_symbolic, area_value),
    )
    compiler.add_target(
        f"{element_name}.cth",
        "capacitance",
        _binary("*", cth_symbolic, cth_value, area_symbolic, area_value),
    )


def scale_mosfet_instance_symbolic(
    instance_symbolic: Mapping[str, Expression], scale: float
) -> dict[str, Expression]:
    """Scales MOSFET instance symbolic parameters by ``scale ** power``."""
    if scale == 1.0 or not instance_symbolic:
        return dict(instance_symbolic)
    return {
        key: (
            Binary("*", expression, Number(math.pow(scale, power)))
            if (power := _MOSFET_SCALED.get(key, 0))
            else expression
        )
        for key, expression in instance_symbolic.items()
    }


def _record_nominal_temperature(
    compiler: StatisticalCompiler,
    element_name: str,
    symbolic: Mapping[str, Expression],
    *,
    allow_tref: bool = False,
) -> None:
    """Records `nominal_temperature` target when `TNOM`/`T_MEASURED`/`TREF` varies."""
    keys = (
        models.NOMINAL_TEMPERATURES_WITH_TREF
        if allow_tref
        else models.NOMINAL_TEMPERATURES
    )
    for key in keys:
        if key in symbolic:
            compiler.add_target(
                element_name,
                "nominal_temperature",
                _offset(symbolic[key], ZERO_CELSIUS),
            )
            return


def _record_own_temperature(
    compiler: StatisticalCompiler,
    element_name: str,
    card_symbolic: Mapping[str, Expression],
    instance_symbolic: Mapping[str, Expression],
) -> None:
    """Records per-device `temperature` or `temperature_offset` target when varying."""
    for source in (card_symbolic, instance_symbolic):
        for key in ("t_abs", "temp"):
            if key in source:
                compiler.add_target(
                    element_name,
                    "temperature",
                    _offset(source[key], ZERO_CELSIUS),
                )
        for key in ("t_rel_global", "dtemp"):
            if key in source:
                compiler.add_target(element_name, "temperature_offset", source[key])


def _area_expression(
    instance: Mapping[str, float], instance_symbolic: Mapping[str, Expression]
) -> tuple[float, Expression | None]:
    """Computes effective `area * m` nominal value and optional symbolic expression."""
    area_value = instance.get("area", 1.0)
    copies_value = instance.get("m", 1.0)
    area_symbolic = instance_symbolic.get("area")
    copies_symbolic = instance_symbolic.get("m")
    return (
        area_value * copies_value,
        _binary("*", area_symbolic, area_value, copies_symbolic, copies_value),
    )


def _area_scaled_symbolic(  # noqa: PLR0913 - area scaling specification
    model: str,
    values: Mapping[str, float],
    symbolic_values: Mapping[str, Expression],
    *,
    area_value: float,
    area_symbolic: Expression | None,
    multiplied: tuple[str, ...],
    divided: tuple[str, ...],
) -> dict[str, Expression]:
    """Applies area scaling to symbolic card parameters, populating nonzero defaults."""
    if area_value == 1.0 and area_symbolic is None:
        return dict(symbolic_values)
    table = {
        "diode": models.DIODE,
        "jfet": models.JFET,
        "bipolar_transistor": models.BIPOLAR,
        "vbic": models.VBIC,
    }[model]
    defaults = models._defaults(model)  # noqa: SLF001 - shared cached default lookup
    filled = dict(values)
    given = {table[name][0] for name in values if name in table}
    for name in (*multiplied, *divided):
        if name not in table:
            continue
        parameter, factor = table[name]
        default = defaults.get(parameter)
        if parameter in given or not isinstance(default, float) or default == 0.0:
            continue
        filled[name] = default / factor
        given.add(parameter)
    scaled: dict[str, Expression] = {}
    for name, nominal_value in filled.items():
        expression = symbolic_values.get(name)
        if name in multiplied:
            result = _binary("*", expression, nominal_value, area_symbolic, area_value)
        elif name in divided:
            result = _binary("/", expression, nominal_value, area_symbolic, area_value)
        else:
            result = expression
        if result is not None:
            scaled[name] = result
    return scaled


def _record_mapped(  # noqa: PLR0913 - mapped table recording parameters
    compiler: StatisticalCompiler,
    element_name: str,
    symbolic: Mapping[str, Expression],
    table: models.Table,
    *,
    skipped: frozenset[str] = frozenset(),
    allow_tref: bool = False,
) -> None:
    """Records mapped SI targets for all symbolic SPICE parameters in `symbolic`."""
    nominal_keys = (
        models.NOMINAL_TEMPERATURES_WITH_TREF
        if allow_tref
        else models.NOMINAL_TEMPERATURES
    )
    _record_nominal_temperature(
        compiler,
        element_name,
        {key: value for key, value in symbolic.items() if key not in skipped},
        allow_tref=allow_tref,
    )
    for name, expression in symbolic.items():
        if name in skipped or name in nominal_keys or name not in table:
            continue
        jolty_name, factor = table[name]
        compiler.add_target(element_name, jolty_name, _scaled(expression, factor))


def _record_jfet(context: _DeviceContext) -> None:
    """Records symbolic targets for a JFET device."""
    area_value, area_symbolic = _area_expression(
        context.instance, context.symbolic_instance
    )
    scaled = _area_scaled_symbolic(
        "jfet",
        context.values,
        context.symbolic_values,
        area_value=area_value,
        area_symbolic=area_symbolic,
        multiplied=("beta", "is", "cgs", "cgd"),
        divided=("rd", "rs"),
    )
    _record_mapped(context.compiler, context.element_name, scaled, models.JFET)


def _diode_geometry_expressions(
    context: _DeviceContext,
    *,
    geometric: bool,
    geometry: Mapping[str, float],
) -> tuple[float, Expression | None, float, Expression | None]:
    """Computes nominal and symbolic `(area, perimeter)` for a diode placement."""
    sized = "w" in context.instance or "l" in context.instance
    area_value = context.instance.get("area", 1.0 if sized else geometry["area"])
    area_symbolic = (
        context.symbolic_instance.get("area")
        if "area" in context.instance
        else (None if sized else context.symbolic_values.get("area"))
    )
    perimeter_value = context.instance.get("pj", 0.0 if sized else geometry["pj"])
    perimeter_symbolic = (
        context.symbolic_instance.get("pj")
        if "pj" in context.instance
        else (None if sized else context.symbolic_values.get("pj"))
    )
    if geometric and "w" in context.instance and "l" in context.instance:
        scale = context.instance.get("scale", 1.0)
        offset_value = geometry["xw"]
        offset_symbolic = context.symbolic_values.get("xw")
        width_value = context.instance["w"] + offset_value
        length_value = context.instance["l"] + offset_value
        width_symbolic = _binary(
            "+",
            context.symbolic_instance.get("w"),
            context.instance["w"],
            offset_symbolic,
            offset_value,
        )
        length_symbolic = _binary(
            "+",
            context.symbolic_instance.get("l"),
            context.instance["l"],
            offset_symbolic,
            offset_value,
        )
        area_value = width_value * length_value * (scale**2)
        area_symbolic = _scaled(
            _binary("*", width_symbolic, width_value, length_symbolic, length_value),
            scale**2,
        )
        perimeter_value = (2.0 * width_value + 2.0 * length_value) * scale
        perimeter_symbolic = _scaled(
            _binary("+", width_symbolic, width_value, length_symbolic, length_value),
            2.0 * scale,
        )
    return area_value, area_symbolic, perimeter_value, perimeter_symbolic


def _record_diode(context: _DeviceContext) -> None:
    """Records symbolic targets for a level-1 or level-3 diode device."""
    for discrete in ("level", *models.DIODE_SELECTORS, *models.METAL_AND_POLY):
        if discrete in context.symbolic_values or discrete in context.symbolic_instance:
            _reject_stochastic(context.statement, discrete, f"of `{context.card.name}`")
    level = context.values.get("level", 1.0)
    geometric = level == 3.0  # noqa: PLR2004 - geometric diode level
    geometry = {
        name: context.values.get(name, default)
        for name, default in models.DIODE_GEOMETRY.items()
    }
    raw_kept = {
        name: value
        for name, value in context.values.items()
        if name not in models.DIODE_IGNORED
        and name not in models.DIODE_GEOMETRY
        and name not in models.DIODE_SELECTORS
        and name != "level"
    }
    kept_symbolic: dict[str, Expression] = {}
    for name, expression in context.symbolic_values.items():
        if name in models.DIODE_IGNORED or name in models.DIODE_GEOMETRY:
            continue
        if name in models.DIODE_KNEES and raw_kept[name] < models.MIN_CURRENT:
            _reject_stochastic(
                context.statement,
                name,
                f"of `{context.card.name}` below MIN_CURRENT",
            )
        kept_symbolic[name] = (
            _clamp_min(expression, models.MIN_CURRENT) or expression
            if name in {"is", "js"}
            else expression
        )
    kept_values = {
        name: (max(value, models.MIN_CURRENT) if name in {"is", "js"} else value)
        for name, value in raw_kept.items()
        if name not in models.DIODE_KNEES or value >= models.MIN_CURRENT
    }
    copies_value = context.instance.get("m", 1.0)
    copies_symbolic = context.symbolic_instance.get("m")
    area_value, area_symbolic, perimeter_value, perimeter_symbolic = (
        _diode_geometry_expressions(context, geometric=geometric, geometry=geometry)
    )
    bottom_value = area_value * copies_value
    bottom_symbolic = _binary(
        "*", area_symbolic, area_value, copies_symbolic, copies_value
    )
    scaled_symbolic = _area_scaled_symbolic(
        "diode",
        kept_values,
        kept_symbolic,
        area_value=bottom_value,
        area_symbolic=bottom_symbolic,
        multiplied=models.DIODE_BY_AREA,
        divided=("rs",),
    )
    scaled_values = models._area_scaled(  # noqa: SLF001 - shared area scaling helper
        "diode", kept_values, bottom_value, models.DIODE_BY_AREA, ("rs",)
    )
    sidewall_value = perimeter_value * copies_value
    sidewall_symbolic = _binary(
        "*", perimeter_symbolic, perimeter_value, copies_symbolic, copies_value
    )
    if sidewall_value > 0.0:
        scaled_symbolic = _area_scaled_symbolic(
            "diode",
            scaled_values,
            scaled_symbolic,
            area_value=sidewall_value,
            area_symbolic=sidewall_symbolic,
            multiplied=models.DIODE_BY_PERIMETER,
            divided=("rsw",),
        )
        scaled_values = models._area_scaled(  # noqa: SLF001 - shared area helper
            "diode", scaled_values, sidewall_value, models.DIODE_BY_PERIMETER, ("rsw",)
        )
    else:
        for name in (*models.DIODE_BY_PERIMETER, "rsw"):
            scaled_symbolic.pop(name, None)
            scaled_values.pop(name, None)
        if sidewall_symbolic is not None:
            _reject_stochastic(
                context.statement, "pj", f"of `{context.card.name}` with nominal PJ = 0"
            )
    breakdown_value = copies_value * (area_value if geometric else 1.0)
    breakdown_symbolic = (
        _binary("*", copies_symbolic, copies_value, area_symbolic, area_value)
        if geometric
        else copies_symbolic
    )
    scaled_symbolic = _area_scaled_symbolic(
        "diode",
        scaled_values,
        scaled_symbolic,
        area_value=breakdown_value,
        area_symbolic=breakdown_symbolic,
        multiplied=("ibv", "ib"),
        divided=(),
    )
    _record_mapped(
        context.compiler,
        context.element_name,
        scaled_symbolic,
        models.DIODE,
        allow_tref=True,
    )


def _record_bipolar(context: _DeviceContext) -> None:
    """Records symbolic targets for a level-1 Gummel-Poon bipolar transistor."""
    for discrete in ("level", "subs", *models.BIPOLAR_TEMPERATURE_MODELS):
        if discrete in context.symbolic_values:
            _reject_stochastic(context.statement, discrete, f"of `{context.card.name}`")
    skipped = {
        *models.BIPOLAR_UNDEFINED,
        *models.BIPOLAR_POWER_COEFFICIENTS,
        "level",
        "subs",
        *models.BIPOLAR_TEMPERATURE_MODELS,
        *(
            name
            for name in models.BIPOLAR_SATURATION_CURRENTS
            if name in context.values
        ),
    }
    kept_values: dict[str, float] = {}
    kept_symbolic: dict[str, Expression] = {}
    for name, value in context.values.items():
        if name in skipped:
            continue
        if name in models.BIPOLAR_ABSENT_AT_ZERO and value == 0.0:
            if name in context.symbolic_values:
                _reject_stochastic(
                    context.statement,
                    name,
                    f"of `{context.card.name}` with nominal 0",
                )
            continue
        if name in {"nkf", "nk"}:
            kept_values[name] = min(value, models.BIPOLAR_MAX_HIGH_INJECTION_EXPONENT)
            if name in context.symbolic_values:
                expression = context.symbolic_values[name]
                kept_symbolic[name] = (
                    _clamp_max(expression, models.BIPOLAR_MAX_HIGH_INJECTION_EXPONENT)
                    or expression
                )
        else:
            kept_values[name] = value
            if name in context.symbolic_values:
                kept_symbolic[name] = context.symbolic_values[name]
    area_value, area_symbolic = _area_expression(
        context.instance, context.symbolic_instance
    )
    scaled = _area_scaled_symbolic(
        "bipolar_transistor",
        kept_values,
        kept_symbolic,
        area_value=area_value,
        area_symbolic=area_symbolic,
        multiplied=_BIPOLAR_MULTIPLIED,
        divided=_BIPOLAR_DIVIDED,
    )
    _record_mapped(
        context.compiler,
        context.element_name,
        scaled,
        models.BIPOLAR,
        allow_tref=True,
    )


def _scale_vbic_flicker(
    scaled_values: Mapping[str, float],
    scaled_symbolic: dict[str, Expression],
    copies_value: float,
    copies_symbolic: Expression | None,
) -> None:
    """Scales VBIC flicker-noise coefficients in `scaled_symbolic` by `copies`."""
    for coefficient, exponent in (("kfn", "afn"), ("kf", "af")):
        if coefficient not in scaled_values:
            continue
        exponent_value = scaled_values.get(exponent, 1.0)
        exponent_symbolic = scaled_symbolic.get(exponent)
        one_minus_exponent = (
            None
            if exponent_symbolic is None
            else Binary("-", Number(1.0), exponent_symbolic)
        )
        factor_symbolic = _power(
            copies_symbolic, copies_value, one_minus_exponent, 1.0 - exponent_value
        )
        scaled_coefficient = _binary(
            "*",
            scaled_symbolic.get(coefficient),
            scaled_values[coefficient],
            factor_symbolic,
            math.pow(copies_value, 1.0 - exponent_value),
        )
        if scaled_coefficient is not None:
            scaled_symbolic[coefficient] = scaled_coefficient


def _record_vbic(context: _DeviceContext) -> None:
    """Records symbolic targets for a VBIC bipolar transistor."""
    for discrete in ("level", "qbm"):
        if discrete in context.symbolic_values:
            _reject_stochastic(context.statement, discrete, f"of `{context.card.name}`")
    ignored = models.VBIC_IGNORED | models.VBIC_SELF_HEATING | {"level", "qbm"}
    kept_values: dict[str, float] = {}
    kept_symbolic: dict[str, Expression] = {}
    for name, value in context.values.items():
        if name in ignored:
            continue
        if name in models.VBIC_INFINITE_AT_ZERO and value <= 0.0:
            if name in context.symbolic_values:
                _reject_stochastic(
                    context.statement,
                    name,
                    f"of `{context.card.name}` with non-positive nominal",
                )
            continue
        if name in models.VBIC_RESISTANCES:
            kept_values[name] = max(value, models.VBIC_MIN_RESISTANCE)
            if name in context.symbolic_values:
                expression = context.symbolic_values[name]
                kept_symbolic[name] = (
                    _clamp_min(expression, models.VBIC_MIN_RESISTANCE) or expression
                )
        else:
            kept_values[name] = value
            if name in context.symbolic_values:
                kept_symbolic[name] = context.symbolic_values[name]
    copies_value = context.instance.get("m", 1.0)
    copies_symbolic = context.symbolic_instance.get("m")
    area_value, area_symbolic = _area_expression(
        context.instance, context.symbolic_instance
    )
    scaled_symbolic = _area_scaled_symbolic(
        "vbic",
        kept_values,
        kept_symbolic,
        area_value=area_value,
        area_symbolic=area_symbolic,
        multiplied=models.VBIC_MULTIPLIED,
        divided=models.VBIC_RESISTANCES,
    )
    scaled_values = models._area_scaled(  # noqa: SLF001 - shared area scaling helper
        "vbic", kept_values, area_value, models.VBIC_MULTIPLIED, models.VBIC_RESISTANCES
    )
    _scale_vbic_flicker(scaled_values, scaled_symbolic, copies_value, copies_symbolic)
    _record_mapped(
        context.compiler,
        context.element_name,
        scaled_symbolic,
        models.VBIC,
        allow_tref=True,
    )


def _record_mosfet(context: _DeviceContext) -> None:
    """Records symbolic targets for a level-1 or level-3 MOSFET."""
    for discrete in ("level", "tpg"):
        if discrete in context.symbolic_values:
            _reject_stochastic(context.statement, discrete, f"of `{context.card.name}`")
    level = context.values.get("level", 1.0)
    card_only = {
        key: expression
        for key, expression in context.symbolic_values.items()
        if key not in {"level", "tpg", "l", "w"}
        and not (level == 3.0 and key == "lambda")  # noqa: PLR2004 - level-3 MOSFET
    }
    _record_mapped(
        context.compiler,
        context.element_name,
        card_only,
        models.MOSFET,
        skipped=models.NOISE,
    )
    geometry = {
        key: context.symbolic_values[key]
        for key in ("l", "w")
        if key in context.symbolic_values and key not in context.symbolic_instance
    }
    _record_mapped(
        context.compiler,
        context.element_name,
        geometry | dict(context.symbolic_instance),
        models.MOSFET_INSTANCE,
    )


def _record_vdmos(context: _DeviceContext) -> None:
    """Records symbolic targets for a VDMOS device."""
    card_only = {
        key: expression
        for key, expression in context.symbolic_values.items()
        if key != "phi"
    }
    _record_mapped(
        context.compiler,
        context.element_name,
        card_only,
        models.VDMOS,
        skipped=models.VDMOS_SELF_HEATING,
    )
    _record_mapped(
        context.compiler,
        context.element_name,
        context.symbolic_instance,
        models.VDMOS_INSTANCE,
    )


def _bsim_scale_expression(  # noqa: PLR0913 - BSIM SI scaling context
    bsim: models.Bsim,
    base: str,
    nominal_value: float,
    expression: Expression | None,
    *,
    centimeters_nominal: float,
    centimeters_symbolic: Expression | None,
) -> Expression | None:
    """Applies BSIM density or mobility SI unit conversion to `expression`."""
    if base in bsim.densities:
        if expression is None:
            return None
        threshold = bsim.densities[base]
        return Binary(
            "*",
            expression,
            Conditional(
                Binary(">", expression, Number(threshold)),
                Number(1.0),
                Number(1e6),
            ),
        )
    if base == "nsub":
        return _scaled(expression, 1e6)
    if base == "u0":
        return _binary(
            "*", expression, nominal_value, centimeters_symbolic, centimeters_nominal
        )
    return expression


def _bsim_aliased(
    bsim: models.Bsim,
    values: Mapping[str, float],
    symbolic_values: Mapping[str, Expression],
) -> tuple[dict[str, float], dict[str, Expression]]:
    """Resolves BSIM aliases while dropping unread parameters."""
    aliased_values: dict[str, float] = {}
    aliased_symbolic: dict[str, Expression] = {}
    for spice, value in values.items():
        canonical = bsim.aliases.get(spice, spice)
        if canonical in bsim.unread:
            continue
        aliased_values[canonical] = value
        if spice in symbolic_values:
            aliased_symbolic[canonical] = symbolic_values[spice]
        else:
            aliased_symbolic.pop(canonical, None)
    return aliased_values, aliased_symbolic


def _record_bsim(
    context: _DeviceContext,
    family: Literal["bsim3", "bsim4"],
) -> None:
    """Records symbolic targets for a BSIM3 or BSIM4 transistor."""
    bsim = models.BSIMS[family]
    fixed_modes = (
        frozenset(models.BSIM3_OFF)
        if family == "bsim3"
        else frozenset(models.BSIM4_MODES)
    )
    for discrete in (
        "level",
        "binunit",
        "paramchk",
        *fixed_modes,
        *bsim.selectors,
    ):
        if discrete in context.symbolic_values or discrete in context.symbolic_instance:
            _reject_stochastic(context.statement, discrete, f"of `{context.card.name}`")
    binunit = context.values.get("binunit", 1.0)
    binned_extent = 1e-6 if binunit == 1.0 else 1.0
    default_mobility = 0.067 if context.card.kind[0] == "n" else 0.025
    mobility_nominal = context.values.get("u0", default_mobility)
    mobility_symbolic = context.symbolic_values.get("u0")
    centimeters_nominal = 1e-4 if mobility_nominal > 1.0 else 1.0
    centimeters_symbolic = (
        Conditional(
            Binary(">", mobility_symbolic, Number(1.0)), Number(1e-4), Number(1.0)
        )
        if mobility_symbolic is not None
        else None
    )
    aliased_values, aliased_symbolic = _bsim_aliased(
        bsim, context.values, context.symbolic_values
    )
    _record_nominal_temperature(
        context.compiler, context.element_name, aliased_symbolic
    )
    ignored = (
        models.NOMINAL_TEMPERATURES
        | set(bsim.selectors)
        | {"level", "binunit", *fixed_modes}
    )
    for name, value in aliased_values.items():
        if name in ignored:
            continue
        expression = aliased_symbolic.get(name)
        if name in bsim.settings:
            context.compiler.add_target(
                context.element_name, bsim.settings[name], expression
            )
        elif name in bsim.card:
            scaled = _bsim_scale_expression(
                bsim,
                name,
                value,
                expression,
                centimeters_nominal=centimeters_nominal,
                centimeters_symbolic=centimeters_symbolic,
            )
            context.compiler.add_target(context.element_name, name, scaled)
        elif name[:1] in {"l", "w", "p"} and name[1:] in bsim.binned:
            extent = binned_extent * binned_extent if name[0] == "p" else binned_extent
            scaled = _bsim_scale_expression(
                bsim,
                name[1:],
                value,
                expression,
                centimeters_nominal=centimeters_nominal,
                centimeters_symbolic=centimeters_symbolic,
            )
            context.compiler.add_target(
                context.element_name, name, _scaled(scaled, extent)
            )
    instance_unread = (
        frozenset({"geo"}) if family == "bsim3" else models.BSIM4_INSTANCE_UNREAD
    )
    filtered_instance = {
        key: expression
        for key, expression in context.symbolic_instance.items()
        if key not in instance_unread
    }
    _record_mapped(
        context.compiler, context.element_name, filtered_instance, bsim.instance
    )


def check_condition_deterministic(
    condition: Expression,
    compiler: StatisticalCompiler,
    scope: Mapping[str, float],
    statement: Statement,
) -> None:
    """Rejects `.if`/`.elseif` conditions whose outcome depends on statistical draws."""
    if compiler.lower(condition, scope, call_name=".if") is not None:
        message = (
            "a conditional `.if` block reads a statistical expression, which changes "
            "circuit structure across Monte Carlo samples"
        )
        raise statement.fail(message)


def check_binned_deterministic(  # noqa: PLR0913 - bin selection validation inputs
    card: ModelCard,
    instance_symbolic: Mapping[str, Expression],
    compiler: StatisticalCompiler,
    *,
    scope: Mapping[str, float],
    functions: Mapping[str, Function],
    statement: Statement,
) -> None:
    """Rejects binned MOSFETs whose bin selection depends on statistical draws."""
    for dimension in ("l", "w", "nf"):
        if dimension in instance_symbolic:
            _reject_stochastic(statement, dimension, f"selecting bin of `{card.name}`")
    for bound in ("lmin", "lmax", "wmin", "wmax"):
        if bound in card.parameters and (
            compiler.lower(
                inlined(card.parameters[bound], functions),
                scope,
                call_name=f"{card.name.casefold()}.{bound}",
            )
            is not None
        ):
            _reject_stochastic(statement, bound, f"selecting bin of `{card.name}`")


_WAVEFORM_FIELDS: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "pulse": (
        "waveform.pulse",
        ("initial", "pulsed", "delay", "rise", "fall", "width", "period"),
    ),
    "sin": (
        "waveform.sine",
        ("offset", "amplitude", "frequency", "delay", "damping", "phase"),
    ),
    "exp": (
        "waveform.exponential",
        (
            "initial",
            "pulsed",
            "rise_delay",
            "rise_time_constant",
            "fall_delay",
            "fall_time_constant",
        ),
    ),
}
"""Waveform target prefixes and ordered field names for SPICE transient waveforms."""


def record_waveform_targets(
    compiler: StatisticalCompiler,
    element_name: str,
    waveform_kind: str,
    argument_expressions: Sequence[Expression | None],
) -> None:
    """Records symbolic targets for an independent source's time-domain waveform."""
    if not any(expression is not None for expression in argument_expressions):
        return
    if waveform_kind == "pwl":
        for index, expression in enumerate(argument_expressions):
            point_index, coordinate = divmod(index, 2)
            compiler.add_target(
                element_name,
                f"waveform.piecewise.points.{point_index}.{coordinate}",
                expression,
            )
        return
    entry = _WAVEFORM_FIELDS.get(waveform_kind)
    if entry is None:
        return
    prefix, fields = entry
    for field_name, expression in zip(fields, argument_expressions, strict=False):
        scaled = (
            _scaled(expression, math.pi / 180.0)
            if waveform_kind == "sin" and field_name == "phase"
            else expression
        )
        compiler.add_target(element_name, f"{prefix}.{field_name}", scaled)


def single_keyword_symbolic(
    evaluate_word: Callable[[str], tuple[float, Expression | None]],
    statement: Statement,
    name: str,
    words: Sequence[str],
) -> tuple[float, Expression | None]:
    """Evaluates a single-valued keyword returning `(nominal, symbolic_expression)`."""
    if not words:
        message = f"`{name.upper()}` is not assigned a value"
        raise statement.fail(message)
    if len(words) != 1:
        message = f"`{name.upper()}` accepts one value, not `{' '.join(words)}`"
        raise statement.fail(message)
    return evaluate_word(words[0])


def instance_keywords_symbolic(
    evaluate_word: Callable[[str], tuple[float, Expression | None]],
    statement: Statement,
    assigned: Mapping[str, list[str]],
    *,
    flags: frozenset[str] = frozenset({"off", "on"}),
    multi_valued: frozenset[str] = frozenset(),
) -> tuple[dict[str, float], dict[str, Expression]]:
    """Evaluates element keyword assignments returning `(values, symbolic_map)`."""
    values: dict[str, float] = {}
    symbolic: dict[str, Expression] = {}
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
            values[key] = evaluated[0][0]
            if evaluated[0][1] is not None:
                symbolic[key] = evaluated[0][1]
            continue
        value_word, *extra = words
        stray = [word for word in extra if word.casefold() not in flags]
        if stray:
            message = f"`{key.upper()}` accepts one value, not `{' '.join(words)}`"
            raise statement.fail(message)
        resolved_value, expression = evaluate_word(value_word)
        values[key] = resolved_value
        if expression is not None:
            symbolic[key] = expression
    return values, symbolic

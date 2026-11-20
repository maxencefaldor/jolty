# Copyright 2026 Maxence Faldor
"""Parser, evaluator, and translator for SPICE expressions.

SPICE literals accept scale suffixes such as ``4.7k`` or ``1MEG``. Functions follow
PSpice and ngspice semantics: ``PWR(x, y)`` computes ``|x| ^ y``, ``LIMIT(x, a, b)``
clamps ``x`` to ``[a, b]``, and ``&`` and ``|`` are logical operators. Identifiers are
case-insensitive.

Examples:
    >>> expression = parse("LIMIT(Gain*V(VC+,VC-), -1m, 1m)")
    >>> evaluate(parse("2 * Gain + 1k"), {"gain": 3.0})
    1006.0
    >>> translate(expression, {"gain": 1e3}, str.lower, str.lower)
    'limit(1000.0 * v(vc+, vc-), -0.001, 0.001)'
"""

import bisect
import contextlib
import functools
import itertools
import math
import re
from collections.abc import Callable, Generator, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, override

from jolty import _core
from jolty.errors import SpiceError

_SCALES: Final = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "mil": 25.4e-6,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}
"""SPICE scale suffixes, optionally followed by unit letters such as ``10uF``."""

_NUMBER: Final = re.compile(
    r"(?P<mantissa>[-+]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)"
    r"(?P<suffix>meg|mil|[tgkmunpf])?[a-z]*",
    re.IGNORECASE | re.ASCII,
)


class UndefinedError(SpiceError):
    """Error raised when an expression references an undefined name."""

    def __init__(self, message: str, name: str) -> None:
        """Initializes an undefined-name error with `message` and missing `name`."""
        super().__init__(message)
        self.name = name


def number(text: str) -> float | None:
    """Parses a numeric literal with an optional SPICE scale suffix, or returns `None`.

    Returns `None` for non-finite values, including literals that overflow IEEE-754
    double precision on parsing or scale multiplication.

    Args:
        text: Candidate numeric literal with optional SPICE scale suffix and unit.

    Returns:
        Parsed finite floating-point value, or `None` when `text` is non-numeric or
        non-finite.

    Examples:
        >>> number("4.7k"), number("1MEG"), number("1uF"), number("-2.5E-3")
        (4700.0, 1000000.0, 1e-06, -0.0025)
        >>> number("x") is None
        True
    """
    match = _NUMBER.fullmatch(text)
    if match is None:
        return None
    scale = _SCALES[match["suffix"].lower()] if match["suffix"] else 1.0
    result = float(match["mantissa"]) * scale
    return result if math.isfinite(result) else None


@dataclass(frozen=True, slots=True)
class Number:
    """Numeric constant expression node."""

    value: float


@dataclass(frozen=True, slots=True)
class Name:
    """Parameter or constant reference identified by its case-folded name."""

    name: str


@dataclass(frozen=True, slots=True)
class Voltage:
    """Node voltage ``V(a)`` or differential voltage ``V(a, b)`` expression node."""

    positive: str
    negative: str | None


@dataclass(frozen=True, slots=True)
class Current:
    """Branch current ``I(Vx)`` through a voltage source."""

    source: str


class _Compound:
    """Mixin giving compound expression nodes iterative equality, hash, and repr."""

    __slots__ = ()

    @override
    def __eq__(self, other: object) -> bool:
        if (
            type(self) is type(other)
            and isinstance(self, Call | Unary | Binary | Conditional)
            and isinstance(other, Call | Unary | Binary | Conditional)
        ):
            return _equal(self, other)
        return NotImplemented

    @override
    def __hash__(self) -> int:
        if isinstance(self, Call | Unary | Binary | Conditional):
            return _hash(self)
        return super().__hash__()

    @override
    def __repr__(self) -> str:
        if isinstance(self, Call | Unary | Binary | Conditional):
            return _repr(self)
        return super().__repr__()


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Call(_Compound):
    """Function invocation with a case-folded name and argument tuple."""

    function: str
    arguments: tuple[Expression, ...]


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Unary(_Compound):
    """Unary sign or logical negation applied to one operand."""

    operator: str
    operand: Expression


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Binary(_Compound):
    """Binary operation combining left and right operands."""

    operator: str
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Conditional(_Compound):
    """Ternary conditional expression ``c ? a : b``."""

    condition: Expression
    then: Expression
    otherwise: Expression


type Expression = (
    Number | Name | Voltage | Current | Call | Unary | Binary | Conditional
)


@dataclass(frozen=True, slots=True)
class Function:
    """User-defined SPICE ``.func`` declaration with parameter names and body.

    Parameter names are stored case-folded and referenced within `body`.
    """

    parameters: tuple[str, ...]
    body: Expression


_TOKEN: Final = re.compile(
    r"\s*(?:(?P<number>(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?[a-z]*)"
    r"|(?P<name>[a-z_][a-z0-9_.$]*)"
    r"|(?P<symbol>\*\*|==|!=|<>|<=|>=|&&|\|\||[-+*/^<>!~&|?:,(){}]))",
    re.IGNORECASE | re.ASCII,
)

_LEVELS: Final = (
    {"||": "||", "|": "||"},
    {"&&": "&&", "&": "&&"},
    {"==": "==", "!=": "!=", "<>": "!="},
    {"<": "<", "<=": "<=", ">": ">", ">=": ">="},
    {"+": "+", "-": "-"},
    {"*": "*", "/": "/"},
)
"""Binary operators grouped by precedence (loosest first) with canonical forms."""


_BINARY: Final[dict[str, tuple[int, str]]] = {
    symbol: (precedence, operator)
    for precedence, operators in enumerate(_LEVELS)
    for symbol, operator in operators.items()
}
"""Mapping from binary operator symbols to `(precedence, canonical_operator)`."""

_UNARY: Final = {"-": "-", "+": "+", "!": "!", "~": "!"}
"""Unary operator symbols mapped to their canonical operator."""

_MAX_DEPTH: Final = 100
"""Maximum nesting depth accepted by the expression parser."""

_MAX_CORE_DEPTH: Final = 256
"""Maximum parenthesis depth passed to the Rust expression engine.

Matches `jolty_expressions::parser::MAX_DEPTH` (`256`) so SPICE behavioral and
parameter translation raises `SpiceError` directly before crossing into `_core`.
"""

_MAX_INLINED_NODES: Final = 100_000
"""Maximum expression node count allowed while inlining user functions."""

_MAX_POLYNOMIAL_DEGREE: Final = 64
"""Largest polynomial degree integrated in closed form."""

_MAX_TRANSLATED_LENGTH: Final = 1_000_000
"""Maximum character length allowed for a translated expression string.

Guards `translate` against exponential blowups in duplicating forms (`int`,
`pwrs`, and odd-integer `_power`), where each nesting level doubles the operand
string (`2^d`). One million characters far exceeds real SPICE behavioral
expressions while stopping exponential growth well before memory exhaustion.
"""

_ATOM_PRECEDENCE: Final = 9
"""Precedence rank of atomic terms (literals, calls, parenthesized groups)."""

_UNARY_PRECEDENCE: Final = 7
"""Precedence rank of prefix unary operators (`-`, `+`, `!`), matching Rust."""

_BIN_PRECEDENCE: Final[dict[str, int]] = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 4,
    "<=": 4,
    ">": 4,
    ">=": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
}
"""Precedence ranks (`1..=6`, loosest first) of left-associative binary operators.

Matches `jolty_expressions::parser::binary_operator` so `translate` omits
redundant parentheses on flat operator chains.
"""


_ENCLOSING: Final = {"{": "}", "'": "'", '"': '"'}
"""Delimiter pairs that enclose SPICE expressions (braces or quotes).

Supports nested delimiters such as HSPICE and SKY130 forms like ``{"-tol_nfom/2"}``.
"""


def _is_enclosed(text: str) -> bool:
    """Returns whether `text` is wrapped in one matched pair of delimiters."""
    if len(text) <= 1:
        return False
    opening = text[0]
    closing = _ENCLOSING.get(opening)
    if closing is None or text[-1] != closing:
        return False
    if opening == "{":
        depth = 0
        for index, character in enumerate(text):
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return index == len(text) - 1
        return False
    return text.find(closing, 1) == len(text) - 1


def nesting_depth(text: str) -> int:
    """Returns the maximum parenthesis nesting depth in `text`.

    Args:
        text: Expression string to inspect.

    Returns:
        Maximum parenthesis nesting depth in `text`.
    """
    depth = 0
    deepest = 0
    for character in text:
        if character == "(":
            depth += 1
            deepest = max(deepest, depth)
        elif character == ")":
            depth = max(0, depth - 1)
    return deepest


def check_core_depth(text: str) -> None:
    """Checks that `text` stays within `_MAX_CORE_DEPTH` nested parentheses.

    Args:
        text: Translated expression string checked against `_MAX_CORE_DEPTH`.

    Raises:
        SpiceError: When the translated expression exceeds the safe nesting limit.
    """
    depth = nesting_depth(text)
    if depth > _MAX_CORE_DEPTH:
        message = (
            f"expression nests {depth} parentheses deep, exceeding the limit of "
            f"{_MAX_CORE_DEPTH}"
        )
        raise SpiceError(message)


def parse(text: str) -> Expression:
    """Parses a SPICE expression string, stripping outer braces or quotes.

    Args:
        text: SPICE expression string, optionally enclosed in braces or quotes.

    Returns:
        Parsed syntax tree for `text`.

    Raises:
        SpiceError: When `text` is not a valid expression.
    """
    stripped = text.strip()
    while _is_enclosed(stripped):
        stripped = stripped[1:-1].strip()
    parser = _Parser(stripped)
    expression = parser.conditional()
    if parser.position < len(stripped.rstrip()):
        expected = "an operator"
        raise parser.fail(expected)
    return expression


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0
        self.depth = 0

    def fail(self, expected: str) -> SpiceError:
        """Builds a `SpiceError` reporting that `expected` was not found at cursor."""
        rest = self.text[self.position :].strip() or "the end"
        return SpiceError(f"expected {expected} in `{self.text}`, not `{rest}`")

    def _too_deep(self) -> SpiceError:
        message = f"expression exceeds maximum nesting depth of {_MAX_DEPTH}"
        return SpiceError(message)

    @contextlib.contextmanager
    def _nested(self) -> Generator[None]:
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            raise self._too_deep()
        try:
            yield
        finally:
            self.depth -= 1

    def scan(self) -> tuple[str, str, int] | None:
        """Returns the next token kind, text, and end index, or `None` at end."""
        match = _TOKEN.match(self.text, self.position)
        kind = None if match is None else match.lastgroup
        if match is None or kind is None:
            return None
        return kind, self.text[match.start(kind) : match.end()], match.end()

    def peek(self) -> tuple[str, str] | None:
        scanned = self.scan()
        return None if scanned is None else (scanned[0], scanned[1])

    def take(self) -> tuple[str, str] | None:
        scanned = self.scan()
        if scanned is None:
            return None
        kind, text, self.position = scanned
        return kind, text

    def accept(self, symbol: str) -> bool:
        if self.peek() == ("symbol", symbol):
            self.take()
            return True
        return False

    def expect(self, symbol: str) -> None:
        if not self.accept(symbol):
            expected = f"`{symbol}`"
            raise self.fail(expected)

    def conditional(self) -> Expression:
        condition = self.binary()
        if not self.accept("?"):
            return condition
        with self._nested():
            then = self.conditional()
            self.expect(":")
            otherwise = self.conditional()
        return Conditional(condition, then, otherwise)

    def binary(self) -> Expression:
        operands: list[Expression] = [self.unary()]
        operators: list[tuple[int, str]] = []

        def reduce_top() -> None:
            _, pending = operators.pop()
            right = operands.pop()
            operands.append(Binary(pending, operands.pop(), right))

        while (token := self.peek()) is not None and token[0] == "symbol":
            entry = _BINARY.get(token[1])
            if entry is None:
                break
            precedence, operator = entry
            self.take()
            while operators and operators[-1][0] >= precedence:
                reduce_top()
            operators.append((precedence, operator))
            operands.append(self.unary())
        while operators:
            reduce_top()
        return operands[0]

    def unary(self) -> Expression:
        operators: list[str] = []
        while (token := self.peek()) is not None and token[0] == "symbol":
            operator = _UNARY.get(token[1])
            if operator is None:
                break
            self.take()
            operators.append(operator)
            if self.depth + len(operators) > _MAX_DEPTH:
                raise self._too_deep()
        result = self.power()
        for operator in reversed(operators):
            result = Unary(operator, result)
        return result

    def power(self) -> Expression:
        base = self.primary()
        if self.accept("**") or self.accept("^"):
            with self._nested():
                return Binary("^", base, self.unary())
        return base

    def primary(self) -> Expression:
        token = self.take()
        match token:
            case ("number", text):
                value = number(text)
                if value is None:
                    message = f"number `{text}` is not finite in `{self.text}`"
                    raise SpiceError(message)
                return Number(value)
            # Brace-enclosed parameter inside an expression, as in PSpice: `{VTHRES}+1`.
            case ("symbol", "(" | "{" as opening):
                with self._nested():
                    inner = self.conditional()
                    self.expect(")" if opening == "(" else "}")
                return inner
            case ("name", name) if self.peek() == ("symbol", "("):
                return self.call(name.casefold())
            case ("name", name):
                return Name(name.casefold())
        expected = "an operand"
        raise self.fail(expected)

    def call(self, function: str) -> Expression:
        self.take()
        if function in {"v", "i"}:
            # Node names may contain signs (such as V(VC+,COM)) and are read verbatim.
            end = self.text.find(")", self.position)
            if end < 0:
                expected = "`)`"
                raise self.fail(expected)
            written = self.text[self.position : end]
            names = [part.strip() for part in written.split(",")]
            self.position = end + 1
            if all(names):
                match function, names:
                    case "v", [positive]:
                        return Voltage(positive, None)
                    case "v", [positive, negative]:
                        return Voltage(positive, negative)
                    case "i", [source]:
                        return Current(source)
            expected = f"one or two nodes in `{function}(…)`"
            raise self.fail(expected)
        with self._nested():
            arguments = [self.conditional()]
            while self.accept(","):
                arguments.append(self.conditional())
            self.expect(")")
            return Call(function, tuple(arguments))


def children(expression: Expression) -> tuple[Expression, ...]:
    """Returns the direct sub-expressions of `expression` in evaluation order.

    Args:
        expression: Expression node whose direct operands are requested.

    Returns:
        Direct child expression nodes in evaluation order, or an empty tuple for leaves.
    """
    match expression:
        case Unary(_, operand):
            return (operand,)
        case Binary(_, left, right):
            return (left, right)
        case Conditional(condition, then, otherwise):
            return (condition, then, otherwise)
        case Call(_, arguments):
            return arguments
    return ()


def _label(expression: Expression) -> str:
    """Returns the operator or function name identifying a compound node."""
    match expression:
        case Call(function, _):
            return function
        case Unary(operator, _) | Binary(operator, _, _):
            return operator
        case _:
            return ""


def _equal(first: Expression, second: Expression) -> bool:
    """Checks structural equality iteratively without Python recursion limits."""
    stack: list[tuple[Expression, Expression]] = [(first, second)]
    while stack:
        left, right = stack.pop()
        if left is right:
            continue
        if type(left) is not type(right):
            return False
        if isinstance(left, Number | Name | Voltage | Current):
            if left != right:
                return False
            continue
        if _label(left) != _label(right):
            return False
        left_children = children(left)
        right_children = children(right)
        if len(left_children) != len(right_children):
            return False
        stack.extend(zip(left_children, right_children, strict=True))
    return True


def walk(expression: Expression) -> Iterator[Expression]:
    """Yields every node of `expression` iteratively without recursion.

    Args:
        expression: Root expression node to traverse.

    Returns:
        Iterator yielding every expression node reachable from `expression`.
    """
    stack: list[Expression] = [expression]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(children(current))


def fold[Result](
    expression: Expression,
    leaf: Callable[[Expression], Result | None],
    combine: Callable[[Expression, list[Result]], Result],
    *,
    before: Callable[[Expression], None] | None = None,
) -> Result:
    """Folds `expression` bottom-up iteratively without recursion.

    When `leaf(node)` returns a non-`None` value, that value is used directly
    without descending into the children of `node`. Otherwise `before(node)` (when
    provided) runs before descending left-to-right, and `combine(node, built)`
    combines the folded child results.

    Args:
        expression: Root expression node to fold.
        leaf: Callback returning a direct result to short-circuit descent, or `None` to
            descend.
        combine: Callback combining a node and folded child results into a single value.
        before: Optional pre-order hook invoked on non-short-circuited nodes before
            descending.

    Returns:
        Folded result computed at the root `expression`.
    """
    frames: list[tuple[Expression, bool]] = [(expression, False)]
    results: list[Result] = []
    while frames:
        current, ready = frames.pop()
        if not ready:
            direct = leaf(current)
            if direct is not None:
                results.append(direct)
                continue
            if before is not None:
                before(current)
            subexpressions = children(current)
            if not subexpressions:
                results.append(combine(current, []))
                continue
            frames.append((current, True))
            frames.extend((child, False) for child in reversed(subexpressions))
            continue
        arity = len(children(current))
        built = results[-arity:] if arity else []
        if arity:
            del results[-arity:]
        results.append(combine(current, built))
    return results[0]


def rebuild(expression: Expression, subexpressions: list[Expression]) -> Expression:
    """Rebuilds `expression` with `subexpressions` in place of its children.

    Args:
        expression: Template expression node supplying kind and operator or function.
        subexpressions: Replacement child expressions in `children(expression)` order.

    Returns:
        Reconstructed expression node, reusing `expression` when children are unchanged.
    """
    current = children(expression)
    if len(current) == len(subexpressions) and all(
        new is old for new, old in zip(subexpressions, current, strict=True)
    ):
        return expression
    match expression:
        case Unary(operator, _):
            return Unary(operator, subexpressions[0])
        case Binary(operator, _, _):
            return Binary(operator, subexpressions[0], subexpressions[1])
        case Conditional():
            return Conditional(subexpressions[0], subexpressions[1], subexpressions[2])
        case Call(function, _):
            return Call(function, tuple(subexpressions))
    return expression


def _no_replace(_node: Expression) -> Expression | None:
    """Returns `None` so `rewrite` rebuilds every node bottom-up by default."""
    return None


def rewrite(
    expression: Expression,
    replace: Callable[[Expression], Expression | None] | None = None,
    *,
    before: Callable[[Expression], None] | None = None,
    combine: Callable[[Expression, list[Expression]], Expression] = rebuild,
) -> Expression:
    """Rewrites `expression` bottom-up iteratively without recursion.

    Args:
        expression: Root expression node to rewrite.
        replace: Optional leaf callback returning a replacement subtree to skip
            descending, or `None` to descend.
        before: Optional pre-order hook invoked before descending into child nodes.
        combine: Bottom-up constructor building each rewritten node from transformed
            children.

    Returns:
        Rewritten expression tree.
    """
    leaf = _no_replace if replace is None else replace
    return fold(expression, leaf, combine, before=before)


def _hash_leaf(node: Expression) -> int | None:
    """Hashes atomic expression nodes directly; returns None for compound nodes."""
    if isinstance(node, Number | Name | Voltage | Current):
        return hash(node)
    return None


def _hash_combine(node: Expression, child_hashes: list[int]) -> int:
    """Combines child hashes into the structural hash of a compound node."""
    return hash((type(node), _label(node), tuple(child_hashes)))


def _hash(expression: Expression) -> int:
    """Computes a structural hash iteratively without Python recursion limits."""
    return fold(expression, _hash_leaf, _hash_combine)


def _push_call_repr(
    parts: list[str],
    stack: list[Expression | str],
    function: str,
    arguments: tuple[Expression, ...],
) -> None:
    """Appends a `Call` header and schedules its arguments for iterative `repr`."""
    parts.append(f"Call(function={function!r}, arguments=(")
    if not arguments:
        parts.append("))")
    elif len(arguments) == 1:
        stack.extend((",))", arguments[0]))
    else:
        stack.append("))")
        for index in range(len(arguments) - 1, -1, -1):
            stack.append(arguments[index])
            if index > 0:
                stack.append(", ")


def _repr(expression: Expression) -> str:
    """Formats a structural representation iteratively without recursion limits."""
    parts: list[str] = []
    stack: list[Expression | str] = [expression]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            parts.append(item)
            continue
        match item:
            case Number() | Name() | Voltage() | Current():
                parts.append(repr(item))
            case Unary(operator, operand):
                parts.append(f"Unary(operator={operator!r}, operand=")
                stack.extend((")", operand))
            case Binary(operator, left, right):
                parts.append(f"Binary(operator={operator!r}, left=")
                stack.extend((")", right, ", right=", left))
            case Conditional(condition, then, otherwise):
                parts.append("Conditional(condition=")
                stack.extend(
                    (")", otherwise, ", otherwise=", then, ", then=", condition)
                )
            case Call(function, arguments):
                _push_call_repr(parts, stack, function, arguments)
    return "".join(parts)


def _check_budget(count: int) -> int:
    """Increments the node count and fails when it passes `_MAX_INLINED_NODES`."""
    next_count = count + 1
    if next_count > _MAX_INLINED_NODES:
        message = (
            f"inlining user functions exceeds {_MAX_INLINED_NODES} expression nodes"
        )
        raise SpiceError(message)
    return next_count


def _inline_call(
    function: str,
    arguments: list[Expression],
    functions: Mapping[str, Function],
    calling: tuple[str, ...],
) -> tuple[Expression, tuple[str, ...]]:
    """Substitutes `arguments` into the body of `function` and extends `calling`."""
    defined = functions[function]
    given = dict(zip(defined.parameters, arguments, strict=True))
    if not given:
        return defined.body, (*calling, function)

    def replace(node: Expression) -> Expression | None:
        return given.get(node.name) if isinstance(node, Name) else None

    return rewrite(defined.body, replace), (*calling, function)


def _check_call(
    function: str,
    argument_count: int,
    functions: Mapping[str, Function],
    calling: tuple[str, ...],
) -> None:
    """Checks that `function` is not already in `calling` and takes `argument_count`."""
    if function in calling:
        cycle = " calls ".join(
            f"`{name}`" for name in (*calling[calling.index(function) :], function)
        )
        message = f"a function calls itself: {cycle}"
        raise SpiceError(message)
    defined = functions[function]
    if argument_count != len(defined.parameters):
        message = (
            f"`{function}` takes {len(defined.parameters)} arguments, not "
            f"{argument_count}"
        )
        raise SpiceError(message)


def inlined(
    expression: Expression,
    functions: Mapping[str, Function],
    calling: tuple[str, ...] = (),
) -> Expression:
    """Returns `expression` with all calls to user-defined `functions` expanded inline.

    Each matching call is replaced by its function body with parameters bound to the
    call arguments, tracking active frames in `calling`. User functions may call other
    user functions, but direct or indirect recursion is rejected.

    Args:
        expression: Expression whose user-defined function calls are expanded.
        functions: Case-folded user function definitions available in scope.
        calling: Active call stack of function names used to detect recursion.

    Returns:
        Expression with all calls into `functions` substituted inline.

    Raises:
        SpiceError: When a function recurses, receives the wrong argument count, or
            inlining exceeds `_MAX_INLINED_NODES` nodes.

    Examples:
        >>> twice = Function(("x",), parse("2 * x"))
        >>> inlined(parse("twice(twice(y)) + 1"), {"twice": twice})
        Binary(operator='+', left=Binary(operator='*', left=Number(value=2.0), \
right=Binary(operator='*', left=Number(value=2.0), right=Name(name='y'))), \
right=Number(value=1.0))
    """
    if not functions or isinstance(expression, Number | Name | Voltage | Current):
        return expression
    frames: list[tuple[Expression, tuple[str, ...], bool]] = [
        (expression, calling, False)
    ]
    results: list[Expression] = []
    node_count = 0
    while frames:
        current, active, ready = frames.pop()
        if not ready:
            is_user = isinstance(current, Call) and current.function in functions
            if is_user:
                _check_call(current.function, len(current.arguments), functions, active)
            subexpressions = children(current)
            if not subexpressions and not is_user:
                node_count = _check_budget(node_count)
                results.append(current)
                continue
            frames.append((current, active, True))
            frames.extend((child, active, False) for child in reversed(subexpressions))
            continue
        arity = len(children(current))
        built = results[-arity:] if arity else []
        if arity:
            del results[-arity:]
        if isinstance(current, Call) and current.function in functions:
            body, next_calling = _inline_call(
                current.function, built, functions, active
            )
            node_count = _check_budget(node_count)
            frames.append((body, next_calling, False))
        else:
            node_count = _check_budget(node_count)
            results.append(rebuild(current, built))
    return results[0]


_CONSTANTS: Final = {
    "pi": math.pi,
    "e": math.e,
    "temp": 27.0,
    "vt": (27.0 + 273.15) * 8.6173303e-5,
    "gmin": 1e-12,
}
"""Built-in constants defined for every expression, matching ngspice PSpice defaults.

Includes ``TEMP`` (temperature in degrees Celsius), ``VT`` (thermal voltage using
the ngspice constant), and ``GMIN`` (minimum conductance).
"""

_FUNCTIONS: Final[dict[str, tuple[int | None, str]]] = {
    **{
        name: (1, name)
        for name in (
            *(
                "abs",
                "sqrt",
                "exp",
                "ln",
                "log10",
                "floor",
                "ceil",
                "sgn",
                "u",
                "uramp",
            ),
            *("sin", "cos", "tan", "asin", "acos", "atan"),
            *("sinh", "cosh", "tanh", "asinh", "acosh", "atanh"),
        )
    },
    "log": (1, "ln({0})"),
    "sign": (1, "sgn({0})"),
    "stp": (1, "u({0})"),
    "int": (1, "(sgn({0}) * floor(abs({0})))"),
    "min": (2, "min"),
    "max": (2, "max"),
    "pow": (2, "pow(abs({0}), {1})"),
    "pwr": (2, "pow(abs({0}), {1})"),
    "pwrs": (2, "(sgn({0}) * pow(abs({0}), {1}))"),
    "limit": (3, "limit"),
    "if": (3, "if"),
    "table": (None, "pwl"),
    "pwl": (None, "pwl"),
    "rounded_table": (None, "rounded_table"),
}
"""Supported SPICE functions mapped to `(arity, jolty_name_or_template)`.

An arity of `None` accepts any argument count. PSpice ``PWR`` operates on the
magnitude of the base, while ``POW`` follows `_power`. Inside expressions, ``TABLE``
maps to ngspice ``pwl`` (extended past endpoints), whereas source-level ``TABLE``
elements built by [`library`][jolty.spice.library] use ``rounded_table``.
"""


def substituted(
    expression: Expression, variable: Expression, replacement: Expression
) -> Expression:
    """Replaces every occurrence of `variable` in `expression` with `replacement`.

    Args:
        expression: Expression tree in which occurrences are replaced.
        variable: Target subexpression matched structurally for replacement.
        replacement: Expression substituted for each occurrence of `variable`.

    Returns:
        Expression with every occurrence of `variable` replaced by `replacement`.
    """

    def replace(node: Expression) -> Expression | None:
        return replacement if node == variable else None

    return rewrite(expression, replace)


type Polynomial = dict[int, float]
"""Univariate polynomial mapping each exponent to its numeric coefficient."""


def _combine_polynomial(
    expression: Expression, computed: Mapping[int, Polynomial | None]
) -> Polynomial | None:
    """Combines child polynomials from `computed` for a compound `expression`."""
    result: Polynomial | None = None
    match expression:
        case Unary("-", operand):
            found = computed.get(id(operand))
            result = None if found is None else _scaled(found, -1.0)
        case Unary("+", operand):
            result = computed.get(id(operand))
        case Binary("+" | "-" as operator, left, right):
            first, second = computed.get(id(left)), computed.get(id(right))
            if first is not None and second is not None:
                sign = 1.0 if operator == "+" else -1.0
                result = _added(first, _scaled(second, sign))
        case Binary("*", left, right):
            first, second = computed.get(id(left)), computed.get(id(right))
            if first is not None and second is not None:
                result = _multiplied(first, second)
        case Binary("/", left, Number(divisor)) if divisor != 0:
            found = computed.get(id(left))
            result = None if found is None else _scaled(found, 1 / divisor)
        case Binary("^", base, Number(power)) | Call("pow", (base, Number(power))):
            result = _power_polynomial(computed.get(id(base)), power)
    return result


def _polynomial_map(
    expression: Expression, variable: Expression
) -> dict[int, Polynomial | None]:
    """Computes bottom-up polynomials of `variable` for every reachable subtree.

    Maps each node's `id` to its polynomial dictionary (up to
    `_MAX_POLYNOMIAL_DEGREE`), or `None` when the subtree is not a polynomial.
    Runs iteratively in a single `O(N)` pass with no recursion or repeated rescans.
    """
    computed: dict[int, Polynomial | None] = {}

    def leaf(current: Expression) -> bool | None:
        node_id = id(current)
        if node_id in computed:
            return True
        if isinstance(current, Number):
            computed[node_id] = {0: current.value}
            return True
        if current == variable:
            computed[node_id] = {1: 1.0}
            return True
        if isinstance(current, Name | Voltage | Current):
            computed[node_id] = None
            return True
        return None

    def combine(current: Expression, _built: list[bool]) -> bool:
        computed[id(current)] = _combine_polynomial(current, computed)
        return True

    fold(expression, leaf, combine)
    return computed


def _polynomial_integral(polynomial: Polynomial, variable: Expression) -> Expression:
    """Integrates a polynomial of `variable` term by term."""
    terms: list[Expression] = []
    for power, coefficient in sorted(polynomial.items()):
        term: Expression = Number(coefficient / (power + 1))
        for _ in range(power + 1):
            term = Binary("*", term, variable)
        terms.append(term)
    if not terms:
        return Number(0.0)
    return functools.reduce(lambda left, right: Binary("+", left, right), terms)


def _power_polynomial(base: Polynomial | None, power: float) -> Polynomial | None:
    """Raises `base` to integer `power`, up to `_MAX_POLYNOMIAL_DEGREE`."""
    if base is None or not (
        power.is_integer() and 0 <= power <= _MAX_POLYNOMIAL_DEGREE
    ):
        return None
    exponent = int(power)
    if base and max(base) * exponent > _MAX_POLYNOMIAL_DEGREE:
        return None
    product: Polynomial = {0: 1.0}
    for _ in range(exponent):
        multiplied = _multiplied(product, base)
        if multiplied is None:
            return None
        product = multiplied
    return product


def _scaled(polynomial: Polynomial, factor: float) -> Polynomial:
    """Scales every coefficient of `polynomial` by `factor`."""
    return {power: coefficient * factor for power, coefficient in polynomial.items()}


def _added(first: Polynomial, second: Polynomial) -> Polynomial:
    """Returns the sum of two polynomials."""
    total = dict(first)
    for power, coefficient in second.items():
        total[power] = total.get(power, 0.0) + coefficient
    return total


def _multiplied(first: Polynomial, second: Polynomial) -> Polynomial | None:
    """Multiplies two polynomials, returning `None` past `_MAX_POLYNOMIAL_DEGREE`."""
    if first and second and max(first) + max(second) > _MAX_POLYNOMIAL_DEGREE:
        return None
    product: Polynomial = {}
    for (one, a), (other, b) in itertools.product(first.items(), second.items()):
        product[one + other] = product.get(one + other, 0.0) + a * b
    return product


_ANTIDERIVATIVES: Final[dict[str, Callable[[Expression], Expression]]] = {
    "exp": lambda argument: Call("exp", (argument,)),
    "sinh": lambda argument: Call("cosh", (argument,)),
    "cosh": lambda argument: Call("sinh", (argument,)),
    "tanh": lambda argument: Call("ln", (Call("cosh", (argument,)),)),
    "sin": lambda argument: Unary("-", Call("cos", (argument,))),
    "cos": lambda argument: Call("sin", (argument,)),
}
"""Elementary antiderivatives for functions integrable over all real arguments."""


class _NotIntegrableError(Exception):
    """Internal control-flow signal when a subtree has no closed-form integral."""


def _check_integrable(current: Expression) -> None:
    """Raises `_NotIntegrableError` when a non-leaf node cannot be integrated."""
    match current:
        case (
            Unary("-" | "+", _)
            | Binary("+" | "-", _, _)
            | Binary("*", Number(), _)
            | Binary("*", _, Number())
        ):
            return
        case Binary("/", _, Number(divisor)) if divisor != 0:
            return
    raise _NotIntegrableError


def _combine_integral(expression: Expression, built: list[Expression]) -> Expression:
    """Builds the integrated parent node from already-integrated child results."""
    match expression:
        case Unary(operator, _):
            return Unary(operator, built[0])
        case Binary("+" | "-" as operator, _, _):
            return Binary(operator, built[0], built[1])
        case Binary("*", Number() as factor, _):
            return Binary("*", factor, built[1])
        case Binary("*", _, Number() as factor):
            return Binary("*", factor, built[0])
        case Binary("/", _, Number() as divisor):
            return Binary("/", built[0], divisor)
    raise _NotIntegrableError


def integral(
    expression: Expression, variable: Expression, depth: int = 0
) -> Expression | None:
    """Computes a closed-form antiderivative of `expression` for `variable`.

    Assumes `expression` contains only numeric constants and `variable`. Supports sums
    of polynomials in `variable` and scalar multiples of ``exp``, ``sinh``, ``cosh``,
    ``tanh``, ``sin``, or ``cos`` applied to linear functions of `variable`. Polynomial
    powers are integrated term by term, and elementary functions are divided by the
    linear slope. Returns `None` when no closed-form antiderivative is supported.

    Args:
        expression: Integrand expression containing constants and `variable`.
        variable: Integration variable expression node.
        depth: Nesting depth guard returning `None` past `_MAX_DEPTH`.

    Returns:
        Closed-form antiderivative expression, or `None` when unsupported.
    """
    if depth > _MAX_DEPTH:
        return None
    polynomials = _polynomial_map(expression, variable)

    def leaf(current: Expression) -> Expression | None:
        polynomial = polynomials.get(id(current))
        if polynomial is not None:
            return _polynomial_integral(polynomial, variable)
        if (
            isinstance(current, Call)
            and current.function in _ANTIDERIVATIVES
            and len(current.arguments) == 1
        ):
            argument = current.arguments[0]
            line = polynomials.get(id(argument))
            if line is None or not set(line) <= {0, 1} or line.get(1, 0.0) == 0:
                raise _NotIntegrableError
            antiderivative = _ANTIDERIVATIVES[current.function](argument)
            return Binary("/", antiderivative, Number(line[1]))
        return None

    try:
        return fold(expression, leaf, _combine_integral, before=_check_integrable)
    except _NotIntegrableError:
        return None


def names(expression: Expression) -> set[str]:
    """Returns the set of parameter and constant names referenced in `expression`.

    Args:
        expression: Expression tree to scan for identifier references.

    Returns:
        Case-folded parameter and constant names referenced in `expression`.
    """
    return {node.name for node in walk(expression) if isinstance(node, Name)}


STATISTICAL: Final = {"agauss": 3, "gauss": 3, "aunif": 2, "unif": 2, "limit": 2}
"""ngspice statistical functions mapped to their expected argument counts.

Matches numparam ``xpressn.c``, where the first argument is the nominal value.
Three-argument ``limit`` is the PSpice clamping function rather than statistical.
"""

type Draw = Callable[[str, tuple[float, ...]], float]
"""Callback sampling a statistical function from its name and numeric arguments."""


def nominal(_function: str, arguments: tuple[float, ...]) -> float:
    """Returns the nominal value (first argument) of a statistical function call.

    Args:
        _function: Statistical function name, unused during nominal evaluation.
        arguments: Evaluated numeric arguments, with the nominal value first.

    Returns:
        Nominal value from `arguments[0]`.
    """
    return arguments[0]


def evaluate(
    expression: Expression, parameters: Mapping[str, float], draw: Draw = nominal
) -> float:
    """Evaluates a parameter-only `expression` using `parameters` and `draw`.

    Statistical calls take values returned by `draw` (defaulting to `nominal`),
    evaluated in source order with inner calls first.

    Args:
        expression: Parameter-only expression to evaluate.
        parameters: Case-folded parameter values in scope.
        draw: Statistical sampling callback invoked on statistical function calls.

    Returns:
        Evaluated floating-point value of `expression`.

    Raises:
        SpiceError: When `expression` references a voltage, current, unknown
            identifier, or unknown function.
    """
    match expression:
        case Number(value) if math.isfinite(value):
            return float(value)
        case Unary("-", Number(value)) if math.isfinite(value):
            return -float(value)
        case Unary("+", Number(value)) if math.isfinite(value):
            return float(value)
        case Name(name) if name in parameters:
            value = parameters[name]
            if math.isfinite(value):
                return float(value)
        case Name(name) if name in _CONSTANTS:
            return _CONSTANTS[name]

    def unread(name: str) -> str:
        message = f"a parameter's expression reads no voltage or current, not `{name}`"
        raise SpiceError(message)

    drawn = _drawn(expression, parameters, draw)
    written = translate(drawn, parameters, unread, unread)
    check_core_depth(written)
    try:
        return _core.constant(written)
    except ValueError as error:
        raise SpiceError(str(error)) from None


def is_statistical(expression: Expression) -> bool:
    """Returns whether `expression` is a call to a SPICE statistical function.

    Args:
        expression: Expression node to classify.

    Returns:
        `True` when `expression` invokes a SPICE statistical distribution function.
    """
    if not isinstance(expression, Call):
        return False
    function = expression.function
    return function in STATISTICAL and not (
        function == "limit" and len(expression.arguments) != STATISTICAL[function]
    )


def _drawn(
    expression: Expression, parameters: Mapping[str, float], draw: Draw
) -> Expression:
    """Replaces statistical function calls in `expression` with sampled numeric values.

    Raises:
        SpiceError: When a statistical function receives an invalid argument count.
    """

    def check(node: Expression) -> None:
        if is_statistical(node) and isinstance(node, Call):
            expected = STATISTICAL[node.function]
            if len(node.arguments) != expected:
                message = (
                    f"`{node.function}` takes {expected} arguments, not "
                    f"{len(node.arguments)}"
                )
                raise SpiceError(message)

    def combine(node: Expression, built: list[Expression]) -> Expression:
        if is_statistical(node) and isinstance(node, Call):
            values = tuple(evaluate(argument, parameters, draw) for argument in built)
            return Number(draw(node.function, values))
        return rebuild(node, built)

    return rewrite(expression, before=check, combine=combine)


def _check_translated_length(length: int) -> None:
    """Checks that a translated expression stays within `_MAX_TRANSLATED_LENGTH`."""
    if length > _MAX_TRANSLATED_LENGTH:
        message = (
            f"translated expression length {length} exceeds the limit of "
            f"{_MAX_TRANSLATED_LENGTH} characters"
        )
        raise SpiceError(message)


def translate(
    expression: Expression,
    parameters: Mapping[str, float],
    node: Callable[[str], str],
    current: Callable[[str], str],
    *,
    symbolic: Mapping[str, str] | None = None,
) -> str:
    """Translates `expression` into the jolty expression language.

    Substitutes numeric values for parameters, emits ``p(<name>)`` for names present in
    `symbolic`, maps non-ground nodes through `node`, and maps voltage-source currents
    through `current`. Emits parentheses only where operator precedence or associativity
    requires them.

    Args:
        expression: SPICE expression tree to format.
        parameters: Case-folded parameter values substituted as numeric literals.
        node: Mapper converting non-ground SPICE node names to jolty net names.
        current: Mapper converting SPICE voltage-source names to jolty instance names.
        symbolic: Optional mapping from case-folded names to jolty parameter identifiers
            emitted as ``p(<identifier>)`` instead of numeric literals.

    Returns:
        Expression formatted in the jolty expression language.

    Raises:
        SpiceError: When `expression` references an unknown parameter or function, or
            when the translated text exceeds `_MAX_TRANSLATED_LENGTH`.
    """

    def leaf(part: Expression) -> tuple[str, int] | None:
        if isinstance(part, Number | Name | Voltage | Current):
            leaf_text = _leaf(part, parameters, node, current, symbolic)
            _check_translated_length(len(leaf_text))
            return leaf_text, _ATOM_PRECEDENCE
        return None

    def combine(part: Expression, built: list[tuple[str, int]]) -> tuple[str, int]:
        formatted = _format_node(part, built)
        _check_translated_length(len(formatted[0]))
        return formatted

    return fold(expression, leaf, combine)[0]


def _format_unary(operator: str, operand: tuple[str, int]) -> tuple[str, int]:
    """Formats a unary operation with minimal parentheses and space separation."""
    operand_text, operand_precedence = operand
    if operand_precedence < _UNARY_PRECEDENCE:
        _check_translated_length(len(operator) + len(operand_text) + 2)
        return f"{operator}({operand_text})", _UNARY_PRECEDENCE
    spacer = " " if operand_precedence == _UNARY_PRECEDENCE else ""
    _check_translated_length(len(operator) + len(spacer) + len(operand_text))
    return f"{operator}{spacer}{operand_text}", _UNARY_PRECEDENCE


def _format_binary(
    operator: str, left: tuple[str, int], right: tuple[str, int]
) -> tuple[str, int]:
    """Formats a left-associative binary operation with minimal parentheses."""
    precedence = _BIN_PRECEDENCE[operator]
    left_text, left_precedence = left
    right_text, right_precedence = right
    wrap_left = left_precedence < precedence
    wrap_right = right_precedence <= precedence or (
        right_precedence == _UNARY_PRECEDENCE and right_text.startswith(("+", "-"))
    )
    predicted = (
        len(left_text)
        + (2 if wrap_left else 0)
        + len(operator)
        + 2
        + len(right_text)
        + (2 if wrap_right else 0)
    )
    _check_translated_length(predicted)
    left_out = f"({left_text})" if wrap_left else left_text
    right_out = f"({right_text})" if wrap_right else right_text
    return f"{left_out} {operator} {right_out}", precedence


def _format_node(
    expression: Expression, built: list[tuple[str, int]]
) -> tuple[str, int]:
    """Formats a non-leaf `expression` from `(text, precedence)` child entries."""
    match expression:
        case Unary(operator, _):
            return _format_unary(operator, built[0])
        case Binary("^", _, _) | Call("pow", (_, _)):
            return _power(built[0][0], built[1][0]), _ATOM_PRECEDENCE
        case Binary(operator, _, _):
            return _format_binary(operator, built[0], built[1])
        case Conditional():
            condition, then_branch, otherwise = (entry[0] for entry in built)
            predicted = len(condition) + len(then_branch) + len(otherwise) + 8
            _check_translated_length(predicted)
            return f"if({condition}, {then_branch}, {otherwise})", _ATOM_PRECEDENCE
        case Call(function, _):
            return _call(function, [entry[0] for entry in built]), _ATOM_PRECEDENCE
    message = f"unexpected expression node: {type(expression).__name__}"
    raise SpiceError(message)


def _power(base: str, exponent: str) -> str:
    """Formats `base` raised to `exponent` in the jolty expression language.

    Preserves the sign of a negative base when `exponent` is a constant odd integer,
    matching ngspice parameter and behavioral-source evaluation (``xpressn.c``, lines
    136 to 137 and 786 to 803; ``ptfuncs.c``, lines 69 to 87). All other exponents
    apply ``pow`` to ``abs(base)`` to avoid NaN results on negative bases.
    """
    power = math.nan
    if nesting_depth(exponent) <= _MAX_CORE_DEPTH:
        try:
            power = _core.constant(exponent)
        except ValueError:
            power = math.nan
    magnitude_length = len(base) + len(exponent) + 12
    is_odd_integer = power.is_integer() and bool(power % 2)
    if is_odd_integer:
        _check_translated_length(len(base) + magnitude_length + 10)
    else:
        _check_translated_length(magnitude_length)
    magnitude = f"pow(abs({base}), {exponent})"
    if is_odd_integer:
        return f"(sgn({base}) * {magnitude})"
    return magnitude


def _leaf(
    leaf: Number | Name | Voltage | Current,
    parameters: Mapping[str, float],
    node: Callable[[str], str],
    current: Callable[[str], str],
    symbolic: Mapping[str, str] | None = None,
) -> str:
    """Formats a leaf expression node in the jolty expression language.

    Raises:
        UndefinedError: When a `Name` node matches no parameter or built-in constant.
    """
    match leaf:
        case Number(value):
            return _number(value)
        case Name(name) if symbolic is not None and name in symbolic:
            return f"p({symbolic[name]})"
        case Name(name) if name in parameters:
            return _number(parameters[name])
        case Name(name) if name in _CONSTANTS:
            return _number(_CONSTANTS[name])
        case Name(name):
            message = f"no parameter is named `{name}`"
            raise UndefinedError(message, name)
        case Voltage(positive, negative):
            return _voltage(positive, negative, node)
        case Current(source):
            return f"i({current(source)})"


def _voltage(positive: str, negative: str | None, node: Callable[[str], str]) -> str:
    """Formats ``V(a, b)`` in the jolty expression language, omitting ground ``0``."""
    match positive, negative:
        case "0", None | "0":
            return "0.0"
        case "0", str(other):
            return f"(-v({node(other)}))"
        case _, None | "0":
            return f"v({node(positive)})"
        case _, str(other):
            return f"v({node(positive)}, {node(other)})"


def _call(function: str, arguments: list[str]) -> str:
    if function not in _FUNCTIONS:
        message = f"no function is named `{function}`"
        raise SpiceError(message)
    arity, written = _FUNCTIONS[function]
    if arity is not None and len(arguments) != arity:
        message = f"`{function}` takes {arity} arguments, not {len(arguments)}"
        raise SpiceError(message)
    if "{" in written:
        predicted = len(written)
        for index, argument in enumerate(arguments):
            placeholder = f"{{{index}}}"
            count = written.count(placeholder)
            predicted += count * (len(argument) - len(placeholder))
        _check_translated_length(predicted)
        return written.format(*arguments)
    predicted = (
        len(written)
        + 2
        + sum(len(argument) for argument in arguments)
        + 2 * max(0, len(arguments) - 1)
    )
    _check_translated_length(predicted)
    return f"{written}({', '.join(arguments)})"


def _number(value: float) -> str:
    """Formats numeric `value` for jolty expressions, parenthesizing negatives."""
    written = repr(value)
    return f"({written})" if written.startswith("-") else written


_NO_FUNCTIONS: Final[Mapping[str, Function]] = MappingProxyType({})
"""Empty user-function mapping used when `evaluate_dual` receives no `.func` table."""

_LN_10: Final = math.log(10.0)
"""Natural logarithm of 10 used for `log10` differentiation."""

_MAX_TABLE_ROUNDING: Final = 0.5
"""Maximum corner-rounding fraction accepted by `rounded_table`."""

_MIN_TABLE_ARGUMENTS: Final = 3
"""Minimum argument count for `table` and `pwl` (input plus one `(x, y)` pair)."""

_MIN_ROUNDED_TABLE_ARGUMENTS: Final = 4
"""Minimum argument count for `rounded_table` (input, rounding, plus one pair)."""


def _scale_derivatives[Seed](
    derivatives: Mapping[Seed, float], factor: float
) -> dict[Seed, float]:
    """Scales each partial derivative in `derivatives` by `factor`, dropping zeros.

    Returns an empty dictionary without touching `factor` when `derivatives` is empty,
    preventing `0 * inf` or `0 * nan` on constant operands.
    """
    if not derivatives:
        return {}
    result: dict[Seed, float] = {}
    for seed, slope in derivatives.items():
        scaled = slope * factor
        if scaled != 0.0:
            result[seed] = scaled
    return result


def _combine_derivatives[Seed](
    first: Mapping[Seed, float],
    first_scale: float,
    second: Mapping[Seed, float],
    second_scale: float,
) -> dict[Seed, float]:
    """Computes `first_scale * first + second_scale * second`, dropping exact zeros."""
    if not first:
        return _scale_derivatives(second, second_scale)
    if not second:
        return _scale_derivatives(first, first_scale)
    result = _scale_derivatives(first, first_scale)
    for seed, slope in second.items():
        combined = result.get(seed, 0.0) + slope * second_scale
        if combined == 0.0:
            result.pop(seed, None)
        else:
            result[seed] = combined
    return result


def _safe_divide(numerator: float, denominator: float) -> float:
    """Computes IEEE-754 floating-point division without raising `ZeroDivisionError`."""
    if denominator == 0.0:
        if numerator == 0.0 or math.isnan(numerator):
            return math.nan
        sign = math.copysign(1.0, numerator) * math.copysign(1.0, denominator)
        return math.copysign(math.inf, sign)
    return numerator / denominator


def _safe_pow(base: float, exponent: float) -> float:
    """Computes IEEE-754 `base ** exponent` for `base >= 0` without raising."""
    if math.isnan(base) or math.isnan(exponent):
        return math.nan
    if exponent == 0.0:
        return 1.0
    if base == 0.0:
        return 0.0 if exponent > 0.0 else math.inf
    try:
        return math.pow(base, exponent)
    except OverflowError:
        return math.inf
    except ValueError:
        return math.nan


def _sign(value: float) -> float:
    """Returns `-1.0`, `0.0`, or `1.0` according to the sign of `value` (or `NaN`)."""
    if math.isnan(value):
        return math.nan
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _unit_step(value: float) -> float:
    """Returns `0.0` below zero, `0.5` at zero, and `1.0` above zero (or `NaN`)."""
    if math.isnan(value):
        return math.nan
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return 0.0
    return 0.5


@dataclass(frozen=True, slots=True)
class Dual[Seed]:
    """Primal scalar value paired with exact first-order partial derivatives.

    `derivatives` maps each differentiated seed to its nonzero partial derivative.
    Omitted seeds have derivative zero.

    Examples:
        >>> x = Dual.seed(3.0, "x")
        >>> (x * x + 2.0 * x).derivative("x")
        8.0
    """

    value: float
    derivatives: Mapping[Seed, float] = field(default_factory=dict)

    @classmethod
    def constant(cls, value: float) -> Dual[Seed]:
        """Returns a constant dual number with zero derivatives."""
        return cls(float(value))

    @classmethod
    def seed(cls, value: float, key: Seed) -> Dual[Seed]:
        """Returns a dual number seeded with unit derivative for `key`."""
        return cls(float(value), {key: 1.0})

    def derivative(self, key: Seed) -> float:
        """Returns the partial derivative with respect to `key` (`0.0` when absent)."""
        return self.derivatives.get(key, 0.0)

    def __pos__(self) -> Dual[Seed]:
        """Returns `self` unchanged."""
        return self

    def __neg__(self) -> Dual[Seed]:
        """Returns the negated dual number."""
        return Dual(-self.value, _scale_derivatives(self.derivatives, -1.0))

    def __add__(self, other: Dual[Seed] | float) -> Dual[Seed]:
        """Returns the sum of `self` and `other`."""
        if isinstance(other, Dual):
            return Dual(
                self.value + other.value,
                _combine_derivatives(self.derivatives, 1.0, other.derivatives, 1.0),
            )
        return Dual(self.value + float(other), self.derivatives)

    def __radd__(self, other: float) -> Dual[Seed]:
        """Returns the sum of scalar `other` and `self`."""
        return Dual(float(other) + self.value, self.derivatives)

    def __sub__(self, other: Dual[Seed] | float) -> Dual[Seed]:
        """Returns the difference `self - other`."""
        if isinstance(other, Dual):
            return Dual(
                self.value - other.value,
                _combine_derivatives(self.derivatives, 1.0, other.derivatives, -1.0),
            )
        return Dual(self.value - float(other), self.derivatives)

    def __rsub__(self, other: float) -> Dual[Seed]:
        """Returns the difference `other - self` for scalar `other`."""
        return Dual(
            float(other) - self.value,
            _scale_derivatives(self.derivatives, -1.0),
        )

    def __mul__(self, other: Dual[Seed] | float) -> Dual[Seed]:
        """Returns the product `self * other`."""
        if isinstance(other, Dual):
            return Dual(
                self.value * other.value,
                _combine_derivatives(
                    self.derivatives, other.value, other.derivatives, self.value
                ),
            )
        factor = float(other)
        return Dual(self.value * factor, _scale_derivatives(self.derivatives, factor))

    def __rmul__(self, other: float) -> Dual[Seed]:
        """Returns the product `other * self` for scalar `other`."""
        factor = float(other)
        return Dual(factor * self.value, _scale_derivatives(self.derivatives, factor))

    def __truediv__(self, other: Dual[Seed] | float) -> Dual[Seed]:
        """Returns the quotient `self / other`."""
        if isinstance(other, Dual):
            quotient = _safe_divide(self.value, other.value)
            left_scale = _safe_divide(1.0, other.value) if self.derivatives else 0.0
            right_scale = (
                _safe_divide(-quotient, other.value) if other.derivatives else 0.0
            )
            return Dual(
                quotient,
                _combine_derivatives(
                    self.derivatives, left_scale, other.derivatives, right_scale
                ),
            )
        divisor = float(other)
        quotient = _safe_divide(self.value, divisor)
        scale = _safe_divide(1.0, divisor) if self.derivatives else 0.0
        return Dual(quotient, _scale_derivatives(self.derivatives, scale))

    def __rtruediv__(self, other: float) -> Dual[Seed]:
        """Returns the quotient `other / self` for scalar `other`."""
        numerator = float(other)
        quotient = _safe_divide(numerator, self.value)
        scale = _safe_divide(-quotient, self.value) if self.derivatives else 0.0
        return Dual(quotient, _scale_derivatives(self.derivatives, scale))


type Scalar[Seed] = float | Dual[Seed]
"""Plain floating-point constant or first-order dual value."""

type DualDraw[Seed] = Callable[[str, tuple[Dual[Seed], ...]], Dual[Seed]]
"""Callback evaluating a SPICE statistical call in dual-number mode."""


def nominal_dual[Seed](_function: str, arguments: tuple[Dual[Seed], ...]) -> Dual[Seed]:
    """Returns the nominal value (first argument) of a statistical call in dual mode.

    Args:
        _function: Statistical function name, unused during nominal evaluation.
        arguments: Evaluated dual arguments, with the nominal expression first.

    Returns:
        First argument `arguments[0]` preserving its primal value and derivatives.
    """
    return arguments[0]


def primal[Seed](value: Scalar[Seed]) -> float:
    """Returns the primal floating-point value of a scalar or `Dual`."""
    return value.value if isinstance(value, Dual) else float(value)


def is_finite[Seed](value: Scalar[Seed]) -> bool:
    """Returns whether `value` and all of its partial derivatives are finite."""
    if isinstance(value, Dual):
        return math.isfinite(value.value) and all(
            math.isfinite(slope) for slope in value.derivatives.values()
        )
    return math.isfinite(value)


def has_derivatives[Seed](value: Scalar[Seed]) -> bool:
    """Returns whether `value` carries at least one nonzero partial derivative."""
    return isinstance(value, Dual) and bool(value.derivatives)


def compact[Seed](value: Dual[Seed]) -> Scalar[Seed]:
    """Unwraps a derivative-free `Dual` to a plain `float` to keep constants fast."""
    return value if value.derivatives else value.value


def dual_min[Seed](left: Scalar[Seed], right: Scalar[Seed]) -> Scalar[Seed]:
    """Returns the smaller of `left` and `right`, preserving active derivatives."""
    return right if primal(right) < primal(left) else left


def dual_max[Seed](left: Scalar[Seed], right: Scalar[Seed]) -> Scalar[Seed]:
    """Returns the larger of `left` and `right`, preserving active derivatives."""
    return right if primal(right) > primal(left) else left


def dual_power[Seed](base: Scalar[Seed], exponent: Scalar[Seed]) -> Scalar[Seed]:
    """Computes `base ** exponent` for non-negative `base` in `float` or `Dual`.

    Args:
        base: Non-negative real scalar or dual base.
        exponent: Real scalar or dual exponent.

    Returns:
        Exact `base ** exponent` as a `float` when both operands are floats, or a
        `Dual` propagating exact base and exponent derivatives otherwise.
    """
    if not isinstance(base, Dual) and not isinstance(exponent, Dual):
        return math.pow(base, exponent)
    left = base if isinstance(base, Dual) else Dual[Seed](float(base))
    right = exponent if isinstance(exponent, Dual) else Dual[Seed](float(exponent))
    return compact(_dual_spice_power(left, right, signed=False, odd_signed=True))


def _dual_spice_power[Seed](
    base: Dual[Seed],
    exponent: Dual[Seed],
    *,
    signed: bool,
    odd_signed: bool,
) -> Dual[Seed]:
    """Computes SPICE `^`/`pow`, `pwr`, or `pwrs` with exact first-order derivatives."""
    a, b = base.value, exponent.value
    if math.isnan(a) or math.isnan(b):
        return Dual(
            math.nan,
            _combine_derivatives(
                base.derivatives, math.nan, exponent.derivatives, math.nan
            ),
        )
    if a > 0.0:
        value = _safe_pow(a, b)
        base_slope = (
            0.0 if not base.derivatives or b == 0.0 else b * _safe_pow(a, b - 1.0)
        )
        exponent_slope = value * math.log(a) if exponent.derivatives else 0.0
        return Dual(
            value,
            _combine_derivatives(
                base.derivatives, base_slope, exponent.derivatives, exponent_slope
            ),
        )
    if a == 0.0:
        magnitude = _safe_pow(0.0, b)
        value = _sign(0.0) * magnitude if signed else magnitude
        if (
            not base.derivatives
            or (not signed and not odd_signed)
            or b == 0.0
            or b > 1.0
        ):
            base_slope = 0.0
        elif b == 1.0:
            base_slope = 1.0
        else:
            base_slope = b * _safe_pow(0.0, b - 1.0)
        return Dual(value, _scale_derivatives(base.derivatives, base_slope))
    magnitude = _safe_pow(-a, b)
    is_negative = signed or (
        odd_signed and math.isfinite(b) and b.is_integer() and bool(b % 2)
    )
    value = -magnitude if is_negative else magnitude
    if not base.derivatives or b == 0.0:
        base_slope = 0.0
    else:
        magnitude_slope = b * _safe_pow(-a, b - 1.0)
        base_slope = magnitude_slope if is_negative else -magnitude_slope
    exponent_slope = value * math.log(-a) if exponent.derivatives else 0.0
    return Dual(
        value,
        _combine_derivatives(
            base.derivatives, base_slope, exponent.derivatives, exponent_slope
        ),
    )


def _safe_exp(value: float) -> float:
    """Computes `math.exp(value)`, returning `+inf` on overflow."""
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _safe_sinh(value: float) -> float:
    """Computes `math.sinh(value)`, returning signed infinity on overflow."""
    try:
        return math.sinh(value)
    except OverflowError:
        return math.copysign(math.inf, value)


def _safe_cosh(value: float) -> float:
    """Computes `math.cosh(value)`, returning `+inf` on overflow."""
    try:
        return math.cosh(value)
    except OverflowError:
        return math.inf


def _sqrt_pair(value: float) -> tuple[float, float]:
    """Returns `(sqrt(value), d(sqrt)/dx)` safely for all IEEE-754 inputs."""
    if value < 0.0 or math.isnan(value):
        return math.nan, math.nan
    root = math.sqrt(value)
    return root, _safe_divide(0.5, root)


def _exp_pair(value: float) -> tuple[float, float]:
    """Returns `(exp(value), exp(value))` safely for all IEEE-754 inputs."""
    computed = _safe_exp(value)
    return computed, computed


def _ln_pair(value: float) -> tuple[float, float]:
    """Returns `(ln(value), 1 / value)` safely for all IEEE-754 inputs."""
    if value == 0.0:
        return -math.inf, _safe_divide(1.0, value)
    if value < 0.0 or math.isnan(value):
        return math.nan, math.nan
    return math.log(value), 1.0 / value


def _log10_pair(value: float) -> tuple[float, float]:
    """Returns `(log10(value), 1 / (value * ln(10)))` safely for all IEEE-754 inputs."""
    if value == 0.0:
        return -math.inf, _safe_divide(1.0, value * _LN_10)
    if value < 0.0 or math.isnan(value):
        return math.nan, math.nan
    return math.log10(value), 1.0 / (value * _LN_10)


def _sin_pair(value: float) -> tuple[float, float]:
    """Returns `(sin(value), cos(value))`, or `(NaN, NaN)` on non-finite inputs."""
    if not math.isfinite(value):
        return math.nan, math.nan
    return math.sin(value), math.cos(value)


def _cos_pair(value: float) -> tuple[float, float]:
    """Returns `(cos(value), -sin(value))`, or `(NaN, NaN)` on non-finite inputs."""
    if not math.isfinite(value):
        return math.nan, math.nan
    return math.cos(value), -math.sin(value)


def _tan_pair(value: float) -> tuple[float, float]:
    """Returns `(tan(value), 1 + tan(value)^2)`, or `(NaN, NaN)` when non-finite."""
    if not math.isfinite(value):
        return math.nan, math.nan
    tangent = math.tan(value)
    return tangent, 1.0 + tangent * tangent


def _asin_pair(value: float) -> tuple[float, float]:
    """Returns `(asin(value), 1 / sqrt(1 - value^2))` safely."""
    if not (-1.0 <= value <= 1.0):
        return math.nan, math.nan
    return math.asin(value), _safe_divide(1.0, math.sqrt(max(0.0, 1.0 - value * value)))


def _acos_pair(value: float) -> tuple[float, float]:
    """Returns `(acos(value), -1 / sqrt(1 - value^2))` safely."""
    if not (-1.0 <= value <= 1.0):
        return math.nan, math.nan
    return (
        math.acos(value),
        _safe_divide(-1.0, math.sqrt(max(0.0, 1.0 - value * value))),
    )


def _atan_pair(value: float) -> tuple[float, float]:
    """Returns `(atan(value), 1 / (1 + value^2))` safely."""
    if math.isnan(value):
        return math.nan, math.nan
    return math.atan(value), 0.0 if math.isinf(value) else 1.0 / (1.0 + value * value)


def _sinh_pair(value: float) -> tuple[float, float]:
    """Returns `(sinh(value), cosh(value))` safely."""
    return _safe_sinh(value), _safe_cosh(value)


def _cosh_pair(value: float) -> tuple[float, float]:
    """Returns `(cosh(value), sinh(value))` safely."""
    return _safe_cosh(value), _safe_sinh(value)


def _tanh_pair(value: float) -> tuple[float, float]:
    """Returns `(tanh(value), 1 - tanh(value)^2)` safely."""
    hyperbolic = math.tanh(value)
    return hyperbolic, 1.0 - hyperbolic * hyperbolic


def _asinh_pair(value: float) -> tuple[float, float]:
    """Returns `(asinh(value), 1 / sqrt(value^2 + 1))` safely."""
    if math.isnan(value):
        return math.nan, math.nan
    slope = 0.0 if math.isinf(value) else 1.0 / math.hypot(value, 1.0)
    return math.asinh(value), slope


def _acosh_pair(value: float) -> tuple[float, float]:
    """Returns `(acosh(value), 1 / sqrt(value^2 - 1))` safely."""
    if value < 1.0 or math.isnan(value):
        return math.nan, math.nan
    slope = (
        0.0
        if math.isinf(value)
        else _safe_divide(1.0, math.sqrt(max(0.0, (value - 1.0) * (value + 1.0))))
    )
    return math.acosh(value), slope


def _atanh_pair(value: float) -> tuple[float, float]:
    """Returns `(atanh(value), 1 / (1 - value^2))` safely."""
    if abs(value) == 1.0:
        return math.copysign(math.inf, value), math.inf
    if not (-1.0 < value < 1.0):
        return math.nan, math.nan
    return math.atanh(value), 1.0 / (1.0 - value * value)


_UNARY_DUAL_FUNCTIONS: Final[dict[str, Callable[[float], tuple[float, float]]]] = {
    "abs": lambda value: (abs(value), _sign(value)),
    "sqrt": _sqrt_pair,
    "exp": _exp_pair,
    "ln": _ln_pair,
    "log": _ln_pair,
    "log10": _log10_pair,
    "sin": _sin_pair,
    "cos": _cos_pair,
    "tan": _tan_pair,
    "asin": _asin_pair,
    "acos": _acos_pair,
    "atan": _atan_pair,
    "sinh": _sinh_pair,
    "cosh": _cosh_pair,
    "tanh": _tanh_pair,
    "asinh": _asinh_pair,
    "acosh": _acosh_pair,
    "atanh": _atanh_pair,
}
"""Smooth unary SPICE functions mapped to `(value, slope)` evaluators."""

_UNARY_STEP_FUNCTIONS: Final[dict[str, Callable[[float], float]]] = {
    "floor": lambda value: float(math.floor(value)) if math.isfinite(value) else value,
    "ceil": lambda value: float(math.ceil(value)) if math.isfinite(value) else value,
    "int": lambda value: float(math.trunc(value)) if math.isfinite(value) else value,
    "sgn": _sign,
    "sign": _sign,
    "u": _unit_step,
    "stp": _unit_step,
}
"""Piecewise-constant unary SPICE functions with zero derivative away from jumps."""

_COMPARISONS: Final[dict[str, Callable[[float, float], bool]]] = {
    "<": lambda left, right: left < right,
    "<=": lambda left, right: left <= right,
    ">": lambda left, right: left > right,
    ">=": lambda left, right: left >= right,
    "==": lambda left, right: left == right,
    "!=": lambda left, right: left != right,
    "&&": lambda left, right: left != 0.0 and right != 0.0,
    "||": lambda left, right: left != 0.0 or right != 0.0,
}
"""Comparison and logical operators returning `1.0` or `0.0` with zero derivative."""


def _dual_unary[Seed](operator: str, operand: Dual[Seed]) -> Dual[Seed]:
    """Applies unary operator `+`, `-`, or `!` to `operand`."""
    if operator == "+":
        return operand
    if operator == "-":
        return -operand
    if operator == "!":
        return Dual(1.0 if operand.value == 0.0 else 0.0)
    message = f"unexpected unary operator `{operator}`"
    raise SpiceError(message)


def _dual_binary[Seed](
    operator: str, left: Dual[Seed], right: Dual[Seed]
) -> Dual[Seed]:
    """Applies binary operator `operator` to `left` and `right`."""
    match operator:
        case "+":
            return left + right
        case "-":
            return left - right
        case "*":
            return left * right
        case "/":
            return left / right
        case "^":
            return _dual_spice_power(left, right, signed=False, odd_signed=True)
    comparison = _COMPARISONS.get(operator)
    if comparison is not None:
        return Dual(1.0 if comparison(left.value, right.value) else 0.0)
    message = f"unexpected binary operator `{operator}`"
    raise SpiceError(message)


def _dual_limit[Seed](
    value: Dual[Seed], first: Dual[Seed], second: Dual[Seed]
) -> Dual[Seed]:
    """Clamps `value` between `first` and `second` in either order, keeping slope."""
    low, high = (second, first) if first.value > second.value else (first, second)
    if value.value > high.value:
        return high
    if value.value < low.value:
        return low
    return value


def _extract_table_points[Seed](
    function: str, raw_points: list[Dual[Seed]], *, rounded: bool
) -> list[tuple[Dual[Seed], Dual[Seed]]]:
    """Validates and pairs ascending `(abscissa, ordinate)` entries for table calls."""
    if not raw_points or len(raw_points) % 2 != 0:
        qualifier = " a rounding from 0 to 0.5," if rounded else ""
        message = (
            f"`{function}` takes an input,{qualifier} then constant points, in pairs, "
            "with ascending abscissae"
        )
        raise SpiceError(message)
    points = [
        (raw_points[index], raw_points[index + 1])
        for index in range(0, len(raw_points), 2)
    ]
    for index, (abscissa, ordinate) in enumerate(points):
        if not (math.isfinite(abscissa.value) and math.isfinite(ordinate.value)):
            message = f"`{function}` points must be finite"
            raise SpiceError(message)
        if index > 0 and not (points[index - 1][0].value < abscissa.value):
            message = f"`{function}` requires strictly ascending abscissae"
            raise SpiceError(message)
    return points


def _dual_table[Seed](function: str, arguments: list[Dual[Seed]]) -> Dual[Seed]:
    """Evaluates `table` or `pwl` (linearly extended beyond endpoints) in dual mode."""
    if len(arguments) < _MIN_TABLE_ARGUMENTS:
        message = (
            f"`{function}` takes an input, then constant points, in pairs, "
            "with ascending abscissae"
        )
        raise SpiceError(message)
    input_value = arguments[0]
    points = _extract_table_points(function, arguments[1:], rounded=False)
    if math.isnan(input_value.value):
        return Dual(math.nan, _scale_derivatives(input_value.derivatives, math.nan))
    if len(points) == 1:
        return points[0][1]
    if input_value.value <= points[0][0].value:
        left, right = points[0], points[1]
    elif input_value.value >= points[-1][0].value:
        left, right = points[-2], points[-1]
    else:
        abscissae = [point[0].value for point in points]
        upper = min(
            max(bisect.bisect_right(abscissae, input_value.value), 1),
            len(points) - 1,
        )
        left, right = points[upper - 1], points[upper]
    slope = (right[1] - left[1]) / (right[0] - left[0])
    return left[1] + (input_value - left[0]) * slope


def _dual_rounded_table[Seed](arguments: list[Dual[Seed]]) -> Dual[Seed]:
    """Evaluates `rounded_table` matching `jolty_expressions::tape::Points` in dual."""
    if len(arguments) < _MIN_ROUNDED_TABLE_ARGUMENTS:
        message = (
            "`rounded_table` takes an input, a rounding from 0 to 0.5, then constant "
            "points, in pairs, with ascending abscissae"
        )
        raise SpiceError(message)
    input_value = arguments[0]
    rounding = arguments[1]
    if not (0.0 <= rounding.value <= _MAX_TABLE_ROUNDING):
        message = (
            "`rounded_table` takes an input, a rounding from 0 to 0.5, then constant "
            "points, in pairs, with ascending abscissae"
        )
        raise SpiceError(message)
    points = _extract_table_points("rounded_table", arguments[2:], rounded=True)
    if math.isnan(input_value.value):
        return Dual(math.nan, _scale_derivatives(input_value.derivatives, math.nan))
    last = len(points) - 1
    if last == 0:
        before_x = points[0][0] - 1.0
        after_x = points[0][0] + 1.0
    else:
        before_x = points[0][0] * 2.0 - points[1][0]
        after_x = points[last][0] * 2.0 - points[last - 1][0]
    xs = [before_x, *(point[0] for point in points), after_x]
    ys = [points[0][1], *(point[1] for point in points), points[last][1]]
    end = len(xs) - 1

    def segment(anchor: int, start: int, stop: int) -> Dual[Seed]:
        slope = (ys[stop] - ys[start]) / (xs[stop] - xs[start])
        return ys[anchor] + (input_value - xs[anchor]) * slope

    if input_value.value <= 0.5 * (xs[0].value + xs[1].value):
        return segment(0, 0, 1)
    if input_value.value >= 0.5 * (xs[end - 1].value + xs[end].value):
        return segment(end, end - 1, end)
    primal_xs = [entry.value for entry in xs]
    upper = bisect.bisect_right(primal_xs, input_value.value)
    pivot = (
        upper - 1
        if input_value.value < 0.5 * (xs[upper - 1].value + xs[upper].value)
        else upper
    )
    below = xs[pivot] - xs[pivot - 1]
    above = xs[pivot + 1] - xs[pivot]
    shorter = above if above.value < below.value else below
    domain = rounding * shorter
    if input_value.value < xs[pivot].value - domain.value:
        return segment(pivot, pivot - 1, pivot)
    if input_value.value >= xs[pivot].value + domain.value:
        return segment(pivot, pivot, pivot + 1)
    lower_slope = (ys[pivot] - ys[pivot - 1]) / below
    upper_slope = (ys[pivot + 1] - ys[pivot]) / above
    upper_x = xs[pivot] + domain
    upper_y = ys[pivot] + upper_slope * domain
    curvature = (upper_slope - lower_slope) / (domain * 4.0)
    linear = upper_slope - curvature * upper_x * 2.0
    constant = upper_y - curvature * upper_x * upper_x - linear * upper_x
    return curvature * input_value * input_value + linear * input_value + constant


def _dual_uramp[Seed](operand: Dual[Seed]) -> Dual[Seed]:
    """Evaluates `uramp(operand) = max(operand, 0)` with exact active slope."""
    if math.isnan(operand.value):
        return Dual(math.nan, _scale_derivatives(operand.derivatives, math.nan))
    return operand if operand.value > 0.0 else Dual(0.0)


def _dual_min[Seed](arguments: list[Dual[Seed]]) -> Dual[Seed]:
    """Evaluates `min(a, b)` returning the active dual branch."""
    return arguments[1] if arguments[1].value < arguments[0].value else arguments[0]


def _dual_max[Seed](arguments: list[Dual[Seed]]) -> Dual[Seed]:
    """Evaluates `max(a, b)` returning the active dual branch."""
    return arguments[1] if arguments[1].value > arguments[0].value else arguments[0]


def _dual_if[Seed](arguments: list[Dual[Seed]]) -> Dual[Seed]:
    """Evaluates `if(condition, when_true, when_false)` selecting the active branch."""
    return arguments[2] if arguments[0].value == 0.0 else arguments[1]


def _dual_pow[Seed](arguments: list[Dual[Seed]]) -> Dual[Seed]:
    """Evaluates SPICE `pow(base, exponent)` in dual mode."""
    return _dual_spice_power(arguments[0], arguments[1], signed=False, odd_signed=True)


def _dual_pwr[Seed](arguments: list[Dual[Seed]]) -> Dual[Seed]:
    """Evaluates SPICE `pwr(base, exponent)` in dual mode."""
    return _dual_spice_power(arguments[0], arguments[1], signed=False, odd_signed=False)


def _dual_pwrs[Seed](arguments: list[Dual[Seed]]) -> Dual[Seed]:
    """Evaluates SPICE `pwrs(base, exponent)` in dual mode."""
    return _dual_spice_power(arguments[0], arguments[1], signed=True, odd_signed=False)


def _dual_special_call[Seed](function: str, arguments: list[Dual[Seed]]) -> Dual[Seed]:
    """Evaluates multi-argument and special built-in SPICE functions in dual mode."""
    handlers: dict[str, Callable[[list[Dual[Seed]]], Dual[Seed]]] = {
        "uramp": lambda values: _dual_uramp(values[0]),
        "min": _dual_min,
        "max": _dual_max,
        "pow": _dual_pow,
        "pwr": _dual_pwr,
        "pwrs": _dual_pwrs,
        "limit": lambda values: _dual_limit(values[0], values[1], values[2]),
        "if": _dual_if,
        "table": lambda values: _dual_table("table", values),
        "pwl": lambda values: _dual_table("pwl", values),
        "rounded_table": _dual_rounded_table,
    }
    handler = handlers.get(function)
    if handler is None:
        message = f"no function is named `{function}`"
        raise SpiceError(message)
    return handler(arguments)


def _dual_call[Seed](function: str, arguments: list[Dual[Seed]]) -> Dual[Seed]:
    """Evaluates built-in SPICE `function` on dual `arguments`."""
    if function not in _FUNCTIONS:
        message = f"no function is named `{function}`"
        raise SpiceError(message)
    arity, _ = _FUNCTIONS[function]
    if arity is not None and len(arguments) != arity:
        message = f"`{function}` takes {arity} arguments, not {len(arguments)}"
        raise SpiceError(message)
    if (smooth := _UNARY_DUAL_FUNCTIONS.get(function)) is not None:
        operand = arguments[0]
        value, slope = smooth(operand.value)
        return Dual(value, _scale_derivatives(operand.derivatives, slope))
    if (step := _UNARY_STEP_FUNCTIONS.get(function)) is not None:
        return Dual(step(arguments[0].value))
    return _dual_special_call(function, arguments)


def _dual_leaf[Seed](
    node: Number | Name | Voltage | Current,
    parameters: Mapping[str, Scalar[Seed]],
) -> Dual[Seed]:
    """Resolves a leaf expression node into a `Dual` value."""
    match node:
        case Number(value):
            return Dual(float(value))
        case Name(name) if name in parameters:
            found = parameters[name]
            return found if isinstance(found, Dual) else Dual(float(found))
        case Name(name) if name in _CONSTANTS:
            return Dual(_CONSTANTS[name])
        case Name(name):
            message = f"no parameter is named `{name}`"
            raise UndefinedError(message, name)
        case Voltage(positive, _):
            message = (
                "a parameter's expression reads no voltage or current, not "
                f"`{positive}`"
            )
            raise SpiceError(message)
        case Current(source):
            message = (
                f"a parameter's expression reads no voltage or current, not `{source}`"
            )
            raise SpiceError(message)


def _dual_fast[Seed](
    expression: Expression, parameters: Mapping[str, Scalar[Seed]]
) -> Dual[Seed] | None:
    """Fast-paths literals, signed numbers, and known names in `evaluate_dual`."""
    match expression:
        case Number(value):
            return Dual(float(value))
        case Unary("-", Number(value)):
            return Dual(-float(value))
        case Unary("+", Number(value)):
            return Dual(float(value))
        case Name(name) if name in parameters:
            found = parameters[name]
            return found if isinstance(found, Dual) else Dual(float(found))
        case Name(name) if name in _CONSTANTS:
            return Dual(_CONSTANTS[name])
    return None


def _validate_dual_call(
    node: Call,
    functions: Mapping[str, Function],
    calling: tuple[str, ...],
) -> None:
    """Checks user-function recursion and call arity before descending into a `Call`."""
    if node.function in functions:
        _check_call(node.function, len(node.arguments), functions, calling)
        if len(calling) >= _MAX_DEPTH:
            message = f"function calls nest more than {_MAX_DEPTH} levels deep"
            raise SpiceError(message)
    elif is_statistical(node):
        expected = STATISTICAL[node.function]
        if len(node.arguments) != expected:
            message = (
                f"`{node.function}` takes {expected} arguments, not "
                f"{len(node.arguments)}"
            )
            raise SpiceError(message)


def _reduce_dual_node[Seed](
    node: Expression,
    built: list[Dual[Seed]],
    draw: DualDraw[Seed] | None,
) -> Dual[Seed]:
    """Combines evaluated child dual values for a non-user-function compound node."""
    match node:
        case Call(function, _) if is_statistical(node):
            arguments = tuple(built)
            return (
                nominal_dual(function, arguments)
                if draw is None
                else draw(function, arguments)
            )
        case Call(function, _):
            return _dual_call(function, built)
        case Unary(operator, _):
            return _dual_unary(operator, built[0])
        case Binary(operator, _, _):
            return _dual_binary(operator, built[0], built[1])
        case Conditional():
            return built[2] if built[0].value == 0.0 else built[1]
    message = f"unexpected expression node: {type(node).__name__}"
    raise SpiceError(message)


def evaluate_dual[Seed](
    expression: Expression,
    parameters: Mapping[str, Scalar[Seed]],
    draw: DualDraw[Seed] | None = None,
    *,
    functions: Mapping[str, Function] = _NO_FUNCTIONS,
) -> Dual[Seed]:
    """Evaluates parameter `expression` with exact first-order dual derivatives.

    Propagates derivatives simultaneously with respect to every seeded `Dual` entry in
    `parameters` and any statistical call seeded by `draw` (defaulting to
    `nominal_dual`). User-defined `.func` calls in `functions` evaluate via iterative
    dual substitution without recursion.

    Args:
        expression: Parameter-only expression tree to evaluate.
        parameters: Case-folded parameter values as plain floats or `Dual` seeds.
        draw: Optional dual statistical callback (defaults to `nominal_dual`).
        functions: Case-folded user `.func` definitions visible in scope.

    Returns:
        `Dual` pairing the evaluated floating-point value with exact partial
        derivatives.

    Raises:
        SpiceError: When `expression` references a voltage, current, unknown
            identifier, unknown function, recursive `.func`, or invalid table.
    """
    quick = _dual_fast(expression, parameters)
    if quick is not None:
        return quick

    frames: list[
        tuple[Expression, Mapping[str, Scalar[Seed]], tuple[str, ...], bool]
    ] = [(expression, parameters, (), False)]
    results: list[Dual[Seed]] = []
    while frames:
        current, scope, calling, ready = frames.pop()
        if not ready:
            if isinstance(current, Number | Name | Voltage | Current):
                results.append(_dual_leaf(current, scope))
                continue
            if isinstance(current, Call):
                _validate_dual_call(current, functions, calling)
            frames.append((current, scope, calling, True))
            frames.extend(
                (child, scope, calling, False) for child in reversed(children(current))
            )
            continue
        arity = len(children(current))
        built = results[-arity:] if arity else []
        if arity:
            del results[-arity:]
        if isinstance(current, Call) and current.function in functions:
            defined = functions[current.function]
            call_scope: Mapping[str, Scalar[Seed]] = {
                **scope,
                **dict(zip(defined.parameters, built, strict=True)),
            }
            frames.append(
                (defined.body, call_scope, (*calling, current.function), False)
            )
        else:
            results.append(_reduce_dual_node(current, built, draw))
    return results[0]

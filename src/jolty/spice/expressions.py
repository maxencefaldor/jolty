# Copyright 2026 Maxence Faldor
"""SPICE's expressions: read, evaluated, and written in jolty's language.

SPICE writes numbers with scale suffixes, such as ``4.7k`` or ``1MEG``, and its
functions with PSpice's meanings, which ngspice shares: ``PWR(x, y)`` is
``|x| ^ y``, ``LIMIT(x, a, b)`` holds ``x`` between ``a`` and ``b``, ``&`` and ``|`` are
logical. Names are case-insensitive.

Examples:
    >>> expression = parse("LIMIT(Gain*V(VC+,VC-), -1m, 1m)")
    >>> evaluate(parse("2 * Gain + 1k"), {"gain": 3.0})
    1006.0
    >>> translate(expression, {"gain": 1e3}, str.lower, str.lower)
    'limit(1000.0 * v(vc+, vc-), -0.001, 0.001)'
"""

import functools
import itertools
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
"""SPICE's scale suffixes, which a unit's letters may follow, as in ``10uF``."""

_NUMBER: Final = re.compile(
    r"(?P<mantissa>[-+]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)"
    r"(?P<suffix>meg|mil|[tgkmunpf])?[a-z]*",
    re.IGNORECASE | re.ASCII,
)


class UndefinedError(SpiceError):
    """An expression reads a name that no parameter or constant has."""

    def __init__(self, message: str, name: str) -> None:
        """The error that `name` is undefined, stated by `message`."""
        super().__init__(message)
        self.name = name


def number(text: str) -> float | None:
    """The number `text` writes, with its scale suffix, or `None` if it is not one.

    Returns `None` for non-finite values, including literals that overflow IEEE-754
    double precision on parsing or scale multiplication.

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
    """A constant."""

    value: float


@dataclass(frozen=True, slots=True)
class Name:
    """A parameter or a constant, by its case-folded name."""

    name: str


@dataclass(frozen=True, slots=True)
class Voltage:
    """``V(a)``, a node's voltage, or ``V(a, b)``, one node's over another's."""

    positive: str
    negative: str | None


@dataclass(frozen=True, slots=True)
class Current:
    """``I(Vx)``, the current through a voltage source."""

    source: str


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Call:
    """A function, by its case-folded name, of its arguments."""

    function: str
    arguments: tuple[Expression, ...]

    @override
    def __eq__(self, other: object) -> bool:
        return _equal(self, other) if isinstance(other, Call) else NotImplemented

    @override
    def __hash__(self) -> int:
        return _hash(self)

    @override
    def __repr__(self) -> str:
        return _repr(self)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Unary:
    """A sign or a negation of an operand."""

    operator: str
    operand: Expression

    @override
    def __eq__(self, other: object) -> bool:
        return _equal(self, other) if isinstance(other, Unary) else NotImplemented

    @override
    def __hash__(self) -> int:
        return _hash(self)

    @override
    def __repr__(self) -> str:
        return _repr(self)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Binary:
    """An operator between two operands."""

    operator: str
    left: Expression
    right: Expression

    @override
    def __eq__(self, other: object) -> bool:
        return _equal(self, other) if isinstance(other, Binary) else NotImplemented

    @override
    def __hash__(self) -> int:
        return _hash(self)

    @override
    def __repr__(self) -> str:
        return _repr(self)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Conditional:
    """``c ? a : b``."""

    condition: Expression
    then: Expression
    otherwise: Expression

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Conditional):
            return _equal(self, other)
        return NotImplemented

    @override
    def __hash__(self) -> int:
        return _hash(self)

    @override
    def __repr__(self) -> str:
        return _repr(self)


type Expression = (
    Number | Name | Voltage | Current | Call | Unary | Binary | Conditional
)


@dataclass(frozen=True, slots=True)
class Function:
    """A user's function, SPICE's ``.func``: its parameters and its body.

    The parameters are case-folded names, which the body, an expression, reads.
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
"""The binary operators at each level of precedence, loosest first, and what each is."""


_BINARY: Final[dict[str, tuple[int, str]]] = {
    symbol: (precedence, operator)
    for precedence, operators in enumerate(_LEVELS)
    for symbol, operator in operators.items()
}
"""Each binary operator symbol mapped to `(precedence, canonical_operator)`."""

_UNARY: Final = {"-": "-", "+": "+", "!": "!", "~": "!"}
"""Unary operator symbols mapped to their canonical operator."""

_MAX_DEPTH: Final = 100
"""The maximum nesting depth accepted by the expression parser."""

_MAX_CORE_DEPTH: Final = 256
"""The maximum parenthesis depth passed to the Rust expression engine.

Matches `jolty_expressions::parser::MAX_DEPTH` (`256`) so SPICE behavioral and
parameter translation raises `SpiceError` directly before crossing into `_core`.
"""

_MAX_INLINED_NODES: Final = 100_000
"""The maximum number of expression nodes built while inlining user functions."""

_MAX_POLYNOMIAL_DEGREE: Final = 64
"""The largest polynomial degree integrated in closed form."""

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
"""The marks that may enclose an expression: braces, or single or double quotes, as
HSPICE writes them, one within another, as in SKY130's ``{"-tol_nfom/2"}``."""


def _is_enclosed(text: str) -> bool:
    """Whether `text` starts and ends with one matched pair of enclosing marks."""
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
    """The maximum parenthesis nesting depth of `text`."""
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
    """The expression `text` writes, with or without braces or quotes around it.

    Raises:
        SpiceError: When `text` is not an expression.
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
        """An error: `expected` is not where the parser is."""
        rest = self.text[self.position :].strip() or "the end"
        return SpiceError(f"expected {expected} in `{self.text}`, not `{rest}`")

    def _too_deep(self) -> SpiceError:
        message = f"expression exceeds maximum nesting depth of {_MAX_DEPTH}"
        return SpiceError(message)

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            raise self._too_deep()

    def _leave(self) -> None:
        self.depth -= 1

    def scan(self) -> tuple[str, str, int] | None:
        """The next token's kind and text, and where it ends, if one is left."""
        match = _TOKEN.match(self.text, self.position)
        kind = None if match is None else match.lastgroup
        if match is None or kind is None:
            return None
        return kind, self.text[match.start(kind) : match.end()], match.end()

    def peek(self) -> tuple[str, str] | None:
        scanned = self.scan()
        if scanned is None:
            return None
        kind, text, _ = scanned
        return kind, text

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
        self._enter()
        try:
            then = self.conditional()
            self.expect(":")
            otherwise = self.conditional()
        finally:
            self._leave()
        return Conditional(condition, then, otherwise)

    def binary(self) -> Expression:
        operands: list[Expression] = [self.unary()]
        operators: list[tuple[int, str]] = []
        while (token := self.peek()) is not None and token[0] == "symbol":
            entry = _BINARY.get(token[1])
            if entry is None:
                break
            precedence, operator = entry
            self.take()
            while operators and operators[-1][0] >= precedence:
                _, pending = operators.pop()
                right = operands.pop()
                left = operands.pop()
                operands.append(Binary(pending, left, right))
            operators.append((precedence, operator))
            operands.append(self.unary())
        while operators:
            _, pending = operators.pop()
            right = operands.pop()
            left = operands.pop()
            operands.append(Binary(pending, left, right))
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
            self._enter()
            try:
                exponent = self.unary()
            finally:
                self._leave()
            return Binary("^", base, exponent)
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
            # A parameter in braces within an expression, as in PSpice: `{VTHRES} + 1`.
            case ("symbol", "(" | "{" as opening):
                self._enter()
                try:
                    inner = self.conditional()
                    self.expect(")" if opening == "(" else "}")
                finally:
                    self._leave()
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
            # Node names may hold signs, as in V(VC+,COM): they are read as written.
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
        self._enter()
        try:
            arguments = [self.conditional()]
            while self.accept(","):
                arguments.append(self.conditional())
            self.expect(")")
            return Call(function, tuple(arguments))
        finally:
            self._leave()


def children(expression: Expression) -> tuple[Expression, ...]:
    """The direct sub-expressions of `expression`, in order."""
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
    """The distinguishing tag of a compound expression node."""
    match expression:
        case Call(function, _):
            return function
        case Unary(operator, _) | Binary(operator, _, _):
            return operator
        case _:
            return ""


def _equal(first: Expression, second: Expression) -> bool:
    """Iterative structural equality that never hits Python's recursion limit."""
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


def _hash(expression: Expression) -> int:
    """Iterative structural hash that never hits Python's recursion limit."""
    frames: list[tuple[Expression, bool]] = [(expression, False)]
    hashes: list[int] = []
    while frames:
        current, ready = frames.pop()
        if not ready:
            if isinstance(current, Number | Name | Voltage | Current):
                hashes.append(hash(current))
                continue
            subexpressions = children(current)
            frames.append((current, True))
            frames.extend((child, False) for child in reversed(subexpressions))
            continue
        arity = len(children(current))
        if arity:
            child_hashes = tuple(hashes[-arity:])
            del hashes[-arity:]
        else:
            child_hashes = ()
        hashes.append(hash((type(current), _label(current), child_hashes)))
    return hashes[0]


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
        return
    if len(arguments) == 1:
        stack.append(",))")
        stack.append(arguments[0])
        return
    stack.append("))")
    for index in range(len(arguments) - 1, -1, -1):
        stack.append(arguments[index])
        if index > 0:
            stack.append(", ")


def _repr(expression: Expression) -> str:
    """Iterative structural representation that never hits Python's recursion limit."""
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
                stack.append(")")
                stack.append(operand)
            case Binary(operator, left, right):
                parts.append(f"Binary(operator={operator!r}, left=")
                stack.append(")")
                stack.append(right)
                stack.append(", right=")
                stack.append(left)
            case Conditional(condition, then, otherwise):
                parts.append("Conditional(condition=")
                stack.append(")")
                stack.append(otherwise)
                stack.append(", otherwise=")
                stack.append(then)
                stack.append(", then=")
                stack.append(condition)
            case Call(function, arguments):
                _push_call_repr(parts, stack, function, arguments)
    return "".join(parts)


def rebuild(expression: Expression, subexpressions: list[Expression]) -> Expression:
    """Rebuilds `expression` with `subexpressions` in place of its children."""
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
    """Checks cycle and arity for `function` and returns its substituted body."""
    defined = functions[function]
    given = dict(zip(defined.parameters, arguments, strict=True))
    return _substituted(defined.body, given), (*calling, function)


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
    """`expression` with each call of one of the user's `functions` written out.

    A call becomes its function's body, the function's parameters replaced by the
    call's arguments, within the functions `calling`. A function may call another, but
    none itself, directly or through others.

    Examples:
        >>> twice = Function(("x",), parse("2 * x"))
        >>> inlined(parse("twice(twice(y)) + 1"), {"twice": twice})
        Binary(operator='+', left=Binary(operator='*', left=Number(value=2.0), \
right=Binary(operator='*', left=Number(value=2.0), right=Name(name='y'))), \
right=Number(value=1.0))

    Raises:
        SpiceError: When a function calls itself, naming the functions through which,
            a call gives a function another number of arguments than it takes, or
            inlining exceeds `_MAX_INLINED_NODES` nodes.
    """
    if not functions:
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


def _substituted(expression: Expression, given: Mapping[str, Expression]) -> Expression:
    """`expression` with each name `given` replaced by the expression it is given."""
    if not given:
        return expression
    frames: list[tuple[Expression, bool]] = [(expression, False)]
    results: list[Expression] = []
    while frames:
        current, ready = frames.pop()
        if not ready:
            if isinstance(current, Name) and current.name in given:
                results.append(given[current.name])
                continue
            subexpressions = children(current)
            if not subexpressions:
                results.append(current)
                continue
            frames.append((current, True))
            frames.extend((child, False) for child in reversed(subexpressions))
            continue
        arity = len(children(current))
        built = results[-arity:]
        del results[-arity:]
        results.append(rebuild(current, built))
    return results[0]


_CONSTANTS: Final = {
    "pi": math.pi,
    "e": math.e,
    "temp": 27.0,
    "vt": (27.0 + 273.15) * 8.6173303e-5,
    "gmin": 1e-12,
}
"""The names every expression knows, as ngspice defines PSpice's: ``TEMP``, the
temperature in degrees Celsius, ``VT``, the thermal voltage there by ngspice's constant,
and ``GMIN``, the minimum conductance."""

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
"""Each function: its number of arguments, `None` for any, and how jolty writes it.

jolty writes a function by its name, or by a template of its arguments. PSpice's
``PWR`` is of the magnitude of the base, and ``POW`` as `_power` writes it. A
``TABLE`` within an expression is ngspice's ``pwl``, extended beyond its ends; a
``TABLE`` source's, made by [`library`][jolty.spice.library], is a ``rounded_table``.
"""


def substituted(
    expression: Expression, variable: Expression, replacement: Expression
) -> Expression:
    """`expression` with each occurrence of `variable` replaced by `replacement`."""
    frames: list[tuple[Expression, bool]] = [(expression, False)]
    results: list[Expression] = []
    while frames:
        current, ready = frames.pop()
        if not ready:
            if current == variable:
                results.append(replacement)
                continue
            subexpressions = children(current)
            if not subexpressions:
                results.append(current)
                continue
            frames.append((current, True))
            frames.extend((child, False) for child in reversed(subexpressions))
            continue
        arity = len(children(current))
        built = results[-arity:]
        del results[-arity:]
        results.append(rebuild(current, built))
    return results[0]


type Polynomial = dict[int, float]
"""A polynomial of one variable: each power's coefficient."""


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


def _push_polynomial_children(
    current: Expression,
    frames: list[tuple[Expression, bool]],
    computed: dict[int, Polynomial | None],
) -> None:
    """Schedules polynomial child subtrees or marks non-polynomial nodes `None`."""
    node_id = id(current)
    match current:
        case Unary("-" | "+", operand):
            frames.append((current, True))
            frames.append((operand, False))
        case Binary("+" | "-" | "*", left, right):
            frames.append((current, True))
            frames.append((right, False))
            frames.append((left, False))
        case Binary("/", left, Number(divisor)) if divisor != 0:
            frames.append((current, True))
            frames.append((left, False))
        case Binary("^", base, Number()) | Call("pow", (base, Number())):
            frames.append((current, True))
            frames.append((base, False))
        case Call(function, (argument,)) if function in _ANTIDERIVATIVES:
            computed[node_id] = None
            frames.append((argument, False))
        case _:
            computed[node_id] = None


def _polynomial_map(
    expression: Expression, variable: Expression
) -> dict[int, Polynomial | None]:
    """Computes bottom-up polynomials of `variable` for every reachable subtree.

    Maps each node's `id` to its polynomial dictionary (up to
    `_MAX_POLYNOMIAL_DEGREE`), or `None` when the subtree is not a polynomial.
    Runs iteratively in a single `O(N)` pass with no recursion or repeated rescans.
    """
    computed: dict[int, Polynomial | None] = {}
    frames: list[tuple[Expression, bool]] = [(expression, False)]
    while frames:
        current, ready = frames.pop()
        node_id = id(current)
        if node_id in computed:
            continue
        if not ready:
            if isinstance(current, Number):
                computed[node_id] = {0: current.value}
            elif current == variable:
                computed[node_id] = {1: 1.0}
            else:
                _push_polynomial_children(current, frames, computed)
            continue
        computed[node_id] = _combine_polynomial(current, computed)
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
    """`base` raised to integer `power`, up to `_MAX_POLYNOMIAL_DEGREE`."""
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
    """`polynomial` times `factor`."""
    return {power: coefficient * factor for power, coefficient in polynomial.items()}


def _added(first: Polynomial, second: Polynomial) -> Polynomial:
    """The sum of two polynomials."""
    total = dict(first)
    for power, coefficient in second.items():
        total[power] = total.get(power, 0.0) + coefficient
    return total


def _multiplied(first: Polynomial, second: Polynomial) -> Polynomial | None:
    """The product of two polynomials, or `None` past `_MAX_POLYNOMIAL_DEGREE`."""
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
"""The antiderivative of each function whose antiderivative is elementary and holds
for every argument, of its argument."""


def _rebuild_integral(expression: Expression, results: list[Expression]) -> Expression:
    """Builds the integrated parent node from already-integrated child `results`."""
    match expression:
        case Unary(operator, _):
            return Unary(operator, results.pop())
        case Binary("+" | "-" as operator, _, _):
            right = results.pop()
            left = results.pop()
            return Binary(operator, left, right)
        case Binary("*", Number() as factor, _) | Binary("*", _, Number() as factor):
            return Binary("*", factor, results.pop())
        case Binary("/", _, Number() as divisor):
            return Binary("/", results.pop(), divisor)
    message = f"unexpected expression node: {type(expression).__name__}"
    raise SpiceError(message)


def _step_integral(
    current: Expression,
    polynomials: Mapping[int, Polynomial | None],
    frames: list[tuple[Expression, bool]],
    results: list[Expression],
) -> bool:
    """Expands one non-polynomial node during iterative integration.

    Returns True when `current` admits closed-form integration, False otherwise.
    """
    match current:
        case Unary("-" | "+", operand):
            frames.append((current, True))
            frames.append((operand, False))
        case Binary("+" | "-", left, right):
            frames.append((current, True))
            frames.append((right, False))
            frames.append((left, False))
        case Binary("*", Number(), other) | Binary("*", other, Number()):
            frames.append((current, True))
            frames.append((other, False))
        case Binary("/", other, Number(divisor)) if divisor != 0:
            frames.append((current, True))
            frames.append((other, False))
        case Call(function, (argument,)) if function in _ANTIDERIVATIVES:
            line = polynomials.get(id(argument))
            if line is None or not set(line) <= {0, 1} or line.get(1, 0.0) == 0:
                return False
            antiderivative = _ANTIDERIVATIVES[function](argument)
            results.append(Binary("/", antiderivative, Number(line[1])))
        case _:
            return False
    return True


def integral(
    expression: Expression, variable: Expression, depth: int = 0
) -> Expression | None:
    """An antiderivative of `expression` with respect to `variable`, in closed form.

    `expression` holds numbers and `variable` alone, as one whose other terms are
    evaluated does. A closed form is found where `expression` is a sum of a polynomial
    of the variable and of numbers times ``exp``, ``sinh``, ``cosh``, ``tanh``, ``sin``
    or ``cos`` of a linear function of it: each power integrated term by term, and each
    function by its antiderivative over the line's slope; `None` where no closed form
    exists.
    """
    if depth > _MAX_DEPTH:
        return None
    polynomials = _polynomial_map(expression, variable)
    frames: list[tuple[Expression, bool]] = [(expression, False)]
    results: list[Expression] = []
    while frames:
        current, ready = frames.pop()
        if ready:
            results.append(_rebuild_integral(current, results))
            continue
        polynomial = polynomials.get(id(current))
        if polynomial is not None:
            results.append(_polynomial_integral(polynomial, variable))
            continue
        if not _step_integral(current, polynomials, frames, results):
            return None
    return results[0]


def names(expression: Expression) -> set[str]:
    """The names `expression` reads, of parameters or constants."""
    found: set[str] = set()
    stack: list[Expression] = [expression]
    while stack:
        current = stack.pop()
        if isinstance(current, Name):
            found.add(current.name)
        else:
            stack.extend(children(current))
    return found


STATISTICAL: Final = {"agauss": 3, "gauss": 3, "aunif": 2, "unif": 2, "limit": 2}
"""ngspice's statistical functions, by name, and their number of arguments, the first
the nominal value each draws about (numparam's ``xpressn.c``). ``limit`` of any other
number of arguments is PSpice's, which holds its first between its next two."""

type Draw = Callable[[str, tuple[float, ...]], float]
"""The value a statistical function, by name, draws with its arguments' values."""


def nominal(_function: str, arguments: tuple[float, ...]) -> float:
    """The nominal value of a statistical function: its first argument, undrawn."""
    return arguments[0]


def evaluate(
    expression: Expression, parameters: Mapping[str, float], draw: Draw = nominal
) -> float:
    """The value of `expression`, which reads no voltage or current, with `parameters`.

    Each statistical function it calls takes the value `draw` draws, its nominal one
    unless given, in the order the expression writes them, inner calls first.

    Raises:
        SpiceError: When the expression reads a voltage, a current or an unknown name,
            or calls an unknown function.
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


def _is_statistical(expression: Expression) -> bool:
    """Whether `expression` is a call to one of SPICE's statistical functions."""
    if not isinstance(expression, Call):
        return False
    function = expression.function
    return function in STATISTICAL and not (
        function == "limit" and len(expression.arguments) != STATISTICAL[function]
    )


def _drawn(
    expression: Expression, parameters: Mapping[str, float], draw: Draw
) -> Expression:
    """`expression` with each statistical function's call replaced by what it draws.

    Raises:
        SpiceError: When a statistical function takes the wrong number of arguments.
    """
    frames: list[tuple[Expression, bool]] = [(expression, False)]
    results: list[Expression] = []
    while frames:
        current, ready = frames.pop()
        if not ready:
            if _is_statistical(current) and isinstance(current, Call):
                expected = STATISTICAL[current.function]
                if len(current.arguments) != expected:
                    message = (
                        f"`{current.function}` takes {expected} arguments, not "
                        f"{len(current.arguments)}"
                    )
                    raise SpiceError(message)
            subexpressions = children(current)
            if not subexpressions:
                results.append(current)
                continue
            frames.append((current, True))
            frames.extend((child, False) for child in reversed(subexpressions))
            continue
        arity = len(children(current))
        built = results[-arity:]
        del results[-arity:]
        if _is_statistical(current) and isinstance(current, Call):
            values = tuple(evaluate(argument, parameters, draw) for argument in built)
            results.append(Number(draw(current.function, values)))
        else:
            results.append(rebuild(current, built))
    return results[0]


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
) -> str:
    """`expression` in jolty's language.

    Each parameter is its value, each node but the ground a terminal named by `node`,
    and each source whose current it reads a port named by `current`. Emits
    parentheses only where operator precedence or associativity requires them.

    Raises:
        SpiceError: When the expression names an unknown parameter or function, or
            the translated text exceeds `_MAX_TRANSLATED_LENGTH`.
    """
    frames: list[tuple[Expression, bool]] = [(expression, False)]
    results: list[tuple[str, int]] = []
    while frames:
        part, ready = frames.pop()
        if not ready:
            if isinstance(part, Number | Name | Voltage | Current):
                leaf_text = _leaf(part, parameters, node, current)
                _check_translated_length(len(leaf_text))
                results.append((leaf_text, _ATOM_PRECEDENCE))
                continue
            subexpressions = children(part)
            if not subexpressions:
                if isinstance(part, Call):
                    results.append((_call(part.function, []), _ATOM_PRECEDENCE))
                    continue
                message = f"unexpected expression node: {type(part).__name__}"
                raise SpiceError(message)
            frames.append((part, True))
            frames.extend((child, False) for child in reversed(subexpressions))
            continue
        arity = len(children(part))
        built = results[-arity:]
        del results[-arity:]
        formatted = _format_node(part, built)
        _check_translated_length(len(formatted[0]))
        results.append(formatted)
    return results[0][0]


def _format_unary(operator: str, operand: tuple[str, int]) -> tuple[str, int]:
    """Formats a unary operation with minimal parentheses and space separation."""
    operand_text, operand_prec = operand
    if operand_prec < _UNARY_PRECEDENCE:
        _check_translated_length(len(operator) + len(operand_text) + 2)
        return f"{operator}({operand_text})", _UNARY_PRECEDENCE
    if operand_prec == _UNARY_PRECEDENCE:
        _check_translated_length(len(operator) + len(operand_text) + 1)
        return f"{operator} {operand_text}", _UNARY_PRECEDENCE
    _check_translated_length(len(operator) + len(operand_text))
    return f"{operator}{operand_text}", _UNARY_PRECEDENCE


def _format_binary(
    operator: str, left: tuple[str, int], right: tuple[str, int]
) -> tuple[str, int]:
    """Formats a left-associative binary operation with minimal parentheses."""
    precedence = _BIN_PRECEDENCE[operator]
    left_text, left_prec = left
    right_text, right_prec = right
    wrap_left = left_prec < precedence
    wrap_right = right_prec <= precedence or (
        right_prec == _UNARY_PRECEDENCE and right_text.startswith(("+", "-"))
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
    """`base` to the power of `exponent`, in jolty's language, as ngspice takes it.

    A negative base to a whole power keeps its sign where the power is odd, as ngspice's
    parameters and behavioral sources raise it (``xpressn.c``, lines 136 to 137 and 786
    to 803; ``ptfuncs.c``, lines 69 to 87), where the exponent is a constant; any other
    power is of the magnitude of the base, where ngspice's behavioral sources give 0 and
    its parameters no number.
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
) -> str:
    """A number, a name, a voltage or a current, in jolty's language.

    Raises:
        SpiceError: When no parameter or constant is named as the leaf is.
    """
    match leaf:
        case Number(value):
            return _number(value)
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
    """``V(a, b)`` in jolty's language, the ground, ``0``, left out."""
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
    """`value` as jolty's language writes it: a negative one in parentheses."""
    written = repr(value)
    return f"({written})" if written.startswith("-") else written

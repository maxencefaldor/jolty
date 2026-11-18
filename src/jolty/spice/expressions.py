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
    'limit((1000.0 * v(vc+, vc-)), (-0.001), 0.001)'
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
    re.IGNORECASE,
)


class UndefinedError(SpiceError):
    """An expression reads a name that no parameter or constant has."""

    def __init__(self, message: str, name: str) -> None:
        """The error that `name` is undefined, stated by `message`."""
        super().__init__(message)
        self.name = name


def number(text: str) -> float | None:
    """The number `text` writes, with its scale suffix, or `None` if it is not one.

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
    return float(match["mantissa"]) * scale


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


@dataclass(frozen=True, slots=True, eq=False)
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


@dataclass(frozen=True, slots=True, eq=False)
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


@dataclass(frozen=True, slots=True, eq=False)
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


@dataclass(frozen=True, slots=True, eq=False)
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
    re.IGNORECASE,
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

The compiled recursive-descent parser overflows the stack around 750 nested
parentheses on an 8 MiB Linux thread stack, so Python refuses deeper expressions
before crossing into the extension.
"""

_MAX_INLINED_NODES: Final = 100_000
"""The maximum number of expression nodes built while inlining user functions."""

_MAX_POLYNOMIAL_DEGREE: Final = 64
"""The largest polynomial degree integrated in closed form."""


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
                    expected = "a number"
                    raise self.fail(expected)
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


def _polynomial(  # noqa: C901, PLR0911 - one case per form a polynomial takes
    expression: Expression, variable: Expression, depth: int = 0
) -> Polynomial | None:
    """`expression` as a polynomial of `variable`, if it is one of its numbers.

    Sums, differences, products, quotients by a number and whole powers of polynomials
    are polynomials up to `_MAX_POLYNOMIAL_DEGREE`; a whole power is SPICE's ``^`` or
    ``POW``, which keep a negative base's sign at an odd power, as `_power` writes
    them.
    """
    if depth > _MAX_DEPTH:
        return None

    def of(part: Expression) -> Polynomial | None:
        return _polynomial(part, variable, depth + 1)

    match expression:
        case Number(value):
            return {0: value}
        case _ if expression == variable:
            return {1: 1.0}
        case Unary("-", operand):
            found = of(operand)
            return None if found is None else _scaled(found, -1.0)
        case Unary("+", operand):
            return of(operand)
        case Binary("+" | "-" as operator, left, right):
            first, second = of(left), of(right)
            if first is None or second is None:
                return None
            sign = 1.0 if operator == "+" else -1.0
            return _added(first, _scaled(second, sign))
        case Binary("*", left, right):
            first, second = of(left), of(right)
            if first is None or second is None:
                return None
            return _multiplied(first, second)
        case Binary("/", left, Number(divisor)) if divisor != 0:
            found = of(left)
            return None if found is None else _scaled(found, 1 / divisor)
        case Binary("^", base, Number(power)) | Call("pow", (base, Number(power))):
            return _power_polynomial(of(base), power)
    return None


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


def integral(  # noqa: C901, PLR0911 - one case per form the integral takes
    expression: Expression, variable: Expression, depth: int = 0
) -> Expression | None:
    """An antiderivative of `expression` with respect to `variable`, in closed form.

    `expression` holds numbers and `variable` alone, as one whose other terms are
    evaluated does. A closed form is found where it is a sum of a polynomial of the
    variable and of numbers times ``exp``, ``sinh``, ``cosh``, ``tanh``, ``sin`` or
    ``cos`` of a linear function of it: each power integrated term by term, and each
    function by its antiderivative over the line's slope; `None` where it is not.
    """
    if depth > _MAX_DEPTH:
        return None
    polynomial = _polynomial(expression, variable)
    if polynomial is not None:
        terms: list[Expression] = []
        for power, coefficient in sorted(polynomial.items()):
            term: Expression = Number(coefficient / (power + 1))
            for _ in range(power + 1):
                term = Binary("*", term, variable)
            terms.append(term)
        return functools.reduce(lambda left, right: Binary("+", left, right), terms)

    def of(part: Expression) -> Expression | None:
        return integral(part, variable, depth + 1)

    match expression:
        case Unary("-" | "+" as operator, operand):
            found = of(operand)
            return None if found is None else Unary(operator, found)
        case Binary("+" | "-" as operator, left, right):
            first, second = of(left), of(right)
            if first is None or second is None:
                return None
            return Binary(operator, first, second)
        case Binary("*", Number() as factor, other) | Binary(
            "*", other, Number() as factor
        ):
            found = of(other)
            return None if found is None else Binary("*", factor, found)
        case Binary("/", other, Number(divisor) as number) if divisor != 0:
            found = of(other)
            return None if found is None else Binary("/", found, number)
        case Call(function, (argument,)) if function in _ANTIDERIVATIVES:
            line = _polynomial(argument, variable)
            if line is None or not set(line) <= {0, 1} or line.get(1, 0.0) == 0:
                return None
            antiderivative = _ANTIDERIVATIVES[function](argument)
            return Binary("/", antiderivative, Number(line[1]))
    return None


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


def translate(
    expression: Expression,
    parameters: Mapping[str, float],
    node: Callable[[str], str],
    current: Callable[[str], str],
) -> str:
    """`expression` in jolty's language.

    Each parameter is its value, each node but the ground a terminal named by `node`,
    and each source whose current it reads a port named by `current`.

    Raises:
        SpiceError: When the expression names an unknown parameter or function.
    """
    frames: list[tuple[Expression, bool]] = [(expression, False)]
    results: list[str] = []
    while frames:
        part, ready = frames.pop()
        if not ready:
            if isinstance(part, Number | Name | Voltage | Current):
                results.append(_leaf(part, parameters, node, current))
                continue
            subexpressions = children(part)
            if not subexpressions:
                if isinstance(part, Call):
                    results.append(_call(part.function, []))
                    continue
                message = f"unexpected expression node: {type(part).__name__}"
                raise SpiceError(message)
            frames.append((part, True))
            frames.extend((child, False) for child in reversed(subexpressions))
            continue
        arity = len(children(part))
        built = results[-arity:]
        del results[-arity:]
        results.append(_format_node(part, built))
    return results[0]


def _format_node(expression: Expression, built: list[str]) -> str:
    """Formats a non-leaf `expression` from its already-translated `built` children."""
    match expression:
        case Unary(operator, _):
            return f"({operator}{built[0]})"
        case Binary("^", _, _) | Call("pow", (_, _)):
            return _power(built[0], built[1])
        case Binary(operator, _, _):
            return f"({built[0]} {operator} {built[1]})"
        case Conditional():
            return f"if({built[0]}, {built[1]}, {built[2]})"
        case Call(function, _):
            return _call(function, built)
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
    magnitude = f"pow(abs({base}), {exponent})"
    if power.is_integer() and power % 2:
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
        return written.format(*arguments)
    return f"{written}({', '.join(arguments)})"


def _number(value: float) -> str:
    """`value` as jolty's language writes it: a negative one in parentheses."""
    written = repr(value)
    return f"({written})" if value < 0 else written

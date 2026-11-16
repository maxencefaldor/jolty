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

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

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


@dataclass(frozen=True, slots=True)
class Call:
    """A function, by its case-folded name, of its arguments."""

    function: str
    arguments: tuple[Expression, ...]


@dataclass(frozen=True, slots=True)
class Unary:
    """A sign or a negation of an operand."""

    operator: str
    operand: Expression


@dataclass(frozen=True, slots=True)
class Binary:
    """An operator between two operands."""

    operator: str
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class Conditional:
    """``c ? a : b``."""

    condition: Expression
    then: Expression
    otherwise: Expression


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


_ENCLOSING: Final = {"{": "}", "'": "'", '"': '"'}
"""The marks that may enclose an expression: braces, or single or double quotes, as
HSPICE writes them, one within another, as in SKY130's ``{"-tol_nfom/2"}``."""


def parse(text: str) -> Expression:
    """The expression `text` writes, with or without braces or quotes around it.

    Raises:
        SpiceError: When `text` is not an expression.
    """
    stripped = text.strip()
    while len(stripped) > 1 and _ENCLOSING.get(stripped[0]) == stripped[-1]:
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

    def fail(self, expected: str) -> SpiceError:
        """An error: `expected` is not where the parser is."""
        rest = self.text[self.position :].strip() or "the end"
        return SpiceError(f"expected {expected} in `{self.text}`, not `{rest}`")

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
        condition = self.level(0)
        if not self.accept("?"):
            return condition
        then = self.conditional()
        self.expect(":")
        return Conditional(condition, then, self.conditional())

    def level(self, level: int) -> Expression:
        if level == len(_LEVELS):
            return self.unary()
        operators = _LEVELS[level]
        left = self.level(level + 1)
        while (token := self.peek()) is not None and token[0] == "symbol":
            operator = operators.get(token[1])
            if operator is None:
                break
            self.take()
            left = Binary(operator, left, self.level(level + 1))
        return left

    def unary(self) -> Expression:
        for symbol, operator in (("-", "-"), ("+", "+"), ("!", "!"), ("~", "!")):
            if self.accept(symbol):
                return Unary(operator, self.unary())
        return self.power()

    def power(self) -> Expression:
        base = self.primary()
        if self.accept("**") or self.accept("^"):
            return Binary("^", base, self.unary())
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
            # Node names may hold signs, as in V(VC+,COM): they are read as written.
            end = self.text.find(")", self.position)
            if end < 0:
                expected = "`)`"
                raise self.fail(expected)
            written = self.text[self.position : end]
            names = [part.strip() for part in written.split(",")]
            self.position = end + 1
            match function, names:
                case "v", [positive]:
                    return Voltage(positive, None)
                case "v", [positive, negative]:
                    return Voltage(positive, negative)
                case "i", [source]:
                    return Current(source)
            expected = f"one or two nodes in `{function}(…)`"
            raise self.fail(expected)
        arguments = [self.conditional()]
        while self.accept(","):
            arguments.append(self.conditional())
        self.expect(")")
        return Call(function, tuple(arguments))


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
            or a call gives a function another number of arguments than it takes.
    """

    def inner(part: Expression) -> Expression:
        return inlined(part, functions, calling)

    match expression:
        case Call(function, arguments) if function in functions:
            if function in calling:
                cycle = " calls ".join(
                    f"`{name}`"
                    for name in (*calling[calling.index(function) :], function)
                )
                message = f"a function calls itself: {cycle}"
                raise SpiceError(message)
            defined = functions[function]
            if len(arguments) != len(defined.parameters):
                message = (
                    f"`{function}` takes {len(defined.parameters)} arguments, not "
                    f"{len(arguments)}"
                )
                raise SpiceError(message)
            given = dict(zip(defined.parameters, map(inner, arguments), strict=True))
            body = _substituted(defined.body, given)
            return inlined(body, functions, (*calling, function))
        case Call(function, arguments):
            return Call(function, tuple(map(inner, arguments)))
        case Unary(operator, operand):
            return Unary(operator, inner(operand))
        case Binary(operator, left, right):
            return Binary(operator, inner(left), inner(right))
        case Conditional(condition, then, otherwise):
            return Conditional(inner(condition), inner(then), inner(otherwise))
    return expression


def _substituted(expression: Expression, given: Mapping[str, Expression]) -> Expression:
    """`expression` with each name `given` replaced by the expression it is given."""

    def inner(part: Expression) -> Expression:
        return _substituted(part, given)

    match expression:
        case Name(name) if name in given:
            return given[name]
        case Call(function, arguments):
            return Call(function, tuple(map(inner, arguments)))
        case Unary(operator, operand):
            return Unary(operator, inner(operand))
        case Binary(operator, left, right):
            return Binary(operator, inner(left), inner(right))
        case Conditional(condition, then, otherwise):
            return Conditional(inner(condition), inner(then), inner(otherwise))
    return expression


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
``PWR`` and ngspice's power are of the magnitude of the base. A ``TABLE`` within an
expression is ngspice's ``pwl``, extended beyond its ends; a ``TABLE`` source's, made by
[`library`][jolty.spice.library], is a ``rounded_table``.
"""


def names(expression: Expression) -> set[str]:
    """The names `expression` reads, of parameters or constants."""
    match expression:
        case Name(name):
            return {name}
        case Unary(_, operand):
            return names(operand)
        case Binary(_, left, right):
            return names(left) | names(right)
        case Conditional(condition, then, otherwise):
            return names(condition) | names(then) | names(otherwise)
        case Call(_, arguments):
            found: set[str] = set()
            for argument in arguments:
                found |= names(argument)
            return found
    return set()


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

    def unread(name: str) -> str:
        message = f"a parameter's expression reads no voltage or current, not `{name}`"
        raise SpiceError(message)

    drawn = _drawn(expression, parameters, draw)
    written = translate(drawn, parameters, unread, unread)
    try:
        return _core.constant(written)
    except ValueError as error:
        raise SpiceError(str(error)) from None


def _drawn(
    expression: Expression, parameters: Mapping[str, float], draw: Draw
) -> Expression:
    """`expression` with each statistical function's call replaced by what it draws.

    Raises:
        SpiceError: When a statistical function takes the wrong number of arguments.
    """

    def drawn(part: Expression) -> Expression:
        return _drawn(part, parameters, draw)

    match expression:
        case Call(function, arguments) if function in STATISTICAL and not (
            function == "limit" and len(arguments) != STATISTICAL[function]
        ):
            if len(arguments) != STATISTICAL[function]:
                message = (
                    f"`{function}` takes {STATISTICAL[function]} arguments, not "
                    f"{len(arguments)}"
                )
                raise SpiceError(message)
            values = tuple(
                evaluate(drawn(argument), parameters, draw) for argument in arguments
            )
            return Number(draw(function, values))
        case Call(function, arguments):
            return Call(function, tuple(drawn(argument) for argument in arguments))
        case Unary(operator, operand):
            return Unary(operator, drawn(operand))
        case Binary(operator, left, right):
            return Binary(operator, drawn(left), drawn(right))
        case Conditional(condition, then, otherwise):
            return Conditional(drawn(condition), drawn(then), drawn(otherwise))
    return expression


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

    def written(part: Expression) -> str:
        return translate(part, parameters, node, current)

    match expression:
        case Unary(operator, operand):
            return f"({operator}{written(operand)})"
        case Binary("^", left, right):
            return f"pow(abs({written(left)}), {written(right)})"
        case Binary(operator, left, right):
            return f"({written(left)} {operator} {written(right)})"
        case Conditional(condition, then, otherwise):
            return f"if({written(condition)}, {written(then)}, {written(otherwise)})"
        case Call(function, arguments):
            return _call(function, [written(part) for part in arguments])
        case leaf:
            return _leaf(leaf, parameters, node, current)


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

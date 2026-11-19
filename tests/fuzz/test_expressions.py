# Copyright 2026 Maxence Faldor
"""Hypothesis fuzz tests for the SPICE and Rust `_core` expression parsers."""

import contextlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from fuzz.helpers import FUZZ, one_of_strings
from jolty import _core
from jolty.errors import SpiceError
from jolty.spice import expressions
from jolty.spice.expressions import nesting_depth

_SPICE_SUFFIXES: Final = (
    "",
    "t",
    "g",
    "meg",
    "k",
    "mil",
    "m",
    "u",
    "n",
    "p",
    "f",
    "kohm",
    "uF",
    "MEG",
)
_CORE_BINARY_OPERATORS: Final = (
    "+",
    "-",
    "*",
    "/",
    "^",
    "<",
    "<=",
    ">",
    ">=",
    "==",
    "!=",
    "&&",
    "||",
)
_SPICE_BINARY_OPERATORS: Final = (*_CORE_BINARY_OPERATORS, "**", "<>", "&", "|")
_SPICE_UNARY_OPERATORS: Final = ("-", "+", "!", "~")
_CORE_UNARY_FUNCTIONS: Final = (
    "abs",
    "sqrt",
    "exp",
    "ln",
    "log",
    "log10",
    "floor",
    "ceil",
    "sgn",
    "u",
    "uramp",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
)
_SPICE_UNARY_FUNCTIONS: Final = (*_CORE_UNARY_FUNCTIONS, "sign", "stp", "int")
_SPICE_BINARY_FUNCTIONS: Final = ("min", "max", "pow", "pwr", "pwrs")
_SPICE_PARAMETERS: Final[dict[str, float]] = {
    "alpha": 2.5,
    "beta": -1.25,
    "gain": 10.0,
    "offset": 0.5,
}
_USER_FUNCTIONS: Final[dict[str, expressions.Function]] = {
    "user_double": expressions.Function(("x",), expressions.parse("x + x")),
    "user_loop": expressions.Function(("x",), expressions.parse("user_loop(x)")),
    "user_zero": expressions.Function((), expressions.parse("42")),
}
_EXPRESSION_SOUP_TOKENS: Final = (
    "0",
    "1",
    "-1",
    "1e500",
    "1e-500",
    "1e308",
    "1k",
    "2.5MEG",
    "3mil",
    "4uF",
    "1\u212a",
    "1\u0660",
    "alpha",
    "unknown_name",
    "pi",
    "time",
    "inf",
    "nan",
    "v(a)",
    "v(a,b)",
    "v()",
    "v(,)",
    "i(v1)",
    "i()",
    "i(a,b)",
    "agauss(1,2,3)",
    "agauss(1)",
    "limit(1,2,3)",
    "limit(1,2)",
    "limit(1)",
    "pow(1,2)",
    "pow(-2, 1e30)",
    "table(1,0,0,1,1)",
    "table(1,0,0,0,1)",
    "pwl(1)",
    "rounded_table(1,0.25,0,0,1,1)",
    "user_double(1)",
    "user_loop(1)",
    "user_zero()",
    "{1}",
    "'1'",
    '"1"',
    "+",
    "-",
    "*",
    "/",
    "^",
    "**",
    "?",
    ":",
    "(",
    ")",
    "{",
    "}",
    "'",
    '"',
    ",",
    "!",
    "~",
    "&",
    "&&",
    "|",
    "||",
    "==",
    "!=",
    "<>",
    "<",
    "<=",
    ">",
    ">=",
    " ",
    "\t",
    "\n",
    "$",
    ";",
)


@st.composite
def _spice_number_literals(draw: st.DrawFn) -> str:
    """Generates syntactically valid SPICE numeric literals with scale suffixes."""
    mantissa = draw(
        st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False)
    )
    suffix = draw(st.sampled_from(_SPICE_SUFFIXES))
    return f"{mantissa:.4g}{suffix}"


@dataclass(frozen=True, slots=True)
class _ExpressionDialect:
    """Operator and function vocabulary shared by an expression grammar."""

    unary_operators: tuple[str, ...]
    binary_operators: tuple[str, ...]
    unary_functions: tuple[str, ...]
    binary_functions: tuple[str, ...]


_SPICE_DIALECT: Final = _ExpressionDialect(
    unary_operators=_SPICE_UNARY_OPERATORS,
    binary_operators=_SPICE_BINARY_OPERATORS,
    unary_functions=_SPICE_UNARY_FUNCTIONS,
    binary_functions=_SPICE_BINARY_FUNCTIONS,
)
_CORE_DIALECT: Final = _ExpressionDialect(
    unary_operators=("-", "+", "!"),
    binary_operators=_CORE_BINARY_OPERATORS,
    unary_functions=_CORE_UNARY_FUNCTIONS,
    binary_functions=("min", "max", "pow"),
)


def _expression_grammar(
    leaves: st.SearchStrategy[str],
    dialect: _ExpressionDialect,
    extra_branches: Callable[[st.SearchStrategy[str]], list[st.SearchStrategy[str]]],
) -> st.SearchStrategy[str]:
    """Builds a recursive expression strategy over shared operator and call forms."""

    def extend(children: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
        return one_of_strings(
            [
                st.tuples(st.sampled_from(dialect.unary_operators), children).map(
                    lambda pair: f"({pair[0]}{pair[1]})"
                ),
                st.tuples(
                    children, st.sampled_from(dialect.binary_operators), children
                ).map(lambda triple: f"({triple[0]} {triple[1]} {triple[2]})"),
                st.tuples(children, children, children).map(
                    lambda triple: f"({triple[0]} ? {triple[1]} : {triple[2]})"
                ),
                st.tuples(st.sampled_from(dialect.unary_functions), children).map(
                    lambda pair: f"{pair[0]}({pair[1]})"
                ),
                st.tuples(
                    st.sampled_from(dialect.binary_functions), children, children
                ).map(lambda triple: f"{triple[0]}({triple[1]}, {triple[2]})"),
                st.tuples(children, children, children).map(
                    lambda triple: f"limit({triple[0]}, {triple[1]}, {triple[2]})"
                ),
                *extra_branches(children),
            ]
        )

    return st.recursive(leaves, extend, max_leaves=8)


def _valid_spice_expressions(*, with_variables: bool) -> st.SearchStrategy[str]:
    """Generates grammar-guided valid SPICE expression strings."""
    base_leaves: list[st.SearchStrategy[str]] = [
        _spice_number_literals(),
        st.sampled_from(["pi", "e", "temp", "vt", "gmin", *_SPICE_PARAMETERS]),
    ]
    if with_variables:
        base_leaves.append(
            st.sampled_from(
                [
                    "V(in)",
                    "V(out, ref)",
                    "V(0, ref)",
                    "V(node_a, 0)",
                    "V(vc+, com)",
                    "I(V1)",
                ]
            )
        )

    def extra_branches(
        children: st.SearchStrategy[str],
    ) -> list[st.SearchStrategy[str]]:
        branches: list[st.SearchStrategy[str]] = [
            st.tuples(children, children, children).map(
                lambda triple: f"if({triple[0]}, {triple[1]}, {triple[2]})"
            ),
            children.map(lambda inner: f"{{{inner}}}"),
            children.map(lambda inner: f"table({inner}, -1, -1, 0, 0, 1, 1)"),
        ]
        if not with_variables:
            branches.append(
                st.sampled_from(
                    [
                        "agauss(1.0, 0.1, 3.0)",
                        "gauss(2.0, 0.05, 3.0)",
                        "aunif(1.0, 0.2)",
                        "unif(2.0, 0.1)",
                        "limit(1.0, 0.1)",
                    ]
                )
            )
        return branches

    return _expression_grammar(
        one_of_strings(base_leaves), _SPICE_DIALECT, extra_branches
    )


def _valid_core_expressions(*, with_variables: bool) -> st.SearchStrategy[str]:
    """Generates grammar-guided valid expressions in jolty's Rust syntax."""
    numbers = st.floats(
        min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False
    ).map(lambda value: f"{value:.6g}")
    base_leaves: list[st.SearchStrategy[str]] = [numbers, st.just("pi")]
    if with_variables:
        base_leaves.append(st.sampled_from(["time", "v(a)", "v(a, b)", "i(p1)"]))
    return _expression_grammar(
        one_of_strings(base_leaves),
        _CORE_DIALECT,
        lambda children: [
            children.map(
                lambda inner: f"rounded_table({inner}, 0.25, -2, -1, 0, 0, 2, 1)"
            )
        ],
    )


def _expression_token_soup() -> st.SearchStrategy[str]:
    """Generates adversarial sequences of expression tokens and punctuation."""
    return st.lists(
        st.sampled_from(_EXPRESSION_SOUP_TOKENS), min_size=0, max_size=35
    ).map("".join)


def _exercise_spice_expression(text: str) -> None:
    try:
        parsed = expressions.parse(text)
    except SpiceError:
        return
    assert isinstance(expressions.names(parsed), set)
    assert expressions.inlined(parsed, {}) == parsed
    with contextlib.suppress(SpiceError):
        expressions.inlined(parsed, _USER_FUNCTIONS)
    expressions.substituted(parsed, expressions.Name("alpha"), expressions.Number(1.0))
    expressions.integral(parsed, expressions.Voltage("a", None))
    with contextlib.suppress(SpiceError):
        expressions.evaluate(parsed, _SPICE_PARAMETERS)
    with contextlib.suppress(SpiceError):
        expressions.translate(parsed, _SPICE_PARAMETERS, str.lower, str.lower)


def _exercise_core_expression(text: str) -> None:
    with contextlib.suppress(ValueError):
        _core.constant(text)
    try:
        tree = _core.expression_tree(text)
    except ValueError:
        return
    assert json.loads(tree) is not None or tree == "null"


@FUZZ
@example("")
@example("4.7k")
@example("1\u212a")
@example("1\u0660")
@example("1e500")
@example("1e-500")
@given(st.text(max_size=120))
def test_number_never_raises_on_arbitrary_text(text: str) -> None:
    first = expressions.number(text)
    second = expressions.number(text)
    assert first is None or isinstance(first, float)
    if first is None:
        assert second is None
    elif math.isnan(first):
        assert second is not None
        assert math.isnan(second)
    else:
        assert first == second


@FUZZ
@example("")
@example("(" * 120 + "1" + ")" * 120)
@example("-" * 1100 + "1")
@example("{1} + {2}")
@example("'1' + '2'")
@example("v()")
@example("i()")
@example("v(a, )")
@example("1e500")
@example("1/0")
@example("0/0")
@example("1" + "+1" * 500)
@given(st.text(max_size=200))
def test_spice_expression_arbitrary_text_only_raises_spice_error(text: str) -> None:
    _exercise_spice_expression(text)


@FUZZ
@given(_expression_token_soup())
def test_spice_expression_token_soup_only_raises_spice_error(text: str) -> None:
    _exercise_spice_expression(text)


@FUZZ
@given(_valid_spice_expressions(with_variables=False))
def test_valid_spice_expressions_parse_and_evaluate_deterministically(
    text: str,
) -> None:
    parsed = expressions.parse(text)
    assert expressions.parse(f"{{{text}}}") == parsed
    try:
        first = expressions.evaluate(parsed, _SPICE_PARAMETERS)
    except SpiceError:
        return
    second = expressions.evaluate(parsed, _SPICE_PARAMETERS)
    if math.isnan(first):
        assert math.isnan(second)
    else:
        assert first == second


@FUZZ
@given(_valid_spice_expressions(with_variables=True))
def test_valid_spice_expressions_with_variables_translate_to_core(text: str) -> None:
    parsed = expressions.parse(text)
    node_map = {
        "in": "n_in",
        "out": "n_out",
        "ref": "n_ref",
        "node_a": "n_a",
        "vc+": "n_vc_plus",
        "com": "n_com",
    }
    translated = expressions.translate(
        parsed,
        _SPICE_PARAMETERS,
        lambda node: node_map[node.casefold()],
        lambda source: source.casefold(),
    )
    error_message: str | None = None
    try:
        tree_json = _core.expression_tree(translated)
    except ValueError as error:
        error_message = str(error)
    else:
        assert json.loads(tree_json) is not None or tree_json == "null"
    if error_message is not None:
        assert "levels deep" in error_message


@FUZZ
@example("")
@example("1/0")
@example("0/0")
@example("table(1, 0, 0, 1, 1)")
@example("table(1, 0, 0, 0, 1)")
@example("table(0, 0/0, 1, 1, 2)")
@example("rounded_table(1, 0.25, 0, 0, 1, 1)")
@example("(" * 1000 + "1" + ")" * 1000)
@example("-" * 1000 + "1")
@given(st.text(max_size=150))
def test_core_expression_arbitrary_text_only_raises_value_error(text: str) -> None:
    _exercise_core_expression(text)


@FUZZ
@given(_expression_token_soup())
def test_core_expression_token_soup_only_raises_value_error(text: str) -> None:
    _exercise_core_expression(text)


@FUZZ
@given(_valid_core_expressions(with_variables=False))
def test_valid_core_expressions_evaluate_deterministically(text: str) -> None:
    first = _core.constant(text)
    second = _core.constant(text)
    if math.isnan(first):
        assert math.isnan(second)
    else:
        assert first == second
    json.loads(_core.expression_tree(text))


@FUZZ
@given(_valid_core_expressions(with_variables=True))
def test_valid_core_expressions_with_variables_produce_json_tree(text: str) -> None:
    tree = _core.expression_tree(text)
    assert tree
    json.loads(tree)


def test_expression_parser_regression_bugs() -> None:
    # Deeply nested parentheses and unaries raise SpiceError, not RecursionError.
    for deep in (
        "(" * 120 + "1" + ")" * 120,
        "-" * 1100 + "1",
        "1 ? " * 120 + "1" + " : 2" * 120,
        "1||1&&1==1<1+1*(" * 105 + "1" + ")" * 105,
    ):
        with pytest.raises(SpiceError, match="maximum nesting depth"):
            expressions.parse(deep)

    # Multiple braced sub-expressions must not strip mismatched outer braces.
    assert expressions.evaluate(expressions.parse("{1} + {2}"), {}) == 3.0
    assert expressions.evaluate(expressions.parse("{ {1} + {2} }"), {}) == 3.0
    assert expressions.evaluate(expressions.parse('{"-1/2"}'), {}) == -0.5

    # Empty V() / I() node names are rejected at parse time.
    for invalid in ("v()", "V( )", "v(a, )", "v(, b)", "i()", "I( )"):
        with pytest.raises(SpiceError):
            expressions.parse(invalid)

    # Flat long binary chains never raise RecursionError in AST walkers, and evaluate
    # guards parenthesis depth before crossing into the Rust parser.
    long_sum = expressions.parse("1" + "+1" * 1500)
    second_sum = expressions.parse("1" + "+1" * 1500)
    assert long_sum == second_sum
    assert hash(long_sum) == hash(second_sum)
    assert expressions.names(long_sum) == set()
    assert expressions.inlined(long_sum, {}) is long_sum
    substituted = expressions.substituted(
        long_sum, expressions.Name("x"), expressions.Number(0.0)
    )
    assert substituted is long_sum
    assert substituted == second_sum
    empty_call = expressions.Call("f", ())
    assert empty_call == expressions.Call("f", ())
    assert empty_call != expressions.Call("g", ())
    assert hash(empty_call) == hash(expressions.Call("f", ()))
    translated = expressions.translate(long_sum, {}, str.lower, str.lower)
    assert nesting_depth(translated) == 0
    assert expressions.evaluate(long_sum, {}) == 1501.0
    with pytest.raises(SpiceError, match="parentheses deep"):
        expressions.check_core_depth("(" * 300 + "1" + ")" * 300)

    # Zero-argument call AST inlines cleanly; empty parse syntax raises SpiceError.
    with pytest.raises(SpiceError):
        expressions.parse("user_zero()")
    assert (
        expressions.evaluate(
            expressions.inlined(expressions.Call("user_zero", ()), _USER_FUNCTIONS), {}
        )
        == 42.0
    )
    with pytest.raises(SpiceError):
        expressions.inlined(expressions.parse("user_loop(1)"), _USER_FUNCTIONS)

    # Exponential `.func` expansion stops at the node budget instead of hanging.
    chain: dict[str, expressions.Function] = {
        "f0": expressions.Function(("x",), expressions.parse("x + x"))
    }
    for index in range(1, 24):
        previous = f"f{index - 1}"
        chain[f"f{index}"] = expressions.Function(
            ("x",), expressions.parse(f"{previous}(x) + {previous}(x)")
        )
    with pytest.raises(SpiceError, match="inlining user functions exceeds"):
        expressions.inlined(expressions.parse("f23(1)"), chain)

    # High polynomial exponents return None instead of hanging or blowing recursion.
    variable = expressions.Voltage("a", None)
    assert expressions.integral(expressions.parse("v(a)^100000"), variable) is None
    assert expressions.integral(expressions.parse("v(a)^1e20"), variable) is None


def test_core_expression_depth_and_non_finite_table_regression_bugs() -> None:
    # Rust expression parser guards recursion depth and non-finite table points.
    for text, match, fn in (
        ("(" * 1000 + "1" + ")" * 1000, "levels deep", _core.constant),
        ("-" * 1000 + "v(a)", "levels deep", _core.expression_tree),
        ("2 ^ " * 300 + "2", "levels deep", _core.constant),
        ("table(0, 0/0, 1, 1, 2)", "ascending abscissae", _core.constant),
        ("table(v(a), 0, 1/0, 1, 2)", "ascending abscissae", _core.expression_tree),
    ):
        with pytest.raises(ValueError, match=match):
            fn(text)

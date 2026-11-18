# Copyright 2026 Maxence Faldor
"""Every engine constant and test tolerance names its source.

Enforces two policies from `AGENTS.md`:

1. Every named `const` declaration in `crates/jolty-engine/src/`,
   `crates/jolty-numerics/src/`, and `crates/jolty-checks/src/` must carry a
   `///` doc comment explaining where its value comes from: a literature or
   standard reference, a verified source file line, a mathematical derivation,
   a measured machine benchmark, or an explicit jolty design origin.
2. Every tolerance comparison site across `tests/**/*.py` (`pytest.approx`,
   `assert_allclose`, `assert_almost_equal`, `assert_array_almost_equal`,
   `isclose`, `allclose`, `compare(..., tolerance=...)`, `_agree(...)`, and
   direct `assert` error bounds against a positive float literal below 1.0 or
   a named tolerance constant) must have a genuine local rationale explaining
   why that tolerance has its value.
"""

import ast
import re
import textwrap
import tokenize
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Final

from workspace import ROOT

CONSTANT_CRATES: Final[tuple[str, ...]] = (
    "jolty-engine",
    "jolty-numerics",
    "jolty-checks",
)
"""Workspace crates whose `src/` `const` items must document their origin."""

_CONSTANT_DECLARATION: Final = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?const\s+(?!fn\b|unsafe\b)([A-Za-z_][A-Za-z0-9_]*)\s*:",
)

_CONSTANT_SOURCE_PATTERN: Final = re.compile(
    r"(?i)(?:"
    r"\b(?:source|origin|derivation|measured|sentinel|additive identity)\b"
    r"|\b(?:ngspice|spice|numpy|ieee|iec|ietf|fnv|faer)\b"
    r"|\b(?:hairer|higham|kelley|farrell|ruiz|golub|duff|davis|nagel|seydel|aprille|wilson)\b"
    r"|draft-eastlake-fnv|agents\.md"
    r"|\b(?:transcendental|workspace|longest step|corpus|rounding|ratio of 1)\b"
    r"|2\^|2\u00b3\u2079|2\u2075\u00b3|1 <<|C\u2081"
    r"|compiled_base\.c|loops_utils\.h\.src|traninit\.c|dctran\.c|cktop\.c"
    r"|cktntask\.c|ckttrunc\.c|dctrcurv\.c|fourier\.c"
    r")",
)

_TOLERANCE_CALL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "approx",
        "assert_allclose",
        "assert_almost_equal",
        "assert_array_almost_equal",
        "isclose",
        "allclose",
        "_agree",
    }
)

_TOLERANCE_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "rel",
        "abs",
        "rtol",
        "atol",
        "rel_tol",
        "abs_tol",
        "tolerance",
        "allowance",
    }
)

_MAGNITUDE_CALL_NAMES: Final[frozenset[str]] = frozenset({"abs", "fabs", "norm"})

_ERROR_NAME_FRAGMENTS: Final[tuple[str, ...]] = (
    "err",
    "residual",
    "mismatch",
    "deviation",
    "diff",
    "drift",
    "discrepancy",
    "disagreement",
    "stray",
)

_SUPPRESSION_COMMENT: Final = re.compile(
    r"(?i)^\s*(?:noqa|type:\s*ignore|ty:|fmt:|pylint:|pragma:|todo|fixme)\b",
)

_TOLERANCE_RATIONALE_PATTERN: Final = re.compile(
    r"(?i)(?:"
    r"\b(?:round(?:ing|ings|ed)?|eps(?:ilon)?|ulp|ieee-?754|floating-point"
    r"|truncat(?:ion|ed|es)?|cancell?ation|condition(?:ing|ed)?|precision|machine)\b"
    r"|\b(?:reltol|vntol|abstol|trtol|rtol|atol|rel_tol|abs_tol|tight"
    r"|tolerance|allowance|disagreement|discrepancy)\b"
    r"|(?<![A-Za-z0-9_])(?:rel|abs|rtol|atol)="
    r"|\b(?:newton|lu|qz|krylov|gmres|svd|eigenvalue|lineariz(?:ation|ed)"
    r"|jacobian|adjoint|residual|converg(?:ence|e|ed|es))\b"
    r"|\b(?:trapezoidal|bdf|lte|discretiz(?:ation|ed)|quadrature|interpolat(?:ion|ed)"
    r"|chord sag|finite[- ]difference|central[- ]difference|forward[- ]difference"
    r"|second[- ]order|resample|resampling)\b"
    r"|o\((?:h|step|eps|dt)"
    r"|\b(?:standard error|monte carlo|sigma|variance|quantil(?:e|es)|confidence"
    r"|bootstrap|z-score|sample size|sampling|chi-square|uniform support"
    r"|p\(max|median std)\b"
    r"|\b(?:codata|boltzmann|2019 si|gmin|shunt|thermal voltage|decimal"
    r"|serializ(?:ation|ed)?|formatting|quantiz(?:ation|ed)?)\b"
    r"|\b(?:closed[- ]loop|closed[- ]form|midpoint|cutoff|tail|leakage"
    r"|nanovolts?|dot-product)\b"
    r"|\b(?:pytest\.approx|assert_allclose|max_step|err(?:s|or|ors)?|deviation)\b"
    r")",
)


def _documentation_lines_before(lines: list[str], declaration_index: int) -> list[str]:
    """Collect the contiguous `///` doc comment attached to a `const` declaration."""
    cursor = declaration_index - 1
    while cursor >= 0 and lines[cursor].strip().startswith("#["):
        cursor -= 1
    collected: list[str] = []
    while cursor >= 0 and lines[cursor].strip().startswith("///"):
        collected.append(lines[cursor].strip().removeprefix("///").strip())
        cursor -= 1
    collected.reverse()
    return collected


def find_uncited_rust_constants(source: str, label: str) -> list[str]:
    """Return descriptions of named Rust `const` items lacking a documented source."""
    lines = source.splitlines()
    violations: list[str] = []
    for line_index, line in enumerate(lines):
        match = _CONSTANT_DECLARATION.match(line)
        if match is None:
            continue
        name = match.group(1)
        if name == "_":
            continue
        documentation = " ".join(_documentation_lines_before(lines, line_index))
        line_number = line_index + 1
        if not documentation:
            violations.append(f"{label}:{line_number}: `const {name}` has no `///` doc")
        elif _CONSTANT_SOURCE_PATTERN.search(documentation) is None:
            violations.append(
                f"{label}:{line_number}: `const {name}` doc names no source: "
                f"{documentation}"
            )
    return violations


def _rust_source_files() -> list[Path]:
    """Return every `.rs` source file under the audited crates."""
    files: list[Path] = []
    for crate_name in CONSTANT_CRATES:
        source_root = ROOT / "crates" / crate_name / "src"
        files.extend(sorted(source_root.rglob("*.rs")))
    return files


def _callable_short_name(function: ast.expr) -> str | None:
    """Return the unqualified function or attribute name of a call target."""
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _extract_comments(source: str) -> dict[int, str]:
    """Map 1-based line numbers to non-suppression comment text."""
    comments: dict[int, str] = {}
    for token in tokenize.generate_tokens(StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        text = token.string.lstrip("#").strip()
        if text and _SUPPRESSION_COMMENT.match(text) is None:
            comments[token.start[0]] = text
    return comments


def _extract_constant_documentation(
    module: ast.Module, comments: dict[int, str]
) -> dict[str, str]:
    """Collect preceding comments and attached docstrings for `ALL_CAPS` constants."""
    documented: dict[str, str] = {}
    statements = module.body
    for index, statement in enumerate(statements):
        names: list[str] = []
        if isinstance(statement, ast.Assign):
            names.extend(
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name) and target.id.isupper()
            )
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id.isupper()
        ):
            names.append(statement.target.id)
        if not names:
            continue
        end_line = statement.end_lineno or statement.lineno
        pieces = [
            comments[line_number]
            for line_number in range(max(1, statement.lineno - 4), end_line + 1)
            if line_number in comments
        ]
        if index + 1 < len(statements):
            next_statement = statements[index + 1]
            if (
                isinstance(next_statement, ast.Expr)
                and isinstance(next_statement.value, ast.Constant)
                and isinstance(next_statement.value.value, str)
            ):
                pieces.append(next_statement.value.value)
        combined = " ".join(pieces)
        for name in names:
            documented[name] = combined
    return documented


def _has_error_metric(expression: ast.expr) -> bool:
    """Return whether `expression` computes an absolute magnitude or named error."""
    for child in ast.walk(expression):
        if isinstance(child, ast.Call):
            if _callable_short_name(child.func) in _MAGNITUDE_CALL_NAMES:
                return True
        elif isinstance(child, ast.Name):
            lowered = child.id.lower()
            if any(fragment in lowered for fragment in _ERROR_NAME_FRAGMENTS):
                return True
        elif isinstance(child, ast.Attribute):
            lowered = child.attr.lower()
            if any(fragment in lowered for fragment in _ERROR_NAME_FRAGMENTS):
                return True
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            lowered = child.value.lower()
            if any(fragment in lowered for fragment in _ERROR_NAME_FRAGMENTS):
                return True
    return False


def _has_small_float_literal(expression: ast.expr) -> bool:
    """Return whether `expression` contains a float literal in `(0, 1)` outside `**`."""
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Pow):
        return _has_small_float_literal(expression.left)
    if isinstance(expression, ast.Constant) and isinstance(expression.value, float):
        return 0.0 < expression.value < 1.0
    return any(
        _has_small_float_literal(child)
        for child in ast.iter_child_nodes(expression)
        if isinstance(child, ast.expr)
    )


def _referenced_uppercase_names(node: ast.AST) -> set[str]:
    """Return all `ALL_CAPS` identifier names referenced inside `node`."""
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and child.id.isupper()
    }


def _is_tolerance_call(call: ast.Call) -> str | None:
    """Return the matched call kind if `call` is a tolerance comparison, else None."""
    name = _callable_short_name(call.func)
    if name in _TOLERANCE_CALL_NAMES:
        return name
    if name == "compare" and any(
        keyword.arg in {"tolerance", "allowance"} for keyword in call.keywords
    ):
        return "compare"
    return None


def _is_tolerance_bound(
    expression: ast.expr,
    constant_documentation: dict[str, str],
) -> bool:
    """Return whether `expression` is a literal `< 1` bound or documented constant."""
    return _has_small_float_literal(expression) or any(
        name in constant_documentation
        for name in _referenced_uppercase_names(expression)
    )


@dataclass(frozen=True, slots=True)
class _ToleranceSite:
    """Single tolerance comparison site located within a Python test statement."""

    start_line: int
    end_line: int
    kind: str
    node: ast.AST


@dataclass(frozen=True, slots=True)
class _ScopeContext:
    """Enclosing function or class scope metadata during AST traversal."""

    docstring: str
    parameters: frozenset[str]
    is_helper: bool


@dataclass(frozen=True, slots=True)
class _ModuleContext:
    """File-level comments and documented `ALL_CAPS` constants."""

    comments: dict[int, str]
    constant_documentation: dict[str, str]


def _extract_compare_sites(
    comparison: ast.Compare,
    constant_documentation: dict[str, str],
) -> list[_ToleranceSite]:
    """Return direct `assert` comparison tolerance sites inside `comparison`."""
    sites: list[_ToleranceSite] = []
    span_start = comparison.lineno
    span_end = comparison.end_lineno or comparison.lineno
    left_operand = comparison.left
    for operator, right_operand in zip(
        comparison.ops, comparison.comparators, strict=True
    ):
        if (
            isinstance(operator, (ast.Lt, ast.LtE))
            and _has_error_metric(left_operand)
            and _is_tolerance_bound(right_operand, constant_documentation)
        ):
            sites.append(
                _ToleranceSite(span_start, span_end, "assert_compare", right_operand)
            )
        elif (
            isinstance(operator, (ast.Gt, ast.GtE))
            and _has_error_metric(right_operand)
            and _is_tolerance_bound(left_operand, constant_documentation)
        ):
            sites.append(
                _ToleranceSite(span_start, span_end, "assert_compare", left_operand)
            )
        left_operand = right_operand
    return sites


def _collect_statement_sites(
    statement: ast.stmt,
    constant_documentation: dict[str, str],
) -> list[_ToleranceSite]:
    """Return tolerance sites directly belonging to `statement`."""
    sites: list[_ToleranceSite] = []

    def walk_expression(node: ast.AST) -> None:
        if isinstance(node, ast.stmt) and node is not statement:
            return
        if isinstance(node, ast.Call):
            kind = _is_tolerance_call(node)
            if kind is not None:
                sites.append(
                    _ToleranceSite(
                        node.lineno, node.end_lineno or node.lineno, kind, node
                    )
                )
                return
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            sites.extend(_extract_compare_sites(node.test, constant_documentation))
        for child in ast.iter_child_nodes(node):
            walk_expression(child)

    walk_expression(statement)
    return sites


def _block_group_start_line(
    statements: list[ast.stmt],
    index: int,
    lines: list[str],
    comments: dict[int, str],
) -> int:
    """Walk upward across contiguous `assert`/call lines sharing a header comment."""
    top_line = statements[index].lineno
    cursor = index - 1
    while cursor >= 0:
        previous = statements[cursor]
        previous_end = previous.end_lineno or previous.lineno
        gap_lines = lines[previous_end : top_line - 1]
        if any(line.strip() == "" for line in gap_lines):
            break
        if any(
            line_number in comments for line_number in range(previous_end + 1, top_line)
        ):
            break
        if isinstance(previous, (ast.Assert, ast.Expr)):
            top_line = previous.lineno
            cursor -= 1
        else:
            break
    return top_line


def _forwards_helper_parameter(
    node: ast.AST,
    scope: _ScopeContext,
) -> bool:
    """Return whether a helper call forwards one of its own tolerance parameters."""
    if not isinstance(node, ast.Call) or not _has_rationale_text(scope.docstring):
        return False
    return any(
        keyword.arg in _TOLERANCE_KEYWORDS
        and any(
            isinstance(child, ast.Name) and child.id in scope.parameters
            for child in ast.walk(keyword.value)
        )
        for keyword in node.keywords
    )


def _sub_statement_blocks(statement: ast.stmt) -> list[list[ast.stmt]]:
    """Return nested statement lists owned by a compound AST statement."""
    blocks: list[list[ast.stmt]] = []
    for attribute in ("body", "orelse", "finalbody"):
        child_block = getattr(statement, attribute, None)
        if isinstance(child_block, list):
            blocks.append(child_block)
    if isinstance(statement, ast.Try):
        blocks.extend(handler.body for handler in statement.handlers)
    if isinstance(statement, ast.Match):
        blocks.extend(case.body for case in statement.cases)
    return blocks


def _has_rationale_text(text: str) -> bool:
    """Return whether `text` contains genuine tolerance rationale terminology."""
    return bool(text) and _TOLERANCE_RATIONALE_PATTERN.search(text) is not None


def _comments_in_range(
    comments: dict[int, str],
    start_line: int,
    end_line: int,
) -> list[str]:
    """Return all non-suppression comments within `[start_line, end_line]`."""
    return [
        comments[line_number]
        for line_number in range(max(1, start_line), end_line + 1)
        if line_number in comments
    ]


def _site_is_cited(
    site: _ToleranceSite,
    statement_span: tuple[int, int],
    module_context: _ModuleContext,
    scope: _ScopeContext,
) -> bool:
    """Return whether a tolerance site has a documented local source/rationale."""
    group_start, statement_end = statement_span
    nearby = _comments_in_range(module_context.comments, group_start - 3, statement_end)
    node_local = _comments_in_range(
        module_context.comments, site.start_line - 3, site.end_line
    )
    if any(_has_rationale_text(comment) for comment in (*nearby, *node_local)):
        return True
    if _has_rationale_text(scope.docstring):
        return True
    if any(
        name in module_context.constant_documentation
        and _has_rationale_text(module_context.constant_documentation[name])
        for name in _referenced_uppercase_names(site.node)
    ):
        return True
    return scope.is_helper and _forwards_helper_parameter(site.node, scope)


def find_uncited_test_tolerances(
    source: str,
    label: str,
) -> tuple[int, list[str]]:
    """Return `(total_sites, violations)` for Python test tolerance sites."""
    module = ast.parse(source, filename=label)
    lines = source.splitlines()
    comments = _extract_comments(source)
    module_context = _ModuleContext(
        comments=comments,
        constant_documentation=_extract_constant_documentation(module, comments),
    )
    total_sites = 0
    violations: list[str] = []

    def visit_block(statements: list[ast.stmt], scope: _ScopeContext) -> None:
        nonlocal total_sites
        for index, statement in enumerate(statements):
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters = frozenset(
                    argument.arg
                    for argument in (
                        *statement.args.posonlyargs,
                        *statement.args.args,
                        *statement.args.kwonlyargs,
                    )
                )
                nested_scope = _ScopeContext(
                    docstring=ast.get_docstring(statement) or "",
                    parameters=parameters,
                    is_helper=not statement.name.startswith("test_"),
                )
                visit_block(statement.body, nested_scope)
                continue
            if isinstance(statement, ast.ClassDef):
                class_docstring = ast.get_docstring(statement) or ""
                nested_scope = _ScopeContext(
                    docstring=class_docstring or scope.docstring,
                    parameters=scope.parameters,
                    is_helper=scope.is_helper,
                )
                visit_block(statement.body, nested_scope)
                continue

            group_start = _block_group_start_line(statements, index, lines, comments)
            for site in _collect_statement_sites(
                statement, module_context.constant_documentation
            ):
                total_sites += 1
                statement_span = (group_start, statement.end_lineno or site.end_line)
                if not _site_is_cited(site, statement_span, module_context, scope):
                    snippet = lines[site.start_line - 1].strip()
                    violations.append(
                        f"{label}:{site.start_line}: uncited `{site.kind}` "
                        f"tolerance: {snippet}"
                    )

            for child_block in _sub_statement_blocks(statement):
                visit_block(child_block, scope)

    root_scope = _ScopeContext(docstring="", parameters=frozenset(), is_helper=False)
    visit_block(module.body, root_scope)
    return total_sites, violations


def _python_test_files() -> list[Path]:
    """Return every `.py` file under `tests/` except this guard module itself."""
    tests_root = ROOT / "tests"
    return sorted(
        path for path in tests_root.rglob("*.py") if path.name != "test_tolerances.py"
    )


def test_every_engine_constant_names_its_source() -> None:
    files = _rust_source_files()
    assert files, "expected Rust source files under the audited crates"
    violations: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        violations.extend(
            find_uncited_rust_constants(path.read_text(encoding="utf-8"), relative)
        )
    assert not violations, "\n".join(violations)


def test_constant_guard_flags_undocumented_and_uncited_constants() -> None:
    sample = textwrap.dedent(
        """\
        const BARE: f64 = 0.25;
        /// The damping factor.
        const UNCITED: f64 = 0.5;
        /// Safe anonymous compile-time assertion.
        const _: () = {};
        /// Step shrink factor after non-convergence.
        /// Source: ngspice 47 (`dctran.c`, line 733: `CKTdelta/8`).
        const CITED: f64 = 0.125;
        /// Internal helper constant function, not a value constant.
        const fn helper() -> bool { true }
        """
    )
    findings = find_uncited_rust_constants(sample, "synthetic.rs")
    assert len(findings) == 2
    assert "BARE" in findings[0]
    assert "no `///` doc" in findings[0]
    assert "UNCITED" in findings[1]
    assert "names no source" in findings[1]


def test_every_test_tolerance_names_its_source() -> None:
    files = _python_test_files()
    assert files, "expected Python test files under tests/"
    total_sites = 0
    violations: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        count, file_violations = find_uncited_test_tolerances(
            path.read_text(encoding="utf-8"), relative
        )
        total_sites += count
        violations.extend(file_violations)
    assert total_sites > 350, f"expected >350 tolerance sites, found {total_sites}"
    assert not violations, "\n".join(violations)


def test_tolerance_guard_flags_uncited_tolerance_sites() -> None:
    sample = textwrap.dedent(
        """\
        import math
        import numpy as np
        import pytest

        def test_uncited_cases(x, a, b, scale):
            assert x == pytest.approx(1.0, rel=1e-5)
            assert x == pytest.approx(1.0)
            np.testing.assert_allclose(a, b, rtol=1e-4)
            assert math.isclose(a, b, rel_tol=1e-6)
            assert np.allclose(a, b, rtol=1e-5)
            compare(a, b, "v", scale=1.0, tolerance=1e-6)
            assert abs(a - b) <= 1e-6
            assert abs(a - b) <= 1e-6 * scale
            # Check output matches the reference value.
            assert x == pytest.approx(2.0, rel=1e-4)
            # noqa: S101
            assert x == pytest.approx(3.0, rel=1e-4)
            # Adding a .param invalidates caches.
            assert x == pytest.approx(4.0, rel=1e-4)
        """
    )
    total, violations = find_uncited_test_tolerances(sample, "synthetic_uncited.py")
    assert total == 11
    assert len(violations) == 11, "\n".join(violations)


def test_tolerance_guard_accepts_cited_tolerance_sites() -> None:
    sample = textwrap.dedent(
        """\
        import math
        import numpy as np
        import pytest

        AGREEMENT = 2e-6
        \"\"\"Twice ngspice relative tolerance (reltol=1e-6) for DC solves.\"\"\"

        def check_close(left, right, tolerance):
            \"\"\"Compare arrays within `tolerance` covering Newton convergence.\"\"\"
            np.testing.assert_allclose(left, right, rtol=tolerance)

        def test_preceding_and_inline(x, a, b, scale):
            # IEEE-754 rounding (~4500 eps) for linear division.
            assert x == pytest.approx(1.0, rel=1e-12)
            assert a == pytest.approx(1.0, rel=1e-12)
            assert math.isclose(a, b, rel_tol=1e-12)  # IEEE-754 rounding bound.
            compare(a, b, "v", scale=scale, tolerance=AGREEMENT)
            # Physical range checks and square-root exponents are not tolerances.
            assert 0.6 < x < 0.8
            assert x ** 0.5 < 10.0

        def test_function_docstring_citation(x, y):
            \"\"\"Closed-form test; rel=1e-9 covers dense QZ eigenvalue rounding.\"\"\"
            assert x == pytest.approx(1.0, rel=1e-9)
            assert abs(x - y) <= 1e-9
        """
    )
    total, violations = find_uncited_test_tolerances(sample, "synthetic_cited.py")
    assert total == 7
    assert not violations, "\n".join(violations)

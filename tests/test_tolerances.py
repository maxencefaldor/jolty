# Copyright 2026 Maxence Faldor
"""Every engine constant names its source (`AGENTS.md`, "Constants").

Every named `const` declaration in `crates/jolty-engine/src/`,
`crates/jolty-numerics/src/`, and `crates/jolty-checks/src/` must carry a `///`
doc comment explaining where its value comes from: a literature or standard
reference, a verified source file line, a mathematical derivation, a measured
machine benchmark, or an explicit jolty design origin.
"""

import re
import textwrap
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
    r"|compiled_base\.c|loops_utils\.h\.src|traninit\.c|dctran\.c|cktop\.c|cktntask\.c|ckttrunc\.c|dctrcurv\.c|fourier\.c"
    r")",
)


def _doc_lines_before(lines: list[str], declaration_index: int) -> list[str]:
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
        doc_text = " ".join(_doc_lines_before(lines, line_index))
        line_number = line_index + 1
        if not doc_text:
            violations.append(f"{label}:{line_number}: `const {name}` has no `///` doc")
        elif _CONSTANT_SOURCE_PATTERN.search(doc_text) is None:
            violations.append(
                f"{label}:{line_number}: `const {name}` doc names no source: {doc_text}"
            )
    return violations


def _rust_source_files() -> list[Path]:
    """Return every `.rs` source file under the audited crates."""
    files: list[Path] = []
    for crate_name in CONSTANT_CRATES:
        source_root = ROOT / "crates" / crate_name / "src"
        files.extend(sorted(source_root.rglob("*.rs")))
    return files


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

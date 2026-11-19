# Copyright 2026 Maxence Faldor
"""Coverage matrix runner, evaluator, and Markdown renderer.

``scripts/bench --coverage`` invokes this module through ``benchmarks/speed/bench.py``.
Each row in ``capabilities.py`` executes in jolty, ngspice (Sparse 1.3 and KLU), or the
simulators selected by ``--simulator`` via ``benchmarks/simulators.py``. Cells are
scored against each row's known answer:

- ``agreed``: simulator ran and every extracted value lies within tolerance of the known
  answer.
- ``disagreed``: simulator ran, and at least one value falls outside tolerance.
- ``failed``: simulator accepted the fixture but did not complete, recording stderr.
- ``unsupported``: simulator rejected the fixture or lacks a matching statement, citing
  its error or manual section.
- ``solved``: simulator ran, and the row has no known answer yet.
- ``reference``: defining simulator whose output supplies the known answer.
- ``documented``: simulator unavailable locally documents the capability in its manual.
- ``unknown``: no consulted manual documents support for the row.

ngspice executes fixture decks via `jolty.spice.ngspice.runs` (one ASCII raw file per
command), while Xyce writes requested reads as CSV via ``.print``. Simulator output
patterns distinguish explicit refusals from solver failures.

Usage::

    scripts/bench --coverage                 every row, in jolty and ngspice
    scripts/bench --coverage op ac           only these rows
    scripts/bench --coverage --simulator xyce-serial   Xyce's column, where it runs

Results are merged into ``benchmarks/coverage/coverage.json`` (updating only executed
rows and simulators) and rendered into ``benchmarks/coverage/matrix.md``. See
``benchmarks/coverage/README.md`` for fixture authoring details.
"""

import csv
import re
import time
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Final

import documented
import numpy as np
from capabilities import (
    ROOM,
    Category,
    Documented,
    Fixture,
    Known,
    Read,
    Spice,
    Values,
    fixtures,
)
from measuring import (
    ROOT,
    Provenance,
    load,
    load_average,
    markdown_table,
    provenance,
    save,
    shown,
)
from pydantic import BaseModel, Field
from simulators import Ngspice, Simulator, Unsupported, Xyce, runner

from jolty.errors import JoltyError
from jolty.spice import ngspice, xyce

COVERAGE: Final = Path(__file__).resolve().parent
"""Directory containing the coverage suite and generated artifacts."""

RESULTS: Final = COVERAGE / "coverage.json"
"""JSON data file storing evaluated coverage cells."""

MATRIX: Final = COVERAGE / "matrix.md"
"""Rendered Markdown coverage matrix output path."""

TIMEOUT: Final = 60
"""Per-fixture timeout in seconds for external SPICE simulators."""

SIMULATORS: Final = (
    Simulator.JOLTY,
    Simulator.NGSPICE_SPARSE,
    Simulator.NGSPICE_KLU,
    Simulator.XYCE_SERIAL,
)
"""Executable simulators supported by the coverage matrix."""


class Status(StrEnum):
    """Outcome classification for a coverage matrix cell."""

    AGREED = "agreed"
    DISAGREED = "disagreed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    SOLVED = "solved"
    REFERENCE = "reference"
    DOCUMENTED = "documented"
    UNKNOWN = "unknown"


class Cell(BaseModel):
    """Evaluation result of one simulator on one coverage row."""

    status: Status
    evidence: str = ""
    found: dict[str, tuple[float, float]] | None = None


class SimulatorProvenance(Provenance):
    """Provenance record for a simulator run, adding version, host load, and date."""

    version: str
    load: float
    date: str


class FixtureResults(BaseModel):
    """Serialized coverage row metadata, reference values, and per-simulator cells."""

    category: Category
    name: str
    source: str
    known: dict[str, tuple[float, float]] | None
    cells: dict[str, Cell] = Field(default_factory=dict)


class SuiteResults(BaseModel):
    """Root schema for ``coverage.json``, holding simulator provenance and row cells."""

    simulators: dict[str, SimulatorProvenance] = Field(default_factory=dict)
    fixtures: dict[str, FixtureResults] = Field(default_factory=dict)


def _pairs(values: Values) -> dict[str, tuple[float, float]]:
    return {name: (value.real, value.imag) for name, value in values.items()}


def _complex(pairs: Mapping[str, tuple[float, float]]) -> Values:
    return {name: complex(*pair) for name, pair in pairs.items()}


def judge(known: Known, found: Values, reference: Values | None) -> tuple[Status, str]:
    """Evaluates whether `found` agrees with `known` or the `reference` simulator run.

    Scales each error by the largest known magnitude of the same quantity kind (such as
    ``v`` or ``i``) in the row and marks agreement when all errors stay within
    `known.tolerance`.
    """
    expected = known.values if known.values is not None else reference
    if expected is None:
        return Status.SOLVED, ""
    missing = sorted(set(expected) - set(found))
    if missing:
        return Status.DISAGREED, f"read no {', '.join(missing)}"
    scales: dict[str, float] = {}
    for name, value in expected.items():
        kind = name.split("(")[0]
        scales[kind] = max(scales.get(kind, 0.0), abs(value))
    worst, said = 0.0, ""
    for name, value in expected.items():
        scale = scales[name.split("(")[0]] or 1.0
        error = abs(found[name] - value) / scale
        if error > worst:
            worst = error
            said = f"{name}: {_number(found[name])} for {_number(value)}"
    if worst <= known.tolerance:
        return Status.AGREED, f"largest error {worst:.2g} of the scale"
    return Status.DISAGREED, f"{said}, an error of {worst:.3g} of the scale"


def _number(value: complex) -> str:
    """Formats `value` compactly, omitting the imaginary part when zero."""
    if value.imag == 0:
        return f"{value.real:.7g}"
    return f"{value.real:.7g}{value.imag:+.7g}j"


class _NotInstalledError(Exception):
    """Exception raised when a requested simulator binary is not installed locally."""


def run(fixture: Fixture, simulator: Simulator) -> Cell:
    """Runs `fixture` in `simulator` and returns an unjudged `Cell`.

    Strips the repository root prefix from error text so paths stay portable across
    hosts.
    """
    try:
        match simulator:
            case Simulator.JOLTY:
                found = run_jolty(fixture)
            case Simulator.NGSPICE_SPARSE | Simulator.NGSPICE_KLU:
                found = run_ngspice(fixture, Ngspice(simulator))
            case Simulator.XYCE_SERIAL:
                found = run_xyce(fixture)
            case _:
                message = f"the matrix runs no {simulator}"
                raise ValueError(message)
    except Unsupported as refused:
        return Cell(status=Status.UNSUPPORTED, evidence=_relative(str(refused), ROOT))
    except Documented as manual:
        return Cell(status=Status.DOCUMENTED, evidence=str(manual))
    except _NotInstalledError as missing:
        return Cell(status=Status.UNKNOWN, evidence=str(missing))
    except Failed as failure:
        return Cell(status=Status.FAILED, evidence=_relative(str(failure), ROOT))
    return Cell(status=Status.SOLVED, found=_pairs(found))


def _relative(message: str, root: Path) -> str:
    """Rewrites paths under `root` inside `message` as repository-relative paths."""
    return message.replace(f"{root}/", "")


def rejudge(document: SuiteResults, every: Mapping[str, Fixture]) -> None:
    """Judges every cell that ran against its row's known answer, or its reference."""
    for key, line in document.fixtures.items():
        fixture = every.get(key)
        if fixture is None or fixture.known is None:
            continue
        reference = line.cells.get(fixture.known.reference or "")
        expected = (
            None
            if reference is None or reference.found is None
            else _complex(reference.found)
        )
        for simulator, cell in line.cells.items():
            if cell.found is None:
                continue
            if simulator == fixture.known.reference:
                cell.status, cell.evidence = Status.REFERENCE, ""
                continue
            cell.status, cell.evidence = judge(
                fixture.known, _complex(cell.found), expected
            )


def measure(names: Sequence[str], chosen: Sequence[Simulator]) -> None:
    """Runs the rows `names` in the simulators `chosen`, and renders the matrix.

    Every row runs without names, and the `DEFAULTS` without simulators chosen; Xyce
    runs where it is on the path.

    Raises:
        SystemExit: When a row named does not exist, or a simulator chosen does not run
            the matrix.
    """
    every = {fixture.key: fixture for fixture in fixtures()}
    unknown = [name for name in names if name not in every]
    if unknown:
        message = f"no row is named {', '.join(unknown)}; there are {', '.join(every)}"
        raise SystemExit(message)
    strange = [simulator for simulator in chosen if simulator not in SIMULATORS]
    if strange:
        message = (
            f"the matrix runs no {', '.join(strange)}; it runs {', '.join(SIMULATORS)}"
        )
        raise SystemExit(message)
    simulators = list(chosen) or list(DEFAULTS)
    document = (
        SuiteResults.model_validate(load(RESULTS))
        if RESULTS.is_file()
        else SuiteResults()
    )
    measured, average = provenance(), load_average()
    for simulator in simulators:
        document.simulators[simulator] = SimulatorProvenance(
            **measured.model_dump(),
            version=runner(simulator).version() or "",
            load=average,
            date=time.strftime("%Y-%m-%d"),
        )
    for name in names or list(every):
        fixture = every[name]
        line = document.fixtures.get(name) or FixtureResults(
            category=fixture.category, name=fixture.name, source="", known=None
        )
        line.category, line.name = fixture.category, fixture.name
        answer = fixture.known
        line.source = "" if answer is None else answer.source
        line.known = (
            None if answer is None or answer.values is None else _pairs(answer.values)
        )
        found = []
        for simulator in simulators:
            cell = run(fixture, simulator)
            line.cells[simulator] = cell
            found.append(f"{simulator} {cell.status}")
        document.fixtures[name] = line
        print(f"{name:24} {' | '.join(found)}", flush=True)
    document.fixtures = {
        key: document.fixtures[key] for key in every if key in document.fixtures
    }
    rejudge(document, every)
    save(RESULTS, document)
    render(document)
    print(f"wrote {shown(MATRIX)}")


DEFAULTS: Final = (
    Simulator.JOLTY,
    Simulator.NGSPICE_SPARSE,
    Simulator.NGSPICE_KLU,
)
"""The simulators a run runs unless it names others: those installed with jolty's
tests, ngspice with Sparse 1.3 and with KLU."""


COLUMNS: Final = (
    (Simulator.JOLTY, "jolty"),
    (Simulator.NGSPICE_SPARSE, "ngspice 47"),
    (Simulator.NGSPICE_KLU, "ngspice 47, KLU"),
    (Simulator.XYCE_SERIAL, "Xyce 7.10"),
    *((name, title) for name, title in documented.TITLES.items()),
)
"""The matrix's columns: each simulator configuration, those run first."""

SHORT: Final = {
    Status.AGREED: "agreed",
    Status.DISAGREED: "**disagreed**",
    Status.FAILED: "**failed**",
    Status.UNSUPPORTED: "no",
    Status.SOLVED: "solved",
    Status.REFERENCE: "reference",
    Status.DOCUMENTED: "documented",
    Status.UNKNOWN: "?",
}
"""Display label for each `Status` in rendered Markdown tables."""

TITLES: Final = {
    Category.ANALYSIS: "Analyses",
    Category.DEVICE: "Devices",
    Category.SYNTAX: "Syntax",
    Category.OUTPUT: "Outputs",
}
"""Section heading for each `Category` in the Markdown matrix."""

OFFERS: Final = frozenset(
    {
        Status.AGREED,
        Status.DISAGREED,
        Status.SOLVED,
        Status.REFERENCE,
        Status.DOCUMENTED,
    }
)
"""Statuses indicating that a simulator implements the requested capability."""


def cells(document: SuiteResults, key: str) -> dict[str, Cell]:
    """Returns all executed and documented cells for row `key`."""
    found = dict(document.fixtures[key].cells)
    for name, cell in documented.cells(key).items():
        found[name] = Cell(status=Status(cell[0]), evidence=cell[1])
    return found


def render(document: SuiteResults) -> None:
    """Writes the matrix: its tables, its counts, what each lacks, and the evidence."""
    text = ["# The coverage matrix", ""]
    text.append(
        "Generated by `scripts/bench --coverage` from `coverage.json`; "
        "[README.md](README.md) explains how each cell is found."
    )
    text.append("")
    text.extend(_provenance(document))
    for category in Category:
        keys = [
            key
            for key in document.fixtures
            if document.fixtures[key].category == category
        ]
        if not keys:
            continue
        rows = []
        for key in keys:
            found = cells(document, key)
            written = [
                SHORT[found[name].status] if name in found else "?"
                for name, _ in COLUMNS
            ]
            rows.append([f"[{document.fixtures[key].name}](#{key})", *written])
        text.extend(
            [
                f"## {TITLES[category]}",
                "",
                markdown_table(["Row", *(title for _, title in COLUMNS)], rows),
                "",
            ]
        )
    text.extend(_counts(document))
    text.extend(_lacks(document))
    text.extend(_evidence(document))
    MATRIX.write_text("\n".join(text).rstrip() + "\n", encoding="utf-8")


def _provenance(document: SuiteResults) -> list[str]:
    rows = [
        [
            name,
            ran.version,
            f"`{ran.commit[:12]}`",
            str(ran.machine.get("processor") or ran.machine.get("cpu") or ""),
            f"{ran.load:.2f}",
            ran.date,
        ]
        for name, ran in document.simulators.items()
    ]
    return [
        markdown_table(
            ["Simulator", "Version", "Commit", "Machine", "Load", "Date"], rows
        ),
        "",
        (
            "The documented columns cite their sources in "
            "[documented.py](documented.py); the fixtures are not timed."
        ),
        "",
    ]


def _counts(document: SuiteResults) -> list[str]:
    rows = []
    for category in Category:
        keys = [
            key
            for key in document.fixtures
            if document.fixtures[key].category == category
        ]
        counts = [
            f"{
                sum(
                    cells(document, key).get(name, Cell(status=Status.UNKNOWN)).status
                    in OFFERS
                    for key in keys
                )
            } of {len(keys)}"
            for name, _ in COLUMNS
        ]
        rows.append([TITLES[category], *counts])
    return [
        "## Counts",
        "",
        (
            "Rows each simulator offers, right or not, by category; no total and no "
            "weights."
        ),
        "",
        markdown_table(["Category", *(title for _, title in COLUMNS)], rows),
        "",
    ]


def _lacks(document: SuiteResults) -> list[str]:
    text = ["## What each simulator lacks that another has", ""]
    for name, title in COLUMNS[:4]:
        lacking = []
        for key, line in document.fixtures.items():
            found = cells(document, key)
            own = found.get(name)
            if own is None or own.status is not Status.UNSUPPORTED:
                continue
            others = sorted(
                other
                for other, cell in found.items()
                if other != name and cell.status in OFFERS
            )
            if others:
                lacking.append(f"- [{line.name}](#{key}): {', '.join(others)}")
        text.extend([f"### {title}", "", *(lacking or ["- nothing another has"]), ""])
    return text


def _evidence(document: SuiteResults) -> list[str]:
    text = ["## Evidence", ""]
    for key, line in document.fixtures.items():
        text.extend(
            [
                f"### {key}",
                "",
                f"{line.name}; the known answer: {line.source or 'none yet'}.",
                "",
            ]
        )
        for name, cell in cells(document, key).items():
            # Simulator diagnostic strings are quoted verbatim as inline code.
            quoted = cell.status in {Status.UNSUPPORTED, Status.FAILED}
            text_evidence = cell.evidence.replace(". ", "; ")
            evidence = f"`` {text_evidence} ``" if quoted else cell.evidence
            said = f": {evidence}" if cell.evidence else ""
            text.append(f"- {name}, {cell.status}{said}")
        text.append("")
    return text


class Failed(Exception):  # noqa: N818 - a finding, not a failure of the suite
    """Exception raised when a simulator accepts a fixture but fails to complete."""


NGSPICE_REFUSALS: Final = re.compile(
    r"unknown|not (?:yet )?supported|no such|unrecognized|could not find|"
    r"undefined|not available|cannot",
    re.IGNORECASE,
)
"""Pattern matching ngspice messages that reject an unsupported feature (such as
"unknown subckt" or "not supported with option KLU"). Other errors count as failures."""

XYCE_REFUSALS: Final = re.compile(
    r"Unrecognized|not supported|unknown|undefined|No analysis specified|Netlist error|"
    r"Unable to find|not a valid|is not allowed",
    re.IGNORECASE,
)
"""Pattern matching Xyce netlist rejection messages (such as "Unrecognized dot line" or
"Netlist error"). Non-matching errors count as solver failures."""


def _said(printed: str, refusals: re.Pattern[str]) -> str:
    """Extracts unique refusal or error lines from output, else trailing lines."""
    lines = [" ".join(line.split()) for line in printed.splitlines() if line.strip()]
    said = [line for line in lines if refusals.search(line) or "rror" in line]
    return " / ".join(dict.fromkeys(said or lines[-3:]))[:600]


def _refused(printed: str, refusals: re.Pattern[str]) -> Exception:
    """Classifies simulator output as `Unsupported` on `refusals`, else `Failed`."""
    said = _said(printed, refusals)
    return Unsupported(said) if refusals.search(printed) else Failed(said)


def run_jolty(fixture: Fixture) -> Values:
    """Runs a row's fixture in jolty.

    Raises:
        Unsupported: When jolty offers no way to ask the row, or refuses the fixture.
        Documented: When jolty's coverage is documented rather than run offline.
        Failed: When jolty takes the fixture and does not finish it.
    """
    if isinstance(fixture.jolty, Unsupported | Documented):
        raise fixture.jolty
    try:
        return fixture.jolty()
    except JoltyError as error:
        raise Failed(" ".join(str(error).split())) from None


def _at(scale: np.ndarray, values: np.ndarray, read: Read) -> complex:
    """Extracts `read` by interpolating at `read.time` or indexing `read.index`."""
    if read.time is not None:
        if values.size == 1:
            return complex(values[0])
        real = np.interp(read.time, scale.real, values.real)
        imaginary = np.interp(read.time, scale.real, values.imag)
        return complex(real, imaginary)
    return complex(values[read.index])


PRINTED: Final = "printed:"
"""Prefix marking reads parsed from stdout (e.g. ``tcross = 1.69e-06``) not vectors."""


def _printed(printed: str, read: str) -> complex:
    """Parses a numeric stdout entry formatted as ``name = value`` or ``name: value``.

    Raises:
        Failed: When the expected name does not appear in `printed`.
    """
    name = read.removeprefix(PRINTED)
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    found = re.search(
        rf"(?<![\w.]){re.escape(name)}\s*[:=]\s*{number}",
        printed,
        re.IGNORECASE,
    )
    if found is None:
        message = f"printed no {name}"
        raise Failed(message)
    return complex(float(found.group(1)))


def _spice(fixture: Fixture) -> Spice:
    if isinstance(fixture.spice, Unsupported):
        raise fixture.spice
    return fixture.spice


def run_ngspice(fixture: Fixture, runner: Ngspice) -> Values:
    """Runs a row's fixture in ngspice, by `runner`, with its solver.

    Raises:
        Unsupported: When ngspice has no statement for the row, or refuses the
            fixture.
        Failed: When ngspice takes the fixture and does not finish it.
    """
    spice = _spice(fixture)
    if isinstance(spice.ngspice, Unsupported | Documented):
        raise spice.ngspice
    if ngspice.EXECUTABLE is None:
        message = "ngspice is not installed"
        raise _NotInstalledError(message)
    tight = "\n".join(part for part in (ngspice.OPTIONS, spice.options) if part)
    try:
        plots = runner.runs(
            list(spice.deck),
            spice.ngspice,
            options=tight,
            temperature=spice.temperature,
            compatibility=spice.compatibility,
            libraries=spice.libraries,
            sections=spice.sections,
        )
    except JoltyError as error:
        raise _refused(str(error), NGSPICE_REFUSALS) from None
    values: Values = {}
    for name, read in spice.reads.items():
        plot = plots[read.plot]
        if read.ngspice.startswith(PRINTED):
            values[name] = _printed(plot.printed, read.ngspice)
            continue
        vector = plot.vectors.get(read.ngspice.lower())
        if vector is None:
            message = (
                f"ngspice wrote no {read.ngspice}; its plot {plot.plot!r} holds "
                f"{', '.join(sorted(plot.vectors))}"
            )
            raise Failed(message)
        scale = next(iter(plot.vectors.values()))
        values[name] = _at(scale, vector, read)
    return values if spice.derive is None else spice.derive(values)


def _columns(reads: Mapping[str, Read]) -> list[str]:
    """Builds Xyce ``.print`` expressions, splitting complex reads into real/imag."""
    columns: list[str] = []
    for read in reads.values():
        if read.xyce is None or not read.printed or read.xyce.startswith(PRINTED):
            continue
        parts = (
            [f"{read.xyce[0]}R{read.xyce[1:]}", f"{read.xyce[0]}I{read.xyce[1:]}"]
            if read.complex
            else [read.xyce]
        )
        columns.extend(part for part in parts if part not in columns)
    return columns


def run_xyce(fixture: Fixture) -> Values:
    """Runs a row's fixture in Xyce, the program on the path, by its runner.

    The deck's lines go to Xyce as written, with the fixture's statements, its
    temperature where Xyce's thermal voltage is jolty's, and a ``.print`` of every
    read in CSV.

    Raises:
        Unsupported: When Xyce has no statement for the row, or refuses the fixture.
        Failed: When Xyce takes the fixture and does not finish it.
    """
    spice = _spice(fixture)
    if isinstance(spice.xyce, Unsupported | Documented):
        raise spice.xyce
    statements, kind = spice.xyce
    columns = _columns(spice.reads)
    if spice.thermal:
        nominal = xyce.option_temperature(ROOM)
        temperature = xyce.option_temperature(spice.temperature)
    else:
        nominal, temperature = ROOM - 273.15, spice.temperature - 273.15
    deck = [
        *spice.deck,
        *statements,
        f".options device temp={temperature!r} tnom={nominal!r}",
        ".options nonlin reltol=1e-6 abstol=1e-14",
        f".print {kind} format=csv file={xyce.VALUES} {' '.join(columns)}",
        ".end",
    ]
    if xyce.EXECUTABLE is None:
        message = "Xyce is not installed"
        raise _NotInstalledError(message)
    try:
        executed = Xyce().execute(deck, timeout=TIMEOUT)
    except JoltyError as error:
        raise Failed(str(error)) from None
    printed = executed.printed
    if executed.code != 0 or executed.values is None:
        raise _refused(printed, XYCE_REFUSALS)
    lines = list(csv.reader(executed.values.splitlines()))
    header = [cell.strip().upper() for cell in lines[0]]
    data = np.array([[float(cell) for cell in line] for line in lines[1:] if line])
    found = {column: data[:, index] for index, column in enumerate(header)}
    scale = data[:, 0]
    values: Values = {}
    for name, read in spice.reads.items():
        if read.xyce is None:
            continue
        if read.xyce.startswith(PRINTED):
            values[name] = _printed(printed, read.xyce)
            continue
        key = read.xyce.upper()
        if read.complex:
            real = found[f"{key[0]}R{key[1:]}"]
            imaginary = found[f"{key[0]}I{key[1:]}"]
            vector = real + 1j * imaginary
        else:
            vector = found[key].astype(complex)
        values[name] = _at(scale, vector, read) ** read.power
    return values if spice.derive is None else spice.derive(values)

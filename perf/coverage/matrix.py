# Copyright 2026 Maxence Faldor
"""The coverage matrix: what each simulator offers, each cell a probe that ran.

``scripts/bench --coverage`` runs it through ``perf/bench.py``. Each row of
``probes.py`` runs in jolty's and ngspice's configurations, or the simulators
``--simulator`` names, and each cell is judged against the row's known answer:

- ``agrees``: it ran, and every number it read lies within the row's tolerance of the
  known answer;
- ``disagrees``: it ran, and a number lies outside it, which the cell gives;
- ``fails``: it took the request and did not finish, with its message;
- ``not offered``: it refused the request, with its message, or has no statement that
  asks it, with where its documentation says what it offers;
- ``runs``: it ran, and the row has no known answer to judge it by yet;
- ``reference``: it is the implementation that defines the row's model, whose run is
  the known answer;
- ``documented``: a simulator that cannot run here offers it, by its manual's section;
- ``unknown``: no document found says whether it does.

Usage::

    scripts/bench --coverage                 every row, in jolty and ngspice
    scripts/bench --coverage op ac           only these rows
    scripts/bench --coverage --simulator xyce   Xyce's column, where Xyce is on the path

The cells go to ``perf/coverage/coverage.json``, a run replacing only the cells of the
rows and simulators it ran, and the matrix is rendered from it into
``perf/coverage/matrix.md``. ``perf/coverage/README.md`` says how each probe is made.
"""

import time
from collections.abc import Mapping, Sequence
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Final

import documented
import runners
from probes import Category, Documented, Known, NotOffered, Row, Values, rows
from pydantic import BaseModel, Field, JsonValue

from jolty.spice import ngspice, xyce

COVERAGE: Final = Path(__file__).resolve().parent
"""The ``perf/coverage`` directory."""

RESULTS: Final = COVERAGE / "coverage.json"
"""The cells, the matrix's source."""

MATRIX: Final = COVERAGE / "matrix.md"
"""The matrix, rendered."""


class Simulator(StrEnum):
    """A simulator configuration that runs here, named as the suites name them."""

    JOLTY = "jolty"
    NGSPICE = "ngspice"
    NGSPICE_KLU = "ngspice-klu"
    XYCE = "xyce"


class Status(StrEnum):
    """What a cell found."""

    AGREES = "agrees"
    DISAGREES = "disagrees"
    FAILS = "fails"
    ABSENT = "not offered"
    RUNS = "runs"
    REFERENCE = "reference"
    DOCUMENTED = "documented"
    UNKNOWN = "unknown"


class Cell(BaseModel):
    """One simulator's finding on one row."""

    status: Status
    evidence: str = ""
    found: dict[str, tuple[float, float]] | None = None


class Ran(BaseModel):
    """Where and how a simulator's cells were found."""

    version: str
    commit: str
    machine: dict[str, JsonValue]
    load: float
    date: str


class Line(BaseModel):
    """A row as the results hold it: its names, its answer, and each cell."""

    category: Category
    name: str
    source: str
    known: dict[str, tuple[float, float]] | None
    cells: dict[str, Cell] = Field(default_factory=dict)


class Document(BaseModel):
    """The matrix's source: every row's cells, and where each simulator's ran."""

    simulators: dict[str, Ran] = Field(default_factory=dict)
    rows: dict[str, Line] = Field(default_factory=dict)


def _pairs(values: Values) -> dict[str, tuple[float, float]]:
    return {name: (value.real, value.imag) for name, value in values.items()}


def _complex(pairs: Mapping[str, tuple[float, float]]) -> Values:
    return {name: complex(*pair) for name, pair in pairs.items()}


def judge(known: Known, found: Values, reference: Values | None) -> tuple[Status, str]:
    """Whether `found` agrees with the known answer, or `reference`'s run of it.

    Each number is judged against the largest known number of its kind in the row,
    the kind its name begins with, such as ``v`` or ``i``: within `known`'s tolerance
    of it, it agrees.
    """
    expected = known.values if known.values is not None else reference
    if expected is None:
        return Status.RUNS, ""
    missing = sorted(set(expected) - set(found))
    if missing:
        return Status.DISAGREES, f"read no {', '.join(missing)}"
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
        return Status.AGREES, f"largest error {worst:.2g} of the scale"
    return Status.DISAGREES, f"{said}, an error of {worst:.3g} of the scale"


def _number(value: complex) -> str:
    """A number in its shortest form, complex only when it has an imaginary part."""
    if value.imag == 0:
        return f"{value.real:.7g}"
    return f"{value.real:.7g}{value.imag:+.7g}j"


def probe(row: Row, simulator: Simulator, root: Path) -> Cell:
    """Runs `row`'s probe in `simulator`, its cell not yet judged.

    Its messages name the files of the repository at `root` from there.
    """
    try:
        match simulator:
            case Simulator.JOLTY:
                found = runners.jolty(row)
            case Simulator.NGSPICE:
                found = runners.run_ngspice(row)
            case Simulator.NGSPICE_KLU:
                found = runners.run_ngspice(row, ".options klu")
            case Simulator.XYCE:
                if xyce.EXECUTABLE is None:
                    return Cell(status=Status.UNKNOWN, evidence="Xyce is not installed")
                found = runners.run_xyce(row)
    except NotOffered as refused:
        return Cell(status=Status.ABSENT, evidence=_relative(str(refused), root))
    except Documented as manual:
        return Cell(status=Status.DOCUMENTED, evidence=str(manual))
    except runners.Failed as failure:
        return Cell(status=Status.FAILS, evidence=_relative(str(failure), root))
    return Cell(status=Status.RUNS, found=_pairs(found))


def _relative(message: str, root: Path) -> str:
    """A message with the paths within `root` written from it, as on any machine."""
    return message.replace(f"{root}/", "")


def rejudge(document: Document, every: Mapping[str, Row]) -> None:
    """Judges every cell that ran against its row's known answer, or its reference."""
    for key, line in document.rows.items():
        row = every.get(key)
        if row is None or row.known is None:
            continue
        reference = line.cells.get(row.known.reference or "")
        expected = (
            None
            if reference is None or reference.found is None
            else _complex(reference.found)
        )
        for simulator, cell in line.cells.items():
            if cell.found is None:
                continue
            if simulator == row.known.reference:
                cell.status, cell.evidence = Status.REFERENCE, ""
                continue
            cell.status, cell.evidence = judge(
                row.known, _complex(cell.found), expected
            )


def versions(simulator: Simulator) -> str:
    """The version of the program a simulator configuration runs."""
    match simulator:
        case Simulator.JOLTY:
            return f"jolty {metadata.version('jolty')}"
        case Simulator.NGSPICE | Simulator.NGSPICE_KLU:
            return ngspice.version() or ""
        case Simulator.XYCE:
            return xyce.version() or ""


class Context(BaseModel):
    """Where the probes run, as the speed suite reads it: its commit and machine."""

    commit: str
    machine: dict[str, JsonValue]
    load: float
    root: Path


def measure(
    names: Sequence[str],
    configurations: Sequence[str],
    context: Context,
) -> None:
    """Runs the rows `names` in the simulators named, and renders the matrix.

    Every row runs without names, and jolty's and ngspice's configurations without
    `configurations`; Xyce runs where it is on the path.

    Raises:
        SystemExit: When a row or a simulator named does not exist.
    """
    every = {row.key: row for row in rows()}
    unknown = [name for name in names if name not in every]
    if unknown:
        message = f"no row is named {', '.join(unknown)}; there are {', '.join(every)}"
        raise SystemExit(message)
    known = {simulator.value for simulator in Simulator}
    strange = [name for name in configurations if name not in known]
    if strange:
        message = (
            f"no simulator is named {', '.join(strange)}; there are {', '.join(known)}"
        )
        raise SystemExit(message)
    chosen = [Simulator(name) for name in configurations] or list(DEFAULTS)
    document = (
        Document.model_validate_json(RESULTS.read_text(encoding="utf-8"))
        if RESULTS.is_file()
        else Document()
    )
    for simulator in chosen:
        document.simulators[simulator] = Ran(
            version=versions(simulator),
            commit=context.commit,
            machine=context.machine,
            load=context.load,
            date=time.strftime("%Y-%m-%d"),
        )
    for name in names or list(every):
        row = every[name]
        line = document.rows.get(name) or Line(
            category=row.category, name=row.name, source="", known=None
        )
        line.category, line.name = row.category, row.name
        answer = row.known
        line.source = "" if answer is None else answer.source
        line.known = (
            None if answer is None or answer.values is None else _pairs(answer.values)
        )
        cells = []
        for simulator in chosen:
            cell = probe(row, simulator, context.root)
            line.cells[simulator] = cell
            cells.append(f"{simulator} {cell.status}")
        document.rows[name] = line
        print(f"{name:24} {' | '.join(cells)}", flush=True)
    document.rows = {key: document.rows[key] for key in every if key in document.rows}
    rejudge(document, every)
    RESULTS.write_text(document.model_dump_json(indent=1) + "\n", encoding="utf-8")
    render(document)
    shown = [path.relative_to(context.root) for path in (RESULTS, MATRIX)]
    print(f"wrote {shown[0]} and {shown[1]}")


DEFAULTS: Final = (Simulator.JOLTY, Simulator.NGSPICE, Simulator.NGSPICE_KLU)
"""The configurations a run probes unless it names others: those installed with
jolty's tests, ngspice's Sparse 1.3 and KLU."""


COLUMNS: Final = (
    ("jolty", "jolty"),
    ("ngspice", "ngspice 47"),
    ("ngspice-klu", "ngspice 47, KLU"),
    ("xyce", "Xyce 7.10"),
    *((name, title) for name, title in documented.TITLES.items()),
)
"""The matrix's columns: each simulator configuration, those run first."""

SHORT: Final = {
    Status.AGREES: "agrees",
    Status.DISAGREES: "**disagrees**",
    Status.FAILS: "**fails**",
    Status.ABSENT: "no",
    Status.RUNS: "runs",
    Status.REFERENCE: "reference",
    Status.DOCUMENTED: "documented",
    Status.UNKNOWN: "?",
}
"""How the table writes each status."""

TITLES: Final = {
    Category.ANALYSIS: "Analyses",
    Category.DEVICE: "Devices",
    Category.SYNTAX: "Syntax",
    Category.OUTPUT: "Outputs",
}
"""Each category's heading."""

OFFERS: Final = frozenset(
    {Status.AGREES, Status.DISAGREES, Status.RUNS, Status.REFERENCE, Status.DOCUMENTED}
)
"""The statuses of a simulator that offers what a row asks, right or not."""


def cells(document: Document, key: str) -> dict[str, Cell]:
    """Every cell of a row: those that ran, and the documented ones."""
    found = dict(document.rows[key].cells)
    for name, cell in documented.cells(key).items():
        found[name] = Cell(status=Status(cell[0]), evidence=cell[1])
    return found


def render(document: Document) -> None:
    """Writes the matrix: its tables, its counts, what each lacks, and the evidence."""
    text = ["# The coverage matrix", ""]
    text.append(
        "Generated by `scripts/bench --coverage` from `coverage.json`; "
        "[README.md](README.md) says how each cell is found."
    )
    text.append("")
    text.extend(_provenance(document))
    for category in Category:
        keys = [key for key in document.rows if document.rows[key].category == category]
        if not keys:
            continue
        text.extend([f"## {TITLES[category]}", ""])
        text.append("| Row | " + " | ".join(title for _, title in COLUMNS) + " |")
        text.append("| --- |" + " --- |" * len(COLUMNS))
        for key in keys:
            found = cells(document, key)
            written = [
                SHORT[found[name].status] if name in found else "?"
                for name, _ in COLUMNS
            ]
            text.append(
                f"| [{document.rows[key].name}](#{key}) | " + " | ".join(written) + " |"
            )
        text.append("")
    text.extend(_counts(document))
    text.extend(_lacks(document))
    text.extend(_evidence(document))
    MATRIX.write_text("\n".join(text).rstrip() + "\n", encoding="utf-8")


def _provenance(document: Document) -> list[str]:
    text = ["| Simulator | Version | Commit | Machine | Load | Date |"]
    text.append("| --- | --- | --- | --- | --- | --- |")
    for name, ran in document.simulators.items():
        machine = str(ran.machine.get("processor") or ran.machine.get("cpu") or "")
        text.append(
            f"| {name} | {ran.version} | `{ran.commit[:12]}` | {machine} | "
            f"{ran.load:.2f} | {ran.date} |"
        )
    text.append("")
    text.append(
        "The documented columns cite their sources in "
        "[documented.py](documented.py); the probes are not timed."
    )
    text.append("")
    return text


def _counts(document: Document) -> list[str]:
    text = ["## Counts", ""]
    text.append(
        "Rows each simulator offers, right or not, by category; no total and no "
        "weights."
    )
    text.append("")
    text.append("| Category | " + " | ".join(title for _, title in COLUMNS) + " |")
    text.append("| --- |" + " --- |" * len(COLUMNS))
    for category in Category:
        keys = [key for key in document.rows if document.rows[key].category == category]
        counts = []
        for name, _ in COLUMNS:
            offered = sum(
                cells(document, key).get(name, Cell(status=Status.UNKNOWN)).status
                in OFFERS
                for key in keys
            )
            counts.append(f"{offered} of {len(keys)}")
        text.append(f"| {TITLES[category]} | " + " | ".join(counts) + " |")
    text.append("")
    return text


def _lacks(document: Document) -> list[str]:
    text = ["## What each simulator lacks that another has", ""]
    for name, title in COLUMNS[:4]:
        lacking = []
        for key, line in document.rows.items():
            found = cells(document, key)
            own = found.get(name)
            if own is None or own.status is not Status.ABSENT:
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


def _evidence(document: Document) -> list[str]:
    text = ["## Evidence", ""]
    for key, line in document.rows.items():
        text.extend(
            [
                f"### {key}",
                "",
                f"{line.name}; the known answer: {line.source or 'none yet'}.",
                "",
            ]
        )
        for name, cell in cells(document, key).items():
            # A simulator's own words are quoted as code, as it wrote them.
            quoted = cell.status in {Status.ABSENT, Status.FAILS}
            evidence = f"`` {cell.evidence} ``" if quoted else cell.evidence
            said = f": {evidence}" if cell.evidence else ""
            text.append(f"- {name}, {cell.status}{said}")
        text.append("")
    return text

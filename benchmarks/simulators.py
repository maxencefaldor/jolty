# Copyright 2026 Maxence Faldor
"""Simulator configurations, runners, and output parsers shared by benchmark suites.

Each `Simulator` identifies one program and solver configuration: jolty, ngspice with
Sparse 1.3 or KLU, and Xyce in serial or parallel MPI mode. Runners (`Jolty`, `Ngspice`,
and `Xyce`) wrap `jolty.simulate`, `jolty.spice.ngspice`, and `jolty.spice.xyce`, expose
`Runner` metadata, and execute decks uniformly across suites.

Additional helpers export circuit netlists (`netlist`) and SPICE analysis commands
(`command`). Output parsers extract ngspice ``rusage`` statistics (`RUSAGE`),
deduplicated diagnostics (`trouble`), Xyce transient step counts (`XYCE_STEPS`),
progress
(`XYCE_PROGRESS`), and failure summaries (`stopped`).
"""

import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Final, Literal, Protocol, Self, Unpack

from jolty import ir, results
from jolty.plan import (
    Ac,
    Analysis,
    DcSweep,
    Method,
    Noise,
    OperatingPoint,
    Options,
    PoleZero,
    Transient,
)
from jolty.simulate import Simulation
from jolty.spice import export, ngspice, xyce
from jolty.units import FREQUENCY, TIME, VOLTAGE, Quantity


class Simulator(StrEnum):
    """Simulator identifier pairing a program with a solver or execution mode."""

    JOLTY = "jolty"
    NGSPICE_SPARSE = "ngspice-sparse"
    NGSPICE_KLU = "ngspice-klu"
    XYCE_SERIAL = "xyce-serial"
    XYCE_MPI = "xyce-mpi"


class Unsupported(Exception):  # noqa: N818 - a finding, not a failure of the suite
    """Exception raised when a simulator lacks a feature requested by a fixture."""


class Runner(Protocol):
    """Protocol for simulator runners exposing simulator identity and version."""

    @property
    def simulator(self) -> Simulator:
        """Simulator configuration executed by the runner."""
        ...

    def version(self) -> str | None:
        """Returns the installed simulator version string, or None when unavailable."""
        ...


@dataclass(frozen=True, slots=True)
class Jolty:
    """In-process runner for the jolty simulation engine."""

    @property
    def simulator(self) -> Simulator:
        """Simulator identifier for jolty."""
        return Simulator.JOLTY

    def version(self) -> str:
        """Returns the installed jolty package version string."""
        return f"jolty {metadata.version('jolty')}"

    def run(
        self, circuit: ir.Circuit, analyses: Mapping[str, Analysis], options: Options
    ) -> Analyzed:
        """Elaborates `circuit` and runs `analyses` sequentially by name.

        Raises:
            JoltyError: When the circuit fails to elaborate or an analysis fails.
        """
        simulation = Simulation(circuit)
        found: dict[str, results.AnalysisResult] = {}
        analysis_seconds = 0.0
        for name, analysis in analyses.items():
            started = time.perf_counter()
            found[name] = simulation.run(analysis, options)
            analysis_seconds += time.perf_counter() - started
        return Analyzed(found, analysis_seconds)


@dataclass(frozen=True, slots=True)
class Analyzed:
    """Named jolty analysis results paired with total analysis wall time."""

    found: dict[str, results.AnalysisResult]
    analysis_seconds: float
    """Wall time inside ``Simulation.run`` calls, including Python result marshaling
    and excluding circuit elaboration."""


KLU: Final = ".options klu"
"""SPICE options directive selecting the KLU direct linear solver in ngspice."""

SOLVERS: Final = {
    Simulator.NGSPICE_SPARSE: "Using SPARSE 1.3 as Direct Linear Solver",
    Simulator.NGSPICE_KLU: "Using KLU as Direct Linear Solver",
}
"""Expected ngspice stdout lines confirming the active direct linear solver."""


@dataclass(frozen=True, slots=True)
class Ngspice:
    """Batch-mode ngspice runner using either Sparse 1.3 (default) or KLU."""

    simulator: Literal[Simulator.NGSPICE_SPARSE, Simulator.NGSPICE_KLU] = (
        Simulator.NGSPICE_SPARSE
    )

    def version(self) -> str | None:
        """Returns the ngspice version string, or None when not installed."""
        return ngspice.version()

    @property
    def options(self) -> tuple[str, ...]:
        """SPICE option lines selecting the configured solver (empty for Sparse 1.3)."""
        return (KLU,) if self.simulator is Simulator.NGSPICE_KLU else ()

    @property
    def solver(self) -> str:
        """Expected ngspice stdout line identifying the active direct linear solver."""
        return SOLVERS[self.simulator]

    def execute(
        self,
        deck: Sequence[str],
        control: Sequence[str],
        *,
        timeout: float | None = ngspice.TIMEOUT,
    ) -> ngspice.Executed:
        """Runs `control` commands on netlist `deck` in one ngspice invocation.

        Appends solver options and wraps `control` in ``.control`` before calling
        `jolty.spice.ngspice.execute`. Enforces `timeout` in seconds when provided, or
        waits without a timeout when None.

        Raises:
            JoltyError: As `jolty.spice.ngspice.execute` does.
        """
        full_deck = [*deck, *self.options, ".control", *control, ".endc", ".end"]
        return ngspice.execute(full_deck, timeout=timeout)

    def runs(
        self,
        deck: list[str],
        analyses: Sequence[str],
        **running: Unpack[ngspice.Running],
    ) -> tuple[ngspice.Result, ...]:
        """Runs `analyses` sequentially in one ngspice process and returns their plots.

        Delegates to `jolty.spice.ngspice.runs`, appending solver options after
        any options provided in `running` (or `ngspice.OPTIONS` by default).

        Raises:
            JoltyError: As `jolty.spice.ngspice.runs` does.
        """
        options = "\n".join([running.get("options", ngspice.OPTIONS), *self.options])
        return ngspice.runs(deck, analyses, **(running | {"options": options}))


@dataclass(frozen=True, slots=True)
class Xyce:
    """Xyce runner for either the serial binary on PATH or an MPI parallel build."""

    simulator: Literal[Simulator.XYCE_SERIAL, Simulator.XYCE_MPI] = (
        Simulator.XYCE_SERIAL
    )
    launch: tuple[str, ...] | None = None
    """Command prefix launching Xyce, defaulting to serial Xyce on PATH when None."""

    @classmethod
    def from_parallel_build(cls, build: Path, ranks: int) -> Self:
        """Constructs an MPI Xyce runner for `build` across `ranks` MPI ranks."""
        return cls(Simulator.XYCE_MPI, ("mpirun", "-np", str(ranks), str(build)))

    def version(self) -> str | None:
        """Returns the serial Xyce version string from PATH, or None when absent."""
        return xyce.version()

    def run(
        self,
        deck: list[str],
        analysis: str,
        *,
        printed: Sequence[str] | None = None,
        options: Sequence[str] = xyce.OPTIONS,
        timeout: float | None = xyce.TIMEOUT,
    ) -> xyce.Result:
        """Runs a single analysis of `deck` via `jolty.spice.xyce.run`.

        Raises:
            JoltyError: As `jolty.spice.xyce.run` does.
        """
        return xyce.run(
            deck,
            analysis,
            printed=printed,
            options=options,
            launch=self.launch,
            timeout=timeout,
        )

    def execute(
        self, deck: Sequence[str], *, timeout: float | None = xyce.TIMEOUT
    ) -> xyce.Executed:
        """Executes netlist lines `deck` via `jolty.spice.xyce.execute`.

        Raises:
            JoltyError: As `jolty.spice.xyce.execute` does.
        """
        return xyce.execute(deck, launch=self.launch, timeout=timeout)


def runner(simulator: Simulator) -> Jolty | Ngspice | Xyce:
    """Returns a runner instance for `simulator`, excluding `Simulator.XYCE_MPI`.

    Call `Xyce.from_parallel_build` instead for MPI Xyce, which requires an explicit
    binary path and rank count.

    Raises:
        ValueError: When `simulator` is `Simulator.XYCE_MPI`.
    """
    match simulator:
        case Simulator.JOLTY:
            return Jolty()
        case Simulator.NGSPICE_SPARSE | Simulator.NGSPICE_KLU:
            return Ngspice(simulator)
        case Simulator.XYCE_SERIAL:
            return Xyce()
        case Simulator.XYCE_MPI:
            message = "Xyce's parallel build runs from its path and its ranks"
            raise ValueError(message)


def netlist(
    circuit: ir.Circuit,
    analyses: Iterable[Analysis],
    *,
    dialect: export.Dialect = export.NGSPICE,
) -> list[str]:
    """Exports `circuit` as a SPICE netlist configured for `analyses` in `dialect`.

    Applies the small-signal excitation level when `analyses` includes an AC or noise
    run, and enables dynamic memristor updates when `analyses` includes a transient
    run. Defaults to the ngspice dialect.
    """
    chosen = list(analyses)
    excited = next(
        (each.source for each in chosen if isinstance(each, Ac | Noise)), None
    )
    transient = any(isinstance(each, Transient) for each in chosen)
    return export.netlist(
        circuit, excited=excited, transient=transient, dialect=dialect
    )


def dc_command(element: str, start: float, stop: float, step: float) -> str:
    """Formats a SPICE DC sweep command: ``dc <element> <start> <stop> <step>``."""
    return f"dc {element} {start!r} {stop!r} {step!r}"


def tran_command(step: float, stop: float, max_step: float | None = None) -> str:
    """Formats a SPICE transient command: ``tran <step> <stop> 0 <max_step>``."""
    longest = step if max_step is None else max_step
    return f"tran {step!r} {stop!r} 0 {longest!r}"


def ngspice_element(flat: ir.Circuit, path: str) -> str:
    """Returns the SPICE element name for instance `path` in flattened `flat`.

    Raises:
        KeyError: When `flat` contains no instance at `path`.
    """
    top = flat.components[flat.top]
    return export.element(_instance(top, path.replace("/", "_")))


def ngspice_method_flags(method: Method) -> tuple[str, ...]:
    """Returns ngspice ``.options`` tokens for integration `method`, empty for trap."""
    if method == "gear2":
        return ("method=gear", "maxord=2")
    if method == "backward_euler":
        return ("method=gear", "maxord=1")
    return ()


def ngspice_options_line(flags: Sequence[str], *, fallback: str = "") -> str:
    """Joins ``.options`` `flags` into an option line, or `fallback` when empty."""
    return f".options {' '.join(flags)}" if flags else fallback


def command(analysis: Analysis, flat: ir.Circuit) -> str:
    """Translates `analysis` into an ngspice control command (add a dot for Xyce).

    Resolves swept, noise, and pole-zero names against flattened circuit `flat`.
    Uses `max_step` (or one fiftieth of `stop`, matching SPICE's default) for transient
    step and output intervals, and formats AC or noise grids in points per decade.

    Raises:
        TypeError: When `analysis` is not supported by external-simulator suites.
    """
    top = flat.components[flat.top]
    match analysis:
        case OperatingPoint():
            return "op"
        case DcSweep(source=source):
            given = analysis.numbers(VOLTAGE)
            step = (given[-1] - given[0]) / (len(given) - 1)
            return dc_command(ngspice_element(flat, source), given[0], given[-1], step)
        case Ac(frequencies=frequencies):
            return f"ac {_decades(frequencies)}"
        case Noise(frequencies=frequencies, output=output, source=source):
            element = ngspice_element(flat, source or "")
            return f"noise v({output}) {element} {_decades(frequencies)}"
        case Transient(stop=stop, max_step=max_step):
            end = stop.to(TIME, "a stop")
            longest = end / 50 if max_step is None else max_step.to(TIME, "a step")
            return tran_command(longest, end)
        case PoleZero(source=str(source), output=str(output)):
            positive = _instance(top, source).terminals["positive"]
            ground = next(
                port.name for port in top.ports if port.role is ir.Role.GROUND
            )
            node = "0" if positive == ground else positive
            return f"pz {node} 0 {output} 0 cur pol"
    message = f"no suite runs {type(analysis).__name__} in another simulator"
    raise TypeError(message)


def _decades(frequencies: Sequence[Quantity]) -> str:
    """Formats logarithmic `frequencies` as a SPICE grid (``dec N first last``)."""
    first = frequencies[0].to(FREQUENCY, "a frequency")
    last = frequencies[-1].to(FREQUENCY, "a frequency")
    per_decade = round((len(frequencies) - 1) / math.log10(last / first))
    return f"dec {per_decade} {first!r} {last!r}"


def _instance(top: ir.Component, name: str) -> ir.Instance:
    """Returns instance `name` from top component `top` of a flattened circuit.

    Raises:
        KeyError: When `top` contains no instance matching `name`.
    """
    found = top.instance(name)
    if found is None:
        message = f"the circuit has no instance {name!r}"
        raise KeyError(message)
    return found


RUSAGE: Final = {
    "analysis_seconds": re.compile(
        r"^Total analysis time \(seconds\) = (\S+)", re.MULTILINE
    ),
    "iterations": re.compile(r"^Total iterations = (\d+)", re.MULTILINE),
    "transient iterations": re.compile(r"^Transient iterations = (\d+)", re.MULTILINE),
    "accepted": re.compile(r"^Accepted timepoints = (\d+)", re.MULTILINE),
    "rejected": re.compile(r"^Rejected timepoints = (\d+)", re.MULTILINE),
}
"""Patterns parsing cumulative ngspice ``rusage all`` metrics (analysis time, total and
transient Newton iterations, and accepted or rejected time points)."""

TROUBLE: Final = re.compile(
    r"error|warning|abort|too small|singular|converge|fail|trouble", re.IGNORECASE
)
"""Pattern matching simulator diagnostic lines for errors or convergence issues."""


NUMBER: Final = re.compile(r"\b\d[\d.]*(?:e[-+]?\d+)?\b", re.IGNORECASE)
"""Pattern matching numeric literals in simulator output outside identifier names."""


def trouble(printed: str) -> str:
    """Extracts unique diagnostic lines from simulator output `printed`.

    Deduplicates lines differing only in numbers (such as repeated netlist warnings)
    by keeping the first occurrence of each pattern.
    """
    once: dict[str, str] = {}
    for line in printed.splitlines():
        if TROUBLE.search(line):
            once.setdefault(NUMBER.sub("#", line.strip()), line.strip())
    return " | ".join(once.values())[:2000]


XYCE_STEPS: Final = {
    "accepted": re.compile(r"^\s*Successful Step\s+(\d+)", re.MULTILINE),
    "rejected": re.compile(r"^\s*Failed Steps\s+(\d+)", re.MULTILINE),
}
"""Patterns extracting accepted and rejected transient step counts from Xyce."""

XYCE_PROGRESS: Final = re.compile(r"Percent complete: (\S+) %")
"""Regular expression extracting transient completion percentages reported by Xyce."""


def stopped(message: str) -> str:
    """Summarizes the failure cause and last Xyce progress from error `message`.

    Combines the first line of `message`, unique diagnostic lines from subsequent lines,
    and the final Xyce transient completion percentage when present.
    """
    first, *rest = message.splitlines() or [""]
    said = [first.removesuffix(":"), *filter(None, [trouble("\n".join(rest))])]
    reached = XYCE_PROGRESS.findall(message)
    if reached:
        said.append(f"{float(reached[-1]):.3g}% of its transient done")
    return "; ".join(said)

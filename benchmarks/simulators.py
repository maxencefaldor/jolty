# Copyright 2026 Maxence Faldor
"""The simulators the suites run their fixtures in, and one runner for each.

A simulator is one program in one configuration (`Simulator`): jolty; ngspice with
its default direct solver, Sparse 1.3, or with KLU; and Xyce, serially or its parallel
build on MPI ranks. Each has one runner, `Jolty`, `Ngspice` or `Xyce`, built on
`jolty.simulate`, `jolty.spice.ngspice` and `jolty.spice.xyce`, which names its
simulator, states its version (`Runner`), and runs a deck in it the one way every
suite does; each suite reads of the run what it measures: the speed suite its times,
the corpus its outcome, the matrix its cell, the precision suite its points.

Beside them, what the suites write of a plan's analyses for another simulator, the
netlist they need (`netlist`) and each analysis as ngspice's command (`command`); what
they read of what ngspice printed, its ``rusage`` statistics (`RUSAGE`) and the lines
that say what went wrong (`trouble`), and of what Xyce printed, its transient's steps
(`XYCE_STEPS`) and how far it came (`XYCE_PROGRESS`); and what stopped a run that
failed (`stopped`).
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
    """A simulator: one program in one configuration, as every suite names it."""

    JOLTY = "jolty"
    NGSPICE_SPARSE = "ngspice-sparse"
    NGSPICE_KLU = "ngspice-klu"
    XYCE_SERIAL = "xyce-serial"
    XYCE_MPI = "xyce-mpi"


class Unsupported(Exception):  # noqa: N818 - a finding, not a failure of the suite
    """A simulator does not offer what a fixture asks, for the reason given."""


class Runner(Protocol):
    """What runs fixtures in one simulator: it names it, and states its version."""

    @property
    def simulator(self) -> Simulator:
        """The simulator it runs."""
        ...

    def version(self) -> str | None:
        """The version and build of the program, or None when it is not installed."""
        ...


@dataclass(frozen=True, slots=True)
class Jolty:
    """jolty's engine, in this process."""

    @property
    def simulator(self) -> Simulator:
        """The simulator it runs, jolty."""
        return Simulator.JOLTY

    def version(self) -> str:
        """The engine's version."""
        return f"jolty {metadata.version('jolty')}"

    def run(
        self, circuit: ir.Circuit, analyses: Mapping[str, Analysis], options: Options
    ) -> Analysed:
        """Elaborates `circuit`, and runs the `analyses` on it in turn, by name.

        Raises:
            JoltyError: When the circuit does not elaborate, or an analysis fails.
        """
        simulation = Simulation(circuit)
        found: dict[str, results.AnalysisResult] = {}
        analysis_seconds = 0.0
        for name, analysis in analyses.items():
            started = time.perf_counter()
            found[name] = simulation.run(analysis, options)
            analysis_seconds += time.perf_counter() - started
        return Analysed(found, analysis_seconds)


@dataclass(frozen=True, slots=True)
class Analysed:
    """What jolty found of a circuit's analyses, by name, and how long they took."""

    found: dict[str, results.AnalysisResult]
    analysis_seconds: float
    """The wall time of the analyses alone, each ``Simulation.run``: the engine's work
    and its results made in Python, the circuit's elaboration left out."""


KLU: Final = ".options klu"
"""The options line that makes ngspice solve by KLU rather than by Sparse 1.3."""

SOLVERS: Final = {
    Simulator.NGSPICE_SPARSE: "Using SPARSE 1.3 as Direct Linear Solver",
    Simulator.NGSPICE_KLU: "Using KLU as Direct Linear Solver",
}
"""What ngspice prints of the direct solver each of its configurations runs, once an
analysis."""


@dataclass(frozen=True, slots=True)
class Ngspice:
    """ngspice in batch mode, with Sparse 1.3, its default direct solver, or KLU."""

    simulator: Literal[Simulator.NGSPICE_SPARSE, Simulator.NGSPICE_KLU] = (
        Simulator.NGSPICE_SPARSE
    )

    def version(self) -> str | None:
        """The version and build of ngspice, or None when it is not installed."""
        return ngspice.version()

    @property
    def options(self) -> tuple[str, ...]:
        """The options lines that choose its solver: none for Sparse 1.3."""
        return (KLU,) if self.simulator is Simulator.NGSPICE_KLU else ()

    @property
    def solver(self) -> str:
        """What ngspice prints of the solver it runs, once an analysis."""
        return SOLVERS[self.simulator]

    def execute(
        self,
        lines: Sequence[str],
        control: Sequence[str],
        *,
        timeout: float | None = ngspice.TIMEOUT,
    ) -> ngspice.Executed:
        """Runs the `control` commands on the netlist `lines`, in one ngspice.

        The deck is the netlist, then its solver's options, then a control section of
        the commands; it runs by `jolty.spice.ngspice.execute`, for at most `timeout`
        seconds, or as long as it takes if None.

        Raises:
            JoltyError: As `jolty.spice.ngspice.execute` does.
        """
        deck = [*lines, *self.options, ".control", *control, ".endc", ".end"]
        return ngspice.execute(deck, timeout=timeout)

    def runs(
        self,
        lines: list[str],
        analyses: Sequence[str],
        **running: Unpack[ngspice.Running],
    ) -> tuple[ngspice.Result, ...]:
        """The plot of each of `analyses`, run in turn in one ngspice.

        They run by `jolty.spice.ngspice.runs`, its solver's options after the options
        `running` gives, or the tight ones.

        Raises:
            JoltyError: As `jolty.spice.ngspice.runs` does.
        """
        options = "\n".join([running.get("options", ngspice.OPTIONS), *self.options])
        return ngspice.runs(lines, analyses, **(running | {"options": options}))


@dataclass(frozen=True, slots=True)
class Xyce:
    """Xyce, serially, the program on the path, or its parallel build on MPI ranks."""

    simulator: Literal[Simulator.XYCE_SERIAL, Simulator.XYCE_MPI] = (
        Simulator.XYCE_SERIAL
    )
    launch: tuple[str, ...] | None = None
    """The command that starts it, the serial Xyce on the path unless given."""

    @classmethod
    def from_parallel_build(cls, build: Path, ranks: int) -> Self:
        """The parallel build at `build`, on `ranks` MPI ranks."""
        return cls(Simulator.XYCE_MPI, ("mpirun", "-np", str(ranks), str(build)))

    def version(self) -> str | None:
        """Xyce's version, as the serial Xyce on the path states it, or None."""
        return xyce.version()

    def run(
        self,
        lines: list[str],
        analysis: str,
        *,
        printed: Sequence[str] | None = None,
        options: Sequence[str] = xyce.OPTIONS,
        timeout: float | None = xyce.TIMEOUT,
    ) -> xyce.Result:
        """Runs one analysis of the deck `lines` by `jolty.spice.xyce.run`.

        Raises:
            JoltyError: As `jolty.spice.xyce.run` does.
        """
        return xyce.run(
            lines,
            analysis,
            printed=printed,
            options=options,
            launch=self.launch,
            timeout=timeout,
        )

    def execute(
        self, deck: Sequence[str], *, timeout: float | None = xyce.TIMEOUT
    ) -> xyce.Executed:
        """Runs the lines `deck` by `jolty.spice.xyce.execute`.

        Raises:
            JoltyError: As `jolty.spice.xyce.execute` does.
        """
        return xyce.execute(deck, launch=self.launch, timeout=timeout)


def runner(simulator: Simulator) -> Jolty | Ngspice | Xyce:
    """The runner of `simulator`, but Xyce's parallel build's.

    `Xyce.from_parallel_build` makes that one, from its path and its ranks.

    Raises:
        ValueError: For Xyce's parallel build, whose path and ranks it needs.
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
    """The netlist of `circuit` that another simulator runs the `analyses` on.

    The source a small-signal response or a noise analysis excites carries its
    small-signal level, and a transient's netlist moves its memristors' states, in the
    `dialect` of the simulator, ngspice's unless given.
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
    """The SPICE element name for the instance at `path` in flattened circuit `flat`.

    Raises:
        KeyError: When `flat` holds no instance at `path`.
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
    """`analysis` as ngspice's command in a control section.

    Xyce's statement is the same after a dot. `flat` is the circuit flattened, whose
    instances a sweep, a noise analysis or a pole-zero analysis names. A transient's
    longest step is its plan's, or a fiftieth of it, SPICE's default, and its output's
    interval the same; a response runs at its frequencies a decade.

    Raises:
        TypeError: When the analysis is none that the suites run in another simulator.
    """
    top = flat.components[flat.top]
    match analysis:
        case OperatingPoint():
            return "op"
        case DcSweep(source=source, values=levels):
            given = [level.to(VOLTAGE, "a level") for level in levels]
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
    """Frequencies spaced evenly by decade as SPICE gives them: ``dec N first last``."""
    first = frequencies[0].to(FREQUENCY, "a frequency")
    last = frequencies[-1].to(FREQUENCY, "a frequency")
    per_decade = round((len(frequencies) - 1) / math.log10(last / first))
    return f"dec {per_decade} {first!r} {last!r}"


def _instance(top: ir.Component, id: str) -> ir.Instance:
    """The instance `id` of the flattened circuit's `top`.

    Raises:
        KeyError: When it has none.
    """
    found = top.instance(id)
    if found is None:
        message = f"the circuit has no instance {id!r}"
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
"""What ngspice's ``rusage all`` prints, summed over the analyses it has run: its
analysis time, which its clock measures as wall time, its Newton iterations, those of
its transients, and the time points it accepted and rejected."""

TROUBLE: Final = re.compile(
    r"error|warning|abort|too small|singular|converge|fail|trouble", re.IGNORECASE
)
"""What marks the lines of a simulator's output that say what went wrong."""


NUMBER: Final = re.compile(r"\b\d[\d.]*(?:e[-+]?\d+)?\b", re.IGNORECASE)
"""A number in a line of what a simulator printed, such as a line of its netlist or a
time, but not a digit within a name."""


def trouble(printed: str) -> str:
    """The lines of what a simulator `printed` that say what went wrong, once each.

    Lines that differ only in their numbers, such as a warning at each of many lines of
    the netlist, are one, and the first stands for them.
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
"""What the table of Xyce's timings states of a transient's steps, accepted and
rejected."""

XYCE_PROGRESS: Final = re.compile(r"Percent complete: (\S+) %")
"""What Xyce prints of how far its transient has come."""


def stopped(message: str) -> str:
    """What stopped a run, from the `message` of its error: what went wrong, how far.

    The first line of the message, then each line after it that says what went wrong,
    and the last of Xyce's reports of how far its transient came.
    """
    first, *rest = message.splitlines() or [""]
    said = [first.removesuffix(":"), *filter(None, [trouble("\n".join(rest))])]
    reached = XYCE_PROGRESS.findall(message)
    if reached:
        said.append(f"{float(reached[-1]):.3g}% of its transient done")
    return "; ".join(said)

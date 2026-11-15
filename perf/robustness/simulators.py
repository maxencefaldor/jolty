# Copyright 2026 Maxence Faldor
"""The simulators that run the corpus, each run in a process of its own.

A simulator is a named configuration of one program (`corpus.Simulator`): ``jolty``, the
engine at its defaults, and ``ngspice-sparse``, ngspice in batch mode at its defaults,
whose direct solver is Sparse 1.3. Each runs a case in a child process that a timeout
stops, and returns what it found, the voltages the case compares, its time and its
Newton iterations, or its message.

Run as a script, this module is the child that runs jolty: ``simulators.py run CASE``
runs a case and ``simulators.py reference CASE`` finds its reference, every DC solution
jolty finds by deflation at tight tolerances; each prints one line of JSON.
"""

import dataclasses
import json
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from corpus import (
    ROBUSTNESS,
    Built,
    Case,
    Completes,
    Impasse,
    Oscillates,
    Point,
    Reference,
    Run,
    Settles,
    Simulator,
    Solution,
    Solutions,
    Sweep,
    Tran,
    Unsupported,
    cases,
)

from jolty import _core, ir
from jolty.errors import JoltyError
from jolty.plan import DcSweep, OperatingPoint, OperatingPoints, Options, Transient
from jolty.simulate import Simulation
from jolty.spice import ngspice
from jolty.units import CURRENT, VOLTAGE, K, Quantity, s

TIGHT: Final = Options(
    relative_tolerance=1e-6,
    voltage_tolerance=ngspice.VOLTAGE_TOLERANCE,
    current_tolerance=ngspice.CURRENT_TOLERANCE,
    charge_tolerance=1e-18,
)
"""The tolerances the reference's DC solutions are found at: the differential tests'
(``tests/differential.py``'s ``TIGHT``), as tight as ngspice's tightest."""

_REACHED: Final = re.compile(r"at t = (\S+) s")
"""Where jolty's message says a transient stopped: its step fell below the smallest
``at t = ... s`` (``SimulationError::StepTooSmall``)."""


def every() -> dict[str, Simulator]:
    """Every simulator the corpus knows, by name, installed or not."""
    known: list[Simulator] = [Jolty(), Ngspice()]
    return {simulator.name: simulator for simulator in known}


def _child(mode: str, case: Case, timeout: float) -> tuple[dict[str, Any], float]:
    """What this module, run as a script in `mode` on `case`, printed, and its time.

    Raises:
        subprocess.TimeoutExpired: When it runs past `timeout`.
        JoltyError: When it prints no result, as when it crashes.
    """
    started = time.perf_counter()
    done = subprocess.run(  # noqa: S603 - this module, run on the case named
        [sys.executable, str(ROBUSTNESS / "simulators.py"), mode, case.name],
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )
    seconds = time.perf_counter() - started
    lines = done.stdout.strip().splitlines()
    if done.returncode != 0 or not lines:
        message = f"exited with {done.returncode}: {done.stderr.strip()[-2000:]}"
        raise JoltyError(message)
    return json.loads(lines[-1]), seconds


class Jolty:
    """jolty's engine at its defaults, each case in a Python process of its own."""

    name = "jolty"

    def version(self) -> str:
        """The engine's version."""
        from importlib.metadata import version  # noqa: PLC0415 - read once

        return f"jolty {version('jolty')}"

    def run(self, case: Case, timeout: float) -> Run:
        """Runs `case` in a child process, stopped after `timeout` seconds."""
        try:
            document, seconds = _child("run", case, timeout)
        except subprocess.TimeoutExpired:
            return Run(
                completed=False, message=f"ran past {timeout:g} s", timed_out=True
            )
        except JoltyError as error:
            return Run(completed=False, message=f"the process failed: {error}")
        return dataclasses.replace(Run.of(document), seconds=seconds)


def reference(case: Case, timeout: float) -> Reference:
    """The reference `case`'s runs are judged against, found by jolty in a child.

    Every DC solution jolty finds by deflation at tight tolerances: at the operating
    point, at each level of a sweep, or of the circuit settled at a transient's end;
    none for a check that needs none.
    """
    if isinstance(case.check, Impasse | Completes):
        return Reference()
    try:
        document, _ = _child("reference", case, timeout)
    except subprocess.TimeoutExpired:
        return Reference(message=f"the search for DC solutions ran past {timeout:g} s")
    except JoltyError as error:
        return Reference(message=f"the search for DC solutions failed: {error}")
    points = document.get("points")
    if not isinstance(points, list):
        return Reference(message=str(document.get("message", "")))
    return Reference(tuple(tuple(point) for point in points))


def _found(case: Case) -> Run:
    """Runs `case` in jolty, in this process: what the child of `Jolty.run` does."""
    try:
        built = case.problem.jolty()
        simulation = Simulation(built.circuit, built.temperature * K)
    except (Unsupported, JoltyError) as error:
        return Run(completed=False, message=str(error), unsupported=True)
    diagnostics = tuple(
        f"{diagnostic.rule}: {' '.join(diagnostic.message.split())}"
        for diagnostic in simulation.check()
    )
    started = time.perf_counter()
    try:
        run = _analysis(case, built, simulation)
    except JoltyError as error:
        message = " ".join(str(error).split())
        reached = _REACHED.search(message)
        return Run(
            completed=False,
            message=message,
            analysis_seconds=time.perf_counter() - started,
            reached=float(reached.group(1)) if reached else None,
            diagnostics=diagnostics,
        )
    return dataclasses.replace(
        run, analysis_seconds=time.perf_counter() - started, diagnostics=diagnostics
    )


def _analysis(case: Case, built: Built, simulation: Simulation) -> Run:
    """Runs `case`'s analysis on `simulation`, and reads the voltages it compares.

    Raises:
        JoltyError: When the analysis fails.
    """
    options = built.options
    match case.analysis:
        case Point():
            point = simulation.run(OperatingPoint(), options)
            values = {net: point.voltage(net) for net in built.nets}
            return Run(completed=True, strategy=point.strategy, values=(values,))
        case Sweep(source=source) as sweep:
            unit = _level(built, source)
            levels = [level * unit for level in sweep.levels]
            swept = simulation.run(DcSweep(source, levels), options)
            columns = {net: swept.voltage(net) for net in built.nets}
            rows = tuple(
                {net: float(column[row]) for net, column in columns.items()}
                for row in range(len(levels))
            )
            return Run(completed=True, values=rows)
        case Tran(stop=stop, method=method) as transient:
            waveforms = simulation.run(
                Transient(stop * s, max_step=transient.longest * s, method=method),
                options,
            )
            end = {net: float(waveforms.voltage(net)[-1]) for net in built.nets}
            work = waveforms.statistics.work
            return Run(
                completed=True,
                iterations=None if work is None else work.iterations,
                strategy=waveforms.statistics.initial,
                values=(end,),
                reached=float(waveforms.times[-1]),
            )


def _level(built: Built, source: str) -> Quantity:
    """The unit of the levels of the source `source`: a volt, or an ampere.

    Raises:
        Unsupported: When `source` is no independent source of the circuit's top.
    """
    top = built.circuit.components[built.circuit.top]
    instance = top.instance(source)
    model = (
        instance.of.device
        if instance is not None and isinstance(instance.of, ir.DeviceOf)
        else None
    )
    if model == "voltage_source":
        return Quantity(1.0, VOLTAGE)
    if model == "current_source":
        return Quantity(1.0, CURRENT)
    message = f"`{source}` is no independent source of the circuit's top"
    raise Unsupported(message)


def _solutions(
    built: Built, overrides: dict[str, dict[str, float]] | None
) -> list[Solution]:
    """Every DC solution jolty finds of `built`, by deflation, at tight tolerances.

    Raises:
        JoltyError: When it finds none.
    """
    simulation = Simulation(built.circuit, built.temperature * K, overrides)
    tight = dataclasses.replace(
        built.options,
        relative_tolerance=TIGHT.relative_tolerance,
        voltage_tolerance=TIGHT.voltage_tolerance,
        current_tolerance=TIGHT.current_tolerance,
        charge_tolerance=TIGHT.charge_tolerance,
    )
    found = simulation.run(OperatingPoints(), tight)
    return [{net: point.voltage(net) for net in built.nets} for point in found.points]


def _reference(case: Case) -> dict[str, object]:
    """The reference of `case`, as the child of `reference` prints it."""
    try:
        match case.check:
            case Solutions():
                built = case.problem.jolty()
                if isinstance(case.analysis, Sweep):
                    source = case.analysis.source
                    points = [
                        _solutions(built, {source: {"waveform.dc": level}})
                        for level in case.analysis.levels
                    ]
                else:
                    points = [_solutions(built, None)]
            case Settles() | Oscillates():
                settled = case.problem.settled()
                if settled is None:
                    return {"message": "the case gives no settled circuit"}
                points = [_solutions(settled, None)]
            case _:
                return {"message": "the case needs no reference"}
    except (Unsupported, JoltyError) as error:
        return {"message": " ".join(str(error).split())}
    return {"points": points}


_TOTAL_ITERATIONS: Final = re.compile(r"^Total iterations = (\d+)", re.MULTILINE)
"""How ngspice's ``rusage all`` states the Newton iterations of every analysis run."""

_ANALYSIS_TIME: Final = re.compile(
    r"^Total analysis time \(seconds\) = (\S+)", re.MULTILINE
)
"""How ngspice's ``rusage all`` states the time its analyses took."""

_NOTES: Final = (
    ("Transient op started", "transient operating point"),
    ("source stepping", "source stepping"),
    ("gmin stepping", "gmin stepping"),
)
"""What ngspice prints as it tries each way to an operating point, and its name, the
last tried first."""

_TROUBLE: Final = re.compile(
    r"error|warning|abort|too small|singular|converge|fail|trouble", re.IGNORECASE
)
"""The lines of ngspice's output that say what went wrong."""

_DIGITS: Final = 1e-12
"""How near ngspice's last time must be to a transient's stop, relatively, to have
reached it: an ASCII raw file prints 15 significant digits."""


class Ngspice:
    """ngspice in batch mode at its defaults: Sparse 1.3, and SPICE's tolerances.

    It runs each case's deck with the case's options, at the temperature where its
    thermal voltage is jolty's, as the differential tests run it, the analysis and
    ``rusage all`` in a control section.
    """

    name = "ngspice-sparse"

    def version(self) -> str | None:
        """The version of ngspice, or None when it is not installed."""
        if ngspice.EXECUTABLE is None:
            return None
        done = subprocess.run(  # noqa: S603 - ngspice states its version
            [ngspice.EXECUTABLE, "--version"],
            capture_output=True,
            check=False,
            text=True,
        )
        lines = [line.strip("* ") for line in done.stdout.splitlines()]
        kept = [line for line in lines if line.startswith(("ngspice", "Compiled"))]
        return "; ".join(kept)

    def run(self, case: Case, timeout: float) -> Run:
        """Runs `case` in ngspice, stopped after `timeout` seconds."""
        executable = ngspice.EXECUTABLE
        if executable is None:
            return Run(
                completed=False, message="ngspice is not installed", unsupported=True
            )
        try:
            spiced = case.problem.spice()
        except (Unsupported, JoltyError) as error:
            return Run(completed=False, message=str(error), unsupported=True)
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            deck = [*spiced.lines, *_options(case), ".control", "set filetype=ascii"]
            deck += [_command(case), "write result.raw all", "rusage all"]
            deck += [".endc", ".end"]
            path = directory / "case.cir"
            path.write_text("\n".join(deck) + "\n", encoding="utf-8")
            started = time.perf_counter()
            try:
                done = subprocess.run(  # noqa: S603 - ngspice runs the deck written here
                    [executable, "-b", path.name],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=timeout,
                    cwd=directory,
                )
            except subprocess.TimeoutExpired:
                return Run(
                    completed=False, message=f"ran past {timeout:g} s", timed_out=True
                )
            seconds = time.perf_counter() - started
            printed = done.stdout + done.stderr
            raw = directory / "result.raw"
            plots = (
                ngspice.read(raw.read_text(encoding="utf-8"), printed)
                if raw.is_file()
                else ()
            )
        return _judged(case, spiced.vectors, printed, plots, seconds)


def _judged(
    case: Case,
    vectors: Mapping[str, str],
    printed: str,
    plots: tuple[ngspice.Raw, ...],
    seconds: float,
) -> Run:
    """What ngspice found of `case`, from what it `printed` and the `plots` it wrote."""
    iterations = _TOTAL_ITERATIONS.search(printed)
    analysis = _ANALYSIS_TIME.search(printed)
    counted = Run(
        completed=False,
        seconds=seconds,
        iterations=int(iterations.group(1)) if iterations else None,
        analysis_seconds=float(analysis.group(1)) if analysis else None,
        strategy=next((name for note, name in _NOTES if note in printed), "direct"),
    )
    trouble = [line.strip() for line in printed.splitlines() if _TROUBLE.search(line)]
    message = " | ".join(dict.fromkeys(trouble))[:2000] or "ngspice wrote no results"
    failed = dataclasses.replace(counted, message=message)
    if not plots or plots[-1].plot == "Constant values":
        return failed
    plot = plots[-1]
    missing = [vector for vector in vectors.values() if vector not in plot.vectors]
    if missing:
        return dataclasses.replace(
            counted, message=f"ngspice wrote no {', '.join(missing[:5])}"
        )
    columns = {net: plot.real(vector) for net, vector in vectors.items()}
    rows = len(next(iter(plot.vectors.values())))
    values = tuple(
        {net: float(column[row]) for net, column in columns.items()}
        for row in range(rows)
    )
    aborted = "simulation(s) aborted" in printed
    if not isinstance(case.analysis, Tran):
        return (
            failed
            if aborted
            else dataclasses.replace(counted, completed=True, values=values)
        )
    reached = float(plot.real("time")[-1])
    if aborted or reached < case.analysis.stop * (1 - _DIGITS):
        return dataclasses.replace(failed, reached=reached)
    return dataclasses.replace(
        counted, completed=True, values=values[-1:], reached=reached
    )


def _options(case: Case) -> list[str]:
    """The options line of the case's options, and the temperature line.

    ngspice runs where its thermal voltage is jolty's at the case's temperature, its
    models' nominal temperature where it is jolty's, as `jolty.spice.ngspice.runs`
    runs it.
    """
    options = case.options
    named = {
        "reltol": options.relative_tolerance,
        "vntol": options.voltage_tolerance,
        "abstol": options.current_tolerance,
        "chgtol": options.charge_tolerance,
        "gmin": options.gmin,
        "itl1": options.dc_iterations,
        "itl4": options.transient_iterations,
        "trtol": options.truncation_tolerance,
    }
    given = [f"{key}={value!r}" for key, value in named.items() if value is not None]
    if isinstance(case.analysis, Tran) and case.analysis.method != "trapezoidal":
        given.append("method=gear")
        if case.analysis.method == "backward_euler":
            given.append("maxord=1")
    temperature = case.problem.temperature()
    nominal = ngspice.spice_celsius(_core.REFERENCE_TEMPERATURE)
    lines = [f".options {' '.join(given)}"] if given else []
    lines.append(
        f".options temp={ngspice.spice_celsius(temperature)!r} tnom={nominal!r}"
    )
    return lines


def _command(case: Case) -> str:
    """The case's analysis as an ngspice command."""
    match case.analysis:
        case Point():
            return "op"
        case Sweep(source=source, start=start, stop=stop, step=step):
            return f"dc {source} {start!r} {stop!r} {step!r}"
        case Tran(step=step, stop=stop) as transient:
            return f"tran {step!r} {stop!r} 0 {transient.longest!r}"


def main(arguments: list[str]) -> None:
    """Runs a case, or finds its reference, in jolty, and prints it as JSON."""
    mode, name = arguments
    case = cases()[name]
    found = _found(case).to_json() if mode == "run" else _reference(case)
    print(json.dumps(found, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1:])

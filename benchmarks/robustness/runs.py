# Copyright 2026 Maxence Faldor
"""How the corpus runs a fixture in each simulator, each run in a process of its own.

Each simulator runs a fixture by its runner of ``benchmarks/simulators.py`` (`RUNS`):
jolty at its defaults, and ngspice in batch mode at its defaults, whose direct solver
is Sparse 1.3. Each runs a fixture in a child process that a timeout stops, and returns
what it found, the voltages the fixture compares, its time and its Newton iterations,
or its message.

Run as a script, this module is the child that runs jolty: ``runs.py run FIXTURE``
runs a fixture and ``runs.py reference FIXTURE`` finds its reference, every DC solution
jolty finds by deflation at tight tolerances; each prints one line of JSON.
"""

import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any, Final

from corpus import (
    ROBUSTNESS,
    Built,
    Completes,
    Fixture,
    FixtureRun,
    Impasse,
    Oscillates,
    Point,
    Reference,
    Runs,
    Settles,
    Solution,
    Solutions,
    Sweep,
    Transient,
    fixtures,
)
from simulators import RUSAGE, Ngspice, Simulator, Unsupported, trouble

from jolty import _core, ir, plan
from jolty.errors import JoltyError
from jolty.foundry import SPICE_OPTIONS
from jolty.simulate import Simulation
from jolty.spice import ngspice
from jolty.units import CURRENT, VOLTAGE, K, Quantity, s

_REACHED: Final = re.compile(r"at t = (\S+) s")
"""Where jolty's message says a transient stopped: its step fell below the smallest
``at t = ... s`` (``SimulationError::StepTooSmall``)."""


def _child(mode: str, fixture: Fixture, timeout: float) -> tuple[dict[str, Any], float]:
    """What this module, run as a script in `mode` on `fixture`, printed, and its time.

    The child imports as this process does, from the same path.

    Raises:
        subprocess.TimeoutExpired: When it runs past `timeout`.
        JoltyError: When it prints no result, as when it crashes.
    """
    path = os.pathsep.join(entry for entry in sys.path if entry)
    started = time.perf_counter()
    done = subprocess.run(  # noqa: S603 - this module, run on the fixture named
        [sys.executable, str(ROBUSTNESS / "runs.py"), mode, fixture.name],
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
        env=os.environ | {"PYTHONPATH": path},
    )
    seconds = time.perf_counter() - started
    lines = done.stdout.strip().splitlines()
    if done.returncode != 0 or not lines:
        message = f"exited with {done.returncode}: {done.stderr.strip()[-2000:]}"
        raise JoltyError(message)
    return json.loads(lines[-1]), seconds


def jolty(fixture: Fixture, timeout: float) -> FixtureRun:
    """Runs `fixture` in jolty at its defaults, in a child process.

    The child is stopped after `timeout` seconds.
    """
    try:
        document, seconds = _child("run", fixture, timeout)
    except subprocess.TimeoutExpired:
        return FixtureRun(
            completed=False, message=f"ran past {timeout:g} s", timed_out=True
        )
    except JoltyError as error:
        return FixtureRun(completed=False, message=f"the process failed: {error}")
    return dataclasses.replace(FixtureRun.of(document), seconds=seconds)


def reference(fixture: Fixture, timeout: float) -> Reference:
    """The reference `fixture`'s runs are judged against, found by jolty in a child.

    Every DC solution jolty finds by deflation at tight tolerances: at the operating
    point, at each level of a sweep, or of the circuit settled at a transient's end;
    none for an expectation that needs none.
    """
    if isinstance(fixture.expectation, Impasse | Completes):
        return Reference()
    try:
        document, _ = _child("reference", fixture, timeout)
    except subprocess.TimeoutExpired:
        return Reference(message=f"the search for DC solutions ran past {timeout:g} s")
    except JoltyError as error:
        return Reference(message=f"the search for DC solutions failed: {error}")
    points = document.get("points")
    if not isinstance(points, list):
        return Reference(message=str(document.get("message", "")))
    return Reference(tuple(tuple(point) for point in points))


def _found(fixture: Fixture) -> FixtureRun:
    """Runs `fixture` in jolty, in this process: what the child of `jolty` does."""
    try:
        built = fixture.circuit.jolty()
        simulation = Simulation(built.circuit, built.temperature * K)
    except (Unsupported, JoltyError) as error:
        return FixtureRun(completed=False, message=str(error), unsupported=True)
    diagnostics = tuple(
        f"{diagnostic.rule}: {' '.join(diagnostic.message.split())}"
        for diagnostic in simulation.check()
    )
    started = time.perf_counter()
    try:
        run = _analysis(fixture, built, simulation)
    except JoltyError as error:
        message = " ".join(str(error).split())
        reached = _REACHED.search(message)
        return FixtureRun(
            completed=False,
            message=message,
            analysis_seconds=time.perf_counter() - started,
            reached=float(reached.group(1)) if reached else None,
            diagnostics=diagnostics,
        )
    return dataclasses.replace(
        run, analysis_seconds=time.perf_counter() - started, diagnostics=diagnostics
    )


def _analysis(fixture: Fixture, built: Built, simulation: Simulation) -> FixtureRun:
    """Runs `fixture`'s analysis on `simulation`, and reads the voltages it compares.

    Raises:
        JoltyError: When the analysis fails.
    """
    options = built.options
    match fixture.analysis:
        case Point():
            point = simulation.run(plan.OperatingPoint(), options)
            values = {net: point.voltage(net) for net in built.nets}
            return FixtureRun(
                completed=True,
                iterations=point.iterations,
                strategy=point.strategy,
                values=(values,),
            )
        case Sweep(source=source) as sweep:
            unit = _level(built, source)
            levels = [level * unit for level in sweep.levels]
            swept = simulation.run(plan.DcSweep(source, levels), options)
            columns = {net: swept.voltage(net) for net in built.nets}
            rows = tuple(
                {net: float(column[row]) for net, column in columns.items()}
                for row in range(len(levels))
            )
            return FixtureRun(
                completed=True,
                iterations=sum(swept.iterations),
                strategy=", ".join(dict.fromkeys(swept.strategies)),
                values=rows,
            )
        case Transient(stop=stop, method=method) as transient:
            waveforms = simulation.run(
                plan.Transient(stop * s, max_step=transient.longest * s, method=method),
                options,
            )
            end = {net: float(waveforms.voltage(net)[-1]) for net in built.nets}
            work = waveforms.statistics.work
            return FixtureRun(
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
        instance.of.model
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
        relative_tolerance=plan.TIGHT.relative_tolerance,
        voltage_tolerance=plan.TIGHT.voltage_tolerance,
        current_tolerance=plan.TIGHT.current_tolerance,
        charge_tolerance=plan.TIGHT.charge_tolerance,
    )
    found = simulation.run(plan.OperatingPoints(), tight)
    return [{net: point.voltage(net) for net in built.nets} for point in found.points]


def _reference(fixture: Fixture) -> dict[str, object]:
    """The reference of `fixture`, as the child of `reference` prints it."""
    try:
        match fixture.expectation:
            case Solutions():
                built = fixture.circuit.jolty()
                if isinstance(fixture.analysis, Sweep):
                    source = fixture.analysis.source
                    points = [
                        _solutions(built, {source: {"waveform.dc": level}})
                        for level in fixture.analysis.levels
                    ]
                else:
                    points = [_solutions(built, None)]
            case Settles() | Oscillates():
                settled = fixture.circuit.settled()
                if settled is None:
                    return {"message": "the fixture gives no settled circuit"}
                points = [_solutions(settled, None)]
            case _:
                return {"message": "the fixture needs no reference"}
    except (Unsupported, JoltyError) as error:
        return {"message": " ".join(str(error).split())}
    return {"points": points}


_NOTES: Final = (
    ("Transient op started", "transient operating point"),
    ("source stepping", "source stepping"),
    ("gmin stepping", "gmin stepping"),
)
"""What ngspice prints as it tries each way to an operating point, and its name, the
last tried first."""

_DIGITS: Final = 1e-12
"""How near ngspice's last time must be to a transient's stop, relatively, to have
reached it: an ASCII raw file prints 15 significant digits."""


def ngspice_sparse(fixture: Fixture, timeout: float) -> FixtureRun:
    """Runs `fixture` in ngspice at its defaults, Sparse 1.3 and SPICE's tolerances.

    It runs the fixture's deck with the fixture's options, at the temperature where its
    thermal voltage is jolty's, as the differential tests run it, the analysis and
    ``rusage all`` in a control section, stopped after `timeout` seconds.
    """
    if ngspice.EXECUTABLE is None:
        return FixtureRun(
            completed=False, message="ngspice is not installed", unsupported=True
        )
    try:
        spiced = fixture.circuit.spice()
    except (Unsupported, JoltyError) as error:
        return FixtureRun(completed=False, message=str(error), unsupported=True)
    control = [
        "set filetype=ascii",
        _command(fixture),
        f"write {ngspice.VALUES} all",
        "rusage all",
    ]
    try:
        executed = Ngspice().execute(
            [*spiced.lines, *_options(fixture)], control, timeout=timeout
        )
    except ngspice.NgspiceTimeoutError:
        return FixtureRun(
            completed=False, message=f"ran past {timeout:g} s", timed_out=True
        )
    plots = (
        ()
        if executed.values is None
        else ngspice.read(executed.values, executed.printed)
    )
    return _judged(fixture, spiced.vectors, executed.printed, plots, executed.seconds)


RUNS: Final[Mapping[Simulator, Runs]] = {
    Simulator.JOLTY: jolty,
    Simulator.NGSPICE_SPARSE: ngspice_sparse,
}
"""How each simulator the corpus runs runs a fixture: another joins it by one more."""


def _judged(
    fixture: Fixture,
    vectors: Mapping[str, str],
    printed: str,
    plots: tuple[ngspice.Result, ...],
    seconds: float,
) -> FixtureRun:
    """What ngspice found of `fixture`, from what it `printed` and its `plots`."""
    iterations = RUSAGE["iterations"].search(printed)
    analysis = RUSAGE["analysis_seconds"].search(printed)
    counted = FixtureRun(
        completed=False,
        seconds=seconds,
        iterations=int(iterations.group(1)) if iterations else None,
        analysis_seconds=float(analysis.group(1)) if analysis else None,
        strategy=next((name for note, name in _NOTES if note in printed), "direct"),
    )
    message = trouble(printed) or "ngspice wrote no results"
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
    if not isinstance(fixture.analysis, Transient):
        return (
            failed
            if aborted
            else dataclasses.replace(counted, completed=True, values=values)
        )
    reached = float(plot.real("time")[-1])
    if aborted or reached < fixture.analysis.stop * (1 - _DIGITS):
        return dataclasses.replace(failed, reached=reached)
    return dataclasses.replace(
        counted, completed=True, values=values[-1:], reached=reached
    )


def _options(fixture: Fixture) -> list[str]:
    """The options line of the fixture's options, and the temperature line.

    ngspice runs where its thermal voltage is jolty's at the fixture's temperature, its
    models' nominal temperature where it is jolty's, as `jolty.spice.ngspice.runs`
    runs it.
    """
    options = fixture.options
    given = [
        f"{SPICE_OPTIONS[name]}={value!r}"
        for name, value in ((name, getattr(options, name)) for name in SPICE_OPTIONS)
        if value is not None
    ]
    if (
        isinstance(fixture.analysis, Transient)
        and fixture.analysis.method != "trapezoidal"
    ):
        given.append("method=gear")
        if fixture.analysis.method == "backward_euler":
            given.append("maxord=1")
    temperature = fixture.circuit.temperature()
    nominal = ngspice.option_temperature(_core.REFERENCE_TEMPERATURE)
    lines = [f".options {' '.join(given)}"] if given else []
    lines.append(
        f".options temp={ngspice.option_temperature(temperature)!r} tnom={nominal!r}"
    )
    return lines


def _command(fixture: Fixture) -> str:
    """The fixture's analysis as an ngspice command."""
    match fixture.analysis:
        case Point():
            return "op"
        case Sweep(source=source, start=start, stop=stop, step=step):
            return f"dc {source} {start!r} {stop!r} {step!r}"
        case Transient(step=step, stop=stop) as transient:
            return f"tran {step!r} {stop!r} 0 {transient.longest!r}"


def main(arguments: list[str]) -> None:
    """Runs a fixture, or finds its reference, in jolty, and prints it as JSON."""
    mode, name = arguments
    fixture = fixtures()[name]
    found = _found(fixture).to_json() if mode == "run" else _reference(fixture)
    print(json.dumps(found, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1:])

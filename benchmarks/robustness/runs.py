# Copyright 2026 Maxence Faldor
"""Subprocess runners executing robustness fixtures under timeouts across simulators.

Registered runners in `RUNS` invoke jolty and batch-mode ngspice (Sparse 1.3) at
default options through ``benchmarks/simulators.py``. Each run executes in a separate
timed child process and captures compared voltages, runtime, iteration counts, or error
diagnostics.

Invoked directly as a script, this module serves as the jolty worker: ``runs.py run
FIXTURE`` executes a fixture and ``runs.py reference FIXTURE`` enumerates
DC solutions via deflation. Both commands emit a single JSON line on stdout.
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
from simulators import (
    RUSAGE,
    Ngspice,
    Simulator,
    Unsupported,
    dc_command,
    ngspice_method_flags,
    ngspice_options_line,
    tran_command,
    trouble,
)

from jolty import _core, ir, plan
from jolty.errors import JoltyError
from jolty.foundry import SPICE_OPTIONS
from jolty.simulate import Simulation
from jolty.spice import ngspice
from jolty.units import CURRENT, VOLTAGE, K, Quantity, V, s

_REACHED: Final = re.compile(r"at t = (\S+) s")
"""Pattern extracting the final timestamp from ``SimulationError::StepTooSmall``."""


def _child(mode: str, fixture: Fixture, timeout: float) -> tuple[dict[str, Any], float]:
    """Runs this module in a child process for `mode` and `fixture`, returning JSON.

    Propagates `sys.path` via ``PYTHONPATH`` so the child resolves identical imports.

    Raises:
        subprocess.TimeoutExpired: When the child process exceeds `timeout` seconds.
        JoltyError: When the child exits non-zero or produces no stdout output.
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
    return dataclasses.replace(FixtureRun.from_document(document), seconds=seconds)


def reference(fixture: Fixture, timeout: float) -> Reference:
    """Computes the reference DC solutions for `fixture` in a timed jolty child process.

    Enumerates all DC solutions via deflation at tight tolerances for operating points,
    sweep levels, or final-level settling circuits. Returns an empty `Reference` when
    no reference is needed.
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
            initial = {
                net: volts * V for net, volts in built.initial_conditions.items()
            }
            analysis = plan.Transient(
                stop * s,
                max_step=transient.longest * s,
                method=method,
                initial_conditions=initial,
            )
            waveforms = simulation.run(analysis, options)
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
    """Returns the unit quantity (1 V or 1 A) for independent `source` in `built`.

    Raises:
        Unsupported: When `source` is not a top-level independent source.
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
    built: Built, overrides: dict[str, dict[str, Quantity]] | None
) -> list[Solution]:
    """Enumerates all DC solutions of `built` via deflation at tight tolerances.

    Raises:
        JoltyError: When deflation finds no DC solution.
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
    """Builds the JSON reference payload printed by the `reference` child worker."""
    try:
        match fixture.expectation:
            case Solutions():
                built = fixture.circuit.jolty()
                if isinstance(fixture.analysis, Sweep):
                    source = fixture.analysis.source
                    unit = _level(built, source)
                    points = [
                        _solutions(built, {source: {"waveform.dc": level * unit}})
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
"""ngspice stdout markers mapped to operating-point strategies (last-tried first)."""

_DIGITS: Final = 1e-12
"""Relative tolerance verifying ngspice reached transient stop (15-digit ASCII raw)."""


def ngspice_sparse(fixture: Fixture, timeout: float) -> FixtureRun:
    """Runs `fixture` in batch ngspice with Sparse 1.3 and default SPICE tolerances.

    Executes the fixture deck with fixture options at jolty's thermal-voltage equivalent
    temperature, running the analysis and ``rusage all`` in a ``.control`` block under
    `timeout`.
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
"""Registry mapping each supported robustness simulator to its execution callable."""


def _judged(
    fixture: Fixture,
    vectors: Mapping[str, str],
    printed: str,
    plots: tuple[ngspice.Result, ...],
    seconds: float,
) -> FixtureRun:
    """Builds a `FixtureRun` for `fixture` from ngspice `printed` output and `plots`."""
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
    """Builds ngspice ``.options`` lines for tolerances, method, and temperature.

    Matches jolty's thermal voltage at both simulation and nominal model temperatures,
    consistent with `jolty.spice.ngspice.runs`.
    """
    options = fixture.options
    given = [
        f"{SPICE_OPTIONS[name]}={value!r}"
        for name, value in ((name, getattr(options, name)) for name in SPICE_OPTIONS)
        if value is not None
    ]
    if isinstance(fixture.analysis, Transient):
        given.extend(ngspice_method_flags(fixture.analysis.method))
    temperature = fixture.circuit.temperature()
    nominal = ngspice.option_temperature(_core.REFERENCE_TEMPERATURE)
    option_line = ngspice_options_line(given)
    lines = [option_line] if option_line else []
    lines.append(
        f".options temp={ngspice.option_temperature(temperature)!r} tnom={nominal!r}"
    )
    return lines


def _command(fixture: Fixture) -> str:
    """Translates the analysis of `fixture` into an ngspice control command."""
    match fixture.analysis:
        case Point():
            return "op"
        case Sweep(source=source, start=start, stop=stop, step=step):
            return dc_command(source, start, stop, step)
        case Transient(step=step, stop=stop) as transient:
            return tran_command(step, stop, transient.longest)


def main(arguments: list[str]) -> None:
    """Runs a fixture, or finds its reference, in jolty, and prints it as JSON."""
    mode, name = arguments
    fixture = fixtures()[name]
    found = _found(fixture).to_json() if mode == "run" else _reference(fixture)
    print(json.dumps(found, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1:])

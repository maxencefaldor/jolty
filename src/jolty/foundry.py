# Copyright 2026 Maxence Faldor
"""Analyses of circuits that hold a foundry's devices, run in ngspice with its models.

The engine solves no foundry's device, such as a SKY130 transistor, so
[`Simulation`][jolty.simulate.Simulation] runs each analysis of a circuit that holds one
here. The circuit is written as a SPICE netlist, ngspice reads its kit's model library
at a process corner, and runs the analysis at the simulation's temperature with the
plan's options, the others at SPICE's defaults, which are the engine's. The vectors it
writes become a result of the same kind, whose unknowns are the circuit's nets and the
currents of its sources and inductors, as jolty names them: a device's own unknowns,
and its terminals' currents, ngspice keeps to itself.

An operating point, a DC sweep in even steps, a small-signal response and a transient
run this way. ngspice steps a transient where it chooses, and lands on the times the
analysis names only where its steps do. Any other analysis of such a circuit fails, and
says so.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
from pydantic import JsonValue

from jolty import ir, results
from jolty.errors import JoltyError, SimulationError
from jolty.plan import Ac, Analysis, DcSweep, OperatingPoint, Options, Transient
from jolty.spice import export, sky130
from jolty.units import CURRENT, FREQUENCY, TIME, VOLTAGE

STRATEGY: Final = "ngspice"
"""The strategy of an operating point ngspice found."""

_OPTIONS: Final = {
    "relative_tolerance": "reltol",
    "voltage_tolerance": "vntol",
    "current_tolerance": "abstol",
    "charge_tolerance": "chgtol",
    "gmin": "gmin",
    "dc_iterations": "itl1",
    "transient_iterations": "itl4",
    "truncation_tolerance": "trtol",
}
"""ngspice's name of each of jolty's options."""

_METHODS: Final = {
    "trapezoidal": "method=trap",
    "gear2": "method=gear maxord=2",
    "backward_euler": "method=gear maxord=1",
}
"""ngspice's options for each integration method."""

_EVEN: Final = 1e-9
"""How far, as a fraction of a step, a DC sweep's levels may lie from even steps."""


def run(  # noqa: PLR0913 - the analysis, and the conditions it runs at
    circuit: ir.Circuit,
    analysis: Analysis,
    options: Options,
    *,
    temperature: float,
    process: str | None,
    unknowns: Sequence[results.Unknown],
) -> results.Result:
    """Runs `analysis` of `circuit` in ngspice, at `temperature` and at `process`.

    The temperature is in kelvins, and the process is the corner of the kit's model
    library, such as ``ss``, its typical one unless given. The result holds those of
    `unknowns` that ngspice names.

    Raises:
        SimulationError: When ngspice does not run the analysis, or cannot read the
            circuit, saying why.
    """
    corner = sky130.TYPICAL if process is None else process
    try:
        return _run(
            circuit,
            analysis,
            options=options,
            temperature=temperature,
            process=corner,
            unknowns=unknowns,
        )
    except JoltyError as error:
        if isinstance(error, SimulationError):
            raise
        raise SimulationError(str(error)) from None


def _run(  # noqa: PLR0913 - as `run`
    circuit: ir.Circuit,
    analysis: Analysis,
    *,
    options: Options,
    temperature: float,
    process: str,
    unknowns: Sequence[results.Unknown],
) -> results.Result:
    if process not in sky130.CORNERS:
        message = f"SKY130 has no process corner {process!r}: it has {sky130.CORNERS}"
        raise SimulationError(message)
    flat = export.flatten(circuit)
    excited = _source(flat, analysis.source)[0].id if isinstance(analysis, Ac) else None
    lines = export.netlist(flat, excited=excited)
    read = _readable(flat, unknowns)
    running: export.Running = {
        "sections": ((sky130.root() / sky130.LIBRARY, process),),
        "compatibility": "hsa",
        "options": _options(options, analysis),
        "temperature": temperature,
    }
    match analysis:
        case OperatingPoint():
            (raw,) = export.ngspice_runs(lines, ("op",), **running)
            return _operating_point(raw, read)
        case DcSweep():
            command, levels, quantity = _sweep(flat, analysis)
            (raw,) = export.ngspice_runs(lines, (command,), **running)
            values = _table(raw, read)
            if values.shape[0] != len(levels):
                message = (
                    f"ngspice swept {analysis.source} through {values.shape[0]} "
                    f"levels, not {len(levels)}"
                )
                raise SimulationError(message)
            return results.Sweep(
                unknowns=tuple(unknown for unknown, _ in read),
                source=analysis.source,
                quantity=quantity,
                levels=levels,
                values=values,
            )
        case Ac():
            frequencies = [
                frequency.to(FREQUENCY, "a frequency")
                for frequency in analysis.frequencies
            ]
            commands = (
                "op",
                *(f"ac lin 1 {frequency!r} {frequency!r}" for frequency in frequencies),
            )
            point, *responses = export.ngspice_runs(lines, commands, **running)
            return results.Response(
                unknowns=tuple(unknown for unknown, _ in read),
                source=analysis.source,
                frequencies=np.array(frequencies),
                values=np.array(
                    [
                        [raw.vectors[vector][0] for _, vector in read]
                        for raw in responses
                    ]
                ).reshape(len(frequencies), len(read)),
                operating_point=_operating_point(point, read),
            )
        case Transient(periodic=False):
            return _transient(lines, analysis, running, read)
    message = (
        f"ngspice runs no {type(analysis).__name__} analysis of a circuit that holds a "
        "foundry's device here: only an operating point, a DC sweep, a small-signal "
        "response and a transient that is not periodic"
    )
    raise SimulationError(message)


def overridden(
    circuit: ir.Circuit, overrides: Mapping[str, Mapping[str, float]]
) -> ir.Circuit:
    """`circuit` with the parameters `overrides` replaces, as the engine replaces them.

    They replace a device's parameters by its instance's path, then by the parameter's
    path, such as ``waveform.dc``. A circuit with any is flattened into its devices,
    their ids their paths joined by underscores, as a netlist writes them.
    """
    if not overrides:
        return circuit
    replaced = {path.replace("/", "_"): values for path, values in overrides.items()}
    flat = export.flatten(circuit)
    top = flat.components[flat.top]
    instances = []
    for instance in top.instances:
        parameters = dict(instance.parameters)
        for path, value in replaced.get(instance.id, {}).items():
            key, *rest = path.split(".")
            parameters[key] = _replaced(parameters.get(key), rest, value)
        instances.append(instance.model_copy(update={"parameters": parameters}))
    placing = top.model_copy(update={"instances": tuple(instances)})
    return flat.model_copy(
        update={"components": {**flat.components, flat.top: placing}}
    )


def _replaced(found: JsonValue, keys: Sequence[str], value: float) -> JsonValue:
    """`found` with `value` at the path of `keys` within it."""
    if not keys:
        return value
    key, *rest = keys
    match found:
        case dict():
            return {**found, key: _replaced(found.get(key), rest, value)}
        case list() if key.isdecimal() and int(key) < len(found):
            entries = list(found)
            entries[int(key)] = _replaced(entries[int(key)], rest, value)
            return entries
    message = f"no parameter at {'.'.join(keys)} to replace"
    raise SimulationError(message)


def _readable(
    circuit: ir.Circuit, unknowns: Sequence[results.Unknown]
) -> list[tuple[results.Unknown, str]]:
    """Each of `unknowns` that ngspice names, and the name of its vector."""
    read = []
    for unknown in unknowns:
        try:
            read.append((unknown, export.vector(circuit, unknown)))
        except JoltyError:
            continue
    return read


def _options(options: Options, analysis: Analysis) -> str:
    """The options line ngspice reads: the plan's options, and a transient's method."""
    given = [
        f"{_OPTIONS[name]}={value!r}"
        for name, value in ((name, getattr(options, name)) for name in _OPTIONS)
        if value is not None
    ]
    if isinstance(analysis, Transient):
        given.append(_METHODS[analysis.method])
    return f".options {' '.join(given)}" if given else "*"


def _table(
    raw: export.Raw, read: Sequence[tuple[results.Unknown, str]]
) -> results.Real:
    """The real values of each of `read`'s vectors, a column each."""
    return np.column_stack([raw.real(vector) for _, vector in read]).reshape(
        -1, len(read)
    )


def _operating_point(
    raw: export.Raw, read: Sequence[tuple[results.Unknown, str]]
) -> results.OperatingPoint:
    return results.OperatingPoint(
        unknowns=tuple(unknown for unknown, _ in read),
        values=_table(raw, read)[0],
        strategy=STRATEGY,
    )


def _source(flat: ir.Circuit, path: str) -> tuple[ir.Instance, results.Quantity]:
    """The independent source at `path` in the flattened circuit `flat`, and its level.

    Its level is a voltage or a current.

    Raises:
        SimulationError: When no independent source is at `path`.
    """
    found = flat.components[flat.top].instance(path.replace("/", "_"))
    match found:
        case ir.Instance(of=ir.DeviceOf(device="voltage_source")):
            return found, "voltage"
        case ir.Instance(of=ir.DeviceOf(device="current_source")):
            return found, "current"
    message = f"`{path}` is not an independent source"
    raise SimulationError(message)


def _sweep(
    flat: ir.Circuit, sweep: DcSweep
) -> tuple[str, results.Real, results.Quantity]:
    """The command ngspice runs `sweep` by, its levels, and whether they are voltages.

    Raises:
        SimulationError: When the swept instance is no independent source, or the
            levels do not rise or fall in even steps, as ngspice sweeps a source.
    """
    found, quantity = _source(flat, sweep.source)
    dimension = VOLTAGE if quantity == "voltage" else CURRENT
    given = [level.to(dimension, "a sweep level") for level in sweep.levels]
    levels = np.array(given)
    if len(levels) < 2:  # noqa: PLR2004 - a sweep's first and last levels
        message = "ngspice sweeps a source through two levels or more"
        raise SimulationError(message)
    first, last = given[0], given[-1]
    step = (last - first) / (len(given) - 1)
    even = levels[0] + step * np.arange(len(levels))
    if step == 0 or np.any(np.abs(levels - even) > _EVEN * abs(step)):
        message = f"ngspice sweeps {sweep.source} in even steps, and its levels are not"
        raise SimulationError(message)
    command = f"dc {export.element(found)} {first!r} {last!r} {step!r}"
    return command, levels, quantity


_STATISTICS: Final = {
    "iterations": re.compile(r"Transient iterations\s*=\s*(\d+)"),
    "accepted": re.compile(r"Accepted timepoints\s*=\s*(\d+)"),
    "rejected": re.compile(r"Rejected timepoints\s*=\s*(\d+)"),
}
"""What ngspice's ``rusage`` prints of a transient: its Newton iterations, and the time
points it accepted and rejected."""


def _transient(
    lines: list[str],
    transient: Transient,
    running: export.Running,
    read: Sequence[tuple[results.Unknown, str]],
) -> results.Waveforms:
    stop = transient.stop.to(TIME, "a transient's stop time")
    longest = (
        stop / 50
        if transient.max_step is None
        else transient.max_step.to(TIME, "a transient's longest step")
    )
    command = (
        f"tran {longest!r} {stop!r} 0 {longest!r}\nrusage traniter accept rejected"
    )
    (raw,) = export.ngspice_runs(lines, (command,), **running)
    counted = {
        name: [int(found) for found in pattern.findall(raw.printed)]
        for name, pattern in _STATISTICS.items()
    }
    if any(len(found) != 1 for found in counted.values()):
        message = f"ngspice printed no statistics of its transient:\n{raw.printed}"
        raise SimulationError(message)
    return results.Waveforms(
        unknowns=tuple(unknown for unknown, _ in read),
        times=raw.real("time"),
        values=_table(raw, read),
        statistics=results.Statistics(
            initial=STRATEGY,
            accepted=counted["accepted"][0],
            rejected=counted["rejected"][0],
            iterations=counted["iterations"][0],
        ),
    )

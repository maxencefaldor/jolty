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

An operating point, a DC sweep in even steps, a small-signal response, a transient and
the noise at a voltage run this way. ngspice steps a transient where it chooses, and
lands on the times the analysis names only where its steps do. It names each source of
noise its own way, such as ``id`` for a transistor's channel and ``1overf`` for its
flicker noise, and refers noise to a source of its own choosing when the analysis names
none, which the result then leaves out. Any other analysis of such a circuit fails, and
says so.

Given a seed, ngspice draws the kit's mismatch, each transistor's parameters apart from
the others', as a Monte Carlo sample of the process does; the same seed draws the same.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from pydantic import JsonValue

from jolty import ir, results
from jolty.errors import JoltyError, SimulationError
from jolty.plan import (
    Ac,
    Analysis,
    Conditions,
    DcSweep,
    Noise,
    OperatingPoint,
    Options,
    Transient,
)
from jolty.spice import export, ngspice, sky130
from jolty.units import CURRENT, FREQUENCY, TIME, VOLTAGE, Quantity

STRATEGY: Final = "ngspice"
"""The strategy of an operating point ngspice found."""

SPICE_OPTIONS: Final = {
    "relative_tolerance": "reltol",
    "voltage_tolerance": "vntol",
    "current_tolerance": "abstol",
    "charge_tolerance": "chgtol",
    "gmin": "gmin",
    "max_dc_iterations": "itl1",
    "max_transient_iterations": "itl4",
    "truncation_tolerance": "trtol",
}
"""ngspice's ``.options`` name of each option of [`Options`][jolty.plan.Options] it
reads alike, by the option's name."""
"""ngspice's name of each of jolty's options."""

_METHODS: Final = {
    "trapezoidal": "method=trap",
    "gear2": "method=gear maxord=2",
    "backward_euler": "method=gear maxord=1",
}
"""ngspice's options for each integration method."""

_EVEN: Final = 1e-9
"""How far, as a fraction of a step, a DC sweep's levels may lie from even steps."""


def run(
    circuit: ir.Circuit,
    analysis: Analysis,
    options: Options,
    conditions: Conditions,
    unknowns: Sequence[results.Unknown],
) -> results.AnalysisResult:
    """Runs `analysis` of `circuit` in ngspice, with `options`, at `conditions`.

    The result holds those of `unknowns` that ngspice names.

    Raises:
        SimulationError: When ngspice does not run the analysis, or cannot read the
            circuit, saying why.
    """
    try:
        return _Deck.from_circuit(circuit, analysis, options, conditions, unknowns).run(
            analysis
        )
    except JoltyError as error:
        if isinstance(error, SimulationError):
            raise
        raise SimulationError(str(error)) from None


@dataclass(frozen=True, slots=True)
class _Deck:
    """A circuit flattened into its devices, as ngspice runs one of its analyses."""

    flat: ir.Circuit
    paths: Mapping[str, str]
    """Each device's path, by its id in `flat`."""
    lines: list[str]
    read: list[tuple[results.Unknown, str]]
    """Each unknown ngspice names, and the name of its vector."""
    running: ngspice.Running

    @classmethod
    def from_circuit(
        cls,
        circuit: ir.Circuit,
        analysis: Analysis,
        options: Options,
        conditions: Conditions,
        unknowns: Sequence[results.Unknown],
    ) -> _Deck:
        """The deck of `analysis` of `circuit`, which excites the source it reads.

        Raises:
            SimulationError: When the kit has no such process corner, or the analysis
                names no independent source where it names one.
        """
        kit = sky130.running(
            conditions.process, mismatched=conditions.mismatch is not None
        )
        flat, paths = _flattened(circuit, conditions.overrides)
        excited = _excited(flat, analysis)
        running: ngspice.Running = {
            **kit,
            "options": _options(options, analysis),
            "temperature": conditions.temperature,
            "seed": conditions.mismatch,
        }
        return cls(
            flat,
            paths,
            export.netlist(
                flat, excited=excited, transient=isinstance(analysis, Transient)
            ),
            _readable(flat, unknowns),
            running,
        )

    def runs(self, commands: Sequence[str]) -> tuple[ngspice.Result, ...]:
        """The plot of each of `commands`, run in turn in one ngspice."""
        return ngspice.runs(self.lines, commands, **self.running)

    def unknowns(self) -> tuple[results.Unknown, ...]:
        return tuple(unknown for unknown, _ in self.read)

    def run(self, analysis: Analysis) -> results.AnalysisResult:
        match analysis:
            case OperatingPoint():
                (raw,) = self.runs(("op",))
                return self.operating_point(raw)
            case DcSweep():
                return self.sweep(analysis)
            case Ac():
                frequencies = _frequencies(analysis.frequencies)
                point, *responses = self.runs(("op", *_points("ac", frequencies)))
                return results.AcResult(
                    unknowns=self.unknowns(),
                    source=analysis.source,
                    frequencies=np.array(frequencies),
                    values=np.array(
                        [
                            [raw.vectors[vector][0] for _, vector in self.read]
                            for raw in responses
                        ]
                    ).reshape(len(frequencies), len(self.read)),
                    operating_point=self.operating_point(point),
                )
            case Transient(periodic=False):
                return self.transient(analysis)
            case Noise():
                return self.noise(analysis)
        message = (
            f"ngspice runs no {type(analysis).__name__} analysis of a circuit that "
            "holds a foundry's device here: only an operating point, a DC sweep, a "
            "small-signal response, a transient that is not periodic and noise"
        )
        raise SimulationError(message)

    def operating_point(self, raw: ngspice.Result) -> results.OperatingPointResult:
        return results.OperatingPointResult(
            unknowns=self.unknowns(),
            values=_table(raw, self.read)[0],
            strategy=STRATEGY,
        )

    def sweep(self, sweep: DcSweep) -> results.DcSweepResult:
        command, levels, nature = _sweep(self.flat, sweep)
        (raw,) = self.runs((command,))
        values = _table(raw, self.read)
        if values.shape[0] != len(levels):
            message = (
                f"ngspice swept {sweep.source} through {values.shape[0]} levels, not "
                f"{len(levels)}"
            )
            raise SimulationError(message)
        return results.DcSweepResult(
            unknowns=self.unknowns(),
            source=sweep.source,
            nature=nature,
            swept=levels,
            values=values,
        )

    def transient(self, transient: Transient) -> results.TransientResult:
        stop = transient.stop.to(TIME, "a transient's stop time")
        longest = (
            stop / 50
            if transient.max_step is None
            else transient.max_step.to(TIME, "a transient's longest step")
        )
        steps = f"tran {longest!r} {stop!r} 0 {longest!r}"
        (raw,) = self.runs((f"{steps}\nrusage traniter accept rejected",))
        counted = {
            name: [int(found) for found in pattern.findall(raw.printed)]
            for name, pattern in _STATISTICS.items()
        }
        if any(len(found) != 1 for found in counted.values()):
            message = f"ngspice printed no statistics of its transient:\n{raw.printed}"
            raise SimulationError(message)
        return results.TransientResult(
            unknowns=self.unknowns(),
            times=raw.real("time"),
            values=_table(raw, self.read),
            statistics=results.Statistics(
                initial=STRATEGY,
                accepted=counted["accepted"][0],
                rejected=counted["rejected"][0],
                # ngspice steps across an event rather than landing on it.
                landed=0,
                iterations=counted["iterations"][0],
            ),
        )

    def noise(self, noise: Noise) -> results.NoiseResult:
        """The noise at a voltage, each device's sources apart, referred to a source.

        Raises:
            SimulationError: When the output is not a voltage, or the circuit holds no
                independent source to refer its noise to.
        """
        output = self.output(noise)
        frequencies = _frequencies(noise.frequencies)
        source = _referred(self.flat, noise)
        reads = f"noise {output} {export.element(source)}"
        noises = [
            f"{reads} lin 1 {frequency!r} {frequency!r} 1" for frequency in frequencies
        ]
        noises[0] = f"set sqrnoise\n{noises[0]}"
        gains = _points("ac", frequencies) if noise.source is not None else []
        point, *found = self.runs(("op", *gains, *noises))
        responses, spectra = found[: len(gains)], found[len(gains) :]
        densities = np.array([raw.real("onoise_spectrum")[0] for raw in spectra])
        sources, contributions = self.contributions(spectra)
        given = None
        if noise.source is not None:
            _, nature = _source(self.flat, noise.source)
            transfer = _difference(output)
            given = results.NoiseInput(
                source=noise.source,
                nature=nature,
                gains=np.array(
                    [
                        raw.vectors[transfer[0]][0]
                        - (raw.vectors[transfer[1]][0] if transfer[1] else 0)
                        for raw in responses
                    ]
                ),
            )
        return results.NoiseResult(
            unknowns=self.unknowns(),
            output=noise.output,
            reference=noise.reference,
            nature="voltage",
            frequencies=np.array(frequencies),
            densities=densities,
            sources=sources,
            contributions=contributions,
            input=given,
            operating_point=self.operating_point(point),
        )

    def output(self, noise: Noise) -> str:
        """The output of `noise` as ngspice names it, ``v(out)`` or ``v(out,ref)``.

        Raises:
            SimulationError: When the output is not a voltage.
        """
        named = {unknown.name: vector for unknown, vector in self.read}
        nodes = [noise.output, *([] if noise.reference is None else [noise.reference])]
        vectors = [named.get(node, "") for node in nodes]
        if not all(vector.startswith("v(") for vector in vectors):
            message = (
                "ngspice takes the noise at a voltage, over the ground or a node, and "
                f"{' over '.join(nodes)} is not one"
            )
            raise SimulationError(message)
        return f"v({','.join(vector[2:-1] for vector in vectors)})"

    def contributions(
        self, spectra: Sequence[ngspice.Result]
    ) -> tuple[tuple[results.NoiseSource, ...], results.Real]:
        """Each device's sources of noise that add any, and what each adds.

        Raises:
            SimulationError: When ngspice names noise of a device the netlist does not
                hold.
        """
        devices = sorted(
            _devices(self.flat, self.paths).items(), key=lambda pair: -len(pair[0])
        )
        found: dict[results.NoiseSource, list[float]] = {}
        for name in spectra[0].vectors:
            rest = name.removeprefix("onoise")[1:]
            if not name.startswith("onoise") or rest == "spectrum":
                continue
            matched = next(
                (
                    (path, rest[len(device) + 1 :])
                    for device, path in devices
                    if rest == device or rest.startswith((f"{device}_", f"{device}."))
                ),
                None,
            )
            if matched is None:
                message = (
                    f"ngspice names noise of {rest}, which the netlist does not hold"
                )
                raise SimulationError(message)
            path, kind = matched
            values = [raw.real(name)[0] for raw in spectra]
            if kind and any(values):
                found[results.NoiseSource(path, kind)] = values
        sources = tuple(found)
        contributions = np.array([found[source] for source in sources]).reshape(
            len(sources), len(spectra)
        )
        return sources, contributions


def _flattened(
    circuit: ir.Circuit, overrides: Mapping[str, Mapping[str, float]]
) -> tuple[ir.Circuit, dict[str, str]]:
    """`circuit` flattened into its devices, with the parameters `overrides` replaces.

    Each device's id is its path, its ids joined by underscores, as a netlist writes
    it; the mapping gives each device's path by that id.
    """
    paths = {
        path.replace("/", "_"): path
        for path, instance in _devices_within(circuit, circuit.top, "")
    }
    replaced = {path.replace("/", "_"): values for path, values in overrides.items()}
    flat = export.flatten(circuit)
    top = flat.components[flat.top]
    instances = []
    for instance in top.instances:
        parameters = dict(instance.parameters)
        for path, value in replaced.get(instance.name, {}).items():
            key, *rest = path.split(".")
            parameters[key] = _replaced(parameters.get(key), rest, value)
        instances.append(instance.model_copy(update={"parameters": parameters}))
    placing = top.model_copy(update={"instances": tuple(instances)})
    flattened = flat.model_copy(
        update={"components": {**flat.components, flat.top: placing}}
    )
    return flattened, paths


def _devices_within(
    circuit: ir.Circuit, component: str, prefix: str
) -> list[tuple[str, ir.Instance]]:
    """Every device within `component`, by path."""
    found = []
    for instance in circuit.components[component].instances:
        path = f"{prefix}{instance.name}"
        if isinstance(instance.of, ir.ComponentOf):
            found.extend(_devices_within(circuit, instance.of.component, f"{path}/"))
        else:
            found.append((path, instance))
    return found


def _devices(flat: ir.Circuit, paths: Mapping[str, str]) -> dict[str, str]:
    """Each device's path, by the name ngspice gives it in its noise's vectors."""
    named = {}
    for instance in flat.components[flat.top].instances:
        element = export.element(instance).lower()
        match instance:
            case ir.Instance(
                of=ir.DeviceOf(model="foundry_mosfet"),
                parameters={"model": str(model)},
            ):
                named[sky130.inner(element, model)] = paths[instance.name]
            case _:
                named[element] = paths[instance.name]
    return named


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
        f"{SPICE_OPTIONS[name]}={value!r}"
        for name, value in ((name, getattr(options, name)) for name in SPICE_OPTIONS)
        if value is not None
    ]
    if isinstance(analysis, Transient):
        given.append(_METHODS[analysis.method])
    return f".options {' '.join(given)}" if given else "*"


def _table(
    raw: ngspice.Result, read: Sequence[tuple[results.Unknown, str]]
) -> results.Real:
    """The real values of each of `read`'s vectors, a column each."""
    return np.column_stack([raw.real(vector) for _, vector in read]).reshape(
        -1, len(read)
    )


def _source(flat: ir.Circuit, path: str) -> tuple[ir.Instance, results.Nature]:
    """The independent source at `path` in the flattened circuit `flat`, and its level.

    Its level is a voltage or a current.

    Raises:
        SimulationError: When no independent source is at `path`.
    """
    found = flat.components[flat.top].instance(path.replace("/", "_"))
    match found:
        case ir.Instance(of=ir.DeviceOf(model="voltage_source")):
            return found, "voltage"
        case ir.Instance(of=ir.DeviceOf(model="current_source")):
            return found, "current"
    message = f"`{path}` is not an independent source"
    raise SimulationError(message)


def _sweep(
    flat: ir.Circuit, sweep: DcSweep
) -> tuple[str, results.Real, results.Nature]:
    """The command ngspice runs `sweep` by, its levels, and whether they are voltages.

    Raises:
        SimulationError: When the swept instance is no independent source, or the
            levels do not rise or fall in even steps, as ngspice sweeps a source.
    """
    found, nature = _source(flat, sweep.source)
    dimension = VOLTAGE if nature == "voltage" else CURRENT
    given = [level.to(dimension, "a sweep level") for level in sweep.values]
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
    return command, levels, nature


_STATISTICS: Final = {
    "iterations": re.compile(r"Transient iterations\s*=\s*(\d+)"),
    "accepted": re.compile(r"Accepted timepoints\s*=\s*(\d+)"),
    "rejected": re.compile(r"Rejected timepoints\s*=\s*(\d+)"),
}
"""What ngspice's ``rusage`` prints of a transient: its Newton iterations, and the time
points it accepted and rejected."""


def _excited(flat: ir.Circuit, analysis: Analysis) -> str | None:
    """The id of the source a small-signal response or a noise analysis excites."""
    match analysis:
        case Ac():
            return _source(flat, analysis.source)[0].name
        case Noise():
            return _referred(flat, analysis).name
    return None


def _referred(flat: ir.Circuit, noise: Noise) -> ir.Instance:
    """The independent source `noise` is referred to: its own, or the circuit's first.

    Raises:
        SimulationError: When the analysis names no independent source, or names none
            and the circuit holds none.
    """
    if noise.source is not None:
        return _source(flat, noise.source)[0]
    found = next(
        (
            instance
            for instance in flat.components[flat.top].instances
            if instance.of
            in (
                ir.DeviceOf(model="voltage_source"),
                ir.DeviceOf(model="current_source"),
            )
        ),
        None,
    )
    if found is None:
        message = (
            "ngspice refers noise to an independent source, and the circuit holds none"
        )
        raise SimulationError(message)
    return found


def _frequencies(given: Sequence[Quantity]) -> list[float]:
    return [frequency.to(FREQUENCY, "a frequency") for frequency in given]


def _points(analysis: str, frequencies: Sequence[float]) -> list[str]:
    """A small-signal command of `analysis` at each of `frequencies`, in turn."""
    return [
        f"{analysis} lin 1 {frequency!r} {frequency!r}" for frequency in frequencies
    ]


def _difference(output: str) -> tuple[str, str]:
    """The vectors whose difference is the voltage ``v(out,reference)`` names."""
    nodes = output.removeprefix("v(").removesuffix(")").split(",")
    first, *rest = nodes
    return f"v({first})", f"v({rest[0]})" if rest else ""

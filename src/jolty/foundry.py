# Copyright 2026 Maxence Faldor
"""Foundry-device circuit analyses executed in ngspice with process design kit models.

The native engine does not solve foundry devices such as SKY130 transistors, so
[`Simulation`][jolty.simulate.Simulation] dispatches circuits containing foundry devices
to this module. Each circuit is exported as a SPICE netlist, loaded with the kit model
library at the requested process corner, and simulated at the configured temperature and
plan options. Output vectors are converted into standard jolty results containing net
voltages and source or inductor currents. Internal device unknowns and individual
terminal currents are not exported by ngspice.

Supported analyses are operating point, evenly spaced DC sweep, small-signal AC
response, transient, and voltage noise. During transient runs, ngspice chooses adaptive
time steps and hits requested times only when a step lands on them. Noise sources use
ngspice naming conventions such as ``id`` for channel thermal noise and ``1overf`` for
flicker noise. When a noise analysis omits an input reference source, ngspice selects an
internal reference source whose input-referred data the result omits. Unsupported
analyses raise a simulation error.

When a mismatch seed is provided, ngspice samples per-instance device mismatch
parameters reproducibly for that seed.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from pydantic import JsonValue

from jolty import ir, results, unknowns
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
    voltages,
)
from jolty.spice import export, ngspice, sky130
from jolty.units import FREQUENCY, TIME, Quantity

STRATEGY: Final = "ngspice"
"""Operating-point strategy name recorded for ngspice solutions."""

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
"""Mapping from [`Options`][jolty.plan.Options] field names to ngspice ``.options``
parameter names."""

_METHODS: Final = {
    "trapezoidal": "method=trap",
    "gear2": "method=gear maxord=2",
    "backward_euler": "method=gear maxord=1",
}
"""Mapping from integration method names to ngspice ``.options`` clauses."""

_EVEN: Final = 1e-9
"""Maximum relative deviation allowed from uniform step spacing in a DC sweep."""


def run(
    circuit: ir.Circuit,
    analysis: Analysis,
    options: Options,
    conditions: Conditions,
    unknowns: Sequence[results.Unknown],
) -> results.AnalysisResult:
    """Executes `analysis` on `circuit` in ngspice with `options` and `conditions`.

    The returned result contains all unknowns from `unknowns` that ngspice exports.

    Args:
        circuit: Circuit containing one or more foundry devices.
        analysis: Supported analysis specification to execute.
        options: Solver tolerance, iteration, and nodeset options.
        conditions: Temperature, process corner, mismatch seed, and parameter overrides.
        unknowns: Circuit unknowns requested for extraction from ngspice output.

    Returns:
        Standard `AnalysisResult` populated from exported ngspice vectors.

    Raises:
        SimulationError: If ngspice does not support the analysis, fails to run, or
            cannot read the netlist.
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
    """Flattened circuit deck prepared for ngspice execution."""

    flat: ir.Circuit
    paths: Mapping[str, str]
    """Mapping from flattened device name in `flat` to original hierarchical path."""
    lines: list[str]
    read: list[tuple[results.Unknown, str]]
    """Sequence of supported unknowns paired with their ngspice vector names."""
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
        """Builds an ngspice deck for `analysis` on `circuit`, exciting any AC source.

        Raises:
            SimulationError: If the process corner is invalid or an analysis references
                a missing or non-independent source.
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
        transient = isinstance(analysis, Transient)
        initial = analysis.initial_conditions if transient else {}
        return cls(
            flat,
            paths,
            [
                *export.netlist(flat, excited=excited, transient=transient),
                *export.node_voltages(
                    ".nodeset", flat, voltages(options.nodesets, "a nodeset")
                ),
                *export.node_voltages(
                    ".ic", flat, voltages(initial, "an initial condition")
                ),
            ],
            _readable(flat, unknowns),
            running,
        )

    def runs(self, commands: Sequence[str]) -> tuple[ngspice.Result, ...]:
        """Executes `commands` sequentially in a single ngspice invocation."""
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
            case Transient():
                return self.transient(analysis)
            case Noise():
                return self.noise(analysis)
        message = (
            f"ngspice runs no {type(analysis).__name__} analysis of a circuit that "
            "holds a foundry's device here: only an operating point, a DC sweep, a "
            "small-signal response, a transient and noise"
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
        # The ngspice `uic` flag starts the transient from initial conditions alone.
        start = "" if transient.operating_point else " uic"
        steps = f"tran {longest!r} {stop!r} 0 {longest!r}{start}"
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
                initial=STRATEGY if transient.operating_point else None,
                accepted=counted["accepted"][0],
                rejected=counted["rejected"][0],
                # Adaptive ngspice time stepping crosses events instead of landing.
                landed=0,
                iterations=counted["iterations"][0],
            ),
        )

    def noise(self, noise: Noise) -> results.NoiseResult:
        """Computes voltage noise density, per-device contributions, and input noise.

        Raises:
            SimulationError: If the noise output is not a voltage or the circuit
                contains no independent source for input referral.
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
        """Formats the `noise` output for ngspice as ``v(out)`` or ``v(out,ref)``.

        Raises:
            SimulationError: If the noise output does not refer to a voltage node.
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
        """Extracts nonzero per-device noise sources and their spectral contributions.

        Raises:
            SimulationError: If ngspice reports noise for a device absent from the
                netlist.
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


def _index_flat[T](entries: Iterable[tuple[str, T]], what: str) -> dict[str, T]:
    """Maps each path's flattened ngspice name to its value, rejecting collisions."""
    indexed: dict[str, T] = {}
    for path, value in entries:
        flat_name = path.replace("/", "_")
        if flat_name in indexed:
            message = f"{what} `{path}` collides on flattened name `{flat_name}`"
            raise SimulationError(message)
        indexed[flat_name] = value
    return indexed


def _flattened(
    circuit: ir.Circuit, overrides: Mapping[str, Mapping[str, float]]
) -> tuple[ir.Circuit, dict[str, str]]:
    """Flattens `circuit` into devices with parameter `overrides` applied.

    Returns the flattened circuit and a mapping from underscore-joined flattened device
    identifiers back to their original hierarchical paths.
    """
    flat = export.flatten(circuit)
    paths = _index_flat(
        (
            (path, path)
            for path, instance in circuit.walk()
            if isinstance(instance.of, ir.DeviceOf)
        ),
        "device",
    )
    replaced = _index_flat(overrides.items(), "override")
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


def _devices(flat: ir.Circuit, paths: Mapping[str, str]) -> dict[str, str]:
    """Maps each ngspice noise-vector device name to its original hierarchical path."""
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
    """Returns a copy of `found` with `value` written at nested path `keys`."""
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
    """Pairs each ngspice-readable unknown from `unknowns` with its vector name."""
    read = []
    for unknown in unknowns:
        try:
            read.append((unknown, export.vector(circuit, unknown)))
        except JoltyError:
            continue
    return read


def _options(options: Options, analysis: Analysis) -> str:
    """Builds the ngspice ``.options`` line from `options` and any transient method."""
    given = [
        f"{SPICE_OPTIONS[name]}={value!r}"
        for name, value in ((name, getattr(options, name)) for name in SPICE_OPTIONS)
        if value is not None
    ]
    if isinstance(analysis, Transient):
        given.append(_METHODS[analysis.method])
    return f".options {' '.join(given)}" if given else ngspice.NO_OPTIONS


def _table(
    raw: ngspice.Result, read: Sequence[tuple[results.Unknown, str]]
) -> results.Real:
    """Extracts the real values of each vector in `read` as columns of a 2D array."""
    return np.column_stack([raw.real(vector) for _, vector in read]).reshape(
        -1, len(read)
    )


def _source(flat: ir.Circuit, path: str) -> tuple[ir.Instance, results.Nature]:
    """Resolves the independent source at `path` in `flat` and its electrical nature.

    The returned nature is ``"voltage"`` or ``"current"``.

    Raises:
        SimulationError: If `path` does not refer to an independent source.
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
    """Builds the ngspice ``dc`` command, swept levels, and source nature for `sweep`.

    Raises:
        SimulationError: If the swept instance is not an independent source or the
            sweep values are not uniformly spaced.
    """
    found, nature = _source(flat, sweep.source)
    dimension = unknowns.DIMENSIONS[nature]
    given = [level.to(dimension, "a sweep level") for level in sweep.values]
    levels = np.array(given)
    if len(levels) < 2:  # noqa: PLR2004 - minimum two sweep endpoints
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
"""Regular expressions parsing ngspice ``rusage`` transient iteration and timepoint
counts."""


def _excited(flat: ir.Circuit, analysis: Analysis) -> str | None:
    """Returns the flattened source name excited by an AC or noise analysis, if any."""
    match analysis:
        case Ac():
            return _source(flat, analysis.source)[0].name
        case Noise():
            return _referred(flat, analysis).name
    return None


def _referred(flat: ir.Circuit, noise: Noise) -> ir.Instance:
    """Returns the reference source for `noise`, or the circuit's first source.

    Raises:
        SimulationError: If the named source is not independent, or if `noise` omits a
            source and the circuit contains no independent sources.
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
    """Builds single-frequency ngspice commands for `analysis` over `frequencies`."""
    return [
        f"{analysis} lin 1 {frequency!r} {frequency!r}" for frequency in frequencies
    ]


def _difference(output: str) -> tuple[str, str]:
    """Splits a ``v(out)`` or ``v(out,ref)`` expression into vector names."""
    nodes = output.removeprefix("v(").removesuffix(")").split(",")
    first, *rest = nodes
    return f"v({first})", f"v({rest[0]})" if rest else ""

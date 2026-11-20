# Copyright 2026 Maxence Faldor
"""Elaborated circuits and differentiable analysis functions (JVP, VJP, batched VJP)."""

import functools
import json
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from jolty import _core, checks, devices, ir
from jolty.builder import Component, circuit
from jolty.devices import Device, ParameterValue
from jolty.errors import ElaborationError, SimulationError
from jolty.plan import (
    Ac,
    DcSweep,
    OperatingPoint,
    Options,
    PeriodicSteadyState,
    Transient,
    differentiated,
)
from jolty.simulate._decode import (
    elaborating,
    engine_pool,
    kelvins,
    points_of,
    substitutes_json,
    swept_dimension,
)
from jolty.units import VOLTAGE, Quantity

type Parameters = Mapping[str, Mapping[str, float]]
"""Device parameter mapping keyed by instance path and parameter path in SI units."""

type Outputs = dict[str, npt.NDArray[np.float64] | npt.NDArray[np.complex128] | float]
"""Analysis output mapping from unknown name to values, plus `PERIOD` if autonomous."""

type Differentiable = OperatingPoint | DcSweep | Ac | Transient | PeriodicSteadyState
"""Analysis types supported by [`Function`][jolty.simulate.Function]."""

type Gradient = dict[str, dict[str, float | complex]]
"""Parameter gradient mapping keyed by instance path and parameter name."""

type Pullback = Callable[[Mapping[str, npt.ArrayLike]], Gradient]
"""Adjoint pullback from [`vjp`][jolty.simulate.vjp] returning parameter gradients."""

PERIOD: Final = "the period"
"""Output key for an oscillator's period in seconds (distinct from unknown names)."""


class Elaborated:
    """Elaborated circuit shared across simulations with different parameter values.

    Devices evaluate at a single temperature (27 °C by default) with any models in
    `substitutes` replacing matching component instances. Creating a simulation via
    [`Simulation.from_elaborated`][jolty.simulate.Simulation.from_elaborated] applies
    parameter overrides and rebuilds device kernels bit-identically without repeating
    topology elaboration. Parameter values that change matrix structure (such as a
    resistance below 1 Ω introducing a branch-current unknown) cache a dedicated
    restructured elaboration shared across matching runs.

    Examples:
        >>> from jolty import Builder, Role, Simulation, component, devices
        >>> from jolty.plan import OperatingPoint
        >>> from jolty.units import V, kohm
        >>> @component
        ... def divider(c: Builder) -> None:
        ...     ground = c.port("ground", Role.GROUND)
        ...     source = c.place(devices.voltage_source(5 * V), name="V1")
        ...     upper = c.place(devices.resistor(1 * kohm), name="R1")
        ...     lower = c.place(devices.resistor(1 * kohm), name="R2")
        ...     c.connect(source.positive, upper.a)
        ...     c.connect(upper.b, lower.a, name="out")
        ...     c.connect(lower.b, source.negative, ground)
        >>> elaborated = Elaborated(divider())
        >>> sample = Simulation.from_elaborated(
        ...     elaborated, {"R1": {"resistance": 3 * kohm}}
        ... )
        >>> sample.run(OperatingPoint()).voltage("out")
        1.25
    """

    circuit: ir.Circuit
    """Canonical circuit representation."""

    temperature: Quantity | None
    """Ambient temperature for device evaluation, defaulting to 27 °C when None."""

    substituted: frozenset[str]
    """Instance paths of components replaced by substitute models."""

    def __init__(
        self,
        circuit_or_top: ir.Circuit | Component,
        temperature: Quantity | None = None,
        substitutes: Mapping[str, Device] | None = None,
    ) -> None:
        """Elaborates a circuit or a circuit rooted at a top component.

        Raises:
            ElaborationError: Lists every problem reported by the engine, naming the
                instance and build-script line.
            DimensionError: When `temperature` does not carry temperature dimensions.
        """
        self.circuit = (
            circuit_or_top
            if isinstance(circuit_or_top, ir.Circuit)
            else circuit(circuit_or_top)
        )
        self.temperature = temperature
        self.substituted = frozenset(substitutes or {})
        self._kelvins = kelvins(temperature)
        with elaborating():
            self._core = _core.Elaborated(
                self.circuit.model_dump_json(), substitutes_json(substitutes)
            )
        self._structures: dict[_core.StructureKey, _core.Elaborated] = {}
        self._structure_lock = threading.Lock()

    def simulator(
        self, overrides: Mapping[str, Mapping[str, ParameterValue]] | None = None
    ) -> _core.Simulator:
        """Builds an engine simulator from this elaborated circuit with `overrides`.

        Args:
            overrides: Optional nested mapping of instance paths to parameter values
                validated against each device's declared parameter dimensions.

        Returns:
            Engine `Simulator` initialized with `overrides` at the elaborated
            temperature.

        Raises:
            DimensionError: When an override for a known parameter has the wrong
                physical dimension or kind.
            ElaborationError: Lists unknown parameters in `overrides` or devices whose
                overridden parameters fall out of range.
        """
        return self._simulator_si(devices.convert_overrides(self.circuit, overrides))

    def _simulator_si(
        self, overrides: Mapping[str, Mapping[str, float]] | None = None
    ) -> _core.Simulator:
        """Builds an engine simulator from pre-converted SI base-unit `overrides`."""
        with elaborating():
            values = self._core.values(
                json.dumps(overrides or {}, allow_nan=False), self._kelvins
            )
            key = self._core.structure_of(values)
            shared = self._core
            if not key.unchanged:
                with self._structure_lock:
                    cached = self._structures.get(key)
                    if cached is None:
                        cached = self._core.restructured(values)
                        self._structures[key] = cached
                    shared = cached
            return _core.Simulator.from_elaborated(shared, values)

    def check(self, parameters: Mapping[str, Iterable[str]]) -> None:
        """Checks that the circuit defines every instance and parameter in `parameters`.

        Args:
            parameters: Mapping from instance paths to iterables of parameter names.

        Raises:
            ElaborationError: Lists missing instances and unrecognized parameter names.
        """
        overrides = {
            path: dict.fromkeys(names, 0.0) for path, names in parameters.items()
        }
        with elaborating():
            self._core.values(json.dumps(overrides, allow_nan=False), self._kelvins)

    def variants(self, runs: Iterable[Parameters]) -> _core.Variants:
        """Compiles explicit parameter `runs` into engine variants for this circuit.

        Args:
            runs: Per-run parameter override mappings in SI base units.

        Returns:
            Compiled explicit engine `Variants` at the elaborated temperature.

        Raises:
            ElaborationError: Lists unrecognized parameter names referenced by `runs`.
        """
        listed: list[tuple[str, float | None]] = [
            (json.dumps(run, allow_nan=False), self._kelvins) for run in runs
        ]
        with elaborating():
            return _core.Variants.explicit(self._core, listed)


class Function:
    """Differentiable circuit analysis as a callable function of device parameters.

    Calling ``function(parameters)`` evaluates the analysis at the supplied overrides
    and returns each unknown's values across analysis points. Forward and reverse
    differentiation work via [`jvp`][jolty.simulate.jvp] and
    [`vjp`][jolty.simulate.vjp], mirroring ``jax.jvp`` and ``jax.vjp``. The circuit
    elaborates once and rebuilds device kernels per call with the requested overrides.
    """

    elaborated: Elaborated
    """Elaborated circuit shared across calls."""

    analysis: Differentiable
    """Analysis evaluated by each call."""

    options: Options
    """Solver options applied to each evaluation."""

    def __init__(
        self,
        circuit_or_top: ir.Circuit | Component,
        analysis: Differentiable,
        options: Options | None = None,
    ) -> None:
        """Initializes a differentiable analysis function for `circuit_or_top`.

        Raises:
            ElaborationError: Lists every problem reported during elaboration.
            TypeError: When `analysis` is not a differentiable analysis type.
        """
        if not isinstance(
            analysis, OperatingPoint | DcSweep | Ac | Transient | PeriodicSteadyState
        ):
            message = (
                "a function runs an operating point, a DC sweep, a small-signal "
                f"response, a transient or a periodic steady state, not "
                f"{type(analysis).__name__}"
            )
            raise TypeError(message)
        self.elaborated = Elaborated(circuit_or_top)
        self.analysis = analysis
        self.options = Options() if options is None else options

    def __call__(self, parameters: Parameters) -> Outputs:
        """Evaluates the analysis at `parameters` and returns the unknown outputs.

        Raises:
            ElaborationError: Lists unknown parameters or out-of-range device values.
            SimulationError: When the analysis fails to solve.
        """
        core, outcome = self._run(
            parameters,
            lambda core, analysis, settings: core.run(
                analysis, settings, engine_pool()
            ),
        )
        return self._outputs(core, outcome)

    def _encoded(self) -> str:
        """Encodes the differentiated analysis specification as JSON."""
        of = None if isinstance(self.analysis, OperatingPoint) else self.analysis
        dimension = (
            swept_dimension(self.elaborated.circuit, of)
            if isinstance(of, DcSweep)
            else VOLTAGE
        )
        return json.dumps(differentiated(of, dimension), allow_nan=False)

    def _settings(self) -> str:
        """Encodes solver options as JSON for the engine."""
        return json.dumps(self.options.to_json(), allow_nan=False)

    def _outputs(self, core: _core.Simulator, outcome: _core.AnalysisResult) -> Outputs:
        """Extracts unknown arrays and optional oscillator period from `outcome`."""
        rows, period = _values(outcome)
        found: Outputs = {
            name: rows[:, column] for column, (name, _) in enumerate(core.unknowns)
        }
        if period is not None and self.oscillates:
            found[PERIOD] = float(period)
        return found

    @property
    def oscillates(self) -> bool:
        """True when the analysis is autonomous PSS whose outputs include the period."""
        analysis = self.analysis
        return isinstance(analysis, PeriodicSteadyState) and analysis.period is None

    def _named(self, parameters: Parameters) -> list[tuple[str, str]]:
        """Flattens `parameters` into ordered `(instance, parameter)` pairs."""
        return [(path, name) for path, named in parameters.items() for name in named]

    def _run[T](
        self, parameters: Parameters, run: Callable[[_core.Simulator, str, str], T]
    ) -> tuple[_core.Simulator, T]:
        """Creates a simulator at `parameters` and executes `run` against it."""
        core = self.elaborated._simulator_si(parameters)  # noqa: SLF001 - same-module SI fast path for Function
        try:
            return core, run(core, self._encoded(), self._settings())
        except _core.SimulationError as error:
            raise SimulationError(str(error)) from None

    def pushed(
        self, parameters: Parameters, tangents: Parameters
    ) -> tuple[Outputs, Outputs]:
        """Computes primal outputs and forward-mode directional derivatives (JVP).

        Args:
            parameters: Parameter overrides defining the evaluation point.
            tangents: Directional parameter perturbations in SI base units.

        Returns:
            Pair of primal `Outputs` and forward-pushed tangent `Outputs`.
        """
        names = self._named(parameters)
        given = {(path, name) for path, named in tangents.items() for name in named}
        if unknown := sorted(given - set(names)):
            message = f"a tangent moves only the parameters given, not {unknown}"
            raise ValueError(message)
        tangent = [float(tangents.get(path, {}).get(name, 0.0)) for path, name in names]
        core, found = self._run(
            parameters,
            lambda core, analysis, settings: core.jvp(
                analysis, names, [tangent], settings, engine_pool()
            ),
        )
        outputs = self._outputs(core, found["result"])
        moved = found["tangents"][:, :, 0]
        pushed: Outputs = {
            name: moved[:, column] for column, (name, _) in enumerate(core.unknowns)
        }
        if (period := found["period"]) is not None and self.oscillates:
            pushed[PERIOD] = float(period[0])
        return outputs, pushed

    def pulled(self, parameters: Parameters) -> tuple[Outputs, Pullback]:
        """Computes primal outputs and a reverse-mode adjoint pullback (VJP).

        Args:
            parameters: Parameter overrides defining the evaluation point.

        Returns:
            Pair of primal `Outputs` and adjoint `Pullback` callable.
        """
        names = self._named(parameters)
        core, (outcome, pullback) = self._run(
            parameters,
            lambda core, analysis, settings: core.vjp(
                analysis, names, settings, engine_pool()
            ),
        )
        outputs = self._outputs(core, outcome)
        columns = {name: column for column, (name, _) in enumerate(core.unknowns)}
        points = _values(outcome)[0].shape[0]
        phasors = isinstance(self.analysis, Ac)
        settings = self._settings()

        def pull(cotangent: Mapping[str, npt.ArrayLike]) -> Gradient:
            """Evaluates parameter gradients for `cotangent`, keyed by instance."""
            values, period = _cotangents(
                cotangent,
                columns,
                (points, len(columns)),
                phasors=phasors,
                oscillates=self.oscillates,
            )
            try:
                found = core.pull_back(
                    pullback, values, float(period), settings, engine_pool()
                )
            except _core.SimulationError as error:
                raise SimulationError(str(error)) from None
            pulled: Gradient = {}
            for (path, name), value in zip(names, found.tolist(), strict=True):
                pulled.setdefault(path, {})[name] = value
            return pulled

        return outputs, pull

    @functools.cached_property
    def unknowns(self) -> tuple[str, ...]:
        """Circuit unknown names in engine order."""
        return tuple(name for name, _ in self.elaborated.simulator().unknowns)

    @property
    def points(self) -> int:
        """Number of points per run in a batched evaluation."""
        return points_of(self.analysis)

    def batch(self, names: Sequence[tuple[str, str]], values: npt.ArrayLike) -> Outputs:
        """Evaluates the analysis across a batch of parameter rows.

        Args:
            names: Ordered `(instance, parameter)` pairs naming matrix columns.
            values: Two-dimensional array of parameter values shaped
                `(runs, len(names))`.

        Returns:
            Batched unknown `Outputs` indexed by run and analysis point.
        """
        outputs, _ = self._batched(names, values, [])
        return outputs

    def batch_vjp(
        self, names: Sequence[tuple[str, str]], values: npt.ArrayLike
    ) -> tuple[Outputs, BatchPullback]:
        """Evaluates a batch of runs and returns its outputs and `BatchPullback`.

        Args:
            names: Ordered `(instance, parameter)` pairs naming matrix columns.
            values: Two-dimensional array of parameter values shaped
                `(runs, len(names))`.

        Returns:
            Pair of batched `Outputs` and `BatchPullback` over all runs.
        """
        outputs, core = self._batched(names, values, list(names))
        pullback = BatchPullback(
            core,
            self.unknowns,
            self.points,
            phasors=isinstance(self.analysis, Ac),
            oscillates=self.oscillates,
            settings=self._settings(),
        )
        return outputs, pullback

    def _batched(
        self,
        names: Sequence[tuple[str, str]],
        values: npt.ArrayLike,
        parameters: list[tuple[str, str]],
    ) -> tuple[Outputs, _core.BatchPullback]:
        """Executes a batch at `values` for `names` and returns outputs and pullback."""
        rows = np.asarray(values, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(names):  # noqa: PLR2004 - a matrix
            message = f"values are a row of {len(names)} names by run, not {rows.shape}"
            raise ValueError(message)
        runs: list[dict[str, dict[str, float]]] = []
        for row in rows.tolist():
            run: dict[str, dict[str, float]] = {}
            for (path, name), value in zip(names, row, strict=True):
                run.setdefault(path, {})[name] = value
            runs.append(run)
        variants = self.elaborated.variants(runs)
        base = self.elaborated.simulator()
        columns = list(range(len(base.unknowns)))
        try:
            found, pullback = base.batch_vjp(
                self._encoded(),
                self._settings(),
                parameters,
                variants,
                (0, len(runs)),
                columns,
                engine_pool(),
            )
        except _core.SimulationError as error:
            raise SimulationError(str(error)) from None
        if found["problems"]:
            problems = found["problems"][min(found["problems"])]
            raise ElaborationError([checks.problem(each) for each in problems])
        if found["errors"]:
            first = min(found["errors"])
            message = f"batch run {first}: {found['errors'][first]}"
            raise SimulationError(message)
        traces = found["traces"]
        outputs: Outputs = {
            name: traces[:, :, column] for column, name in enumerate(self.unknowns)
        }
        if self.oscillates:
            outputs[PERIOD] = found["periods"]
        return outputs, pullback


class BatchPullback:
    """Per-run adjoint pullback from [`batch_vjp`][jolty.simulate.Function.batch_vjp].

    Calling a `BatchPullback` with output cotangents returns parameter gradients for
    each selected batch row.
    """

    def __init__(  # noqa: PLR0913 - engine pullback handle and output layout metadata
        self,
        core: _core.BatchPullback,
        unknowns: Sequence[str],
        points: int,
        *,
        phasors: bool,
        oscillates: bool,
        settings: str,
    ) -> None:
        """Wraps an engine `core` batch pullback for `unknowns` over `points`."""
        self._core = core
        self._columns = {name: column for column, name in enumerate(unknowns)}
        self._points = points
        self._phasors = phasors
        self._oscillates = oscillates
        self._settings = settings

    def __call__(
        self, cotangent: Mapping[str, npt.ArrayLike], runs: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Evaluates parameter gradients for `cotangent` by run and parameter."""
        rows = np.atleast_1d(
            np.arange(self._core.runs) if runs is None else np.asarray(runs, np.intp)
        )
        values, periods = _cotangents(
            cotangent,
            self._columns,
            (len(rows), self._points, len(self._columns)),
            phasors=self._phasors,
            oscillates=self._oscillates,
        )
        try:
            return self._core.pull_back(
                rows.tolist(), values, periods.tolist(), self._settings, engine_pool()
            )
        except _core.SimulationError as error:
            raise SimulationError(str(error)) from None


def _cotangents(
    cotangent: Mapping[str, npt.ArrayLike],
    columns: Mapping[str, int],
    shape: tuple[int, ...],
    *,
    phasors: bool,
    oscillates: bool,
) -> tuple[
    npt.NDArray[np.float64] | npt.NDArray[np.complex128], npt.NDArray[np.float64]
]:
    """Packs named `cotangent` arrays into engine columns plus period cotangents."""
    values: npt.NDArray[np.float64] | npt.NDArray[np.complex128] = (
        np.zeros(shape, np.complex128) if phasors else np.zeros(shape)
    )
    periods = np.zeros(shape[:-2])
    for name, given in cotangent.items():
        if name == PERIOD and oscillates:
            periods[...] = np.asarray(given, dtype=np.float64)
        elif name in columns:
            values[..., columns[name]] = given
        else:
            message = f"the function gives no output `{name}`"
            raise ValueError(message)
    return values, periods


def _values(
    outcome: _core.AnalysisResult,
) -> tuple[npt.NDArray[np.float64] | npt.NDArray[np.complex128], float | None]:
    """Extracts the 2D unknown array and optional period from an engine `outcome`."""
    match outcome["kind"]:
        case "periodic_steady_state":
            return np.atleast_2d(outcome["values"]), outcome["period"]
        case "operating_point" | "dc_sweep" | "ac" | "transient":
            return np.atleast_2d(outcome["values"]), None
        case other:
            message = f"a function runs no {other} analysis"
            raise SimulationError(message)


def function(
    circuit_or_top: ir.Circuit | Component,
    analysis: Differentiable,
    options: Options | None = None,
) -> Function:
    """Creates a differentiable [`Function`][jolty.simulate.Function] for `analysis`.

    Calling ``function(circuit, analysis)(parameters)`` evaluates `analysis` with
    `options` at `parameters` and returns unknown outputs.

    Args:
        circuit_or_top: Circuit IR or top-level component to elaborate.
        analysis: Differentiable analysis specification evaluated per call.
        options: Optional solver tolerances, iteration limits, and nodesets.

    Returns:
        Callable `Function` evaluating `analysis` at parameter overrides.

    Raises:
        ElaborationError: Lists every problem reported during circuit elaboration.
    """
    return Function(circuit_or_top, analysis, options)


def jvp(
    function: Function, parameters: Parameters, tangents: Parameters
) -> tuple[Outputs, Outputs]:
    """Computes outputs of `function` at `parameters` and derivatives along `tangents`.

    Mirrors ``jax.jvp(f, primals, tangents)``: `parameters` sets the differentiation
    point and `tangents` specifies SI-unit perturbations, holding omitted parameters
    fixed. Exact derivatives are computed by the direct method with one linear solve
    per analysis point.

    Args:
        function: Differentiable circuit analysis function to evaluate.
        parameters: Parameter overrides defining the primal evaluation point.
        tangents: Directional parameter perturbations in SI base units.

    Returns:
        Pair of primal `Outputs` and directional derivative `Outputs`.

    Raises:
        ValueError: When `tangents` references a parameter absent from `parameters`.
        ElaborationError: When parameter overrides fail circuit elaboration.
        SimulationError: When the analysis or Jacobian linear solve fails.
    """
    return function.pushed(parameters, tangents)


def vjp(function: Function, parameters: Parameters) -> tuple[Outputs, Pullback]:
    """Computes outputs of `function` at `parameters` and its adjoint pullback.

    Mirrors ``jax.vjp(f, primals)``: the pullback accepts per-output cotangent arrays
    (omitted outputs default to zero) and returns parameter gradients keyed by
    instance and parameter name. The adjoint method solves one transposed linear
    system per analysis point regardless of parameter count. For a scalar loss, pass
    its derivatives with respect to outputs as cotangents to obtain the parameter
    gradient. Over frequency, cotangents and gradients are complex ``v^T dx/dp``, so
    a real loss of phasors ``a + jb`` uses the real part of the pullback applied to
    ``dL/da - j dL/db``.

    Args:
        function: Differentiable circuit analysis function to evaluate.
        parameters: Parameter overrides defining the primal evaluation point.

    Returns:
        Pair of primal `Outputs` and reverse-mode adjoint `Pullback`.

    Raises:
        ElaborationError: When parameter overrides fail circuit elaboration.
        SimulationError: When the analysis fails, or for transient analyses containing
            delay lines or discrete logic unsupported by the adjoint solver.
    """
    return function.pulled(parameters)

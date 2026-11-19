# Copyright 2026 Maxence Faldor
"""Elaborated circuits and differentiable analysis functions (JVP, VJP, batched VJP)."""

import functools
import json
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from jolty import _core, checks, ir
from jolty.builder import Component, circuit
from jolty.devices import Device
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
    swept_level,
)
from jolty.units import VOLTAGE, Quantity

type Parameters = Mapping[str, Mapping[str, float]]
"""Numbers of devices' parameters, by instance path, such as ``X1/R1``, then by
parameter path, such as ``resistance`` or ``waveform.dc``, in SI base units, as a
simulation's overrides name them."""

type Outputs = dict[str, npt.NDArray[np.float64] | npt.NDArray[np.complex128] | float]
"""What a [`Function`][jolty.simulate.Function] gives: each unknown's values at every
point of its analysis, by the unknown's name, in SI base units, phasors over
frequency; and an oscillator's period under [`PERIOD`][jolty.simulate.PERIOD]."""

type Differentiable = OperatingPoint | DcSweep | Ac | Transient | PeriodicSteadyState
"""An analysis a [`Function`][jolty.simulate.Function] runs."""

type Gradient = dict[str, dict[str, float | complex]]
"""Each parameter's value of a cotangent, by instance then parameter."""

type Pullback = Callable[[Mapping[str, npt.ArrayLike]], Gradient]
"""A function's pullback, as [`vjp`][jolty.simulate.vjp] gives it: each parameter's
value of a cotangent on its outputs, by instance then parameter."""

PERIOD: Final = "the period"
"""The key of an oscillator's period among a function's outputs, in seconds: no
unknown's name, as names hold no whitespace."""


class Elaborated:
    """A circuit elaborated once, which simulations of it at any parameters share.

    Its devices run at one temperature, 27 °C unless given, with the models of
    `substitutes` standing in for component instances. A simulation made from it by
    [`Simulation.from_elaborated`][jolty.simulate.Simulation.from_elaborated] replaces
    some of its devices' numbers, as a Monte Carlo sample does, and makes its devices
    again from them, as elaborating the circuit with them would, bit for bit, without
    elaborating it anew.
    Numbers that give a device another structure, as a resistance below 1 Ω carries its
    current as an unknown of its own, run on the circuit elaborated once for that
    structure, which every simulation whose numbers give it shares.

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
        ...     elaborated, {"R1": {"resistance": 3000.0}}
        ... )
        >>> sample.run(OperatingPoint()).voltage("out")
        1.25
    """

    circuit: ir.Circuit
    """The circuit, in the canonical format."""

    temperature: Quantity | None
    """The temperature its devices run at; 27 °C when it is None."""

    substituted: frozenset[str]
    """The paths of the component instances a model stands in for."""

    def __init__(
        self,
        circuit_or_top: ir.Circuit | Component,
        temperature: Quantity | None = None,
        substitutes: Mapping[str, Device] | None = None,
    ) -> None:
        """Elaborates a circuit, or the circuit with a component at its top.

        Raises:
            ElaborationError: With every problem the engine finds, each naming the
                instance and the line of the build script that placed it.
            DimensionError: When the temperature is not one.
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
        self, overrides: Mapping[str, Mapping[str, float]] | None = None
    ) -> _core.Simulator:
        """The engine's simulator of the circuit with the numbers of `overrides`.

        Raises:
            ElaborationError: With every number `overrides` names that the circuit's
                devices do not write, and every device whose numbers they put out of
                range.
        """
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
        """Checks that the circuit has each instance and number `parameters` names.

        Raises:
            ElaborationError: With every instance the circuit lacks and every number its
                devices do not write.
        """
        overrides = {
            path: dict.fromkeys(names, 0.0) for path, names in parameters.items()
        }
        with elaborating():
            self._core.values(json.dumps(overrides, allow_nan=False), self._kelvins)

    def variants(self, runs: Iterable[Parameters]) -> _core.Variants:
        """The engine's variants of a batch of `runs`, each the numbers of one run.

        Raises:
            ElaborationError: With every number a run names that the circuit's
                devices do not write.
        """
        listed: list[tuple[str, float | None]] = [
            (json.dumps(run, allow_nan=False), self._kelvins) for run in runs
        ]
        with elaborating():
            return _core.Variants.explicit(self._core, listed)


class Function:
    """An analysis of a circuit as a function of its devices' parameters.

    Called with parameters, ``f(params)``, it gives the analysis's outputs at them, each
    unknown's values over the analysis's points, the circuit's other parameters as they
    are; [`jvp`][jolty.simulate.jvp] and [`vjp`][jolty.simulate.vjp] differentiate it,
    as ``jax.jvp`` and ``jax.vjp`` differentiate a function.
    The circuit is elaborated once, and each call makes its devices again from the
    parameters given, as a [`Simulation`][jolty.simulate.Simulation] with them as
    overrides does.
    """

    elaborated: Elaborated
    """The circuit, elaborated once."""

    analysis: Differentiable
    """The analysis it runs."""

    options: Options
    """The options the analysis runs with."""

    def __init__(
        self,
        circuit_or_top: ir.Circuit | Component,
        analysis: Differentiable,
        options: Options | None = None,
    ) -> None:
        """The function that runs `analysis` on a circuit with `options`.

        Raises:
            ElaborationError: With every problem the engine finds in the circuit.
            TypeError: When `analysis` is of a kind a function does not run.
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

    def __call__(self, params: Parameters) -> Outputs:
        """The analysis's outputs with the parameters `params`.

        Raises:
            ElaborationError: With every number `params` names that the circuit's
                devices do not write, and every device it puts out of range.
            SimulationError: When the analysis fails.
        """
        core, outcome = self._run(
            params,
            lambda core, analysis, settings: core.run(
                analysis, settings, engine_pool()
            ),
        )
        return self._outputs(core, outcome)

    def _encoded(self) -> str:
        """The analysis, differentiated, in the engine's format."""
        of = None if isinstance(self.analysis, OperatingPoint) else self.analysis
        level = (
            swept_level(self.elaborated.circuit, of)
            if isinstance(of, DcSweep)
            else VOLTAGE
        )
        return json.dumps(differentiated(of, level), allow_nan=False)

    def _settings(self) -> str:
        """The options in the engine's format."""
        return json.dumps(self.options.to_json(), allow_nan=False)

    def _outputs(self, core: _core.Simulator, outcome: _core.AnalysisResult) -> Outputs:
        """The outputs of `outcome`, which `core` found."""
        rows, period = _values(outcome)
        found: Outputs = {
            name: rows[:, column] for column, (name, _) in enumerate(core.unknowns)
        }
        if period is not None and self.oscillates:
            found[PERIOD] = float(period)
        return found

    @property
    def oscillates(self) -> bool:
        """Whether its analysis is an oscillator's, whose outputs hold its period."""
        analysis = self.analysis
        return isinstance(analysis, PeriodicSteadyState) and analysis.period is None

    def _named(self, params: Parameters) -> list[tuple[str, str]]:
        """The parameters `params` names, each its instance's path and its name."""
        return [(path, name) for path, named in params.items() for name in named]

    def _run[T](
        self, params: Parameters, run: Callable[[_core.Simulator, str, str], T]
    ) -> tuple[_core.Simulator, T]:
        """The simulator at `params`, and what `run` finds of it."""
        core = self.elaborated.simulator(params)
        try:
            return core, run(core, self._encoded(), self._settings())
        except _core.SimulationError as error:
            raise SimulationError(str(error)) from None

    def pushed(
        self, params: Parameters, tangents: Parameters
    ) -> tuple[Outputs, Outputs]:
        """As [`jvp`][jolty.simulate.jvp] finds it."""
        names = self._named(params)
        given = {(path, name) for path, named in tangents.items() for name in named}
        if unknown := sorted(given - set(names)):
            message = f"a tangent moves only the parameters given, not {unknown}"
            raise ValueError(message)
        tangent = [float(tangents.get(path, {}).get(name, 0.0)) for path, name in names]
        core, found = self._run(
            params,
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

    def pulled(self, params: Parameters) -> tuple[Outputs, Pullback]:
        """As [`vjp`][jolty.simulate.vjp] finds it."""
        names = self._named(params)
        core, (outcome, pullback) = self._run(
            params,
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
            """Each parameter's value of `cotangent`, by instance then parameter."""
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
        """The names of the circuit's unknowns, its outputs, in the engine's order."""
        return tuple(name for name, _ in self.elaborated.simulator().unknowns)

    @property
    def points(self) -> int:
        """How many points each output of a run of a batch has."""
        return points_of(self.analysis)

    def batch(self, names: Sequence[tuple[str, str]], values: npt.ArrayLike) -> Outputs:
        """The analysis's outputs on each run of a batch, shaped by run and point."""
        outputs, _ = self._batched(names, values, [])
        return outputs

    def batch_vjp(
        self, names: Sequence[tuple[str, str]], values: npt.ArrayLike
    ) -> tuple[Outputs, BatchPullback]:
        """The outputs of a [`batch`][jolty.simulate.Function.batch] and pullback."""
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
        """The outputs of a batch at `values` of `names`, and the engine's pullback."""
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
    """The pullback of a [`batch`][jolty.simulate.Function.batch], a run each.

    [`Function.batch_vjp`][jolty.simulate.Function.batch_vjp] gives it; called with a
    cotangent on the batch's outputs, it gives each row's value on each parameter.
    """

    def __init__(  # noqa: PLR0913 - what a batch's pullback reads of its function
        self,
        core: _core.BatchPullback,
        unknowns: Sequence[str],
        points: int,
        *,
        phasors: bool,
        oscillates: bool,
        settings: str,
    ) -> None:
        """The pullback `core` of a batch of a function of `unknowns` at `points`."""
        self._core = core
        self._columns = {name: column for column, name in enumerate(unknowns)}
        self._points = points
        self._phasors = phasors
        self._oscillates = oscillates
        self._settings = settings

    def __call__(
        self, cotangent: Mapping[str, npt.ArrayLike], runs: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Each row's value of `cotangent` on each parameter, by row then parameter."""
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
    """The values `cotangent` gives the unknowns at `columns`, and the periods."""
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
    """The values of every unknown at each point of `outcome`, and its period."""
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
    """The analysis `analysis` of a circuit as a function of its parameters.

    ``function(circuit, analysis)(params)`` runs it with `options` at the parameters
    `params` and gives its outputs; see [`Function`][jolty.simulate.Function].

    Raises:
        ElaborationError: With every problem the engine finds in the circuit.
    """
    return Function(circuit_or_top, analysis, options)


def jvp(
    f: Function, params: Parameters, tangents: Parameters
) -> tuple[Outputs, Outputs]:
    """The outputs of `f` at `params`, and their derivative along `tangents`.

    As ``jax.jvp(f, primals, tangents)``: the parameters `params` names are those
    differentiated, and `tangents` moves some of them, in SI base units, the others
    held. The derivatives, by output as the outputs are, are exact, by the direct
    method: one solve at each point of the analysis.

    Raises:
        ValueError: When a tangent moves a parameter `params` does not give.
        ElaborationError: As calling `f` raises it.
        SimulationError: When the analysis or a Jacobian's solve fails.
    """
    return f.pushed(params, tangents)


def vjp(f: Function, params: Parameters) -> tuple[Outputs, Pullback]:
    """The outputs of `f` at `params`, and its pullback.

    As ``jax.vjp(f, primals)``: the pullback takes a cotangent on the outputs, by
    output as they are, each unknown's at every point, those it leaves out zero, and
    gives its value on each parameter of `params`, by instance then parameter, by the
    adjoint method: one transposed solve at each point of the analysis, however many
    the parameters. A scalar loss's derivatives with respect to the outputs are its
    cotangent, and the pullback gives its gradient. Over frequency, the cotangents and
    their values are complex, `vᵀ dx/dp`: a real loss of the phasors `a + jb` has the
    real part of the pullback of `∂L/∂a - j ∂L/∂b`.

    Raises:
        ElaborationError: As calling `f` raises it.
        SimulationError: When the analysis fails, or over time on a circuit with delay
            lines or logic, whose points the adjoint method cannot evaluate again as
            they were.
    """
    return f.pulled(params)

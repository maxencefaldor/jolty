# Copyright 2026 Maxence Faldor
"""Elaborated circuits and differentiable analysis functions (JVP, VJP, batched VJP)."""

import functools
import json
import math
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, cast

import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

from jolty import _core, checks, devices, ir
from jolty.builder import Component, circuit
from jolty.devices import Device, ParameterValue
from jolty.errors import ElaborationError, SimulationError
from jolty.plan import (
    LOOP_GAIN,
    Ac,
    DcSweep,
    Noise,
    OperatingPoint,
    Options,
    PeriodicSteadyState,
    Stability,
    Transient,
    differentiated,
    noise_outputs,
)
from jolty.simulate._decode import (
    elaborating,
    engine_pool,
    kelvins,
    points_of,
    substitutes_json,
    swept_dimension,
)
from jolty.spice import BinBounds, SubcircuitDerivatives, SubcircuitPlacement
from jolty.units import VOLTAGE, Quantity

type Parameters = Mapping[str, Mapping[str, float]]
"""Device parameter mapping keyed by instance path and parameter path in SI units."""

type Outputs = dict[str, npt.NDArray[np.float64] | npt.NDArray[np.complex128] | float]
"""Analysis output mapping from output name to values, plus `PERIOD` if autonomous."""

type Differentiable = (
    OperatingPoint | DcSweep | Ac | Stability | Noise | Transient | PeriodicSteadyState
)
"""Analysis types supported by [`Function`][jolty.simulate.Function]."""

type Gradient = dict[str, dict[str, float | complex]]
"""Parameter gradient mapping keyed by instance path and parameter name."""

type Pullback = Callable[[Mapping[str, npt.ArrayLike]], Gradient]
"""Adjoint pullback from [`vjp`][jolty.simulate.vjp] returning parameter gradients."""

PERIOD: Final = "the period"
"""Output key for an oscillator's period in seconds (distinct from unknown names)."""

_BASELINE_RELATIVE_TOLERANCE: Final = 1e-12
"""Relative tolerance when checking baseline subcircuit parameters against a circuit."""

_BASELINE_ABSOLUTE_TOLERANCE: Final = 1e-15
"""Absolute tolerance when checking baseline subcircuit parameters against a circuit."""


def _split_canonical(
    value: JsonValue,
    path: str,
    floats: dict[str, float],
    discrete: dict[str, object],
) -> None:
    """Separates continuous `float` leaves from discrete values in defaulted JSON."""
    if type(value) is float:
        floats[path] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else key
            _split_canonical(item, child, floats, discrete)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{path}.{index}" if path else str(index)
            _split_canonical(item, child, floats, discrete)
    else:
        discrete[path] = value


def _canonical_parameters(
    model: str, raw: Mapping[str, JsonValue]
) -> tuple[dict[str, JsonValue], dict[str, float], dict[str, object]]:
    """Defaults `raw` via the engine schema and splits continuous vs discrete leaves."""
    canonical = cast(
        "dict[str, JsonValue]",
        json.loads(_core.defaulted(model, json.dumps(raw, allow_nan=False))),
    )
    floats: dict[str, float] = {}
    discrete: dict[str, object] = {}
    _split_canonical(canonical, "", floats, discrete)
    return canonical, floats, discrete


def _component_devices(
    placed: Component, prefix: str
) -> dict[str, tuple[str, Mapping[str, JsonValue]]]:
    """Collects `(model, parameters)` for every primitive device inside `placed`."""
    child_map = {child.name: child for child in placed.children}
    found: dict[str, tuple[str, Mapping[str, JsonValue]]] = {}
    for instance in placed.definition.instances:
        path = f"{prefix}{instance.name}"
        match instance.of:
            case ir.DeviceOf(model=model):
                found[path] = (model, instance.parameters)
            case ir.ComponentOf(component=inner):
                found |= _component_devices(child_map[inner], f"{path}{ir.SEPARATOR}")
    return found


@dataclass(frozen=True, slots=True)
class _BoundSubcircuit:
    """Validated subcircuit placement bound to a component instance path."""

    path: str
    placement: SubcircuitPlacement
    models: Mapping[str, str]
    base_floats: Mapping[str, Mapping[str, float]]
    base_discrete: Mapping[str, Mapping[str, object]]

    def expand(
        self,
        overrides: Mapping[str, float],
        *,
        with_respect_to: Sequence[str],
    ) -> tuple[dict[str, dict[str, float]], SubcircuitDerivatives]:
        """Evaluates `placement` at `overrides` and diffs continuous parameters."""
        evaluated = self.placement.derivatives(
            self.path, with_respect_to=with_respect_to, **overrides
        )
        prefix = f"{self.path}{ir.SEPARATOR}"
        current = _component_devices(evaluated.component, prefix)
        current_models = {
            device_path: model for device_path, (model, _) in current.items()
        }
        if current_models != self.models:
            message = (
                f"subcircuit `{self.path}` changed its device topology under overrides"
            )
            raise ValueError(message)
        diff: dict[str, dict[str, float]] = {}
        for device_path, (model, raw_parameters) in current.items():
            _, floats, discrete = _canonical_parameters(model, raw_parameters)
            if (
                discrete != self.base_discrete[device_path]
                or floats.keys() != self.base_floats[device_path].keys()
            ):
                message = (
                    f"subcircuit `{self.path}` changed discrete model settings on "
                    f"`{device_path}`"
                )
                raise ValueError(message)
            base_device = self.base_floats[device_path]
            changed = {
                key: value for key, value in floats.items() if value != base_device[key]
            }
            if changed:
                diff[device_path] = changed
        return diff, evaluated


def _validate_bound_paths(paths: Sequence[str]) -> None:
    """Rejects empty or nested overlapping bound subcircuit paths."""
    for path in paths:
        if not path:
            message = "subcircuit placement path must be non-empty"
            raise ValueError(message)
    for first in paths:
        prefix = f"{first}{ir.SEPARATOR}"
        for second in paths:
            if second.startswith(prefix):
                message = (
                    f"cannot bind nested subcircuit `{second}` inside bound "
                    f"subcircuit `{first}`"
                )
                raise ValueError(message)


def _bind_subcircuits(
    circuit_ir: ir.Circuit,
    subcircuits: Mapping[str, SubcircuitPlacement] | None,
) -> tuple[dict[str, _BoundSubcircuit], dict[str, dict[str, JsonValue]]]:
    """Validates and binds `subcircuits` against devices in `circuit_ir`."""
    if subcircuits is None or not subcircuits:
        return {}, {}
    _validate_bound_paths(tuple(subcircuits))
    walked = list(circuit_ir.walk())
    component_paths = {
        path for path, instance in walked if isinstance(instance.of, ir.ComponentOf)
    }
    circuit_devices = {
        path: (instance.of.model, instance.parameters)
        for path, instance in walked
        if isinstance(instance.of, ir.DeviceOf)
    }
    bound: dict[str, _BoundSubcircuit] = {}
    canonical_devices: dict[str, dict[str, JsonValue]] = {}
    for path, placement in subcircuits.items():
        if path not in component_paths:
            message = f"circuit has no subcircuit placement at `{path}`"
            raise ValueError(message)
        prefix = f"{path}{ir.SEPARATOR}"
        actual = {
            device_path: entry
            for device_path, entry in circuit_devices.items()
            if device_path.startswith(prefix)
        }
        baseline = placement.derivatives(path, with_respect_to=())
        expected = _component_devices(baseline.component, prefix)
        actual_models = {
            device_path: model for device_path, (model, _) in actual.items()
        }
        expected_models = {
            device_path: model for device_path, (model, _) in expected.items()
        }
        if actual_models != expected_models:
            message = (
                f"subcircuit binding `{path}` does not match the devices placed at "
                f"`{path}`"
            )
            raise ValueError(message)
        base_floats: dict[str, Mapping[str, float]] = {}
        base_discrete: dict[str, Mapping[str, object]] = {}
        for device_path, (model, actual_raw) in actual.items():
            actual_canonical, actual_floats, actual_discrete = _canonical_parameters(
                model, actual_raw
            )
            _, expected_floats, expected_discrete = _canonical_parameters(
                model, expected[device_path][1]
            )
            if (
                actual_discrete != expected_discrete
                or actual_floats.keys() != expected_floats.keys()
                or any(
                    not math.isclose(
                        actual_floats[key],
                        expected_floats[key],
                        rel_tol=_BASELINE_RELATIVE_TOLERANCE,
                        abs_tol=_BASELINE_ABSOLUTE_TOLERANCE,
                    )
                    for key in actual_floats
                )
            ):
                message = (
                    f"subcircuit binding `{path}` baseline parameters do not match "
                    f"`{device_path}` in the circuit"
                )
                raise ValueError(message)
            canonical_devices[device_path] = actual_canonical
            base_floats[device_path] = MappingProxyType(actual_floats)
            base_discrete[device_path] = MappingProxyType(actual_discrete)
        bound[path] = _BoundSubcircuit(
            path=path,
            placement=placement,
            models=MappingProxyType(actual_models),
            base_floats=MappingProxyType(base_floats),
            base_discrete=MappingProxyType(base_discrete),
        )
    return bound, canonical_devices


def _canonical_ancestors(paths: Iterable[str]) -> set[str]:
    """Returns every strict ancestor component path containing a path in `paths`."""
    ancestors: set[str] = set()
    for device_path in paths:
        parts = device_path.split(ir.SEPARATOR)
        for depth in range(1, len(parts)):
            ancestors.add(ir.SEPARATOR.join(parts[:depth]))
    return ancestors


def _fresh_component_name(base: str, existing: Mapping[str, ir.Component]) -> str:
    """Returns a unique component definition name derived from `base`."""
    suffix = 1
    while f"{base}__{suffix}" in existing:
        suffix += 1
    return f"{base}__{suffix}"


def _canonicalize_circuit(
    circuit_ir: ir.Circuit,
    canonical_devices: Mapping[str, dict[str, JsonValue]],
) -> ir.Circuit:
    """Populates engine-defaulted parameters on bound subcircuit primitive devices.

    Elaborated circuits index numeric overrides from each device's stored IR
    parameters. Defaulting bound subcircuit devices ensures continuous parameters
    omitted at baseline (such as default `multiplier = 1` or bin-specific parameters
    absent from the initial bin) remain overridable and differentiable without
    re-elaboration, while leaving unbound components unchanged.
    """
    if not canonical_devices:
        return circuit_ir
    ancestors = _canonical_ancestors(canonical_devices)
    components = dict(circuit_ir.components)

    def rewrite_instance(
        instance: ir.Instance, prefix: str
    ) -> tuple[ir.Instance, bool]:
        path = f"{prefix}{instance.name}"
        match instance.of:
            case ir.DeviceOf():
                canonical = canonical_devices.get(path)
                if canonical is not None:
                    return instance.model_copy(update={"parameters": canonical}), True
            case ir.ComponentOf(component=inner) if path in ancestors:
                child_name = rewrite(inner, f"{path}{ir.SEPARATOR}")
                if child_name != inner:
                    return (
                        instance.model_copy(
                            update={"of": ir.ComponentOf(component=child_name)}
                        ),
                        True,
                    )
        return instance, False

    def rewrite(component_name: str, prefix: str) -> str:
        definition = circuit_ir.components[component_name]
        rebuilt_pairs = [
            rewrite_instance(instance, prefix) for instance in definition.instances
        ]
        if not any(changed for _, changed in rebuilt_pairs):
            return component_name
        rebuilt = definition.model_copy(
            update={"instances": tuple(item for item, _ in rebuilt_pairs)}
        )
        target_name = (
            component_name
            if not prefix
            else _fresh_component_name(component_name, components)
        )
        components[target_name] = rebuilt
        return target_name

    rewrite(circuit_ir.top, "")
    return circuit_ir.model_copy(update={"components": components})


def _check_subcircuit_conflicts(
    paths: Iterable[str], bound: Mapping[str, _BoundSubcircuit]
) -> None:
    """Rejects overriding both a bound subcircuit path and an inner device under it."""
    path_set = set(paths)
    active_bound = path_set & bound.keys()
    for subcircuit_path in active_bound:
        prefix = f"{subcircuit_path}{ir.SEPARATOR}"
        for other in path_set:
            if other.startswith(prefix):
                message = (
                    f"cannot override both subcircuit `{subcircuit_path}` and inner "
                    f"device `{other}`"
                )
                raise ValueError(message)


def _contract_subcircuit(
    evaluated: SubcircuitDerivatives,
    primitive_gradients: Mapping[str, Mapping[str, float | complex]],
    *,
    phasors: bool,
) -> dict[str, float | complex]:
    """Pulls real or complex primitive gradients back to subcircuit argument slopes."""
    pulled: dict[str, float | complex] = dict.fromkeys(
        evaluated.arguments, 0j if phasors else 0.0
    )
    for device_path, device_cotangents in primitive_gradients.items():
        device_jacobian = evaluated.derivatives.get(device_path)
        if device_jacobian is None:
            continue
        for parameter_name, cotangent in device_cotangents.items():
            slopes = device_jacobian.get(parameter_name)
            if slopes is None:
                continue
            for argument, slope in slopes.items():
                pulled[argument] = pulled[argument] + cotangent * slope
    return pulled


def _collect_row_entries(
    evaluated: Mapping[str, SubcircuitDerivatives],
    primitive_index: dict[tuple[str, str], int],
    column_index: Mapping[tuple[str, str], int],
) -> tuple[tuple[int, int, float], ...]:
    """Collects sparse `(primitive_col, output_col, slope)` triples for one run."""
    entries: list[tuple[int, int, float]] = []
    for subcircuit_path, subcircuit in evaluated.items():
        for device_path, device_params in subcircuit.derivatives.items():
            for parameter_name, slopes in device_params.items():
                primitive_column = primitive_index.setdefault(
                    (device_path, parameter_name), len(primitive_index)
                )
                for argument, slope in slopes.items():
                    entries.append(
                        (
                            primitive_column,
                            column_index[subcircuit_path, argument],
                            slope,
                        )
                    )
    return tuple(entries)


def _plain_runs(
    names: Sequence[tuple[str, str]], rows: npt.NDArray[np.float64]
) -> list[dict[str, dict[str, float]]]:
    """Unpacks a parameter matrix into per-run nested override dictionaries."""
    runs: list[dict[str, dict[str, float]]] = []
    for row in rows.tolist():
        run: dict[str, dict[str, float]] = {}
        for (path, name), value in zip(names, row, strict=True):
            run.setdefault(path, {})[name] = value
        runs.append(run)
    return runs


@dataclass(frozen=True, slots=True)
class _RowComposition:
    """Sparse chain-rule pullback composer for batched subcircuit gradients."""

    output_columns: int
    direct_columns: tuple[tuple[int, int], ...]
    row_entries: tuple[tuple[tuple[int, int, float], ...], ...]

    def apply(
        self,
        rows: npt.NDArray[np.intp],
        raw: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
        *,
        phasors: bool,
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Contracts engine primitive gradients `raw` back to caller column layout."""
        shape = (len(rows), self.output_columns)
        result: npt.NDArray[np.float64] | npt.NDArray[np.complex128] = (
            np.zeros(shape, np.complex128) if phasors else np.zeros(shape, np.float64)
        )
        for output_column, primitive_column in self.direct_columns:
            result[:, output_column] = raw[:, primitive_column]
        for output_row, run_index in enumerate(rows.tolist()):
            for primitive_column, output_column, slope in self.row_entries[run_index]:
                result[output_row, output_column] += (
                    raw[output_row, primitive_column] * slope
                )
        return result


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
        *,
        subcircuits: Mapping[str, SubcircuitPlacement] | None = None,
    ) -> None:
        """Initializes a differentiable analysis function for `circuit_or_top`.

        Args:
            circuit_or_top: Circuit IR or top-level component to elaborate.
            analysis: Differentiable analysis specification evaluated per call.
            options: Optional solver tolerances, iteration limits, and nodesets.
            subcircuits: Optional mapping from placed SPICE subcircuit instance paths
                to [`SubcircuitPlacement`][jolty.spice.SubcircuitPlacement] bindings,
                enabling exact differentiation through SPICE subcircuit parameters.

        Raises:
            ElaborationError: Lists every problem reported during elaboration.
            TypeError: When `analysis` is not a differentiable analysis type.
            ValueError: When a `subcircuits` binding does not match the circuit.
        """
        if not isinstance(
            analysis,
            OperatingPoint
            | DcSweep
            | Ac
            | Stability
            | Noise
            | Transient
            | PeriodicSteadyState,
        ):
            message = (
                "a function runs an operating point, a DC sweep, a small-signal "
                "response, a stability analysis, a noise analysis, a transient or a "
                "periodic steady "
                f"state, not {type(analysis).__name__}"
            )
            raise TypeError(message)
        raw_circuit = (
            circuit_or_top
            if isinstance(circuit_or_top, ir.Circuit)
            else circuit(circuit_or_top)
        )
        bound, canonical_devices = _bind_subcircuits(raw_circuit, subcircuits)
        self.elaborated = Elaborated(
            _canonicalize_circuit(raw_circuit, canonical_devices)
        )
        self.analysis = analysis
        self.options = Options() if options is None else options
        self._subcircuits = bound

    def check(self, parameters: Mapping[str, Iterable[str]]) -> None:
        """Validates every device and bound subcircuit parameter named in `parameters`.

        Args:
            parameters: Mapping from instance or bound subcircuit paths to iterables of
                parameter names.

        Raises:
            ElaborationError: Lists unrecognized primitive device instances or names.
            SpiceError: When a bound subcircuit rejects an unknown or duplicate name.
            ValueError: When `parameters` overrides both a subcircuit and an inner
                device under it.
        """
        if not self._subcircuits:
            self.elaborated.check(parameters)
            return
        _check_subcircuit_conflicts(parameters, self._subcircuits)
        primitive: dict[str, Iterable[str]] = {}
        for path, names in parameters.items():
            bound = self._subcircuits.get(path)
            if bound is None:
                primitive[path] = names
            else:
                bound.placement.derivatives(path, with_respect_to=tuple(names))
        if primitive:
            self.elaborated.check(primitive)

    def bins(self, parameters: Parameters | None = None) -> dict[str, BinBounds]:
        """Returns active MOSFET bin rectangles for all bound subcircuits.

        Args:
            parameters: Optional overrides keyed by bound subcircuit path.

        Returns:
            Mapping from hierarchical MOSFET device paths to their active
            [`BinBounds`][jolty.spice.BinBounds].
        """
        if not self._subcircuits:
            return {}
        active: dict[str, BinBounds] = {}
        overrides = parameters or {}
        for path, bound in self._subcircuits.items():
            evaluated = bound.placement.derivatives(
                path, with_respect_to=(), **overrides.get(path, {})
            )
            active.update(evaluated.bins)
        return active

    def _expand_point(
        self, parameters: Parameters, *, differentiate: bool
    ) -> tuple[dict[str, dict[str, float]], dict[str, SubcircuitDerivatives]]:
        """Expands bound subcircuit entries in `parameters` into primitive overrides."""
        if not self._subcircuits:
            return {path: dict(values) for path, values in parameters.items()}, {}
        _check_subcircuit_conflicts(parameters, self._subcircuits)
        expanded: dict[str, dict[str, float]] = {}
        evaluated: dict[str, SubcircuitDerivatives] = {}
        for path, values in parameters.items():
            bound = self._subcircuits.get(path)
            if bound is None:
                expanded[path] = dict(values)
                continue
            seeds = tuple(values) if differentiate else ()
            diff, subcircuit_derivatives = bound.expand(values, with_respect_to=seeds)
            evaluated[path] = subcircuit_derivatives
            expanded.update(diff)
        return expanded, evaluated

    def __call__(self, parameters: Parameters) -> Outputs:
        """Evaluates the analysis at `parameters` and returns the unknown outputs.

        Raises:
            ElaborationError: Lists unknown parameters or out-of-range device values.
            SimulationError: When the analysis fails to solve.
        """
        expanded, _ = self._expand_point(parameters, differentiate=False)
        core, outcome = self._run(
            expanded,
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

    def _output_names(self, core: _core.Simulator) -> tuple[str, ...]:
        """Returns output names for a specific simulator `core` in column order."""
        if isinstance(self.analysis, Stability):
            return (LOOP_GAIN,)
        if isinstance(self.analysis, Noise):
            return noise_outputs(self.analysis)
        return tuple(name for name, _ in core.unknowns)

    def _outputs(self, core: _core.Simulator, outcome: _core.AnalysisResult) -> Outputs:
        """Extracts output arrays and optional oscillator period from `outcome`."""
        rows, period = _values(outcome)
        found: Outputs = {
            name: rows[:, column]
            for column, name in enumerate(self._output_names(core))
        }
        if period is not None and self.oscillates:
            found[PERIOD] = float(period)
        return found

    @property
    def _is_complex(self) -> bool:
        """True when the analysis produces complex frequency-domain outputs."""
        return isinstance(self.analysis, Ac | Stability)

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
        expanded, evaluated = self._expand_point(parameters, differentiate=True)
        primitive_names, primitive_tangent = self._push_tangents(
            parameters, tangents, evaluated
        )
        if not primitive_names:
            core, outcome = self._run(
                expanded,
                lambda core, analysis, settings: core.run(
                    analysis, settings, engine_pool()
                ),
            )
            outputs = self._outputs(core, outcome)
            zeros: Outputs = {
                name: (np.zeros_like(value) if isinstance(value, np.ndarray) else 0.0)
                for name, value in outputs.items()
            }
            return outputs, zeros
        core, found = self._run(
            expanded,
            lambda core, analysis, settings: core.jvp(
                analysis, primitive_names, [primitive_tangent], settings, engine_pool()
            ),
        )
        outputs = self._outputs(core, found["result"])
        moved = found["tangents"][:, :, 0]
        pushed: Outputs = {
            name: moved[:, column]
            for column, name in enumerate(self._output_names(core))
        }
        if (period := found["period"]) is not None and self.oscillates:
            pushed[PERIOD] = float(period[0])
        return outputs, pushed

    def _push_tangents(
        self,
        parameters: Parameters,
        tangents: Parameters,
        evaluated: Mapping[str, SubcircuitDerivatives],
    ) -> tuple[list[tuple[str, str]], list[float]]:
        """Maps direct and subcircuit tangents into flat engine `(names, tangent)`."""
        if not evaluated:
            names = self._named(parameters)
            tangent = [
                float(tangents.get(path, {}).get(name, 0.0)) for path, name in names
            ]
            return names, tangent
        combined: dict[tuple[str, str], float] = {}
        for path, named in parameters.items():
            subcircuit = evaluated.get(path)
            if subcircuit is None:
                for name in named:
                    combined[path, name] = float(tangents.get(path, {}).get(name, 0.0))
                continue
            pushed = subcircuit.push(
                {name: float(value) for name, value in tangents.get(path, {}).items()}
            )
            for device_path, device_tangents in pushed.items():
                for parameter_name, delta in device_tangents.items():
                    combined[device_path, parameter_name] = (
                        combined.get((device_path, parameter_name), 0.0) + delta
                    )
        return list(combined), list(combined.values())

    def pulled(self, parameters: Parameters) -> tuple[Outputs, Pullback]:
        """Computes primal outputs and a reverse-mode adjoint pullback (VJP).

        Args:
            parameters: Parameter overrides defining the evaluation point.

        Returns:
            Pair of primal `Outputs` and adjoint `Pullback` callable.
        """
        expanded, evaluated = self._expand_point(parameters, differentiate=True)
        primitive_names = self._pull_names(parameters, evaluated)
        phasors = self._is_complex
        if not primitive_names:
            core, outcome = self._run(
                expanded,
                lambda core, analysis, settings: core.run(
                    analysis, settings, engine_pool()
                ),
            )
            outputs = self._outputs(core, outcome)
            columns = {
                name: column for column, name in enumerate(self._output_names(core))
            }
            points = _values(outcome)[0].shape[0]

            def zero_pull(cotangent: Mapping[str, npt.ArrayLike]) -> Gradient:
                _cotangents(
                    cotangent,
                    columns,
                    (points, len(columns)),
                    phasors=phasors,
                    oscillates=self.oscillates,
                )
                return {
                    path: dict.fromkeys(named, 0j if phasors else 0.0)
                    for path, named in parameters.items()
                }

            return outputs, zero_pull
        core, (outcome, pullback) = self._run(
            expanded,
            lambda core, analysis, settings: core.vjp(
                analysis, primitive_names, settings, engine_pool()
            ),
        )
        outputs = self._outputs(core, outcome)
        columns = {name: column for column, name in enumerate(self._output_names(core))}
        points = _values(outcome)[0].shape[0]
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
            primitive_gradients: Gradient = {}
            for (path, name), value in zip(
                primitive_names, found.tolist(), strict=True
            ):
                primitive_gradients.setdefault(path, {})[name] = value
            if not evaluated:
                return primitive_gradients
            pulled: Gradient = {}
            for path, named in parameters.items():
                subcircuit = evaluated.get(path)
                if subcircuit is None:
                    pulled[path] = {
                        name: primitive_gradients[path][name] for name in named
                    }
                else:
                    pulled[path] = _contract_subcircuit(
                        subcircuit, primitive_gradients, phasors=phasors
                    )
            return pulled

        return outputs, pull

    def _pull_names(
        self,
        parameters: Parameters,
        evaluated: Mapping[str, SubcircuitDerivatives],
    ) -> list[tuple[str, str]]:
        """Collects deduplicated engine `(device, parameter)` targets for VJP."""
        if not evaluated:
            return self._named(parameters)
        ordered: dict[tuple[str, str], None] = {}
        for path, named in parameters.items():
            subcircuit = evaluated.get(path)
            if subcircuit is None:
                for name in named:
                    ordered[path, name] = None
                continue
            for device_path, device_parameters in subcircuit.derivatives.items():
                for parameter_name in device_parameters:
                    ordered[device_path, parameter_name] = None
        return list(ordered)

    @functools.cached_property
    def unknowns(self) -> tuple[str, ...]:
        """Circuit unknown names in engine order."""
        return tuple(name for name, _ in self.elaborated.simulator().unknowns)

    @property
    def outputs(self) -> tuple[str, ...]:
        """Output array names returned by this function in engine column order."""
        if isinstance(self.analysis, Stability):
            return (LOOP_GAIN,)
        if isinstance(self.analysis, Noise):
            return noise_outputs(self.analysis)
        return self.unknowns

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
            Batched output `Outputs` indexed by run and analysis point.
        """
        outputs, _, _ = self._batched(names, values, differentiate=False)
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
        outputs, core, composition = self._batched(names, values, differentiate=True)
        pullback = BatchPullback(
            core,
            self.outputs,
            self.points,
            phasors=self._is_complex,
            oscillates=self.oscillates,
            settings=self._settings(),
            composition=composition,
        )
        return outputs, pullback

    def _expand_batch(
        self,
        names: Sequence[tuple[str, str]],
        rows: npt.NDArray[np.float64],
        *,
        differentiate: bool,
    ) -> tuple[
        list[dict[str, dict[str, float]]],
        list[tuple[str, str]],
        _RowComposition | None,
    ]:
        """Expands batch rows through any bound subcircuits and builds chain rule."""
        has_bound = bool(self._subcircuits) and any(
            path in self._subcircuits for path, _ in names
        )
        if not has_bound:
            return (
                _plain_runs(names, rows),
                list(names) if differentiate else [],
                None,
            )
        _check_subcircuit_conflicts((path for path, _ in names), self._subcircuits)
        column_index = {pair: index for index, pair in enumerate(names)}
        primitive_index: dict[tuple[str, str], int] = {}
        direct_columns: list[tuple[int, int]] = []
        if differentiate:
            for output_column, pair in enumerate(names):
                if pair[0] not in self._subcircuits:
                    primitive_column = primitive_index.setdefault(
                        pair, len(primitive_index)
                    )
                    direct_columns.append((output_column, primitive_column))
        runs: list[dict[str, dict[str, float]]] = []
        row_entries: list[tuple[tuple[int, int, float], ...]] = []
        for point in _plain_runs(names, rows):
            expanded, evaluated = self._expand_point(point, differentiate=differentiate)
            runs.append(expanded)
            if differentiate:
                row_entries.append(
                    _collect_row_entries(evaluated, primitive_index, column_index)
                )
        composition = (
            _RowComposition(
                output_columns=len(names),
                direct_columns=tuple(direct_columns),
                row_entries=tuple(row_entries),
            )
            if differentiate
            else None
        )
        return runs, list(primitive_index), composition

    def _batched(
        self,
        names: Sequence[tuple[str, str]],
        values: npt.ArrayLike,
        *,
        differentiate: bool,
    ) -> tuple[Outputs, _core.BatchPullback, _RowComposition | None]:
        """Executes a batch at `values` for `names` and returns outputs and pullback."""
        rows = np.asarray(values, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(names):  # noqa: PLR2004 - a matrix
            message = f"values are a row of {len(names)} names by run, not {rows.shape}"
            raise ValueError(message)
        runs, primitive_parameters, composition = self._expand_batch(
            names, rows, differentiate=differentiate
        )
        variants = self.elaborated.variants(runs)
        base = self.elaborated.simulator()
        output_names = self.outputs
        columns = list(range(len(output_names)))
        try:
            found, pullback = base.batch_vjp(
                self._encoded(),
                self._settings(),
                primitive_parameters,
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
            name: traces[:, :, column] for column, name in enumerate(output_names)
        }
        if self.oscillates:
            outputs[PERIOD] = found["periods"]
        return outputs, pullback, composition


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
        composition: _RowComposition | None = None,
    ) -> None:
        """Wraps an engine `core` batch pullback for `unknowns` over `points`."""
        self._core = core
        self._columns = {name: column for column, name in enumerate(unknowns)}
        self._points = points
        self._phasors = phasors
        self._oscillates = oscillates
        self._settings = settings
        self._composition = composition

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
            raw = self._core.pull_back(
                rows.tolist(), values, periods.tolist(), self._settings, engine_pool()
            )
        except _core.SimulationError as error:
            raise SimulationError(str(error)) from None
        if self._composition is None:
            return raw
        return self._composition.apply(rows, raw, phasors=self._phasors)


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
    """Extracts the 2D output array and optional period from an engine `outcome`."""
    match outcome["kind"]:
        case "periodic_steady_state":
            return np.atleast_2d(outcome["values"]), outcome["period"]
        case "operating_point" | "dc_sweep" | "ac" | "transient":
            return np.atleast_2d(outcome["values"]), None
        case "stability":
            return outcome["gains"][:, np.newaxis], None
        case "noise":
            densities = np.asarray(outcome["densities"], dtype=np.float64)
            if (noise_input := outcome["input"]) is not None:
                referred = _core.referred(
                    densities, np.asarray(noise_input["gains"], dtype=np.complex128)
                )
                return np.column_stack([densities, referred]), None
            return densities[:, np.newaxis], None
        case other:
            message = f"a function runs no {other} analysis"
            raise SimulationError(message)


def function(
    circuit_or_top: ir.Circuit | Component,
    analysis: Differentiable,
    options: Options | None = None,
    *,
    subcircuits: Mapping[str, SubcircuitPlacement] | None = None,
) -> Function:
    """Creates a differentiable [`Function`][jolty.simulate.Function] for `analysis`.

    Calling ``function(circuit, analysis)(parameters)`` evaluates `analysis` with
    `options` at `parameters` and returns unknown outputs. Supplying `subcircuits`
    enables exact differentiation through SPICE subcircuit arguments (such as PDK
    MOSFET `w`, `l`, `nf`, or `mult`) composed with the engine adjoint solver.

    Args:
        circuit_or_top: Circuit IR or top-level component to elaborate.
        analysis: Differentiable analysis specification evaluated per call.
        options: Optional solver tolerances, iteration limits, and nodesets.
        subcircuits: Optional mapping from placed subcircuit paths to
            [`SubcircuitPlacement`][jolty.spice.SubcircuitPlacement] bindings.

    Returns:
        Callable `Function` evaluating `analysis` at parameter overrides.

    Raises:
        ElaborationError: Lists every problem reported during circuit elaboration.
        ValueError: When a `subcircuits` binding does not match the circuit.
    """
    return Function(circuit_or_top, analysis, options, subcircuits=subcircuits)


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

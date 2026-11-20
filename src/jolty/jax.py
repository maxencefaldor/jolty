# Copyright 2026 Maxence Faldor
"""JAX integration for circuit simulation and differentiation (`jit`, `grad`, `vmap`).

[`function`][jolty.jax.function] wraps a circuit analysis as a JAX callable taking the
same nested parameter mapping as [`jolty.function`][jolty.simulate.function] and
returning a dictionary of output arrays. Each invocation runs through
[`Function.batch`][jolty.simulate.Function.batch] inside `jax.pure_callback` on the CPU
so `jax.vmap` maps directly to a multithreaded engine batch with bit-exact results.

Reverse-mode differentiation uses `jax.custom_vjp` backed by
[`Function.batch_vjp`][jolty.simulate.Function.batch_vjp], requiring one transposed
solve per analysis point regardless of parameter count. Supported analyses are
operating points, DC sweeps, small-signal responses, stability loop gains, noise
spectra, transients, and periodic steady states.

Forward passes cache pullbacks for the most recent `saved_pullbacks` calls per
function; evicted entries recompute the forward pass bit-identically during the
backward pass. Only first-order reverse mode is supported (`jax.jvp` and higher-order
derivatives such as `jax.hessian` are rejected).

Optional installation requires ``pip install 'jolty[jax]'``.

Examples:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jolty import Builder, Role, component, devices
    >>> from jolty.jax import function
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
    >>> solve = function(divider(), OperatingPoint())
    >>> round(float(solve({"R1": {"resistance": 3000.0}})["out"][0]), 6)
    1.25
    >>> gradient = jax.grad(
    ...     lambda parameters: jnp.sum((solve(parameters)["out"] - 2.0) ** 2)
    ... )({"R1": {"resistance": 1000.0}})
    >>> round(float(gradient["R1"]["resistance"]), 9)
    -0.00125
"""

import itertools
import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import numpy.typing as npt

try:
    import jax
    import jax.numpy as jnp
    from jax.typing import ArrayLike
except ModuleNotFoundError as error:
    root_module = (error.name or "").split(".", 1)[0]
    if root_module not in {"jax", "jaxlib"}:
        raise
    message = (
        "jolty.jax needs the optional `jax` dependency; "
        "install it with `pip install 'jolty[jax]'`"
    )
    raise ImportError(message) from error

from jolty import ir
from jolty.builder import Component
from jolty.plan import Ac, Options, PeriodicSteadyState, Stability, Transient
from jolty.simulate import (
    LOOP_GAIN,
    NOISE_DENSITY,
    PERIOD,
    REFERRED_NOISE,
    BatchPullback,
    Differentiable,
    Function,
)

type JaxParameters = Mapping[str, Mapping[str, ArrayLike]]
"""Nested mapping of instance path to parameter path and real scalar SI value."""

type JaxOutputs = dict[str, jax.Array]
"""Dictionary mapping each output name of a JAX-wrapped analysis to a JAX array."""

type _Name = tuple[str, str]
"""Parameter identifier pairing an instance path with a parameter path."""

type _Structure = tuple[tuple[str, tuple[str, ...]], ...]
"""Canonical sorted tuple of instance paths and their sorted parameter-path tuples."""

_HANDLE_INTEGERS: Final = 2
"""Number of integer fields in a pullback handle: forward-pass key and row index."""


@dataclass(frozen=True, slots=True)
class _Dtypes:
    """Active JAX dtypes (64-bit when `jax_enable_x64` is enabled, else 32-bit).

    Stores the real scalar dtype, output dtype (complex for AC), and handle dtype.
    """

    real: np.dtype
    output: np.dtype
    handle: np.dtype


class _Saved:
    """Bounded LRU cache storing pullbacks from the most recent forward passes."""

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._kept: OrderedDict[int, BatchPullback] = OrderedDict()
        self._keys = itertools.count()
        self._lock = threading.Lock()

    def keep(self, pullback: BatchPullback, bound: int) -> int:
        """Stores `pullback`, evicts old entries past `max_entries`, and returns a key.

        Returned keys wrap modulo `bound` to fit within the handle integer dtype.
        """
        with self._lock:
            key = next(self._keys) % bound
            if self._max_entries > 0:
                self._kept[key] = pullback
                while len(self._kept) > self._max_entries:
                    self._kept.popitem(last=False)
            return key

    def get(self, key: int) -> BatchPullback | None:
        """Returns the cached pullback for `key`, or ``None`` if evicted."""
        with self._lock:
            return self._kept.get(key)


def _static(target: Function) -> None:
    """Validates that every output batch from `target` has a static point count.

    Raises:
        ValueError: When a transient or driven periodic steady state omits
            breakpoints, or when the circuit contains foundry devices.
    """
    analysis = target.analysis
    driven = isinstance(analysis, Transient) or (
        isinstance(analysis, PeriodicSteadyState) and analysis.period is not None
    )
    if driven and not analysis.breakpoints:
        message = (
            f"a JAX {type(analysis).__name__} requires non-empty `breakpoints` to "
            "sample outputs"
        )
        raise ValueError(message)
    if target.elaborated.simulator().foundry_devices:
        message = (
            "jolty.jax runs the circuits the engine simulates, not foundry devices"
        )
        raise ValueError(message)


def _flatten(parameters: JaxParameters) -> tuple[_Structure, list[jax.Array]]:
    """Extracts the canonical structure and flat scalar list from `parameters`.

    Raises:
        TypeError: When `parameters` is not a two-level mapping of strings to real
            scalars (`bool` and `complex` are rejected).
        ValueError: When any parameter leaf is not a scalar.
    """
    if not isinstance(parameters, Mapping):
        message = f"parameters are a mapping, not {type(parameters).__name__}"
        raise TypeError(message)
    instances: list[tuple[str, list[tuple[str, jax.Array]]]] = []
    for path, numbers in parameters.items():
        if not isinstance(path, str) or not isinstance(numbers, Mapping):
            message = "parameters map each instance's path to its parameters' mapping"
            raise TypeError(message)
        named: list[tuple[str, jax.Array]] = []
        for name, value in numbers.items():
            leaf = jnp.asarray(value)
            real = jnp.issubdtype(leaf.dtype, jnp.floating) or jnp.issubdtype(
                leaf.dtype, jnp.integer
            )
            if not isinstance(name, str) or isinstance(value, bool) or not real:
                message = f"parameter `{path}.{name}` is a real number, not {value!r}"
                raise TypeError(message)
            if leaf.ndim != 0:
                message = f"parameter `{path}.{name}` is a scalar, not {leaf.shape}"
                raise ValueError(message)
            named.append((name, leaf))
        named.sort(key=lambda pair: pair[0])
        instances.append((path, named))
    instances.sort(key=lambda instance: instance[0])
    structure = tuple(
        (path, tuple(name for name, _ in named)) for path, named in instances
    )
    return structure, [leaf for _, named in instances for _, leaf in named]


def _rows(theta: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], tuple[int, ...]]:
    """Reshapes a batched parameter array into a 2-D row matrix and batch shape."""
    array = np.asarray(theta, dtype=np.float64)
    shape = array.shape[:-1]
    return array.reshape(math.prod(shape), array.shape[-1]), shape


def _pulled(  # noqa: PLR0913 - backward pass reads full forward-pass state
    target: Function,
    names: tuple[_Name, ...],
    saved: _Saved,
    *,
    rows: npt.NDArray[np.float64],
    held: npt.NDArray[np.integer],
    cotangent: Mapping[str, npt.NDArray[np.generic]],
) -> npt.NDArray[np.float64]:
    """Computes parameter gradients by pulling `cotangent` back across every row.

    Each row `rows[i]` references a cached pullback through `held[i]`. Rows whose
    forward-pass pullback was evicted re-run the forward pass together as one batch.
    """
    pulled = np.zeros((len(rows), len(names)))
    for key in np.unique(held[:, 0]):
        members = np.flatnonzero(held[:, 0] == key)
        kept = saved.get(int(key))
        if kept is None:
            _, kept = target.batch_vjp(names, rows[members])
            runs = np.arange(len(members))
        else:
            runs = held[members, 1]
        given = {name: value[members] for name, value in cotangent.items()}
        pulled[members] = np.real(kept(given, runs))
    return pulled


def _runner(
    target: Function,
    names: tuple[_Name, ...],
    dtypes: _Dtypes,
    saved: _Saved,
) -> jax.custom_vjp[JaxOutputs]:
    """Builds the `jax.custom_vjp` callable mapping a parameter vector to outputs.

    Outputs from `jax.pure_callback` match the declared `ShapeDtypeStruct` layout and
    are explicitly cast to satisfy static type checking.
    """
    shapes = {
        name: jax.ShapeDtypeStruct((target.points,), dtypes.output)
        for name in target.outputs
    }
    if target.oscillates:
        shapes[PERIOD] = jax.ShapeDtypeStruct((), dtypes.real)
    gradient = jax.ShapeDtypeStruct((len(names),), dtypes.real)
    handles = jax.ShapeDtypeStruct((_HANDLE_INTEGERS,), dtypes.handle)
    bound = np.iinfo(dtypes.handle).max

    def shaped(outputs: Mapping[str, npt.ArrayLike], batch: tuple[int, ...]) -> object:
        return {
            name: np.asarray(outputs[name])
            .astype(spec.dtype)
            .reshape(batch + spec.shape)
            for name, spec in shapes.items()
        }

    def forward(theta: npt.ArrayLike) -> object:
        rows, batch = _rows(theta)
        return shaped(target.batch(names, rows), batch)

    def saving(theta: npt.ArrayLike) -> object:
        rows, batch = _rows(theta)
        outputs, pullback = target.batch_vjp(names, rows)
        key = saved.keep(pullback, bound)
        held = np.stack([np.full(len(rows), key), np.arange(len(rows))], axis=-1)
        held = held.astype(dtypes.handle).reshape((*batch, _HANDLE_INTEGERS))
        return shaped(outputs, batch), held

    def backward(
        theta: npt.ArrayLike,
        held: npt.ArrayLike,
        cotangent: Mapping[str, npt.ArrayLike],
    ) -> object:
        rows, batch = _rows(theta)
        given = {
            name: np.asarray(value).reshape((len(rows), *shapes[name].shape))
            for name, value in cotangent.items()
        }
        handles = np.asarray(held).reshape(len(rows), _HANDLE_INTEGERS)
        pulled = _pulled(target, names, saved, rows=rows, held=handles, cotangent=given)
        return pulled.astype(dtypes.real).reshape((*batch, len(names)))

    @jax.custom_vjp
    def run(theta: jax.Array) -> JaxOutputs:
        found = jax.pure_callback(forward, shapes, theta, vmap_method="broadcast_all")
        return cast("JaxOutputs", found)

    def run_saving(theta: jax.Array) -> tuple[JaxOutputs, tuple[jax.Array, jax.Array]]:
        found = jax.pure_callback(
            saving, (shapes, handles), theta, vmap_method="broadcast_all"
        )
        outputs, held = cast("tuple[JaxOutputs, jax.Array]", found)
        return outputs, (theta, held)

    def run_backward(
        residuals: tuple[jax.Array, jax.Array], cotangent: JaxOutputs
    ) -> tuple[jax.Array]:
        theta, held = residuals
        found = jax.pure_callback(
            backward, gradient, theta, held, cotangent, vmap_method="broadcast_all"
        )
        return (cast("jax.Array", found),)

    run.defvjp(run_saving, run_backward)
    return run


def function(
    circuit_or_top: ir.Circuit | Component,
    analysis: Differentiable,
    options: Options | None = None,
    *,
    saved_pullbacks: int = 16,
) -> Callable[[JaxParameters], JaxOutputs]:
    """Wraps a circuit `analysis` as a differentiable JAX function.

    The returned callable accepts ``{instance: {parameter: value}}`` mappings of real
    scalars matching [`jolty.function`][jolty.simulate.function] and returns outputs at
    [`Function.points`][jolty.simulate.Function.points]. Output points correspond to
    one operating point, DC sweep levels, AC complex phasor frequencies, stability loop
    gain frequencies (under [`LOOP_GAIN`][jolty.jax.LOOP_GAIN]), noise spectral density
    frequencies (under [`NOISE_DENSITY`][jolty.jax.NOISE_DENSITY] and optional
    [`REFERRED_NOISE`][jolty.jax.REFERRED_NOISE]), or transient and driven periodic
    steady-state `breakpoints`. Autonomous oscillators return zero-point outputs plus a
    scalar period under [`PERIOD`][jolty.simulate.PERIOD]. For AC and stability
    analyses, gradients take the real part of the pullback with respect to JAX complex
    cotangents.

    The callable supports `jax.jit`, `jax.grad`, `jax.value_and_grad`, `jax.vjp`,
    `jax.jacrev`, `jax.remat`, and `jax.vmap` (including empty and nested batches).
    Computation uses `float64` when `jax_enable_x64` is enabled and `float32`
    otherwise, dispatching each call as one
    [`Function.batch`][jolty.simulate.Function.batch]. Forward passes cache up to
    `saved_pullbacks` pullbacks before recomputing on backward passes, and engine
    exceptions propagate through JAX callbacks with their original type and message.

    Args:
        circuit_or_top: Circuit IR or top-level `Component` to simulate.
        analysis: Differentiable analysis specification with static point count.
        options: Optional solver tolerance, iteration, and nodeset options.
        saved_pullbacks: Maximum forward-pass pullbacks retained in LRU cache.

    Returns:
        JAX-transformable callable mapping `JaxParameters` to `JaxOutputs`.

    Raises:
        ElaborationError: When the circuit fails to elaborate or the first call
            references parameters not present on circuit devices.
        ValueError: When the circuit contains foundry devices or a transient or driven
            periodic steady state omits `breakpoints`.
    """
    target = Function(circuit_or_top, analysis, options)
    _static(target)
    saved = _Saved(saved_pullbacks)
    lock = threading.Lock()
    runners: dict[tuple[_Structure, _Dtypes], jax.custom_vjp[JaxOutputs]] = {}

    def evaluate(parameters: JaxParameters) -> JaxOutputs:
        structure, leaves = _flatten(parameters)
        real = np.dtype(jax.dtypes.canonicalize_dtype(np.float64))
        output = (
            np.dtype(jax.dtypes.canonicalize_dtype(np.complex128))
            if isinstance(target.analysis, Ac | Stability)
            else real
        )
        dtypes = _Dtypes(
            real, output, np.dtype(jax.dtypes.canonicalize_dtype(np.int64))
        )
        key = (structure, dtypes)
        with lock:
            run = runners.get(key)
        if run is None:
            target.elaborated.check(dict(structure))
            names = tuple((path, name) for path, named in structure for name in named)
            with lock:
                run = runners.get(key)
                if run is None:
                    run = _runner(target, names, dtypes, saved)
                    runners[key] = run
        theta = (
            jnp.stack([leaf.astype(real) for leaf in leaves])
            if leaves
            else jnp.empty((0,), dtype=real)
        )
        return run(theta)

    return evaluate


__all__ = [
    "LOOP_GAIN",
    "NOISE_DENSITY",
    "PERIOD",
    "REFERRED_NOISE",
    "JaxOutputs",
    "JaxParameters",
    "function",
]

# Copyright 2026 Maxence Faldor
"""Run and differentiate circuits inside JAX (`jit`, `grad`, `vjp`, `vmap`).

[`function`][jolty.jax.function] wraps a circuit's analysis as a JAX function of a
nested mapping of parameters, the same mapping
[`jolty.function`][jolty.simulate.function] takes, to a dictionary of outputs. Each
call runs as a batch of [`Function.batch`][jolty.simulate.Function.batch], inside
`jax.pure_callback`, on the CPU outside XLA's compiled code, so `jax.vmap` maps to one
batch of the engine across its threads, and a mapped call finds, row by row, the bits
of an unmapped one. Its reverse mode, registered with `jax.custom_vjp`, is
[`Function.batch_vjp`][jolty.simulate.Function.batch_vjp]'s pullback: one transposed
solve at each point of a run however many the parameters, for operating points, DC
sweeps, small-signal responses, transients and periodic steady states.

The forward pass of a derivative keeps its runs' pullbacks for the backward pass that
follows, the most recent `saved_pullbacks` forward passes' of each function; a
backward pass whose pullbacks are no longer kept runs its forward pass again, which
finds the same bits. Only the first order of reverse mode is supported: `jax.jvp` and
derivatives of derivatives, such as `jax.hessian`, are refused.

Install the optional extra with ``pip install 'jolty[jax]'`` to use this module.

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
from jolty.plan import Ac, Options, PeriodicSteadyState, Transient
from jolty.simulate import PERIOD, BatchPullback, Differentiable, Function

type JaxParameters = Mapping[str, Mapping[str, ArrayLike]]
"""Parameters by instance path, then by parameter path, each a real scalar, in SI."""

type JaxOutputs = dict[str, jax.Array]
"""Each output of a JAX-wrapped analysis, by its name."""

type _Name = tuple[str, str]
"""A parameter, as its instance's path and its own."""

type _Structure = tuple[tuple[str, tuple[str, ...]], ...]
"""The parameters a call names: each instance's path, with its parameters' paths, both
sorted, an instance named with none among them."""

_HANDLE_INTEGERS: Final = 2
"""The integers of a run's handle on its pullback: its forward pass's key, its row."""


@dataclass(frozen=True, slots=True)
class _Dtypes:
    """The dtypes JAX computes in, those of 64 bits with `jax_enable_x64`, else 32.

    The real numbers', the outputs', complex over frequency, and the handles'.
    """

    real: np.dtype
    output: np.dtype
    handle: np.dtype


class _Saved:
    """The pullbacks of the latest forward passes, up to `max_entries`."""

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._kept: OrderedDict[int, BatchPullback] = OrderedDict()
        self._keys = itertools.count()
        self._lock = threading.Lock()

    def keep(self, pullback: BatchPullback, bound: int) -> int:
        """Keeps `pullback`, the oldest dropped past `max_entries`, and gives its key.

        The key is below `bound`, the largest a handle's integers hold.
        """
        with self._lock:
            key = next(self._keys) % bound
            if self._max_entries > 0:
                self._kept[key] = pullback
                while len(self._kept) > self._max_entries:
                    self._kept.popitem(last=False)
            return key

    def get(self, key: int) -> BatchPullback | None:
        """The pullback kept at `key`, if it still is."""
        with self._lock:
            return self._kept.get(key)


def _static(target: Function) -> None:
    """Checks that each output of `target`'s batches has a fixed number of points.

    Raises:
        ValueError: For a transient or a driven periodic steady state without
            breakpoints, the times its outputs are read at, or for a circuit of a
            foundry's devices, which the engine does not run.
    """
    analysis = target.analysis
    driven = isinstance(analysis, Transient) or (
        isinstance(analysis, PeriodicSteadyState) and analysis.period is not None
    )
    if driven and not analysis.breakpoints:
        message = (
            f"a JAX {type(analysis).__name__} reads its outputs at its `breakpoints`, "
            "so it needs some"
        )
        raise ValueError(message)
    if target.elaborated.simulator().foundry_devices:
        message = (
            "jolty.jax runs the circuits the engine simulates, not foundry devices"
        )
        raise ValueError(message)


def _flatten(parameters: JaxParameters) -> tuple[_Structure, list[jax.Array]]:
    """The structure of `parameters`, and each parameter's scalar, in its order.

    Raises:
        TypeError: When `parameters` is not a two-level mapping of strings to real
            scalars, `bool` and `complex` refused.
        ValueError: When a parameter is not a scalar.
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
    """The rows of a batch of parameter vectors, a row per run, and its batch shape."""
    array = np.asarray(theta, dtype=np.float64)
    shape = array.shape[:-1]
    return array.reshape(math.prod(shape), array.shape[-1]), shape


def _pulled(  # noqa: PLR0913 - a backward pass reads its forward pass's whole state
    target: Function,
    names: tuple[_Name, ...],
    saved: _Saved,
    *,
    rows: npt.NDArray[np.float64],
    held: npt.NDArray[np.integer],
    cotangent: Mapping[str, npt.NDArray[np.generic]],
) -> npt.NDArray[np.float64]:
    """Each row's value of `cotangent` on each parameter of `names`, row by parameter.

    Row `i`, the parameters `rows[i]`, holds the handle `held[i]` on its pullback; the
    rows of a forward pass whose pullback is no longer kept run it again, as one batch.
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
    """The `custom_vjp` function of the parameter vector of `names` to the outputs.

    `jax.pure_callback` returns what its callback does, shaped as its specification
    says, which its type does not state, so each call's result is cast to it.
    """
    shapes = {
        name: jax.ShapeDtypeStruct((target.points,), dtypes.output)
        for name in target.unknowns
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
    """A JAX function that runs and differentiates `analysis` of a circuit.

    It takes a mapping ``{instance: {parameter: value}}`` of real scalars, as
    [`jolty.function`][jolty.simulate.function] does, and gives each output, by its
    name, at the analysis's [`points`][jolty.simulate.Function.points]: an operating
    point's one, a DC sweep's levels, a response's frequencies, in complex phasors, and
    a transient's or a driven periodic steady state's `breakpoints`, which it needs; an
    oscillator's outputs hold no point, and its period, a scalar, is under
    [`PERIOD`][jolty.simulate.PERIOD]. Over frequency, a real loss's cotangent on the
    phasors is JAX's own, and the gradient the real part of their pullback.

    It runs under `jax.jit`, `jax.grad`, `jax.value_and_grad`, `jax.vjp`,
    `jax.jacrev`, `jax.remat` and `jax.vmap`, empty and nested batches included, in
    `float64` with `jax_enable_x64` and in `float32` without, and runs each call,
    mapped or not, as one batch of [`Function.batch`][jolty.simulate.Function.batch].
    A derivative's forward pass keeps its runs' pullbacks for its backward pass, the
    latest `saved_pullbacks` forward passes' at most, and else runs it again. An
    exception of the engine within a call reaches the caller as JAX's callbacks raise
    it, its name and message kept.

    Raises:
        ElaborationError: When the circuit does not elaborate, or, at the first call
            with them, naming every parameter its devices do not write.
        ValueError: For a circuit of a foundry's devices, or a transient or a driven
            periodic steady state without breakpoints.
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
            if isinstance(target.analysis, Ac)
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


__all__ = ["PERIOD", "JaxOutputs", "JaxParameters", "function"]

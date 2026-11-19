# Copyright 2026 Maxence Faldor
"""Differentiable circuit functions, forward-mode JVPs, and reverse-mode VJPs.

Engine tests check JVPs against VJPs across all analyses to solver rounding, plus closed
forms and finite differences. The Python suite verifies `jolty.function`, `jolty.jvp`,
and `jolty.vjp` match `jax.jvp`/`jax.vjp` calling conventions and batched evaluation.
"""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pytest

from jolty import function, jvp, vjp
from jolty.errors import ElaborationError
from jolty.plan import Ac, DcSweep, OperatingPoint, Transient
from jolty.simulate import Differentiable, Outputs, Parameters
from jolty.units import MHz, V, kHz, ns, us
from sample_circuits import CROSSBAR_WEIGHTS, crossbar

TRANSPOSED = 1e-11
"""Relative JVP/VJP transpose tolerance against the sum of term magnitudes.

The engine dot-product test measures 7.4e-16 per derivative
(`crates/jolty-engine/tests/sensitivity.rs`). Across every output and parameter, this
coarser scale still exposes any tangent or cotangent dropped across Python."""

NAMES = [(path, "resistance") for path in CROSSBAR_WEIGHTS]
"""The weights' parameters, in the order a batch's rows give them."""

ROWS = np.array(
    [
        [named["resistance"] * (1 + run / 4) for named in CROSSBAR_WEIGHTS.values()]
        for run in range(3)
    ]
)
"""A batch of three runs of the weights, a row each."""

BATCHED = [
    OperatingPoint(),
    DcSweep("V0", [0.1 * V, 0.2 * V, 0.4 * V]),
    Ac("V1", [1 * kHz, 1 * MHz]),
]
"""The analyses whose batches read their outputs at the points their calls do."""


def inner(
    left: Mapping[str, npt.ArrayLike], right: Mapping[str, npt.ArrayLike]
) -> tuple[complex, float]:
    """Computes the inner product of `left` and `right` plus its term-magnitude sum."""
    total, magnitude = 0j, 0.0
    for name, values in left.items():
        products = np.asarray(values) * np.asarray(right[name])
        total += complex(np.sum(products))
        magnitude += float(np.sum(np.abs(products)))
    return total, magnitude


@pytest.mark.parametrize(
    "analysis",
    [
        OperatingPoint(),
        DcSweep("V0", [0.1 * V, 0.2 * V, 0.4 * V]),
        Ac("V1", [1 * kHz, 1 * MHz]),
        Transient(2 * us, max_step=20 * ns),
    ],
    ids=["operating_point", "dc_sweep", "ac", "transient"],
)
def test_the_pullback_is_the_jvps_transpose(analysis: Differentiable) -> None:
    f = function(crossbar(), analysis)
    tangents = {
        path: {name: value * (index + 1) / 10 for name, value in named.items()}
        for index, (path, named) in enumerate(CROSSBAR_WEIGHTS.items())
    }
    outputs, pushed = jvp(f, CROSSBAR_WEIGHTS, tangents)
    same, pull = vjp(f, CROSSBAR_WEIGHTS)
    for name, values in outputs.items():
        np.testing.assert_array_equal(values, same[name])
    cotangent = {
        name: np.linspace(1.0, 2.0, np.size(values)) * (index + 1)
        for index, (name, values) in enumerate(outputs.items())
    }
    forward, magnitude = inner(cotangent, pushed)
    pulled = pull(cotangent)
    backward = sum(
        complex(pulled[path][name]) * tangents[path][name]
        for path, named in CROSSBAR_WEIGHTS.items()
        for name in named
    )
    terms = sum(
        abs(complex(pulled[path][name]) * tangents[path][name])
        for path, named in CROSSBAR_WEIGHTS.items()
        for name in named
    )
    assert abs(forward - backward) <= TRANSPOSED * max(magnitude, terms)


def test_a_losss_gradient_over_the_weights_is_its_finite_differences() -> None:
    # A central difference's truncation, the relative step squared, 1e-8, times the
    # loss's curvature, of order one in those units, with the solves' rounding over the
    # step, 1e-16 / 1e-4: held to 1e-6 of the gradient's largest.
    f = function(crossbar(), OperatingPoint())
    targets = {"out0": 0.05, "out1": -0.02}

    def loss(outputs: Outputs) -> float:
        return sum(
            float(np.sum((np.asarray(outputs[name]) - target) ** 2))
            for name, target in targets.items()
        )

    outputs, pull = vjp(f, CROSSBAR_WEIGHTS)
    gradient = pull(
        {
            name: 2 * (np.asarray(outputs[name]) - target)
            for name, target in targets.items()
        }
    )
    step = 1e-4
    largest = max(abs(gradient[path]["resistance"]) for path in CROSSBAR_WEIGHTS)
    for path, named in CROSSBAR_WEIGHTS.items():
        value = named["resistance"]
        moved = [
            loss(
                f(
                    {
                        **CROSSBAR_WEIGHTS,
                        path: {"resistance": value * (1 + sign * step)},
                    }
                )
            )
            for sign in (1, -1)
        ]
        finite = (moved[0] - moved[1]) / (2 * step * value)
        # Central difference O(step^2) = 1e-8 plus rounding eps / step ~ 1e-12.
        assert abs(gradient[path]["resistance"] - finite) <= 1e-6 * largest, path


def test_a_tangent_moves_only_the_parameters_given() -> None:
    f = function(crossbar(), OperatingPoint())
    with pytest.raises(ValueError, match="only the parameters given"):
        jvp(f, {"W00": {"resistance": 1e3}}, {"W11": {"resistance": 1.0}})


def test_a_parameter_is_a_number_the_circuit_writes() -> None:
    f = function(crossbar(), OperatingPoint())
    with pytest.raises(ElaborationError):
        vjp(f, {"W00": {"nothing": 1.0}})


def parameters(row: npt.ArrayLike) -> Parameters:
    """Converts one batch weight row into a scalar-call parameter mapping."""
    return {
        path: {name: float(value)}
        for (path, name), value in zip(NAMES, np.asarray(row).tolist(), strict=True)
    }


@pytest.mark.parametrize("analysis", BATCHED, ids=["operating_point", "dc_sweep", "ac"])
def test_a_batch_finds_each_run_as_a_call_does(analysis: Differentiable) -> None:
    f = function(crossbar(), analysis)
    outputs = f.batch(NAMES, ROWS)
    for row, values in enumerate(ROWS):
        for name, found in f(parameters(values)).items():
            np.testing.assert_array_equal(np.asarray(outputs[name])[row], found)


@pytest.mark.parametrize("analysis", BATCHED, ids=["operating_point", "dc_sweep", "ac"])
def test_a_batch_pullback_pulls_each_row_back_as_a_vjp_does(
    analysis: Differentiable,
) -> None:
    f = function(crossbar(), analysis)
    outputs, pullback = f.batch_vjp(NAMES, ROWS)
    runs = [2, 0, 2]
    cotangent = {
        name: np.stack(
            [
                np.linspace(1.0, 2.0, np.shape(values)[1]) * (row + index + 1)
                for row in range(3)
            ]
        )
        for index, (name, values) in enumerate(outputs.items())
    }
    pulled = pullback(cotangent, runs)
    for row, run in enumerate(runs):
        _, pull = vjp(f, parameters(ROWS[run]))
        alone = pull({name: values[row] for name, values in cotangent.items()})
        np.testing.assert_array_equal(
            pulled[row], [alone[path][name] for path, name in NAMES]
        )


def test_a_batchs_values_are_a_row_of_its_names_by_run() -> None:
    f = function(crossbar(), OperatingPoint())
    with pytest.raises(ValueError, match="a row of 4 names by run"):
        f.batch(NAMES, ROWS[:, :3])


def test_a_batch_pullback_takes_the_outputs_and_the_runs_its_batch_has() -> None:
    _, pullback = function(crossbar(), OperatingPoint()).batch_vjp(NAMES, ROWS)
    with pytest.raises(ValueError, match="gives no output `nothing`"):
        pullback({"nothing": np.ones((3, 1))})
    with pytest.raises(ValueError, match="not one at 3"):
        pullback({}, [3])

# Copyright 2026 Maxence Faldor
"""Tests for the JAX bridge (`jolty.jax.function`).

Checks exact bit-for-bit parity with `jolty.vjp`, agreement with central finite
differences across all five differentiable analyses, `jax.jit`, `jax.grad`,
`jax.value_and_grad`, `jax.vjp`, `jax.jacrev`, `jax.remat`, single-batch `jax.vmap`
dispatch across cores, empty batches, nested `vmap`, mixed `in_axes`, structural
unknown changes (both added and dropped unknowns), unsorted and duplicate breakpoints,
linear interpolation transpose adjointness, both `jax_enable_x64=True` and `False`,
input and analysis validation, runtime failure messages, optional import error
handling, and free-threaded `3.15t` execution without re-enabling the GIL.
"""

import builtins
import importlib.util
import math
import sys
import sysconfig
from collections.abc import Callable, Mapping
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jolty.jax as jolty_jax_module
from jolty import (
    Builder,
    Net,
    Options,
    Role,
    Simulation,
    component,
    devices,
    results,
)
from jolty import function as jolty_function
from jolty import vjp as jolty_vjp
from jolty.errors import ElaborationError, SimulationError
from jolty.jax import PERIOD
from jolty.jax import function as jax_function
from jolty.plan import (
    Ac,
    DcSweep,
    Noise,
    OperatingPoint,
    OperatingPoints,
    PeriodicSteadyState,
    PoleZero,
    Stability,
    TransferFunction,
    Transient,
)
from jolty.units import F, H, MHz, V, kHz, kohm, nF, ns, ohm, pF, us
from jolty.waveforms import Pulse, Sine

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))

WEIGHTS: dict[str, dict[str, float]] = {
    f"W{row}{column}": {"resistance": 1e3 * (1 + row + 2 * column)}
    for row in range(2)
    for column in range(2)
}


@component
def crossbar(c: Builder) -> None:
    """Two-input, two-output loaded resistive crossbar."""
    ground = c.port("ground", Role.GROUND)
    for row, level in enumerate((0.3 * V, -0.2 * V)):
        source = c.place(devices.voltage_source(level), name=f"V{row}")
        c.connect(source.positive, name=f"in{row}")
        c.connect(source.negative, ground)
    for column in range(2):
        load = c.place(devices.resistor(1 * kohm), name=f"RL{column}")
        hold = c.place(devices.capacitor(1 * nF), name=f"CL{column}")
        c.connect(load.a, hold.a, name=f"out{column}")
        c.connect(load.b, hold.b, ground)
        for row in range(2):
            weight = c.place(devices.resistor(1 * kohm), name=f"W{row}{column}")
            c.connect(weight.a, Net(f"in{row}"))
            c.connect(weight.b, Net(f"out{column}"))


@component
def low_resistance_crossbar(c: Builder) -> None:
    """Crossbar whose nominal `W00` is below 1 ohm so `W00:current` is an unknown."""
    ground = c.port("ground", Role.GROUND)
    for row, level in enumerate((0.3 * V, -0.2 * V)):
        source = c.place(devices.voltage_source(level), name=f"V{row}")
        c.connect(source.positive, name=f"in{row}")
        c.connect(source.negative, ground)
    for column in range(2):
        load = c.place(devices.resistor(1 * kohm), name=f"RL{column}")
        hold = c.place(devices.capacitor(1 * nF), name=f"CL{column}")
        c.connect(load.a, hold.a, name=f"out{column}")
        c.connect(load.b, hold.b, ground)
        for row in range(2):
            nominal = 0.5 * ohm if (row == 0 and column == 0) else 1 * kohm
            weight = c.place(devices.resistor(nominal), name=f"W{row}{column}")
            c.connect(weight.a, Net(f"in{row}"))
            c.connect(weight.b, Net(f"out{column}"))


@component
def driven_rc_filter(c: Builder) -> None:
    """Sinusoidally driven RC low-pass filter for driven periodic steady state."""
    ground = c.port("ground", Role.GROUND)
    drive = Sine(offset=0 * V, amplitude=1 * V, frequency=1 * MHz)
    source = c.place(devices.voltage_source(drive), name="V1")
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    capacitor = c.place(devices.capacitor(100e-12 * F), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def van_der_pol_tank(c: Builder) -> None:
    """Autonomous LC tank oscillator with smooth cubic negative conductance."""
    ground = c.port("ground", Role.GROUND)
    inductance, capacitance = 1e-6, 1e-9
    impedance = math.sqrt(inductance / capacitance)
    expression = (
        f"-{0.2 / impedance!r} * v(x) * (1 - (v(x) ^ 2 + ({impedance!r} * i(p)) ^ 2))"
    )
    capacitor = c.place(devices.capacitor(capacitance * F), name="C1")
    inductor = c.place(devices.inductor(inductance * H), name="L1")
    regulation = c.place(devices.behavioral_current_source(expression), name="B1")
    c.connect(capacitor.a, inductor.a, regulation.positive, regulation.x, name="t")
    c.connect(inductor.b, regulation.p_positive, name="l")
    c.connect(capacitor.b, regulation.negative, regulation.p_negative, ground)


@component
def delayed_line_circuit(c: Builder) -> None:
    """Pulse into a lossless transmission line whose transient VJP is refused."""
    ground = c.port("ground", Role.GROUND)
    drive = Pulse(
        initial=0 * V,
        pulsed=1 * V,
        delay=1 * ns,
        rise=0.5 * ns,
        fall=0.5 * ns,
        width=5 * ns,
        period=20 * ns,
    )
    source = c.place(devices.voltage_source(drive), name="V1")
    feed = c.place(devices.resistor(50 * ohm), name="R1")
    line = c.place(devices.transmission_line(50 * ohm, 2 * ns), name="T1")
    load = c.place(devices.resistor(50 * ohm), name="RL")
    hold = c.place(devices.capacitor(1 * pF), name="CL")
    c.connect(source.positive, feed.a, name="in")
    c.connect(feed.b, line.positive1, name="near")
    c.connect(line.positive2, load.a, hold.a, name="out")
    c.connect(source.negative, line.negative1, line.negative2, load.b, hold.b, ground)


@component
def quadratic_fold_circuit(c: Builder) -> None:
    """Quadratic behavioral load: real DC root exists iff resistance <= 500 ohm."""
    ground = c.port("ground", Role.GROUND)
    resistor = c.place(devices.resistor(100 * ohm), name="R1")
    current = c.place(
        devices.behavioral_current_source("1e-3 * (v(out) ^ 2 + 1)"), name="B1"
    )
    c.connect(resistor.a, current.positive, current.out, name="out")
    c.connect(resistor.b, current.negative, ground)


def _as_jax_parameters(
    parameters: Mapping[str, Mapping[str, float]],
    *,
    dtype: np.dtype[np.floating] | type[np.floating] | jax.typing.DTypeLike,
) -> dict[str, dict[str, jax.Array]]:
    return {
        instance: {
            name: jnp.asarray(value, dtype=dtype) for name, value in inner.items()
        }
        for instance, inner in parameters.items()
    }


def _expect_real_scalar(value: object) -> float:
    assert isinstance(value, float)
    return value


def _central_difference(
    loss_function: Callable[[dict[str, dict[str, float]]], float],
    parameters: dict[str, dict[str, float]],
    relative_step: float = 1e-4,
) -> dict[str, dict[str, float]]:
    gradients: dict[str, dict[str, float]] = {}
    for instance, inner in parameters.items():
        gradients[instance] = {}
        for name, value in inner.items():
            step = value * relative_step
            plus = {**parameters, instance: {**inner, name: value + step}}
            minus = {**parameters, instance: {**inner, name: value - step}}
            gradients[instance][name] = (
                float(loss_function(plus)) - float(loss_function(minus))
            ) / (2.0 * step)
    return gradients


def test_operating_point_matches_jolty_vjp_bit_for_bit_and_finite_differences() -> None:
    analysis = OperatingPoint()
    wrapped = jax_function(crossbar(), analysis)
    native = jolty_function(crossbar(), analysis)

    with jax.enable_x64():
        targets = {"out0": 0.05, "out1": -0.02}

        def loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            outputs = wrapped(parameters)
            return jnp.sum((outputs["out0"] - targets["out0"]) ** 2) + jnp.sum(
                (outputs["out1"] - targets["out1"]) ** 2
            )

        jax_weights = _as_jax_parameters(WEIGHTS, dtype=jnp.float64)
        eager_outputs = wrapped(jax_weights)
        jitted_outputs = jax.jit(wrapped)(jax_weights)
        for name in ("out0", "out1"):
            np.testing.assert_array_equal(
                np.asarray(eager_outputs[name]), np.asarray(jitted_outputs[name])
            )

        value, eager_gradient = jax.value_and_grad(loss)(jax_weights)
        jitted_value, jitted_gradient = jax.jit(jax.value_and_grad(loss))(jax_weights)
        np.testing.assert_allclose(
            float(value), float(jitted_value), rtol=1e-15, atol=0.0
        )

        native_outputs, pullback = jolty_vjp(native, WEIGHTS)
        for name in ("out0", "out1"):
            np.testing.assert_array_equal(
                np.asarray(eager_outputs[name]), np.asarray(native_outputs[name])
            )
        cotangent = {
            name: 2.0 * (np.asarray(native_outputs[name]) - target)
            for name, target in targets.items()
        }
        expected = pullback(cotangent)
        for instance, inner in WEIGHTS.items():
            for name in inner:
                expected_value = _expect_real_scalar(expected[instance][name])
                np.testing.assert_array_equal(
                    float(eager_gradient[instance][name]), expected_value
                )
                np.testing.assert_array_equal(
                    float(jitted_gradient[instance][name]), expected_value
                )

        finite_difference = _central_difference(
            lambda current: float(loss(_as_jax_parameters(current, dtype=jnp.float64))),
            WEIGHTS,
        )
        largest = max(
            abs(float(eager_gradient[instance]["resistance"])) for instance in WEIGHTS
        )
        for instance in WEIGHTS:
            assert (
                abs(
                    float(eager_gradient[instance]["resistance"])
                    - finite_difference[instance]["resistance"]
                )
                <= 1e-6 * largest
            )


def test_dc_sweep_matches_jolty_vjp_bit_for_bit() -> None:
    analysis = DcSweep("V0", [0.1 * V, 0.2 * V, 0.4 * V])
    wrapped = jax_function(crossbar(), analysis)
    native = jolty_function(crossbar(), analysis)

    with jax.enable_x64():
        weights_output0 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        weights_output1 = jnp.array([-0.5, 0.25, 1.5], dtype=jnp.float64)

        def loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            outputs = wrapped(parameters)
            return jnp.sum(
                weights_output0 * outputs["out0"] + weights_output1 * outputs["out1"]
            )

        gradient = jax.jit(jax.grad(loss))(
            _as_jax_parameters(WEIGHTS, dtype=jnp.float64)
        )
        _, pullback = jolty_vjp(native, WEIGHTS)
        expected = pullback(
            {
                "out0": np.asarray(weights_output0),
                "out1": np.asarray(weights_output1),
            }
        )
        for instance in WEIGHTS:
            expected_value = _expect_real_scalar(expected[instance]["resistance"])
            np.testing.assert_array_equal(
                float(gradient[instance]["resistance"]), expected_value
            )


def test_ac_complex_wirtinger_sign_matches_jolty_vjp_and_finite_differences() -> None:
    """Checks symmetric |V|^2, asymmetric Re(V) + 2*Im(V), and pure-imaginary VJP."""
    analysis = Ac("V1", [1 * kHz, 100 * kHz, 1 * MHz])
    wrapped = jax_function(crossbar(), analysis)
    native = jolty_function(crossbar(), analysis)

    with jax.enable_x64():
        jax_weights = _as_jax_parameters(WEIGHTS, dtype=jnp.float64)

        def loss_asymmetric(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            phasor = wrapped(parameters)["out0"]
            return jnp.sum(phasor.real + 2.0 * phasor.imag)

        def loss_magnitude_squared(
            parameters: dict[str, dict[str, jax.Array]],
        ) -> jax.Array:
            phasor = wrapped(parameters)["out0"]
            return jnp.sum(phasor.real**2 + phasor.imag**2)

        gradient_asymmetric = jax.jit(jax.grad(loss_asymmetric))(jax_weights)
        gradient_magnitude = jax.jit(jax.grad(loss_magnitude_squared))(jax_weights)

        native_outputs, pullback = jolty_vjp(native, WEIGHTS)
        cotangent_asymmetric = np.full(3, 1.0 - 2.0j, dtype=np.complex128)
        expected_asymmetric = pullback({"out0": cotangent_asymmetric})
        cotangent_magnitude = 2.0 * np.conj(np.asarray(native_outputs["out0"]))
        expected_magnitude = pullback({"out0": cotangent_magnitude})

        finite_difference_asymmetric = _central_difference(
            lambda current: float(
                loss_asymmetric(_as_jax_parameters(current, dtype=jnp.float64))
            ),
            WEIGHTS,
        )
        finite_difference_magnitude = _central_difference(
            lambda current: float(
                loss_magnitude_squared(_as_jax_parameters(current, dtype=jnp.float64))
            ),
            WEIGHTS,
        )
        largest_asymmetric = max(
            abs(float(gradient_asymmetric[instance]["resistance"]))
            for instance in WEIGHTS
        )
        largest_magnitude = max(
            abs(float(gradient_magnitude[instance]["resistance"]))
            for instance in WEIGHTS
        )

        for instance in WEIGHTS:
            np.testing.assert_array_equal(
                float(gradient_asymmetric[instance]["resistance"]),
                float(np.real(expected_asymmetric[instance]["resistance"])),
            )
            np.testing.assert_array_equal(
                float(gradient_magnitude[instance]["resistance"]),
                float(np.real(expected_magnitude[instance]["resistance"])),
            )
            assert (
                abs(
                    float(gradient_asymmetric[instance]["resistance"])
                    - finite_difference_asymmetric[instance]["resistance"]
                )
                <= 1e-6 * largest_asymmetric
            )
            assert (
                abs(
                    float(gradient_magnitude[instance]["resistance"])
                    - finite_difference_magnitude[instance]["resistance"]
                )
                <= 1e-6 * largest_magnitude
            )

        # Direct jax.vjp with a purely imaginary cotangent.
        primal, vjp_function = jax.vjp(wrapped, jax_weights)
        zeros = {name: jnp.zeros_like(array) for name, array in primal.items()}
        imaginary_cotangent = jnp.full(3, -1.0j, dtype=jnp.complex128)
        (pulled_imaginary,) = vjp_function({**zeros, "out0": imaginary_cotangent})
        expected_imaginary = pullback({"out0": np.full(3, -1.0j, dtype=np.complex128)})
        for instance in WEIGHTS:
            np.testing.assert_array_equal(
                float(pulled_imaginary[instance]["resistance"]),
                float(np.real(expected_imaginary[instance]["resistance"])),
            )


def test_transient_breakpoints_match_public_jolty_vjp_and_finite_differences() -> None:
    breakpoints = (0.5 * us, 1.0 * us, 1.5 * us, 2.0 * us)
    analysis = Transient(2 * us, max_step=50 * ns, breakpoints=breakpoints)
    wrapped = jax_function(crossbar(), analysis)
    native = jolty_function(crossbar(), analysis)

    with jax.enable_x64():
        breakpoint_weights = jnp.array([1.0, -0.5, 2.0, 0.75], dtype=jnp.float64)

        def loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            outputs = wrapped(parameters)
            return jnp.sum(breakpoint_weights * outputs["out0"])

        jax_weights = _as_jax_parameters(WEIGHTS, dtype=jnp.float64)
        gradient = jax.jit(jax.grad(loss))(jax_weights)

        transient_result = Simulation(crossbar(), overrides=WEIGHTS).run(analysis)
        assert isinstance(transient_result, results.TransientResult)
        times = transient_result.times
        cotangent_series = np.zeros(len(times), dtype=np.float64)
        for target_seconds, weight in zip(
            (0.5e-6, 1.0e-6, 1.5e-6, 2.0e-6),
            np.asarray(breakpoint_weights),
            strict=True,
        ):
            match_index = int(np.searchsorted(times, target_seconds))
            assert times[match_index] == target_seconds
            cotangent_series[match_index] = float(weight)

        _, pullback = jolty_vjp(native, WEIGHTS)
        expected = pullback({"out0": cotangent_series})
        for instance in WEIGHTS:
            expected_value = _expect_real_scalar(expected[instance]["resistance"])
            np.testing.assert_array_equal(
                float(gradient[instance]["resistance"]), expected_value
            )

        finite_difference = _central_difference(
            lambda current: float(loss(_as_jax_parameters(current, dtype=jnp.float64))),
            WEIGHTS,
        )
        largest = max(
            abs(float(gradient[instance]["resistance"])) for instance in WEIGHTS
        )
        for instance in WEIGHTS:
            assert (
                abs(
                    float(gradient[instance]["resistance"])
                    - finite_difference[instance]["resistance"]
                )
                <= 1e-5 * largest
            )


def test_linear_interpolation_sample_and_scatter_are_exact_transposes() -> None:
    times = np.array([0.0, 1.0, 3.0, 6.0, 10.0], dtype=np.float64)
    values = np.array(
        [
            [1.0, -2.0, 0.5],
            [2.5, 0.0, -1.5],
            [-0.5, 3.0, 2.0],
            [4.0, -1.0, 1.0],
            [0.25, 2.5, -3.0],
        ],
        dtype=np.float64,
    )
    # Mix interior off-grid breakpoints, unsorted order, exact grid hits, duplicates.
    breakpoints = (4.5, 1.0, 0.25, 4.5, 8.2, 10.0, 2.1)
    base_indices = np.array([0, 2], dtype=np.intp)
    simulator_indices = np.array([2, 0], dtype=np.intp)
    indices = (base_indices, simulator_indices)

    sampled = jolty_jax_module._sample_at_breakpoints(  # noqa: SLF001 - direct unit test of transpose pair
        times, values, breakpoints, indices, width=3
    )
    assert np.all(np.isnan(sampled[:, 1]))

    cotangent_at_breakpoints = np.array(
        [
            [0.7, 0.0, -1.1],
            [-0.4, 0.0, 0.9],
            [1.5, 0.0, 0.3],
            [0.2, 0.0, 0.8],
            [-0.6, 0.0, 1.2],
            [0.5, 0.0, -0.5],
            [1.0, 0.0, 0.4],
        ],
        dtype=np.float64,
    )
    full_cotangent = np.zeros_like(values)
    jolty_jax_module._scatter_breakpoint_cotangents(  # noqa: SLF001 - direct unit test of transpose pair
        times, breakpoints, cotangent_at_breakpoints, indices, full_cotangent
    )

    forward_inner = float(
        np.sum(sampled[:, base_indices] * cotangent_at_breakpoints[:, base_indices])
    )
    adjoint_inner = float(np.sum(values * full_cotangent))
    np.testing.assert_allclose(forward_inner, adjoint_inner, rtol=1e-15, atol=0.0)

    with pytest.raises(SimulationError, match="outside solved time range"):
        jolty_jax_module._sample_at_breakpoints(  # noqa: SLF001 - bounds check
            times, values, (12.0,), indices, width=3
        )
    with pytest.raises(SimulationError, match="outside solved time range"):
        jolty_jax_module._scatter_breakpoint_cotangents(  # noqa: SLF001 - bounds check
            times, (12.0,), np.zeros((1, 3), dtype=np.float64), indices, full_cotangent
        )


def test_unsorted_and_duplicate_breakpoints_work_single_and_vmapped() -> None:
    breakpoints = (1.5 * us, 0.5 * us, 1.5 * us, 1.0 * us)
    transient = Transient(2 * us, max_step=50 * ns, breakpoints=breakpoints)
    wrapped_transient = jax_function(crossbar(), transient)

    periodic_breakpoints = (0.75 * us, 0.25 * us, 0.75 * us)
    periodic = PeriodicSteadyState(
        period=1 * us, max_step=25 * ns, breakpoints=periodic_breakpoints
    )
    wrapped_periodic = jax_function(driven_rc_filter(), periodic)

    with jax.enable_x64():
        jax_weights = _as_jax_parameters(WEIGHTS, dtype=jnp.float64)
        single_output = wrapped_transient(jax_weights)["out0"]
        assert single_output.shape == (4,)
        np.testing.assert_array_equal(
            np.asarray(single_output[0]), np.asarray(single_output[2])
        )

        weights_vector = jnp.array([1.2, -0.7, 0.4, 2.0], dtype=jnp.float64)

        def transient_loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            return jnp.sum(weights_vector * wrapped_transient(parameters)["out0"])

        gradient = jax.grad(transient_loss)(jax_weights)
        finite_difference = _central_difference(
            lambda current: float(
                transient_loss(_as_jax_parameters(current, dtype=jnp.float64))
            ),
            WEIGHTS,
        )
        largest = max(
            abs(float(gradient[instance]["resistance"])) for instance in WEIGHTS
        )
        for instance in WEIGHTS:
            assert (
                abs(
                    float(gradient[instance]["resistance"])
                    - finite_difference[instance]["resistance"]
                )
                <= 1e-5 * largest
            )

        batched_resistances = jnp.array([1000.0, 1800.0, 2500.0], dtype=jnp.float64)
        batched_inputs = {
            "W00": {"resistance": batched_resistances},
            "W01": {"resistance": jnp.asarray(3000.0, dtype=jnp.float64)},
        }
        in_axes = ({"W00": {"resistance": 0}, "W01": {"resistance": None}},)
        vmapped_output = jax.vmap(wrapped_transient, in_axes=in_axes)(batched_inputs)[
            "out0"
        ]
        assert vmapped_output.shape == (3, 4)
        for row_index in range(3):
            expected_single = wrapped_transient(
                {
                    "W00": {"resistance": batched_resistances[row_index]},
                    "W01": {"resistance": jnp.asarray(3000.0, dtype=jnp.float64)},
                }
            )["out0"]
            np.testing.assert_array_equal(
                np.asarray(vmapped_output[row_index]), np.asarray(expected_single)
            )

        periodic_parameters = {
            "R1": {"resistance": jnp.array([900.0, 1300.0], dtype=jnp.float64)},
            "C1": {"capacitance": jnp.asarray(120e-12, dtype=jnp.float64)},
        }
        periodic_axes = ({"R1": {"resistance": 0}, "C1": {"capacitance": None}},)
        periodic_output = jax.vmap(wrapped_periodic, in_axes=periodic_axes)(
            periodic_parameters
        )["out"]
        assert periodic_output.shape == (2, 3)
        np.testing.assert_array_equal(
            np.asarray(periodic_output[:, 0]), np.asarray(periodic_output[:, 2])
        )


def test_driven_periodic_steady_state_gradient() -> None:
    analysis = PeriodicSteadyState(
        period=1 * us,
        max_step=25 * ns,
        breakpoints=(0.25 * us, 0.5 * us, 0.75 * us),
    )
    wrapped = jax_function(driven_rc_filter(), analysis)
    parameters = {"R1": {"resistance": 1.2e3}, "C1": {"capacitance": 120e-12}}

    with jax.enable_x64():

        def loss(current: dict[str, dict[str, jax.Array]]) -> jax.Array:
            return jnp.sum(wrapped(current)["out"] ** 2)

        gradient = jax.jit(jax.grad(loss))(
            _as_jax_parameters(parameters, dtype=jnp.float64)
        )
        finite_difference = _central_difference(
            lambda current: float(loss(_as_jax_parameters(current, dtype=jnp.float64))),
            parameters,
        )
        for instance, inner in parameters.items():
            for name in inner:
                assert abs(
                    float(gradient[instance][name]) - finite_difference[instance][name]
                ) <= 1e-4 * abs(finite_difference[instance][name])


def test_autonomous_periodic_steady_state_period_gradient_and_scaling_law() -> None:
    """L * dT/dL + C * dT/dC = T on the van der Pol tank."""
    options = Options(relative_tolerance=1e-6)
    analysis = PeriodicSteadyState()
    wrapped = jax_function(van_der_pol_tank(), analysis, options)
    inductance, capacitance = 1e-6, 1e-9
    parameters = {"L1": {"inductance": inductance}, "C1": {"capacitance": capacitance}}

    with jax.enable_x64():
        jax_parameters = _as_jax_parameters(parameters, dtype=jnp.float64)
        outputs = wrapped(jax_parameters)
        assert set(outputs) == {PERIOD}
        assert outputs[PERIOD].shape == ()

        def period_function(current: dict[str, dict[str, jax.Array]]) -> jax.Array:
            return wrapped(current)[PERIOD]

        period, gradient = jax.jit(jax.value_and_grad(period_function))(jax_parameters)
        period_value = float(period)
        assert period.shape == ()
        euler_sum = inductance * float(
            gradient["L1"]["inductance"]
        ) + capacitance * float(gradient["C1"]["capacitance"])
        assert abs(euler_sum - period_value) <= 1e-9 * period_value


def test_vmap_dispatches_single_engine_batch_and_matches_loop_bit_for_bit() -> None:
    analysis = OperatingPoint()
    wrapped = jax_function(crossbar(), analysis)
    batch_size = 6

    with jax.enable_x64():
        resistance00 = jnp.linspace(800.0, 2000.0, batch_size, dtype=jnp.float64)
        resistance01 = jnp.linspace(1200.0, 3000.0, batch_size, dtype=jnp.float64)
        batched = {
            "W00": {"resistance": resistance00},
            "W01": {"resistance": resistance01},
            "W10": {"resistance": jnp.asarray(2000.0, dtype=jnp.float64)},
            "W11": {"resistance": jnp.asarray(4000.0, dtype=jnp.float64)},
        }
        in_axes = (
            {
                "W00": {"resistance": 0},
                "W01": {"resistance": 0},
                "W10": {"resistance": None},
                "W11": {"resistance": None},
            },
        )

        call_shapes: list[tuple[int, ...]] = []
        original_forward = jolty_jax_module._forward_batch  # noqa: SLF001 - verifies batch dispatch

        def spy_forward(
            plan: jolty_jax_module._AnalysisPlan,
            theta_host: np.ndarray,
            names: tuple[tuple[str, str], ...],
            output_dtype: np.dtype,
        ) -> np.ndarray:
            call_shapes.append(tuple(np.asarray(theta_host).shape))
            return original_forward(plan, theta_host, names, output_dtype)

        with patch.object(jolty_jax_module, "_forward_batch", side_effect=spy_forward):
            vmapped_outputs = jax.jit(jax.vmap(wrapped, in_axes=in_axes))(batched)

        assert call_shapes == [(batch_size, 4)]

        for row_index in range(batch_size):
            single = {
                "W00": {"resistance": resistance00[row_index]},
                "W01": {"resistance": resistance01[row_index]},
                "W10": {"resistance": jnp.asarray(2000.0, dtype=jnp.float64)},
                "W11": {"resistance": jnp.asarray(4000.0, dtype=jnp.float64)},
            }
            expected = wrapped(single)
            for name in ("out0", "out1"):
                np.testing.assert_array_equal(
                    np.asarray(vmapped_outputs[name][row_index]),
                    np.asarray(expected[name]),
                )

        def batch_loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            outputs = jax.vmap(wrapped, in_axes=in_axes)(parameters)
            return jnp.sum(outputs["out0"] ** 2 + 2.0 * outputs["out1"] ** 2)

        gradient_over_batch = jax.jit(jax.grad(batch_loss))(batched)

        def single_loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            outputs = wrapped(parameters)
            return jnp.sum(outputs["out0"] ** 2 + 2.0 * outputs["out1"] ** 2)

        per_example_gradients = jax.jit(
            jax.vmap(jax.grad(single_loss), in_axes=in_axes)
        )(batched)

        np.testing.assert_array_equal(
            np.asarray(gradient_over_batch["W00"]["resistance"]),
            np.asarray(per_example_gradients["W00"]["resistance"]),
        )
        np.testing.assert_allclose(
            float(gradient_over_batch["W10"]["resistance"]),
            float(jnp.sum(per_example_gradients["W10"]["resistance"])),
            rtol=1e-14,
            atol=0.0,
        )


def test_vmap_forward_and_gradient_across_all_analyses() -> None:
    with jax.enable_x64():
        dc_wrapped = jax_function(
            crossbar(), DcSweep("V0", [0.1 * V, 0.3 * V, 0.5 * V])
        )
        resistances = jnp.array([900.0, 1500.0, 2200.0], dtype=jnp.float64)
        batched = {"W00": {"resistance": resistances}}
        dc_gradient = jax.jit(
            jax.grad(
                lambda current: jnp.sum(jax.vmap(dc_wrapped)(current)["out0"] ** 2)
            )
        )(batched)
        for index in range(3):
            single_gradient = jax.grad(
                lambda current: jnp.sum(dc_wrapped(current)["out0"] ** 2)
            )({"W00": {"resistance": resistances[index]}})
            np.testing.assert_array_equal(
                float(dc_gradient["W00"]["resistance"][index]),
                float(single_gradient["W00"]["resistance"]),
            )

        ac_wrapped = jax_function(crossbar(), Ac("V1", [1 * kHz, 100 * kHz, 1 * MHz]))

        def ac_loss(current: dict[str, dict[str, jax.Array]]) -> jax.Array:
            phasor = jax.vmap(ac_wrapped)(current)["out0"]
            return jnp.sum(phasor.real + 2.0 * phasor.imag)

        def single_ac_loss(current: dict[str, dict[str, jax.Array]]) -> jax.Array:
            phasor = ac_wrapped(current)["out0"]
            return jnp.sum(phasor.real + 2.0 * phasor.imag)

        ac_gradient = jax.jit(jax.grad(ac_loss))(batched)
        for index in range(3):
            single_gradient = jax.grad(single_ac_loss)(
                {"W00": {"resistance": resistances[index]}}
            )
            np.testing.assert_array_equal(
                float(ac_gradient["W00"]["resistance"][index]),
                float(single_gradient["W00"]["resistance"]),
            )

        transient_wrapped = jax_function(
            crossbar(),
            Transient(1 * us, max_step=50 * ns, breakpoints=(0.5 * us, 1.0 * us)),
        )
        transient_gradient = jax.jit(
            jax.grad(
                lambda current: jnp.sum(
                    jax.vmap(transient_wrapped)(current)["out0"] ** 2
                )
            )
        )(batched)
        for index in range(3):
            single_gradient = jax.grad(
                lambda current: jnp.sum(transient_wrapped(current)["out0"] ** 2)
            )({"W00": {"resistance": resistances[index]}})
            np.testing.assert_array_equal(
                float(transient_gradient["W00"]["resistance"][index]),
                float(single_gradient["W00"]["resistance"]),
            )

        driven_wrapped = jax_function(
            driven_rc_filter(),
            PeriodicSteadyState(
                period=1 * us, max_step=50 * ns, breakpoints=(0.25 * us, 0.75 * us)
            ),
        )
        driven_resistances = jnp.array([800.0, 1400.0], dtype=jnp.float64)
        driven_gradient = jax.jit(
            jax.grad(
                lambda current: jnp.sum(jax.vmap(driven_wrapped)(current)["out"] ** 2)
            )
        )({"R1": {"resistance": driven_resistances}})
        for index in range(2):
            single_gradient = jax.grad(
                lambda current: jnp.sum(driven_wrapped(current)["out"] ** 2)
            )({"R1": {"resistance": driven_resistances[index]}})
            np.testing.assert_array_equal(
                float(driven_gradient["R1"]["resistance"][index]),
                float(single_gradient["R1"]["resistance"]),
            )

        autonomous_wrapped = jax_function(
            van_der_pol_tank(),
            PeriodicSteadyState(),
            Options(relative_tolerance=1e-6),
        )
        inductances = jnp.array([0.9e-6, 1.1e-6], dtype=jnp.float64)
        autonomous_periods = jax.vmap(autonomous_wrapped)(
            {"L1": {"inductance": inductances}}
        )[PERIOD]
        assert autonomous_periods.shape == (2,)
        autonomous_gradient = jax.jit(
            jax.grad(
                lambda current: jnp.sum(jax.vmap(autonomous_wrapped)(current)[PERIOD])
            )
        )({"L1": {"inductance": inductances}})
        for index in range(2):
            single_period, single_gradient = jax.value_and_grad(
                lambda current: autonomous_wrapped(current)[PERIOD]
            )({"L1": {"inductance": inductances[index]}})
            np.testing.assert_array_equal(
                float(autonomous_periods[index]), float(single_period)
            )
            np.testing.assert_array_equal(
                float(autonomous_gradient["L1"]["inductance"][index]),
                float(single_gradient["L1"]["inductance"]),
            )


def test_empty_batch_vmap_forward_and_gradient() -> None:
    with jax.enable_x64():
        operating_point_wrapped = jax_function(crossbar(), OperatingPoint())
        ac_wrapped = jax_function(crossbar(), Ac("V1", [1 * kHz, 1 * MHz]))
        autonomous_wrapped = jax_function(van_der_pol_tank(), PeriodicSteadyState())

        for shape in ((0,), (0, 3)):
            empty = jnp.empty(shape, dtype=jnp.float64)
            if len(shape) == 1:
                map_operating_point = jax.vmap(operating_point_wrapped)
                map_ac = jax.vmap(ac_wrapped)
                map_autonomous = jax.vmap(autonomous_wrapped)
            else:
                map_operating_point = jax.vmap(jax.vmap(operating_point_wrapped))
                map_ac = jax.vmap(jax.vmap(ac_wrapped))
                map_autonomous = jax.vmap(jax.vmap(autonomous_wrapped))

            operating_point_output = map_operating_point(
                {"W00": {"resistance": empty}}
            )["out0"]
            assert operating_point_output.shape == (*shape, 1)
            assert operating_point_output.dtype == jnp.float64
            operating_point_gradient = jax.grad(
                lambda current, mapper=map_operating_point: jnp.sum(
                    mapper(current)["out0"]
                )
            )({"W00": {"resistance": empty}})
            assert operating_point_gradient["W00"]["resistance"].shape == shape

            ac_output = map_ac({"W00": {"resistance": empty}})["out0"]
            assert ac_output.shape == (*shape, 2)
            assert ac_output.dtype == jnp.complex128
            ac_gradient = jax.grad(
                lambda current, mapper=map_ac: jnp.sum(mapper(current)["out0"].real)
            )({"W00": {"resistance": empty}})
            assert ac_gradient["W00"]["resistance"].shape == shape

            autonomous_output = map_autonomous({"L1": {"inductance": empty}})[PERIOD]
            assert autonomous_output.shape == shape
            assert autonomous_output.dtype == jnp.float64
            autonomous_gradient = jax.grad(
                lambda current, mapper=map_autonomous: jnp.sum(mapper(current)[PERIOD])
            )({"L1": {"inductance": empty}})
            assert autonomous_gradient["L1"]["inductance"].shape == shape


def test_nested_vmap_repeated_vjp_and_zero_cotangent_shortcut() -> None:
    analysis = OperatingPoint()
    wrapped = jax_function(crossbar(), analysis)

    with jax.enable_x64():
        grid_first = jnp.array([1000.0, 2000.0], dtype=jnp.float64)
        grid_second = jnp.array([1500.0, 2500.0, 3500.0], dtype=jnp.float64)

        nested = jax.jit(
            jax.vmap(
                jax.vmap(
                    lambda first, second: wrapped(
                        {"W00": {"resistance": first}, "W01": {"resistance": second}}
                    )["out0"],
                    in_axes=(None, 0),
                ),
                in_axes=(0, None),
            )
        )
        output_grid = nested(grid_first, grid_second)
        assert output_grid.shape == (2, 3, 1)
        for row_index in range(2):
            for column_index in range(3):
                single = wrapped(
                    {
                        "W00": {"resistance": grid_first[row_index]},
                        "W01": {"resistance": grid_second[column_index]},
                    }
                )["out0"]
                np.testing.assert_array_equal(
                    np.asarray(output_grid[row_index, column_index]),
                    np.asarray(single),
                )

        jax_weights = _as_jax_parameters(WEIGHTS, dtype=jnp.float64)
        primal, vjp_function = jax.vjp(wrapped, jax_weights)
        zeros = {name: jnp.zeros_like(array) for name, array in primal.items()}
        (gradient_first,) = vjp_function(
            {**zeros, "out0": jnp.ones_like(primal["out0"])}
        )
        (gradient_second,) = vjp_function(
            {**zeros, "out0": 3.0 * jnp.ones_like(primal["out0"])}
        )
        for instance in WEIGHTS:
            np.testing.assert_allclose(
                float(gradient_second[instance]["resistance"]),
                3.0 * float(gradient_first[instance]["resistance"]),
                rtol=1e-14,
            )

        # Zero cotangent shortcuts without invoking the simulator elaboration/vjp.
        with patch.object(
            jolty_jax_module, "_build_overrides", side_effect=AssertionError
        ):
            (gradient_zero,) = vjp_function(zeros)
        for instance in WEIGHTS:
            assert float(gradient_zero[instance]["resistance"]) == 0.0


def test_remat_checkpoint_and_jacrev_across_analyses() -> None:
    with jax.enable_x64():
        jax_weights = _as_jax_parameters(WEIGHTS, dtype=jnp.float64)
        operating_point_wrapped = jax_function(crossbar(), OperatingPoint())

        def remat_loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            outputs = jax.checkpoint(operating_point_wrapped)(parameters)
            return jnp.sum(outputs["out0"] ** 2 + outputs["out1"] ** 2)

        def plain_loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            outputs = operating_point_wrapped(parameters)
            return jnp.sum(outputs["out0"] ** 2 + outputs["out1"] ** 2)

        remat_value, remat_gradient = jax.jit(jax.value_and_grad(remat_loss))(
            jax_weights
        )
        plain_value, plain_gradient = jax.jit(jax.value_and_grad(plain_loss))(
            jax_weights
        )
        np.testing.assert_allclose(float(remat_value), float(plain_value), rtol=1e-15)
        for instance in WEIGHTS:
            np.testing.assert_array_equal(
                float(remat_gradient[instance]["resistance"]),
                float(plain_gradient[instance]["resistance"]),
            )

        # jacrev across OperatingPoint, DcSweep, Ac, Transient, and driven periodic.
        operating_point_jacobian = jax.jacrev(
            lambda current: operating_point_wrapped(current)["out0"]
        )(jax_weights)
        assert operating_point_jacobian["W00"]["resistance"].shape == (1,)

        dc_wrapped = jax_function(crossbar(), DcSweep("V0", [0.1 * V, 0.3 * V]))
        dc_jacobian = jax.jacrev(lambda current: dc_wrapped(current)["out0"])(
            jax_weights
        )
        assert dc_jacobian["W00"]["resistance"].shape == (2,)

        ac_wrapped = jax_function(crossbar(), Ac("V1", [1 * kHz, 1 * MHz]))
        ac_jacobian = jax.jacrev(lambda current: ac_wrapped(current)["out0"].real)(
            jax_weights
        )
        assert ac_jacobian["W00"]["resistance"].shape == (2,)

        transient_wrapped = jax_function(
            crossbar(),
            Transient(1 * us, max_step=50 * ns, breakpoints=(0.5 * us, 1.0 * us)),
        )
        transient_jacobian = jax.jacrev(
            lambda current: transient_wrapped(current)["out0"]
        )(jax_weights)
        assert transient_jacobian["W00"]["resistance"].shape == (2,)

        driven_wrapped = jax_function(
            driven_rc_filter(),
            PeriodicSteadyState(
                period=1 * us, max_step=50 * ns, breakpoints=(0.25 * us, 0.75 * us)
            ),
        )
        driven_jacobian = jax.jacrev(lambda current: driven_wrapped(current)["out"])(
            {"R1": {"resistance": jnp.asarray(1.2e3, dtype=jnp.float64)}}
        )
        assert driven_jacobian["R1"]["resistance"].shape == (2,)


def test_structural_changes_added_and_dropped_unknowns() -> None:
    with jax.enable_x64():
        # Direction 1: adding an unknown (1000 ohm -> 0.5 ohm).
        wrapped_added = jax_function(crossbar(), OperatingPoint())
        low_resistance = {**WEIGHTS, "W00": {"resistance": 0.5}}

        def added_loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            return jnp.sum(wrapped_added(parameters)["out0"] ** 2)

        added_gradient = jax.jit(jax.grad(added_loss))(
            _as_jax_parameters(low_resistance, dtype=jnp.float64)
        )
        finite_difference = _central_difference(
            lambda current: float(
                added_loss(_as_jax_parameters(current, dtype=jnp.float64))
            ),
            low_resistance,
        )
        largest = max(
            abs(float(added_gradient[instance]["resistance"]))
            for instance in low_resistance
        )
        for instance in low_resistance:
            assert (
                abs(
                    float(added_gradient[instance]["resistance"])
                    - finite_difference[instance]["resistance"]
                )
                <= 1e-5 * largest
            )

        # Direction 2: dropping an unknown (0.5 ohm -> 1000 ohm drops `W00:current`).
        wrapped_dropped = jax_function(low_resistance_crossbar(), OperatingPoint())
        high_parameters = _as_jax_parameters(WEIGHTS, dtype=jnp.float64)
        single_output = wrapped_dropped(high_parameters)
        assert "W00:current" in single_output
        assert bool(jnp.all(jnp.isnan(single_output["W00:current"])))
        assert bool(jnp.all(jnp.isfinite(single_output["out0"])))

        # Batched vmap also places NaN in the dropped unknown column.
        batched_resistances = jnp.array([0.5, 1000.0], dtype=jnp.float64)
        batched_output = jax.vmap(wrapped_dropped)(
            {"W00": {"resistance": batched_resistances}}
        )
        assert bool(jnp.isfinite(batched_output["W00:current"][0, 0]))
        assert bool(jnp.isnan(batched_output["W00:current"][1, 0]))

        # Zero cotangent on dropped column succeeds cleanly.
        primal, vjp_function = jax.vjp(wrapped_dropped, high_parameters)
        zeros = {name: jnp.zeros_like(array) for name, array in primal.items()}
        (gradient_valid,) = vjp_function(
            {**zeros, "out0": jnp.ones_like(primal["out0"])}
        )
        assert math.isfinite(float(gradient_valid["W00"]["resistance"]))

        # Non-zero cotangent on dropped column triggers SimulationError via callback.
        callback_errors = (SimulationError, ValueError, jax.errors.JaxRuntimeError)
        with pytest.raises(callback_errors, match=r"SimulationError.*W00:current"):
            vjp_function({**zeros, "W00:current": jnp.ones_like(primal["W00:current"])})

        # NaN cotangent on dropped column also triggers SimulationError via callback.
        with pytest.raises(callback_errors, match=r"SimulationError.*W00:current"):
            vjp_function(
                {
                    **zeros,
                    "W00:current": jnp.full_like(primal["W00:current"], jnp.nan),
                }
            )

        # Direct host-side check verifies unwrapped SimulationError type directly.
        plan = jolty_jax_module._build_analysis_plan(  # noqa: SLF001 - direct unit check
            low_resistance_crossbar(), OperatingPoint(), None
        )
        names = tuple((instance, "resistance") for instance in sorted(WEIGHTS))
        theta_row = np.array(
            [WEIGHTS[instance]["resistance"] for instance in sorted(WEIGHTS)],
            dtype=np.float64,
        )
        dropped_cotangent = np.zeros((1, len(plan.unknowns)), dtype=np.float64)
        dropped_cotangent[0, plan.unknowns.index("W00:current")] = 1.0
        with pytest.raises(SimulationError, match="W00:current"):
            jolty_jax_module._backward_batch(  # noqa: SLF001 - direct unit check
                plan, theta_row, dropped_cotangent, names, np.dtype(np.float64)
            )

        # Complex AC dropped unknown places NaN + NaNj and rejects non-zero / NaN.
        wrapped_dropped_ac = jax_function(
            low_resistance_crossbar(), Ac("V1", [1 * kHz, 1 * MHz])
        )
        ac_output = wrapped_dropped_ac(high_parameters)
        dropped_phasor = np.asarray(ac_output["W00:current"])
        assert np.all(np.isnan(dropped_phasor.real))
        assert np.all(np.isnan(dropped_phasor.imag))
        ac_primal, ac_vjp = jax.vjp(wrapped_dropped_ac, high_parameters)
        ac_zeros = {name: jnp.zeros_like(array) for name, array in ac_primal.items()}
        with pytest.raises(callback_errors, match=r"SimulationError.*W00:current"):
            ac_vjp(
                {
                    **ac_zeros,
                    "W00:current": jnp.full_like(
                        ac_primal["W00:current"], complex(0.0, 1.0)
                    ),
                }
            )


def test_float32_and_integer_input_dtypes() -> None:
    analysis = Ac("V1", [1 * kHz, 1 * MHz])
    wrapped = jax_function(crossbar(), analysis)
    with jax.enable_x64(new_val=False):
        parameters_float32 = _as_jax_parameters(WEIGHTS, dtype=jnp.float32)
        outputs = wrapped(parameters_float32)
        assert outputs["out0"].dtype == jnp.complex64
        gradient = jax.jit(
            jax.grad(lambda current: jnp.sum(wrapped(current)["out0"].real))
        )(parameters_float32)
        assert gradient["W00"]["resistance"].dtype == jnp.float32

    with jax.enable_x64():
        # float32 input under x64 enabled promotes cleanly to float64 / complex128.
        mixed_output = wrapped(
            {"W00": {"resistance": jnp.asarray(1000.0, dtype=jnp.float32)}}
        )
        assert mixed_output["out0"].dtype == jnp.complex128
        # Integer input forward works cleanly.
        integer_output = wrapped({"W00": {"resistance": 1000}})
        jax_integer_output = wrapped(
            {"W00": {"resistance": jnp.asarray(1000, dtype=jnp.int32)}}
        )
        np.testing.assert_array_equal(
            np.asarray(integer_output["out0"]),
            np.asarray(jax_integer_output["out0"]),
        )


def test_empty_parameters_key_order_aliasing_and_runner_cache() -> None:
    wrapped = jax_function(crossbar(), OperatingPoint())

    with jax.enable_x64():
        empty_output = wrapped({})
        assert empty_output["out0"].shape == (1,)
        empty_gradient = jax.grad(lambda current: jnp.sum(wrapped(current)["out0"]))({})
        assert empty_gradient == {}

        reversed_keys = {
            "W11": {"resistance": jnp.asarray(4000.0, dtype=jnp.float64)},
            "W00": {"resistance": jnp.asarray(1000.0, dtype=jnp.float64)},
        }
        sorted_keys = {
            "W00": {"resistance": jnp.asarray(1000.0, dtype=jnp.float64)},
            "W11": {"resistance": jnp.asarray(4000.0, dtype=jnp.float64)},
        }
        gradient_reversed = jax.grad(lambda current: jnp.sum(wrapped(current)["out0"]))(
            reversed_keys
        )
        gradient_sorted = jax.grad(lambda current: jnp.sum(wrapped(current)["out0"]))(
            sorted_keys
        )
        np.testing.assert_array_equal(
            float(gradient_reversed["W00"]["resistance"]),
            float(gradient_sorted["W00"]["resistance"]),
        )
        np.testing.assert_array_equal(
            float(gradient_reversed["W11"]["resistance"]),
            float(gradient_sorted["W11"]["resistance"]),
        )

        # Shared/aliased parameter leaf accumulates both partial derivatives.
        def tied(resistance: jax.Array) -> jax.Array:
            return jnp.sum(
                wrapped(
                    {
                        "W00": {"resistance": resistance},
                        "W10": {"resistance": resistance},
                    }
                )["out0"]
            )

        resistance_value = jnp.asarray(1500.0, dtype=jnp.float64)
        gradient_tied = float(jax.grad(tied)(resistance_value))
        gradient_separate = jax.grad(lambda current: jnp.sum(wrapped(current)["out0"]))(
            {
                "W00": {"resistance": resistance_value},
                "W10": {"resistance": resistance_value},
            }
        )
        np.testing.assert_allclose(
            gradient_tied,
            float(gradient_separate["W00"]["resistance"])
            + float(gradient_separate["W10"]["resistance"]),
            rtol=1e-14,
        )

        # Repeated eager calls hit the runner cache without re-running validation.
        wrapped({"W00": {"resistance": jnp.asarray(1200.0, dtype=jnp.float64)}})
        with patch.object(
            jolty_jax_module, "_validate_parameter_names", side_effect=AssertionError
        ):
            cached_output = wrapped(
                {"W00": {"resistance": jnp.asarray(1600.0, dtype=jnp.float64)}}
            )
            assert bool(jnp.isfinite(cached_output["out0"][0]))


def test_construction_time_analysis_refusals() -> None:
    with pytest.raises(ValueError, match="at least one sweep level"):
        jax_function(crossbar(), DcSweep("V0", []))
    with pytest.raises(ValueError, match="at least one frequency"):
        jax_function(crossbar(), Ac("V1", []))
    with pytest.raises(ValueError, match="stop time must be positive"):
        jax_function(crossbar(), Transient(stop=0 * us, breakpoints=(0.1 * us,)))
    with pytest.raises(ValueError, match="stop time must be positive"):
        jax_function(crossbar(), Transient(stop=-1 * us, breakpoints=(0.1 * us,)))
    with pytest.raises(ValueError, match="breakpoints"):
        jax_function(crossbar(), Transient(stop=2 * us))
    with pytest.raises(ValueError, match=r"must lie in \(0,"):
        jax_function(crossbar(), Transient(stop=2 * us, breakpoints=(0 * us,)))
    with pytest.raises(ValueError, match=r"must lie in \(0,"):
        jax_function(crossbar(), Transient(stop=2 * us, breakpoints=(-0.5 * us,)))
    with pytest.raises(ValueError, match=r"must lie in \(0,"):
        jax_function(crossbar(), Transient(stop=2 * us, breakpoints=(2.5 * us,)))
    with pytest.raises(ValueError, match="period must be positive"):
        jax_function(
            driven_rc_filter(),
            PeriodicSteadyState(period=0 * us, breakpoints=(0.1 * us,)),
        )
    with pytest.raises(ValueError, match="period must be positive"):
        jax_function(
            driven_rc_filter(),
            PeriodicSteadyState(period=-1 * us, breakpoints=(0.1 * us,)),
        )
    with pytest.raises(ValueError, match="breakpoints"):
        jax_function(driven_rc_filter(), PeriodicSteadyState(period=1 * us))
    with pytest.raises(ValueError, match=r"must lie in \(0,"):
        jax_function(
            driven_rc_filter(),
            PeriodicSteadyState(period=1 * us, breakpoints=(0 * us,)),
        )
    with pytest.raises(ValueError, match=r"must lie in \(0,"):
        jax_function(
            driven_rc_filter(),
            PeriodicSteadyState(period=1 * us, breakpoints=(1.5 * us,)),
        )
    with pytest.raises(ValueError, match="breakpoints"):
        jax_function(
            van_der_pol_tank(),
            PeriodicSteadyState(breakpoints=(0.1 * us,)),
        )

    for unsupported in (
        OperatingPoints(),
        Noise("out0", [1 * kHz]),
        Stability("V0", [1 * kHz]),
        PoleZero(source="V0", output="out0"),
        TransferFunction(source="V0", output="out0"),
    ):
        with pytest.raises(ValueError, match="unsupported analysis"):
            jax_function(crossbar(), unsupported)  # ty: ignore[invalid-argument-type] - the wrong type is tested

    class _FoundrySimulatorStub:
        foundry_devices = ("M1",)

    with (
        patch(
            "jolty.simulate.Elaborated.simulator",
            return_value=_FoundrySimulatorStub(),
        ),
        pytest.raises(ValueError, match="foundry devices"),
    ):
        jax_function(crossbar(), OperatingPoint())


def test_invalid_parameter_and_autodiff_refusals() -> None:
    wrapped = jax_function(crossbar(), OperatingPoint())

    with pytest.raises(TypeError, match="mapping of instance path"):
        wrapped([1000.0])  # ty: ignore[invalid-argument-type] - the wrong type is tested
    with pytest.raises(TypeError, match="mapping of instance path"):
        wrapped(None)  # ty: ignore[invalid-argument-type] - the wrong type is tested
    with pytest.raises(TypeError, match="mapping of instance path"):
        wrapped({1: {"resistance": 1000.0}})  # ty: ignore[invalid-argument-type] - the wrong type is tested
    with pytest.raises(TypeError, match="mapping of instance path"):
        wrapped({"W00": 1000.0})  # ty: ignore[invalid-argument-type] - the wrong type is tested
    with pytest.raises(TypeError, match="must be a string"):
        wrapped({"W00": {1: 1000.0}})  # ty: ignore[invalid-argument-type] - the wrong type is tested
    with pytest.raises(TypeError, match="not bool"):
        wrapped({"W00": {"resistance": True}})
    with pytest.raises(TypeError, match="real numeric dtype"):
        wrapped({"W00": {"resistance": jnp.asarray(1, dtype=jnp.bool_)}})
    with pytest.raises(TypeError, match="real numeric dtype"):
        wrapped({"W00": {"resistance": 1000.0 + 1.0j}})
    with pytest.raises(ValueError, match="must be a scalar"):
        wrapped({"W00": {"resistance": jnp.array([1000.0, 2000.0])}})
    with pytest.raises(ElaborationError):
        wrapped({"NONEXISTENT": {}})
    with pytest.raises(ElaborationError):
        jax.jit(jax.grad(lambda current: jnp.sum(wrapped(current)["out0"])))(
            {"W00": {"nonexistent": jnp.asarray(1.0)}}
        )

    # Autodiff refusals: forward-mode jvp and second-order grad.
    with jax.enable_x64():
        with pytest.raises(TypeError, match="forward-mode"):
            jax.jvp(
                wrapped,
                ({"W00": {"resistance": jnp.asarray(1000.0)}},),
                ({"W00": {"resistance": jnp.asarray(1.0)}},),
            )
        with pytest.raises(ValueError, match="Pure callbacks do not support JVP"):
            jax.grad(
                jax.grad(
                    lambda value: jnp.sum(
                        wrapped({"W00": {"resistance": value}})["out0"]
                    )
                )
            )(jnp.asarray(1000.0, dtype=jnp.float64))


def test_runtime_errors_under_eager_jit_and_vmap() -> None:
    wrapped = jax_function(crossbar(), OperatingPoint())
    callback_errors = (
        ElaborationError,
        SimulationError,
        ValueError,
        jax.errors.JaxRuntimeError,
    )

    with jax.enable_x64():
        # Negative resistance triggers ElaborationError in eager, jit, and vmap.
        with pytest.raises(callback_errors, match=r"ElaborationError.*positive"):
            wrapped({"W00": {"resistance": jnp.asarray(-1.0, dtype=jnp.float64)}})
        with pytest.raises(callback_errors, match=r"ElaborationError.*positive"):
            jax.jit(wrapped)(
                {"W00": {"resistance": jnp.asarray(-1.0, dtype=jnp.float64)}}
            )
        with pytest.raises(callback_errors, match=r"ElaborationError.*positive"):
            jax.vmap(wrapped)(
                {"W00": {"resistance": jnp.array([1000.0, -1.0], dtype=jnp.float64)}}
            )

        # NaN and Inf parameters trigger ValueError via JSON allow_nan=False.
        for bad_value in (float("nan"), float("inf")):
            with pytest.raises(callback_errors, match="JSON compliant"):
                wrapped(
                    {"W00": {"resistance": jnp.asarray(bad_value, dtype=jnp.float64)}}
                )
            with pytest.raises(callback_errors, match="JSON compliant"):
                jax.vmap(wrapped)(
                    {
                        "W00": {
                            "resistance": jnp.array(
                                [1000.0, bad_value], dtype=jnp.float64
                            )
                        }
                    }
                )

        # Batched forward simulation error reports the failing batch run index.
        failing_wrapped = jax_function(
            quadratic_fold_circuit(),
            OperatingPoint(),
            Options(max_dc_iterations=10),
        )
        with pytest.raises(callback_errors, match=r"SimulationError.*batch run 1:"):
            jax.vmap(failing_wrapped)(
                {"R1": {"resistance": jnp.array([100.0, 1000.0], dtype=jnp.float64)}}
            )

        # Delay-line transient VJP is refused cleanly on single and vmapped runs.
        line_wrapped = jax_function(
            delayed_line_circuit(),
            Transient(6 * ns, max_step=0.2 * ns, breakpoints=(3 * ns, 6 * ns)),
        )
        with pytest.raises(callback_errors, match="SimulationError"):
            jax.grad(lambda current: jnp.sum(line_wrapped(current)["out"]))(
                {"RL": {"resistance": jnp.asarray(50.0, dtype=jnp.float64)}}
            )
        with pytest.raises(callback_errors, match=r"SimulationError.*batch run 0:"):
            jax.grad(lambda current: jnp.sum(jax.vmap(line_wrapped)(current)["out"]))(
                {"RL": {"resistance": jnp.array([50.0, 75.0], dtype=jnp.float64)}}
            )


def test_optional_import_error_masks_only_jax_and_jaxlib() -> None:
    assert jolty_jax_module.__file__ is not None
    module_path = Path(jolty_jax_module.__file__)

    def load_with_blocked(blocked_module: str) -> None:
        real_import = builtins.__import__

        def fake_import(
            name: str,
            globals_mapping: Mapping[str, object] | None = None,
            locals_mapping: Mapping[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "jax" or name.startswith("jax."):
                error = ModuleNotFoundError(f"No module named {blocked_module!r}")
                error.name = blocked_module
                raise error
            return real_import(name, globals_mapping, locals_mapping, fromlist, level)

        spec = importlib.util.spec_from_file_location("_jolty_jax_probe", module_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with patch.object(builtins, "__import__", side_effect=fake_import):
            spec.loader.exec_module(module)

    with pytest.raises(ImportError, match=r"jolty\[jax\]"):
        load_with_blocked("jax")
    with pytest.raises(ImportError, match=r"jolty\[jax\]"):
        load_with_blocked("jaxlib.xla_extension")
    with pytest.raises(ModuleNotFoundError, match="scipy"):
        load_with_blocked("scipy")


@pytest.mark.skipif(not FREE_THREADED, reason="the interpreter has a GIL to keep")
def test_jax_bridge_keeps_gil_disabled_on_free_threaded_python() -> None:
    assert not sys._is_gil_enabled()  # noqa: SLF001 - CPython's only way to ask
    wrapped = jax_function(crossbar(), OperatingPoint())
    autonomous = jax_function(
        van_der_pol_tank(), PeriodicSteadyState(), Options(relative_tolerance=1e-6)
    )
    with jax.enable_x64():
        single_gradient = jax.grad(
            lambda current: jnp.sum(wrapped(current)["out0"] ** 2)
        )(_as_jax_parameters(WEIGHTS, dtype=jnp.float64))
        assert np.isfinite(float(single_gradient["W00"]["resistance"]))

        batched_gradient = jax.grad(
            lambda current: jnp.sum(jax.vmap(wrapped)(current)["out0"] ** 2)
        )({"W00": {"resistance": jnp.linspace(800.0, 2000.0, 8, dtype=jnp.float64)}})
        assert bool(jnp.all(jnp.isfinite(batched_gradient["W00"]["resistance"])))

        autonomous_gradient = jax.grad(
            lambda current: jnp.sum(jax.vmap(autonomous)(current)[PERIOD])
        )({"L1": {"inductance": jnp.array([0.9e-6, 1.1e-6], dtype=jnp.float64)}})
        assert bool(jnp.all(jnp.isfinite(autonomous_gradient["L1"]["inductance"])))

    assert not sys._is_gil_enabled()  # noqa: SLF001 - CPython's only way to ask

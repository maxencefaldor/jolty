# Copyright 2026 Maxence Faldor
"""Tests for the JAX bridge (`jolty.jax.function`).

Verifies bit-for-bit parity with `jolty.vjp` and central finite-difference agreement
across all six differentiable analyses. Covers `jax.jit`, `jax.grad`,
`jax.value_and_grad`, `jax.vjp`, `jax.jacrev`, `jax.remat`, and batched `jax.vmap`
matching scalar loops bit for bit. Exercises empty batches, nested `vmap`, mixed
`in_axes`, pullbacks retained from forward passes, and structural unknown changes.
Validates unsorted and duplicate breakpoints, `jax_enable_x64` modes, input checks,
runtime errors, optional import handling, and free-threaded `3.15t` execution without
re-enabling the GIL.
"""

import builtins
import importlib.util
import math
import sys
import sysconfig
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

import jolty.jax as jolty_jax_module
from jolty import (
    Builder,
    Component,
    Options,
    Role,
    Simulation,
    component,
    devices,
    results,
)
from jolty import function as jolty_function
from jolty import vjp as jolty_vjp
from jolty.errors import ElaborationError, SimulationError, SpiceError
from jolty.jax import LOOP_GAIN, NOISE_DENSITY, PERIOD, REFERRED_NOISE, JaxOutputs
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
    decade,
)
from jolty.settings import Settings
from jolty.simulate import BatchPullback, Elaborated, Function, Outputs
from jolty.spice import Library, SubcircuitPlacement, kits, sky130
from jolty.units import F, H, Hz, MHz, V, kHz, kohm, mA, ns, ohm, pF, uA, um, us
from jolty.waveforms import Pulse, Sine
from sample_circuits import CROSSBAR_WEIGHTS as WEIGHTS
from sample_circuits import crossbar

CENTRAL_DIFF_RTOL: Final = 1e-6
"""Relative tolerance for `step=1e-4` central differences (O(step^2) + rounding)."""

TRANSIENT_CENTRAL_DIFF_RTOL: Final = 1e-5
"""Relative tolerance for `step=1e-4` transient differences with step-shift noise."""

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


@component
def driven_rc_filter(c: Builder) -> None:
    """Builds a sinusoidally driven RC low-pass filter for periodic steady state."""
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
    """Builds an autonomous LC tank oscillator with smooth cubic conductance."""
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
    """Builds a pulse-driven lossless line whose transient VJP is refused."""
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
    """Builds a quadratic load with a real DC root when resistance <= 500 ohm."""
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
            step = relative_step if value == 0.0 else value * relative_step
            plus = {**parameters, instance: {**inner, name: value + step}}
            minus = {**parameters, instance: {**inner, name: value - step}}
            gradients[instance][name] = (
                float(loss_function(plus)) - float(loss_function(minus))
            ) / (2.0 * step)
    return gradients


def _assert_matches_central_difference(
    loss_function: Callable[[dict[str, dict[str, jax.Array]]], jax.Array],
    gradient: Mapping[str, Mapping[str, jax.Array]],
    parameters: dict[str, dict[str, float]],
    *,
    relative_tolerance: float,
) -> None:
    """Asserts gradient agreement with central differences within tolerance."""
    finite_difference = _central_difference(
        lambda current: float(
            loss_function(_as_jax_parameters(current, dtype=jnp.float64))
        ),
        parameters,
    )
    largest = max(
        abs(float(gradient[instance][name]))
        for instance, inner in parameters.items()
        for name in inner
    )
    for instance, inner in parameters.items():
        for name in inner:
            assert (
                abs(float(gradient[instance][name]) - finite_difference[instance][name])
                <= relative_tolerance * largest
            )


def test_operating_point_matches_jolty_vjp_bit_for_bit_and_finite_differences() -> None:
    analysis = OperatingPoint()
    wrapped = jax_function(crossbar(), analysis)
    target = jolty_function(crossbar(), analysis)

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
        # XLA JIT fusion vs eager f64 sum rounding (~4.5 eps).
        np.testing.assert_allclose(
            float(value), float(jitted_value), rtol=1e-15, atol=0.0
        )

        target_outputs, pullback = jolty_vjp(target, WEIGHTS)
        for name in ("out0", "out1"):
            np.testing.assert_array_equal(
                np.asarray(eager_outputs[name]), np.asarray(target_outputs[name])
            )
        cotangent = {
            name: 2.0 * (np.asarray(target_outputs[name]) - target_value)
            for name, target_value in targets.items()
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

        _assert_matches_central_difference(
            loss, eager_gradient, WEIGHTS, relative_tolerance=CENTRAL_DIFF_RTOL
        )


def test_dc_sweep_matches_jolty_vjp_bit_for_bit() -> None:
    analysis = DcSweep("V0", [0.1 * V, 0.2 * V, 0.4 * V])
    wrapped = jax_function(crossbar(), analysis)
    target = jolty_function(crossbar(), analysis)

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
        _, pullback = jolty_vjp(target, WEIGHTS)
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
    target = jolty_function(crossbar(), analysis)

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

        target_outputs, pullback = jolty_vjp(target, WEIGHTS)
        cotangent_asymmetric = np.full(3, 1.0 - 2.0j, dtype=np.complex128)
        expected_asymmetric = pullback({"out0": cotangent_asymmetric})
        cotangent_magnitude = 2.0 * np.conj(np.asarray(target_outputs["out0"]))
        expected_magnitude = pullback({"out0": cotangent_magnitude})

        for instance in WEIGHTS:
            np.testing.assert_array_equal(
                float(gradient_asymmetric[instance]["resistance"]),
                float(np.real(expected_asymmetric[instance]["resistance"])),
            )
            np.testing.assert_array_equal(
                float(gradient_magnitude[instance]["resistance"]),
                float(np.real(expected_magnitude[instance]["resistance"])),
            )
        _assert_matches_central_difference(
            loss_asymmetric,
            gradient_asymmetric,
            WEIGHTS,
            relative_tolerance=CENTRAL_DIFF_RTOL,
        )
        _assert_matches_central_difference(
            loss_magnitude_squared,
            gradient_magnitude,
            WEIGHTS,
            relative_tolerance=CENTRAL_DIFF_RTOL,
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
    target = jolty_function(crossbar(), analysis)

    with jax.enable_x64():
        breakpoint_weights = jnp.array([1.0, -0.5, 2.0, 0.75], dtype=jnp.float64)

        def loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            outputs = wrapped(parameters)
            return jnp.sum(breakpoint_weights * outputs["out0"])

        jax_weights = _as_jax_parameters(WEIGHTS, dtype=jnp.float64)
        gradient = jax.jit(jax.grad(loss))(jax_weights)

        weight_overrides = {
            instance: {
                parameter: value * ohm for parameter, value in parameters.items()
            }
            for instance, parameters in WEIGHTS.items()
        }
        transient_result = Simulation(crossbar(), overrides=weight_overrides).run(
            analysis
        )
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

        _, pullback = jolty_vjp(target, WEIGHTS)
        expected = pullback({"out0": cotangent_series})
        for instance in WEIGHTS:
            expected_value = _expect_real_scalar(expected[instance]["resistance"])
            np.testing.assert_array_equal(
                float(gradient[instance]["resistance"]), expected_value
            )

        _assert_matches_central_difference(
            loss, gradient, WEIGHTS, relative_tolerance=TRANSIENT_CENTRAL_DIFF_RTOL
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
        _assert_matches_central_difference(
            transient_loss,
            gradient,
            WEIGHTS,
            relative_tolerance=TRANSIENT_CENTRAL_DIFF_RTOL,
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
                # Central difference (step=1e-4) over PSS shooting (reltol=1e-3).
                assert abs(
                    float(gradient[instance][name]) - finite_difference[instance][name]
                ) <= 1e-4 * abs(finite_difference[instance][name])


def test_autonomous_periodic_steady_state_period_gradient_and_scaling_law() -> None:
    """Verifies the scaling law L * dT/dL + C * dT/dC = T on the van der Pol tank."""
    options = Options(relative_tolerance=1e-6)
    analysis = PeriodicSteadyState()
    wrapped = jax_function(van_der_pol_tank(), analysis, options)
    inductance, capacitance = 1e-6, 1e-9
    parameters = {"L1": {"inductance": inductance}, "C1": {"capacitance": capacitance}}

    with jax.enable_x64():
        jax_parameters = _as_jax_parameters(parameters, dtype=jnp.float64)
        outputs = wrapped(jax_parameters)
        assert outputs[PERIOD].shape == ()
        waveforms = {name: array for name, array in outputs.items() if name != PERIOD}
        assert waveforms
        assert all(array.shape == (0,) for array in waveforms.values())

        def period_function(current: dict[str, dict[str, jax.Array]]) -> jax.Array:
            return wrapped(current)[PERIOD]

        period, gradient = jax.jit(jax.value_and_grad(period_function))(jax_parameters)
        period_value = float(period)
        assert period.shape == ()
        euler_sum = inductance * float(
            gradient["L1"]["inductance"]
        ) + capacitance * float(gradient["C1"]["capacitance"])
        # Euler homogeneity identity for period at reltol=1e-6 (cf. test_periodic.py).
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
        original_batch = Function.batch

        def spy_batch(
            target: Function,
            names: Sequence[tuple[str, str]],
            values: npt.ArrayLike,
        ) -> Outputs:
            call_shapes.append(np.asarray(values).shape)
            return original_batch(target, names, values)

        with patch.object(Function, "batch", spy_batch):
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
        # Shared-parameter batch sum vs per-example sum: f64 accumulation (~45 eps).
        np.testing.assert_allclose(
            float(gradient_over_batch["W10"]["resistance"]),
            float(jnp.sum(per_example_gradients["W10"]["resistance"])),
            rtol=1e-14,
            atol=0.0,
        )


def test_gradient_keeps_its_forward_pullbacks_and_recomputes_once_dropped() -> None:
    calls: list[int] = []
    original_batch_vjp = Function.batch_vjp

    def spy_batch_vjp(
        target: Function,
        names: Sequence[tuple[str, str]],
        values: npt.ArrayLike,
    ) -> tuple[Outputs, BatchPullback]:
        calls.append(len(np.asarray(values)))
        return original_batch_vjp(target, names, values)

    with jax.enable_x64():
        parameters = _as_jax_parameters(WEIGHTS, dtype=jnp.float64)
        gradients = {}
        for saved in (16, 0):
            wrapped = jax_function(crossbar(), OperatingPoint(), saved_pullbacks=saved)

            def loss(
                current: dict[str, dict[str, jax.Array]],
                wrapped: Callable[..., dict[str, jax.Array]] = wrapped,
            ) -> jax.Array:
                return jnp.sum(wrapped(current)["out0"] ** 2)

            calls.clear()
            with patch.object(Function, "batch_vjp", spy_batch_vjp):
                gradients[saved] = jax.grad(loss)(parameters)
            assert calls == ([1] if saved else [1, 1])
        for instance in WEIGHTS:
            np.testing.assert_array_equal(
                np.asarray(gradients[16][instance]["resistance"]),
                np.asarray(gradients[0][instance]["resistance"]),
            )


def test_vmap_forward_and_gradient_across_all_analyses() -> None:
    def check_vmap_grad(
        wrapped: Callable[[dict[str, dict[str, jax.Array]]], dict[str, jax.Array]],
        instance: str,
        values: jax.Array,
        reduce_output: Callable[[dict[str, jax.Array]], jax.Array],
    ) -> None:
        batched = {instance: {"resistance": values}}
        vmapped_gradient = jax.jit(
            jax.grad(lambda current: reduce_output(jax.vmap(wrapped)(current)))
        )(batched)
        for index in range(len(values)):
            single_gradient = jax.grad(lambda current: reduce_output(wrapped(current)))(
                {instance: {"resistance": values[index]}}
            )
            np.testing.assert_array_equal(
                float(vmapped_gradient[instance]["resistance"][index]),
                float(single_gradient[instance]["resistance"]),
            )

    with jax.enable_x64():
        resistances = jnp.array([900.0, 1500.0, 2200.0], dtype=jnp.float64)
        check_vmap_grad(
            jax_function(crossbar(), DcSweep("V0", [0.1 * V, 0.3 * V, 0.5 * V])),
            "W00",
            resistances,
            lambda outputs: jnp.sum(outputs["out0"] ** 2),
        )
        check_vmap_grad(
            jax_function(crossbar(), Ac("V1", [1 * kHz, 100 * kHz, 1 * MHz])),
            "W00",
            resistances,
            lambda outputs: jnp.sum(outputs["out0"].real + 2.0 * outputs["out0"].imag),
        )
        check_vmap_grad(
            jax_function(
                crossbar(),
                Transient(1 * us, max_step=50 * ns, breakpoints=(0.5 * us, 1.0 * us)),
            ),
            "W00",
            resistances,
            lambda outputs: jnp.sum(outputs["out0"] ** 2),
        )
        check_vmap_grad(
            jax_function(
                driven_rc_filter(),
                PeriodicSteadyState(
                    period=1 * us,
                    max_step=50 * ns,
                    breakpoints=(0.25 * us, 0.75 * us),
                ),
            ),
            "R1",
            jnp.array([800.0, 1400.0], dtype=jnp.float64),
            lambda outputs: jnp.sum(outputs["out"] ** 2),
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
        cases = [
            (
                jax_function(crossbar(), OperatingPoint()),
                "W00",
                "resistance",
                "out0",
                (1,),
                jnp.float64,
                jnp.sum,
            ),
            (
                jax_function(crossbar(), Ac("V1", [1 * kHz, 1 * MHz])),
                "W00",
                "resistance",
                "out0",
                (2,),
                jnp.complex128,
                lambda array: jnp.sum(array.real),
            ),
            (
                jax_function(van_der_pol_tank(), PeriodicSteadyState()),
                "L1",
                "inductance",
                PERIOD,
                (),
                jnp.float64,
                jnp.sum,
            ),
        ]
        for shape in ((0,), (0, 3)):
            empty = jnp.empty(shape, dtype=jnp.float64)
            for wrapped, instance, parameter, key, trailing, dtype, reduce_fn in cases:
                mapper = (
                    jax.vmap(wrapped)
                    if len(shape) == 1
                    else jax.vmap(jax.vmap(wrapped))
                )
                inputs = {instance: {parameter: empty}}
                output = mapper(inputs)[key]
                assert output.shape == (*shape, *trailing)
                assert output.dtype == dtype
                gradient = jax.grad(
                    lambda current, m=mapper, k=key, r=reduce_fn: r(m(current)[k])
                )(inputs)
                assert gradient[instance][parameter].shape == shape


def test_nested_vmap_repeated_vjps_and_a_zero_cotangent() -> None:
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
            # Exact VJP linearity under 3x cotangent scaling; f64 rounding (~45 eps).
            np.testing.assert_allclose(
                float(gradient_second[instance]["resistance"]),
                3.0 * float(gradient_first[instance]["resistance"]),
                rtol=1e-14,
            )

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
        # Rematerialized forward value vs plain JIT forward: f64 rounding (~4.5 eps).
        np.testing.assert_allclose(float(remat_value), float(plain_value), rtol=1e-15)
        for instance in WEIGHTS:
            np.testing.assert_array_equal(
                float(remat_gradient[instance]["resistance"]),
                float(plain_gradient[instance]["resistance"]),
            )

        # jacrev across OperatingPoint, DcSweep, Ac, Transient, and driven periodic.
        driven_inputs = {"R1": {"resistance": jnp.asarray(1.2e3, dtype=jnp.float64)}}
        jacrev_cases = [
            (operating_point_wrapped, "out0", jax_weights, "W00", (1,)),
            (
                jax_function(crossbar(), DcSweep("V0", [0.1 * V, 0.3 * V])),
                "out0",
                jax_weights,
                "W00",
                (2,),
            ),
            (
                jax_function(crossbar(), Ac("V1", [1 * kHz, 1 * MHz])),
                "out0",
                jax_weights,
                "W00",
                (2,),
            ),
            (
                jax_function(
                    crossbar(),
                    Transient(
                        1 * us, max_step=50 * ns, breakpoints=(0.5 * us, 1.0 * us)
                    ),
                ),
                "out0",
                jax_weights,
                "W00",
                (2,),
            ),
            (
                jax_function(
                    driven_rc_filter(),
                    PeriodicSteadyState(
                        period=1 * us,
                        max_step=50 * ns,
                        breakpoints=(0.25 * us, 0.75 * us),
                    ),
                ),
                "out",
                driven_inputs,
                "R1",
                (2,),
            ),
        ]
        for wrapped, key, target_inputs, instance, expected_shape in jacrev_cases:
            jacobian = jax.jacrev(
                lambda current, f=wrapped, k=key: jnp.real(f(current)[k])
            )(target_inputs)
            assert jacobian[instance]["resistance"].shape == expected_shape


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
        # Central difference at step=1e-4 across sub-ohm MNA branch current switch.
        _assert_matches_central_difference(
            added_loss, added_gradient, low_resistance, relative_tolerance=1e-5
        )

        # Direction 2: dropping an unknown (0.5 ohm -> 1000 ohm drops `W00:current`).
        wrapped_dropped = jax_function(crossbar(0.5 * ohm), OperatingPoint())
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

        # The batch's own pullback raises the unwrapped SimulationError directly.
        target = jolty_function(crossbar(0.5 * ohm), OperatingPoint())
        names = [(instance, "resistance") for instance in sorted(WEIGHTS)]
        row = [[WEIGHTS[instance]["resistance"] for instance in sorted(WEIGHTS)]]
        _, batch_pullback = target.batch_vjp(names, row)
        with pytest.raises(SimulationError, match="W00:current"):
            batch_pullback({"W00:current": [[1.0]]})

        # Complex AC dropped unknown places NaN + NaNj and rejects non-zero / NaN.
        wrapped_dropped_ac = jax_function(
            crossbar(0.5 * ohm), Ac("V1", [1 * kHz, 1 * MHz])
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
        # Chain-rule sum for tied vs separate gradients: f64 rounding (~45 eps).
        np.testing.assert_allclose(
            gradient_tied,
            float(gradient_separate["W00"]["resistance"])
            + float(gradient_separate["W10"]["resistance"]),
            rtol=1e-14,
        )

        # Repeated eager calls hit the runner cache without re-running validation.
        wrapped({"W00": {"resistance": jnp.asarray(1200.0, dtype=jnp.float64)}})
        with patch.object(Elaborated, "check", side_effect=AssertionError):
            cached_output = wrapped(
                {"W00": {"resistance": jnp.asarray(1600.0, dtype=jnp.float64)}}
            )
            assert bool(jnp.isfinite(cached_output["out0"][0]))


def test_construction_time_analysis_refusals() -> None:
    with pytest.raises(ValueError, match="breakpoints"):
        jax_function(crossbar(), Transient(stop=2 * us))
    with pytest.raises(ValueError, match="breakpoints"):
        jax_function(driven_rc_filter(), PeriodicSteadyState(period=1 * us))

    for unsupported in (
        OperatingPoints(),
        PoleZero(source="V0", output="out0"),
        TransferFunction(source="V0", output="out0"),
    ):
        with pytest.raises(TypeError, match="a function runs"):
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

    bad_inputs: list[tuple[object, type[Exception], str]] = [
        ([1000.0], TypeError, "a mapping, not list"),
        (None, TypeError, "a mapping, not NoneType"),
        ({1: {"resistance": 1000.0}}, TypeError, "each instance's path"),
        ({"W00": 1000.0}, TypeError, "each instance's path"),
        ({"W00": {1: 1000.0}}, TypeError, "is a real number"),
        ({"W00": {"resistance": True}}, TypeError, "is a real number"),
        (
            {"W00": {"resistance": jnp.asarray(1, dtype=jnp.bool_)}},
            TypeError,
            "is a real number",
        ),
        ({"W00": {"resistance": 1000.0 + 1.0j}}, TypeError, "is a real number"),
        (
            {"W00": {"resistance": jnp.array([1000.0, 2000.0])}},
            ValueError,
            "is a scalar",
        ),
    ]
    for bad_input, error_type, pattern in bad_inputs:
        with pytest.raises(error_type, match=pattern):
            wrapped(bad_input)  # ty: ignore[invalid-argument-type] - invalid parameter structure is tested
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


def test_concurrent_first_call_builds_runner_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped = jax_function(crossbar(), OperatingPoint())
    workers = 4
    barrier = threading.Barrier(workers)
    original_check = Elaborated.check

    def synced_check(self: Elaborated, parameters: Mapping[str, Iterable[str]]) -> None:
        original_check(self, parameters)
        barrier.wait(timeout=10)

    def evaluate_once() -> JaxOutputs:
        with jax.enable_x64():
            return wrapped(_as_jax_parameters(WEIGHTS, dtype=jnp.float64))

    monkeypatch.setattr(Elaborated, "check", synced_check)
    original_runner = jolty_jax_module._runner  # noqa: SLF001 - count builds
    with (
        patch.object(jolty_jax_module, "_runner", wraps=original_runner) as runner_spy,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = [pool.submit(evaluate_once) for _ in range(workers)]
        results_list = [future.result() for future in futures]
    assert runner_spy.call_count == 1
    first = np.asarray(results_list[0]["out0"])
    for output in results_list[1:]:
        np.testing.assert_array_equal(np.asarray(output["out0"]), first)


THREE_STAGE_POLES: Final = (
    (90 * kohm, 1e3),
    (10 * kohm, 120e3),
    (1 * kohm, 7e6),
)
"""Three-stage amplifier loads and pole frequencies placing crossings off grid nodes."""


@component
def three_stage_feedback(c: Builder) -> None:
    """Builds a three-stage unity-feedback amplifier broken by floating probe ``VP``."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), name="VIN")
    c.connect(source.negative, ground)
    stages = [
        c.place(devices.voltage_controlled_current_source(1 * mA / V), name=f"G{index}")
        for index in range(1, len(THREE_STAGE_POLES) + 1)
    ]
    driving = source.positive
    for index, (stage, (load, pole)) in enumerate(
        zip(stages, THREE_STAGE_POLES, strict=True), 1
    ):
        resistor = c.place(devices.resistor(load), name=f"R{index}")
        capacitance = 1 / (2 * math.pi * pole * load.value) * F
        capacitor = c.place(devices.capacitor(capacitance), name=f"C{index}")
        c.connect(stage.control_positive, driving)
        c.connect(stage.positive, resistor.b, capacitor.b, ground)
        c.connect(stage.negative, resistor.a, capacitor.a, name=f"n{index}")
        driving = stage.negative
    for stage in stages[1:]:
        c.connect(stage.control_negative, ground)
    buffer = c.place(devices.voltage_controlled_voltage_source(1.0), name="E1")
    c.connect(buffer.control_positive, driving)
    c.connect(buffer.control_negative, buffer.negative, ground)
    probe = c.place(devices.voltage_source(0 * V), name="VP")
    c.connect(buffer.positive, probe.negative, name="out")
    c.connect(probe.positive, stages[0].control_negative, name="fb")


@component
def mosfet_feedback_amplifier(c: Builder) -> None:
    """Builds a nonlinear MOSFET shunt-shunt feedback amplifier with probe ``VP``."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(3.3 * V), name="VDD")
    drain_load = c.place(devices.resistor(5 * kohm), name="RD")
    transistor = c.place(
        devices.mosfet(
            width=20 * um,
            length=1 * um,
            threshold_voltage=0.7 * V,
            transconductance_parameter=100 * uA / V**2,
            channel_length_modulation=0.02 / V,
        ),
        name="M1",
    )
    probe = c.place(devices.voltage_source(0 * V), name="VP")
    feedback = c.place(devices.resistor(20 * kohm), name="RF")
    bias = c.place(devices.resistor(20 * kohm), name="RG")
    load_cap = c.place(devices.capacitor(2 * pF), name="CL")
    feedback_cap = c.place(devices.capacitor(0.5 * pF), name="CF")
    nonlinear_cap = c.place(
        devices.behavioral_capacitor("1e-12 * (v(x) + 0.3 * (v(x) ^ 2))"),
        name="BC1",
    )
    c.connect(supply.positive, drain_load.a, name="vdd")
    c.connect(
        drain_load.b,
        transistor.drain,
        probe.negative,
        load_cap.a,
        nonlinear_cap.positive,
        nonlinear_cap.x,
        name="drain",
    )
    c.connect(probe.positive, feedback.a, feedback_cap.a, name="fb")
    c.connect(feedback.b, feedback_cap.b, bias.a, transistor.gate, name="gate")
    c.connect(
        supply.negative,
        transistor.source,
        transistor.bulk,
        bias.b,
        load_cap.b,
        nonlinear_cap.negative,
        ground,
    )


def _unwrapped_phase_degrees(gains: jax.Array) -> jax.Array:
    """Computes smooth unwrapped phase in degrees along a monotonic frequency grid."""
    steps = jnp.angle(gains[1:] / gains[:-1])
    return jnp.rad2deg(jnp.cumsum(jnp.concatenate([jnp.angle(gains[:1]), steps])))


def _differentiable_phase_margin(gains: jax.Array) -> jax.Array:
    """Linearly interpolates phase margin in degrees at the 0 dB crossing."""
    log_magnitude = jnp.log10(jnp.abs(gains))
    phase_degrees = _unwrapped_phase_degrees(gains)
    crossing = jax.lax.stop_gradient(
        jnp.argmax((log_magnitude[:-1] >= 0.0) & (log_magnitude[1:] < 0.0))
    )
    fraction = log_magnitude[crossing] / (
        log_magnitude[crossing] - log_magnitude[crossing + 1]
    )
    return (
        180.0
        + phase_degrees[crossing]
        + fraction * (phase_degrees[crossing + 1] - phase_degrees[crossing])
    )


def _differentiable_gain_margin(gains: jax.Array) -> jax.Array:
    """Linearly interpolates gain margin in dB at the -180 degree phase crossing."""
    log_magnitude = jnp.log10(jnp.abs(gains))
    phase_degrees = _unwrapped_phase_degrees(gains)
    crossing = jax.lax.stop_gradient(
        jnp.argmax((phase_degrees[:-1] >= -180.0) & (phase_degrees[1:] < -180.0))
    )
    fraction = (phase_degrees[crossing] + 180.0) / (
        phase_degrees[crossing] - phase_degrees[crossing + 1]
    )
    return -20.0 * (
        log_magnitude[crossing]
        + fraction * (log_magnitude[crossing + 1] - log_magnitude[crossing])
    )


def test_stability_jax_grad_of_loop_gain_phase_margin_and_gain_margin() -> None:
    frequencies = decade(10 * kHz, 10 * MHz, 15)
    analysis = Stability("VP", frequencies)
    wrapped = jax_function(three_stage_feedback(), analysis)
    target = jolty_function(three_stage_feedback(), analysis)

    load_first, pole_first = THREE_STAGE_POLES[0]
    load_second, pole_second = THREE_STAGE_POLES[1]
    point: dict[str, dict[str, float]] = {
        "R1": {"resistance": load_first.value},
        "C1": {"capacitance": 1 / (2 * math.pi * pole_first * load_first.value)},
        "C2": {"capacitance": 1 / (2 * math.pi * pole_second * load_second.value)},
    }

    with jax.enable_x64():
        jax_point = _as_jax_parameters(point, dtype=jnp.float64)

        def asymmetric_loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            gains = wrapped(parameters)[LOOP_GAIN]
            return jnp.sum(gains.real + 2.0 * gains.imag)

        def phase_margin_objective(
            parameters: dict[str, dict[str, jax.Array]],
        ) -> jax.Array:
            return _differentiable_phase_margin(wrapped(parameters)[LOOP_GAIN])

        def gain_margin_objective(
            parameters: dict[str, dict[str, jax.Array]],
        ) -> jax.Array:
            return _differentiable_gain_margin(wrapped(parameters)[LOOP_GAIN])

        # Exact Wirtinger pullback matches jolty.vjp bit for bit.
        asymmetric_gradient = jax.jit(jax.grad(asymmetric_loss))(jax_point)
        _, pullback = jolty_vjp(target, point)
        cotangent = np.full(len(frequencies), 1.0 - 2.0j, dtype=np.complex128)
        expected_asymmetric = pullback({LOOP_GAIN: cotangent})
        for instance, inner in point.items():
            for name in inner:
                np.testing.assert_array_equal(
                    float(asymmetric_gradient[instance][name]),
                    float(np.real(expected_asymmetric[instance][name])),
                )

        # Phase margin and gain margin gradients match central finite differences.
        pm_value, pm_gradient = jax.jit(jax.value_and_grad(phase_margin_objective))(
            jax_point
        )
        gm_value, gm_gradient = jax.jit(jax.value_and_grad(gain_margin_objective))(
            jax_point
        )
        assert 10.0 < float(pm_value) < 35.0
        assert 10.0 < float(gm_value) < 30.0
        # Increasing dominant-pole compensation capacitor C1 improves phase margin.
        assert float(pm_gradient["C1"]["capacitance"]) > 0.0

        pm_finite = _central_difference(
            lambda moved: float(
                phase_margin_objective(_as_jax_parameters(moved, dtype=jnp.float64))
            ),
            point,
        )
        gm_finite = _central_difference(
            lambda moved: float(
                gain_margin_objective(_as_jax_parameters(moved, dtype=jnp.float64))
            ),
            point,
        )
        for instance, inner in point.items():
            for name in inner:
                np.testing.assert_allclose(
                    float(pm_gradient[instance][name]),
                    pm_finite[instance][name],
                    rtol=CENTRAL_DIFF_RTOL,
                )
                np.testing.assert_allclose(
                    float(gm_gradient[instance][name]),
                    gm_finite[instance][name],
                    rtol=CENTRAL_DIFF_RTOL,
                )


def test_stability_jax_vmap_jacrev_nonlinear_bias_and_probe_errors() -> None:
    frequencies = [1 * kHz, 1 * MHz, 100 * MHz]
    analysis = Stability("VP", frequencies)
    wrapped = jax_function(mosfet_feedback_amplifier(), analysis)

    with jax.enable_x64():
        # Nonlinear bias-coupled gradients (including VP.waveform.dc) vs finite diffs.
        nonlinear_point: dict[str, dict[str, float]] = {
            "VDD": {"waveform.dc": 3.3},
            "VP": {"waveform.dc": 0.0},
            "M1": {"width": 20e-6},
            "RF": {"resistance": 20e3},
        }

        def magnitude_loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            gains = wrapped(parameters)[LOOP_GAIN]
            return jnp.sum(gains.real**2 + gains.imag**2)

        gradient = jax.jit(jax.grad(magnitude_loss))(
            _as_jax_parameters(nonlinear_point, dtype=jnp.float64)
        )
        assert abs(float(gradient["VP"]["waveform.dc"])) > 0.1
        finite = _central_difference(
            lambda moved: float(
                magnitude_loss(_as_jax_parameters(moved, dtype=jnp.float64))
            ),
            nonlinear_point,
        )
        for instance, inner in nonlinear_point.items():
            for name in inner:
                np.testing.assert_allclose(
                    float(gradient[instance][name]),
                    finite[instance][name],
                    rtol=CENTRAL_DIFF_RTOL,
                )

        # vmap forward and grad match scalar calls bit for bit.
        widths = jnp.array([15e-6, 20e-6, 30e-6], dtype=jnp.float64)
        batched = {"M1": {"width": widths}}
        vmapped_outputs = jax.jit(jax.vmap(wrapped))(batched)[LOOP_GAIN]
        assert vmapped_outputs.shape == (3, len(frequencies))
        assert vmapped_outputs.dtype == jnp.complex128
        vmapped_gradient = jax.jit(
            jax.grad(
                lambda current: jnp.sum(
                    jnp.abs(jax.vmap(wrapped)(current)[LOOP_GAIN]) ** 2
                )
            )
        )(batched)
        for index in range(len(widths)):
            single_input = {"M1": {"width": widths[index]}}
            np.testing.assert_array_equal(
                np.asarray(vmapped_outputs[index]),
                np.asarray(wrapped(single_input)[LOOP_GAIN]),
            )
            single_gradient = jax.grad(
                lambda current: jnp.sum(jnp.abs(wrapped(current)[LOOP_GAIN]) ** 2)
            )(single_input)
            np.testing.assert_array_equal(
                float(vmapped_gradient["M1"]["width"][index]),
                float(single_gradient["M1"]["width"]),
            )

        # Empty batch preserves complex128 output shape and gradient shape.
        for shape in ((0,), (0, 2)):
            empty = jnp.empty(shape, dtype=jnp.float64)
            mapper = (
                jax.vmap(wrapped) if len(shape) == 1 else jax.vmap(jax.vmap(wrapped))
            )
            empty_inputs = {"M1": {"width": empty}}
            empty_out = mapper(empty_inputs)[LOOP_GAIN]
            assert empty_out.shape == (*shape, len(frequencies))
            assert empty_out.dtype == jnp.complex128
            empty_grad = jax.grad(
                lambda current, m=mapper: jnp.sum(m(current)[LOOP_GAIN].real)
            )(empty_inputs)
            assert empty_grad["M1"]["width"].shape == shape

        # jacrev differentiates real and imaginary parts of loop gain.
        jacobian = jax.jacrev(
            lambda current: jnp.concatenate(
                [wrapped(current)[LOOP_GAIN].real, wrapped(current)[LOOP_GAIN].imag]
            )
        )({"M1": {"width": jnp.asarray(20e-6, dtype=jnp.float64)}})
        assert jacobian["M1"]["width"].shape == (2 * len(frequencies),)

        # Invalid probe raises SimulationError through JAX callback.
        bad_probe = jax_function(
            mosfet_feedback_amplifier(), Stability("RF", [1 * kHz])
        )
        with pytest.raises(
            (SimulationError, jax.errors.JaxRuntimeError),
            match="not a floating voltage source",
        ):
            bad_probe({"M1": {"width": jnp.asarray(20e-6, dtype=jnp.float64)}})


@component
def noisy_common_source_amplifier(c: Builder) -> None:
    """Builds a common-source stage with thermal, flicker, and post-pole noise."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(3.3 * V), name="VDD")
    gate_bias = c.place(devices.voltage_source(1.2 * V), name="VG")
    gate_resistor = c.place(devices.resistor(10 * kohm), name="RG")
    drain_resistor = c.place(devices.resistor(10 * kohm), name="RD")
    load_capacitor = c.place(devices.capacitor(2 * pF), name="CL")
    output_resistor = c.place(devices.resistor(2 * kohm), name="RO")
    load_resistor = c.place(devices.resistor(50 * kohm), name="RL")
    transistor = c.place(
        devices.mosfet(
            width=2 * um,
            length=1 * um,
            threshold_voltage=0.7 * V,
            transconductance_parameter=100 * uA / V**2,
            channel_length_modulation=0.02 / V,
            flicker_coefficient=1e-33,
            flicker_exponent=1.2,
        ),
        name="M1",
    )
    c.connect(supply.positive, drain_resistor.a, name="vdd")
    c.connect(gate_bias.positive, gate_resistor.a, name="source")
    c.connect(gate_resistor.b, transistor.gate, name="gate")
    c.connect(
        drain_resistor.b,
        transistor.drain,
        load_capacitor.a,
        output_resistor.a,
        name="drain",
    )
    c.connect(output_resistor.b, load_resistor.a, name="out")
    c.connect(
        supply.negative,
        gate_bias.negative,
        transistor.source,
        transistor.bulk,
        load_capacitor.b,
        load_resistor.b,
        ground,
    )


def test_noise_jax_grad_matches_vjp_and_finite_differences() -> None:
    frequencies = [10 * Hz, 10 * kHz, 10 * MHz]
    analysis = Noise("out", frequencies, source="VG")
    wrapped = jax_function(noisy_common_source_amplifier(), analysis)
    target = jolty_function(noisy_common_source_amplifier(), analysis)
    output_only = jax_function(
        noisy_common_source_amplifier(), Noise("out", frequencies)
    )

    point: dict[str, dict[str, float]] = {
        "VG": {"waveform.dc": 1.2},
        "RG": {"resistance": 10e3},
        "RD": {"resistance": 10e3},
        "CL": {"capacitance": 2e-12},
        "M1": {
            "width": 2e-6,
            "threshold_voltage": 0.7,
            "flicker_coefficient": 1e-33,
            "flicker_exponent": 1.2,
        },
    }

    with jax.enable_x64():
        jax_point = _as_jax_parameters(point, dtype=jnp.float64)

        # Output-only noise exposes only NOISE_DENSITY.
        solo = output_only(jax_point)
        assert list(solo) == [NOISE_DENSITY]
        assert solo[NOISE_DENSITY].shape == (len(frequencies),)
        assert solo[NOISE_DENSITY].dtype == jnp.float64

        # Exact linear loss matches jolty.vjp bit for bit across both noise outputs.
        weights_density = jnp.array([1e14, 1e16, 1e17], dtype=jnp.float64)
        weights_referred = jnp.array([2e14, -1e16, 5e16], dtype=jnp.float64)

        def combined_loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            spectrum = wrapped(parameters)
            return jnp.sum(weights_density * spectrum[NOISE_DENSITY]) + jnp.sum(
                weights_referred * spectrum[REFERRED_NOISE]
            )

        jax_gradient = jax.jit(jax.grad(combined_loss))(jax_point)
        _, pullback = jolty_vjp(target, point)
        expected_vjp = pullback(
            {
                NOISE_DENSITY: np.asarray(weights_density),
                REFERRED_NOISE: np.asarray(weights_referred),
            }
        )
        for instance, inner in point.items():
            for name in inner:
                np.testing.assert_array_equal(
                    float(jax_gradient[instance][name]),
                    float(np.real(expected_vjp[instance][name])),
                )

        # Log-density objectives for both output and input-referred noise match
        # central finite differences across bias, passive, and flicker parameters.
        for key in (NOISE_DENSITY, REFERRED_NOISE):

            def log_noise_objective(
                parameters: dict[str, dict[str, jax.Array]],
                output_key: str = key,
            ) -> jax.Array:
                return jnp.sum(jnp.log(wrapped(parameters)[output_key]))

            value, gradient = jax.jit(jax.value_and_grad(log_noise_objective))(
                jax_point
            )
            assert np.isfinite(float(value))
            finite = _central_difference(
                lambda moved, fn=log_noise_objective: float(
                    fn(_as_jax_parameters(moved, dtype=jnp.float64))
                ),
                point,
            )
            for instance, inner in point.items():
                for name in inner:
                    np.testing.assert_allclose(
                        float(gradient[instance][name]),
                        finite[instance][name],
                        rtol=CENTRAL_DIFF_RTOL,
                    )


def test_noise_jax_vmap_jacrev_and_errors() -> None:
    frequencies = [10 * Hz, 10 * kHz, 10 * MHz]
    analysis = Noise("out", frequencies, source="VG")
    wrapped = jax_function(noisy_common_source_amplifier(), analysis)

    with jax.enable_x64():
        # vmap forward and grad match scalar calls bit for bit on both outputs.
        widths = jnp.array([1.5e-6, 2.0e-6, 3.0e-6], dtype=jnp.float64)
        batched = {"M1": {"width": widths}}
        vmapped = jax.jit(jax.vmap(wrapped))(batched)
        for key in (NOISE_DENSITY, REFERRED_NOISE):
            assert vmapped[key].shape == (3, len(frequencies))
            assert vmapped[key].dtype == jnp.float64
        vmapped_gradient = jax.jit(
            jax.grad(
                lambda current: jnp.sum(
                    jnp.log(jax.vmap(wrapped)(current)[NOISE_DENSITY])
                    + jnp.log(jax.vmap(wrapped)(current)[REFERRED_NOISE])
                )
            )
        )(batched)
        for index in range(len(widths)):
            single_input = {"M1": {"width": widths[index]}}
            single_output = wrapped(single_input)
            for key in (NOISE_DENSITY, REFERRED_NOISE):
                np.testing.assert_array_equal(
                    np.asarray(vmapped[key][index]),
                    np.asarray(single_output[key]),
                )
            single_gradient = jax.grad(
                lambda current: jnp.sum(
                    jnp.log(wrapped(current)[NOISE_DENSITY])
                    + jnp.log(wrapped(current)[REFERRED_NOISE])
                )
            )(single_input)
            np.testing.assert_array_equal(
                float(vmapped_gradient["M1"]["width"][index]),
                float(single_gradient["M1"]["width"]),
            )

        # Empty batch preserves float64 output shapes and gradient shapes.
        for shape in ((0,), (0, 2)):
            empty = jnp.empty(shape, dtype=jnp.float64)
            mapper = (
                jax.vmap(wrapped) if len(shape) == 1 else jax.vmap(jax.vmap(wrapped))
            )
            empty_inputs = {"M1": {"width": empty}}
            empty_out = mapper(empty_inputs)
            for key in (NOISE_DENSITY, REFERRED_NOISE):
                assert empty_out[key].shape == (*shape, len(frequencies))
                assert empty_out[key].dtype == jnp.float64
            empty_grad = jax.grad(
                lambda current, m=mapper: (
                    jnp.sum(m(current)[NOISE_DENSITY])
                    + jnp.sum(m(current)[REFERRED_NOISE])
                )
            )(empty_inputs)
            assert empty_grad["M1"]["width"].shape == shape

        # jacrev differentiates both output density and referred noise together.
        jacobian = jax.jacrev(
            lambda current: jnp.concatenate(
                [wrapped(current)[NOISE_DENSITY], wrapped(current)[REFERRED_NOISE]]
            )
        )({"M1": {"width": jnp.asarray(2e-6, dtype=jnp.float64)}})
        assert jacobian["M1"]["width"].shape == (2 * len(frequencies),)

        # Invalid input source raises SimulationError through JAX callback.
        bad_source = jax_function(
            noisy_common_source_amplifier(),
            Noise("out", [1 * kHz], source="RD"),
        )
        with pytest.raises(
            (SimulationError, jax.errors.JaxRuntimeError),
            match="not an independent source",
        ):
            bad_source({"M1": {"width": jnp.asarray(2e-6, dtype=jnp.float64)}})

# SPICE and PDK subcircuit differentiation through JAX.

SUBCIRCUIT_CENTRAL_DIFF_RTOL: Final = 1e-5
"""Relative tolerance for `step=1e-4` nonlinear BSIM4 central differences."""


def _synthetic_pdk_stage() -> tuple[Component, dict[str, SubcircuitPlacement]]:
    """Builds a two-bin BSIM4 and geometric-resistor common-source stage for JAX."""
    loaded = Library()
    loaded.read(
        ".option scale=1e-6\n"
        ".model nmos.0 nmos level=54 version=4.5 "
        "lmin=0.1u lmax=0.2u wmin=0.5u wmax=2.0u "
        "vth0=0.4 toxe=4e-9 u0=400 vsat=1e5 rdsw=200 cgso=2e-10 cgdo=2e-10\n"
        ".model nmos.1 nmos level=54 version=4.5 "
        "lmin=0.2u lmax=0.5u wmin=0.5u wmax=2.0u "
        "vth0=0.45 toxe=4e-9 u0=380 vsat=1e5 cgso=2e-10 cgdo=2e-10\n"
        ".model rpoly r rsh=200 narrow=0.05u short=0.1u\n"
        ".subckt nfet d g s b w=1.0 l=0.15 nf=1\n"
        "M1 d g s b nmos w={w} l={l} nf={nf} "
        "ad={w * 0.27} as={w * 0.27} pd={2 * (w + 0.27)} ps={2 * (w + 0.27)}\n"
        ".ends\n"
        ".subckt rload a b w=1.0 l=20.0\n"
        "R1 a b rpoly w={w} l={l}\n"
        ".ends\n",
        "jax_pdk.lib",
    )
    mosfet_placement = loaded.placement("nfet", w=1.0, l=0.15, m=1.0)
    load_placement = loaded.placement("rload", w=1.0, l=20.0)

    @component
    def stage(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(1.8 * V), name="VDD")
        gate = c.place(devices.voltage_source(0.9 * V), name="VG")
        load = c.place(load_placement.component(), name="RL")
        mosfet = c.place(mosfet_placement.component(), name="M1")
        capacitor = c.place(devices.capacitor(0.1 * pF), name="CL")
        c.connect(supply.positive, load.a, name="vdd")
        c.connect(load.b, mosfet.d, capacitor.a, name="drain")
        c.connect(gate.positive, mosfet.g, name="gate")
        c.connect(
            supply.negative,
            gate.negative,
            mosfet.s,
            mosfet.b,
            capacitor.b,
            ground,
        )

    return stage(), {"M1": mosfet_placement, "RL": load_placement}


def test_subcircuit_jax_jit_grad_jacrev_and_ac() -> None:
    """Verifies `jit`, `value_and_grad`, `jacrev`, and AC gradients on subcircuits."""
    stage, subcircuits = _synthetic_pdk_stage()
    op_wrapped = jax_function(stage, OperatingPoint(), subcircuits=subcircuits)
    op_target = jolty_function(stage, OperatingPoint(), subcircuits=subcircuits)
    ac_analysis = Ac("VG", [1 * MHz, 100 * MHz])
    ac_wrapped = jax_function(stage, ac_analysis, subcircuits=subcircuits)
    ac_target = jolty_function(stage, ac_analysis, subcircuits=subcircuits)

    point: dict[str, dict[str, float]] = {
        "M1": {"w": 1.1, "l": 0.16, "m": 1.5},
        "RL": {"w": 1.2, "l": 18.0},
        "CL": {"capacitance": 0.12e-12},
    }

    with jax.enable_x64():
        jax_point = _as_jax_parameters(point, dtype=jnp.float64)

        def op_loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            drain = op_wrapped(parameters)["drain"]
            return jnp.sum((drain - 1.0) ** 2)

        value, gradient = jax.jit(jax.value_and_grad(op_loss))(jax_point)
        target_outputs, pullback = jolty_vjp(op_target, point)
        expected = pullback(
            {"drain": 2.0 * (np.asarray(target_outputs["drain"]) - 1.0)}
        )
        # XLA JIT scalar arithmetic rounding (~4.5 eps).
        np.testing.assert_allclose(
            float(value),
            float(np.sum((np.asarray(target_outputs["drain"]) - 1.0) ** 2)),
            rtol=1e-15,
            atol=0.0,
        )
        for instance, inner in point.items():
            for name in inner:
                np.testing.assert_array_equal(
                    float(gradient[instance][name]),
                    _expect_real_scalar(expected[instance][name]),
                )
        _assert_matches_central_difference(
            op_loss,
            gradient,
            point,
            relative_tolerance=SUBCIRCUIT_CENTRAL_DIFF_RTOL,
        )

        # `jax.jacrev` agrees with `jolty.vjp` on drain voltage.
        jacobian = jax.jacrev(lambda current: op_wrapped(current)["drain"][0])(
            jax_point
        )
        unit_expected = pullback({"drain": np.array([1.0])})
        for instance, inner in point.items():
            for name in inner:
                np.testing.assert_array_equal(
                    float(jacobian[instance][name]),
                    _expect_real_scalar(unit_expected[instance][name]),
                )

        # AC Wirtinger gradient matches `jolty.vjp` bit-for-bit and finite differences.
        def ac_loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            phasor = ac_wrapped(parameters)["drain"]
            return jnp.sum(phasor.real + 2.0 * phasor.imag)

        ac_gradient = jax.jit(jax.grad(ac_loss))(jax_point)
        _, ac_pullback = jolty_vjp(ac_target, point)
        ac_expected = ac_pullback(
            {"drain": np.full(2, 1.0 - 2.0j, dtype=np.complex128)}
        )
        for instance, inner in point.items():
            for name in inner:
                np.testing.assert_array_equal(
                    float(ac_gradient[instance][name]),
                    float(np.real(ac_expected[instance][name])),
                )
        _assert_matches_central_difference(
            ac_loss,
            ac_gradient,
            point,
            relative_tolerance=SUBCIRCUIT_CENTRAL_DIFF_RTOL,
        )

        # Invalid subcircuit parameter raises `SpiceError` during structure check.
        with pytest.raises(SpiceError, match="has no parameter `bogus`"):
            op_wrapped({"M1": {"bogus": jnp.asarray(1.0, dtype=jnp.float64)}})


def test_subcircuit_jax_vmap_across_bins_and_empty() -> None:
    """Verifies `jax.vmap` on bound subcircuits across BSIM4 bins and empty batches."""
    stage, subcircuits = _synthetic_pdk_stage()
    op_wrapped = jax_function(stage, OperatingPoint(), subcircuits=subcircuits)

    with jax.enable_x64():
        batched_point = {
            "M1": {
                "w": jnp.array([0.9, 1.2, 1.0], dtype=jnp.float64),
                "l": jnp.array([0.14, 0.17, 0.35], dtype=jnp.float64),
                "m": jnp.array([1.0, 1.5, 2.0], dtype=jnp.float64),
            },
            "RL": {
                "w": jnp.array([1.0, 1.1, 1.2], dtype=jnp.float64),
                "l": jnp.array([18.0, 20.0, 22.0], dtype=jnp.float64),
            },
            "CL": {
                "capacitance": jnp.array(
                    [0.1e-12, 0.12e-12, 0.15e-12], dtype=jnp.float64
                )
            },
        }
        batched_gradients = jax.jit(
            jax.grad(lambda current: jnp.sum(jax.vmap(op_wrapped)(current)["drain"]))
        )(batched_point)
        for index in range(3):
            row = {
                instance: {name: values[index] for name, values in inner.items()}
                for instance, inner in batched_point.items()
            }
            scalar_gradient = jax.grad(
                lambda current: jnp.sum(op_wrapped(current)["drain"])
            )(row)
            for instance, inner in batched_point.items():
                for name in inner:
                    np.testing.assert_array_equal(
                        float(batched_gradients[instance][name][index]),
                        float(scalar_gradient[instance][name]),
                    )

        # Empty 0-length `jax.vmap` works cleanly on forward and backward passes.
        empty_point = {
            "M1": {
                "w": jnp.empty((0,), dtype=jnp.float64),
                "l": jnp.empty((0,), dtype=jnp.float64),
            }
        }
        empty_grad = jax.grad(
            lambda current: jnp.sum(jax.vmap(op_wrapped)(current)["drain"])
        )(empty_point)
        assert empty_grad["M1"]["w"].shape == (0,)
        assert empty_grad["M1"]["l"].shape == (0,)


@pytest.fixture(scope="session")
def sky130_library(pytestconfig: pytest.Config) -> Library:
    """Loads the SKY130 typical-corner library from pytest's pinned PDK cache."""
    assert pytestconfig.cache is not None
    settings = Settings(cache=pytestconfig.cache.mkdir("downloads"))
    return kits.read(sky130.KIT, ((sky130.LIBRARY, sky130.TYPICAL),), settings=settings)


def test_sky130_nfet_jax_jit_value_and_grad_matches_vjp_and_finite_differences(
    sky130_library: Library,
) -> None:
    """Verifies `jax.jit(jax.value_and_grad)` on a real SKY130 BSIM4 MOSFET stage."""
    baseline = {"w": 0.92, "l": 0.165, "m": 2.0}
    placement = sky130_library.placement("sky130_fd_pr__nfet_01v8", **baseline)

    @component
    def stage(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(1.8 * V), name="VDD")
        gate = c.place(devices.voltage_source(0.85 * V), name="VG")
        load = c.place(devices.resistor(5 * kohm), name="RL")
        capacitor = c.place(devices.capacitor(0.2 * pF), name="CL")
        mosfet = c.place(placement.component(), name="M1")
        c.connect(supply.positive, load.a, name="vdd")
        c.connect(load.b, mosfet.d, capacitor.a, name="drain")
        c.connect(gate.positive, mosfet.g, name="gate")
        c.connect(
            supply.negative,
            gate.negative,
            mosfet.s,
            mosfet.b,
            capacitor.b,
            ground,
        )

    circuit_top = stage()
    point: dict[str, dict[str, float]] = {"M1": dict(baseline)}

    op_wrapped = jax_function(
        circuit_top, OperatingPoint(), subcircuits={"M1": placement}
    )
    op_target = jolty_function(
        circuit_top, OperatingPoint(), subcircuits={"M1": placement}
    )
    ac_analysis = Ac("VG", [1 * MHz, 100 * MHz])
    ac_wrapped = jax_function(circuit_top, ac_analysis, subcircuits={"M1": placement})
    ac_target = jolty_function(circuit_top, ac_analysis, subcircuits={"M1": placement})

    with jax.enable_x64():
        jax_point = _as_jax_parameters(point, dtype=jnp.float64)

        def op_loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            return jnp.sum(op_wrapped(parameters)["drain"] ** 2)

        _, op_gradient = jax.jit(jax.value_and_grad(op_loss))(jax_point)
        op_outputs, op_pullback = jolty_vjp(op_target, point)
        op_expected = op_pullback({"drain": 2.0 * np.asarray(op_outputs["drain"])})
        for name in baseline:
            np.testing.assert_array_equal(
                float(op_gradient["M1"][name]),
                _expect_real_scalar(op_expected["M1"][name]),
            )
        _assert_matches_central_difference(
            op_loss,
            op_gradient,
            point,
            relative_tolerance=SUBCIRCUIT_CENTRAL_DIFF_RTOL,
        )

        def ac_loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
            phasor = ac_wrapped(parameters)["drain"]
            return jnp.sum(phasor.real**2 + phasor.imag**2)

        _, ac_gradient = jax.jit(jax.value_and_grad(ac_loss))(jax_point)
        ac_outputs, ac_pullback = jolty_vjp(ac_target, point)
        ac_expected = ac_pullback(
            {"drain": 2.0 * np.conj(np.asarray(ac_outputs["drain"]))}
        )
        for name in baseline:
            np.testing.assert_array_equal(
                float(ac_gradient["M1"][name]),
                float(np.real(ac_expected["M1"][name])),
            )
        _assert_matches_central_difference(
            ac_loss,
            ac_gradient,
            point,
            relative_tolerance=SUBCIRCUIT_CENTRAL_DIFF_RTOL,
        )

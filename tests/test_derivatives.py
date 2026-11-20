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

from jolty import Builder, Role, component, devices, function, jvp, vjp
from jolty.errors import ElaborationError, SimulationError
from jolty.plan import (
    LOOP_GAIN,
    NOISE_DENSITY,
    REFERRED_NOISE,
    Ac,
    DcSweep,
    Noise,
    OperatingPoint,
    Stability,
    Transient,
)
from jolty.simulate import Differentiable, Outputs, Parameters
from jolty.units import Hz, MHz, V, kHz, kohm, nF, ns, ohm, pF, uA, um, us
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


@component
def _feedback_loop(c: Builder) -> None:
    """Builds an inverting feedback loop broken by floating probe ``VP``."""
    ground = c.port("ground", Role.GROUND)
    amplifier = c.place(devices.voltage_controlled_voltage_source(-1000.0), name="E1")
    output_resistor = c.place(devices.resistor(100 * ohm), name="RO")
    probe = c.place(devices.voltage_source(0 * V), name="VP")
    upper = c.place(devices.resistor(9 * kohm), name="R1")
    lower = c.place(devices.resistor(1 * kohm), name="R2")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(amplifier.positive, output_resistor.a, name="e")
    c.connect(output_resistor.b, probe.negative, name="y")
    c.connect(probe.positive, upper.a, name="x")
    c.connect(upper.b, lower.a, capacitor.a, amplifier.control_positive, name="in")
    c.connect(
        amplifier.negative,
        amplifier.control_negative,
        lower.b,
        capacitor.b,
        ground,
    )


def test_stability_function_jvp_vjp_and_batched_pullback() -> None:
    analysis = Stability("VP", [1 * kHz, 100 * kHz, 1 * MHz])
    target = function(_feedback_loop(), analysis)
    assert target.outputs == (LOOP_GAIN,)
    assert target.points == 3

    point: dict[str, dict[str, float]] = {
        "RO": {"resistance": 100.0},
        "R1": {"resistance": 9000.0},
        "R2": {"resistance": 1000.0},
        "C1": {"capacitance": 1e-9},
    }
    tangents: dict[str, dict[str, float]] = {
        "RO": {"resistance": 15.0},
        "R1": {"resistance": -700.0},
        "R2": {"resistance": 120.0},
        "C1": {"capacitance": 2e-10},
    }
    outputs, pushed = jvp(target, point, tangents)
    same, pull = vjp(target, point)
    assert list(outputs) == [LOOP_GAIN]
    np.testing.assert_array_equal(outputs[LOOP_GAIN], same[LOOP_GAIN])
    cotangent = {
        LOOP_GAIN: np.array([1.2 - 0.7j, -0.5 + 1.4j, 0.8 + 0.3j], dtype=np.complex128)
    }
    forward, magnitude = inner(cotangent, pushed)
    pulled = pull(cotangent)
    backward = sum(
        complex(pulled[path][name]) * tangents[path][name]
        for path, named in point.items()
        for name in named
    )
    terms = sum(
        abs(complex(pulled[path][name]) * tangents[path][name])
        for path, named in point.items()
        for name in named
    )
    assert abs(forward - backward) <= TRANSPOSED * max(magnitude, terms)

    # Verify batch and batch_vjp match single-point calls bit for bit.
    names = [
        ("RO", "resistance"),
        ("R1", "resistance"),
        ("R2", "resistance"),
        ("C1", "capacitance"),
    ]
    batch_rows = np.array(
        [
            [100.0, 9000.0, 1000.0, 1e-9],
            [120.0, 8000.0, 1200.0, 1.5e-9],
            [80.0, 10000.0, 900.0, 0.8e-9],
        ]
    )
    batched = target.batch(names, batch_rows)
    batch_outputs, batch_pull = target.batch_vjp(names, batch_rows)
    np.testing.assert_array_equal(batched[LOOP_GAIN], batch_outputs[LOOP_GAIN])
    for row_index, values in enumerate(batch_rows):
        scalar_point = {
            path: {name: float(value)}
            for (path, name), value in zip(names, values.tolist(), strict=True)
        }
        np.testing.assert_array_equal(
            np.asarray(batched[LOOP_GAIN])[row_index], target(scalar_point)[LOOP_GAIN]
        )

    runs = [2, 0, 2]
    batch_cotangent = {
        LOOP_GAIN: np.stack(
            [cotangent[LOOP_GAIN] * (row + 1) for row in range(len(runs))]
        )
    }
    pulled_batch = batch_pull(batch_cotangent, runs)
    for row_index, run_index in enumerate(runs):
        scalar_point = {
            path: {name: float(value)}
            for (path, name), value in zip(
                names, batch_rows[run_index].tolist(), strict=True
            )
        }
        _, single_pull = vjp(target, scalar_point)
        alone = single_pull({LOOP_GAIN: batch_cotangent[LOOP_GAIN][row_index]})
        np.testing.assert_array_equal(
            pulled_batch[row_index], [alone[path][name] for path, name in names]
        )

    with pytest.raises(SimulationError, match="not a floating voltage source"):
        function(_feedback_loop(), Stability("R1", [1 * kHz]))(point)


@component
def _noisy_mosfet_stage(c: Builder) -> None:
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


def test_noise_function_jvp_vjp_and_finite_differences() -> None:
    frequencies = [10 * Hz, 10 * kHz, 10 * MHz]
    output_only = function(_noisy_mosfet_stage(), Noise("out", frequencies))
    assert output_only.outputs == (NOISE_DENSITY,)
    assert output_only.points == 3

    analysis = Noise("out", frequencies, source="VG")
    target = function(_noisy_mosfet_stage(), analysis)
    assert target.outputs == (NOISE_DENSITY, REFERRED_NOISE)
    assert target.points == 3

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
    tangents: dict[str, dict[str, float]] = {
        "VG": {"waveform.dc": 0.15},
        "RG": {"resistance": 500.0},
        "RD": {"resistance": -800.0},
        "CL": {"capacitance": 3e-13},
        "M1": {
            "width": 3e-7,
            "threshold_voltage": -0.05,
            "flicker_coefficient": 2e-34,
            "flicker_exponent": 0.1,
        },
    }

    # Output-only noise rejects referred cotangent and satisfies JVP/VJP transpose.
    solo_outputs, solo_pushed = jvp(output_only, point, tangents)
    solo_same, solo_pull = vjp(output_only, point)
    assert list(solo_outputs) == [NOISE_DENSITY]
    np.testing.assert_array_equal(solo_outputs[NOISE_DENSITY], solo_same[NOISE_DENSITY])
    with pytest.raises(ValueError, match="gives no output `referred`"):
        solo_pull({REFERRED_NOISE: np.ones(3)})
    solo_cotangent = {NOISE_DENSITY: np.array([1.5, -0.7, 2.3])}
    solo_forward, solo_magnitude = inner(solo_cotangent, solo_pushed)
    solo_pulled = solo_pull(solo_cotangent)
    solo_backward = sum(
        float(np.real(solo_pulled[path][name])) * tangents[path][name]
        for path, named in point.items()
        for name in named
    )
    solo_terms = sum(
        abs(float(np.real(solo_pulled[path][name])) * tangents[path][name])
        for path, named in point.items()
        for name in named
    )
    assert abs(solo_forward - solo_backward) <= TRANSPOSED * max(
        solo_magnitude, solo_terms
    )

    # Full noise (output density + input-referred density) transpose and finite diffs.
    outputs, pushed = jvp(target, point, tangents)
    same, pull = vjp(target, point)
    assert list(outputs) == [NOISE_DENSITY, REFERRED_NOISE]
    for key in (NOISE_DENSITY, REFERRED_NOISE):
        np.testing.assert_array_equal(outputs[key], same[key])
    cotangent = {
        NOISE_DENSITY: np.array([1.2, -0.5, 0.8]),
        REFERRED_NOISE: np.array([-0.4, 1.7, 0.6]),
    }
    forward, magnitude = inner(cotangent, pushed)
    pulled = pull(cotangent)
    backward = sum(
        float(np.real(pulled[path][name])) * tangents[path][name]
        for path, named in point.items()
        for name in named
    )
    terms = sum(
        abs(float(np.real(pulled[path][name])) * tangents[path][name])
        for path, named in point.items()
        for name in named
    )
    assert abs(forward - backward) <= TRANSPOSED * max(magnitude, terms)

    # Verify both output density and referred noise pullbacks against central diffs.
    step = 1e-4
    for key in (NOISE_DENSITY, REFERRED_NOISE):
        nominal = np.asarray(outputs[key], dtype=np.float64)
        weights = np.array([1.0, 2.0, 3.0]) / nominal
        gradient = pull({key: weights})
        for path, named in point.items():
            for name, value in named.items():
                delta = value * step
                plus = {**point, path: {**named, name: value + delta}}
                minus = {**point, path: {**named, name: value - delta}}
                loss_plus = float(np.sum(weights * np.asarray(target(plus)[key])))
                loss_minus = float(np.sum(weights * np.asarray(target(minus)[key])))
                finite = (loss_plus - loss_minus) / (2.0 * delta)
                # Central difference O(step^2) = 1e-8 plus rounding eps / step ~ 1e-12.
                np.testing.assert_allclose(
                    float(np.real(gradient[path][name])), finite, rtol=1e-5
                )


def test_noise_function_batch_and_batch_vjp() -> None:
    frequencies = [10 * Hz, 10 * kHz, 10 * MHz]
    target = function(_noisy_mosfet_stage(), Noise("out", frequencies, source="VG"))
    names = [
        ("VG", "waveform.dc"),
        ("RD", "resistance"),
        ("CL", "capacitance"),
        ("M1", "width"),
    ]
    batch_rows = np.array(
        [
            [1.2, 10e3, 2e-12, 2e-6],
            [1.3, 8e3, 1.5e-12, 3e-6],
            [1.1, 12e3, 2.5e-12, 1.5e-6],
        ]
    )
    batched = target.batch(names, batch_rows)
    batch_outputs, batch_pull = target.batch_vjp(names, batch_rows)
    for key in (NOISE_DENSITY, REFERRED_NOISE):
        np.testing.assert_array_equal(batched[key], batch_outputs[key])
        for row_index, values in enumerate(batch_rows):
            scalar_point = {
                path: {name: float(value)}
                for (path, name), value in zip(names, values.tolist(), strict=True)
            }
            np.testing.assert_array_equal(
                np.asarray(batched[key])[row_index], target(scalar_point)[key]
            )

    cotangent = {
        NOISE_DENSITY: np.array([1.2, -0.5, 0.8]),
        REFERRED_NOISE: np.array([-0.4, 1.7, 0.6]),
    }
    runs = [2, 0, 2]
    batch_cotangent = {
        NOISE_DENSITY: np.stack(
            [cotangent[NOISE_DENSITY] * (row + 1) for row in range(len(runs))]
        ),
        REFERRED_NOISE: np.stack(
            [cotangent[REFERRED_NOISE] * (row + 2) for row in range(len(runs))]
        ),
    }
    pulled_batch = batch_pull(batch_cotangent, runs)
    for row_index, run_index in enumerate(runs):
        scalar_point = {
            path: {name: float(value)}
            for (path, name), value in zip(
                names, batch_rows[run_index].tolist(), strict=True
            )
        }
        _, single_pull = vjp(target, scalar_point)
        alone = single_pull(
            {
                NOISE_DENSITY: batch_cotangent[NOISE_DENSITY][row_index],
                REFERRED_NOISE: batch_cotangent[REFERRED_NOISE][row_index],
            }
        )
        np.testing.assert_array_equal(
            pulled_batch[row_index], [alone[path][name] for path, name in names]
        )

    with pytest.raises(SimulationError, match="not an independent source"):
        function(_noisy_mosfet_stage(), Noise("out", [1 * kHz], source="RD"))(
            {"RD": {"resistance": 10e3}}
        )

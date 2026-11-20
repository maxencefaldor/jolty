# Copyright 2026 Maxence Faldor
"""Differentiable circuit functions, forward-mode JVPs, and reverse-mode VJPs.

Engine tests check JVPs against VJPs across all analyses to solver rounding, plus closed
forms and finite differences. The Python suite verifies `jolty.function`, `jolty.jvp`,
and `jolty.vjp` match `jax.jvp`/`jax.vjp` calling conventions and batched evaluation.
"""

from collections.abc import Mapping
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

from jolty import (
    Builder,
    Component,
    Role,
    Simulation,
    component,
    devices,
    function,
    jvp,
    vjp,
)
from jolty.errors import ElaborationError, SimulationError, SpiceError
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
from jolty.settings import Settings
from jolty.simulate import Differentiable, Function, Outputs, Parameters
from jolty.spice import BinBounds, Library, kits, sky130
from jolty.units import GHz, Hz, MHz, V, kHz, kohm, nF, ns, ohm, pF, uA, um, us
from jolty.waveforms import Sine
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

# SPICE and PDK subcircuit differentiation.

SUBCIRCUIT_FD_STEP: Final = 1e-4
"""Relative central-difference step for nonlinear BSIM4 and diode checks."""

SUBCIRCUIT_FD_RTOL: Final = 1e-5
"""Relative tolerance for `step=1e-4` nonlinear central differences.

Covers `O(step^2)` truncation (`~1e-8` times curvature) plus BSIM4/diode Newton
convergence noise divided by `2 * step`."""


def _synthetic_pdk_library() -> Library:
    """Builds a two-bin BSIM4 and geometric-resistor SPICE library for testing."""
    loaded = Library()
    loaded.read(
        ".option scale=1e-6\n"
        ".model nmos.0 nmos level=54 version=4.5 "
        "lmin=0.1u lmax=0.2u wmin=0.5u wmax=2.0u "
        "vth0=0.4 toxe=4e-9 u0=400 vsat=1e5 rdsw=200 cgso=2e-10 cgdo=2e-10\n"
        ".model nmos.1 nmos level=54 version=4.5 "
        "lmin=0.2u lmax=0.5u wmin=0.5u wmax=2.0u "
        "vth0=0.45 toxe=4e-9 u0=380 vsat=1e5 eta0=0.12 cgso=2e-10 cgdo=2e-10\n"
        ".model rpoly r rsh=200 narrow=0.05u short=0.1u\n"
        ".subckt nfet d g s b w=1.0 l=0.15 nf=1 unused=1.0\n"
        "M1 d g s b nmos w={w} l={l} nf={nf} "
        "ad={w * 0.27} as={w * 0.27} pd={2 * (w + 0.27)} ps={2 * (w + 0.27)}\n"
        ".ends\n"
        ".subckt rload a b w=1.0 l=20.0\n"
        "R1 a b rpoly w={w} l={l}\n"
        ".ends\n",
        "synthetic_pdk.lib",
    )
    return loaded


def _synthetic_common_source(
    transistor: Component,
    load: Component,
    *,
    transient_drive: bool = False,
) -> Component:
    """Builds a common-source stage with subcircuit MOSFET, load resistor, and `CL`."""

    @component
    def stage(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(1.8 * V), name="VDD")
        gate_source = (
            devices.voltage_source(
                Sine(offset=0.9 * V, amplitude=0.05 * V, frequency=1 * MHz)
            )
            if transient_drive
            else devices.voltage_source(0.9 * V)
        )
        gate = c.place(gate_source, name="VG")
        placed_load = c.place(load, name="RL")
        placed_mosfet = c.place(transistor, name="M1")
        capacitor = c.place(devices.capacitor(0.1 * pF), name="CL")
        c.connect(supply.positive, placed_load.a, name="vdd")
        c.connect(placed_load.b, placed_mosfet.d, capacitor.a, name="drain")
        c.connect(gate.positive, placed_mosfet.g, name="gate")
        c.connect(
            supply.negative,
            gate.negative,
            placed_mosfet.s,
            placed_mosfet.b,
            capacitor.b,
            ground,
        )

    return stage()


def _assert_transpose_agrees(
    target: Function, point: Parameters, tangents: Parameters
) -> tuple[Outputs, Mapping[str, Mapping[str, float | complex]]]:
    """Checks `jvp`/`vjp` primal equality and transpose identity within `TRANSPOSED`."""
    outputs, pushed = jvp(target, point, tangents)
    same, pull = vjp(target, point)
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
        for path, named in point.items()
        for name in named
    )
    terms = sum(
        abs(complex(pulled[path][name]) * tangents[path][name])
        for path, named in point.items()
        for name in named
    )
    assert abs(forward - backward) <= TRANSPOSED * max(magnitude, terms)
    return outputs, pulled


@pytest.mark.parametrize(
    ("analysis", "transient_drive"),
    [
        (OperatingPoint(), False),
        (DcSweep("VG", [0.8 * V, 0.9 * V, 1.0 * V]), False),
        (Ac("VG", [1 * MHz, 100 * MHz]), False),
        (Transient(2 * us, max_step=20 * ns), True),
    ],
    ids=["operating_point", "dc_sweep", "ac", "transient"],
)
def test_subcircuit_jvp_and_vjp_transpose_across_analyses(
    analysis: Differentiable, *, transient_drive: bool
) -> None:
    loaded = _synthetic_pdk_library()
    mosfet_placement = loaded.placement("nfet", w=1.0, l=0.15, m=1.0)
    load_placement = loaded.placement("rload", w=1.0, l=20.0)
    stage = _synthetic_common_source(
        mosfet_placement.component(),
        load_placement.component(),
        transient_drive=transient_drive,
    )
    f = function(
        stage,
        analysis,
        subcircuits={"M1": mosfet_placement, "RL": load_placement},
    )
    point: Parameters = {
        "M1": {"w": 1.1, "l": 0.16, "m": 1.5, "unused": 2.0},
        "RL": {"w": 1.2, "l": 18.0},
        "CL": {"capacitance": 0.12e-12},
    }
    tangents: Parameters = {
        "M1": {"w": 0.08, "l": -0.01, "m": 0.2, "unused": 3.0},
        "RL": {"w": -0.05, "l": 1.5},
        "CL": {"capacitance": 0.02e-12},
    }
    _assert_transpose_agrees(f, point, tangents)


def test_subcircuit_vjp_matches_fresh_simulation_central_differences_across_bins() -> (
    None
):
    """Verifies subcircuit VJP against fresh Simulation central differences across bins.

    Checks both bin 0 (`rdsw=200`) and bin 1 (`rdsw` omitted, reset to engine default)
    to confirm default restoration matches a fresh build within rounding (`rel=1e-12`).
    """
    loaded = _synthetic_pdk_library()
    mosfet_placement = loaded.placement("nfet", w=1.0, l=0.15, m=1.0)
    load_placement = loaded.placement("rload", w=1.0, l=20.0)
    stage = _synthetic_common_source(
        mosfet_placement.component(), load_placement.component()
    )
    f = function(
        stage,
        OperatingPoint(),
        subcircuits={"M1": mosfet_placement, "RL": load_placement},
    )

    def fresh_drain(point: Parameters) -> float:
        rebuilt = _synthetic_common_source(
            mosfet_placement.component(**point["M1"]),
            load_placement.component(**point["RL"]),
        )
        return Simulation(rebuilt).run(OperatingPoint()).voltage("drain")

    for expected_model, length in (("nmos.0", 0.15), ("nmos.1", 0.35)):
        point: Parameters = {
            "M1": {"w": 1.0, "l": length, "m": 1.5},
            "RL": {"w": 1.0, "l": 20.0},
        }
        active_bins = f.bins(point)
        assert active_bins["M1/M1"].model == expected_model
        assert active_bins["M1/M1"].contains(length * um, 1.0 * um)

        outputs, pull = vjp(f, point)
        # Restored bin defaults match a fresh Simulation build to solver rounding.
        assert float(np.asarray(outputs["drain"])[0]) == pytest.approx(
            fresh_drain(point), rel=1e-12
        )
        gradient = pull({"drain": np.array([1.0])})
        for path, named in point.items():
            for name, value in named.items():
                step = value * SUBCIRCUIT_FD_STEP
                plus = {**point, path: {**named, name: value + step}}
                minus = {**point, path: {**named, name: value - step}}
                finite = (fresh_drain(plus) - fresh_drain(minus)) / (2.0 * step)
                found = float(np.real(gradient[path][name]))
                assert abs(found - finite) <= SUBCIRCUIT_FD_RTOL * max(
                    abs(found), abs(finite), 1.0
                ), (expected_model, path, name)

    # Differentiating only an unused argument yields exact zeros cleanly.
    unused_point: Parameters = {"M1": {"unused": 2.5}}
    unused_out, unused_pushed = jvp(f, unused_point, {"M1": {"unused": 1.0}})
    assert float(np.asarray(unused_pushed["drain"])[0]) == 0.0
    same_out, unused_pull = vjp(f, unused_point)
    np.testing.assert_array_equal(unused_out["drain"], same_out["drain"])
    assert unused_pull({"drain": np.array([1.0])}) == {"M1": {"unused": 0.0}}


@pytest.mark.parametrize("analysis", BATCHED, ids=["operating_point", "dc_sweep", "ac"])
def test_subcircuit_batch_and_batch_vjp_across_bins_and_empty(
    analysis: Differentiable,
) -> None:
    loaded = _synthetic_pdk_library()
    mosfet_placement = loaded.placement("nfet", w=1.0, l=0.15)
    load_placement = loaded.placement("rload", w=1.0, l=20.0)
    stage = _synthetic_common_source(
        mosfet_placement.component(), load_placement.component()
    )
    # Match `DcSweep`/`Ac` excitation source name to `VG` on `stage`.
    adapted: Differentiable
    if isinstance(analysis, DcSweep):
        adapted = DcSweep("VG", [0.8 * V, 0.9 * V, 1.0 * V])
    elif isinstance(analysis, Ac):
        adapted = Ac("VG", [1 * MHz, 100 * MHz])
    else:
        adapted = analysis
    f = function(
        stage,
        adapted,
        subcircuits={"M1": mosfet_placement, "RL": load_placement},
    )
    names = [("M1", "w"), ("M1", "l"), ("RL", "l"), ("CL", "capacitance")]
    # Rows 0 and 1 lie in bin 0 (`l <= 0.2`); row 2 crosses into bin 1 (`l = 0.35`).
    batch_rows = np.array(
        [
            [0.9, 0.14, 18.0, 0.1e-12],
            [1.2, 0.17, 22.0, 0.15e-12],
            [1.0, 0.35, 20.0, 0.12e-12],
        ]
    )
    outputs = f.batch(names, batch_rows)
    for row_index, values in enumerate(batch_rows):
        scalar_point: dict[str, dict[str, float]] = {}
        for (path, name), value in zip(names, values.tolist(), strict=True):
            scalar_point.setdefault(path, {})[name] = value
        for output_name, found in f(scalar_point).items():
            np.testing.assert_array_equal(
                np.asarray(outputs[output_name])[row_index], found
            )

    batch_outputs, pullback = f.batch_vjp(names, batch_rows)
    runs = [2, 0, 2]
    cotangent = {
        name: np.stack(
            [
                np.linspace(1.0, 2.0, np.shape(values)[1]) * (row + index + 1)
                for row in range(len(runs))
            ]
        )
        for index, (name, values) in enumerate(batch_outputs.items())
    }
    pulled = pullback(cotangent, runs)
    for row_index, run_index in enumerate(runs):
        scalar_point = {}
        for (path, name), value in zip(
            names, batch_rows[run_index].tolist(), strict=True
        ):
            scalar_point.setdefault(path, {})[name] = value
        _, scalar_pull = vjp(f, scalar_point)
        alone = scalar_pull(
            {name: values[row_index] for name, values in cotangent.items()}
        )
        np.testing.assert_array_equal(
            pulled[row_index], [alone[path][name] for path, name in names]
        )

    # Empty 0-row batch returns empty outputs and gradients with expected dtypes.
    empty_rows = np.empty((0, len(names)), dtype=np.float64)
    empty_outputs = f.batch(names, empty_rows)
    assert np.asarray(empty_outputs["drain"]).shape == (0, f.points)
    _, empty_pullback = f.batch_vjp(names, empty_rows)
    expected_dtype = np.complex128 if isinstance(adapted, Ac) else np.float64
    empty_gradients = empty_pullback(
        {"drain": np.empty((0, f.points), dtype=expected_dtype)}
    )
    assert empty_gradients.shape == (0, len(names))
    assert empty_gradients.dtype == expected_dtype


def test_subcircuit_binding_validation_errors() -> None:
    loaded = _synthetic_pdk_library()
    loaded.read(
        ".subckt inner a b rval=1k\n"
        "R1 a b {rval}\n"
        ".ends\n"
        ".subckt outer a b rval=1k\n"
        "X1 a b inner rval={rval}\n"
        ".ends\n"
        ".subckt switched a b mode=0\n"
        ".if (mode > 0)\n"
        "R1 a b 1k\n"
        "R2 a b 2k\n"
        ".else\n"
        "R1 a b 1k\n"
        ".endif\n"
        ".ends\n",
        "errors.lib",
    )
    mosfet_placement = loaded.placement("nfet", w=1.0, l=0.15)
    load_placement = loaded.placement("rload", w=1.0, l=20.0)
    stage = _synthetic_common_source(
        mosfet_placement.component(), load_placement.component()
    )

    with pytest.raises(ValueError, match="must be non-empty"):
        function(stage, OperatingPoint(), subcircuits={"": mosfet_placement})
    with pytest.raises(ValueError, match="no subcircuit placement at `missing`"):
        function(stage, OperatingPoint(), subcircuits={"missing": mosfet_placement})
    with pytest.raises(ValueError, match="does not match the devices placed"):
        function(stage, OperatingPoint(), subcircuits={"M1": load_placement})
    with pytest.raises(ValueError, match="baseline parameters do not match"):
        function(
            stage,
            OperatingPoint(),
            subcircuits={"M1": loaded.placement("nfet", w=1.2, l=0.15)},
        )

    f = function(stage, OperatingPoint(), subcircuits={"M1": mosfet_placement})
    with pytest.raises(ValueError, match="cannot override both subcircuit `M1`"):
        f({"M1": {"w": 1.0}, "M1/M1": {"width": 1e-6}})
    with pytest.raises(SpiceError, match="has no parameter `bogus`"):
        f.check({"M1": ("bogus",)})

    # Reject nested overlapping bound subcircuit paths.
    outer_placement = loaded.placement("outer", rval=1000.0)
    inner_placement = loaded.placement("inner", rval=1000.0)

    @component
    def nested_bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), name="V1")
        block = c.place(outer_placement.component(), name="U1")
        c.connect(source.positive, block.a, name="in")
        c.connect(block.b, source.negative, ground)

    with pytest.raises(ValueError, match="cannot bind nested subcircuit `U1/X1`"):
        function(
            nested_bench(),
            OperatingPoint(),
            subcircuits={"U1": outer_placement, "U1/X1": inner_placement},
        )

    # Reject parameter override that alters subcircuit device topology.
    switched_placement = loaded.placement("switched", mode=0.0)

    @component
    def switched_bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), name="V1")
        block = c.place(switched_placement.component(), name="S1")
        c.connect(source.positive, block.a, name="in")
        c.connect(block.b, source.negative, ground)

    switched_fn = function(
        switched_bench(), OperatingPoint(), subcircuits={"S1": switched_placement}
    )
    with pytest.raises(ValueError, match="changed its device topology"):
        switched_fn({"S1": {"mode": 1.0}})


# Real SKY130 PDK end-to-end differentiation.


@pytest.fixture(scope="session")
def sky130_typical(pytestconfig: pytest.Config) -> Library:
    """Loads the SKY130 typical-corner library from pytest's pinned PDK cache."""
    assert pytestconfig.cache is not None
    settings = Settings(cache=pytestconfig.cache.mkdir("downloads"))
    return kits.read(sky130.KIT, ((sky130.LIBRARY, sky130.TYPICAL),), settings=settings)


def test_sky130_nfet_derivatives_and_active_bin_bounds(
    sky130_typical: Library,
) -> None:
    """Verifies SKY130 `nfet_01v8` exact Jacobian and bin 71 `BinBounds`.

    Scale conversion `w, l -> meters` is exact up to IEEE-754 rounding (`rel=1e-14`).
    """
    evaluated = sky130_typical.derivatives(
        "sky130_fd_pr__nfet_01v8", "M1", w=1.0, l=0.15
    )
    device_path = "M1/msky130_fd_pr__nfet_01v8"
    assert evaluated.bins[device_path] == BinBounds(
        model="sky130_fd_pr__nfet_01v8__model.71",
        min_length=1.5e-7,
        max_length=1.8e-7,
        min_width=8.4e-7,
        max_width=1.0e-6,
        finger_divisor=1.0,
    )
    assert evaluated.bins[device_path].contains(0.165 * um, 0.92 * um)
    assert not evaluated.bins[device_path].contains(0.20 * um, 0.92 * um)
    assert evaluated.derivatives[device_path]["width"]["w"] == pytest.approx(
        1e-6, rel=1e-14
    )
    assert evaluated.derivatives[device_path]["length"]["l"] == pytest.approx(
        1e-6, rel=1e-14
    )


def test_sky130_nfet_common_source_operating_point_and_ac_gradients(
    sky130_typical: Library,
) -> None:
    """Verifies end-to-end SKY130 BSIM4 `(w, l, m)` gradients for OP and AC gain."""
    # `(w=0.92, l=0.165)` lies strictly inside bin 71 (`[0.84, 1.0] x [0.15, 0.18]` um).
    baseline = {"w": 0.92, "l": 0.165, "m": 2.0}
    placement = sky130_typical.placement("sky130_fd_pr__nfet_01v8", **baseline)

    def build_stage(overrides: Mapping[str, float]) -> Component:
        @component
        def stage(c: Builder) -> None:
            ground = c.port("ground", Role.GROUND)
            supply = c.place(devices.voltage_source(1.8 * V), name="VDD")
            gate = c.place(devices.voltage_source(0.85 * V), name="VG")
            load = c.place(devices.resistor(5 * kohm), name="RL")
            capacitor = c.place(devices.capacitor(0.2 * pF), name="CL")
            mosfet = c.place(placement.component(**overrides), name="M1")
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

        return stage()

    base_circuit = build_stage({})
    tangents: Parameters = {"M1": {"w": 0.05, "l": -0.01, "m": 0.25}}

    # Operating-point gradients vs JVP transpose and fresh central differences.
    op_fn = function(base_circuit, OperatingPoint(), subcircuits={"M1": placement})
    assert (
        op_fn.bins()["M1/msky130_fd_pr__nfet_01v8"].model
        == "sky130_fd_pr__nfet_01v8__model.71"
    )
    point: Parameters = {"M1": dict(baseline)}
    _assert_transpose_agrees(op_fn, point, tangents)
    _, op_pull = vjp(op_fn, point)
    op_gradient = op_pull({"drain": np.array([1.0])})["M1"]
    for name, value in baseline.items():
        step = value * SUBCIRCUIT_FD_STEP
        plus = (
            Simulation(build_stage({name: value + step}))
            .run(OperatingPoint())
            .voltage("drain")
        )
        minus = (
            Simulation(build_stage({name: value - step}))
            .run(OperatingPoint())
            .voltage("drain")
        )
        finite = (plus - minus) / (2.0 * step)
        found = float(np.real(op_gradient[name]))
        assert abs(found - finite) <= SUBCIRCUIT_FD_RTOL * max(
            abs(found), abs(finite), 1.0
        ), ("op", name)

    # AC gain gradients vs JVP transpose and fresh Simulation central differences.
    ac_analysis = Ac("VG", [1 * MHz, 100 * MHz])
    ac_fn = function(base_circuit, ac_analysis, subcircuits={"M1": placement})
    ac_out, _ = _assert_transpose_agrees(ac_fn, point, tangents)
    _, ac_pull = vjp(ac_fn, point)

    drain_phasor = np.asarray(ac_out["drain"])
    ac_gradient = ac_pull({"drain": 2.0 * np.conj(drain_phasor)})["M1"]

    def ac_gain_energy(overrides: Mapping[str, float]) -> float:
        phasor = Simulation(build_stage(overrides)).run(ac_analysis).voltage("drain")
        return float(np.sum(np.abs(phasor) ** 2))

    for name, value in baseline.items():
        step = value * SUBCIRCUIT_FD_STEP
        finite = (
            ac_gain_energy({name: value + step}) - ac_gain_energy({name: value - step})
        ) / (2.0 * step)
        found = float(np.real(ac_gradient[name]))
        assert abs(found - finite) <= SUBCIRCUIT_FD_RTOL * max(
            abs(found), abs(finite), 1.0
        ), ("ac", name)


def test_sky130_mim_capacitor_and_diffusion_resistor_ac_gradients(
    sky130_typical: Library,
) -> None:
    """Verifies end-to-end SKY130 MIM cap and diffusion resistor `(w, l)` gradients."""
    resistor_base = {"w": 1.5, "l": 20.0}
    capacitor_base = {"w": 10.0, "l": 10.0}
    resistor_placement = sky130_typical.placement(
        "sky130_fd_pr__res_generic_nd", **resistor_base
    )
    capacitor_placement = sky130_typical.placement(
        "sky130_fd_pr__cap_mim_m3_1", **capacitor_base
    )

    def build_filter(
        resistor_overrides: Mapping[str, float],
        capacitor_overrides: Mapping[str, float],
    ) -> Component:
        @component
        def rc_filter(c: Builder) -> None:
            ground = c.port("ground", Role.GROUND)
            # Positive 0.5 V DC bias keeps the substrate diodes reverse-biased.
            source = c.place(devices.voltage_source(0.5 * V), name="V1")
            resistor = c.place(
                resistor_placement.component(**resistor_overrides), name="R1"
            )
            capacitor = c.place(
                capacitor_placement.component(**capacitor_overrides), name="C1"
            )
            c.connect(source.positive, resistor.t1, name="in")
            c.connect(resistor.t2, capacitor.c0, name="out")
            c.connect(source.negative, resistor.b, capacitor.c1, ground)

        return rc_filter()

    ac_analysis = Ac("V1", [10 * MHz, 100 * MHz, 1 * GHz])
    base_filter = build_filter({}, {})
    f = function(
        base_filter,
        ac_analysis,
        subcircuits={"R1": resistor_placement, "C1": capacitor_placement},
    )
    point: Parameters = {"R1": dict(resistor_base), "C1": dict(capacitor_base)}
    tangents: Parameters = {
        "R1": {"w": 0.1, "l": -1.0},
        "C1": {"w": -0.5, "l": 0.5},
    }
    outputs, _ = _assert_transpose_agrees(f, point, tangents)
    _, pull = vjp(f, point)

    out_phasor = np.asarray(outputs["out"])
    gradient = pull({"out": 2.0 * np.conj(out_phasor)})

    def transfer_energy(
        resistor_overrides: Mapping[str, float],
        capacitor_overrides: Mapping[str, float],
    ) -> float:
        phasor = (
            Simulation(build_filter(resistor_overrides, capacitor_overrides))
            .run(ac_analysis)
            .voltage("out")
        )
        return float(np.sum(np.abs(phasor) ** 2))

    for path, named in point.items():
        for name, value in named.items():
            step = value * SUBCIRCUIT_FD_STEP
            if path == "R1":
                plus = transfer_energy({name: value + step}, {})
                minus = transfer_energy({name: value - step}, {})
            else:
                plus = transfer_energy({}, {name: value + step})
                minus = transfer_energy({}, {name: value - step})
            finite = (plus - minus) / (2.0 * step)
            found = float(np.real(gradient[path][name]))
            assert abs(found - finite) <= SUBCIRCUIT_FD_RTOL * max(
                abs(found), abs(finite), 1.0
            ), (path, name)

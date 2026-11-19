# Copyright 2026 Maxence Faldor

import cmath
import math

import exact
import fixtures as precision_fixtures
import mpmath
import numpy as np
import pytest
import suite
from mpmath import mp, mpf
from simulators import stopped

from differential import SKIP, SKIP_XYCE
from jolty import _core
from jolty.plan import DcSweep


def test_phi_agrees_with_mpmath_on_both_sides_of_its_series() -> None:
    # Within the series, at its edge on either side, far out, and undamped.
    arguments = np.array(
        [-1e-9, -0.3, 0.5j, -0.999, -1.0, -1.001, -3.0, 2j, -50.0, -1e6, 40j]
    )
    first, second = exact.phi(arguments)
    for argument, one, two in zip(arguments, first, second, strict=True):
        z = mpmath.mpc(argument)
        exact_first = (mpmath.exp(z) - 1) / z
        exact_second = (mpmath.exp(z) - 1 - z) / z**2
        # Each is within its rounding: seven of a direct form's cancellation, and a
        # few roundings beside, as ROUNDINGS counts them.
        assert abs(one - complex(exact_first)) <= 16 * exact.EPSILON * abs(exact_first)
        assert abs(two - complex(exact_second)) <= (
            16 * exact.EPSILON * abs(exact_second)
        )


def test_an_rc_follows_its_closed_form_within_its_bound() -> None:
    resistance, capacitance, rise = 1e3, 1e-9, 1e-9
    tau = mpf(resistance) * mpf(capacitance)
    modal = exact.linear(
        ["v(out)"], [[-1 / tau]], [1 / tau], [[mpf(1)]], [(0.0, 0.0), (rise, 1.0)]
    )
    times = np.array([0.0, rise / 3, rise, 2 * rise, 1e-6, 5e-6, 2e-5])
    values, bound = modal.at(times)

    def ramp(time: object) -> object:
        """The response to a unit ramp from zero, ``t - τ(1 - e^{-t/τ})``, in mpmath."""
        at = mpf(time)
        return at - tau * (1 - mpmath.exp(-at / tau)) if at > 0 else mpf(0)

    # The response to a ramp to 1 V over `rise`, then held: two ramps' difference,
    # the second's time taken in mpmath, where the difference loses nothing.
    expected = [
        (mpf(ramp(time)) - mpf(ramp(mpf(time) - mpf(rise)))) / mpf(rise)
        for time in times
    ]
    for value, limit, reference in zip(values[0], bound[0], expected, strict=True):
        assert abs(mpf(float(value)) - reference) <= mpf(float(limit))


def test_the_tree_solver_agrees_with_the_general_one() -> None:
    generator = np.random.default_rng(7)
    count = 10
    parents = (0, *(int(generator.integers(0, node)) for node in range(1, count)))
    branches = tuple(float(value) for value in 10 ** -generator.uniform(1, 4, count))
    capacitances = tuple(
        float(value) for value in 10 ** generator.uniform(-12, -9, count)
    )
    network = exact.Tree(parents, branches, capacitances, 1 / 50)
    level = [(0.0, 0.0), (1e-9, 1.0)]
    solved = exact.tree(network, list(range(count)), level)
    # The same network's state equations, ẋ = -C⁻¹G x + C⁻¹ g₀ e₀ u, in mpmath.
    conductances = [[mpf(0)] * count for _ in range(count)]
    conductances[0][0] += mpf(network.source)
    for node in range(1, count):
        parent, g = parents[node], mpf(branches[node])
        conductances[node][node] += g
        conductances[parent][parent] += g
        conductances[node][parent] -= g
        conductances[parent][node] -= g
    states = [
        [-value / mpf(capacitances[row]) for value in conductances[row]]
        for row in range(count)
    ]
    inputs = [mpf(network.source) / mpf(capacitances[0]), *[mpf(0)] * (count - 1)]
    read = [[mpf(row == column) for column in range(count)] for row in range(count)]
    general = exact.linear(solved.outputs, states, inputs, read, level)
    # Tree eigen-rates vs mpmath state eigenvalues cast to f64 (~45 eps rounding).
    np.testing.assert_allclose(
        np.sort(solved.rates.real), np.sort(general.rates.real), rtol=1e-14
    )
    times = np.concatenate([np.linspace(0, 2e-9, 9), np.geomspace(3e-9, 1e-4, 30)])
    ours, our_bound = solved.at(times)
    theirs, their_bound = general.at(times)
    # Each is within its own bound of the exact answer, so within both of the other.
    assert np.all(np.abs(ours - theirs) <= our_bound + their_bound)


def test_a_tree_s_nodes_settle_at_the_source_s_level() -> None:
    network = exact.Tree((0, 0, 1), (0.0, 1e-3, 2e-3), (1e-12, 1e-10, 1e-9), 1e-2)
    solved = exact.tree(network, [0, 1, 2], [(0.0, 0.0), (1e-9, 1.0)])
    settled, bound = solved.at(np.array([1.0]))
    assert np.all(np.abs(settled - 1) <= bound)


def test_a_butterworth_ladder_s_poles_lie_on_its_circle() -> None:
    order, _, cutoff = precision_fixtures.BUTTERWORTH
    found = exact.poles(
        precision_fixtures.butterworth_states(precision_fixtures.butterworth_values())
    )
    circle = [
        cutoff * cmath.exp(1j * math.pi * (2 * k + order - 1) / (2 * order))
        for k in range(1, order + 1)
    ]
    # The element values are rounded to doubles, so the ladder's poles lie on the
    # circle to about that rounding, which a fifth-order ladder's sensitivity
    # amplifies by less than a thousand.
    assert len(found) == order
    for pole in found:
        # 5th-order Butterworth pole rounding amplified <1000x (<= 1e-13 * cutoff).
        assert min(abs(pole - each) for each in circle) <= 1e-13 * cutoff


def test_the_diode_s_voltage_balances_its_currents() -> None:
    thermal = mpf(_core.BOLTZMANN) * mpf(_core.REFERENCE_TEMPERATURE)
    thermal /= mpf(_core.ELEMENTARY_CHARGE)
    levels = np.array([0.1, 0.7, 5.0])
    voltages = exact.diode(levels, 1e3, 1e-14, thermal, 1e-12)
    for level, voltage in zip(levels, voltages, strict=True):
        v = mpf(float(voltage))
        through = (mpf(float(level)) - v) / 1000
        junction = mpf(1e-14) * (mpmath.exp(v / thermal) - 1) + mpf(1e-12) * v
        # The voltage is rounded to a double: the junction's current moves by its
        # conductance times that rounding, half an epsilon of the voltage.
        slope = mpf(1e-14) / thermal * mpmath.exp(v / thermal) + mpf(1e-12) + mpf(1e-3)
        assert abs(through - junction) <= slope * v * mpf(2) ** -52


def test_a_score_counts_the_digits_of_the_worst_point_up_to_its_limit() -> None:
    reference = np.array([1.0, 0.5, 0.0])
    simulated = reference + np.array([1e-6, 1e-4, 1e-3])
    compared = precision_fixtures.Compared(
        simulated, reference, np.full(3, 1e-12), np.arange(3.0)
    )
    output = precision_fixtures.ScoredOutput("v(out)", "mixed", 1.0, "V")
    mixed = suite.score(compared, output)["mixed"]
    # Exact -log10 of synthetic error counts (1e-3 -> 3, 1e-12 -> 12); float rounding.
    assert mixed.digits == pytest.approx(3.0)
    assert mixed.at == 2.0
    assert mixed.limit == pytest.approx(12.0)
    # A zero exact value has no relative error.
    assert "relative" not in suite.score(compared, output)
    capped = precision_fixtures.Compared(
        reference[:2] + 1e-15, reference[:2], np.full(2, 1e-10), np.arange(2.0)
    )
    relative = suite.score(
        capped, precision_fixtures.ScoredOutput("v", "relative", 0.0, "V")
    )["relative"]
    # Exact -log10(1e-10 / 0.5) = -log10(2e-10); default pytest.approx covers rounding.
    assert relative.digits == relative.limit == pytest.approx(-math.log10(2e-10))


def test_every_fixture_names_what_it_scores_and_how_its_answer_is_found() -> None:
    every = precision_fixtures.fixtures()
    assert len(every) >= 10
    for fixture in every.values():
        assert fixture.outputs
        assert fixture.reference
        assert mp.dps >= 50


@SKIP
def test_both_simulators_run_a_fixture_and_score_it() -> None:
    prepared = suite.Prepared.from_fixture(precision_fixtures.fixtures()["rc_step"])
    setting = suite.Setting("scaled", suite.DEFAULTS["relative_tolerance"] / 100)
    output = prepared.fixture.outputs[0]
    for ran in (
        suite.run_jolty(prepared, setting, 2),
        suite.run_ngspice(prepared, setting, 2, suite.Ngspice()),
    ):
        assert len(ran.seconds) == 2
        assert all(seconds > 0 for seconds in ran.seconds)
        assert ran.counts["accepted"] > 0
        compared = prepared.fixture.compare(ran.found)[output.name]
        mixed = suite.score(compared, output)["mixed"]
        # Both take trapezoidal steps of at most a fiftieth of the run, a tenth of the
        # time constant, whose error is at most about (h/τ)²/12 = 8e-4 of the 1 V
        # swing, and far above the exact answer's rounding.
        assert 3 < mixed.digits < mixed.limit


def test_equal_error_takes_each_simulator_s_fastest_run_to_the_target() -> None:
    output = precision_fixtures.ScoredOutput("v(out)", "mixed", 1.0, "V")

    def scored(
        simulator: suite.Simulator, tolerance: float, digits: float, seconds: float
    ) -> suite.FixtureRun:
        return suite.FixtureRun(
            simulator=simulator,
            sweep="reltol",
            relative_tolerance=tolerance,
            load=1.0,
            status="scored",
            seconds=seconds,
            scores={
                "v(out)": {"mixed": suite.Scored(digits=digits, limit=15.0, at=0.0)}
            },
        )

    runs = [
        scored(suite.Simulator.JOLTY, 1e-3, 3.0, 1e-4),
        scored(suite.Simulator.JOLTY, 1e-5, 5.5, 3e-4),
        scored(suite.Simulator.JOLTY, 1e-7, 6.0, 9e-4),
        scored(suite.Simulator.NGSPICE_SPARSE, 1e-3, 4.2, 2e-4),
        scored(suite.Simulator.NGSPICE_SPARSE, 1e-5, 4.0, 1e-4),
        suite.FixtureRun(
            simulator=suite.Simulator.NGSPICE_SPARSE,
            sweep="reltol",
            relative_tolerance=1e-7,
            load=1.0,
            status="failed",
            message="timestep too small",
        ),
        suite.FixtureRun(
            simulator=suite.Simulator.NGSPICE_KLU,
            sweep="defaults",
            relative_tolerance=1e-3,
            load=1.0,
            status="unsupported",
        ),
    ]
    document = suite.SuiteResults(
        commit="0",
        machine={},
        versions={},
        repeats=1,
        fixtures={
            "rc": suite.FixtureResults(
                describes="", reference="", outputs=[output], seconds=0.0, runs=runs
            )
        },
    )
    # The target is the most digits every simulator that ran reached: ngspice's 4.2.
    *_, row = suite.equal_error_table(document, None).splitlines()
    assert row == (
        "| `rc` | v(out), mixed | 4.2 | 300 µs, reltol 1e-05, 5.5 digits "
        "| 200 µs, reltol 0.001, 4.2 digits; 1 failed | unsupported |"
    )
    # A target beyond a simulator's reach marks it with the most it reached.
    *_, row = suite.equal_error_table(document, 5.0).splitlines()
    assert "| never: 4.2 digits at most; 1 failed |" in row


def test_xyce_s_tolerances_follow_each_sweep() -> None:
    assert suite.Setting("defaults", 1e-3).xyce_options() == ()
    # Along the relative tolerance's sweep, each group's relative tolerance is the
    # setting's and its absolute ones Xyce's own; scaled, those are scaled with it.
    alone = suite.Setting(suite.Sweep.RELTOL, 1e-5).xyce_options()
    scaled = suite.Setting(suite.Sweep.SCALED, 1e-5).xyce_options()
    for group, (one, other) in zip(
        suite.XYCE_DEFAULTS, zip(alone, scaled, strict=True), strict=True
    ):
        defaults = suite.XYCE_DEFAULTS[group]
        assert one == f".options {group} " + " ".join(
            f"{name}={1e-5 if name == 'reltol' else value!r}"
            for name, value in defaults.items()
        )
        assert other == f".options {group} " + " ".join(
            f"{name}={1e-5 if name == 'reltol' else value * 1e-2!r}"
            for name, value in defaults.items()
        )


@SKIP_XYCE
def test_xyce_runs_a_fixture_and_scores_it() -> None:
    prepared = suite.Prepared.from_fixture(precision_fixtures.fixtures()["rc_step"])
    setting = suite.Setting("scaled", suite.DEFAULTS["relative_tolerance"] / 100)
    output = prepared.fixture.outputs[0]
    ran = suite.run_xyce(prepared, setting, 2, suite.Xyce())
    assert len(ran.seconds) == 2
    assert ran.counts["accepted"] > 0
    mixed = suite.score(prepared.fixture.compare(ran.found)[output.name], output)[
        "mixed"
    ]
    # Xyce's trapezoidal steps, its default, are at most a fiftieth of the run, as
    # jolty's and ngspice's are: an error of at most about 8e-4 of the 1 V swing.
    assert 3 < mixed.digits < mixed.limit


def test_jolty_s_sweep_is_scored_at_its_levels() -> None:
    prepared = suite.Prepared.from_fixture(precision_fixtures.fixtures()["diode"])
    sweep = prepared.fixture.analysis
    assert isinstance(sweep, DcSweep)
    ran = suite.run_jolty(
        prepared, suite.Setting("defaults", suite.DEFAULTS["relative_tolerance"]), 1
    )
    np.testing.assert_array_equal(
        ran.found.points, [each.value for each in sweep.values]
    )


def test_a_stopped_run_says_each_kind_of_trouble_once_and_how_far_it_came() -> None:
    said = "\n".join(
        [
            "Xyce ran for more than 300 s:",
            *(
                f"Netlist warning in file circuit.cir at or near line {n}"
                for n in (4, 6)
            ),
            " Device MOSFET level 1: Duplicated model and device name instance MX1_M1",
            "***** Percent complete: 5.00002 %",
            "***** Percent complete: 6.00002 %",
        ]
    )
    assert stopped(said) == (
        "Xyce ran for more than 300 s; Netlist warning in file circuit.cir at or "
        "near line 4; 6% of its transient done"
    )


def test_ladder_100_exact_root_pivot_does_not_divide_by_zero() -> None:
    """`exact._Elimination` handles exact 50-digit roots (`pivots[0] == 0`) cleanly."""
    with mp.workdps(exact.DIGITS):
        # 3-node path A = [[2, -1, 0], [-1, 3, -1], [0, -1, 2]] has exact eigenvalue
        # mu = 1 where d_2 = 1, d_1 = 1, and root pivot d_0 = 0 identically.
        diagonal = [mpf(2), mpf(3), mpf(2)]
        off = [mpf(0), mpf(-1), mpf(-1)]
        elimination = exact._Elimination((0, 0, 1), diagonal, off)  # noqa: SLF001 - white-box zero-pivot check
        assert elimination.newton(mpf(1)) == mpf(0)
        eigenvector = elimination.vector(mpf(1))
        expected = mpf(1) / mp.sqrt(3)
        for component in eigenvector:
            # 50-digit mpmath arithmetic precision: 1e-45 bound (< 1e5 * 10^-50).
            assert abs(component - expected) < mpf("1e-45")

    fixture = precision_fixtures.fixtures()["ladder_100"]
    ran = suite.run_jolty(
        suite.Prepared.from_fixture(fixture),
        suite.Setting("defaults", suite.DEFAULTS["relative_tolerance"]),
        1,
    )
    output = fixture.outputs[0]
    compared = fixture.compare(ran.found)[output.name]
    digits = suite.score(compared, output)[output.score].digits
    # Default-tolerance ladder_100 reaches ~2.74 digits vs 50-digit exact reference.
    assert 2.0 < digits < 4.0

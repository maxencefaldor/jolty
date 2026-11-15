# Copyright 2026 Maxence Faldor
"""A batch of runs finds, on each run, what a simulation at its values alone finds.

[`Simulation.batch`][jolty.simulate.Simulation.batch] runs a plan's analyses on every
run of its variants in one call to the engine, across its threads. Each run's
measures, traced unknowns and failures are those of a simulation made for its
values alone, bit for bit, on any number of threads; a run whose values make problems
is reported and the batch goes on; a run whose values give the circuit another
structure measures the unknowns of the names its measures read.

The runs compared with simulations made alone take their numbers as those do, through
the same JSON text, so that both read the same numbers; a Monte Carlo sample is drawn
in the engine and read by no parser, so its draws are compared with
`jolty.variation.sample`'s.
"""

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest

from jolty import (
    Ac,
    Builder,
    DcSweep,
    OperatingPoint,
    Role,
    Simulation,
    Transient,
    circuit,
    component,
    devices,
    ir,
    results,
    simulate,
)
from jolty.errors import ElaborationError
from jolty.plan import (
    Axis,
    Explicit,
    Grid,
    MonteCarlo,
    Product,
    Variant,
    decade,
    runs,
)
from jolty.simulate import Elaborated
from jolty.specs import Measure, MeasureError, Reading, gain, maximum, value
from jolty.units import CURRENT, VOLTAGE, Hz, MHz, V, celsius, kHz, kohm, nF, ohm, us
from jolty.variation import sample, toleranced
from jolty.waveforms import Pulse


@component
def clipper(c: Builder) -> None:
    """A divider whose output a diode clips and a capacitor holds, toleranced parts."""
    ground = c.port("ground", Role.GROUND)
    pulse = devices.voltage_source(
        Pulse(
            initial=0 * V,
            pulsed=2 * V,
            rise=1 * us,
            fall=1 * us,
            width=20 * us,
            period=50 * us,
        )
    )
    source = c.place(pulse.toleranced("waveform.pulse.pulsed", 0.05), id="V1")
    upper = c.place(devices.resistor(1 * kohm, tolerance=0.05), id="R1")
    lower = c.place(devices.resistor(2 * kohm, tolerance=0.05), id="R2")
    diode = c.place(devices.diode(), id="D1")
    held = c.place(devices.capacitor(1 * nF, tolerance=0.1), id="C1")
    c.connect(source.positive, upper.a, name="in")
    c.connect(upper.b, lower.a, diode.anode, held.a, name="out")
    c.connect(lower.b, diode.cathode, held.b, source.negative, ground)


TIMES = [step * 2.5 * us for step in range(1, 17)]
"""The times the transient steps onto, which the batch traces it at."""

BODE = Ac("V1", decade(1 * Hz, 10 * MHz, 4))
STEP = Transient(stop=40 * us, times=TIMES)
ANALYSES = {"op": OperatingPoint(), "bode": BODE, "step": STEP}

MEASUREMENTS = (
    value("op", "out"),
    gain("bode", "out", at=1 * kHz),
    maximum("step", "out"),
)


def _alone(
    built: ir.Circuit, variant: Variant
) -> tuple[
    Simulation, results.OperatingPointResult, results.AcResult, results.TransientResult
]:
    """A simulation made for `variant` alone, and what its analyses find."""
    elaborated = Elaborated(built, variant.temperature)
    simulation = Simulation.of(elaborated, variant.overrides)
    point = simulation.run(OperatingPoint())
    response = simulation.run(BODE)
    waveforms = simulation.run(STEP)
    return simulation, point, response, waveforms


def _bits(array: np.ndarray) -> np.ndarray:
    """The bits of every number of `array`, so that NaNs compare too."""
    return np.ascontiguousarray(array).view(np.uint8)


VARIANTS = Explicit(
    [
        Variant(),
        Variant({"R1": {"resistance": 1500.0}}),
        Variant(temperature=celsius(85)),
        Variant({"R2": {"resistance": 900.0}, "C1": {"capacitance": 2e-9}}),
        Variant({"V1": {"waveform.pulse.pulsed": 1.5}}, celsius(-40)),
    ]
)


def test_a_batch_finds_what_each_run_alone_finds() -> None:
    built = circuit(clipper())
    found = Simulation(built).batch(
        ANALYSES, VARIANTS, measures=MEASUREMENTS, traced=["out", "in"]
    )
    assert found.runs == len(VARIANTS.variants)
    assert found.analyses == ("op", "bode", "step")
    assert not found.failed.any()
    op, bode, step = found.traced
    assert (op.along, bode.along, step.along) == ("point", "frequency", "time")
    for index, variant in enumerate(VARIANTS.variants):
        simulation, point, response, waveforms = _alone(built, variant)
        solved = {"op": point, "bode": response, "step": waveforms}
        expected = [each.read(solved[each.analysis]).value for each in MEASUREMENTS]
        assert found.measured[index].tolist() == expected, f"run {index}"
        assert (found.causes[index] == -1).all()
        assert op.values[index, 0].tolist() == [
            point.voltage("out"),
            point.voltage("in"),
        ]
        assert np.array_equal(bode.values_of("out")[index], response.voltage("out"))
        # Each time reaches the engine as JSON text, which it reads to within an ulp.
        stepped = np.abs(waveforms.times[:, np.newaxis] - step.axis).argmin(axis=0)
        assert waveforms.times[stepped] == pytest.approx(step.axis, rel=2.3e-16)
        assert np.array_equal(
            step.values_of("in")[index], waveforms.voltage("in")[stepped]
        )
        temperature = simulation.conditions.temperature
        assert found.values[index, found.parameters.index("temperature")] == (
            temperature
        )


def test_a_batch_finds_the_same_on_any_number_of_threads() -> None:
    simulation = Simulation(clipper())
    variants = Product(
        Explicit([Variant(), Variant(temperature=celsius(85))]), MonteCarlo(12, seed=5)
    )
    found = []
    for count in (1, 2, 3, 8):
        with simulate.threads(count):
            found.append(
                simulation.batch(
                    ANALYSES, variants, measures=MEASUREMENTS, traced=["out"]
                )
            )
    first, *others = found
    for other in others:
        for array in ("values", "measured", "causes", "ended"):
            assert np.array_equal(
                _bits(getattr(other, array)), _bits(getattr(first, array))
            )
        for mine, theirs in zip(other.traced, first.traced, strict=True):
            assert np.array_equal(_bits(mine.values), _bits(theirs.values))
        assert other.reasons == first.reasons


def _same(found: results.BatchResult, other: results.BatchResult) -> None:
    """Checks that two batch results hold the same bits, and the same of the rest."""
    for array in ("values", "measured", "causes", "ended"):
        mine, theirs = getattr(found, array), getattr(other, array)
        assert mine.shape == theirs.shape, array
        assert np.array_equal(_bits(mine), _bits(theirs)), array
    for mine, theirs in zip(found.traced, other.traced, strict=True):
        assert np.array_equal(_bits(mine.values), _bits(theirs.values))
        assert np.array_equal(_bits(mine.axis), _bits(theirs.axis))
    rest = ("analyses", "parameters", "measures", "units", "reasons")
    for name in (*rest, "problems", "errors"):
        assert getattr(found, name) == getattr(other, name), name


def _peak(result: results.AnalysisResult) -> Reading:
    """The output's highest voltage, unread above 0.656 V, why naming the voltage."""
    if not isinstance(result, results.TransientResult):
        raise TypeError(result)
    volts = float(result.voltage("out").max())
    if volts > 0.656:  # About half the samples' peaks lie above.
        message = f"out peaks at {volts:.6f} V"
        raise MeasureError(message)
    return Reading(volts, VOLTAGE)


def test_ranges_of_a_batch_hold_its_rows_and_join_into_it() -> None:
    simulation = Simulation(clipper())
    variants = Product(
        MonteCarlo(15, seed=5),
        Explicit([Variant(), Variant({"R1": {"resistance": -1.0}})]),
    )
    measures = [
        *MEASUREMENTS,
        Measure("step", "peak of out", None, _peak),
        value("op", "nowhere"),
    ]

    def batch(taken: range | None = None) -> results.BatchResult:
        return simulation.batch(
            ANALYSES, variants, measures=measures, traced=["out"], runs=taken
        )

    whole = batch()
    # Every other run fails, and some of the others read no peak, each for a reason of
    # its own.
    assert whole.failed.tolist() == [False, True] * 15
    assert len(whole.reasons) > 4
    cuts = [0, 4, 11, 15, 16, 30]
    parts = []
    for start, stop in itertools.pairwise(cuts):
        part = batch(range(start, stop))
        # A measure Python reads takes the unit of its first reading: none where
        # a range reads none.
        assert all(
            mine in {None, theirs}
            for mine, theirs in zip(part.units, whole.units, strict=True)
        )
        _same(part, replace(whole.rows(start, stop), units=part.units))
        parts.append(part)
    _same(results.BatchResult.joined(parts), whole)
    _same(results.BatchResult.joined([whole.rows(0, 7), whole.rows(7, 30)]), whole)
    with pytest.raises(ValueError, match="is not a range of the variants' 30 runs"):
        batch(range(20, 31))


def test_a_batch_names_its_reasons_in_the_order_its_runs_first_meet_them() -> None:
    found = results.BatchResult(
        ("op",),
        (),
        np.empty((3, 0)),
        np.ones((3, 1), dtype=np.bool_),
        ("a in op", "b in op"),
        (None, None),
        np.full((3, 2), np.nan),
        np.array([[-1, 2], [0, 2], [3, -1]], dtype=np.int32),
        ("first", "unused", "second", "first"),
    )
    assert found.reasons == ("second", "first")
    assert found.causes.tolist() == [[-1, 0], [1, 0], [1, -1]]
    assert found.rows(2, 3).reasons == ("first",)
    other = results.BatchResult(
        ("op",),
        (),
        np.empty((1, 0)),
        np.ones((1, 1), dtype=np.bool_),
        ("a in op", "b in op"),
        (VOLTAGE, None),
        np.full((1, 2), np.nan),
        np.full((1, 2), -1, dtype=np.int32),
        (),
    )
    with pytest.raises(ValueError, match="read a in op in different units: 'V', 'A'"):
        results.BatchResult.joined([other, replace(other, units=(CURRENT, None))])


def test_a_batch_draws_the_samples_variation_draws() -> None:
    built = circuit(clipper())
    found = Simulation(built).batch({"op": OperatingPoint()}, MonteCarlo(40, seed=9))
    varied = toleranced(built)
    assert found.parameters == tuple(
        f"{parameter.path} {parameter.parameter}" for parameter in varied
    )
    for index in range(40):
        drawn = sample(varied, 9, index)
        assert found.values[index].tolist() == [
            drawn[parameter.path][parameter.parameter] for parameter in varied
        ]


def test_a_grid_steps_its_last_axis_fastest_and_a_product_its_second() -> None:
    simulation = Simulation(clipper())
    grid = Grid(
        [
            Axis.temperature([celsius(0), celsius(50)]),
            Axis("R1", "resistance", [1 * kohm, 2 * kohm, 3 * kohm]),
        ]
    )
    found = simulation.batch({"op": OperatingPoint()}, grid)
    assert found.parameters == ("temperature", "R1 resistance")
    assert found.values.tolist() == [
        [273.15 + celsius, resistance]
        for celsius in (0.0, 50.0)
        for resistance in (1e3, 2e3, 3e3)
    ]
    product = Product(grid, MonteCarlo(4))
    assert runs(product) == 24
    found = simulation.batch({"op": OperatingPoint()}, product)
    assert found.runs == 24
    # Run 9 is the grid's run 2, then the samples' run 1, whose R1 replaces the grid's.
    drawn = sample(toleranced(circuit(clipper())), 0, 1)
    assert found.parameter("R1 resistance")[9] == drawn["R1"]["resistance"]
    assert found.parameter("temperature")[9] == pytest.approx(273.15)


def test_a_run_that_fails_is_reported_and_the_batch_goes_on() -> None:
    built = circuit(clipper())
    variants = Explicit(
        [
            Variant({"R1": {"resistance": -1.0 if index % 3 == 1 else 1e3}})
            for index in range(7)
        ]
    )
    found = Simulation(built).batch(
        ANALYSES, variants, measures=[value("op", "out"), value("op", "nowhere")]
    )
    failed = [index for index in range(7) if index % 3 == 1]
    assert np.nonzero(found.failed)[0].tolist() == failed
    for index in range(7):
        if index in failed:
            assert "resistance must be positive" in (found.failure(index) or "")
            assert found.reason(index, 0) == "the analysis op failed"
            assert math.isnan(found.measured[index, 0])
        else:
            assert found.failure(index) is None
            assert found.reason(index, 0) is None
        assert found.reason(index, 1) == (
            "the analysis op failed"
            if index in failed
            else "the circuit has no unknown named 'nowhere'"
        )
    with pytest.raises(ElaborationError, match="no device at `R9`"):
        Simulation(built).batch(
            ANALYSES, Explicit([Variant({"R9": {"resistance": 1.0}})])
        )


@component
def shunted(c: Builder) -> None:
    """A shunt placed before its source.

    Below 1 Ω, the shunt's current is an unknown, which comes before the source's.
    """
    ground = c.port("ground", Role.GROUND)
    shunt = c.place(devices.resistor(2 * ohm), id="R1")
    source = c.place(devices.voltage_source(5 * V), id="V1")
    load = c.place(devices.resistor(1 * kohm), id="R2")
    c.connect(source.positive, load.a, name="top")
    c.connect(load.b, shunt.a, name="out")
    c.connect(shunt.b, source.negative, ground)


def test_a_run_of_another_structure_measures_its_own_unknowns() -> None:
    built = circuit(shunted())
    variants = Explicit(
        [
            Variant({"R1": {"resistance": 0.5 if index % 2 else 2.0}})
            for index in range(6)
        ]
    )
    seen: dict[int, results.Run] = {}
    found = Simulation(built).batch(
        {"op": OperatingPoint()},
        variants,
        measures=[value("op", "V1:current")],
        each=lambda run: seen.setdefault(run.index, run),
    )
    assert sorted(seen) == list(range(6))
    for index, variant in enumerate(variants.variants):
        alone = Simulation.of(Elaborated(built), variant.overrides).run(
            OperatingPoint()
        )
        assert found.measured[index, 0] == alone.current("V1"), f"run {index}"
        point = seen[index].results["op"]
        assert isinstance(point, results.OperatingPointResult)
        assert point.unknowns == alone.unknowns
        assert point.current("V1") == alone.current("V1")


def test_a_measurement_only_python_reads_reads_each_run() -> None:
    def doubled(result: results.AnalysisResult) -> Reading:
        if not isinstance(result, results.OperatingPointResult):
            raise TypeError(result)
        return Reading(2 * result.voltage("out"), VOLTAGE)

    built = circuit(clipper())
    found = Simulation(built).batch(
        {"op": OperatingPoint()},
        VARIANTS,
        measures=[
            value("op", "out"),
            Measure("op", "twice out", None, doubled),
        ],
    )
    assert found.units == (VOLTAGE, VOLTAGE)
    assert (found.measured[:, 1] == 2 * found.measured[:, 0]).all()


def test_a_sweep_of_a_parameter_names_it() -> None:
    @component
    def divider(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(5 * V), id="V1")
        upper = c.place(devices.resistor(1 * kohm), id="R1")
        lower = c.place(devices.resistor(1 * kohm), id="R2")
        c.connect(source.positive, upper.a)
        c.connect(upper.b, lower.a, name="out")
        c.connect(lower.b, source.negative, ground)

    levels = [1 * kohm, 2 * kohm, 4 * kohm]
    sweep = Simulation(divider()).run(DcSweep("R2", levels, parameter="resistance"))
    assert (sweep.source, sweep.parameter, sweep.quantity) == ("R2", "resistance", None)
    # The divider's output, 5 V times R2 over R1 and R2, within a part in 10¹⁵.
    expected = [5 * lower / (1e3 + lower) for lower in (1e3, 2e3, 4e3)]
    assert sweep.voltage("out") == pytest.approx(expected, rel=1e-15)
    at = value("sweep", "out", at=3 * kohm).read(sweep).value
    assert at == pytest.approx((expected[1] + expected[2]) / 2, rel=1e-15)

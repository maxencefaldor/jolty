# Copyright 2026 Maxence Faldor

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from jolty import (
    Ac,
    Builder,
    DcSweep,
    Noise,
    OperatingPoint,
    OperatingPoints,
    PeriodicSteadyState,
    PoleZero,
    Role,
    Simulation,
    TransferFunction,
    Transient,
    component,
    devices,
    files,
    results,
    service,
    specs,
)
from jolty.errors import JoltyError
from jolty.plan import decade
from jolty.settings import Settings
from jolty.units import Hz, V, kHz, kohm, ms, nF, ohm, uF, uH, us
from jolty.unknowns import Unknown
from jolty.waveforms import Sine

_ADVICE_WORDS = (
    "use ",
    "pass ",
    "try ",
    "instead",
    "should",
    "give ",
    "did you mean",
)


def _assert_factual(message: str) -> None:
    lowered = message.casefold()
    for word in _ADVICE_WORDS:
        assert word not in lowered, f"advice word {word!r} in message: {message}"


@component
def rc_low_pass(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), name="V1")
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def driven_sine_rc(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(
        devices.voltage_source(Sine(offset=0 * V, amplitude=1 * V, frequency=1 * kHz)),
        name="V1",
    )
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    capacitor = c.place(devices.capacitor(1 * uF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def rlc_underdamped(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), name="V1")
    resistor = c.place(devices.resistor(10 * ohm), name="R1")
    inductor = c.place(devices.inductor(10 * uH), name="L1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, inductor.a, name="mid")
    c.connect(inductor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@dataclass(frozen=True, slots=True)
class _Fixture:
    rc_path: Path
    rc_results: dict[str, results.AnalysisResult]
    rlc_path: Path
    rlc_results: dict[str, results.AnalysisResult]
    sine_path: Path
    sine_results: dict[str, results.AnalysisResult]
    loop_path: Path
    loop_results: dict[str, results.AnalysisResult]


def _run_measure(
    request: service.MeasureRequest, settings: Settings | None = None
) -> service.MeasureResult:
    return asyncio.run(service.measure(request, settings or Settings()))


@pytest.fixture
def saved_results(tmp_path: Path) -> _Fixture:
    rc_sim = Simulation(rc_low_pass())
    rc_path = tmp_path / "rc.results.npz"
    files.save_results(
        rc_path,
        {
            "op": rc_sim.run(OperatingPoint()),
            "states": rc_sim.run(OperatingPoints()),
            "sweep": rc_sim.run(DcSweep("V1", [0 * V, 1 * V, 2 * V])),
            "bode": rc_sim.run(
                Ac("V1", decade(10 * Hz, 10 * kHz * 1000, per_decade=20))
            ),
            "step": rc_sim.run(
                Transient(
                    stop=10 * us,
                    max_step=0.05 * us,
                    initial_conditions={"out": 0 * V},
                    operating_point=False,
                )
            ),
            "noise": rc_sim.run(
                Noise(
                    "out", decade(10 * Hz, 1 * kHz * 1000, per_decade=10), source="V1"
                )
            ),
            "tf": rc_sim.run(TransferFunction("V1", "out")),
        },
    )

    rlc_sim = Simulation(rlc_underdamped())
    rlc_path = tmp_path / "rlc.results.npz"
    files.save_results(
        rlc_path,
        {
            "rlc": rlc_sim.run(
                Transient(
                    stop=5 * us,
                    max_step=0.01 * us,
                    initial_conditions={"out": 0 * V, "mid": 0 * V},
                    operating_point=False,
                )
            ),
            "pz": rlc_sim.run(PoleZero(source="V1", output="out")),
        },
    )

    sine_sim = Simulation(driven_sine_rc())
    sine_path = tmp_path / "sine.results.npz"
    files.save_results(
        sine_path,
        {
            "sine": sine_sim.run(Transient(stop=3 * ms, max_step=0.01 * ms)),
            "pss": sine_sim.run(PeriodicSteadyState(period=1 * ms, max_step=0.01 * ms)),
        },
    )

    # Synthetic 3-pole loop gain: T(s) = 100 / (1 + s / (2*pi*1k))^3
    # Crosses 0 dB around 4.54 kHz with positive phase margin and crosses -180 deg
    # with positive gain margin.
    frequencies = np.geomspace(1.0, 1e6, 400)
    jw = 1j * 2.0 * math.pi * frequencies
    loop_gains = np.asarray(
        100.0 / (1.0 + jw / (2.0 * math.pi * 1e3)) ** 3, dtype=np.complex128
    )
    unknowns = (Unknown("out", "voltage"),)
    point = results.OperatingPointResult(unknowns, np.array([0.0]), "direct")
    stability_result = results.StabilityResult(
        unknowns=unknowns,
        probe="VP",
        frequencies=frequencies,
        gains=loop_gains,
        operating_point=point,
    )
    notch_ac = results.AcResult(
        unknowns=unknowns,
        source="V1",
        frequencies=np.array([100.0, 1000.0, 10000.0]),
        values=np.array([[1.0 + 0j], [0.0 + 0j], [1.0 + 0j]], dtype=np.complex128),
        operating_point=point,
    )
    loop_path = tmp_path / "loop.results.npz"
    files.save_results(loop_path, {"loop": stability_result, "notch": notch_ac})

    return _Fixture(
        rc_path=rc_path,
        rc_results=files.load_results(rc_path),
        rlc_path=rlc_path,
        rlc_results=files.load_results(rlc_path),
        sine_path=sine_path,
        sine_results=files.load_results(sine_path),
        loop_path=loop_path,
        loop_results=files.load_results(loop_path),
    )


def test_every_measure_matches_in_memory_specs_to_the_bit(
    saved_results: _Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "analyses.jsonl"
    monkeypatch.setenv("JOLTY_ANALYSIS_LOG", str(log_path))
    settings = Settings.load()
    assert settings.analysis_log == log_path

    rc_path, rc_loaded = saved_results.rc_path, saved_results.rc_results
    rlc_path, rlc_loaded = saved_results.rlc_path, saved_results.rlc_results
    sine_path, sine_loaded = saved_results.sine_path, saved_results.sine_results
    loop_path, loop_loaded = saved_results.loop_path, saved_results.loop_results

    cases: list[
        tuple[service.MeasureRequest, specs.Measure, dict[str, results.AnalysisResult]]
    ] = [
        (
            service.MeasureRequest(
                results=rc_path, analysis="op", measure="value", signal="out"
            ),
            specs.value("op", "out"),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path,
                analysis="op",
                measure="value",
                signal="out",
                reference="in",
            ),
            specs.value("op", "out", reference="in"),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path, analysis="sweep", measure="value", signal="out", at=1.5
            ),
            specs.value("sweep", "out", at=1.5 * V),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path,
                analysis="step",
                measure="value",
                signal="out",
                at=2e-6,
            ),
            specs.value("step", "out", at=2 * us),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path,
                analysis="bode",
                measure="gain",
                signal="out",
                at=100e3,
            ),
            specs.gain("bode", "out", at=100 * kHz),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path,
                analysis="bode",
                measure="gain",
                signal="out",
                over="in",
                at=100e3,
            ),
            specs.gain("bode", "out", at=100 * kHz, over="in"),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path,
                analysis="bode",
                measure="phase",
                signal="out",
                at=100e3,
            ),
            specs.phase("bode", "out", at=100 * kHz),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path,
                analysis="bode",
                measure="bandwidth",
                signal="out",
            ),
            specs.bandwidth("bode", "out"),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path,
                analysis="bode",
                measure="bandwidth",
                signal="out",
                drop=6.0,
            ),
            specs.bandwidth("bode", "out", drop=6.0),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=loop_path, analysis="loop", measure="crossover"
            ),
            specs.crossover("loop"),
            loop_loaded,
        ),
        (
            service.MeasureRequest(
                results=loop_path, analysis="loop", measure="phase_margin"
            ),
            specs.phase_margin("loop"),
            loop_loaded,
        ),
        (
            service.MeasureRequest(
                results=loop_path, analysis="loop", measure="gain_margin"
            ),
            specs.gain_margin("loop"),
            loop_loaded,
        ),
        (
            service.MeasureRequest(
                results=loop_path, analysis="loop", measure="gain", at=1e3
            ),
            specs.gain("loop", at=1 * kHz),
            loop_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path,
                analysis="step",
                measure="maximum",
                signal="out",
                start=1e-6,
                stop=8e-6,
            ),
            specs.maximum("step", "out", start=1 * us, stop=8 * us),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path,
                analysis="step",
                measure="minimum",
                signal="out",
            ),
            specs.minimum("step", "out"),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=sine_path,
                analysis="pss",
                measure="peak_to_peak",
                signal="out",
            ),
            specs.peak_to_peak("pss", "out"),
            sine_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path, analysis="sweep", measure="mean", signal="out"
            ),
            specs.mean("sweep", "out"),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=sine_path, analysis="pss", measure="rms", signal="out"
            ),
            specs.rms("pss", "out"),
            sine_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path,
                analysis="step",
                measure="rise_time",
                signal="out",
                low=0.1,
                high=0.9,
            ),
            specs.rise_time("step", "out", low=0.1, high=0.9),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rlc_path, analysis="rlc", measure="overshoot", signal="out"
            ),
            specs.overshoot("rlc", "out"),
            rlc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rlc_path,
                analysis="rlc",
                measure="settling_time",
                signal="out",
                band=0.05,
            ),
            specs.settling_time("rlc", "out", band=0.05),
            rlc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path,
                analysis="step",
                measure="crossing",
                signal="out",
                level=0.5,
                edge="rising",
            ),
            specs.crossing("step", "out", 0.5 * V, edge="rising"),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=sine_path,
                analysis="sine",
                measure="frequency",
                signal="out",
            ),
            specs.frequency("sine", "out"),
            sine_loaded,
        ),
        (
            service.MeasureRequest(
                results=sine_path,
                analysis="sine",
                measure="frequency",
                signal="out",
                level=0.0,
            ),
            specs.frequency("sine", "out", level=0.0 * V),
            sine_loaded,
        ),
        (
            service.MeasureRequest(
                results=sine_path,
                analysis="sine",
                measure="distortion",
                signal="out",
                fundamental=1e3,
                harmonics=5,
            ),
            specs.distortion("sine", "out", fundamental=1 * kHz, harmonics=5),
            sine_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path, analysis="noise", measure="noise_density", at=1e3
            ),
            specs.noise_density("noise", at=1 * kHz),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path,
                analysis="noise",
                measure="noise_density",
                at=1e3,
                referred=True,
            ),
            specs.noise_density("noise", at=1 * kHz, referred=True),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path,
                analysis="noise",
                measure="integrated_noise",
                start=100.0,
                stop=100e3,
            ),
            specs.integrated_noise("noise", start=100 * Hz, stop=100 * kHz),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path,
                analysis="noise",
                measure="integrated_noise",
                referred=True,
            ),
            specs.integrated_noise("noise", referred=True),
            rc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rc_path, analysis="states", measure="solutions"
            ),
            specs.solutions("states"),
            rc_loaded,
        ),
        (
            service.MeasureRequest(results=rlc_path, analysis="pz", measure="damping"),
            specs.damping("pz"),
            rlc_loaded,
        ),
        (
            service.MeasureRequest(
                results=rlc_path, analysis="pz", measure="dominant_pole"
            ),
            specs.dominant_pole("pz"),
            rlc_loaded,
        ),
    ]

    for request, spec_measure, loaded in cases:
        analysis_name = request.analysis
        assert analysis_name is not None
        expected = spec_measure.read(loaded[analysis_name])
        measured = _run_measure(request, settings)
        assert measured.ok
        assert measured.analysis == analysis_name
        assert measured.measure == request.measure
        # Bit-for-bit equality with in-memory Measure reading.
        assert measured.value == expected.value
        assert math.copysign(1.0, measured.value) == math.copysign(1.0, expected.value)
        assert measured.description == str(spec_measure)
        assert measured.describe() == f"{measured.description}: {measured.formatted}"

    # Periodic steady state `period` measurement.
    pss_period = _run_measure(
        service.MeasureRequest(results=sine_path, analysis="pss", measure="period"),
        settings,
    )
    pss_loaded = sine_loaded["pss"]
    assert isinstance(pss_loaded, results.PeriodicSteadyStateResult)
    assert pss_period.value == pss_loaded.period
    assert pss_period.unit == "s"
    assert pss_period.formatted == "1 ms"
    assert pss_period.describe() == "period in pss: 1 ms"

    # Infinite dB reading serializes as "-Infinity" in JSON.
    notch = _run_measure(
        service.MeasureRequest(
            results=loop_path, analysis="notch", measure="gain", signal="out", at=1000.0
        ),
        settings,
    )
    assert notch.value == -math.inf
    assert notch.unit == "dB"
    dumped = json.loads(notch.model_dump_json())
    assert dumped["value"] == "-Infinity"

    # Zero simulations ran: analysis_log was never created or written.
    assert not log_path.exists()


def test_measure_defaults_to_sole_analysis_in_file(tmp_path: Path) -> None:
    point = results.OperatingPointResult(
        (Unknown("out", "voltage"),), np.array([2.5]), "direct"
    )
    path = tmp_path / "single.results.npz"
    files.save_results(path, {"op": point})

    measured = _run_measure(
        service.MeasureRequest(results=path, measure="value", signal="out")
    )
    assert measured.analysis == "op"
    assert measured.value == 2.5
    assert measured.unit == "V"
    assert measured.formatted == "2.5 V"


def test_measure_factual_errors_for_missing_files_analyses_and_unknowns(
    saved_results: _Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = saved_results.rc_path

    # Missing file.
    with pytest.raises(JoltyError, match="no such results file") as missing_file:
        _run_measure(
            service.MeasureRequest(
                results=tmp_path / "missing.results.npz",
                analysis="op",
                measure="value",
                signal="out",
            )
        )
    _assert_factual(str(missing_file.value))

    # Ambiguous multi-analysis file without --analysis.
    with pytest.raises(JoltyError) as ambiguous:
        _run_measure(
            service.MeasureRequest(results=path, measure="value", signal="out")
        )
    assert "op" in str(ambiguous.value)
    assert "bode" in str(ambiguous.value)
    assert "analysis is unset" in str(ambiguous.value)
    _assert_factual(str(ambiguous.value))

    # Missing named analysis reports available analyses.
    with pytest.raises(JoltyError) as missing_analysis:
        _run_measure(
            service.MeasureRequest(
                results=path, analysis="nope", measure="value", signal="out"
            )
        )
    assert "holds no analysis named 'nope'" in str(missing_analysis.value)
    assert "op" in str(missing_analysis.value)
    assert "bode" in str(missing_analysis.value)
    _assert_factual(str(missing_analysis.value))

    # Missing unknown reports available unknowns (and respects max_unknowns).
    with pytest.raises(JoltyError) as missing_unknown:
        _run_measure(
            service.MeasureRequest(
                results=path, analysis="op", measure="value", signal="bogus"
            ),
            Settings(max_unknowns=2),
        )
    assert "no unknown named 'bogus'" in str(missing_unknown.value)
    assert "unknowns in op:" in str(missing_unknown.value)
    assert "1 more" in str(missing_unknown.value)
    _assert_factual(str(missing_unknown.value))

    # Missing required signal names available unknowns.
    with pytest.raises(JoltyError) as missing_signal:
        _run_measure(
            service.MeasureRequest(results=path, analysis="op", measure="value")
        )
    assert "unknowns in op:" in str(missing_signal.value)
    assert "out" in str(missing_signal.value)
    _assert_factual(str(missing_signal.value))

    # AC response without signal names available unknowns.
    with pytest.raises(JoltyError) as missing_ac_signal:
        _run_measure(
            service.MeasureRequest(
                results=path, analysis="bode", measure="gain", at=1e3
            )
        )
    assert "unknowns in bode:" in str(missing_ac_signal.value)
    assert "out" in str(missing_ac_signal.value)
    _assert_factual(str(missing_ac_signal.value))

    # Empty loaded mapping (defensive path when a results map holds no analyses).
    monkeypatch.setattr(service, "_load", lambda _path: {})
    with pytest.raises(JoltyError, match="holds no analyses") as empty_unset:
        _run_measure(
            service.MeasureRequest(results=path, measure="value", signal="out")
        )
    _assert_factual(str(empty_unset.value))
    with pytest.raises(JoltyError, match="holds no analyses") as empty_named:
        _run_measure(
            service.MeasureRequest(
                results=path, analysis="op", measure="value", signal="out"
            )
        )
    _assert_factual(str(empty_named.value))


def test_measure_factual_errors_for_options_and_result_kinds(
    saved_results: _Fixture,
) -> None:
    rc_path = saved_results.rc_path
    sine_path = saved_results.sine_path

    # Irrelevant options rejected cleanly.
    with pytest.raises(JoltyError, match="measure value takes no drop") as irrelevant:
        _run_measure(
            service.MeasureRequest(
                results=rc_path, analysis="op", measure="value", signal="out", drop=3.0
            )
        )
    _assert_factual(str(irrelevant.value))

    with pytest.raises(
        JoltyError, match="measure value takes no referred"
    ) as bad_referred:
        _run_measure(
            service.MeasureRequest(
                results=rc_path,
                analysis="op",
                measure="value",
                signal="out",
                referred=True,
            )
        )
    _assert_factual(str(bad_referred.value))

    # Missing required scalar options.
    for request, expected in [
        (
            service.MeasureRequest(
                results=rc_path, analysis="bode", measure="gain", signal="out"
            ),
            "takes a frequency as at",
        ),
        (
            service.MeasureRequest(
                results=rc_path, analysis="noise", measure="noise_density"
            ),
            "takes a frequency as at",
        ),
        (
            service.MeasureRequest(
                results=rc_path, analysis="step", measure="crossing", signal="out"
            ),
            "takes a level",
        ),
        (
            service.MeasureRequest(
                results=sine_path, analysis="sine", measure="distortion", signal="out"
            ),
            "takes a fundamental frequency",
        ),
    ]:
        with pytest.raises(JoltyError, match=expected) as err:
            _run_measure(request)
        _assert_factual(str(err.value))

    # `period` on non-PSS result.
    with pytest.raises(
        JoltyError, match="period reads a periodic steady state, not a transient"
    ) as wrong_period:
        _run_measure(
            service.MeasureRequest(results=rc_path, analysis="step", measure="period")
        )
    _assert_factual(str(wrong_period.value))

    # `value` on TransferFunctionResult (verifies _KINDS completeness).
    with pytest.raises(JoltyError, match="a transfer function") as wrong_tf:
        _run_measure(
            service.MeasureRequest(
                results=rc_path, analysis="tf", measure="value", signal="out"
            )
        )
    _assert_factual(str(wrong_tf.value))

    # Invalid numeric parameters convert ValueError into JoltyError.
    with pytest.raises(JoltyError, match="positive") as bad_drop:
        _run_measure(
            service.MeasureRequest(
                results=rc_path,
                analysis="bode",
                measure="bandwidth",
                signal="out",
                drop=-1.0,
            )
        )
    _assert_factual(str(bad_drop.value))

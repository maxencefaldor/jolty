# Copyright 2026 Maxence Faldor

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
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
class _Bundle:
    path: Path
    results: dict[str, results.AnalysisResult]


def _run_measure(
    request: service.MeasureRequest, settings: Settings | None = None
) -> service.MeasureResult:
    return asyncio.run(service.measure(request, settings or Settings()))


def _assert_factual_error(
    request: service.MeasureRequest,
    expected: Sequence[str],
    settings: Settings | None = None,
) -> None:
    with pytest.raises(JoltyError) as raised:
        _run_measure(request, settings)
    message = str(raised.value)
    for fragment in expected:
        assert fragment in message
    _assert_factual(message)


@pytest.fixture(scope="module")
def saved_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, _Bundle]:
    tmp_path = tmp_path_factory.mktemp("service_results")
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

    return {
        "rc": _Bundle(rc_path, files.load_results(rc_path)),
        "rlc": _Bundle(rlc_path, files.load_results(rlc_path)),
        "sine": _Bundle(sine_path, files.load_results(sine_path)),
        "loop": _Bundle(loop_path, files.load_results(loop_path)),
    }


type _MeasureEntry = tuple[str, str, specs.Measure, Mapping[str, object]]
type _ErrorEntry = tuple[str | None, Mapping[str, object], tuple[str, ...]]


def _out(**extra: object) -> dict[str, object]:
    return {"signal": "out", **extra}


_MEASURE_CASES: list[_MeasureEntry] = [
    ("rc", "value", specs.value("op", "out"), _out()),
    ("rc", "value", specs.value("op", "out", reference="in"), _out(reference="in")),
    ("rc", "value", specs.value("sweep", "out", at=1.5 * V), _out(at=1.5)),
    ("rc", "value", specs.value("step", "out", at=2 * us), _out(at=2e-6)),
    ("rc", "gain", specs.gain("bode", "out", at=100 * kHz), _out(at=100e3)),
    (
        "rc",
        "gain",
        specs.gain("bode", "out", at=100 * kHz, over="in"),
        _out(over="in", at=100e3),
    ),
    ("rc", "phase", specs.phase("bode", "out", at=100 * kHz), _out(at=100e3)),
    ("rc", "bandwidth", specs.bandwidth("bode", "out"), _out()),
    ("rc", "bandwidth", specs.bandwidth("bode", "out", drop=6.0), _out(drop=6.0)),
    ("loop", "crossover", specs.crossover("loop"), {}),
    ("loop", "phase_margin", specs.phase_margin("loop"), {}),
    ("loop", "gain_margin", specs.gain_margin("loop"), {}),
    ("loop", "gain", specs.gain("loop", at=1 * kHz), {"at": 1e3}),
    (
        "rc",
        "maximum",
        specs.maximum("step", "out", start=1 * us, stop=8 * us),
        _out(start=1e-6, stop=8e-6),
    ),
    ("rc", "minimum", specs.minimum("step", "out"), _out()),
    ("sine", "peak_to_peak", specs.peak_to_peak("pss", "out"), _out()),
    ("rc", "mean", specs.mean("sweep", "out"), _out()),
    ("sine", "rms", specs.rms("pss", "out"), _out()),
    (
        "rc",
        "rise_time",
        specs.rise_time("step", "out", low=0.1, high=0.9),
        _out(low=0.1, high=0.9),
    ),
    ("rlc", "overshoot", specs.overshoot("rlc", "out"), _out()),
    (
        "rlc",
        "settling_time",
        specs.settling_time("rlc", "out", band=0.05),
        _out(band=0.05),
    ),
    (
        "rc",
        "crossing",
        specs.crossing("step", "out", 0.5 * V, edge="rising"),
        _out(level=0.5, edge="rising"),
    ),
    ("sine", "frequency", specs.frequency("sine", "out"), _out()),
    (
        "sine",
        "frequency",
        specs.frequency("sine", "out", level=0.0 * V),
        _out(level=0.0),
    ),
    (
        "sine",
        "distortion",
        specs.distortion("sine", "out", fundamental=1 * kHz, harmonics=5),
        _out(fundamental=1e3, harmonics=5),
    ),
    ("rc", "noise_density", specs.noise_density("noise", at=1 * kHz), {"at": 1e3}),
    (
        "rc",
        "noise_density",
        specs.noise_density("noise", at=1 * kHz, referred=True),
        {"at": 1e3, "referred": True},
    ),
    (
        "rc",
        "integrated_noise",
        specs.integrated_noise("noise", start=100 * Hz, stop=100 * kHz),
        {"start": 100.0, "stop": 100e3},
    ),
    (
        "rc",
        "integrated_noise",
        specs.integrated_noise("noise", referred=True),
        {"referred": True},
    ),
    ("rc", "solutions", specs.solutions("states"), {}),
    ("rlc", "damping", specs.damping("pz"), {}),
    ("rlc", "dominant_pole", specs.dominant_pole("pz"), {}),
]


def _measure_id(case: _MeasureEntry) -> str:
    _, measure, spec, options = case
    suffix = f"-{'-'.join(options)}" if options else ""
    return f"{spec.analysis}-{measure}{suffix}"


@pytest.mark.parametrize("case", _MEASURE_CASES, ids=_measure_id)
def test_every_measure_matches_in_memory_specs_to_the_bit(
    case: _MeasureEntry,
    saved_results: dict[str, _Bundle],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "analyses.jsonl"
    monkeypatch.setenv("JOLTY_ANALYSIS_LOG", str(log_path))
    settings = Settings.load()
    assert settings.analysis_log == log_path

    bundle_key, measure, spec, options = case
    bundle = saved_results[bundle_key]
    analysis = spec.analysis
    request = service.MeasureRequest.model_validate(
        {"results": bundle.path, "analysis": analysis, "measure": measure, **options}
    )
    expected = spec.read(bundle.results[analysis])
    measured = _run_measure(request, settings)
    assert measured.ok
    assert measured.analysis == analysis
    assert measured.measure == measure
    # Bit-for-bit equality with in-memory Measure reading.
    assert measured.value == expected.value
    assert math.copysign(1.0, measured.value) == math.copysign(1.0, expected.value)
    assert measured.description == str(spec)
    assert measured.describe() == f"{measured.description}: {measured.formatted}"
    # Zero simulations ran: analysis_log was never created or written.
    assert not log_path.exists()


def test_measure_pss_period_and_infinite_db_notch(
    saved_results: dict[str, _Bundle],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "analyses.jsonl"
    monkeypatch.setenv("JOLTY_ANALYSIS_LOG", str(log_path))
    settings = Settings.load()
    assert settings.analysis_log == log_path

    sine_bundle = saved_results["sine"]
    pss_period = _run_measure(
        service.MeasureRequest(
            results=sine_bundle.path, analysis="pss", measure="period"
        ),
        settings,
    )
    pss_loaded = sine_bundle.results["pss"]
    assert isinstance(pss_loaded, results.PeriodicSteadyStateResult)
    assert pss_period.value == pss_loaded.period
    assert pss_period.unit == "s"
    assert pss_period.formatted == "1 ms"
    assert pss_period.describe() == "period in pss: 1 ms"

    # Infinite dB reading serializes as "-Infinity" in JSON.
    notch = _run_measure(
        service.MeasureRequest(
            results=saved_results["loop"].path,
            analysis="notch",
            measure="gain",
            signal="out",
            at=1000.0,
        ),
        settings,
    )
    assert notch.value == -math.inf
    assert notch.unit == "dB"
    dumped = json.loads(notch.model_dump_json())
    assert dumped["value"] == "-Infinity"
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


_ERROR_CASES: dict[str, _ErrorEntry] = {
    "missing-file": (
        None,
        {"analysis": "op", "measure": "value", "signal": "out"},
        ("no such results file",),
    ),
    "ambiguous-analysis": (
        "rc",
        {"measure": "value", "signal": "out"},
        ("op", "bode", "analysis is unset"),
    ),
    "missing-analysis": (
        "rc",
        {"analysis": "nope", "measure": "value", "signal": "out"},
        ("holds no analysis named 'nope'", "op", "bode"),
    ),
    "missing-unknown-truncated": (
        "rc",
        {"analysis": "op", "measure": "value", "signal": "bogus"},
        ("no unknown named 'bogus'", "unknowns in op:", "1 more"),
    ),
    "missing-op-signal": (
        "rc",
        {"analysis": "op", "measure": "value"},
        ("unknowns in op:", "out"),
    ),
    "missing-ac-signal": (
        "rc",
        {"analysis": "bode", "measure": "gain", "at": 1e3},
        ("unknowns in bode:", "out"),
    ),
    "irrelevant-drop": (
        "rc",
        {"analysis": "op", "measure": "value", "signal": "out", "drop": 3.0},
        ("measure value takes no drop",),
    ),
    "irrelevant-referred": (
        "rc",
        {"analysis": "op", "measure": "value", "signal": "out", "referred": True},
        ("measure value takes no referred",),
    ),
    "missing-at-gain": (
        "rc",
        {"analysis": "bode", "measure": "gain", "signal": "out"},
        ("takes a frequency as at",),
    ),
    "missing-at-noise-density": (
        "rc",
        {"analysis": "noise", "measure": "noise_density"},
        ("takes a frequency as at",),
    ),
    "missing-level-crossing": (
        "rc",
        {"analysis": "step", "measure": "crossing", "signal": "out"},
        ("takes a level",),
    ),
    "missing-fundamental-distortion": (
        "sine",
        {"analysis": "sine", "measure": "distortion", "signal": "out"},
        ("takes a fundamental frequency",),
    ),
    "period-on-transient": (
        "rc",
        {"analysis": "step", "measure": "period"},
        ("period reads a periodic steady state, not a transient",),
    ),
    "value-on-transfer-function": (
        "rc",
        {"analysis": "tf", "measure": "value", "signal": "out"},
        ("a transfer function",),
    ),
    "negative-drop": (
        "rc",
        {"analysis": "bode", "measure": "bandwidth", "signal": "out", "drop": -1.0},
        ("positive",),
    ),
}


@pytest.mark.parametrize("name", list(_ERROR_CASES))
def test_measure_factual_errors(
    name: str,
    saved_results: dict[str, _Bundle],
    tmp_path: Path,
) -> None:
    bundle, fields, expected = _ERROR_CASES[name]
    path = (
        tmp_path / "missing.results.npz"
        if bundle is None
        else saved_results[bundle].path
    )
    settings = Settings(max_unknowns=2) if name == "missing-unknown-truncated" else None
    request = service.MeasureRequest.model_validate({"results": path, **fields})
    _assert_factual_error(request, expected, settings)


@pytest.mark.parametrize("analysis", [None, "op"], ids=["unset", "named"])
def test_measure_factual_errors_for_empty_results_file(
    analysis: str | None,
    saved_results: dict[str, _Bundle],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "_load", lambda _path: {})
    request = service.MeasureRequest(
        results=saved_results["rc"].path,
        analysis=analysis,
        measure="value",
        signal="out",
    )
    _assert_factual_error(request, ("holds no analyses",))

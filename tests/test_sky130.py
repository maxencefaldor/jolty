# Copyright 2026 Maxence Faldor
"""SKY130's transistors, which ngspice simulates with the kit's models.

The tests that run ngspice fetch jolty's pinned release of the kit into pytest's cache
the first time, and are skipped when ngspice is not installed.
"""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from differential import SKIP
from jolty import (
    Builder,
    Role,
    Simulation,
    SimulationError,
    _core,
    circuit,
    component,
    devices,
    results,
)
from jolty.cli import main
from jolty.errors import DimensionError, JoltyError
from jolty.plan import (
    Ac,
    DcSweep,
    Noise,
    OperatingPoint,
    Sensitivity,
    Transient,
    linear,
)
from jolty.settings import Settings
from jolty.spice import export, sky130
from jolty.units import Hz, MHz, V, celsius, kHz, kohm, ns, um
from jolty.waveforms import Pulse


@pytest.fixture(scope="session")
def kit(pytestconfig: pytest.Config) -> Settings:
    """Settings whose cache, pytest's, holds the kit's release."""
    assert pytestconfig.cache is not None
    settings = Settings(cache=pytestconfig.cache.mkdir("downloads"))
    sky130.root(settings)
    return settings


@pytest.fixture
def cached(kit: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Points jolty's cache at the one that holds the kit's release."""
    monkeypatch.setenv("JOLTY_CACHE", str(kit.cache))


def test_a_transistor_names_the_kits_model_and_its_size() -> None:
    placed = sky130.mosfet("pfet_01v8_hvt", 2 * um, 0.5 * um, fingers=2, multiplier=3)
    assert placed.model == "foundry_mosfet"
    assert placed.terminals == ("drain", "gate", "source", "bulk")
    assert placed.parameters == {
        "pdk": "sky130",
        "model": "sky130_fd_pr__pfet_01v8_hvt",
        "width": 2e-6,
        "length": 5e-7,
        "fingers": 2,
        "multiplier": 3,
    }
    with pytest.raises(ValueError, match="SKY130 has no transistor 'nfet_01v9'"):
        sky130.mosfet("nfet_01v9", 1 * um, 1 * um)
    with pytest.raises(ValueError, match="width is positive, not -1e-06 m"):
        sky130.mosfet("nfet_01v8", -1 * um, 1 * um)
    with pytest.raises(ValueError, match="fingers is 1 or more, not 0"):
        sky130.mosfet("nfet_01v8", 1 * um, 1 * um, fingers=0)
    with pytest.raises(DimensionError):
        sky130.mosfet("nfet_01v8", 1 * V, 1 * um)


def test_a_netlist_estimates_the_diffusions_as_the_kits_symbols_do() -> None:
    # Two fingers of 1 µm share a drain between two sources: AD = 1 µm * 0.29 µm, AS
    # twice that, PD = 2 * (1 + 0.29) µm, PS twice that, NRD = NRS = 0.29 / 2.
    written = sky130.written(
        sky130.mosfet("nfet_01v8", 2 * um, 0.15 * um, fingers=2).parameters
    )
    assert written == (
        "sky130_fd_pr__nfet_01v8 L=0.15 W=2.0 nf=2 ad=0.29 as=0.58 pd=2.58 ps=5.16 "
        "nrd=0.145 nrs=0.145 sa=0 sb=0 sd=0 mult=1 m=1"
    )
    with pytest.raises(ValueError, match="not a SKY130 transistor's parameters"):
        sky130.written({"pdk": "gf180", "model": "nfet_03v3"})


@component
def inverter(c: Builder) -> None:
    """A SKY130 inverter, its p-channel transistor of two fingers, driven at `in`."""
    ground = c.port("0", Role.GROUND)
    supply = c.place(devices.voltage_source(1.8 * V), id="VDD")
    drive = c.place(
        devices.voltage_source(
            Pulse(
                0 * V, 1.8 * V, delay=1 * ns, rise=0.1 * ns, fall=0.1 * ns, width=5 * ns
            )
        ),
        id="VIN",
    )
    lower = c.place(sky130.mosfet("nfet_01v8", 1 * um, 0.15 * um), id="MN")
    upper = c.place(sky130.mosfet("pfet_01v8", 2 * um, 0.15 * um, fingers=2), id="MP")
    c.connect(supply.positive, upper.source, upper.bulk, name="vdd")
    c.connect(drive.positive, lower.gate, upper.gate, name="in")
    c.connect(lower.drain, upper.drain, name="out")
    c.connect(supply.negative, drive.negative, lower.source, lower.bulk, ground)


def test_the_engine_checks_a_foundrys_devices_and_leaves_them_to_ngspice() -> None:
    simulation = Simulation(inverter())
    assert simulation.foundry == ("MN", "MP")
    assert [unknown.name for unknown in simulation.unknowns] == [
        "vdd",
        "in",
        "out",
        "VDD:current",
        "VIN:current",
    ]
    assert simulation.check() == ()
    with pytest.raises(SimulationError, match="ngspice simulates with its foundry's"):
        simulation.check_operating_point()
    with pytest.raises(ValueError, match="the circuit holds none"):
        Simulation(divider(), process="ss")


@component
def divider(c: Builder) -> None:
    ground = c.port("0", Role.GROUND)
    supply = c.place(devices.voltage_source(1 * V), id="V1")
    upper = c.place(devices.resistor(1 * kohm), id="R1")
    c.connect(supply.positive, upper.a)
    c.connect(supply.negative, upper.b, ground)


@component
def mixed(c: Builder) -> None:
    """A SKY130 transistor beside a level-1 one."""
    ground = c.port("0", Role.GROUND)
    foundry = c.place(sky130.mosfet("nfet_01v8", 1 * um, 0.15 * um), id="M1")
    native = c.place(devices.mosfet(), id="M2")
    for transistor in (foundry, native):
        c.connect(transistor.drain, transistor.gate, transistor.source, ground)
        c.connect(transistor.bulk, ground)


def test_a_netlist_keeps_a_kits_scale_from_a_native_mosfet() -> None:
    with pytest.raises(JoltyError, match="whose kit scales the sizes of every MOSFET"):
        export.netlist(circuit(mixed()))
    held = sky130.mosfet("nfet_01v8", 1 * um, 0.15 * um).held_at(celsius(50))

    @component
    def warm(c: Builder) -> None:
        ground = c.port("0", Role.GROUND)
        transistor = c.place(held, id="M1")
        c.connect(transistor.drain, transistor.gate, transistor.source, ground)
        c.connect(transistor.bulk, ground)

    with pytest.raises(JoltyError, match="at its own temperature, which ngspice's"):
        export.netlist(circuit(warm()))


def test_a_pdk_root_holds_the_kits_library(tmp_path: Path) -> None:
    with pytest.raises(JoltyError, match=r"holds no sky130A/libs\.tech/ngspice"):
        sky130.root(Settings(pdk_root=tmp_path))
    library = tmp_path / sky130.LIBRARY
    library.parent.mkdir(parents=True)
    library.write_text("* the library\n", encoding="utf-8")
    assert sky130.root(Settings(pdk_root=tmp_path)) == tmp_path


@SKIP
@pytest.mark.usefixtures("cached")
def test_an_inverter_switches_in_ngspice() -> None:
    simulation = Simulation(inverter())
    out = simulation.unknowns[2]
    assert out.name == "out"
    point = simulation.run(OperatingPoint())
    assert point.strategy == "ngspice"
    # The input pulse starts low, so the output starts high.
    assert point.voltage("out") == pytest.approx(1.8, abs=1e-6)
    sweep = simulation.run(DcSweep("VIN", linear(0 * V, 1.8 * V, 37)))
    transfer = sweep.values[:, results.column(sweep.unknowns, "out")]
    assert np.all(np.diff(transfer) <= 1e-9)
    assert transfer[0] == pytest.approx(1.8, abs=1e-3)
    assert transfer[-1] == pytest.approx(0, abs=1e-3)
    threshold = np.interp(0.9, transfer[::-1], sweep.levels[::-1])
    assert 0.6 < threshold < 1.2
    transient = simulation.run(Transient(stop=10 * ns, max_step=0.05 * ns))
    assert transient.statistics.accepted == len(transient.times)
    output = transient.values[:, results.column(transient.unknowns, "out")]
    assert output[0] == pytest.approx(1.8, abs=1e-3)
    during = (transient.times > 3e-9) & (transient.times < 6e-9)
    assert np.all(output[during] < 0.05)
    assert output[-1] == pytest.approx(1.8, abs=1e-2)
    with pytest.raises(SimulationError, match=r"runs no Sensitivity analysis"):
        simulation.run(Sensitivity(["out"]))
    with pytest.raises(SimulationError, match="in even steps, and its levels are not"):
        simulation.run(DcSweep("VIN", [0 * V, 0.1 * V, 0.3 * V]))


@component
def amplifier(c: Builder) -> None:
    """A common-source amplifier, its load a resistor."""
    ground = c.port("0", Role.GROUND)
    supply = c.place(devices.voltage_source(1.8 * V), id="VDD")
    bias = c.place(devices.voltage_source(0.7 * V), id="VIN")
    load = c.place(devices.resistor(10 * kohm), id="RL")
    transistor = c.place(sky130.mosfet("nfet_01v8", 2 * um, 0.5 * um), id="M1")
    c.connect(supply.positive, load.a, name="vdd")
    c.connect(bias.positive, transistor.gate, name="in")
    c.connect(load.b, transistor.drain, name="out")
    c.connect(
        supply.negative, bias.negative, transistor.source, transistor.bulk, ground
    )


@SKIP
@pytest.mark.usefixtures("cached")
def test_the_small_signal_gain_is_the_transfer_curves_slope() -> None:
    simulation = Simulation(amplifier())
    step = 1e-3
    sweep = simulation.run(
        DcSweep("VIN", linear((0.7 - step) * V, (0.7 + step) * V, 3))
    )
    out = sweep.values[:, results.column(sweep.unknowns, "out")]
    slope = (out[-1] - out[0]) / (2 * step)
    response = simulation.run(Ac("VIN", [1 * kHz]))
    gain = response.values[0, results.column(response.unknowns, "out")]
    assert slope < -0.5
    assert gain.real == pytest.approx(slope, rel=1e-4)
    assert abs(gain.imag) < 1e-6 * abs(gain.real)
    assert response.operating_point.voltage("out") == pytest.approx(out[1], abs=1e-9)


@SKIP
@pytest.mark.usefixtures("cached")
def test_corners_temperatures_and_overrides_reach_ngspice() -> None:
    def current(simulation: Simulation) -> float:
        return simulation.run(OperatingPoint()).current("VDD")

    built = amplifier()
    typical = current(Simulation(built))
    fast, slow = (current(Simulation(built, process=corner)) for corner in ("ff", "ss"))
    assert fast < typical < slow < 0
    hot = current(Simulation(built, celsius(125)))
    assert hot != pytest.approx(typical, rel=1e-3)
    loaded = current(Simulation(built, overrides={"RL": {"resistance": 20e3}}))
    assert abs(loaded) < abs(typical)
    with pytest.raises(SimulationError, match="SKY130 has no process corner 'xx'"):
        Simulation(amplifier(), process="xx").run(OperatingPoint())


@SKIP
@pytest.mark.usefixtures("cached")
def test_a_plan_runs_at_the_kits_corners(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = tmp_path / "amplifier.py"
    script.write_text(
        textwrap.dedent(
            """
            from jolty import (
                Builder, Corner, MonteCarlo, Plan, Role, component, devices,
            )
            from jolty.plan import OperatingPoint
            from jolty.spice import sky130
            from jolty.units import V, kohm, um


            @component
            def amplifier(c: Builder) -> None:
                ground = c.port("0", Role.GROUND)
                supply = c.place(devices.voltage_source(1.8 * V), id="VDD")
                bias = c.place(devices.voltage_source(0.7 * V), id="VIN")
                load = c.place(devices.resistor(10 * kohm), id="RL")
                transistor = c.place(
                    sky130.mosfet("nfet_01v8", 2 * um, 0.5 * um), id="M1"
                )
                c.connect(supply.positive, load.a, name="vdd")
                c.connect(bias.positive, transistor.gate, name="in")
                c.connect(load.b, transistor.drain, name="out")
                c.connect(
                    supply.negative, bias.negative, transistor.source,
                    transistor.bulk, ground,
                )


            def build():
                return amplifier()


            def plan():
                return Plan(
                    {"bias": OperatingPoint()},
                    process="tt",
                    corners={"slow": Corner(process="ss")},
                    monte_carlo=MonteCarlo(2),
                )
            """
        ),
        encoding="utf-8",
    )
    assert _run("simulate", str(script)) == 0
    text = capsys.readouterr().out
    assert "bias: operating point, found by ngspice" in text
    assert "corner slow, at 27 °C, process ss: every analysis ended" in text
    assert (
        "monte carlo: 2 samples of 0 toleranced parameters and the mismatch of 1 "
        "foundry's devices, at slow, seed 0; 0 failed"
    ) in text


def _run(*arguments: str) -> int:
    try:
        main(arguments)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else 0
    return 0


@component
def holder(c: Builder) -> None:
    """The amplifier within a component instance."""
    ground = c.port("0", Role.GROUND)
    placed = c.place(amplifier(), id="X1")
    c.connect(placed["0"], ground)


@SKIP
@pytest.mark.usefixtures("cached")
def test_a_nested_source_is_swept_and_excited_with_overrides() -> None:
    simulation = Simulation(holder(), overrides={"X1/RL": {"resistance": 20e3}})
    sweep = simulation.run(DcSweep("X1/VIN", linear(0.6 * V, 0.8 * V, 3)))
    assert sweep.source == "X1/VIN"
    out = sweep.values[:, results.column(sweep.unknowns, "X1/out")]
    response = simulation.run(Ac("X1/VIN", [1 * kHz]))
    gain = response.values[0, results.column(response.unknowns, "X1/out")]
    assert gain.real < 0
    assert response.operating_point.voltage("X1/out") == pytest.approx(out[1])
    loaded = response.operating_point.current("X1/VDD")
    alone = Simulation(holder()).run(DcSweep("X1/VIN", linear(0.6 * V, 0.8 * V, 3)))
    assert abs(loaded) < abs(
        alone.values[1, results.column(alone.unknowns, "X1/VDD:current")]
    )
    with pytest.raises(SimulationError, match="`X1/RL` is not an independent source"):
        simulation.run(Ac("X1/RL", [1 * kHz]))


@component
def quiet(c: Builder) -> None:
    """A divider beside a transistor that its noise does not reach."""
    ground = c.port("0", Role.GROUND)
    supply = c.place(devices.voltage_source(1 * V), id="V1")
    upper = c.place(devices.resistor(1 * kohm), id="R1")
    lower = c.place(devices.resistor(2 * kohm), id="R2")
    rail = c.place(devices.voltage_source(1.8 * V), id="VDD")
    transistor = c.place(sky130.mosfet("nfet_01v8", 1 * um, 0.15 * um), id="M1")
    c.connect(supply.positive, upper.a, name="in")
    c.connect(upper.b, lower.a, name="out")
    c.connect(rail.positive, transistor.drain, transistor.gate, name="vdd")
    c.connect(
        supply.negative,
        lower.b,
        rail.negative,
        transistor.source,
        transistor.bulk,
        ground,
    )


@SKIP
@pytest.mark.usefixtures("cached")
def test_the_noise_of_a_divider_is_its_resistances_in_parallel() -> None:
    # Each resistor's 4kTR reaches the output divided by the other's share: R1's by
    # 2/3, R2's by 1/3, and their sum is 4kT times 1 kΩ in parallel with 2 kΩ.
    spectrum = Simulation(quiet()).run(Noise("out", [1 * kHz, 1 * MHz], source="V1"))
    four_kt = 4 * _core.BOLTZMANN * _core.REFERENCE_TEMPERATURE
    expected = four_kt * 2e3 / 3
    np.testing.assert_allclose(spectrum.densities, expected, rtol=1e-4)
    assert spectrum.sources == (
        results.NoiseSource("R1", "thermal"),
        results.NoiseSource("R2", "thermal"),
    )
    np.testing.assert_allclose(
        spectrum.contributions,
        [[four_kt * 1e3 * 4 / 9] * 2, [four_kt * 2e3 / 9] * 2],
        rtol=1e-4,
    )
    assert spectrum.input is not None
    np.testing.assert_allclose(spectrum.input.gains, 2 / 3, rtol=1e-9)
    np.testing.assert_allclose(spectrum.referred(), expected * 9 / 4, rtol=1e-4)


@SKIP
@pytest.mark.usefixtures("cached")
def test_a_transistors_noise_adds_its_channels_and_its_flicker() -> None:
    simulation = Simulation(amplifier())
    spectrum = simulation.run(Noise("out", [10 * Hz, 1 * MHz]))
    named = {(source.instance, source.name) for source in spectrum.sources}
    assert {("M1", "id"), ("M1", "1overf"), ("RL", "thermal")} <= named
    assert spectrum.input is None
    np.testing.assert_allclose(
        spectrum.contributions.sum(axis=0), spectrum.densities, rtol=1e-6
    )
    flicker = spectrum.contribution("M1", "1overf")
    assert flicker[0] > 100 * flicker[1]
    with pytest.raises(SimulationError, match="takes the noise at a voltage"):
        simulation.run(Noise("VDD:current", [1 * kHz]))


@SKIP
@pytest.mark.usefixtures("cached")
def test_a_seed_draws_the_kits_mismatch() -> None:
    def current(mismatch: int | None) -> float:
        simulation = Simulation(amplifier(), mismatch=mismatch)
        return simulation.run(OperatingPoint()).current("VDD")

    first, again, other, nominal = current(1), current(1), current(2), current(None)
    assert first == again
    assert first != pytest.approx(other, rel=1e-6)
    assert first != pytest.approx(nominal, rel=1e-6)
    assert Simulation(divider(), mismatch=1).conditions.mismatch == 1

# Copyright 2026 Maxence Faldor
"""Differential tests on vendors' macromodels, read from their SPICE files.

Each test bench places a part read by jolty's SPICE importer, and ngspice simulates the
same bench with the same file included: in its PSpice compatibility mode for the files
written for PSpice, whose syntax its default mode does not read. The files are ngspice's
own examples, downloaded from a mirror of its repository at a fixed commit and checked
against their digests.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from differential import (
    SKIP,
    TIGHT,
    closest,
    compare,
    compared,
    netlist,
    ngspice,
    size,
    vector,
)
from jolty import (
    Ac,
    Builder,
    Component,
    DcSweep,
    OperatingPoint,
    Options,
    Role,
    Simulation,
    Transient,
    devices,
    spice,
)
from jolty.plan import decade, linear
from jolty.spice.ngspice import OPTIONS, VOLTAGE_TOLERANCE, Compatibility
from jolty.units import Hz, MHz, Quantity, V, celsius, kHz, kohm, ms, us
from jolty.waveforms import Sine

pytestmark = SKIP

_MIRROR = (
    "https://raw.githubusercontent.com/imr/ngspice/"
    "a80f6e3e95d51534905b1f23410a951802666656/examples/"
)
"""ngspice's examples, at a fixed commit of a mirror of its repository."""


@dataclass(frozen=True)
class Part:
    """A vendor's macromodel: where its file is, its digest, and its subcircuit."""

    path: str
    sha256: str
    subcircuit: str


TL072 = Part(
    "probe/TL072.301",
    "74e89d558163615ac7a19f0c783101a6f8af77fb6d80bd3c20c6ab66426561cd",
    "TL072",
)
MCP6041 = Part(
    "p-to-n-examples/MCP6041.txt",
    "056c6a2bfd39f2258b652d85bb892d537d6cb7c2132374bce38bcef82a3d8d67",
    "MCP6041",
)
AD22057 = Part(
    "p-to-n-examples/ad22057n.lib",
    "f3dd259368087bb20a5510748a35d1b70d852088fc53d703641894fb757e685b",
    "AD22057N",
)
OPA171 = Part(
    "p-to-n-examples/OPA171.txt",
    "a9d1083026345165b77dad0c919d261568415ff06eaa03265aed983efdfaf5ba",
    "OPA171",
)


@pytest.fixture(scope="session")
def downloads(pytestconfig: pytest.Config) -> Path:
    """Where the parts' files are kept between runs."""
    assert pytestconfig.cache is not None
    return pytestconfig.cache.mkdir("downloads")


def _file(part: Part, downloads: Path) -> Path:
    return spice.fetch(_MIRROR + part.path, part.sha256, downloads)


def _inverting(
    part: Component,
    supplies: tuple[float, float],
    pins: tuple[str, str, str, str, str],
    signal: Sine,
) -> Component:
    """An amplifier of gain -2 around `part`, its non-inverting input at mid-supply.

    The part's `pins` are its non-inverting and inverting inputs, its supplies and its
    output.
    """
    plus, minus, positive, negative, output = pins
    c = Builder("inverting")
    ground = c.port("ground", Role.GROUND)
    upper = c.place(devices.voltage_source(supplies[0] * V), id="VP")
    lower = c.place(devices.voltage_source(supplies[1] * V), id="VN")
    middle = c.place(devices.voltage_source(sum(supplies) / 2 * V), id="VREF")
    source = c.place(devices.voltage_source(signal), id="V1")
    feed = c.place(devices.resistor(10 * kohm), id="RIN")
    feedback = c.place(devices.resistor(20 * kohm), id="RF")
    load = c.place(devices.resistor(100 * kohm), id="RL")
    amplifier = c.place(part, id="U1")
    c.connect(upper.positive, amplifier[positive], name="supply")
    c.connect(lower.positive, amplifier[negative], name="return")
    c.connect(middle.positive, amplifier[plus], load.b, name="reference")
    c.connect(source.positive, feed.a, name="in")
    c.connect(feed.b, feedback.a, amplifier[minus], name="inverting")
    c.connect(feedback.b, load.a, amplifier[output], name="out")
    c.connect(
        upper.negative,
        lower.negative,
        middle.negative,
        source.negative,
        amplifier[spice.GROUND],
        ground,
    )
    return c.finish()


def _tl072(downloads: Path) -> tuple[Component, Path]:
    file = _file(TL072, downloads)
    part = spice.load(file).component(TL072.subcircuit)
    signal = Sine(
        offset=0.5 * V, amplitude=2 * V, frequency=10 * kHz, phase=math.pi / 2
    )
    return _inverting(part, (15, -15), ("1", "2", "3", "4", "5"), signal), file


def _mcp6041(downloads: Path) -> tuple[Component, Path]:
    file = _file(MCP6041, downloads)
    part = spice.load(file).component(MCP6041.subcircuit)
    signal = Sine(
        offset=2.6 * V, amplitude=0.5 * V, frequency=500 * Hz, phase=math.pi / 2
    )
    return _inverting(part, (5, 0), ("1", "2", "3", "4", "5"), signal), file


def _opa171(downloads: Path) -> tuple[Component, Path]:
    file = _file(OPA171, downloads)
    part = spice.load(file).component(OPA171.subcircuit)
    signal = Sine(
        offset=0.5 * V, amplitude=1 * V, frequency=10 * kHz, phase=math.pi / 2
    )
    pins = ("IN+", "IN-", "VCC", "VEE", "OUT")
    return _inverting(part, (15, -15), pins, signal), file


def _ad22057(downloads: Path) -> tuple[Component, Path]:
    """The difference amplifier with its two stages joined, from a single supply."""
    file = _file(AD22057, downloads)
    part = spice.load(file).component(AD22057.subcircuit)
    c = Builder("difference")
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), id="VCC")
    reference = c.place(devices.voltage_source(0.1 * V), id="V2")
    signal = Sine(
        offset=0.12 * V, amplitude=0.05 * V, frequency=1 * kHz, phase=math.pi / 2
    )
    source = c.place(devices.voltage_source(signal), id="V1")
    load = c.place(devices.resistor(10 * kohm), id="RL")
    offset = c.place(devices.resistor(100 * kohm), id="ROFF")
    amplifier = c.place(part, id="U1")
    c.connect(supply.positive, amplifier["99"], name="supply")
    c.connect(source.positive, amplifier["1"], name="plus")
    c.connect(reference.positive, amplifier["2"], name="minus")
    c.connect(amplifier["30"], amplifier["31"], name="between")
    c.connect(amplifier["40"], offset.a, name="offset")
    c.connect(amplifier["49"], load.a, name="out")
    c.connect(
        supply.negative,
        reference.negative,
        source.negative,
        load.b,
        offset.b,
        amplifier["50"],
        amplifier[spice.GROUND],
        ground,
    )
    return c.finish(), file


@dataclass(frozen=True)
class Bench:
    """A test bench, the analyses both engines run on it, and their options.

    Both run its operating point, a sweep of its input `V1` from the first to the last
    of `sweep`'s levels, its frequency response if `ac`, and a `transient` of a duration
    with a maximal step.
    """

    build: Callable[[Path], tuple[Component, Path]]
    sweep: tuple[float, float, int] | None
    transient: tuple[Quantity, Quantity] | None
    ac: bool = True
    options: Options = TIGHT
    spice_options: str = OPTIONS
    compatibility: Compatibility | None = "ps"

    def runs(self, analysis: str) -> bool:
        """Whether both engines run `analysis`, ``op``, ``dc``, ``ac`` or ``tran``."""
        match analysis:
            case "dc":
                return self.sweep is not None
            case "ac":
                return self.ac
            case "tran":
                return self.transient is not None
        return True


BENCHES = {
    # SPICE2's syntax, which ngspice reads without PSpice's mode, in which it would give
    # each diode without a series resistance one of 100 µΩ.
    "tl072": Bench(_tl072, (-10, 10, 81), (300 * us, 0.2 * us), compatibility=None),
    "mcp6041": Bench(_mcp6041, (1.5, 3.5, 81), (6 * ms, 2 * us)),
    # The amplifier's gains of a million leave its currents' last digits unresolved
    # below the default absolute tolerance, in both engines.
    "ad22057": Bench(
        _ad22057,
        (0.0, 0.3, 61),
        (3 * ms, 1 * us),
        options=replace(TIGHT, current_tolerance=1e-12),
        spice_options=(
            f".options reltol=1e-6 vntol={VOLTAGE_TOLERANCE!r} abstol=1e-12"
            " chgtol=1e-18"
        ),
    ),
    # ngspice finds this amplifier's operating point at its default tolerances only. It
    # runs the amplifier's switches as XSPICE's, which stamp the negative of their
    # conductance in an AC analysis.
    "opa171": Bench(_opa171, None, None, ac=False, spice_options=""),
}
"""The benches, by name."""


def _benches(analysis: str) -> list[str]:
    """The benches whose `analysis` both engines run."""
    return [name for name, bench in BENCHES.items() if bench.runs(analysis)]


@pytest.mark.parametrize("name", _benches("op"))
def test_operating_points_agree(name: str, downloads: Path) -> None:
    bench = BENCHES[name]
    top, file = bench.build(downloads)
    simulation = Simulation(top)
    point = simulation.run(OperatingPoint(), bench.options)
    raw = ngspice.run(
        netlist(simulation.circuit),
        "op",
        libraries=(file,),
        compatibility=bench.compatibility,
        options=bench.spice_options,
    )
    for column, unknown in compared(simulation):
        spice_value = raw.real(vector(simulation.circuit, unknown))[0]
        compare(
            point.values[column],
            spice_value,
            unknown,
            scale=abs(spice_value),
            tolerance=1e-6,
        )


_TEMPERATURES = [
    pytest.param(name, degrees, id=f"{name}-{degrees}C")
    for name in _benches("op")
    for degrees in (-40, 125)
    # ngspice scales a transistor's base-collector saturation current by its area
    # twice, which the AD22057's leakage at 125 °C makes visible.
    if (name, degrees) != ("ad22057", 125)
]
"""The benches, and the temperatures in degrees Celsius both also run them at."""


@pytest.mark.parametrize(("name", "degrees"), _TEMPERATURES)
def test_operating_points_agree_across_temperatures(
    name: str, degrees: float, downloads: Path
) -> None:
    bench = BENCHES[name]
    top, file = bench.build(downloads)
    temperature = celsius(degrees)
    simulation = Simulation(top, temperature)
    point = simulation.run(OperatingPoint(), bench.options)
    raw = ngspice.run(
        netlist(simulation.circuit),
        "op",
        libraries=(file,),
        compatibility=bench.compatibility,
        options=bench.spice_options,
        temperature=temperature.value,
    )
    for column, unknown in compared(simulation):
        spice_value = raw.real(vector(simulation.circuit, unknown))[0]
        compare(
            point.values[column],
            spice_value,
            unknown,
            scale=abs(spice_value),
            tolerance=1e-6,
        )


@pytest.mark.parametrize("name", _benches("dc"))
def test_dc_sweeps_agree(name: str, downloads: Path) -> None:
    bench = BENCHES[name]
    top, file = bench.build(downloads)
    simulation = Simulation(top)
    assert bench.sweep is not None
    first, last, count = bench.sweep
    sweep = simulation.run(
        DcSweep("V1", linear(first * V, last * V, count)), bench.options
    )
    step = (last - first) / (count - 1)
    raw = ngspice.run(
        netlist(simulation.circuit),
        f"dc VV1 {first!r} {last!r} {step!r}",
        libraries=(file,),
        compatibility=bench.compatibility,
        options=bench.spice_options,
    )
    for column, unknown in compared(simulation):
        spice_values = raw.real(vector(simulation.circuit, unknown))
        compare(
            sweep.values[:, column],
            spice_values,
            unknown,
            scale=size(spice_values),
            tolerance=1e-6,
        )


@pytest.mark.parametrize("name", _benches("ac"))
def test_ac_responses_agree(name: str, downloads: Path) -> None:
    bench = BENCHES[name]
    top, file = bench.build(downloads)
    simulation = Simulation(top)
    response = simulation.run(Ac("V1", decade(10 * Hz, 10 * MHz, 20)), bench.options)
    raw = ngspice.run(
        netlist(simulation.circuit, excited="V1"),
        "ac dec 20 10 10meg",
        libraries=(file,),
        compatibility=bench.compatibility,
        options=bench.spice_options,
    )
    for column, unknown in compared(simulation):
        spice_values = raw.vectors[vector(simulation.circuit, unknown)]
        compare(
            response.values[:, column],
            spice_values,
            unknown,
            scale=size(spice_values),
            tolerance=1e-6,
        )


@pytest.mark.parametrize("name", _benches("tran"))
def test_transients_agree(name: str, downloads: Path) -> None:
    bench = BENCHES[name]
    top, file = bench.build(downloads)
    simulation = Simulation(top)
    assert bench.transient is not None
    duration, maximal = bench.transient
    waveforms = simulation.run(
        Transient(stop=duration, max_step=maximal), bench.options
    )
    step, stop = maximal.value, duration.value
    raw = ngspice.run(
        netlist(simulation.circuit),
        f"tran {step!r} {stop!r} 0 {step!r}",
        libraries=(file,),
        compatibility=bench.compatibility,
        options=bench.spice_options,
    )
    times = raw.real("time")
    for column, unknown in compared(simulation):
        spice_values = raw.real(vector(simulation.circuit, unknown))
        jolty = closest(
            waveforms.times, waveforms.values[:, column], times, spice_values, step
        )
        compare(jolty, spice_values, unknown, scale=size(spice_values), tolerance=1e-3)

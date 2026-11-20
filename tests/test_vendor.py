# Copyright 2026 Maxence Faldor
"""Differential tests on vendor macromodels loaded from SPICE subcircuit files.

Each bench places a component parsed by jolty's SPICE importer and runs ngspice on the
same netlist. PSpice-syntax models run ngspice in PSpice compatibility mode because
default ngspice rejects their syntax. All model files come from ngspice's own examples,
fetched from a pinned repository commit and verified against SHA digests.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from differential import (
    SKIP,
    assert_ac_agrees,
    assert_dc_sweep_agrees,
    assert_op_agrees,
    assert_transient_agrees,
    dc_command,
    netlist,
    ngspice,
    transient_command,
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
from jolty.plan import TIGHT, decade, linear
from jolty.spice.ngspice import OPTIONS, VOLTAGE_TOLERANCE, Compatibility
from jolty.units import Hz, MHz, Quantity, V, celsius, kHz, kohm, ms, pA, us
from jolty.waveforms import Sine

pytestmark = SKIP

_MIRROR = (
    "https://raw.githubusercontent.com/imr/ngspice/"
    "a80f6e3e95d51534905b1f23410a951802666656/examples/"
)
"""Pinned ngspice examples repository mirror URL and commit."""


@dataclass(frozen=True)
class Part:
    """A vendor's macromodel: where its file is, its digest, and its subcircuit."""

    path: str
    digest: str
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
    """Returns the cache directory storing downloaded vendor model files."""
    assert pytestconfig.cache is not None
    return pytestconfig.cache.mkdir("downloads")


def _file(part: Part, downloads: Path) -> Path:
    return spice.fetch(_MIRROR + part.path, part.digest, downloads)


def _inverting(
    part: Component,
    supplies: tuple[float, float],
    pins: tuple[str, str, str, str, str],
    signal: Sine,
) -> Component:
    """Builds a gain -2 inverting amplifier around `part` with mid-supply reference.

    The `pins` tuple orders non-inverting input, inverting input, supplies, and output.
    """
    plus, minus, positive, negative, output = pins
    c = Builder("inverting")
    ground = c.port("ground", Role.GROUND)
    upper = c.place(devices.voltage_source(supplies[0] * V), name="VP")
    lower = c.place(devices.voltage_source(supplies[1] * V), name="VN")
    middle = c.place(devices.voltage_source(sum(supplies) / 2 * V), name="VREF")
    source = c.place(devices.voltage_source(signal), name="V1")
    feed = c.place(devices.resistor(10 * kohm), name="RIN")
    feedback = c.place(devices.resistor(20 * kohm), name="RF")
    load = c.place(devices.resistor(100 * kohm), name="RL")
    amplifier = c.place(part, name="U1")
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
    """Builds the single-supply AD22057 difference amplifier with both stages joined."""
    file = _file(AD22057, downloads)
    part = spice.load(file).component(AD22057.subcircuit)
    c = Builder("difference")
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(5 * V), name="VCC")
    reference = c.place(devices.voltage_source(0.1 * V), name="V2")
    signal = Sine(
        offset=0.12 * V, amplitude=0.05 * V, frequency=1 * kHz, phase=math.pi / 2
    )
    source = c.place(devices.voltage_source(signal), name="V1")
    load = c.place(devices.resistor(10 * kohm), name="RL")
    offset = c.place(devices.resistor(100 * kohm), name="ROFF")
    amplifier = c.place(part, name="U1")
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
        """Returns whether both simulators run `analysis` (`op`, `dc`, `ac`, `tran`)."""
        match analysis:
            case "dc":
                return self.sweep is not None
            case "ac":
                return self.ac
            case "tran":
                return self.transient is not None
        return True


BENCHES = {
    # Uses SPICE2 syntax, which ngspice parses outside PSpice mode; PSpice mode would
    # add a 100 µΩ series resistor to every diode lacking one.
    "tl072": Bench(_tl072, (-10, 10, 81), (300 * us, 0.2 * us), compatibility=None),
    "mcp6041": Bench(_mcp6041, (1.5, 3.5, 81), (6 * ms, 2 * us)),
    # The amplifier's gains of a million leave its currents' last digits unresolved
    # below the default absolute tolerance, in both engines.
    "ad22057": Bench(
        _ad22057,
        (0.0, 0.3, 61),
        (3 * ms, 1 * us),
        options=replace(TIGHT, current_tolerance=1 * pA),
        spice_options=(
            f".options reltol=1e-6 vntol={VOLTAGE_TOLERANCE.value!r} abstol=1e-12"
        ),
    ),
    # ngspice finds the operating point of this amplifier at default tolerances only.
    # XSPICE switches in the model stamp negated conductance during AC analysis.
    "opa171": Bench(_opa171, None, None, ac=False, spice_options=""),
}
"""The benches, by name."""


def _benches(analysis: str) -> list[str]:
    """Returns names of all benches enabled for `analysis` on both simulators."""
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
    assert_op_agrees(simulation, point, raw)


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
    assert_op_agrees(simulation, point, raw)


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
    raw = ngspice.run(
        netlist(simulation.circuit),
        dc_command("V1", sweep),
        libraries=(file,),
        compatibility=bench.compatibility,
        options=bench.spice_options,
    )
    assert_dc_sweep_agrees(simulation, sweep, raw)


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
    assert_ac_agrees(simulation, response, raw)


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
    raw = ngspice.run(
        netlist(simulation.circuit),
        transient_command(duration.value, maximal.value),
        libraries=(file,),
        compatibility=bench.compatibility,
        options=bench.spice_options,
    )
    assert_transient_agrees(simulation, waveforms, raw, window=maximal.value)

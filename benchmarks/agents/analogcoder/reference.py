# Copyright 2026 Maxence Faldor
"""Reference circuit designs for all twenty-four AnalogCoder benchmark problems.

Problems 1 through 15 adapt the textbook topologies from AnalogCoder
(github.com/laiyao1/AnalogCoder, MIT License) with updated sizing and biasing.
Problems 16 through 24 are custom reference designs built around a shared two-stage
op-amp using the same MOSFET models. The `LEVELS` mapping records the DC input bias
levels for each problem.
"""

import runpy
from pathlib import Path

from jolty import Builder, Component, Role, component, devices
from jolty.builder import Net
from jolty.units import Mohm, V, fF, kohm, nF, ohm, pF, uF, um

_parts = runpy.run_path(str(Path(__file__).parent / "given" / "parts.py"))


def nmos(width: float, length: float = 1.0) -> devices.Device:
    """Returns the given NMOS sized `width` by `length` micrometers."""
    made = _parts["nmos"](width * um, length * um)
    if not isinstance(made, devices.Device):
        raise TypeError(made)
    return made


def pmos(width: float, length: float = 1.0) -> devices.Device:
    """Returns the given PMOS sized `width` by `length` micrometers."""
    made = _parts["pmos"](width * um, length * um)
    if not isinstance(made, devices.Device):
        raise TypeError(made)
    return made


def ports(c: Builder, *names: str) -> dict[str, Net]:
    """Declares supply ports ``vdd`` and ``ground`` plus each port in `names`."""
    found = {"vdd": c.port("vdd", Role.SUPPLY), "ground": c.port("ground", Role.GROUND)}
    return found | {name: c.port(name) for name in names}


def mosfet(
    c: Builder, id: str, terminals: tuple[Net, Net, Net], made: devices.Device
) -> None:
    """Places `made` as `id` on its drain, gate and source, its bulk on its source."""
    drain, gate, source = terminals
    placed = c.place(made, name=id)
    c.connect(placed.drain, drain)
    c.connect(placed.gate, gate)
    c.connect(placed.source, placed.bulk, source)


def two_terminal(c: Builder, id: str, made: devices.Device, a: Net, b: Net) -> None:
    """Places the two-terminal `made` as `id`, from `a` to `b`."""
    placed = c.place(made, name=id)
    terminals = [getattr(placed, name) for name in made.terminals]
    c.connect(terminals[0], a)
    c.connect(terminals[1], b)


def resistor(c: Builder, id: str, kilohms: float, a: Net, b: Net) -> None:
    """Places a resistor of `kilohms` kΩ named `id` between `a` and `b`."""
    two_terminal(c, id, devices.resistor(kilohms * kohm), a, b)


def capacitor(c: Builder, id: str, made: devices.Device, a: Net, b: Net) -> None:
    """Places capacitor `made` named `id` between `a` and `b`."""
    two_terminal(c, id, made, a, b)


def dc_source(
    circuit_builder: Builder, name: str, volts: float, net: Net, ground: Net
) -> None:
    """Places a DC voltage source holding `net` at `volts` above `ground`."""
    two_terminal(circuit_builder, name, devices.voltage_source(volts * V), net, ground)


def common_source(c: Builder) -> None:
    """Builds problem 1: a common-source amplifier with a 1 kΩ load."""
    p = ports(c, "vin", "vout")
    mosfet(c, "M1", (p["vout"], p["vin"], p["ground"]), nmos(50))
    resistor(c, "R1", 1, p["vout"], p["vdd"])


def three_stages(c: Builder) -> None:
    """Builds problem 2: three cascaded common-source stages with 1 kΩ loads."""
    p = ports(c, "vin", "vout")
    drains = [c.net("drain1"), c.net("drain2"), p["vout"]]
    gates = [p["vin"], *drains[:2]]
    for index, (gate, drain) in enumerate(zip(gates, drains, strict=True), 1):
        mosfet(c, f"M{index}", (drain, gate, p["ground"]), nmos(50))
        resistor(c, f"R{index}", 1, drain, p["vdd"])


def source_follower(c: Builder) -> None:
    """Builds problem 3: a source follower driving a 1 kΩ load."""
    p = ports(c, "vin", "vout")
    mosfet(c, "M1", (p["vdd"], p["vin"], p["vout"]), nmos(50))
    resistor(c, "R1", 1, p["vout"], p["ground"])


def common_gate(c: Builder) -> None:
    """Builds problem 4: a common-gate amplifier driven at the source."""
    p = ports(c, "vin", "vbias", "vout")
    mosfet(c, "M1", (p["vout"], p["vbias"], p["vin"]), nmos(50))
    resistor(c, "R1", 1, p["vout"], p["vdd"])


def cascode(c: Builder) -> None:
    """Builds problem 5: a two-NMOS cascode amplifier with a 1 kΩ load."""
    p = ports(c, "vin", "vbias", "vout")
    middle = c.net("middle")
    mosfet(c, "M1", (middle, p["vin"], p["ground"]), nmos(50))
    mosfet(c, "M2", (p["vout"], p["vbias"], middle), nmos(50))
    resistor(c, "R1", 1, p["vout"], p["vdd"])


def resistive_inverter(c: Builder) -> None:
    """Builds problem 6: an NMOS inverter with a 1 kΩ resistive load."""
    common_source(c)


def cmos_inverter(c: Builder) -> None:
    """Builds problem 7: a complementary CMOS inverter."""
    p = ports(c, "vin", "vout")
    mosfet(c, "M1", (p["vout"], p["vin"], p["ground"]), nmos(50))
    mosfet(c, "M2", (p["vout"], p["vin"], p["vdd"]), pmos(100))


def current_sink(c: Builder) -> None:
    """Builds problem 8: an NMOS current sink loaded by the test bench."""
    p = ports(c, "vbias", "vout")
    mosfet(c, "M1", (p["vout"], p["vbias"], p["ground"]), nmos(50))


def miller(c: Builder) -> None:
    """Builds problem 9: two common-source stages with a Miller capacitor."""
    p = ports(c, "vin", "vout")
    first = c.net("first")
    mosfet(c, "M1", (first, p["vin"], p["ground"]), nmos(50))
    resistor(c, "R1", 1, first, p["vdd"])
    mosfet(c, "M2", (p["vout"], first, p["ground"]), nmos(10))
    resistor(c, "R2", 1, p["vout"], p["vdd"])
    capacitor(c, "CC", devices.capacitor(10 * pF), first, p["vout"])


def diode_load(c: Builder) -> None:
    """Builds problem 10: a common-source amplifier with a diode-connected PMOS load."""
    p = ports(c, "vin", "vout")
    mosfet(c, "M1", (p["vout"], p["vin"], p["ground"]), nmos(50))
    mosfet(c, "M2", (p["vout"], p["vout"], p["vdd"]), pmos(100))


def active_load_pair(c: Builder) -> None:
    """Builds problem 11: a PMOS-mirror-loaded differential pair with dual outputs."""
    p = ports(c, "vinp", "vinn", "vbias", "voutp", "vout")
    tail = c.net("tail")
    mosfet(c, "M1", (p["voutp"], p["vinp"], tail), nmos(50))
    mosfet(c, "M2", (p["vout"], p["vinn"], tail), nmos(50))
    mosfet(c, "M3", (tail, p["vbias"], p["ground"]), nmos(100))
    mosfet(c, "M4", (p["voutp"], p["voutp"], p["vdd"]), pmos(100))
    mosfet(c, "M5", (p["vout"], p["voutp"], p["vdd"]), pmos(100))


def cascode_mirror(c: Builder) -> None:
    """Builds problem 12: a four-NMOS cascode current mirror."""
    p = ports(c, "iref", "iout")
    reference, output = c.net("reference"), c.net("output")
    mosfet(c, "M1", (p["iref"], p["iref"], reference), nmos(50))
    mosfet(c, "M2", (reference, reference, p["ground"]), nmos(50))
    mosfet(c, "M3", (p["iout"], p["iref"], output), nmos(50))
    mosfet(c, "M4", (output, reference, p["ground"]), nmos(50))


def resistive_pair(c: Builder) -> None:
    """Builds problem 13: a differential pair with resistive loads and one output."""
    p = ports(c, "vinp", "vinn", "vout")
    tail, bias, other = c.net("tail"), c.net("bias"), c.net("other")
    dc_source(c, "VB", 1.0, bias, p["ground"])
    mosfet(c, "M1", (other, p["vinp"], tail), nmos(50))
    mosfet(c, "M2", (p["vout"], p["vinn"], tail), nmos(50))
    mosfet(c, "M3", (tail, bias, p["ground"]), nmos(100))
    resistor(c, "R1", 2, other, p["vdd"])
    resistor(c, "R2", 2, p["vout"], p["vdd"])


def two_stage(c: Builder) -> None:
    """Builds problem 14: a mirror-loaded pair driving PMOS stages with NMOS loads."""
    p = ports(c, "vinp", "vinn", "vbias1", "vbias2", "vbias3", "voutp", "vout")
    tail, near, far = c.net("tail"), c.net("near"), c.net("far")
    mosfet(c, "M1", (near, p["vinn"], tail), nmos(50))
    mosfet(c, "M2", (far, p["vinp"], tail), nmos(50))
    mosfet(c, "M3", (tail, p["vbias1"], p["ground"]), nmos(100))
    mosfet(c, "M4", (near, near, p["vdd"]), pmos(50))
    mosfet(c, "M5", (far, near, p["vdd"]), pmos(50))
    mosfet(c, "M6", (p["voutp"], near, p["vdd"]), pmos(100))
    mosfet(c, "M7", (p["voutp"], p["vbias2"], p["ground"]), nmos(100))
    mosfet(c, "M8", (p["vout"], far, p["vdd"]), pmos(100))
    mosfet(c, "M9", (p["vout"], p["vbias3"], p["ground"]), nmos(100))


def telescopic(c: Builder) -> None:
    """Builds problem 15: a telescopic cascode differential pair with dual outputs."""
    names = ("vinp", "vinn", "vbias1", "vbias2", "vbias3", "vbias4", "voutp", "vout")
    p = ports(c, *names)
    tail = c.net("tail")
    lower = [c.net("lower1"), c.net("lower2")]
    upper = [c.net("upper1"), c.net("upper2")]
    outputs = [p["voutp"], p["vout"]]
    inputs = [p["vinp"], p["vinn"]]
    for side in range(2):
        mosfet(c, f"MI{side}", (lower[side], inputs[side], tail), nmos(50))
        mosfet(c, f"MC{side}", (outputs[side], p["vbias1"], lower[side]), nmos(50))
        mosfet(c, f"ML{side}", (outputs[side], p["vbias2"], upper[side]), pmos(100))
        mosfet(c, f"MS{side}", (upper[side], p["vbias3"], p["vdd"]), pmos(100))
    mosfet(c, "MT", (tail, p["vbias4"], p["ground"]), nmos(50))


@component
def opamp(c: Builder) -> None:
    """Builds a two-stage op-amp with a mirror-loaded NMOS pair and a PMOS output stage.

    Each stage carries 200 µA biased at 0.7 V on the NMOS tail sources, achieving an
    output swing from 0.2 V to 4.7 V with a 1 pF Miller compensation capacitor. A 1 pF
    output load capacitor models real parasitic capacitance so transient time steps
    converge reliably during rapid output transitions.
    """
    p = ports(c, "inp", "inn", "out")
    tail, near, far, bias = c.net("tail"), c.net("near"), c.net("far"), c.net("bias")
    dc_source(c, "VB", 0.7, bias, p["ground"])
    mosfet(c, "M1", (near, p["inn"], tail), nmos(50))
    mosfet(c, "M2", (far, p["inp"], tail), nmos(50))
    mosfet(c, "M3", (tail, bias, p["ground"]), nmos(100))
    mosfet(c, "M4", (near, near, p["vdd"]), pmos(50))
    mosfet(c, "M5", (far, near, p["vdd"]), pmos(50))
    mosfet(c, "M6", (p["out"], far, p["vdd"]), pmos(100))
    mosfet(c, "M7", (p["out"], bias, p["ground"]), nmos(100))
    capacitor(c, "CC", devices.capacitor(1 * pF), far, p["out"])
    capacitor(c, "CL", devices.capacitor(1 * pF), p["out"], p["ground"])


def amplified(c: Builder, p: dict[str, Net], inp: Net, inn: Net, out: Net) -> None:
    """Places the op-amp on the supply of the ports `p`."""
    placed = c.place(opamp(), name="A1")
    c.connect(placed.vdd, p["vdd"])
    c.connect(placed.ground, p["ground"])
    c.connect(placed.inp, inp)
    c.connect(placed.inn, inn)
    c.connect(placed.out, out)


def phase_shift(c: Builder) -> None:
    """Builds problem 16: an inverting gain-33 stage behind three RC phase sections."""
    p = ports(c, "vout")
    reference, inn = c.net("reference"), c.net("inn")
    dc_source(c, "VR", 2.5, reference, p["ground"])
    amplified(c, p, reference, inn, p["vout"])
    resistor(c, "RF", 330, inn, p["vout"])
    nodes = [p["vout"], c.net("n1"), c.net("n2"), c.net("n3")]
    for index in range(3):
        made = devices.capacitor(1 * nF)
        capacitor(c, f"C{index + 1}", made, nodes[index], nodes[index + 1])
    resistor(c, "R1", 10, nodes[1], reference)
    resistor(c, "R2", 10, nodes[2], reference)
    resistor(c, "R3", 10, nodes[3], inn)


def wien(c: Builder) -> None:
    """Builds problem 17: a Wien-bridge oscillator with a gain-3.3 amplifier."""
    p = ports(c, "vout")
    reference, inp, inn, series = (
        c.net("reference"),
        c.net("inp"),
        c.net("inn"),
        c.net("series"),
    )
    dc_source(c, "VR", 2.5, reference, p["ground"])
    amplified(c, p, inp, inn, p["vout"])
    resistor(c, "RS", 10, p["vout"], series)
    capacitor(c, "CS", devices.capacitor(1 * nF), series, inp)
    resistor(c, "RP", 10, inp, reference)
    capacitor(c, "CP", devices.capacitor(1 * nF), inp, reference)
    resistor(c, "RF", 23, p["vout"], inn)
    resistor(c, "RG", 10, inn, reference)


def integrator(c: Builder) -> None:
    """Builds problem 18: an inverting integrator (10 kΩ, 3 µF, 10 MΩ DC feedback)."""
    p = ports(c, "vin", "vout")
    reference, inn = c.net("reference"), c.net("inn")
    dc_source(c, "VR", 2.5, reference, p["ground"])
    amplified(c, p, reference, inn, p["vout"])
    resistor(c, "R1", 10, p["vin"], inn)
    capacitor(c, "CF", devices.capacitor(3 * uF), inn, p["vout"])
    two_terminal(c, "RDC", devices.resistor(10 * Mohm), inn, p["vout"])


def differentiator(c: Builder) -> None:
    """Builds problem 19: a differentiator with 3 µF, 100 Ω damping, and 10 kΩ feedback.

    Adding the 100 Ω series resistor preserves loop phase margin against the op-amp
    pole and prevents ringing.
    """
    p = ports(c, "vin", "vout")
    reference, inn, series = c.net("reference"), c.net("inn"), c.net("series")
    dc_source(c, "VR", 2.5, reference, p["ground"])
    amplified(c, p, reference, inn, p["vout"])
    capacitor(c, "C1", devices.capacitor(3 * uF), p["vin"], series)
    two_terminal(c, "RS", devices.resistor(100 * ohm), series, inn)
    resistor(c, "RF", 10, inn, p["vout"])


def adder(c: Builder) -> None:
    """Builds problem 20: an inverting two-input 10 kΩ summer biased around 2.5 V."""
    p = ports(c, "vin1", "vin2", "vout")
    reference, inn = c.net("reference"), c.net("inn")
    dc_source(c, "VR", 2.5, reference, p["ground"])
    amplified(c, p, reference, inn, p["vout"])
    resistor(c, "R1", 10, p["vin1"], inn)
    resistor(c, "R2", 10, p["vin2"], inn)
    resistor(c, "RF", 10, inn, p["vout"])


def subtractor(c: Builder) -> None:
    """Builds problem 21: a four-resistor 10 kΩ difference amplifier."""
    p = ports(c, "vin1", "vin2", "vout")
    inp, inn = c.net("inp"), c.net("inn")
    amplified(c, p, inp, inn, p["vout"])
    resistor(c, "R1", 10, p["vin1"], inn)
    resistor(c, "R2", 10, inn, p["vout"])
    resistor(c, "R3", 10, p["vin2"], inp)
    resistor(c, "R4", 10, inp, p["ground"])


def schmitt(c: Builder) -> None:
    """Builds problem 22: a non-inverting Schmitt trigger (20 kΩ over 10 kΩ input)."""
    p = ports(c, "vin", "vout")
    inp, inn = c.net("inp"), c.net("inn")
    amplified(c, p, inp, inn, p["vout"])
    resistor(c, "R1", 10, p["vin"], inp)
    resistor(c, "RF", 20, inp, p["vout"])
    resistor(c, "RA", 10, p["vdd"], inn)
    resistor(c, "RB", 10, inn, p["ground"])


def inverter(c: Builder, p: dict[str, Net], id: str, gate: Net, drain: Net) -> None:
    """Places a CMOS inverter from `gate` to `drain` with a 50 fF output load capacitor.

    The 50 fF load replaces missing intrinsic MOSFET capacitances so transient time
    steps remain smooth across switching transitions.
    """
    mosfet(c, f"{id}N", (drain, gate, p["ground"]), nmos(10))
    mosfet(c, f"{id}P", (drain, gate, p["vdd"]), pmos(20))
    capacitor(c, f"{id}C", devices.capacitor(50 * fF), drain, p["ground"])


@component
def ring(c: Builder, capacitance: float) -> None:
    """Builds a three-stage current-starved ring oscillator controlled by `control`.

    The `control` voltage sets current through a 2 µm by 5 µm NMOS under each stage
    and a mirrored PMOS above. Each stage drives `capacitance` farads (with 20 % extra
    on stage one to break startup symmetry without noise), buffered to ``out``.
    Internal starved stage nodes carry 10 fF to ground in place of omitted intrinsic
    transistor capacitances.
    """
    p = ports(c, "control", "out")
    mirror = c.net("mirror")
    mosfet(c, "MB", (mirror, p["control"], p["ground"]), nmos(2, 5))
    mosfet(c, "MM", (mirror, mirror, p["vdd"]), pmos(4, 5))
    stages = [c.net(f"stage{index}") for index in range(3)]
    for index, (gate, drain) in enumerate(
        zip(stages[-1:] + stages[:-1], stages, strict=True)
    ):
        low, high = c.net(f"low{index}"), c.net(f"high{index}")
        mosfet(c, f"MS{index}", (low, p["control"], p["ground"]), nmos(2, 5))
        mosfet(c, f"MN{index}", (drain, gate, low), nmos(10))
        mosfet(c, f"MP{index}", (drain, gate, high), pmos(20))
        mosfet(c, f"MT{index}", (high, mirror, p["vdd"]), pmos(4, 5))
        for node, name in ((low, "L"), (high, "H")):
            capacitor(
                c, f"C{name}{index}", devices.capacitor(10 * fF), node, p["ground"]
            )
        heavier = 1.2 if index == 0 else 1.0
        load = devices.capacitor(heavier * capacitance * 1e12 * pF)
        capacitor(c, f"C{index}", load, drain, p["ground"])
    inverter(c, p, "MO", stages[-1], p["out"])


def placed_ring(
    c: Builder, p: dict[str, Net], capacitance: float, control: Net, out: Net
) -> None:
    """Places a ring of `capacitance` on the supply of the ports `p`."""
    placed = c.place(ring(capacitance), name="VCO")
    c.connect(placed.vdd, p["vdd"])
    c.connect(placed.ground, p["ground"])
    c.connect(placed.control, control)
    c.connect(placed.out, out)


def vco(c: Builder) -> None:
    """Builds problem 23: a current-starved ring VCO operating near 2 MHz."""
    p = ports(c, "vin", "vout")
    placed_ring(c, p, 10e-12, p["vin"], p["vout"])


def pll(c: Builder) -> None:
    """Builds problem 24: an XOR phase detector, lag-lead filter, and ring VCO.

    The ring oscillator runs near 11 MHz at mid-supply control voltage, matching the
    average phase-detector output while unlocked.
    """
    p = ports(c, "clk_ref", "clk_p")
    names = ("ref_bar", "out_bar", "detected", "pulled", "control", "middle")
    ref_bar, out_bar, detected, pulled, control, middle = (c.net(n) for n in names)
    inverter(c, p, "IR", p["clk_ref"], ref_bar)
    inverter(c, p, "IO", p["clk_p"], out_bar)
    pairs = ((p["clk_ref"], p["clk_p"]), (ref_bar, out_bar))
    for index, (first, second) in enumerate(pairs):
        between = c.net(f"between{index}")
        mosfet(c, f"MXN{index}A", (detected, first, between), nmos(10))
        mosfet(c, f"MXN{index}B", (between, second, p["ground"]), nmos(10))
    mosfet(c, "MXP0A", (pulled, p["clk_ref"], p["vdd"]), pmos(20))
    mosfet(c, "MXP0B", (pulled, p["clk_p"], p["vdd"]), pmos(20))
    mosfet(c, "MXP1A", (detected, ref_bar, pulled), pmos(20))
    mosfet(c, "MXP1B", (detected, out_bar, pulled), pmos(20))
    capacitor(c, "CX", devices.capacitor(50 * fF), detected, p["ground"])
    resistor(c, "R1", 45, detected, control)
    resistor(c, "R2", 5, control, middle)
    capacitor(c, "C1", devices.capacitor(10 * pF), middle, p["ground"])
    capacitor(c, "C2", devices.capacitor(1 * pF), control, p["ground"])
    placed_ring(c, p, 0.42e-12, control, p["clk_p"])


BUILDERS = {
    1: common_source,
    2: three_stages,
    3: source_follower,
    4: common_gate,
    5: cascode,
    6: resistive_inverter,
    7: cmos_inverter,
    8: current_sink,
    9: miller,
    10: diode_load,
    11: active_load_pair,
    12: cascode_mirror,
    13: resistive_pair,
    14: two_stage,
    15: telescopic,
    16: phase_shift,
    17: wien,
    18: integrator,
    19: differentiator,
    20: adder,
    21: subtractor,
    22: schmitt,
    23: vco,
    24: pll,
}
"""Mapping from AnalogCoder problem number to circuit builder function."""

LEVELS = {
    1: {"vin": 1.5},
    2: {"vin": 1.65},
    3: {"vin": 4.0},
    4: {"vin": 0.0, "vbias": 1.5},
    5: {"vin": 1.5, "vbias": 2.5},
    8: {"vbias": 1.5},
    9: {"vin": 1.5},
    10: {"vin": 2.5},
    11: {"vinp": 2.5, "vinn": 2.5, "vbias": 1.5},
    12: {"iref": 1e-3},
    13: {"vinp": 2.0, "vinn": 2.0},
    14: {"vinp": 2.5, "vinn": 2.5, "vbias1": 0.7, "vbias2": 0.7, "vbias3": 0.7},
    15: {
        "vinp": 2.0,
        "vinn": 2.0,
        "vbias1": 3.0,
        "vbias2": 3.0,
        "vbias3": 4.3,
        "vbias4": 0.7,
    },
    18: {"vin": 2.5},
    19: {"vin": 2.5},
    20: {"vin1": 2.5, "vin2": 2.5},
    21: {"vin1": 2.0, "vin2": 4.0},
}
"""Mapping from problem number to required DC input bias levels."""


def build(number: int) -> Component:
    """Constructs and returns the reference `Component` for problem `number`."""
    function = BUILDERS[number]
    return component(function)()

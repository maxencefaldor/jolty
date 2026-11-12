# Copyright 2026 Maxence Faldor
"""A design for each of AnalogCoder's problems, which the scorer must pass.

The first fifteen are the textbook circuits of the samples of AnalogCoder
(github.com/laiyao1/AnalogCoder, MIT License), sized and biased anew; the rest are this
file's own, around one two-stage op-amp of the same transistors. ``LEVELS`` gives each
design's input levels, by problem.
"""

import runpy
from pathlib import Path

from jolty import Builder, Component, Role, component, devices
from jolty.builder import Net
from jolty.units import Mohm, V, fF, kohm, nF, ohm, pF, uF, um

_parts = runpy.run_path(str(Path(__file__).parent / "given" / "parts.py"))


def nmos(width: float, length: float = 1.0) -> devices.Device:
    """The given NMOS, `width` by `length` micrometres."""
    made = _parts["nmos"](width * um, length * um)
    if not isinstance(made, devices.Device):
        raise TypeError(made)
    return made


def pmos(width: float, length: float = 1.0) -> devices.Device:
    """The given PMOS, `width` by `length` micrometres."""
    made = _parts["pmos"](width * um, length * um)
    if not isinstance(made, devices.Device):
        raise TypeError(made)
    return made


def ports(c: Builder, *names: str) -> dict[str, Net]:
    """The supply's ports, ``vdd`` and ``ground``, then `names`."""
    found = {"vdd": c.port("vdd", Role.SUPPLY), "ground": c.port("ground", Role.GROUND)}
    return found | {name: c.port(name) for name in names}


def mosfet(
    c: Builder, id: str, terminals: tuple[Net, Net, Net], made: devices.Device
) -> None:
    """Places `made` as `id` on its drain, gate and source, its bulk on its source."""
    drain, gate, source = terminals
    placed = c.place(made, id=id)
    c.connect(placed.drain, drain)
    c.connect(placed.gate, gate)
    c.connect(placed.source, placed.bulk, source)


def two_terminal(c: Builder, id: str, made: devices.Device, a: Net, b: Net) -> None:
    """Places the two-terminal `made` as `id`, from `a` to `b`."""
    placed = c.place(made, id=id)
    terminals = [getattr(placed, name) for name in made.terminals]
    c.connect(terminals[0], a)
    c.connect(terminals[1], b)


def resistor(c: Builder, id: str, kilohms: float, a: Net, b: Net) -> None:
    """A resistor of `kilohms` from `a` to `b`."""
    two_terminal(c, id, devices.resistor(kilohms * kohm), a, b)


def capacitor(c: Builder, id: str, made: devices.Device, a: Net, b: Net) -> None:
    """The capacitor `made` from `a` to `b`."""
    two_terminal(c, id, made, a, b)


def level(c: Builder, id: str, volts: float, net: Net, ground: Net) -> None:
    """A constant source holding `net` at `volts` over `ground`."""
    two_terminal(c, id, devices.voltage_source(volts * V), net, ground)


def common_source(c: Builder) -> None:
    """Problem 1: a common-source amplifier with a 1 kΩ load."""
    p = ports(c, "vin", "vout")
    mosfet(c, "M1", (p["vout"], p["vin"], p["ground"]), nmos(50))
    resistor(c, "R1", 1, p["vout"], p["vdd"])


def three_stages(c: Builder) -> None:
    """Problem 2: three common-source stages, each with a 1 kΩ load."""
    p = ports(c, "vin", "vout")
    drains = [c.net("drain1"), c.net("drain2"), p["vout"]]
    gates = [p["vin"], *drains[:2]]
    for index, (gate, drain) in enumerate(zip(gates, drains, strict=True), 1):
        mosfet(c, f"M{index}", (drain, gate, p["ground"]), nmos(50))
        resistor(c, f"R{index}", 1, drain, p["vdd"])


def source_follower(c: Builder) -> None:
    """Problem 3: a source follower into 1 kΩ."""
    p = ports(c, "vin", "vout")
    mosfet(c, "M1", (p["vdd"], p["vin"], p["vout"]), nmos(50))
    resistor(c, "R1", 1, p["vout"], p["ground"])


def common_gate(c: Builder) -> None:
    """Problem 4: a common-gate amplifier, its input at its source."""
    p = ports(c, "vin", "vbias", "vout")
    mosfet(c, "M1", (p["vout"], p["vbias"], p["vin"]), nmos(50))
    resistor(c, "R1", 1, p["vout"], p["vdd"])


def cascode(c: Builder) -> None:
    """Problem 5: a cascode of two NMOS into 1 kΩ."""
    p = ports(c, "vin", "vbias", "vout")
    middle = c.net("middle")
    mosfet(c, "M1", (middle, p["vin"], p["ground"]), nmos(50))
    mosfet(c, "M2", (p["vout"], p["vbias"], middle), nmos(50))
    resistor(c, "R1", 1, p["vout"], p["vdd"])


def resistive_inverter(c: Builder) -> None:
    """Problem 6: an NMOS inverter with a 1 kΩ load."""
    common_source(c)


def cmos_inverter(c: Builder) -> None:
    """Problem 7: a CMOS inverter."""
    p = ports(c, "vin", "vout")
    mosfet(c, "M1", (p["vout"], p["vin"], p["ground"]), nmos(50))
    mosfet(c, "M2", (p["vout"], p["vin"], p["vdd"]), pmos(100))


def current_sink(c: Builder) -> None:
    """Problem 8: an NMOS current source, its load the bench's."""
    p = ports(c, "vbias", "vout")
    mosfet(c, "M1", (p["vout"], p["vbias"], p["ground"]), nmos(50))


def miller(c: Builder) -> None:
    """Problem 9: two common-source stages, a Miller capacitor across the second."""
    p = ports(c, "vin", "vout")
    first = c.net("first")
    mosfet(c, "M1", (first, p["vin"], p["ground"]), nmos(50))
    resistor(c, "R1", 1, first, p["vdd"])
    mosfet(c, "M2", (p["vout"], first, p["ground"]), nmos(10))
    resistor(c, "R2", 1, p["vout"], p["vdd"])
    capacitor(c, "CC", devices.capacitor(10 * pF), first, p["vout"])


def diode_load(c: Builder) -> None:
    """Problem 10: a common-source amplifier with a diode-connected PMOS load."""
    p = ports(c, "vin", "vout")
    mosfet(c, "M1", (p["vout"], p["vin"], p["ground"]), nmos(50))
    mosfet(c, "M2", (p["vout"], p["vout"], p["vdd"]), pmos(100))


def active_load_pair(c: Builder) -> None:
    """Problem 11: a differential pair, its load a PMOS mirror, and two outputs."""
    p = ports(c, "vinp", "vinn", "vbias", "voutp", "vout")
    tail = c.net("tail")
    mosfet(c, "M1", (p["voutp"], p["vinp"], tail), nmos(50))
    mosfet(c, "M2", (p["vout"], p["vinn"], tail), nmos(50))
    mosfet(c, "M3", (tail, p["vbias"], p["ground"]), nmos(100))
    mosfet(c, "M4", (p["voutp"], p["voutp"], p["vdd"]), pmos(100))
    mosfet(c, "M5", (p["vout"], p["voutp"], p["vdd"]), pmos(100))


def cascode_mirror(c: Builder) -> None:
    """Problem 12: a cascode current mirror of four NMOS."""
    p = ports(c, "iref", "iout")
    reference, output = c.net("reference"), c.net("output")
    mosfet(c, "M1", (p["iref"], p["iref"], reference), nmos(50))
    mosfet(c, "M2", (reference, reference, p["ground"]), nmos(50))
    mosfet(c, "M3", (p["iout"], p["iref"], output), nmos(50))
    mosfet(c, "M4", (output, reference, p["ground"]), nmos(50))


def resistive_pair(c: Builder) -> None:
    """Problem 13: a differential pair with resistive loads and one output."""
    p = ports(c, "vinp", "vinn", "vout")
    tail, bias, other = c.net("tail"), c.net("bias"), c.net("other")
    level(c, "VB", 1.0, bias, p["ground"])
    mosfet(c, "M1", (other, p["vinp"], tail), nmos(50))
    mosfet(c, "M2", (p["vout"], p["vinn"], tail), nmos(50))
    mosfet(c, "M3", (tail, bias, p["ground"]), nmos(100))
    resistor(c, "R1", 2, other, p["vdd"])
    resistor(c, "R2", 2, p["vout"], p["vdd"])


def two_stage(c: Builder) -> None:
    """Problem 14: a mirror-loaded pair, then PMOS stages with NMOS loads."""
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
    """Problem 15: a telescopic cascode pair with two outputs."""
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
    """A two-stage op-amp: a mirror-loaded NMOS pair, then a PMOS stage.

    Its stages carry 200 µA each, biased by 0.7 V on its NMOS sources, its output
    swings from 0.2 V to 4.7 V, and a Miller capacitor of 1 pF sets its bandwidth. Its
    output holds 1 pF, as a real one would: without it, nothing holds the output
    and its stage's input together as they swing, and no time step converges.
    """
    p = ports(c, "inp", "inn", "out")
    tail, near, far, bias = c.net("tail"), c.net("near"), c.net("far"), c.net("bias")
    level(c, "VB", 0.7, bias, p["ground"])
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
    placed = c.place(opamp(), id="A1")
    c.connect(placed.vdd, p["vdd"])
    c.connect(placed.ground, p["ground"])
    c.connect(placed.inp, inp)
    c.connect(placed.inn, inn)
    c.connect(placed.out, out)


def phase_shift(c: Builder) -> None:
    """Problem 16: an inverting amplifier of gain 33 behind three RC sections."""
    p = ports(c, "vout")
    reference, inn = c.net("reference"), c.net("inn")
    level(c, "VR", 2.5, reference, p["ground"])
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
    """Problem 17: a Wien bridge around a non-inverting amplifier of gain 3.3."""
    p = ports(c, "vout")
    reference, inp, inn, series = (
        c.net("reference"),
        c.net("inp"),
        c.net("inn"),
        c.net("series"),
    )
    level(c, "VR", 2.5, reference, p["ground"])
    amplified(c, p, inp, inn, p["vout"])
    resistor(c, "RS", 10, p["vout"], series)
    capacitor(c, "CS", devices.capacitor(1 * nF), series, inp)
    resistor(c, "RP", 10, inp, reference)
    capacitor(c, "CP", devices.capacitor(1 * nF), inp, reference)
    resistor(c, "RF", 23, p["vout"], inn)
    resistor(c, "RG", 10, inn, reference)


def integrator(c: Builder) -> None:
    """Problem 18: R1 of 10 kΩ into Cf of 3 µF, 10 MΩ across Cf for its bias."""
    p = ports(c, "vin", "vout")
    reference, inn = c.net("reference"), c.net("inn")
    level(c, "VR", 2.5, reference, p["ground"])
    amplified(c, p, reference, inn, p["vout"])
    resistor(c, "R1", 10, p["vin"], inn)
    capacitor(c, "CF", devices.capacitor(3 * uF), inn, p["vout"])
    two_terminal(c, "RDC", devices.resistor(10 * Mohm), inn, p["vout"])


def differentiator(c: Builder) -> None:
    """Problem 19: C1 of 3 µF into Rf of 10 kΩ, through 100 Ω.

    Without the 100 Ω, the op-amp's own pole and C1's into Rf would leave its loop no
    phase margin, and it would ring.
    """
    p = ports(c, "vin", "vout")
    reference, inn, series = c.net("reference"), c.net("inn"), c.net("series")
    level(c, "VR", 2.5, reference, p["ground"])
    amplified(c, p, reference, inn, p["vout"])
    capacitor(c, "C1", devices.capacitor(3 * uF), p["vin"], series)
    two_terminal(c, "RS", devices.resistor(100 * ohm), series, inn)
    resistor(c, "RF", 10, inn, p["vout"])


def adder(c: Builder) -> None:
    """Problem 20: an inverting adder of two 10 kΩ inputs around 2.5 V."""
    p = ports(c, "vin1", "vin2", "vout")
    reference, inn = c.net("reference"), c.net("inn")
    level(c, "VR", 2.5, reference, p["ground"])
    amplified(c, p, reference, inn, p["vout"])
    resistor(c, "R1", 10, p["vin1"], inn)
    resistor(c, "R2", 10, p["vin2"], inn)
    resistor(c, "RF", 10, inn, p["vout"])


def subtractor(c: Builder) -> None:
    """Problem 21: a difference amplifier of four 10 kΩ resistors."""
    p = ports(c, "vin1", "vin2", "vout")
    inp, inn = c.net("inp"), c.net("inn")
    amplified(c, p, inp, inn, p["vout"])
    resistor(c, "R1", 10, p["vin1"], inn)
    resistor(c, "R2", 10, inn, p["vout"])
    resistor(c, "R3", 10, p["vin2"], inp)
    resistor(c, "R4", 10, inp, p["ground"])


def schmitt(c: Builder) -> None:
    """Problem 22: positive feedback from 20 kΩ against a 10 kΩ input, around 2.5 V."""
    p = ports(c, "vin", "vout")
    inp, inn = c.net("inp"), c.net("inn")
    amplified(c, p, inp, inn, p["vout"])
    resistor(c, "R1", 10, p["vin"], inp)
    resistor(c, "RF", 20, inp, p["vout"])
    resistor(c, "RA", 10, p["vdd"], inn)
    resistor(c, "RB", 10, inn, p["ground"])


def inverter(c: Builder, p: dict[str, Net], id: str, gate: Net, drain: Net) -> None:
    """A CMOS inverter of strong transistors, from `gate` to `drain`, which holds 50 fF.

    The transistors' own capacitance would hold the output; without any, the output
    would jump as the input crossed the middle, and no time step would converge.
    """
    mosfet(c, f"{id}N", (drain, gate, p["ground"]), nmos(10))
    mosfet(c, f"{id}P", (drain, gate, p["vdd"]), pmos(20))
    capacitor(c, f"{id}C", devices.capacitor(50 * fF), drain, p["ground"])


@component
def ring(c: Builder, capacitance: float) -> None:
    """A ring of three inverters, each starved by a current `control` sets.

    `control` drives an NMOS of 2 µm by 5 µm under each inverter, and a mirror of the
    same current a PMOS above it; each inverter's output holds `capacitance`, in farads,
    and ``out`` buffers the third's. The first holds 20% more: a ring of three equal
    stages would rest at its balance as its supply switches on, with no noise to start
    it. The nodes between the transistors of each stage hold 10 fF, as the
    transistors' own capacitance would: with the current starved, nothing else would
    hold them.
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
    placed = c.place(ring(capacitance), id="VCO")
    c.connect(placed.vdd, p["vdd"])
    c.connect(placed.ground, p["ground"])
    c.connect(placed.control, control)
    c.connect(placed.out, out)


def vco(c: Builder) -> None:
    """Problem 23: a current-starved ring oscillator, near 2 MHz."""
    p = ports(c, "vin", "vout")
    placed_ring(c, p, 10e-12, p["vin"], p["vout"])


def pll(c: Builder) -> None:
    """Problem 24: an XOR phase detector, a lag-lead filter and a ring oscillator.

    The ring runs near 11 MHz with its control at mid-supply, where the detector's
    average rests while the loop is unlocked.
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
"""Each problem's design, by number."""

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
"""Each design's input levels, by problem."""


def build(number: int) -> Component:
    """The design of problem `number`."""
    function = BUILDERS[number]
    return component(function)()

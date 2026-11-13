# Copyright 2026 Maxence Faldor
"""The candidates for a two-terminal current source, each a component of ports a and b.

Each candidate takes its resistors as arguments, so the experiments can tune them.
"""

from jolty import Builder, Component, Role, component, devices
from jolty.parts import q2n3904, q2n3906
from jolty.units import V, kohm, ohm


@component
def resistor_only(c: Builder, resistance: float = 12e3) -> None:
    """A resistor: the baseline, its current Ohm's law."""
    a, b = c.port("a"), c.port("b")
    load = c.place(devices.resistor(resistance * ohm), id="R1")
    c.connect(a, load.a)
    c.connect(load.b, b)


@component
def limiter(c: Builder, sense: float = 650.0, bias: float = 100e3) -> None:
    """The two-transistor limiter: Q2 steals Q1's base drive once Rs holds a Vbe."""
    a, b = c.port("a"), c.port("b")
    first, second = c.place(q2n3904(), id="Q1"), c.place(q2n3904(), id="Q2")
    feed = c.place(devices.resistor(bias * ohm), id="RB")
    resistor = c.place(devices.resistor(sense * ohm), id="RS")
    c.connect(a, feed.a, first.collector)
    c.connect(feed.b, first.base, second.collector, name="drive")
    c.connect(first.emitter, second.base, resistor.a, name="sensed")
    c.connect(resistor.b, second.emitter, b)


@component
def ptat(c: Builder, resistance: float = 1.9e3, ratio: int = 4) -> None:
    """A self-biased ΔVbe reference, its current proportional to absolute temperature.

    A PNP mirror forces equal currents into one NPN and `ratio` in parallel over R, so
    the current is Vt·ln(ratio)/R. A megohm starts it.
    """
    a, b = c.port("a"), c.port("b")
    left, right = c.place(q2n3906(), id="QP1"), c.place(q2n3906(), id="QP2")
    single = c.place(q2n3904(), id="QN1")
    wide = [c.place(q2n3904(), id=f"QN{index}") for index in range(2, 2 + ratio)]
    degeneration = c.place(devices.resistor(resistance * ohm), id="R1")
    start = c.place(devices.resistor(1e6 * ohm), id="RS")
    c.connect(a, left.emitter, right.emitter, start.a)
    c.connect(
        left.base,
        right.base,
        right.collector,
        *(q.collector for q in wide),
        name="mirror",
    )
    c.connect(
        left.collector,
        single.collector,
        single.base,
        *(q.base for q in wide),
        start.b,
        name="gate",
    )
    c.connect(*(q.emitter for q in wide), degeneration.a, name="tail")
    c.connect(single.emitter, degeneration.b, b)


@component
def bench(c: Builder, candidate: Component, volts: float = 12.0) -> None:
    """The candidate across a source of `volts`, a to its positive end."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(volts * V), id="V1")
    placed = c.place(candidate, id="X1")
    c.connect(source.positive, placed.a, name="a")
    c.connect(placed.b, source.negative, ground)


@component
def mirrored_limiter(
    c: Builder, sense: float = 650.0, share: float = 10.0, top: float = 100.0
) -> None:
    """The limiter, its drive fed by a PNP mirror of its own current, not a resistor.

    A diode-connected PNP carries the whole current from a; a second PNP, its emitter
    resistor `share` times larger, mirrors a fixed fraction of it into Q1's base and
    Q2's collector, so Q2's current, and so its Vbe, no longer follows the line. A
    megohm starts it.
    """
    a, b = c.port("a"), c.port("b")
    main, tap = c.place(q2n3906(), id="QP1"), c.place(q2n3906(), id="QP2")
    upper = c.place(devices.resistor(top * ohm), id="RE1")
    lower = c.place(devices.resistor(top * share * ohm), id="RE2")
    first, second = c.place(q2n3904(), id="Q1"), c.place(q2n3904(), id="Q2")
    resistor = c.place(devices.resistor(sense * ohm), id="RS")
    start = c.place(devices.resistor(1 * kohm * 1000), id="RST")
    c.connect(a, upper.a, lower.a, start.a)
    c.connect(upper.b, main.emitter, name="top")
    c.connect(lower.b, tap.emitter, name="tap")
    c.connect(main.base, main.collector, tap.base, first.collector, name="mirror")
    c.connect(tap.collector, first.base, second.collector, start.b, name="drive")
    c.connect(first.emitter, second.base, resistor.a, name="sensed")
    c.connect(resistor.b, second.emitter, b)


@component
def compensated(
    c: Builder, sense: float = 1300.0, bias: float = 1e6, resistance: float = 260.0
) -> None:
    """A limiter and a ΔVbe cell in parallel, their drifts cancelling.

    About half the current falls with temperature and half rises.
    """
    a, b = c.port("a"), c.port("b")
    falling = c.place(limiter(sense, bias), id="XCTAT")
    rising = c.place(ptat(resistance, 4), id="XPTAT")
    c.connect(a, falling.a, rising.a)
    c.connect(b, falling.b, rising.b)

# Copyright 2026 Maxence Faldor
"""Two-terminal current source candidates exposing ports ``a`` and ``b``.

Each candidate accepts resistor values as parameters for tuning experiments.
"""

from jolty import Builder, Component, Role, component, devices
from jolty.parts import q2n3904, q2n3906
from jolty.units import V, kohm, ohm


@component
def resistor_only(c: Builder, resistance: float = 12e3) -> None:
    """Builds a single-resistor baseline whose current follows Ohm's law."""
    a, b = c.port("a"), c.port("b")
    load = c.place(devices.resistor(resistance * ohm), name="R1")
    c.connect(a, load.a)
    c.connect(load.b, b)


@component
def limiter(c: Builder, sense: float = 650.0, bias: float = 100e3) -> None:
    """Builds a two-transistor limiter where Q2 shunts Q1 base drive at one Vbe."""
    a, b = c.port("a"), c.port("b")
    first, second = c.place(q2n3904(), name="Q1"), c.place(q2n3904(), name="Q2")
    feed = c.place(devices.resistor(bias * ohm), name="RB")
    resistor = c.place(devices.resistor(sense * ohm), name="RS")
    c.connect(a, feed.a, first.collector)
    c.connect(feed.b, first.base, second.collector, name="drive")
    c.connect(first.emitter, second.base, resistor.a, name="sensed")
    c.connect(resistor.b, second.emitter, b)


@component
def ptat(c: Builder, resistance: float = 1.9e3, ratio: int = 4) -> None:
    """Builds a self-biased ΔVbe current source proportional to absolute temperature.

    A PNP mirror equalizes currents between one NPN and `ratio` parallel NPNs over R,
    producing Vt·ln(ratio)/R with a 1 MΩ startup resistor.
    """
    a, b = c.port("a"), c.port("b")
    left, right = c.place(q2n3906(), name="QP1"), c.place(q2n3906(), name="QP2")
    single = c.place(q2n3904(), name="QN1")
    wide = [c.place(q2n3904(), name=f"QN{index}") for index in range(2, 2 + ratio)]
    degeneration = c.place(devices.resistor(resistance * ohm), name="R1")
    start = c.place(devices.resistor(1e6 * ohm), name="RS")
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
    """Builds a test bench placing `candidate` across a `volts` DC source."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(volts * V), name="V1")
    placed = c.place(candidate, name="X1")
    c.connect(source.positive, placed.a, name="a")
    c.connect(placed.b, source.negative, ground)


@component
def mirrored_limiter(
    c: Builder, sense: float = 650.0, share: float = 10.0, top: float = 100.0
) -> None:
    """Builds a limiter driven by a PNP current mirror instead of a bias resistor.

    A diode-connected PNP carries total current while a second PNP with a `share`-times
    larger emitter resistor mirrors a fixed fraction into Q1 base and Q2 collector.
    Decoupling Q2 bias from the supply stabilizes Vbe across supply voltage, aided by a
    1 MΩ startup resistor.
    """
    a, b = c.port("a"), c.port("b")
    main, tap = c.place(q2n3906(), name="QP1"), c.place(q2n3906(), name="QP2")
    upper = c.place(devices.resistor(top * ohm), name="RE1")
    lower = c.place(devices.resistor(top * share * ohm), name="RE2")
    first, second = c.place(q2n3904(), name="Q1"), c.place(q2n3904(), name="Q2")
    resistor = c.place(devices.resistor(sense * ohm), name="RS")
    start = c.place(devices.resistor(1 * kohm * 1000), name="RST")
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
    """Builds a limiter and a ΔVbe cell in parallel to cancel temperature drift.

    Roughly half of the output current has negative drift and half has positive drift.
    """
    a, b = c.port("a"), c.port("b")
    falling = c.place(limiter(sense, bias), name="XCTAT")
    rising = c.place(ptat(resistance, 4), name="XPTAT")
    c.connect(a, falling.a, rising.a)
    c.connect(b, falling.b, rising.b)

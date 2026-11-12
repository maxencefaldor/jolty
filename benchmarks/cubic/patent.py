# Copyright 2026 Maxence Faldor
"""The patented circuit, the scorer's reference: US 6,160,427, FIG. 6.

Cipriani and Takeshian's compact cubic function generator, reconstructed from the
patent's text: diodes D4 and D5 carry the reference current, so Q4 and Q5, stacked
across them, carry currents whose product is its square; diodes D6 and D7 carry the
input, Q6 mirrors it from Q4, so Q5 carries the reference's square over the input, and
Q7 and Q8, stacked across D6 and D7, leave Q8 the input's cube over the reference's
square. No voltage drop crosses more than two transistors. Every diode is a
diode-connected transistor, as on the patent's chip.
"""

import runpy
from pathlib import Path

from jolty import Builder, Role, component

npn = runpy.run_path(str(Path(__file__).resolve().parent / "given" / "parts.py"))["npn"]
"""The given transistor, which every transistor and diode of the patent is."""


@component
def cubic(c: Builder) -> None:
    """The patented cubic function generator, its ports the task's."""
    supply = c.port("supply", Role.SUPPLY)
    reference = c.port("reference", Role.INPUT)
    signal = c.port("input", Role.INPUT)
    output = c.port("output", Role.OUTPUT)
    ground = c.port("ground", Role.GROUND)
    q4, q5, q6, q7, q8 = (c.place(npn(), id=f"Q{n}") for n in (4, 5, 6, 7, 8))
    d4, d5, d6, d7 = (c.place(npn(), id=f"D{n}") for n in (4, 5, 6, 7))
    c.connect(reference, d4.collector, d4.base, q4.base)
    c.connect(d4.emitter, d5.collector, d5.base, name="reference_middle")
    c.connect(signal, d6.collector, d6.base, q7.base)
    c.connect(d6.emitter, d7.collector, d7.base, q6.base, name="input_middle")
    c.connect(q4.emitter, q5.base, q6.collector, name="square")
    c.connect(q5.collector, q7.emitter, q8.base, name="cube")
    c.connect(supply, q4.collector, q7.collector)
    c.connect(output, q8.collector)
    c.connect(d5.emitter, d7.emitter, q5.emitter, q6.emitter, q8.emitter, ground)

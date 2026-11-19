# Copyright 2026 Maxence Faldor
"""Reference patented cubic function generator reconstructed from US 6,160,427, FIG. 6.

Cipriani and Takeshian's compact translinear generator biases diodes D4 and D5 with
the reference current so stacked transistors Q4 and Q5 maintain an emitter current
product equal to the squared reference current. Diodes D6 and D7 carry the input
current while Q6 mirrors input current into Q4, forcing Q5 to carry the squared
reference current divided by the input current. Stacked transistors Q7 and Q8 across D6
and D7 then produce an output current in Q8 equal to the input cube divided by the
squared reference current. No branch spans more than two base-emitter drops, and every
diode uses a diode-connected NPN transistor as on the patented die.
"""

import runpy
from pathlib import Path

from jolty import Builder, Role, component

npn = runpy.run_path(str(Path(__file__).resolve().parent / "given" / "parts.py"))["npn"]
"""Given integrated NPN transistor model used for every transistor and diode."""


@component
def cubic(c: Builder) -> None:
    """Builds the patented cubic function generator with the benchmark task ports."""
    supply = c.port("supply", Role.SUPPLY)
    reference = c.port("reference", Role.INPUT)
    signal = c.port("input", Role.INPUT)
    output = c.port("output", Role.OUTPUT)
    ground = c.port("ground", Role.GROUND)
    q4, q5, q6, q7, q8 = (c.place(npn(), name=f"Q{n}") for n in (4, 5, 6, 7, 8))
    d4, d5, d6, d7 = (c.place(npn(), name=f"D{n}") for n in (4, 5, 6, 7))
    c.connect(reference, d4.collector, d4.base, q4.base)
    c.connect(d4.emitter, d5.collector, d5.base, name="reference_middle")
    c.connect(signal, d6.collector, d6.base, q7.base)
    c.connect(d6.emitter, d7.collector, d7.base, q6.base, name="input_middle")
    c.connect(q4.emitter, q5.base, q6.collector, name="square")
    c.connect(q5.collector, q7.emitter, q8.base, name="cube")
    c.connect(supply, q4.collector, q7.collector)
    c.connect(output, q8.collector)
    c.connect(d5.emitter, d7.emitter, q5.emitter, q6.emitter, q8.emitter, ground)

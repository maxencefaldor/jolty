# Copyright 2026 Maxence Faldor
"""The voltage-current converter task's scorer, which the agent never sees.

It places the agent's component on Koza's test fixture for the voltage-current converter
of US patent 6,166,529 (Genetic Programming IV, sections 15.3.1.4 and 15.3.5.4) and
runs his four fitness cases: one second each, the output current read at 31 times, 1/30
s apart. Each case's mean absolute error against 1 mA/V times the inputs' difference is
divided by the patented circuit's on that case, as Koza measured it: 41, 36, 26 and 44
µA, which he gives in microvolts, as many across the 1 Ω load. The fitness is the four
ratios' mean, as Koza weighted it, so the patent's is 1, and the circuit genetic
programming evolved reached 0.619. The work meets the task when it builds from the
allowed parts alone, its fitness is at most 1, and its output current settles within
40 µs after ``VS`` steps by 10 mV at three points of its range: a check Koza's fitness
lacks, since its steps of 1/30 s cannot see an oscillation.

Koza's added input is ``plus`` and his subtracted one ``minus``; his evolved circuit
sources a current from the first and sinks one into the second, so the output current
flows out of ``output`` when ``plus`` is the higher.
"""

import runpy
import sys
from pathlib import Path

import numpy as np
from scoring import count_devices, disallowed
from verdict import Verdict

from jolty import Builder, Component, Role, Simulation, circuit, component, devices
from jolty.plan import Transient
from jolty.units import Hz, V, ns, ohm, s, us
from jolty.waveforms import Piecewise, Pulse, Sine

TRANSCONDUCTANCE = 1e-3
"""The output current per volt of the inputs' difference, in amperes per volt."""
TIMES = 31
"""The times each case is read at, 1/30 s apart from 0 s to 1 s."""
PATENT = (41e-6, 36e-6, 26e-6, 44e-6)
"""The patented circuit's mean absolute error in each case, in amperes, per Koza."""
EVOLVED = 0.619
"""The fitness of the circuit genetic programming evolved, as Koza found."""
ALLOWED = {"resistor", "capacitor"}
"""The devices the design may hold besides the given transistors."""
STEP = 10e-3
"""The step of ``VS`` the output current must settle after, in volts."""
SETTLING = ((6.0, 4.0, 0.0), (10.0, 2.0, 1.0), (2.0, 10.0, -1.0))
"""Where it must settle: ``plus``, ``minus`` and ``VS`` before the step, in volts."""
UNSETTLED = 1e-6
"""The swing of the output current, in amperes, beyond which it has not settled."""
SETTLED = 40e-6
"""The time from which the output current must have settled, in seconds."""


def ramp(start: float, end: float) -> devices.Device:
    """A source that moves linearly from `start` to `end` volts over one second."""
    return devices.voltage_source(Piecewise([(0 * s, start * V), (1 * s, end * V)]))


def sine(middle: float, swing: float, frequency: float) -> devices.Device:
    """A source that swings `swing` volts about `middle` at `frequency` hertz."""
    return devices.voltage_source(
        Sine(offset=middle * V, amplitude=swing * V, frequency=frequency * Hz)
    )


def cases() -> list[tuple[devices.Device, devices.Device]]:
    """Koza's four fitness cases: the ``plus`` source, then the ``minus`` source."""
    return [
        (ramp(10, 2), ramp(2, 10)),
        (sine(5.5, 3.5, 4.5), sine(6, 3.5, 6.4)),
        (sine(6, 4, 2), sine(6, 4, 2)),
        (sine(4, 1, 4.5), sine(4, 1, 6.4)),
    ]


def swing() -> devices.Device:
    """Koza's ``VS``: 1 V about 0 V at 3.2 Hz."""
    return devices.voltage_source(
        Sine(offset=0 * V, amplitude=1 * V, frequency=3.2 * Hz)
    )


def step(level: float) -> devices.Device:
    """A ``VS`` that steps from `level` to 10 mV above it at 1 µs, over 1 ns."""
    return devices.voltage_source(
        Pulse(
            initial=level * V,
            pulsed=(level + STEP) * V,
            delay=1 * us,
            rise=1 * ns,
            fall=1 * ns,
            width=1 * s,
        )
    )


@component
def bench(
    c: Builder,
    made: Component,
    inputs: tuple[devices.Device, devices.Device],
    output: devices.Device,
) -> None:
    """`made` on Koza's fixture, driven by `inputs` and `output`.

    `inputs` drive ``plus`` and ``minus``, and `output` is ``VS``.
    """
    ground = c.port("ground", Role.GROUND)
    placed = c.place(made, id="X1")
    supply = c.place(devices.voltage_source(-15 * V), id="VEE")
    c.connect(supply.positive, placed.vee)
    c.connect(supply.negative, placed.ground, ground)
    for name, source in zip(("plus", "minus"), inputs, strict=True):
        driven = c.place(source, id=f"V{name.upper()}")
        series = c.place(devices.resistor(1e-6 * ohm), id=f"R{name.upper()}")
        c.connect(driven.positive, series.a, name=name)
        c.connect(series.b, placed[name])
        c.connect(driven.negative, ground)
    held = c.place(output, id="VS")
    load = c.place(devices.resistor(1 * ohm), id="RLOAD")
    c.connect(placed.output, held.positive, name="output")
    c.connect(held.negative, load.a, name="load")
    c.connect(load.b, ground)


def errors(made: Component) -> list[float]:
    """The mean absolute error of `made`'s output current in each case, in amperes."""
    times = np.linspace(0.0, 1.0, TIMES)
    found = []
    for inputs in cases():
        waveforms = Simulation(circuit(bench(made, inputs, swing()))).run(
            Transient(
                stop=1 * s,
                max_step=(1 / (10 * (TIMES - 1))) * s,
                times=[time * s for time in times[1:]],
            )
        )

        def read(values: np.ndarray, at: np.ndarray = waveforms.times) -> np.ndarray:
            return np.interp(times, at, values)

        output = read(waveforms.current("RLOAD", "a"))
        wanted = TRANSCONDUCTANCE * (
            read(waveforms.voltage("plus")) - read(waveforms.voltage("minus"))
        )
        found.append(float(np.mean(np.abs(output - wanted))))
    return found


def unsettled(made: Component) -> list[str]:
    """Where `made`'s output current does not settle after ``VS`` steps by 10 mV.

    The fitness cases' steps are milliseconds long, too long to see an oscillation; this
    reads the current over the last 10 µs of 50, at steps of 10 ns.
    """
    found = []
    for plus, minus, level in SETTLING:
        inputs = (devices.voltage_source(plus * V), devices.voltage_source(minus * V))
        waveforms = Simulation(circuit(bench(made, inputs, step(level)))).run(
            Transient(stop=50 * us, max_step=10 * ns)
        )
        late = waveforms.times >= SETTLED
        current = waveforms.current("RLOAD", "a")[late]
        swing = float(np.ptp(current))
        if swing > UNSETTLED:
            found.append(
                f"with plus at {plus} V, minus at {minus} V and VS stepped from "
                f"{level} V, the output current still swings by {swing:.3g} A"
            )
    return found


def score(task: Path, work: Path) -> Verdict:
    """The verdict on the voltage-current converter in `work`."""
    parts = runpy.run_path(str(task / "given" / "parts.py"))
    given = [parts["npn"](), parts["pnp"]()]
    design = work / "design.py"
    if not design.is_file():
        return Verdict(met=False, reasons=(f"there is no {design}",))
    sys.path.insert(0, str(work))
    made = runpy.run_path(str(design))["build"]()
    if not isinstance(made, Component):
        return Verdict(met=False, reasons=("build() returns no component",))
    built = circuit(bench(made, cases()[0], swing()))
    reasons = disallowed(built, given, ALLOWED)
    reasons += unsettled(made)
    found = errors(made)
    fitness = float(
        np.mean([mine / theirs for mine, theirs in zip(found, PATENT, strict=True)])
    )
    if fitness > 1:
        reasons.append(f"the fitness is {fitness:.4g}, more than the patent's 1")
    measurements = {
        "fitness": fitness,
        "evolved fitness": EVOLVED,
        "devices": float(count_devices(built, "X1")),
    }
    for case, (mine, theirs) in enumerate(zip(found, PATENT, strict=True), start=1):
        measurements[f"error {case}"] = mine
        measurements[f"patent's error {case}"] = theirs
    return Verdict(met=not reasons, measurements=measurements, reasons=tuple(reasons))

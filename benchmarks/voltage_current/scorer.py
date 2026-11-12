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
allowed parts alone and its fitness is at most 1.

Koza's added input is ``plus`` and his subtracted one ``minus``; his evolved circuit
sources a current from the first and sinks one into the second, so the output current
flows out of ``output`` when ``plus`` is the higher.
"""

import runpy
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scoring import count_devices
from verdict import Verdict

from jolty import Builder, Component, Role, Simulation, circuit, component, devices, ir
from jolty.plan import Transient
from jolty.units import Hz, V, ohm, s
from jolty.waveforms import Piecewise, Sine

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


@component
def bench(c: Builder, made: Component, case: int) -> None:
    """`made` on Koza's fixture, its inputs driven as in `case`."""
    ground = c.port("ground", Role.GROUND)
    placed = c.place(made, id="X1")
    supply = c.place(devices.voltage_source(-15 * V), id="VEE")
    c.connect(supply.positive, placed.vee)
    c.connect(supply.negative, placed.ground, ground)
    for name, source in zip(("plus", "minus"), cases()[case], strict=True):
        driven = c.place(source, id=f"V{name.upper()}")
        series = c.place(devices.resistor(1e-6 * ohm), id=f"R{name.upper()}")
        c.connect(driven.positive, series.a, name=name)
        c.connect(series.b, placed[name])
        c.connect(driven.negative, ground)
    swing = c.place(
        devices.voltage_source(Sine(offset=0 * V, amplitude=1 * V, frequency=3.2 * Hz)),
        id="VS",
    )
    load = c.place(devices.resistor(1 * ohm), id="RLOAD")
    c.connect(placed.output, swing.positive, name="output")
    c.connect(swing.negative, load.a, name="load")
    c.connect(load.b, ground)


def errors(made: Component) -> list[float]:
    """The mean absolute error of `made`'s output current in each case, in amperes."""
    times = np.linspace(0.0, 1.0, TIMES)
    found = []
    for case in range(len(PATENT)):
        waveforms = Simulation(circuit(bench(made, case))).run(
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


def disallowed(built: ir.Circuit, given: Mapping[str, devices.Device]) -> list[str]:
    """The devices of the design that are neither a given transistor nor allowed."""
    transistors = [(part.model, part.parameters) for part in given.values()]
    found = []
    for name, made in built.components.items():
        if name == built.top:
            continue
        for instance in made.instances:
            if not isinstance(instance.of, ir.DeviceOf):
                continue
            kind = instance.of.device
            if kind == "bipolar_transistor":
                if (kind, instance.parameters) not in transistors:
                    found.append(f"{instance.id} is not a given transistor")
            elif kind not in ALLOWED:
                found.append(f"{instance.id} is a {kind}, which the task forbids")
    return found


def score(task: Path, work: Path) -> Verdict:
    """The verdict on the voltage-current converter in `work`."""
    parts = runpy.run_path(str(task / "given" / "parts.py"))
    given = {"npn": parts["npn"](), "pnp": parts["pnp"]()}
    design = work / "design.py"
    if not design.is_file():
        return Verdict(met=False, reasons=(f"there is no {design}",))
    sys.path.insert(0, str(work))
    made = runpy.run_path(str(design))["build"]()
    if not isinstance(made, Component):
        return Verdict(met=False, reasons=("build() returns no component",))
    built = circuit(bench(made, 0))
    reasons = disallowed(built, given)
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

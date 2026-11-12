# Copyright 2026 Maxence Faldor
"""The cubic task's scorer, which the agent never sees.

It places the agent's component on the task's bench and measures, in four cases, the
mean absolute error of its output current against the input's cube over the reference's
square: a DC sweep, and a sine at 1 MHz, 100 MHz and 1 GHz. Each error is divided by the
patented circuit's, ``patent.py``, on the same case, as Koza weighted his fitness cases,
and the score is their mean. The work meets the task when it builds from the allowed
parts alone and scores at most 1.
"""

import runpy
import sys
from pathlib import Path

import numpy as np
from scoring import count_devices
from verdict import Verdict

from jolty import Builder, Component, Role, Simulation, circuit, component, devices, ir
from jolty.plan import DcSweep, Transient, linear
from jolty.units import A, Hz, V, s, uA
from jolty.waveforms import Sine

REFERENCE = 100e-6
"""The reference current, in amperes."""
FREQUENCIES = (1e6, 1e8, 1e9)
"""The sine cases' frequencies, in hertz."""
ALLOWED = {"resistor", "capacitor"}
"""The devices the design may hold besides the given transistor."""


@component
def bench(c: Builder, made: Component, frequency: float = 0.0) -> None:
    """`made` from 2 V, its input a sine at `frequency`, or 0 A to sweep when 0."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(2 * V), id="VS")
    level = (
        Sine(offset=100 * uA, amplitude=90 * uA, frequency=frequency * Hz)
        if frequency
        else 0 * A
    )
    signal = c.place(devices.current_source(level), id="IIN")
    reference = c.place(devices.current_source(REFERENCE * A), id="IREF")
    meter = c.place(devices.voltage_source(0 * V), id="VM")
    placed = c.place(made, id="X1")
    c.connect(
        supply.positive,
        signal.positive,
        reference.positive,
        meter.positive,
        placed.supply,
        name="supply",
    )
    c.connect(signal.negative, placed.input, name="input")
    c.connect(reference.negative, placed.reference, name="reference")
    c.connect(meter.negative, placed.output, name="output")
    c.connect(supply.negative, placed.ground, ground)


def errors(made: Component) -> list[float]:
    """The mean absolute error of `made`'s output in each case, in amperes."""
    swept = Simulation(circuit(bench(made))).run(
        DcSweep("IIN", linear(0 * A, 200 * uA, 41))
    )
    found = [_error(swept.levels, np.abs(swept.current("VM")))]
    for frequency in FREQUENCIES:
        period = 1 / frequency
        waveforms = Simulation(circuit(bench(made, frequency))).run(
            Transient(stop=4 * period * s, max_step=period / 200 * s)
        )
        late = waveforms.times >= 2 * period
        level = 100e-6 + 90e-6 * np.sin(2 * np.pi * frequency * waveforms.times)
        found.append(_error(level[late], np.abs(waveforms.current("VM"))[late]))
    return found


def _error(level: np.ndarray, output: np.ndarray) -> float:
    return float(np.mean(np.abs(output - level**3 / REFERENCE**2)))


def disallowed(built: ir.Circuit, transistor: devices.Device) -> list[str]:
    """The devices of the design that are neither `transistor` nor allowed."""
    found = []
    for name, made in built.components.items():
        if name == built.top:
            continue
        for instance in made.instances:
            if not isinstance(instance.of, ir.DeviceOf):
                continue
            kind = instance.of.device
            if kind == transistor.model:
                if instance.parameters != transistor.parameters:
                    found.append(f"{instance.id} is not the given transistor")
            elif kind not in ALLOWED:
                found.append(f"{instance.id} is a {kind}, which the task forbids")
    return found


def score(task: Path, work: Path) -> Verdict:
    """The verdict on the cubic `work` against `task`."""
    patent = runpy.run_path(str(task / "patent.py"))["cubic"]()
    transistor = runpy.run_path(str(task / "given" / "parts.py"))["npn"]()
    design = work / "design.py"
    if not design.is_file():
        return Verdict(met=False, reasons=(f"there is no {design}",))
    sys.path.insert(0, str(work))
    made = runpy.run_path(str(design))["build"]()
    if not isinstance(made, Component):
        return Verdict(met=False, reasons=("build() returns no component",))
    built = circuit(bench(made))
    reasons = disallowed(built, transistor)
    reference, found = errors(patent), errors(made)
    ratios = [mine / theirs for mine, theirs in zip(found, reference, strict=True)]
    total = float(np.mean(ratios))
    if total > 1:
        reasons.append(f"the score is {total:.4g}, more than the patent's 1")
    names = ["dc", *(f"{frequency:.0e} Hz" for frequency in FREQUENCIES)]
    measurements = {"score": total, "devices": float(count_devices(built, "X1"))}
    for name, mine, theirs in zip(names, found, reference, strict=True):
        measurements[f"error {name}"] = mine
        measurements[f"patent's error {name}"] = theirs
    return Verdict(met=not reasons, measurements=measurements, reasons=tuple(reasons))

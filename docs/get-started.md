# Get started

The page installs jolty, writes a first circuit, simulates it and reads the results.

## Install

jolty builds from source.
It needs [uv](https://docs.astral.sh/uv/), which installs Python 3.15, and a [Rust toolchain](https://rustup.rs/), which builds the engine:

```sh
git clone https://github.com/maxencefaldor/jolty
cd jolty
uv sync
```

The command `uv sync` builds the engine, installs jolty and every tool the repository uses, and rebuilds the engine whenever its Rust sources change.
Commands then run through uv, such as `uv run jolty --version`.

## Write a circuit

A script defines `build()`, which returns the top component of the circuit.
The script below is a first-order low-pass filter: a resistor in series and a capacitor to ground, driven by a step.

```python title="low_pass.py"
from jolty import Builder, Component, Plan, Role, component, devices
from jolty.plan import Ac, OperatingPoint, Transient, decade
from jolty.units import MHz, Quantity, V, kHz, kohm, nF, ns, us
from jolty.waveforms import Piecewise


@component
def low_pass(c: Builder, resistance: Quantity, capacitance: Quantity) -> None:
    """A series resistor, then a capacitor to ground."""
    source = c.port("input")
    output = c.port("output")
    ground = c.port("ground", Role.GROUND)
    resistor = c.place(devices.resistor(resistance), id="R")
    capacitor = c.place(devices.capacitor(capacitance), id="C")
    c.connect(source, resistor.a)
    c.connect(resistor.b, capacitor.a, output)
    c.connect(capacitor.b, ground)


@component
def bench(c: Builder) -> None:
    """The filter, driven by a 1 V step at 1 µs."""
    ground = c.port("ground", Role.GROUND)
    step = Piecewise([(1 * us, 0 * V), (1 * us + 1 * ns, 1 * V)])
    source = c.place(devices.voltage_source(step), id="V1")
    stage = c.place(low_pass(1 * kohm, 1 * nF), id="X1")
    c.connect(source.positive, stage.input, name="in")
    c.connect(stage.output, name="out")
    c.connect(source.negative, stage.ground, ground)


def build() -> Component:
    """The filter on its bench."""
    return bench()


def plan() -> Plan:
    """The operating point, the step response and the frequency response."""
    return Plan(
        {
            "op": OperatingPoint(),
            "step": Transient(stop=10 * us),
            "bode": Ac("V1", decade(1 * kHz, 100 * MHz, 10)),
        }
    )
```

The component `low_pass` takes its resistance and capacitance as parameters, so `low_pass(1 * kohm, 1 * nF)` and `low_pass(2 * kohm, 1 * nF)` are two components.
The component `bench` places one of them as the instance `X1` and drives it.
The function `plan()` names three analyses; the [Simulation](guide/simulation.md) page describes each one.

## Build it

The command `jolty build` runs `build()`, saves the circuit beside the script as `low_pass.jolty.json`, and checks that the engine can simulate it:

```sh
uv run jolty build low_pass.py
```

```text
saved low_pass.jolty.json: 2 components, 3 devices, 3 unknowns

took 0.27 s
```

The engine solves for three unknowns: the voltages of the nets `in` and `out`, and the current through the source `V1`.
When the circuit has problems, the command lists every one, each with the script line that placed the instance, and exits with 1.

## Simulate it

The command `jolty simulate` builds the circuit, runs every analysis of `plan()`, saves the results as `low_pass.results.npz` and reports what each analysis found:

```sh
uv run jolty simulate low_pass.py
```

```text
step: transient to 10 µs in 67 steps, 0 rejected
              first  min        max       last
  in          0 V    0 V        1 V       1 V
  out         0 V    0 V        999.9 mV  999.9 mV
  V1:current  0 A    -999.5 µA  0 A       -119.6 nA
```

The output charges to 999.9 mV in the nine time constants after the step, and the source's current is negative while it delivers power.

## Read the results

The results file holds every value of every analysis.
`jolty.files.load_results` reads it back, and each result reads its unknowns by name:

```python
from pathlib import Path

from jolty.files import load_results

results = load_results(Path("low_pass.results.npz"))
step = results["step"]
times, out = step.times, step.voltage("out")
gain = abs(results["bode"].voltage("out"))
```

The [Results](guide/results.md) page describes each kind of result.

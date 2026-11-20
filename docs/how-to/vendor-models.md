# Read and simulate a vendor SPICE model

Import a vendor SPICE subcircuit or model card into jolty, connect its pins, and simulate it alongside native devices.

## 1. Load a SPICE file or text into a library

Use `spice.load("model.lib")` to read a file on disk, or `spice.Library().read(...)` for inline SPICE text.
To pin a downloadable vendor file by SHA-256 digest, fetch it first with `spice.fetch(url, digest)`.

```python
from jolty import spice

NETLIST = """
.subckt gain_stage in out
Rin in 0 10k
Eamp out 0 in 0 4
.ends
"""

library = spice.Library()
library.read(NETLIST, "gain_stage.lib")
names = library.subcircuits  # ('gain_stage',)
```

## 2. Place the subcircuit and wire its pins

Call `library.component(name, **parameters)` to turn a subcircuit into a placeable jolty component.
Ports keep the pin order declared on `.subckt`.
Subcircuits that reference SPICE node `0` expose an extra `spice.GROUND` port that you wire to circuit ground.

```python
from jolty import Builder, OperatingPoint, Role, Simulation, component, devices
from jolty.units import V, kohm

stage = library.component("gain_stage")


@component
def bench(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0.5 * V), name="V1")
    dut = c.place(stage, name="U1")
    load = c.place(devices.resistor(2 * kohm), name="RL")
    c.connect(source.positive, dut["in"], name="in")
    c.connect(dut["out"], load.a, name="out")
    c.connect(source.negative, dut[spice.GROUND], load.b, ground)


point = Simulation(bench()).run(OperatingPoint())
round(point.voltage("out"), 6)  # 2.0
```

## 3. Override subcircuit parameters

Pass keyword arguments to `library.component(name, ...)` to override `.subckt` defaults cleanly.

```python
PARAM_NETLIST = """
.subckt attenuator in out params: ratio=0.5
Rin in 0 10k
Eout out 0 in 0 {ratio}
.ends
"""

param_library = spice.Library()
param_library.read(PARAM_NETLIST, "attenuator.lib")
quarter = param_library.component("attenuator", ratio=0.25)


@component
def divider_bench(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(4 * V), name="V1")
    dut = c.place(quarter, name="U1")
    c.connect(source.positive, dut["in"], name="in")
    c.connect(dut["out"], name="out")
    c.connect(source.negative, dut[spice.GROUND], ground)


round(Simulation(divider_bench()).run(OperatingPoint()).voltage("out"), 6)  # 1.0
```

## 4. Handle unsupported syntax and verify against ngspice

Every unrecognized statement or model parameter raises a `SpiceError` naming the exact file and line number.
See [SPICE models](../guide/spice.md) for supported syntax, PSpice compatibility details, and open process design kits.

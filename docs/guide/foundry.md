# Foundry processes

A chip's transistors are its foundry's, and only the foundry's models say how they behave.
jolty places SkyWater's SKY130 transistors as foundry devices: the engine checks them but does not solve them, and ngspice simulates every analysis of a circuit that holds one with the process design kit's own models.

## Placing a transistor

`sky130.mosfet(transistor, width, length, fingers=1, multiplier=1)` places one of the kit's transistors, named by its model without the library's prefix, its width all its fingers together:

```python
from jolty import Builder, Role, component, devices
from jolty.spice import sky130
from jolty.units import V, um


@component
def inverter(c: Builder) -> None:
    ground = c.port("0", Role.GROUND)
    supply = c.place(devices.voltage_source(1.8 * V), name="VDD")
    lower = c.place(sky130.mosfet("nfet_01v8", 1 * um, 0.15 * um), name="MN")
    upper = c.place(sky130.mosfet("pfet_01v8", 2 * um, 0.15 * um, fingers=2), name="MP")
    c.connect(supply.positive, upper.source, upper.bulk, name="vdd")
    c.connect(lower.gate, upper.gate, name="in")
    c.connect(lower.drain, upper.drain, name="out")
    c.connect(supply.negative, lower.source, lower.bulk, ground)
```

Its terminals are `drain`, `gate`, `source` and `bulk`.
`sky130.TRANSISTORS` lists the transistors: the 1.8 V core ones, of standard, low and high threshold, the native ones of 3.3 V and 5 V, and those whose gates stand 5 V and whose drains stand 10.5 V or 16 V, as their names say.
The kit's models bin their sizes, so a width or a length that no bin holds fails when ngspice runs.

A build elaborates the circuit and checks its structure as any other, through the branches the transistor declares, a MOSFET's: the engine solves none of a foundry's devices, so the checks at the operating point do not run.

## Simulating in ngspice

`jolty simulate` and `Simulation.run` hand each analysis of a circuit that holds a foundry's device to ngspice, which reads the kit's model library and runs it at the plan's temperature, with its options, in its HSPICE compatibility mode, as the kit asks.
The circuit is written as a [SPICE netlist](spice.md#writing-spice), each transistor an instance of the kit's subcircuit, its sizes in micrometres, as the kit's library scales them, and its diffusions' areas, perimeters and squares estimated as the kit's xschem symbols estimate them, from a diffusion 0.29 µm long beside each finger.
ngspice reads the whole library, which takes seconds, before each analysis.

Its results are jolty's: reports, plots, specs and corners read them as they read the engine's.
They hold the circuit's nets and the currents of its sources and inductors, as jolty names them; a device's own unknowns and its terminals' currents ngspice keeps to itself, whatever the options [record](results.md#what-a-result-records), so a rating of a current or a power goes unjudged, and an operating point is found by `ngspice`.

An operating point, a DC sweep in even steps, a small-signal response, a transient and the noise at a voltage run this way.
A transient takes the steps ngspice chooses, and lands on the times its analysis names only where they fall.
A noise analysis gives each device's sources of noise as ngspice names them, such as a transistor's `id`, its channel's thermal noise, and `1overf`, its flicker noise; ngspice refers noise to a source, so it takes the circuit's first independent source when the analysis names none, and the result then leaves the referral out.
Any other analysis of such a circuit fails, and says so.

A netlist holds no level-1 or level-3 MOSFET beside a foundry's, since the kit scales the sizes of every MOSFET, and no foundry's device at a temperature of its own.

## Process corners

A plan's `process` is the corner of the kit's model library its foundry's devices run at, `tt`, the typical one, unless given, and each [corner](simulation.md#corners) may name its own:

```python
Plan(
    {"bias": OperatingPoint(), "gain": Ac("VIN", decade(1 * kHz, 1 * GHz, 10))},
    corners={
        "slow": Corner(process="ss", temperature=celsius(125)),
        "fast": Corner(process="ff", temperature=celsius(-40)),
    },
)
```

`sky130.CORNERS` lists them: `tt`, then `ff`, `ss`, `sf` and `fs`, fast and slow, the n-channel transistors first where they differ, and `ll`, `hh`, `hl` and `lh`, which move the resistors and capacitors too.
A process corner of a circuit that holds no foundry's device is an error.

## Mismatch

A [Monte Carlo](simulation.md#monte-carlo) analysis draws the kit's mismatch on each sample, as the kit models it at the corner's process: each transistor's threshold, oxide and mobility apart from the others', by how far its area lies from the kit's, from a seed of the sample's own, as well as the tolerances of jolty's own devices, which the netlist carries to ngspice.
A sample draws the same mismatch at every corner, so a corner runs the same transistors, and the report counts the foundry's devices whose mismatch each sample draws.
`Simulation(circuit, mismatch=seed)` draws one sample of it.

## The process design kit

jolty pins one release of the kit, `sky130.VERSION`, as open_pdks builds it and ciel publishes it.
`sky130.root()` fetches its ngspice models, about 20 MB, into the [`cache`](../reference/configuration.md) the first time, each archive checked against its SHA-256 digest, and reads them from there after.
The setting `pdk_root` names a kit installed elsewhere instead, the directory that holds `sky130A`, as open_pdks, volare and ciel install it.

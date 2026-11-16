# Parts and costs

Real parts carry their makers' numbers and ratings, a catalog lists the parts a circuit can be built from, a bill of materials prices a component with them, and a KiCad netlist places them on a board.

## Real parts

`jolty.parts` holds parts that makers sell, each a function that returns its model, numbered as its maker numbers it and rated as its datasheet states:

| Function | Part | Ratings |
| --- | --- | --- |
| `q2n3904()` | 2N3904, NPN transistor | 40 V collector to emitter, 60 V collector to base, 6 V emitter to base, 200 mA, 625 mW, -55 °C to 150 °C |
| `q2n3906()` | 2N3906, PNP transistor | 40 V emitter to collector and base to collector, 5 V base to emitter, 200 mA, 625 mW, -55 °C to 150 °C |
| `d1n4148()` | 1N4148, signal diode | 100 V reverse, 300 mA forward, 500 mW, -55 °C to 175 °C |
| `irf511()` | IRF511, n-channel power MOSFET | 80 V drain to source, ±20 V gate to source, 5.6 A, 43 W, -55 °C to 175 °C |
| `irf9230()` | IRF9230, p-channel power MOSFET | 200 V source to drain, ±20 V gate to source, 6.5 A, 75 W, -55 °C to 150 °C |
| `irfz44()` | IRFZ44, n-channel power MOSFET | 60 V drain to source, ±20 V gate to source, 50 A, 150 W, -55 °C to 175 °C |
| `mtp50p03hdl()` | MTP50P03HDL, logic-level p-channel power MOSFET | 30 V source to drain, ±15 V gate to source, 50 A, 125 W, -55 °C to 150 °C |

The transistors and the diode take the SPICE parameters their makers publish.
The power MOSFETs' SPICE models may not be redistributed, so `power_mosfet(sheet)` fits each, by one stated recipe, to the figures its `Datasheet` states, and fits any other from its own; the power ratings are at a case of 25 °C, which only a heat sink holds.
The temperature ratings bound each junction's.

```python
from jolty import parts

amplifier = c.place(parts.q2n3904(), name="Q1")
switch = c.place(parts.irf511(), name="X1")
```

A simulation holds each part to its ratings, as the [checks](checks.md#on-analyses-results) describe.

Each part takes `heated=True` to [heat](devices.md#self-heating) by what it dissipates: a terminal or port `thermal` joins its junction to a thermal net, which a thermal resistance holds to the ambient, as its datasheet's junction-to-ambient one, 200 K/W for the 2N3904 in its TO-92, or a network of them for its case and heat sink.
A power MOSFET's channel and body diode share its junction.

```python
amplifier = c.place(parts.q2n3904(heated=True), name="Q1")
mount = c.place(devices.thermal_resistance(200 * K / W), name="RTH")
c.connect(amplifier.thermal, mount.a, name="junction")
c.connect(mount.b, ground)
```

Its junction's temperature is then held to its rating as it heats, and an unheated part's is the simulation's.

## Catalogs

A catalog file holds JSON: a currency, then parts, each at a unit price, and each a resistor, a capacitor or an inductor of one value and relative tolerance, a part its maker numbers, or both.

```json
{
  "format": "jolty-catalog",
  "version": 2,
  "currency": "USD",
  "parts": [
    {"kind": "resistor", "value": 4700.0, "tolerance": 0.01, "price": 0.004, "number": "RC0603FR-074K7L", "package": "0603", "footprint": "Resistor_SMD:R_0603_1608Metric"},
    {"kind": "capacitor", "value": 1e-9, "tolerance": 0.1, "price": 0.003},
    {"number": "2N3904", "price": 0.12, "package": "TO-92", "footprint": "Package_TO_SOT_THT:TO-92_Inline", "pins": {"emitter": ["1"], "base": ["2"], "collector": ["3"]}},
    {"number": "IRF511", "price": 1.1, "package": "TO-220AB", "lifecycle": "obsolete"}
  ]
}
```

Values are in SI base units.
A part's `number`, its maker's part number, and its `package` are optional, and its `lifecycle` is `active` unless it states `not-recommended`, for parts still made but not for new designs, or `obsolete`, for parts no longer made.
Its `footprint` and `pins`, also optional, place it on a [board](#boards): the footprint as KiCad's libraries name it, `Library:Name`, and the pins as the pads of that footprint each terminal or port of its instance joins, by name, as its maker's datasheet numbers its package's pins.
In Python, `Catalog.load(path)` reads one, and `catalog.save(path)` writes one.

## Bills of materials

`jolty cost` prices the instances of a saved circuit's top component, or of the component at `--instance`, such as a design placed in its test bench:

```sh
jolty cost bench.jolty.json --catalog parts.json --instance X1
```

An instance that names a part number, a device or a whole component instance such as a fitted power MOSFET, takes the cheapest part of that number, which must be its device, of its value and within its tolerance if the part states them.
Each other resistor, capacitor and inductor takes the cheapest part of its value whose tolerance is within its own, the tolerance its script gives it.
No instance takes an obsolete part.
An instance that states no tolerance, that comes from a lot, whose shared spread no part states, that no orderable part matches, or that is another device without a number, stays unpriced: the bill says why, the total leaves it out, and the command exits with 1.
The MCP server's `cost_circuit` tool does the same, and `jolty.catalog.bill` in Python.

## Boards

`jolty export-kicad` writes a saved circuit's top component, or the component at `--instance`, as a KiCad netlist, which KiCad's PCB editor imports with **File > Import > Netlist** to lay out its board:

```sh
jolty export-kicad bench.jolty.json --catalog parts.json --instance X1
```

Each instance its [bill of materials](#bills-of-materials) prices becomes a footprint, its catalog part's, whose pads join the nets of the terminals its part's pins name.
A resistor, a capacitor or an inductor whose part states no pins joins `a` to pad 1 and `b` to pad 2, as KiCad's two-pad footprints number them.
A terminal on a thermal net, such as a heated part's `thermal`, joins no pad, and an instance whose every terminal is on one, such as a thermal resistance, has no footprint; a terminal its part's pins give no pads, as `"light": []`, joins none either.
The component is elaborated alone, so it has a ground port, whose name its ground net takes; each other port's net takes the port's name, and each net inside its path, such as `X1/mid`.
Each footprint's reference designator is its device's letter, R, C, L, T for coupled inductors, D, or Q for a transistor, or U for any other part, numbered in the order the components place them, and its fields name the instance it builds and its maker's number.
Its identifier comes from its instance's path, so a footprint keeps it from one netlist of a component to the next.

The netlist is written only when every instance is placed, and otherwise the command lists what keeps each off the board and exits with 1: no part prices it, its part states no footprint, or its part's pins name a terminal it lacks or one on a thermal net, leave out one on an electrical net, or put a pad on two nets.
The netlist is saved beside the circuit, as `bench.X1.net` here, unless `--output` says where.
The MCP server's `export_kicad_netlist` tool does the same, and `jolty.kicad.netlist` in Python.

## Preferred values

The E series of IEC 60063 space preferred values evenly on a log scale: E3 to E24 for tolerances from 40 % down to 5 %, E48 and E96 for 2 % and 1 %, and E192 for 0.5 % and better.
`values(series, low, high)` lists a series' values between two bounds, and `neighbours(series, value)` the values nearest a value below and above it:

```python
neighbours("E96", 1234.0)  # (1210.0, 1240.0)
```

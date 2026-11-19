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

The transistors and diode use manufacturer-published SPICE parameters.
Vendor power-MOSFET SPICE models cannot be redistributed, so `power_mosfet(sheet)` fits each device to its `Datasheet` specifications using a fixed recipe and supports custom datasheets.
Power ratings assume a 25 °C case held by a heat sink, while temperature ratings bound the semiconductor junction.

```python
from jolty import parts

amplifier = c.place(parts.q2n3904(), name="Q1")
switch = c.place(parts.irf511(), name="X1")
```

Simulations verify every part against its ratings, as described in [checks](checks.md#on-analysis-results).

Passing `heated=True` enables [self-heating](devices.md#self-heating) from dissipated power, exposing a `thermal` terminal that connects to a thermal network (such as a 200 K/W junction-to-ambient resistance for a TO-92 2N3904, or a case and heat-sink network).
A power MOSFET's channel and body diode share one thermal junction.

```python
amplifier = c.place(parts.q2n3904(heated=True), name="Q2")
mount = c.place(devices.thermal_resistance(200 * K / W), name="RTH")
c.connect(amplifier.thermal, mount.a, name="junction")
c.connect(mount.b, ground)
```

Heated parts check junction temperature against ratings as self-heating develops, while unheated parts use the simulation ambient temperature.

## Catalogs

A catalog JSON file defines a currency and a list of unit-priced parts: passive values with relative tolerances, numbered manufacturer parts, or both.

```json
{
  "format": "jolty-catalog",
  "version": 3,
  "currency": "USD",
  "parts": [
    {"kind": "resistor", "value": 4700.0, "tolerance": 0.01, "price": 0.004, "part_number": "RC0603FR-074K7L", "package": "0603", "footprint": "Resistor_SMD:R_0603_1608Metric"},
    {"kind": "capacitor", "value": 1e-9, "tolerance": 0.1, "price": 0.003},
    {"part_number": "2N3904", "price": 0.12, "package": "TO-92", "footprint": "Package_TO_SOT_THT:TO-92_Inline", "pins": {"emitter": ["1"], "base": ["2"], "collector": ["3"]}},
    {"part_number": "IRF511", "price": 1.1, "package": "TO-220AB", "lifecycle": "obsolete"}
  ]
}
```

Values are in SI base units.
A part's `part_number` and `package` are optional, and `lifecycle` defaults to `active` (other values: `not-recommended` for existing designs only, or `obsolete` for discontinued parts).
Optional `footprint` (`Library:Name` in KiCad) and `pins` (mapping terminal or port names to datasheet pad numbers) enable [board export](#boards).
In Python, `Catalog.load(path)` reads a catalog and `catalog.save(path)` writes one.

## Bills of materials

`jolty cost` prices the top component of a saved circuit, or the subcircuit at `--instance` (such as a design placed inside a test bench):

```sh
jolty cost bench.jolty.json --catalog parts.json --instance X1
```

Instances specifying a part number (individual devices or whole subcircuits like fitted power MOSFETs) select the cheapest matching catalog entry with compatible kind, value, and tolerance.
Unnumbered resistors, capacitors, and inductors select the cheapest catalog entry matching their value whose tolerance meets or beats the script's tolerance.
Obsolete parts are never selected.
Instances lacking a tolerance, belonging to a lot, matching no orderable part, or lacking a required part number remain unpriced: the bill reports each cause, excludes them from the total, and exits with 1.
The MCP tool `cost_circuit` and Python helper `jolty.catalog.bill` provide the same pricing.

## Boards

`jolty export-kicad` exports a saved circuit's top component (or `--instance`) as a KiCad netlist for **File > Import > Netlist** in KiCad's PCB editor:

```sh
jolty export-kicad bench.jolty.json --catalog parts.json --instance X1
```

Each priced instance in the [bill of materials](#bills-of-materials) becomes its catalog footprint with pads wired to the nets matching its pin map.
Passive parts without explicit `pins` map `a` to pad 1 and `b` to pad 2, matching KiCad two-pad footprints.
Terminals on thermal nets (such as `thermal`) map to no pad, and purely thermal devices (such as thermal resistances) emit no footprint; terminals mapped to an empty list (like `"light": []`) likewise join no pad.
The exported component elaborates standalone, naming its ground net after the ground port, external nets after port names, and internal nets by hierarchical path (such as `X1/mid`).
Reference designators use standard prefixes (`R`, `C`, `L`, `T` for coupled inductors, `D`, `Q` for transistors, `U` otherwise) numbered in placement order, storing instance path and part number in footprint fields.
Each footprint UUID derives deterministically from the instance path so repeated exports preserve board identity.

Export writes a netlist only when every electrical instance maps cleanly; otherwise it lists every unplaced reason (unpriced part, missing footprint, unknown or thermal pin, unmapped electrical terminal, or duplicate pad) and exits with 1.
The netlist saves next to the circuit (for example, `bench.X1.net`) unless `--output` specifies another path.
The MCP tool `export_kicad_netlist` and Python helper `jolty.kicad.netlist` produce identical output.

## Preferred values

IEC 60063 E-series values space logarithmically: E3 through E24 cover 40% to 5% tolerances, E48 and E96 cover 2% and 1%, and E192 covers 0.5% and tighter.
`values(series, low, high)` lists all series values within bounds, and `neighbors(series, value)` returns the nearest lower and upper series values:

```python
from jolty.catalog import neighbors

neighbors("E96", 1234.0)  # (1210.0, 1240.0)
```

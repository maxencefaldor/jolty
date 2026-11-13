# Parts and costs

Real parts carry their makers' numbers and ratings, a catalog lists the parts a circuit can be built from, and a bill of materials prices a component with them.

## Real parts

`jolty.parts` holds parts that makers sell, each a function that returns its model, numbered as its maker numbers it and rated as its datasheet states:

| Function | Part | Ratings |
| --- | --- | --- |
| `q2n3904()` | 2N3904, NPN transistor | 40 V collector to emitter, 60 V collector to base, 6 V emitter to base, 200 mA, 625 mW |
| `q2n3906()` | 2N3906, PNP transistor | 40 V emitter to collector and base to collector, 5 V base to emitter, 200 mA, 625 mW |
| `d1n4148()` | 1N4148, signal diode | 100 V reverse, 300 mA forward, 500 mW |
| `irf511()` | IRF511, n-channel power MOSFET | 80 V drain to source, ±20 V gate to source, 5.6 A, 43 W |
| `irf9230()` | IRF9230, p-channel power MOSFET | 200 V source to drain, ±20 V gate to source, 6.5 A, 75 W |
| `irfz44()` | IRFZ44, n-channel power MOSFET | 60 V drain to source, ±20 V gate to source, 50 A, 150 W |
| `mtp50p03hdl()` | MTP50P03HDL, logic-level p-channel power MOSFET | 30 V source to drain, ±15 V gate to source, 50 A, 125 W |

The transistors and the diode take the SPICE parameters their makers publish.
The power MOSFETs' SPICE models may not be redistributed, so `power_mosfet(sheet)` fits each, by one stated recipe, to the figures its `Datasheet` states, and fits any other from its own; the power ratings are at a case of 25 °C, which only a heat sink holds.

```python
from jolty import parts

amplifier = c.place(parts.q2n3904(), id="Q1")
switch = c.place(parts.irf511(), id="X1")
```

A simulation holds each part to its ratings, as the [checks](checks.md#on-analyses-results) describe.

## Catalogs

A catalog file holds JSON: a currency, then parts, each at a unit price, and each a resistor, a capacitor or an inductor of one value and relative tolerance, a part its maker numbers, or both.

```json
{
  "format": "jolty-catalog",
  "version": 1,
  "currency": "USD",
  "parts": [
    {"device": "resistor", "value": 4700.0, "tolerance": 0.01, "price": 0.004, "number": "RC0603FR-074K7L", "package": "0603"},
    {"device": "capacitor", "value": 1e-9, "tolerance": 0.1, "price": 0.003},
    {"number": "2N3904", "price": 0.12, "package": "TO-92"},
    {"number": "IRF511", "price": 1.1, "package": "TO-220AB", "lifecycle": "obsolete"}
  ]
}
```

Values are in SI base units.
A part's `number`, its maker's part number, and its `package` are optional, and its `lifecycle` is `active` unless it states `not-recommended`, for parts still made but not for new designs, or `obsolete`, for parts no longer made.
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
The MCP server's `price_circuit` tool does the same, and `jolty.catalog.bill` in Python.

## Preferred values

The E series of IEC 60063 space preferred values evenly on a log scale: E3 to E24 for tolerances from 40 % down to 5 %, E48 and E96 for 2 % and 1 %, and E192 for 0.5 % and better.
`values(series, low, high)` lists a series' values between two bounds, and `neighbours(series, value)` the values nearest a value below and above it:

```python
neighbours("E96", 1234.0)  # (1210.0, 1240.0)
```

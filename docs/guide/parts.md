# Parts and costs

A catalog lists the parts a circuit can be built from, and a bill of materials prices a component with them.

## Catalogs

A catalog file holds JSON: a currency, then parts, each a resistor, a capacitor or an inductor of one value and relative tolerance, at a unit price.

```json
{
  "format": "jolty-catalog",
  "version": 1,
  "currency": "USD",
  "parts": [
    {"device": "resistor", "value": 4700.0, "tolerance": 0.01, "price": 0.004, "number": "RC0603FR-074K7L"},
    {"device": "capacitor", "value": 1e-9, "tolerance": 0.1, "price": 0.003}
  ]
}
```

Values are in SI base units, and `number`, the manufacturer's part number, is optional.
In Python, `Catalog.load(path)` reads one, and `catalog.save(path)` writes one.

## Bills of materials

`jolty cost` prices the device instances of a saved circuit's top component, or of the component at `--instance`, such as a design placed in its test bench:

```sh
jolty cost bench.jolty.json --catalog parts.json --instance X1
```

Each resistor, capacitor and inductor takes the cheapest part of its value whose tolerance is within its own, the tolerance its script gives it.
An instance that states no tolerance, whose value no part has, or that is another device, stays unpriced: the bill says why, the total leaves it out, and the command exits with 1.
The MCP server's `price_circuit` tool does the same, and `jolty.catalog.bill` in Python.

## Preferred values

The E series of IEC 60063 space preferred values evenly on a log scale: E3 to E24 for tolerances from 40 % down to 5 %, E48 and E96 for 2 % and 1 %, and E192 for 0.5 % and better.
`values(series, low, high)` lists a series' values between two bounds, and `neighbours(series, value)` the values nearest a value below and above it:

```python
neighbours("E96", 1234.0)  # (1210.0, 1240.0)
```

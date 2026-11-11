# Units

Values in jolty carry their dimension, so a value in farads cannot go where ohms belong.

## Quantities

A script writes a value as a number times a unit, such as `10 * kohm` or `4.7 * uF`.
The result is a `Quantity`: its value in SI base units and its `Dimension`, the exponents of the kilogram, metre, second, ampere and kelvin.

Arithmetic combines dimensions, so `5 * V / (1 * kohm)` is a current, and `1 * kohm * (1 * uF)` is a time.
Adding or subtracting quantities of different dimensions raises a `DimensionError`.

## Checked where they are used

Every device checks the dimension of each parameter as it reads it.
`devices.resistor(1 * uF)` raises a `DimensionError` that says a resistance takes a quantity in Ω, on the line that called it.
Values must also be finite.

## The units

The module `jolty.units` defines the units below; the micro prefix is written `u` in names.

| Quantity | Units |
| --- | --- |
| Time | `s`, `ms`, `us`, `ns`, `ps` |
| Frequency | `Hz`, `kHz`, `MHz`, `GHz` |
| Voltage | `V`, `kV`, `mV`, `uV` |
| Current | `A`, `mA`, `uA`, `nA`, `pA` |
| Resistance | `ohm`, `kohm`, `Mohm` |
| Conductance | `S`, `mS`, `uS` |
| Capacitance | `F`, `mF`, `uF`, `nF`, `pF`, `fF` |
| Inductance | `H`, `mH`, `uH`, `nH` |
| Power | `W`, `mW`, `uW` |
| Temperature | `K` |

## Printed with prefixes

A quantity prints with the SI prefix that puts its value between 1 and 1000, to four significant digits: `str(4700 * ohm)` is `4.7 kΩ`.
Reports print every value the same way.

The circuit format, the engine and the results files hold plain numbers in SI base units.

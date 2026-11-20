# Units

Values in jolty carry their dimension, so a value in farads cannot go where ohms belong.

## Quantities

A script writes a value as a number times a unit, such as `10 * kohm` or `4.7 * uF`.
The result is a `Quantity`: its value in SI base units and its `Dimension`, the exponents of the kilogram, meter, second, ampere and kelvin.

Arithmetic combines dimensions, so `5 * V / (1 * kohm)` is a current, and `1 * kohm * (1 * uF)` is a time.
Powers may be fractions, as a noise density's square root of a hertz is: `Hz ** Fraction(1, 2)` is `rtHz`.
Adding, subtracting, or comparing quantities of different dimensions raises a `DimensionError`.

## Checked where they are used

Every device, option, override, variant, axis, and sweep checks parameter dimensions before simulation runs.
`devices.resistor(1 * uF)` raises a `DimensionError` that says a resistance takes a quantity in Ω, on the line that called it.
Values must also be finite.

## The units

The module `jolty.units` defines the units below; the micro prefix is written `u` in names.

| Quantity | Units |
| --- | --- |
| Time | `s`, `ms`, `us`, `ns`, `ps` |
| Frequency | `Hz`, `kHz`, `MHz`, `GHz` |
| Voltage | `V`, `kV`, `mV`, `uV`, `nV` |
| Current | `A`, `mA`, `uA`, `nA`, `pA`, `fA` |
| Charge | `C`, `mC`, `uC`, `nC`, `pC`, `fC` |
| Resistance | `ohm`, `kohm`, `Mohm` |
| Conductance | `S`, `mS`, `uS`, `nS`, `pS` |
| Capacitance | `F`, `mF`, `uF`, `nF`, `pF`, `fF` |
| Inductance | `H`, `mH`, `uH`, `nH` |
| Power | `W`, `mW`, `uW` |
| Temperature | `K` |
| Energy | `J`, `mJ` |
| Noise density | `rtHz`, the square root of a hertz: `4 * nV / rtHz` |

## Printed with prefixes

A quantity prints with the SI prefix that puts its value between 1 and 1000, to four significant digits: `str(4700 * ohm)` is `4.7 kΩ`.
Reports print every value the same way.

The circuit format, the engine and the results files hold plain numbers in SI base units.

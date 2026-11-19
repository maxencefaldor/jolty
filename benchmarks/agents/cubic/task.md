# A compact cubic function generator

Design a circuit whose output current is the cube of its input current over the square of a reference current: `Iout = Iin³ / Iref²`.
The circuit must operate from a 2 V supply and remain accurate from DC to 1 GHz.

Write `design.py`, a script whose `build()` returns your circuit as a component with five ports: `supply`, `reference`, `input`, `output` and `ground`.
The scoring bench holds `supply` at 2 V above `ground`, drives 100 µA from `supply` into `reference`, and drives `Iin` from `supply` into `input`.
The bench holds `output` at 2 V through an ammeter, measuring `Iout` as the current flowing into `output`.

Build it only from the transistor of `parts.py`, `npn()`, an integrated NPN transistor of a fast bipolar process, and from resistors and capacitors.
Every transistor of an integrated circuit matches every other, so a diode is an `npn()` whose base joins its collector.
No other transistor, diode, source or behavioral element may appear in your circuit.

Your design is scored across four cases: a DC sweep of `Iin` from 0 to 200 µA in 5 µA steps, plus three sinusoidal transient tests.
Each transient drives `Iin = 100 µA + 90 µA · sin(2π f t)` at `f` = 1 MHz, 100 MHz, and 1 GHz, evaluated over the third and fourth periods.
In each case the scorer measures the mean absolute error of `Iout` against `Iin³ / Iref²`, divides it by a reference circuit's error on the same case in the same conditions, and averages the four ratios: the score, lower being better.
The design passes when it uses only allowed parts and achieves a score of at most 1 across all operating conditions below, both with nominal parts and on at least 95 % of 100 Monte Carlo boards.
The circuit must pass from 0 °C to 70 °C across a ±5 % supply variation.
The scorer evaluates 27 °C at nominal supply alongside 0 °C and 70 °C at both −5 % and +5 % supply.
On each integrated board, resistors share a common ±20 % process shift while matching within 0.1 % (3σ normal spread) regardless of declared tolerances.
Each capacitor varies within ±5 % of nominal regardless of declared tolerance, while transistors remain at nominal parameters.
At an equal score, fewer parts are better.

You have jolty, a compiler and simulator for electronic circuits: its command line, `jolty`, its MCP server, `jolty-mcp`, and its Python package.
Work as long as you need, and verify what you claim.

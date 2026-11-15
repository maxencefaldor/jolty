# A compact cubic function generator

Design a circuit whose output current is the cube of its input current over the square of a reference current: `Iout = Iin³ / Iref²`.
It must work from a 2 V supply and stay accurate from DC to 1 GHz.

Write `design.py`, a script whose `build()` returns your circuit as a component with five ports: `supply`, `reference`, `input`, `output` and `ground`.
The bench that scores it holds `supply` at 2 V over `ground`, drives 100 µA from the supply into `reference`, drives the input current `Iin` from the supply into `input`, and holds `output` at the supply's 2 V through an ammeter: `Iout` is the current into `output`.

Build it only from the transistor of `parts.py`, `npn()`, an integrated NPN transistor of a fast bipolar process, and from resistors and capacitors.
Every transistor of an integrated circuit matches every other, so a diode is an `npn()` whose base joins its collector.
No other transistor, diode, source or behavioral element may appear in your circuit.

Your design is scored in four cases: a DC sweep of `Iin` from 0 to 200 µA in 5 µA steps, and `Iin = 100 µA + 90 µA · sin(2π f t)` at f = 1 MHz, 100 MHz and 1 GHz, each read over its third and fourth periods.
In each case the scorer measures the mean absolute error of `Iout` against `Iin³ / Iref²`, divides it by a reference circuit's error on the same case in the same conditions, and averages the four ratios: the score, lower being better.
The design meets the task when it holds only the parts allowed and its score is at most 1, with its parts exact in each of the conditions below, and on at least 95 % of 100 boards built of it, each in all of them.
It must meet the task from 0 °C to 70 °C with its supply anywhere within 5 % of its level: the scorer checks it at 27 °C with the supply exact, and at 0 °C and at 70 °C with the supply 5 % low and 5 % high.
On each board, as on one die, the resistors lie together anywhere within 20 % of their values, and each within 0.1 % of the others, three standard deviations of a normal spread, whatever tolerance you give them; every capacitor lies anywhere within 5 % of its value, and the transistors stay as given.
At an equal score, fewer parts are better.

You have jolty, a compiler and simulator for electronic circuits: its command line, `jolty`, its MCP server, `jolty-mcp`, and its Python package.
Work as long as you need, and verify what you claim.

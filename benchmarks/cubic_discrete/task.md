# A cubic function generator from discrete transistors

Design a circuit whose output current is the cube of its input current, scaled: with the input `Iin = 100 µA/V · Vin`, the output `Iout` should make `10.5 kV/A · Iout = Vin³`, so `Iout = Iin³ / (102.5 µA)²`.
It must work from a 2 V supply, and it has no reference current: it makes its own scale.

Write `design.py`, a script whose `build()` returns your circuit as a component with four ports: `input`, `output`, `vcc` and `ground`.
The bench that scores it holds `vcc` at 2 V over `ground`, drives `Iin` from the supply into `input` through 1 µΩ, and holds `output` at the supply's 2 V through an ammeter and 1 µΩ: `Iout` is the current into `output`.

Build it only from the transistors of `parts.py`, `npn()`, a 2N3904, and `pnp()`, a 2N3906, and from resistors and capacitors.
No other transistor, diode, source or behavioral element may appear in your circuit.

Your design is scored in four cases, each a transient read at 101 times evenly spaced from its start to its end:

- `Vin` rises linearly from 0 V to 1.26 V over 1 ms;
- `Vin` makes one cycle of a 1 kHz sine between 0 V and 1.26 V, starting at 0.63 V and rising;
- `Vin` falls linearly from 1.26 V to 0 V over 10 ms;
- `Vin` holds 1 V for 10 ms.

The scorer reads the error `10.5 kV/A · Iout − Vin³` at every time, and averages its absolute value over the 404 readings.
The design meets the task when it holds only the parts allowed and that mean absolute error is at most 6.76 mV, a reference circuit's, both with its parts exact and on at least 95 % of 100 boards built of it.
On each board, every resistor lies anywhere within 1 % of its value and every capacitor within 5 %, whatever tolerance you give them, while the transistors stay as given.
Lower is better, and at an equal error, fewer parts are better.

You have jolty, a compiler and simulator for electronic circuits: its command line, `jolty`, its MCP server, `jolty-mcp`, and its Python package.
Work as long as you need, and verify what you claim.

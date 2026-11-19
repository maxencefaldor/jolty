# A cubic function generator from discrete transistors

Design a circuit whose output current is the cube of its input current, scaled: with the input `Iin = 100 µA/V · Vin`, the output `Iout` should make `10.5 kV/A · Iout = Vin³`, so `Iout = Iin³ / (102.5 µA)²`.
The circuit must operate from a 2 V supply without an external reference current, generating an internal current scale.

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
The design passes when it uses only allowed parts and keeps mean absolute error at most 6.76 mV (the reference circuit level) across all operating conditions below.
Passing requires meeting the bound both with nominal parts and on at least 95 % of 100 Monte Carlo boards.
The circuit must pass from 0 °C to 70 °C across a ±5 % supply variation.
The scorer evaluates 27 °C at nominal supply alongside 0 °C and 70 °C at both −5 % and +5 % supply.
On each board, every resistor lies anywhere within 1 % of its value and every capacitor within 5 %, whatever tolerance you give them.
All 2N3904 parts on a board share one reel and all 2N3906 parts share another, with datasheet bounds at 10 mA of 0.65 V to 0.85 V for base-emitter saturation voltage and 100 to 300 for current gain.
The nominal models produce 160 and 0.743 V for the 2N3904, and 163 and 0.791 V for the 2N3906.
On a reel, each transistor's current gain lies within 2 % of any other's and its base-emitter voltage within 2 mV, as a matched pair's from adjacent die do.
Lower is better, and at an equal error, fewer parts are better.

You have jolty, a compiler and simulator for electronic circuits: its command line, `jolty`, its MCP server, `jolty-mcp`, and its Python package.
Work as long as you need, and verify what you claim.

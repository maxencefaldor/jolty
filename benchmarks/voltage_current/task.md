# A voltage-current converter

Design a circuit that turns the difference of two input voltages into a stable output current: 1 mA for each volt by which `plus` exceeds `minus`, whatever the voltage at the output.
The inputs swing between 2 V and 10 V, and the circuit's only supply is −15 V: it has no positive supply, so the inputs may sit above every rail it has.

Write `design.py`, a script whose `build()` returns your circuit as a component with five ports: `plus`, `minus`, `output`, `vee` and `ground`.
The bench that scores it holds `vee` at −15 V over `ground` and drives `plus` and `minus` from ideal voltage sources through 1 µΩ each.
`output` returns to `ground` through a voltage source `VS` and a 1 Ω load, in that order: `VS` holds `output` at `0 V + 1 V · sin(2π · 3.2 Hz · t)` over the load's top.
The output current is the current from `output` through `VS` and the load to `ground`; it should be `1 mA/V · (v(plus) − v(minus))`.

Build it only from the transistors of `parts.py`, `npn()`, a 2N3904, and `pnp()`, a 2N3906, and from resistors and capacitors.
No other transistor, diode, source or behavioral element may appear in your circuit.

Your design is scored in four cases, each a transient of one second read at 31 times, 1/30 s apart from 0 s to 1 s:

| Case | `plus` | `minus` |
| --- | --- | --- |
| 1 | falls linearly from 10 V to 2 V | rises linearly from 2 V to 10 V |
| 2 | a 4.5 Hz sine about 5.5 V, from 2 V to 9 V | a 6.4 Hz sine about 6 V, from 2.5 V to 9.5 V |
| 3 | a 2 Hz sine about 6 V, from 2 V to 10 V | the same as `plus` |
| 4 | a 4.5 Hz sine about 4 V, from 3 V to 5 V | a 6.4 Hz sine about 4 V, from 3 V to 5 V |

Each sine starts at its middle at 0 s and rises, as `VS` does.
In each case the scorer takes the mean absolute error of the output current over the 31 times, and divides it by a reference circuit's error on that case: 41 µA, 36 µA, 26 µA and 44 µA.
The fitness is the mean of the four ratios, lower being better; the reference circuit's is 1.
Its output current must also be stable: after `VS` steps by 10 mV in 1 ns, the current must settle, swinging by less than 1 µA from 40 µs to 50 µs after the start, with `plus` and `minus` at 6 V and 4 V, at 10 V and 2 V with `VS` from 1 V, and at 2 V and 10 V with `VS` from −1 V before its step.
There, its mean over those 10 µs must lie within 44 µA, the reference circuit's largest mean error over a case, of 1 mA/V times the difference between `plus` and `minus`.
The design meets the task when it holds only the parts allowed, and its fitness is at most 1 and its output current stable and accurate where it settles, with its parts exact in each of the conditions below, and on at least 95 % of 100 boards built of it, each in all of them.
It must meet the task from 0 °C to 70 °C with its supply anywhere within 5 % of its level: the scorer checks it at 27 °C with the supply exact, and at 0 °C and at 70 °C with the supply 5 % low and 5 % high.
On each board, every resistor lies anywhere within 1 % of its value and every capacitor within 5 %, whatever tolerance you give them.
The board's 2N3904s come from one reel and its 2N3906s from another, each reel anywhere its datasheet allows: a current gain at 10 mA from 100 to 300, and a base-emitter saturation voltage at 10 mA from 0.65 V to 0.85 V, where the models give 160 and 0.743 V for the 2N3904, and 163 and 0.791 V for the 2N3906.
On a reel, each transistor's current gain lies within 2 % of any other's and its base-emitter voltage within 2 mV, as a matched pair's from adjacent die do.
At an equal fitness, fewer parts are better.

You have jolty, a compiler and simulator for electronic circuits: its command line, `jolty`, its MCP server, `jolty-mcp`, and its Python package.
Work as long as you need, and verify what you claim.

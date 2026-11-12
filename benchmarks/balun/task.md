# A balun from a 1 V supply

Design a balun that runs from a single 1 V supply: from one input it makes two outputs, each half the input's amplitude, `in_phase` in phase with it and `out_of_phase` 180° from it, both at the same DC level, from 1 MHz to 100 MHz.

Write `design.py`, a script whose `build()` returns your circuit as a component with five ports: `input`, `in_phase`, `out_of_phase`, `vcc` and `ground`.
The bench that scores it holds `vcc` at 1 V over `ground`.
It drives `input` from a source through 1 µF and 75 Ω in series, so your circuit sets its input's DC level itself, and loads each output with 150 Ω to the 1 V supply.
The difference `out_of_phase − in_phase` should then equal the source's signal, with the same amplitude, 180° from it.

Build it only from the transistors of `parts.py`, `npn()`, a 2N3904, and `pnp()`, a 2N3906, and from resistors and capacitors.
No other transistor, diode, source or behavioral element may appear in your circuit.

The scorer measures, relative to the source's signal:

- from an AC analysis at 41 frequencies, 20 a decade from 1 MHz to 100 MHz: each output's gain against 0.5 (−6.02 dB) and its phase against 0° or 180°, and the difference's gain against 1 (0 dB) and its phase against 180°;
- from the operating point: the DC offset between the outputs;
- from a transient of 200 ns, the source a 10 MHz sine of 10 mV: the difference's total harmonic distortion, harmonics 2 to 9, over its second period.

The design meets the task when it holds only the parts allowed and, with its parts exact in each of the conditions below, and on at least 95 % of 100 boards built of it, each in all of them:

- the difference's gain is within 0.60 dB of 0 dB on average over the 41 frequencies;
- its total harmonic distortion is at most −33.9 dB;
- each output's gain is within 3 dB of −6.02 dB, and each output's phase, and the difference's, within 10° of its own, at every one of the 41 frequencies;
- the DC offset between the outputs is at most 1.72 mV.

It must meet the task from 0 °C to 70 °C with its supply anywhere within 5 % of its level: the scorer checks it at 27 °C with the supply exact, and at 0 °C and at 70 °C with the supply 5 % low and 5 % high.
On each board, every resistor lies anywhere within 1 % of its value and every capacitor within 5 %, whatever tolerance you give them, and each transistor's forward beta strays as far as its datasheet's current gain at 10 mA, 100 to 300, allows: a 2N3904's within 57 % of its model's, a 2N3906's within 38 %, each part apart from every other.

A reference circuit reaches 0.60 dB and −33.9 dB; lower is better on both, and at equal results, fewer parts are better.

You have jolty, a compiler and simulator for electronic circuits: its command line, `jolty`, its MCP server, `jolty-mcp`, and its Python package.
Work as long as you need, and verify what you claim.

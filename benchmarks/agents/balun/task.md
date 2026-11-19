# A balun from a 1 V supply

Design a balun operating from a single 1 V supply across 1 MHz to 100 MHz.
From one input, the circuit produces two half-amplitude outputs at equal DC levels: `in_phase` at 0° and `out_of_phase` at 180°.

Write `design.py`, a script whose `build()` returns your circuit as a component with five ports: `input`, `in_phase`, `out_of_phase`, `vcc` and `ground`.
The bench that scores it holds `vcc` at 1 V over `ground`.
The bench drives `input` through 1 µF and 75 Ω in series, leaving the circuit to establish input DC bias, and loads each output with 150 Ω to the 1 V supply.
The difference `out_of_phase - in_phase` should then equal the source's signal, with the same amplitude, 180° from it.

Build it only from the transistors of `parts.py`, `npn()`, a 2N3904, and `pnp()`, a 2N3906, and from resistors and capacitors.
No other transistor, diode, source or behavioral element may appear in your circuit.

The scorer measures, relative to the source's signal:

- AC analysis at 41 frequencies (20 per decade, 1 MHz to 100 MHz): single-ended gain versus 0.5 (-6.02 dB) and phase (0°/180°), plus differential gain versus 1 (0 dB) and phase (180°);
- from the operating point: the DC offset between the outputs;
- from a transient of 200 ns, the source a 10 MHz sine of 10 mV: the difference's total harmonic distortion, harmonics 2 to 9, over its second period.

The design meets the task when it holds only the parts allowed and, with its parts exact in each of the conditions below, and on at least 95 % of 100 boards built of it, each in all of them:

- the difference's gain is within 0.60 dB of 0 dB on average over the 41 frequencies;
- total harmonic distortion of the difference is at most -33.9 dB;
- at all 41 frequencies, each output's gain stays within 3 dB of -6.02 dB, and each single-ended and differential phase stays within 10° of target;
- the DC offset between the outputs is at most 1.72 mV.

The circuit must pass from 0 °C to 70 °C across a ±5 % supply variation.
The scorer evaluates 27 °C at nominal supply alongside 0 °C and 70 °C at both -5 % and +5 % supply.
On each board, every resistor lies anywhere within 1 % of its value and every capacitor within 5 %, whatever tolerance you give them.
All 2N3904 parts on a board share one reel and all 2N3906 parts share another, with datasheet bounds at 10 mA of 0.65 V to 0.85 V for base-emitter saturation voltage and 100 to 300 for current gain.
The nominal models produce 160 and 0.743 V for the 2N3904, and 163 and 0.791 V for the 2N3906.
On a reel, each transistor's current gain lies within 2 % of any other's and its base-emitter voltage within 2 mV, as a matched pair's from adjacent die do.

A reference circuit reaches 0.60 dB and -33.9 dB; lower is better on both, and at equal results, fewer parts are better.

You have jolty, a compiler and simulator for electronic circuits: its command line, `jolty`, its MCP server, `jolty-mcp`, and its Python package.
Work as long as you need, and verify what you claim.

# A linear voltage-controlled oscillator

Design an oscillator whose frequency is proportional to a control voltage: 1 kHz per volt from 0.5 V to 5 V.

- Each frequency lies within 1 % of the straight line through the origin that fits them best: its linearity.
- The slope is within 5 % of 1 kHz/V.
- Use at most two op-amps of `examples/common.py`, its transistors, diodes, resistors and capacitors, as few as you can.

Use jolty to find the best design you can, and report what limits its linearity.

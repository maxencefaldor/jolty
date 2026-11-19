# A two-terminal current source

Design a two-terminal current source across terminals `a` and `b` carrying 1 mA from `a` to `b` for applied voltages from 3 V to 24 V.

- Current stays within 1% of the 12 V value across the full 3 V to 24 V range (line regulation).
- At 12 V, current stays within 2% of the 27 °C value from -40 °C to 85 °C.
- The circuit draws no auxiliary supply and uses as few parts as possible from `examples/common.py` (2N3904, 2N3906, 1N4148, and resistors).

Use jolty to find the best design you can.
Report the design, line regulation, temperature drift, and experimental findings.

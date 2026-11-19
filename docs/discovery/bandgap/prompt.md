# A reference without an op-amp

Design a voltage reference near 1.2 V from a 5 V supply without an op-amp.

- Output drift stays below 50 ppm/K from -40 °C to 125 °C (a span under 10 mV).
- Line regulation stays below 1 mV per volt across 4.5 V to 5.5 V.
- The reference self-starts reliably and uses ten parts or fewer from `examples/common.py` (transistors, diodes, and resistors) plus the 2N5457 JFET.

Use jolty to find the best design you can, and report what limits performance.

# A clean, steady sine oscillator

Design a 1 kHz sine oscillator around the op-amp from `examples/common.py`.

- Oscillation frequency stays within 1% of 1 kHz.
- Peak amplitude targets 2 V within 5% and holds within 5% from -40 °C to 85 °C.
- Total harmonic distortion stays below 0.5%.
- The oscillator self-starts and settles within 100 ms.

Use as few parts as possible from `examples/common.py` (resistors, capacitors, and parts) plus a 2N5457 JFET.
Use jolty to find the best design you can, and report what limits performance.

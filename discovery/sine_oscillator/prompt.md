# A clean, steady sine oscillator

Design a 1 kHz sine oscillator around the op-amp of `examples/common.py`.

- Its frequency is within 1 % of 1 kHz.
- Its amplitude is 2 V peak, within 5 %, and holds within 5 % from -40 °C to 85 °C.
- Its total harmonic distortion is below 0.5 %.
- It starts by itself and settles within 100 ms.

Use any of the parts of `examples/common.py`, resistors, capacitors and a 2N5457 JFET, as few as you can.
Use jolty to find the best design you can, and report what limits it.

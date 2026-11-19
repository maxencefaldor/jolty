# Findings: a NOR gate of three memristors

The prompt requires a three-memristor VTEAM NOR gate driven by a single pulse up to 10 ns, holding every output within 0.1 and every input within 0.05 across ±10% pulse amplitude variation.
The textbook MAGIC gate operates only from 1.30 V to 1.32 V (±0.8%), whereas a 2.2 kΩ input shunt widens the window to 1.14 V through 1.91 V (±25%) and passes all bounds.

## Method

`candidates.py` implements the Kvatinsky et al. MAGIC NOR gate plus a four-gate bench testing all input pairs under a shared pulse.
`explore.py` sweeps pulse amplitude in 10 mV steps, runs a transient at each step, and inspects post-pulse memristor states to identify passing windows and boundary failure modes.
Each transient window is compared against the static threshold-only window computed from initial device resistances before switching begins.

## Results

| Gate | Pulse | Works from | to | Margin | Fails below | Fails above |
| --- | --- | --- | --- | --- | --- | --- |
| MAGIC | 10 ns | 1.30 V | 1.32 V | ±0.8 % | output of 01 and 10 | inputs of 00 |
| MAGIC | 100 ns | 1.10 V | 1.16 V | ±2.7 % | output of 01 and 10 | inputs of 00 |
| MAGIC | 1 µs | 1.03 V | 1.08 V | ±2.4 % | output of 01 and 10 | inputs of 00 |
| Shunted by 20 kΩ | 10 ns | 1.27 V | 1.38 V | ±4.2 % | output of 01 and 10 | inputs of 00 |
| Shunted by 5 kΩ | 10 ns | 1.21 V | 1.57 V | ±12.9 % | output of 01 and 10 | inputs of 00 |
| Shunted by 3 kΩ | 10 ns | 1.17 V | 1.75 V | ±19.9 % | output of 01 and 10 | inputs of 00 |
| Shunted by 2.5 kΩ | 10 ns | 1.15 V | 1.84 V | ±23.1 % | output of 01 and 10 | inputs of 00 |
| **Shunted by 2.2 kΩ** | **10 ns** | **1.14 V** | **1.91 V** | **±25.2 %** | **output of 01 and 10** | **output of 00** |
| Shunted by 2 kΩ | 10 ns | 1.13 V | 1.80 V | ±22.9 % | output of 01 and 10 | output of 00 |
| Shunted by 1.5 kΩ | 10 ns | 1.10 V | 1.52 V | ±16 % | output of 01 and 10 | output of 00 |
| Shunted by 1 kΩ | 10 ns | 1.05 V | 1.24 V | ±8.3 % | output of 01 and 10 | output of 00 |

## What the experiments showed

1. **MAGIC's window is a tight threshold coincidence.**
    With one input on (`01`/`10`), two 1 kΩ resistors sit in series so the output switches off only above twice the 0.5 V off-threshold.
    With both inputs off (`00`), 50 kΩ over 1 kΩ places 98% of the pulse across the inputs and flips them above the 1 V threshold.
    With an on-threshold twice the off-threshold, static thresholds permit only 0.995 V to 1.02 V.
1. **Pulse duration shifts the window without widening it enough.**
    At 10 ns, cubic overdrive pushes the active window to 1.30 V through 1.32 V because outputs must switch fully while inputs barely move.
    Longer 100 ns pulses pull the window down and widen margin to ±2.7%, still well short of ±10%.
1. **A shunt resistor across the inputs separates the two failure bounds.**
    The shunt diverts voltage off the high-resistance `00` inputs (raising the upper bound) while barely affecting the already-1 kΩ `01`/`10` input pair.
    An overly small shunt leaves enough pulse voltage on the `00` output to flip it off, whereas an overly large shunt allows the `00` inputs to turn on.
    Static thresholds balance at 2.1 kΩ (matching 2 kΩ across inputs), while full transients peak at 2.2 kΩ where the upper limit transitions from input disturbance to `00` output disturbance.

## Conclusion

`design.py` implements the MAGIC NOR gate with a 2.2 kΩ input shunt driven by a 10 ns, 1.525 V pulse, operating across 1.14 V to 1.91 V (±25%).
At ±10% nominal pulse amplitude, every output finishes within 0.002 of its NOR target, no input drifts past 0.006, and ngspice agrees within `1e-4`.

## Limits

The agent that wrote the prompt also designed the gate with knowledge of MAGIC and judged the output, making this a capability showcase rather than a blind benchmark.
All memristors share identical nominal VTEAM parameters; device-to-device variability, cycle-to-cycle jitter, and softer physical thresholds would narrow the real margin.
The shunt resistor draws write current continuously and would require switching or sharing in a crossbar array, which this single-gate bench omits.

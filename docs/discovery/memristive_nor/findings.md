# Findings: a NOR gate of three memristors

The prompt asks for a NOR gate of three VTEAM memristors and one pulse of at most 10 ns, every output within 0.1 of its NOR and every input within 0.05 of its bit, with the pulse's amplitude 10 % off either way.
MAGIC's gate, the textbook answer, works only between 1.30 V and 1.32 V, ±0.8 %; a 2.2 kΩ resistor across its inputs widens that to 1.14 V to 1.91 V, ±25 %, and meets the prompt.

## Method

`candidates.py` writes MAGIC's NOR gate with its inputs and its output oriented as Kvatinsky et al. give them, and a bench of four gates, one per pair of inputs, written by one pulse.
`explore.py` scans the pulse's amplitude in steps of 10 mV, runs a transient of each, and reads every memristor's state where the pulse has ended; a gate works where every output and input lies within its bound, and what fails a step beyond each end of the range is named.
Beside each range stands the one the thresholds alone allow, before any state moves, from the resistances each gate starts with.

## Results

| Gate | Pulse | Works from | to | Margin | Fails below | Fails above |
| --- | --- | --- | --- | --- | --- | --- |
| MAGIC's | 10 ns | 1.30 V | 1.32 V | ±0.8 % | output of 01 and 10 | inputs of 00 |
| MAGIC's | 100 ns | 1.10 V | 1.16 V | ±2.7 % | output of 01 and 10 | inputs of 00 |
| MAGIC's | 1 µs | 1.03 V | 1.08 V | ±2.4 % | output of 01 and 10 | inputs of 00 |
| Shunted by 20 kΩ | 10 ns | 1.27 V | 1.38 V | ±4.2 % | output of 01 and 10 | inputs of 00 |
| Shunted by 5 kΩ | 10 ns | 1.21 V | 1.57 V | ±12.9 % | output of 01 and 10 | inputs of 00 |
| Shunted by 3 kΩ | 10 ns | 1.17 V | 1.75 V | ±19.9 % | output of 01 and 10 | inputs of 00 |
| Shunted by 2.5 kΩ | 10 ns | 1.15 V | 1.84 V | ±23.1 % | output of 01 and 10 | inputs of 00 |
| **Shunted by 2.2 kΩ** | **10 ns** | **1.14 V** | **1.91 V** | **±25.2 %** | **output of 01 and 10** | **output of 00** |
| Shunted by 2 kΩ | 10 ns | 1.13 V | 1.80 V | ±22.9 % | output of 01 and 10 | output of 00 |
| Shunted by 1.5 kΩ | 10 ns | 1.10 V | 1.52 V | ±16 % | output of 01 and 10 | output of 00 |
| Shunted by 1 kΩ | 10 ns | 1.05 V | 1.24 V | ±8.3 % | output of 01 and 10 | output of 00 |

## What the experiments showed

1. **MAGIC's window is a coincidence of thresholds.**
    A gate with one input on starts as two 1 kΩ resistances in series, so its output turns off only once the pulse passes twice its 0.5 V threshold; a gate with both inputs off starts as 50 kΩ over 1 kΩ, so its inputs take 98 % of the pulse and turn on past 1 V. With these memristors, whose on threshold is twice their off one, the thresholds alone leave 0.995 V to 1.02 V.
1. **Time moves the window but does not open it.**
    In 10 ns the output turns only when driven well past its threshold, its speed growing as the cube of the excess, while inputs barely past theirs move little: the window rises to 1.30 V and 1.32 V. Longer pulses bring it back toward the thresholds' and widen it to ±2.7 % at 100 ns, but no pulse length reaches ±10 %.
1. **A resistor across the inputs separates the two bounds.**
    It takes the voltage off a gate whose inputs are both off, raising the upper bound, while a gate with one input on, already 1 kΩ, barely changes.
    Too small, it leaves so much of the pulse across the output of a gate with both inputs off that the output turns off; too large, the inputs still turn on.
    The thresholds alone balance at 2.1 kΩ, where the inputs' share of the pulse is 2 kΩ; the transients put the best at 2.2 kΩ, where the upper bound passes from the inputs to the output.

## Conclusion

`design.py` is the gate: MAGIC's NOR of three memristors with 2.2 kΩ across its inputs, written by a 10 ns pulse of 1.525 V, working from 1.14 V to 1.91 V, ±25 %.
At 10 % below and above the nominal amplitude, every output ends within 0.002 of its NOR and no input moves by more than 0.006, and ngspice agrees within 1e-4 at each.

## Limits

The agent that wrote this prompt also designed the gate, knowing MAGIC's, and judged the result itself: this is a showcase of what jolty lets an agent see, not a blind result.
The memristors are VTEAM's model with one set of parameters and no spread; a real memristor's thresholds and resistances vary from device to device and from one write to the next, which would narrow the window, and VTEAM's thresholds are sharper than many devices'.
The shunt draws current through every write, and a gate in a crossbar would need it switched out, or shared, which this bench does not model.

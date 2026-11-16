# Logic

Logic devices compute levels rather than voltages.
Their pins join logic nets, which hold a level, `0`, `1` or unknown, and an event scheduler simulates them: it jumps from one change to the next instead of integrating, so logic costs nothing between its edges.
Where a logic pin meets an analog circuit, the engine joins the two through a connect element, so one circuit mixes both.

## Logic devices

| Function | Pins | Behavior |
| --- | --- | --- |
| `logic_cell(inputs, outputs, rise, fall, transition)` | the inputs, then the outputs | each output a function of the inputs |
| `logic_gate(gate, inputs=2, rise, fall, transition)` | `a`, `b` and on, then `y` | `and`, `or`, `nand`, `nor`, `xor`, `xnor`, `not` or `buffer` |
| `logic_source(initial, changes, transition)` | `output` | a level at time zero, then each change |

A cell's output is its truth table, a `0` or a `1` for each row of the inputs, the first input the most significant bit, or a function of the inputs as booleans:

```python
from jolty.units import ps

nand = devices.logic_cell(
    ["a", "b"],
    {"y": lambda a, b: not (a and b)},
    rise=40 * ps,
    fall=30 * ps,
    transition=20 * ps,
)
```

An output fed unknown inputs is known only when every level they could take gives it the same level: a NAND with a `0` on one input is `1`, whatever the other.

An output reaches its new level `rise` after an input changes when it rises, and `fall` after when it falls.
The delay is inertial, as a gate's is: a pulse at the inputs shorter than the delay never reaches the output.
`transition` is how long an edge takes to swing from ground to supply.

A logic source's `changes` are times and levels in time order, each the time its edge crosses half the supply; a level is `0`, `1`, or `None` for unknown.

## Logic nets

A net that only logic pins join is a logic net.
Its level is its outputs' level when they agree, and unknown when they disagree or when no output drives it.
Two rules state these facts: [`driver-conflict`](../reference/rules.md#driver-conflict) for a net several outputs drive, and [`floating-input`](../reference/rules.md#floating-input) for one that inputs read and nothing drives.

At the operating point, the logic settles at once, every delay taken as zero.
A loop that would oscillate, such as an enabled ring of inverters, settles unknown.

## Mixed circuits

An `electrical=True` cell has the terminals `supply` and `ground` too, and its pins may join electrical nets.
Each such pin joins the net through a connect element of its own:

- an input reads the net through a threshold detector, at `threshold` of the supply above ground, one half unless given, and loads the net with `input_capacitance` to ground;
- an output drives the net through a logic driver: a voltage between ground and supply that ramps over the transition, crossing half the supply when the output's level changes, and holds half the supply while the level is unknown.

```python
gate = devices.logic_gate(
    "not", rise=40 * ps, transition=20 * ps, electrical=True, input_capacitance=2 * fF
)
inverter = c.place(gate, name="U1")
c.connect(inverter.supply, vdd)
c.connect(inverter.ground, ground)
c.connect(inverter.a, analog_input)  # an electrical net: a threshold detector reads it
c.connect(inverter.y, load.a)  # an electrical net: a logic driver drives it
```

A logic pin on an electrical net of a cell without a supply is a problem.
The net between a pin and its connect element is a logic net of its own, named after the instance and the pin, such as `U1:a`.

In a transient, the analog steps land where a detector crosses its threshold, and where a driver starts to ramp, so each crossing is found within the relative tolerance of its step.
At the operating point, the logic and the analog circuit settle each other until they agree.

## Levels in results

Every result holds the logic nets' levels beside its unknowns, and `level(net)` reads them, as a number: `0`, `1`, or NaN for unknown.

```python
point.level("carry")  # a float
sweep.level("carry")  # an array, one level per level of the source
trace = waveforms.level("carry")  # a Trace
changes = trace.times, trace.levels  # the level at time zero, then each change
sampled = trace.at(times)  # the level at each of times
```

Reports list each logic net's level at an operating point, and its first and last level and how often it changed over a sweep or a transient.
Plots draw logic nets in a panel of their own, one lane each, unknown levels hatched.

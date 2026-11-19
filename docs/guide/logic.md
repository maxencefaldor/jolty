# Logic

Logic devices compute discrete levels rather than continuous voltages.
Logic pins connect to logic nets holding `0`, `1`, or unknown, and an event scheduler advances directly from transition to transition at zero cost between edges.
Where a logic pin meets an analog net, the engine inserts a connect element so one circuit mixes both domains cleanly.

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

An output reaches its new level `rise` after an input change when rising and `fall` after when falling.
Delay is inertial (matching physical gates): input pulses shorter than the delay never propagate to the output.
`transition` sets the duration for an edge to swing from ground to supply.

A logic source's `changes` list time-ordered `(time, level)` pairs where edges cross half-supply, using `0`, `1`, or `None` for unknown.

## Logic nets

A net connected solely to logic pins is a logic net.
A logic net takes the common output level when all drivers agree, and becomes unknown when drivers disagree or no output drives it.
Two rules report these cases: [`driver-conflict`](../reference/rules.md#driver-conflict) for multi-driver conflicts, and [`floating-input`](../reference/rules.md#floating-input) for undriven inputs.

At the operating point, logic settles instantaneously with zero delay.
Oscillating feedback loops (such as an enabled odd ring of inverters) settle to unknown.

## Mixed circuits

An `electrical=True` cell exposes `supply` and `ground` terminals and allows pins to connect to electrical nets.
Each pin on an electrical net inserts a dedicated connect element:

- inputs sense the net through a threshold detector at `threshold` of supply above ground (default 0.5) and load the net with `input_capacitance` to ground;
- outputs drive the net through a logic driver that ramps between ground and supply over `transition`, crossing half-supply at the event time and holding half-supply while unknown.

<!-- not run: a fragment of a component that defines the nets `vdd` and `analog_input` and places `load` -->

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

Connecting a logic pin to an electrical net on a cell without a supply is an error.
The net between a pin and its connect element forms a dedicated logic net named after the instance and pin, such as `U1:a`.

In a transient, analog steps land where detectors cross thresholds and where drivers start ramping so every crossing meets relative tolerance.
At the operating point, logic and analog domains iterate until both agree.

## Levels in results

Every result stores logic-net levels alongside electrical unknowns, accessible via `logic_level(net)` as `0`, `1`, or `NaN` for unknown.

<!-- not run: it reads the results of a circuit with a logic net `carry`, which the page does not build -->

```python
point.logic_level("carry")  # a float
sweep.logic_level("carry")  # an array, one logic level per value of the source
trace = waveforms.logic_level("carry")  # a LogicTrace
changes = trace.times, trace.levels  # the level at time zero, then each change
sampled = trace.at(times)  # the level at each of times
```

Reports list each logic net's operating-point level, or its initial level, final level, and transition count across a sweep or transient.
Plots render logic nets in a dedicated panel with one lane per net and hatched regions for unknown levels.

# Checks

Every build checks the circuit's structure before any analysis, and reports each fact it finds the way a compiler reports an error.

## What a check finds

The checks read the flattened circuit as a graph: nodes joined by the branches of its devices.
Each device declares its branches: a resistor or a diode conducts, a capacitor stores charge and is open at DC, an inductor is a short at DC, a voltage source fixes a voltage, and a current source fixes a current.
A controlled source's control inputs only sense their nodes.

From that graph alone, the checks find what makes a circuit's equations singular or meaningless: a terminal that connects to nothing, a source shorted by its own net, a loop of voltage sources, or a group of nodes with no DC path to the ground.
The [rules reference](../reference/rules.md) gives each rule's exact condition.

A check also reads each resistor's, capacitor's and inductor's value against the E series of IEC 60063, the preferred values parts are made in: a resistance in none from E3 to E192, or a capacitance or inductance in none from E3 to E24, is one no standard part has, which the rule `non-standard-value` states:

```text
warn[non-standard-value]: the resistance of `R1`, 4800.0 Ω, is in no E series from E3 to E192
  = R1, divider.py:9 in divider()
```

## At the operating point

A simulation also checks the circuit at its plan's operating point, solved with the plan's options, and linearized there, before reading any analysis.

The engine places gmin, a conductance of 1e-12 S, across every junction, so that Newton iteration has a slope to follow where a junction is off.
A net that little else holds, such as one between reverse-biased junctions, takes whatever voltage gmin's picoamperes set, which is the simulator's, not the circuit's.
The rule `gmin-dependence` names every net whose voltage doubling gmin would move, to first order, by more than a hundredth of the larger of that voltage and 1 V:

```text
warn[gmin-dependence]: the operating point depends on gmin: doubling it moves `n` from 999.9899999999999 V by -999.9899999999998 V
```

A circuit whose DC equations have more than one solution can rest in more than one state: a latch in either of its own, a bias circuit that can also rest with no current, which never starts.
The rule `multiple-operating-points` states it when an operating-points analysis would find a solution besides the operating point, naming the nets that differ most between the two:

```text
warn[multiple-operating-points]: the circuit has another DC solution, where `x` at 1.948683389235564 V rather than 0.0513167012238326 V
```

A pole of the linearized circuit with a positive real part is a natural mode that grows away from the operating point, so the circuit does not rest there: an amplifier whose feedback makes it oscillate, or a latch balanced between its states.
The rule `unstable-pole` states it, with the nets where the growing mode is largest, though the plan asks for no stability or pole-zero analysis:

```text
warn[unstable-pole]: the operating point is unstable: its poles at 49999.999999999985 ± 998749.2177719085j rad/s have a positive real part, so an oscillation near 158955.87491756308 Hz grows from it, largest at `n`
```

An oscillator raises it by design, so it is a warning, which a plan for one can ignore.
The poles are found by the QZ algorithm on the dense linearized circuit, which takes about a second at 500 unknowns, so a circuit of more unknowns is checked at its operating point for gmin-dependence alone.

## On analyses' results

Once its analyses end, a simulation checks their results too.
A stability analysis's loop gain raises `phase-margin` when it falls through 0 dB with less than 45° of phase margin, and `gain-margin` when its phase falls through -180° with less than 6 dB of gain margin, each measured as the specs' `phase_margin` and `gain_margin` measure them:

```text
warn[phase-margin]: the loop broken at `VP` has a phase margin of 41.69037114357252° at its crossover, 1039270.2892072713 Hz
```

A spec on the margin states a design's own bound; the rules state the usual ones for any loop a plan measures.

The [ratings](devices.md#ratings) of every device and component instance are held to every operating point and transient the plan solves, at the furthest point of each: a stress beyond a rating raises `over-rating`, an error, and one within it but past 80 % of a bound away from zero, a temperature's from 0 °C, raises `derating`, a warning, as derating guidelines keep that margin for heat, ageing and spread.
A component instance's stresses are at its ports, and its power is all it dissipates; an instance a plan's fidelity replaces by a device is held to its own ratings, not to those of the instances inside it, and a rating of the voltage or the current of a terminal on a logic or a thermal net is not checked, as the terminal has no voltage or current.
A temperature is the simulation's, or the device's own where it has one, raised by the thermal net that an instance's terminal `thermal` joins where it [heats](devices.md#self-heating), and stated in degrees Celsius:

```text
error[over-rating]: the voltage of `C1` at `a` over `b` reaches 20.0 V at 2e-06 s in `tran`, beyond its rating of 16.0 V
  = C1, supply.py:10 in supply()
error[over-rating]: the temperature of `Q1` reaches 159.797947444 °C at the operating point of `op`, beyond its rating of 150.0 °C
  = Q1, follower.py:12 in follower()
```

## Diagnostics

Each diagnostic names its level and rule, states the fact, and lists the instances involved with the script line that placed each one:

```text
error[no-dc-path]: `mid` and `out` have no DC path to `ground`
  = C1, coupled.py:9 in coupled()
  = R1, coupled.py:10 in coupled()
  = C2, coupled.py:11 in coupled()

1 error, 0 warnings
```

A diagnostic states a fact and never how to change the circuit: a capacitor-coupled node may be a mistake, or a stage the agent will bias later.
The command `jolty explain no-dc-path` prints the exact condition behind a rule.

## Levels

A check reports each rule at a level, as ty names them:

- `error`: the fact surely breaks the circuit, and the command exits with 1.
- `warn`: the fact may break the circuit; the command still exits with 0.
- `ignore`: the check leaves the rule out.

Each rule has a default level, and the settings set others in the `[tool.jolty.rules]` table of `pyproject.toml`, where `all` sets every rule and a rule's own entry overrides it:

```toml title="pyproject.toml"
[tool.jolty.rules]
all = "error"
shorted-device = "ignore"
```

The options `--error`, `--warn` and `--ignore` then apply over the settings, for one command:

```sh
uv run jolty build coupled.py --warn no-dc-path
```

## When checks run

`jolty build` and `jolty simulate` check every circuit they build, unless given `--no-check`.
`jolty check` checks a saved circuit file, with no script to run.
A report lists the errors first, up to the setting `max_diagnostics`, and counts the rest.

A circuit the engine cannot elaborate, such as one with a negative resistance or no ground port, is not checked: the command lists every reason it does not elaborate instead, in the same form as a diagnostic but with no rule, since no level turns a problem off.

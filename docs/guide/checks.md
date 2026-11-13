# Checks

Every build checks the circuit's structure before any analysis, and reports each fact it finds the way a compiler reports an error.

## What a check finds

The checks read the flattened circuit as a graph: nodes joined by the branches of its devices.
Each device declares its branches: a resistor or a diode conducts, a capacitor stores charge and is open at DC, an inductor is a short at DC, a voltage source fixes a voltage, and a current source fixes a current.
A controlled source's control inputs only sense their nodes.

From that graph alone, the checks find what makes a circuit's equations singular or meaningless: a terminal that connects to nothing, a source shorted by its own net, a loop of voltage sources, or a group of nodes with no DC path to the ground.
The [rules reference](../reference/rules.md) gives each rule's exact condition.

## At the operating point

A simulation also checks the circuit at its plan's operating point, solved with the plan's options, and linearized there, before reading any analysis.

The engine places gmin, a conductance of 1e-12 S, across every junction, so that Newton iteration has a slope to follow where a junction is off.
A net that little else holds, such as one between reverse-biased junctions, takes whatever voltage gmin's picoamperes set, which is the simulator's, not the circuit's.
The rule `gmin-dependence` names every net whose voltage doubling gmin would move, to first order, by more than a hundredth of the larger of that voltage and 1 V:

```text
warn[gmin-dependence]: the operating point depends on gmin: doubling it moves `n` from 999.9899999999999 V by -999.9899999999998 V
```

A pole of the linearized circuit with a positive real part is a natural mode that grows away from the operating point, so the circuit does not rest there: an amplifier whose feedback makes it oscillate, or a latch balanced between its states.
The rule `unstable-pole` states it, with the nets where the growing mode is largest, though the plan asks for no stability or pole-zero analysis:

```text
warn[unstable-pole]: the operating point is unstable: its poles at 50000.000000000015 ± 998749.2177719092j rad/s have a positive real part, so an oscillation near 158955.8749175632 Hz grows from it, largest at `n`
```

An oscillator raises it by design, so it is a warning, which a plan for one can ignore.
The poles are found by the QZ algorithm on the dense linearized circuit, which takes about a second at 500 unknowns, so a circuit of more unknowns is not checked at its operating point.

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

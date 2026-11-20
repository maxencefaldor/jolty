# Checks

Every build checks the circuit's structure before any analysis, and reports each fact it finds the way a compiler reports an error.

## What a check finds

The checks read the flattened circuit as a graph: nodes joined by the branches of its devices.
Each device declares its branches.
A resistor or diode conducts, a capacitor stores charge and is open at DC, and an inductor is a short at DC.
A voltage source fixes a voltage, and a current source fixes a current.
A controlled source's control inputs only sense their nodes.

From that graph alone, checks identify issues that make circuit equations singular or meaningless: unconnected terminals, shorted sources, voltage-source loops, and node groups lacking a DC path to ground.
The [rules reference](../reference/rules.md) specifies each rule's exact condition.

Checks also compare every resistor, capacitor, and inductor value against IEC 60063 preferred E-series values.
A resistance outside E3 through E192, or a capacitance or inductance outside E3 through E24, matches no standard part and triggers `non-standard-value`:

<!-- output of: jolty build divider.py --no-schematic -->

```text
warn[non-standard-value]: the resistance of `R1`, 4800.0 Ω, is in no E series from E3 to E192
  = R1, divider.py:13 in divider()
```

## At the operating point

A simulation also checks the circuit at its plan's operating point, solved with the plan's options, and linearized there, before reading any analysis.
The checks take the operating point the analyses solved, which the simulation remembers, so they solve it themselves only when no analysis did.

The engine places gmin, a conductance of 1e-12 S, across every junction, so that Newton iteration has a slope to follow where a junction is off.
A net that little else holds, such as one between reverse-biased junctions, takes whatever voltage gmin's picoamperes set, which is the simulator's, not the circuit's.
The rule `gmin-dependence` names every net whose voltage doubling gmin would move, to first order, by more than a hundredth of the larger of that voltage and 1 V:

<!-- output of: jolty simulate leakage.py --no-schematic --no-plot -->

```text
warn[gmin-dependence]: the operating point depends on gmin: to first order, doubling it moves `n` from 999.9899999999999 V by -999.9899999999998 V
```

A circuit whose DC equations admit multiple solutions can settle in more than one state, such as a bistable latch or a bias circuit with an unintended zero-current state.
The rule `multiple-operating-points` reports any additional DC solution found beyond the primary operating point, naming the nets that differ most between solutions:

<!-- output of: jolty simulate bistable.py --no-schematic --no-plot -->

```text
warn[multiple-operating-points]: the circuit has another DC solution, where `x` at 1.9486832970234704 V rather than 0.05131670197609366 V
```

A linearized pole with a positive real part represents a growing natural mode, indicating that the circuit will drift away from the operating point (such as an unstable amplifier or a latch balanced between states).
The rule `unstable-pole` reports growing modes and the nets with the largest amplitude even when the plan omits stability or pole-zero analyses:

<!-- output of: jolty simulate oscillator.py --no-schematic --no-plot -->

```text
warn[unstable-pole]: the operating point is unstable: its poles at 50000.0 ± 998749.217771909j rad/s have a positive real part, so an oscillation near 158955.87491756317 Hz grows from it, largest at `n`
```

An oscillator raises it by design, so it is a warning, which a plan for one can ignore.
The poles are found by the QZ algorithm on the dense linearized circuit, which takes about a second at 500 unknowns, so a circuit of more unknowns is checked at its operating point for gmin-dependence alone.
A circuit with a [transmission line](devices.md#transmission-lines), whose delay gives it infinitely many poles, is not checked for them either.

## On analysis results

Once analyses finish, a simulation checks their results as well.
A stability analysis triggers `low-phase-margin` when loop gain crosses 0 dB with less than 45° of phase margin, and `low-gain-margin` when phase crosses -180° with less than 6 dB of gain margin, matching `phase_margin` and `gain_margin` specs:

<!-- output of: jolty simulate loop.py --no-schematic --no-plot -->

```text
warn[low-phase-margin]: the loop broken at `VP` has a phase margin of 18.171071365739664° at its crossover, 327206.47668315953 Hz
```

A margin spec defines a design-specific bound, while these rules enforce standard defaults on every measured loop.

Every device and component instance's [ratings](devices.md#ratings) are checked at the peak excursion of each operating point and transient.
Exceeding a rating raises `over-rating` (an error), while exceeding 80% of a bound away from zero (or from 0 °C for temperature) raises `near-rating` (a warning) to reflect standard derating margins.
Component stresses evaluate at ports, and component power equals total internal dissipation.
Instances replaced by a characterized model check only the top-level instance ratings rather than internal device ratings, and ratings on logic or thermal nets are skipped because those terminals carry no electrical voltage or current.
Current and power ratings read terminal currents, which operating points and transients [record](results.md#what-a-result-records) automatically regardless of the plan's `record` filter.
Device temperature starts from the simulation or per-device temperature, adds any thermal-net rise for [self-heated](devices.md#self-heating) instances, and is reported in degrees Celsius:

<!-- output of: jolty simulate supply.py --no-schematic --no-plot -->

```text
error[over-rating]: the temperature of `Q1` reaches 157.829574664 °C at the operating point of `op`, beyond its rating of 150.0 °C
  = Q1, supply.py:25 in supply()

error[over-rating]: the voltage of `C1` at `a` over `b` reaches 19.99999999999999 V at 2e-06 s in `tran`, beyond its rating of 16.0 V
  = C1, supply.py:20 in supply()
```

## Diagnostics

Each diagnostic names its level and rule, states the fact, and lists the instances involved with the script line that placed each one:

<!-- output of: jolty build coupled.py --no-schematic -->

```text
error[no-dc-path]: `mid` and `out` have no DC path to `ground`
  = C1, coupled.py:13 in coupled()
  = R1, coupled.py:14 in coupled()
  = C2, coupled.py:15 in coupled()

1 error, 0 warnings
```

A diagnostic states a fact and never how to change the circuit: a capacitor-coupled node may be a mistake, or a stage the agent will bias later.
The command `jolty explain no-dc-path` prints the exact condition behind a rule.

## Levels

A check reports each rule at a level, as ty names them:

- `error`: the fact surely breaks the circuit, and the command exits with 1.
- `warn`: the fact may break the circuit; the command still exits with 0.
- `ignore`: the check omits the rule.

Each rule has a default level, which `[tool.jolty.rules]` in `pyproject.toml` can override using `all` as a baseline and per-rule entries for specific overrides:

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

A circuit the engine cannot elaborate (such as one with a negative resistance or no ground port) skips rule checks.
Instead, the command lists every elaboration problem in the same format without a rule name, since elaboration problems cannot be silenced.

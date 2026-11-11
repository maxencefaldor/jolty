# Specs

Specs state what a circuit must do, as requirements on measurements of its plan's results.
`jolty simulate` checks them after the analyses and states, for each, the value it read and how far it lies inside its bounds.

## Requirements

A plan script may define `specs()`, which returns `Specs`, a list of requirements:

```python
from jolty import Specs
from jolty.specs import bandwidth, gain, overshoot, rise_time
from jolty.units import kHz, ns, us


def specs() -> Specs:
    return Specs(
        [
            gain("bode", "out", at=1 * kHz).within(0, 0.1),
            bandwidth("bode", "out").between(150 * kHz, 170 * kHz),
            (rise_time("step", "out") <= 2.5 * us).with_margin(100 * ns),
            overshoot("step", "out") <= 1,
        ]
    )
```

A measurement names the analysis whose result it reads, such as `bode`, and the signal it reads, such as `out`.
A signal is an unknown, such as the voltage `out` or `X1/mid` or the current `V1:current`, and `reference=` makes a voltage relative to another net's.
Comparing a measurement makes a requirement: `>=` a minimum, `<=` a maximum, `.between(minimum, maximum)`, or `.within(target, tolerance)`.
Bounds are quantities, such as `2.5 * us`; a gain is in decibels, a phase in degrees and an overshoot in percent, so their bounds are numbers.

## Measurements

| Measurement | Reads | From |
| --- | --- | --- |
| `value(analysis, signal, at=None)` | the signal's value, interpolated `at` a time or a level | an operating point, a transient or a DC sweep |
| `maximum`, `minimum`, `peak_to_peak`, `average`, `rms` | the statistic over a window from `start` to `stop`, by default the whole axis | a transient or a DC sweep |
| `rise_time(analysis, signal, low=0.1, high=0.9)` | the time from `low` to `high` of the signal's step, rising or falling | a transient |
| `overshoot(analysis, signal)` | how far the signal passes its final value, in percent of its step | a transient |
| `settling_time(analysis, signal, band=0.02)` | the time until the signal stays within `band` of its step from its final value | a transient |
| `crossing(analysis, signal, level, edge="either")` | the first time the signal crosses `level` | a transient |
| `frequency(analysis, signal, level=None)` | how often the signal rises through `level`, by default the midpoint of its range | a transient |
| `gain(analysis, output, at, over=None)` | the gain in decibels at a frequency, of `output` over `over` or per unit excitation | a small-signal response |
| `phase(analysis, output, at, over=None)` | the phase in degrees at a frequency, unwrapped from the lowest frequency | a small-signal response |
| `bandwidth(analysis, output, drop=3.0, over=None)` | the first frequency where the gain falls `drop` decibels below its gain at the lowest frequency | a small-signal response |
| `phase_margin(analysis, output, over=None)` | 180° plus the phase where the gain first falls through 0 dB | a small-signal response |

Every transient measurement takes a window, `start` and `stop`.
A step runs from the signal's value at the window's start to its value at the window's end, so a window ends once the signal settles.
Values between the points of a result are interpolated linearly, and a small-signal response's over log frequency.

## Verdicts

Each requirement's verdict is one of four:

- **passed:** the value lies within its bounds;
- **marginal:** the value lies within its bounds, by less than the requirement's margin, set with `.with_margin(margin)`;
- **failed:** the value lies outside its bounds;
- **unmeasured:** its analysis failed or is not in the plan, or the measurement cannot be read, such as a crossing that never happens.

A requirement holds at the plan's own conditions and at each of its [corners](simulation.md#corners), and its verdict is the worst of them: failed, then unmeasured, then marginal, then passed, and the least slack among equals.
A verdict states the value read and the slack, how far the value lies inside its bounds, negative outside them, the corner it comes from, and the script line that stated the requirement:

```text
failed: rise time of out in step ≤ 2 µs: read 2.192 µs at the corner hot, slack -192.5 ns
  = rc_filter.py:62 in specs()
```

`jolty simulate` exits with 1 when a requirement fails or is unmeasured, as when an analysis fails at the plan's own conditions or at a corner.
With `--output-format json`, the verdicts come as data, values and slacks in SI base units.

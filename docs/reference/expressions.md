# Expressions

A behavioral source's output is an expression of the voltages and currents it reads, in SI units.

## Variables

An expression reads three kinds of variable:

- `v(a)` is the voltage of terminal `a`, and `v(a, b)` the voltage of `a` over `b`.
- `i(p)` is the current through the source's internal short-circuit sensing port `p`.
- `time` is the time, in seconds: 0 at the operating point.

Numbers are plain, such as `2.5e-3`, and `pi` is π.
Every part without a variable is computed once, when the expression is read.

## Operators

From loosest to tightest:

| Operators | Meaning |
| --- | --- |
| `c ? a : b` | `a` if `c` is true, else `b`; groups to the right |
| `\|\|` | or |
| `&&` | and |
| `==`, `!=` | equal, not equal |
| `<`, `<=`, `>`, `>=` | comparisons |
| `+`, `-` | sum, difference |
| `*`, `/` | product, quotient |
| `-`, `+`, `!` | sign, negation |
| `^` | power; groups to the right, so `2 ^ 3 ^ 2` is 512 |

The power binds tighter than a sign, so `-2 ^ 2` is -4.
A comparison or a logical operator is 1 when true and 0 when false, and any nonzero value is true.

## Functions

| Function | Value |
| --- | --- |
| `abs(x)`, `sqrt(x)`, `exp(x)` | absolute value, square root, exponential |
| `ln(x)`, `log(x)`, `log10(x)` | natural logarithm, twice, and decimal logarithm |
| `sin(x)`, `cos(x)`, `tan(x)`, `asin(x)`, `acos(x)`, `atan(x)` | trigonometric functions, in radians |
| `sinh(x)`, `cosh(x)`, `tanh(x)`, `asinh(x)`, `acosh(x)`, `atanh(x)` | hyperbolic functions |
| `floor(x)`, `ceil(x)` | the integers below and above |
| `sgn(x)` | -1, 0 or 1 |
| `u(x)` | the step: 0 below zero, ½ at zero, 1 above |
| `uramp(x)` | the ramp, `max(x, 0)` |
| `min(a, b)`, `max(a, b)` | the smaller, the larger |
| `pow(a, b)` | `a ^ b` |
| `limit(x, a, b)` | `x` held between `a` and `b`, in either order |
| `if(c, a, b)` | `c ? a : b` |
| `table(x, x1, y1, x2, y2, …)` | the piecewise-linear interpolation of constant points with ascending abscissae, constant beyond the first and the last |
| `pwl(x, x1, y1, x2, y2, …)` | the same, extended beyond the first and the last along the first and the last segments |
| `rounded_table(x, r, x1, y1, x2, y2, …)` | `table` with each corner (including the endpoints) rounded by a parabola spanning `r` times the shorter adjacent segment, for `r` in `[0, 0.5]`, matching ngspice PSpice `TABLE` sources |

A function outside its domain, such as `sqrt(-1)`, is not a number, and the analysis fails.

## Derivatives

The simulator takes every derivative exactly.
Where an expression branches, through a conditional, `min`, `max`, `limit` or a table, the derivative follows the branch taken.

## Jumps

An expression's value jumps whenever an active comparison, `u`, `sgn`, `floor`, or `ceil` changes branch.
Transient analysis locates each jump between time points, steps directly onto the transition, and restarts integration as at a waveform corner.

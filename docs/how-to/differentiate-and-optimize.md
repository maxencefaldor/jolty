# Size a circuit with gradients and JAX

Compute exact analytical parameter gradients for any primary analysis and size component values to hit a target specification.

## 1. Read exact sensitivities from one solve

Use `Sensitivity` when you want the derivative of chosen outputs with respect to every numeric parameter in a circuit.

```python
from jolty import Builder, Role, Sensitivity, Simulation, component, devices
from jolty.units import V, kohm


@component
def divider(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), name="V1")
    upper = c.place(devices.resistor(1 * kohm), name="R1")
    lower = c.place(devices.resistor(1 * kohm), name="R2")
    c.connect(source.positive, upper.a, name="in")
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


sensitivity = Simulation(divider()).run(Sensitivity(["out"]))
round(sensitivity.derivative("out", "R2", "resistance"), 7)  # 0.00125
```

## 2. Optimize parameters with `jolty.function` and `jolty.vjp`

Wrap a circuit and analysis with `jolty.function` to differentiate with respect to a chosen subset of SI parameters.
`jolty.vjp` returns all parameter gradients in one transposed solve regardless of parameter count.

```python
from jolty import OperatingPoint, function, vjp

evaluate = function(divider(), OperatingPoint())
parameters = {"R1": {"resistance": 1000.0}, "R2": {"resistance": 1000.0}}
target = 3.0

for _ in range(15):
    outputs, pullback = vjp(evaluate, parameters)
    error = outputs["out"] - target
    gradient = pullback({"out": 2.0 * error})
    for name in ("R1", "R2"):
        parameters[name]["resistance"] -= 1.5e5 * float(gradient[name]["resistance"])

round(float(evaluate(parameters)["out"][0]), 3)  # 3.0
```

`jolty.function`, `jolty.jvp`, and `jolty.vjp` work across operating point, DC sweep, AC, stability, noise, transient, and periodic steady-state analyses.

## 3. Differentiate and batch inside JAX

Install the optional `jolty[jax]` extra to expose the same differentiable map as a `jax.custom_vjp`.
It composes directly with `jax.jit`, `jax.grad`, `jax.value_and_grad`, and `jax.vmap`.

```python
import jax
import jax.numpy as jnp

import jolty.jax

jax_eval = jolty.jax.function(divider(), OperatingPoint())


def loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
    out = jax_eval(parameters)["out"]
    return jnp.sum((out - 3.0) ** 2)


start = {
    "R1": {"resistance": jnp.asarray(1000.0)},
    "R2": {"resistance": jnp.asarray(1000.0)},
}
value_and_grad = jax.jit(jax.value_and_grad(loss))
loss_value, gradients = value_and_grad(start)
round(float(loss_value), 3)  # 0.25
```

For batched transients or driven periodic steady states under `jax.vmap`, pass explicit `breakpoints` so every sample returns a fixed waveform shape.

## 4. Size across specs and corners with smooth minimax

Compose `jolty.function`, `slack_derivatives`, and `scipy.optimize.minimize` to maximize worst-case requirement slack across operating corners.
Log-sum-exp smoothly approximates the worst normalized slack while keeping exact analytical gradients from one backward pass per corner.

```python
import math

import numpy as np
from scipy.optimize import minimize

from jolty import Ac, Builder, Role, component, devices, function
from jolty.plan import decade
from jolty.simulate import slack_derivatives
from jolty.specs import bandwidth, gain
from jolty.units import K, MHz, V, celsius, kHz, kohm, nF


@component
def rc_filter(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(1 * V), name="V1")
    resistor = c.place(
        devices.resistor(2 * kohm, temperature_coefficient=4e-3 / K), name="R1"
    )
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


bode = Ac("V1", decade(1 * kHz, 10 * MHz, 20))
corners = [
    function(rc_filter(), bode, temperature=celsius(temperature))
    for temperature in (-40, 125)
]
stopband_limit = -14.0
requirements = (
    bandwidth("bode", "out") >= 80 * kHz,
    gain("bode", "out", at=1 * MHz) <= stopband_limit,
)
scales = (80e3, abs(stopband_limit))


def objective(log_values: np.ndarray) -> tuple[float, np.ndarray]:
    resistance, capacitance = math.exp(log_values[0]), math.exp(log_values[1])
    parameters = {"R1": {"resistance": resistance}, "C1": {"capacitance": capacitance}}
    slacks, gradients = [], []
    for corner in corners:
        for item, scale in zip(
            slack_derivatives(corner, parameters, requirements), scales, strict=True
        ):
            slack = 0.0 if item.verdict.slack is None else item.verdict.slack
            slacks.append(slack / scale)
            gradients.append(
                [
                    item.derivatives["R1"]["resistance"] * resistance / scale,
                    item.derivatives["C1"]["capacitance"] * capacitance / scale,
                ]
            )
    beta = 20.0
    scaled = -beta * np.array(slacks)
    shift = float(np.max(scaled))
    weights = np.exp(scaled - shift)
    total = float(np.sum(weights))
    return (shift + math.log(total)) / beta, -(weights / total) @ np.array(gradients)


bounds = [(math.log(200.0), math.log(10e3)), (math.log(0.2e-9), math.log(5e-9))]
solved = minimize(
    objective,
    [math.log(2e3), math.log(1e-9)],
    method="L-BFGS-B",
    jac=True,
    bounds=bounds,
)
sized = {
    "R1": {"resistance": math.exp(solved.x[0])},
    "C1": {"capacitance": math.exp(solved.x[1])},
}
all(
    item.verdict.status == "passed"
    for corner in corners
    for item in slack_derivatives(corner, sized, requirements)
)  # True
```

See [Sensitivities](../guide/simulation.md#sensitivities) for AC complex cotangents, transient adjoints, and oscillator period derivatives.

## 4. Differentiate SPICE and PDK subcircuits

SPICE and PDK subcircuits evaluate parameter expressions and geometry formulas before building primitive devices.
Bind each placed subcircuit with `Library.placement` and pass `subcircuits={path: placement}` to `jolty.function` or `jolty.jax.function`.
Exact dual numbers propagate subcircuit arguments through SPICE expressions and compose with the engine adjoint solver.

```python
from jolty.spice import Library

library = Library()
library.read(
    ".model rsheet r rsh=1000\n"
    ".subckt sized a b w=1u l=2u\n"
    "R1 a b rsheet w={w} l={l}\n"
    ".ends\n",
    "sized.lib",
)
upper = library.placement("sized", w=1e-6, l=2e-6)


@component
def sized_divider(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(4 * V), name="V1")
    top = c.place(upper.component(), name="X1")
    bottom = c.place(devices.resistor(2 * kohm), name="R2")
    c.connect(source.positive, top.a, name="in")
    c.connect(top.b, bottom.a, name="out")
    c.connect(bottom.b, source.negative, ground)


sized_eval = function(sized_divider(), OperatingPoint(), subcircuits={"X1": upper})
outputs, pullback = vjp(sized_eval, {"X1": {"w": 1e-6, "l": 2e-6}})
gradient = pullback({"out": [1.0]})
round(float(outputs["out"][0]), 6)  # 2.0
round(float(gradient["X1"]["w"]), 1)  # 1000000.0
round(float(gradient["X1"]["l"]), 1)  # -500000.0
```

For binned BSIM3 or BSIM4 transistors, call `sized_eval.bins(parameters)` or `library.derivatives(...)` to inspect each device's active `(min_length, max_length, min_width, max_width)` rectangle.

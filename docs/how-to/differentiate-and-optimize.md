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
See [Sensitivities](../guide/simulation.md#sensitivities) for AC complex cotangents, transient adjoints, and oscillator period derivatives.

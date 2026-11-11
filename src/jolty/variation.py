# Copyright 2026 Maxence Faldor
"""Monte Carlo samples of a circuit's tolerances.

A device may give some of its parameters a tolerance, such as a resistor's 1 %. Each
sample draws every such parameter of every device instance, independently, around its
nominal value: evenly within a ``uniform`` tolerance, and normally within a ``normal``
one, the tolerance three standard deviations. Two placements of one component vary
apart, since each is its own instance. A sample replaces the parameters by instance
path, as the engine's overrides do.

Examples:
    >>> import numpy as np
    >>> from jolty import Builder, Role, circuit, component, devices
    >>> from jolty.units import V, kohm
    >>> @component
    ... def divider(c: Builder) -> None:
    ...     ground = c.port("ground", Role.GROUND)
    ...     source = c.place(devices.voltage_source(5 * V), id="V1")
    ...     upper = c.place(devices.resistor(1 * kohm, tolerance=0.01), id="R1")
    ...     c.connect(source.positive, upper.a)
    ...     c.connect(upper.b, source.negative, ground)
    >>> (varied,) = toleranced(circuit(divider()))
    >>> (varied.path, varied.parameter, varied.nominal)
    ('R1', 'resistance', 1000.0)
    >>> drawn = sample((varied,), np.random.default_rng(0))["R1"]["resistance"]
    >>> 990 <= drawn <= 1010
    True
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from jolty import ir


@dataclass(frozen=True, slots=True)
class Varied:
    """One parameter of one device instance that varies within its tolerance."""

    path: str
    """The instance's path, such as ``X1/R1``."""
    parameter: str
    nominal: float
    tolerance: ir.Tolerance


def toleranced(circuit: ir.Circuit) -> tuple[Varied, ...]:
    """Every parameter of every device instance of `circuit` that has a tolerance.

    They come in the order of the hierarchy, each instance's parameters by name, so a
    seed draws the same samples from the same circuit.

    Raises:
        ValueError: When a tolerance's parameter has no number.
    """
    return tuple(_walk(circuit, circuit.top, ""))


def sample(
    varied: Iterable[Varied], generator: np.random.Generator
) -> dict[str, dict[str, float]]:
    """One sample of the `varied` parameters, drawn from `generator`.

    It holds each parameter's value, by instance path then by parameter name.
    """
    drawn: dict[str, dict[str, float]] = {}
    for parameter in varied:
        relative = parameter.tolerance.relative
        match parameter.tolerance.distribution:
            case "uniform":
                deviation = generator.uniform(-relative, relative)
            case "normal":
                deviation = generator.normal(0.0, relative / 3)
        value = parameter.nominal * (1 + float(deviation))
        drawn.setdefault(parameter.path, {})[parameter.parameter] = value
    return drawn


def _walk(circuit: ir.Circuit, component: str, prefix: str) -> Iterable[Varied]:
    for instance in circuit.components[component].instances:
        path = f"{prefix}{instance.id}"
        match instance.of:
            case ir.ComponentOf(component=inner):
                yield from _walk(circuit, inner, f"{path}/")
            case ir.DeviceOf():
                for name, tolerance in sorted(instance.tolerances.items()):
                    match instance.parameters.get(name):
                        case float() | int() as nominal:
                            yield Varied(path, name, float(nominal), tolerance)
                        case _:
                            message = f"{path} gives {name} a tolerance, not a number"
                            raise ValueError(message)

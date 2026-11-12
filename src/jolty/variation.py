# Copyright 2026 Maxence Faldor
"""Monte Carlo samples of a circuit's tolerances.

A device may give some of its parameters a tolerance, such as a resistor's 1 %, or a
supply's DC level, ``waveform.dc``. Each sample draws every such parameter of every
device instance, independently, around its nominal value: evenly within a ``uniform``
tolerance, and normally within a ``normal`` one, the tolerance three standard
deviations. Two placements of one component vary apart, since each is its own instance.
A part from a lot adds the lot's deviation, drawn once per sample for each parameter
and shared by every part in the lot, as the parts on one die drift together. A sample
replaces the parameters by instance path, then by parameter path, as the engine's
overrides do.

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
from typing import Literal

import numpy as np

from jolty import ir


@dataclass(frozen=True, slots=True)
class Varied:
    """One parameter of one device instance that varies within its tolerance."""

    path: str
    """The instance's path, such as ``X1/R1``."""
    parameter: str
    """The parameter's path, such as ``resistance`` or ``waveform.dc``."""
    nominal: float
    tolerance: ir.Tolerance


def toleranced(circuit: ir.Circuit) -> tuple[Varied, ...]:
    """Every parameter of every device instance of `circuit` that has a tolerance.

    They come in the order of the hierarchy, each instance's parameters by path, so a
    seed draws the same samples from the same circuit.

    Raises:
        ValueError: When a tolerance's parameter has no number.
    """
    return tuple(_walk(circuit, circuit.top, ""))


def sample(
    varied: Iterable[Varied], generator: np.random.Generator
) -> dict[str, dict[str, float]]:
    """One sample of the `varied` parameters, drawn from `generator`.

    It holds each parameter's value, by instance path then by parameter path. Each
    parameter draws its own deviation, then the deviation of its lot, if it has one and
    no parameter before it in the lot drew it.
    """
    drawn: dict[str, dict[str, float]] = {}
    lots: dict[tuple[str, str], float] = {}
    for parameter in varied:
        tolerance = parameter.tolerance
        deviation = _deviation(tolerance.relative, tolerance.distribution, generator)
        if (lot := tolerance.lot) is not None:
            key = (lot.name, parameter.parameter)
            if key not in lots:
                lots[key] = _deviation(lot.relative, lot.distribution, generator)
            deviation += lots[key]
        value = parameter.nominal * (1 + deviation)
        drawn.setdefault(parameter.path, {})[parameter.parameter] = value
    return drawn


def _deviation(
    relative: float,
    distribution: Literal["uniform", "normal"],
    generator: np.random.Generator,
) -> float:
    """A relative deviation within `relative`, spread by `distribution`."""
    match distribution:
        case "uniform":
            return float(generator.uniform(-relative, relative))
        case "normal":
            return float(generator.normal(0.0, relative / 3))


def _walk(circuit: ir.Circuit, component: str, prefix: str) -> Iterable[Varied]:
    for instance in circuit.components[component].instances:
        path = f"{prefix}{instance.id}"
        match instance.of:
            case ir.ComponentOf(component=inner):
                yield from _walk(circuit, inner, f"{path}/")
            case ir.DeviceOf():
                for name, tolerance in sorted(instance.tolerances.items()):
                    match ir.parameter(instance.parameters, name):
                        case float() | int() as nominal:
                            yield Varied(path, name, float(nominal), tolerance)
                        case _:
                            message = f"{path} gives {name} a tolerance, not a number"
                            raise ValueError(message)

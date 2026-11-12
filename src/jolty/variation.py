# Copyright 2026 Maxence Faldor
"""Monte Carlo samples of a circuit's tolerances, and their extremes.

A device may give some of its parameters a tolerance, such as a resistor's 1 %, or a
supply's DC level, ``waveform.dc``. Each sample draws every such parameter of every
device instance, independently, around its nominal value, by a factor its
[distribution][jolty.ir.Distribution] spreads. Two placements of one component vary
apart, since each is its own instance. A part from a lot is scaled by the lot's factor
too, drawn once per sample for each parameter and shared by every part in the lot, as
the parts on one die drift together. A sample replaces the parameters by instance path,
then by parameter path, as the engine's overrides do. A worst case moves each parameter,
and each lot's spread, to an extreme of its tolerance instead, or a step toward one.

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

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

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
    parameter draws its own factor, then the factor of its lot, if it has one and no
    parameter before it in the lot drew it.
    """
    drawn: dict[str, dict[str, float]] = {}
    lots: dict[tuple[str, str], float] = {}
    for parameter in varied:
        tolerance = parameter.tolerance
        factor = _factor(tolerance, generator)
        if (lot := tolerance.lot) is not None:
            key = (lot.name, parameter.parameter)
            if key not in lots:
                lots[key] = _factor(lot, generator)
            factor *= lots[key]
        drawn.setdefault(parameter.path, {})[parameter.parameter] = (
            parameter.nominal * factor
        )
    return drawn


type Lot = tuple[str, str]
"""A lot's spread of one parameter: the lot's name and the parameter's path."""


def lot(varied: Varied) -> Lot | None:
    """The lot whose spread `varied` shares, by name and parameter; None for none."""
    shared = varied.tolerance.lot
    return None if shared is None else (shared.name, varied.parameter)


def moved(
    varied: Sequence[Varied], index: int, fraction: float
) -> dict[str, dict[str, float]]:
    """The parameter of `varied` at `index` moved `fraction` of its tolerance up.

    The others keep their nominal values, which the circuit holds already.
    """
    parameter = varied[index]
    value = parameter.nominal * edge(parameter.tolerance, fraction)
    return {parameter.path: {parameter.parameter: value}}


def extreme(
    varied: Iterable[Varied],
    directions: Sequence[float],
    lots: Mapping[Lot, float],
) -> dict[str, dict[str, float]]:
    """Each of `varied` at the extreme of its tolerance in its direction.

    A direction is 1 up, -1 down, or 0 for the nominal value; each part is scaled by its
    lot's spread too, at the extreme in the lot's direction in `lots`, 0 unless given.
    The values come by instance path, then by parameter path.
    """
    found: dict[str, dict[str, float]] = {}
    for parameter, direction in zip(varied, directions, strict=True):
        factor = edge(parameter.tolerance, direction)
        if (shared := parameter.tolerance.lot) is not None:
            key = (shared.name, parameter.parameter)
            factor *= edge(shared, lots.get(key, 0.0))
        found.setdefault(parameter.path, {})[parameter.parameter] = (
            parameter.nominal * factor
        )
    return found


def edge(spread: ir.Tolerance | ir.Lot, direction: float) -> float:
    """The factor `direction` of the way from nominal to the upper extreme of `spread`.

    The extreme is its relative tolerance, three standard deviations of a normal one, or
    its factor ``1 + relative`` for a logarithmic one; -1 is the lower extreme.

    Examples:
        >>> edge(ir.Tolerance(relative=0.05), -1)
        0.95
        >>> edge(ir.Tolerance(relative=1.0, distribution="log_uniform"), -1)
        0.5
    """
    match spread.distribution:
        case "uniform" | "normal":
            return 1 + direction * spread.relative
        case "log_uniform" | "log_normal":
            return math.pow(1 + spread.relative, direction)


def _factor(spread: ir.Tolerance | ir.Lot, generator: np.random.Generator) -> float:
    """A factor drawn from `generator` within `spread`, as its distribution has it."""
    relative = spread.relative
    match spread.distribution:
        case "uniform":
            return 1 + float(generator.uniform(-relative, relative))
        case "normal":
            return 1 + float(generator.normal(0.0, relative / 3))
        case "log_uniform":
            return math.pow(1 + relative, float(generator.uniform(-1.0, 1.0)))
        case "log_normal":
            return math.pow(1 + relative, float(generator.normal(0.0, 1 / 3)))


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

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

The engine draws every sample, so a sample drawn here is the one a batch of runs draws:
each factor from the seed, the sample's index and the name of its part, or of its lot,
alone. Sample `i` is the same however many samples are drawn, and a part added to the
circuit leaves every other part's draws as they were.

Examples:
    >>> from jolty import Builder, Role, circuit, component, devices
    >>> from jolty.units import V, kohm
    >>> @component
    ... def divider(c: Builder) -> None:
    ...     ground = c.port("ground", Role.GROUND)
    ...     source = c.place(devices.voltage_source(5 * V), name="V1")
    ...     upper = c.place(devices.resistor(1 * kohm, tolerance=0.01), name="R1")
    ...     c.connect(source.positive, upper.a)
    ...     c.connect(upper.b, source.negative, ground)
    >>> (varied,) = toleranced(circuit(divider()))
    >>> (varied.instance, varied.parameter, varied.nominal)
    ('R1', 'resistance', 1000.0)
    >>> drawn = sample((varied,), seed=0, index=0)["R1"]["resistance"]
    >>> 990 <= drawn <= 1010
    True
"""

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from jolty import _core, ir


@dataclass(frozen=True, slots=True)
class Toleranced:
    """One parameter of one device instance that varies within its tolerance."""

    instance: str
    """The instance's path, such as ``X1/R1``."""
    parameter: str
    """The parameter's path, such as ``resistance`` or ``waveform.dc``."""
    nominal: float
    tolerance: ir.Tolerance


def toleranced(circuit: ir.Circuit) -> tuple[Toleranced, ...]:
    """Every parameter of every device instance of `circuit` that has a tolerance.

    They come in the order of the hierarchy, each instance's parameters by path, as the
    engine lists them.

    Raises:
        ValueError: When a tolerance's parameter has no number, or when a component
            contains itself through a cycle of placements.
    """
    return tuple(
        Toleranced(
            path, parameter, nominal, ir.Tolerance.model_validate_json(tolerance)
        )
        for path, parameter, nominal, tolerance in _core.toleranced(
            circuit.model_dump_json()
        )
    )


def payload(varied: Iterable[Toleranced]) -> list[JsonValue]:
    """Encodes `varied` as the JSON payload expected by the engine."""
    return [
        {
            "instance": parameter.instance,
            "parameter": parameter.parameter,
            "nominal": parameter.nominal,
            "tolerance": parameter.tolerance.model_dump(mode="json"),
        }
        for parameter in varied
    ]


def sample(
    varied: Iterable[Toleranced], seed: int, index: int
) -> dict[str, dict[str, float]]:
    """The sample `index` of `varied`, drawn from `seed` as the engine draws it.

    It holds each parameter's value, by instance path then by parameter path: its
    nominal value times its own factor and, if it has a lot, the lot's, each drawn from
    the seed, the index, and the name of the part's instance and parameter, or of the
    lot and the parameter, alone.

    Raises:
        OverflowError: When the seed or the index is negative, or 2**64 or more.
    """
    chosen = tuple(varied)
    encoded = json.dumps(payload(chosen), allow_nan=False)
    drawn: dict[str, dict[str, float]] = {}
    for parameter, value in zip(
        chosen, _core.sample(encoded, seed, index), strict=True
    ):
        drawn.setdefault(parameter.instance, {})[parameter.parameter] = value
    return drawn


type LotMember = tuple[str, str]
"""A lot's spread of one parameter: the lot's name and the parameter's path."""


def lot(varied: Toleranced) -> LotMember | None:
    """The lot whose spread `varied` shares, by name and parameter; None for none."""
    shared = varied.tolerance.lot
    return None if shared is None else (shared.name, varied.parameter)


def moved(
    varied: Sequence[Toleranced], index: int, fraction: float
) -> dict[str, dict[str, float]]:
    """The parameter of `varied` at `index` moved `fraction` of its tolerance up.

    The others keep their nominal values, which the circuit holds already.
    """
    parameter = varied[index]
    value = parameter.nominal * edge(parameter.tolerance, fraction)
    return {parameter.instance: {parameter.parameter: value}}


def extreme(
    varied: Iterable[Toleranced],
    directions: Sequence[float],
    lots: Mapping[LotMember, float],
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
        found.setdefault(parameter.instance, {})[parameter.parameter] = (
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

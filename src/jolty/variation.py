# Copyright 2026 Maxence Faldor
"""Monte Carlo tolerance sampling and worst-case parameter extremes.

A device can attach a tolerance to any numeric parameter, such as a 1 % resistance or a
supply DC level (``waveform.dc``). Each Monte Carlo sample scales every toleranced
parameter independently around its nominal value according to its
[distribution][jolty.ir.Distribution]. Separate placements of the same component vary
independently, while parts sharing a lot also multiply by one per-sample lot factor.
Worst-case analysis moves parameters and lots to tolerance extremes instead.

Sampling runs inside the Rust engine so standalone draws match batch runs bit for bit.
Each draw depends strictly on the seed, sample index, and instance or lot parameter
name, so sample `i` is invariant to total sample count or unrelated added parts.

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
    >>> (parameter,) = toleranced(circuit(divider()))
    >>> (parameter.instance, parameter.parameter, parameter.nominal)
    ('R1', 'resistance', 1000.0)
    >>> drawn = sample((parameter,), seed=0, index=0)["R1"]["resistance"]
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
    """Single device parameter that varies within an assigned tolerance."""

    instance: str
    """Hierarchical path of the device instance, such as ``X1/R1``."""
    parameter: str
    """Dotted parameter path, such as ``resistance`` or ``waveform.dc``."""
    nominal: float
    tolerance: ir.Tolerance


def toleranced(circuit: ir.Circuit) -> tuple[Toleranced, ...]:
    """Collects every toleranced device parameter in `circuit` in hierarchy order.

    Entries appear in hierarchy order with each instance's parameters sorted by path,
    matching the engine listing.

    Args:
        circuit: Circuit IR inspected for device parameter tolerances.

    Returns:
        Hierarchy-ordered tuple of `Toleranced` parameter descriptors.

    Raises:
        ValueError: When a toleranced parameter is non-numeric or when component
            placements form a cycle.
    """
    return tuple(
        Toleranced(
            path, parameter, nominal, ir.Tolerance.model_validate_json(tolerance)
        )
        for path, parameter, nominal, tolerance in _core.toleranced(
            circuit.model_dump_json()
        )
    )


def payload(toleranced: Iterable[Toleranced]) -> list[JsonValue]:
    """Encodes `toleranced` as the JSON payload expected by the engine.

    Args:
        toleranced: Toleranced parameter descriptors to serialize.

    Returns:
        JSON-compatible list of tolerance dictionaries for the Rust engine.
    """
    return [
        {
            "instance": parameter.instance,
            "parameter": parameter.parameter,
            "nominal": parameter.nominal,
            "tolerance": parameter.tolerance.model_dump(mode="json"),
        }
        for parameter in toleranced
    ]


def sample(
    toleranced: Iterable[Toleranced], seed: int, index: int
) -> dict[str, dict[str, float]]:
    """Draws Monte Carlo sample `index` of `toleranced` using `seed` in the engine.

    The returned nested mapping stores parameter values keyed by instance path and
    parameter path. Each value multiplies nominal by an independent instance factor
    plus any shared lot factor, seeded by `seed`, `index`, and the parameter key.

    Args:
        toleranced: Toleranced parameter descriptors to sample.
        seed: Unsigned 64-bit pseudorandom generator seed.
        index: Zero-based Monte Carlo sample index.

    Returns:
        Nested dictionary `{instance: {parameter: value}}` in SI base units.

    Raises:
        OverflowError: When `seed` or `index` is negative or at least ``2**64``.
    """
    chosen = tuple(toleranced)
    encoded = json.dumps(payload(chosen), allow_nan=False)
    drawn: dict[str, dict[str, float]] = {}
    for parameter, value in zip(
        chosen, _core.sample(encoded, seed, index), strict=True
    ):
        drawn.setdefault(parameter.instance, {})[parameter.parameter] = value
    return drawn


type LotMember = tuple[str, str]
"""Identifier pairing a shared lot name with a parameter path."""


def lot(toleranced: Toleranced) -> LotMember | None:
    """Returns the shared `(lot_name, parameter)` key for `toleranced`, or None.

    Args:
        toleranced: Single toleranced parameter descriptor to inspect.

    Returns:
        `(lot_name, parameter)` tuple when assigned to a lot, or ``None``.
    """
    shared = toleranced.tolerance.lot
    return None if shared is None else (shared.name, toleranced.parameter)


def moved(
    toleranced: Sequence[Toleranced], index: int, fraction: float
) -> dict[str, dict[str, float]]:
    """Builds an override moving `toleranced[index]` by `fraction` of its tolerance.

    All other circuit parameters remain at their nominal values.

    Args:
        toleranced: Indexed sequence of toleranced parameter descriptors.
        index: Zero-based position of the parameter to perturb.
        fraction: Signed fraction of tolerance range (`-1` to `1`).

    Returns:
        Single-entry override `{instance: {parameter: value}}` in SI base units.
    """
    parameter = toleranced[index]
    value = parameter.nominal * edge(parameter.tolerance, fraction)
    return {parameter.instance: {parameter.parameter: value}}


def extreme(
    toleranced: Iterable[Toleranced],
    directions: Sequence[float],
    lots: Mapping[LotMember, float],
) -> dict[str, dict[str, float]]:
    """Computes parameter overrides placing every entry of `toleranced` at an extreme.

    Each entry in `directions` uses ``1`` for upper, ``-1`` for lower, or ``0`` for
    nominal. Parts belonging to a lot also multiply by the lot extreme looked up in
    `lots` (defaulting to ``0``). Results are nested by instance path and parameter
    path.

    Args:
        toleranced: Toleranced parameters paired elementwise with `directions`.
        directions: Signed per-instance tolerance directions (`-1`, `0`, `1`).
        lots: Signed lot directions keyed by `(lot_name, parameter)`.

    Returns:
        Nested parameter override dictionary `{instance: {parameter: value}}`.
    """
    found: dict[str, dict[str, float]] = {}
    for parameter, direction in zip(toleranced, directions, strict=True):
        factor = edge(parameter.tolerance, direction)
        if (shared := parameter.tolerance.lot) is not None:
            key = (shared.name, parameter.parameter)
            factor *= edge(shared, lots.get(key, 0.0))
        found.setdefault(parameter.instance, {})[parameter.parameter] = (
            parameter.nominal * factor
        )
    return found


def edge(spread: ir.Tolerance | ir.Lot, direction: float) -> float:
    """Computes the multiplicative scale factor at `direction` along `spread`.

    Linear tolerances evaluate ``1 + direction * relative`` (where ``relative`` is
    three standard deviations for normal distributions), and logarithmic tolerances
    evaluate ``(1 + relative) ** direction``. Passing ``-1`` selects the lower extreme.

    Args:
        spread: Instance `Tolerance` or shared `Lot` specification.
        direction: Signed excursion multiplier along the tolerance spread.

    Returns:
        Multiplicative scale factor applied to the nominal parameter value.

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

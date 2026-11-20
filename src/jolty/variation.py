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
    >>> from jolty.units import V, kohm, ohm
    >>> @component
    ... def divider(c: Builder) -> None:
    ...     ground = c.port("ground", Role.GROUND)
    ...     source = c.place(devices.voltage_source(5 * V), name="V1")
    ...     upper = c.place(devices.resistor(1 * kohm, tolerance=0.01), name="R1")
    ...     c.connect(source.positive, upper.a)
    ...     c.connect(upper.b, source.negative, ground)
    >>> (parameter,) = toleranced(circuit(divider()))
    >>> (parameter.instance, parameter.parameter, parameter.nominal == 1 * kohm)
    ('R1', 'resistance', True)
    >>> drawn = sample((parameter,), seed=0, index=0)["R1"]["resistance"]
    >>> 990 * ohm <= drawn <= 1010 * ohm
    True
"""

import json
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from jolty import _core, devices, ir
from jolty.devices import ParameterValue
from jolty.units import Quantity


@dataclass(frozen=True, slots=True)
class Toleranced:
    """Single device parameter that varies within an assigned tolerance."""

    instance: str
    """Hierarchical path of the device instance, such as ``X1/R1``."""
    parameter: str
    """Dotted parameter path, such as ``resistance`` or ``waveform.dc``."""
    nominal: ParameterValue
    tolerance: ir.Tolerance


def toleranced(circuit: ir.Circuit) -> tuple[Toleranced, ...]:
    """Collects every toleranced device parameter in `circuit` in hierarchy order.

    Entries appear in hierarchy order with each instance's parameters sorted by path,
    matching the engine listing. Dimensioned parameters store their `nominal` as a
    [`Quantity`][jolty.units.Quantity] in SI base units.

    Args:
        circuit: Circuit IR inspected for device parameter tolerances.

    Returns:
        Hierarchy-ordered tuple of `Toleranced` parameter descriptors.

    Raises:
        ValueError: When a toleranced parameter is non-numeric or when component
            placements form a cycle.
    """
    raw = _core.toleranced(circuit.model_dump_json())
    if not raw:
        return ()
    return tuple(
        Toleranced(
            path,
            parameter,
            (
                devices.wrap_parameter(target, parameter, nominal)
                if (target := circuit.find(path)) is not None
                else nominal
            ),
            ir.Tolerance.model_validate_json(tolerance),
        )
        for path, parameter, nominal, tolerance in raw
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
            "nominal": (
                parameter.nominal.value
                if isinstance(parameter.nominal, Quantity)
                else parameter.nominal
            ),
            "tolerance": parameter.tolerance.model_dump(mode="json"),
        }
        for parameter in toleranced
    ]


def _resolved_path(prefix: str, suffix: str) -> str:
    """Joins `prefix` and `suffix` with `/` when both are non-empty."""
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}{ir.SEPARATOR}{suffix}"


def _filter_statistical(
    path: str,
    model: ir.Statistical | None,
    replaced: frozenset[str],
    prefixes: tuple[str, ...],
) -> ir.Statistical | None:
    """Filters `model` targets falling under `replaced` or `prefixes`."""
    if model is None or not model.targets:
        return None
    if not replaced:
        return model
    kept = tuple(
        target
        for target in model.targets
        if (resolved := _resolved_path(path, target.instance)) not in replaced
        and not resolved.startswith(prefixes)
    )
    if not kept:
        return None
    return ir.Statistical(steps=model.steps, targets=kept)


def statistical(
    circuit: ir.Circuit, substituted: Collection[str] = ()
) -> tuple[tuple[str, ir.Statistical], ...]:
    """Collects every component statistical model in `circuit` in hierarchy order.

    Skips component subtrees replaced in `substituted` and filters out any hoisted
    targets whose resolved device paths fall beneath a substituted component.

    Args:
        circuit: Circuit IR inspected for component statistical models.
        substituted: Component instance paths replaced by substitute device models.

    Returns:
        Hierarchy-ordered tuple of `(placement_path, statistical)` pairs.
    """
    replaced = frozenset(substituted)
    prefixes = tuple(f"{inside}{ir.SEPARATOR}" for inside in substituted)
    found: list[tuple[str, ir.Statistical]] = []

    def record(path: str, component_name: str) -> ir.Component | None:
        component = circuit.components.get(component_name)
        if component is None:
            return None
        kept = _filter_statistical(path, component.statistical, replaced, prefixes)
        if kept is not None:
            found.append((path, kept))
        return component

    def walk(component_name: str, prefix: str, active: set[str]) -> None:
        if component_name in active:
            return
        component = record(prefix.rstrip(ir.SEPARATOR), component_name)
        if component is None:
            return
        active.add(component_name)
        try:
            for instance in component.instances:
                if not isinstance(instance.of, ir.ComponentOf):
                    continue
                path = f"{prefix}{instance.name}"
                if path not in replaced:
                    walk(instance.of.component, f"{path}{ir.SEPARATOR}", active)
        finally:
            active.remove(component_name)

    walk(circuit.top, "", set())
    return tuple(found)


def statistical_payload(
    circuit: ir.Circuit, substituted: Collection[str] = ()
) -> list[JsonValue]:
    """Encodes component statistical models in `circuit` for the Rust engine.

    Args:
        circuit: Circuit IR whose component statistical models are serialized.
        substituted: Component instance paths replaced by substitute device models.

    Returns:
        JSON-compatible list of `{"path": ..., "statistical": ...}` dictionaries.
    """
    return [
        {"path": path, "statistical": model.model_dump(mode="json")}
        for path, model in statistical(circuit, substituted)
    ]


def sample(
    toleranced: Iterable[Toleranced], seed: int, index: int
) -> dict[str, dict[str, ParameterValue]]:
    """Draws Monte Carlo sample `index` of `toleranced` using `seed` in the engine.

    The returned nested mapping stores parameter values keyed by instance path and
    parameter path, preserving [`Quantity`][jolty.units.Quantity] dimensions from each
    `Toleranced.nominal`. Each value multiplies nominal by an independent instance
    factor plus any shared lot factor, seeded by `seed`, `index`, and the parameter key.

    Args:
        toleranced: Toleranced parameter descriptors to sample.
        seed: Unsigned 64-bit pseudorandom generator seed.
        index: Zero-based Monte Carlo sample index.

    Returns:
        Nested dictionary `{instance: {parameter: value}}` matching each parameter's
        dimension or dimensionless scalar type.

    Raises:
        OverflowError: When `seed` or `index` is negative or at least ``2**64``.
    """
    chosen = tuple(toleranced)
    encoded = json.dumps(payload(chosen), allow_nan=False)
    drawn: dict[str, dict[str, ParameterValue]] = {}
    for parameter, value in zip(
        chosen, _core.sample(encoded, seed, index), strict=True
    ):
        wrapped: ParameterValue = (
            Quantity(value, parameter.nominal.dimension)
            if isinstance(parameter.nominal, Quantity)
            else value
        )
        drawn.setdefault(parameter.instance, {})[parameter.parameter] = wrapped
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
) -> dict[str, dict[str, ParameterValue]]:
    """Builds an override moving `toleranced[index]` by `fraction` of its tolerance.

    All other circuit parameters remain at their nominal values.

    Args:
        toleranced: Indexed sequence of toleranced parameter descriptors.
        index: Zero-based position of the parameter to perturb.
        fraction: Signed fraction of tolerance range (`-1` to `1`).

    Returns:
        Single-entry override `{instance: {parameter: value}}` preserving parameter
        dimensions.
    """
    parameter = toleranced[index]
    value = parameter.nominal * edge(parameter.tolerance, fraction)
    return {parameter.instance: {parameter.parameter: value}}


def extreme(
    toleranced: Iterable[Toleranced],
    directions: Sequence[float],
    lots: Mapping[LotMember, float],
) -> dict[str, dict[str, ParameterValue]]:
    """Computes parameter overrides placing every entry of `toleranced` at an extreme.

    Each entry in `directions` uses ``1`` for upper, ``-1`` for lower, or ``0`` for
    nominal. Parts belonging to a lot also multiply by the lot extreme looked up in
    `lots` (defaulting to ``0``). Results are nested by instance path and parameter
    path, preserving parameter dimensions.

    Args:
        toleranced: Toleranced parameters paired elementwise with `directions`.
        directions: Signed per-instance tolerance directions (`-1`, `0`, `1`).
        lots: Signed lot directions keyed by `(lot_name, parameter)`.

    Returns:
        Nested parameter override dictionary `{instance: {parameter: value}}`.
    """
    found: dict[str, dict[str, ParameterValue]] = {}
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

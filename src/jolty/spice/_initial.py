# Copyright 2026 Maxence Faldor
"""SPICE initial conditions (``.ic``, ``IC=``) and nodesets (``.nodeset``)."""

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from jolty import ir
from jolty.builder import Component
from jolty.errors import SpiceError
from jolty.spice._params import Subcircuit
from jolty.spice.expressions import Function, evaluate, inlined, number, parse
from jolty.spice.netlist import Statement, tokens

GROUND: Final = "ground"
"""The port through which a subcircuit reaches SPICE's global ground, node ``0``."""

INITIAL: Final = frozenset({".ic", ".nodeset"})
"""The statements that give nodes' voltages to start from: a transient's initial
conditions, ``.ic``, and an operating point's nodesets, ``.nodeset``."""

_VOLTAGE_ABSOLUTE_TOLERANCE: Final = 1e-12
"""Absolute tolerance, in volts, when checking initial voltage conditions agree."""

_VOLTAGE_RELATIVE_TOLERANCE: Final = 1e-9
"""Relative tolerance when checking whether initial voltage conditions agree."""


@dataclass(frozen=True, slots=True)
class DeviceVoltage:
    """A capacitor's initial voltage across two nets of a subcircuit (``IC=``)."""

    element: str
    positive: str
    negative: str
    voltage: float


@dataclass(frozen=True, slots=True)
class InitialState:
    """Initial conditions and nodesets collected inside one translated subcircuit."""

    initial_conditions: tuple[tuple[str, float], ...] = ()
    nodesets: tuple[tuple[str, float], ...] = ()
    capacitors: tuple[DeviceVoltage, ...] = ()
    inductors: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class Built:
    """A translated subcircuit: component, net spellings, and initial state."""

    component: Component
    nets: Mapping[str, str]
    initial: InitialState
    temperature_dependent: bool = False


def check_net_name(node: str, statement: Statement) -> None:
    """Rejects node names colliding with `GROUND` or containing `ir.SEPARATOR`.

    Args:
        node: Written SPICE node name to validate.
        statement: Statement used for error attribution on failure.
    """
    if node.casefold() == GROUND:
        message = f"a node named `{node}`, which jolty names the ground port"
        raise statement.fail(message)
    if ir.SEPARATOR in node:
        message = (
            f"a node named `{node}`, whose `{ir.SEPARATOR}` separates the levels "
            "of jolty's paths"
        )
        raise statement.fail(message)


def node_of(written: str) -> str | None:
    """Extracts the single node name from ``V(node)``, or returns None otherwise.

    Args:
        written: Token expected in ``V(node)`` form.

    Returns:
        Trimmed node name inside ``V(...)``, or `None` when malformed or differential.
    """
    if written[:2].casefold() != "v(" or not written.endswith(")"):
        return None
    inside = written[2:-1].strip()
    return inside if inside and "," not in inside else None


def parse_voltages(
    statement: Statement,
    scope: Mapping[str, float],
    functions: Mapping[str, Function],
) -> dict[str, float]:
    """Parses ``V(node)=value`` voltage entries from an ``.ic`` or ``.nodeset`` line.

    Args:
        statement: Logical ``.ic`` or ``.nodeset`` statement to parse.
        scope: Case-folded parameter values in scope for voltage expressions.
        functions: Case-folded user-defined functions inlined before evaluation.

    Returns:
        Mapping from written SPICE node names to evaluated voltages in volts.

    Raises:
        SpiceError: When an entry is malformed or evaluates to a non-finite value.
    """
    words = tokens(statement.text)[1:]
    found: dict[str, float] = {}
    for index in range(0, len(words), 3):
        written = words[index : index + 3]
        match written:
            case [voltage, "=", value] if (node := node_of(voltage)) is not None:
                given = number(value)
                resolved = (
                    evaluate(inlined(parse(value), functions), scope)
                    if given is None
                    else given
                )
                if not math.isfinite(resolved):
                    message = f"initial voltage for `{node}` is not finite: `{value}`"
                    raise statement.fail(message)
                found[node] = resolved
            case _:
                message = (
                    f"a node's voltage is written `V(node)=value`, not "
                    f"`{' '.join(written)}`"
                )
                raise statement.fail(message)
    return found


def port_map(
    subcircuit: Subcircuit,
    built: Built,
    connections: Mapping[str, str] | None,
) -> dict[str, str]:
    """Maps `subcircuit`'s canonical pin names to the external nets in `connections`.

    Args:
        subcircuit: Subcircuit definition supplying formal pin names.
        built: Translated subcircuit build mapping case-folded pins to canonical nets.
        connections: Optional mapping from pin names to connected external net names.

    Returns:
        Mapping from canonical subcircuit port net names to external net names.
    """
    if connections is None or not connections:
        return {}
    canonical = {pin.casefold(): built.nets[pin.casefold()] for pin in subcircuit.pins}
    mapped: dict[str, str] = {}
    for pin, target in connections.items():
        pin_name = canonical.get(pin.casefold())
        if pin_name is None:
            message = f"`{subcircuit.name}` has no pin `{pin}`"
            raise SpiceError(message)
        mapped[pin_name] = GROUND if target in {"0", GROUND} else target
    return mapped


def remap_net(net: str, mapping: Mapping[str, str], prefix: str) -> str:
    """Remaps a net from a child subcircuit into its parent's namespace.

    Args:
        net: Canonical net name inside the child subcircuit.
        mapping: Port mapping from child port nets to parent net names.
        prefix: Hierarchy prefix prepended to unmapped internal nets.

    Returns:
        Corresponding net name in the parent namespace.
    """
    if net == GROUND:
        return GROUND
    if net in mapping:
        return mapping[net]
    return f"{prefix}{net}"


def remap_capacitor(
    capacitor: DeviceVoltage, mapping: Mapping[str, str], prefix: str
) -> DeviceVoltage:
    """Remaps a capacitor's ``IC=`` voltage endpoints into its parent's namespace.

    Args:
        capacitor: Child capacitor initial-voltage record.
        mapping: Port mapping from child port nets to parent net names.
        prefix: Hierarchy prefix prepended to element and internal net names.

    Returns:
        Remapped `DeviceVoltage` expressed in the parent namespace.
    """
    return DeviceVoltage(
        element=f"{prefix}{capacitor.element}",
        positive=remap_net(capacitor.positive, mapping, prefix),
        negative=remap_net(capacitor.negative, mapping, prefix),
        voltage=capacitor.voltage,
    )


def prefixed_inductors(
    inductors: tuple[tuple[str, float], ...], prefix: str
) -> dict[str, float]:
    """Prefixes inductor current unknown names with `prefix`.

    Args:
        inductors: Pairs of `(unknown_name, initial_current)` collected inside a build.
        prefix: Hierarchy prefix prepended to each unknown name.

    Returns:
        Mapping from prefixed inductor unknown names to initial currents in amperes.
    """
    return {f"{prefix}{unknown_name}": current for unknown_name, current in inductors}


def collect_node_voltages(  # noqa: PLR0913 - subcircuit, build, local/top maps and path
    subcircuit: Subcircuit,
    built: Built,
    local_entries: Sequence[tuple[str, float]],
    top_level: Mapping[str, float],
    *,
    path: str,
    connections: Mapping[str, str] | None,
) -> tuple[dict[str, str], str, list[tuple[str, float]]]:
    """Gathers and remaps ``.ic`` or ``.nodeset`` voltages for a subcircuit.

    Args:
        subcircuit: Subcircuit definition being queried.
        built: Nominal build of `subcircuit` supplying canonical net names.
        local_entries: Inner `(net, voltage)` pairs collected inside `built`.
        top_level: Library-level written node voltages overriding inner declarations.
        path: Hierarchical placement path, or empty when `subcircuit` is top-level.
        connections: Optional mapping from subcircuit pin names to external nets.

    Returns:
        Tuple of `(port_mapping, hierarchy_prefix, remapped_entries)`.
    """
    mapping = port_map(subcircuit, built, connections)
    prefix = f"{path}{ir.SEPARATOR}" if path else ""
    entries: list[tuple[str, float]] = [
        (remap_net(net, mapping, prefix), voltage) for net, voltage in local_entries
    ]
    if not path:
        for node, voltage in top_level.items():
            resolved = GROUND if node == "0" else built.nets.get(node.casefold(), node)
            entries.append((remap_net(resolved, mapping, prefix), voltage))
    return mapping, prefix, entries


def resolve_node_voltages(
    entries: Sequence[tuple[str, float]],
    what: str,
) -> dict[str, float]:
    """Resolves ``.ic`` or ``.nodeset`` entries, checking ground consistency.

    Args:
        entries: Ordered `(net, voltage)` pairs where later entries override earlier
            assignments.
        what: Noun phrase describing the directive kind in error messages.

    Returns:
        Mapping from non-ground net names to resolved voltages in volts.

    Raises:
        SpiceError: When ground is assigned a non-zero voltage.
    """
    known: dict[str, float] = {}
    for net, voltage in entries:
        if net == GROUND:
            if not math.isclose(
                voltage, 0.0, rel_tol=0.0, abs_tol=_VOLTAGE_ABSOLUTE_TOLERANCE
            ):
                message = (
                    f"{what} on node `0`, SPICE's global ground, is {voltage!r} V, "
                    "while ground is 0 V"
                )
                raise SpiceError(message)
            continue
        known[net] = voltage
    return known


def _capacitor_graph(
    capacitors: Sequence[DeviceVoltage],
) -> tuple[dict[str, list[tuple[str, float, str]]], list[str]]:
    """Builds the potential-difference graph across capacitor ``IC=`` edges."""
    adjacency: dict[str, list[tuple[str, float, str]]] = {}
    ordered_nodes: list[str] = []
    seen_nodes: set[str] = set()
    for capacitor in capacitors:
        if capacitor.positive == capacitor.negative:
            if not math.isclose(
                capacitor.voltage,
                0.0,
                rel_tol=0.0,
                abs_tol=_VOLTAGE_ABSOLUTE_TOLERANCE,
            ):
                message = (
                    f"initial voltage of shorted capacitor `{capacitor.element}` is "
                    f"{capacitor.voltage!r} V, while its terminals share "
                    f"`{capacitor.positive}`"
                )
                raise SpiceError(message)
            continue
        adjacency.setdefault(capacitor.negative, []).append(
            (capacitor.positive, capacitor.voltage, capacitor.element)
        )
        adjacency.setdefault(capacitor.positive, []).append(
            (capacitor.negative, -capacitor.voltage, capacitor.element)
        )
        for node in (capacitor.negative, capacitor.positive):
            if node not in seen_nodes:
                seen_nodes.add(node)
                ordered_nodes.append(node)
    return adjacency, ordered_nodes


def _propagate_voltages(
    known: dict[str, float],
    adjacency: Mapping[str, Sequence[tuple[str, float, str]]],
    seeds: Sequence[str],
    visited: set[str],
) -> None:
    """Propagates known node voltages across capacitor ``IC=`` edges, checking loops."""
    queue = deque(seeds)
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        base = known[current]
        for neighbor, delta, element in adjacency.get(current, ()):
            proposed = base + delta
            if neighbor not in known:
                known[neighbor] = proposed
                queue.append(neighbor)
                continue
            existing = known[neighbor]
            if not math.isclose(
                existing,
                proposed,
                rel_tol=_VOLTAGE_RELATIVE_TOLERANCE,
                abs_tol=_VOLTAGE_ABSOLUTE_TOLERANCE,
            ):
                label = "node `0` (ground)" if neighbor == GROUND else f"`{neighbor}`"
                message = (
                    f"contradictory initial conditions on {label}: "
                    f"{existing!r} V and {proposed!r} V across `{element}`"
                )
                raise SpiceError(message)


def solve_initial_voltages(
    entries: Sequence[tuple[str, float]],
    capacitors: Sequence[DeviceVoltage],
) -> dict[str, float]:
    """Solves ``.ic`` node voltages and capacitor ``IC=`` voltage differences.

    Args:
        entries: Explicit `(net, voltage)` pairs from ``.ic`` statements.
        capacitors: Capacitor differential voltage constraints from ``IC=`` parameters.

    Returns:
        Mapping from constrained non-ground net names to solved voltages in volts.

    Raises:
        SpiceError: When a shorted capacitor has a non-zero ``IC=``, ground is assigned
            a non-zero voltage, or two paths disagree on a net's initial voltage.
    """
    known: dict[str, float] = {
        GROUND: 0.0,
        **resolve_node_voltages(entries, "an initial condition"),
    }
    if not capacitors:
        return {net: voltage for net, voltage in known.items() if net != GROUND}
    adjacency, ordered_nodes = _capacitor_graph(capacitors)
    visited: set[str] = set()
    _propagate_voltages(known, adjacency, list(known), visited)
    for node in ordered_nodes:
        if node not in known:
            known[node] = 0.0
            _propagate_voltages(known, adjacency, [node], visited)
    return {net: voltage for net, voltage in known.items() if net != GROUND}

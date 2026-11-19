# Copyright 2026 Maxence Faldor
"""Rating stress evaluation and result-level diagnostics."""

import functools
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
from pydantic import JsonValue

from jolty import checks, ir, results, specs
from jolty.plan import Conditions
from jolty.units import ZERO_CELSIUS

_UNITS: Final[dict[ir.Stress, str]] = {
    "voltage": "V",
    "current": "A",
    "power": "W",
    "temperature": "°C",
}


@dataclass(frozen=True, slots=True)
class Joined:
    """Where a terminal of a device, a component instance or a logic device joins."""

    node: str | None
    """The unknown that is its voltage, by name; None on the ground."""
    thermal: bool
    """Whether it is thermal: its voltage a temperature above the ambient, and its
    current the heat flowing into it."""


def rated_instances(
    circuit: ir.Circuit, substituted: Collection[str]
) -> Iterator[tuple[str, ir.Instance]]:
    """Every rated instance of `circuit`, by path, each before those it places.

    The walk skips the inside of each component instance whose path is among
    `substituted`, a model standing in for it, and of each component that holds no
    rated instance.
    """
    prefixes = tuple(f"{path}/" for path in substituted)
    holds: dict[str, bool] = {}

    def holding(component: str) -> bool:
        """Whether `component` holds a rated instance, at any depth."""
        if component not in holds:
            holds[component] = any(
                instance.ratings
                or (
                    isinstance(instance.of, ir.ComponentOf)
                    and holding(instance.of.component)
                )
                for instance in circuit.components[component].instances
            )
        return holds[component]

    for path, instance in circuit.walk(
        descend=lambda inner: (
            isinstance(inner.of, ir.ComponentOf) and holding(inner.of.component)
        )
    ):
        if prefixes and path.startswith(prefixes):
            continue
        if instance.ratings:
            yield path, instance


def _own(ambient: float, parameters: Mapping[str, JsonValue]) -> float:
    """The temperature of an instance with `parameters`, the simulation's `ambient`.

    It is the instance's own, held or offset from the simulation's, if it has one.
    """
    match parameters:
        case {"temperature": int() | float() as held}:
            return float(held)
        case {"temperature_offset": int() | float() as offset}:
            return ambient + offset
    return ambient


@dataclass(frozen=True, slots=True)
class _Terminals:
    """The voltage at and the current into each reached terminal, over a result."""

    joined: Callable[[str, str], Joined | None]
    ambient: float
    overrides: Mapping[str, Mapping[str, float]]
    columns: Mapping[str, int]
    values: results.Real
    currents: results.Real
    recording: results.Recording

    def stress(
        self, path: str, instance: ir.Instance, rating: ir.Rating
    ) -> results.Real | None:
        """The stress `rating` limits on the instance at `path`, at each point.

        There is none when a rated terminal is on a logic or a thermal net, or the
        result recorded no current into it and the stress needs one.
        """
        if rating.stress == "temperature":
            replaced = {**instance.parameters, **self.overrides.get(path, {})}
            return _own(self.ambient, replaced) + self._rise(path)
        named = rating.terminals or tuple(
            terminal
            for terminal in instance.terminals
            if self._electrical(path, terminal)
        )
        if not all(self._electrical(path, terminal) for terminal in named):
            return None
        if rating.stress == "voltage":
            first, second = named
            return self._voltage(path, first) - self._voltage(path, second)
        columns = [self.recording.find(path, terminal) for terminal in named]
        if None in columns:
            return None
        currents = [self.currents[:, column] for column in columns]
        if rating.stress == "current":
            (current,) = currents
            return current
        return sum(
            (
                self._voltage(path, terminal) * current
                for terminal, current in zip(named, currents, strict=True)
            ),
            start=np.zeros(self.values.shape[0]),
        )

    def _electrical(self, path: str, terminal: str) -> bool:
        """Whether `terminal` of the instance at `path` is on an electrical net."""
        found = self.joined(path, terminal)
        return found is not None and not found.thermal

    def _rise(self, path: str) -> results.Real:
        """The temperature above the ambient of the instance at `path`."""
        found = self.joined(path, "thermal")
        if found is None or not found.thermal or found.node is None:
            return np.zeros(self.values.shape[0])
        return self.values[:, self.columns[found.node]]

    def _voltage(self, path: str, terminal: str) -> results.Real:
        found = self.joined(path, terminal)
        if found is None or found.node is None:
            return np.zeros(self.values.shape[0])
        return self.values[:, self.columns[found.node]]


@dataclass(frozen=True, slots=True)
class Stressed:
    """A rated stress of a device over one analysis's points, judged by its rating."""

    analysis: str
    path: str
    instance: ir.Instance
    rating: ir.Rating
    stress: results.Real
    times: results.Real | None
    """The time of each point, or None at an operating point."""

    def diagnostics(self) -> list[checks.Diagnostic]:
        """The diagnostics of over-rating and near-rating, one per bound at most."""
        found: list[checks.Diagnostic] = []
        high = int(np.argmax(self.stress))
        low = int(np.argmin(self.stress))
        # Derating keeps its margin from zero, a temperature's from 0 °C.
        origin = ZERO_CELSIUS if self.rating.stress == "temperature" else 0.0
        for point, bound, sign in (
            (high, self.rating.maximum, 1),
            (low, self.rating.minimum, -1),
        ):
            if bound is None:
                continue
            value = float(self.stress[point])
            reach = sign * (bound - origin)
            if sign * value > sign * bound:
                found.append(self._diagnostic(checks.OVER_RATING, point, bound, value))
            elif reach > 0 and sign * (value - origin) > checks.DERATING_FACTOR * reach:
                found.append(self._diagnostic(checks.NEAR_RATING, point, bound, value))
        return found

    def _diagnostic(
        self, rule: checks.Rule, point: int, bound: float, value: float
    ) -> checks.Diagnostic:
        unit = _UNITS[self.rating.stress]
        terminals = [f"`{terminal}`" for terminal in self.rating.terminals]
        described = {
            "voltage": f"the voltage of `{self.path}` at {' over '.join(terminals)}",
            "current": f"the current into `{self.path}` at {''.join(terminals)}",
            "power": f"the power `{self.path}` dissipates",
            "temperature": f"the temperature of `{self.path}`",
        }[self.rating.stress]
        if self.rating.stress == "temperature":
            # In degrees Celsius, as datasheets state them, rounded past the
            # conversion's own error.
            bound, value = (round(each - ZERO_CELSIUS, 9) for each in (bound, value))
        where = (
            f"at the operating point of `{self.analysis}`"
            if self.times is None
            else f"at {float(self.times[point])!r} s in `{self.analysis}`"
        )
        moves = "reaches" if value >= 0 else "falls to"
        if rule is checks.OVER_RATING:
            beyond = "beyond" if value > bound else "below"
            judged = f"{beyond} its rating of {bound!r} {unit}"
        else:
            judged = (
                f"past {checks.DERATING_FACTOR:.0%} of its rating of {bound!r} {unit}"
            )
        return checks.Diagnostic(
            rule=rule.name,
            level=rule.level,
            message=f"{described} {moves} {value!r} {unit} {where}, {judged}",
            instances=(
                checks.Location(
                    path=self.path, source_location=self.instance.source_location
                ),
            ),
            nets=(),
        )


def check_results(
    circuit: ir.Circuit,
    substituted: Collection[str],
    conditions: Conditions,
    joined: Callable[[str, str], Joined | None],
    solved: Mapping[str, results.AnalysisResult],
) -> tuple[checks.Diagnostic, ...]:
    """Every diagnostic of the rules on the analyses' results `solved`."""
    found = list(specs.diagnose(solved))
    cached_joined = functools.cache(joined)
    rated = [
        (path, instance, rating)
        for path, instance in rated_instances(circuit, substituted)
        if any(
            cached_joined(path, terminal) is not None for terminal in instance.terminals
        )
        for rating in instance.ratings
    ]
    for name, result in solved.items():
        match result:
            case results.OperatingPointResult():
                values = result.values[np.newaxis, :]
                currents = result.currents.reshape(1, -1)
                times = None
            case results.TransientResult():
                values, currents, times = result.values, result.currents, result.times
            case _:
                continue
        columns = {
            unknown.name: column for column, unknown in enumerate(result.unknowns)
        }
        terminals = _Terminals(
            cached_joined,
            conditions.temperature,
            conditions.overrides,
            columns,
            values,
            currents,
            result.recording,
        )
        for path, instance, rating in rated:
            stress = terminals.stress(path, instance, rating)
            if stress is not None:
                judged = Stressed(name, path, instance, rating, stress, times)
                found.extend(judged.diagnostics())
    return tuple(found)

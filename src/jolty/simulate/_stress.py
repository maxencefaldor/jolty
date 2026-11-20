# Copyright 2026 Maxence Faldor
"""Rating stress evaluation and result-level diagnostics."""

import functools
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
from pydantic import JsonValue

from jolty import checks, devices, ir, results, specs
from jolty.plan import Conditions
from jolty.units import TEMPERATURE, ZERO_CELSIUS

_UNITS: Final[dict[ir.Stress, str]] = {
    "voltage": "V",
    "current": "A",
    "power": "W",
    "temperature": "°C",
}


@dataclass(frozen=True, slots=True)
class Joined:
    """Connection target of a device, component instance, or logic terminal."""

    node: str | None
    """Name of the connected node voltage unknown, or None when grounded."""
    thermal: bool
    """True for thermal terminals (voltage is temperature rise; current is heat)."""


def rated_instances(
    circuit: ir.Circuit, substituted: Collection[str]
) -> Iterator[tuple[str, ir.Instance]]:
    """Yields every rated instance in `circuit` in preorder by hierarchical path.

    The traversal skips subtrees replaced by `substituted` models as well as
    components containing no rated instances.

    Args:
        circuit: Elaborated circuit whose component hierarchy is traversed.
        substituted: Component instance paths replaced by substitute device models.

    Returns:
        Iterator of `(path, instance)` pairs for active rated instances.
    """
    prefixes = tuple(f"{path}/" for path in substituted)
    holds: dict[str, bool] = {}

    def holding(component: str) -> bool:
        """Returns whether `component` contains a rated instance at any depth."""
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
    """Computes the instance base temperature from `parameters` and `ambient`.

    Uses an explicit instance ``temperature`` or ``temperature_offset`` when present,
    defaulting to the simulation ambient temperature.
    """
    match parameters:
        case {"temperature": int() | float() as held}:
            return float(held)
        case {"temperature_offset": int() | float() as offset}:
            return ambient + offset
    return ambient


@dataclass(frozen=True, slots=True)
class _Terminals:
    """Terminal voltage, current, and temperature-rise evaluator over one result."""

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
        """Computes the stress trajectory for `rating` on `instance` at `path`.

        Returns None when a rated terminal connects to a logic or thermal net, or when
        the result did not record a required terminal current.
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
        """Returns whether `terminal` on `path` connects to an electrical net."""
        found = self.joined(path, terminal)
        return found is not None and not found.thermal

    def _rise(self, path: str) -> results.Real:
        """Returns the temperature rise above ambient for the instance at `path`."""
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
    """Rated stress trajectory of one device over an analysis against its bounds."""

    analysis: str
    path: str
    instance: ir.Instance
    rating: ir.Rating
    stress: results.Real
    times: results.Real | None
    """Time point array for transient analyses, or None for an operating point."""

    def diagnostics(self) -> list[checks.Diagnostic]:
        """Collects over-rating and near-rating diagnostics, at most one per bound.

        Returns:
            Rating diagnostics triggered at maximum or minimum stress extremes.
        """
        found: list[checks.Diagnostic] = []
        high = int(np.argmax(self.stress))
        low = int(np.argmin(self.stress))
        # Derating measures margin from zero, or from 0 °C for temperatures.
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
            # Convert to Celsius as stated in datasheets, rounding past float noise.
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
    """Evaluates result-level rules across `solved` and returns all diagnostics.

    Args:
        circuit: Elaborated circuit containing device rating declarations.
        substituted: Component instance paths replaced by substitute device models.
        conditions: Simulation temperature and parameter overrides.
        joined: Lookup returning the connected `Joined` node for an instance terminal.
        solved: Mapping from analysis names to completed analysis results.

    Returns:
        Stability-margin and device-stress diagnostics collected across `solved`.
    """
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
            conditions.temperature.to(TEMPERATURE, "a temperature"),
            devices.convert_overrides(circuit, conditions.overrides),
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

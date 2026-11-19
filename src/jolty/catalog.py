# Copyright 2026 Maxence Faldor
"""Orderable parts, IEC 60063 preferred values, and bills of materials.

A [`Catalog`][jolty.catalog.Catalog] lists purchasable parts with unit prices,
packages, footprints, and lifecycle states. Each catalog file stores one catalog as
JSON. A [`Bill`][jolty.catalog.Bill] prices every circuit instance or a single
component subtree.

Numbered instances select the cheapest matching orderable part with that part number.
Unnumbered resistors, capacitors, and inductors select the cheapest orderable part
matching their kind, value, and tolerance. Obsolete parts are never selected, and
unpriced instances record the reason no part matched.

IEC 60063 E-series tables space preferred values logarithmically per decade, from E24
(24 values per decade, 5 % tolerance) to E192 (0.5 % tolerance and tighter).

Examples:
    >>> values("E12", 1e3, 2e3)
    (1000.0, 1200.0, 1500.0, 1800.0)
    >>> neighbors("E24", 4.8e3)
    (4700.0, 5100.0)
"""

import bisect
import math
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from jolty import _core, ir
from jolty.units import format_si

type PartKind = Literal["resistor", "capacitor", "inductor"]
"""Device kind that a catalog prices by value and tolerance."""

type Series = Literal["E3", "E6", "E12", "E24", "E48", "E96", "E192"]
"""IEC 60063 E-series identifier for preferred values."""

type Lifecycle = Literal["active", "not-recommended", "obsolete"]
"""Manufacturer lifecycle status: active, not recommended for new designs, or
obsolete."""

VALUED: Final[dict[PartKind, str]] = {
    "resistor": "resistance",
    "capacitor": "capacitance",
    "inductor": "inductance",
}
"""Mapping from each catalog-priced device kind to its primary value parameter."""

SYMBOLS: Final[dict[PartKind, str]] = {
    "resistor": "Ω",
    "capacitor": "F",
    "inductor": "H",
}
"""Mapping from each catalog-priced device kind to its SI unit symbol."""


@cache
def mantissas(series: Series) -> tuple[float, ...]:
    """Returns the normalized decade mantissas of `series` from 1 to 10.

    E3 through E24 use historical IEC 60063 values. E48 through E192 round decade
    powers to three significant digits, except for 9.20 in E192.

    Examples:
        >>> mantissas("E96")[:4]
        (1.0, 1.02, 1.05, 1.07)
    """
    return tuple(_core.mantissas(series))


def values(series: Series, low: float, high: float) -> tuple[float, ...]:
    """Returns every `series` value from `low` to `high` inclusive in ascending order.

    Raises:
        ValueError: When `low` is not positive or exceeds `high`.
    """
    if not 0 < low <= high:
        message = f"values lie between two positive bounds, not {low!r} and {high!r}"
        raise ValueError(message)
    first = math.floor(math.log10(low)) - 1
    last = math.floor(math.log10(high)) + 1
    found = (
        _value(mantissa, decade)
        for decade in range(first, last + 1)
        for mantissa in mantissas(series)
    )
    return tuple(value for value in found if _within(value, low, high))


def neighbors(series: Series, value: float) -> tuple[float, float]:
    """Returns the `series` values immediately at or below and at or above `value`.

    Raises:
        ValueError: When `value` is not positive.
    """
    if value <= 0:
        message = f"a value of an E series is positive, not {value!r}"
        raise ValueError(message)
    around = values(series, value / 10, value * 10)
    index = bisect.bisect_left(around, value)
    if index < len(around) and _same(around[index], value):
        return around[index], around[index]
    return around[index - 1], around[index]


def _value(mantissa: float, decade: int) -> float:
    """Returns `mantissa` scaled by ``10 ** decade`` and rounded to mantissa digits."""
    return float(f"{mantissa}e{decade}")


def _same(value: float, other: float) -> bool:
    """Returns whether two values are equal within rounding precision."""
    return math.isclose(value, other, rel_tol=1e-9)


def _within(value: float, low: float, high: float) -> bool:
    return (low < value or _same(value, low)) and (value < high or _same(value, high))


type Pins = dict[str, tuple[Annotated[str, Field(min_length=1)], ...]]
"""Mapping from instance terminal or port names to package pad tuples.

For example, ``{"collector": ("3",), "base": ("2",), "emitter": ("1",)}`` maps BJT
terminals to package pads; an empty tuple denotes an unconnected terminal."""


class Part(ir.FrozenModel):
    """Orderable part entry priced by device value and tolerance, part number, or both.

    A part without a device kind matches any device or component carrying its part
    number. Package footprint and pin mappings support [KiCad netlist][jolty.kicad]
    export.
    """

    kind: PartKind | None = Field(default=None, exclude_if=lambda kind: kind is None)
    value: float | None = Field(
        default=None,
        gt=0,
        exclude_if=lambda value: value is None,
        description="Nominal component value in SI base units.",
    )
    tolerance: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        exclude_if=lambda tolerance: tolerance is None,
        description="Relative manufacturing tolerance, such as 0.01 for 1 %.",
    )
    price: float = Field(ge=0, description="Unit price in the catalog currency.")
    part_number: ir.PartNumber | None = Field(
        default=None,
        exclude_if=lambda part_number: part_number is None,
        description="Manufacturer part number.",
    )
    package: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda package: package is None,
        description="Physical package name, such as 0603 or TO-92.",
    )
    footprint: str | None = Field(
        default=None,
        pattern=r"^[^:]+:[^:]+$",
        exclude_if=lambda footprint: footprint is None,
        description="KiCad library footprint name, such as "
        "Resistor_SMD:R_0603_1608Metric.",
    )
    pins: Pins | None = Field(
        default=None,
        exclude_if=lambda pins: pins is None,
        description="Package pad numbers mapped by instance terminal or port name.",
    )
    lifecycle: Lifecycle = Field(
        default="active", exclude_if=lambda lifecycle: lifecycle == "active"
    )

    @model_validator(mode="after")
    def _identified(self) -> Self:
        """Validates part identification fields and checks for duplicate pads."""
        valued = (self.kind, self.value, self.tolerance)
        if any(each is None for each in valued) and any(
            each is not None for each in valued
        ):
            message = "a part states its kind, value and tolerance together"
            raise ValueError(message)
        if self.kind is None and self.part_number is None:
            message = "a part states a kind, value and tolerance, or a part number"
            raise ValueError(message)
        for terminal, pads in (self.pins or {}).items():
            if len(set(pads)) < len(pads):
                message = f"a part's pins give {terminal} a pad twice"
                raise ValueError(message)
        return self

    @property
    def orderable(self) -> bool:
        """Orderability status; true when the part lifecycle is not obsolete."""
        return self.lifecycle != "obsolete"

    def describe(self) -> str:
        """Formats a human-readable summary of the part excluding price."""
        said = []
        if self.kind is not None and self.value is not None:
            said.append(
                f"{self.kind} of {format_si(self.value, SYMBOLS[self.kind])} "
                f"within {self.tolerance:.6g}"
            )
        if self.part_number is not None:
            said.append(self.part_number)
        if self.package is not None:
            said.append(f"in {self.package}")
        if self.lifecycle == "not-recommended":
            said.append("not recommended for new designs")
        return ", ".join(said)


class Catalog(ir.FrozenModel):
    """Collection of orderable parts priced in a single currency."""

    format: Literal["jolty-catalog"] = "jolty-catalog"
    version: Literal[3] = 3
    currency: str = Field(description="Currency code for all part prices, such as USD.")
    parts: tuple[Part, ...]

    @classmethod
    def load(cls, path: Path) -> Catalog:
        """Loads and validates a catalog from the JSON file at `path`."""
        return cls.model_validate_json(path.read_bytes())

    def save(self, path: Path) -> None:
        """Writes the catalog to `path` as indented JSON."""
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def cheapest(self, kind: PartKind, value: float, tolerance: float) -> Part | None:
        """Returns the cheapest orderable part matching `kind`, `value`, and tolerance.

        Price ties break in favor of the first listed part. Returns ``None`` when no
        orderable part matches.
        """
        return _cheapest(
            part for part in self.matching(kind, value, tolerance) if part.orderable
        )

    def matching(
        self, kind: PartKind, value: float, tolerance: float
    ) -> tuple[Part, ...]:
        """Returns all parts matching `kind`, `value`, and `tolerance`."""
        return tuple(
            part for part, held in self._valued(kind, tolerance) if _same(held, value)
        )

    def values(self, kind: PartKind, tolerance: float) -> tuple[float, ...]:
        """Returns sorted distinct values of orderable parts within `tolerance`."""
        found = sorted(
            held for part, held in self._valued(kind, tolerance) if part.orderable
        )
        distinct: list[float] = []
        for value in found:
            if not distinct or not _same(distinct[-1], value):
                distinct.append(value)
        return tuple(distinct)

    def part_for(self, instance: ir.Instance) -> Part | str:
        """Returns the cheapest orderable part for `instance`, or a failure reason.

        Numbered instances select a matching part with that part number; if the part
        declares a device kind, the instance value and tolerance also match. Unnumbered
        resistors, capacitors, and inductors match by kind, value, and tolerance.
        """
        if instance.part is not None:
            return _numbered(self, instance, instance.part)
        model = (
            instance.of.model
            if isinstance(instance.of, ir.DeviceOf)
            else instance.of.component
        )
        match model:
            case "resistor" | "capacitor" | "inductor":
                kind: PartKind = model
            case _:
                return (
                    f"a {model} is priced by its part number alone, and it states none"
                )
        found = _valued(instance, kind)
        if isinstance(found, str):
            return found
        value, tolerance = found
        matching = self.matching(kind, value, tolerance)
        part = _cheapest(part for part in matching if part.orderable)
        if part is not None:
            return part
        described = (
            f"{kind} of {format_si(value, SYMBOLS[kind])} within {tolerance:.6g}"
        )
        if matching:
            return f"every {described} in the catalog is obsolete"
        return f"no {described} is in the catalog"

    def _valued(self, kind: PartKind, tolerance: float) -> Iterator[tuple[Part, float]]:
        """Yields each part of `kind` within `tolerance` alongside its nominal value."""
        for part in self.parts:
            if (
                part.kind == kind
                and part.value is not None
                and part.tolerance is not None
                and part.tolerance <= tolerance * (1 + 1e-9)
            ):
                yield part, part.value


class Line(ir.FrozenModel):
    """Priced bill-of-materials entry pairing an instance path with a catalog part."""

    instance: str = Field(description="Hierarchical path of the priced instance.")
    part: Part


class Unpriced(ir.FrozenModel):
    """Unpriced bill-of-materials entry for an instance without a matching part."""

    instance: str = Field(description="Hierarchical path of the unpriced instance.")
    reason: str = Field(
        description="Explanation for why no catalog part matches the instance."
    )


class Bill(ir.FrozenModel):
    """Bill of materials listing priced parts, unpriced instances, and total cost.

    Entries cover every numbered device or component instance together with all
    remaining device instances outside numbered components.
    """

    currency: str
    lines: tuple[Line, ...]
    unpriced: tuple[Unpriced, ...] = Field(
        description="Instances excluded from the total because no catalog part matches."
    )

    @property
    def total(self) -> float:
        """Sum of unit prices across all priced bill-of-materials lines."""
        return math.fsum(line.part.price for line in self.lines)

    def describe(self) -> str:
        """Formats the bill of materials as a plain-text report."""
        head = (
            f"{len(self.lines)} parts for {self.total:.6g} {self.currency}, "
            f"{len(self.unpriced)} instances unpriced"
        )
        rows = [
            f"  {line.instance}: {line.part.describe()}, {line.part.price:.6g} "
            f"{self.currency}"
            for line in self.lines
        ]
        rows.extend(
            f"  {unpriced.instance}: unpriced: {unpriced.reason}"
            for unpriced in self.unpriced
        )
        return "\n".join([head, *rows])


def bill(circuit: ir.Circuit, catalog: Catalog, instance: str = "") -> Bill:
    """Builds a bill of materials for the component at `instance` (top by default).

    Raises:
        KeyError: When no component instance exists at `instance`.
    """
    if not instance:
        component = circuit.top
    else:
        found = circuit.find(instance)
        if found is None or not isinstance(found.of, ir.ComponentOf):
            message = f"no component instance is at {instance!r}"
            raise KeyError(message)
        component = found.of.component
    prefix = f"{instance}/" if instance else ""
    lines: list[Line] = []
    unpriced: list[Unpriced] = []
    for path, placed in built(circuit, component, prefix):
        part = catalog.part_for(placed)
        if isinstance(part, Part):
            lines.append(Line(instance=path, part=part))
        else:
            unpriced.append(Unpriced(instance=path, reason=part))
    return Bill(currency=catalog.currency, lines=tuple(lines), unpriced=tuple(unpriced))


def built(
    circuit: ir.Circuit, component: str, prefix: str = ""
) -> Iterator[tuple[str, ir.Instance]]:
    """Yields each billable instance inside `component` with its path after `prefix`.

    Yielded entries include numbered devices or components plus unnumbered device
    instances outside them, in placement order.
    """
    for path, instance in circuit.walk(
        component, prefix=prefix, descend=lambda placed: placed.part is None
    ):
        if instance.part is not None or isinstance(instance.of, ir.DeviceOf):
            yield path, instance


def _valued(instance: ir.Instance, kind: PartKind) -> tuple[float, float] | str:
    """Extracts the required value and tolerance for `instance`, or a failure reason."""
    parameter = VALUED[kind]
    value = instance.parameters.get(parameter)
    tolerance = instance.tolerances.get(parameter)
    if not isinstance(value, float | int):
        return f"its {parameter} is not a number"
    if tolerance is None:
        return f"its {parameter} has no tolerance, which a part must meet"
    if tolerance.lot is not None:
        return (
            f"its {parameter} comes from the lot {tolerance.lot.name}, whose shared "
            "spread no part of the catalog states"
        )
    return float(value), tolerance.relative


def _numbered(catalog: Catalog, instance: ir.Instance, number: str) -> Part | str:
    """Selects the cheapest orderable part numbered `number` for `instance`, or reason.

    A catalog part that specifies a device kind only matches an instance of that kind,
    matching value, and compatible tolerance when present.
    """
    numbered = [part for part in catalog.parts if part.part_number == number]
    if not numbered:
        return f"no part numbered {number} is in the catalog"
    fitting = [part for part in numbered if _fits(part, instance)]
    if not fitting:
        return (
            f"the catalog's parts numbered {number} are not its device, of its value "
            "within its tolerance"
        )
    part = _cheapest(part for part in fitting if part.orderable)
    if part is None:
        return f"every part numbered {number} in the catalog is obsolete"
    return part


def _fits(part: Part, instance: ir.Instance) -> bool:
    """Returns whether part-number-matched `part` satisfies `instance` constraints."""
    if part.kind is None or part.value is None or part.tolerance is None:
        return True
    if instance.of != ir.DeviceOf(model=part.kind):
        return False
    found = _valued(instance, part.kind)
    if isinstance(found, tuple):
        value, tolerance = found
        return _same(part.value, value) and part.tolerance <= tolerance * (1 + 1e-9)
    value = instance.parameters.get(VALUED[part.kind])
    return isinstance(value, float | int) and _same(part.value, float(value))


def _cheapest(parts: Iterable[Part]) -> Part | None:
    """Returns the lowest-priced part from `parts`, breaking ties by list order."""
    return min(parts, key=lambda part: part.price, default=None)

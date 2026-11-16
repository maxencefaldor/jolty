# Copyright 2026 Maxence Faldor
"""Orderable parts, preferred values, and bills of materials.

A [`Catalog`][jolty.catalog.Catalog] lists parts a circuit can be built from, each at a
unit price: a resistor, a capacitor or an inductor of one value and tolerance, a part
its maker numbers, or both, in a package, at a stage of its lifecycle. A catalog file
holds one as JSON. A [`Bill`][jolty.catalog.Bill] prices the instances of a circuit, or
of one component instance in it. An instance that stands for a numbered part, a device
or a whole component instance, takes the cheapest part of that number; any other device
instance takes the cheapest part of its device, of its value, whose tolerance is within
its own. No instance takes an obsolete part. An instance that states no tolerance, that
comes from a lot, whose shared spread no part states, or that no part matches, stays
unpriced, and the bill says why.

The E series of IEC 60063 space preferred values evenly on a log scale: E24 with 24
values a decade, for tolerances of 5 %, to E192 for 0.5 % and better.

Examples:
    >>> values("E12", 1e3, 2e3)
    (1000.0, 1200.0, 1500.0, 1800.0)
    >>> neighbours("E24", 4.8e3)
    (4700.0, 5100.0)
"""

import bisect
import math
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jolty import _core, ir
from jolty.units import format_si

type PartKind = Literal["resistor", "capacitor", "inductor"]
"""A device a catalog prices."""

type Series = Literal["E3", "E6", "E12", "E24", "E48", "E96", "E192"]
"""An E series of preferred values."""

type Lifecycle = Literal["active", "not-recommended", "obsolete"]
"""Where a part stands in its maker's lifecycle: made, made but not recommended for new
designs, or no longer made."""

VALUED: Final[dict[PartKind, str]] = {
    "resistor": "resistance",
    "capacitor": "capacitance",
    "inductor": "inductance",
}
"""The parameter that holds the value of each device a catalog prices."""

SYMBOLS: Final[dict[PartKind, str]] = {
    "resistor": "Ω",
    "capacitor": "F",
    "inductor": "H",
}
"""The symbol of the unit of each device's value."""


@cache
def mantissas(series: Series) -> tuple[float, ...]:
    """The values of `series` in the decade from 1 to 10.

    E3 to E24 are IEC 60063's historical values; E48 to E192 round 10 to the power of
    each step to three significant digits, but for E192's 9.20.

    Examples:
        >>> mantissas("E96")[:4]
        (1.0, 1.02, 1.05, 1.07)
    """
    return tuple(_core.mantissas(series))


def values(series: Series, low: float, high: float) -> tuple[float, ...]:
    """Every value of `series` from `low` to `high`, both included, increasing.

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


def neighbours(series: Series, value: float) -> tuple[float, float]:
    """The values of `series` nearest `value` at or below it and at or above it.

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
    """`mantissa` times 10 to the power `decade`, rounded to its own digits."""
    return float(f"{mantissa}e{decade}")


def _same(value: float, other: float) -> bool:
    """Whether two values are one, but for rounding."""
    return math.isclose(value, other, rel_tol=1e-9)


def _within(value: float, low: float, high: float) -> bool:
    return (low < value or _same(value, low)) and (value < high or _same(value, high))


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


type Pins = dict[str, tuple[Annotated[str, Field(min_length=1)], ...]]
"""The pads of a part's package each terminal or port of its instance joins, by name,
such as ``{"collector": ("3",), "base": ("2",), "emitter": ("1",)}``; a terminal of no
pads joins none."""


class Part(_Model):
    """A part at a unit price: a device of one value and tolerance, numbered, or both.

    A part that states no device is any device or component its number names. Its
    footprint and pins place it on a board, as a [KiCad netlist][jolty.kicad] writes it.
    """

    kind: PartKind | None = Field(default=None, exclude_if=lambda kind: kind is None)
    value: float | None = Field(
        default=None,
        gt=0,
        exclude_if=lambda value: value is None,
        description="Its value, in SI base units.",
    )
    tolerance: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        exclude_if=lambda tolerance: tolerance is None,
        description="Its relative tolerance, such as 0.01 for 1 %.",
    )
    price: float = Field(ge=0, description="Its unit price, in the catalog's currency.")
    number: ir.PartNumber | None = Field(
        default=None,
        exclude_if=lambda number: number is None,
        description="Its maker's part number.",
    )
    package: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda package: package is None,
        description="Its package, such as 0603 or TO-92.",
    )
    footprint: str | None = Field(
        default=None,
        pattern=r"^[^:]+:[^:]+$",
        exclude_if=lambda footprint: footprint is None,
        description="Its footprint, as KiCad's libraries name it, such as "
        "Resistor_SMD:R_0603_1608Metric.",
    )
    pins: Pins | None = Field(
        default=None,
        exclude_if=lambda pins: pins is None,
        description="The pads each terminal or port of its instance joins, by name.",
    )
    lifecycle: Lifecycle = Field(
        default="active", exclude_if=lambda lifecycle: lifecycle == "active"
    )

    @model_validator(mode="after")
    def _identified(self) -> Self:
        """Holds a part to a device, a value and a tolerance together, or a number."""
        valued = (self.kind, self.value, self.tolerance)
        if any(each is None for each in valued) and any(
            each is not None for each in valued
        ):
            message = "a part states its kind, value and tolerance together"
            raise ValueError(message)
        if self.kind is None and self.number is None:
            message = "a part states a kind, value and tolerance, or a number"
            raise ValueError(message)
        for terminal, pads in (self.pins or {}).items():
            if len(set(pads)) < len(pads):
                message = f"a part's pins give {terminal} a pad twice"
                raise ValueError(message)
        return self

    @property
    def orderable(self) -> bool:
        """Whether its maker still makes it."""
        return self.lifecycle != "obsolete"

    def describe(self) -> str:
        """The part as text, its price aside."""
        said = []
        if self.kind is not None and self.value is not None:
            said.append(
                f"{self.kind} of {format_si(self.value, SYMBOLS[self.kind])} "
                f"within {self.tolerance:.6g}"
            )
        if self.number is not None:
            said.append(self.number)
        if self.package is not None:
            said.append(f"in {self.package}")
        if self.lifecycle == "not-recommended":
            said.append("not recommended for new designs")
        return ", ".join(said)


class Catalog(_Model):
    """The parts a circuit can be built from, priced in one currency."""

    format: Literal["jolty-catalog"] = "jolty-catalog"
    version: Literal[2] = 2
    currency: str = Field(description="The currency of every price, such as USD.")
    parts: tuple[Part, ...]

    @classmethod
    def load(cls, path: Path) -> Catalog:
        """The catalog in the file at `path`."""
        return cls.model_validate_json(path.read_bytes())

    def save(self, path: Path) -> None:
        """Writes the catalog to `path` as JSON."""
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def cheapest(self, kind: PartKind, value: float, tolerance: float) -> Part | None:
        """The cheapest orderable part of `kind` of `value`, within `tolerance`.

        Of parts at one price, the first listed. None when no part matches.
        """
        return _cheapest(
            part for part in self.matching(kind, value, tolerance) if part.orderable
        )

    def matching(
        self, kind: PartKind, value: float, tolerance: float
    ) -> tuple[Part, ...]:
        """Every part of `kind` of `value` within `tolerance`, orderable or not."""
        return tuple(
            part for part, held in self._valued(kind, tolerance) if _same(held, value)
        )

    def values(self, kind: PartKind, tolerance: float) -> tuple[float, ...]:
        """Each value of `kind` an orderable part within `tolerance` has, in order."""
        found = sorted(
            held for part, held in self._valued(kind, tolerance) if part.orderable
        )
        distinct: list[float] = []
        for value in found:
            if not distinct or not _same(distinct[-1], value):
                distinct.append(value)
        return tuple(distinct)

    def part_for(self, instance: ir.Instance) -> Part | str:
        """The cheapest orderable part that builds `instance`, or why none does.

        A numbered instance takes a part of its number, which a part that states a
        device builds only as that device would, of its value and within the
        instance's tolerance if it states one. A resistor, a capacitor or an inductor
        takes a part of its value within its tolerance. Any other instance takes none.
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
        """Each part of `kind` within `tolerance`, orderable or not, and its value."""
        for part in self.parts:
            if (
                part.kind == kind
                and part.value is not None
                and part.tolerance is not None
                and part.tolerance <= tolerance * (1 + 1e-9)
            ):
                yield part, part.value


class Line(_Model):
    """An instance, and the part that builds it."""

    instance: str = Field(description="The instance's path.")
    part: Part


class Unpriced(_Model):
    """An instance that no part of the catalog builds."""

    instance: str = Field(description="The instance's path.")
    reason: str = Field(description="Why no part builds it.")


class Bill(_Model):
    """The parts that build the instances of a circuit, and what they cost.

    Its instances are the numbered ones, devices or component instances, and the other
    device instances outside them.
    """

    currency: str
    lines: tuple[Line, ...]
    unpriced: tuple[Unpriced, ...] = Field(
        description="Instances no part builds, which the total leaves out."
    )

    @property
    def total(self) -> float:
        """The sum of the parts' prices."""
        return math.fsum(line.part.price for line in self.lines)

    def describe(self) -> str:
        """The bill as text."""
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
    """The bill of materials of the component at `instance`, the top by default.

    Raises:
        KeyError: When no component instance is at `instance`.
    """
    component = circuit.top
    for id in filter(None, instance.split("/")):
        placed = next(
            (
                each
                for each in circuit.components[component].instances
                if each.name == id
            ),
            None,
        )
        if placed is None or not isinstance(placed.of, ir.ComponentOf):
            message = f"no component instance is at {instance!r}"
            raise KeyError(message)
        component = placed.of.component
    prefix = f"{instance}/" if instance else ""
    lines: list[Line] = []
    unpriced: list[Unpriced] = []
    for path, placed in built(circuit, component, prefix):
        found = catalog.part_for(placed)
        if isinstance(found, Part):
            lines.append(Line(instance=path, part=found))
        else:
            unpriced.append(Unpriced(instance=path, reason=found))
    return Bill(currency=catalog.currency, lines=tuple(lines), unpriced=tuple(unpriced))


def built(
    circuit: ir.Circuit, component: str, prefix: str = ""
) -> Iterator[tuple[str, ir.Instance]]:
    """Each instance within `component` a part builds, by its path after `prefix`.

    They are the numbered instances, devices or component instances, and the device
    instances outside them, in the order the components place them.
    """
    for instance in circuit.components[component].instances:
        path = f"{prefix}{instance.name}"
        match instance:
            case ir.Instance(part=None, of=ir.ComponentOf(component=inner)):
                yield from built(circuit, inner, f"{path}/")
            case _:
                yield path, instance


def _valued(instance: ir.Instance, kind: PartKind) -> tuple[float, float] | str:
    """The value and the tolerance a part of `instance` must have, or why none has."""
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
    """The cheapest part numbered `number` that builds `instance`, or why none does.

    A part that states a device builds only an instance of that device, of its value,
    and within the instance's tolerance if it states one.
    """
    numbered = [part for part in catalog.parts if part.number == number]
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
    """Whether `part`, numbered as `instance` is, builds it as a device would."""
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
    """The cheapest of `parts`, the first listed of those at one price, if any."""
    return min(parts, key=lambda part: part.price, default=None)

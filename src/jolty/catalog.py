# Copyright 2026 Maxence Faldor
"""Orderable parts, preferred values, and bills of materials.

A [`Catalog`][jolty.catalog.Catalog] lists parts a circuit can be built from, each a
resistor, a capacitor or an inductor of one value and tolerance at a unit price. A
catalog file holds one as JSON. A [`Bill`][jolty.catalog.Bill] prices the device
instances of a circuit, or of one component instance in it: each instance takes the
cheapest part of its device, of its value, whose tolerance is within its own. An
instance that states no tolerance, that comes from a lot, whose shared spread no part
states, or that no part matches, stays unpriced, and the bill says why.

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
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from jolty import _core, ir
from jolty.units import format_si

type Device = Literal["resistor", "capacitor", "inductor"]
"""A device a catalog prices."""

type Series = Literal["E3", "E6", "E12", "E24", "E48", "E96", "E192"]
"""An E series of preferred values."""

VALUED: Final[dict[Device, str]] = {
    "resistor": "resistance",
    "capacitor": "capacitance",
    "inductor": "inductance",
}
"""The parameter that holds the value of each device a catalog prices."""

_SYMBOLS: Final[dict[Device, str]] = {
    "resistor": "Ω",
    "capacitor": "F",
    "inductor": "H",
}


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


class Part(_Model):
    """An orderable part: a device of one value and tolerance, at a unit price."""

    device: Device
    value: float = Field(gt=0, description="Its value, in SI base units.")
    tolerance: float = Field(
        gt=0, lt=1, description="Its relative tolerance, such as 0.01 for 1 %."
    )
    price: float = Field(ge=0, description="Its unit price, in the catalog's currency.")
    number: str | None = Field(
        default=None, description="Its manufacturer's part number."
    )


class Catalog(_Model):
    """The parts a circuit can be built from, priced in one currency."""

    format: Literal["jolty-catalog"] = "jolty-catalog"
    version: Literal[1] = 1
    currency: str = Field(description="The currency of every price, such as USD.")
    parts: tuple[Part, ...]

    @classmethod
    def load(cls, path: Path) -> Catalog:
        """The catalog in the file at `path`."""
        return cls.model_validate_json(path.read_bytes())

    def save(self, path: Path) -> None:
        """Writes the catalog to `path` as JSON."""
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def cheapest(self, device: Device, value: float, tolerance: float) -> Part | None:
        """The cheapest part of `device` of `value`, its tolerance within `tolerance`.

        Of parts at one price, the first listed. None when no part matches.
        """
        matching = [
            part
            for part in self.parts
            if part.device == device
            and _same(part.value, value)
            and part.tolerance <= tolerance * (1 + 1e-9)
        ]
        return min(matching, key=lambda part: part.price, default=None)

    def values(self, device: Device, tolerance: float) -> tuple[float, ...]:
        """Every value of `device` that a part within `tolerance` has, increasing."""
        found = sorted(
            part.value
            for part in self.parts
            if part.device == device and part.tolerance <= tolerance * (1 + 1e-9)
        )
        distinct: list[float] = []
        for value in found:
            if not distinct or not _same(distinct[-1], value):
                distinct.append(value)
        return tuple(distinct)


class Line(_Model):
    """A device instance, and the part that builds it."""

    instance: str = Field(description="The instance's path.")
    part: Part


class Unpriced(_Model):
    """A device instance that no part of the catalog builds."""

    instance: str = Field(description="The instance's path.")
    reason: str = Field(description="Why no part builds it.")


class Bill(_Model):
    """The parts that build the device instances of a circuit, and what they cost."""

    currency: str
    lines: tuple[Line, ...]
    unpriced: tuple[Unpriced, ...] = Field(
        description="Device instances no part builds, which the total leaves out."
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
            f"  {line.instance}: {line.part.device} of "
            f"{format_si(line.part.value, _SYMBOLS[line.part.device])} within "
            f"{line.part.tolerance:.6g}, {line.part.price:.6g} {self.currency}"
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
            (each for each in circuit.components[component].instances if each.id == id),
            None,
        )
        if placed is None or not isinstance(placed.of, ir.ComponentOf):
            message = f"no component instance is at {instance!r}"
            raise KeyError(message)
        component = placed.of.component
    prefix = f"{instance}/" if instance else ""
    lines: list[Line] = []
    unpriced: list[Unpriced] = []
    for path, device, placed in _devices(circuit, component, prefix):
        found = _price(catalog, device, placed)
        if isinstance(found, Part):
            lines.append(Line(instance=path, part=found))
        else:
            unpriced.append(Unpriced(instance=path, reason=found))
    return Bill(currency=catalog.currency, lines=tuple(lines), unpriced=tuple(unpriced))


def _devices(
    circuit: ir.Circuit, component: str, prefix: str
) -> Iterable[tuple[str, str, ir.Instance]]:
    """Every device instance within `component`: its path, its device and itself."""
    for instance in circuit.components[component].instances:
        path = f"{prefix}{instance.id}"
        match instance.of:
            case ir.ComponentOf(component=inner):
                yield from _devices(circuit, inner, f"{path}/")
            case ir.DeviceOf(device=device):
                yield path, device, instance


def _price(catalog: Catalog, device: str, instance: ir.Instance) -> Part | str:
    """The cheapest part that builds the instance of `device`, or why none does."""
    match device:
        case "resistor" | "capacitor" | "inductor":
            kind: Device = device
        case _:
            return (
                f"the catalog prices resistors, capacitors and inductors, not {device}s"
            )
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
    part = catalog.cheapest(kind, float(value), tolerance.relative)
    if part is None:
        return (
            f"no {device} of {format_si(float(value), _SYMBOLS[kind])} within "
            f"{tolerance.relative:.6g} is in the catalog"
        )
    return part

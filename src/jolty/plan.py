# Copyright 2026 Maxence Faldor
"""Simulation plans: the analyses to run on a circuit, how, and in which conditions.

A plan is kept apart from the circuit: the circuit says what exists, the plan how to
simulate it. A plan runs its analyses at its temperature, then again at each of its
corners: another temperature, or other arguments to the build script's ``build()``, and
on Monte Carlo samples of the circuit's tolerances, and on the runs of its variants,
each at its own conditions and at every corner. Variants are the values of many runs of
one circuit, each a pure function of the run's index: listed
([`Explicit`][jolty.plan.Explicit]), on a [`Grid`][jolty.plan.Grid] of parameters and
temperatures, drawn as [`MonteCarlo`][jolty.plan.MonteCarlo] samples, or every one of
one at every one of another ([`Product`][jolty.plan.Product]). Its fidelity chooses, for
the component instances its scopes match, logic models characterized from their
circuits.

Every plan object validates through Pydantic, reporting every invalid field and entry
at once as a [`ValidationError`][pydantic.ValidationError], as circuit and results
files do.

Examples:
    >>> from jolty.units import us
    >>> plan = Plan({"step": Transient(stop=10 * us)})
"""

import decimal
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass as stdlib_dataclass
from dataclasses import field, fields
from decimal import Decimal
from types import MappingProxyType
from typing import Final, Literal, LiteralString, Never

from pydantic import (
    ConfigDict,
    JsonValue,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass, rebuild_dataclass
from pydantic_core import InitErrorDetails, PydanticCustomError

from jolty import _core, ir
from jolty.errors import DimensionError
from jolty.spice import PeriodicSteadyStateCard, ngspice
from jolty.units import (
    FREQUENCY,
    TEMPERATURE,
    TIME,
    VOLTAGE,
    Dimension,
    Quantity,
    s,
)

DECADE_DIGITS: Final = 40
"""The decimal digits a decade sweep's frequencies are computed to before they are
rounded to floats: so many more than a float's 17 that a million steps' rounding stays
far below a float's, and each frequency rounds to the float nearest its exact value."""

_SEED_LIMIT: Final = 2**64
"""The exclusive upper bound of a 64-bit unsigned seed."""

_CONFIG: Final = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
"""Shared Pydantic configuration for every plan dataclass."""

type Method = Literal["trapezoidal", "gear2", "backward_euler"]
"""An integration method: trapezoidal by default, Gear2 for stiff circuits that ring,
backward Euler for the most damping."""


def _fail(kind: LiteralString, message: str) -> Never:
    """Raises a Pydantic custom error holding literal `message` safe from braces."""
    raise PydanticCustomError(kind, "{message}", {"message": message})


def _detail(
    kind: LiteralString,
    message: str,
    value: object,
    location: tuple[int | str, ...] = (),
) -> InitErrorDetails:
    """One structured validation error entry at `location` with literal `message`."""
    return InitErrorDetails(
        type=PydanticCustomError(kind, "{message}", {"message": message}),
        loc=location,
        input=value,
    )


def _raise_if_errors(title: str, found: Sequence[InitErrorDetails]) -> None:
    """Raises `ValidationError` holding every entry of `found`, if any."""
    if found:
        raise ValidationError.from_exception_data(title, list(found))


def _check_integer(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    minimum_message: str | None = None,
) -> int:
    """Validates `value` is a strict integer (not bool) within `[minimum, maximum)`."""
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("int_type", f"{name} is an integer, not {value!r}")
    if minimum is not None and value < minimum:
        message = (
            f"{name} is {minimum} or more, not {value}"
            if minimum_message is None
            else minimum_message
        )
        _fail("int_range", message)
    if maximum is not None and value >= maximum:
        message = f"{name} lies from {minimum or 0} to {maximum - 1}, not {value}"
        _fail("int_range", message)
    return value


def _check_float(
    value: object,
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    """Validates `value` is a strict real number (not bool), finite and in range."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail("float_type", f"{name} is a real number, not {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        _fail("finite", f"{name} is finite, not {converted!r}")
    if positive and converted <= 0.0:
        _fail("positive", f"{name} is positive, not {converted!r}")
    if non_negative and converted < 0.0:
        _fail("non_negative", f"{name} is 0 or more, not {converted!r}")
    return converted


def _check_boolean(value: object, name: str) -> bool:
    """Validates `value` is a strict boolean."""
    if not isinstance(value, bool):
        _fail("bool_type", f"{name} is a boolean, not {value!r}")
    return value


def _check_string(value: object, name: str, *, non_empty: bool = False) -> str:
    """Validates `value` is a string, and non-empty when `non_empty`."""
    if not isinstance(value, str):
        _fail("string_type", f"{name} is a string, not {value!r}")
    if non_empty and not value:
        _fail("empty", f"{name} is non-empty, not ''")
    return value


def _check_method(value: object) -> Method:
    """Validates `value` is one of the supported integration methods."""
    match value:
        case "trapezoidal":
            return "trapezoidal"
        case "gear2":
            return "gear2"
        case "backward_euler":
            return "backward_euler"
        case _:
            message = (
                "an integration method is 'trapezoidal', 'gear2' or 'backward_euler', "
                f"not {value!r}"
            )
            _fail("method", message)


def _check_quantity(
    value: object,
    dimension: Dimension,
    what: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> Quantity:
    """Validates `value` is a `Quantity` of `dimension`, finite and in range.

    Raises:
        DimensionError: When `value` is a `Quantity` with another dimension.
        PydanticCustomError: When `value` is not a `Quantity`, non-finite, or outside
            the requested sign range.
    """
    if not isinstance(value, Quantity):
        _fail("quantity_type", f"{what} is a Quantity, not {value!r}")
    if value.dimension != dimension:
        message = f"{what} takes a quantity in {dimension}, not {value}"
        raise DimensionError(message)
    if not math.isfinite(value.value):
        _fail("finite", f"{what} is finite, not {value}")
    if positive and value.value <= 0.0:
        _fail("positive", f"{what} is positive, not {value}")
    if non_negative and value.value < 0.0:
        _fail("non_negative", f"{what} is at least 0 {dimension}, not {value}")
    return value


def _check_voltage_map(value: object, title: str, what: str) -> dict[str, Quantity]:
    """Validates a mapping of net names to finite voltage quantities."""
    if not isinstance(value, Mapping):
        _fail("mapping_type", f"{title} is a mapping of net names to voltages")
    errors: list[InitErrorDetails] = []
    checked: dict[str, Quantity] = {}
    for net, level in value.items():
        if not isinstance(net, str):
            errors.append(
                _detail(
                    "string_type",
                    f"a net name is a string, not {net!r}",
                    net,
                    (repr(net),),
                )
            )
            continue
        if not isinstance(level, Quantity):
            errors.append(
                _detail(
                    "quantity_type",
                    f"{what} for {net!r} is a Quantity, not {level!r}",
                    level,
                    (net,),
                )
            )
            continue
        if level.dimension != VOLTAGE:
            message = f"{what} takes a quantity in {VOLTAGE}, not {level}"
            raise DimensionError(message)
        if not math.isfinite(level.value):
            errors.append(
                _detail(
                    "finite",
                    f"{what} of {net!r} is finite, not {level}",
                    level,
                    (net,),
                )
            )
            continue
        checked[net] = level
    _raise_if_errors(title, errors)
    return checked


def _check_frequencies(
    value: object,
    title: str,
    *,
    strictly_positive: bool,
    empty_message: str,
) -> tuple[Quantity, ...]:
    """Validates a non-empty sequence of finite frequency quantities."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        _fail(
            "sequence_type",
            f"frequencies is a sequence of quantities, not {value!r}",
        )
    items = tuple(value)
    if not items:
        _fail("empty", empty_message)
    errors: list[InitErrorDetails] = []
    checked: list[Quantity] = []
    for index, frequency in enumerate(items):
        if not isinstance(frequency, Quantity):
            errors.append(
                _detail(
                    "quantity_type",
                    f"a frequency is a Quantity, not {frequency!r}",
                    frequency,
                    (index,),
                )
            )
            continue
        if frequency.dimension != FREQUENCY:
            message = f"a frequency takes a quantity in {FREQUENCY}, not {frequency}"
            raise DimensionError(message)
        if not math.isfinite(frequency.value):
            errors.append(
                _detail(
                    "finite",
                    f"frequencies are finite, not {frequency}",
                    frequency,
                    (index,),
                )
            )
        elif strictly_positive and frequency.value <= 0.0:
            errors.append(
                _detail(
                    "positive",
                    f"noise frequencies are positive, not {frequency}",
                    frequency,
                    (index,),
                )
            )
        elif not strictly_positive and frequency.value < 0.0:
            errors.append(
                _detail(
                    "non_negative",
                    f"frequencies are at least 0 Hz, not {frequency}",
                    frequency,
                    (index,),
                )
            )
        else:
            checked.append(frequency)
    _raise_if_errors(title, errors)
    return tuple(checked)


def _check_breakpoints(
    value: object,
    title: str,
    *,
    upper: Quantity | None,
    upper_name: str,
    require_period: bool = False,
) -> tuple[Quantity, ...]:
    """Validates breakpoint times against `(0, upper]` when `upper` is known."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        _fail(
            "sequence_type",
            f"breakpoints is a sequence of times, not {value!r}",
        )
    items = tuple(value)
    if require_period and items:
        message = (
            "an oscillator's period is found, not fixed, so its periodic steady "
            "state has no times to step onto"
        )
        _fail("oscillator_breakpoints", message)
    errors: list[InitErrorDetails] = []
    checked: list[Quantity] = []
    limit = upper.value if upper is not None and upper.value > 0.0 else None
    for index, time in enumerate(items):
        if not isinstance(time, Quantity):
            errors.append(
                _detail(
                    "quantity_type",
                    f"a time to step onto is a Quantity, not {time!r}",
                    time,
                    (index,),
                )
            )
            continue
        if time.dimension != TIME:
            message = f"a time to step onto takes a quantity in {TIME}, not {time}"
            raise DimensionError(message)
        if not math.isfinite(time.value):
            errors.append(
                _detail(
                    "finite",
                    f"a time to step onto is finite, not {time}",
                    time,
                    (index,),
                )
            )
        elif not (time.value > 0.0 and (limit is None or time.value <= limit)):
            errors.append(
                _detail(
                    "breakpoint_range",
                    f"the times to step onto lie after 0 and at most {upper_name}, "
                    f"not {time}",
                    time,
                    (index,),
                )
            )
        else:
            checked.append(time)
    _raise_if_errors(title, errors)
    return tuple(checked)


def _check_string_sequence(
    value: object, title: str, item_noun: str
) -> tuple[str, ...]:
    """Validates a sequence of strings, rejecting a bare string."""
    if isinstance(value, str):
        message = (
            f"{title} is a sequence of {item_noun}, such as [{value!r}], "
            f"not the string {value!r}"
        )
        _fail("sequence_type", message)
    if isinstance(value, bytes) or not isinstance(value, Sequence):
        message = f"{title} is a sequence of {item_noun}, not {value!r}"
        _fail("sequence_type", message)
    items = tuple(value)
    errors: list[InitErrorDetails] = []
    checked: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str):
            errors.append(
                _detail(
                    "string_type",
                    f"{title} is a sequence of {item_noun}, not {item!r}",
                    item,
                    (index,),
                )
            )
        else:
            checked.append(item)
    _raise_if_errors(title, errors)
    return tuple(checked)


def _check_differentiated(value: object) -> Differentiated:
    """Validates `value` is `None` or a differentiable analysis."""
    if value is None or isinstance(
        value, DcSweep | Ac | Transient | PeriodicSteadyState
    ):
        return value
    message = (
        "a differentiated analysis is a DcSweep, Ac, Transient, "
        f"PeriodicSteadyState or None, not {value!r}"
    )
    _fail("analysis_type", message)


def _check_variants(value: object, subject: str) -> Variants:
    """Validates `value` is a `Variants` instance."""
    if isinstance(value, MonteCarlo | Grid | Explicit | Product):
        return value
    _fail("variants_type", f"{subject}, not {value!r}")


def _check_rising_grid(
    value: object, name: str, *, strictly_positive: bool
) -> tuple[float, ...]:
    """Validates a strictly increasing sequence of finite floats."""
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        _fail("sequence_type", f"the {name} is a sequence of numbers, not {value!r}")
    above = "above zero" if strictly_positive else "from zero or more"
    raw = tuple(value)
    numbers: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float):
            _fail("float_type", f"the {name} holds real numbers, not {item!r}")
        converted = float(item)
        if not math.isfinite(converted):
            _fail("finite", f"the {name} are finite, not {raw}")
        numbers.append(converted)
    values = tuple(numbers)
    rising = all(a < b for a, b in itertools.pairwise(values))
    first = values[0] if values else math.nan
    starts = first > 0.0 if strictly_positive else first >= 0.0
    if not values or not rising or not starts:
        _fail("grid_order", f"the {name} rise {above}, not {values}")
    return values


@stdlib_dataclass(frozen=True, slots=True)
class _IdentifierRule:
    """Describes the messages for an identifier-keyed mapping field."""

    title: str
    kind: str
    type_message: str
    non_empty_message: str | None = None


def _check_identifier_mapping[T](
    value: object,
    expected_type: type[T] | tuple[type[object], ...],
    rule: _IdentifierRule,
) -> dict[str, T]:
    """Validates a mapping keyed by Python identifiers to instances of `expected_type`.

    Reports every invalid key and every wrong-typed value at once.
    """
    if not isinstance(value, Mapping):
        _fail("mapping_type", f"{rule.title} is a mapping, not {value!r}")
    if rule.non_empty_message is not None and not value:
        _fail("empty", rule.non_empty_message)
    errors: list[InitErrorDetails] = []
    checked: dict[str, T] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.isidentifier():
            errors.append(
                _detail(
                    "identifier",
                    f"{rule.kind} is named by a Python identifier, not {key!r}",
                    key,
                    (str(key),),
                )
            )
        if not isinstance(item, expected_type):
            errors.append(
                _detail(
                    "value_type",
                    rule.type_message.format(key=key, item=repr(item)),
                    item,
                    (str(key),),
                )
            )
        elif isinstance(key, str) and key.isidentifier():
            checked[key] = item
    _raise_if_errors(rule.title, errors)
    return checked


def voltages(given: Mapping[str, Quantity], what: str) -> dict[str, float]:
    """The voltage of each net `given` names, in volts, each checked as `what`.

    Raises:
        DimensionError: When one is not a voltage.
    """
    return {net: voltage.to(VOLTAGE, what) for net, voltage in given.items()}


def _raise_one(title: str, detail: InitErrorDetails) -> Never:
    """Raises `ValidationError` holding a single `detail`."""
    raise ValidationError.from_exception_data(title, [detail])


def _check_record(record: object) -> tuple[str, ...] | None:
    """Validates `record` as `None` or a sequence of string patterns."""
    if record is None:
        return None
    if isinstance(record, str):
        message = (
            "a record is a sequence of instance paths, such as "
            f"[{record!r}], not the string {record!r}"
        )
        _raise_one("record", _detail("record_type", message, record))
    if isinstance(record, bytes) or not isinstance(record, Sequence):
        message = f"a record is a sequence of instance paths, not {record!r}"
        _raise_one("record", _detail("record_type", message, record))
    patterns = tuple(record)
    errors: list[InitErrorDetails] = []
    checked: list[str] = []
    for index, pattern in enumerate(patterns):
        if not isinstance(pattern, str):
            errors.append(
                _detail(
                    "string_type",
                    f"a record is a sequence of instance paths, not {patterns!r}",
                    pattern,
                    (index,),
                )
            )
        else:
            checked.append(pattern)
    _raise_if_errors("record", errors)
    return tuple(checked)


def recorded(record: Sequence[str] | None) -> tuple[str, ...] | None:
    """`record` as a tuple of patterns, or None when it is left out.

    Raises:
        ValidationError: When `record` is a string, or holds anything but strings.
    """
    return _check_record(record)


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Recorded:
    """An analysis whose results can keep the devices' detail beside the unknowns.

    Its `record` names the instances whose detail its results keep, as a plan's does,
    and replaces the plan's when set (see [`Plan`][jolty.plan.Plan]). With `regions`,
    its results keep besides each other transistor's region of operation and its
    margins to the regions' boundaries alone, which an operating point's `region` reads.
    """

    record: Sequence[str] | None = field(default=None, kw_only=True)
    regions: bool = field(default=False, kw_only=True)

    @field_validator("record", mode="before")
    @classmethod
    def _validate_record(cls, value: object) -> tuple[str, ...] | None:
        return _check_record(value)

    @field_validator("regions", mode="before")
    @classmethod
    def _validate_regions(cls, value: object) -> bool:
        return _check_boolean(value, "regions")


@dataclass(frozen=True, slots=True, config=_CONFIG)
class OperatingPoint(Recorded):
    """The DC operating point, with every source at its level at time zero."""

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        return "operating_point"


@dataclass(frozen=True, slots=True, config=_CONFIG)
class OperatingPoints(Recorded):
    """The circuit's DC solutions, up to `max_solutions`, the operating point first.

    Once the operating point is found, the circuit's equations deflated by the
    solutions found so far lead Newton iteration to another, until none does, so a
    latch has a solution per state and one balanced between them, and a self-biased
    reference its zero-current state beside its working one.
    """

    max_solutions: int = 8

    @field_validator("max_solutions", mode="before")
    @classmethod
    def _validate_max_solutions(cls, value: object) -> int:
        return _check_integer(
            value,
            "max_solutions",
            minimum=1,
            minimum_message=(
                f"a search for operating points finds 1 or more, not {value}"
            ),
        )

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        return {"operating_points": {"max_solutions": self.max_solutions}}


@dataclass(frozen=True, slots=True, config=_CONFIG)
class DcSweep(Recorded):
    """The operating point at each level of one independent source, or of a parameter.

    The source is named by its path, such as ``V1`` or ``bias/I1``. With a
    `parameter`, named by its path among the instance's parameters, such as
    ``resistance``, the sweep sets that number of any device to each level instead, the
    device made again from its parameters with the number replaced.
    """

    source: str
    values: Sequence[Quantity]
    parameter: str | None = None

    @field_validator("source", mode="before")
    @classmethod
    def _validate_source(cls, value: object) -> str:
        return _check_string(value, "source", non_empty=True)

    @field_validator("values", mode="before")
    @classmethod
    def _validate_values(cls, value: object) -> tuple[Quantity, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            _fail(
                "sequence_type",
                f"a DC sweep's values is a sequence of quantities, not {value!r}",
            )
        items = tuple(value)
        if not items:
            _fail("empty", "a DC sweep has at least one level")
        errors: list[InitErrorDetails] = []
        checked: list[Quantity] = []
        first_dimension: Dimension | None = None
        for index, level in enumerate(items):
            if not isinstance(level, Quantity):
                errors.append(
                    _detail(
                        "quantity_type",
                        f"a sweep level is a Quantity, not {level!r}",
                        level,
                        (index,),
                    )
                )
                continue
            if first_dimension is None:
                first_dimension = level.dimension
            elif level.dimension != first_dimension:
                message = (
                    f"a sweep level takes a quantity in {first_dimension}, not {level}"
                )
                raise DimensionError(message)
            if not math.isfinite(level.value):
                errors.append(
                    _detail(
                        "finite",
                        f"a DC sweep's levels are finite, not {level}",
                        level,
                        (index,),
                    )
                )
            else:
                checked.append(level)
        _raise_if_errors("values", errors)
        return tuple(checked)

    @field_validator("parameter", mode="before")
    @classmethod
    def _validate_parameter(cls, value: object) -> str | None:
        if value is None:
            return None
        return _check_string(value, "parameter", non_empty=True)

    def to_json(self, dimension: Dimension) -> JsonValue:
        """The analysis in the engine's format, checked against `dimension`."""
        values: list[JsonValue] = [
            value.to(dimension, "a sweep level") for value in self.values
        ]
        sweep: dict[str, JsonValue] = {"source": self.source, "values": values}
        if self.parameter is not None:
            sweep["parameter"] = self.parameter
        return {"dc_sweep": sweep}


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Ac(Recorded):
    """The small-signal response to a unit excitation of a source, at each frequency."""

    source: str
    frequencies: Sequence[Quantity]

    @field_validator("source", mode="before")
    @classmethod
    def _validate_source(cls, value: object) -> str:
        return _check_string(value, "source", non_empty=True)

    @field_validator("frequencies", mode="before")
    @classmethod
    def _validate_frequencies(cls, value: object) -> tuple[Quantity, ...]:
        return _check_frequencies(
            value,
            "frequencies",
            strictly_positive=False,
            empty_message="a small-signal analysis has at least one frequency",
        )

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        frequencies: list[JsonValue] = [
            frequency.to(FREQUENCY, "a frequency") for frequency in self.frequencies
        ]
        return {"ac": {"source": self.source, "frequencies": frequencies}}


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Transient(Recorded):
    """The response over time, from the operating point at zero until `stop`.

    The analysis steps onto each of `breakpoints` exactly, besides the times it chooses.
    Its `initial_conditions` hold nets, by path such as ``out`` or ``amp/mid``, at the
    voltage they start from, as SPICE's ``.ic``: each is held there while the operating
    point is solved, then left free. A net that voltage sources and inductors join to
    the ground, or to a net named before it, is left to them, and its voltage there must
    be the one stated. Without its `operating_point`, as SPICE's ``UIC``, the transient
    starts from its initial conditions alone, every other unknown at zero, so a circuit
    starts from rest or from a state no DC solution has, such as an oscillator's kick.

    Examples:
        >>> from jolty.units import V, us
        >>> started = Transient(10 * us, initial_conditions={"out": 1 * V})
        >>> started.to_json()["transient"]["initial_conditions"]
        {'out': 1.0}
    """

    stop: Quantity
    max_step: Quantity | None = None
    method: Method = "trapezoidal"
    breakpoints: Sequence[Quantity] = ()
    initial_conditions: Mapping[str, Quantity] = field(default_factory=dict)
    operating_point: bool = True

    @field_validator("stop", mode="before")
    @classmethod
    def _validate_stop(cls, value: object) -> Quantity:
        return _check_quantity(value, TIME, "a transient's stop time", positive=True)

    @field_validator("max_step", mode="before")
    @classmethod
    def _validate_max_step(cls, value: object) -> Quantity | None:
        if value is None:
            return None
        return _check_quantity(value, TIME, "a transient's longest step", positive=True)

    @field_validator("method", mode="before")
    @classmethod
    def _validate_method(cls, value: object) -> Method:
        return _check_method(value)

    @field_validator("breakpoints", mode="before")
    @classmethod
    def _validate_breakpoints(
        cls, value: object, info: ValidationInfo
    ) -> tuple[Quantity, ...]:
        stop = info.data.get("stop")
        return _check_breakpoints(
            value,
            "breakpoints",
            upper=stop if isinstance(stop, Quantity) else None,
            upper_name="the stop time",
        )

    @field_validator("initial_conditions", mode="before")
    @classmethod
    def _validate_initial_conditions(cls, value: object) -> dict[str, Quantity]:
        return _check_voltage_map(value, "initial_conditions", "an initial condition")

    @field_validator("operating_point", mode="before")
    @classmethod
    def _validate_operating_point(cls, value: object) -> bool:
        return _check_boolean(value, "operating_point")

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        transient: dict[str, JsonValue] = {
            "stop": self.stop.to(TIME, "a transient's stop time"),
            "method": self.method,
        }
        if self.breakpoints:
            transient["breakpoints"] = [
                time.to(TIME, "a time to step onto") for time in self.breakpoints
            ]
        if self.max_step is not None:
            transient["max_step"] = self.max_step.to(TIME, "a transient's longest step")
        if self.initial_conditions:
            transient["initial_conditions"] = dict[str, JsonValue](
                voltages(self.initial_conditions, "an initial condition")
            )
        if not self.operating_point:
            transient["operating_point"] = False
        return {"transient": transient}


@dataclass(frozen=True, slots=True, config=_CONFIG)
class PeriodicSteadyState(Recorded):
    """One period of the steady state a circuit repeats, found by shooting.

    With a `period`, that of every source, shooting starts from the operating point;
    without one, the circuit is an oscillator whose sources hold steady, and shooting
    finds its period too, starting where a transient from its operating point, kicked
    along its fastest-growing natural mode, settles onto its orbit. The period's steps
    are at most `max_step`, by default a fiftieth of the period, and land on each of
    `breakpoints`, times within the period.

    Examples:
        >>> from jolty.units import us
        >>> PeriodicSteadyState(period=1 * us).to_json()
        {'periodic_steady_state': {'method': 'trapezoidal', 'period': 1e-06}}
        >>> PeriodicSteadyState().to_json()
        {'periodic_steady_state': {'method': 'trapezoidal'}}
    """

    period: Quantity | None = None
    max_step: Quantity | None = None
    method: Method = "trapezoidal"
    breakpoints: Sequence[Quantity] = ()

    @classmethod
    def from_spice(cls, card: PeriodicSteadyStateCard) -> PeriodicSteadyState:
        """Converts a SPICE ``.pss`` card into a [`PeriodicSteadyState`] analysis.

        An autonomous card leaves `period` as ``None`` so shooting solves for the
        oscillator's period; a driven card sets `period` in seconds. When `max_step`
        is omitted on a driven card with `points` set, `max_step` is
        ``period / points``.
        """
        period = None if card.autonomous or card.period is None else card.period * s
        if card.max_step is not None:
            max_step = card.max_step * s
        elif (
            not card.autonomous and card.period is not None and card.points is not None
        ):
            max_step = (card.period / card.points) * s
        else:
            max_step = None
        return cls(
            period=period,
            max_step=max_step,
            method=card.method or "trapezoidal",
        )

    @field_validator("period", mode="before")
    @classmethod
    def _validate_period(cls, value: object) -> Quantity | None:
        if value is None:
            return None
        return _check_quantity(
            value, TIME, "a periodic steady state's period", positive=True
        )

    @field_validator("max_step", mode="before")
    @classmethod
    def _validate_max_step(cls, value: object) -> Quantity | None:
        if value is None:
            return None
        return _check_quantity(value, TIME, "a period's longest step", positive=True)

    @field_validator("method", mode="before")
    @classmethod
    def _validate_method(cls, value: object) -> Method:
        return _check_method(value)

    @field_validator("breakpoints", mode="before")
    @classmethod
    def _validate_breakpoints(
        cls, value: object, info: ValidationInfo
    ) -> tuple[Quantity, ...]:
        period_present = "period" in info.data
        period = info.data.get("period")
        return _check_breakpoints(
            value,
            "breakpoints",
            upper=period if isinstance(period, Quantity) else None,
            upper_name="the period",
            require_period=period_present and period is None,
        )

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        steady: dict[str, JsonValue] = {"method": self.method}
        if self.period is not None:
            steady["period"] = self.period.to(TIME, "a periodic steady state's period")
        if self.breakpoints:
            steady["breakpoints"] = [
                time.to(TIME, "a time to step onto") for time in self.breakpoints
            ]
        if self.max_step is not None:
            steady["max_step"] = self.max_step.to(TIME, "a period's longest step")
        return {"periodic_steady_state": steady}


type Differentiated = DcSweep | Ac | Transient | PeriodicSteadyState | None
"""An analysis a sensitivity or a mismatch analysis differentiates, None for the
operating point."""


def differentiated(of: Differentiated, level: Dimension) -> JsonValue:
    """The analysis `of`, differentiated, in the engine's format.

    The levels of a DC sweep are checked against `level`.

    Examples:
        >>> differentiated(None, VOLTAGE)
        'operating_point'
    """
    match of:
        case None:
            return "operating_point"
        case DcSweep():
            return of.to_json(level)
        case Ac() | Transient() | PeriodicSteadyState():
            return of.to_json()


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Sensitivity:
    """How unknowns move with every number of every device's parameters.

    Each output is an unknown by name, a voltage such as ``out`` or a current such as
    ``V1:current``. The derivatives are taken at the operating point, or at each point
    of the analysis `of`: each level of a DC sweep, each frequency of a small-signal
    response, each time of a transient, and each point of a periodic steady state's
    period, at its fraction of the period, the state the period starts from moving so
    that it still repeats, and an oscillator's period with it. They are the analysis's
    Jacobian, which [`jvp`][jolty.simulate.jvp] finds a parameter at a time and
    [`vjp`][jolty.simulate.vjp] an output at a time; the analysis takes whichever costs
    the fewer solves.
    """

    outputs: Sequence[str]
    of: Differentiated = None

    @field_validator("outputs", mode="before")
    @classmethod
    def _validate_outputs(cls, value: object) -> tuple[str, ...]:
        return _check_string_sequence(value, "outputs", "unknown names")

    @field_validator("of", mode="before")
    @classmethod
    def _validate_of(cls, value: object) -> Differentiated:
        return _check_differentiated(value)

    def to_json(self, level: Dimension = VOLTAGE) -> JsonValue:
        """The analysis in the engine's format.

        The levels of a DC sweep it differentiates are checked against `level`.
        """
        outputs: list[JsonValue] = list(self.outputs)
        return {
            "sensitivity": {"outputs": outputs, "of": differentiated(self.of, level)}
        }


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Mismatch:
    """How far unknowns spread, to first order, with the numbers samples draw at random.

    Each output is an unknown by name, as a [`Sensitivity`][jolty.plan.Sensitivity]'s
    is, and spreads at the operating point, or at each point of the analysis `of`, with
    every random variable ([`RandomVariable`][jolty.ir.RandomVariable]): each part's own
    factor and each lot's of the circuit's tolerances, as a Monte Carlo sample draws
    them, and the `random_variables` provided besides, such as a process kit's
    statistical functions
    ([`Library.random_variables`][jolty.spice.Library.random_variables]). Its
    derivative with respect to each is the sensitivity analysis's, summed over the
    numbers the random variable moves; the random variables are drawn apart from one
    another, so to first order the output's standard deviation is the root of the sum
    of each derivative times its random variable's standard deviation, squared, and each
    random variable's share of its variance is its own term's.
    """

    outputs: Sequence[str]
    of: Differentiated = None
    random_variables: Sequence[ir.RandomVariable] = ()

    @field_validator("outputs", mode="before")
    @classmethod
    def _validate_outputs(cls, value: object) -> tuple[str, ...]:
        return _check_string_sequence(value, "outputs", "unknown names")

    @field_validator("of", mode="before")
    @classmethod
    def _validate_of(cls, value: object) -> Differentiated:
        return _check_differentiated(value)

    @field_validator("random_variables", mode="before")
    @classmethod
    def _validate_random_variables(cls, value: object) -> tuple[ir.RandomVariable, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            _fail(
                "sequence_type",
                f"random_variables is a sequence of RandomVariable, not {value!r}",
            )
        items = tuple(value)
        errors: list[InitErrorDetails] = []
        checked: list[ir.RandomVariable] = []
        for index, variable in enumerate(items):
            if not isinstance(variable, ir.RandomVariable):
                errors.append(
                    _detail(
                        "random_variable_type",
                        f"random_variables holds RandomVariable objects, "
                        f"not {variable!r}",
                        variable,
                        (index,),
                    )
                )
                continue
            checked.append(variable)
            for path, named in variable.derivatives.items():
                for parameter, derivative in named.items():
                    if not math.isfinite(derivative):
                        errors.append(
                            _detail(
                                "finite",
                                f"derivative of {path} {parameter} is finite, "
                                f"not {derivative!r}",
                                derivative,
                                (index, path, parameter),
                            )
                        )
        _raise_if_errors("random_variables", errors)
        return tuple(checked)

    def to_json(
        self, level: Dimension = VOLTAGE, toleranced: Sequence[JsonValue] = ()
    ) -> JsonValue:
        """The analysis in the engine's format, with the circuit's `toleranced` numbers.

        The levels of a DC sweep it differentiates are checked against `level`.
        """
        outputs: list[JsonValue] = list(self.outputs)
        provided: list[JsonValue] = [
            variable.model_dump(mode="json") for variable in self.random_variables
        ]
        return {
            "mismatch": {
                "outputs": outputs,
                "of": differentiated(self.of, level),
                "toleranced": list(toleranced),
                "random_variables": provided,
            }
        }


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Stability:
    """The loop gain through a probe at each frequency, by Middlebrook's injections.

    The probe is an independent voltage source in series with the loop, named by its
    path, such as ``VP``: the loop's signal enters its ``positive`` terminal from its
    ``negative`` one, and neither terminal is the ground, through which the injected
    current returns. One voltage injected in series and one current injected from the
    ground compute the loop gain however the two sides load each other (Middlebrook,
    1975). Negative feedback makes the loop gain positive at low frequencies.
    """

    probe: str
    frequencies: Sequence[Quantity]

    @field_validator("probe", mode="before")
    @classmethod
    def _validate_probe(cls, value: object) -> str:
        return _check_string(value, "probe", non_empty=True)

    @field_validator("frequencies", mode="before")
    @classmethod
    def _validate_frequencies(cls, value: object) -> tuple[Quantity, ...]:
        return _check_frequencies(
            value,
            "frequencies",
            strictly_positive=False,
            empty_message="a stability analysis has at least one frequency",
        )

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        frequencies: list[JsonValue] = [
            frequency.to(FREQUENCY, "a frequency") for frequency in self.frequencies
        ]
        return {"stability": {"probe": self.probe, "frequencies": frequencies}}


@dataclass(frozen=True, slots=True, config=_CONFIG)
class TransferFunction:
    """The small-signal transfer function at DC, as SPICE's ``.tf`` reports it.

    From an independent source, named by its path such as ``V1``, to an output: a net's
    voltage, over the ground or over the net `reference` names, or a voltage source's
    current, such as ``V2:current``. It computes the gain, the output per volt or ampere
    of the source, the resistance the source sees, and the resistance the output sees
    with the source at its level: a voltage source a short, a current source open.

    Examples:
        >>> TransferFunction("V1", "out").to_json()
        {'transfer_function': {'source': 'V1', 'output': 'out'}}
    """

    source: str
    output: str
    reference: str | None = None

    @field_validator("source", mode="before")
    @classmethod
    def _validate_source(cls, value: object) -> str:
        return _check_string(value, "source", non_empty=True)

    @field_validator("output", mode="before")
    @classmethod
    def _validate_output(cls, value: object) -> str:
        return _check_string(value, "output", non_empty=True)

    @field_validator("reference", mode="before")
    @classmethod
    def _validate_reference(cls, value: object, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        checked = _check_string(value, "reference", non_empty=True)
        if checked == info.data.get("output"):
            _fail("same_node", f"the output `{checked}` is taken over itself")
        return checked

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        named: dict[str, JsonValue] = {"source": self.source, "output": self.output}
        if self.reference is not None:
            named["reference"] = self.reference
        return {"transfer_function": named}


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Noise:
    """The noise density at an output at each frequency, from every device's noise.

    The output is an unknown by name, a voltage such as ``out`` or a current such as
    ``V1:current``, or a voltage over a `reference` node, for a differential output.
    Resistances make thermal noise, junctions shot noise, and diodes and transistors
    with a flicker coefficient flicker noise, as SPICE models them. With a `source`,
    an independent source such as ``V1``, the gain from its level to the output refers
    the noise back to it, as an input's noise.
    """

    output: str
    frequencies: Sequence[Quantity]
    reference: str | None = None
    source: str | None = None

    @field_validator("output", mode="before")
    @classmethod
    def _validate_output(cls, value: object) -> str:
        return _check_string(value, "output", non_empty=True)

    @field_validator("frequencies", mode="before")
    @classmethod
    def _validate_frequencies(cls, value: object) -> tuple[Quantity, ...]:
        return _check_frequencies(
            value,
            "frequencies",
            strictly_positive=True,
            empty_message="a noise analysis has at least one frequency",
        )

    @field_validator("reference", mode="before")
    @classmethod
    def _validate_reference(cls, value: object, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        checked = _check_string(value, "reference", non_empty=True)
        if checked == info.data.get("output"):
            _fail("same_node", f"the output `{checked}` is taken over itself")
        return checked

    @field_validator("source", mode="before")
    @classmethod
    def _validate_source(cls, value: object) -> str | None:
        if value is None:
            return None
        return _check_string(value, "source", non_empty=True)

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        frequencies: list[JsonValue] = [
            frequency.to(FREQUENCY, "a frequency") for frequency in self.frequencies
        ]
        noise: dict[str, JsonValue] = {
            "output": self.output,
            "frequencies": frequencies,
        }
        if self.reference is not None:
            noise["reference"] = self.reference
        if self.source is not None:
            noise["source"] = self.source
        return {"noise": noise}


@dataclass(frozen=True, slots=True, config=_CONFIG)
class PoleZero:
    """The poles of the linearized circuit, and the zeros of a transfer function.

    The circuit is linearized at its operating point, and the transfer function is from
    a `source` to an `output`, with both set. The poles are the circuit's natural
    frequencies, the complex frequencies at which its small-signal equations have a
    solution without any excitation. The output is an unknown by name, or a voltage
    over a `reference` node.
    """

    source: str | None = None
    output: str | None = None
    reference: str | None = None

    @field_validator("source", "output", mode="before")
    @classmethod
    def _validate_endpoint(cls, value: object, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _check_string(value, info.field_name or "name", non_empty=True)

    @field_validator("reference", mode="before")
    @classmethod
    def _validate_reference(cls, value: object, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        checked = _check_string(value, "reference", non_empty=True)
        if checked == info.data.get("output"):
            _fail("same_node", f"the output `{checked}` is taken over itself")
        return checked

    @model_validator(mode="after")
    def _validate_pairing(self) -> PoleZero:
        both_set = self.source is not None and self.output is not None
        none_set = (
            self.source is None and self.output is None and self.reference is None
        )
        if not (both_set or none_set):
            _fail(
                "pairing",
                "a pole-zero analysis's zeros require a source and an output, "
                "both or neither",
            )
        return self

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        found = {
            "source": self.source,
            "output": self.output,
            "reference": self.reference,
        }
        named: dict[str, JsonValue] = {
            key: value for key, value in found.items() if value is not None
        }
        return {"pole_zero": named}


type Analysis = (
    OperatingPoint
    | OperatingPoints
    | DcSweep
    | Ac
    | Transient
    | PeriodicSteadyState
    | Sensitivity
    | Mismatch
    | Stability
    | Noise
    | PoleZero
    | TransferFunction
)
"""Any analysis."""

_ANALYSIS_TYPES: Final = (
    OperatingPoint,
    OperatingPoints,
    DcSweep,
    Ac,
    Transient,
    PeriodicSteadyState,
    Sensitivity,
    Mismatch,
    Stability,
    Noise,
    PoleZero,
    TransferFunction,
)
"""Concrete runtime tuple of every `Analysis` class."""


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Parallelism:
    """How a run spreads its own work across the engine's threads.

    Each kind of work spreads in tasks of at least its threshold, its least size per
    task, once it fills such a task for every thread the run may occupy, and stays on
    the run's own thread for less: the frequencies of a run's small-signal analyses and
    the right-hand sides of its sensitivities, in tasks of at least
    `min_solves_per_task`; each group of its device instances, in tasks of at least
    `min_instances_per_task`, but a linear one's, which are evaluated once a run, for
    their entries; and the work of its Newton iteration and its transient's step control
    on each unknown and each stored entry of its matrix, in tasks of at least
    `min_unknowns_per_task`, in whole chunks of 1,024 its threads take as they free.
    Each block of its matrix of more than `max_unknowns_per_interior` unknowns is
    ordered by nested dissection into interiors of at most as many and the separators
    between them, which factor at once, wherever that costs its elimination fewer
    multiply-adds than the minimum degree ordering. A run spreads over at most
    `max_threads` of the engine's threads, the setting `threads` or
    [`threads`][jolty.simulate.threads], and over all of them by default; a batch runs
    each run on one thread whenever it has a run for every thread. Each setting left out
    but `max_threads` takes its break-even on the threads the run may occupy,
    [`BREAK_EVENS`][jolty.plan.BREAK_EVENS]. Every result is the same bits however the
    work is split, and no setting but `max_unknowns_per_interior` changes any: that one
    orders the factorization's operations otherwise, so results agree with those of
    another to within its rounding, as they do on another number of threads where its
    default dissects a block.

    Examples:
        >>> Parallelism(min_solves_per_task=16).to_json()
        {'min_solves_per_task': 16}
        >>> Parallelism(max_threads=1).to_json()
        {'max_threads': 1}
    """

    max_threads: int | None = None
    min_solves_per_task: int | None = None
    min_instances_per_task: int | None = None
    min_unknowns_per_task: int | None = None
    max_unknowns_per_interior: int | None = None

    @field_validator(
        "max_threads",
        "min_solves_per_task",
        "min_instances_per_task",
        "min_unknowns_per_task",
        "max_unknowns_per_interior",
        mode="before",
    )
    @classmethod
    def _validate_threshold(cls, value: object, info: ValidationInfo) -> int | None:
        if value is None:
            return None
        return _check_integer(value, info.field_name or "threshold", minimum=1)

    def to_json(self) -> dict[str, JsonValue]:
        """The settings in the engine's format, without those left to defaults."""
        values = {setting.name: getattr(self, setting.name) for setting in fields(self)}
        return {name: value for name, value in values.items() if value is not None}


DEFAULTS: Final[Mapping[str, float]] = MappingProxyType(
    {
        name: value
        for name, value in json.loads(_core.default_options()).items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
)
"""The engine's default of each numerical option of [`Options`][jolty.plan.Options], by
name, SPICE's: what an option left unset takes, as the engine defines it; those of its
[`Parallelism`][jolty.plan.Parallelism] are [`BREAK_EVENS`][jolty.plan.BREAK_EVENS].

Examples:
    >>> DEFAULTS["charge_tolerance"]
    1e-14
"""

BREAK_EVENS: Final[Mapping[str, tuple[int | None, ...]]] = MappingProxyType(
    {name: tuple(values) for name, values in json.loads(_core.break_evens()).items()}
)
"""The break-evens each threshold of [`Parallelism`][jolty.plan.Parallelism] left unset
takes, as the engine defines them: under ``"threads"``, the numbers of threads they were
measured on, and under each threshold's name, its break-even on each, none where no size
measured paid; a run takes the one measured on the most threads it may occupy or fewer.

Examples:
    >>> BREAK_EVENS["threads"]
    (2, 4, 8, 64)
"""


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Options:
    """Numerical settings; each one left out takes the engine's default, SPICE's own.

    Tolerances are in SI base units: volts, amperes and coulombs. The `parallelism`
    says how a run spreads its own work across threads, which changes no result but
    for the last digits its dissected factorization's order of operations produces.

    Examples:
        >>> Options(gmin=1e-15).to_json()
        {'gmin': 1e-15}
        >>> Options(parallelism=Parallelism(min_solves_per_task=16)).to_json()
        {'parallelism': {'min_solves_per_task': 16}}
    """

    relative_tolerance: float | None = None
    voltage_tolerance: float | None = None
    current_tolerance: float | None = None
    charge_tolerance: float | None = None
    gmin: float | None = None
    max_dc_iterations: int | None = None
    max_transient_iterations: int | None = None
    truncation_tolerance: float | None = None
    max_memory: float | None = None
    profile: bool | None = None
    """Whether a transient measures where its time goes, into its statistics' profile;
    it changes no other result."""
    parallelism: Parallelism = field(default_factory=Parallelism)
    nodesets: Mapping[str, Quantity] = field(default_factory=dict)
    """The voltage each net named, by path, is held at while an operating point is
    first solved from zero, then released and solved again from there, as SPICE's
    ``.nodeset``: a guess that leads it to one solution of several. A net that voltage
    sources and inductors join to the ground, or to a net named before it, is left to
    them."""

    @field_validator(
        "relative_tolerance",
        "voltage_tolerance",
        "current_tolerance",
        "charge_tolerance",
        "truncation_tolerance",
        "max_memory",
        mode="before",
    )
    @classmethod
    def _validate_positive(cls, value: object, info: ValidationInfo) -> float | None:
        if value is None:
            return None
        return _check_float(value, info.field_name or "option", positive=True)

    @field_validator("gmin", mode="before")
    @classmethod
    def _validate_gmin(cls, value: object) -> float | None:
        if value is None:
            return None
        return _check_float(value, "gmin", non_negative=True)

    @field_validator("max_dc_iterations", "max_transient_iterations", mode="before")
    @classmethod
    def _validate_iterations(cls, value: object, info: ValidationInfo) -> int | None:
        if value is None:
            return None
        return _check_integer(value, info.field_name or "option", minimum=1)

    @field_validator("profile", mode="before")
    @classmethod
    def _validate_profile(cls, value: object) -> bool | None:
        if value is None:
            return None
        return _check_boolean(value, "profile")

    @field_validator("parallelism", mode="before")
    @classmethod
    def _validate_parallelism(cls, value: object) -> object:
        if isinstance(value, Parallelism | Mapping):
            return value
        _fail("parallelism_type", f"parallelism is a Parallelism, not {value!r}")

    @field_validator("nodesets", mode="before")
    @classmethod
    def _validate_nodesets(cls, value: object) -> dict[str, Quantity]:
        return _check_voltage_map(value, "nodesets", "a nodeset")

    def to_json(self) -> dict[str, JsonValue]:
        """The options in the engine's format, without those left to defaults."""
        values: dict[str, JsonValue] = {
            setting.name: getattr(self, setting.name)
            for setting in fields(self)
            if setting.name not in {"parallelism", "nodesets"}
        }
        spread = self.parallelism.to_json()
        if spread:
            values["parallelism"] = spread
        if self.nodesets:
            values["nodesets"] = dict[str, JsonValue](
                voltages(self.nodesets, "a nodeset")
            )
        return {name: value for name, value in values.items() if value is not None}


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Conditions:
    """What a simulation runs its circuit at.

    A Monte Carlo sample and a corner each run at conditions of their own.
    """

    temperature: float
    """The temperature, in kelvins."""
    process: str | None = None
    """The process corner of a foundry's devices, such as ``ss``, its kit's typical one
    unless set."""
    overrides: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    """The parameters that replace the circuit's, by instance path, then by parameter
    path, such as ``waveform.dc``, in SI base units."""
    mismatch: int | None = None
    """The seed a foundry's devices draw their kit's mismatch from, if they draw it."""

    @field_validator("temperature", mode="before")
    @classmethod
    def _validate_temperature(cls, value: object) -> float:
        return _check_float(value, "temperature")

    @field_validator("process", mode="before")
    @classmethod
    def _validate_process(cls, value: object) -> str | None:
        if value is None:
            return None
        return _check_string(value, "process", non_empty=True)

    @field_validator("mismatch", mode="before")
    @classmethod
    def _validate_mismatch(cls, value: object) -> int | None:
        if value is None:
            return None
        return _check_integer(value, "mismatch", minimum=0, maximum=_SEED_LIMIT)


@stdlib_dataclass(frozen=True, slots=True, init=False)
class Corner:
    """Conditions a plan also runs its analyses at.

    A corner holds a temperature and a foundry's process corner, the plan's unless set,
    and arguments to the build script's ``build()``, which builds the circuit again with
    them, such as a lower supply voltage.

    Examples:
        >>> from jolty.units import V, celsius
        >>> corner = Corner(temperature=celsius(-40), supply=3 * V)
        >>> str(corner.arguments["supply"])
        '3 V'
        >>> Corner(process="ss").process
        'ss'
    """

    temperature: Quantity | None
    process: str | None
    arguments: Mapping[str, object]

    def __init__(
        self,
        *,
        temperature: Quantity | None = None,
        process: str | None = None,
        **arguments: object,
    ) -> None:
        """A corner at `temperature` and `process`, building with `arguments`.

        Raises:
            DimensionError: When `temperature` has another dimension than temperature.
            ValidationError: When `temperature` is not a finite `Quantity` or `process`
                is not a non-empty string.
        """
        errors: list[InitErrorDetails] = []
        if temperature is not None:
            try:
                _check_quantity(temperature, TEMPERATURE, "a corner's temperature")
            except PydanticCustomError as error:
                errors.append(
                    InitErrorDetails(
                        type=error, loc=("temperature",), input=temperature
                    )
                )
        if process is not None:
            try:
                _check_string(process, "process", non_empty=True)
            except PydanticCustomError as error:
                errors.append(
                    InitErrorDetails(type=error, loc=("process",), input=process)
                )
        _raise_if_errors("Corner", errors)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "process", process)
        object.__setattr__(self, "arguments", arguments)


@dataclass(frozen=True, slots=True, config=_CONFIG)
class MonteCarlo:
    """Samples of the circuit's tolerances, each a run of the plan's analyses.

    Each sample runs at the plan's own conditions and at each of its corners, on the
    same parts wherever the corner's circuit holds the same toleranced parameters, and
    a requirement's verdict on it is its worst there. A `seed` draws the same samples
    from the same circuit: sample `i` from the seed and `i` alone, as the engine draws
    it. As variants, sample `i` is run `i`.
    """

    samples: int
    seed: int = 0

    @field_validator("samples", mode="before")
    @classmethod
    def _validate_samples(cls, value: object) -> int:
        return _check_integer(
            value,
            "samples",
            minimum=1,
            minimum_message=(
                f"a Monte Carlo analysis draws a sample or more, not {value}"
            ),
        )

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            _fail("int_type", f"seed is an integer, not {value!r}")
        if not 0 <= value < _SEED_LIMIT:
            message = f"a Monte Carlo seed lies from 0 to 2**64 - 1, not {value}"
            _fail("seed_range", message)
        return value


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Variant:
    """The values one run of a batch takes: numbers and a temperature.

    The numbers replace parameters of devices, by instance path then by parameter path,
    such as ``waveform.dc``, in SI base units, as a simulation's overrides do; the
    temperature is the simulation's unless set.

    Examples:
        >>> from jolty.units import celsius
        >>> Variant({"R1": {"resistance": 2e3}}, celsius(85)).overrides
        {'R1': {'resistance': 2000.0}}
    """

    overrides: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    temperature: Quantity | None = None

    @field_validator("overrides", mode="before")
    @classmethod
    def _validate_overrides(cls, value: object) -> dict[str, dict[str, float]]:
        if not isinstance(value, Mapping):
            _fail("mapping_type", f"overrides is a mapping, not {value!r}")
        errors: list[InitErrorDetails] = []
        numbers: dict[str, dict[str, float]] = {}
        for path, named in value.items():
            if not isinstance(path, str):
                errors.append(
                    _detail(
                        "string_type",
                        f"an instance path is a string, not {path!r}",
                        path,
                        (repr(path),),
                    )
                )
                continue
            if not isinstance(named, Mapping):
                errors.append(
                    _detail(
                        "mapping_type",
                        f"overrides for {path!r} is a mapping, not {named!r}",
                        named,
                        (path,),
                    )
                )
                continue
            inner: dict[str, float] = {}
            for name, number in named.items():
                if not isinstance(name, str):
                    errors.append(
                        _detail(
                            "string_type",
                            f"a parameter name is a string, not {name!r}",
                            name,
                            (path, repr(name)),
                        )
                    )
                    continue
                if isinstance(number, bool) or not isinstance(number, int | float):
                    errors.append(
                        _detail(
                            "float_type",
                            f"{name} is a real number, not {number!r}",
                            number,
                            (path, name),
                        )
                    )
                    continue
                converted = float(number)
                if not math.isfinite(converted):
                    errors.append(
                        _detail(
                            "finite",
                            f"{name} is finite, not {converted!r}",
                            number,
                            (path, name),
                        )
                    )
                    continue
                inner[name] = converted
            numbers[path] = inner
        _raise_if_errors("overrides", errors)
        return numbers

    @field_validator("temperature", mode="before")
    @classmethod
    def _validate_temperature(cls, value: object) -> Quantity | None:
        if value is None:
            return None
        return _check_quantity(value, TEMPERATURE, "a variant's temperature")


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Explicit:
    """Runs at listed values, run `i` at the variant `i`.

    Examples:
        >>> from jolty.units import celsius
        >>> len(
        ...     Explicit([Variant(temperature=celsius(t)) for t in (-40, 125)]).variants
        ... )
        2
    """

    variants: Sequence[Variant]

    @field_validator("variants", mode="before")
    @classmethod
    def _validate_variants(cls, value: object) -> tuple[Variant, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            _fail(
                "sequence_type",
                f"explicit variants is a sequence of variants, not {value!r}",
            )
        listed = tuple(value)
        if not listed:
            _fail("empty", "explicit variants list a run or more")
        errors: list[InitErrorDetails] = []
        checked: list[Variant] = []
        for index, variant in enumerate(listed):
            if not isinstance(variant, Variant):
                errors.append(
                    _detail(
                        "variant_type",
                        f"explicit variants are variants, not {variant!r}",
                        variant,
                        (index,),
                    )
                )
            else:
                checked.append(variant)
        _raise_if_errors("variants", errors)
        return tuple(checked)


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Axis:
    """What one axis of a grid steps, and its values in order.

    It steps a number of a device's parameters, named by the instance's path and the
    parameter's, such as ``R1`` and ``resistance``, its values in SI base units in the
    dimension of the first; or the temperature, with neither named.

    Examples:
        >>> from jolty.units import celsius, kohm
        >>> Axis("R1", "resistance", [1 * kohm, 2 * kohm]).numbers()
        [1000.0, 2000.0]
        >>> Axis.temperature([celsius(-40), celsius(125)]).numbers()
        [233.14999999999998, 398.15]
    """

    instance: str | None
    parameter: str | None
    values: Sequence[Quantity]

    @field_validator("instance", mode="before")
    @classmethod
    def _validate_instance(cls, value: object) -> str | None:
        if value is None:
            return None
        return _check_string(value, "instance", non_empty=True)

    @field_validator("parameter", mode="before")
    @classmethod
    def _validate_parameter(cls, value: object, info: ValidationInfo) -> str | None:
        checked = (
            None if value is None else _check_string(value, "parameter", non_empty=True)
        )
        if "instance" in info.data and (info.data["instance"] is None) != (
            checked is None
        ):
            _fail(
                "axis_pairing",
                "an axis names an instance and its parameter, or neither",
            )
        return checked

    @field_validator("values", mode="before")
    @classmethod
    def _validate_values(
        cls, value: object, info: ValidationInfo
    ) -> tuple[Quantity, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            _fail(
                "sequence_type",
                f"an axis's values is a sequence of quantities, not {value!r}",
            )
        items = tuple(value)
        if not items:
            _fail("empty", "an axis steps a value or more")
        is_temperature = (
            info.data.get("instance") is None
            and info.data.get("parameter") is None
            and "instance" in info.data
            and "parameter" in info.data
        )
        errors: list[InitErrorDetails] = []
        checked: list[Quantity] = []
        expected: Dimension | None = TEMPERATURE if is_temperature else None
        for index, item in enumerate(items):
            if not isinstance(item, Quantity):
                errors.append(
                    _detail(
                        "quantity_type",
                        f"an axis's value is a Quantity, not {item!r}",
                        item,
                        (index,),
                    )
                )
                continue
            if expected is None:
                expected = item.dimension
            elif item.dimension != expected:
                message = f"an axis's value takes a quantity in {expected}, not {item}"
                raise DimensionError(message)
            if not math.isfinite(item.value):
                errors.append(
                    _detail(
                        "finite",
                        f"an axis's value is finite, not {item}",
                        item,
                        (index,),
                    )
                )
            else:
                checked.append(item)
        _raise_if_errors("values", errors)
        return tuple(checked)

    @classmethod
    def temperature(cls, values: Sequence[Quantity]) -> Axis:
        """The axis that steps the temperature through `values`."""
        return cls(None, None, values)

    def numbers(self) -> list[float]:
        """The values in SI base units: kelvins for the temperature.

        Raises:
            DimensionError: When a value is not of the axis's dimension.
        """
        dimension = TEMPERATURE if self.instance is None else self.values[0].dimension
        return [value.to(dimension, "an axis's value") for value in self.values]


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Grid:
    """Runs on a grid: every combination of its axes' values, the last axis fastest.

    Run `i` takes, from the last axis back, the value of each axis at `i` modulo its
    length, `i` then divided by it, as digits in mixed radix. Each line along the last
    axis is a chain of warm runs when `warm`: each run of it starts from the operating
    point of the one before, as a DC sweep's levels do.

    Examples:
        >>> from jolty.units import celsius, kohm
        >>> grid = Grid(
        ...     [
        ...         Axis.temperature([celsius(0), celsius(85)]),
        ...         Axis("R1", "resistance", [1 * kohm, 2 * kohm, 3 * kohm]),
        ...     ]
        ... )
        >>> len(grid.axes)
        2
    """

    axes: Sequence[Axis]
    warm: bool = False

    @field_validator("axes", mode="before")
    @classmethod
    def _validate_axes(cls, value: object) -> tuple[Axis, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            _fail(
                "sequence_type", f"a grid's axes is a sequence of axes, not {value!r}"
            )
        listed = tuple(value)
        errors: list[InitErrorDetails] = []
        checked: list[Axis] = []
        for index, axis in enumerate(listed):
            if not isinstance(axis, Axis):
                errors.append(
                    _detail(
                        "axis_type",
                        f"a grid's axes are axes, not {axis!r}",
                        axis,
                        (index,),
                    )
                )
            else:
                checked.append(axis)
        _raise_if_errors("axes", errors)
        return tuple(checked)

    @field_validator("warm", mode="before")
    @classmethod
    def _validate_warm(cls, value: object) -> bool:
        return _check_boolean(value, "warm")


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Product:
    """Every run of `first` at every run of `second`.

    Run `i` is run `i // len(second)` of `first`, then run `i % len(second)` of
    `second`, whose values replace those of `first` where both hold one, so corners,
    samples and a sweep make one batch. Its chains are those of `second`.
    """

    first: Variants
    second: Variants

    @field_validator("first", "second", mode="before")
    @classmethod
    def _validate_part(cls, value: object) -> Variants:
        return _check_variants(value, "a product is of variants")


type Variants = MonteCarlo | Grid | Explicit | Product
"""The values of many runs of one circuit, each a pure function of the run's index."""

rebuild_dataclass(Product)  # ty: ignore[invalid-argument-type] - @pydantic.dataclasses.dataclass attaches __pydantic_complete__ at runtime


def runs(variants: Variants) -> int:
    """How many runs `variants` make.

    Examples:
        >>> from jolty.units import celsius
        >>> grid = Grid([Axis.temperature([celsius(0), celsius(85)])])
        >>> runs(Product(grid, MonteCarlo(100)))
        200
    """
    match variants:
        case MonteCarlo(samples=samples):
            return samples
        case Grid(axes=axes):
            return math.prod(len(axis.values) for axis in axes)
        case Explicit(variants=listed):
            return len(listed)
        case Product(first=first, second=second):
            return runs(first) * runs(second)


@dataclass(frozen=True, slots=True, config=_CONFIG)
class WorstCase:
    """Each requirement at the extremes of the circuit's tolerances that worsen it.

    At the plan's own conditions and at each corner, a run per toleranced parameter,
    moved a `step` of its tolerance, finds which way each requirement's value moves with
    it. Then, for each bound of each requirement, every parameter goes to the extreme of
    its tolerance that moves the value toward the bound, and a lot's shared spread to
    the extreme its parts move it toward together, as PSpice's worst-case analysis sets
    them, and the requirement is judged there. A normal tolerance's extreme is its three
    standard deviations.
    """

    step: float = 0.1

    @field_validator("step", mode="before")
    @classmethod
    def _validate_step(cls, value: object) -> float:
        converted = _check_float(value, "step")
        if not 0.0 < converted <= 1.0:
            message = (
                f"a worst case moves a step above 0 and at most 1, not {converted!r}"
            )
            _fail("step_range", message)
        return converted


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Characterized:
    """A logic model characterized from a component's own circuit, at `supply`.

    The component's ports are its inputs, its outputs, one supply and one ground. Its
    timing is measured over a grid of input transitions, `slews` times the cell's own
    output transition, and of output loads, `fanouts` times its mean input capacitance.
    Each run, `spot_checks` of the instances the scope matches, drawn with `seed`, are
    simulated again at their circuit's fidelity, and each of their outputs' changes must
    lie within the model's measured error, plus `tolerance` times its mean delay, of the
    model's.

    Examples:
        >>> from jolty.units import V
        >>> Characterized(1.8 * V).fanouts
        (0.0, 1.0, 2.0, 4.0)
    """

    supply: Quantity
    slews: Sequence[float] = (0.5, 1.0, 2.0, 4.0)
    fanouts: Sequence[float] = (0.0, 1.0, 2.0, 4.0)
    spot_checks: int = 1
    tolerance: float = 0.1
    seed: int = 0

    @field_validator("supply", mode="before")
    @classmethod
    def _validate_supply(cls, value: object) -> Quantity:
        return _check_quantity(
            value, VOLTAGE, "a characterization's supply", positive=True
        )

    @field_validator("slews", mode="before")
    @classmethod
    def _validate_slews(cls, value: object) -> tuple[float, ...]:
        return _check_rising_grid(value, "slews", strictly_positive=True)

    @field_validator("fanouts", mode="before")
    @classmethod
    def _validate_fanouts(cls, value: object) -> tuple[float, ...]:
        return _check_rising_grid(value, "fanouts", strictly_positive=False)

    @field_validator("spot_checks", mode="before")
    @classmethod
    def _validate_spot_checks(cls, value: object) -> int:
        return _check_integer(
            value,
            "spot_checks",
            minimum=0,
            minimum_message=f"spot_checks is zero or more, not {value}",
        )

    @field_validator("tolerance", mode="before")
    @classmethod
    def _validate_tolerance(cls, value: object) -> float:
        return _check_float(value, "tolerance", non_negative=True)

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            _fail("int_type", f"seed is an integer, not {value!r}")
        if not 0 <= value < _SEED_LIMIT:
            message = f"a seed lies from 0 to 2**64 - 1, not {value}"
            _fail("seed_range", message)
        return value


TIGHT: Final = Options(
    relative_tolerance=ngspice.RELATIVE_TOLERANCE,
    voltage_tolerance=ngspice.VOLTAGE_TOLERANCE,
    current_tolerance=ngspice.CURRENT_TOLERANCE,
)
"""jolty's options as tight as the ones [`jolty.spice.ngspice`][jolty.spice.ngspice]
sets for ngspice, which the comparisons of the two run with.

The charge tolerance is both engines' default, 1e-14 C, as both read it alike: the least
charge the relative tolerance is taken of, so a charge is held to 1e-20 C at least."""


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Plan:
    """Named analyses to run on a circuit, the options they share, and its conditions.

    Each analysis is named by a Python identifier, such as ``step`` or ``bode``, which
    names its results too. The circuit's devices are simulated at `temperature`, 27 °C
    unless set, a foundry's devices at its `process` corner, its kit's typical unless
    set, then at each of the `corners`, each named by an identifier too, on the
    samples of a `monte_carlo` analysis, on the runs of its `variants`, and at the
    extremes of a `worst_case` analysis. The samples and the variants each run at the
    plan's own conditions and at every corner, as one batch of runs per circuit.

    Its `fidelity` maps scopes, patterns of instance paths such as ``adder/*``, where
    ``*`` matches within a level of the hierarchy and ``**`` across levels, to the
    logic models characterized for the component instances they match; an instance
    within another that a scope matches takes the outer one's model.

    Every result holds the circuit's unknowns. Its `record` names the instances whose
    detail the results keep besides, by path or by patterns written as scopes are, where
    a pattern after ``!`` takes back what the patterns before it named, the last that
    matches an instance deciding, as in a ``.gitignore``: ``["X1/**", "!X1/bias/**"]``
    records every instance within ``X1`` but its bias network, ``["**"]`` every
    instance, and ``[]``, as when the record is left out, none. A device recorded keeps
    its terminals' currents and charges, its derivatives and its variables; a component
    instance or a logic device recorded keeps its terminals' currents and charges, each
    the sum of the device terminals behind it. An analysis that records
    ([`Recorded`][jolty.plan.Recorded]) and names a record of its own keeps that one
    in place of the plan's, and an operating point and a transient also record the
    instances rated for a current or a power, whose ratings the checks hold them to.
    """

    analyses: Mapping[str, Analysis]
    options: Options = field(default_factory=Options)
    temperature: Quantity | None = None
    corners: Mapping[str, Corner] = field(default_factory=dict)
    monte_carlo: MonteCarlo | None = None
    worst_case: WorstCase | None = None
    fidelity: Mapping[str, Characterized] = field(default_factory=dict)
    process: str | None = None
    record: Sequence[str] | None = None
    variants: Variants | None = None

    @field_validator("analyses", mode="before")
    @classmethod
    def _validate_analyses(cls, value: object) -> dict[str, Analysis]:
        return _check_identifier_mapping(
            value,
            _ANALYSIS_TYPES,
            _IdentifierRule(
                title="analyses",
                kind="an analysis",
                type_message="the analysis {key} is {item}, not an Analysis",
                non_empty_message="a plan runs at least one analysis",
            ),
        )

    @field_validator("options", mode="before")
    @classmethod
    def _validate_options(cls, value: object) -> object:
        if isinstance(value, Options | Mapping):
            return value
        _fail("options_type", f"options is an Options, not {value!r}")

    @field_validator("temperature", mode="before")
    @classmethod
    def _validate_temperature(cls, value: object) -> Quantity | None:
        if value is None:
            return None
        return _check_quantity(value, TEMPERATURE, "a plan's temperature")

    @field_validator("corners", mode="before")
    @classmethod
    def _validate_corners(cls, value: object) -> dict[str, Corner]:
        return _check_identifier_mapping(
            value,
            Corner,
            _IdentifierRule(
                title="corners",
                kind="a corner",
                type_message="the corner {key} is {item}, not a Corner",
            ),
        )

    @field_validator("monte_carlo", mode="before")
    @classmethod
    def _validate_monte_carlo(cls, value: object) -> object:
        if value is None or isinstance(value, MonteCarlo | Mapping):
            return value
        _fail("monte_carlo_type", f"monte_carlo is a MonteCarlo, not {value!r}")

    @field_validator("worst_case", mode="before")
    @classmethod
    def _validate_worst_case(cls, value: object) -> object:
        if value is None or isinstance(value, WorstCase | Mapping):
            return value
        _fail("worst_case_type", f"worst_case is a WorstCase, not {value!r}")

    @field_validator("fidelity", mode="before")
    @classmethod
    def _validate_fidelity(cls, value: object) -> dict[str, Characterized]:
        if not isinstance(value, Mapping):
            _fail("mapping_type", f"fidelity is a mapping, not {value!r}")
        errors: list[InitErrorDetails] = []
        checked: dict[str, Characterized] = {}
        for scope, model in value.items():
            if not isinstance(scope, str) or not scope:
                errors.append(
                    _detail(
                        "string_type",
                        f"a fidelity scope is a non-empty string, not {scope!r}",
                        scope,
                        (str(scope),),
                    )
                )
            if not isinstance(model, Characterized):
                errors.append(
                    _detail(
                        "characterized_type",
                        f"the scope {scope} takes {model!r}, not a Characterized model",
                        model,
                        (str(scope),),
                    )
                )
            elif isinstance(scope, str) and scope:
                checked[scope] = model
        _raise_if_errors("fidelity", errors)
        return checked

    @field_validator("process", mode="before")
    @classmethod
    def _validate_process(cls, value: object) -> str | None:
        if value is None:
            return None
        return _check_string(value, "process", non_empty=True)

    @field_validator("record", mode="before")
    @classmethod
    def _validate_record(cls, value: object) -> tuple[str, ...] | None:
        return _check_record(value)

    @field_validator("variants", mode="before")
    @classmethod
    def _validate_variants(cls, value: object) -> Variants | None:
        if value is None:
            return None
        return _check_variants(value, "a plan's variants are variants")

    def process_at(self, corner: Corner | None = None) -> str | None:
        """The process corner of the plan's own run, or of `corner`'s.

        None stands for the typical corner of the kit of each foundry's device.
        """
        if corner is not None and corner.process is not None:
            return corner.process
        return self.process

    def temperature_at(self, corner: Corner | None = None) -> Quantity | None:
        """The temperature of the plan's own run, or of `corner`'s; None for 27 °C."""
        if corner is not None and corner.temperature is not None:
            return corner.temperature
        return self.temperature


def decade(start: Quantity, stop: Quantity, per_decade: int) -> list[Quantity]:
    """Frequencies from `start` to `stop`, `per_decade` of them spaced evenly a decade.

    The `k`th of `n` is `start · r^k`, with `r = (stop / start)^(1/n)`, computed in
    decimal arithmetic to `DECADE_DIGITS` digits, step after step, and only then rounded
    to the nearest float, so that every platform finds the same frequencies, where its
    own `log10` and `pow` round differently.

    Examples:
        >>> from jolty.units import Hz
        >>> [str(f) for f in decade(1 * Hz, 100 * Hz, 2)]
        ['1 Hz', '3.162 Hz', '10 Hz', '31.62 Hz', '100 Hz']
    """
    low, high = (
        start.to(FREQUENCY, "a sweep's start"),
        stop.to(FREQUENCY, "a sweep's stop"),
    )
    if not 0 < low < high or per_decade < 1:
        message = "a decade sweep rises from a positive start, a point or more a decade"
        raise ValueError(message)
    with decimal.localcontext(prec=DECADE_DIGITS):
        span = Decimal(high) / Decimal(low)
        count = max(1, round(span.log10() * per_decade))
        ratio = span ** (Decimal(1) / count)
        frequencies: list[Quantity] = []
        frequency = Decimal(low)
        for _ in range(count + 1):
            frequencies.append(Quantity(float(frequency), FREQUENCY))
            frequency *= ratio
        return frequencies


def linear(start: Quantity, stop: Quantity, points: int) -> list[Quantity]:
    """`points` quantities evenly spaced from `start` to `stop`, both included.

    Examples:
        >>> from jolty.units import V
        >>> [str(level) for level in linear(0 * V, 1 * V, 3)]
        ['0 V', '500 mV', '1 V']
    """
    if points < 2:  # noqa: PLR2004 - a start and a stop
        message = f"a linear sweep has at least two points, not {points}"
        raise ValueError(message)
    step = (stop - start) / (points - 1)
    return [start + step * index for index in range(points)]

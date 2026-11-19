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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass as stdlib_dataclass
from dataclasses import field, fields
from decimal import Decimal
from types import MappingProxyType
from typing import Annotated, Final, Literal, LiteralString

from pydantic import (
    BeforeValidator,
    ConfigDict,
    JsonValue,
    ValidationError,
    ValidationInfo,
    model_validator,
)
from pydantic.dataclasses import dataclass, rebuild_dataclass
from pydantic_core import InitErrorDetails, PydanticCustomError

from jolty import _core, ir
from jolty.errors import DimensionError, detail, fail, raise_if_errors
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

_MAX_SEED_EXCLUSIVE: Final = 2**64
"""The exclusive upper bound of a 64-bit unsigned seed."""

_CONFIG: Final = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
"""Shared Pydantic configuration for every plan dataclass."""

type Method = Literal["trapezoidal", "gear2", "backward_euler"]
"""An integration method: trapezoidal by default, Gear2 for stiff circuits that ring,
backward Euler for the most damping."""


def _integer(
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    range_message: str | None = None,
    optional: bool = False,
) -> BeforeValidator:
    """Builds a strict integer field validator over `[minimum, maximum)`."""

    def validate(value: object, info: ValidationInfo) -> int | None:
        if optional and value is None:
            return None
        name = info.field_name or "option"
        if isinstance(value, bool) or not isinstance(value, int):
            fail("int_type", f"{name} is an integer, not {value!r}")
        if (minimum is not None and value < minimum) or (
            maximum is not None and value >= maximum
        ):
            message = (
                f"{name} is {minimum} or more, not {value}"
                if range_message is None
                else range_message.format(value=value)
            )
            fail("int_range", message)
        return value

    return BeforeValidator(validate)


def _seed(label: str, *, optional: bool = False) -> BeforeValidator:
    """Builds a 64-bit unsigned integer seed validator."""
    return _integer(
        minimum=0,
        maximum=_MAX_SEED_EXCLUSIVE,
        range_message=f"{label} lies from 0 to 2**64 - 1, not {{value}}",
        optional=optional,
    )


def _check_float(
    value: object,
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    """Validates `value` is a strict real number (not bool), finite and in range."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        fail("float_type", f"{name} is a real number, not {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        fail("finite", f"{name} is finite, not {converted!r}")
    if positive and converted <= 0.0:
        fail("positive", f"{name} is positive, not {converted!r}")
    if non_negative and converted < 0.0:
        fail("non_negative", f"{name} is 0 or more, not {converted!r}")
    return converted


def _real(
    *,
    positive: bool = False,
    non_negative: bool = False,
    optional: bool = False,
) -> BeforeValidator:
    """Builds a strict finite float field validator."""

    def validate(value: object, info: ValidationInfo) -> float | None:
        if optional and value is None:
            return None
        return _check_float(
            value,
            info.field_name or "option",
            positive=positive,
            non_negative=non_negative,
        )

    return BeforeValidator(validate)


def _boolean(*, optional: bool = False) -> BeforeValidator:
    """Builds a strict boolean field validator."""

    def validate(value: object, info: ValidationInfo) -> bool | None:
        if optional and value is None:
            return None
        if not isinstance(value, bool):
            name = info.field_name or "flag"
            fail("bool_type", f"{name} is a boolean, not {value!r}")
        return value

    return BeforeValidator(validate)


def _check_non_empty(value: object, name: str) -> str:
    """Validates `value` is a non-empty string."""
    if not isinstance(value, str):
        fail("string_type", f"{name} is a string, not {value!r}")
    if not value:
        fail("empty", f"{name} is non-empty, not ''")
    return value


def _text(*, optional: bool = False) -> BeforeValidator:
    """Builds a non-empty string field validator."""

    def validate(value: object, info: ValidationInfo) -> str | None:
        return (
            None
            if optional and value is None
            else _check_non_empty(value, info.field_name or "name")
        )

    return BeforeValidator(validate)


def _validate_reference(value: object, info: ValidationInfo) -> str | None:
    """Validates an optional reference net distinct from `output`."""
    if value is None:
        return None
    checked = _check_non_empty(value, "reference")
    if checked == info.data.get("output"):
        fail("same_node", f"the output `{checked}` is taken over itself")
    return checked


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
                "an integration method is 'trapezoidal', 'gear2' or "
                f"'backward_euler', not {value!r}"
            )
            error = PydanticCustomError("method", "{message}", {"message": message})
            raise error


def _check_quantity(
    value: object, dimension: Dimension, what: str, *, positive: bool = False
) -> Quantity:
    """Validates `value` is a finite `Quantity` of `dimension`.

    Raises:
        DimensionError: When `value` is a `Quantity` with another dimension.
        PydanticCustomError: When `value` is not a `Quantity`, non-finite, or
            non-positive when `positive` is true.
    """
    if not isinstance(value, Quantity):
        fail("quantity_type", f"{what} is a Quantity, not {value!r}")
    if value.dimension != dimension:
        message = f"{what} takes a quantity in {dimension}, not {value}"
        raise DimensionError(message)
    if not math.isfinite(value.value):
        fail("finite", f"{what} is finite, not {value}")
    if positive and value.value <= 0.0:
        fail("positive", f"{what} is positive, not {value}")
    return value


def _quantity(
    dimension: Dimension,
    what: str,
    *,
    positive: bool = False,
    optional: bool = False,
) -> BeforeValidator:
    """Builds a `Quantity` field validator enforcing `dimension` and finiteness."""

    def validate(value: object) -> Quantity | None:
        return (
            None
            if optional and value is None
            else _check_quantity(value, dimension, what, positive=positive)
        )

    return BeforeValidator(validate)


def _model(
    expected: type[object], label: str, *, optional: bool = False
) -> BeforeValidator:
    """Builds a validator accepting `expected`, a mapping, or None when `optional`."""

    def validate(value: object, info: ValidationInfo) -> object:
        if optional and value is None:
            return None
        if not isinstance(value, (expected, Mapping)):
            name = info.field_name or label
            fail("model_type", f"{name} is {label}, not {value!r}")
        return value

    return BeforeValidator(validate)


def _instance(
    expected: type[object] | tuple[type[object], ...],
    kind: LiteralString,
    message: str,
    *,
    optional: bool = False,
) -> BeforeValidator:
    """Builds a validator checking `value` is an instance of `expected`."""

    def validate(value: object) -> object:
        if optional and value is None:
            return None
        if not isinstance(value, expected):
            fail(kind, message.format(value=repr(value)))
        return value

    return BeforeValidator(validate)


def _as_sequence(
    value: object, not_sequence: str, *, string_error: str | None = None
) -> tuple[object, ...]:
    """Validates `value` is a non-string, non-bytes `Sequence`."""
    if isinstance(value, str) and string_error is not None:
        fail("sequence_type", string_error.format(value=repr(value)))
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        fail("sequence_type", not_sequence.format(value=repr(value)))
    return tuple(value)


def _collect_items[T](
    items: tuple[object, ...], title: str, check_item: Callable[[object], T]
) -> tuple[T, ...]:
    """Validates every entry of `items`, collecting all per-item errors."""
    errors: list[InitErrorDetails] = []
    checked: list[T] = []
    for index, item in enumerate(items):
        try:
            checked.append(check_item(item))
        except PydanticCustomError as error:
            errors.append(InitErrorDetails(type=error, loc=(index,), input=item))
    raise_if_errors(title, errors)
    return tuple(checked)


def _collect_mapping[T](
    value: object,
    title: str,
    key_noun: str,
    check_value: Callable[[str, object], T],
) -> dict[str, T]:
    """Validates a string-keyed mapping, collecting all key and entry errors."""
    if not isinstance(value, Mapping):
        fail("mapping_type", f"{title} is a mapping, not {value!r}")
    errors: list[InitErrorDetails] = []
    checked: dict[str, T] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            errors.append(
                detail(
                    "string_type",
                    f"{key_noun} is a string, not {key!r}",
                    key,
                    (repr(key),),
                )
            )
            continue
        try:
            checked[key] = check_value(key, item)
        except PydanticCustomError as error:
            errors.append(InitErrorDetails(type=error, loc=(key,), input=item))
        except ValidationError as error:
            errors.extend(
                detail("nested", entry["msg"], entry["input"], (key, *entry["loc"]))
                for entry in error.errors()
            )
    raise_if_errors(title, errors)
    return checked


def _voltage_map(title: str, what: str) -> BeforeValidator:
    """Builds a validator for a mapping of net names to finite voltage quantities."""

    def check_level(_net: str, level: object) -> Quantity:
        return _check_quantity(level, VOLTAGE, what)

    def validate(value: object) -> dict[str, Quantity]:
        return _collect_mapping(value, title, "a net name", check_level)

    return BeforeValidator(validate)


def _frequencies(empty_message: str, *, strictly_positive: bool) -> BeforeValidator:
    """Builds a validator for a non-empty sequence of finite frequency quantities."""

    def check_one(item: object) -> Quantity:
        frequency = _check_quantity(item, FREQUENCY, "a frequency")
        if strictly_positive and frequency.value <= 0.0:
            fail("positive", f"noise frequencies are positive, not {frequency}")
        if not strictly_positive and frequency.value < 0.0:
            fail("non_negative", f"frequencies are at least 0 Hz, not {frequency}")
        return frequency

    def validate(value: object) -> tuple[Quantity, ...]:
        items = _as_sequence(
            value, "frequencies is a sequence of quantities, not {value}"
        )
        if not items:
            fail("empty", empty_message)
        return _collect_items(items, "frequencies", check_one)

    return BeforeValidator(validate)


def _breakpoints(
    upper_field: str, upper_name: str, *, require_upper: bool = False
) -> BeforeValidator:
    """Builds a validator for breakpoint times within `(0, upper]`."""

    def validate(value: object, info: ValidationInfo) -> tuple[Quantity, ...]:
        items = _as_sequence(value, "breakpoints is a sequence of times, not {value}")
        upper = info.data.get(upper_field)
        if require_upper and upper_field in info.data and upper is None and items:
            message = (
                "an oscillator's period is found, not fixed, so its periodic steady "
                "state has no times to step onto"
            )
            fail("oscillator_breakpoints", message)
        limit = (
            upper.value if isinstance(upper, Quantity) and upper.value > 0.0 else None
        )

        def check_time(item: object) -> Quantity:
            time = _check_quantity(item, TIME, "a time to step onto")
            if not (time.value > 0.0 and (limit is None or time.value <= limit)):
                message = (
                    f"the times to step onto lie after 0 and at most {upper_name}, "
                    f"not {time}"
                )
                fail("breakpoint_range", message)
            return time

        return _collect_items(items, "breakpoints", check_time)

    return BeforeValidator(validate)


def _check_quantity_sequence(
    value: object,
    title: str,
    what: str,
    *,
    empty_message: str,
    expected: Dimension | None = None,
) -> tuple[Quantity, ...]:
    """Validates a non-empty sequence of finite quantities sharing one dimension."""
    items = _as_sequence(value, f"{title} is a sequence of quantities, not {{value}}")
    if not items:
        fail("empty", empty_message)
    dimension = expected

    def check_one(item: object) -> Quantity:
        nonlocal dimension
        if not isinstance(item, Quantity):
            fail("quantity_type", f"{what} is a Quantity, not {item!r}")
        if dimension is None:
            dimension = item.dimension
        return _check_quantity(item, dimension, what)

    return _collect_items(items, "values", check_one)


def _check_typed_sequence[T](
    value: object,
    expected: type[T],
    title: str,
    noun: str,
    *,
    empty_message: str | None = None,
) -> tuple[T, ...]:
    """Validates a sequence whose entries are instances of `expected`."""
    string_error = (
        f"{title} is a sequence of {noun}, such as [{{value}}], "
        "not the string {value}"
        if expected is str
        else None
    )
    items = _as_sequence(
        value,
        f"{title} is a sequence of {noun}, not {{value}}",
        string_error=string_error,
    )
    if empty_message is not None and not items:
        fail("empty", empty_message)

    def check_one(item: object) -> T:
        if not isinstance(item, expected):
            kind: LiteralString = "string_type" if expected is str else "item_type"
            fail(kind, f"{title} is a sequence of {noun}, not {item!r}")
        return item

    return _collect_items(items, title, check_one)


def _sequence_of[T](
    expected: type[T],
    title: str,
    noun: str,
    *,
    empty_message: str | None = None,
) -> BeforeValidator:
    """Builds a validator for a sequence whose items are instances of `expected`."""

    def validate(value: object) -> tuple[T, ...]:
        return _check_typed_sequence(
            value, expected, title, noun, empty_message=empty_message
        )

    return BeforeValidator(validate)


def _rising_grid(name: str, *, strictly_positive: bool) -> BeforeValidator:
    """Builds a validator for a strictly increasing sequence of finite floats."""
    above = "above zero" if strictly_positive else "from zero or more"

    def validate(value: object) -> tuple[float, ...]:
        raw = _as_sequence(value, f"the {name} is a sequence of numbers, not {{value}}")
        values = tuple(_check_float(item, f"the {name}") for item in raw)
        rising = all(a < b for a, b in itertools.pairwise(values))
        first = values[0] if values else math.nan
        starts = first > 0.0 if strictly_positive else first >= 0.0
        if not values or not rising or not starts:
            fail("grid_order", f"the {name} rise {above}, not {values}")
        return values

    return BeforeValidator(validate)


@stdlib_dataclass(frozen=True, slots=True)
class _MappingRule:
    """Describes key and value error messages for a validated mapping field."""

    title: str
    key_message: str
    type_message: str
    non_empty_message: str | None = None
    identifier_keys: bool = True


def _valid_mapping_key(key: object, *, identifier_keys: bool) -> bool:
    """Whether `key` satisfies a mapping rule's key constraint."""
    if not isinstance(key, str):
        return False
    return key.isidentifier() if identifier_keys else bool(key)


def _mapping(
    expected_type: type[object] | tuple[type[object], ...], rule: _MappingRule
) -> BeforeValidator:
    """Builds a validator for a string-keyed mapping, reporting every error at once."""
    key_kind: LiteralString = "identifier" if rule.identifier_keys else "string_type"

    def validate(value: object) -> dict[str, object]:
        if not isinstance(value, Mapping):
            fail("mapping_type", f"{rule.title} is a mapping, not {value!r}")
        if rule.non_empty_message is not None and not value:
            fail("empty", rule.non_empty_message)
        errors: list[InitErrorDetails] = []
        checked: dict[str, object] = {}
        for key, item in value.items():
            key_ok = _valid_mapping_key(key, identifier_keys=rule.identifier_keys)
            if not key_ok:
                message = rule.key_message.format(key=repr(key))
                errors.append(detail(key_kind, message, key, (str(key),)))
            if not isinstance(item, expected_type):
                message = rule.type_message.format(key=key, item=repr(item))
                errors.append(detail("value_type", message, item, (str(key),)))
            elif key_ok and isinstance(key, str):
                checked[key] = item
        raise_if_errors(rule.title, errors)
        return checked

    return BeforeValidator(validate)


def _check_random_variables(value: object) -> tuple[ir.RandomVariable, ...]:
    """Validates a sequence of `RandomVariable` objects with finite derivatives."""
    items = _as_sequence(
        value, "random_variables is a sequence of RandomVariable, not {value}"
    )
    errors: list[InitErrorDetails] = []
    checked: list[ir.RandomVariable] = []
    for index, variable in enumerate(items):
        if not isinstance(variable, ir.RandomVariable):
            errors.append(
                detail(
                    "random_variable_type",
                    f"random_variables holds RandomVariable objects, not {variable!r}",
                    variable,
                    (index,),
                )
            )
            continue
        checked.append(variable)
        errors.extend(
            detail(
                "finite",
                f"derivative of {path} {parameter} is finite, not {derivative!r}",
                derivative,
                (index, path, parameter),
            )
            for path, named in variable.derivatives.items()
            for parameter, derivative in named.items()
            if not math.isfinite(derivative)
        )
    raise_if_errors("random_variables", errors)
    return tuple(checked)


def _check_overrides(value: object) -> dict[str, dict[str, float]]:
    """Validates a nested mapping of instance paths to finite float overrides."""

    def check_parameter(name: str, number: object) -> float:
        return _check_float(number, name)

    def check_instance(path: str, named: object) -> dict[str, float]:
        return _collect_mapping(
            named, f"overrides for {path!r}", "a parameter name", check_parameter
        )

    return _collect_mapping(value, "overrides", "an instance path", check_instance)


def _check_sweep_values(value: object) -> tuple[Quantity, ...]:
    """Validates a non-empty sequence of finite same-dimension DC sweep levels."""
    return _check_quantity_sequence(
        value,
        "a DC sweep's values",
        "a sweep level",
        empty_message="a DC sweep has at least one level",
    )


def _check_worst_case_step(value: object) -> float:
    """Validates a worst-case probe step in `(0, 1]`."""
    converted = _check_float(value, "step")
    if not 0.0 < converted <= 1.0:
        message = f"a worst case moves a step above 0 and at most 1, not {converted!r}"
        fail("step_range", message)
    return converted


def _validate_axis_parameter(value: object, info: ValidationInfo) -> str | None:
    """Validates `Axis.parameter` is paired with `Axis.instance`."""
    checked = None if value is None else _check_non_empty(value, "parameter")
    if "instance" in info.data and (info.data["instance"] is None) != (checked is None):
        fail("axis_pairing", "an axis names an instance and its parameter, or neither")
    return checked


def _validate_axis_values(value: object, info: ValidationInfo) -> tuple[Quantity, ...]:
    """Validates `Axis.values`, enforcing temperature units for temperature axes."""
    is_temperature = (
        "instance" in info.data
        and "parameter" in info.data
        and info.data["instance"] is None
        and info.data["parameter"] is None
    )
    return _check_quantity_sequence(
        value,
        "an axis's values",
        "an axis's value",
        empty_message="an axis steps a value or more",
        expected=TEMPERATURE if is_temperature else None,
    )


def _check_record(record: object) -> tuple[str, ...] | None:
    """Validates `record` as `None` or a sequence of string patterns."""
    if record is None:
        return None
    return _check_typed_sequence(record, str, "a record", "instance paths")


def _check_product_part(value: object) -> Variants:
    """Validates one operand of a `Product` is a `Variants` instance."""
    if not isinstance(value, MonteCarlo | Grid | Explicit | Product):
        fail("variants_type", f"a product is of variants, not {value!r}")
    return value


_STRICT_BOOL: Final = _boolean()
_NON_EMPTY: Final = _text()
_OPTIONAL_NON_EMPTY: Final = _text(optional=True)
_REFERENCE: Final = BeforeValidator(_validate_reference)
_METHOD: Final = BeforeValidator(_check_method)
_RECORD: Final = BeforeValidator(_check_record)
_OPTIONAL_POSITIVE_FLOAT: Final = _real(positive=True, optional=True)
_OPTIONAL_POSITIVE_INT: Final = _integer(minimum=1, optional=True)
_OUTPUTS: Final = _sequence_of(str, "outputs", "unknown names")


def voltages(given: Mapping[str, Quantity], what: str) -> dict[str, float]:
    """The voltage of each net `given` names, in volts, each checked as `what`.

    Raises:
        DimensionError: When one is not a voltage.
    """
    return {net: voltage.to(VOLTAGE, what) for net, voltage in given.items()}


def recorded(record: Sequence[str] | None) -> tuple[str, ...] | None:
    """`record` as a tuple of patterns, or None when it is left out.

    Raises:
        ValidationError: When `record` is a string, or holds anything but strings.
    """
    try:
        return _check_record(record)
    except PydanticCustomError as error:
        title = "record"
        raise ValidationError.from_exception_data(
            title, [InitErrorDetails(type=error, loc=(), input=record)]
        ) from None


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Recorded:
    """An analysis whose results can keep the devices' detail beside the unknowns.

    Its `record` names the instances whose detail its results keep, as a plan's does,
    and replaces the plan's when set (see [`Plan`][jolty.plan.Plan]). With `regions`,
    its results keep besides each other transistor's region of operation and its
    margins to the regions' boundaries alone, which an operating point's `region` reads.
    """

    record: Annotated[Sequence[str] | None, _RECORD] = field(default=None, kw_only=True)
    regions: Annotated[bool, _STRICT_BOOL] = field(default=False, kw_only=True)


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

    max_solutions: Annotated[
        int,
        _integer(
            minimum=1,
            range_message="a search for operating points finds 1 or more, not {value}",
        ),
    ] = 8

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

    source: Annotated[str, _NON_EMPTY]
    values: Annotated[Sequence[Quantity], BeforeValidator(_check_sweep_values)]
    parameter: Annotated[str | None, _OPTIONAL_NON_EMPTY] = None

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

    source: Annotated[str, _NON_EMPTY]
    frequencies: Annotated[
        Sequence[Quantity],
        _frequencies(
            "a small-signal analysis has at least one frequency",
            strictly_positive=False,
        ),
    ]

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

    stop: Annotated[Quantity, _quantity(TIME, "a transient's stop time", positive=True)]
    max_step: Annotated[
        Quantity | None,
        _quantity(TIME, "a transient's longest step", positive=True, optional=True),
    ] = None
    method: Annotated[Method, _METHOD] = "trapezoidal"
    breakpoints: Annotated[
        Sequence[Quantity], _breakpoints("stop", "the stop time")
    ] = ()
    initial_conditions: Annotated[
        Mapping[str, Quantity],
        _voltage_map("initial_conditions", "an initial condition"),
    ] = field(default_factory=dict)
    operating_point: Annotated[bool, _STRICT_BOOL] = True

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

    period: Annotated[
        Quantity | None,
        _quantity(
            TIME, "a periodic steady state's period", positive=True, optional=True
        ),
    ] = None
    max_step: Annotated[
        Quantity | None,
        _quantity(TIME, "a period's longest step", positive=True, optional=True),
    ] = None
    method: Annotated[Method, _METHOD] = "trapezoidal"
    breakpoints: Annotated[
        Sequence[Quantity],
        _breakpoints("period", "the period", require_upper=True),
    ] = ()

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

_DIFFERENTIATED: Final = _instance(
    (DcSweep, Ac, Transient, PeriodicSteadyState),
    "analysis_type",
    "a differentiated analysis is a DcSweep, Ac, Transient, PeriodicSteadyState "
    "or None, not {value}",
    optional=True,
)


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

    outputs: Annotated[Sequence[str], _OUTPUTS]
    of: Annotated[Differentiated, _DIFFERENTIATED] = None

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

    outputs: Annotated[Sequence[str], _OUTPUTS]
    of: Annotated[Differentiated, _DIFFERENTIATED] = None
    random_variables: Annotated[
        Sequence[ir.RandomVariable], BeforeValidator(_check_random_variables)
    ] = ()

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

    probe: Annotated[str, _NON_EMPTY]
    frequencies: Annotated[
        Sequence[Quantity],
        _frequencies(
            "a stability analysis has at least one frequency", strictly_positive=False
        ),
    ]

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

    source: Annotated[str, _NON_EMPTY]
    output: Annotated[str, _NON_EMPTY]
    reference: Annotated[str | None, _REFERENCE] = None

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

    output: Annotated[str, _NON_EMPTY]
    frequencies: Annotated[
        Sequence[Quantity],
        _frequencies(
            "a noise analysis has at least one frequency", strictly_positive=True
        ),
    ]
    reference: Annotated[str | None, _REFERENCE] = None
    source: Annotated[str | None, _OPTIONAL_NON_EMPTY] = None

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

    source: Annotated[str | None, _OPTIONAL_NON_EMPTY] = None
    output: Annotated[str | None, _OPTIONAL_NON_EMPTY] = None
    reference: Annotated[str | None, _REFERENCE] = None

    @model_validator(mode="after")
    def _validate_pairing(self) -> PoleZero:
        both_set = self.source is not None and self.output is not None
        none_set = (
            self.source is None and self.output is None and self.reference is None
        )
        if not (both_set or none_set):
            fail(
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

    max_threads: Annotated[int | None, _OPTIONAL_POSITIVE_INT] = None
    min_solves_per_task: Annotated[int | None, _OPTIONAL_POSITIVE_INT] = None
    min_instances_per_task: Annotated[int | None, _OPTIONAL_POSITIVE_INT] = None
    min_unknowns_per_task: Annotated[int | None, _OPTIONAL_POSITIVE_INT] = None
    max_unknowns_per_interior: Annotated[int | None, _OPTIONAL_POSITIVE_INT] = None

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

    relative_tolerance: Annotated[float | None, _OPTIONAL_POSITIVE_FLOAT] = None
    voltage_tolerance: Annotated[float | None, _OPTIONAL_POSITIVE_FLOAT] = None
    current_tolerance: Annotated[float | None, _OPTIONAL_POSITIVE_FLOAT] = None
    charge_tolerance: Annotated[float | None, _OPTIONAL_POSITIVE_FLOAT] = None
    gmin: Annotated[float | None, _real(non_negative=True, optional=True)] = None
    max_dc_iterations: Annotated[int | None, _OPTIONAL_POSITIVE_INT] = None
    max_transient_iterations: Annotated[int | None, _OPTIONAL_POSITIVE_INT] = None
    truncation_tolerance: Annotated[float | None, _OPTIONAL_POSITIVE_FLOAT] = None
    max_memory: Annotated[float | None, _OPTIONAL_POSITIVE_FLOAT] = None
    profile: Annotated[bool | None, _boolean(optional=True)] = None
    """Whether a transient measures where its time goes, into its statistics' profile;
    it changes no other result."""
    parallelism: Annotated[Parallelism, _model(Parallelism, "a Parallelism")] = field(
        default_factory=Parallelism
    )
    nodesets: Annotated[
        Mapping[str, Quantity], _voltage_map("nodesets", "a nodeset")
    ] = field(default_factory=dict)
    """The voltage each net named, by path, is held at while an operating point is
    first solved from zero, then released and solved again from there, as SPICE's
    ``.nodeset``: a guess that leads it to one solution of several. A net that voltage
    sources and inductors join to the ground, or to a net named before it, is left to
    them."""

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

    temperature: Annotated[float, _real()]
    """The temperature, in kelvins."""
    process: Annotated[str | None, _OPTIONAL_NON_EMPTY] = None
    """The process corner of a foundry's devices, such as ``ss``, its kit's typical one
    unless set."""
    overrides: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    """The parameters that replace the circuit's, by instance path, then by parameter
    path, such as ``waveform.dc``, in SI base units."""
    mismatch: Annotated[int | None, _seed("mismatch", optional=True)] = None
    """The seed a foundry's devices draw their kit's mismatch from, if they draw it."""


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
                _check_non_empty(process, "process")
            except PydanticCustomError as error:
                errors.append(
                    InitErrorDetails(type=error, loc=("process",), input=process)
                )
        raise_if_errors("Corner", errors)
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

    samples: Annotated[
        int,
        _integer(
            minimum=1,
            range_message="a Monte Carlo analysis draws a sample or more, not {value}",
        ),
    ]
    seed: Annotated[int, _seed("a Monte Carlo seed")] = 0


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

    overrides: Annotated[
        Mapping[str, Mapping[str, float]], BeforeValidator(_check_overrides)
    ] = field(default_factory=dict)
    temperature: Annotated[
        Quantity | None,
        _quantity(TEMPERATURE, "a variant's temperature", optional=True),
    ] = None


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

    variants: Annotated[
        Sequence[Variant],
        _sequence_of(
            Variant,
            "explicit variants",
            "variants",
            empty_message="explicit variants list a run or more",
        ),
    ]


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

    instance: Annotated[str | None, _OPTIONAL_NON_EMPTY]
    parameter: Annotated[str | None, BeforeValidator(_validate_axis_parameter)]
    values: Annotated[Sequence[Quantity], BeforeValidator(_validate_axis_values)]

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

    axes: Annotated[Sequence[Axis], _sequence_of(Axis, "a grid's axes", "axes")]
    warm: Annotated[bool, _STRICT_BOOL] = False


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Product:
    """Every run of `first` at every run of `second`.

    Run `i` is run `i // len(second)` of `first`, then run `i % len(second)` of
    `second`, whose values replace those of `first` where both hold one, so corners,
    samples and a sweep make one batch. Its chains are those of `second`.
    """

    first: Annotated[Variants, BeforeValidator(_check_product_part)]
    second: Annotated[Variants, BeforeValidator(_check_product_part)]


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

    step: Annotated[float, BeforeValidator(_check_worst_case_step)] = 0.1


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

    supply: Annotated[
        Quantity, _quantity(VOLTAGE, "a characterization's supply", positive=True)
    ]
    slews: Annotated[Sequence[float], _rising_grid("slews", strictly_positive=True)] = (
        0.5,
        1.0,
        2.0,
        4.0,
    )
    fanouts: Annotated[
        Sequence[float], _rising_grid("fanouts", strictly_positive=False)
    ] = (0.0, 1.0, 2.0, 4.0)
    spot_checks: Annotated[
        int,
        _integer(minimum=0, range_message="spot_checks is zero or more, not {value}"),
    ] = 1
    tolerance: Annotated[float, _real(non_negative=True)] = 0.1
    seed: Annotated[int, _seed("a seed")] = 0


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

    analyses: Annotated[
        Mapping[str, Analysis],
        _mapping(
            _ANALYSIS_TYPES,
            _MappingRule(
                "analyses",
                "an analysis is named by a Python identifier, not {key}",
                "the analysis {key} is {item}, not an Analysis",
                "a plan runs at least one analysis",
            ),
        ),
    ]
    options: Annotated[Options, _model(Options, "an Options")] = field(
        default_factory=Options
    )
    temperature: Annotated[
        Quantity | None, _quantity(TEMPERATURE, "a plan's temperature", optional=True)
    ] = None
    corners: Annotated[
        Mapping[str, Corner],
        _mapping(
            Corner,
            _MappingRule(
                "corners",
                "a corner is named by a Python identifier, not {key}",
                "the corner {key} is {item}, not a Corner",
            ),
        ),
    ] = field(default_factory=dict)
    monte_carlo: Annotated[
        MonteCarlo | None, _model(MonteCarlo, "a MonteCarlo", optional=True)
    ] = None
    worst_case: Annotated[
        WorstCase | None, _model(WorstCase, "a WorstCase", optional=True)
    ] = None
    fidelity: Annotated[
        Mapping[str, Characterized],
        _mapping(
            Characterized,
            _MappingRule(
                "fidelity",
                "a fidelity scope is a non-empty string, not {key}",
                "the scope {key} takes {item}, not a Characterized model",
                identifier_keys=False,
            ),
        ),
    ] = field(default_factory=dict)
    process: Annotated[str | None, _OPTIONAL_NON_EMPTY] = None
    record: Annotated[Sequence[str] | None, _RECORD] = None
    variants: Annotated[
        Variants | None,
        _instance(
            (MonteCarlo, Grid, Explicit, Product),
            "variants_type",
            "a plan's variants are variants, not {value}",
            optional=True,
        ),
    ] = None

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

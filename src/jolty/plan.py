# Copyright 2026 Maxence Faldor
"""Simulation plans defining analyses, solver options, and operating conditions.

A plan is separate from the circuit: the circuit defines structure, while the plan
defines how to simulate the circuit. A plan runs its analyses at its nominal
temperature and repeats them at each corner, which may vary temperature, process
corner, or arguments passed to ``build()``. Plans can also evaluate Monte Carlo
tolerance samples and structured variant sweeps across nominal and corner conditions.

Variants parameterize multiple runs of one circuit as pure functions of the run index:
explicit lists ([`Explicit`][jolty.plan.Explicit]), parameter and temperature grids
([`Grid`][jolty.plan.Grid]), random tolerance draws
([`MonteCarlo`][jolty.plan.MonteCarlo]), and Cartesian combinations
([`Product`][jolty.plan.Product]). Plan fidelity maps instance scopes to logic models
characterized from the underlying component circuits.

Every plan object validates through Pydantic, collecting all invalid fields and
entries into a single [`ValidationError`][pydantic.ValidationError] consistent with
circuit and results serialization.

Examples:
    >>> from jolty.units import us
    >>> plan = Plan({"step": Transient(stop=10 * us)})
"""

__lazy_modules__ = ["decimal", "itertools", "jolty.spice._pss"]

import decimal
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass as stdlib_dataclass
from dataclasses import field, fields
from decimal import Decimal
from types import MappingProxyType
from typing import Annotated, Final, Literal, LiteralString, ReadOnly, TypedDict

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
from jolty.devices import ParameterValue
from jolty.errors import DimensionError, detail, fail, raise_if_errors
from jolty.spice import ngspice
from jolty.spice._pss import PeriodicSteadyStateCard
from jolty.units import (
    CHARGE,
    CONDUCTANCE,
    CURRENT,
    FREQUENCY,
    TEMPERATURE,
    TIME,
    VOLTAGE,
    Dimension,
    Quantity,
    finite,
    s,
)

DECADE_DIGITS: Final = 40
"""Decimal precision used when computing decade sweep frequencies before rounding.

Using 40 decimal digits (far beyond 64-bit float's 17 digits) prevents accumulated
rounding drift over large sweeps so each frequency rounds to the nearest float.
"""

LOOP_GAIN: Final = _core.LOOP_GAIN
"""Output name for loop gain `T` of a [`Stability`][jolty.plan.Stability] analysis."""

_MAX_SEED_EXCLUSIVE: Final = 2**64
"""Exclusive upper bound of a 64-bit unsigned integer seed."""

_CONFIG: Final = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
"""Shared Pydantic configuration for every plan dataclass."""

type Method = Literal["trapezoidal", "gear2", "backward_euler"]
"""Transient integration method: trapezoidal (default), Gear2 for stiff ringing
circuits, or backward Euler for maximum damping."""


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
    value: object,
    dimension: Dimension,
    what: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> Quantity:
    """Validates `value` is a finite `Quantity` of `dimension`.

    Raises:
        DimensionError: When `value` is a `Quantity` with another dimension.
        PydanticCustomError: When `value` is not a `Quantity`, non-finite, or
            out of range when `positive` or `non_negative` is true.
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
    if non_negative and value.value < 0.0:
        fail("non_negative", f"{what} is 0 or more, not {value}")
    return value


def _quantity(
    dimension: Dimension,
    what: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
    optional: bool = False,
) -> BeforeValidator:
    """Builds a `Quantity` field validator enforcing `dimension` and finiteness."""

    def validate(value: object) -> Quantity | None:
        return (
            None
            if optional and value is None
            else _check_quantity(
                value, dimension, what, positive=positive, non_negative=non_negative
            )
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
    """Key and value validation rules and error messages for a mapping field."""

    title: str
    key_message: str
    type_message: str
    non_empty_message: str | None = None
    identifier_keys: bool = True


def _valid_mapping_key(key: object, *, identifier_keys: bool) -> bool:
    """Returns whether `key` satisfies a mapping rule's key constraint."""
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


def _check_parameter_value(value: object, name: str) -> ParameterValue:
    """Validates `value` is a finite `Quantity` or plain real number (not bool)."""
    if isinstance(value, Quantity):
        if not math.isfinite(value.value):
            fail("finite", f"{name} is finite, not {value}")
        return value
    if not isinstance(value, bool) and isinstance(value, int | float):
        return _check_float(value, name)
    kind: LiteralString = "parameter_value"
    template: LiteralString = "{message}"
    message = f"{name} is a Quantity or real number, not {value!r}"
    raise PydanticCustomError(kind, template, {"message": message})


_UNSET: Final = object()
"""Sentinel distinguishing an uninitialized sequence element kind from ``None``."""


def _check_parameter_sequence(
    value: object,
    title: str,
    what: str,
    *,
    empty_message: str,
    expected: Dimension | None = None,
) -> tuple[ParameterValue, ...]:
    """Validates a non-empty sequence of uniform finite quantities or plain numbers."""
    if expected is not None:
        return _check_quantity_sequence(
            value, title, what, empty_message=empty_message, expected=expected
        )
    items = _as_sequence(
        value, f"{title} is a sequence of quantities or numbers, not {{value}}"
    )
    if not items:
        fail("empty", empty_message)
    first_kind: Dimension | object | None = _UNSET

    def check_one(item: object) -> ParameterValue:
        nonlocal first_kind
        checked = _check_parameter_value(item, what)
        if first_kind is _UNSET:
            first_kind = checked.dimension if isinstance(checked, Quantity) else None
            return checked
        if isinstance(first_kind, Dimension):
            if not isinstance(checked, Quantity):
                message = f"{what} takes a quantity in {first_kind}, not {checked!r}"
                raise DimensionError(message)
            return _check_quantity(checked, first_kind, what)
        if isinstance(checked, Quantity):
            message = f"{what} takes a plain number, not {checked}"
            raise DimensionError(message)
        return checked

    return _collect_items(items, "values", check_one)


def _check_overrides(value: object) -> dict[str, dict[str, ParameterValue]]:
    """Validates a nested mapping of instance paths to finite parameter values."""

    def check_parameter(name: str, entry: object) -> ParameterValue:
        return _check_parameter_value(entry, name)

    def check_instance(path: str, named: object) -> dict[str, ParameterValue]:
        return _collect_mapping(
            named, f"overrides for {path!r}", "a parameter name", check_parameter
        )

    return _collect_mapping(value, "overrides", "an instance path", check_instance)


def _check_sweep_values(value: object) -> tuple[ParameterValue, ...]:
    """Validates a non-empty sequence of finite uniform DC sweep levels."""
    return _check_parameter_sequence(
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


def _validate_axis_values(
    value: object, info: ValidationInfo
) -> tuple[ParameterValue, ...]:
    """Validates `Axis.values`, enforcing temperature units for temperature axes."""
    is_temperature = (
        "instance" in info.data
        and "parameter" in info.data
        and info.data["instance"] is None
        and info.data["parameter"] is None
    )
    return _check_parameter_sequence(
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
_OUTPUTS: Final = _sequence_of(str, "outputs", "output names")


def voltages(given: Mapping[str, Quantity], what: str) -> dict[str, float]:
    """Converts named net voltages in `given` to volts, validated as `what`.

    Args:
        given: Mapping from net path to voltage `Quantity`.
        what: Context label included in `DimensionError` messages.

    Returns:
        Dictionary mapping each net path to its scalar potential in volts.

    Raises:
        DimensionError: When any value in `given` does not have voltage dimension.
    """
    return {net: voltage.to(VOLTAGE, what) for net, voltage in given.items()}


def recorded(record: Sequence[str] | None) -> tuple[str, ...] | None:
    """Normalizes `record` to a tuple of patterns, or `None` when omitted.

    Args:
        record: Optional sequence of gitignore-style instance path patterns.

    Returns:
        Validated pattern tuple, or ``None`` when `record` is ``None``.

    Raises:
        ValidationError: When `record` is a string or contains non-string entries.
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
    """Base class for analyses that can record per-device details alongside unknowns.

    When set, `record` selects instance paths whose detailed currents, charges, and
    derivatives are retained, overriding the plan-level `record` setting (see
    [`Plan`][jolty.plan.Plan]). Setting `regions` additionally records each transistor's
    operating region and boundary margins for inspection on operating-point results.
    """

    record: Annotated[Sequence[str] | None, _RECORD] = field(default=None, kw_only=True)
    regions: Annotated[bool, _STRICT_BOOL] = field(default=False, kw_only=True)


@dataclass(frozen=True, slots=True, config=_CONFIG)
class OperatingPoint(Recorded):
    """DC operating-point analysis evaluated with every source at time zero."""

    def to_json(self) -> JsonValue:
        """Serializes the analysis into the simulation engine's JSON format.

        Returns:
            Engine JSON tag ``'operating_point'``.
        """
        return "operating_point"


@dataclass(frozen=True, slots=True, config=_CONFIG)
class OperatingPoints(Recorded):
    """DC multi-solution search finding up to `max_solutions` operating points.

    Starting from the initial operating point, deflated Newton iteration suppresses
    previously found solutions to locate additional equilibria until convergence stops.
    Multi-solution search reveals every stable and metastable state of a latch, as well
    as both the zero-current and active states of a self-biased reference.
    """

    max_solutions: Annotated[
        int,
        _integer(
            minimum=1,
            range_message="a search for operating points finds 1 or more, not {value}",
        ),
    ] = 8

    def to_json(self) -> JsonValue:
        """Serializes the analysis into the simulation engine's JSON format.

        Returns:
            Engine JSON mapping ``{'operating_points': {'max_solutions': ...}}``.
        """
        return {"operating_points": {"max_solutions": self.max_solutions}}


@dataclass(frozen=True, slots=True, config=_CONFIG)
class DcSweep(Recorded):
    """DC operating-point sweep over levels of a source or device parameter.

    The target `source` is specified by instance path, such as ``V1`` or ``bias/I1``.
    When `parameter` names a device parameter path such as ``resistance``, each sweep
    step rebuilds the device with that parameter set to the corresponding sweep value.
    """

    source: Annotated[str, _NON_EMPTY]
    values: Annotated[Sequence[ParameterValue], BeforeValidator(_check_sweep_values)]
    parameter: Annotated[str | None, _OPTIONAL_NON_EMPTY] = None

    @model_validator(mode="after")
    def _validate_source_levels(self) -> DcSweep:
        if self.parameter is None and any(
            not isinstance(item, Quantity) for item in self.values
        ):
            bad = next(item for item in self.values if not isinstance(item, Quantity))
            fail("quantity_type", f"a sweep level is a Quantity, not {bad!r}")
        return self

    def numbers(self, dimension: Dimension | None = None) -> list[float]:
        """Converts sweep values into SI base-unit floats, checked against `dimension`.

        When `dimension` is a [`Dimension`][jolty.units.Dimension], every level must
        be a quantity with that dimension. When `dimension` is ``None`` and `parameter`
        is set, levels must be plain dimensionless numbers; when `parameter` is
        ``None``, every level is converted in its own dimension.

        Args:
            dimension: Optional expected SI `Dimension` for each sweep level, or
                ``None`` for a dimensionless swept parameter.

        Returns:
            List of scalar sweep levels in SI base units.

        Raises:
            DimensionError: When a level does not match `dimension`.
        """
        if dimension is None and self.parameter is None:
            first = self.values[0]
            expected = first.dimension if isinstance(first, Quantity) else None
        else:
            expected = dimension
        return [
            _convert_sweep_value(value, expected, "a sweep level")
            for value in self.values
        ]

    def to_json(self, dimension: Dimension | None) -> JsonValue:
        """Serializes the DC sweep into engine JSON, validating against `dimension`.

        Args:
            dimension: Expected physical dimension of the swept source or parameter, or
                ``None`` for a dimensionless device parameter.

        Returns:
            Engine JSON mapping ``{'dc_sweep': ...}`` with SI sweep levels.
        """
        values: list[JsonValue] = list(self.numbers(dimension))
        sweep: dict[str, JsonValue] = {"source": self.source, "values": values}
        if self.parameter is not None:
            sweep["parameter"] = self.parameter
        return {"dc_sweep": sweep}


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Ac(Recorded):
    """Small-signal AC frequency response to a unit excitation of a source."""

    source: Annotated[str, _NON_EMPTY]
    frequencies: Annotated[
        Sequence[Quantity],
        _frequencies(
            "a small-signal analysis has at least one frequency",
            strictly_positive=False,
        ),
    ]

    def to_json(self) -> JsonValue:
        """Serializes the analysis into the simulation engine's JSON format.

        Returns:
            Engine JSON mapping ``{'ac': ...}`` with frequencies in hertz.
        """
        frequencies: list[JsonValue] = [
            frequency.to(FREQUENCY, "a frequency") for frequency in self.frequencies
        ]
        return {"ac": {"source": self.source, "frequencies": frequencies}}


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Transient(Recorded):
    """Time-domain transient response from time zero until `stop`.

    The integrator steps directly onto every time in `breakpoints` in addition to
    adaptively chosen time steps. The `initial_conditions` mapping fixes named nets
    (such as ``out`` or ``amp/mid``) to initial voltages during operating-point
    solution before releasing them, matching SPICE ``.ic``. Nets rigidly tied through
    voltage sources or inductors to ground or to an earlier net follow those
    constraints and match the stated voltage. Setting `operating_point` to `False`
    (matching SPICE ``UIC``) skips the DC solve and starts from `initial_conditions`
    with all other unknowns at zero, enabling startup or oscillator kick simulations.

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
        """Serializes the analysis into the simulation engine's JSON format.

        Returns:
            Engine JSON mapping ``{'transient': ...}`` with times in seconds.
        """
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
    """Single period of a circuit's periodic steady state computed via shooting.

    When `period` is given, shooting starts from the operating point for driven
    circuits sharing that source period. When `period` is `None`, the circuit is
    treated as an autonomous oscillator with DC sources, and shooting solves for the
    unknown period after a startup transient kicked along the fastest-growing mode.
    Time steps stay at or below `max_step` (defaulting to one fiftieth of the period)
    and hit every time listed in `breakpoints`.

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

        An autonomous card leaves `period` as `None` so shooting solves for the
        oscillator period, whereas a driven card sets `period` in seconds. When
        `max_step` is omitted on a driven card with `points` set, `max_step` defaults to
        ``period / points``.

        Args:
            card: Parsed SPICE ``.pss`` analysis card.

        Returns:
            Equivalent `PeriodicSteadyState` analysis specification.
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
        """Serializes the analysis into the simulation engine's JSON format.

        Returns:
            Engine JSON mapping ``{'periodic_steady_state': ...}`` in SI base units.
        """
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


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Stability:
    """Small-signal loop-gain stability analysis using Middlebrook injection probes.

    The `probe` names an independent voltage source placed in series with the feedback
    loop (such as ``VP``), with loop signal entering ``positive`` from ``negative`` and
    neither terminal tied to ground. Combining series voltage injection with shunt
    current injection from ground yields exact loop gain regardless of loading
    impedances (Middlebrook, 1975). Negative feedback produces positive low-frequency
    loop gain.
    """

    probe: Annotated[str, _NON_EMPTY]
    frequencies: Annotated[
        Sequence[Quantity],
        _frequencies(
            "a stability analysis has at least one frequency", strictly_positive=False
        ),
    ]

    def to_json(self) -> JsonValue:
        """Serializes the analysis into the simulation engine's JSON format.

        Returns:
            Engine JSON mapping ``{'stability': ...}`` with frequencies in hertz.
        """
        frequencies: list[JsonValue] = [
            frequency.to(FREQUENCY, "a frequency") for frequency in self.frequencies
        ]
        return {"stability": {"probe": self.probe, "frequencies": frequencies}}


type Differentiated = DcSweep | Ac | Stability | Transient | PeriodicSteadyState | None
"""Analysis type supported for differentiation, or `None` for the operating point."""

_DIFFERENTIATED: Final = _instance(
    (DcSweep, Ac, Stability, Transient, PeriodicSteadyState),
    "analysis_type",
    "a differentiated analysis is a DcSweep, Ac, Stability, Transient, "
    "PeriodicSteadyState or None, not {value}",
    optional=True,
)


def differentiated(of: Differentiated, level: Dimension | None) -> JsonValue:
    """Serializes the differentiated analysis `of` into the engine's JSON format.

    DC sweep levels are validated against `level`.

    Args:
        of: Underlying analysis to differentiate, or ``None`` for DC.
        level: Expected physical dimension when `of` is a `DcSweep`, or ``None`` for
            a dimensionless swept parameter.

    Returns:
        Engine JSON payload representing the differentiated target analysis.

    Examples:
        >>> differentiated(None, VOLTAGE)
        'operating_point'
    """
    match of:
        case None:
            return "operating_point"
        case DcSweep():
            return of.to_json(level)
        case Ac() | Stability() | Transient() | PeriodicSteadyState():
            return of.to_json()


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Sensitivity:
    """First-order sensitivity of analysis outputs with respect to device parameters.

    Each entry in `outputs` names a circuit unknown such as a node voltage (``out``) or
    branch current (``V1:current``), or [`LOOP_GAIN`][jolty.plan.LOOP_GAIN] when
    differentiating a [`Stability`][jolty.plan.Stability] analysis (the default when
    `outputs` is omitted for `Stability`). Derivatives are evaluated at the operating
    point when `of` is `None`, or at every point of a DC sweep, AC sweep, stability
    analysis, transient, or periodic steady state. For periodic steady states,
    derivatives are taken at fixed period fractions while adjusting the periodic initial
    state and oscillator period to preserve periodicity. The engine computes the full
    Jacobian using whichever of [`jvp`][jolty.simulate.jvp] (per parameter) or
    [`vjp`][jolty.simulate.vjp] (per output) requires fewer linear solves.
    """

    outputs: Annotated[Sequence[str], _OUTPUTS] = ()
    of: Annotated[Differentiated, _DIFFERENTIATED] = None

    @model_validator(mode="after")
    def _default_stability_output(self) -> Sensitivity:
        if not self.outputs and isinstance(self.of, Stability):
            object.__setattr__(self, "outputs", (LOOP_GAIN,))
        return self

    def to_json(self, level: Dimension | None = VOLTAGE) -> JsonValue:
        """Serializes the sensitivity analysis into engine JSON.

        When differentiating a DC sweep, sweep levels are validated against `level`.

        Args:
            level: Expected physical dimension when `self.of` is a `DcSweep`, or
                ``None`` for a dimensionless swept parameter.

        Returns:
            Engine JSON mapping ``{'sensitivity': ...}``.
        """
        outputs: list[JsonValue] = list(self.outputs)
        return {
            "sensitivity": {"outputs": outputs, "of": differentiated(self.of, level)}
        }


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Mismatch:
    """First-order statistical spread of analysis outputs under random variations.

    Each entry in `outputs` names an unknown (or [`LOOP_GAIN`][jolty.plan.LOOP_GAIN]
    when differentiating [`Stability`][jolty.plan.Stability], defaulted when omitted)
    evaluated at the operating point or across every point of the analysis `of`,
    matching [`Sensitivity`][jolty.plan.Sensitivity]. Variations include per-part and
    per-lot circuit tolerances (matching Monte Carlo draws) along with explicit
    [`RandomVariable`][jolty.ir.RandomVariable] inputs in `random_variables`, such as
    process-kit statistical models from
    [`Library.random_variables`][jolty.spice.Library.random_variables].

    Each random variable's output derivative sums the sensitivity contributions of its
    affected parameters. Independent random variables combine in quadrature: first-order
    output standard deviation equals the root-sum-square of each derivative scaled by
    its standard deviation, and each squared term gives that variable's variance share.
    """

    outputs: Annotated[Sequence[str], _OUTPUTS] = ()
    of: Annotated[Differentiated, _DIFFERENTIATED] = None
    random_variables: Annotated[
        Sequence[ir.RandomVariable], BeforeValidator(_check_random_variables)
    ] = ()

    @model_validator(mode="after")
    def _default_stability_output(self) -> Mismatch:
        if not self.outputs and isinstance(self.of, Stability):
            object.__setattr__(self, "outputs", (LOOP_GAIN,))
        return self

    def to_json(
        self, level: Dimension | None = VOLTAGE, toleranced: Sequence[JsonValue] = ()
    ) -> JsonValue:
        """Serializes the mismatch analysis and `toleranced` entries into engine JSON.

        When differentiating a DC sweep, sweep levels are validated against `level`.

        Args:
            level: Expected physical dimension when `self.of` is a `DcSweep`.
            toleranced: Serialized circuit tolerance descriptors from `variation`.

        Returns:
            Engine JSON mapping ``{'mismatch': ...}``.
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
class TransferFunction:
    """DC small-signal transfer function analysis matching SPICE ``.tf``.

    Evaluates the linearized transfer relationship from an independent `source` (such
    as ``V1``) to an `output`, which may be a node voltage (relative to ground or
    `reference`) or a branch current (such as ``V2:current``). Results report the DC
    gain, input resistance seen by `source`, and output resistance with `source` set
    to zero (shorted voltage source or open current source).

    Examples:
        >>> TransferFunction("V1", "out").to_json()
        {'transfer_function': {'source': 'V1', 'output': 'out'}}
    """

    source: Annotated[str, _NON_EMPTY]
    output: Annotated[str, _NON_EMPTY]
    reference: Annotated[str | None, _REFERENCE] = None

    def to_json(self) -> JsonValue:
        """Serializes the analysis into the simulation engine's JSON format.

        Returns:
            Engine JSON mapping ``{'transfer_function': ...}``.
        """
        named: dict[str, JsonValue] = {"source": self.source, "output": self.output}
        if self.reference is not None:
            named["reference"] = self.reference
        return {"transfer_function": named}


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Noise:
    """Small-signal noise spectral density at an output across positive frequencies.

    The `output` names a circuit unknown (voltage or current), or a differential node
    voltage when `reference` is provided. Resistors contribute thermal noise, junctions
    contribute shot noise, and diodes and transistors with flicker coefficients
    contribute flicker noise per standard SPICE models. When an independent `source` is
    specified, the transfer gain refers output noise back to an equivalent input noise.
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
        """Serializes the analysis into the simulation engine's JSON format.

        Returns:
            Engine JSON mapping ``{'noise': ...}`` with frequencies in hertz.
        """
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
    """Poles of the linearized circuit and optional zeros of a transfer function.

    The circuit is linearized at its operating point to compute natural pole
    frequencies where the homogeneous small-signal system has nontrivial solutions.
    Setting both `source` and `output` (optionally over `reference`) additionally
    computes transfer-function zeros.
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
        """Serializes the analysis into the simulation engine's JSON format.

        Returns:
            Engine JSON mapping ``{'pole_zero': ...}``.
        """
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
"""Union of all supported simulation analysis types."""

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
    """Multithreaded execution thresholds for distributing work within a single run.

    Each workload splits into parallel tasks of at least its configured minimum size
    once enough work exists to occupy every available thread; smaller workloads run on
    the calling thread. Small-signal frequency sweeps and sensitivity right-hand sides
    partition by `min_solves_per_task`. Nonlinear device groups partition by
    `min_instances_per_task`, while linear devices evaluate once per run. Newton
    iteration and transient step control partition unknowns and matrix entries by
    `min_unknowns_per_task` in dynamic chunks of 1,024. Matrix blocks exceeding
    `max_unknowns_per_interior` use nested dissection with parallel separator
    factorization whenever elimination requires fewer multiply-adds than minimum-degree
    ordering.

    A single run uses at most `max_threads` worker threads, defaulting to the global
    [`threads`][jolty.simulate.threads] limit; batches run each simulation on one thread
    whenever runs saturate available threads. Any omitted threshold takes its measured
    thread-count break-even from [`BREAK_EVENS`][jolty.plan.BREAK_EVENS]. All settings
    except `max_unknowns_per_interior` produce bit-identical results regardless of
    partitioning. Changing `max_unknowns_per_interior` reorders factorization
    operations, so results agree within floating-point rounding.

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
        """Serializes configured parallelism overrides into JSON, omitting defaults.

        Returns:
            Dictionary of explicitly set parallelism thresholds for the engine.
        """
        values = {setting.name: getattr(self, setting.name) for setting in fields(self)}
        return {name: value for name, value in values.items() if value is not None}


class OptionDefaults(TypedDict):
    """Read-only engine defaults for numerical `Options` fields."""

    relative_tolerance: ReadOnly[float]
    voltage_tolerance: ReadOnly[Quantity]
    current_tolerance: ReadOnly[Quantity]
    charge_tolerance: ReadOnly[Quantity]
    gmin: ReadOnly[Quantity]
    max_dc_iterations: ReadOnly[int]
    max_transient_iterations: ReadOnly[int]
    truncation_tolerance: ReadOnly[float]
    max_memory: ReadOnly[float]


_OPTION_DIMENSIONS: Final[Mapping[str, Dimension]] = {
    "voltage_tolerance": VOLTAGE,
    "current_tolerance": CURRENT,
    "charge_tolerance": CHARGE,
    "gmin": CONDUCTANCE,
}
"""Physical SI dimensions of dimensioned [`Options`][jolty.plan.Options] fields."""


def _load_defaults() -> OptionDefaults:
    """Loads engine defaults from `_core`, wrapping dimensioned fields as quantities."""
    decoded = json.loads(_core.default_options())
    if not isinstance(decoded, dict):
        message = f"engine defaults is a mapping, not {decoded!r}"
        raise TypeError(message)

    def number(name: str) -> float:
        entry = decoded[name]
        if isinstance(entry, bool) or not isinstance(entry, int | float):
            message = f"engine default `{name}` is numeric, not {entry!r}"
            raise TypeError(message)
        return float(entry)

    def count(name: str) -> int:
        entry = decoded[name]
        if isinstance(entry, bool) or not isinstance(entry, int):
            message = f"engine default `{name}` is an integer, not {entry!r}"
            raise TypeError(message)
        return entry

    return OptionDefaults(
        relative_tolerance=number("relative_tolerance"),
        voltage_tolerance=Quantity(number("voltage_tolerance"), VOLTAGE),
        current_tolerance=Quantity(number("current_tolerance"), CURRENT),
        charge_tolerance=Quantity(number("charge_tolerance"), CHARGE),
        gmin=Quantity(number("gmin"), CONDUCTANCE),
        max_dc_iterations=count("max_dc_iterations"),
        max_transient_iterations=count("max_transient_iterations"),
        truncation_tolerance=number("truncation_tolerance"),
        max_memory=number("max_memory"),
    )


DEFAULTS: Final[OptionDefaults] = _load_defaults()
"""Engine default values for numerical fields of [`Options`][jolty.plan.Options].

Dimensioned tolerances and `gmin` are [`Quantity`][jolty.units.Quantity] values, while
dimensionless options and counts are plain numbers. Defaults for
[`Parallelism`][jolty.plan.Parallelism] thresholds are listed separately in
[`BREAK_EVENS`][jolty.plan.BREAK_EVENS].

Examples:
    >>> str(DEFAULTS["charge_tolerance"])
    '10 fC'
"""

BREAK_EVENS: Final[Mapping[str, tuple[int | None, ...]]] = MappingProxyType(
    {name: tuple(values) for name, values in json.loads(_core.break_evens()).items()}
)
"""Measured break-even thresholds for [`Parallelism`][jolty.plan.Parallelism] fields.

The ``"threads"`` entry lists measured thread counts, and each threshold name maps to
the corresponding break-even sizes (`None` where parallelism never paid off). A run
selects the entry for the highest measured thread count not exceeding its limit.

Examples:
    >>> BREAK_EVENS["threads"]
    (2, 4, 8, 64)
"""


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Options:
    """Numerical solver options, defaulting omitted fields to SPICE engine defaults.

    Dimensioned fields (`voltage_tolerance`, `current_tolerance`, `charge_tolerance`,
    and `gmin`) take [`Quantity`][jolty.units.Quantity] values in volts, amperes,
    coulombs, and siemens respectively. The `parallelism` field configures thread
    distribution without altering numerical results beyond rounding differences from
    dissected factorization order.

    Examples:
        >>> from jolty.units import pS
        >>> Options(gmin=1e-3 * pS).to_json()
        {'gmin': 1e-15}
        >>> Options(parallelism=Parallelism(min_solves_per_task=16)).to_json()
        {'parallelism': {'min_solves_per_task': 16}}
    """

    relative_tolerance: Annotated[float | None, _OPTIONAL_POSITIVE_FLOAT] = None
    voltage_tolerance: Annotated[
        Quantity | None,
        _quantity(VOLTAGE, "voltage_tolerance", positive=True, optional=True),
    ] = None
    current_tolerance: Annotated[
        Quantity | None,
        _quantity(CURRENT, "current_tolerance", positive=True, optional=True),
    ] = None
    charge_tolerance: Annotated[
        Quantity | None,
        _quantity(CHARGE, "charge_tolerance", positive=True, optional=True),
    ] = None
    gmin: Annotated[
        Quantity | None,
        _quantity(CONDUCTANCE, "gmin", non_negative=True, optional=True),
    ] = None
    max_dc_iterations: Annotated[int | None, _OPTIONAL_POSITIVE_INT] = None
    max_transient_iterations: Annotated[int | None, _OPTIONAL_POSITIVE_INT] = None
    truncation_tolerance: Annotated[float | None, _OPTIONAL_POSITIVE_FLOAT] = None
    max_memory: Annotated[float | None, _OPTIONAL_POSITIVE_FLOAT] = None
    profile: Annotated[bool | None, _boolean(optional=True)] = None
    """Flag enabling transient timing profiles without altering simulation results."""
    parallelism: Annotated[Parallelism, _model(Parallelism, "a Parallelism")] = field(
        default_factory=Parallelism
    )
    nodesets: Annotated[
        Mapping[str, Quantity], _voltage_map("nodesets", "a nodeset")
    ] = field(default_factory=dict)
    """Initial voltage guesses keyed by net path for operating-point convergence.

    Matches SPICE ``.nodeset``: named nets are held at their target voltages during an
    initial DC solve from zero, then released for the final solve. Nets rigidly tied by
    voltage sources or inductors to ground or an earlier net follow those constraints.
    """

    def to_json(self) -> dict[str, JsonValue]:
        """Serializes configured solver options into engine JSON, omitting defaults.

        Returns:
            Dictionary of explicitly configured solver options in SI base units.
        """
        values: dict[str, JsonValue] = {}
        for setting in fields(self):
            if setting.name in {"parallelism", "nodesets"}:
                continue
            entry = getattr(self, setting.name)
            if entry is None:
                continue
            dimension = _OPTION_DIMENSIONS.get(setting.name)
            values[setting.name] = (
                entry.to(dimension, setting.name) if dimension is not None else entry
            )
        spread = self.parallelism.to_json()
        if spread:
            values["parallelism"] = spread
        if self.nodesets:
            values["nodesets"] = dict[str, JsonValue](
                voltages(self.nodesets, "a nodeset")
            )
        return values


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Conditions:
    """Operating conditions applied during a single circuit simulation run.

    Monte Carlo samples and plan corners each execute with dedicated condition sets.
    """

    temperature: Annotated[
        Quantity, _quantity(TEMPERATURE, "a condition's temperature")
    ]
    """Simulation temperature as a temperature `Quantity`."""
    process: Annotated[str | None, _OPTIONAL_NON_EMPTY] = None
    """Foundry process corner such as ``ss``, defaulting to the kit typical corner."""
    overrides: Annotated[
        Mapping[str, Mapping[str, ParameterValue]], BeforeValidator(_check_overrides)
    ] = field(default_factory=dict)
    """Parameter overrides keyed by instance path and parameter path."""
    mismatch: Annotated[int | None, _seed("mismatch", optional=True)] = None
    """Optional 64-bit seed controlling foundry device mismatch draws."""


@stdlib_dataclass(frozen=True, slots=True, init=False)
class Corner:
    """Alternative operating condition evaluated alongside nominal plan conditions.

    A corner may override temperature, foundry process corner, and keyword arguments
    passed to ``build()`` to rebuild the circuit (for example, at a lower supply).

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
        """Initializes a corner at `temperature` and `process` with build `arguments`.

        Raises:
            DimensionError: When `temperature` does not have temperature dimension.
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
    """Random tolerance sampling specification across all plan analyses and corners.

    Each sample executes at nominal plan conditions and at every corner, reusing the
    same drawn device values whenever a corner shares identical toleranced parameters.
    Requirement verdicts for a sample take the worst outcome across all conditions. Each
    sample index `i` derives deterministically from `seed` and `i` alone, matching run
    index `i` when used as variants.
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
    """Parameter overrides and optional temperature for a single batch run.

    Parameter overrides map instance paths to parameter paths (such as ``waveform.dc``)
    with [`Quantity`][jolty.units.Quantity] values for dimensioned parameters and plain
    numbers for dimensionless parameters. When `temperature` is unset, the run uses the
    simulation temperature.

    Examples:
        >>> from jolty.units import celsius, kohm
        >>> Variant({"R1": {"resistance": 2 * kohm}}, celsius(85)).overrides["R1"][
        ...     "resistance"
        ... ] == 2 * kohm
        True
    """

    overrides: Annotated[
        Mapping[str, Mapping[str, ParameterValue]], BeforeValidator(_check_overrides)
    ] = field(default_factory=dict)
    temperature: Annotated[
        Quantity | None,
        _quantity(TEMPERATURE, "a variant's temperature", optional=True),
    ] = None


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Explicit:
    """Explicit sequence of variant runs, where run `i` uses variant `i`.

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
    """Single sweep axis of a grid defining the swept target and ordered values.

    An axis sweeps either a device parameter (identified by `instance` and `parameter`
    paths, validated against the target device's declared parameter dimension before
    simulation) or temperature when both `instance` and `parameter` are `None`.

    Examples:
        >>> from jolty.units import celsius, kohm
        >>> Axis("R1", "resistance", [1 * kohm, 2 * kohm]).numbers()
        [1000.0, 2000.0]
        >>> Axis.temperature([celsius(-40), celsius(125)]).numbers()
        [233.14999999999998, 398.15]
    """

    instance: Annotated[str | None, _OPTIONAL_NON_EMPTY]
    parameter: Annotated[str | None, BeforeValidator(_validate_axis_parameter)]
    values: Annotated[Sequence[ParameterValue], BeforeValidator(_validate_axis_values)]

    @classmethod
    def temperature(cls, values: Sequence[Quantity]) -> Axis:
        """Builds a temperature sweep `Axis` across `values`.

        Args:
            values: Non-empty sequence of temperature `Quantity` sweep points.

        Returns:
            Temperature sweep `Axis` with `instance` and `parameter` set to ``None``.
        """
        return cls(None, None, values)

    def numbers(self) -> list[float]:
        """Converts axis values into SI base-unit floats, using kelvins for temperature.

        Returns:
            List of scalar axis values converted to SI base units.

        Raises:
            DimensionError: When any value does not match the axis dimension.
        """
        if self.instance is None:
            expected: Dimension | None = TEMPERATURE
        else:
            first = self.values[0]
            expected = first.dimension if isinstance(first, Quantity) else None
        return [
            _convert_sweep_value(value, expected, "an axis's value")
            for value in self.values
        ]


def _convert_sweep_value(
    value: ParameterValue, dimension: Dimension | None, what: str
) -> float:
    """Converts one axis or sweep `value` to SI base units matching `dimension`."""
    if dimension is not None:
        if not isinstance(value, Quantity):
            message = f"{what} takes a quantity in {dimension}, not {value!r}"
            raise DimensionError(message)
        return value.to(dimension, what)
    if isinstance(value, Quantity):
        message = f"{what} takes a plain number, not {value}"
        raise DimensionError(message)
    return finite(value, what)


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Grid:
    """Cartesian product of parameter and temperature axes varying last axis fastest.

    Run index `i` selects values from the last axis backward using mixed-radix indexing.
    When `warm` is true, each line along the last axis forms a warm-start chain where
    every run starts from the preceding operating point, mirroring a DC sweep.

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
    """Cartesian product combining every run of `first` with every run of `second`.

    Run `i` pairs run ``i // len(second)`` of `first` with run ``i % len(second)`` of
    `second`, letting `second` override overlapping entries so corners, samples, and
    sweeps compose into one batch. Warm-start chains follow those of `second`.
    """

    first: Annotated[Variants, BeforeValidator(_check_product_part)]
    second: Annotated[Variants, BeforeValidator(_check_product_part)]


type Variants = MonteCarlo | Grid | Explicit | Product
"""Union of variant specifications mapping run indices to parameter overrides."""

# @pydantic.dataclasses.dataclass attaches __pydantic_complete__ at runtime.
rebuild_dataclass(Product)  # ty: ignore[invalid-argument-type]


def runs(variants: Variants) -> int:
    """Computes the total number of runs generated by `variants`.

    Args:
        variants: Variant specification (`MonteCarlo`, `Grid`, `Explicit`, `Product`).

    Returns:
        Total number of simulation runs defined by `variants`.

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
    """Tolerance-corner analysis evaluating each requirement at worsening extremes.

    At nominal conditions and every corner, one perturbed run per toleranced parameter
    (offset by `step` of its tolerance) determines each requirement's sensitivity sign.
    For every requirement bound, each parameter and shared lot spread moves to the
    extreme pushing the requirement toward that bound, following PSpice worst-case
    conventions. Normal tolerances use three standard deviations as their extremes.
    """

    step: Annotated[float, BeforeValidator(_check_worst_case_step)] = 0.1


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Characterized:
    """Logic timing model characterized from a component circuit at `supply`.

    The target component exposes input ports, output ports, one supply, and one ground.
    Characterization sweeps input transitions (`slews` times nominal output slew) and
    capacitive loads (`fanouts` times mean input capacitance). On each run,
    `spot_checks` matched instances selected using `seed` re-simulate at transistor
    level and validate output transitions against measured error plus `tolerance` times
    mean delay.

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
"""Strict solver options matching the tolerances configured for ngspice comparisons.

Matches [`jolty.spice.ngspice`][jolty.spice.ngspice] relative, voltage, and current
tolerances while keeping the shared 1e-14 C default charge tolerance (at least 1e-20 C
absolute accuracy).
"""


@dataclass(frozen=True, slots=True, config=_CONFIG)
class Plan:
    """Named analyses, shared solver options, and operating conditions for a circuit.

    Each analysis uses a Python identifier key (such as ``step`` or ``bode``) that also
    names its results. Analyses run at `temperature` (defaulting to 27 °C) and `process`
    (defaulting to the foundry kit typical corner), then repeat across all named
    `corners`. Optional `monte_carlo`, `variants`, and `worst_case` configurations also
    execute across nominal and corner conditions, running one batch per distinct
    circuit.

    The `fidelity` mapping binds instance path scopes (such as ``adder/*``, where ``*``
    matches one hierarchy level and ``**`` spans levels) to characterized logic models.
    Nested instances inside a matched scope inherit the outer scope's model.

    Every analysis records circuit unknowns, while `record` selects additional devices
    using gitignore-style patterns where ``!`` excludes earlier matches and the last
    match wins. For example, ``["X1/**", "!X1/bias/**"]`` records ``X1`` excluding its
    bias network, ``["**"]`` records every instance, and ``[]`` or `None` records none.
    Recorded primitive devices store terminal currents, charges, derivatives, and state
    variables, while recorded components and logic devices sum terminal currents and
    charges across enclosed devices. Any [`Recorded`][jolty.plan.Recorded] analysis
    specifying a local `record` overrides the plan setting; operating points and
    transients always record current- or power-rated instances for rating checks.
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
        """Returns the active process corner for nominal execution or for `corner`.

        Returns `None` to indicate each foundry kit's typical process corner.

        Args:
            corner: Optional `Corner` overriding the nominal process corner.

        Returns:
            Active foundry process corner name, or ``None`` for the kit typical corner.
        """
        if corner is not None and corner.process is not None:
            return corner.process
        return self.process

    def temperature_at(self, corner: Corner | None = None) -> Quantity | None:
        """Returns the active temperature for nominal execution or for `corner`.

        Returns `None` to indicate the default 27 °C simulation temperature.

        Args:
            corner: Optional `Corner` overriding the nominal plan temperature.

        Returns:
            Active simulation temperature `Quantity`, or ``None`` for default 27 °C.
        """
        if corner is not None and corner.temperature is not None:
            return corner.temperature
        return self.temperature


def decade(start: Quantity, stop: Quantity, per_decade: int) -> list[Quantity]:
    """Generates logarithmically spaced frequencies from `start` to `stop`.

    Spans `per_decade` points per decade using `start · r^k` with
    `r = (stop / start)^(1/n)`. Steps compute sequentially in `DECADE_DIGITS`-digit
    decimal arithmetic before rounding to the nearest float so results match across
    platforms regardless of native `log10` or `pow` rounding.

    Args:
        start: Positive lower frequency bound in hertz.
        stop: Upper frequency bound strictly above `start`.
        per_decade: Target number of logarithmically spaced steps per decade.

    Returns:
        Ascending list of frequency `Quantity` points from `start` to `stop`.

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
    """Generates `points` evenly spaced quantities from `start` to `stop` inclusive.

    Args:
        start: Starting `Quantity` at index 0.
        stop: Ending `Quantity` of matching dimension at index `points - 1`.
        points: Total number of evenly spaced points (at least 2).

    Returns:
        List of `points` evenly spaced `Quantity` values from `start` to `stop`.

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

# Copyright 2026 Maxence Faldor
"""Simulation plans: the analyses to run on a circuit, how, and in which conditions.

A plan is kept apart from the circuit: the circuit says what exists, the plan how to
simulate it. A plan runs its analyses at its temperature, then again at each of its
corners: another temperature, or other arguments to the build script's ``build()``, and
on Monte Carlo samples of the circuit's tolerances, each at its own conditions and at
every corner. Its fidelity chooses, for the
component instances its scopes match, logic models characterized from their circuits.

Examples:
    >>> from jolty.units import us
    >>> plan = Plan({"step": Transient(stop=10 * us)})
"""

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Literal

from pydantic import JsonValue

from jolty.units import FREQUENCY, TEMPERATURE, TIME, VOLTAGE, Dimension, Quantity

type Method = Literal["trapezoidal", "gear2", "backward_euler"]
"""An integration method: trapezoidal by default, Gear2 for stiff circuits that ring,
backward Euler for the most damping."""


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """The DC operating point, with every source at its level at time zero."""

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        return "operating_point"


@dataclass(frozen=True, slots=True)
class DcSweep:
    """The operating point at each level of one independent source.

    The source is named by its path, such as ``V1`` or ``bias/I1``.
    """

    source: str
    levels: Sequence[Quantity]

    def to_json(self, level: Dimension) -> JsonValue:
        """The analysis in the engine's format, its levels checked against `level`."""
        levels: list[JsonValue] = [
            value.to(level, "a sweep level") for value in self.levels
        ]
        return {"dc_sweep": {"source": self.source, "levels": levels}}


@dataclass(frozen=True, slots=True)
class Ac:
    """The small-signal response to a unit excitation of a source, at each frequency."""

    source: str
    frequencies: Sequence[Quantity]

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        frequencies: list[JsonValue] = [
            frequency.to(FREQUENCY, "a frequency") for frequency in self.frequencies
        ]
        return {"ac": {"source": self.source, "frequencies": frequencies}}


@dataclass(frozen=True, slots=True)
class Transient:
    """The response over time, from the operating point at zero until `stop`.

    The analysis steps onto each of `times` exactly, besides the times it chooses.
    """

    stop: Quantity
    max_step: Quantity | None = None
    method: Method = "trapezoidal"
    times: Sequence[Quantity] = ()

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        transient: dict[str, JsonValue] = {
            "stop": self.stop.to(TIME, "a transient's stop time"),
            "method": self.method,
        }
        if self.times:
            transient["times"] = [
                time.to(TIME, "a time to step onto") for time in self.times
            ]
        if self.max_step is not None:
            transient["max_step"] = self.max_step.to(TIME, "a transient's longest step")
        return {"transient": transient}


@dataclass(frozen=True, slots=True)
class Sensitivity:
    """How unknowns move with every number of every device's parameters.

    Each output is an unknown by name, a voltage such as ``out`` or a current such as
    ``V1:current``. The derivatives are taken at the operating point, or at each point
    of the analysis `of`: each level of a DC sweep and each frequency of a small-signal
    response, where one solve per output serves every parameter, however many, and each
    time of a transient, which carries each parameter's derivatives along its steps.
    """

    outputs: Sequence[str]
    of: DcSweep | Ac | Transient | None = None

    def to_json(self, level: Dimension = VOLTAGE) -> JsonValue:
        """The analysis in the engine's format.

        The levels of a DC sweep it differentiates are checked against `level`.
        """
        outputs: list[JsonValue] = list(self.outputs)
        match self.of:
            case None:
                of: JsonValue = "operating_point"
            case DcSweep():
                of = self.of.to_json(level)
            case Ac() | Transient():
                of = self.of.to_json()
        return {"sensitivity": {"outputs": outputs, "of": of}}


@dataclass(frozen=True, slots=True)
class Stability:
    """The loop gain through a probe at each frequency, by Middlebrook's injections.

    The probe is an independent voltage source in series with the loop, named by its
    path, such as ``VP``: the loop's signal enters its ``positive`` terminal from its
    ``negative`` one, and neither terminal is the ground, through which the injected
    current returns. One voltage injected in series and one current injected from the
    ground give the loop gain however the two sides load each other (Middlebrook, 1975).
    Negative feedback makes the loop gain positive at low frequencies.
    """

    probe: str
    frequencies: Sequence[Quantity]

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        frequencies: list[JsonValue] = [
            frequency.to(FREQUENCY, "a frequency") for frequency in self.frequencies
        ]
        return {"stability": {"probe": self.probe, "frequencies": frequencies}}


@dataclass(frozen=True, slots=True)
class Noise:
    """The noise density at an output at each frequency, from every device's noise.

    The output is an unknown by name, a voltage such as ``out`` or a current such as
    ``V1:current``, or a voltage over a `reference` node, for a differential output.
    Resistances make thermal noise, junctions shot noise, and diodes and transistors
    with a flicker coefficient flicker noise, as SPICE models them. Given a `source`,
    an independent source such as ``V1``, the gain from its level to the output refers
    the noise back to it, as an input's noise.
    """

    output: str
    frequencies: Sequence[Quantity]
    reference: str | None = None
    source: str | None = None

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


@dataclass(frozen=True, slots=True)
class PoleZero:
    """The poles of the linearized circuit, and the zeros of a transfer function.

    The circuit is linearized at its operating point, and the transfer function is from
    a `source` to an `output`, given both. The poles are the circuit's natural
    frequencies, the complex frequencies at which its small-signal equations have a
    solution without any excitation. The output is an unknown by name, or a voltage
    over a `reference` node.
    """

    source: str | None = None
    output: str | None = None
    reference: str | None = None

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        found = {
            "source": self.source,
            "output": self.output,
            "reference": self.reference,
        }
        given: dict[str, JsonValue] = {
            key: value for key, value in found.items() if value is not None
        }
        return {"pole_zero": given}


type Analysis = (
    OperatingPoint
    | DcSweep
    | Ac
    | Transient
    | Sensitivity
    | Stability
    | Noise
    | PoleZero
)
"""Any analysis."""


@dataclass(frozen=True, slots=True)
class Options:
    """Numerical settings; each one left out takes the engine's default, SPICE's own.

    Tolerances are in SI base units: volts, amperes and coulombs.
    """

    relative_tolerance: float | None = None
    voltage_tolerance: float | None = None
    current_tolerance: float | None = None
    charge_tolerance: float | None = None
    gmin: float | None = None
    dc_iterations: int | None = None
    transient_iterations: int | None = None
    truncation_tolerance: float | None = None

    def to_json(self) -> JsonValue:
        """The options in the engine's format, without those left to its defaults."""
        values = {setting.name: getattr(self, setting.name) for setting in fields(self)}
        return {name: value for name, value in values.items() if value is not None}


@dataclass(frozen=True, slots=True, init=False)
class Corner:
    """Conditions a plan also runs its analyses at.

    A corner gives a temperature, the plan's unless given, and arguments to the build
    script's ``build()``, which builds the circuit again with them, such as a lower
    supply voltage.

    Examples:
        >>> from jolty.units import V, celsius
        >>> corner = Corner(temperature=celsius(-40), supply=3 * V)
        >>> str(corner.arguments["supply"])
        '3 V'
    """

    temperature: Quantity | None
    arguments: Mapping[str, object]

    def __init__(
        self, *, temperature: Quantity | None = None, **arguments: object
    ) -> None:
        """A corner at `temperature`, building the circuit with `arguments`.

        Raises:
            DimensionError: When the temperature is not one.
        """
        if temperature is not None:
            temperature.to(TEMPERATURE, "a corner's temperature")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "arguments", arguments)


@dataclass(frozen=True, slots=True)
class MonteCarlo:
    """Samples of the circuit's tolerances, each a run of the plan's analyses.

    Each sample runs at the plan's own conditions and at each of its corners, on the
    same parts wherever the corner's circuit holds the same toleranced parameters, and
    a requirement's verdict on it is its worst there. A `seed` draws the same samples
    from the same circuit.
    """

    samples: int
    seed: int = 0

    def __post_init__(self) -> None:
        """Checks that there is a sample or more.

        Raises:
            ValueError: When there is none.
        """
        if self.samples < 1:
            message = (
                f"a Monte Carlo analysis draws a sample or more, not {self.samples}"
            )
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        """Checks the settings, and keeps the grids as tuples.

        Raises:
            DimensionError: When the supply is not a voltage.
            ValueError: When a grid does not rise, from above zero for the transitions
                and from zero or more for the fanouts, or the spot checks or the
                tolerance are negative.
        """
        if self.supply.to(VOLTAGE, "a characterization's supply") <= 0:
            message = f"a characterization's supply is positive, not {self.supply}"
            raise ValueError(message)
        for name, grid, above in (
            ("slews", self.slews, "above zero"),
            ("fanouts", self.fanouts, "from zero or more"),
        ):
            values = tuple(float(value) for value in grid)
            rising = all(a < b for a, b in itertools.pairwise(values))
            first = values[0] if values else math.nan
            starts = first > 0 if name == "slews" else first >= 0
            if not rising or not starts:
                message = f"the {name} rise {above}, not {values}"
                raise ValueError(message)
            object.__setattr__(self, name, values)
        if self.spot_checks < 0 or self.tolerance < 0:
            message = "the spot checks and the tolerance are zero or more"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class Plan:
    """Named analyses to run on a circuit, the options they share, and its conditions.

    Each analysis is named by a Python identifier, such as ``step`` or ``bode``, which
    names its results too. The circuit's devices are simulated at `temperature`, 27 °C
    unless given, then at each of the `corners`, each named by an identifier too, and on
    the samples of a `monte_carlo` analysis.

    Its `fidelity` maps scopes, patterns of instance paths such as ``adder/*``, where
    ``*`` matches within a level of the hierarchy and ``**`` across levels, to the
    logic models characterized for the component instances they match; an instance
    within another that a scope matches takes the outer one's model.
    """

    analyses: Mapping[str, Analysis]
    options: Options = field(default_factory=Options)
    temperature: Quantity | None = None
    corners: Mapping[str, Corner] = field(default_factory=dict)
    monte_carlo: MonteCarlo | None = None
    fidelity: Mapping[str, Characterized] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Checks that the plan runs an analysis or more, all named by identifiers.

        Raises:
            ValueError: When the plan runs no analysis, or a name is not an identifier.
            TypeError: When a corner is not one.
            DimensionError: When the temperature is not one.
        """
        if self.temperature is not None:
            self.temperature.to(TEMPERATURE, "a plan's temperature")
        if not self.analyses:
            message = "a plan runs at least one analysis"
            raise ValueError(message)
        for kind, name in [
            *(("an analysis", name) for name in self.analyses),
            *(("a corner", name) for name in self.corners),
        ]:
            if not name.isidentifier():
                message = f"{kind} is named by a Python identifier, not {name!r}"
                raise ValueError(message)
        for name, corner in self.corners.items():
            if not isinstance(corner, Corner):
                message = f"the corner {name} is {corner!r}, not a Corner"
                raise TypeError(message)
        for scope, model in self.fidelity.items():
            if not isinstance(model, Characterized):
                message = (
                    f"the scope {scope} takes {model!r}, not a Characterized model"
                )
                raise TypeError(message)

    def temperature_at(self, corner: Corner | None = None) -> Quantity | None:
        """The temperature of the plan's own run, or of `corner`'s; None for 27 °C."""
        if corner is not None and corner.temperature is not None:
            return corner.temperature
        return self.temperature


def decade(start: Quantity, stop: Quantity, per_decade: int) -> list[Quantity]:
    """Frequencies from `start` to `stop`, `per_decade` of them spaced evenly a decade.

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
    first, last = math.log10(low), math.log10(high)
    count = max(1, round((last - first) * per_decade))
    return [
        Quantity(10 ** (first + (last - first) * step / count), FREQUENCY)
        for step in range(count + 1)
    ]


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

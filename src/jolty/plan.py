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

Examples:
    >>> from jolty.units import us
    >>> plan = Plan({"step": Transient(stop=10 * us)})
"""

import decimal
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from decimal import Decimal
from types import MappingProxyType
from typing import Final, Literal

from pydantic import JsonValue

from jolty import _core, ir
from jolty.spice import ngspice
from jolty.units import (
    FREQUENCY,
    TEMPERATURE,
    TIME,
    VOLTAGE,
    Dimension,
    Quantity,
    finite,
)

DECADE_DIGITS: Final = 40
"""The decimal digits a decade sweep's frequencies are computed to before they are
rounded to floats: so many more than a float's 17 that a million steps' rounding stays
far below a float's, and each frequency rounds to the float nearest its exact value."""

type Method = Literal["trapezoidal", "gear2", "backward_euler"]
"""An integration method: trapezoidal by default, Gear2 for stiff circuits that ring,
backward Euler for the most damping."""


def voltages(given: Mapping[str, Quantity], what: str) -> dict[str, float]:
    """The voltage of each net `given` names, in volts, each checked as `what`.

    Raises:
        DimensionError: When one is not a voltage.
    """
    return {net: voltage.to(VOLTAGE, what) for net, voltage in given.items()}


def recorded(record: Sequence[str] | None) -> tuple[str, ...] | None:
    """`record` as a tuple of patterns, or None when it is left out.

    Raises:
        TypeError: When the record is a string, or holds anything but strings.
    """
    if record is None:
        return None
    if isinstance(record, str):
        message = (
            "a record is a sequence of instance paths, such as "
            f"[{record!r}], not the string {record!r}"
        )
        raise TypeError(message)
    patterns = tuple(record)
    if not all(isinstance(pattern, str) for pattern in patterns):
        message = f"a record is a sequence of instance paths, not {patterns!r}"
        raise TypeError(message)
    return patterns


@dataclass(frozen=True, slots=True)
class Recorded:
    """An analysis whose results can keep the devices' detail beside the unknowns.

    Its `record` names the instances whose detail its results keep, as a plan's does,
    and replaces the plan's when given (see [`Plan`][jolty.plan.Plan]). With `regions`,
    its results keep besides each other transistor's region of operation and its
    margins to the regions' boundaries alone, which an operating point's `region` reads.
    """

    record: Sequence[str] | None = field(default=None, kw_only=True)
    regions: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        """Keeps the record as a tuple.

        Raises:
            TypeError: When the record is a string, or holds anything but strings.
        """
        object.__setattr__(self, "record", recorded(self.record))


@dataclass(frozen=True, slots=True)
class OperatingPoint(Recorded):
    """The DC operating point, with every source at its level at time zero."""

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        return "operating_point"


@dataclass(frozen=True, slots=True)
class OperatingPoints(Recorded):
    """The circuit's DC solutions, up to `max_solutions`, the operating point first.

    Once the operating point is found, the circuit's equations deflated by the
    solutions found so far lead Newton iteration to another, until none does, so a
    latch has a solution per state and one balanced between them, and a self-biased
    reference its zero-current state beside its working one.
    """

    max_solutions: int = 8

    def __post_init__(self) -> None:
        """Checks that the search finds a solution or more.

        Raises:
            ValueError: When it finds none.
            TypeError: When the record is a string, or holds anything but strings.
        """
        Recorded.__post_init__(self)
        if self.max_solutions < 1:
            message = (
                "a search for operating points finds 1 or more, "
                f"not {self.max_solutions}"
            )
            raise ValueError(message)

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        return {"operating_points": {"max_solutions": self.max_solutions}}


@dataclass(frozen=True, slots=True)
class DcSweep(Recorded):
    """The operating point at each level of one independent source, or of a parameter.

    The source is named by its path, such as ``V1`` or ``bias/I1``. Given a
    `parameter`, named by its path among the instance's parameters, such as
    ``resistance``, the sweep sets that number of any device to each level instead, the
    device made again from its parameters with the number replaced.
    """

    source: str
    values: Sequence[Quantity]
    parameter: str | None = None

    def to_json(self, dimension: Dimension) -> JsonValue:
        """The analysis in the engine's format, checked against `dimension`."""
        values: list[JsonValue] = [
            value.to(dimension, "a sweep level") for value in self.values
        ]
        sweep: dict[str, JsonValue] = {"source": self.source, "values": values}
        if self.parameter is not None:
            sweep["parameter"] = self.parameter
        return {"dc_sweep": sweep}


@dataclass(frozen=True, slots=True)
class Ac(Recorded):
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
class Transient(Recorded):
    """The response over time, from the operating point at zero until `stop`.

    The analysis steps onto each of `breakpoints` exactly, besides the times it chooses.
    Its `initial_conditions` give nets, by path such as ``out`` or ``amp/mid``, the
    voltage they start from, as SPICE's ``.ic``: each is held there while the operating
    point is solved, then left free. A net that voltage sources and inductors join to
    the ground, or to a net named before it, is left to them, and its voltage there must
    be the one given. Without its `operating_point`, as SPICE's ``UIC``, the transient
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


@dataclass(frozen=True, slots=True)
class PeriodicSteadyState(Recorded):
    """One period of the steady state a circuit repeats, found by shooting.

    Given a `period`, that of every source, shooting starts from the operating point;
    without one, the circuit is an oscillator whose sources hold steady, and shooting
    finds its period too, starting where a transient from its operating point, kicked
    along its fastest-growing natural mode, settles onto its orbit. The period's steps
    are at most `max_step`, by default a fiftieth of the period, and land on each of
    `breakpoints`, times within a given period.

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


@dataclass(frozen=True, slots=True)
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

    def to_json(self, level: Dimension = VOLTAGE) -> JsonValue:
        """The analysis in the engine's format.

        The levels of a DC sweep it differentiates are checked against `level`.
        """
        outputs: list[JsonValue] = list(self.outputs)
        return {
            "sensitivity": {"outputs": outputs, "of": differentiated(self.of, level)}
        }


@dataclass(frozen=True, slots=True)
class Mismatch:
    """How far unknowns spread, to first order, with the numbers samples draw at random.

    Each output is an unknown by name, as a [`Sensitivity`][jolty.plan.Sensitivity]'s
    is, and spreads at the operating point, or at each point of the analysis `of`, with
    every random variable ([`RandomVariable`][jolty.ir.RandomVariable]): each part's own
    factor and each lot's of the circuit's tolerances, as a Monte Carlo sample draws
    them, and the `random_variables` given besides, such as a process kit's statistical
    functions ([`Library.random_variables`][jolty.spice.Library.random_variables]). Its
    derivative with respect to each is the sensitivity analysis's, summed over the
    numbers the random variable moves; the random variables are drawn apart from one
    another, so to first order the output's standard deviation is the root of the sum
    of each derivative times its random variable's standard deviation, squared, and each
    random variable's share of its variance is its own term's.
    """

    outputs: Sequence[str]
    of: Differentiated = None
    random_variables: Sequence[ir.RandomVariable] = ()

    def to_json(
        self, level: Dimension = VOLTAGE, toleranced: Sequence[JsonValue] = ()
    ) -> JsonValue:
        """The analysis in the engine's format, with the circuit's `toleranced` numbers.

        The levels of a DC sweep it differentiates are checked against `level`.
        """
        outputs: list[JsonValue] = list(self.outputs)
        given: list[JsonValue] = [
            variable.model_dump(mode="json") for variable in self.random_variables
        ]
        return {
            "mismatch": {
                "outputs": outputs,
                "of": differentiated(self.of, level),
                "toleranced": list(toleranced),
                "random_variables": given,
            }
        }


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
class TransferFunction:
    """The small-signal transfer function at DC, as SPICE's ``.tf`` reports it.

    From an independent source, named by its path such as ``V1``, to an output: a net's
    voltage, over the ground or over the net `reference` names, or a voltage source's
    current, such as ``V2:current``. It gives the gain, the output per volt or ampere of
    the source, the resistance the source sees, and the resistance the output sees with
    the source at its level: a voltage source a short, a current source open.

    Examples:
        >>> TransferFunction("V1", "out").to_json()
        {'transfer_function': {'source': 'V1', 'output': 'out'}}
    """

    source: str
    output: str
    reference: str | None = None

    def to_json(self) -> JsonValue:
        """The analysis in the engine's format."""
        given: dict[str, JsonValue] = {"source": self.source, "output": self.output}
        if self.reference is not None:
            given["reference"] = self.reference
        return {"transfer_function": given}


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


@dataclass(frozen=True, slots=True)
class Parallelism:
    """How a run spreads its own work across the engine's threads.

    Each kind of work spreads in tasks of at least its threshold, its least size per
    task, once it fills such a task for every thread the run may use, and stays on the
    run's own thread for less: the
    frequencies of a run's small-signal analyses and the right-hand sides of its
    sensitivities, in tasks of at least `min_solves_per_task`; each group of its device
    instances, in tasks of at least `min_instances_per_task`, but a linear one's, which
    are evaluated once a run, for their entries; and the work of its Newton
    iteration and its transient's step control on each unknown and each stored entry of
    its matrix, in tasks of at least `min_unknowns_per_task`, in whole chunks of 1,024
    its threads take as they free. Each block of its matrix of more than
    `max_unknowns_per_interior` unknowns is ordered by nested dissection into interiors
    of at most as many and the separators between them, which factor at once, wherever
    that costs its elimination fewer multiply-adds than the minimum degree ordering. A
    run spreads over at most `max_threads` of the engine's threads, the setting
    `threads` or [`threads`][jolty.simulate.threads], and over all of them by default; a
    batch runs each run on one thread whenever it has a run for every thread. Each
    setting left out but `max_threads` takes its break-even on the threads the run may
    use, [`BREAK_EVENS`][jolty.plan.BREAK_EVENS]. Every result is the same bits however
    the work is split, and no setting but `max_unknowns_per_interior` changes any: that
    one orders the factorization's operations otherwise, so results agree with those of
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

    def __post_init__(self) -> None:
        """Checks that each setting given is a count of one or more.

        Raises:
            ValueError: When one is below one.
        """
        for setting in fields(self):
            value = getattr(self, setting.name)
            if value is not None and value < 1:
                message = f"{setting.name} is 1 or more, not {value}"
                raise ValueError(message)

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
measured paid; a run takes the one measured on the most threads it may use or fewer.

Examples:
    >>> BREAK_EVENS["threads"]
    (2, 4, 8, 64)
"""


@dataclass(frozen=True, slots=True)
class Options:
    """Numerical settings; each one left out takes the engine's default, SPICE's own.

    Tolerances are in SI base units: volts, amperes and coulombs. The `parallelism`
    says how a run spreads its own work across threads, which changes no result but
    for the last digits its dissected factorization's order of operations gives.

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


@dataclass(frozen=True, slots=True)
class Conditions:
    """What a simulation runs its circuit at.

    A Monte Carlo sample and a corner each run at conditions of their own.
    """

    temperature: float
    """The temperature, in kelvins."""
    process: str | None = None
    """The process corner of a foundry's devices, such as ``ss``, its kit's typical one
    unless given."""
    overrides: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    """The parameters that replace the circuit's, by instance path, then by parameter
    path, such as ``waveform.dc``, in SI base units."""
    mismatch: int | None = None
    """The seed a foundry's devices draw their kit's mismatch from, if they draw it."""


@dataclass(frozen=True, slots=True, init=False)
class Corner:
    """Conditions a plan also runs its analyses at.

    A corner gives a temperature and a foundry's process corner, the plan's unless
    given, and arguments to the build script's ``build()``, which builds the circuit
    again with them, such as a lower supply voltage.

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
            DimensionError: When the temperature is not one.
        """
        if temperature is not None:
            temperature.to(TEMPERATURE, "a corner's temperature")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "process", process)
        object.__setattr__(self, "arguments", arguments)


@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        """Checks that there is a sample or more, and that the seed is 64 bits.

        Raises:
            ValueError: When there is no sample, or the seed is negative or 2**64 or
                more.
        """
        if self.samples < 1:
            message = (
                f"a Monte Carlo analysis draws a sample or more, not {self.samples}"
            )
            raise ValueError(message)
        if not 0 <= self.seed < 2**64:
            message = f"a Monte Carlo seed lies from 0 to 2**64 - 1, not {self.seed}"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class Variant:
    """The values one run of a batch takes: numbers and a temperature.

    The numbers replace parameters of devices, by instance path then by parameter path,
    such as ``waveform.dc``, in SI base units, as a simulation's overrides do; the
    temperature is the simulation's unless given.

    Examples:
        >>> from jolty.units import celsius
        >>> Variant({"R1": {"resistance": 2e3}}, celsius(85)).overrides
        {'R1': {'resistance': 2000.0}}
    """

    overrides: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    temperature: Quantity | None = None

    def __post_init__(self) -> None:
        """Keeps the numbers as dictionaries of floats, and checks the temperature.

        Raises:
            DimensionError: When the temperature is not one.
            ValueError: When a number is infinite or not a number.
        """
        if self.temperature is not None:
            self.temperature.to(TEMPERATURE, "a variant's temperature")
        numbers = {
            path: {name: finite(float(value), name) for name, value in named.items()}
            for path, named in self.overrides.items()
        }
        object.__setattr__(self, "overrides", numbers)


@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        """Keeps the variants as a tuple, and checks there is one or more.

        Raises:
            ValueError: When there is none.
            TypeError: When one is not a variant.
        """
        variants = tuple(self.variants)
        if not variants:
            message = "explicit variants list a run or more"
            raise ValueError(message)
        for variant in variants:
            if not isinstance(variant, Variant):
                message = f"explicit variants are variants, not {variant!r}"
                raise TypeError(message)
        object.__setattr__(self, "variants", variants)


@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        """Keeps the values as a tuple, and checks them.

        Raises:
            ValueError: When there is no value, or the axis names an instance but no
                parameter or a parameter but no instance.
            DimensionError: When the values of the temperature are not temperatures,
                or a parameter's are not all of one dimension.
        """
        values = tuple(self.values)
        object.__setattr__(self, "values", values)
        if (self.instance is None) != (self.parameter is None):
            message = "an axis names an instance and its parameter, or neither"
            raise ValueError(message)
        if not values:
            message = "an axis steps a value or more"
            raise ValueError(message)
        self.numbers()

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


@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        """Keeps the axes as a tuple.

        Raises:
            TypeError: When one is not an axis.
        """
        axes = tuple(self.axes)
        for axis in axes:
            if not isinstance(axis, Axis):
                message = f"a grid's axes are axes, not {axis!r}"
                raise TypeError(message)
        object.__setattr__(self, "axes", axes)


@dataclass(frozen=True, slots=True)
class Product:
    """Every run of `first` at every run of `second`.

    Run `i` is run `i // len(second)` of `first`, then run `i % len(second)` of
    `second`, whose values replace those of `first` where both give one, so corners,
    samples and a sweep make one batch. Its chains are those of `second`.
    """

    first: Variants
    second: Variants

    def __post_init__(self) -> None:
        """Checks that both are variants.

        Raises:
            TypeError: When one is not.
        """
        for part in (self.first, self.second):
            if not isinstance(part, MonteCarlo | Grid | Explicit | Product):
                message = f"a product is of variants, not {part!r}"
                raise TypeError(message)


type Variants = MonteCarlo | Grid | Explicit | Product
"""The values of many runs of one circuit, each a pure function of the run's index."""


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


@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        """Checks the step.

        Raises:
            ValueError: When it is not above 0 and at most 1.
        """
        if not 0 < self.step <= 1:
            message = (
                f"a worst case moves a step above 0 and at most 1, not {self.step}"
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


TIGHT: Final = Options(
    relative_tolerance=ngspice.RELATIVE_TOLERANCE,
    voltage_tolerance=ngspice.VOLTAGE_TOLERANCE,
    current_tolerance=ngspice.CURRENT_TOLERANCE,
)
"""jolty's options as tight as the ones [`jolty.spice.ngspice`][jolty.spice.ngspice]
gives ngspice, which the comparisons of the two run with.

The charge tolerance is both engines' default, 1e-14 C, as both read it alike: the least
charge the relative tolerance is taken of, so a charge is held to 1e-20 C at least."""


@dataclass(frozen=True, slots=True)
class Plan:
    """Named analyses to run on a circuit, the options they share, and its conditions.

    Each analysis is named by a Python identifier, such as ``step`` or ``bode``, which
    names its results too. The circuit's devices are simulated at `temperature`, 27 °C
    unless given, a foundry's devices at its `process` corner, its kit's typical unless
    given, then at each of the `corners`, each named by an identifier too, on the
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
    instead, and an operating point and a transient also record the instances rated for
    a current or a power, whose ratings the checks hold them to.
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

    def __post_init__(self) -> None:
        """Checks that the plan runs an analysis or more, all named by identifiers.

        Raises:
            ValueError: When the plan runs no analysis, or a name is not an identifier.
            TypeError: When a corner is not one, the variants are not variants, or the
                record is a string or holds anything but strings.
            DimensionError: When the temperature is not one.
        """
        object.__setattr__(self, "record", recorded(self.record))
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
        if self.variants is not None and not isinstance(
            self.variants, MonteCarlo | Grid | Explicit | Product
        ):
            message = f"a plan's variants are variants, not {self.variants!r}"
            raise TypeError(message)

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

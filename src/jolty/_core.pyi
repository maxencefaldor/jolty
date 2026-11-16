# Copyright 2026 Maxence Faldor
# The types of the Rust extension module; its documentation lives with the Rust code.

from collections.abc import Callable
from typing import Literal, Self, TypedDict, final

import numpy as np
import numpy.typing as npt

__version__: str
EXPRESSIONS: str
BOLTZMANN: float
ELEMENTARY_CHARGE: float
REFERENCE_TEMPERATURE: float
REGION: str
SAMPLES_PER_PERIOD: int

class ElaborationError(Exception):
    args: tuple[list[Problem]]

class SimulationError(Exception): ...

class MeasureError(Exception):
    args: tuple[str, list[float]]

type Real = npt.NDArray[np.float64]
type Complex = npt.NDArray[np.complex128]
type Signal = tuple[int, int | None]
type Transfer = tuple[Signal, Signal | None] | None
type Window = tuple[float | None, float | None]
type Nature = Literal["voltage", "current", "temperature", "state"]

class RecordedValues(TypedDict):
    currents: npt.NDArray[np.float64]
    charges: npt.NDArray[np.float64]
    conductances: npt.NDArray[np.float64]
    capacitances: npt.NDArray[np.float64]
    variables: npt.NDArray[np.float64]

class SolvedPoint(RecordedValues):
    values: npt.NDArray[np.float64]
    logic: npt.NDArray[np.float64]
    strategy: str
    iterations: int

class OperatingPointResult(SolvedPoint):
    kind: Literal["operating_point"]

class OperatingPointsResult(TypedDict):
    kind: Literal["operating_points"]
    points: list[SolvedPoint]

class DcSweepResult(RecordedValues):
    kind: Literal["dc_sweep"]
    source: str
    parameter: str | None
    nature: Nature | None
    swept: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    logic: npt.NDArray[np.float64]
    strategies: list[str]
    iterations: list[int]

class AcResult(TypedDict):
    kind: Literal["ac"]
    source: str
    frequencies: npt.NDArray[np.float64]
    values: npt.NDArray[np.complex128]
    currents: npt.NDArray[np.complex128]
    operating_point: SolvedPoint

class Work(TypedDict):
    iterations: int
    factorizations: int
    refactorizations: int
    fallbacks: int
    reuses: int

class Profile(TypedDict):
    preparing: float
    evaluating: float
    assembling: float
    factoring: float
    solving: float
    step_control: float
    recording: float
    other: float

class Statistics(TypedDict):
    initial: str
    accepted: int
    rejected: int
    landed: int
    iterations: int
    work: Work
    profile: Profile | None

class Waveforms(RecordedValues):
    times: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    logic: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
    statistics: Statistics

class TransientResult(Waveforms):
    kind: Literal["transient"]

class PeriodicSteadyStateResult(Waveforms):
    kind: Literal["periodic_steady_state"]
    period: float
    periods: int

class SensitivityResult(TypedDict):
    kind: Literal["sensitivity"]
    outputs: list[str]
    parameters: list[tuple[str, str]]
    values: npt.NDArray[np.float64]
    derivatives: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    period_derivatives: npt.NDArray[np.float64] | None
    result: AnalysisResult

class StabilityResult(TypedDict):
    kind: Literal["stability"]
    probe: str
    frequencies: npt.NDArray[np.float64]
    gains: npt.NDArray[np.complex128]
    operating_point: SolvedPoint

class NoiseInput(TypedDict):
    source: str
    nature: Nature
    gains: npt.NDArray[np.complex128]

class NoiseResult(TypedDict):
    kind: Literal["noise"]
    output: str
    reference: str | None
    nature: Nature
    frequencies: npt.NDArray[np.float64]
    densities: npt.NDArray[np.float64]
    input: NoiseInput | None
    sources: list[tuple[str, str]]
    contributions: npt.NDArray[np.float64]
    operating_point: SolvedPoint

class PoleZeroResult(TypedDict):
    kind: Literal["pole_zero"]
    source: str | None
    output: str | None
    reference: str | None
    poles: npt.NDArray[np.complex128]
    zeros: npt.NDArray[np.complex128] | None
    operating_point: SolvedPoint

class TransferFunctionResult(TypedDict):
    kind: Literal["transfer_function"]
    source: str
    input: Nature
    output: str
    reference: str | None
    gain: float
    input_resistance: float
    output_resistance: float
    operating_point: SolvedPoint

type AnalysisResult = (
    OperatingPointResult
    | DcSweepResult
    | AcResult
    | TransientResult
    | PeriodicSteadyStateResult
    | SensitivityResult
    | StabilityResult
    | NoiseResult
    | PoleZeroResult
    | OperatingPointsResult
    | TransferFunctionResult
)

class SourceLocation(TypedDict):
    file: str
    line: int
    function: str | None

class Location(TypedDict):
    path: str
    source_location: SourceLocation | None

class Problem(TypedDict):
    message: str
    instance: Location | None

class Diagnostic(TypedDict):
    rule: str
    message: str
    instances: list[Location]
    nets: list[str]

class Excursion(TypedDict):
    instance: str
    bound: Literal["input transition", "load", "supply"]
    minimum: float
    maximum: float
    value: float
    time: float
    count: int

def rules() -> list[tuple[str, Literal["error", "warn", "ignore"], str]]: ...
def terminals(model: str, parameters: str) -> list[str]: ...
def constant(expression: str) -> float: ...
def expression_tree(expression: str) -> str: ...
def defaulted(model: str, parameters: str) -> str: ...
def default_options() -> str: ...
def break_evens() -> str: ...
def level(waveform: str, time: float) -> float: ...
def models() -> list[str]: ...
def schema(model: str) -> str: ...
def mantissas(series: str) -> list[float]: ...

@final
class Pool:
    def __new__(cls, threads: int | None = None) -> Self: ...
    @property
    def threads(self) -> int: ...

def sample(varied: str, seed: int, index: int) -> list[float]: ...
def toleranced(circuit: str) -> list[tuple[str, str, float, str]]: ...
def variate(variate: str, seed: int, index: int, scope: str, call: str) -> float: ...

@final
class Values: ...

@final
class StructureKey:
    @property
    def unchanged(self) -> bool: ...

@final
class Elaborated:
    def __new__(cls, circuit: str, substitutes: str) -> Self: ...
    def values(self, overrides: str, temperature: float) -> Values: ...
    def structure_of(self, values: Values) -> StructureKey: ...
    def restructured(self, values: Values) -> Elaborated: ...

@final
class Simulator:
    def __new__(
        cls, circuit: str, temperature: float, overrides: str, substitutes: str
    ) -> Self: ...
    @staticmethod
    def from_elaborated(elaborated: Elaborated, values: Values) -> Simulator: ...
    @property
    def unknowns(
        self,
    ) -> list[tuple[str, Literal["voltage", "current", "temperature"]]]: ...
    @property
    def nets(self) -> list[str]: ...
    @property
    def instances(self) -> list[str]: ...
    def recording(
        self, record: str, regions: bool
    ) -> tuple[
        list[
            tuple[str, list[str], list[bool], int, int, list[str], list[str], list[str]]
        ],
        list[tuple[str, bool]],
    ]: ...
    def terminal(
        self, instance: str, terminal: str
    ) -> tuple[str | None, bool] | None: ...
    @property
    def devices(self) -> int: ...
    @property
    def foundry_devices(self) -> list[str]: ...
    def check(self) -> list[Diagnostic]: ...
    def check_operating_point(self, options: str, pool: Pool) -> list[Diagnostic]: ...
    def excursions(self) -> list[Excursion]: ...
    def run(self, analysis: str, options: str, pool: Pool) -> AnalysisResult: ...
    @property
    def elaborated(self) -> Elaborated: ...
    def batch(
        self,
        analyses: list[tuple[str, str]],
        variants: Variants,
        runs: tuple[int, int],
        measures: list[tuple[int, Measure]],
        traces: list[tuple[int, list[int]]],
        pool: Pool,
        each: Callable[
            [int, Simulator | None, list[Problem] | None, list[AnalysisResult | str]],
            None,
        ]
        | None = None,
    ) -> BatchResult: ...

class BatchResult(TypedDict):
    values: npt.NDArray[np.float64]
    parameters: list[tuple[str, str] | None]
    measured: npt.NDArray[np.float64]
    causes: npt.NDArray[np.int32]
    reasons: list[tuple[str, list[float], str]]
    ended: npt.NDArray[np.bool_]
    problems: dict[int, list[Problem]]
    errors: dict[int, list[tuple[int, str]]]
    traces: list[npt.NDArray[np.float64] | npt.NDArray[np.complex128]]

@final
class Variants:
    @staticmethod
    def explicit(
        elaborated: Elaborated, runs: list[tuple[str, float | None]]
    ) -> Variants: ...
    @staticmethod
    def grid(
        elaborated: Elaborated,
        axes: list[tuple[tuple[str, str] | None, list[float]]],
        warm: bool,
    ) -> Variants: ...
    @staticmethod
    def monte_carlo(
        elaborated: Elaborated, varied: str, seed: int, samples: int
    ) -> Variants: ...
    @staticmethod
    def product(first: Variants, second: Variants) -> Variants: ...
    def __len__(self) -> int: ...
    @property
    def parameters(self) -> list[tuple[str, str] | None]: ...
    def overrides(
        self, index: int
    ) -> tuple[dict[str, dict[str, float]], float | None]: ...
    def sample(self, index: int) -> tuple[int, int] | None: ...

def integrate(
    frequencies: Real,
    densities: Real,
    start: float | None = None,
    stop: float | None = None,
) -> float: ...
def density(frequencies: Real, densities: Real, at: float) -> float: ...
def referred(densities: Real, gains: Complex) -> Real: ...
def dampings(roots: Complex) -> Real: ...
def natural_frequencies(roots: Complex) -> Real: ...

@final
class Measure:
    @staticmethod
    def value(signal: Signal, at: float | None) -> Measure: ...
    @staticmethod
    def gain(transfer: Transfer, at: float) -> Measure: ...
    @staticmethod
    def phase(transfer: Transfer, at: float) -> Measure: ...
    @staticmethod
    def bandwidth(transfer: Transfer, drop: float) -> Measure: ...
    @staticmethod
    def crossover(transfer: Transfer) -> Measure: ...
    @staticmethod
    def phase_margin(transfer: Transfer) -> Measure: ...
    @staticmethod
    def gain_margin(transfer: Transfer) -> Measure: ...
    @staticmethod
    def maximum(signal: Signal, window: Window) -> Measure: ...
    @staticmethod
    def minimum(signal: Signal, window: Window) -> Measure: ...
    @staticmethod
    def peak_to_peak(signal: Signal, window: Window) -> Measure: ...
    @staticmethod
    def mean(signal: Signal, window: Window) -> Measure: ...
    @staticmethod
    def rms(signal: Signal, window: Window) -> Measure: ...
    @staticmethod
    def rise_time(
        signal: Signal, window: Window, low: float, high: float
    ) -> Measure: ...
    @staticmethod
    def overshoot(signal: Signal, window: Window) -> Measure: ...
    @staticmethod
    def settling_time(signal: Signal, window: Window, band: float) -> Measure: ...
    @staticmethod
    def crossing(
        signal: Signal,
        window: Window,
        level: float,
        edge: Literal["rising", "falling", "either"],
    ) -> Measure: ...
    @staticmethod
    def frequency(signal: Signal, window: Window, level: float | None) -> Measure: ...
    @staticmethod
    def distortion(
        signal: Signal, window: Window, fundamental: float, harmonics: int
    ) -> Measure: ...
    @staticmethod
    def noise_density(at: float, referred: bool) -> Measure: ...
    @staticmethod
    def integrated_noise(
        start: float | None,
        stop: float | None,
        referred: bool,
    ) -> Measure: ...
    @staticmethod
    def solutions() -> Measure: ...
    @staticmethod
    def damping() -> Measure: ...
    @staticmethod
    def dominant_pole() -> Measure: ...
    def read_operating_point(self, values: Real) -> float: ...
    def read_dc_sweep(self, swept: Real, values: Real) -> float: ...
    def read_transient(self, times: Real, values: Real) -> float: ...
    def read_ac(self, frequencies: Real, values: Complex) -> float: ...
    def read_stability(self, frequencies: Real, gains: Complex) -> float: ...
    def read_noise(
        self, frequencies: Real, densities: Real, gains: Complex | None
    ) -> float: ...
    def read_pole_zero(self, poles: Complex) -> float: ...
    def read_operating_points(self, count: int) -> float: ...

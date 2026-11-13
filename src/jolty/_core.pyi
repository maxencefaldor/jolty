# Copyright 2026 Maxence Faldor
# The types of the Rust extension module; its documentation lives with the Rust code.

from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt

__version__: str
EXPRESSIONS: str
BOLTZMANN: float
ELEMENTARY_CHARGE: float
REFERENCE_TEMPERATURE: float

class ElaborationError(Exception):
    args: tuple[list[ProblemOutcome]]

class SimulationError(Exception): ...

class Observed(TypedDict):
    currents: npt.NDArray[np.float64]
    charges: npt.NDArray[np.float64]
    conductances: npt.NDArray[np.float64]
    capacitances: npt.NDArray[np.float64]
    variables: npt.NDArray[np.float64]

class SolvedPoint(Observed):
    values: npt.NDArray[np.float64]
    logic: npt.NDArray[np.float64]
    strategy: str

class OperatingPointOutcome(SolvedPoint):
    kind: Literal["operating_point"]

class OperatingPointsOutcome(TypedDict):
    kind: Literal["operating_points"]
    points: list[SolvedPoint]

class SweepOutcome(Observed):
    kind: Literal["dc_sweep"]
    source: str
    quantity: Literal["voltage", "current", "temperature", "state"]
    levels: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    logic: npt.NDArray[np.float64]

class ResponseOutcome(TypedDict):
    kind: Literal["ac"]
    source: str
    frequencies: npt.NDArray[np.float64]
    values: npt.NDArray[np.complex128]
    currents: npt.NDArray[np.complex128]
    operating_point: SolvedPoint

class StatisticsOutcome(TypedDict):
    initial: str
    accepted: int
    rejected: int
    iterations: int
    periods: int | None

class WaveformsOutcome(Observed):
    kind: Literal["transient"]
    times: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    logic: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
    statistics: StatisticsOutcome

class SensitivityOutcome(TypedDict):
    kind: Literal["sensitivity"]
    outputs: list[str]
    parameters: list[tuple[str, str]]
    values: npt.NDArray[np.float64]
    derivatives: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    result: Outcome

class StabilityOutcome(TypedDict):
    kind: Literal["stability"]
    probe: str
    frequencies: npt.NDArray[np.float64]
    gains: npt.NDArray[np.complex128]
    operating_point: SolvedPoint

class NoiseInputOutcome(TypedDict):
    source: str
    quantity: Literal["voltage", "current", "temperature", "state"]
    gains: npt.NDArray[np.complex128]

class NoiseOutcome(TypedDict):
    kind: Literal["noise"]
    output: str
    reference: str | None
    quantity: Literal["voltage", "current", "temperature", "state"]
    frequencies: npt.NDArray[np.float64]
    densities: npt.NDArray[np.float64]
    input: NoiseInputOutcome | None
    sources: list[tuple[str, str]]
    contributions: npt.NDArray[np.float64]
    operating_point: SolvedPoint

class PoleZeroOutcome(TypedDict):
    kind: Literal["pole_zero"]
    source: str | None
    output: str | None
    reference: str | None
    poles: npt.NDArray[np.complex128]
    zeros: npt.NDArray[np.complex128] | None
    operating_point: SolvedPoint

type Outcome = (
    OperatingPointOutcome
    | SweepOutcome
    | ResponseOutcome
    | WaveformsOutcome
    | SensitivityOutcome
    | StabilityOutcome
    | NoiseOutcome
    | PoleZeroOutcome
    | OperatingPointsOutcome
)

class SourceOutcome(TypedDict):
    file: str
    line: int
    function: str | None

class InstanceOutcome(TypedDict):
    path: str
    source: SourceOutcome | None

class ProblemOutcome(TypedDict):
    message: str
    instance: InstanceOutcome | None

class DiagnosticOutcome(TypedDict):
    rule: str
    message: str
    instances: list[InstanceOutcome]
    nets: list[str]

class ExcursionOutcome(TypedDict):
    path: str
    bound: Literal["input transition", "load", "supply"]
    low: float
    high: float
    value: float
    time: float
    count: int

def rules() -> list[tuple[str, Literal["error", "warn", "ignore"], str]]: ...
def terminals(model: str, parameters: str) -> list[str]: ...
def constant(expression: str) -> float: ...
def expression_tree(expression: str) -> str: ...
def defaulted(model: str, parameters: str) -> str: ...
def mantissas(series: str) -> list[float]: ...

class Simulator:
    def __init__(
        self, circuit: str, temperature: float, overrides: str, substitutes: str
    ) -> None: ...
    @property
    def unknowns(
        self,
    ) -> list[tuple[str, Literal["voltage", "current", "temperature"]]]: ...
    @property
    def nets(self) -> list[str]: ...
    @property
    def instances(self) -> list[str]: ...
    def recording(
        self, record: str
    ) -> tuple[
        list[tuple[str, list[str], list[bool], int, int, list[str]]],
        list[tuple[str, bool]],
    ]: ...
    def terminal(
        self, instance: str, terminal: str
    ) -> tuple[str | None, bool] | None: ...
    @property
    def devices(self) -> int: ...
    @property
    def foundry_devices(self) -> list[str]: ...
    def check(self) -> list[DiagnosticOutcome]: ...
    def check_operating_point(self, options: str) -> list[DiagnosticOutcome]: ...
    def excursions(self) -> list[ExcursionOutcome]: ...
    def run(self, analysis: str, options: str) -> Outcome: ...

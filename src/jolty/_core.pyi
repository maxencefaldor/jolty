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

class ElaborationError(Exception): ...
class SimulationError(Exception): ...

class SolvedPoint(TypedDict):
    values: npt.NDArray[np.float64]
    logic: npt.NDArray[np.float64]
    strategy: str

class OperatingPointOutcome(TypedDict):
    kind: Literal["operating_point"]
    values: npt.NDArray[np.float64]
    logic: npt.NDArray[np.float64]
    strategy: str

class SweepOutcome(TypedDict):
    kind: Literal["dc_sweep"]
    source: str
    quantity: Literal["voltage", "current"]
    levels: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    logic: npt.NDArray[np.float64]

class ResponseOutcome(TypedDict):
    kind: Literal["ac"]
    source: str
    frequencies: npt.NDArray[np.float64]
    values: npt.NDArray[np.complex128]
    operating_point: SolvedPoint

class StatisticsOutcome(TypedDict):
    initial: str
    accepted: int
    rejected: int
    iterations: int

class WaveformsOutcome(TypedDict):
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
    derivatives: npt.NDArray[np.float64]
    operating_point: SolvedPoint

type Outcome = (
    OperatingPointOutcome
    | SweepOutcome
    | ResponseOutcome
    | WaveformsOutcome
    | SensitivityOutcome
)

class SourceOutcome(TypedDict):
    file: str
    line: int
    function: str | None

class InstanceOutcome(TypedDict):
    path: str
    source: SourceOutcome | None

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

class Simulator:
    def __init__(
        self, circuit: str, temperature: float, overrides: str, substitutes: str
    ) -> None: ...
    @property
    def unknowns(self) -> list[tuple[str, Literal["voltage", "current"]]]: ...
    @property
    def nets(self) -> list[str]: ...
    @property
    def devices(self) -> int: ...
    def check(self) -> list[DiagnosticOutcome]: ...
    def excursions(self) -> list[ExcursionOutcome]: ...
    def run(self, analysis: str, options: str) -> Outcome: ...

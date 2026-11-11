# Copyright 2026 Maxence Faldor
# The types of the Rust extension module; its documentation lives with the Rust code.

from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt

__version__: str

class ElaborationError(Exception): ...
class SimulationError(Exception): ...

class SolvedPoint(TypedDict):
    values: npt.NDArray[np.float64]
    strategy: str

class OperatingPointOutcome(TypedDict):
    kind: Literal["operating_point"]
    values: npt.NDArray[np.float64]
    strategy: str

class SweepOutcome(TypedDict):
    kind: Literal["dc_sweep"]
    levels: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]

class ResponseOutcome(TypedDict):
    kind: Literal["ac"]
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
    statistics: StatisticsOutcome

type Outcome = OperatingPointOutcome | SweepOutcome | ResponseOutcome | WaveformsOutcome

class Simulator:
    def __init__(self, circuit: str) -> None: ...
    @property
    def unknowns(self) -> list[tuple[str, Literal["voltage", "current"]]]: ...
    @property
    def devices(self) -> int: ...
    def run(self, analysis: str, options: str) -> Outcome: ...

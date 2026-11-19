# Copyright 2026 Maxence Faldor
"""jolty: a compiler and simulator for electronic circuits, built for AI agents.

Circuits are authored as Python components using [`component`][jolty.builder.component]
and [`Builder`][jolty.builder.Builder], with physical values in
[`jolty.units`][jolty.units]. A [`Simulation`][jolty.simulate.Simulation] runs
[`jolty.plan`][jolty.plan] analyses in the Rust engine, and
[`jolty.specs`][jolty.specs] defines pass/fail criteria for the results.
"""

__lazy_modules__ = [
    "jolty",
    "jolty.builder",
    "jolty.errors",
    "jolty.ir",
    "jolty.plan",
    "jolty.simulate",
    "jolty.specs",
]

from importlib.metadata import version

from jolty import devices, files, parts, plan, specs, spice, units, waveforms
from jolty.builder import (
    Builder,
    Component,
    Connectable,
    Net,
    Placed,
    Terminal,
    circuit,
    component,
)
from jolty.errors import (
    BuildError,
    CharacterizationError,
    DimensionError,
    DownloadError,
    ElaborationError,
    JoltyError,
    PlanError,
    ScriptError,
    SimulationError,
    SpiceError,
)
from jolty.ir import Role
from jolty.plan import (
    Ac,
    Characterized,
    Corner,
    DcSweep,
    Mismatch,
    MonteCarlo,
    Noise,
    OperatingPoint,
    OperatingPoints,
    Options,
    PeriodicSteadyState,
    Plan,
    PoleZero,
    Sensitivity,
    Stability,
    TransferFunction,
    Transient,
    WorstCase,
)
from jolty.simulate import Simulation, function, jvp, vjp
from jolty.specs import Specs

__version__ = version("jolty")

__all__ = [
    "Ac",
    "BuildError",
    "Builder",
    "CharacterizationError",
    "Characterized",
    "Component",
    "Connectable",
    "Corner",
    "DcSweep",
    "DimensionError",
    "DownloadError",
    "ElaborationError",
    "JoltyError",
    "Mismatch",
    "MonteCarlo",
    "Net",
    "Noise",
    "OperatingPoint",
    "OperatingPoints",
    "Options",
    "PeriodicSteadyState",
    "Placed",
    "Plan",
    "PlanError",
    "PoleZero",
    "Role",
    "ScriptError",
    "Sensitivity",
    "Simulation",
    "SimulationError",
    "Specs",
    "SpiceError",
    "Stability",
    "Terminal",
    "TransferFunction",
    "Transient",
    "WorstCase",
    "__version__",
    "circuit",
    "component",
    "devices",
    "files",
    "function",
    "jvp",
    "parts",
    "plan",
    "specs",
    "spice",
    "units",
    "vjp",
    "waveforms",
]

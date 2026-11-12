# Copyright 2026 Maxence Faldor
"""jolty: a compiler and simulator for electronic circuits, built for AI agents.

Circuits are written as Python components with [`component`][jolty.builder.component]
and a [`Builder`][jolty.builder.Builder], with values in the units of
[`jolty.units`][jolty.units]. A [`Simulation`][jolty.simulate.Simulation] runs the
analyses of [`jolty.plan`][jolty.plan] on them with the Rust engine, and
[`jolty.specs`][jolty.specs] states what their results must show.
"""

from importlib.metadata import version

from jolty import devices, files, plan, specs, spice, units, waveforms
from jolty.builder import Builder, Component, Net, Placed, Terminal, circuit, component
from jolty.errors import (
    BuildError,
    CharacterizationError,
    DimensionError,
    DownloadError,
    ElaborationError,
    JoltyError,
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
    MonteCarlo,
    Noise,
    OperatingPoint,
    OperatingPoints,
    Options,
    Plan,
    PoleZero,
    Sensitivity,
    Stability,
    Transient,
    WorstCase,
)
from jolty.simulate import Simulation
from jolty.specs import Specs

__version__ = version("jolty")

__all__ = [
    "Ac",
    "BuildError",
    "Builder",
    "CharacterizationError",
    "Characterized",
    "Component",
    "Corner",
    "DcSweep",
    "DimensionError",
    "DownloadError",
    "ElaborationError",
    "JoltyError",
    "MonteCarlo",
    "Net",
    "Noise",
    "OperatingPoint",
    "OperatingPoints",
    "Options",
    "Placed",
    "Plan",
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
    "Transient",
    "WorstCase",
    "__version__",
    "circuit",
    "component",
    "devices",
    "files",
    "plan",
    "specs",
    "spice",
    "units",
    "waveforms",
]

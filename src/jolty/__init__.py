# Copyright 2026 Maxence Faldor
"""jolty: a compiler and simulator for electronic circuits, built for AI agents.

Circuits are written as Python components with [`component`][jolty.builder.component]
and a [`Builder`][jolty.builder.Builder], with values in the units of
[`jolty.units`][jolty.units]. A [`Simulation`][jolty.simulate.Simulation] runs the
analyses of [`jolty.plan`][jolty.plan] on them with the Rust engine.
"""

from importlib.metadata import version

from jolty import devices, files, plan, spice, units, waveforms
from jolty.builder import Builder, Component, Net, Placed, Terminal, circuit, component
from jolty.errors import (
    BuildError,
    DimensionError,
    DownloadError,
    ElaborationError,
    JoltyError,
    ScriptError,
    SimulationError,
    SpiceError,
)
from jolty.ir import Role
from jolty.plan import Ac, DcSweep, OperatingPoint, Options, Plan, Transient
from jolty.simulate import Simulation

__version__ = version("jolty")

__all__ = [
    "Ac",
    "BuildError",
    "Builder",
    "Component",
    "DcSweep",
    "DimensionError",
    "DownloadError",
    "ElaborationError",
    "JoltyError",
    "Net",
    "OperatingPoint",
    "Options",
    "Placed",
    "Plan",
    "Role",
    "ScriptError",
    "Simulation",
    "SimulationError",
    "SpiceError",
    "Terminal",
    "Transient",
    "__version__",
    "circuit",
    "component",
    "devices",
    "files",
    "plan",
    "spice",
    "units",
    "waveforms",
]

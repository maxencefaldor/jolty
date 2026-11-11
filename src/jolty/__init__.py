# Copyright 2026 Maxence Faldor
"""jolty: a compiler and simulator for electronic circuits, built for AI agents.

Circuits are written as Python components with :func:`component` and a
:class:`Builder`, with values in the units of :mod:`jolty.units`. A :class:`Simulation`
runs the analyses of :mod:`jolty.plan` on them with the Rust engine.
"""

from importlib.metadata import version

from jolty import devices, files, plan, units, waveforms
from jolty.builder import Builder, Component, Net, Placed, Terminal, circuit, component
from jolty.errors import (
    BuildError,
    DimensionError,
    ElaborationError,
    JoltyError,
    ScriptError,
    SimulationError,
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
    "Terminal",
    "Transient",
    "__version__",
    "circuit",
    "component",
    "devices",
    "files",
    "plan",
    "units",
    "waveforms",
]

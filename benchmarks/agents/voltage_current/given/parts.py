# Copyright 2026 Maxence Faldor
"""The two transistors of the voltage-current converter: a 2N3904 and a 2N3906."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from discrete_parts import npn, pnp

__all__ = ["npn", "pnp"]

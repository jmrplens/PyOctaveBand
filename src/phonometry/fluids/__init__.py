#  Copyright (c) 2026. Jose Manuel Requena Plens
"""fluids domain of phonometry (see module docstrings).

The state of the medium a sound travels through, computed from the conditions
that were measured rather than assumed. Every other package may import this one
without an architecture edge, the way they import :mod:`phonometry.filters`,
:mod:`phonometry.signals` and :mod:`phonometry.metrology`: a medium is not a
domain of application but something every domain needs, and eleven identical
edges would record nothing.

What lives here is the *physics* of a fluid. A simplified formula a measurement
standard prints inside its own procedure stays in that standard's module, where
its clause can be cited beside it, and a constant frozen by a conformance row
never moves at all. Those three are different things, and keeping them apart is
what lets better physics reach a caller without any measurement silently
ceasing to reproduce the standard it claims.
"""

from __future__ import annotations

from ._state import (
    Fluid,
    FluidAssumptionWarning,
    FluidPropertyUnavailable,
    FluidWarning,
    characteristic_impedance,
)
from .air import (
    DEFAULT_CO2_MOLE_FRACTION,
    DEFAULT_RELATIVE_HUMIDITY_PERCENT,
    DEFAULT_STATIC_PRESSURE_PA,
    air,
)

__all__ = [
    "DEFAULT_CO2_MOLE_FRACTION",
    "DEFAULT_RELATIVE_HUMIDITY_PERCENT",
    "DEFAULT_STATIC_PRESSURE_PA",
    "Fluid",
    "FluidAssumptionWarning",
    "FluidPropertyUnavailable",
    "FluidWarning",
    "air",
    "characteristic_impedance",
]

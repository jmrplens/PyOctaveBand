#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Compatibility shim for the documentation figure/animation generators.

The 2D acoustic FDTD engine that used to live here was promoted to the
public library as :mod:`phonometry.simulation.fdtd`; this module re-exports
it so the committed animation scripts keep importing ``fdtd2d`` unchanged.
The physics (grid setup, leapfrog stepping, sources, sponge layers) is the
library code itself, so the committed clips only change when the library
changes.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")))

from phonometry.simulation.fdtd import (
    FDTD2D,
    CWSource,
    Field2D,
    GaussianPulse,
    PlaneWaveSource,
    SignalSource,
    Source,
    fdtd_simulation,
)

__all__ = [
    "FDTD2D",
    "CWSource",
    "Field2D",
    "GaussianPulse",
    "PlaneWaveSource",
    "SignalSource",
    "Source",
    "fdtd_simulation",
]


#  Copyright (c) 2026. Jose M. Requena-Plens
"""simulation domain of phonometry (see module docstrings)."""

from __future__ import annotations

from .fdtd import (
    FDTD2D,
    CWSource,
    FDTDResult,
    GaussianPulse,
    PlaneWaveSource,
    SignalSource,
    fdtd_simulation,
)

__all__ = [
    "FDTD2D",
    "CWSource",
    "FDTDResult",
    "GaussianPulse",
    "PlaneWaveSource",
    "SignalSource",
    "fdtd_simulation",
]

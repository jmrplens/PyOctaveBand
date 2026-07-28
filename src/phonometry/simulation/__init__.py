#  Copyright (c) 2026. Jose M. Requena-Plens
"""simulation domain of phonometry (see module docstrings)."""

from __future__ import annotations

from .elastic_fdtd import (
    AIR,
    ALUMINIUM,
    CONCRETE,
    STEEL,
    WATER,
    ElasticFDTD2D,
    ElasticFDTDResult,
    ExplosionSource,
    ForceSource,
    Material,
    elastic_fdtd_simulation,
    scholte_speed,
)
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
    "AIR",
    "ALUMINIUM",
    "CONCRETE",
    "FDTD2D",
    "STEEL",
    "WATER",
    "CWSource",
    "ElasticFDTD2D",
    "ElasticFDTDResult",
    "ExplosionSource",
    "FDTDResult",
    "ForceSource",
    "GaussianPulse",
    "Material",
    "PlaneWaveSource",
    "SignalSource",
    "elastic_fdtd_simulation",
    "fdtd_simulation",
    "scholte_speed",
]

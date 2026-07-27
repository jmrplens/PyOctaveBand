#  Copyright (c) 2026. Jose M. Requena-Plens
"""noise_control domain of phonometry (see module docstrings)."""

from __future__ import annotations

from .._plot.geometry import (
    plot_plenum_geometry,
    plot_silencer_geometry,
)
from .enclosures import EnclosureResult, enclosure_insertion_loss
from .hvac import (
    HvacSpectrumResult,
    elbow_insertion_loss,
    end_reflection_loss,
    flow_noise_bend,
    flow_noise_straight_duct,
    plenum_attenuation,
)
from .silencers import (
    ReactiveSilencerResult,
    cascade,
    duct_matrix,
    expansion_chamber,
    extended_tube_chamber,
    helmholtz_impedance,
    helmholtz_resonator,
    insertion_loss,
    quarter_wave_impedance,
    quarter_wave_resonator,
    shunt_matrix,
    transmission_loss,
)

__all__ = [
    "EnclosureResult",
    "HvacSpectrumResult",
    "ReactiveSilencerResult",
    "cascade",
    "duct_matrix",
    "elbow_insertion_loss",
    "enclosure_insertion_loss",
    "end_reflection_loss",
    "expansion_chamber",
    "extended_tube_chamber",
    "flow_noise_bend",
    "flow_noise_straight_duct",
    "helmholtz_impedance",
    "helmholtz_resonator",
    "insertion_loss",
    "plenum_attenuation",
    "plot_plenum_geometry",
    "plot_silencer_geometry",
    "quarter_wave_impedance",
    "quarter_wave_resonator",
    "shunt_matrix",
    "transmission_loss",
]

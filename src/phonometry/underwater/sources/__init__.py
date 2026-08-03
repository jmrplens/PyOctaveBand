#  Copyright (c) 2026. Jose Manuel Requena Plens
"""underwater.sources subdomain of phonometry: what makes the sound.

Ship radiated noise (ISO 17208), shipping traffic, impact pile driving and the
wind and thermal ambient noise the sea makes on its own.
"""

from __future__ import annotations

from .ambient_noise import (
    AmbientNoiseResult,
    ocean_ambient_noise,
    thermal_noise_spectrum,
    wind_noise_spectrum,
)
from .pile_driving_noise import (
    PileStrikeResult,
    StrikeSelSpectrum,
    cumulative_sel,
    cumulative_sel_identical,
    pile_strike_metrics,
    single_strike_sel,
    strike_sel_spectrum,
)
from .ship_radiated_noise import (
    ShipSourceLevelResult,
    hydrophone_depths,
    monopole_source_level,
    radiated_noise_level,
    source_level_uncertainty,
)
from .ship_traffic_noise import (
    VESSEL_CLASSES,
    ShipTrafficSpectrum,
    ship_source_spectrum,
)

__all__ = [
    "VESSEL_CLASSES",
    "AmbientNoiseResult",
    "PileStrikeResult",
    "ShipSourceLevelResult",
    "ShipTrafficSpectrum",
    "StrikeSelSpectrum",
    "cumulative_sel",
    "cumulative_sel_identical",
    "hydrophone_depths",
    "monopole_source_level",
    "ocean_ambient_noise",
    "pile_strike_metrics",
    "radiated_noise_level",
    "ship_source_spectrum",
    "single_strike_sel",
    "source_level_uncertainty",
    "strike_sel_spectrum",
    "thermal_noise_spectrum",
    "wind_noise_spectrum",
]

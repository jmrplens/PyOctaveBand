#  Copyright (c) 2026. Jose Manuel Requena Plens
"""aircraft domain of phonometry (see module docstrings)."""

from __future__ import annotations

from .._compat import _namespace_dir, _namespace_shim
from .airport_noise import (
    FlyoverResult,
    NoiseContourResult,
    NpdLevelResult,
    duration_correction,
    engine_installation_correction,
    event_level,
    impedance_adjustment,
    lateral_attenuation,
    noise_contour,
    noise_fraction,
    npd_curve,
    npd_level,
    start_of_roll_directivity,
)
from .anp_fleet import (
    AnpAircraft,
    AnpDatabase,
    AnpNpdCurves,
    AnpProfile,
    load_anp_database,
)
from .atmospheric_absorption import AircraftBandAttenuation, sae_band_attenuation
from .certification import (
    NOY_BANDS,
    EPNLResult,
    effective_perceived_noise_level,
    epnl_from_pnlt,
    perceived_noise_level,
    perceived_noisiness,
    tone_correction,
)
from .measurement_system import verify_aircraft_noise_system
from .rotorcraft_noise import (
    FlightPathKinematics,
    MeanGroundPlaneResult,
    RotorcraftEventResult,
    RotorcraftHemisphere,
    RotorcraftNoiseContourResult,
    TerrainScreeningResult,
    atmospheric_adjustment,
    diffraction_attenuation,
    flight_condition_weights,
    flight_path_kinematics,
    ground_effect_adjustment,
    hemisphere_source_level,
    interpolated_source_level,
    mean_flow_resistivity,
    mean_ground_plane,
    rotorcraft_event_level,
    rotorcraft_noise_contour,
    spherical_spreading_adjustment,
    terrain_screening_adjustment,
)

__all__ = [
    "NOY_BANDS",
    "AircraftBandAttenuation",
    "AnpAircraft",
    "AnpDatabase",
    "AnpNpdCurves",
    "AnpProfile",
    "EPNLResult",
    "FlightPathKinematics",
    "FlyoverResult",
    "MeanGroundPlaneResult",
    "NoiseContourResult",
    "NpdLevelResult",
    "RotorcraftEventResult",
    "RotorcraftHemisphere",
    "RotorcraftNoiseContourResult",
    "TerrainScreeningResult",
    "atmospheric_adjustment",
    "diffraction_attenuation",
    "duration_correction",
    "effective_perceived_noise_level",
    "engine_installation_correction",
    "epnl_from_pnlt",
    "event_level",
    "flight_condition_weights",
    "flight_path_kinematics",
    "ground_effect_adjustment",
    "hemisphere_source_level",
    "impedance_adjustment",
    "interpolated_source_level",
    "lateral_attenuation",
    "load_anp_database",
    "mean_flow_resistivity",
    "mean_ground_plane",
    "noise_contour",
    "noise_fraction",
    "npd_curve",
    "npd_level",
    "perceived_noise_level",
    "perceived_noisiness",
    "rotorcraft_event_level",
    "rotorcraft_noise_contour",
    "sae_band_attenuation",
    "spherical_spreading_adjustment",
    "start_of_roll_directivity",
    "terrain_screening_adjustment",
    "tone_correction",
    "verify_aircraft_noise_system",
]

#: No public name left this namespace in 4.0, but the modules did, so
#: ``aircraft.aircraft_noise`` has to keep resolving to its alias module until 5.0: the
#: import registers it, the attribute read needs this.
__getattr__ = _namespace_shim(__name__)
__dir__ = _namespace_dir(__name__, __all__)

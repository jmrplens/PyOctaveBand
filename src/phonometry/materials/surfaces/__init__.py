#  Copyright (c) 2026. Jose Manuel Requena Plens
"""materials.surfaces subdomain of phonometry: surfaces characterised where they are.

The other three families take a specimen: a sample in a tube, a panel in a
reverberation room, a layer under a load. This one is for a surface that
cannot be sampled and has to be measured where it lies, which is a discipline
of its own with its own standards, its own geometry and its own uncertainty.
ISO 13472 and the road are what it holds today.
"""

from __future__ import annotations

from .road_absorption import (
    DEFAULT_MIC_HEIGHT,
    DEFAULT_SOURCE_HEIGHT,
    DEFAULT_SPEED_OF_SOUND,
    PART1_FREQUENCY_RANGE,
    SPOT_FREQUENCY_RANGE,
    SPOT_NARROW_BAND_RANGE,
    InsituAbsorptionResult,
    RoadAbsorptionWarning,
    absorption_reference_corrected,
    adrienne_window,
    check_spot_frequency_range,
    geometric_spreading_factor,
    geometric_spreading_factor_angle,
    insitu_absorption_coefficient,
    insitu_absorption_from_reflection,
    insitu_absorption_spectrum,
    insitu_reflection_factor,
    max_sampled_area_radius,
    msa_major_axis,
    one_third_octave_absorption,
    power_reflection_coefficient,
    reflected_path_delay,
    spot_internal_loss_correction,
    spot_microphone_spacing_bounds,
    spot_tube_upper_frequency,
)

__all__ = [
    "DEFAULT_MIC_HEIGHT",
    "DEFAULT_SOURCE_HEIGHT",
    "DEFAULT_SPEED_OF_SOUND",
    "PART1_FREQUENCY_RANGE",
    "SPOT_FREQUENCY_RANGE",
    "SPOT_NARROW_BAND_RANGE",
    "InsituAbsorptionResult",
    "RoadAbsorptionWarning",
    "absorption_reference_corrected",
    "adrienne_window",
    "check_spot_frequency_range",
    "geometric_spreading_factor",
    "geometric_spreading_factor_angle",
    "insitu_absorption_coefficient",
    "insitu_absorption_from_reflection",
    "insitu_absorption_spectrum",
    "insitu_reflection_factor",
    "max_sampled_area_radius",
    "msa_major_axis",
    "one_third_octave_absorption",
    "power_reflection_coefficient",
    "reflected_path_delay",
    "spot_internal_loss_correction",
    "spot_microphone_spacing_bounds",
    "spot_tube_upper_frequency",
]

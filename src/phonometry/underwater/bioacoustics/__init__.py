#  Copyright (c) 2026. Jose Manuel Requena Plens
"""underwater.bioacoustics subdomain of phonometry: who hears the sound.

Marine mammal audiograms and the regulatory auditory weighting and exposure
criteria built on them.
"""

from __future__ import annotations

from .audiograms import (
    AUDIOGRAM_GROUPS,
    BEST_HEARING_FREQUENCY_KHZ,
    ORCA_AUDIOGRAM_RANGE_KHZ,
    AudiogramParameters,
    AudiogramResult,
    audiogram_parameters,
    group_audiogram,
    orca_audiogram,
)
from .weighting import (
    WEIGHTING_GUIDANCE,
    AuditoryWeightingResult,
    ExposureCriteria,
    WeightedExposureResult,
    WeightingParameters,
    auditory_weighting,
    exposure_criteria,
    hearing_groups,
    weighted_exposure,
    weighting_parameters,
)

__all__ = [
    "AUDIOGRAM_GROUPS",
    "BEST_HEARING_FREQUENCY_KHZ",
    "ORCA_AUDIOGRAM_RANGE_KHZ",
    "WEIGHTING_GUIDANCE",
    "AudiogramParameters",
    "AudiogramResult",
    "AuditoryWeightingResult",
    "ExposureCriteria",
    "WeightedExposureResult",
    "WeightingParameters",
    "audiogram_parameters",
    "auditory_weighting",
    "exposure_criteria",
    "group_audiogram",
    "hearing_groups",
    "orca_audiogram",
    "weighted_exposure",
    "weighting_parameters",
]

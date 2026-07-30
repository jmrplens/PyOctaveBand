#  Copyright (c) 2026. Jose M. Requena-Plens
"""hearing domain of phonometry (see module docstrings)."""

from __future__ import annotations

from .noise_induced_hearing_loss import (
    HtlanResult,
    NiptsResult,
    NoiseInducedHearingLossWarning,
    combine_age_and_noise,
    htlan,
    nipts,
)
from .objective_intelligibility import STOIResult, stoi
from .occupational_exposure import (
    COVERAGE_FACTOR,
    INSTRUMENT_U2,
    ExposureResult,
    OccupationalExposureWarning,
    Task,
    TaskContribution,
    full_day_exposure,
    job_based_exposure,
    minimum_cumulative_duration_hours,
    table_c4_contribution,
    task_based_exposure,
)
from .sii import (
    SII_METHODS,
    SIIProcedure,
    SIIResult,
    StandardSpeechSpectrum,
    sii_procedure,
    speech_intelligibility_index,
    standard_speech_spectra,
    standard_speech_spectrum,
)
from .sti import STIResult, STIWarning, sti_from_impulse_response, stipa, stipa_signal
from .threshold import (
    AUDIOMETRIC_FREQUENCIES,
    FIELDS,
    SEXES,
    AgeThresholdResult,
    age_threshold,
    reference_threshold,
)

__all__ = [
    "AUDIOMETRIC_FREQUENCIES",
    "COVERAGE_FACTOR",
    "FIELDS",
    "INSTRUMENT_U2",
    "SEXES",
    "SII_METHODS",
    "AgeThresholdResult",
    "ExposureResult",
    "HtlanResult",
    "NiptsResult",
    "NoiseInducedHearingLossWarning",
    "OccupationalExposureWarning",
    "SIIProcedure",
    "SIIResult",
    "STIResult",
    "STIWarning",
    "STOIResult",
    "StandardSpeechSpectrum",
    "Task",
    "TaskContribution",
    "age_threshold",
    "combine_age_and_noise",
    "full_day_exposure",
    "htlan",
    "job_based_exposure",
    "minimum_cumulative_duration_hours",
    "nipts",
    "reference_threshold",
    "sii_procedure",
    "speech_intelligibility_index",
    "standard_speech_spectra",
    "standard_speech_spectrum",
    "sti_from_impulse_response",
    "stipa",
    "stipa_signal",
    "stoi",
    "table_c4_contribution",
    "task_based_exposure",
]

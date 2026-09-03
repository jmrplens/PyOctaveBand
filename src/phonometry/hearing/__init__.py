#  Copyright (c) 2026. Jose Manuel Requena Plens
"""hearing domain of phonometry (see module docstrings).

Narrowed in 4.0 to hearing conservation: audiometric thresholds, noise-induced
hearing loss and occupational exposure. Speech intelligibility (STI, SII, STOI)
moved to :mod:`phonometry.speech`, which evaluates a transmission channel
rather than an ear.
"""

from __future__ import annotations

from .hearing_protectors import (
    HML_REFERENCE_C_MINUS_A,
    HML_REFERENCE_D,
    HML_REFERENCE_NOISES,
    PINK_NOISE_A_WEIGHTED,
    PROTECTION_PERFORMANCES,
    PROTECTOR_A_WEIGHTING,
    PROTECTOR_OCTAVE_BANDS,
    AssumedProtectionResult,
    HMLRatingResult,
    ProtectedLevelResult,
    SNRRatingResult,
    assumed_protection_value,
    hml_protected_level,
    hml_rating,
    octave_band_protected_level,
    snr_protected_level,
    snr_rating,
)
from .noise_induced_hearing_loss import (
    HtlanResult,
    NiptsResult,
    NoiseInducedHearingLossWarning,
    combine_age_and_noise,
    htlan,
    nipts,
)
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
    "PROTECTOR_A_WEIGHTING",
    "COVERAGE_FACTOR",
    "FIELDS",
    "HML_REFERENCE_C_MINUS_A",
    "HML_REFERENCE_D",
    "HML_REFERENCE_NOISES",
    "INSTRUMENT_U2",
    "PROTECTOR_OCTAVE_BANDS",
    "PINK_NOISE_A_WEIGHTED",
    "PROTECTION_PERFORMANCES",
    "SEXES",
    "AgeThresholdResult",
    "AssumedProtectionResult",
    "ExposureResult",
    "HMLRatingResult",
    "HtlanResult",
    "NiptsResult",
    "NoiseInducedHearingLossWarning",
    "OccupationalExposureWarning",
    "ProtectedLevelResult",
    "SNRRatingResult",
    "Task",
    "TaskContribution",
    "age_threshold",
    "assumed_protection_value",
    "combine_age_and_noise",
    "full_day_exposure",
    "hml_protected_level",
    "hml_rating",
    "htlan",
    "job_based_exposure",
    "minimum_cumulative_duration_hours",
    "nipts",
    "octave_band_protected_level",
    "reference_threshold",
    "snr_protected_level",
    "snr_rating",
    "table_c4_contribution",
    "task_based_exposure",
]

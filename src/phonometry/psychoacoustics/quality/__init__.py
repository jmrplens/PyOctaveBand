#  Copyright (c) 2026. Jose Manuel Requena Plens
"""psychoacoustics.quality subdomain of phonometry: what a sound is like.

The attributes a listener hears beyond loudness (sharpness, roughness,
fluctuation strength, tonality and tone audibility) and the annoyance
models built on them.
"""

from __future__ import annotations

from .annoyance import (
    PsychoacousticAnnoyanceResult,
    psychoacoustic_annoyance,
    psychoacoustic_annoyance_from_signal,
)
from .fluctuation_strength import (
    FluctuationStrengthResult,
    fluctuation_strength,
    fluctuation_strength_am_noise,
)
from .fluctuation_strength_ecma import (
    EcmaFluctuationStrength,
    fluctuation_strength_ecma,
)
from .roughness_ecma import (
    EcmaRoughness,
    roughness_ecma,
)
from .sharpness import (
    sharpness_din,
    sharpness_din_from_specific,
)
from .tonality import (
    TonalityWarning,
    ToneAssessment,
    prominence_ratio,
    tone_to_noise_ratio,
)
from .tonality_ecma import (
    EcmaTonality,
    tonality_ecma,
)
from .tone_audibility import (
    HANNING_BANDWIDTH_FACTOR,
    NO_TONE_AUDIBILITY,
    ToneAudibilityResult,
    analyze_spectrum,
    assess_tones,
    audibility_from_levels,
    audibility_uncertainty,
    combined_tone_level,
    critical_band_corners,
    critical_band_level,
    critical_bandwidth_engineering,
    energy_sum_level,
    masking_index,
    mean_audibility,
    mean_audibility_uncertainty,
    mean_narrowband_level,
    resolve_tones_separately,
    tone_audibility,
    tone_level,
    two_tone_separation_frequency,
)

__all__ = [
    "HANNING_BANDWIDTH_FACTOR",
    "NO_TONE_AUDIBILITY",
    "EcmaFluctuationStrength",
    "EcmaRoughness",
    "EcmaTonality",
    "FluctuationStrengthResult",
    "PsychoacousticAnnoyanceResult",
    "TonalityWarning",
    "ToneAssessment",
    "ToneAudibilityResult",
    "analyze_spectrum",
    "assess_tones",
    "audibility_from_levels",
    "audibility_uncertainty",
    "combined_tone_level",
    "critical_band_corners",
    "critical_band_level",
    "critical_bandwidth_engineering",
    "energy_sum_level",
    "fluctuation_strength",
    "fluctuation_strength_am_noise",
    "fluctuation_strength_ecma",
    "masking_index",
    "mean_audibility",
    "mean_audibility_uncertainty",
    "mean_narrowband_level",
    "prominence_ratio",
    "psychoacoustic_annoyance",
    "psychoacoustic_annoyance_from_signal",
    "resolve_tones_separately",
    "roughness_ecma",
    "sharpness_din",
    "sharpness_din_from_specific",
    "tonality_ecma",
    "tone_audibility",
    "tone_level",
    "tone_to_noise_ratio",
    "two_tone_separation_frequency",
]

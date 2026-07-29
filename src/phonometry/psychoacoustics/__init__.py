#  Copyright (c) 2026. Jose M. Requena-Plens
"""psychoacoustics domain of phonometry (see module docstrings)."""

from __future__ import annotations

from .erb_scale import (
    CAM_C,
    ERB_C1,
    ERB_C2,
    cam_from_frequency,
    erb_bandwidth,
    frequency_from_cam,
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
from .loudness_contours import (
    EqualLoudnessContours,
    equal_loudness_contour,
    equal_loudness_contours,
    hearing_threshold,
    loudness_level,
)
from .loudness_ecma import EcmaLoudness, loudness_ecma
from .loudness_moore_glasberg import (
    MooreGlasbergLoudness,
    loudness_moore_glasberg,
    loudness_moore_glasberg_from_spectrum,
    loudness_moore_glasberg_from_third_octave,
)
from .loudness_moore_glasberg_time import (
    MooreGlasbergTimeVaryingLoudness,
    loudness_moore_glasberg_time,
)
from .loudness_zwicker import (
    ZwickerLoudness,
    loudness_zwicker,
    loudness_zwicker_from_spectrum,
)
from .psychoacoustic_annoyance import (
    PsychoacousticAnnoyanceResult,
    psychoacoustic_annoyance,
    psychoacoustic_annoyance_from_signal,
)
from .roughness_ecma import EcmaRoughness, roughness_ecma
from .sharpness import sharpness_din, sharpness_din_from_specific
from .tonality import (
    TonalityWarning,
    ToneAssessment,
    prominence_ratio,
    tone_to_noise_ratio,
)
from .tonality_ecma import EcmaTonality, tonality_ecma
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
    "CAM_C",
    "ERB_C1",
    "ERB_C2",
    "HANNING_BANDWIDTH_FACTOR",
    "NO_TONE_AUDIBILITY",
    "EcmaFluctuationStrength",
    "EcmaLoudness",
    "EcmaRoughness",
    "EcmaTonality",
    "EqualLoudnessContours",
    "FluctuationStrengthResult",
    "MooreGlasbergLoudness",
    "MooreGlasbergTimeVaryingLoudness",
    "PsychoacousticAnnoyanceResult",
    "TonalityWarning",
    "ToneAssessment",
    "ToneAudibilityResult",
    "ZwickerLoudness",
    "analyze_spectrum",
    "assess_tones",
    "audibility_from_levels",
    "audibility_uncertainty",
    "cam_from_frequency",
    "combined_tone_level",
    "critical_band_corners",
    "critical_band_level",
    "critical_bandwidth_engineering",
    "energy_sum_level",
    "equal_loudness_contour",
    "equal_loudness_contours",
    "erb_bandwidth",
    "fluctuation_strength",
    "fluctuation_strength_am_noise",
    "fluctuation_strength_ecma",
    "frequency_from_cam",
    "hearing_threshold",
    "loudness_ecma",
    "loudness_level",
    "loudness_moore_glasberg",
    "loudness_moore_glasberg_from_spectrum",
    "loudness_moore_glasberg_from_third_octave",
    "loudness_moore_glasberg_time",
    "loudness_zwicker",
    "loudness_zwicker_from_spectrum",
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

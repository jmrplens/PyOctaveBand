#  Copyright (c) 2026. Jose Manuel Requena Plens
"""psychoacoustics domain of phonometry (see module docstrings).

Two families since 4.0, along the split every textbook on sound quality
makes: :mod:`~phonometry.psychoacoustics.loudness` for how loud a sound is
(the four models and the contours they are read against) and
:mod:`~phonometry.psychoacoustics.quality` for what it is like once its
loudness is known (sharpness, roughness, fluctuation strength, tonality and
the annoyance models built on them). The ERB scale stays at the root: both
families measure on it. Every public name is still exported here, so
``from phonometry import psychoacoustics`` reads as it did.
"""

from __future__ import annotations

from .._compat import _namespace_dir, _namespace_shim
from .erb_scale import (
    CAM_C,
    ERB_C1,
    ERB_C2,
    cam_from_frequency,
    erb_bandwidth,
    frequency_from_cam,
)
from .loudness import (
    EcmaLoudness,
    EqualLoudnessContours,
    MooreGlasbergLoudness,
    MooreGlasbergTimeVaryingLoudness,
    ZwickerLoudness,
    equal_loudness_contour,
    equal_loudness_contours,
    hearing_threshold,
    loudness_ecma,
    loudness_level,
    loudness_moore_glasberg,
    loudness_moore_glasberg_from_spectrum,
    loudness_moore_glasberg_from_third_octave,
    loudness_moore_glasberg_time,
    loudness_zwicker,
    loudness_zwicker_from_spectrum,
)
from .quality import (
    HANNING_BANDWIDTH_FACTOR,
    NO_TONE_AUDIBILITY,
    EcmaFluctuationStrength,
    EcmaRoughness,
    EcmaTonality,
    FluctuationStrengthResult,
    PsychoacousticAnnoyanceResult,
    TonalityWarning,
    ToneAssessment,
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
    fluctuation_strength,
    fluctuation_strength_am_noise,
    fluctuation_strength_ecma,
    masking_index,
    mean_audibility,
    mean_audibility_uncertainty,
    mean_narrowband_level,
    prominence_ratio,
    psychoacoustic_annoyance,
    psychoacoustic_annoyance_from_signal,
    resolve_tones_separately,
    roughness_ecma,
    sharpness_din,
    sharpness_din_from_specific,
    tonality_ecma,
    tone_audibility,
    tone_level,
    tone_to_noise_ratio,
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

#: No public name left this namespace in 4.0, but the modules did, so
#: ``psychoacoustics.loudness_zwicker`` has to keep resolving to its alias
#: module until 5.0: the import registers it, the attribute read needs this.
__getattr__ = _namespace_shim(__name__)
__dir__ = _namespace_dir(__name__, __all__)

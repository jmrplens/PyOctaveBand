#  Copyright (c) 2026. Jose Manuel Requena Plens
"""psychoacoustics.loudness subdomain of phonometry: how loud a sound is.

The four loudness models (ISO 532-1 Zwicker, ISO 532-2 Moore-Glasberg,
ISO 532-3 time-varying Moore-Glasberg and ECMA-418-2) and the ISO 226
equal-loudness contours that put a level in phons.
"""

from __future__ import annotations

from .contours import (
    EqualLoudnessContours,
    equal_loudness_contour,
    equal_loudness_contours,
    hearing_threshold,
    loudness_level,
)
from .ecma import (
    EcmaLoudness,
    loudness_ecma,
)
from .moore_glasberg import (
    MooreGlasbergLoudness,
    loudness_moore_glasberg,
    loudness_moore_glasberg_from_spectrum,
    loudness_moore_glasberg_from_third_octave,
)
from .moore_glasberg_time import (
    MooreGlasbergTimeVaryingLoudness,
    loudness_moore_glasberg_time,
)
from .zwicker import (
    ZwickerLoudness,
    loudness_zwicker,
    loudness_zwicker_from_spectrum,
)

__all__ = [
    "EcmaLoudness",
    "EqualLoudnessContours",
    "MooreGlasbergLoudness",
    "MooreGlasbergTimeVaryingLoudness",
    "ZwickerLoudness",
    "equal_loudness_contour",
    "equal_loudness_contours",
    "hearing_threshold",
    "loudness_ecma",
    "loudness_level",
    "loudness_moore_glasberg",
    "loudness_moore_glasberg_from_spectrum",
    "loudness_moore_glasberg_from_third_octave",
    "loudness_moore_glasberg_time",
    "loudness_zwicker",
    "loudness_zwicker_from_spectrum",
]

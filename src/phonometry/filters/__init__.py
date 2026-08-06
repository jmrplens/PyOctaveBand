#  Copyright (c) 2026. Jose Manuel Requena Plens
"""filters domain of phonometry (see module docstrings)."""

from __future__ import annotations

from .._compat import _namespace_dir, _namespace_shim
from .compliance import (
    FilterComplianceResult,
    class_limits,
    filter_class_compliance,
    verify_filter_class,
)
from .core import (
    BlockProcessing,
    FilterBankWarning,
    FilterDesign,
    LevelCalibration,
    OctaveFilterBank,
    ResponsePlot,
    octave_filter,
)
from .equalizer import EQResponseResult, EQSection, ParametricEQ, parametric_eq
from .frequencies import (
    nominal_frequencies,
    normalized_frequencies,
)
from .weighting import (
    TimeWeighting,
    WeightingFilter,
    linkwitz_riley,
    time_weighting,
    weighting_filter,
)
from .weighting_compliance import verify_weighting_class, weighting_class_limits

__all__ = [
    "BlockProcessing",
    "EQResponseResult",
    "EQSection",
    "FilterBankWarning",
    "FilterComplianceResult",
    "FilterDesign",
    "LevelCalibration",
    "OctaveFilterBank",
    "ParametricEQ",
    "ResponsePlot",
    "TimeWeighting",
    "WeightingFilter",
    "class_limits",
    "filter_class_compliance",
    "linkwitz_riley",
    "nominal_frequencies",
    "normalized_frequencies",
    "octave_filter",
    "parametric_eq",
    "time_weighting",
    "verify_filter_class",
    "verify_weighting_class",
    "weighting_class_limits",
    "weighting_filter",
]

#: The IEC 61265 aircraft measurement-system check verifies a certification
#: chain, so it moved to :mod:`phonometry.aircraft`; reading it from here
#: still works until 5.0.
_MOVED_TO = ("phonometry.aircraft",)
_MOVED_NAMES: dict[str, tuple[str, ...]] = {"phonometry.aircraft": ("verify_aircraft_noise_system",)}
__getattr__ = _namespace_shim(__name__, _MOVED_TO, only=_MOVED_NAMES)
__dir__ = _namespace_dir(__name__, __all__, _MOVED_TO, _MOVED_NAMES)

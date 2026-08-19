#  Copyright (c) 2026. Jose Manuel Requena Plens
"""filters domain of phonometry (see module docstrings)."""

from __future__ import annotations

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
    TimeWeightedEnvelope,
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
    "TimeWeightedEnvelope",
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

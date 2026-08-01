#  Copyright (c) 2026. Jose Manuel Requena Plens
"""filters domain of phonometry (see module docstrings)."""

from __future__ import annotations

from .compliance import (
    FilterComplianceResult,
    class_limits,
    filter_class_compliance,
    verify_aircraft_noise_system,
    verify_filter_class,
    verify_weighting_class,
    weighting_class_limits,
)
from .core import FilterBankWarning, OctaveFilterBank, octave_filter, octavefilter
from .equalizer import EQResponseResult, EQSection, ParametricEQ, parametric_eq
from .frequencies import (
    getansifrequencies,
    nominal_frequencies,
    normalized_frequencies,
    normalizedfreq,
)
from .weighting import (
    TimeWeighting,
    WeightingFilter,
    linkwitz_riley,
    time_weighting,
    weighting_filter,
)

__all__ = [
    "EQResponseResult",
    "EQSection",
    "FilterBankWarning",
    "FilterComplianceResult",
    "OctaveFilterBank",
    "ParametricEQ",
    "TimeWeighting",
    "WeightingFilter",
    "class_limits",
    "filter_class_compliance",
    "getansifrequencies",
    "linkwitz_riley",
    "nominal_frequencies",
    "normalized_frequencies",
    "normalizedfreq",
    "octave_filter",
    "octavefilter",
    "parametric_eq",
    "time_weighting",
    "verify_aircraft_noise_system",
    "verify_filter_class",
    "verify_weighting_class",
    "weighting_class_limits",
    "weighting_filter",
]

#  Copyright (c) 2026. Jose Manuel Requena Plens
"""metrology domain of phonometry (see module docstrings).

Narrowed in 4.0 to the transverse metrology: calibration, GUM uncertainty,
data qualification and the IEC 61043 intensity-instrument class check. The
filter banks and weightings moved to :mod:`phonometry.filters` and the general
signal analysis to :mod:`phonometry.signals`; reading either from here still
works until 5.0.
"""

from __future__ import annotations

from .._compat import _namespace_dir, _namespace_shim
from .calibration import CalibrationWarning, calculate_sensitivity, sensitivity
from .data_qualification import (
    LevelCrossingResult,
    PeakStatisticsResult,
    StationarityTestResult,
    TrendTestResult,
    level_crossing_rate,
    peak_statistics,
    stationarity_test,
    trend_test,
)
from .intensity_compliance import (
    IntensityInstrumentComplianceResult,
    instrument_class_from_components,
    intensity_class_compliance,
    phase_mismatch_from_residual_index,
    residual_index_from_phase_mismatch,
    residual_index_limits,
    verify_intensity_class,
)
from .uncertainty import (
    MonteCarloResult,
    Quantity,
    UncertaintyResult,
    UncertaintyWarning,
    combine_uncertainty,
    monte_carlo,
    rectangular,
    triangular,
    u_shaped,
)

#: Names that left this namespace in 4.0 keep resolving from here until 5.0.
_MOVED_TO = ("phonometry.filters", "phonometry.signals")
__getattr__ = _namespace_shim(__name__, _MOVED_TO)

__all__ = [
    "CalibrationWarning",
    "IntensityInstrumentComplianceResult",
    "LevelCrossingResult",
    "MonteCarloResult",
    "PeakStatisticsResult",
    "Quantity",
    "StationarityTestResult",
    "TrendTestResult",
    "UncertaintyResult",
    "UncertaintyWarning",
    "calculate_sensitivity",
    "combine_uncertainty",
    "instrument_class_from_components",
    "intensity_class_compliance",
    "level_crossing_rate",
    "monte_carlo",
    "peak_statistics",
    "phase_mismatch_from_residual_index",
    "rectangular",
    "residual_index_from_phase_mismatch",
    "residual_index_limits",
    "sensitivity",
    "stationarity_test",
    "trend_test",
    "triangular",
    "u_shaped",
    "verify_intensity_class",
]

#: ``__getattr__`` is invisible to ``dir()``; keep the moved names listed
#: while they still resolve.
__dir__ = _namespace_dir(__all__, _MOVED_TO)

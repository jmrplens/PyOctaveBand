#  Copyright (c) 2026. Jose Manuel Requena Plens
"""metrology domain of phonometry (see module docstrings).

Narrowed in 4.0 to the transverse metrology: calibration, GUM uncertainty and
data qualification. The filter banks and weightings moved to
:mod:`phonometry.filters`, the general signal analysis to
:mod:`phonometry.signals` and the IEC 61043 intensity-instrument class check
to :mod:`phonometry.emission.intensity_compliance`, which is what it verifies;
reading any of them from here still works until 5.0.
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
_MOVED_TO = (
    "phonometry.filters",
    "phonometry.signals",
    "phonometry.emission",
    "phonometry.aircraft",
)
#: ``emission`` is a domain of its own; only the IEC 61043 names came from
#: here, so it answers to ``metrology.`` for those and nothing else.
_MOVED_NAMES: dict[str, tuple[str, ...]] = {
    "phonometry.emission": (
        "IntensityInstrumentComplianceResult",
        "instrument_class_from_components",
        "intensity_class_compliance",
        "phase_mismatch_from_residual_index",
        "residual_index_from_phase_mismatch",
        "residual_index_limits",
        "verify_intensity_class",
    ),
    #: The IEC 61265 check passed through here on its way to ``aircraft``.
    "phonometry.aircraft": ("verify_aircraft_noise_system",),
}
__getattr__ = _namespace_shim(__name__, _MOVED_TO, only=_MOVED_NAMES)

__all__ = [
    "CalibrationWarning",
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
    "level_crossing_rate",
    "monte_carlo",
    "peak_statistics",
    "rectangular",
    "sensitivity",
    "stationarity_test",
    "trend_test",
    "triangular",
    "u_shaped",
]

#: ``__getattr__`` is invisible to ``dir()``; keep the moved names listed
#: while they still resolve.
__dir__ = _namespace_dir(__name__, __all__, _MOVED_TO, _MOVED_NAMES)

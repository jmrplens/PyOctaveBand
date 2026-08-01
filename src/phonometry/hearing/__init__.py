#  Copyright (c) 2026. Jose Manuel Requena Plens
"""hearing domain of phonometry (see module docstrings).

Narrowed in 4.0 to hearing conservation: audiometric thresholds, noise-induced
hearing loss and occupational exposure. Speech intelligibility (STI, SII, STOI)
moved to :mod:`phonometry.speech`, which evaluates a transmission channel
rather than an ear; reading those names from here still works until 5.0.
"""

from __future__ import annotations

from .._compat import _namespace_dir, _namespace_shim
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

#: Names that left this namespace in 4.0 keep resolving from here until 5.0.
_MOVED_TO = ("phonometry.speech",)
__getattr__ = _namespace_shim(__name__, _MOVED_TO)

__all__ = [
    "AUDIOMETRIC_FREQUENCIES",
    "COVERAGE_FACTOR",
    "FIELDS",
    "INSTRUMENT_U2",
    "SEXES",
    "AgeThresholdResult",
    "ExposureResult",
    "HtlanResult",
    "NiptsResult",
    "NoiseInducedHearingLossWarning",
    "OccupationalExposureWarning",
    "Task",
    "TaskContribution",
    "age_threshold",
    "combine_age_and_noise",
    "full_day_exposure",
    "htlan",
    "job_based_exposure",
    "minimum_cumulative_duration_hours",
    "nipts",
    "reference_threshold",
    "table_c4_contribution",
    "task_based_exposure",
]

#: ``__getattr__`` is invisible to ``dir()``; keep the moved names listed
#: while they still resolve.
__dir__ = _namespace_dir(__all__, _MOVED_TO)

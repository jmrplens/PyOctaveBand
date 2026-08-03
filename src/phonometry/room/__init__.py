#  Copyright (c) 2026. Jose Manuel Requena Plens
"""room domain of phonometry (see module docstrings)."""

from __future__ import annotations

from .._compat import _namespace_dir, _namespace_shim
from .._plot.geometry import (
    plot_open_plan_geometry,
)
from .acoustics import (
    DecayCurve,
    RoomAcousticsResult,
    decay_curve,
    room_parameters,
)
from .crowd_noise import (
    COMMUNICATION_SNR,
    NORMAL_VOICE_POWER_LEVEL,
    PRIVACY_SNR,
    TALKER_DIRECTIVITY,
    CrowdNoiseResult,
    absorption_per_table,
    crowd_noise,
    crowd_noise_level,
    speech_direct_level,
    speech_to_noise_ratio,
)
from .enclosed_space_absorption import (
    ReverberationResult,
    air_absorption_area,
    enclosed_space_reverberation,
    equivalent_absorption_area,
    hard_object_absorption,
    object_fraction,
    reverberation_time,
)
from .image_source import (
    ImageSourceResult,
    audible_image_count,
    image_source_rir,
    reflection_density,
)
from .impulse_response import (
    ImpulseResponseResult,
    ImpulseResponseWarning,
    ShapedSweepResult,
    golay_impulse_response,
    golay_pair,
    impulse_response,
    inverse_filter,
    mls_impulse_response,
    mls_signal,
    shaped_sweep_signal,
    sweep_signal,
)
from .modes import (
    MODE_KINDS,
    RoomModesResult,
    room_modal_density,
    room_mode_count,
    room_mode_frequency,
    room_modes,
)
from .noise_criteria import (
    NCResult,
    RCResult,
    nc_curve,
    noise_criterion,
    rc_curve,
    room_criterion,
)
from .open_plan import OpenPlanResult, open_plan_metrics
from .reverberation_prediction import (
    ReverberationModelResult,
    arau_puchades_reverberation_time,
    eyring_reverberation_time,
    fitzroy_reverberation_time,
    mean_absorption,
    millington_sette_reverberation_time,
    reverberation_time_models,
    sabine_reverberation_time,
)
from .steady_field import (
    SOURCE_POWER_MODELS,
    SteadyFieldResult,
    critical_distance,
    room_constant,
    schroeder_frequency,
    steady_state_field,
    steady_state_spl,
)

__all__ = [
    "COMMUNICATION_SNR",
    "MODE_KINDS",
    "NORMAL_VOICE_POWER_LEVEL",
    "PRIVACY_SNR",
    "SOURCE_POWER_MODELS",
    "TALKER_DIRECTIVITY",
    "CrowdNoiseResult",
    "DecayCurve",
    "ImageSourceResult",
    "ImpulseResponseResult",
    "ImpulseResponseWarning",
    "NCResult",
    "OpenPlanResult",
    "RCResult",
    "ReverberationModelResult",
    "ReverberationResult",
    "RoomAcousticsResult",
    "RoomModesResult",
    "ShapedSweepResult",
    "SteadyFieldResult",
    "absorption_per_table",
    "air_absorption_area",
    "arau_puchades_reverberation_time",
    "audible_image_count",
    "critical_distance",
    "crowd_noise",
    "crowd_noise_level",
    "decay_curve",
    "enclosed_space_reverberation",
    "equivalent_absorption_area",
    "eyring_reverberation_time",
    "fitzroy_reverberation_time",
    "golay_impulse_response",
    "golay_pair",
    "hard_object_absorption",
    "image_source_rir",
    "impulse_response",
    "inverse_filter",
    "mean_absorption",
    "millington_sette_reverberation_time",
    "mls_impulse_response",
    "mls_signal",
    "nc_curve",
    "noise_criterion",
    "object_fraction",
    "open_plan_metrics",
    "plot_open_plan_geometry",
    "rc_curve",
    "reflection_density",
    "reverberation_time",
    "reverberation_time_models",
    "room_constant",
    "room_criterion",
    "room_modal_density",
    "room_mode_count",
    "room_mode_frequency",
    "room_modes",
    "room_parameters",
    "sabine_reverberation_time",
    "schroeder_frequency",
    "shaped_sweep_signal",
    "speech_direct_level",
    "speech_to_noise_ratio",
    "steady_state_field",
    "steady_state_spl",
    "sweep_signal",
]

#: No public name left this namespace in 4.0, but the modules did, so
#: ``room.room_ir`` has to keep resolving to its alias module until 5.0: the
#: import registers it, the attribute read needs this.
__getattr__ = _namespace_shim(__name__)
__dir__ = _namespace_dir(__name__, __all__)

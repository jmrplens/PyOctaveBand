#  Copyright (c) 2026. Jose Manuel Requena Plens
"""electroacoustics domain of phonometry (see module docstrings)."""

from __future__ import annotations

from .._plot.geometry import (
    plot_piston_geometry,
    plot_sound_reinforcement_geometry,
)
from .distortion import (
    HarmonicDistortionResult,
    ModulationDistortionResult,
    difference_frequency_distortion,
    dynamic_intermodulation_distortion,
    dynamic_range,
    harmonic_analysis,
    harmonic_distortion,
    idle_channel_noise,
    itu_r_468_weighting,
    modulation_distortion,
    sinad,
    thd,
    thd_plus_noise,
    total_difference_frequency_distortion,
    weighted_thd,
)
from .frequency_response import FrequencyResponseResult, coherence, transfer_function
from .loudspeaker import LoudspeakerCharacteristics, loudspeaker_characteristics
from .microphone import MicrophoneCharacteristics, microphone_characteristics
from .piston import (
    PistonDirectivity,
    RadiatingPistonResult,
    piston_directivity,
    piston_directivity_pattern,
    piston_reactance,
    piston_resistance,
    radiating_piston,
)
from .sound_reinforcement import (
    CARDIOID_RELATIVE_DIRECTIVITY,
    DEFAULT_STABILITY_MARGIN,
    FeedbackStabilityResult,
    feedback_loop_gain,
    feedback_stability,
    open_microphone_correction,
)
from .swept_sine import (
    SweptSineDistortionResult,
    swept_sine_distortion,
    synchronized_sweep_signal,
)

__all__ = [
    "CARDIOID_RELATIVE_DIRECTIVITY",
    "DEFAULT_STABILITY_MARGIN",
    "FeedbackStabilityResult",
    "FrequencyResponseResult",
    "HarmonicDistortionResult",
    "LoudspeakerCharacteristics",
    "MicrophoneCharacteristics",
    "ModulationDistortionResult",
    "PistonDirectivity",
    "RadiatingPistonResult",
    "SweptSineDistortionResult",
    "coherence",
    "difference_frequency_distortion",
    "dynamic_intermodulation_distortion",
    "dynamic_range",
    "feedback_loop_gain",
    "feedback_stability",
    "harmonic_analysis",
    "harmonic_distortion",
    "idle_channel_noise",
    "itu_r_468_weighting",
    "loudspeaker_characteristics",
    "microphone_characteristics",
    "modulation_distortion",
    "open_microphone_correction",
    "piston_directivity",
    "piston_directivity_pattern",
    "piston_reactance",
    "piston_resistance",
    "plot_piston_geometry",
    "plot_sound_reinforcement_geometry",
    "radiating_piston",
    "sinad",
    "swept_sine_distortion",
    "synchronized_sweep_signal",
    "thd",
    "thd_plus_noise",
    "total_difference_frequency_distortion",
    "transfer_function",
    "weighted_thd",
]

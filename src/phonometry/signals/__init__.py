#  Copyright (c) 2026. Jose Manuel Requena Plens
"""signal domain of phonometry (see module docstrings)."""

from __future__ import annotations

from .cepstrum import (
    CepstrumResult,
    EchoDetectionResult,
    LifterResult,
    cepstrum,
    echo_detection,
    lifter,
)
from .correlation import (
    AlignedImpulseResponseResult,
    CorrelationResult,
    TimeDelayResult,
    align_impulse_responses,
    correlation,
    correlation_random_error,
    impulse_response_delay,
    time_delay,
)
from .envelope import (
    EnvelopeResult,
    EnvelopeSpectrumResult,
    envelope,
    envelope_spectrum,
)
from .inversion import InverseFilterResult, regularized_inverse_filter
from .levels import laeq, lc_peak, leq, lex_8h, ln_levels, sel, sound_exposure
from .miso import MISOCoherenceResult, miso_coherence
from .multitaper import MultitaperSpectralDensityResult, multitaper_psd
from .phase import (
    PhaseDecompositionResult,
    excess_phase,
    group_delay,
    minimum_phase,
    phase_decomposition,
)
from .spectra import (
    CoherentOutputSpectrumResult,
    CrossSpectralDensityResult,
    SpectralDensityResult,
    coherent_output_spectrum,
    cross_spectral_density,
    fractional_octave_smoothing,
    power_spectral_density,
    resolution_bias_error,
)
from .synchronous_average import (
    SynchronousAverageResult,
    comb_filter_response,
    time_synchronous_average,
)
from .test_signals import (
    ResampledSignalResult,
    ToneBurstResult,
    fractional_delay,
    noise_signal,
    resample_signal,
    tone_burst,
)
from .time_frequency import (
    SpectrogramResult,
    ZoomFFTResult,
    spectrogram,
    zoom_fft,
)
from .windows import WindowMetricsResult, window_metrics

__all__ = [
    "AlignedImpulseResponseResult",
    "CepstrumResult",
    "CoherentOutputSpectrumResult",
    "CorrelationResult",
    "CrossSpectralDensityResult",
    "EchoDetectionResult",
    "EnvelopeResult",
    "EnvelopeSpectrumResult",
    "InverseFilterResult",
    "LifterResult",
    "MISOCoherenceResult",
    "MultitaperSpectralDensityResult",
    "PhaseDecompositionResult",
    "ResampledSignalResult",
    "SpectralDensityResult",
    "SpectrogramResult",
    "SynchronousAverageResult",
    "TimeDelayResult",
    "ToneBurstResult",
    "WindowMetricsResult",
    "ZoomFFTResult",
    "align_impulse_responses",
    "cepstrum",
    "coherent_output_spectrum",
    "comb_filter_response",
    "correlation",
    "correlation_random_error",
    "cross_spectral_density",
    "echo_detection",
    "envelope",
    "envelope_spectrum",
    "excess_phase",
    "fractional_delay",
    "fractional_octave_smoothing",
    "group_delay",
    "impulse_response_delay",
    "laeq",
    "lc_peak",
    "leq",
    "lex_8h",
    "lifter",
    "ln_levels",
    "minimum_phase",
    "miso_coherence",
    "multitaper_psd",
    "noise_signal",
    "phase_decomposition",
    "power_spectral_density",
    "regularized_inverse_filter",
    "resample_signal",
    "resolution_bias_error",
    "sel",
    "sound_exposure",
    "spectrogram",
    "time_delay",
    "time_synchronous_average",
    "tone_burst",
    "window_metrics",
    "zoom_fft",
]

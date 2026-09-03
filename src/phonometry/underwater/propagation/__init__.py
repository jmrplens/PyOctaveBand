#  Copyright (c) 2026. Jose Manuel Requena Plens
"""underwater.propagation subdomain of phonometry: how the sound gets there.

The closed-form propagation loss (spreading plus seawater absorption) and
Weston's regimes, the numerical solvers (normal modes, rays, parabolic
equation), the seabed reflection they bounce off and the sound speed profile
they travel through.
"""

from __future__ import annotations

from .closed_form import (
    PropagationLossResult,
    propagation_loss,
    seawater_absorption,
    spreading_loss,
)
from .numerical import (
    BeamFan,
    EigenrayResult,
    FluidSeabed,
    GaussianBeamResult,
    NormalModeResult,
    ParabolicEquationResult,
    RayTraceResult,
    VolumeAbsorption,
    eigenrays,
    gaussian_beams,
    normal_modes,
    parabolic_equation,
    ray_trace,
)
from .seabed_reflection import (
    BottomLossResult,
    SeabedReflection,
    bottom_reflection_loss,
    critical_angle,
    reflection_coefficient,
    seabed_reflection,
)
from .sound_speed import (
    SoundSpeedProfile,
    sound_speed_profile,
)
from .weston_regimes import (
    WESTON_REGIMES,
    WESTON_SEABEDS,
    WestonPropagationResult,
    WestonRegimeBoundaries,
    WestonSeabed,
    critical_grazing_angle,
    effective_depth,
    loss_parameter,
    reflection_loss_gradient,
    waveguide_cutoff_frequency,
    weston_propagation_loss,
    weston_regime_boundaries,
)

__all__ = [
    "WESTON_REGIMES",
    "WESTON_SEABEDS",
    "BeamFan",
    "BottomLossResult",
    "EigenrayResult",
    "FluidSeabed",
    "GaussianBeamResult",
    "NormalModeResult",
    "ParabolicEquationResult",
    "PropagationLossResult",
    "RayTraceResult",
    "SeabedReflection",
    "SoundSpeedProfile",
    "VolumeAbsorption",
    "WestonPropagationResult",
    "WestonRegimeBoundaries",
    "WestonSeabed",
    "bottom_reflection_loss",
    "critical_angle",
    "critical_grazing_angle",
    "effective_depth",
    "eigenrays",
    "gaussian_beams",
    "loss_parameter",
    "normal_modes",
    "parabolic_equation",
    "propagation_loss",
    "ray_trace",
    "reflection_coefficient",
    "reflection_loss_gradient",
    "seabed_reflection",
    "seawater_absorption",
    "sound_speed_profile",
    "spreading_loss",
    "waveguide_cutoff_frequency",
    "weston_propagation_loss",
    "weston_regime_boundaries",
]

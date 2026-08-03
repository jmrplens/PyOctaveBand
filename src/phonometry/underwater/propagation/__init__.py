#  Copyright (c) 2026. Jose Manuel Requena Plens
"""underwater.propagation subdomain of phonometry: how the sound gets there.

The closed-form transmission loss (spreading plus seawater absorption) and
Weston's regimes, the numerical solvers (normal modes, rays, parabolic
equation), the seabed reflection they bounce off and the sound speed profile
they travel through.
"""

from __future__ import annotations

from .closed_form import (
    TransmissionLossResult,
    seawater_absorption,
    spreading_loss,
    transmission_loss,
)
from .numerical import (
    NormalModeResult,
    ParabolicEquationResult,
    RayTraceResult,
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
    depth_to_pressure,
    sea_water_sound_speed,
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
    "BottomLossResult",
    "NormalModeResult",
    "ParabolicEquationResult",
    "RayTraceResult",
    "SeabedReflection",
    "SoundSpeedProfile",
    "TransmissionLossResult",
    "WestonPropagationResult",
    "WestonRegimeBoundaries",
    "WestonSeabed",
    "bottom_reflection_loss",
    "critical_angle",
    "critical_grazing_angle",
    "depth_to_pressure",
    "effective_depth",
    "loss_parameter",
    "normal_modes",
    "parabolic_equation",
    "ray_trace",
    "reflection_coefficient",
    "reflection_loss_gradient",
    "sea_water_sound_speed",
    "seabed_reflection",
    "seawater_absorption",
    "sound_speed_profile",
    "spreading_loss",
    "transmission_loss",
    "waveguide_cutoff_frequency",
    "weston_propagation_loss",
    "weston_regime_boundaries",
]

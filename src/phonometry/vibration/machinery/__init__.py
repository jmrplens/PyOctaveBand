#  Copyright (c) 2026. Jose Manuel Requena Plens
"""vibration.machinery subdomain of phonometry: machinery condition monitoring."""

from __future__ import annotations

from .diagnostics import (
    FaultFrequencyResult,
    FaultLine,
    bearing_fault_frequencies,
    blade_pass_frequencies,
    combine_fault_lines,
    gear_mesh_frequencies,
    induction_motor_frequencies,
    shaft_rate,
)
from .evaluation import (
    INDUSTRIAL_MACHINE_ZONES,
    OPERATIONAL_LIMIT_HEADROOM,
    SIGNIFICANT_CHANGE_FRACTION,
    TYPICAL_BOUNDARY_LADDER_MM_S,
    TYPICAL_ZONE_BOUNDARY_RANGES_MM_S,
    ZONE_LIMIT_FACTORS,
    MachineZoneLimits,
    VectorChangeResult,
    ZoneBoundaries,
    alarm_limit,
    allowable_velocity,
    evaluation_zone,
    industrial_machine_zone,
    is_significant_change,
    trip_limit,
    vibration_vector_change,
)

__all__ = [
    "FaultFrequencyResult",
    "FaultLine",
    "INDUSTRIAL_MACHINE_ZONES",
    "MachineZoneLimits",
    "OPERATIONAL_LIMIT_HEADROOM",
    "SIGNIFICANT_CHANGE_FRACTION",
    "TYPICAL_BOUNDARY_LADDER_MM_S",
    "TYPICAL_ZONE_BOUNDARY_RANGES_MM_S",
    "VectorChangeResult",
    "ZONE_LIMIT_FACTORS",
    "ZoneBoundaries",
    "alarm_limit",
    "allowable_velocity",
    "bearing_fault_frequencies",
    "blade_pass_frequencies",
    "combine_fault_lines",
    "evaluation_zone",
    "gear_mesh_frequencies",
    "induction_motor_frequencies",
    "industrial_machine_zone",
    "is_significant_change",
    "shaft_rate",
    "trip_limit",
    "vibration_vector_change",
]

#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Deprecated module-path aliases for the phonometry 3.2 package layout.

The 3.2 release grouped the flat top-level modules into domain subpackages
(``phonometry.building``, ``phonometry.underwater``, ...). Every public module
path that moved stays importable for one deprecation cycle through the shims
registered here: ``import phonometry.<old>`` and
``from phonometry.<old> import name`` keep working, warn with the standard
rename notice on attribute access, and delegate to the relocated module.
Pickles produced by 3.1 (whose classes carry old ``__module__`` paths) resolve
the same way. The table and this module are removed in phonometry 4.0.

This generalizes the former ``phonometry.loudness`` PEP 562 shim (that module
file is gone; its entry lives in the table below with its original 3.1 wording
preserved).
"""

from __future__ import annotations

import sys
import types
from importlib import import_module
from typing import Any

from ._internal.warnings import _warn_renamed

#: Old public module path -> relocated module path. One row per moved module.
_MOVED: dict[str, str] = {
    "phonometry.utils": "phonometry._internal.utils",
    "phonometry._warnings": "phonometry._internal.warnings",
    "phonometry.calibration": "phonometry.metrology.calibration",
    "phonometry.compliance": "phonometry.metrology.compliance",
    "phonometry.core": "phonometry.metrology.core",
    "phonometry.filter_design": "phonometry.metrology.filter_design",
    "phonometry.frequencies": "phonometry.metrology.frequencies",
    "phonometry.levels": "phonometry.metrology.levels",
    "phonometry.parametric_filters": "phonometry.metrology.parametric_filters",
    "phonometry.uncertainty": "phonometry.metrology.uncertainty",
    "phonometry.fluctuation_strength": "phonometry.psychoacoustics.fluctuation_strength",
    "phonometry.loudness_contours": "phonometry.psychoacoustics.loudness_contours",
    "phonometry.loudness_ecma": "phonometry.psychoacoustics.loudness_ecma",
    "phonometry.loudness_moore_glasberg": "phonometry.psychoacoustics.loudness_moore_glasberg",
    "phonometry.loudness_moore_glasberg_time": "phonometry.psychoacoustics.loudness_moore_glasberg_time",
    "phonometry.loudness_zwicker": "phonometry.psychoacoustics.loudness_zwicker",
    "phonometry.psychoacoustic_annoyance": "phonometry.psychoacoustics.psychoacoustic_annoyance",
    "phonometry.roughness_ecma": "phonometry.psychoacoustics.roughness_ecma",
    "phonometry.sharpness": "phonometry.psychoacoustics.sharpness",
    "phonometry.tonality": "phonometry.psychoacoustics.tonality",
    "phonometry.tonality_ecma": "phonometry.psychoacoustics.tonality_ecma",
    "phonometry.tone_audibility": "phonometry.psychoacoustics.tone_audibility",
    "phonometry.noise_induced_hearing_loss": "phonometry.hearing.noise_induced_hearing_loss",
    "phonometry.occupational_exposure": "phonometry.hearing.occupational_exposure",
    "phonometry.sii": "phonometry.hearing.sii",
    "phonometry.sti": "phonometry.hearing.sti",
    "phonometry.intensity": "phonometry.emission.intensity",
    "phonometry.sound_power": "phonometry.emission.sound_power",
    "phonometry.sound_power_intensity": "phonometry.emission.sound_power_intensity",
    "phonometry.sound_power_reverberation": "phonometry.emission.sound_power_reverberation",
    "phonometry.vibration_sound_power": "phonometry.emission.vibration_sound_power",
    "phonometry.absorption_rating": "phonometry.materials.absorption_rating",
    "phonometry.absorption_uncertainty": "phonometry.materials.absorption_uncertainty",
    "phonometry.airflow_resistance": "phonometry.materials.airflow_resistance",
    "phonometry.dynamic_stiffness": "phonometry.materials.dynamic_stiffness",
    "phonometry.impedance_tube": "phonometry.materials.impedance_tube",
    "phonometry.road_absorption": "phonometry.materials.road_absorption",
    "phonometry.scattering_diffusion": "phonometry.materials.scattering_diffusion",
    "phonometry.sound_absorption": "phonometry.materials.sound_absorption",
    "phonometry.enclosed_space_absorption": "phonometry.room.enclosed_space_absorption",
    "phonometry.open_plan": "phonometry.room.open_plan",
    "phonometry.reverberation_prediction": "phonometry.room.reverberation_prediction",
    "phonometry.room_acoustics": "phonometry.room.room_acoustics",
    "phonometry.room_ir": "phonometry.room.room_ir",
    "phonometry.room_noise": "phonometry.room.room_noise",
    "phonometry.building_prediction": "phonometry.building.building_prediction",
    "phonometry.building_uncertainty": "phonometry.building.building_uncertainty",
    "phonometry.facade_prediction": "phonometry.building.facade_prediction",
    "phonometry.flanking_transmission": "phonometry.building.flanking_transmission",
    "phonometry.floor_covering_improvement": "phonometry.building.floor_covering_improvement",
    "phonometry.installed_structure_borne": "phonometry.building.installed_structure_borne",
    "phonometry.insulation": "phonometry.building.insulation",
    "phonometry.intensity_insulation": "phonometry.building.intensity_insulation",
    "phonometry.lab_insulation": "phonometry.building.lab_insulation",
    "phonometry.structure_borne_power": "phonometry.building.structure_borne_power",
    "phonometry.survey_insulation": "phonometry.building.survey_insulation",
    "phonometry.human_vibration": "phonometry.vibration.human_vibration",
    "phonometry.mechanical_mobility": "phonometry.vibration.mechanical_mobility",
    "phonometry.multiple_shock_vibration": "phonometry.vibration.multiple_shock_vibration",
    "phonometry.transfer_stiffness": "phonometry.vibration.transfer_stiffness",
    "phonometry.air_absorption": "phonometry.environmental.air_absorption",
    "phonometry.environmental_measurement": "phonometry.environmental.measurement",
    "phonometry.impulse_prominence": "phonometry.environmental.impulse_prominence",
    "phonometry.outdoor_propagation": "phonometry.environmental.outdoor_propagation",
    "phonometry.wind_turbine_noise": "phonometry.environmental.wind_turbine_noise",
    "phonometry.aircraft_atmospheric_absorption": "phonometry.aircraft.atmospheric_absorption",
    "phonometry.aircraft_noise": "phonometry.aircraft.aircraft_noise",
    "phonometry.airport_noise": "phonometry.aircraft.airport_noise",
    "phonometry.rotorcraft_noise": "phonometry.aircraft.rotorcraft_noise",
    "phonometry.numerical_propagation": "phonometry.underwater.numerical_propagation",
    "phonometry.ocean_ambient_noise": "phonometry.underwater.ocean_ambient_noise",
    "phonometry.pile_driving_noise": "phonometry.underwater.pile_driving_noise",
    "phonometry.seabed_reflection": "phonometry.underwater.seabed_reflection",
    "phonometry.ship_radiated_noise": "phonometry.underwater.ship_radiated_noise",
    "phonometry.ship_traffic_noise": "phonometry.underwater.ship_traffic_noise",
    "phonometry.sonar_equation": "phonometry.underwater.sonar_equation",
    "phonometry.underwater_acoustics": "phonometry.underwater.acoustics",
    "phonometry.underwater_propagation": "phonometry.underwater.propagation",
    "phonometry.underwater_sound_speed": "phonometry.underwater.sound_speed",
    "phonometry.distortion": "phonometry.electroacoustics.distortion",
    "phonometry.frequency_response": "phonometry.electroacoustics.frequency_response",
    # <migrate:auto>
}

#: Entries whose deprecation predates 3.2 keep their original wording.
_SINCE: dict[str, str] = {
    "phonometry.loudness": "3.1",
}

#: Renames that were already shimmed before 3.2 (target differs from a plain
#: package move). ``phonometry.loudness`` predates the reorganization.
_MOVED["phonometry.loudness"] = "phonometry.psychoacoustics.loudness_zwicker"


def _make_shim(old: str, new: str) -> types.ModuleType:
    shim = types.ModuleType(old)
    shim.__doc__ = f"Deprecated alias of :mod:`{new}` (removed in phonometry 4.0)."

    def __getattr__(name: str) -> Any:
        target = import_module(new)
        try:
            attr = getattr(target, name)
        except AttributeError:
            raise AttributeError(
                f"module {old!r} has no attribute {name!r}"
            ) from None
        _warn_renamed(
            f"the '{old}' module", f"'{new}'", since=_SINCE.get(old, "3.2")
        )
        return attr

    def __dir__() -> list[str]:
        return dir(import_module(new))

    shim.__getattr__ = __getattr__  # type: ignore[method-assign]
    shim.__dir__ = __dir__  # type: ignore[method-assign]
    return shim


def _install() -> None:
    package = sys.modules["phonometry"]
    for old, new in _MOVED.items():
        if old in sys.modules:  # pragma: no cover - double-import guard
            continue
        shim = _make_shim(old, new)
        sys.modules[old] = shim
        # `import phonometry.utils` also binds the attribute on the package;
        # mirror that so `phonometry.utils` resolves without the import.
        attr = old.rsplit(".", 1)[1]
        if not hasattr(package, attr):
            setattr(package, attr, shim)


_install()

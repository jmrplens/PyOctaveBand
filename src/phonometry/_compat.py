#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Deprecated module-path aliases for the phonometry package layout.

Two generations of aliases live here, each with its own removal date:

* :data:`_MOVED_3X` covers the 3.2 modularization, which grouped the flat
  top-level modules into domain subpackages (``phonometry.building``,
  ``phonometry.underwater``, ...). Removed in 4.0. Its targets follow the
  modules wherever they land, so a 3.x path always resolves in one hop.
* :data:`_MOVED_4X` covers the 4.0 taxonomy, which splits the oversized
  subpackages into domain ones (``phonometry.metrology`` into
  ``phonometry.filters``, ``phonometry.signals`` and a narrowed
  ``phonometry.metrology``; the speech intelligibility of
  ``phonometry.hearing`` into ``phonometry.speech``), and gives the large
  domains a second level (``phonometry.vibration`` into ``structural``,
  ``human`` and ``machinery``). Removed in 5.0.

Every public module path that moved stays importable through the shims
registered here: ``import phonometry.<old>`` and ``from phonometry.<old>
import name`` keep working, warn with the standard rename notice on attribute
access, and delegate to the relocated module. Pickles produced by an earlier
release (whose classes carry old ``__module__`` paths) resolve the same way.

The 4.0 split also moves names *between* subpackage namespaces, and importing
the domain namespace (``from phonometry import metrology``) is the form the
documentation leads with. :func:`_namespace_shim` keeps those attribute reads
working from the namespace they left, with the same notice.

This generalizes the former ``phonometry.loudness`` PEP 562 shim (that module
file is gone; its entry lives in the table below with its original 3.1 wording
preserved).
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from importlib import import_module
from typing import Any

from ._internal.warnings import _warn_renamed

#: Old public module path -> relocated module path. One row per moved module.
_MOVED_3X: dict[str, str] = {
    "phonometry.utils": "phonometry._internal.utils",
    "phonometry._warnings": "phonometry._internal.warnings",
    "phonometry.calibration": "phonometry.metrology.calibration",
    "phonometry.compliance": "phonometry.filters.compliance",
    "phonometry.core": "phonometry.filters.core",
    "phonometry.filter_design": "phonometry.filters.design",
    "phonometry.frequencies": "phonometry.filters.frequencies",
    "phonometry.levels": "phonometry.signals.levels",
    "phonometry.parametric_filters": "phonometry.filters.weighting",
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
    "phonometry.sii": "phonometry.speech.sii",
    "phonometry.sti": "phonometry.speech.sti",
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
    "phonometry.human_vibration": "phonometry.vibration.human.exposure",
    "phonometry.mechanical_mobility":
        "phonometry.vibration.structural.mechanical_mobility",
    "phonometry.multiple_shock_vibration":
        "phonometry.vibration.human.multiple_shock",
    "phonometry.transfer_stiffness":
        "phonometry.vibration.structural.transfer_stiffness",
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
_MOVED_3X["phonometry.loudness"] = "phonometry.psychoacoustics.loudness_zwicker"

#: Old module path -> relocated module path for the 4.0 taxonomy. The
#: oversized ``metrology`` catch-all became three packages: the normalized
#: frequency selectivity in ``filters``, the general signal analysis in
#: ``signal``, and the transverse metrology that gives the package its name.
_MOVED_4X: dict[str, str] = {
    "phonometry.metrology.core": "phonometry.filters.core",
    "phonometry.metrology.filter_design": "phonometry.filters.design",
    "phonometry.metrology.frequencies": "phonometry.filters.frequencies",
    "phonometry.metrology.parametric_filters": "phonometry.filters.weighting",
    "phonometry.metrology.equalizer": "phonometry.filters.equalizer",
    "phonometry.metrology.compliance": "phonometry.filters.compliance",
    "phonometry.metrology.levels": "phonometry.signals.levels",
    "phonometry.metrology.spectra": "phonometry.signals.spectra",
    "phonometry.metrology.time_frequency": "phonometry.signals.time_frequency",
    "phonometry.metrology.cepstrum": "phonometry.signals.cepstrum",
    "phonometry.metrology.correlation": "phonometry.signals.correlation",
    "phonometry.metrology.envelope": "phonometry.signals.envelope",
    "phonometry.metrology.phase": "phonometry.signals.phase",
    "phonometry.metrology.miso": "phonometry.signals.miso",
    "phonometry.metrology.inversion": "phonometry.signals.inversion",
    "phonometry.metrology.synchronous_average":
        "phonometry.signals.synchronous_average",
    "phonometry.metrology.signals": "phonometry.signals.test_signals",
    "phonometry.metrology.random_data":
        "phonometry.metrology.data_qualification",
    "phonometry.hearing.sti": "phonometry.speech.sti",
    "phonometry.hearing.sii": "phonometry.speech.sii",
    "phonometry.hearing.objective_intelligibility":
        "phonometry.speech.objective_intelligibility",
    "phonometry.vibration.mechanical_mobility":
        "phonometry.vibration.structural.mechanical_mobility",
    "phonometry.vibration.point_mobility":
        "phonometry.vibration.structural.point_mobility",
    "phonometry.vibration.junction_transmission":
        "phonometry.vibration.structural.junction_transmission",
    "phonometry.vibration.radiation_efficiency":
        "phonometry.vibration.structural.radiation_efficiency",
    "phonometry.vibration.experimental_sea":
        "phonometry.vibration.structural.experimental_sea",
    "phonometry.vibration.transfer_stiffness":
        "phonometry.vibration.structural.transfer_stiffness",
    "phonometry.vibration.human_vibration": "phonometry.vibration.human.exposure",
    "phonometry.vibration.multiple_shock_vibration":
        "phonometry.vibration.human.multiple_shock",
    "phonometry.vibration.machine_diagnostics":
        "phonometry.vibration.machinery.diagnostics",
}

#: The two generations, each with the release that deprecated it and the one
#: that removes it. Order matters only for readability; the paths are disjoint.
_GENERATIONS: tuple[tuple[dict[str, str], str, str], ...] = (
    (_MOVED_3X, "3.2", "4.0"),
    (_MOVED_4X, "4.0", "5.0"),
)


def _make_shim(old: str, new: str, since: str, removed_in: str) -> types.ModuleType:
    shim = types.ModuleType(old)
    shim.__doc__ = (
        f"Deprecated alias of :mod:`{new}` (removed in phonometry {removed_in})."
    )

    def __getattr__(name: str) -> Any:
        target = import_module(new)
        try:
            attr = getattr(target, name)
        except AttributeError:
            raise AttributeError(
                f"module {old!r} has no attribute {name!r}"
            ) from None
        _warn_renamed(
            f"the '{old}' module",
            f"'{new}'",
            since=_SINCE.get(old, since),
            removed_in=removed_in,
        )
        return attr

    def __dir__() -> list[str]:
        return dir(import_module(new))

    shim.__getattr__ = __getattr__  # type: ignore[method-assign]
    shim.__dir__ = __dir__  # type: ignore[method-assign]
    return shim


def _namespace_shim(
    package: str, targets: tuple[str, ...], *,
    since: str = "4.0", removed_in: str = "5.0"
) -> Callable[[str], Any]:
    """Return a PEP 562 ``__getattr__`` for names that left ``package``.

    The 4.0 taxonomy moves public names between subpackage namespaces, and
    ``from phonometry import metrology`` followed by ``metrology.leq(...)`` is
    the form the documentation leads with, so the read has to keep working
    from the namespace it left. Resolution is by ``__all__`` of the packages
    the names moved to, which keeps the shim honest: a name that stops being
    public anywhere stops resolving here too. A name that was both a module
    and a function resolves to the function, as the pre-split package did:
    ``metrology.cepstrum`` is :func:`phonometry.signals.cepstrum`.

    Only then does a name fall back to the module alias of the same name,
    which is what serves the modules with no public name of their own
    (``metrology.spectra``, ``metrology.levels``). The alias carries its own
    notice on attribute access, so returning it here is silent.

    :param package: The narrowed package, ``__name__`` of its ``__init__``.
    :param targets: Packages the names moved to, in search order.
    :param since: Release that moved the names.
    :param removed_in: Major release that removes the alias.
    :return: The ``__getattr__`` to bind at module level.
    """

    def __getattr__(name: str) -> Any:
        for target in targets:
            module = import_module(target)
            if name in getattr(module, "__all__", ()):
                _warn_renamed(
                    f"'{package}.{name}'",
                    f"'{target}.{name}'",
                    since=since,
                    removed_in=removed_in,
                )
                return getattr(module, name)
        alias = sys.modules.get(f"{package}.{name}")
        if alias is not None:
            return alias
        raise AttributeError(f"module {package!r} has no attribute {name!r}")

    return __getattr__


def _namespace_dir(
    own: list[str] | tuple[str, ...], targets: tuple[str, ...]
) -> Callable[[], list[str]]:
    """Return a ``__dir__`` listing the names a narrowed package still serves.

    ``__getattr__`` is invisible to :func:`dir`, so without this the moved
    names disappear from tab completion and from anything that introspects the
    namespace, one release before they stop working. ``__all__`` is left
    narrow on purpose: ``from phonometry.metrology import *`` gives the 4.0
    API, not the deprecated names.

    :param own: The package's own ``__all__``.
    :param targets: Packages the moved names went to.
    :return: The ``__dir__`` to bind at module level.
    """

    def __dir__() -> list[str]:
        names = set(own)
        for target in targets:
            names |= set(getattr(import_module(target), "__all__", ()))
        return sorted(names)

    return __dir__


def _install() -> None:
    package = sys.modules["phonometry"]
    for table, since, removed_in in _GENERATIONS:
        for old, new in table.items():
            if old in sys.modules:  # pragma: no cover - double-import guard
                continue
            shim = _make_shim(old, new, since, removed_in)
            sys.modules[old] = shim
            # `import phonometry.utils` also binds the attribute on the
            # package; mirror that so `phonometry.utils` resolves without the
            # import. Aliases below a subpackage are served by that package's
            # own shim (:func:`_namespace_shim`), which resolves the moved
            # public names first, so binding them here would shadow a function
            # with a module.
            _, _, attr = old.rpartition(".")
            if old.count(".") == 1 and attr not in vars(package):
                setattr(package, attr, shim)


_install()

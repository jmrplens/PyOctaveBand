#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Deprecated module-path aliases for the phonometry package layout.

:data:`_MOVED_4X` covers the 4.0 taxonomy, which splits the oversized
subpackages into domain ones (``phonometry.metrology`` into
``phonometry.filters``, ``phonometry.signals`` and a narrowed
``phonometry.metrology``; the speech intelligibility of
``phonometry.hearing`` into ``phonometry.speech``), gives the large domains a
second level (``phonometry.vibration`` into ``structural``, ``human`` and
``machinery``) and renames ``phonometry.environmental`` to
``phonometry.environment``. Removed in 5.0.

The 3.2 generation, which grouped the flat top-level modules into domain
subpackages, announced 4.0 as its removal and is gone: ``phonometry.levels``
and its eighty-odd siblings raise ``ModuleNotFoundError`` now. Only the module
paths went: the names they held are still exported flat, so ``from phonometry
import leq`` reads as it always has.

Every public module path that moved stays importable through the shims
registered here: ``import phonometry.<old>`` and ``from phonometry.<old>
import name`` keep working, warn with the standard rename notice on attribute
access, and delegate to the relocated module. Pickles produced by an earlier
release (whose classes carry old ``__module__`` paths) resolve the same way.

The 4.0 split also moves names *between* subpackage namespaces, and importing
the domain namespace (``from phonometry import metrology``) is the form the
documentation leads with. :func:`_namespace_shim` keeps those attribute reads
working from the namespace they left, with the same notice.

"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from importlib import import_module
from typing import Any

from ._internal.warnings import _warn_renamed

#: The one module NT ACOU 112 and ISO/PAS 1996-3 share since 4.0. Three old
#: paths land on it.
_IMPULSIVE_SOUND = "phonometry.environment.assessment.impulsive_sound"

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
    "phonometry.metrology.intensity_compliance":
        "phonometry.emission.intensity_compliance",
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
    # The package itself was renamed, so the whole prefix is an alias.
    "phonometry.environmental": "phonometry.environment",
    "phonometry.environmental.outdoor_propagation":
        "phonometry.environment.propagation.outdoor_propagation",
    "phonometry.environmental.air_absorption":
        "phonometry.environment.propagation.air_absorption",
    "phonometry.environmental.ground_barriers":
        "phonometry.environment.propagation.ground_barriers",
    "phonometry.environmental.atmospheric_refraction":
        "phonometry.environment.propagation.refraction",
    "phonometry.environmental.cnossos_road":
        "phonometry.environment.sources.cnossos_road",
    "phonometry.environmental.cnossos_rail":
        "phonometry.environment.sources.cnossos_rail",
    "phonometry.environmental.wind_turbine_noise":
        "phonometry.environment.sources.wind_turbine",
    "phonometry.environmental.rating":
        "phonometry.environment.assessment.rating",
    "phonometry.environmental.measurement":
        "phonometry.environment.assessment.measurement",
    "phonometry.environmental.impulsive_sound": _IMPULSIVE_SOUND,
    "phonometry.environmental.impulse_prominence": _IMPULSIVE_SOUND,
    "phonometry.environmental.spanish_regulation":
        "phonometry.environment.assessment.spain",
    "phonometry.building.insulation":
        "phonometry.building.measurement.insulation",
    "phonometry.building.lab_insulation":
        "phonometry.building.measurement.lab_insulation",
    "phonometry.building.survey_insulation":
        "phonometry.building.measurement.survey_insulation",
    "phonometry.building.intensity_insulation":
        "phonometry.building.measurement.intensity_insulation",
    "phonometry.building.flanking_transmission":
        "phonometry.building.measurement.flanking_transmission",
    "phonometry.building.heavy_impact":
        "phonometry.building.measurement.heavy_impact",
    "phonometry.building.floor_covering_improvement":
        "phonometry.building.measurement.floor_covering_improvement",
    "phonometry.building.structure_borne_power":
        "phonometry.building.measurement.structure_borne_power",
    "phonometry.building.building_uncertainty":
        "phonometry.building.measurement.uncertainty",
    "phonometry.building.building_prediction":
        "phonometry.building.prediction.simplified_model",
    "phonometry.building.detailed_prediction":
        "phonometry.building.prediction.detailed_model",
    "phonometry.building.facade_prediction":
        "phonometry.building.prediction.facade",
    "phonometry.building.installed_structure_borne":
        "phonometry.building.prediction.installed_structure_borne",
    "phonometry.building.panel_transmission":
        "phonometry.building.prediction.panel_transmission",
    "phonometry.building.aperture_transmission":
        "phonometry.building.prediction.aperture_transmission",
    "phonometry.building.ceiling_plenum":
        "phonometry.building.prediction.ceiling_plenum",
    "phonometry.building.masonry_cavity_wall":
        "phonometry.building.prediction.masonry_cavity_wall",
    "phonometry.building.resilient_layers":
        "phonometry.building.prediction.resilient_layers",
    "phonometry.building.spanish_building_code":
        "phonometry.building.regulation.spain",
    "phonometry.materials.sound_absorption":
        "phonometry.materials.absorbers.sound_absorption",
    "phonometry.materials.impedance_tube":
        "phonometry.materials.absorbers.impedance_tube",
    "phonometry.materials.airflow_resistance":
        "phonometry.materials.absorbers.airflow_resistance",
    "phonometry.materials.biot": "phonometry.materials.absorbers.biot",
    "phonometry.materials.absorption_rating":
        "phonometry.materials.absorbers.rating",
    "phonometry.materials.absorption_uncertainty":
        "phonometry.materials.absorbers.uncertainty",
    "phonometry.materials.porous_absorber":
        "phonometry.materials.absorbers.porous",
    "phonometry.materials.slow_sound_absorber":
        "phonometry.materials.absorbers.slow_sound",
    "phonometry.materials.diffuser_design":
        "phonometry.materials.diffusers.design",
    "phonometry.materials.metadiffuser":
        "phonometry.materials.diffusers.metadiffuser",
    "phonometry.materials.scattering_diffusion":
        "phonometry.materials.diffusers.scattering_diffusion",
    "phonometry.materials.road_absorption":
        "phonometry.materials.surfaces.road_absorption",
    "phonometry.materials.dynamic_stiffness":
        "phonometry.materials.resilient.dynamic_stiffness",
    "phonometry.psychoacoustics.loudness_zwicker":
        "phonometry.psychoacoustics.loudness.zwicker",
    "phonometry.psychoacoustics.loudness_moore_glasberg":
        "phonometry.psychoacoustics.loudness.moore_glasberg",
    "phonometry.psychoacoustics.loudness_moore_glasberg_time":
        "phonometry.psychoacoustics.loudness.moore_glasberg_time",
    "phonometry.psychoacoustics.loudness_ecma":
        "phonometry.psychoacoustics.loudness.ecma",
    "phonometry.psychoacoustics.loudness_contours":
        "phonometry.psychoacoustics.loudness.contours",
    "phonometry.psychoacoustics.sharpness":
        "phonometry.psychoacoustics.quality.sharpness",
    "phonometry.psychoacoustics.roughness_ecma":
        "phonometry.psychoacoustics.quality.roughness_ecma",
    "phonometry.psychoacoustics.fluctuation_strength":
        "phonometry.psychoacoustics.quality.fluctuation_strength",
    "phonometry.psychoacoustics.fluctuation_strength_ecma":
        "phonometry.psychoacoustics.quality.fluctuation_strength_ecma",
    "phonometry.psychoacoustics.tonality":
        "phonometry.psychoacoustics.quality.tonality",
    "phonometry.psychoacoustics.tonality_ecma":
        "phonometry.psychoacoustics.quality.tonality_ecma",
    "phonometry.psychoacoustics.tone_audibility":
        "phonometry.psychoacoustics.quality.tone_audibility",
    "phonometry.psychoacoustics.psychoacoustic_annoyance":
        "phonometry.psychoacoustics.quality.annoyance",
    # NT ACOU 112 and ISO/PAS 1996-3 share their formulae and are one module.
    "phonometry.environment.assessment.impulse_prominence": _IMPULSIVE_SOUND,
    "phonometry.room.room_acoustics": "phonometry.room.acoustics",
    "phonometry.room.room_ir": "phonometry.room.impulse_response",
    "phonometry.room.room_noise": "phonometry.room.noise_criteria",
    "phonometry.room.room_modes": "phonometry.room.modes",
    "phonometry.aircraft.aircraft_noise": "phonometry.aircraft.certification",
    "phonometry.underwater.marine_mammal_audiograms":
        "phonometry.underwater.bioacoustics.audiograms",
    "phonometry.underwater.marine_mammal_weighting":
        "phonometry.underwater.bioacoustics.weighting",
    "phonometry.underwater.numerical_propagation":
        "phonometry.underwater.propagation.numerical",
    "phonometry.underwater.weston_regimes":
        "phonometry.underwater.propagation.weston_regimes",
    "phonometry.underwater.seabed_reflection":
        "phonometry.underwater.propagation.seabed_reflection",
    "phonometry.underwater.sound_speed":
        "phonometry.underwater.propagation.sound_speed",
    "phonometry.underwater.ship_radiated_noise":
        "phonometry.underwater.sources.ship_radiated_noise",
    "phonometry.underwater.ship_traffic_noise":
        "phonometry.underwater.sources.ship_traffic_noise",
    "phonometry.underwater.pile_driving_noise":
        "phonometry.underwater.sources.pile_driving_noise",
    "phonometry.underwater.ocean_ambient_noise":
        "phonometry.underwater.sources.ambient_noise",
    # ``underwater.propagation`` is not in this table: the name now belongs to
    # the family package, which re-exports everything the module of that name
    # exported. An alias would have to shadow a real package to warn, and
    # shadowing it is exactly what must not happen.
}

#: The generations in force, each with the release that deprecated it and the
#: one that removes it. The 3.2 generation was removed in 4.0, as announced.
_GENERATIONS: tuple[tuple[dict[str, str], str, str], ...] = (
    (_MOVED_4X, "4.0", "5.0"),
)


def _make_shim(old: str, new: str, since: str, removed_in: str) -> types.ModuleType:
    shim = types.ModuleType(old)
    shim.__doc__ = (
        f"Deprecated alias of :mod:`{new}` (removed in phonometry {removed_in})."
    )

    def __getattr__(name: str) -> Any:
        if name == "__path__":
            # Never proxy the import machinery's own attribute. A renamed
            # package would otherwise hand out the real package's search path
            # and let ``import phonometry.environmental.propagation`` build a
            # second, independent copy of every submodule: same code, distinct
            # classes, failing isinstance and pickles. The alias serves the
            # modules registered for it and nothing else.
            raise AttributeError(f"module {old!r} has no attribute {name!r}")
        target = import_module(new)
        try:
            attr = getattr(target, name)
        except AttributeError:
            # A renamed package keeps serving the modules that moved out of
            # it: ``environmental.wind_turbine_noise`` is not an attribute of
            # ``environment`` any more, it is an alias of its own. The alias
            # carries the notice, so returning it here is silent.
            alias = sys.modules.get(f"{old}.{name}")
            if alias is not None:
                return alias
            raise AttributeError(
                f"module {old!r} has no attribute {name!r}"
            ) from None
        _warn_renamed(
            f"the '{old}' module",
            f"'{new}'",
            since=since,
            removed_in=removed_in,
        )
        return attr

    def __dir__() -> list[str]:
        names = set(dir(import_module(new))) | set(_alias_modules(old))
        return sorted(names)

    shim.__getattr__ = __getattr__  # type: ignore[method-assign]
    shim.__dir__ = __dir__  # type: ignore[method-assign]
    return shim


def _alias_modules(package: str) -> list[str]:
    """Pre-split module names of ``package`` that only live in the table now.

    ``phonometry.vibration.human_vibration`` is registered in ``sys.modules``
    by :func:`_install`, which is enough for ``import`` but not for the dotted
    read that follows it: the attribute is gone from the package. These are
    the names :func:`_namespace_shim` has to serve and :func:`_namespace_dir`
    has to list.
    """
    prefix = f"{package}."
    return sorted(
        old.removeprefix(prefix) for old in _MOVED_4X
        if old.startswith(prefix) and "." not in old.removeprefix(prefix)
    )


def _namespace_shim(
    package: str, targets: tuple[str, ...] = (), *,
    only: dict[str, tuple[str, ...]] | None = None,
    since: str = "4.0", removed_in: str = "5.0"
) -> Callable[[str], Any]:
    """Return a PEP 562 ``__getattr__`` for what left ``package``.

    The 4.0 taxonomy moves public names between subpackage namespaces, and
    ``from phonometry import metrology`` followed by ``metrology.leq(...)`` is
    the form the documentation leads with, so the read has to keep working
    from the namespace it left. Resolution is by ``__all__`` of the packages
    the names moved to, which keeps the shim honest: a name that stops being
    public anywhere stops resolving here too. A name that was both a module
    and a function resolves to the function, as the pre-split package did:
    ``metrology.cepstrum`` is :func:`phonometry.signals.cepstrum`.

    Only then does a name fall back to the module alias of the same name.
    That is what serves the modules with no public name of their own
    (``metrology.spectra``, ``metrology.levels``) and the whole of a package
    whose modules moved while its names stayed: ``vibration.human_vibration``
    is a module and never was a name, so ``targets`` is empty there and the
    fallback is the only branch that fires. The alias carries its own notice
    on attribute access, so returning it here is silent.

    A target that took only part of what it holds is listed in ``only``: the
    IEC 61265 check went from ``filters`` to ``aircraft``, and without that
    restriction the whole of ``aircraft`` would answer to ``filters.``, with a
    notice claiming names had moved that were never there.

    :param package: The narrowed package, ``__name__`` of its ``__init__``.
    :param targets: Packages the names moved to, in search order. Empty when
        only modules moved.
    :param only: Per target, the names that actually left this package. A
        target absent from the mapping serves its whole ``__all__``.
    :param since: Release that moved the names.
    :param removed_in: Major release that removes the alias.
    :return: The ``__getattr__`` to bind at module level.
    """

    def __getattr__(name: str) -> Any:
        for target in targets:
            allowed = (only or {}).get(target)
            if allowed is not None and name not in allowed:
                continue
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
    package: str, own: list[str] | tuple[str, ...],
    targets: tuple[str, ...] = (),
    only: dict[str, tuple[str, ...]] | None = None,
) -> Callable[[], list[str]]:
    """Return a ``__dir__`` listing the names a narrowed package still serves.

    ``__getattr__`` is invisible to :func:`dir`, so without this the moved
    names disappear from tab completion and from anything that introspects the
    namespace, one release before they stop working. ``__all__`` is left
    narrow on purpose: ``from phonometry.metrology import *`` gives the 4.0
    API, not the deprecated names.

    :param package: The package, ``__name__`` of its ``__init__``.
    :param own: The package's own ``__all__``.
    :param targets: Packages the moved names went to.
    :param only: Per target, the names that actually left this package, as in
        :func:`_namespace_shim`. Listing more would advertise names this
        package never served.
    :return: The ``__dir__`` to bind at module level.
    """

    def __dir__() -> list[str]:
        # Extend the default listing rather than replace it: a package's own
        # dir() carries its dunders and the subpackages it imported, and the
        # subgroups a split introduces are exactly what a reader is looking
        # for there.
        names = set(vars(import_module(package))) | set(own)
        names |= set(_alias_modules(package))
        for target in targets:
            moved = set(getattr(import_module(target), "__all__", ()))
            allowed = (only or {}).get(target)
            names |= moved if allowed is None else moved & set(allowed)
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
            # `import phonometry.environmental` also binds the attribute on
            # the package; mirror that so `phonometry.environmental` resolves
            # without the import. Aliases below a subpackage are served by that package's
            # own shim (:func:`_namespace_shim`), which resolves the moved
            # public names first, so binding them here would shadow a function
            # with a module.
            _, _, attr = old.rpartition(".")
            if old.count(".") == 1 and attr not in vars(package):
                setattr(package, attr, shim)


_install()

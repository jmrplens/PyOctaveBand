#  Copyright (c) 2026. Jose Manuel Requena Plens
"""What each generation of phonometry removed, pinned gone.

Every rename this library has made announced the release that would remove the
old spelling, and every one of those releases has now happened. Nothing here
tests a deprecation, because none is left: 4.0 carries no aliases, no shim
modules and no rename notices. What it tests is that the removals hold, which
is a different thing and needs saying out loud, because a name can come back by
accident. A module re-exported one level up, a constant reintroduced under its
old spelling, a stale file left on disk after a package split: each of those
resurrects a path the documentation says is gone, and none of them fails
anything else in the suite.

Three generations, all removed:

* **3.1** renamed functions, constants and keyword arguments, and named 4.0.
  ``docs/reference/api/index.md`` publishes that list; this file is what holds
  the page to it.
* **3.2** grouped the flat top-level modules into domain subpackages and named
  4.0. ``phonometry.levels`` and its eighty-odd siblings went then.
* **4.0** rebuilt the taxonomy (``metrology`` into ``filters``, ``signals`` and
  a narrowed ``metrology``; the speech intelligibility of ``hearing`` into
  ``speech``; ``environmental`` into ``environment``) and, unlike its
  predecessors, removed the old paths in the same release rather than shimming
  them for one more.
"""

from __future__ import annotations

import importlib
import inspect
import sys

import numpy as np
import pytest

import phonometry as ph

RNG = np.random.default_rng(1234)
SIGNAL = RNG.standard_normal(4800)
FS = 48_000.0


# --------------------------------------------------------------------------- #
# 3.1 renames: the function aliases, the renamed constants and the deprecated
# keywords were removed in 4.0, as their notices said since 3.1. Frozen lists;
# the point is that they stay gone.
# --------------------------------------------------------------------------- #
_REMOVED_FUNCTIONS = [
    ("phonometry", "octavefilter"),
    ("phonometry.filters", "octavefilter"),
    ("phonometry.filters.core", "octavefilter"),
    ("phonometry.filters.frequencies", "getansifrequencies"),
    ("phonometry.filters.frequencies", "normalizedfreq"),
    ("phonometry.metrology.calibration", "calculate_sensitivity"),
    ("phonometry.building.measurement.uncertainty", "coverage_factor"),
    ("phonometry.building.measurement.uncertainty", "expanded_uncertainty"),
]

_REMOVED_CONSTANTS = [
    ("phonometry", "OCTAVE_BANDS_HZ"),
    ("phonometry", "THIRD_OCTAVE_BANDS_HZ"),
    ("phonometry", "BASE_PLATE_BANDS_HZ"),
    ("phonometry", "ExposureWarning"),
    ("phonometry.materials.absorbers.rating", "OCTAVE_BANDS_HZ"),
    ("phonometry.materials.diffusers.scattering_diffusion", "BASE_PLATE_BANDS_HZ"),
    ("phonometry.speech.sii", "BAND_CENTRES"),
    ("phonometry.hearing.occupational_exposure", "ExposureWarning"),
]

#: ``(package, function, keyword)``: the keyword has to be absent from the
#: signature the package publishes today.
_REMOVED_KEYWORDS = [
    ("materials", "adrienne_window", "sample_rate"),
    ("materials", "insitu_reflection_factor", "sample_rate"),
    ("materials", "insitu_absorption_spectrum", "sample_rate"),
    ("environment", "atmospheric_absorption", "humidity"),
    ("environment", "outdoor_propagation_attenuation", "humidity"),
    ("environment", "predicted_receiver_level", "humidity"),
    ("emission", "environmental_correction", "room_volume"),
    ("emission", "sound_power_pressure", "room_volume"),
]


@pytest.mark.parametrize(("module", "name"), _REMOVED_FUNCTIONS + _REMOVED_CONSTANTS)
def test_removed_3_1_alias_is_gone(module: str, name: str) -> None:
    home = importlib.import_module(module)
    with pytest.raises(AttributeError):
        getattr(home, name)


@pytest.mark.parametrize(("package", "func", "keyword"), _REMOVED_KEYWORDS)
def test_removed_3_1_keyword_is_gone(package: str, func: str, keyword: str) -> None:
    home = getattr(ph, package)
    assert keyword not in inspect.signature(getattr(home, func)).parameters


def test_the_canonical_names_the_3_1_aliases_pointed_at_are_all_here() -> None:
    """The removal took the aliases, not the functions they delegated to."""
    for package, name in (
        ("filters", "octave_filter"),
        ("filters", "nominal_frequencies"),
        ("filters", "normalized_frequencies"),
        ("metrology", "sensitivity"),
        ("building", "insulation_coverage_factor"),
        ("building", "insulation_expanded_uncertainty"),
        ("materials", "OCTAVE_BANDS"),
        ("materials", "THIRD_OCTAVE_BANDS"),
        ("materials", "BASE_PLATE_BANDS"),
        ("hearing", "OccupationalExposureWarning"),
    ):
        assert hasattr(getattr(ph, package), name), f"{package}.{name}"


def test_the_plot_renderers_moved_out_of_the_deprecated_module() -> None:
    """``phonometry._plotting`` was the 3.2 re-export; ``_plot`` is the home."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("phonometry._plotting")
    assert callable(importlib.import_module("phonometry._plot.room").plot_excitation)


# --------------------------------------------------------------------------- #
# 3.2 package reorganization: its flat module paths were removed in 4.0, as
# announced. A sample of them, frozen; the point is that they stay gone, and
# that a stale file left on disk would be caught rather than silently served.
# --------------------------------------------------------------------------- #
_REMOVED_FLAT_MODULE_PATHS = [
    "phonometry.core",
    "phonometry.insulation",
    "phonometry.levels",
    "phonometry.loudness",
    "phonometry.room_ir",
    "phonometry.underwater_acoustics",
    "phonometry.utils",
]


@pytest.mark.parametrize("path", _REMOVED_FLAT_MODULE_PATHS)
def test_removed_flat_module_path_raises(path: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(path)
    assert path not in sys.modules
    # The names these modules held never moved; the package that owns each of
    # them publishes it, which is what the flat paths were a shortcut to.
    assert callable(ph.signals.leq)


# --------------------------------------------------------------------------- #
# 4.0 taxonomy: the release that rebuilt it also removed the paths it replaced,
# instead of shimming them for one more generation. A sample of the old paths,
# one per kind of move: a package that split, a package that was renamed, a
# module that gained a second level, and a module that was renamed in place.
# --------------------------------------------------------------------------- #
_REMOVED_4_0_MODULE_PATHS = [
    "phonometry.metrology.core",
    "phonometry.metrology.levels",
    "phonometry.metrology.intensity_compliance",
    "phonometry.hearing.sti",
    "phonometry.hearing.sii",
    "phonometry.environmental",
    "phonometry.environmental.cnossos_road",
    "phonometry.vibration.human_vibration",
    "phonometry.building.spanish_building_code",
    "phonometry.materials.porous_absorber",
    "phonometry.psychoacoustics.loudness_zwicker",
    "phonometry.underwater.ocean_ambient_noise",
    "phonometry.room.room_ir",
    "phonometry.aircraft.aircraft_noise",
]


@pytest.mark.parametrize("path", _REMOVED_4_0_MODULE_PATHS)
def test_removed_4_0_module_path_raises(path: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(path)
    assert path not in sys.modules


def test_a_removed_path_is_not_served_by_its_parent_package() -> None:
    """The three ways a caller can reach for a moved module, all refused.

    ``import a.b`` and ``from a.b import name`` raise ``ModuleNotFoundError``,
    ``from a import b`` raises ``ImportError`` because ``a`` imports and only
    the name lookup inside it fails, and ``a.b`` as an attribute raises
    ``AttributeError``. Three exception types for one removal, which is why the
    CHANGELOG names them separately.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("phonometry.hearing.sti")
    with pytest.raises(ImportError):
        from phonometry.hearing import sti  # noqa: F401
    with pytest.raises(AttributeError):
        _ = ph.hearing.sti
    with pytest.raises(AttributeError):
        _ = ph.environmental


def test_the_names_the_4_0_moves_carried_are_all_reachable() -> None:
    """The paths went; the public names they held did not."""
    assert callable(ph.signals.leq)
    assert callable(ph.filters.octave_filter)
    assert callable(ph.speech.sti_from_impulse_response)
    assert callable(ph.emission.intensity_class_compliance)
    assert callable(ph.environment.lden)
    assert callable(ph.room.impulse_response)


def test_a_namespace_does_not_serve_a_name_that_left_it() -> None:
    """A name that moved between packages is gone from the one it left.

    The 4.0 split let each of these keep answering from its old namespace for
    one release; that indirection went with the shims, so reading one from
    where it used to live now fails where it used to warn.
    """
    for namespace, name in (
        (ph.metrology, "leq"),
        (ph.metrology, "octave_filter"),
        (ph.metrology, "intensity_class_compliance"),
        (ph.hearing, "sti_from_impulse_response"),
        (ph.hearing, "speech_intelligibility_index"),
        (ph.filters, "verify_aircraft_noise_system"),
    ):
        with pytest.raises(AttributeError):
            getattr(namespace, name)

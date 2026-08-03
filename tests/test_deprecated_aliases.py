#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Deprecation shims of the phonometry renames, and what 4.0 removed.

The 3.1 renames (function aliases, renamed constants, deprecated keywords) and
the 3.2 module moves both named 4.0 as their removal, and 4.0 removed them: the
first half of this file pins them gone. What is left alive is the 4.0 taxonomy,
whose aliases warn with the NEP 23 message, delegate to the canonical name and
go in 5.0 (CONTRIBUTING, "Deprecations").
"""

from __future__ import annotations

import pathlib
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

_REMOVED_KEYWORDS = [
    ("adrienne_window", "sample_rate"),
    ("insitu_reflection_factor", "sample_rate"),
    ("insitu_absorption_spectrum", "sample_rate"),
    ("atmospheric_absorption", "humidity"),
    ("outdoor_propagation_attenuation", "humidity"),
    ("predicted_receiver_level", "humidity"),
    ("environmental_correction", "room_volume"),
    ("sound_power_pressure", "room_volume"),
]


@pytest.mark.parametrize(("module", "name"), _REMOVED_FUNCTIONS + _REMOVED_CONSTANTS)
def test_removed_3_1_alias_is_gone(module: str, name: str) -> None:
    import importlib

    home = importlib.import_module(module)
    with pytest.raises(AttributeError):
        getattr(home, name)


@pytest.mark.parametrize(("func", "keyword"), _REMOVED_KEYWORDS)
def test_removed_3_1_keyword_is_gone(func: str, keyword: str) -> None:
    import inspect

    assert keyword not in inspect.signature(getattr(ph, func)).parameters


def test_the_canonical_names_the_3_1_aliases_pointed_at_are_all_here() -> None:
    """The removal took the aliases, not the functions they delegated to."""
    for name in (
        "octave_filter", "nominal_frequencies", "normalized_frequencies",
        "sensitivity", "insulation_coverage_factor",
        "insulation_expanded_uncertainty", "OCTAVE_BANDS", "THIRD_OCTAVE_BANDS",
        "BASE_PLATE_BANDS", "OccupationalExposureWarning",
    ):
        assert hasattr(ph, name), name


def test_the_plot_renderers_moved_out_of_the_deprecated_module() -> None:
    """``phonometry._plotting`` was the 3.2 re-export; ``_plot`` is the home."""
    import importlib

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


def test_pre_split_module_alias_rejects_unknown_names() -> None:
    """A 4.0 alias serves what its target has, and nothing else.

    The alias delegates by ``getattr`` on the relocated module, so a typo has
    to come back as an ``AttributeError`` naming the path the caller wrote,
    not as a silent ``None`` or a confusing message about the new module.
    """
    import importlib

    shim = importlib.import_module("phonometry.metrology.levels")
    with pytest.raises(AttributeError, match="phonometry.metrology.levels"):
        _ = shim.does_not_exist


@pytest.mark.parametrize("path", _REMOVED_FLAT_MODULE_PATHS)
def test_removed_flat_module_path_raises(path: str) -> None:
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(path)
    assert path not in sys.modules
    # The names themselves never moved: the flat API is what they were for.
    assert hasattr(ph, "leq")


# --------------------------------------------------------------------------- #
# 4.0 taxonomy: metrology split into filters + signals + a narrowed metrology,
# and the speech intelligibility of hearing into speech.
# Frozen snapshot of the pre-split module paths; do NOT regenerate from the
# live tree. Removed in 5.0 together with the aliases.
# --------------------------------------------------------------------------- #
_PRE_SPLIT_MODULE_PATHS = [
    "phonometry.hearing.objective_intelligibility",
    "phonometry.hearing.sii",
    "phonometry.hearing.sti",
    "phonometry.metrology.cepstrum",
    "phonometry.metrology.compliance",
    "phonometry.metrology.core",
    "phonometry.metrology.correlation",
    "phonometry.metrology.envelope",
    "phonometry.metrology.equalizer",
    "phonometry.metrology.filter_design",
    "phonometry.metrology.frequencies",
    "phonometry.metrology.inversion",
    "phonometry.metrology.levels",
    "phonometry.metrology.miso",
    "phonometry.metrology.parametric_filters",
    "phonometry.metrology.phase",
    "phonometry.metrology.random_data",
    "phonometry.metrology.signals",
    "phonometry.metrology.spectra",
    "phonometry.metrology.synchronous_average",
    "phonometry.metrology.time_frequency",
    "phonometry.vibration.experimental_sea",
    "phonometry.vibration.human_vibration",
    "phonometry.vibration.junction_transmission",
    "phonometry.vibration.machine_diagnostics",
    "phonometry.vibration.mechanical_mobility",
    "phonometry.vibration.multiple_shock_vibration",
    "phonometry.vibration.point_mobility",
    "phonometry.vibration.radiation_efficiency",
    "phonometry.vibration.transfer_stiffness",
    # The package itself was renamed, so the prefix is an alias as well.
    "phonometry.environmental",
    "phonometry.environmental.air_absorption",
    "phonometry.environmental.atmospheric_refraction",
    "phonometry.environmental.cnossos_rail",
    "phonometry.environmental.cnossos_road",
    "phonometry.environmental.ground_barriers",
    "phonometry.environmental.impulse_prominence",
    "phonometry.environmental.impulsive_sound",
    "phonometry.environmental.measurement",
    "phonometry.environmental.outdoor_propagation",
    "phonometry.environmental.rating",
    "phonometry.environmental.spanish_regulation",
    "phonometry.environmental.wind_turbine_noise",
    "phonometry.building.aperture_transmission",
    "phonometry.building.building_prediction",
    "phonometry.building.building_uncertainty",
    "phonometry.building.ceiling_plenum",
    "phonometry.building.detailed_prediction",
    "phonometry.building.facade_prediction",
    "phonometry.building.flanking_transmission",
    "phonometry.building.floor_covering_improvement",
    "phonometry.building.heavy_impact",
    "phonometry.building.installed_structure_borne",
    "phonometry.building.insulation",
    "phonometry.building.intensity_insulation",
    "phonometry.building.lab_insulation",
    "phonometry.building.masonry_cavity_wall",
    "phonometry.building.panel_transmission",
    "phonometry.building.resilient_layers",
    "phonometry.building.spanish_building_code",
    "phonometry.building.structure_borne_power",
    "phonometry.building.survey_insulation",
    "phonometry.materials.absorption_rating",
    "phonometry.materials.absorption_uncertainty",
    "phonometry.materials.airflow_resistance",
    "phonometry.materials.biot",
    "phonometry.materials.diffuser_design",
    "phonometry.materials.dynamic_stiffness",
    "phonometry.materials.impedance_tube",
    "phonometry.materials.metadiffuser",
    "phonometry.materials.porous_absorber",
    "phonometry.materials.road_absorption",
    "phonometry.materials.scattering_diffusion",
    "phonometry.materials.slow_sound_absorber",
    "phonometry.materials.sound_absorption",
    "phonometry.psychoacoustics.fluctuation_strength",
    "phonometry.psychoacoustics.fluctuation_strength_ecma",
    "phonometry.psychoacoustics.loudness_contours",
    "phonometry.psychoacoustics.loudness_ecma",
    "phonometry.psychoacoustics.loudness_moore_glasberg",
    "phonometry.psychoacoustics.loudness_moore_glasberg_time",
    "phonometry.psychoacoustics.loudness_zwicker",
    "phonometry.psychoacoustics.psychoacoustic_annoyance",
    "phonometry.psychoacoustics.roughness_ecma",
    "phonometry.psychoacoustics.sharpness",
    "phonometry.psychoacoustics.tonality",
    "phonometry.psychoacoustics.tonality_ecma",
    "phonometry.psychoacoustics.tone_audibility",
    "phonometry.underwater.marine_mammal_audiograms",
    "phonometry.underwater.marine_mammal_weighting",
    "phonometry.underwater.numerical_propagation",
    "phonometry.underwater.ocean_ambient_noise",
    "phonometry.underwater.pile_driving_noise",
    "phonometry.underwater.seabed_reflection",
    "phonometry.underwater.ship_radiated_noise",
    "phonometry.underwater.ship_traffic_noise",
    "phonometry.underwater.sound_speed",
    "phonometry.underwater.weston_regimes",
    "phonometry.metrology.intensity_compliance",
    "phonometry.aircraft.aircraft_noise",
    "phonometry.environment.assessment.impulse_prominence",
    "phonometry.room.room_acoustics",
    "phonometry.room.room_ir",
    "phonometry.room.room_modes",
    "phonometry.room.room_noise",
]


@pytest.mark.parametrize("path", _PRE_SPLIT_MODULE_PATHS)
def test_pre_split_module_path_still_imports(path: str) -> None:
    import importlib
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        module = importlib.import_module(path)  # import itself must be silent
    assert module is sys.modules[path]
    public = [name for name in dir(module) if not name.startswith("_")]
    assert public, f"{path} imports but exposes no public names"


def test_the_migration_table_names_real_aliases() -> None:
    """Every row of the curated table must be a live row of the alias tables.

    A global rename over the documentation has rewritten this table's left
    column three times, each time turning a deprecated path into the path it
    resolves to, so the page ended up calling the current path deprecated. The
    check is cheap and the failure is unmistakable.
    """
    import importlib
    import re

    from phonometry._compat import _MOVED_4X

    table = (
        pathlib.Path(__file__).resolve().parent.parent
        / "docs" / "reference" / "api" / "index.md"
    ).read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| `(phonometry[\w.]*)` \| `(phonometry[\w.]*)` \| (\d\.\d) \|$",
        table, re.MULTILINE,
    )
    assert rows, "the migration table is gone from docs/reference/api/index.md"
    # One example per split, or a deleted row passes unnoticed: the table was
    # rebuilt once from a corrupted copy and losing rows is that failure mode.
    documented = {old.split(".")[1] for old, _, _ in rows}
    moved_from = {key.split(".")[1] for key in _MOVED_4X}
    assert moved_from <= documented, (
        f"no example row for {sorted(moved_from - documented)}"
    )
    for old, new, removed_in in rows:
        assert removed_in == "5.0", f"{old} names a removal that already happened"
        assert old in _MOVED_4X, f"{old} is not a deprecated path"
        assert _MOVED_4X[old] == new, (
            f"{old} resolves to {_MOVED_4X[old]}, not to {new}"
        )
        importlib.import_module(new)


def test_a_renamed_package_alias_is_not_a_package() -> None:
    """The alias must not hand out the real package's search path.

    Proxying ``__path__`` would let ``import phonometry.environmental.<sub>``
    build a second, independent copy of every submodule: same source, distinct
    classes, failing isinstance and pickles, and with no notice on the way in.
    """
    import importlib

    from phonometry import environmental  # noqa: F401 - the alias under test

    assert not hasattr(sys.modules["phonometry.environmental"], "__path__")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("phonometry.environmental.propagation")
    # The modules the table names still resolve, and to the same object.
    alias = importlib.import_module("phonometry.environmental.rating")
    assert alias is sys.modules["phonometry.environmental.rating"]


def test_a_module_only_split_still_serves_the_dotted_read() -> None:
    """``vibration.human_vibration.x`` after the import, not only the import.

    No public name left ``vibration`` in 4.0, only its modules did, so the
    package has no names to redirect. The dotted read still has to work: the
    import registers the alias in ``sys.modules`` and the attribute has to
    come from somewhere.
    """
    import importlib

    from phonometry import vibration

    importlib.import_module("phonometry.vibration.human_vibration")
    alias = vibration.human_vibration
    assert alias is sys.modules["phonometry.vibration.human_vibration"]
    with pytest.warns(DeprecationWarning, match="vibration.human.exposure"):
        assert alias.daily_exposure is ph.daily_exposure
    assert "human_vibration" in dir(vibration)
    assert "human_vibration" not in vibration.__all__
    with pytest.raises(AttributeError, match="phonometry.vibration"):
        _ = vibration.not_a_name


@pytest.mark.parametrize("path", _PRE_SPLIT_MODULE_PATHS)
def test_pre_split_dotted_read_resolves(path: str) -> None:
    """Reading the alias off its parent package, which is how code uses it."""
    import importlib
    import warnings

    importlib.import_module(path)
    parent_name, _, leaf = path.rpartition(".")
    parent = importlib.import_module(parent_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        resolved = getattr(parent, leaf)
    # Either the module alias itself, or the public name that shadowed it
    # before the split (``metrology.cepstrum`` was the function, not the
    # module, and stays the function).
    assert resolved is sys.modules[path] or callable(resolved)


def test_pre_split_module_shim_names_the_5_0_removal() -> None:
    """The 4.0 aliases outlive the 3.x ones: they go in 5.0, not in 4.0."""
    shim = sys.modules["phonometry.metrology.levels"]
    with pytest.warns(DeprecationWarning, match="removed in 5.0") as record:
        _ = shim.leq
    assert "phonometry.signals.levels" in str(record[0].message)


def test_narrowed_namespace_still_serves_the_names_that_left() -> None:
    """``metrology.leq`` keeps working: the namespace form is documented."""
    import warnings

    from phonometry import metrology

    with pytest.warns(DeprecationWarning, match="phonometry.signals.leq"):
        assert metrology.leq is ph.leq
    with pytest.warns(DeprecationWarning, match="phonometry.filters.octave_filter"):
        assert metrology.octave_filter is ph.octave_filter
    # Names that stayed resolve without a notice.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert metrology.combine_uncertainty is ph.combine_uncertainty
    with pytest.raises(AttributeError, match="phonometry.metrology"):
        _ = metrology.not_a_name


def test_narrowed_namespace_lists_the_moved_names_in_dir() -> None:
    """A PEP 562 hook is invisible to dir(); the names must not vanish early."""
    from phonometry import filters, metrology, signals

    listed = dir(metrology)
    assert set(metrology.__all__) <= set(listed)
    assert set(filters.__all__) <= set(listed)
    assert set(signals.__all__) <= set(listed)
    assert listed == sorted(listed)
    # __all__ stays narrow, so `import *` gives the 4.0 API, not the aliases.
    assert "leq" not in metrology.__all__


def test_narrowed_hearing_namespace_still_serves_speech() -> None:
    """``hearing.stipa`` keeps working: the namespace form is documented."""
    import warnings

    from phonometry import hearing

    with pytest.warns(DeprecationWarning, match="phonometry.speech.stipa"):
        assert hearing.stipa is ph.stipa
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert hearing.age_threshold is ph.age_threshold
    assert "speech_intelligibility_index" in dir(hearing)
    assert "speech_intelligibility_index" not in hearing.__all__
    with pytest.raises(AttributeError, match="phonometry.hearing"):
        _ = hearing.not_a_name


def test_narrowed_namespace_falls_back_to_the_module_alias() -> None:
    """``metrology.spectra`` has no public name of its own; it is the module."""
    import warnings

    from phonometry import metrology

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert metrology.spectra is sys.modules["phonometry.metrology.spectra"]
    # A name that is both a module and a public function resolves to the
    # function, as the pre-split package did.
    with pytest.warns(DeprecationWarning, match="phonometry.signals.correlation"):
        assert metrology.correlation is ph.correlation


def test_moved_module_shims_warn_and_delegate() -> None:
    import importlib

    from phonometry._compat import _MOVED_4X

    for old, new in _MOVED_4X.items():
        shim = importlib.import_module(old)
        target = importlib.import_module(new)
        public = [n for n in dir(target) if not n.startswith("_")]
        if not public:  # pragma: no cover - all shim targets export names
            continue
        with pytest.warns(DeprecationWarning, match="deprecated since phonometry"):
            attr = getattr(shim, public[0])
        assert attr is getattr(target, public[0])
        assert set(public) <= set(dir(shim))

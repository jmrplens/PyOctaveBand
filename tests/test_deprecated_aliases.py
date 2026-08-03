#  Copyright (c) 2026. Jose Manuel Requena Plens
"""One-cycle deprecation shims of the phonometry renames.

One :func:`pytest.warns` test per alias (CONTRIBUTING, "Deprecations"):
the renamed ``loudness`` module (PEP 562 shim), the legacy snake_case
function aliases and the renamed keyword arguments (scikit-learn
``"deprecated"`` sentinel). Every alias must warn with the NEP 23 message
and delegate to the canonical name.

Two generations coexist, each removed with its own release: the 3.1 renames
and the 3.2 module moves go in 4.0, the 4.0 taxonomy aliases in 5.0.
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
# Renamed module: phonometry.loudness -> phonometry.loudness_zwicker
# --------------------------------------------------------------------------- #
def test_loudness_module_attribute_access_warns_and_delegates() -> None:
    import phonometry.loudness  # noqa: F401  (PEP 562 shim; import is silent)

    shim = sys.modules["phonometry.loudness"]
    target = sys.modules["phonometry.psychoacoustics.loudness.zwicker"]
    with pytest.warns(DeprecationWarning, match=r"loudness\.zwicker"):
        cls = shim.ZwickerLoudness
    assert cls is target.ZwickerLoudness
    with pytest.warns(DeprecationWarning, match="deprecated since phonometry 3.1"):
        func = shim.loudness_zwicker
    assert func is ph.loudness_zwicker
    # __dir__ delegates (and does not warn).
    assert "loudness_zwicker_from_spectrum" in dir(shim)
    with pytest.raises(AttributeError, match="phonometry.loudness"):
        _ = shim.does_not_exist


# --------------------------------------------------------------------------- #
# Legacy snake_case function aliases
# --------------------------------------------------------------------------- #
def test_octavefilter_warns_and_delegates() -> None:
    canonical_spl, canonical_freq = ph.octave_filter(SIGNAL, 48000)
    with pytest.warns(DeprecationWarning, match=r"octave_filter\(\)"):
        spl, freq = ph.octavefilter(SIGNAL, 48000)
    np.testing.assert_allclose(spl, canonical_spl)
    assert freq == canonical_freq


def test_filters_octavefilter_warns_and_delegates() -> None:
    """The alias exported by phonometry.filters keeps the top-level behavior."""
    from phonometry import filters

    assert filters.octave_filter is ph.octave_filter
    assert filters.octavefilter is ph.octavefilter
    canonical_spl, canonical_freq = filters.octave_filter(SIGNAL, 48000)
    with pytest.warns(DeprecationWarning, match=r"octave_filter\(\)"):
        spl, freq = filters.octavefilter(SIGNAL, 48000)
    np.testing.assert_allclose(spl, canonical_spl)
    assert freq == canonical_freq


def test_getansifrequencies_warns_and_delegates() -> None:
    canonical = ph.nominal_frequencies(3, [100, 5000])
    with pytest.warns(DeprecationWarning, match=r"nominal_frequencies\(\)"):
        legacy = ph.getansifrequencies(3, [100, 5000])
    assert legacy == canonical


def test_normalizedfreq_warns_and_delegates() -> None:
    canonical = ph.normalized_frequencies(1)
    with pytest.warns(DeprecationWarning, match=r"normalized_frequencies\(\)"):
        legacy = ph.normalizedfreq(1)
    assert legacy == canonical


def test_calculate_sensitivity_warns_and_delegates() -> None:
    tone = np.sin(2 * np.pi * 1000.0 * np.arange(4800) / FS)
    canonical = ph.sensitivity(tone, target_spl=94.0)
    with pytest.warns(DeprecationWarning, match=r"sensitivity\(\)"):
        legacy = ph.calculate_sensitivity(tone, target_spl=94.0)
    assert legacy == canonical


# --------------------------------------------------------------------------- #
# Renamed keyword: road_absorption sample_rate -> fs
# --------------------------------------------------------------------------- #
def test_adrienne_window_sample_rate_warns_and_forwards() -> None:
    canonical = ph.adrienne_window(FS)
    with pytest.warns(DeprecationWarning, match="'sample_rate' keyword"):
        legacy = ph.adrienne_window(sample_rate=FS)
    np.testing.assert_allclose(legacy, canonical)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="both"):
        ph.adrienne_window(FS, sample_rate=FS)
    with pytest.raises(ValueError, match="missing required argument: 'fs'"):
        ph.adrienne_window()


def test_insitu_reflection_factor_sample_rate_warns_and_forwards() -> None:
    hi = np.zeros(256)
    hi[8] = 1.0
    hr = 0.5 * np.roll(hi, 16)
    delay = 16 / FS
    canonical = ph.insitu_reflection_factor(hi, hr, fs=FS, delay=delay)
    with pytest.warns(DeprecationWarning, match="'sample_rate' keyword"):
        legacy = ph.insitu_reflection_factor(hi, hr, sample_rate=FS, delay=delay)
    np.testing.assert_allclose(legacy, canonical)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="both"):
        ph.insitu_reflection_factor(hi, hr, fs=FS, sample_rate=FS)


def test_insitu_absorption_spectrum_sample_rate_warns_and_forwards() -> None:
    hi = np.zeros(4096)
    hi[16] = 1.0
    hr = 0.5 * np.roll(hi, 32)
    canonical = ph.insitu_absorption_spectrum(hi, hr, FS)
    with pytest.warns(DeprecationWarning, match="'sample_rate' keyword"):
        legacy = ph.insitu_absorption_spectrum(hi, hr, sample_rate=FS)
    np.testing.assert_allclose(legacy.absorption, canonical.absorption)
    with pytest.raises(ValueError, match="missing required argument: 'fs'"):
        ph.insitu_absorption_spectrum(hi, hr)


# --------------------------------------------------------------------------- #
# Renamed keyword: outdoor_propagation humidity -> relative_humidity
# --------------------------------------------------------------------------- #
def test_atmospheric_absorption_humidity_warns_and_forwards() -> None:
    canonical = ph.atmospheric_absorption(200.0, [1000.0], relative_humidity=50.0)
    with pytest.warns(DeprecationWarning, match="'humidity' keyword"):
        legacy = ph.atmospheric_absorption(200.0, [1000.0], humidity=50.0)
    np.testing.assert_allclose(legacy, canonical)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="both"):
        ph.atmospheric_absorption(
            200.0, [1000.0], relative_humidity=50.0, humidity=50.0
        )


def test_outdoor_propagation_attenuation_humidity_warns_and_forwards() -> None:
    canonical = ph.outdoor_propagation_attenuation(
        100.0, 2.0, 4.0, [500.0], relative_humidity=50.0
    )
    with pytest.warns(DeprecationWarning, match="'humidity' keyword"):
        legacy = ph.outdoor_propagation_attenuation(
            100.0, 2.0, 4.0, [500.0], humidity=50.0
        )
    np.testing.assert_allclose(legacy.a_total, canonical.a_total)


def test_predicted_receiver_level_humidity_warns_and_forwards() -> None:
    canonical = ph.predicted_receiver_level(
        [95.0], 100.0, 2.0, 4.0, [500.0], relative_humidity=50.0
    )
    with pytest.warns(DeprecationWarning, match="'humidity' keyword"):
        legacy = ph.predicted_receiver_level(
            [95.0], 100.0, 2.0, 4.0, [500.0], humidity=50.0
        )
    np.testing.assert_allclose(legacy, canonical)


# --------------------------------------------------------------------------- #
# Renamed keyword: sound_power room_volume -> volume
# --------------------------------------------------------------------------- #
def test_environmental_correction_room_volume_warns_and_forwards() -> None:
    canonical = ph.environmental_correction(
        40.0, reverberation_time=1.2, volume=300.0
    )
    with pytest.warns(DeprecationWarning, match="'room_volume' keyword"):
        legacy = ph.environmental_correction(
            40.0, reverberation_time=1.2, room_volume=300.0
        )
    assert legacy == canonical
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="both"):
        ph.environmental_correction(
            40.0, reverberation_time=1.2, volume=300.0, room_volume=300.0
        )


def test_sound_power_pressure_room_volume_warns_and_forwards() -> None:
    levels = np.tile(np.array([90.0, 92.0, 95.0]), (10, 1))
    canonical = ph.sound_power_pressure(
        levels, "hemisphere", radius=2.0, reverberation_time=1.0, volume=2000.0
    )
    with pytest.warns(DeprecationWarning, match="'room_volume' keyword"):
        legacy = ph.sound_power_pressure(
            levels,
            "hemisphere",
            radius=2.0,
            reverberation_time=1.0,
            room_volume=2000.0,
        )
    np.testing.assert_allclose(
        legacy.sound_power_level, canonical.sound_power_level
    )


def test_room_volume_explicit_none_stays_silent() -> None:
    # None was the old default; passing it through the deprecated alias must
    # not warn (only a real value does).
    import warnings

    import phonometry as ph

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ph.environmental_correction(50.0, absorption_area=10.0, room_volume=None)


# --------------------------------------------------------------------------- #
# Renamed constants (unit suffixes dropped) and the renamed warning class,
# aliased through module-level PEP 562 __getattr__ (constants cannot warn as
# wrappers). Each alias exists in its home module and, when re-exported, at
# the package root.
# --------------------------------------------------------------------------- #
def test_octave_bands_hz_warns_and_delegates() -> None:
    from phonometry import absorption_rating

    with pytest.warns(DeprecationWarning, match="use OCTAVE_BANDS"):
        legacy = ph.OCTAVE_BANDS_HZ
    assert legacy is ph.OCTAVE_BANDS
    with pytest.warns(DeprecationWarning, match="deprecated since phonometry 3.1"):
        module_legacy = absorption_rating.OCTAVE_BANDS_HZ
    assert module_legacy is absorption_rating.OCTAVE_BANDS


def test_third_octave_bands_hz_warns_and_delegates() -> None:
    from phonometry import absorption_rating

    with pytest.warns(DeprecationWarning, match="use THIRD_OCTAVE_BANDS"):
        legacy = ph.THIRD_OCTAVE_BANDS_HZ
    assert legacy is ph.THIRD_OCTAVE_BANDS
    with pytest.warns(DeprecationWarning, match="deprecated since phonometry 3.1"):
        module_legacy = absorption_rating.THIRD_OCTAVE_BANDS_HZ
    assert module_legacy is absorption_rating.THIRD_OCTAVE_BANDS


def test_base_plate_bands_hz_warns_and_delegates() -> None:
    from phonometry import scattering_diffusion

    with pytest.warns(DeprecationWarning, match="use BASE_PLATE_BANDS"):
        legacy = ph.BASE_PLATE_BANDS_HZ
    assert legacy is ph.BASE_PLATE_BANDS
    with pytest.warns(DeprecationWarning, match="deprecated since phonometry 3.1"):
        module_legacy = scattering_diffusion.BASE_PLATE_BANDS_HZ
    assert module_legacy is scattering_diffusion.BASE_PLATE_BANDS


def test_band_centres_warns_and_delegates() -> None:
    from phonometry import sii

    with pytest.warns(DeprecationWarning, match="use BAND_CENTERS"):
        legacy = sii.BAND_CENTRES
    assert legacy is sii.BAND_CENTERS


def test_exposure_warning_warns_and_delegates() -> None:
    from phonometry import occupational_exposure

    with pytest.warns(DeprecationWarning, match="use OccupationalExposureWarning"):
        legacy = ph.ExposureWarning
    # Same class object: isinstance/except/filters via the old name still match.
    assert legacy is ph.OccupationalExposureWarning
    with pytest.warns(DeprecationWarning, match="deprecated since phonometry 3.1"):
        module_legacy = occupational_exposure.ExposureWarning
    assert module_legacy is ph.OccupationalExposureWarning


def test_renamed_attribute_shims_reject_unknown_names() -> None:
    from phonometry import absorption_rating, occupational_exposure

    with pytest.raises(AttributeError, match="phonometry"):
        _ = ph.does_not_exist
    with pytest.raises(AttributeError, match="absorption_rating"):
        _ = absorption_rating.does_not_exist
    with pytest.raises(AttributeError, match="occupational_exposure"):
        _ = occupational_exposure.does_not_exist


# --------------------------------------------------------------------------- #
# 3.2 package reorganization: every pre-move public module path must remain
# importable (silently) for one deprecation cycle. Frozen snapshot; do NOT
# regenerate from the live tree (that would defeat its purpose).
# --------------------------------------------------------------------------- #
_PRE_MOVE_MODULE_PATHS = [
    "phonometry.absorption_rating",
    "phonometry.absorption_uncertainty",
    "phonometry.air_absorption",
    "phonometry.aircraft_atmospheric_absorption",
    "phonometry.aircraft_noise",
    "phonometry.airflow_resistance",
    "phonometry.airport_noise",
    "phonometry.building_prediction",
    "phonometry.building_uncertainty",
    "phonometry.calibration",
    "phonometry.compliance",
    "phonometry.core",
    "phonometry.distortion",
    "phonometry.dynamic_stiffness",
    "phonometry.enclosed_space_absorption",
    "phonometry.environmental",
    "phonometry.environmental_measurement",
    "phonometry.facade_prediction",
    "phonometry.filter_design",
    "phonometry.flanking_transmission",
    "phonometry.floor_covering_improvement",
    "phonometry.fluctuation_strength",
    "phonometry.frequencies",
    "phonometry.frequency_response",
    "phonometry.hearing",
    "phonometry.human_vibration",
    "phonometry.impedance_tube",
    "phonometry.impulse_prominence",
    "phonometry.installed_structure_borne",
    "phonometry.insulation",
    "phonometry.intensity",
    "phonometry.intensity_insulation",
    "phonometry.lab_insulation",
    "phonometry.levels",
    "phonometry.loudness",
    "phonometry.loudness_contours",
    "phonometry.loudness_ecma",
    "phonometry.loudness_moore_glasberg",
    "phonometry.loudness_moore_glasberg_time",
    "phonometry.loudness_zwicker",
    "phonometry.mechanical_mobility",
    "phonometry.multiple_shock_vibration",
    "phonometry.noise_induced_hearing_loss",
    "phonometry.numerical_propagation",
    "phonometry.occupational_exposure",
    "phonometry.ocean_ambient_noise",
    "phonometry.open_plan",
    "phonometry.outdoor_propagation",
    "phonometry.parametric_filters",
    "phonometry.pile_driving_noise",
    "phonometry._plotting",
    "phonometry.psychoacoustic_annoyance",
    "phonometry.reverberation_prediction",
    "phonometry.road_absorption",
    "phonometry.room_acoustics",
    "phonometry.room_ir",
    "phonometry.room_noise",
    "phonometry.rotorcraft_noise",
    "phonometry.roughness_ecma",
    "phonometry.scattering_diffusion",
    "phonometry.seabed_reflection",
    "phonometry.sharpness",
    "phonometry.ship_radiated_noise",
    "phonometry.ship_traffic_noise",
    "phonometry.sii",
    "phonometry.sonar_equation",
    "phonometry.sound_absorption",
    "phonometry.sound_power",
    "phonometry.sound_power_intensity",
    "phonometry.sound_power_reverberation",
    "phonometry.sti",
    "phonometry.structure_borne_power",
    "phonometry.survey_insulation",
    "phonometry.tonality",
    "phonometry.tonality_ecma",
    "phonometry.tone_audibility",
    "phonometry.transfer_stiffness",
    "phonometry.uncertainty",
    "phonometry.underwater_acoustics",
    "phonometry.underwater_propagation",
    "phonometry.underwater_sound_speed",
    "phonometry.utils",
    "phonometry.vibration_sound_power",
    "phonometry._warnings",
    "phonometry.wind_turbine_noise",
]


@pytest.mark.parametrize("path", _PRE_MOVE_MODULE_PATHS)
def test_pre_move_module_path_still_imports(path: str) -> None:
    import importlib
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        module = importlib.import_module(path)  # import itself must be silent
    assert module is sys.modules[path]
    # A shim that imports but exposes nothing is as broken as an ImportError:
    # every pre-move path must still surface its public names. ``dir()`` is
    # silent on the PEP 562 shims, so this stays warning-free.
    public = [name for name in dir(module) if not name.startswith("_")]
    assert public, f"{path} imports but exposes no public names"


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

    from phonometry._compat import _MOVED_3X, _MOVED_4X

    table = (
        pathlib.Path(__file__).resolve().parent.parent
        / "docs" / "api-reference.md"
    ).read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| `(phonometry[\w.]*)` \| `(phonometry[\w.]*)` \| (\d\.\d) \|$",
        table, re.MULTILINE,
    )
    assert rows, "the migration table is gone from docs/api-reference.md"
    # One example per split, or a deleted row passes unnoticed: the table was
    # rebuilt once from a corrupted copy and losing rows is that failure mode.
    documented = {old.split(".")[1] for old, _, _ in rows}
    moved_from = {key.split(".")[1] for key in _MOVED_4X}
    assert moved_from <= documented, (
        f"no example row for {sorted(moved_from - documented)}"
    )
    for old, new, removed_in in rows:
        generation = _MOVED_3X if removed_in == "4.0" else _MOVED_4X
        assert old in generation, f"{old} is not a deprecated path of {removed_in}"
        assert generation[old] == new, (
            f"{old} resolves to {generation[old]}, not to {new}"
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


# Frozen snapshot of the ``phonometry._plotting`` re-export surface (the
# renderers as of the 3.2 move); do NOT regenerate from the live tree.
_PLOTTING_RENDERERS = [
    "plot_absorption_uncertainty",
    "plot_age_threshold",
    "plot_airborne_insulation",
    "plot_airborne_prediction",
    "plot_aircraft_band_attenuation",
    "plot_ambient_noise",
    "plot_band_uncertainty",
    "plot_bottom_loss",
    "plot_daily_exposure",
    "plot_decay_curve",
    "plot_diffusion_polar",
    "plot_dynamic_stiffness",
    "plot_ecma_loudness",
    "plot_ecma_roughness",
    "plot_ecma_tonality",
    "plot_enclosed_space_absorption",
    "plot_epnl",
    "plot_excitation",
    "plot_facade_insulation",
    "plot_facade_prediction",
    "plot_fdtd_probes",
    "plot_fdtd_snapshot",
    "plot_floor_covering_improvement",
    "plot_fluctuation_strength",
    "plot_flyover",
    "plot_frequency_response",
    "plot_harmonic_distortion",
    "plot_htlan",
    "plot_impact_insulation",
    "plot_impact_prediction",
    "plot_impact_rating",
    "plot_impedance_tube",
    "plot_impulse_prominence",
    "plot_impulse_response",
    "plot_insitu_absorption",
    "plot_installed_structure_borne",
    "plot_intensity",
    "plot_mobility",
    "plot_monte_carlo",
    "plot_moore_glasberg_loudness",
    "plot_moore_glasberg_time_loudness",
    "plot_multiple_shock",
    "plot_nipts",
    "plot_noise_contour",
    "plot_noise_criterion",
    "plot_normal_modes",
    "plot_npd_level",
    "plot_occupational_exposure",
    "plot_open_plan",
    "plot_outdoor_attenuation",
    "plot_parabolic_equation",
    "plot_pile_strike",
    "plot_psychoacoustic_annoyance",
    "plot_radiated_power",
    "plot_ray_trace",
    "plot_reverberation_models",
    "plot_room_acoustics",
    "plot_room_criterion",
    "plot_rotorcraft_hemisphere",
    "plot_scattering_coefficient",
    "plot_ship_source_level",
    "plot_ship_traffic_spectrum",
    "plot_sii",
    "plot_sonar_equation",
    "plot_sound_power",
    "plot_sound_speed_profile",
    "plot_static_airflow",
    "plot_sti",
    "plot_structure_borne_power",
    "plot_tonal_adjustment",
    "plot_tone_audibility",
    "plot_transfer_stiffness",
    "plot_transmission_loss",
    "plot_uncertainty_budget",
    "plot_vibration_reduction",
    "plot_vibration_sound_power",
    "plot_vibration_weighting",
    "plot_weighted_absorption",
    "plot_weighted_rating",
    "plot_weighted_spectrum",
    "plot_wind_turbine_tonality",
    "plot_zwicker_loudness",
]


def test_plotting_shim_re_exports_every_renderer() -> None:
    """``phonometry._plotting`` silently re-exports the full renderer set."""
    import importlib
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        module = importlib.import_module("phonometry._plotting")
        assert sorted(module.__all__) == _PLOTTING_RENDERERS
        for name in _PLOTTING_RENDERERS:
            assert callable(getattr(module, name)), name


def test_moved_module_shims_warn_and_delegate() -> None:
    import importlib

    from phonometry._compat import _MOVED_3X, _MOVED_4X

    for old, new in {**_MOVED_3X, **_MOVED_4X}.items():
        shim = importlib.import_module(old)
        target = importlib.import_module(new)
        public = [n for n in dir(target) if not n.startswith("_")]
        if not public:  # pragma: no cover - all shim targets export names
            continue
        with pytest.warns(DeprecationWarning, match="deprecated since phonometry"):
            attr = getattr(shim, public[0])
        assert attr is getattr(target, public[0])
        assert set(public) <= set(dir(shim))

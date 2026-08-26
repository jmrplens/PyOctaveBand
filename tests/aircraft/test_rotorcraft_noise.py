#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the rotorcraft hemisphere method (ECAC Doc 32 / NORAH2).

Oracles: the guidance Table 4 (one-third-octave atmospheric attenuation per km at
ICAO reference conditions) for the atmospheric term; closed-form spherical
spreading; analytic ground-effect limits; and exact hemisphere interpolation at
the grid nodes.

Everything here assumes flat ground. What §A.4.4-A.4.5 do when it is not flat,
the mean ground plane, the diffraction over a screening profile and the
heterogeneous event chain, is tested in ``test_rotorcraft_terrain.py``.
"""

from __future__ import annotations

import dataclasses

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pytest

from phonometry.aircraft.rotorcraft_noise import (
    RotorcraftHemisphere,
    hemisphere_source_level,
)
from phonometry.aircraft.rotorcraft_propagation import (
    atmospheric_adjustment,
    ground_effect_adjustment,
    spherical_spreading_adjustment,
)

# NORAH2 guidance Table 4, complete: one-third-octave atmospheric attenuation per
# km (dB) at ICAO reference conditions (25 °C, 70 % RH, 101.325 kPa). Independent
# oracle. Per-band tolerance: the module implements the guidance Eq. 27 pure-tone
# coefficient (which the NORAH2 prototype uses), while Table 4 tabulates the SAE
# (Rickley) band mapping; both agree to ~0.05 dB/km below 800 Hz and to
# ~0.3 dB/km up to 3.15 kHz, then drift apart towards 8-10 kHz (up to
# 2.2 dB/km), which the wider high-band tolerances make explicit.
_TABLE4 = {
    10: 0.0,
    12.5: 0.0,
    16: 0.0,
    20: 0.0,
    25: 0.0,
    31.5: 0.0,
    40: 0.0,
    50: 0.0,
    63: 0.1,
    80: 0.1,
    100: 0.2,
    125: 0.3,
    160: 0.5,
    200: 0.7,
    250: 1.1,
    315: 1.6,
    400: 2.3,
    500: 3.1,
    630: 4.1,
    800: 5.2,
    1000: 6.3,
    1250: 7.5,
    1600: 8.9,
    2000: 10.6,
    2500: 13.0,
    3150: 16.6,
    4000: 22.5,
    5000: 31.0,
    6300: 44.9,
    8000: 67.6,
    10000: 101.0,
}


def _table4_tolerance(freq: float) -> float:
    if freq <= 630.0:
        return 0.1
    if freq <= 3150.0:
        return 0.3
    if freq <= 6300.0:
        return 1.0
    return 2.5


def test_spherical_spreading_matches_inverse_square() -> None:
    assert spherical_spreading_adjustment(60.0) == pytest.approx(0.0)
    assert spherical_spreading_adjustment(600.0) == pytest.approx(-20.0)
    assert spherical_spreading_adjustment(120.0) == pytest.approx(-20.0 * np.log10(2.0))
    with pytest.raises(ValueError, match="distance"):
        spherical_spreading_adjustment(0.0)


@pytest.mark.parametrize("freq", list(_TABLE4))
def test_atmospheric_matches_table4(freq: float) -> None:
    # ΔLa over a 1 km excess path (r = 1060 m) against Table 4, all 31 bands.
    la = atmospheric_adjustment([float(freq)], 1000.0 + 60.0)[0]
    assert la == pytest.approx(-_TABLE4[freq], abs=_table4_tolerance(freq))


def test_atmospheric_low_bands_do_not_warn() -> None:
    # The standard NORAH grid starts at 10 Hz, below the ISO 9613-1 tabulated
    # range; the advisory out-of-range warning is suppressed (alpha ~ 0 there).
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        la = atmospheric_adjustment(np.array(list(_TABLE4), dtype=float), 1060.0)
    assert np.all(la <= 0.0)
    # Above the 10 kHz top of the NORAH grid alpha is large and extrapolated:
    # the advisory warning must still propagate there.
    from phonometry.environment.propagation.air_absorption import (
        AtmosphericAbsorptionWarning,
    )

    with pytest.warns(AtmosphericAbsorptionWarning):
        atmospheric_adjustment([16000.0], 1060.0)


def test_adjustments_honour_reference_distance() -> None:
    # Hemispheres recorded at a non-standard polar distance (e.g. 70 m hover
    # rings) pass their distance through 'reference_distance'.
    assert spherical_spreading_adjustment(
        70.0, reference_distance=70.0
    ) == pytest.approx(0.0)
    assert spherical_spreading_adjustment(
        700.0, reference_distance=70.0
    ) == pytest.approx(-20.0)
    la = atmospheric_adjustment([1000.0], 70.0, reference_distance=70.0)
    assert la[0] == pytest.approx(0.0)


def test_atmospheric_reference_distance_is_zero() -> None:
    la = atmospheric_adjustment([1000.0, 4000.0], 60.0)
    assert np.allclose(la, 0.0)


def test_ground_effect_hard_ground_reinforces_low_frequency() -> None:
    # Over a hard surface the direct and reflected rays are nearly in phase at low
    # frequency and small path difference, giving a positive (up to ~+6 dB) ΔLg.
    lg = ground_effect_adjustment([50.0], 100.0, 1.5, 200.0, flow_resistivity="G")[0]
    assert 0.0 < lg <= 6.02
    # Soft ground absorbs the reflection, so its low-frequency gain is smaller.
    lg_soft = ground_effect_adjustment([50.0], 100.0, 1.5, 200.0, flow_resistivity="D")[
        0
    ]
    assert lg_soft < lg


def test_ground_effect_class_letters_and_values_agree() -> None:
    by_letter = ground_effect_adjustment(
        [500.0, 1000.0], 100.0, 1.5, 300.0, flow_resistivity="G"
    )
    by_value = ground_effect_adjustment(
        [500.0, 1000.0], 100.0, 1.5, 300.0, flow_resistivity=20.0e6
    )
    assert np.allclose(by_letter, by_value)


def test_ground_effect_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="horizontal_distance"):
        ground_effect_adjustment([500.0], 100.0, 1.5, 0.0)
    with pytest.raises(ValueError, match="flow_resistivity"):
        ground_effect_adjustment([500.0], 100.0, 1.5, 300.0, flow_resistivity="Z")


def _synthetic_hemisphere() -> RotorcraftHemisphere:
    az = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    po = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
    fr = np.array([500.0, 1000.0, 2000.0])
    # Level falls linearly with polar angle and is symmetric in azimuth.
    lv = np.zeros((az.size, po.size, fr.size))
    for i in range(az.size):
        for j, t in enumerate(po):
            lv[i, j, :] = np.array([100.0, 98.0, 96.0]) - 0.05 * t
    return RotorcraftHemisphere(fr, az, po, lv)


def test_hemisphere_exact_at_nodes() -> None:
    h = _synthetic_hemisphere()
    assert np.allclose(hemisphere_source_level(h, 0.0, 90.0), [95.5, 93.5, 91.5])
    assert np.allclose(hemisphere_source_level(h, -45.0, 0.0), [100.0, 98.0, 96.0])


def test_hemisphere_energy_bilinear_midpoint() -> None:
    h = _synthetic_hemisphere()
    # Between θ = 0 (100 dB) and θ = 45 (97.75 dB) at φ = 0, band 500 Hz: energy mean.
    got = hemisphere_source_level(h, 0.0, 22.5)[0]
    expected = 10.0 * np.log10(
        0.5 * 10.0 ** (100.0 / 10.0) + 0.5 * 10.0 ** (97.75 / 10.0)
    )
    assert got == pytest.approx(expected)


def test_hemisphere_clips_outside_grid() -> None:
    h = _synthetic_hemisphere()
    # Beyond the grid edges the lookup clamps to the boundary node.
    assert np.allclose(
        hemisphere_source_level(h, 200.0, 300.0),
        hemisphere_source_level(h, 90.0, 180.0),
    )


def test_hemisphere_nearest_bin_fill_for_missing() -> None:
    h = _synthetic_hemisphere()
    lv = h.levels.copy()
    lv[2, 2, 0] = np.nan  # drop φ=0, θ=90, 500 Hz
    hole = RotorcraftHemisphere(h.frequencies, h.azimuth, h.polar, lv)
    out = hemisphere_source_level(hole, 0.0, 90.0)
    # The four axis-neighbours (±45°,90°), (0°,45°), (0°,135°) are all exactly
    # equidistant (ρ = 45°), so the fill is their energetic average (Eq. 14/15).
    neigh = np.array([95.5, 95.5, 97.75, 93.25])  # 100 − 0.05·θ at θ = 90,90,45,135
    assert out[0] == pytest.approx(10.0 * np.log10(np.mean(10.0 ** (neigh / 10.0))))
    assert out[1] == pytest.approx(93.5)  # other bands untouched


def test_hemisphere_partial_cell_keeps_valid_corners() -> None:
    # Gap-fill semantics (Eq. 14/15): the grid is filled FIRST (each empty bin
    # takes its nearest filled bin), then the bilinear interpolation uses all
    # four corners. A cell with three valid corners must therefore keep their
    # contribution instead of snapping to the single bin nearest to the query.
    h = _synthetic_hemisphere()
    lv = h.levels.copy()
    lv[2, 2, 0] = np.nan  # drop φ=0, θ=90 at 500 Hz
    hole = RotorcraftHemisphere(h.frequencies, h.azimuth, h.polar, lv)
    got = hemisphere_source_level(hole, 22.5, 112.5)[
        0
    ]  # centre of the φ 0-45/θ 90-135 cell
    # The dropped corner fills with the energetic mean of its four equidistant
    # neighbours (ρ = 45°: (±45°, 90°), (0°, 45°), (0°, 135°)).
    neigh = np.array([95.5, 95.5, 97.75, 93.25])
    fill = 10.0 * np.log10(np.mean(10.0 ** (neigh / 10.0)))
    corners = np.array(
        [fill, 93.25, 95.5, 93.25]
    )  # (0,90) filled, (0,135), (45,90), (45,135)
    expected = 10.0 * np.log10(np.mean(10.0 ** (corners / 10.0)))
    assert got == pytest.approx(expected)


def test_hemisphere_single_row_grid() -> None:
    # A single measured azimuth row (size-1 axis) interpolates along polar only.
    fr = np.array([500.0, 1000.0])
    po = np.array([80.0, 90.0])
    lv = np.array([[[70.0, 68.0], [74.0, 72.0]]])  # shape (1, 2, 2)
    h = RotorcraftHemisphere(fr, np.array([0.0]), po, lv)
    assert np.allclose(hemisphere_source_level(h, 0.0, 80.0), [70.0, 68.0])
    mid = hemisphere_source_level(h, 30.0, 85.0)  # azimuth clamps to the only row
    expected = 10.0 * np.log10(
        0.5 * 10.0 ** (np.array([70.0, 68.0]) / 10.0)
        + 0.5 * 10.0 ** (np.array([74.0, 72.0]) / 10.0)
    )
    assert np.allclose(mid, expected)


def test_hemisphere_all_nan_band_returns_nan() -> None:
    # A band with no filled bin anywhere cannot be extrapolated: NaN, documented.
    h = _synthetic_hemisphere()
    lv = h.levels.copy()
    lv[:, :, 1] = np.nan
    hole = RotorcraftHemisphere(h.frequencies, h.azimuth, h.polar, lv)
    out = hemisphere_source_level(hole, 10.0, 70.0)
    assert np.isnan(out[1])
    assert np.isfinite(out[0])
    assert np.isfinite(out[2])


def test_hemisphere_plot() -> None:
    assert _synthetic_hemisphere().plot() is not None


# A 3×3×3 sub-grid of real hemisphere levels (dB at 60 m) transcribed from the
# NORAH2 reference database (helicopter A109, approach 60 kt / 3° descent;
# EASA.2020.FC.06, © EASA), azimuth φ ∈ {-10, 0, 10}°, polar θ ∈ {80, 90, 100}°,
# bands 500 / 1000 / 2000 Hz. Used as an independent oracle for the hemisphere
# interpolation (the raw NORAH data is not redistributed).
_A109_PHI = np.array([-10.0, 0.0, 10.0])
_A109_THETA = np.array([80.0, 90.0, 100.0])
_A109_FREQ = np.array([500.0, 1000.0, 2000.0])
_A109_LEVELS = np.array(
    [
        [[79.6, 74.8, 72.2], [77.1, 73.5, 71.2], [76.0, 73.1, 69.9]],
        [[78.7, 73.9, 73.0], [76.0, 72.8, 71.5], [74.3, 71.8, 70.0]],
        [[77.0, 73.9, 73.4], [74.3, 74.1, 71.8], [72.5, 73.9, 70.5]],
    ]
)


def _a109_hemisphere() -> RotorcraftHemisphere:
    return RotorcraftHemisphere(_A109_FREQ, _A109_PHI, _A109_THETA, _A109_LEVELS)


def test_reference_hemisphere_exact_at_nodes() -> None:
    h = _a109_hemisphere()
    assert np.allclose(hemisphere_source_level(h, 0.0, 90.0), [76.0, 72.8, 71.5])
    assert np.allclose(hemisphere_source_level(h, -10.0, 80.0), [79.6, 74.8, 72.2])


def test_reference_hemisphere_energy_bilinear_between_real_nodes() -> None:
    # Midpoint of the φ=0/10, θ=80/90 cell at 500 Hz: energy mean of the 4 corners.
    h = _a109_hemisphere()
    got = hemisphere_source_level(h, 5.0, 85.0)[0]
    corners = np.array([78.7, 77.0, 76.0, 74.3])  # (0,80),(10,80),(0,90),(10,90)
    expected = 10.0 * np.log10(np.mean(10.0 ** (corners / 10.0)))
    assert got == pytest.approx(expected)


# Reference source spectra (dB at 60 m) at the φ = 0°, θ = 90° node for every
# rotorcraft type in the NORAH2 database (approach condition; EASA.2020.FC.06,
# © EASA — factual reference points, the raw database is not redistributed),
# over the 63 Hz - 8 kHz octave grid. Exercises the interpolation pipeline over
# every model and across the frequency range.
_TYPE_BANDS = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
_TYPE_SPECTRA = [
    ("A109", [78.6, 83.6, 86.1, 82.0, 76.7, 70.9, 64.6, 56.1]),
    ("AS350", [75.6, 73.0, 71.8, 72.6, 69.7, 65.8, 59.1, 51.5]),
    ("B412", [85.6, 83.5, 79.4, 76.6, 74.2, 70.3, 62.6, 51.4]),
    ("Cabri", [66.1, 72.2, 69.9, 69.8, 64.8, 61.5, 54.7, 47.3]),
    ("EC120", [76.5, 70.4, 72.5, 69.9, 68.2, 66.3, 62.4, 58.1]),
    ("EC135", [68.7, 79.4, 76.7, 75.2, 72.9, 68.9, 65.5, 62.9]),
    ("R22", [67.6, 72.7, 74.7, 73.9, 70.7, 66.3, 60.1, 54.8]),
    ("R44", [70.9, 80.2, 78.5, 72.0, 68.7, 62.0, 55.6, 45.6]),
    ("R66", [83.1, 81.8, 79.0, 73.6, 68.8, 66.3, 64.3, 62.5]),
    ("S300", [66.6, 66.2, 67.9, 70.2, 65.8, 62.2, 53.3, 43.8]),
    ("S92", [87.1, 87.4, 85.5, 81.4, 78.8, 76.2, 71.3, 65.1]),
]


@pytest.mark.parametrize(("rotorcraft", "spectrum"), _TYPE_SPECTRA)
def test_reference_hemisphere_all_types(rotorcraft: str, spectrum: list[float]) -> None:
    # Place the reference spectrum at the φ=0, θ=90 corner and distinct (offset)
    # values at the other three corners, so recovering it at that exact node
    # genuinely exercises the azimuth/polar indexing and corner weighting.
    spec = np.asarray(spectrum, dtype=np.float64)
    az = np.array([0.0, 10.0])
    po = np.array([80.0, 90.0])
    levels = np.empty((az.size, po.size, spec.size))
    levels[0, 1, :] = spec  # φ=0, θ=90 (the queried node)
    levels[0, 0, :] = spec - 3.0
    levels[1, 1, :] = spec + 2.0
    levels[1, 0, :] = spec - 5.0
    h = RotorcraftHemisphere(_TYPE_BANDS, az, po, levels)
    assert np.allclose(hemisphere_source_level(h, 0.0, 90.0), spec)
    # Physically-plausible helicopter spectrum: mid-frequency content, HF roll-off.
    assert spec[-1] < spec[3]


# Off-node bilinear spot checks against the NORAH2 reference database
# (EASA.2020.FC.06, © EASA — factual reference points from the approach
# hemispheres listed in _TYPE_SPECTRA; the raw database is not redistributed).
# Three query points per rotorcraft type, each in a different grid cell and
# band. Corner order: [L(φ0,θ0), L(φ0,θ1), L(φ1,θ0), L(φ1,θ1)]; the expected
# level is the energy-domain bilinear value (Eq. 13) at the query weights,
# evaluated offline: a pinned regression on real corner data rather than an
# external oracle (the propagation-chain rows below are the external oracle).
_BILINEAR_QUERIES = [
    # (φ, θ, cell φ0/φ1, cell θ0/θ1, band index into _TYPE_BANDS-like triple)
    (5.0, 95.0, (0.0, 10.0), (90.0, 100.0)),
    (-17.0, 117.0, (-20.0, -10.0), (110.0, 120.0)),
    (17.0, 72.0, (10.0, 20.0), (70.0, 80.0)),
]
_BILINEAR_CASES = [
    # type, [(corners, expected)] per query: 500 Hz, 2000 Hz, 125 Hz.
    (
        "A109",
        [
            ([82.0, 80.7, 82.0, 79.5], 81.1693),
            ([70.0, 69.9, 68.8, 68.2], 69.5224),
            ([81.8, 83.7, 81.5, 83.7], 82.1035),
        ],
    ),
    (
        "AS350",
        [
            ([72.6, 71.5, 72.2, 71.0], 71.8687),
            ([68.7, 66.8, 66.4, 65.9], 67.0858),
            ([78.6, 76.8, 78.9, 77.7], 78.5716),
        ],
    ),
    (
        "B412",
        [
            ([76.6, 75.7, 76.9, 76.4], 76.4220),
            ([70.6, 72.1, 69.2, 69.1], 71.0767),
            ([85.8, 84.6, 83.7, 82.9], 84.2671),
        ],
    ),
    (
        "Cabri",
        [
            ([69.8, 70.9, 67.9, 71.2], 70.1285),
            ([63.5, 59.4, 61.4, 59.7], 60.8533),
            ([69.4, 68.0, 71.3, 66.1], 70.2504),
        ],
    ),
    (
        "EC120",
        [
            ([69.9, 67.0, 70.7, 68.0], 69.1438),
            ([67.8, 68.3, 66.6, 66.9], 67.7946),
            ([67.2, 69.8, 67.5, 70.4], 68.1403),
        ],
    ),
    (
        "EC135",
        [
            ([75.2, 73.7, 74.6, 73.1], 74.2248),
            ([69.0, 68.9, 68.8, 68.2], 68.7748),
            ([79.3, 79.2, 79.5, 78.1], 79.2617),
        ],
    ),
    (
        "R22",
        [
            ([73.9, 70.6, 74.6, 71.2], 72.9032),
            ([63.3, 62.4, 63.3, 62.3], 62.6706),
            ([74.6, 74.5, 76.2, 75.7], 75.7006),
        ],
    ),
    (
        "R44",
        [
            ([72.0, 72.1, 71.6, 71.6], 71.8310),
            ([62.7, 61.6, 61.4, 60.7], 61.6740),
            ([80.3, 79.3, 79.6, 79.6], 79.7619),
        ],
    ),
    (
        "R66",
        [
            ([73.6, 72.7, 73.3, 72.5], 73.0477),
            ([67.2, 67.0, 66.1, 65.9], 66.7592),
            ([79.3, 79.2, 77.5, 78.6], 78.2641),
        ],
    ),
    (
        "S300",
        [
            ([70.2, 66.9, 68.8, 67.1], 68.4639),
            ([57.4, 55.7, 58.4, 58.1], 56.9488),
            ([67.8, 69.4, 66.1, 68.0], 67.1040),
        ],
    ),
    (
        "S92",
        [
            ([81.4, 83.1, 79.4, 81.7], 81.5928),
            ([75.5, 72.4, 76.6, 74.6], 74.1725),
            ([95.2, 87.5, 90.9, 89.4], 92.1467),
        ],
    ),
]


@pytest.mark.parametrize(("rotorcraft", "cases"), _BILINEAR_CASES)
def test_reference_bilinear_off_node(rotorcraft: str, cases: list) -> None:
    for (corners, expected), (phi, theta, (a0, a1), (p0, p1)) in zip(
        cases, _BILINEAR_QUERIES, strict=True
    ):
        lv = np.array(corners, dtype=np.float64).reshape(2, 2, 1)
        h = RotorcraftHemisphere(
            np.array([500.0]), np.array([a0, a1]), np.array([p0, p1]), lv
        )
        got = hemisphere_source_level(h, phi, theta)[0]
        assert got == pytest.approx(expected, abs=5e-4), (rotorcraft, phi, theta)


# Full propagation-chain oracle: single-emission rows of the NORAH2 prototype
# ARP cases whose TriOrNear id selects a single nearest hemisphere (no
# flight-condition blending), so L(fc,φ,θ) + ΔLs + ΔLa + ΔLg + A-weighting,
# energy-summed, must reproduce the prototype's tabulated LA. The 31-band
# source spectra below are the hemisphere lookups at the row emission angles,
# derived from the reference database (EASA.2020.FC.06, © EASA — factual
# derived values; the raw database is not redistributed). Case 4 rows
# (R22_H1_APP_STD2_NE, mic (0,0), hr 0.2 m, σ = 1e6 Pa·s/m²) reproduce to
# 0.1 dB; the Case 2 rows (R22_H1_DEP_STD1_NE, mic (-300,-400), hr 1.2 m,
# σ = 2e5) exercise softer ground within 0.5 dB.
_CHAIN_BANDS = np.array(
    [
        10.0,
        12.5,
        16.0,
        20.0,
        25.0,
        31.5,
        40.0,
        50.0,
        63.0,
        80.0,
        100.0,
        125.0,
        160.0,
        200.0,
        250.0,
        315.0,
        400.0,
        500.0,
        630.0,
        800.0,
        1000.0,
        1250.0,
        1600.0,
        2000.0,
        2500.0,
        3150.0,
        4000.0,
        5000.0,
        6300.0,
        8000.0,
        10000.0,
    ]
)
_CHAIN_ROWS = [
    # (label, spectrum dB @60 m, slant m, hs m, hr m, dp m, sigma, LA ref, tol)
    (
        "APP_STD2 t=748.00 (53kts_5deg)",
        [
            72.1,
            66.196,
            63.3,
            77.5,
            58.5,
            63.2,
            79.1,
            67.4,
            75.6,
            71.1,
            81.0,
            76.3,
            67.104,
            65.2,
            76.7,
            79.0,
            82.8,
            86.5,
            88.2,
            85.204,
            85.604,
            83.804,
            75.0,
            73.2,
            71.2,
            69.8,
            70.8,
            70.1,
            73.1,
            73.3,
            73.3,
        ],
        2377.77,
        299.18,
        0.2,
        2358.902,
        1.0e6,
        53.16,
        0.1,
    ),
    (
        "APP_STD2 t=790.00 (53kts_5deg)",
        [
            46.0,
            49.4,
            77.4,
            66.2,
            50.7,
            63.1,
            55.3,
            60.9,
            61.2,
            79.7,
            73.7,
            70.9,
            67.2,
            68.4,
            74.6,
            70.0,
            65.8,
            64.6,
            62.5,
            63.9,
            64.2,
            62.7,
            63.4,
            62.7,
            59.9,
            56.8,
            54.8,
            54.7,
            53.3,
            52.2,
            52.3,
        ],
        1300.08,
        205.02,
        0.2,
        1283.848,
        1.0e6,
        47.21,
        0.1,
    ),
    (
        "APP_STD2 t=825.00 (53kts_12deg)",
        [
            35.8,
            52.2,
            70.0,
            63.6,
            48.4,
            59.2,
            53.1,
            63.3,
            61.2,
            74.6,
            68.7,
            61.8,
            65.7,
            59.7,
            59.7,
            63.6,
            57.9,
            57.7,
            58.6,
            61.0,
            61.9,
            64.7,
            65.7,
            65.6,
            64.1,
            61.6,
            57.9,
            55.8,
            56.4,
            53.4,
            52.9,
        ],
        374.55,
        43.19,
        0.2,
        372.072,
        1.0e6,
        54.93,
        0.1,
    ),
    (
        "APP_STD2 t=831.26 (53kts_12deg)",
        [
            35.8,
            52.2,
            70.0,
            63.6,
            48.4,
            59.2,
            53.1,
            63.3,
            61.2,
            74.6,
            68.7,
            61.8,
            65.7,
            59.7,
            59.7,
            63.6,
            57.9,
            57.7,
            58.6,
            61.0,
            61.9,
            64.7,
            65.7,
            65.6,
            64.1,
            61.6,
            57.9,
            55.8,
            56.4,
            53.4,
            52.9,
        ],
        223.66,
        5.0,
        0.2,
        223.607,
        1.0e6,
        55.87,
        0.1,
    ),
    (
        "DEP_STD1 t=0.00 (54kts_8.5deg)",
        [
            47.992,
            50.306,
            77.582,
            78.82,
            55.472,
            72.727,
            74.719,
            70.448,
            62.609,
            78.785,
            80.74,
            76.042,
            69.093,
            71.592,
            67.773,
            66.552,
            67.625,
            64.61,
            61.615,
            62.097,
            69.004,
            72.176,
            71.897,
            69.595,
            65.737,
            66.233,
            65.226,
            62.767,
            59.982,
            60.554,
            56.737,
        ],
        721.12,
        5.0,
        1.2,
        721.11,
        2.0e5,
        46.86,
        0.5,
    ),
    (
        "DEP_STD1 t=10.00 (54kts_8.5deg)",
        [
            47.947,
            50.414,
            78.59,
            78.512,
            54.966,
            73.537,
            74.053,
            70.789,
            63.086,
            79.737,
            80.806,
            75.942,
            70.201,
            71.781,
            67.13,
            66.899,
            67.466,
            64.435,
            61.714,
            62.613,
            69.192,
            72.564,
            72.477,
            70.021,
            65.789,
            66.575,
            65.735,
            63.389,
            60.461,
            60.856,
            57.021,
        ],
        993.00,
        46.06,
        1.2,
        991.993,
        2.0e5,
        51.00,
        0.5,
    ),
    (
        "DEP_STD1 t=20.00 (54kts_8.5deg)",
        [
            47.895,
            50.327,
            79.023,
            78.209,
            54.829,
            73.844,
            73.682,
            71.056,
            63.942,
            80.128,
            80.606,
            75.558,
            70.72,
            71.661,
            67.061,
            67.086,
            67.329,
            64.267,
            61.627,
            62.9,
            68.977,
            72.529,
            72.645,
            70.228,
            66.056,
            66.648,
            65.864,
            63.656,
            60.789,
            61.004,
            57.228,
        ],
        1267.45,
        87.12,
        1.2,
        1264.532,
        2.0e5,
        47.20,
        0.5,
    ),
]


def _a_weighting_db(frequencies: np.ndarray) -> np.ndarray:
    # IEC 61672-1 analytic A-weighting at the exact band centres.
    f = np.asarray(frequencies, dtype=np.float64)
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217

    def ra(x: np.ndarray) -> np.ndarray:
        return (f4**2 * x**4) / (
            (x**2 + f1**2) * np.sqrt((x**2 + f2**2) * (x**2 + f3**2)) * (x**2 + f4**2)
        )

    return 20.0 * np.log10(ra(f) / ra(np.array(1000.0)))


@pytest.mark.parametrize(
    ("label", "spectrum", "slant", "hs", "hr", "dp", "sigma", "la_ref", "tol"),
    _CHAIN_ROWS,
)
def test_propagation_chain_reproduces_prototype_la(
    label: str,
    spectrum: list[float],
    slant: float,
    hs: float,
    hr: float,
    dp: float,
    sigma: float,
    la_ref: float,
    tol: float,
) -> None:
    level = (
        np.asarray(spectrum, dtype=np.float64)
        + spherical_spreading_adjustment(slant)
        + atmospheric_adjustment(_CHAIN_BANDS, slant)
        + ground_effect_adjustment(_CHAIN_BANDS, hs, hr, dp, flow_resistivity=sigma)
        + _a_weighting_db(_CHAIN_BANDS)
    )
    la = float(10.0 * np.log10(np.sum(10.0 ** (level / 10.0))))
    assert la == pytest.approx(la_ref, abs=tol), label


# --------------------------------------------------------------------------- #
# Hover, idle and taxi derivation (guidance §A.3.5, Table 3)
# --------------------------------------------------------------------------- #

from phonometry.aircraft.rotorcraft_noise import (
    hover_derived_hemisphere,
    hover_ring_hemisphere,
)

#: Twelve ring bearings every 30°; the ring closes periodically at ±180°.
_RING_BEARINGS = np.arange(-180.0, 151.0, 30.0)


def _ring(levels_per_bearing: np.ndarray, bands: int = 2) -> np.ndarray:
    """A ring level table with the same value in every band, shape (B, F)."""
    return np.tile(
        np.asarray(levels_per_bearing, dtype=np.float64)[:, None], (1, bands)
    )


def test_hover_ring_uniform_ring_gives_uniform_hemisphere() -> None:
    h = hover_ring_hemisphere([500.0, 1000.0], _RING_BEARINGS, _ring(np.full(12, 80.0)))
    assert h.distance == 70.0
    assert h.azimuth.size == 19
    assert h.polar.size == 19
    assert np.allclose(h.levels, 80.0)
    # Arbitrary off-node lookups stay at the uniform level.
    assert hemisphere_source_level(h, 37.0, 112.0) == pytest.approx([80.0, 80.0])


def test_hover_ring_constant_phi_closed_form() -> None:
    # Ring level = 60 + bearing/10 dB: every hemisphere bin reads the ring at
    # the bearing +-theta of its own half, in closed form.
    h = hover_ring_hemisphere(
        [500.0, 1000.0], _RING_BEARINGS, _ring(60.0 + _RING_BEARINGS / 10.0)
    )
    ia = {a: i for i, a in enumerate(h.azimuth)}
    ip = {p: i for i, p in enumerate(h.polar)}
    assert h.levels[ia[30.0], ip[60.0], 0] == pytest.approx(66.0)  # ring(+60)
    assert h.levels[ia[-30.0], ip[60.0], 0] == pytest.approx(54.0)  # ring(-60)
    assert h.levels[ia[90.0], ip[150.0], 0] == pytest.approx(75.0)  # ring(+150)
    # The phi = 0 column under the aircraft: energy mean of ring(+-theta),
    # 10*lg((10^6.9 + 10^5.1)/2) at theta = 90.
    mid = 10.0 * np.log10((10.0**6.9 + 10.0**5.1) / 2.0)
    assert h.levels[ia[0.0], ip[90.0], 0] == pytest.approx(mid)
    # Constant in phi within each half: the whole starboard row of theta = 60.
    assert np.allclose(h.levels[[ia[10.0], ia[50.0], ia[90.0]], ip[60.0], :], 66.0)


def test_hover_ring_periodic_energy_interpolation() -> None:
    # Alternating 70/80 dB ring: lookups between bearings blend in the energy
    # domain, and the wrap cell between +150 and -180 interpolates too.
    alternating = np.where(np.arange(12) % 2 == 0, 70.0, 80.0)
    h = hover_ring_hemisphere([500.0, 1000.0], _RING_BEARINGS, _ring(alternating))
    # Rim lookup at theta = 15: halfway between ring(0) = 70 and ring(30) = 80,
    # through the grid nodes at 10 and 20 (both stages are linear in energy).
    expected = 10.0 * np.log10((10.0**7.0 + 10.0**8.0) / 2.0)
    assert hemisphere_source_level(h, 90.0, 15.0)[0] == pytest.approx(expected)
    # Grid node on the wrap: ring(170), two thirds from +150 (80 dB) towards
    # -180 (70 dB).
    ia = {a: i for i, a in enumerate(h.azimuth)}
    ip = {p: i for i, p in enumerate(h.polar)}
    wrap = 10.0 * np.log10((10.0**8.0 + 2.0 * 10.0**7.0) / 3.0)
    assert h.levels[ia[90.0], ip[170.0], 0] == pytest.approx(wrap)


def test_hover_ring_middle_column_survives_inexact_step() -> None:
    # A step of 180/14 deg builds a symmetric axis whose middle node can land
    # a few ulp away from 0.0 in floating point; the phi = 0 column is
    # selected structurally (by index, the axis being symmetric by
    # construction), so it still takes the energy mean of ring(+-theta) and
    # the halves still split port/starboard around it.
    h = hover_ring_hemisphere(
        [500.0, 1000.0],
        _RING_BEARINGS,
        _ring(60.0 + _RING_BEARINGS / 10.0),
        azimuth_step=180.0 / 14.0,
    )
    assert h.azimuth.size == 15
    ip = {p: i for i, p in enumerate(h.polar)}
    mid = 10.0 * np.log10((10.0**6.9 + 10.0**5.1) / 2.0)
    assert h.levels[7, ip[90.0], 0] == pytest.approx(mid)
    assert h.levels[8, ip[60.0], 0] == pytest.approx(66.0)  # starboard side
    assert h.levels[6, ip[60.0], 0] == pytest.approx(54.0)  # port side


def test_hover_ring_bearing_mapping_closed_form() -> None:
    # The NORAH2 reference implementation's reading: the level depends on the
    # horizontal bearing beta = atan2(sin(theta)*sin(phi), cos(theta)).
    h = hover_ring_hemisphere(
        [500.0, 1000.0],
        _RING_BEARINGS,
        _ring(60.0 + _RING_BEARINGS / 10.0),
        mapping="bearing",
    )
    ia = {a: i for i, a in enumerate(h.azimuth)}
    ip = {p: i for i, p in enumerate(h.polar)}
    # The nadir bin reads the ring at bearing 0 (the prototype's convention).
    assert h.levels[ia[0.0], ip[90.0], 0] == pytest.approx(60.0)
    # Behind and below (phi = 0, theta = 120): bearing 180, the -180 row.
    assert h.levels[ia[0.0], ip[120.0], 0] == pytest.approx(42.0)
    # On the rim both mappings coincide: (phi, theta) = (+90, 60) is ring(+60).
    assert h.levels[ia[90.0], ip[60.0], 0] == pytest.approx(66.0)
    # Interior bin (phi, theta) = (+30, 60): beta = atan2(sin60 sin30, cos60),
    # energy-interpolated between ring(+30) and ring(+60).
    beta = np.degrees(np.arctan2(np.sin(np.radians(60.0)) * 0.5, 0.5))
    w = (beta - 30.0) / 30.0
    expected = 10.0 * np.log10((1.0 - w) * 10.0**6.3 + w * 10.0**6.6)
    assert h.levels[ia[30.0], ip[60.0], 0] == pytest.approx(expected)


def test_hover_derived_offsets_are_the_published_constants() -> None:
    # Table 3 Approach 3: +12 / -12 / -2.5 dB from the in-ground-hover disk,
    # uniformly on every band and bin, with NaN bins staying NaN.
    freqs = np.array([500.0, 1000.0])
    az = np.arange(-90.0, 91.0, 10.0)
    po = np.arange(0.0, 181.0, 10.0)
    levels = np.random.default_rng(7).uniform(60.0, 90.0, (19, 19, 2))
    levels[3, 4, 0] = np.nan
    hige = RotorcraftHemisphere(freqs, az, po, levels.copy(), distance=70.0)
    for condition, offset in (
        ("out_of_ground_hover", 12.0),
        ("reduced_rpm_idle", -12.0),
        ("full_rpm_idle", -2.5),
    ):
        got = hover_derived_hemisphere(hige, condition)
        finite = np.isfinite(levels)
        assert np.allclose(got.levels[finite], levels[finite] + offset)
        assert np.isnan(got.levels[3, 4, 0])
        assert got.distance == 70.0
        assert np.array_equal(got.frequencies, freqs)
        assert np.array_equal(got.azimuth, az)
        assert np.array_equal(got.polar, po)
    # The input hemisphere is left untouched.
    assert np.array_equal(hige.levels, levels, equal_nan=True)


def test_hover_derived_approach2_measured_offset() -> None:
    # Approach 2: LA_HOGE(0) = 95.3, LA_HIGE(0) = 88.1 -> offset 7.2 dB,
    # applied bin for bin; the 0-degree lookup rises by exactly that.
    h = hover_ring_hemisphere(
        [500.0, 1000.0], _RING_BEARINGS, _ring(60.0 + _RING_BEARINGS / 10.0)
    )
    hoge = hover_derived_hemisphere(h, "out_of_ground_hover", offset_db=95.3 - 88.1)
    assert np.allclose(hoge.levels - h.levels, 7.2)
    before = hemisphere_source_level(h, 0.0, 0.0)
    after = hemisphere_source_level(hoge, 0.0, 0.0)
    assert after - before == pytest.approx([7.2, 7.2])


def test_hover_helpers_validation() -> None:
    freqs = [500.0, 1000.0]
    ring = _ring(np.full(12, 80.0))
    with pytest.raises(ValueError, match="frequencies"):
        hover_ring_hemisphere([500.0, 0.0], _RING_BEARINGS, ring)
    with pytest.raises(ValueError, match="frequencies"):
        hover_ring_hemisphere([500.0, -1000.0], _RING_BEARINGS, ring)
    with pytest.raises(ValueError, match="bearings"):
        hover_ring_hemisphere(freqs, [0.0], [[80.0, 80.0]])
    with pytest.raises(ValueError, match="bearings"):
        hover_ring_hemisphere(freqs, _RING_BEARINGS[::-1], ring)
    with pytest.raises(ValueError, match="bearings"):
        hover_ring_hemisphere(freqs, _RING_BEARINGS + 200.0, ring)
    with pytest.raises(ValueError, match="levels"):
        hover_ring_hemisphere(freqs, _RING_BEARINGS, ring[:, :1])
    with pytest.raises(ValueError, match="levels"):
        hover_ring_hemisphere(freqs, _RING_BEARINGS, np.where(ring > 0.0, np.nan, ring))
    with pytest.raises(ValueError, match="distance"):
        hover_ring_hemisphere(freqs, _RING_BEARINGS, ring, distance=0.0)
    with pytest.raises(ValueError, match="azimuth_step"):
        hover_ring_hemisphere(freqs, _RING_BEARINGS, ring, azimuth_step=7.0)
    with pytest.raises(ValueError, match="polar_step"):
        hover_ring_hemisphere(freqs, _RING_BEARINGS, ring, polar_step=1000.0)
    with pytest.raises(ValueError, match="mapping"):
        hover_ring_hemisphere(freqs, _RING_BEARINGS, ring, mapping="nearest")
    h = hover_ring_hemisphere(freqs, _RING_BEARINGS, ring)
    with pytest.raises(ValueError, match="condition"):
        hover_derived_hemisphere(h, "hover")
    with pytest.raises(ValueError, match="offset_db"):
        hover_derived_hemisphere(h, "full_rpm_idle", offset_db=float("nan"))


def test_hemisphere_axes_must_span_the_level_grid() -> None:
    # The band axis is the quiet one: a `frequencies` axis that does not match
    # the band axis of `levels` still looks up and still plots, and the plot
    # labels its curve with a frequency the levels were never computed at. The
    # angle axes eventually raise from inside the gap fill, in numpy's words;
    # the construction names the field instead.
    h = hover_ring_hemisphere([500.0, 1000.0], _RING_BEARINGS, _ring(np.full(12, 80.0)))
    extra_band = np.append(h.frequencies, 2000.0)
    with pytest.raises(ValueError, match=r"'frequencies'.*per band"):
        dataclasses.replace(h, frequencies=extra_band)
    short_azimuth = h.azimuth[:-1]
    with pytest.raises(ValueError, match=r"'azimuth'.*per azimuth angle"):
        dataclasses.replace(h, azimuth=short_azimuth)
    extra_polar = np.append(h.polar, 200.0)
    with pytest.raises(ValueError, match=r"'polar'.*per polar angle"):
        dataclasses.replace(h, polar=extra_polar)


# --------------------------------------------------------------------------- #
# Flight-condition interpolation (guidance Eq. 3-10)
# --------------------------------------------------------------------------- #

from phonometry.aircraft.rotorcraft_noise import (
    RotorcraftAtmosphere,
    RotorcraftGround,
    RotorcraftTrackState,
    flight_condition_weights,
    flight_path_kinematics,
    interpolated_source_level,
    rotorcraft_event_level,
    rotorcraft_noise_contour,
)

#: The hard-ground, 0.1 m microphone site the flyover helper measures on.
_HARD_SITE = RotorcraftGround(receiver_height=0.1, flow_resistivity="H")


def _uniform_hemisphere(
    level: float,
    bands: list[float] | None = None,
) -> RotorcraftHemisphere:
    freqs = np.asarray(bands if bands is not None else [50.0], dtype=np.float64)
    az = np.arange(-90.0, 91.0, 10.0)
    po = np.arange(0.0, 181.0, 10.0)
    return RotorcraftHemisphere(
        freqs, az, po, np.full((az.size, po.size, freqs.size), level)
    )


def test_fc_weights_single_hemisphere() -> None:
    assert flight_condition_weights([55.0], [0.0], 80.0, -5.0) == [(0, 1.0)]


def test_fc_weights_exact_condition_wins() -> None:
    w = flight_condition_weights([50.0, 70.0, 60.0], [0.0, 0.0, 10.0], 70.0, 0.0)
    assert w == [(1, 1.0)]


def test_fc_weights_triangle_hand_checked() -> None:
    # Normalised (Eq. 3-6, Ffc = 2, spans 20 kt / 10 deg): points (2.5, 0),
    # (3.5, 0), (3.0, 2); query (3.0, 0.5) -> deltas sqrt(0.5), sqrt(0.5), 1.5
    # -> weights (1/d)/sum(1/d) = 0.404629, 0.404629, 0.190743 (Eq. 7/8).
    w = dict(flight_condition_weights([50.0, 70.0, 60.0], [0.0, 0.0, 10.0], 60.0, 2.5))
    assert w[0] == pytest.approx(0.404629, abs=1e-6)
    assert w[1] == pytest.approx(0.404629, abs=1e-6)
    assert w[2] == pytest.approx(0.190743, abs=1e-6)
    assert sum(w.values()) == pytest.approx(1.0)


def test_fc_weights_outside_hull_nearest_neighbour() -> None:
    # Beyond the measured envelope the nearest condition is adopted unblended
    # (Eq. 9/10), the Doc 32 1st ed. behaviour.
    w = flight_condition_weights([50.0, 70.0, 60.0], [0.0, 0.0, 10.0], 95.0, -10.0)
    assert w == [(1, 1.0)]


def test_fc_weights_collinear_conditions_fall_back_to_nearest() -> None:
    # All-level-flight databases have no triangulation; nearest neighbour.
    w = flight_condition_weights([50.0, 60.0, 70.0], [0.0, 0.0, 0.0], 56.0, 0.0)
    assert w == [(1, 1.0)]


def test_fc_weights_are_unit_invariant() -> None:
    kts = [50.0, 70.0, 60.0]
    ms = [v * 0.514444 for v in kts]
    w1 = flight_condition_weights(kts, [0.0, 0.0, 10.0], 60.0, 2.5)
    w2 = flight_condition_weights(ms, [0.0, 0.0, 10.0], 60.0 * 0.514444, 2.5)
    for (i1, a), (i2, b) in zip(w1, w2, strict=True):
        assert i1 == i2
        assert a == pytest.approx(b, abs=1e-12)


def test_fc_weights_lookup_table_is_honoured() -> None:
    # A square of conditions splits into two triangles either way; the lookup
    # table decides which one blends the query (guidance step 4 admits a
    # precomputed table, and the NORAH database ships one per type).
    v = [50.0, 70.0, 50.0, 70.0]
    g = [0.0, 0.0, 10.0, 10.0]
    w = dict(flight_condition_weights(v, g, 66.0, 2.0, triangles=[[0, 1, 3]]))
    assert set(w) == {0, 1, 3}
    w2 = dict(flight_condition_weights(v, g, 66.0, 2.0, triangles=[[0, 2, 3]]))
    assert set(w2) == {1}  # outside the only listed triangle: nearest


def test_fc_weights_validation() -> None:
    with pytest.raises(ValueError, match=r"'path_angles'.*same shape"):
        flight_condition_weights([50.0, 60.0], [0.0], 55.0, 0.0)
    with pytest.raises(ValueError, match=r"'airspeeds'.*non-empty"):
        flight_condition_weights([[50.0, 60.0]], [[0.0, 1.0]], 55.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        flight_condition_weights([50.0, 60.0], [0.0, 0.0], np.nan, 0.0)
    with pytest.raises(ValueError, match="shape"):
        flight_condition_weights(
            [50.0, 60.0, 70.0], [0.0, 1.0, 2.0], 55.0, 0.0, triangles=[[0, 1]]
        )
    with pytest.raises(ValueError, match="indices"):
        flight_condition_weights(
            [50.0, 60.0, 70.0], [0.0, 1.0, 2.0], 55.0, 0.0, triangles=[[0, 1, 9]]
        )


def test_interpolated_source_level_hand_checked_blend() -> None:
    # The conformance simplex: uniform 100/90/95 dB hemispheres blended with
    # the Eq. 8 weights give 10*lg(sum w*10^(L/10)) = 97.0367 dB by hand.
    hems = [
        _uniform_hemisphere(100.0),
        _uniform_hemisphere(90.0),
        _uniform_hemisphere(95.0),
    ]
    got = interpolated_source_level(
        hems, [50.0, 70.0, 60.0], [0.0, 0.0, 10.0], 60.0, 2.5, 0.0, 90.0
    )
    assert got[0] == pytest.approx(97.0367, abs=1e-3)


def test_interpolated_source_level_validates_band_grids() -> None:
    hems = [_uniform_hemisphere(100.0), _uniform_hemisphere(90.0, [63.0])]
    single = [_uniform_hemisphere(100.0)]
    with pytest.raises(ValueError, match="band grid"):
        interpolated_source_level(hems, [50.0, 70.0], [0.0, 0.0], 60.0, 0.0, 0.0, 90.0)
    with pytest.raises(ValueError, match="equal length"):
        interpolated_source_level(
            single, [50.0, 70.0], [0.0, 0.0], 60.0, 0.0, 0.0, 90.0
        )


def test_hemisphere_mirrored_reverses_azimuth() -> None:
    az = np.arange(-90.0, 91.0, 10.0)
    po = np.arange(0.0, 181.0, 10.0)
    lv = np.zeros((az.size, po.size, 1))
    lv[:, :, 0] = 70.0 + 0.1 * az[:, None]  # port quieter than starboard
    h = RotorcraftHemisphere(np.array([100.0]), az, po, lv)
    m = h.mirrored()
    assert np.allclose(
        hemisphere_source_level(m, 30.0, 90.0), hemisphere_source_level(h, -30.0, 90.0)
    )
    back = m.mirrored()
    assert np.allclose(back.levels, h.levels)
    assert np.allclose(back.azimuth, h.azimuth)


# --------------------------------------------------------------------------- #
# Flight-path kinematics (guidance Eq. 16-21 / Doc 32 Eq. 8-10)
# --------------------------------------------------------------------------- #


def test_kinematics_straight_climb_closed_form() -> None:
    t = np.arange(0.0, 10.5, 0.5)
    vg, gamma, heading = 40.0, np.radians(5.0), np.radians(30.0)
    pos = np.column_stack(
        [
            vg * np.sin(heading) * t,
            vg * np.cos(heading) * t,
            100.0 + vg * np.tan(gamma) * t,
        ]
    )
    kin = flight_path_kinematics(t, pos)
    assert np.allclose(kin.ground_speed, 40.0)
    assert np.allclose(kin.airspeed, 40.0 / np.cos(gamma))
    assert np.allclose(kin.heading, 30.0)
    assert np.allclose(kin.path_angle, 5.0)
    assert np.allclose(kin.curvature, 0.0, atol=1e-12)
    assert np.allclose(kin.bank_angle, 0.0, atol=1e-9)


def test_kinematics_turn_bank_closed_form() -> None:
    # Constant-speed level turn of radius R: K = 1/R and bank = atan(V^2/(gR))
    # (Eq. 18/20). A starboard turn (heading increasing) banks starboard down.
    radius, speed = 500.0, 40.0
    t = np.arange(0.0, 30.1, 0.25)
    ang = speed * t / radius
    pos = np.column_stack(
        [
            radius * np.sin(ang),
            radius * (1.0 - np.cos(ang)) * -1.0 + radius,
            np.full_like(t, 300.0),
        ]
    )
    # circle centred at (0, radius): heading starts at 0 and increases
    kin = flight_path_kinematics(t, pos)
    mid = slice(10, -10)
    assert np.allclose(kin.ground_speed[mid], speed, rtol=1e-3)
    assert np.allclose(kin.curvature[mid], 1.0 / radius, rtol=1e-2)
    expected_bank = np.degrees(np.arctan(speed**2 / (9.80665 * radius)))
    assert np.allclose(kin.bank_angle[mid], expected_bank, atol=0.2)
    assert np.all(kin.bank_angle[mid] > 0.0)


def test_kinematics_descent_negative_path_angle() -> None:
    t = np.arange(0.0, 5.5, 0.5)
    pos = np.column_stack(
        [np.zeros_like(t), 30.0 * t, 300.0 - 30.0 * np.tan(np.radians(6.0)) * t]
    )
    kin = flight_path_kinematics(t, pos)
    assert np.allclose(kin.path_angle, -6.0)


def test_kinematics_hover_is_finite() -> None:
    t = np.arange(0.0, 5.5, 0.5)
    pos = np.column_stack([np.zeros_like(t), np.zeros_like(t), np.full_like(t, 20.0)])
    kin = flight_path_kinematics(t, pos)
    assert np.allclose(kin.ground_speed, 0.0)
    assert np.all(np.isfinite(kin.curvature))
    assert np.allclose(kin.bank_angle, 0.0)


def test_kinematics_validation() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        flight_path_kinematics([0.0, 0.0], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="shape"):
        flight_path_kinematics([0.0, 1.0], [[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="at least two"):
        flight_path_kinematics([0.0], [[0.0, 0.0, 0.0]])


def test_kinematics_columns_must_span_the_track() -> None:
    # The event chain walks the eight columns row by row with no check of its
    # own, so a column of the wrong length has to be refused where it is built.
    t = np.arange(0.0, 5.5, 0.5)
    pos = np.column_stack([np.zeros_like(t), 40.0 * t, np.full_like(t, 100.0)])
    kin = flight_path_kinematics(t, pos)
    short_airspeed = kin.airspeed[:-1]
    with pytest.raises(ValueError, match=r"'airspeed'.*per track point"):
        dataclasses.replace(kin, airspeed=short_airspeed)
    flat_positions = kin.positions[:, 0]
    with pytest.raises(ValueError, match=r"'positions' must have 2 axes"):
        dataclasses.replace(kin, positions=flat_positions)


def test_kinematics_plot() -> None:
    t = np.arange(0.0, 10.5, 0.5)
    pos = np.column_stack([np.zeros_like(t), 40.0 * t, 100.0 + 2.0 * t])
    assert flight_path_kinematics(t, pos).plot() is not None


# --------------------------------------------------------------------------- #
# Single event (Doc 32 §6.1)
# --------------------------------------------------------------------------- #


def _flyover(
    level: float = 100.0,
    bands: list[float] | None = None,
    span: float = 6000.0,
    dt: float = 0.5,
    height: float = 100.1,
    **kwargs: object,
) -> tuple:
    """A constant-speed level flyover along y over the origin receiver."""
    speed = 50.0
    t = np.arange(0.0, 2.0 * span / speed + dt / 2.0, dt)
    pos = np.column_stack([np.zeros_like(t), speed * t - span, np.full_like(t, height)])
    hems = [_uniform_hemisphere(level, bands)]
    kwargs.setdefault("ground", _HARD_SITE)
    res = rotorcraft_event_level(hems, [speed], [0.0], t, pos, (0.0, 0.0), **kwargs)  # type: ignore[arg-type]
    return t, pos, res


def test_event_geometry_overhead() -> None:
    _t, _pos, res = _flyover(bands=[31.5])
    k = int(np.argmin(res.distance))
    assert res.distance[k] == pytest.approx(100.0, abs=1e-9)
    assert res.polar[k] == pytest.approx(90.0, abs=1e-6)
    assert abs(res.azimuth[k]) == pytest.approx(0.0, abs=1e-6)
    # Retarded time (Eq. 22) with c = 346.1 m/s, at every step.
    assert np.allclose(res.times, res.emission_times + res.distance / 346.1)
    # Approaching rows look forward (theta < 90), receding rows rearward.
    assert np.all(res.polar[:k] < 90.0)
    assert np.all(res.polar[k + 1 :] > 90.0)


def test_event_bank_tilt_moves_azimuth() -> None:
    # Rolling starboard-down by 25 deg swings the below-aircraft receiver to
    # phi = +25 deg in the tilted hemisphere frame (guidance §A.3.4).
    _, _, level_res = _flyover(bands=[31.5])
    _, _, banked = _flyover(
        bands=[31.5], track_state=RotorcraftTrackState(bank_angle=25.0)
    )
    k = int(np.argmin(level_res.distance))
    assert level_res.azimuth[k] == pytest.approx(0.0, abs=1e-6)
    assert banked.azimuth[k] == pytest.approx(25.0, abs=1e-6)


def test_event_heading_override_mirrors_polar() -> None:
    # Reversing the heading turns approach geometry into recession: the polar
    # angles mirror about 90 deg (the frame is heading-only, no pitch tilt).
    _, _, fwd = _flyover(bands=[31.5])
    _, _, rev = _flyover(bands=[31.5], track_state=RotorcraftTrackState(heading=180.0))
    assert np.allclose(fwd.polar + rev.polar, 180.0, atol=1e-9)


def test_event_lamax_assembles_the_propagation_chain() -> None:
    # At the closest approach the received level is the source level plus
    # dLs + dLa + dLg + A, each from the public primitives.
    _, _, res = _flyover(bands=[31.5])
    k = int(np.argmin(res.distance))
    freqs = np.array([31.5])
    expected = (
        100.0
        + spherical_spreading_adjustment(100.0)
        + atmospheric_adjustment(freqs, 100.0)[0]
        + ground_effect_adjustment(freqs, 100.1, 0.1, 1e-6, flow_resistivity="H")[0]
    )
    a31 = -39.525  # IEC 61672-1 analytic A-weighting at 31.5 Hz
    assert res.a_levels[k] == pytest.approx(expected + a31, abs=2e-3)
    assert res.la_max == pytest.approx(res.a_levels[k])
    assert res.band_levels[k, 0] == pytest.approx(expected, abs=2e-3)


def test_event_sel_matches_lorentzian_closed_form() -> None:
    # Uniform source, constant dLg: SEL - LASmax = 10*lg(pi*d/V) = 7.982 dB.
    _, _, res = _flyover(bands=[31.5])
    assert res.sel - res.la_max == pytest.approx(7.982, abs=0.1)
    assert res.sel_10db <= res.sel + 1e-9
    # The 10 dB-down window spans r <= sqrt(10)*d, i.e. (2/pi)*atan(3) = 79.5 %
    # of the Lorentzian exposure energy: about 1 dB below the full history.
    assert res.sel - res.sel_10db == pytest.approx(1.0, abs=0.2)


def test_event_level_offset_shifts_levels() -> None:
    _, _, base = _flyover(bands=[31.5])
    _, _, up = _flyover(bands=[31.5], level_offset=3.0)
    assert np.allclose(up.a_levels, base.a_levels + 3.0)
    assert up.sel == pytest.approx(base.sel + 3.0, abs=1e-9)
    assert up.la_max == pytest.approx(base.la_max + 3.0, abs=1e-9)


def test_event_epnl_undefined_off_noy_grid() -> None:
    # A single 31.5 Hz band cannot address the 24 noy bands: EPNL undefined.
    _, _, res = _flyover(bands=[31.5])
    assert np.all(np.isnan(res.pnlt))
    assert np.isnan(res.pnltm)
    assert np.isnan(res.epnl)


_NORAH_BANDS = [
    10.0,
    12.5,
    16.0,
    20.0,
    25.0,
    31.5,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
]


def test_event_epnl_on_the_standard_grid() -> None:
    _, _, res = _flyover(level=90.0, bands=_NORAH_BANDS, span=2000.0)
    assert np.isfinite(res.epnl)
    assert np.isfinite(res.pnltm)
    assert res.pnltm >= np.nanmax(res.pnlt) - 1e-9
    k = int(np.argmin(res.distance))
    assert np.isfinite(res.pnlt[k])


def test_event_atmospheric_methods_agree_at_low_frequency() -> None:
    _, _, iso = _flyover(bands=[31.5], span=2000.0)
    _, _, sae = _flyover(
        bands=[31.5],
        span=2000.0,
        atmosphere=RotorcraftAtmosphere(atmospheric_method="sae"),
    )
    assert sae.la_max == pytest.approx(iso.la_max, abs=0.05)
    assert sae.sel == pytest.approx(iso.sel, abs=0.05)


def test_event_validation() -> None:
    hems = [_uniform_hemisphere(100.0)]
    t = np.array([0.0, 1.0])
    pos = np.array([[0.0, -50.0, 100.0], [0.0, 50.0, 100.0]])
    with pytest.raises(ValueError, match="receiver"):
        rotorcraft_event_level(hems, [50.0], [0.0], t, pos, (0.0, 0.0, 0.0, 0.0))
    unknown_method_atmosphere = RotorcraftAtmosphere(atmospheric_method="exact")
    with pytest.raises(ValueError, match="atmospheric_method"):
        rotorcraft_event_level(
            hems,
            [50.0],
            [0.0],
            t,
            pos,
            (0.0, 0.0),
            atmosphere=unknown_method_atmosphere,
        )
    mismatched_airspeed_state = RotorcraftTrackState(airspeed=[50.0, 50.0, 50.0])
    with pytest.raises(ValueError, match="airspeed"):
        rotorcraft_event_level(
            hems,
            [50.0],
            [0.0],
            t,
            pos,
            (0.0, 0.0),
            track_state=mismatched_airspeed_state,
        )
    with pytest.raises(ValueError, match="level_offset"):
        rotorcraft_event_level(
            hems, [50.0], [0.0], t, pos, (0.0, 0.0), level_offset=[1.0, 2.0, 3.0]
        )
    nan_elevation_ground = RotorcraftGround(ground_elevation=np.nan)
    with pytest.raises(ValueError, match="ground_elevation"):
        rotorcraft_event_level(
            hems, [50.0], [0.0], t, pos, (0.0, 0.0), ground=nan_elevation_ground
        )


def test_event_history_and_spectrum_must_line_up() -> None:
    # Every per-step array is one column of the same time history, and a reader
    # that takes the step of LASmax reads the geometry back out at that index;
    # the band axis of `band_levels` is pinned by `frequencies` the same way.
    _t, _pos, res = _flyover(bands=[31.5], span=1000.0)
    short_distance = res.distance[:-1]
    with pytest.raises(ValueError, match=r"'distance'.*per emission step"):
        dataclasses.replace(res, distance=short_distance)
    extra_band = np.append(res.frequencies, 63.0)
    with pytest.raises(ValueError, match=r"'frequencies'.*per band"):
        dataclasses.replace(res, frequencies=extra_band)


def test_a_non_finite_a_weighted_step_is_refused() -> None:
    """The figure marks LASmax at ``argmax(a_levels)``, which a NaN wins.

    ``argmax`` of an array carrying a NaN answers with the NaN's own index, so
    the marker floats over the break in the curve and is labelled as the
    maximum. ``pnlt`` is the field licensed to hold NaN, and stays exempt.
    """
    _t, _pos, res = _flyover(bands=[31.5], span=1000.0)
    with_gap = np.asarray(res.a_levels, dtype=np.float64).copy()
    with_gap[0] = np.nan
    with pytest.raises(ValueError, match="'a_levels' must be finite"):
        dataclasses.replace(res, a_levels=with_gap)


def test_event_plot() -> None:
    _, _, res = _flyover(level=90.0, bands=_NORAH_BANDS, span=1000.0)
    assert res.plot() is not None


# --------------------------------------------------------------------------- #
# Contour (Doc 32 §6.3)
# --------------------------------------------------------------------------- #


def _contour_inputs() -> tuple:
    speed = 50.0
    t = np.arange(0.0, 40.5, 0.5)
    pos = np.column_stack(
        [np.zeros_like(t), speed * t - 1000.0, np.full_like(t, 150.0)]
    )
    hems = [_uniform_hemisphere(95.0, [125.0])]
    return hems, [speed], [0.0], t, pos


def test_contour_matches_per_point_events() -> None:
    hems, spd, ang, t, pos = _contour_inputs()
    x = np.array([-200.0, 0.0, 150.0])
    y = np.array([-300.0, 100.0])
    for metric, attr in (("exposure", "sel"), ("maximum", "la_max")):
        res = rotorcraft_noise_contour(hems, spd, ang, t, pos, x=x, y=y, metric=metric)
        assert res.level.shape == (2, 3)
        for i, yy in enumerate(y):
            for j, xx in enumerate(x):
                ev = rotorcraft_event_level(hems, spd, ang, t, pos, (xx, yy))
                assert res.level[i, j] == pytest.approx(getattr(ev, attr), abs=1e-9), (
                    metric,
                    xx,
                    yy,
                )


def test_contour_is_symmetric_across_the_track() -> None:
    hems, spd, ang, t, pos = _contour_inputs()
    x = np.array([-400.0, -100.0, 100.0, 400.0])
    y = np.array([-200.0, 0.0, 200.0])
    res = rotorcraft_noise_contour(hems, spd, ang, t, pos, x=x, y=y)
    assert np.allclose(res.level[:, [0, 1]], res.level[:, [3, 2]], atol=1e-9)


def test_contour_validation() -> None:
    hems, spd, ang, t, pos = _contour_inputs()
    with pytest.raises(ValueError, match="metric"):
        rotorcraft_noise_contour(
            hems, spd, ang, t, pos, x=[0.0, 1.0], y=[0.0, 1.0], metric="epnl"
        )
    with pytest.raises(ValueError, match="grid"):
        rotorcraft_noise_contour(hems, spd, ang, t, pos, x=[0.0], y=[0.0, 1.0])


def test_contour_level_grid_must_span_its_coordinates() -> None:
    # Matplotlib refuses these too, but only when someone draws the grid and in
    # its own words (the length of x against the columns of z); the result
    # names the field that disagrees at the moment it is built.
    hems, spd, ang, t, pos = _contour_inputs()
    res = rotorcraft_noise_contour(
        hems,
        spd,
        ang,
        t,
        pos,
        x=np.array([-200.0, 0.0, 150.0]),
        y=np.array([-300.0, 100.0]),
    )
    short_x = res.x[:-1]
    with pytest.raises(ValueError, match=r"'x'.*per grid column"):
        dataclasses.replace(res, x=short_x)
    extra_y = np.append(res.y, 500.0)
    with pytest.raises(ValueError, match=r"'y'.*per grid row"):
        dataclasses.replace(res, y=extra_y)


def test_contour_metric_outside_the_two_the_type_states_is_refused() -> None:
    # The entry point checks the tag at the call, but a grid rewritten by hand
    # reaches the colorbar, which asks only whether it is "exposure" and calls
    # everything else LASmax: the map is labelled with a metric it was never
    # computed in, and nothing raises.
    hems, spd, ang, t, pos = _contour_inputs()
    res = rotorcraft_noise_contour(
        hems,
        spd,
        ang,
        t,
        pos,
        x=np.array([-200.0, 0.0, 150.0]),
        y=np.array([-300.0, 100.0]),
    )
    with pytest.raises(ValueError, match="'metric' must be one of"):
        dataclasses.replace(res, metric="epnl")


def test_contour_plot() -> None:
    hems, spd, ang, t, pos = _contour_inputs()
    res = rotorcraft_noise_contour(
        hems,
        spd,
        ang,
        t,
        pos,
        x=np.linspace(-500.0, 500.0, 5),
        y=np.linspace(-500.0, 500.0, 5),
    )
    assert res.plot() is not None

#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the reverberation-time prediction models.

The source texts (Arau-Puchades 1988; Carrion Isbert; Everest) carry worked
examples, but none is machine-readable as a clean oracle, so the suite anchors
on hand-computed values of the closed-form expressions and on the structural
identities that relate the models: every model collapses to Eyring for a
uniform absorption distribution, and Eyring collapses to Sabine as the
absorption tends to zero.

Reference constants use the Sabine numerator ``k = 24 ln 10 / c0`` with the
default ``c0 = 343 m/s`` (``k = 0.161113...``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
import reference_data as ref

from phonometry import room

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    #: The models called as ``model(volume, [(surface, alpha), ...])``.
    _SurfaceModel = Callable[[float, list[tuple[float, float]]], np.ndarray | float]
    #: The models called as ``model(dimensions, per_axis_alphas)``.
    _AxialModel = Callable[
        [tuple[float, float, float], tuple[float, float, float]], np.ndarray | float
    ]

# A shoebox 8 x 5 x 3 m: V = 120 m3, S = 2(40 + 24 + 15) = 158 m2.
DIMS = (8.0, 5.0, 3.0)
VOLUME = 120.0
SURFACE = 158.0
# Its six boundary surfaces, one entry per wall (two per axis).
UNIFORM_SURFACES = [
    (40.0, 0.2),
    (40.0, 0.2),  # z-pair (floor/ceiling)
    (24.0, 0.2),
    (24.0, 0.2),  # y-pair
    (15.0, 0.2),
    (15.0, 0.2),  # x-pair
]

_K = 24.0 * math.log(10.0) / 343.0


# ---------------------------------------------------------------------------
# Hand-computed closed-form values
# ---------------------------------------------------------------------------
def test_sabine_hand_value() -> None:
    """Sabine: A = 158*0.2 = 31.6, T = k*120/31.6 = 0.611825 s."""
    t = room.sabine_reverberation_time(VOLUME, UNIFORM_SURFACES)
    assert t == pytest.approx(0.6118246547, abs=1e-6)
    assert t == pytest.approx(_K * VOLUME / (SURFACE * 0.2), abs=1e-9)


def test_eyring_hand_value() -> None:
    """Eyring: -158*ln(0.8) = 35.2567, T = k*120/35.2567 = 0.548369 s."""
    t = room.eyring_reverberation_time(VOLUME, UNIFORM_SURFACES)
    assert t == pytest.approx(0.5483686633, abs=1e-6)


def test_millington_hand_value() -> None:
    """Millington (V=100): -(60 ln0.7 + 40 ln0.4) = 58.0521, T = 0.277533 s."""
    t = room.millington_sette_reverberation_time(100.0, [(60.0, 0.3), (40.0, 0.6)])
    assert t == pytest.approx(0.2775330330, abs=1e-6)


def test_mean_absorption() -> None:
    assert room.mean_absorption(UNIFORM_SURFACES) == pytest.approx(0.2)
    # Area-weighted, not arithmetic: 90 % of the area at 0.1, 10 % at 0.9.
    assert room.mean_absorption([(90.0, 0.1), (10.0, 0.9)]) == pytest.approx(0.18)


# ---------------------------------------------------------------------------
# Real worked oracle -- Everest, Master Handbook of Acoustics 4th ed, Fig. 7-22
# ---------------------------------------------------------------------------
def test_sabine_matches_everest_example1() -> None:
    """SI Sabine reproduces the six printed RT60 of Example 1 to <= 0.02 s.

    The imperial worked example (RT60 = 0.049 V / Sa) is converted to SI; the
    residual is the book rounding its 0.049 constant. A genuine numeric oracle
    anchored on measured concrete/gypsum absorption coefficients.
    """
    for i, band in enumerate(ref.EVEREST_EX1_BANDS):
        surfaces = [
            (ref.EVEREST_EX1_FLOOR_AREA, ref.EVEREST_EX1_FLOOR_ALPHA[i]),
            (ref.EVEREST_EX1_SHELL_AREA, ref.EVEREST_EX1_SHELL_ALPHA[i]),
        ]
        t = room.sabine_reverberation_time(ref.EVEREST_EX1_VOLUME, surfaces)
        assert t == pytest.approx(ref.EVEREST_EX1_RT[i], abs=0.02), f"band {band} Hz"


# ---------------------------------------------------------------------------
# Structural identities
# ---------------------------------------------------------------------------
def test_uniform_absorption_all_models_equal_eyring() -> None:
    """Every model collapses to Eyring for a uniform absorption distribution."""
    eyring = room.eyring_reverberation_time(VOLUME, UNIFORM_SURFACES)
    milli = room.millington_sette_reverberation_time(VOLUME, UNIFORM_SURFACES)
    fitz = room.fitzroy_reverberation_time(DIMS, (0.2, 0.2, 0.2))
    arau = room.arau_puchades_reverberation_time(DIMS, (0.2, 0.2, 0.2))
    assert milli == pytest.approx(eyring, abs=1e-12)
    assert fitz == pytest.approx(eyring, abs=1e-12)
    assert arau == pytest.approx(eyring, abs=1e-12)


def test_eyring_reduces_to_sabine_in_the_low_absorption_limit() -> None:
    """As alpha -> 0, -S ln(1-alpha_bar) -> S alpha_bar = A, so Eyring -> Sabine."""
    faint = [(area, 0.001) for area, _ in UNIFORM_SURFACES]
    sabine = room.sabine_reverberation_time(VOLUME, faint)
    eyring = room.eyring_reverberation_time(VOLUME, faint)
    assert eyring == pytest.approx(sabine, rel=1e-3)
    # Eyring is always the shorter time (it never overestimates like Sabine).
    assert eyring < sabine


def test_sabine_overestimates_at_high_absorption() -> None:
    """With strong absorption Sabine gives a longer time than Eyring."""
    strong = [(area, 0.6) for area, _ in UNIFORM_SURFACES]
    assert room.sabine_reverberation_time(
        VOLUME, strong
    ) > room.eyring_reverberation_time(VOLUME, strong)


# ---------------------------------------------------------------------------
# Axial models on a non-uniform (anisotropic) distribution
# ---------------------------------------------------------------------------
def test_arau_and_fitzroy_non_uniform_hand_values() -> None:
    """Absorption 0.5 on the x-pair, 0.1 elsewhere (dims 8x5x3).

    Axial Eyring times are Tx=0.176535, Ty=Tz=1.161393 s. The area-weighted
    geometric mean (Arau) is 0.812147 s, the arithmetic mean (Fitzroy)
    0.974394 s -- hand-computed with c0 = 343 m/s.
    """
    absorptions = (0.5, 0.1, 0.1)
    arau = room.arau_puchades_reverberation_time(DIMS, absorptions)
    fitz = room.fitzroy_reverberation_time(DIMS, absorptions)
    assert arau == pytest.approx(0.8121469281, abs=1e-6)
    assert fitz == pytest.approx(0.9743944340, abs=1e-6)


def test_arau_never_exceeds_fitzroy() -> None:
    """AM-GM: the geometric mean (Arau) <= the arithmetic mean (Fitzroy)."""
    absorptions = (0.5, 0.1, 0.1)
    assert room.arau_puchades_reverberation_time(
        DIMS, absorptions
    ) <= room.fitzroy_reverberation_time(DIMS, absorptions)


def test_axial_means_lie_between_extremes() -> None:
    """Both axial means fall between the fastest and slowest axial time."""
    absorptions = (0.5, 0.1, 0.1)
    tx = _K * VOLUME / (-SURFACE * math.log(1.0 - 0.5))
    ty = _K * VOLUME / (-SURFACE * math.log(1.0 - 0.1))
    for model in (
        room.arau_puchades_reverberation_time,
        room.fitzroy_reverberation_time,
    ):
        t = model(DIMS, absorptions)
        assert tx <= t <= ty


# ---------------------------------------------------------------------------
# Air absorption
# ---------------------------------------------------------------------------
def test_air_absorption_shortens_reverberation() -> None:
    """The 4mV air term adds absorption, so T falls."""
    dry = room.eyring_reverberation_time(VOLUME, UNIFORM_SURFACES)
    humid = room.eyring_reverberation_time(
        VOLUME, UNIFORM_SURFACES, air_attenuation=0.003
    )
    assert humid < dry
    # Explicit denominator: -S ln0.8 + 4*m*V.
    expected = _K * VOLUME / (-SURFACE * math.log(0.8) + 4.0 * 0.003 * VOLUME)
    assert humid == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Per-band arrays and the bundled front-end
# ---------------------------------------------------------------------------
def test_per_band_arrays() -> None:
    alpha = np.array([0.1, 0.2, 0.4])
    surfaces = [(area, alpha) for area, _ in UNIFORM_SURFACES]
    t = room.eyring_reverberation_time(VOLUME, surfaces)
    assert isinstance(t, np.ndarray)
    assert t.shape == (3,)
    # Higher absorption -> shorter time, monotonically.
    assert t[0] > t[1] > t[2]


def test_reverberation_time_models_bundle() -> None:
    freqs = [125.0, 250.0, 500.0, 1000.0]
    res = room.reverberation_time_models(
        DIMS, ([0.1, 0.15, 0.2, 0.3], 0.2, 0.25), frequencies=freqs
    )
    assert isinstance(res, room.ReverberationModelResult)
    assert res.volume == pytest.approx(VOLUME)
    assert res.surface_area == pytest.approx(SURFACE)
    for curve in res.models.values():
        assert curve.shape == (4,)
    # Sabine is the longest estimate band by band.
    assert np.all(res.sabine >= res.eyring - 1e-9)
    assert set(res.models) == {
        "Sabine",
        "Eyring",
        "Millington-Sette",
        "Fitzroy",
        "Arau-Puchades",
    }


def test_model_curves_must_run_over_one_band_axis() -> None:
    """Curves off the band axis are refused when built, not when read.

    The fiche walks ``frequencies`` and reads each curve at that band's index,
    so a curve one entry short raises a bare index error from inside the row
    loop and one entry long is dropped off the end of the table; ``plot()``
    reports matplotlib's two shapes and names neither field. An extra axis is
    the silent one: it counts one entry per band, so ``plot()`` draws the
    column as an ordinary line and returns a figure that looks right.
    """
    good = room.reverberation_time_models(
        DIMS, (0.2, 0.15, 0.3), frequencies=[125.0, 250.0, 500.0, 1000.0]
    )
    per_band = "one value per band"
    curves = ("sabine", "eyring", "millington_sette", "fitzroy", "arau_puchades")
    cases: list[tuple[str, np.ndarray, str]] = [
        ("frequencies", good.frequencies[:-1], per_band),
        ("frequencies", np.append(good.frequencies, 2000.0), per_band),
        ("frequencies", np.column_stack([good.frequencies] * 2), "must have one axis"),
    ]
    for name in curves:
        curve = getattr(good, name)
        cases += [
            (name, curve[:-1], per_band),
            (name, np.append(curve, curve[-1]), per_band),
            (name, np.column_stack([curve] * 2), "must have one axis"),
        ]
    for field, value, fragment in cases:
        with pytest.raises(ValueError, match=rf"'{field}'.*{fragment}"):
            dataclasses.replace(good, **{field: value})


def test_models_single_frequency_is_a_band_axis_of_one() -> None:
    """A scalar ``frequencies`` labels one band, as ``None`` already does.

    The five curves are made 1-D whatever is passed, so a nought-dimensional
    band axis beside them would be a mixture the table cannot walk.
    """
    res = room.reverberation_time_models(DIMS, (0.2, 0.2, 0.2), frequencies=1000.0)
    assert res.frequencies.shape == (1,)
    assert res.frequencies[0] == pytest.approx(1000.0)
    assert res.sabine.shape == (1,)


def test_models_scalar_absorption_broadcasts() -> None:
    res = room.reverberation_time_models(DIMS, (0.2, 0.2, 0.2))
    assert res.frequencies.shape == (1,)
    # Uniform: the four Eyring-family models coincide; Sabine is longer.
    eyring_family = [
        res.eyring[0],
        res.millington_sette[0],
        res.fitzroy[0],
        res.arau_puchades[0],
    ]
    assert np.allclose(eyring_family, res.eyring[0])
    assert res.sabine[0] > res.eyring[0]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_empty_surfaces_raise() -> None:
    with pytest.raises(ValueError, match="at least one surface"):
        room.sabine_reverberation_time(VOLUME, [])


# The old blanket [0, 1) validator rejected alpha >= 1 for every model; that
# was over-strict for Sabine (whose A = sum S_i alpha_i is finite there) and
# for Eyring (which only needs the mean below 1). The tests below encode the
# per-model domains that replaced it.
@pytest.mark.parametrize(
    "model",
    [
        room.sabine_reverberation_time,
        room.eyring_reverberation_time,
        room.millington_sette_reverberation_time,
    ],
)
@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
def test_negative_and_non_finite_alpha_rejected_by_surface_models(
    model: _SurfaceModel, bad: float
) -> None:
    """Negative and non-finite coefficients are rejected by every model."""
    with pytest.raises(ValueError, match="absorption coefficients must be"):
        model(VOLUME, [(10.0, bad)])


@pytest.mark.parametrize(
    "model",
    [room.fitzroy_reverberation_time, room.arau_puchades_reverberation_time],
)
@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
def test_negative_and_non_finite_alpha_rejected_by_axial_models(
    model: _AxialModel, bad: float
) -> None:
    with pytest.raises(ValueError, match="absorption coefficients must be"):
        model(DIMS, (bad, 0.2, 0.2))


@pytest.mark.parametrize(
    "model",
    [
        room.sabine_reverberation_time,
        room.eyring_reverberation_time,
        room.millington_sette_reverberation_time,
    ],
)
def test_alpha_above_unit_error_bound_rejected_by_surface_models(
    model: _SurfaceModel,
) -> None:
    """A coefficient above 2 is flagged as a probable percent-vs-fraction slip.

    (The axial models reject 20.0 too, but through their stricter below-1
    wall-pair-mean check, exercised in the axial rejection test below.)
    """
    with pytest.raises(ValueError, match="unit error"):
        model(VOLUME, [(10.0, 20.0)])


def test_sabine_accepts_the_documented_upper_bound() -> None:
    """The bound is inclusive: alpha exactly 2.0 computes, 2.0 + eps raises."""
    t = room.sabine_reverberation_time(VOLUME, [(SURFACE, 2.0)])
    assert t == pytest.approx(_K * VOLUME / (SURFACE * 2.0), abs=1e-9)
    with pytest.raises(ValueError, match="unit error"):
        room.sabine_reverberation_time(VOLUME, [(SURFACE, 2.0000001)])


def test_sabine_accepts_iso354_alphas_at_and_above_one() -> None:
    """Sabine: A = S alpha stays finite for alpha in {0.2, 1.0, 1.15}.

    Hand check per band: A = 158 alpha, T = k 120 / A, i.e. 0.611825 s,
    0.122365 s and 0.106404 s. The exact 1.0 is the ISO 11654 practical-
    coefficient cap; 1.15 is a routine measured ISO 354 value (edge effect).
    """
    alpha = np.array([0.2, 1.0, 1.15])
    surfaces = [(area, alpha) for area, _ in UNIFORM_SURFACES]
    t = room.sabine_reverberation_time(VOLUME, surfaces)
    np.testing.assert_allclose(t, _K * VOLUME / (SURFACE * alpha), atol=1e-9)
    np.testing.assert_allclose(t, [0.6118246547, 0.1223649309, 0.1064042878], atol=1e-6)


def test_millington_rejects_any_alpha_at_or_above_one() -> None:
    """The per-surface ln(1 - alpha) diverges at 1, so Millington rejects it."""
    for alpha in (1.0, 1.15):
        with pytest.raises(ValueError, match="Millington-Sette requires every"):
            room.millington_sette_reverberation_time(
                100.0, [(60.0, 0.3), (40.0, alpha)]
            )


def test_eyring_accepts_alpha_above_one_when_mean_stays_below_one() -> None:
    """Eyring constrains only the mean: 10 m2 at 1.2 in a 158 m2 room is fine."""
    surfaces = [(10.0, 1.2), (148.0, 0.1)]
    mean = (10.0 * 1.2 + 148.0 * 0.1) / 158.0
    t = room.eyring_reverberation_time(VOLUME, surfaces)
    assert t == pytest.approx(_K * VOLUME / (-158.0 * math.log(1.0 - mean)), abs=1e-9)


def test_eyring_rejects_mean_at_or_above_one() -> None:
    """A mean of exactly 1 or above has no finite Eyring time."""
    for high in (1.1, 1.2):  # means 1.0 and 1.05 with the 0.9 partner below
        with pytest.raises(ValueError, match="mean absorption coefficient"):
            room.eyring_reverberation_time(VOLUME, [(79.0, high), (79.0, 0.9)])


def test_axial_models_reject_wall_pair_mean_at_or_above_one() -> None:
    """Fitzroy/Arau inputs are the means entering ln(1 - alpha_i) directly."""
    for model in (
        room.fitzroy_reverberation_time,
        room.arau_puchades_reverberation_time,
    ):
        for bad in (1.0, 20.0):
            with pytest.raises(ValueError, match="wall-pair mean absorption"):
                model(DIMS, (bad, 0.2, 0.2))


def test_models_bundle_inherits_the_strictest_domain() -> None:
    """The bundle evaluates Millington too, so alpha >= 1 raises its error."""
    with pytest.raises(ValueError, match="Millington-Sette requires every"):
        room.reverberation_time_models(DIMS, (1.0, 0.2, 0.2))


def test_mean_absorption_accepts_iso354_alphas_above_one() -> None:
    """The mean of measured coefficients above 1 is meaningful: 0.21 here."""
    assert room.mean_absorption([(10.0, 1.2), (90.0, 0.1)]) == pytest.approx(0.21)


def test_perfectly_reflecting_room_raises() -> None:
    with pytest.raises(ValueError, match="non-positive"):
        room.sabine_reverberation_time(VOLUME, [(10.0, 0.0)])


def test_non_positive_volume_raises() -> None:
    with pytest.raises(ValueError, match="volume"):
        room.sabine_reverberation_time(0.0, UNIFORM_SURFACES)


def test_negative_air_attenuation_raises() -> None:
    with pytest.raises(ValueError, match="air_attenuation"):
        room.sabine_reverberation_time(VOLUME, UNIFORM_SURFACES, air_attenuation=-1.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_air_attenuation_raises(bad: float) -> None:
    """NaN used to slip through the old m < 0 comparison and return NaN."""
    with pytest.raises(ValueError, match="air_attenuation"):
        room.sabine_reverberation_time(VOLUME, UNIFORM_SURFACES, air_attenuation=bad)


def test_bad_dimensions_raise() -> None:
    with pytest.raises(ValueError, match="three room lengths"):
        room.arau_puchades_reverberation_time((8.0, 5.0), (0.2, 0.2, 0.2))


def test_bad_absorptions_length_raises() -> None:
    with pytest.raises(ValueError, match="three wall pairs"):
        room.fitzroy_reverberation_time(DIMS, (0.2, 0.2))


def test_mismatched_surface_band_counts_raise() -> None:
    """Per-band coefficients with different band counts get a clear error."""
    surfaces = [(40.0, [0.1, 0.2, 0.3]), (40.0, [0.1, 0.2])]
    with pytest.raises(ValueError, match="incompatible shapes"):
        room.eyring_reverberation_time(VOLUME, surfaces)


def test_mismatched_axis_band_counts_raise() -> None:
    """A 6-band x-pair against a 4-band y-pair is reported, not a raw NumPy error."""
    with pytest.raises(ValueError, match="incompatible shapes"):
        room.arau_puchades_reverberation_time(
            DIMS, ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.2, 0.3, 0.4], 0.2)
        )


def test_mismatched_air_band_count_raises() -> None:
    """Air attenuation with a different band count than the surfaces is caught."""
    surfaces = [(158.0, [0.2, 0.3, 0.4])]
    with pytest.raises(ValueError, match="incompatible shapes"):
        room.sabine_reverberation_time(VOLUME, surfaces, air_attenuation=[0.001, 0.002])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def test_plot_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    res = room.reverberation_time_models(
        DIMS, (0.2, 0.15, 0.3), frequencies=[125.0, 250.0]
    )
    assert res.plot() is not None


# ---------------------------------------------------------------------------
# Sabine chain — published Spanish classroom example
# ---------------------------------------------------------------------------


def test_sabine_manual_es_classroom_chain() -> None:
    # Aviles Lopez & Perera Martin, Manual de acustica ambiental y
    # arquitectonica (Paraninfo), Ejemplos 8.1 (pp. 524-525, equivalent
    # absorption areas) and 8.2 (p. 549, Sabine reverberation times) for a
    # 6.0 x 10.2 m classroom: V = 214.2 m3 with the original plaster ceiling
    # and 189.7 m3 with the absorbent false ceiling. Octave bands
    # 125 Hz - 4 kHz. The book evaluates T = 0.16 V / A and rounds to
    # 0.01 s; the library's exact 55.26/c0 constant (0.161) differs by 0.6 %,
    # so 0.04 s covers both effects at the longest published time (4.31 s).
    cases = [
        # (volume m3, A per band m2, published T per band s)
        (
            214.2,
            (34.83, 14.36, 9.74, 7.94, 10.39, 12.67),
            (0.98, 2.39, 3.52, 4.31, 3.30, 2.71),
        ),
        (
            189.7,
            (34.74, 19.80, 25.00, 41.09, 58.44, 47.00),
            (0.87, 1.53, 1.21, 0.74, 0.52, 0.65),
        ),
        (
            189.7,
            (38.34, 27.00, 42.70, 70.49, 92.34, 80.60),
            (0.79, 1.12, 0.71, 0.43, 0.33, 0.38),
        ),
    ]
    for volume, areas, expected in cases:
        for area, t_published in zip(areas, expected, strict=True):
            t = room.sabine_reverberation_time(volume, [(area, 1.0)])
            assert t == pytest.approx(t_published, abs=0.04), (volume, area)

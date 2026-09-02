#  Copyright (c) 2026. Jose Manuel Requena Plens

"""What the building-acoustics ``.plot()`` renderers draw.

Every result in this domain is a one-third-octave curve read against something:
the shifted reference curve of ISO 717-1/-2, the flanking budget of EN 12354, the
companion spectra of ISO 16283, the reproducibility band of ISO 12999-1. That
shared shape is what these tests hold to account, and the figure is where the
sign conventions become visible. Shading falls only on *unfavourable*
deviations, and the sign flips between airborne (measured below the reference)
and impact (measured above it). The ISO 717-2 octave-band -5 dB rule of
Clause 4.3.2 is an offset on the *rating*, not on the curve, so the drawn
reference stays ``ref - shift`` and reads 5 dB above the rating at 500 Hz, with
the offset annotated instead of the curve bent down. The enlarged-range ratings
of ISO 717-1 Annex B draw the measurement over its whole range while the shifted
reference covers only the 16 core bands.

These are the content assertions. The generic plot contract (the soft matplotlib
dependency, the kwarg-forwarding table, external axes) lives in
``tests/test_result_plots.py``.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from result_factories import (
    _airborne_insulation,
    _airborne_prediction,
    _airborne_rating,
    _band_uncertainty,
    _extended_impact_rating,
    _extended_rating,
    _impact_insulation,
    _impact_prediction,
    _impact_rating,
)

import phonometry as ph
from phonometry._plot import common as _plotting


# --------------------------------------------------------------------------
# Weighted ratings (airborne / impact)
# --------------------------------------------------------------------------
def test_airborne_rating_carries_curve_fields() -> None:
    res = _airborne_rating()
    assert res.band_centers is not None
    assert res.band_centers.size == 16
    assert res.measured is not None
    assert res.shifted_reference is not None
    # shifted reference read at 500 Hz (index 7) equals the rating.
    assert round(res.shifted_reference[7]) == res.rating


def test_airborne_rating_shades_only_unfavourable_bands() -> None:
    res = _airborne_rating()
    ax = res.plot()
    # measured curve and shifted reference are drawn as lines.
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.measured)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), res.shifted_reference)
    # a shaded region (fill_between -> collection) is present.
    assert len(ax.collections) >= 1
    mask = _plotting._unfavourable_mask(
        res.measured, res.shifted_reference, impact=False
    )
    # airborne: unfavourable where measured < reference.
    np.testing.assert_array_equal(mask, res.measured < res.shifted_reference)
    assert mask.any()
    plt.close("all")


def test_impact_rating_uses_opposite_sign_mask() -> None:
    res = _impact_rating()
    ax = res.plot()
    mask = _plotting._unfavourable_mask(
        res.measured, res.shifted_reference, impact=True
    )
    np.testing.assert_array_equal(mask, res.measured > res.shifted_reference)
    assert str(res.rating) in ax.get_title()
    plt.close("all")


def test_rating_without_curve_data_raises() -> None:
    bare = ph.building.WeightedRatingResult(
        rating=52, c=-1, ctr=-3, unfavourable_sum=10.0
    )
    with pytest.raises(ValueError, match="no band curve"):
        bare.plot()


def test_rating_plot_forwards_kwargs_without_typeerror() -> None:
    # Regression: _plot_rating used to lack **kwargs, so any styling kwarg
    # forwarded by plot_weighted_rating/plot_impact_rating raised TypeError.
    res = _airborne_rating()
    ax = res.plot(linewidth=2)
    assert ax.lines[0].get_linewidth() == 2.0
    ax2 = _impact_rating().plot(linewidth=2)
    assert ax2.lines[0].get_linewidth() == 2.0
    plt.close("all")


# --------------------------------------------------------------------------
# ISO 717 enlarged-range ratings (Annex B / A.2.1)
# --------------------------------------------------------------------------
def test_extended_rating_plot_full_range_and_terms_in_title() -> None:
    res = _extended_rating()
    ax = res.plot()
    # The measured curve spans the full enlarged range (21 bands), the
    # shifted reference only the 16 core bands.
    assert ax.lines[0].get_xdata().size == 21
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.measured)
    assert ax.lines[1].get_xdata().size == 16
    np.testing.assert_allclose(ax.lines[1].get_ydata(), res.core.shifted_reference)
    # Title carries the core rating and the covered Annex B terms.
    title = ax.get_title()
    assert str(res.rating) in title
    # The terms are set as mathematics, and their range is joined by U+2010:
    # an ASCII hyphen inside $...$ is the binary minus, which mathtext would
    # space out as a subtraction.
    assert "$C_{50‐5000}$" in title
    assert "$C_{\\mathrm{tr},50‐5000}$" in title
    plt.close("all")


def test_extended_impact_rating_plot_title_carries_ci_50_2500() -> None:
    res = _extended_impact_rating()
    assert res.ci_50_2500 is not None
    ax = res.plot()
    title = ax.get_title()
    assert str(res.rating) in title
    assert "$C_{\\mathrm{I},50‐2500}$" in title  # U+2010, not a minus
    # Impact shading uses the opposite sign: measurement above reference.
    assert len(ax.collections) >= 1
    plt.close("all")


def test_extended_rating_plot_forwards_kwargs() -> None:
    ax = _extended_rating().plot(linewidth=2)
    assert ax.lines[0].get_linewidth() == 2.0
    ax2 = _extended_impact_rating().plot(linewidth=2)
    assert ax2.lines[0].get_linewidth() == 2.0
    plt.close("all")


def test_extended_rating_spanish_labels() -> None:
    ax = _extended_rating().plot(language="es")
    labels = [str(ln.get_label()) for ln in ax.lines]
    assert any("Medido" in lbl for lbl in labels)
    assert "reducción acústica" in ax.get_ylabel()
    plt.close("all")


# --------------------------------------------------------------------------
# ISO 717-2 octave-band -5 dB rule: the curve is honest, the rating annotated
# --------------------------------------------------------------------------
_ANNEX_C3_LN_OCTAVE = np.array([65.3, 64.5, 58.0, 55.8, 43.0])


def test_octave_impact_plot_keeps_curve_honest_and_annotates_offset() -> None:
    # ISO 717-2 Annex C, Table C.3 (octave): Ln,w = 54, applying the -5 dB
    # rule of Clause 4.3.2. The drawn shifted-reference curve is genuinely
    # ref - shift (so it reads 59 at 500 Hz); the rating (54) is annotated
    # with the -5 dB note rather than the curve being distorted down.
    res = ph.building.weighted_impact_rating(_ANNEX_C3_LN_OCTAVE)
    assert res.rating == 54
    idx500 = int(np.argmin(np.abs(res.band_centers - 500.0)))
    read_value = float(res.shifted_reference[idx500])
    assert read_value == pytest.approx(res.rating + 5)  # honest curve
    ax = res.plot()
    # reference line (index 1) is undistorted: reads read_value at 500 Hz.
    assert ax.lines[1].get_ydata()[idx500] == pytest.approx(read_value)
    # a marker records the 500 Hz read value.
    marked = [
        ln
        for ln in ax.lines
        if ln.get_ydata().size == 1 and ln.get_ydata()[0] == pytest.approx(read_value)
    ]
    assert marked, "expected a 500 Hz read-value marker"
    # annotation carries both the rating and the -5 dB octave note.
    texts = " ".join(t.get_text() for t in ax.texts).replace("−", "-")
    assert str(res.rating) in texts
    assert "-5" in texts.replace(" ", "")
    plt.close("all")


def test_third_octave_impact_plot_reads_rating_at_500() -> None:
    # For one-third-octave impact there is no -5 dB offset: the curve read
    # value at 500 Hz equals the rating, so no offset note is drawn.
    res = _impact_rating()
    idx500 = int(np.argmin(np.abs(res.band_centers - 500.0)))
    assert round(float(res.shifted_reference[idx500])) == res.rating
    ax = res.plot()
    texts = " ".join(t.get_text() for t in ax.texts).replace("−", "-")
    assert "-5" not in texts.replace(" ", "")
    plt.close("all")


def test_facade_plot_accepts_label_kwarg() -> None:
    # Regression: kwargs used to be forwarded to all four curves, so a user
    # label= collided with the per-curve labels and raised TypeError.
    res = ph.building.facade_insulation(
        [70.0, 72.0, 74.0], [40.0, 41.0, 42.0], [0.5, 0.5, 0.5]
    )
    ax = res.plot(label="my measurement")
    labels = [ln.get_label() for ln in ax.lines]
    assert "my measurement" in labels
    # The companion curves keep their own labels.
    assert any(label.startswith("$D_{2m}$") for label in labels)
    plt.close("all")


# --------------------------------------------------------------------------
# EN 12354 predictions
# --------------------------------------------------------------------------
def test_airborne_prediction_plot_sorted_shares() -> None:
    res = _airborne_prediction()
    ax = res.plot()
    heights = [p.get_height() for p in ax.patches]
    assert heights == sorted(heights, reverse=True)
    assert sum(heights) == pytest.approx(100.0)
    assert f"{res.r_prime_w:.1f}" in ax.get_title()
    plt.close("all")


def test_impact_prediction_plot_terms() -> None:
    res = _impact_prediction()
    ax = res.plot()
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(
        heights,
        [res.ln_w_eq, -res.delta_l_w, res.k_correction, res.l_prime_n_w],
    )
    plt.close("all")


# --------------------------------------------------------------------------
# ISO 16283 field insulation spectra
# --------------------------------------------------------------------------
def test_airborne_insulation_plot_curves() -> None:
    res = _airborne_insulation()
    ax = res.plot()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.dnt)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), res.d)
    assert res.r_prime is not None
    np.testing.assert_allclose(ax.lines[2].get_ydata(), res.r_prime)
    plt.close("all")


def test_impact_insulation_plot_curves_and_label_kwarg() -> None:
    res = _impact_insulation()
    ax = res.plot(label="my measurement")
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.l_n_t)
    labels = [str(ln.get_label()) for ln in ax.lines]
    # user label styles only the primary curve; companions keep theirs.
    assert "my measurement" in labels
    assert any(r"L^{\prime}_\mathrm{n}" in lbl for lbl in labels)
    plt.close("all")


# --------------------------------------------------------------------------
# ISO 12999-1 band uncertainty
# --------------------------------------------------------------------------
def test_band_uncertainty_plot_spectrum() -> None:
    res = _band_uncertainty()
    ax = res.plot()
    freqs, u = res.to_arrays()
    np.testing.assert_allclose(ax.lines[0].get_xdata(), freqs)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), u)
    assert "12999" in ax.get_title()
    plt.close("all")


# --------------------------------------------------------------------------
# ISO 15186-3 low-frequency intensity insulation
# --------------------------------------------------------------------------
def _low_frequency_intensity(*, absorbing: bool = False) -> object:
    """Six bands whose indicator crosses both Clause 6.4.2 limits."""
    return ph.building.low_frequency_intensity_reduction(
        [88.4, 89.1, 90.3, 91.0, 91.4, 91.8],
        [70.0, 73.0, 76.0, 80.0, 82.0, 84.0],
        measurement_area=12.0,
        area=10.0,
        l_p=[81.5, 82.0, 83.6, 87.0, 88.5, 90.0],
        frequencies=[50.0, 63.0, 80.0, 100.0, 125.0, 160.0],
        absorbing_specimen_surface=absorbing,
    )


def test_low_frequency_intensity_plot_draws_index_and_indicator() -> None:
    res = _low_frequency_intensity()
    ax = res.plot()
    # One bar per band, carrying the index itself.
    heights = [patch.get_height() for patch in ax.patches]
    np.testing.assert_allclose(heights[: res.r_i.size], res.r_i)
    # The indicator lives on the twin axis, with its limit beside it.
    (twin,) = [other for other in ax.figure.axes if other is not ax]
    np.testing.assert_allclose(
        twin.lines[0].get_ydata(), res.surface_pressure_intensity
    )
    assert res.indicator_limit == 10.0
    assert twin.lines[1].get_ydata()[0] == res.indicator_limit
    assert "15186-3" in ax.get_title()
    plt.close("all")


def test_low_frequency_intensity_plot_hatches_only_refused_bands() -> None:
    res = _low_frequency_intensity()
    assert res.qualified is not None
    assert res.qualified.any()
    assert not res.qualified.all()
    ax = res.plot()
    hatched = [p for p in ax.patches if p.get_hatch()]
    assert len(hatched) == int((~res.qualified).sum())
    # Each hatch sits on a band the clause refuses, at that band's index.
    refused = res.r_i[~res.qualified]
    np.testing.assert_allclose([p.get_height() for p in hatched], refused)
    plt.close("all")


def test_low_frequency_intensity_plot_draws_the_tighter_limit() -> None:
    res = _low_frequency_intensity(absorbing=True)
    ax = res.plot()
    (twin,) = [other for other in ax.figure.axes if other is not ax]
    assert twin.lines[1].get_ydata()[0] == 6.0
    plt.close("all")


def test_low_frequency_intensity_plot_speaks_spanish() -> None:
    ax = _low_frequency_intensity().plot(language="es")
    assert "Aislamiento" in ax.get_title()
    assert ax.get_xlabel().startswith("Frecuencia")
    labels = [str(t.get_text()) for t in ax.get_legend().get_texts()]
    assert any("no cualificada" in lbl for lbl in labels)
    assert any("límite" in lbl for lbl in labels)
    plt.close("all")


def test_low_frequency_intensity_plot_labels_bands_without_centres() -> None:
    res = ph.building.low_frequency_intensity_reduction(
        [84.0, 86.0], [70.0, 73.0], measurement_area=12.0, area=10.0
    )
    assert res.frequencies is None
    assert res.surface_pressure_intensity is None
    ax = res.plot()
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Band 1", "Band 2"]
    # No indicator measured, so no twin axis and no limit line to draw.
    assert ax.figure.axes == [ax]
    plt.close("all")


def test_low_frequency_element_plot_draws_its_own_quantity() -> None:
    res = ph.building.low_frequency_element_normalized_difference(
        np.full(6, 90.0),
        np.array([62.0, 61.0, 60.0, 58.0, 57.0, 55.0]),
        measurement_area=2.0,
        elements=4,
        l_p=np.array([70.0, 72.0, 74.0, 70.0, 68.0, 66.0]),
        frequencies=[50.0, 63.0, 80.0, 100.0, 125.0, 160.0],
    )
    ax = res.plot()
    heights = [patch.get_height() for patch in ax.patches]
    np.testing.assert_allclose(heights[: res.d_i_n_e.size], res.d_i_n_e)
    assert "15186-3" in ax.get_title()
    assert "element" in ax.get_title()
    plt.close("all")

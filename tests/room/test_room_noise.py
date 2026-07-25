#  Copyright (c) 2026. Jose M. Requena-Plens
"""Tests for :mod:`phonometry.room.room_noise` (ANSI/ASA S12.2-2019 NC and RC Mark II).

The methods are validated against the standard's own tabulated curves: feeding
an NC curve of Table 1 back through the rating returns its NC value (the
tangency rating is exact and the SIL two-step designation of clause 5.2.2
lands on the curve's own speech interference level), and the generated RC
Mark II curves reproduce Table D.1 digit for digit. The spectral tags
(neutral / rumble / hiss) are checked against the deviation rules of clause
D.3, and spectra outside the NC-15 to NC-70 family must be flagged out of
range rather than clamped to a fabricated rating.
"""

from __future__ import annotations

import numpy as np
import pytest
from reference_data import (
    ANSIS12_2_NC40_SELF,
    ANSIS12_2_RC31_63HZ,
    ANSIS12_2_RC35_LMF,
)

from phonometry.room import room_noise as rn


def test_octave_band_layout() -> None:
    assert rn.OCTAVE_BANDS.size == 10
    assert rn.OCTAVE_BANDS[0] == 16.0
    assert rn.OCTAVE_BANDS[-1] == 8000.0
    assert rn.NC_CURVES.shape == (rn.NC_INDICES.size, 10)


def test_nc_table_row_matches_standard() -> None:
    # Table 1: the NC-30 curve, 16 Hz - 8000 Hz.
    np.testing.assert_allclose(
        rn.nc_curve(30.0), [81, 68, 57, 48, 41, 35, 32, 29, 28, 27]
    )


@pytest.mark.parametrize("index", [15.0, 25.0, ANSIS12_2_NC40_SELF, 50.0, 70.0])
def test_nc_curve_returns_its_own_tangency_rating(index: float) -> None:
    # Feeding an NC curve back through the tangency method returns its value.
    result = rn.noise_criterion(rn.nc_curve(index))
    assert result.tangency_rating == pytest.approx(index, abs=1e-9)


def test_nc_curve_sil_designation_matches_its_own_sil() -> None:
    # Clause 5.2.2: a spectrum lying on a Table 1 curve exceeds no band of
    # the NC-(SIL) curve chosen from its SIL, so the designation is NC-(SIL).
    # The NC-40 contour has SIL = (44 + 41 + 39 + 38)/4 = 40.5 dB, which
    # rounds to the curve's own designating number.
    result = rn.noise_criterion(rn.nc_curve(40.0))
    assert result.sil == pytest.approx(40.5)
    assert result.method == "SIL"
    assert result.rating == pytest.approx(40.0)
    assert result.label == "NC-40"
    # A SIL-designated spectrum has no governing band.
    assert np.isnan(result.governing_frequency)
    assert result.out_of_range is None


def test_nc_governing_band_and_monotonicity() -> None:
    # Raising one band above the NC-50 curve lifts the rating and makes that
    # band the governing one (the SIL curve is exceeded there, so the
    # tangency method sets the designation, clause 5.2.3).
    levels = rn.nc_curve(50.0).copy()
    levels[3] += 3.0  # 125 Hz band.
    result = rn.noise_criterion(levels)
    assert result.method == "tangency"
    assert result.rating > 50.0
    assert result.rating == result.tangency_rating
    assert result.governing_frequency == 125.0
    assert result.label == f"NC-{result.rating:g} (125 Hz)"


def test_nc_sil_average_matches_clause_3_2() -> None:
    # SIL = (1/4)(L500 + L1000 + L2000 + L4000), clause 3.2.
    levels = rn.nc_curve(30.0)
    result = rn.noise_criterion(levels)
    expected = float(np.mean(levels[5:9]))
    assert result.sil == pytest.approx(expected)


def test_nc_flat_110db_spectrum_is_above_the_family() -> None:
    # A 110 dB flat spectrum exceeds NC-70 in every band: the standard
    # defines no rating above NC-70, so no number may be fabricated. The
    # governing band is the maximum exceedance over the NC-70 curve
    # (110 - 68 = 42 dB at 4000 Hz, the first of the two tied top bands).
    result = rn.noise_criterion(np.full(10, 110.0))
    assert result.out_of_range == "above"
    assert np.isnan(result.rating)
    assert np.isnan(result.tangency_rating)
    assert result.governing_frequency == 4000.0
    assert result.label == ">NC-70 (4000 Hz)"


def test_nc_sub_nc15_spectrum_is_below_the_family() -> None:
    # A spectrum below the NC-15 curve everywhere touches no curve: the
    # rating is flagged below the family, never clamped to a number.
    result = rn.noise_criterion(np.full(10, 5.0))
    assert result.out_of_range == "below"
    assert np.isnan(result.rating)
    assert np.isnan(result.governing_frequency)
    assert result.label == "<NC-15"


def test_nc_marginal_cases_on_the_family_edges() -> None:
    # Exactly on the NC-70 curve: still inside the family (rating 70).
    on_top = rn.noise_criterion(rn.NC_CURVES[-1].copy())
    assert on_top.out_of_range is None
    assert on_top.rating == pytest.approx(70.0)
    # 1 dB above the NC-70 curve in one band: above the family, and that
    # band governs even though other bands would interpolate higher values.
    over = rn.NC_CURVES[-1].copy()
    over[2] += 1.0  # 63 Hz
    result = rn.noise_criterion(over)
    assert result.out_of_range == "above"
    assert result.governing_frequency == 63.0
    # Exactly on the NC-15 curve: inside the family (tangency 15).
    on_bottom = rn.noise_criterion(rn.NC_CURVES[0].copy())
    assert on_bottom.out_of_range is None
    assert on_bottom.tangency_rating == pytest.approx(15.0)


def test_nc_out_of_range_curve_raises() -> None:
    with pytest.raises(ValueError, match="tabulated range"):
        rn.nc_curve(80.0)


def test_rc_curves_match_table_d1() -> None:
    # Table D.1, generated from the -5 dB/octave rule (Annex D).
    expected = {
        25.0: [55, 55, 45, 40, 35, 30, 25, 20, 15, 10],
        30.0: [55, 55, 50, 45, 40, 35, 30, 25, 20, 15],
        31.0: [56, 56, 51, 46, 41, 36, 31, 26, 21, 16],
        38.0: [63, 63, 58, 53, 48, 43, 38, 33, 28, 23],
        50.0: [75, 75, 70, 65, 60, 55, 50, 45, 40, 35],
    }
    for index, row in expected.items():
        np.testing.assert_allclose(rn.rc_curve(index), row)
    # Pin the inline Table D.1 transcription to the shared reference_data
    # constant used by the CI conformance report (RC-31 curve at 63 Hz).
    assert expected[31.0][2] == ANSIS12_2_RC31_63HZ


def test_rc_low_frequency_floor() -> None:
    # The 31.5 Hz level never drops below 55 dB and 16 Hz equals 31.5 Hz.
    curve = rn.rc_curve(25.0)
    assert curve[1] == 55.0
    assert curve[0] == curve[1]


def test_rc_neutral_spectrum() -> None:
    result = rn.room_criterion(rn.rc_curve(35.0))
    assert result.rating == 35
    assert result.lmf == pytest.approx(ANSIS12_2_RC35_LMF)
    assert result.classification == "N"
    assert result.label == "RC-35(N)"


def test_rc_rumble_tag() -> None:
    # A low band exceeding the RC curve by more than 5 dB -> rumble.
    levels = rn.rc_curve(35.0).copy()
    levels[4] += 8.0  # 250 Hz.
    assert rn.room_criterion(levels).classification == "R"


def test_rc_hiss_tag() -> None:
    # A high band exceeding the RC curve by more than 3 dB -> hiss.
    levels = rn.rc_curve(35.0).copy()
    levels[8] += 5.0  # 4000 Hz.
    assert rn.room_criterion(levels).classification == "H"


def test_rc_combined_rumble_and_hiss() -> None:
    levels = rn.rc_curve(35.0).copy()
    levels[4] += 8.0
    levels[8] += 5.0
    assert rn.room_criterion(levels).classification == "RH"


def test_rc_missing_d4_bands_warn() -> None:
    # Clause D.4 rates a spectrum with at least the 31.5 Hz to 4000 Hz
    # octave bands; a subset missing any of them warns that the absent bands
    # are skipped by the spectral-tag deviation tests.
    with pytest.warns(UserWarning, match="31.5 Hz to 4000 Hz"):
        rn.room_criterion([40.0, 35.0, 30.0], [500.0, 1000.0, 2000.0])


def test_rc_complete_spectrum_does_not_warn() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rn.room_criterion(rn.rc_curve(35.0))


def test_rc_out_of_family_annotation() -> None:
    # Table D.1 tabulates RC-25 through RC-50; ratings outside that family
    # are flagged (the reference curve is an extrapolation of the Annex D
    # rule), while in-family ratings are not.
    assert rn.room_criterion(rn.rc_curve(35.0)).out_of_family is False
    assert rn.room_criterion(rn.rc_curve(55.0)).out_of_family is True
    assert rn.room_criterion(rn.rc_curve(20.0)).out_of_family is True


def test_rc_within_tolerance_stays_neutral() -> None:
    # Deviations of exactly the 5 dB / 3 dB tolerances are not exceedances.
    levels = rn.rc_curve(35.0).copy()
    levels[4] += 5.0   # low band, +5 dB (not > 5).
    levels[8] += 3.0   # high band, +3 dB (not > 3).
    assert rn.room_criterion(levels).classification == "N"


def test_subset_by_frequency() -> None:
    # A subset of the octave bands may be supplied with explicit frequencies.
    freqs = [500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    levels = [40.0, 35.0, 30.0, 27.0, 22.0]
    result = rn.noise_criterion(levels, freqs)
    assert result.rating == pytest.approx(35.0, abs=1e-9)


def test_rc_requires_mid_frequency_bands() -> None:
    # Without 500/1000/2000 Hz the RC rating cannot be computed.
    with pytest.raises(ValueError, match="mid-frequency"):
        rn.room_criterion([50.0, 45.0], [63.0, 125.0])


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="octave-band values"):
        rn.noise_criterion([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="same shape"):
        rn.noise_criterion([1.0, 2.0], [63.0])
    with pytest.raises(ValueError, match="not one of"):
        rn.noise_criterion([50.0], [777.0])
    with pytest.raises(ValueError, match="1-D vector"):
        rn.noise_criterion(np.zeros((2, 10)))
    with pytest.raises(ValueError, match="no valid"):
        rn.noise_criterion([], [])


def test_result_fields_and_copy() -> None:
    result = rn.room_criterion(rn.rc_curve(40.0))
    assert result.frequencies.shape == (10,)
    assert result.levels.shape == (10,)
    assert result.reference_curve.shape == (10,)
    # The returned frequencies must not alias the module constant.
    result.frequencies[0] = 0.0
    assert rn.OCTAVE_BANDS[0] == 16.0


def test_nc_plot_returns_axes() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = rn.noise_criterion(rn.nc_curve(40.0)).plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_rc_plot_returns_axes() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = rn.room_criterion(rn.rc_curve(35.0)).plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")

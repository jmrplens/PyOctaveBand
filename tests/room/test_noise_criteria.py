#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for :mod:`phonometry.room.noise_criteria` (ANSI/ASA S12.2-2019 NC and RC Mark II).

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

import dataclasses

import numpy as np
import pytest
from reference_data import (
    ANSIS12_2_NC40_SELF,
    ANSIS12_2_RC31_63HZ,
    ANSIS12_2_RC35_LMF,
)

from phonometry.room import noise_criteria as rn


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
    levels[4] += 5.0  # low band, +5 dB (not > 5).
    levels[8] += 3.0  # high band, +3 dB (not > 3).
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
    with pytest.raises(
        ValueError, match=r"noise_criterion: 'levels'.*must all have the same shape"
    ):
        rn.noise_criterion([1.0, 2.0], [63.0])
    with pytest.raises(ValueError, match="not one of"):
        rn.noise_criterion([50.0], [777.0])
    two_dimensional = np.zeros((2, 10))
    with pytest.raises(ValueError, match="1-D vector"):
        rn.noise_criterion(two_dimensional)
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
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    ax = rn.noise_criterion(rn.nc_curve(40.0)).plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_rc_plot_returns_axes() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    ax = rn.room_criterion(rn.rc_curve(35.0)).plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")


# ---------------------------------------------------------------------------
# Published NC-rating examples
# ---------------------------------------------------------------------------


def test_nc_rating_manual_es_measured_spectrum() -> None:
    # Aviles Lopez & Perera Martin, Manual de acustica ambiental y
    # arquitectonica (Paraninfo), Ejemplo 8.3 (p. 564): measured octave
    # spectrum 46/44/38/31/27/22/24/21 dB at 63 Hz - 8 kHz. With the classic
    # 5-step NC family the book rates it NC-30 by tangency and refines to
    # NC-27 by sliding the curve down 3 dB. The interpolated ANSI S12.2
    # tangency implemented here lands on the same refined value; 0.5 dB
    # covers the book's whole-dB curve stepping against the interpolation.
    res = rn.noise_criterion(
        [46.0, 44.0, 38.0, 31.0, 27.0, 22.0, 24.0, 21.0],
        [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
    )
    assert res.rating == pytest.approx(27.0, abs=0.5)
    assert res.rating <= 30.0  # the book's unrefined tangency designation


def test_nc_verdict_long_hvac_receiver_spectrum() -> None:
    # Long, Architectural Acoustics 2e (2014), Table 14.9 (pp. 555-558): the
    # combined supply-plus-return receiver spectrum of the worked HVAC duct
    # path, 55/45/32/26/23/25/22/12 dB at 63 Hz - 8 kHz, "meets NC 30". The
    # rating must therefore not exceed 30 (and the spectrum must lie inside
    # the NC family).
    res = rn.noise_criterion(
        [55.0, 45.0, 32.0, 26.0, 23.0, 25.0, 22.0, 12.0],
        [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
    )
    assert res.out_of_range is None
    assert res.rating <= 30.0


# --------------------------------------------------------------------------
# A spectrum that does not run over its own band axis
# --------------------------------------------------------------------------
@pytest.mark.parametrize("trim", [True, False], ids=["short", "long"])
def test_an_nc_rating_refuses_levels_off_the_band_axis(trim: bool) -> None:
    """The rating names a governing band, so the two axes must agree.

    A spectrum of the wrong length would have the rating name a band the
    levels never had.
    """
    bands = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    result = rn.noise_criterion([50.0, 45.0, 40.0, 38.0, 35.0, 32.0, 30.0, 28.0], bands)
    levels = np.asarray(result.levels)
    wrong = levels[:-1] if trim else np.append(levels, levels[-1])
    with pytest.raises(ValueError, match="'levels'"):
        dataclasses.replace(result, levels=wrong)


# --------------------------------------------------------------------------
# The band axis the standard fixes, the spectral tag and the rating/range pair
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["nc", "rc"])
def test_a_rating_refuses_a_band_axis_off_table_1(family: str) -> None:
    """Both families are defined over the ten tabulated octave bands.

    ANSI/ASA S12.2-2019 rates against Table 1 (NC) and Table D.1 (RC), both
    tabulated on the fixed 16 Hz - 8000 Hz octave axis, and the fiche prints
    those nominal labels rather than the stored axis. An axis shifted an
    octave keeps the band count, so nothing downstream notices: every
    measured level would be tabled against the label of its neighbour while
    the plot on the same page draws it at its true frequency.
    """
    result: rn.NCResult | rn.RCResult = (
        rn.noise_criterion(rn.nc_curve(40.0))
        if family == "nc"
        else rn.room_criterion(rn.rc_curve(35.0))
    )
    one_octave_up = rn.OCTAVE_BANDS * 2.0
    with pytest.raises(ValueError, match="'frequencies' must be the ten"):
        dataclasses.replace(result, frequencies=one_octave_up)


def test_an_rc_rating_refuses_the_unimplemented_rv_tag() -> None:
    """``RV`` is a clause D.3.5 designation this library does not rate.

    The vibration/rattle tag of clause D.3.4 needs the Table 6 criterion
    test, so it is refused by name instead of being mapped onto a
    neighbouring letter: the fiche prints one spectral-quality sentence per
    tag and would have boxed ``RC-nn(RV)`` over "Spectral quality: neutral".
    """
    result = rn.room_criterion(rn.rc_curve(35.0))
    with pytest.raises(ValueError, match="classification 'RV'"):
        dataclasses.replace(result, classification="RV")


def test_an_rc_rating_refuses_an_unknown_spectral_tag() -> None:
    """A tag outside N/R/H/RH has no spectral-quality sentence to print."""
    result = rn.room_criterion(rn.rc_curve(35.0))
    with pytest.raises(ValueError, match="'classification' must be one of"):
        dataclasses.replace(result, classification="Q")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("rating", "'rating' must be finite"),
        ("lmf", "'lmf' must be finite"),
        ("reference_curve", "'reference_curve' must contain only finite values"),
    ],
)
def test_an_rc_rating_refuses_a_non_finite_rating_average_or_curve(
    field: str, message: str, bad: float
) -> None:
    """Clause D.4 leaves none of the three undetermined, so none may be stored.

    The rater refuses a spectrum whose 500/1000/2000 Hz bands are not all
    present, so ``LMF`` and the rating rounded from it always exist, and the
    reference curve is generated from that rating by the closed-form
    -5 dB/octave rule of Annex D, which is finite in all ten bands. A
    non-finite value therefore describes no measurement, yet it clears every
    shape guard: the fiche boxes the literal ``RC-nan(N)`` as an accredited
    designation, and ``out_of_family`` stops being a verdict: both sides of
    its chained comparison are false for a ``NaN``, so it reads ``True`` for
    every spectrum rather than answering for the rating.
    """
    result = rn.room_criterion(rn.rc_curve(35.0))
    curve = np.asarray(result.reference_curve, dtype=float).copy()
    curve[3] = bad
    replacement = curve if field == "reference_curve" else bad
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(result, **{field: replacement})


@pytest.mark.parametrize(
    "bad",
    [25.5, 30.0, True],
    ids=["fractional", "integral-float", "bool"],
)
def test_an_rc_rating_refuses_a_rating_that_is_not_an_integer(bad: object) -> None:
    """Finite is too weak: the rating is a designation, not a level.

    Clause D.4 rounds the mid-frequency average to the nearest decibel and
    clause D.3.5 names the curve by that whole number, which ``label`` prints
    verbatim. Every value here is finite, so the finite guard passes it, and
    each one boxes a designation the standard does not define: ``RC-25.5(N)``
    for the fractional average, ``RC-30.0(N)`` for a float that is a whole
    number but does not print as one, and ``RC-True(N)`` for a ``bool``,
    which is an ``int`` in Python and additionally reads out of the
    tabulated family.
    """
    result = rn.room_criterion(rn.rc_curve(35.0))
    with pytest.raises(ValueError, match="'rating' must be an integer"):
        dataclasses.replace(result, rating=bad)


def test_an_rc_rating_admits_a_numpy_integer_and_a_fractional_average() -> None:
    """The integer pin belongs to the designation alone, and only to it.

    ``rating`` is what the ``RC-NN(A)`` label prints, so the pin refuses
    every value that does not print as a whole number of decibels, and
    nothing further: a NumPy integer prints as the same designation. ``lmf``
    is the level clause D.4 rounds *to* that designation, so it keeps its
    tenths and must stay pinned no more tightly than finite.
    """
    result = rn.room_criterion(rn.rc_curve(35.0))
    rated = dataclasses.replace(result, rating=np.int64(35), lmf=35.4)
    assert rated.label == "RC-35(N)"
    assert rated.lmf == pytest.approx(35.4)


def test_an_rc_rating_keeps_the_bands_it_was_never_given() -> None:
    """The finite guard must not reach ``levels``: absent bands are ``NaN``.

    Clause D.4 needs only the mid-frequency bands to rate, so a spectrum may
    be given as a subset of the ten and the rater marks every band it was not
    handed with a ``NaN`` that the fiche renders as an em dash. That is a
    quantity the measurement legitimately left undetermined, unlike the
    rating and the curve derived from it, which are finite here even though
    seven of the ten bands were never measured.
    """
    with pytest.warns(UserWarning, match="31.5 Hz to 4000 Hz octave bands"):
        result = rn.room_criterion([35.0, 32.0, 30.0], [500.0, 1000.0, 2000.0])
    absent = np.isnan(np.asarray(result.levels))
    assert absent.sum() == 7
    assert np.isfinite(np.asarray(result.reference_curve)).all()
    assert np.isfinite([result.rating, result.lmf]).all()


def test_an_nc_rating_outside_the_family_keeps_its_flag() -> None:
    """A NaN rating without ``out_of_range`` boxes ``NC-nan`` on the fiche.

    The rating and the flag are one statement made twice: outside the NC-15
    to NC-70 family of Table 1 there is no NC rating, and the fiche reads the
    flag to print ``>NC-70`` / ``<NC-15`` instead of the number. Clearing the
    flag alone leaves the NaN to be formatted verbatim and to crash the
    verdict's rounding.
    """
    over = rn.NC_CURVES[-1].copy()
    over[2] += 1.0  # 63 Hz, 1 dB above the highest tabulated curve
    result = rn.noise_criterion(over)
    assert result.out_of_range == "above"
    with pytest.raises(ValueError, match="'rating' must be finite"):
        dataclasses.replace(result, out_of_range=None)


def test_an_nc_rating_inside_the_family_cannot_be_flagged_out_of_it() -> None:
    """A flagged result carries no number: the fiche prints the flag instead."""
    result = rn.noise_criterion(rn.nc_curve(40.0))
    assert result.out_of_range is None
    with pytest.raises(ValueError, match="'rating' must be NaN"):
        dataclasses.replace(result, out_of_range="above")

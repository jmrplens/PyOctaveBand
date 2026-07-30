#  Copyright (c) 2026. Jose M. Requena-Plens
"""Tests for the suspended-ceiling plenum flanking path.

Oracles, with the source reference in each test docstring:

* **ASTM E413-22 Table 1** for the reference contour and clauses 5.2 to 5.5 for
  the fitting rules.
* Three **accredited ASTM E1414 laboratory reports** that print the per-band
  normalized ceiling attenuation ``Dn,c`` together with the resulting ceiling
  attenuation class, and (in two of them) the per-band deficiency column and
  its total. Each report is identified in the test docstring that uses it.
* **ISO 140-9:1985 clause 3.3** for the ``Dn,c`` normalization and its
  ``A0 = 10 m2``; ASTM E1414 uses ``A0 = 12 m2``.
* **Vigran (2008) Section 9.2.3** Eqs. (9.13), (9.18), (9.19) and (9.20) for
  the one-dimensional plenum model, whose only anchors are closed-form: the
  printed geometry of Figs. 9.11 to 9.13 and the convergence of Eq. (9.18) to
  Eq. (9.20) as the plenum attenuation vanishes. The model has no published
  numeric output, so no per-band regression is possible.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import (
    CEILING_ATTENUATION_CONTOUR,
    ceiling_attenuation_class,
    normalized_ceiling_attenuation,
    partition_referenced_reduction_index,
    plenum_flanking_reduction_index,
)

# ---------------------------------------------------------------------------
# Printed oracles
# ---------------------------------------------------------------------------

#: ASTM E413-22 Table 1, the reference sound insulation contour (rating zero),
#: over the 16 one-third-octave bands 125 Hz to 4000 Hz.
E413_BANDS = (
    125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0,
    800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0,
)
E413_CONTOUR = (-16, -13, -10, -7, -4, -1, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4)

#: Acoustic Laboratories Australia, report ALA 16-091-5 (16 April 2016),
#: "Airborne sound attenuation between rooms sharing a common ceiling plenum",
#: tested to ASTM E1414/E1414M-11a: a 600 x 600 x 28 mm perforated plaster
#: acoustic tile with 25 mm glasswool, in a Rondo Duo T-bar grid. Table 1 of the
#: report prints Dn,c per one-third octave from 100 Hz to 4 kHz; the data sheet
#: prints the CAC contour, the per-band deficiencies and their total.
ALA_2016_DNC = (
    14.4, 18.6, 21.7, 24.1, 23.4, 30.3, 33.7, 35.2,
    41.6, 44.2, 42.1, 36.8, 35.7, 36.0, 36.9, 37.9,
)
ALA_2016_CAC = 34
ALA_2016_PRINTED_DEFICIENCIES = (
    3.6, 2.4, 2.3, 2.9, 6.6, 2.7, 0.3, 0.0,
    0.0, 0.0, 0.0, 1.2, 2.3, 2.0, 1.1, 0.1,
)
ALA_2016_PRINTED_DEFICIENCY_SUM = 27.5

#: Acoustic Laboratories Australia, report ALA 20-093-2 (8 September 2020),
#: tested to ASTM E1414/E1414M-16: a perforated plaster tile with polyester
#: insulation, ceiling continuous over the partition cap. Table 2 of the report
#: prints Dn,c per one-third octave; the reported class is CAC 39.
ALA_2020_DNC = (
    20.7, 22.1, 25.1, 25.8, 27.1, 35.2, 37.9, 41.2,
    43.4, 46.1, 48.2, 46.4, 45.7, 43.0, 40.7, 41.2,
)
ALA_2020_CAC = 39

#: Intertek report J7488.04-113-11-R0 (19 June 2019), tested to ASTM E1414:
#: 96 x 0,625 in "Better Than Wood" ceiling planks in an Armstrong Prelude
#: grid. Section 10 prints Dn,c and the per-band deficiency count, with
#: CAC 25 and a sum of deficiencies of 24.
INTERTEK_2019_DNC = (8, 13, 15, 15, 19, 23, 24, 21, 23, 26, 26, 27, 29, 32, 34, 36)
INTERTEK_2019_CAC = 25
INTERTEK_2019_DEFICIENCIES = (1, 0, 0, 3, 2, 1, 1, 5, 4, 2, 3, 2, 0, 0, 0, 0)
INTERTEK_2019_DEFICIENCY_SUM = 24


# ---------------------------------------------------------------------------
# ASTM E413 contour
# ---------------------------------------------------------------------------


def test_reference_contour_table() -> None:
    """ASTM E413-22 Table 1, digit for digit over its 16 bands."""
    assert tuple(sorted(CEILING_ATTENUATION_CONTOUR)) == E413_BANDS
    values = tuple(
        CEILING_ATTENUATION_CONTOUR[f] for f in sorted(CEILING_ATTENUATION_CONTOUR)
    )
    assert values == tuple(float(v) for v in E413_CONTOUR)


def test_reference_contour_is_zero_at_500_hz() -> None:
    """Note 2: the tabulated contour has a rating of zero, read at 500 Hz."""
    assert CEILING_ATTENUATION_CONTOUR[500.0] == 0.0


def test_reference_contour_shape() -> None:
    """The contour rises 3 dB per third octave to 400 Hz, then 1 dB, then flat.

    A structural property of ASTM E413-22 Table 1 that a mistranscribed entry
    would break even where the fitted rating happens not to move.
    """
    values = np.asarray(
        [CEILING_ATTENUATION_CONTOUR[f] for f in sorted(CEILING_ATTENUATION_CONTOUR)]
    )
    np.testing.assert_allclose(np.diff(values[:6]), 3.0)
    np.testing.assert_allclose(np.diff(values[6:11]), 1.0)
    np.testing.assert_allclose(np.diff(values[10:]), 0.0)


# ---------------------------------------------------------------------------
# Ceiling attenuation class (ASTM E413 clause 5, invoked by ASTM E1414)
# ---------------------------------------------------------------------------


def test_ala_2016_report_rates_to_its_printed_cac() -> None:
    """ALA 16-091-5: the printed Dn,c spectrum rates to the printed CAC 34."""
    res = ceiling_attenuation_class(ALA_2016_DNC)
    assert res.rating == ALA_2016_CAC
    assert res.deficiency_sum <= 32.0
    assert res.max_deficiency <= 8.0


def test_ala_2016_printed_deficiency_column_pins_the_contour() -> None:
    """ALA 16-091-5 data sheet: the deficiency column at CAC 34, total 27,5 dB.

    The report tabulates its deficiencies from the unrounded Dn,c (ASTM E413-22
    clause 5.2 rounds to integers before fitting), so this checks the contour
    transcription band by band rather than the fitting loop: every entry is
    ``contour + 34 - Dn,c`` floored at zero.
    """
    contour = np.asarray(
        [CEILING_ATTENUATION_CONTOUR[f] for f in sorted(CEILING_ATTENUATION_CONTOUR)]
    )
    deficiencies = np.maximum(contour + ALA_2016_CAC - np.asarray(ALA_2016_DNC), 0.0)
    np.testing.assert_allclose(deficiencies, ALA_2016_PRINTED_DEFICIENCIES, atol=5e-2)
    assert float(np.sum(deficiencies)) == pytest.approx(
        ALA_2016_PRINTED_DEFICIENCY_SUM, abs=5e-2
    )


def test_ala_2020_report_rates_to_its_printed_cac() -> None:
    """ALA 20-093-2: the printed Dn,c spectrum rates to the printed CAC 39.

    This report only rates correctly with the clause 5.2 rounding: on the
    unrounded values the sum of the deficiencies is 32,2 dB and CAC 39 would
    fail. The rounded fit lands exactly on both limits (32,0 dB and 8 dB).
    """
    res = ceiling_attenuation_class(ALA_2020_DNC)
    assert res.rating == ALA_2020_CAC
    assert res.deficiency_sum == pytest.approx(32.0)
    assert res.max_deficiency == pytest.approx(8.0)


def test_intertek_2019_report_rates_to_its_printed_cac() -> None:
    """Intertek J7488.04-113-11-R0: printed CAC 25, sum of deficiencies 24."""
    res = ceiling_attenuation_class(INTERTEK_2019_DNC)
    assert res.rating == INTERTEK_2019_CAC
    assert res.deficiency_sum == pytest.approx(INTERTEK_2019_DEFICIENCY_SUM)


def test_intertek_2019_per_band_deficiencies() -> None:
    """Intertek J7488.04-113-11-R0, Section 10 "number of deficiencies" column.

    The report's Dn,c is already integer, so the clause 5.2 rounding is a
    no-op and every band can be compared with the printed value.
    """
    res = ceiling_attenuation_class(INTERTEK_2019_DNC)
    np.testing.assert_allclose(res.deficiencies, INTERTEK_2019_DEFICIENCIES)
    np.testing.assert_allclose(
        res.shifted_reference,
        np.asarray(E413_CONTOUR, dtype=float) + INTERTEK_2019_CAC,
    )


def test_one_more_decibel_of_shift_always_fails() -> None:
    """Clause 5.3/5.4: the fit is the *highest* contour that still satisfies both limits."""
    contour = np.asarray(
        [CEILING_ATTENUATION_CONTOUR[f] for f in sorted(CEILING_ATTENUATION_CONTOUR)]
    )
    for data, rating in (
        (ALA_2016_DNC, ALA_2016_CAC),
        (ALA_2020_DNC, ALA_2020_CAC),
        (INTERTEK_2019_DNC, INTERTEK_2019_CAC),
    ):
        rounded = np.floor(np.asarray(data, dtype=float) + 0.5)
        excess = np.maximum(contour + rating + 1 - rounded, 0.0)
        assert float(np.sum(excess)) > 32.0 or float(np.max(excess)) > 8.0


def test_contour_shaped_spectrum_rates_two_decibels_above_itself() -> None:
    """A specimen exactly on the contour rates 2 dB above its own 500 Hz value.

    Clause 5.3 keeps raising the contour while the sum of the deficiencies is
    at most 32 dB (clause 5.4.1). Sixteen bands each 2 dB short sum to exactly
    32 dB, and 2 dB is well under the 8 dB single-band limit, so the fit stops
    two steps above the data.
    """
    contour = np.asarray(
        [CEILING_ATTENUATION_CONTOUR[f] for f in sorted(CEILING_ATTENUATION_CONTOUR)]
    )
    res = ceiling_attenuation_class(contour + 40.0)
    assert res.rating == 42
    assert res.deficiency_sum == pytest.approx(32.0)
    assert res.max_deficiency == pytest.approx(2.0)


def test_sum_rule_binds_at_exactly_thirty_two_decibels() -> None:
    """Clause 5.4.1: a total of 33 dB is already too much, 32 dB is not.

    Eleven bands sitting on the contour and five well above it accumulate
    11 dB of deficiency per 1 dB of shift, so the sums step 22, 33, 44: the fit
    stops at +2 dB with 22 dB, because +3 dB would need 33 dB of deficiency and
    the single-band limit (3 dB here) is nowhere near binding.
    """
    contour = np.asarray(
        [CEILING_ATTENUATION_CONTOUR[f] for f in sorted(CEILING_ATTENUATION_CONTOUR)]
    )
    data = contour + 40.0
    data[11:] += 60.0
    res = ceiling_attenuation_class(data)
    assert res.rating == 42
    assert res.deficiency_sum == pytest.approx(22.0)
    assert res.max_deficiency == pytest.approx(2.0)


def test_eight_decibel_rule_can_bind_before_the_sum() -> None:
    """Clause 5.4.2: a single deep dip caps the rating even with a small sum."""
    contour = np.asarray(
        [CEILING_ATTENUATION_CONTOUR[f] for f in sorted(CEILING_ATTENUATION_CONTOUR)]
    )
    data = contour + 40.0
    data[4] -= 8.0
    res = ceiling_attenuation_class(data)
    assert res.rating == 40
    assert res.max_deficiency == pytest.approx(8.0)
    assert res.deficiency_sum == pytest.approx(8.0)


def test_rating_rejects_a_wrong_band_count() -> None:
    with pytest.raises(ValueError, match="16 one-third-octave values"):
        ceiling_attenuation_class([30.0] * 15)


def test_rating_rejects_mismatched_frequencies() -> None:
    with pytest.raises(ValueError, match="contour bands"):
        ceiling_attenuation_class([30.0] * 16, [100.0 * (i + 1) for i in range(16)])


# ---------------------------------------------------------------------------
# Normalization (ISO 140-9:1985 clause 3.3)
# ---------------------------------------------------------------------------


def test_normalization_at_the_reference_area_is_the_level_difference() -> None:
    """ISO 140-9 clause 3.3: Dn,c = D - 10 lg(A/A0) is D when A = A0 = 10 m2."""
    dnc = normalized_ceiling_attenuation([90.0, 85.0], [50.0, 48.0], 10.0)
    np.testing.assert_allclose(dnc, [40.0, 37.0])


def test_normalization_halving_the_absorption_adds_three_decibels() -> None:
    """Halving A raises Dn,c by 10 lg 2 = 3,01 dB."""
    a = normalized_ceiling_attenuation([90.0], [50.0], 10.0)
    b = normalized_ceiling_attenuation([90.0], [50.0], 5.0)
    assert float(b[0] - a[0]) == pytest.approx(10.0 * np.log10(2.0))


def test_astm_reference_area_offset() -> None:
    """ASTM E1414 uses A0 = 12 m2, i.e. 10 lg(12/10) = 0,79 dB above ISO."""
    iso = normalized_ceiling_attenuation([90.0], [50.0], 9.0)
    astm = normalized_ceiling_attenuation([90.0], [50.0], 9.0, reference_area=12.0)
    assert float(astm[0] - iso[0]) == pytest.approx(10.0 * np.log10(1.2))
    assert float(astm[0] - iso[0]) == pytest.approx(0.79, abs=5e-3)


def test_intertek_2019_normalization_reproduces_the_report_column() -> None:
    """Intertek J7488.04-113-11-R0, Section 10, ASTM A0 = 12 m2.

    The report prints both room levels to the nearest integer, so the level
    difference it was computed from carries up to 1 dB of rounding; the
    absorption is printed to 0,1 m2 and Dn,c to the nearest integer. The
    reproduction is therefore checked to 1 dB, which is the printing budget.
    """
    source = (68, 79, 84, 87, 94, 92, 94, 92, 91, 86, 91, 89, 89, 90, 89, 90, 89, 89, 89)
    receive = (59, 76, 75, 74, 80, 77, 74, 69, 67, 66, 69, 64, 64, 64, 60, 57, 56, 54, 55)
    absorption = (
        19.7, 16.2, 13.0, 12.3, 8.6, 12.4, 13.6, 13.0, 11.7, 11.1, 11.4,
        11.0, 10.4, 10.3, 11.7, 12.4, 10.2, 9.0, 9.1,
    )
    printed = (7, 2, 8, 13, 15, 15, 19, 23, 24, 21, 23, 26, 26, 27, 29, 32, 34, 36, 35)
    dnc = normalized_ceiling_attenuation(
        source, receive, absorption, reference_area=12.0
    )
    np.testing.assert_allclose(dnc, printed, atol=1.0)


def test_normalization_rejects_a_non_positive_absorption() -> None:
    with pytest.raises(ValueError, match="absorption_area"):
        normalized_ceiling_attenuation([90.0], [50.0], 0.0)


# ---------------------------------------------------------------------------
# One-dimensional plenum model (Vigran Eqs. (9.13), (9.18) to (9.20))
# ---------------------------------------------------------------------------


def test_undamped_geometry_term_of_the_vigran_example() -> None:
    """Vigran Figs. 9.11 to 9.13: LS = LR = 4,75 m, h = 0,43 m, eps = 2.

    Eq. (9.20) then charges 10 lg(eps**2 LR/(4h)) = 10 lg(11,05) = 10,4 dB
    against the sum of the two ceiling reduction indices.
    """
    res = plenum_flanking_reduction_index(
        [30.0, 35.0], [30.0, 35.0], ceiling_length=4.75, plenum_height=0.43
    )
    assert res.geometry_term == pytest.approx(10.4, abs=0.05)
    np.testing.assert_allclose(res.reduction_index, [60.0 - 10.4, 70.0 - 10.4], atol=0.05)


def test_absorbing_sidewalls_gain_six_decibels() -> None:
    """Eq. (9.19): eps goes from 2 to 1, so the penalty drops by 20 lg 2 = 6,02 dB."""
    kwargs = {"ceiling_length": 4.75, "plenum_height": 0.43}
    reflecting = plenum_flanking_reduction_index([30.0], [30.0], **kwargs)
    absorbing = plenum_flanking_reduction_index(
        [30.0], [30.0], sidewalls="absorbing", **kwargs
    )
    assert float(absorbing.reduction_index[0] - reflecting.reduction_index[0]) == (
        pytest.approx(20.0 * np.log10(2.0))
    )


def test_deeper_plenum_helps_by_ten_lg_of_the_depth_ratio() -> None:
    """Eq. (9.20): the penalty carries 1/h, so doubling h gains 3,01 dB."""
    shallow = plenum_flanking_reduction_index(
        [30.0], [30.0], ceiling_length=4.75, plenum_height=0.4
    )
    deep = plenum_flanking_reduction_index(
        [30.0], [30.0], ceiling_length=4.75, plenum_height=0.8
    )
    assert float(deep.reduction_index[0] - shallow.reduction_index[0]) == pytest.approx(
        10.0 * np.log10(2.0)
    )


def test_transmission_factor_matches_the_reduction_index() -> None:
    """Rcl = -10 lg(tau_cl): the two outputs are the same quantity."""
    res = plenum_flanking_reduction_index(
        [30.0, 35.0, 40.0], [28.0, 33.0, 38.0],
        ceiling_length=4.75, plenum_height=0.43,
    )
    np.testing.assert_allclose(
        res.reduction_index, -10.0 * np.log10(res.transmission_factor)
    )


def test_attenuated_model_converges_to_the_undamped_limit() -> None:
    """Eq. (9.18) collapses to Eq. (9.20) for mS LS, mR LR << 1.

    Vigran states the limit explicitly. The receiving-side exponent also
    carries the leakage term sR tauR/h of Eq. (9.17), so the ceilings are made
    very insulating (tauR = 1e-10) to keep it far below mR.
    """
    undamped = plenum_flanking_reduction_index(
        [50.0], [100.0], ceiling_length=4.75, plenum_height=0.43
    )
    attenuated = plenum_flanking_reduction_index(
        [50.0], [100.0], ceiling_length=4.75, plenum_height=0.43,
        attenuation_source=[1e-5], attenuation_receiving=[1e-5],
    )
    assert float(attenuated.reduction_index[0]) == pytest.approx(
        float(undamped.reduction_index[0]), abs=1e-3
    )
    assert attenuated.model == "attenuated"
    assert undamped.model == "undamped"


def test_plenum_attenuation_increases_the_reduction_index() -> None:
    """A lined plenum attenuates the sideways path: Rcl rises with m."""
    values = [
        float(
            plenum_flanking_reduction_index(
                [30.0], [30.0], ceiling_length=4.75, plenum_height=0.43,
                attenuation_source=[m], attenuation_receiving=[m],
            ).reduction_index[0]
        )
        for m in (0.01, 0.1, 0.5, 1.0)
    ]
    assert values == sorted(values)


def test_attenuation_coefficients_must_come_in_pairs() -> None:
    with pytest.raises(ValueError, match="together"):
        plenum_flanking_reduction_index(
            [30.0], [30.0], ceiling_length=4.75, plenum_height=0.43,
            attenuation_source=[0.1],
        )


def test_partition_reference_shifts_by_the_room_aspect_ratio() -> None:
    """Eq. (9.13): Rcl,p = Rcl + 10 lg(HS/LS)."""
    shifted = partition_referenced_reduction_index([50.0], 2.7, 4.75)
    assert float(shifted[0]) == pytest.approx(50.0 + 10.0 * np.log10(2.7 / 4.75))


def test_plenum_model_rejects_a_non_positive_height() -> None:
    with pytest.raises(ValueError, match="plenum_height"):
        plenum_flanking_reduction_index(
            [30.0], [30.0], ceiling_length=4.75, plenum_height=0.0
        )

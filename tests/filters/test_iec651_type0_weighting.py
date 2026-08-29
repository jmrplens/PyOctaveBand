#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The IEC 651:1979 Table V tolerance masks, and the Type 0 grade.

IEC 61672-1:2013 publishes classes 1 and 2 only. Type 0, the tightest of the
four instrument types of IEC 651:1979 subclause 1.2, survives only in that
superseded edition, so ``verify_weighting_class`` reaches it through
``edition="1979"`` the way ``verify_filter_class`` reaches the IEC 61260:1995
class 0.

The point of carrying it is that the two editions do not see the same errors:
IEC 61672-1 class 1 opens to +2.5/-16 dB at 16 kHz and +3/-inf at 20 kHz,
where Type 0 holds +2/-3 dB at both, so a filter can clear class 1 at the top
of the range with a fault Type 0 refuses. That case is exercised below with a
real design, not a mock.

The mask transcribed inside :mod:`phonometry.filters.weighting_compliance` is
pinned to the independent ``reference_data`` copy shared with the CI
conformance report, so a typo in either surfaces.
"""

import math

import pytest
from reference_data import ANSIS14_TABLE5, IEC651_TABLE5, IEC61672_TABLE3

from phonometry import filters
from phonometry.filters.weighting_compliance import _IEC651_TABLE5

# The three rows where IEC 651 Table V and ANSI S1.4-1983 Table V read the
# Type 0 grade differently: IEC 651 prints "+2; -inf", ANSI prints a
# two-sided +2/-5, +2/-4 and +2/-3.
_TYPE0_EDITION_SPLIT = (10.0, 12.5, 16.0)

# Sample rates the fitted A and C designs are expected to earn Type 0 at.
# 32 kHz is the lowest of them: below it the 20 kHz row falls above Nyquist
# and cannot be demonstrated at all.
_TYPE0_RATES = (32000, 44100, 48000, 96000, 192000)


# ---------------------------------------------------------------------------
# Mask transcription
# ---------------------------------------------------------------------------


def test_table5_matches_reference_data() -> None:
    """The module's Table V equals the independent reference_data copy."""
    assert len(_IEC651_TABLE5) == len(IEC651_TABLE5) == 34
    for row, ref_row in zip(_IEC651_TABLE5, IEC651_TABLE5, strict=True):
        assert row == pytest.approx(ref_row), f"Table V mismatch at {ref_row[0]} Hz"


def test_table5_covers_the_iec61672_row_set() -> None:
    """Table V is tabulated at the same 34 nominal frequencies as Table 3."""
    assert [row[0] for row in _IEC651_TABLE5] == [row[0] for row in IEC61672_TABLE3]


def test_every_type_is_wider_than_the_one_before_it() -> None:
    """Subclause 1.3: the tolerances broaden as the type number rises.

    The verdict logic returns the first type whose margin is not negative, so
    a table where some type were tighter than its predecessor at one row
    would make that ordering a lie.
    """
    for row in _IEC651_TABLE5:
        freq, uppers, lowers = row[0], row[1::2], row[2::2]
        for tighter, looser in zip(uppers, uppers[1:], strict=False):
            assert tighter <= looser, f"upper limits not ordered at {freq} Hz"
        for tighter, looser in zip(lowers, lowers[1:], strict=False):
            assert tighter >= looser, f"lower limits not ordered at {freq} Hz"


def test_type0_differs_from_ansi_type0_at_the_three_lowest_rows() -> None:
    """The IEC and ANSI Type 0 masks are two masks, and stay two masks.

    ANSI S1.4-1983 Table V is two-sided and stricter at 10/12.5/16 Hz, where
    IEC 651 Table V is upper-only; everywhere else the two Type 0 columns
    agree. Merging them would either invent a lower limit IEC 651 never
    published or drop one ANSI does.
    """
    for row, ansi in zip(_IEC651_TABLE5, ANSIS14_TABLE5, strict=True):
        freq, upper, lower = row[0], row[1], row[2]
        assert upper == pytest.approx(ansi[1]), f"upper differs at {freq} Hz"
        if freq in _TYPE0_EDITION_SPLIT:
            assert math.isinf(lower)
            assert math.isfinite(ansi[2]), f"ANSI is two-sided at {freq} Hz"
        else:
            assert lower == pytest.approx(ansi[2]), f"lower differs at {freq} Hz"


def test_class_limits_return_the_type0_column() -> None:
    """The published Type 0 cells, at the rows the two editions disagree on."""
    freqs, lower, upper = filters.weighting_class_limits(0, edition="1979")
    assert len(freqs) == 34
    cells = {
        float(f): (float(u), float(lo))
        for f, u, lo in zip(freqs, upper, lower, strict=True)
    }
    assert cells[10.0][0] == 2.0
    assert math.isinf(cells[10.0][1])
    assert cells[10.0][1] < 0
    assert cells[1000.0] == (0.7, -0.7)  # the reference-frequency row, as printed
    assert cells[5000.0] == (1.0, -1.0)
    assert cells[6300.0] == (1.0, -1.5)
    # The two rows IEC 61672-1 class 1 leaves wide open: +2.5/-16 and +3/-inf.
    assert cells[16000.0] == (2.0, -3.0)
    assert cells[20000.0] == (2.0, -3.0)


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fs", _TYPE0_RATES)
@pytest.mark.parametrize("curve", ["A", "C"])
def test_fitted_design_earns_type0(fs: int, curve: str) -> None:
    """The shipped A and C designs meet the laboratory grade at every rate."""
    result = filters.verify_weighting_class(
        filters.WeightingFilter(fs, curve), edition="1979"
    )
    assert result["overall_class"] == 0
    worst = min(band["margin_class0_db"] for band in result["bands"])
    # The binding rows are the +/-0.7 dB pass-band cells, cleared by ~0.65 dB.
    assert worst > 0.6, f"{curve} at {fs} Hz binds at {worst:+.3f} dB"


def test_plain_bilinear_fails_type0_while_passing_class1() -> None:
    """A fault IEC 61672-1 class 1 cannot see, reported under Type 0.

    The undersampled bilinear design droops badly approaching Nyquist. At
    48 kHz its A response is 6.2 dB low at the 16 kHz row and 15.7 dB low at
    the 20 kHz row - and IEC 61672-1:2013 Table 3 admits both, because class 1
    allows -16 dB at 16 kHz and imposes no lower limit at all at 20 kHz. The
    same filter against IEC 651:1979 Table V is refused Type 0 (+2/-3 dB at
    both rows) and graded Type 1, which is what it is.
    """
    wf = filters.WeightingFilter(48000, "A", high_accuracy=False)

    modern = filters.verify_weighting_class(wf)
    assert modern["overall_class"] == 1
    assert all(band["margin_class1_db"] >= 0 for band in modern["bands"])

    historic = filters.verify_weighting_class(wf, edition="1979")
    assert historic["overall_class"] == 1
    failed = {
        band["freq"] for band in historic["bands"] if band["margin_class0_db"] < 0
    }
    assert failed == {16000.0, 20000.0}
    # Same deviation, opposite verdict: the mask is what differs, not the
    # measurement.
    per_freq = {band["freq"]: band for band in historic["bands"]}
    modern_per_freq = {band["freq"]: band for band in modern["bands"]}
    for freq in failed:
        assert per_freq[freq]["deviation_db"] == modern_per_freq[freq]["deviation_db"]
        assert per_freq[freq]["class"] == 1


def test_band_and_sweep_carry_one_margin_per_type() -> None:
    """Four types in the edition, four margin keys in every verdict."""
    result = filters.verify_weighting_class(
        filters.WeightingFilter(48000, "A"), edition="1979"
    )
    expected = [f"margin_class{cls}_db" for cls in (0, 1, 2, 3)]
    for band in result["bands"]:
        assert list(band) == ["freq", "class", "deviation_db", *expected]
    assert list(result["between_nominals"]) == ["worst_freq", *expected]


def test_b_is_graded_against_table_v_not_the_ansi_mask() -> None:
    """The Table V footnote governs every weighting, B included.

    Under the 2013 edition B has to borrow ANSI S1.4-1983 Table V, whose
    Type 0 column is two-sided at 10 Hz. Under the 1979 edition it uses
    IEC 651 Table V, which is upper-only there - so the mask B is held to
    follows the edition, not the curve.
    """
    wf = filters.WeightingFilter(48000, "B")
    result = filters.verify_weighting_class(wf, edition="1979")
    assert result["overall_class"] == 0
    band_10 = next(band for band in result["bands"] if band["freq"] == 10.0)
    # Upper-only: the margin is the distance to the +2 dB ceiling alone.
    assert band_10["margin_class0_db"] == pytest.approx(
        2.0 - band_10["deviation_db"], abs=1e-9
    )


def test_default_edition_is_unchanged_by_the_new_one() -> None:
    """Omitting ``edition`` still grades against IEC 61672-1:2013 Table 3."""
    wf = filters.WeightingFilter(48000, "A")
    assert filters.verify_weighting_class(wf) == filters.verify_weighting_class(
        wf, edition="2013"
    )


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_unknown_edition_is_refused_by_the_verifier() -> None:
    weighting = filters.WeightingFilter(48000, "A")
    with pytest.raises(ValueError, match="edition must be"):
        filters.verify_weighting_class(weighting, edition="1985")


def test_unknown_edition_is_refused_by_the_limits() -> None:
    with pytest.raises(ValueError, match="edition must be"):
        filters.weighting_class_limits(1, edition="1985")


def test_curve_outside_the_1979_edition_is_refused() -> None:
    """Z arrived with IEC 61672-1; IEC 651 never defined it."""
    weighting = filters.WeightingFilter(48000, "Z")
    with pytest.raises(ValueError, match="Weighting curve must be .*for edition"):
        filters.verify_weighting_class(weighting, edition="1979")


def test_type_outside_the_1979_edition_is_refused() -> None:
    with pytest.raises(ValueError, match="weighting_class must be one of"):
        filters.weighting_class_limits(4, edition="1979")


def test_type0_is_refused_by_the_2013_edition() -> None:
    """IEC 61672-1:2013 publishes no class 0, so asking for one is an error."""
    with pytest.raises(ValueError, match="weighting_class must be one of"):
        filters.weighting_class_limits(0)

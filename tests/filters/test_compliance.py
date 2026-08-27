#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the IEC 61260 filter class verifier (2014 classes 1/2 and the
1995 / ANSI S1.11-2004 edition that adds class 0).
"""

import numpy as np
import pytest
import reference_data as ref

from phonometry import filters
from phonometry.filters.compliance import (
    _PASSBAND_MAX_1995,
    _PASSBAND_MIN_1995,
    _STOPBAND_MIN_1995,
    class_limits,
)


def test_class_limits_table1_anchor_values() -> None:
    """Spot-check the transcription against BS EN 61260-1:2014 Table 1 (b=1)."""
    G = 10 ** (3 / 10)
    # Passband center: class 1 in [-0.4, +0.4]
    lo, hi = class_limits(1.0, 1, np.array([1.0]))
    assert lo[0] == pytest.approx(-0.4)
    assert hi[0] == pytest.approx(0.4)
    # Band edge (inside): max +5.3 (class 1)
    lo, hi = class_limits(1.0, 1, np.array([G**0.5 * 0.999999]))
    assert hi[0] == pytest.approx(5.3, abs=0.05)
    # Just outside the edge: minimum attenuation +1.2 (class 1)
    lo, hi = class_limits(1.0, 1, np.array([G**0.5 * 1.000001]))
    assert lo[0] == pytest.approx(1.2, abs=0.05)
    assert np.isinf(hi[0])
    # One octave out: minimum +16.6 (class 1) / +15.6 (class 2)
    lo1, _ = class_limits(1.0, 1, np.array([G]))
    lo2, _ = class_limits(1.0, 2, np.array([G]))
    assert lo1[0] == pytest.approx(16.6)
    assert lo2[0] == pytest.approx(15.6)
    # Far stopband: minimum +70 (class 1) / +60 (class 2)
    lo1, _ = class_limits(1.0, 1, np.array([G**5]))
    lo2, _ = class_limits(1.0, 2, np.array([G**5]))
    assert lo1[0] == pytest.approx(70.0)
    assert lo2[0] == pytest.approx(60.0)


def test_class_limits_low_side_is_reciprocal() -> None:
    """Formula (10): the low side mirrors the high side at 1/Omega."""
    omega = np.array([1.3, 2.0, 4.0])
    lo_h, hi_h = class_limits(3.0, 1, omega)
    lo_l, hi_l = class_limits(3.0, 1, 1.0 / omega)
    np.testing.assert_allclose(lo_l, lo_h)
    np.testing.assert_allclose(hi_l, hi_h, equal_nan=False)


def test_butter_order6_third_octave_meets_class1() -> None:
    bank = filters.OctaveFilterBank(fs=48000, fraction=3, order=6, limits=[100, 5000])
    result = filters.verify_filter_class(bank)
    assert result["overall_class"] == 1, result


def test_butter_order6_octave_meets_class1() -> None:
    bank = filters.OctaveFilterBank(fs=48000, fraction=1, order=6, limits=[125, 4000])
    result = filters.verify_filter_class(bank)
    assert result["overall_class"] == 1, result


def test_low_order_fails_class1() -> None:
    """A 1st-order bank cannot reach the class stopband attenuations."""
    bank = filters.OctaveFilterBank(fs=48000, fraction=1, order=1, limits=[500, 2000])
    result = filters.verify_filter_class(bank)
    assert result["overall_class"] is None


def test_result_has_per_band_details() -> None:
    bank = filters.OctaveFilterBank(fs=48000, fraction=1, order=6, limits=[500, 2000])
    result = filters.verify_filter_class(bank)
    assert len(result["bands"]) == bank.num_bands
    for band in result["bands"]:
        assert set(band) >= {"freq", "class", "margin_class1_db", "margin_class2_db"}
    # margins must be finite floats
    assert all(np.isfinite(b["margin_class1_db"]) for b in result["bands"])


def test_stateful_bank_matches_stateless_design() -> None:
    """Stateful banks share the SOS design: verification must agree exactly."""
    stateful = filters.OctaveFilterBank(
        fs=48000,
        fraction=1,
        order=6,
        limits=[500, 2000],
        design=filters.FilterDesign(resample=False),
        block_processing=filters.BlockProcessing(stateful=True),
    )
    stateless = filters.OctaveFilterBank(
        fs=48000,
        fraction=1,
        order=6,
        limits=[500, 2000],
        design=filters.FilterDesign(resample=False),
    )
    r_stateful = filters.verify_filter_class(stateful)
    r_stateless = filters.verify_filter_class(stateless)
    assert r_stateful["overall_class"] == r_stateless["overall_class"]
    for a, b in zip(r_stateful["bands"], r_stateless["bands"], strict=True):
        assert a["margin_class1_db"] == pytest.approx(b["margin_class1_db"])


def test_coarse_grid_breakpoints_evaluated_exactly() -> None:
    """The Table 1 breakpoints are evaluated with sosfreqz at their exact
    frequencies, not interpolated off the grid: even the permitted 16-point
    floor reproduces the dense-grid verdict and binding margin (interpolation
    used to yield garbage margins around -190 dB there).
    """
    bank = filters.OctaveFilterBank(
        fs=48000,
        fraction=3,
        order=6,
        limits=[100, 5000],
        design=filters.FilterDesign(filter_type="butter"),
    )
    dense = filters.verify_filter_class(bank)
    coarse = filters.verify_filter_class(bank, num_points=16)
    assert coarse["overall_class"] == dense["overall_class"] == 1
    m_dense = min(b["margin_class1_db"] for b in dense["bands"])
    m_coarse = min(b["margin_class1_db"] for b in coarse["bands"])
    assert m_coarse == pytest.approx(m_dense, abs=0.05)


def test_invalid_inputs_raise() -> None:
    bank = filters.OctaveFilterBank(fs=48000, fraction=1, order=6, limits=[500, 2000])
    band_centre = np.array([1.0])
    with pytest.raises(ValueError, match="num_points"):
        filters.verify_filter_class(bank, num_points=4)
    with pytest.raises(ValueError, match="filter_class"):
        class_limits(1.0, 3, band_centre)
    with pytest.raises(ValueError, match="fraction"):
        class_limits(-1.0, 1, band_centre)


# ---------------------------------------------------------------------------
# IEC 61260:1995 / ANSI S1.11-2004 edition (adds class 0)
# ---------------------------------------------------------------------------


def test_1995_tables_match_reference_data() -> None:
    """The module's 1995 mask reproduces the shared reference_data transcription."""
    assert _PASSBAND_MIN_1995 == ref.IEC61260_1995_PASSBAND_MIN
    assert [tuple(r) for r in _PASSBAND_MAX_1995] == [
        tuple(r) for r in ref.IEC61260_1995_PASSBAND_MAX
    ]
    assert [tuple(r) for r in _STOPBAND_MIN_1995] == [
        tuple(r) for r in ref.IEC61260_1995_STOPBAND_MIN
    ]


def test_1995_class0_anchor_values() -> None:
    """class_limits reproduces the Table 1 class-0 anchors (octave band)."""
    g = 10 ** (3 / 10)
    lo, hi = class_limits(1.0, 0, np.array([1.0]), edition="1995")
    assert (lo[0], hi[0]) == (-0.15, 0.15)  # Omega = 1
    lo, hi = class_limits(1.0, 0, np.array([g**0.5 * 0.999999]), edition="1995")
    assert hi[0] == pytest.approx(4.5, abs=1e-3)  # pass-band edge max
    lo, _ = class_limits(1.0, 0, np.array([g**0.5 * 1.000001]), edition="1995")
    assert lo[0] == pytest.approx(2.3, abs=1e-3)  # stop-band edge min
    lo, _ = class_limits(1.0, 0, np.array([g]), edition="1995")
    assert lo[0] == pytest.approx(18.0, abs=1e-6)  # G**1 min


def test_1995_class0_is_strictest() -> None:
    """At every breakpoint class 0 <= class 1 <= class 2 max, and min ordering."""
    g = 10 ** (3 / 10)
    omega = g ** np.linspace(0, 1.5, 40)
    lo0, hi0 = class_limits(1.0, 0, omega, edition="1995")
    lo1, hi1 = class_limits(1.0, 1, omega, edition="1995")
    lo2, hi2 = class_limits(1.0, 2, omega, edition="1995")
    # Tighter class => smaller (or equal) maximum allowance in the pass-band.
    pb = omega <= g**0.5
    assert np.all(hi0[pb] <= hi1[pb] + 1e-9)
    assert np.all(hi1[pb] <= hi2[pb] + 1e-9)
    # ...and a larger (or equal) minimum: the corridor floor rises with strictness.
    assert np.all(lo0[pb] >= lo1[pb] - 1e-9)
    assert np.all(lo1[pb] >= lo2[pb] - 1e-9)


def test_butter_meets_class0_1995() -> None:
    """The default order-6 Butterworth bank clears the strict 1995 class 0."""
    bank = filters.OctaveFilterBank(fs=48000, fraction=3, order=6)
    result = filters.verify_filter_class(bank, edition="1995")
    assert result["overall_class"] == 0, result
    band = result["bands"][0]
    assert set(band) == {
        "freq",
        "class",
        "checked_to_omega",
        "margin_class0_db",
        "margin_class1_db",
        "margin_class2_db",
    }
    # A class-0 band must clear class 1 and class 2 by at least as much.
    for b in result["bands"]:
        assert b["margin_class0_db"] <= b["margin_class1_db"] + 1e-9
        assert b["margin_class1_db"] <= b["margin_class2_db"] + 1e-9


def test_2014_default_unaffected_by_edition_support() -> None:
    """The default edition still reports only classes 1/2 (no class-0 key)."""
    bank = filters.OctaveFilterBank(fs=48000, fraction=3, order=6)
    result = filters.verify_filter_class(bank)
    assert result["overall_class"] == 1
    assert set(result["bands"][0]) == {
        "freq",
        "class",
        "checked_to_omega",
        "margin_class1_db",
        "margin_class2_db",
    }


def test_range_limited_flag_reports_unverifiable_stopband() -> None:
    """The verdict flags that the mask beyond the processing Nyquist is unchecked.

    Each band's multirate processing Nyquist sits around 1.8-2.0 f_m while
    the octave-band stop-band mask runs to G^4 = 15.85 f_m, so the G^2..G^4
    rows cannot be demonstrated and the verdict must say so instead of
    claiming full Table 1 conformance.
    """
    bank = filters.OctaveFilterBank(fs=48000, fraction=1, order=6, limits=[125, 4000])
    result = filters.verify_filter_class(bank)
    assert result["range_limited"] is True
    for band in result["bands"]:
        # The checked range covers the band edge but not the G^4 mask end.
        assert 10**0.15 < band["checked_to_omega"] < 15.0


def test_1995_rejects_out_of_range_class_and_bad_edition() -> None:
    band_centre = np.array([1.0])
    with pytest.raises(ValueError, match="filter_class"):
        class_limits(1.0, 3, band_centre, edition="1995")
    with pytest.raises(ValueError, match="filter_class"):
        class_limits(1.0, 0, band_centre)  # class 0 invalid for 2014
    with pytest.raises(ValueError, match="edition"):
        class_limits(1.0, 1, band_centre, edition="2020")
    bank = filters.OctaveFilterBank(fs=48000, fraction=1, order=6, limits=[500, 2000])
    with pytest.raises(ValueError, match="edition"):
        filters.verify_filter_class(bank, edition="2020")


def test_map_breakpoint_reproduces_table_f1() -> None:
    """IEC 61260-1:2014 Table F.1: the Formula (9) mapping reproduces every
    printed one-third-octave (b = 3) breakpoint and reciprocal to the five
    printed decimals.
    """
    from reference_data import IEC61260_TABLE_F1

    from phonometry.filters.compliance import _map_breakpoint

    for exponent, (omega, reciprocal) in IEC61260_TABLE_F1.items():
        got = _map_breakpoint(exponent, 3)
        assert got == pytest.approx(omega, abs=5e-6), exponent
        assert 1.0 / got == pytest.approx(reciprocal, abs=5e-6), exponent


# --------------------------------------------------------------------------
# Per-band entries that do not agree
# --------------------------------------------------------------------------
def test_a_filter_verdict_refuses_per_band_entries_that_disagree() -> None:
    """The fiche prints one row per band under the bank's overall class.

    A band list short of an entry gives a sheet whose verdict covers a band
    that is nowhere in its table.
    """
    import dataclasses

    from phonometry.filters.core import OctaveFilterBank

    result = filters.filter_class_compliance(
        OctaveFilterBank(fs=48000, fraction=1, order=4, limits=[500, 16000])
    )
    with pytest.raises(ValueError, match="'bands'"):
        dataclasses.replace(result, bands=result.bands[:-1])


def test_a_filter_verdict_refuses_a_class_its_bands_carry_no_margins_for() -> None:
    """Class 0 exists only in the 1995 edition, so a 2014 verdict has no
    ``margin_class0_db`` keys; an unpinned class would die in a bare
    ``KeyError`` halfway through the corridor figure.
    """
    import dataclasses

    from phonometry.filters.core import OctaveFilterBank

    result = filters.filter_class_compliance(
        OctaveFilterBank(fs=48000, fraction=1, order=4, limits=[500, 16000])
    )
    with pytest.raises(ValueError, match="'overall_class' must be one of"):
        dataclasses.replace(result, overall_class=0)


def test_a_filter_verdict_refuses_an_edition_that_disagrees_with_its_bands() -> None:
    """A 2014 verdict relabelled 1995 would silently draw the other
    edition's corridor under the same title.
    """
    import dataclasses

    from phonometry.filters.core import OctaveFilterBank

    result = filters.filter_class_compliance(
        OctaveFilterBank(fs=48000, fraction=1, order=4, limits=[500, 16000])
    )
    with pytest.raises(ValueError, match="'edition'"):
        dataclasses.replace(result, edition="1995")


def test_a_filter_verdict_refuses_a_later_band_short_of_a_margin_key() -> None:
    """The margin keys are read off every band, not only the first.

    ``verify_filter_class`` fills each entry from the same list of classes, so
    a band list whose entries disagree among themselves is one no bank
    produced. Accepting it on the strength of the first band alone left the
    ``KeyError`` for the reader: the fiche's per-band table and the plot's
    worst-band search read ``margin_class<c>_db`` out of every band, for the
    reference class. The key dropped here is that one, so the band list is
    exactly the one that used to construct and then die mid-figure.
    """
    import copy
    import dataclasses

    from phonometry.filters.core import OctaveFilterBank

    result = filters.filter_class_compliance(
        OctaveFilterBank(fs=48000, fraction=1, order=4, limits=[500, 16000])
    )
    # The producer's own bands all carry the same margin keys, so the guard
    # cannot refuse a verdict a bank emitted.
    assert len({frozenset(band) for band in result.bands}) == 1
    bands = tuple(copy.deepcopy(band) for band in result.bands)
    del bands[1][f"margin_class{result.reference_class()}_db"]
    with pytest.raises(
        ValueError, match=r"entry of 'bands' carries margins for classes \[2\]"
    ):
        dataclasses.replace(result, bands=bands)


def test_a_filter_verdict_refuses_a_non_finite_per_band_value() -> None:
    """Every margin is a ``min`` over the measured attenuation against the
    Table 1 mask, so no bank emits a NaN; one smuggled in prints
    ``Class 1 (+nan dB)`` in the per-band table under a boxed verdict that
    still reads COMPLIES, because the binding margin reads another band.
    """
    import copy
    import dataclasses

    from phonometry.filters.core import OctaveFilterBank

    result = filters.filter_class_compliance(
        OctaveFilterBank(fs=48000, fraction=1, order=4, limits=[500, 16000])
    )
    bands = tuple(copy.deepcopy(band) for band in result.bands)
    bands[0][f"margin_class{result.reference_class()}_db"] = float("nan")
    with pytest.raises(ValueError, match=r"'bands' must carry finite per-band"):
        dataclasses.replace(result, bands=bands)


def test_a_filter_verdict_refuses_a_class_stated_over_no_bands() -> None:
    """A bank with no bands in range is a real outcome, always paired with
    ``overall_class = None``. A class stated over zero bands would print an
    accredited verdict box above a table reportlab then refuses to build,
    complaining about a table with no rows and naming neither the bands nor
    the bank.
    """
    import dataclasses

    import numpy as np

    from phonometry.filters.core import OctaveFilterBank

    result = filters.filter_class_compliance(
        OctaveFilterBank(fs=48000, fraction=1, order=4, limits=[500, 16000])
    )
    empty = {
        "bands": (),
        "sos": (),
        "band_frequencies": np.asarray([], dtype=float),
        "factors": (),
    }
    # The same emptiness with overall_class None is the producer's own output.
    assert dataclasses.replace(result, overall_class=None, **empty).bands == ()
    with pytest.raises(ValueError, match=r"'overall_class' is 1 but 'bands' is empty"):
        dataclasses.replace(result, overall_class=1, **empty)


def test_a_filter_verdict_refuses_an_unknown_edition() -> None:
    """The edition is a pinned tag, refused by name at construction."""
    import dataclasses

    from phonometry.filters.core import OctaveFilterBank

    result = filters.filter_class_compliance(
        OctaveFilterBank(fs=48000, fraction=1, order=4, limits=[500, 16000])
    )
    with pytest.raises(ValueError, match="'edition' must be one of"):
        dataclasses.replace(result, edition="2003")

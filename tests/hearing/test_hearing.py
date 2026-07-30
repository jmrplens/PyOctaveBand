#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for :mod:`phonometry.hearing.threshold` (ISO 7029:2017 and ISO 389-7:2005).

The age-related model is validated against the standard's own boundary
behaviour (at age 18 the median deviation is zero and the spreads equal the
constant polynomial terms of Tables 2-5) and its monotonic/sex trends, and the
reference threshold against the ISO 389-7 Table 1 values.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from reference_data import ISO389_7_REF_FREE_1KHZ, ISO7029_MEDIAN_MALE_60_4KHZ

from phonometry.hearing import threshold as h


def test_frequency_and_table_shapes() -> None:
    assert h.AUDIOMETRIC_FREQUENCIES.size == 11
    assert h.AUDIOMETRIC_FREQUENCIES[0] == 125.0
    assert h.AUDIOMETRIC_FREQUENCIES[-1] == 8000.0
    for table in (h._SU_MALE, h._SL_MALE, h._SU_FEMALE, h._SL_FEMALE):
        assert table.shape == (11, 6)
    for table in (h._MEDIAN_MALE, h._MEDIAN_FEMALE):
        assert table.shape == (11, 2)


@pytest.mark.parametrize("sex", ["male", "female"])
def test_reference_age_is_zero_median(sex: str) -> None:
    # ISO 7029 clause 4.2: at age 18 the median deviation is zero everywhere,
    # and the spreads reduce to the constant terms c0 of Tables 2-5.
    result = h.age_threshold(18, sex, 0.5)
    np.testing.assert_allclose(result.median, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.threshold, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.spread_upper, h._SU[sex][:, 0])
    np.testing.assert_allclose(result.spread_lower, h._SL[sex][:, 0])


def test_median_matches_reference_values() -> None:
    # Reference median deviations computed from the Table 1 formula.
    male = h.age_threshold(60, "male", 0.5)
    assert male.median[4] == pytest.approx(7.8473, abs=1e-3)   # 1000 Hz
    assert male.median[8] == pytest.approx(ISO7029_MEDIAN_MALE_60_4KHZ, abs=1e-3)  # 4000 Hz
    female = h.age_threshold(60, "female", 0.5)
    assert female.median[8] == pytest.approx(15.3218, abs=1e-3)


def test_median_increases_with_age() -> None:
    m = [h.age_threshold(a, "male", 0.5).median[8] for a in (18, 30, 50, 70)]
    assert all(x < y for x, y in pairwise(m))
    assert m[0] == pytest.approx(0.0, abs=1e-12)


def test_fractiles_bracket_the_median() -> None:
    # A worse fractile (0.9) lies above the median; a better one (0.1) below.
    median = h.age_threshold(60, "male", 0.5).threshold
    worse = h.age_threshold(60, "male", 0.9).threshold
    better = h.age_threshold(60, "male", 0.1).threshold
    assert np.all(better < median)
    assert np.all(worse > median)


def test_male_median_exceeds_female_at_high_frequency() -> None:
    male = h.age_threshold(70, "male", 0.5).median[8]     # 4000 Hz
    female = h.age_threshold(70, "female", 0.5).median[8]
    assert male > female


def test_reference_threshold_matches_iso389_7() -> None:
    free = h.reference_threshold("free-field")
    diffuse = h.reference_threshold("diffuse-field")
    # ISO 389-7:2005 Table 1 (audiometric frequencies).
    assert free[4] == pytest.approx(ISO389_7_REF_FREE_1KHZ)  # 1000 Hz free-field
    assert free[10] == pytest.approx(12.6)  # 8000 Hz free-field
    assert diffuse[4] == pytest.approx(0.8)  # 1000 Hz diffuse-field
    # Free- and diffuse-field agree at low frequencies, diverge higher up.
    assert free[0] == diffuse[0]
    assert free[4] != diffuse[4]


def test_subset_by_frequency() -> None:
    result = h.age_threshold(60, "male", 0.5, frequencies=[1000, 4000])
    np.testing.assert_allclose(result.frequencies, [1000.0, 4000.0])
    assert result.median[0] == pytest.approx(7.8473, abs=1e-3)
    ref = h.reference_threshold("free-field", frequencies=[1000, 8000])
    np.testing.assert_allclose(ref, [2.4, 12.6])


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="at least 18"):
        h.age_threshold(10, "male")
    with pytest.raises(ValueError, match="sex must be"):
        h.age_threshold(40, "other")
    with pytest.raises(ValueError, match="fractile"):
        h.age_threshold(40, "male", fractile=1.0)
    with pytest.raises(ValueError, match="audiometric frequency"):
        h.age_threshold(40, "male", frequencies=[777.0])
    with pytest.raises(ValueError, match="field must be"):
        h.reference_threshold("random")


def test_result_fields_and_plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = h.age_threshold(55, "female", 0.75)
    assert result.age == 55.0
    assert result.sex == "female"
    assert result.threshold.shape == (11,)
    ax = result.plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")

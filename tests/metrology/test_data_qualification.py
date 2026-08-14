#  Copyright (c) 2026. Jose Manuel Requena Plens

"""Data qualification: stationarity tests and Rice crossing/peak statistics.

Clean-room oracles from Bendat & Piersol, *Random Data* 4e:

* Sec. 4.5.2 with Table A.6: the reverse-arrangement count of the worked
  N = 8 sequence (A = 14), Example 4.4 (A = 86, accepted at 0.05) and the
  full alpha = 0.05 columns of the percentage-point table (N = 10..100).
* The exact run distribution (Wald & Wolfowitz 1940): classical critical
  values 6/15 for n1 = n2 = 10 at alpha = 0.05.
* Sec. 5.5: N0 = 2 sqrt(m2/m0) (Eq. (5.195)) on synthetic bandlimited
  Gaussian noise against the closed forms of Examples 5.12/5.13, the
  Rayleigh peak exceedance exp(-8) at 4 sigma of Example 5.14, and the
  wide-band irregularity factor of an ideal low-pass spectrum.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phonometry as ph
from phonometry.metrology import data_qualification as rd

# ---------------------------------------------------------------------------
# Reverse arrangement test (B&P Sec. 4.5.2 / Table A.6)
# ---------------------------------------------------------------------------

#: B&P Example 4.4: N = 20 observations whose reverse arrangements A = 86.
EXAMPLE_4_4 = [
    5.2, 6.2, 3.7, 6.4, 3.9, 4.0, 3.9, 5.3, 4.0, 4.6,
    5.9, 6.5, 4.3, 5.7, 3.1, 5.6, 5.2, 3.9, 6.2, 5.0,
]

#: B&P Table A.6, alpha = 0.05 columns: N -> (A_{N;0.975}, A_{N;0.025}).
TABLE_A6_ALPHA_05 = {
    10: (11, 33), 12: (18, 47), 14: (27, 63), 16: (38, 81), 18: (50, 102),
    20: (64, 125), 30: (162, 272), 40: (305, 474), 50: (495, 729),
    60: (731, 1038), 70: (1014, 1400), 80: (1344, 1815), 90: (1721, 2283),
    100: (2145, 2804),
}


def test_reverse_arrangement_count_bp_worked_example() -> None:
    # B&P Sec. 4.5.2 worked sequence (N = 8): A = 3+1+4+4+1+0+1 = 14.
    values = np.array([5.0, 3.0, 8.0, 9.0, 4.0, 1.0, 7.0, 5.0])
    assert rd._reverse_arrangements(values) == 14


def test_example_4_4_accepted_at_5_percent() -> None:
    res = ph.trend_test(EXAMPLE_4_4)
    assert res.statistic == 86
    assert res.bounds == (64, 125)
    assert res.trend_free
    assert res.method == "reverse_arrangements"
    assert res.n == 20
    # Eq. (4.54)-(4.55): mean 95, variance 237.5.
    assert res.mean == pytest.approx(95.0)
    assert res.std == pytest.approx(np.sqrt(237.5))
    assert 0.0 < res.p_value <= 1.0


@pytest.mark.parametrize(("n", "bounds"), sorted(TABLE_A6_ALPHA_05.items()))
def test_table_a6_alpha_05_percentage_points(
    n: int, bounds: tuple[int, int]
) -> None:
    assert rd._reverse_arrangement_bounds(n, 0.05) == bounds


def test_monotonic_sequences_are_rejected() -> None:
    increasing = np.arange(20.0)
    res = ph.trend_test(increasing)
    assert res.statistic == 0  # no pair with x_i > x_j
    assert not res.trend_free
    assert res.p_value < 1e-9
    decreasing = increasing[::-1]
    res = ph.trend_test(decreasing)
    assert res.statistic == 190  # all N(N-1)/2 pairs reversed
    assert not res.trend_free


def test_mahonian_distribution_is_normalized_and_symmetric() -> None:
    cdf = rd._mahonian_cdf(10)
    assert cdf.size == 46  # 0 .. N(N-1)/2 inversions
    assert cdf[-1] == pytest.approx(1.0, abs=1e-12)
    pmf = np.diff(np.concatenate(([0.0], cdf)))
    np.testing.assert_allclose(pmf, pmf[::-1], atol=1e-12)


def test_large_n_p_value_uses_normal_approximation() -> None:
    rng = np.random.default_rng(11)
    values = rng.standard_normal(150)  # beyond the exact-distribution limit
    res = ph.trend_test(values)
    assert 0.0 < res.p_value <= 1.0
    assert res.trend_free


# ---------------------------------------------------------------------------
# Runs test (Wald & Wolfowitz exact distribution)
# ---------------------------------------------------------------------------


def test_runs_classical_critical_values_n20() -> None:
    # Exact conditional distribution, n1 = n2 = 10, alpha = 0.05: the
    # classical two-sided critical values (reject at <= 6 or >= 16 runs).
    assert rd._runs_bounds(10, 10, 0.05) == (6, 15)


def test_runs_moments_and_pmf_normalization() -> None:
    mean, std = rd._runs_moments(10, 10)
    assert mean == pytest.approx(11.0)
    assert std == pytest.approx(2.1764287503, abs=1e-9)
    pmf = rd._runs_pmf(10, 10)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
    assert pmf[:2] == pytest.approx((0.0, 0.0))


def test_runs_test_rejects_alternation_and_trend() -> None:
    alternating = np.tile([1.0, -1.0], 10)  # r = 20: far too many runs
    res = ph.trend_test(alternating, method="runs")
    assert res.statistic == 20
    assert not res.trend_free
    trending = np.arange(20.0)  # r = 2: one run below, one above the median
    res = ph.trend_test(trending, method="runs")
    assert res.statistic == 2
    assert not res.trend_free
    assert res.p_value < 1e-4


def test_runs_test_accepts_random_sequence() -> None:
    rng = np.random.default_rng(3)
    res = ph.trend_test(rng.standard_normal(40), method="runs")
    assert res.trend_free
    assert res.bounds[0] < res.statistic <= res.bounds[1]


def test_runs_test_discards_median_ties() -> None:
    rng = np.random.default_rng(4)
    values = rng.standard_normal(21)
    values[5] = np.median(values)  # odd length: one exact tie
    res = ph.trend_test(values, method="runs")
    assert res.n == 20


def test_trend_test_validation() -> None:
    short = [1.0] * 9
    with pytest.raises(ValueError, match="at least 10"):
        ph.trend_test(short)
    two_dim = np.ones((5, 4))
    with pytest.raises(ValueError, match="one-dimensional"):
        ph.trend_test(two_dim)
    with_nan = np.r_[np.arange(19.0), np.nan]
    with pytest.raises(ValueError, match="finite"):
        ph.trend_test(with_nan)
    good = list(range(12))
    with pytest.raises(ValueError, match="method"):
        ph.trend_test(good, method="chi2")
    with pytest.raises(ValueError, match="alpha"):
        ph.trend_test(good, alpha=1.5)
    constant = np.ones(12)
    with pytest.raises(ValueError, match="distinct from the median"):
        ph.trend_test(constant, method="runs")


# ---------------------------------------------------------------------------
# Stationarity test (B&P Sec. 10.3.1.1)
# ---------------------------------------------------------------------------

FS = 8192.0


def test_stationary_noise_is_accepted() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1 << 16)
    res = ph.stationarity_test(x, FS)
    assert res.stationary
    assert res.n_segments == 20
    assert res.statistic == "mean_square"
    assert res.bounds == (64, 125)
    assert res.segment_values.size == 20
    # Segment centres: half a segment in, one segment apart.
    assert res.segment_times[0] == pytest.approx(res.segment_duration / 2)
    assert np.diff(res.segment_times) == pytest.approx(res.segment_duration)


def test_gain_ramp_is_rejected_like_example_10_3() -> None:
    # A slow +20 % gain increase, the scenario of B&P Example 10.3.
    rng = np.random.default_rng(42)
    n = 1 << 16
    x = rng.standard_normal(n) * np.linspace(1.0, 1.2, n)
    res = ph.stationarity_test(x, FS)
    assert not res.stationary
    assert res.count < res.bounds[0]  # upward trend depresses A
    runs = ph.stationarity_test(x, FS, method="runs")
    assert not runs.stationary


def test_segment_statistics_are_consistent() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal(4000)
    ms = ph.stationarity_test(x, FS, n_segments=10, statistic="mean_square")
    rms = ph.stationarity_test(x, FS, n_segments=10, statistic="rms")
    np.testing.assert_allclose(
        rms.segment_values, np.sqrt(ms.segment_values), rtol=1e-12
    )
    mean = ph.stationarity_test(x, FS, n_segments=10, statistic="mean")
    var = ph.stationarity_test(x, FS, n_segments=10, statistic="variance")
    np.testing.assert_allclose(
        var.segment_values,
        ms.segment_values - mean.segment_values**2,
        rtol=1e-9, atol=1e-12,
    )


def test_stationarity_trailing_samples_are_discarded() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal(1013)  # 1013 = 10 * 101 + 3
    res = ph.stationarity_test(x, FS, n_segments=10)
    assert res.segment_duration == pytest.approx(101 / FS)


def test_stationarity_validation() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(2048)
    with pytest.raises(ValueError, match="statistic"):
        ph.stationarity_test(x, FS, statistic="kurtosis")
    with pytest.raises(ValueError, match="n_segments"):
        ph.stationarity_test(x, FS, n_segments=5)
    with pytest.raises(ValueError, match="n_segments"):
        ph.stationarity_test(x, FS, n_segments=4096)
    with pytest.raises(ValueError, match="fs"):
        ph.stationarity_test(x, 0.0)


# ---------------------------------------------------------------------------
# Rice level crossings (B&P Sec. 5.5.1)
# ---------------------------------------------------------------------------


def _bandlimited_gaussian(
    seed: int, fs: float, n: int, f1: float, f2: float
) -> np.ndarray:
    """Exactly bandlimited unit-variance Gaussian noise (FFT synthesis)."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(
        freqs.size
    )
    spec[(freqs < f1) | (freqs > f2)] = 0.0
    x = np.fft.irfft(spec, n)
    return np.asarray(x / np.std(x))


def test_zero_crossing_rate_matches_rice_for_bandpass_noise() -> None:
    # B&P Example 5.13: bandwidth-limited Gaussian white noise has
    # N0 = 2 sqrt(fc^2 + B^2/12). Statistical tolerance: over T = 25.6 s
    # the record carries about N0*T = 51500 crossings; even under a
    # (pessimistic) Poisson dispersion the count's relative sd is
    # 1/sqrt(N0*T) = 0.44 %, and the measured seed-to-seed sd of this
    # estimator is 0.15 % (bandpass crossings are more regular than
    # Poisson), so 1 % is a > 6 sigma margin - and the seed is fixed.
    fs, n = 20480.0, 1 << 19
    fc, half_band = 1000.0, 200.0
    x = _bandlimited_gaussian(0, fs, n, fc - half_band, fc + half_band)
    res = ph.level_crossing_rate(x, fs)
    n0 = 2.0 * np.sqrt(fc**2 + (2 * half_band) ** 2 / 12.0)
    assert res.zero_crossing_rate == pytest.approx(n0, rel=1e-2)
    assert res.zero_crossing_rate_rice == pytest.approx(n0, rel=1e-2)
    assert res.apparent_frequency == pytest.approx(n0 / 2.0, rel=1e-2)
    # Level dependence, Eq. (5.196): at a = 1 sigma the rate drops by
    # exp(-1/2). The 1-sigma crossing count is about as large as the
    # zero-crossing count, so the same statistical margin applies.
    res_1s = ph.level_crossing_rate(x, fs, levels=[res.sigma])
    assert res_1s.rates[0] == pytest.approx(n0 * np.exp(-0.5), rel=2e-2)
    assert res_1s.rice_rates[0] == pytest.approx(n0 * np.exp(-0.5), rel=2e-2)


def test_zero_crossing_rate_of_lowpass_noise_example_5_12() -> None:
    # B&P Example 5.12: low-pass white noise cutting off at B has
    # N0 = 2 B / sqrt(3), an apparent frequency of 0.577 B.
    fs, n = 20480.0, 1 << 19
    band = 2000.0
    x = _bandlimited_gaussian(1, fs, n, 0.0, band)
    res = ph.level_crossing_rate(x, fs)
    assert res.zero_crossing_rate == pytest.approx(
        2.0 * band / np.sqrt(3.0), rel=1e-2
    )


def test_sine_crosses_zero_at_twice_its_frequency() -> None:
    # A 60 Hz sine has 120 zeros per second (B&P Sec. 5.5.1).
    fs = 8192.0
    t = np.arange(1 << 16) / fs
    x = np.sin(2.0 * np.pi * 60.0 * t)
    res = ph.level_crossing_rate(x, fs)
    assert res.zero_crossing_rate == pytest.approx(120.0, rel=1e-3)
    assert res.apparent_frequency == pytest.approx(60.0, rel=1e-3)


def test_level_crossing_default_levels_and_validation() -> None:
    rng = np.random.default_rng(8)
    x = rng.standard_normal(4096)
    res = ph.level_crossing_rate(x, FS)
    assert res.levels.size == 13
    assert res.levels[0] == pytest.approx(-3.0 * res.sigma)
    assert res.levels[-1] == pytest.approx(3.0 * res.sigma)
    constant = np.ones(4096)
    with pytest.raises(ValueError, match="constant"):
        ph.level_crossing_rate(constant, FS)
    empty_levels: list[float] = []
    with pytest.raises(ValueError, match="levels"):
        ph.level_crossing_rate(x, FS, levels=empty_levels)
    nan_levels = [0.0, np.nan]
    with pytest.raises(ValueError, match="finite"):
        ph.level_crossing_rate(x, FS, levels=nan_levels)


# ---------------------------------------------------------------------------
# Rice peak statistics (B&P Secs. 5.5.2-5.5.4)
# ---------------------------------------------------------------------------


def test_narrowband_peaks_are_rayleigh_example_5_14() -> None:
    fs, n = 20480.0, 1 << 19
    x = _bandlimited_gaussian(2, fs, n, 950.0, 1050.0)
    res = ph.peak_statistics(x, fs)
    assert res.irregularity_factor > 0.99
    # Example 5.14: Prob[peak > 4 sigma] = exp(-8) = 0.00033.
    assert res.peak_exceedance(4.0)[0] == pytest.approx(np.exp(-8.0), rel=1e-2)
    # One maximum per zero-crossing cycle: M matches N0 / 2.
    assert res.peak_rate == pytest.approx(
        res.zero_crossing_rate_rice / 2.0, rel=2e-2
    )


def test_wideband_irregularity_factor_of_lowpass_noise() -> None:
    # Ideal low-pass spectrum: m0 = B, m2 = B^3/3, m4 = B^5/5, so
    # M = B sqrt(3/5) (Eq. (5.211)) and r = N0/(2M) = sqrt(5)/3 = 0.745.
    fs, n = 20480.0, 1 << 19
    band = 2000.0
    x = _bandlimited_gaussian(3, fs, n, 0.0, band)
    res = ph.peak_statistics(x, fs)
    assert res.peak_rate_rice == pytest.approx(
        band * np.sqrt(3.0 / 5.0), rel=1e-2
    )
    assert res.peak_rate == pytest.approx(band * np.sqrt(3.0 / 5.0), rel=2e-2)
    assert res.irregularity_factor == pytest.approx(
        np.sqrt(5.0) / 3.0, rel=1e-2
    )


def test_peak_exceedance_limits_and_density_consistency() -> None:
    z = np.linspace(-4.0, 6.0, 2001)
    # r = 0: the standardized Gaussian (Eq. (5.221)); half the peaks
    # exceed zero.
    gaussian = rd._rice_peak_exceedance(np.zeros(1), 0.0)
    assert gaussian[0] == pytest.approx(0.5, abs=1e-12)
    # r = 1: exactly Rayleigh (Eq. (5.222)).
    rayleigh = rd._rice_peak_exceedance(np.array([2.0]), 1.0)
    assert rayleigh[0] == pytest.approx(np.exp(-2.0), abs=1e-12)
    for r in (0.0, 0.3, 0.745, 0.95, 1.0):
        density = rd._rice_peak_density(z, r)
        assert np.trapezoid(density, z) == pytest.approx(1.0, abs=1e-4)
        # w(z) is minus the derivative of the exceedance (Eq. (5.258)).
        # Skip the grid point at z = 0, where the Rayleigh limit's density
        # has its slope kink and the central difference straddles it.
        exceedance = rd._rice_peak_exceedance(z, r)
        gradient = -np.gradient(exceedance, z)
        keep = np.abs(z) > 2.0 * (z[1] - z[0])
        keep[:10] = False
        keep[-10:] = False
        np.testing.assert_allclose(gradient[keep], density[keep], atol=2e-4)


def test_peak_statistics_validation_and_fields() -> None:
    rng = np.random.default_rng(9)
    x = rng.standard_normal(1 << 14)
    res = ph.peak_statistics(x, FS)
    assert 0.0 < res.irregularity_factor <= 1.0
    assert res.peak_values.size > 0
    assert np.all(np.diff(res.peak_values) >= 0.0)  # sorted
    assert res.duration == pytest.approx((x.size - 1) / FS)
    constant = np.zeros(4096)
    with pytest.raises(ValueError, match="constant"):
        ph.peak_statistics(constant, FS)


# ---------------------------------------------------------------------------
# Plots (English default; ES parity lives in test_metrology_plot_i18n.py)
# ---------------------------------------------------------------------------


def test_trend_test_plot_renders_and_returns_axes() -> None:
    res = ph.trend_test(EXAMPLE_4_4)
    ax = res.plot()
    assert "Trend test" in ax.get_title()
    assert ax.get_xlabel() == "Sample index"
    legend = ax.get_legend().get_texts()[0].get_text()
    assert "Reverse arrangements $A$ = 86" in legend
    assert "no trend" in legend
    plt.close("all")
    rng = np.random.default_rng(3)
    runs = ph.trend_test(rng.standard_normal(40), method="runs")
    ax = runs.plot()
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Runs $r$ =" in t for t in legend_texts)
    assert any("Sequence median" in t for t in legend_texts)
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")
    plt.close("all")


def test_runs_plot_draws_original_classification_median() -> None:
    # Tied data: the median of the original sequence (10) is discarded from
    # the runs test, and the median of what remains (100) is different. The
    # reference line must be the classification median the test used, not a
    # median recomputed on the filtered TrendTestResult.values.
    values = np.array([0.0] * 5 + [10.0] * 21 + [100.0] * 15)
    np.random.default_rng(0).shuffle(values)
    res = ph.trend_test(values, method="runs")
    assert res.median == 10.0
    assert float(np.median(res.values)) == 100.0  # filtered median differs
    ax = res.plot()
    median_lines = [
        ln for ln in ax.get_lines() if ln.get_label() == "Sequence median"
    ]
    assert len(median_lines) == 1
    assert np.allclose(median_lines[0].get_ydata(), 10.0)
    plt.close("all")


def test_plots_render_and_return_axes() -> None:
    rng = np.random.default_rng(10)
    n = 1 << 14
    x = rng.standard_normal(n) * np.linspace(1.0, 1.3, n)
    st = ph.stationarity_test(x, FS)
    ax = st.plot()
    assert "Stationarity test" in ax.get_title()
    assert "Reverse arrangements" in ax.get_legend().get_texts()[0].get_text()
    plt.close("all")
    runs = ph.stationarity_test(x, FS, method="runs")
    ax = runs.plot()
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Runs $r$ =" in t for t in legend_texts)
    assert any("Sequence median" in t for t in legend_texts)
    plt.close("all")
    noise = rng.standard_normal(n)
    ax = ph.level_crossing_rate(noise, FS).plot()
    assert "Level-crossing rate" in ax.get_title()
    assert ax.get_yscale() == "log"
    plt.close("all")
    ax = ph.peak_statistics(noise, FS).plot()
    assert "Peak-height distribution" in ax.get_title()
    plt.close("all")


def test_peak_plot_without_maxima_raises() -> None:
    ramp_result = ph.peak_statistics(
        np.linspace(0.0, 1.0, 4096) ** 2, FS
    )
    assert ramp_result.peak_values.size == 0
    with pytest.raises(ValueError, match="maxima"):
        ramp_result.plot()
    plt.close("all")

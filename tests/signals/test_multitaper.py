#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the Thomson multitaper estimator (Percival & Walden 1993, Ch. 7).

Multitapering is a different bargain from Welch averaging: instead of splitting
the record into segments and averaging their periodograms, it tapers the *whole*
record with K orthogonal Slepian sequences and averages the K eigenspectra. The
oracles follow that construction. The tapers themselves are pinned against
P&W's published concentration tables, Table 382 in quadruple precision and the
Section 7.1 pattern for NW = 4, because every later property is a statement
about lambda_k. Then come the properties: the calibrated white-noise level
sigma^2/(fs/2) of Section 7.2, the variance reduction to 1/K of a single taper
(Section 7.3 item [2]), the adaptive weights tending to uniform for white noise
(Eq. 368a) with nu(f) -> 2K (Eq. 370b), and Parseval in expectation (Comments to
Section 7.4, item [1]). The Monte Carlo bounds are stated as sigma multiples in
each docstring so a tightened tolerance is a deliberate act.

Welch's estimator, the cross-spectrum and the fractional-octave smoother are in
``test_spectra.py``; the two estimators are cross-checked there for calibration
on the same record.
"""

from __future__ import annotations

import dataclasses

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phonometry as ph

FS = 8192.0
N = 32768


def _white(seed: int, rms: float = 1.0, n: int = N) -> np.ndarray:
    return ph.signals.noise_signal(FS, n / FS, color="white", rms=rms, seed=seed)


# ---------------------------------------------------------------------------
# Thomson multitaper estimator (Percival & Walden 1993, Ch. 7)
# ---------------------------------------------------------------------------

#: Percival & Walden (1993), Table 382: the first 15 eigenvalues
#: lambda_k(31, 8/31), computed in quadruple precision. The dpss tapers of
#: scipy must reproduce this published table to double precision.
_PW_TABLE_382 = [
    0.999999999999999999999990704835383,
    0.999999999999999999996662306749001,
    0.999999999999999999432644053636964,
    0.999999999999999939295977113220309,
    0.999999999999995415079273501044881,
    0.999999999999740167030968849737293,
    0.999999999988537646774846704177484,
    0.999999999597048040415404681655238,
    0.999999988540977691075882016331753,
    0.999999734039703946871310445020549,
    0.999994943151896744119676074049113,
    0.999921365649436730155978193674391,
    0.999009548010700813778608335083965,
    0.990177947707393817597071353567426,
    0.929438220819848051673617819974137,
]

#: Percival & Walden (1993), Section 7.1: eigenvalues lambda_k(1024, 4/1024)
#: for NW = 4, printed to the stated digits.
_PW_NW4_N1024 = [
    0.9999999997,
    0.99999997,
    0.9999988,
    0.99997,
    0.9994,
    0.993,
    0.94,
    0.70,
]


def test_dpss_eigenvalues_match_pw_table_382() -> None:
    """Slepian-taper concentrations against P&W's quadruple-precision table."""
    from scipy.signal.windows import dpss

    _, ratios = dpss(31, 8.0, Kmax=15, return_ratios=True)
    assert np.allclose(ratios, _PW_TABLE_382, rtol=0.0, atol=1e-12)


def test_dpss_eigenvalues_match_pw_section_7_1_pattern() -> None:
    """NW = 4 concentrations reproduce P&W's printed near-unity pattern."""
    from scipy.signal.windows import dpss

    _, ratios = dpss(1024, 4.0, Kmax=8, return_ratios=True)
    for computed, printed in zip(ratios, _PW_NW4_N1024, strict=True):
        # Match to the precision P&W printed (half a unit in the last place).
        decimals = len(str(printed).split(".")[1])
        assert computed == pytest.approx(printed, abs=0.5 * 10.0**-decimals)
    assert (ratios == np.sort(ratios)[::-1]).all()


def test_multitaper_result_eigenvalues_are_the_dpss_ratios() -> None:
    res = ph.signals.multitaper_psd(_white(30, n=1024), FS, time_half_bandwidth=4.0)
    assert res.n_tapers == 7
    # P&W print lambda_6 as 0.94: match to half a unit in the last place.
    assert np.allclose(res.eigenvalues, _PW_NW4_N1024[:7], atol=5e-3)


def test_multitaper_white_noise_density_level() -> None:
    """White noise: mean density = sigma^2/(fs/2), one-sided (P&W Sec. 7.2).

    With 2K = 14 chi-square degrees of freedom per bin and about
    N/(4*NW) = 512 independent resolution bands averaged, the relative
    standard error of the mean is below 0.6 %; 3 % is a 5-sigma bound.
    """
    x = _white(31, rms=2.0, n=8192)
    res = ph.signals.multitaper_psd(x, FS)
    expected = 4.0 / (FS / 2.0)
    assert float(np.mean(res.psd[1:-1])) == pytest.approx(expected, rel=0.03)


def test_multitaper_matches_welch_on_the_same_record() -> None:
    """Both estimators are calibrated: same broadband level, either scaling."""
    x = _white(32, n=8192)
    mt = ph.signals.multitaper_psd(x, FS)
    welch = ph.signals.power_spectral_density(x, FS, nperseg=1024)
    assert float(np.mean(mt.psd[1:-1])) == pytest.approx(
        float(np.mean(welch.psd[1:-1])), rel=0.05
    )


def test_multitaper_density_integrates_to_signal_power_parseval() -> None:
    """Unity-weight multitapering preserves Parseval in expectation

    (P&W Comments to Section 7.4, item [1]); with near-unity eigenvalues
    the eigenvalue-weighted estimator integrates to the signal power to a
    fraction of a percent on any single record.
    """
    x = _white(33, n=4096)
    res = ph.signals.multitaper_psd(x, FS, adaptive=False)
    df = float(res.frequencies[1] - res.frequencies[0])
    assert float(np.sum(res.psd) * df) == pytest.approx(float(np.mean(x**2)), rel=5e-3)


def test_multitaper_tone_band_power_recovers_amplitude() -> None:
    """A sinusoid's power A^2/2 concentrates in the 2W resolution band."""
    n = 4096
    t = np.arange(n) / FS
    amp = 3.0
    x = amp * np.sin(2.0 * np.pi * 1024.0 * t)  # exact bin (1024 = 512*df)
    res = ph.signals.multitaper_psd(x, FS, n_tapers=5, adaptive=False)
    df = float(res.frequencies[1] - res.frequencies[0])
    half_w = res.resolution_bandwidth / 2.0
    band = np.abs(res.frequencies - 1024.0) <= 1.5 * half_w
    # The captured fraction is bounded below by the mean concentration
    # lambda_k > 0.999 of the K = 5 tapers.
    assert float(np.sum(res.psd[band]) * df) == pytest.approx(amp**2 / 2.0, rel=5e-3)


@pytest.mark.parametrize("adaptive", [False, True])
def test_multitaper_spectrum_scaling_reads_tone_peak(adaptive: bool) -> None:
    """'spectrum' scaling reads A^2/2 at the tone bin, like Welch."""
    n = 4096
    t = np.arange(n) / FS
    amp = 3.0
    x = amp * np.sin(2.0 * np.pi * 1024.0 * t)
    res = ph.signals.multitaper_psd(x, FS, scaling="spectrum", adaptive=adaptive)
    peak = int(np.argmax(res.psd))
    assert res.frequencies[peak] == pytest.approx(1024.0)
    assert float(res.psd[peak]) == pytest.approx(amp**2 / 2.0, rel=1e-4)


def test_multitaper_variance_is_one_kth_of_single_taper() -> None:
    """var{S_mt} ~ S^2/K (P&W Section 7.3 item [2]), seeded Monte Carlo.

    Across 120 independent white records the per-bin relative variance of
    the K-taper estimate must be ~1/K of the single-taper (K = 1) one.
    With 120 records the chi-squared spread of a variance estimate is
    about 13 % (1 sigma); 25 % is a comfortable 2-sigma tolerance.
    """
    k = 7
    est_mt, est_one = [], []
    for seed in range(120):
        x = _white(600 + seed, n=2048)
        est_mt.append(
            ph.signals.multitaper_psd(x, FS, n_tapers=k, adaptive=False).psd[50:900]
        )
        est_one.append(
            ph.signals.multitaper_psd(x, FS, n_tapers=1, adaptive=False).psd[50:900]
        )
    rel_var_mt = float(np.mean((np.std(est_mt, axis=0) / np.mean(est_mt, axis=0)) ** 2))
    rel_var_one = float(
        np.mean((np.std(est_one, axis=0) / np.mean(est_one, axis=0)) ** 2)
    )
    assert rel_var_one == pytest.approx(1.0, rel=0.25)
    assert rel_var_mt / rel_var_one == pytest.approx(1.0 / k, rel=0.25)


def test_multitaper_adaptive_weights_uniform_for_white_noise() -> None:
    """b_k(f) -> 1 for white noise (P&W Eq. 368a), so weights -> lambda-flat.

    The normalized weights then equal lambda_k/sum(lambda_j) ~ 1/K within
    the departure of the concentrations from unity plus the estimation
    jitter, both far below 0.01 for K = 5, NW = 4.
    """
    res = ph.signals.multitaper_psd(_white(34, n=4096), FS, n_tapers=5)
    mean_weights = np.mean(res.weights[:, 1:-1], axis=1)
    assert np.max(np.abs(mean_weights - 0.2)) < 0.01
    lam_flat = res.eigenvalues / float(np.sum(res.eigenvalues))
    assert np.allclose(mean_weights, lam_flat, atol=2e-3)


def test_multitaper_dof_is_2k_for_white_noise() -> None:
    """nu(f) ~ 2K where the adaptive weights are uniform (P&W Eq. 370b)."""
    res = ph.signals.multitaper_psd(_white(35, n=4096), FS)
    interior = res.degrees_of_freedom[1:-1]
    assert float(np.mean(interior)) == pytest.approx(2.0 * res.n_tapers, rel=0.02)
    assert np.all(interior <= 2.0 * res.n_tapers + 1e-9)
    # DC (and Nyquist for even n) carries a single real component: half nu.
    assert float(res.degrees_of_freedom[0]) < 0.75 * float(np.mean(interior))
    assert float(res.degrees_of_freedom[-1]) < 0.75 * float(np.mean(interior))


def test_multitaper_chi2_interval_covers_true_density() -> None:
    """Monte Carlo coverage of the chi-square interval at ~95 %."""
    true_psd = 1.0 / (FS / 2.0)
    hits, total = 0, 0
    for seed in range(150):
        res = ph.signals.multitaper_psd(_white(800 + seed, n=2048), FS)
        for b in (100, 400, 800):
            hits += int(res.ci_lower[b] <= true_psd <= res.ci_upper[b])
            total += 1
    assert hits / total == pytest.approx(0.95, abs=0.03)


def test_multitaper_adaptive_dof_drops_where_leakage_would_bias() -> None:
    """Adaptive weights spend dof for leakage protection in weak bands.

    For a spectrum with high dynamic range (a strong low-frequency tone
    over faint noise), the adaptive nu(f) in the faint region must fall
    below the full 2K, and the weights there must be non-uniform.
    """
    n = 4096
    t = np.arange(n) / FS
    rng_x = ph.signals.noise_signal(FS, n / FS, color="white", rms=1e-3, seed=99)
    x = 50.0 * np.sin(2.0 * np.pi * 200.0 * t) + rng_x
    res = ph.signals.multitaper_psd(x, FS)
    faint = (res.frequencies > 2000.0) & (res.frequencies < 3800.0)
    assert float(np.mean(res.degrees_of_freedom[faint])) < 1.9 * res.n_tapers


def test_multitaper_defaults_and_result_fields() -> None:
    res = ph.signals.multitaper_psd(_white(36, n=1024), FS)
    assert res.n_tapers == 7  # 2*NW - 1 for the default NW = 4
    assert res.time_half_bandwidth == 4.0
    assert res.resolution_bandwidth == pytest.approx(2.0 * 4.0 * FS / 1024.0)
    assert res.adaptive is True
    assert res.scaling == "density"
    assert res.weights.shape == (7, res.frequencies.size)
    assert np.allclose(np.sum(res.weights, axis=0), 1.0)
    assert res.random_error.shape == res.degrees_of_freedom.shape
    assert np.allclose(res.random_error, np.sqrt(2.0 / res.degrees_of_freedom))
    with pytest.raises(AttributeError, match=r"cannot assign to field 'psd'"):
        res.psd = res.psd * 2.0  # type: ignore[misc]


def test_multitaper_rejects_invalid_inputs() -> None:
    x = _white(37, n=1024)
    two_dimensional = np.zeros((4, 256))
    with pytest.raises(ValueError, match=r"'x' must be one-dimensional"):
        ph.signals.multitaper_psd(two_dimensional, FS)
    with pytest.raises(ValueError, match=r"Signal too short for a spectral estimate"):
        ph.signals.multitaper_psd(x[:16], FS)
    with pytest.raises(ValueError, match=r"'fs' must be a positive, finite number"):
        ph.signals.multitaper_psd(x, -1.0)
    with pytest.raises(ValueError, match=r"'time_half_bandwidth' must be in"):
        ph.signals.multitaper_psd(x, FS, time_half_bandwidth=0.5)
    # One sentence covers both ends of the range, so the rejected value is the
    # only thing that says which end this call was on.
    with pytest.raises(
        ValueError,
        match=r"'n_tapers' must be between 1 and the Shannon number.*\(got 9\)",
    ):
        ph.signals.multitaper_psd(x, FS, n_tapers=9)
    with pytest.raises(
        ValueError,
        match=r"'n_tapers' must be between 1 and the Shannon number.*\(got 0\)",
    ):
        ph.signals.multitaper_psd(x, FS, n_tapers=0)
    silence = np.zeros(1024)
    with pytest.raises(ValueError, match=r"'x' must not be identically zero"):
        ph.signals.multitaper_psd(silence, FS)
    with pytest.raises(ValueError, match=r"'scaling' must be"):
        ph.signals.multitaper_psd(x, FS, scaling="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"'confidence' must be in"):
        ph.signals.multitaper_psd(x, FS, confidence=1.5)


def test_multitaper_columns_must_share_the_frequency_axis() -> None:
    """A column off the rfft grid is refused when built, not when drawn.

    The plot masks ``frequencies`` for the positive bins and applies that
    mask to ``psd``, ``ci_lower`` and ``ci_upper``, so those three surface
    only as numpy's complaint about a boolean index and two axis sizes,
    naming neither the field nor the one that is wrong. The rest is silent:
    ``degrees_of_freedom`` reaches the legend as the mean of its interior
    bins, so a dof array from another estimate relabels the drawn band with
    a nu-bar it does not have, and nothing reads ``random_error`` or
    ``weights`` at all. An extra axis passes every count, so the ranks are
    pinned too.
    """
    good = ph.signals.multitaper_psd(_white(39, n=1024), FS)
    per_bin = "one value per frequency bin"
    cases = (
        ("frequencies", good.frequencies[:-1], per_bin),
        ("psd", good.psd[:-1], per_bin),
        ("psd", np.append(good.psd, good.psd[-1]), per_bin),
        ("ci_lower", good.ci_lower[:-1], per_bin),
        ("ci_upper", good.ci_upper[:-1], per_bin),
        ("degrees_of_freedom", good.degrees_of_freedom[:-1], per_bin),
        (
            "degrees_of_freedom",
            np.append(good.degrees_of_freedom, 999.0),
            per_bin,
        ),
        ("random_error", good.random_error[:-1], per_bin),
        ("psd", np.column_stack([good.psd] * 2), "must have one axis"),
        ("weights", good.weights[0], "must have 2 axes"),
    )
    for field, value, fragment in cases:
        with pytest.raises(ValueError, match=rf"'{field}'.*{fragment}"):
            dataclasses.replace(good, **{field: value})
    for weights in (
        good.weights[:, :-1],
        np.hstack([good.weights, good.weights[:, :1]]),
    ):
        with pytest.raises(ValueError, match=rf"'weights \(axis 1\)'.*{per_bin}"):
            dataclasses.replace(good, weights=weights)


def test_multitaper_eigenvalues_and_weights_must_share_the_taper_axis() -> None:
    """The per-taper pair is refused when built; no plot ever reads it.

    ``lambda_k`` and the weights it drives are two columns of one table, and
    an extra ``lambda_k`` moves the ``sum_j lambda_j`` that a reader
    normalizes by, so the uniform limit lambda_k/sum_j lambda_j comes out
    wrong for every taper, the ones actually present included. An extra row
    of ``weights`` breaks the sum to unity the field's own definition
    asserts. Neither shows on the figure.
    """
    good = ph.signals.multitaper_psd(_white(40, n=1024), FS)
    per_taper = "one value per taper"
    cases = (
        ("eigenvalues", good.eigenvalues[:-1]),
        ("eigenvalues", np.append(good.eigenvalues, good.eigenvalues[-1])),
        ("weights", good.weights[:-1]),
        ("weights", np.vstack([good.weights, good.weights[:1]])),
    )
    for field, value in cases:
        # Both fields are named whichever one was made wrong, so the count is
        # what identifies it, and the count is the one this test chose.
        with pytest.raises(
            ValueError, match=rf"'{field}' \({len(value)}\).*{per_taper}"
        ):
            dataclasses.replace(good, **{field: value})


def test_multitaper_plot_line_and_confidence_band() -> None:
    res = ph.signals.multitaper_psd(_white(38, n=1024), FS)
    ax = res.plot()
    assert ax.lines, "expected the density line"
    labels = [str(c.get_label()) for c in ax.collections]
    assert any("confidence" in lab for lab in labels)
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext
    plt.close("all")

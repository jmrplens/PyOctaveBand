#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the multiple/partial coherence of a MISO system (B&P Ch. 7).

Every oracle is a closed form from Bendat & Piersol, *Random Data* (4th ed.):

* the exact rational conditioned spectra of end-of-chapter Problem 7.2
  (``G22.1 = 4/3``, ``G2y.1 = 4/3``, ``Gyy.1 = 13/3``) and the multiple
  coherence they imply (``0.7``);
* the multiple-coherence signal-to-noise relation ``gamma^2 = SNR/(1+SNR)``
  (Eq. 7.35) for a system with additive output noise of known level;
* the identities for mutually uncorrelated inputs (Eqs. 7.116/7.117): the
  partial coherence of each input equals its ordinary coherence and the
  multiple coherence is their sum;
* the point of the method, shown on correlated inputs: a source that only
  correlates with the true cause carries a large ordinary coherence but a
  partial coherence of (essentially) zero once the true cause is removed;
* the block consistency of the conditioned matrix (Hermitian, positive
  diagonal) and the exact output-power decomposition ``Gyy = sum Gvi + Gnn``.

Statistical tolerances are justified from Bendat & Piersol Section 9.3: the
q-input multiple coherence carries ``nd-(q-1)`` effective averages
(Eq. 9.98), so with several hundred averages the normalized random error is a
few percent.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import signal as sp_signal

import phonometry as ph
from phonometry.signals.miso import _condition, _ordinary_coherences

FS = 8192.0
N = 1 << 19


def _white(seed: int, rms: float = 1.0, n: int = N) -> np.ndarray:
    return ph.noise_signal(FS, n / FS, color="white", rms=rms, seed=seed)


def _fir(x: np.ndarray, taps: list[float]) -> np.ndarray:
    return np.asarray(sp_signal.lfilter(taps, [1.0], x), dtype=np.float64)


def _band(freqs: np.ndarray) -> np.ndarray:
    return (freqs > 200.0) & (freqs < 3500.0)


def _problem_7_2_matrix() -> np.ndarray:
    """Augmented spectral matrix of Bendat & Piersol Problem 7.2 (one bin).

    ``G11=3, G12=1+j, G22=2, Gyy=10, G1y=4+j, G2y=3-j`` (inputs 0, 1; output
    is record 2).
    """
    mat = np.zeros((1, 3, 3), dtype=np.complex128)
    mat[0, 0, 0] = 3.0
    mat[0, 1, 1] = 2.0
    mat[0, 2, 2] = 10.0
    mat[0, 0, 1] = 1.0 + 1.0j
    mat[0, 1, 0] = 1.0 - 1.0j
    mat[0, 0, 2] = 4.0 + 1.0j
    mat[0, 2, 0] = 4.0 - 1.0j
    mat[0, 1, 2] = 3.0 - 1.0j
    mat[0, 2, 1] = 3.0 + 1.0j
    return mat


# ---------------------------------------------------------------------------
# Hand-computable conditioning: Bendat & Piersol Problem 7.2
# ---------------------------------------------------------------------------


def test_problem_7_2_conditioned_spectra_exact() -> None:
    mat = _problem_7_2_matrix()
    partial, coherent, noise = _condition(mat, (0, 1))
    # Coherent output of input 1 (conditioned on input 0): |G2y.1|^2/G22.1
    # = (4/3)^2 / (4/3) = 4/3.  Of input 0: |G1y|^2/G11 = 17/3.
    assert float(coherent[0, 0]) == pytest.approx(17.0 / 3.0, abs=1e-12)
    assert float(coherent[1, 0]) == pytest.approx(4.0 / 3.0, abs=1e-12)
    # Residual output Gyy.2! = Gyy - (Gv1 + Gv2) = 10 - 7 = 3.
    assert float(noise[0]) == pytest.approx(3.0, abs=1e-12)
    # Partial coherence of input 1 (4e total-Gyy denominator): |G2y.1|^2 /
    # (G22.1 * Gyy) = (16/9)/((4/3)*10) = 2/15.
    assert float(partial[1, 0]) == pytest.approx(2.0 / 15.0, abs=1e-12)
    # Input 0 conditioned on nothing: partial == ordinary == 17/30.
    assert float(partial[0, 0]) == pytest.approx(17.0 / 30.0, abs=1e-12)


def test_problem_7_2_ordinary_and_multiple_coherence_exact() -> None:
    mat = _problem_7_2_matrix()
    ordinary = _ordinary_coherences(mat, 2)
    assert float(ordinary[0, 0]) == pytest.approx(17.0 / 30.0, abs=1e-12)
    assert float(ordinary[1, 0]) == pytest.approx(0.5, abs=1e-12)
    partial, _coherent, noise = _condition(mat, (0, 1))
    gyy = mat[0, 2, 2].real
    multiple = (gyy - noise[0]) / gyy
    # gamma^2_{y:x} = 1 - 3/10 = 0.7 = gamma^2_1y + gamma^2_2y.1 (Eq. 7.116).
    assert float(multiple) == pytest.approx(0.7, abs=1e-12)
    assert float(partial.sum(axis=0)[0]) == pytest.approx(0.7, abs=1e-12)


def test_conditioned_matrix_is_hermitian_with_positive_diagonal() -> None:
    x1 = _white(1)
    x2 = 0.6 * x1 + _white(2)
    y = _fir(x1, [1.0, 0.4]) + _fir(x2, [0.3, -0.5]) + _white(3, rms=0.3)
    res = ph.miso_coherence([x1, x2], y, FS, nperseg=2048)
    band = _band(res.frequencies)
    # Conditioned residual noise (a genuine autospectrum) stays non-negative.
    assert np.all(res.noise_psd[band] >= 0.0)
    # Coherent output contributions are non-negative and never exceed Gyy.
    assert np.all(res.coherent_output_spectra[:, band] >= 0.0)
    assert np.all(res.coherent_output_spectra[:, band].sum(axis=0)
                  <= res.output_psd[band] * (1.0 + 1e-9))


# ---------------------------------------------------------------------------
# Multiple coherence: SNR relation (Eq. 7.35)
# ---------------------------------------------------------------------------


def test_multiple_coherence_equals_snr_over_one_plus_snr() -> None:
    # Two independent unit-RMS white inputs summed, plus white output noise of
    # RMS 0.5: every band has Gvv = 2*sigma^2/(fs/2), Gnn = 0.25/(fs/2), so
    # SNR = 2/0.25 = 8 and gamma^2_{y:x} = SNR/(1+SNR) = 8/9.
    x1 = _white(10)
    x2 = _white(11)
    noise = _white(12, rms=0.5)
    y = x1 + x2 + noise
    res = ph.miso_coherence([x1, x2], y, FS, nperseg=1024)
    band = _band(res.frequencies)
    snr = 2.0 / 0.25
    assert float(np.median(res.multiple_coherence[band])) == pytest.approx(
        snr / (1.0 + snr), abs=0.02
    )


def test_multiple_coherence_is_one_for_noiseless_system() -> None:
    x1 = _white(20)
    x2 = _white(21)
    y = _fir(x1, [1.0, 0.5]) + _fir(x2, [0.3, -0.2, 0.1])
    res = ph.miso_coherence([x1, x2], y, FS, nperseg=1024)
    band = _band(res.frequencies)
    # No output noise: multiple coherence is 1 within the estimator bias.
    assert float(np.median(res.multiple_coherence[band])) == pytest.approx(
        1.0, abs=0.01
    )


# ---------------------------------------------------------------------------
# Uncorrelated inputs: partial == ordinary, multiple == sum (Eqs. 7.116/7.117)
# ---------------------------------------------------------------------------


def test_independent_inputs_partial_equals_ordinary() -> None:
    x1 = _white(30)
    x2 = _white(31)
    y = _fir(x1, [1.0, 0.5, -0.3]) + _fir(x2, [0.2, -0.6, 0.4]) + _white(32, rms=0.3)
    res = ph.miso_coherence([x1, x2], y, FS, nperseg=2048)
    band = _band(res.frequencies)
    for i in (0, 1):
        diff = np.abs(res.partial_coherence[i][band]
                      - res.ordinary_coherence[i][band])
        # Both estimates share the estimator bias (Section 9.3); with several
        # hundred averages the identity holds to a few hundredths.
        assert float(np.median(diff)) < 0.02


def test_independent_inputs_multiple_equals_sum_of_ordinary() -> None:
    x1 = _white(40)
    x2 = _white(41)
    y = _fir(x1, [1.0, 0.5]) + _fir(x2, [0.2, -0.6, 0.4]) + _white(42, rms=0.3)
    res = ph.miso_coherence([x1, x2], y, FS, nperseg=2048)
    band = _band(res.frequencies)
    diff = res.multiple_coherence[band] - res.ordinary_coherence[:, band].sum(axis=0)
    # Both sides carry the O(q/nd) coherence bias of Section 9.3 (here nd is a
    # few hundred), so the identity holds to a couple of hundredths.
    assert float(np.median(np.abs(diff))) < 0.02


def test_four_uncorrelated_inputs_general_q_identity() -> None:
    """Four mutually uncorrelated inputs (general q, not the old 2-3 cap).

    Oracle: Bendat & Piersol Eqs. 7.116/7.117. When the ``q`` inputs are
    mutually uncorrelated the conditioning removes nothing, so the partial
    coherence of each input equals its ordinary coherence (Eq. 7.117) and the
    multiple coherence equals the sum of the ordinary coherences (Eq. 7.116),
    bounded by one. The expected relations come from the standard, not from
    the code: four independent white records drive four distinct FIR paths
    into one output with additive measurement noise, so the inputs stay
    mutually uncorrelated by construction and both identities must hold.
    """
    x0 = _white(50)
    x1 = _white(51)
    x2 = _white(52)
    x3 = _white(53)
    y = (
        _fir(x0, [1.0, 0.4, -0.2])
        + _fir(x1, [0.2, -0.6, 0.4])
        + _fir(x2, [0.5, 0.3])
        + _fir(x3, [-0.3, 0.5, 0.1])
        + _white(54, rms=0.6)
    )
    res = ph.miso_coherence([x0, x1, x2, x3], y, FS, nperseg=2048)

    # (a) general q >= 2 no longer raises, and (d) every array carries q = 4.
    assert res.n_inputs == 4
    assert res.ordinary_coherence.shape[0] == 4
    assert res.partial_coherence.shape[0] == 4
    assert res.coherent_output_spectra.shape[0] == 4
    assert res.multiple_coherence.shape == res.frequencies.shape

    band = _band(res.frequencies)
    # (b) per-input partial == ordinary (Eq. 7.117). The extra input widens the
    # Section 9.3 bias slightly over the two-input case, so allow a comfortable
    # few hundredths.
    for i in range(4):
        diff = np.abs(
            res.partial_coherence[i][band] - res.ordinary_coherence[i][band]
        )
        assert float(np.median(diff)) < 0.03
    # (c) multiple == sum of ordinaries, clipped to one (Eq. 7.116).
    expected = np.minimum(res.ordinary_coherence[:, band].sum(axis=0), 1.0)
    diff_multiple = res.multiple_coherence[band] - expected
    assert float(np.median(np.abs(diff_multiple))) < 0.03


# ---------------------------------------------------------------------------
# Correlated inputs: partial coherence removes the shared path (the point)
# ---------------------------------------------------------------------------


def test_correlated_input_ordinary_inflated_but_partial_near_zero() -> None:
    # x2 = 0.8*x1 + independent; output depends ONLY on x1. x2 therefore has a
    # large *ordinary* coherence with y (inherited through the x1 path) but a
    # *partial* coherence of zero once x1 is conditioned out (B&P 7.3).
    x1 = _white(50)
    x2 = 0.8 * x1 + _white(51)
    y = _fir(x1, [1.0, 0.5, -0.3]) + _white(52, rms=0.3)
    res = ph.miso_coherence([x1, x2], y, FS, nperseg=2048)
    band = _band(res.frequencies)
    ordinary_x2 = float(np.median(res.ordinary_coherence[1][band]))
    partial_x2 = float(np.median(res.partial_coherence[1][band]))
    assert ordinary_x2 > 0.2
    assert partial_x2 < 0.01
    assert ordinary_x2 > 20.0 * partial_x2


def test_partial_coherence_recovers_true_conditioned_gain() -> None:
    # y = H1 x1 with x2 = a x1 + u. The conditioned coherent output of x2 is
    # zero, so its share of Gyy vanishes while x1 carries the whole coherent
    # output: Gv1 ~ Gyy - Gnn, Gv2 ~ 0.
    x1 = _white(60)
    x2 = 0.7 * x1 + _white(61)
    y = _fir(x1, [1.0, -0.4, 0.2]) + _white(62, rms=0.2)
    res = ph.miso_coherence([x1, x2], y, FS, nperseg=2048)
    band = _band(res.frequencies)
    share_x1 = res.coherent_output_spectra[0][band]
    share_x2 = res.coherent_output_spectra[1][band]
    assert float(np.median(share_x2 / share_x1)) < 0.02


# ---------------------------------------------------------------------------
# Output-power decomposition and internal consistency
# ---------------------------------------------------------------------------


def test_output_power_decomposition_is_exact() -> None:
    x1 = _white(70)
    x2 = 0.5 * x1 + _white(71)
    x3 = _white(72)
    y = (_fir(x1, [1.0, 0.3]) + _fir(x2, [0.4, -0.5])
         + _fir(x3, [0.2, 0.1]) + _white(73, rms=0.3))
    res = ph.miso_coherence([x1, x2, x3], y, FS, nperseg=2048)
    reconstructed = res.coherent_output_spectra.sum(axis=0) + res.noise_psd
    np.testing.assert_allclose(reconstructed, res.output_psd, rtol=1e-9)


def test_multiple_coherence_equals_sum_of_partials() -> None:
    x1 = _white(80)
    x2 = 0.5 * x1 + _white(81)
    y = _fir(x1, [1.0, 0.3]) + _fir(x2, [0.4, -0.5]) + _white(82, rms=0.3)
    res = ph.miso_coherence([x1, x2], y, FS, nperseg=2048)
    np.testing.assert_allclose(
        res.multiple_coherence, res.partial_coherence.sum(axis=0), atol=1e-12
    )


def test_singular_pivot_conditioned_matrix_keeps_decomposition() -> None:
    # Two perfectly collinear inputs (x2 = x1): after conditioning input 0 the
    # conditioned autospectrum G22.1 is exactly zero, a singular pivot. The
    # power decomposition must still close (sum of contributions + residual =
    # Gyy) with no NaN or inf. G11 = G22 = G12 = 4, G1y = G2y = 4, Gyy = 6
    # (i.e. y = x1 + noise of power 2), so Gv1 = 4, Gv2 = 0, noise = 2.
    mat = np.zeros((1, 3, 3), dtype=np.complex128)
    mat[0, 0, 0] = 4.0
    mat[0, 1, 1] = 4.0
    mat[0, 2, 2] = 6.0
    mat[0, 0, 1] = mat[0, 1, 0] = 4.0
    mat[0, 0, 2] = mat[0, 2, 0] = 4.0
    mat[0, 1, 2] = mat[0, 2, 1] = 4.0
    partial, coherent, noise = _condition(mat, (0, 1))
    assert np.all(np.isfinite(coherent))
    assert np.all(np.isfinite(partial))
    assert float(coherent[0, 0]) == pytest.approx(4.0, abs=1e-12)
    assert float(coherent[1, 0]) == pytest.approx(0.0, abs=1e-12)
    assert float(noise[0]) == pytest.approx(2.0, abs=1e-12)
    reconstructed = coherent.sum(axis=0)[0] + noise[0]
    assert float(reconstructed) == pytest.approx(6.0, abs=1e-12)


def test_near_collinear_inputs_preserve_decomposition() -> None:
    # x2 is x1 filtered plus a tiny independent component, so its conditioned
    # autospectrum is negligible in many bands and dips below the relative
    # pivot floor. The coherent-output accumulation and the Schur update must
    # share that floor, or the decomposition would leak power. This pins the
    # invariant (and finiteness) for a near-singular input matrix.
    x1 = _white(200)
    x2 = _fir(x1, [1.0, 0.2]) + _white(201, rms=1e-4)
    y = _fir(x1, [0.8, -0.3]) + _fir(x2, [0.1, 0.05]) + _white(202, rms=0.2)
    res = ph.miso_coherence([x1, x2], y, FS, nperseg=2048)
    reconstructed = res.coherent_output_spectra.sum(axis=0) + res.noise_psd
    assert np.all(np.isfinite(res.coherent_output_spectra))
    assert np.all(np.isfinite(res.partial_coherence))
    np.testing.assert_allclose(reconstructed, res.output_psd, rtol=1e-9)


def test_ordinary_and_multiple_coherence_are_order_invariant() -> None:
    x1 = _white(90)
    x2 = 0.5 * x1 + _white(91)
    y = _fir(x1, [1.0, 0.3]) + _fir(x2, [0.4, -0.5]) + _white(92, rms=0.3)
    forward = ph.miso_coherence([x1, x2], y, FS, nperseg=2048, order=(0, 1))
    reverse = ph.miso_coherence([x1, x2], y, FS, nperseg=2048, order=(1, 0))
    np.testing.assert_allclose(
        forward.ordinary_coherence, reverse.ordinary_coherence, atol=1e-12
    )
    np.testing.assert_allclose(
        forward.multiple_coherence, reverse.multiple_coherence, atol=1e-12
    )
    # The partial coherences do depend on the order (they are conditioned on
    # different preceding inputs), so the two runs must differ somewhere.
    band = _band(forward.frequencies)
    assert not np.allclose(
        forward.partial_coherence[1][band], reverse.partial_coherence[1][band]
    )


def test_dominant_input_tracks_the_band_with_more_power() -> None:
    # x1 drives the low band (a lowpass path), x2 the high band (a highpass
    # path), so the dominant input flips across the spectrum.
    x1 = _white(100)
    x2 = _white(101)
    lp = sp_signal.butter(4, 500.0, fs=FS, output="sos")
    hp = sp_signal.butter(4, 2000.0, btype="high", fs=FS, output="sos")
    y = (sp_signal.sosfilt(lp, x1) + sp_signal.sosfilt(hp, x2)
         + _white(102, rms=0.05))
    res = ph.miso_coherence([x1, x2], y, FS, nperseg=2048)
    dom = res.dominant_input()
    freqs = res.frequencies
    low = (freqs > 150.0) & (freqs < 400.0)
    high = (freqs > 3000.0) & (freqs < 3800.0)
    assert int(np.round(np.median(dom[low]))) == 0
    assert int(np.round(np.median(dom[high]))) == 1


# ---------------------------------------------------------------------------
# Statistical errors (Section 9.3)
# ---------------------------------------------------------------------------


def test_multiple_coherence_random_error_matches_bendat_piersol() -> None:
    x1 = _white(110)
    x2 = _white(111)
    noise = _white(112, rms=0.5)
    y = x1 + x2 + noise
    res = ph.miso_coherence([x1, x2], y, FS, nperseg=1024)
    band = _band(res.frequencies)
    gamma2 = float(np.median(res.multiple_coherence[band]))
    eff = res.n_averages - (res.n_inputs - 1)
    expected = np.sqrt(2.0) * (1.0 - gamma2) / (np.sqrt(gamma2) * np.sqrt(eff))
    assert float(np.median(res.multiple_coherence_random_error[band])) == (
        pytest.approx(expected, rel=0.05)
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["en", "es"])
def test_plot_returns_two_panels(language: str) -> None:
    x1 = _white(120, n=1 << 17)
    x2 = 0.5 * x1 + _white(121, n=1 << 17)
    y = _fir(x1, [1.0, 0.3]) + _fir(x2, [0.4, -0.5]) + _white(122, rms=0.3, n=1 << 17)
    res = ph.miso_coherence([x1, x2], y, FS, nperseg=1024)
    axes = res.plot(language=language)
    assert np.asarray(axes).size == 2
    plt.close("all")


def test_plot_on_given_axis_draws_spectra_panel() -> None:
    x1 = _white(130, n=1 << 17)
    x2 = _white(131, n=1 << 17)
    y = x1 + x2 + _white(132, rms=0.3, n=1 << 17)
    res = ph.miso_coherence([x1, x2], y, FS, nperseg=1024)
    _fig, ax = plt.subplots()
    returned = res.plot(ax=ax)
    assert returned is ax
    plt.close("all")


# ---------------------------------------------------------------------------
# Input validation (constructed outside the raises blocks)
# ---------------------------------------------------------------------------


def test_rejects_single_input() -> None:
    x1 = _white(140, n=1 << 14)
    y = x1.copy()
    with pytest.raises(ValueError, match="at least two"):
        ph.miso_coherence([x1], y, FS)


def test_rejects_mismatched_length() -> None:
    x1 = _white(170, n=1 << 14)
    x2 = _white(171, n=1 << 13)
    y = _white(172, n=1 << 14)
    with pytest.raises(ValueError, match="same length"):
        ph.miso_coherence([x1, x2], y, FS)


def test_rejects_bad_order() -> None:
    x1 = _white(180, n=1 << 14)
    x2 = _white(181, n=1 << 14)
    y = x1 + x2
    with pytest.raises(ValueError, match="permutation"):
        ph.miso_coherence([x1, x2], y, FS, order=(0, 0))


def test_accepts_2d_input_array() -> None:
    x1 = _white(190, n=1 << 16)
    x2 = _white(191, n=1 << 16)
    y = x1 + x2 + _white(192, rms=0.3, n=1 << 16)
    stacked = np.vstack([x1, x2])
    res = ph.miso_coherence(stacked, y, FS, nperseg=1024)
    assert res.n_inputs == 2

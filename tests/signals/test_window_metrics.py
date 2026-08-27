#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the window figures of merit (Harris 1978).

Clean-room oracles:

* The equivalent noise bandwidth ``n·Σw²/(Σw)²`` has exact closed forms
  for the cosine-sum windows sampled DFT-even: 1 for rectangular, 3/2 for
  Hann (``a₀ = a₁ = 1/2``), 1987/1458 for Hamming (``a₀ = 0.54``,
  ``a₁ = 0.46``) and 1523/882 for Blackman (0.42/0.5/0.08), derived from
  ``ENBW = (a₀² + Σaₖ²/2)/a₀²`` (the cross terms sum to zero over full
  periods).
* The rectangular window's scalloping loss has the exact discrete form
  ``20·lg(N·sin(π/2N))`` (Dirichlet kernel at the half-bin offset), which
  tends to ``20·lg(π/2) = 3.92 dB``; Hann tends to ``-20·lg(8/3π) =
  1.42 dB``.
* Sidelobe levels and main-lobe widths against the classic Harris (1978)
  Table 1 values.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phonometry as ph

N = 1024

#: Exact ENBW closed forms of the cosine-sum windows (DFT-even sampling).
_ENBW_EXACT = [
    ("boxcar", 1.0),
    ("hann", 1.5),
    ("hamming", 1987.0 / 1458.0),  # (2·0.54² + 0.46²) / (2·0.54²)
    ("blackman", 1523.0 / 882.0),  # (2·0.42² + 0.5² + 0.08²) / (2·0.42²)
]


@pytest.mark.parametrize(("window", "expected"), _ENBW_EXACT)
def test_enbw_matches_exact_closed_form(window: str, expected: float) -> None:
    res = ph.signals.window_metrics(window, N)
    assert res.enbw_bins == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("window", "expected"),
    [("boxcar", 1.0), ("hann", 0.5), ("hamming", 0.54)],
)
def test_coherent_gain_exact(window: str, expected: float) -> None:
    # DFT-even cosine sums over full periods leave only the a0 term.
    res = ph.signals.window_metrics(window, N)
    assert res.coherent_gain == pytest.approx(expected, rel=1e-12)


def test_rectangular_scalloping_exact_discrete_form() -> None:
    # |W(1/2)| / |W(0)| = 1 / (N·sin(π/2N)) exactly (Dirichlet kernel).
    res = ph.signals.window_metrics("boxcar", N)
    exact = 20.0 * np.log10(N * np.sin(np.pi / (2.0 * N)))
    assert res.scalloping_loss_db == pytest.approx(exact, rel=1e-9)
    # Large-N limit: 20·lg(π/2) = 3.9224 dB.
    assert res.scalloping_loss_db == pytest.approx(3.9224, abs=1e-3)


def test_hann_scalloping_matches_continuous_limit() -> None:
    # W(1/2)/W(0) -> (2/π)/(1 - 1/4) = 8/3π for the continuous Hann kernel.
    res = ph.signals.window_metrics("hann", N)
    assert res.scalloping_loss_db == pytest.approx(
        -20.0 * np.log10(8.0 / (3.0 * np.pi)), abs=1e-3
    )


@pytest.mark.parametrize(
    ("window", "sidelobe_db", "tol"),
    [
        ("boxcar", -13.3, 0.05),  # Harris Table 1: -13 dB (exact -13.26)
        ("hann", -31.5, 0.05),  # Harris Table 1: -32 dB (exact -31.47)
        ("hamming", -42.7, 0.15),  # Harris Table 1: -43 dB (exact -42.68)
    ],
)
def test_highest_sidelobe_matches_classic_values(
    window: str, sidelobe_db: float, tol: float
) -> None:
    res = ph.signals.window_metrics(window, N)
    assert res.highest_sidelobe_db == pytest.approx(sidelobe_db, abs=tol)


@pytest.mark.parametrize(
    ("window", "width"),
    [("boxcar", 0.89), ("hann", 1.44), ("hamming", 1.30)],
)
def test_mainlobe_3db_width_matches_harris_table(window: str, width: float) -> None:
    res = ph.signals.window_metrics(window, N)
    assert res.mainlobe_width_3db_bins == pytest.approx(width, abs=0.01)


def test_worst_case_processing_loss() -> None:
    # WCPL = scalloping loss + 10·lg(ENBW): 3.92 dB rectangular (no ENBW
    # term), 3.18 dB Hann (Harris Table 1).
    rect = ph.signals.window_metrics("boxcar", N)
    assert rect.worst_case_processing_loss_db == pytest.approx(rect.scalloping_loss_db)
    hann = ph.signals.window_metrics("hann", N)
    assert hann.worst_case_processing_loss_db == pytest.approx(3.18, abs=0.01)


def test_enbw_hz_matches_welch_resolution_bandwidth() -> None:
    # The PSD estimator reports Bₑ = fs·Σw²/(Σw)² per segment; it must be
    # the same number window_metrics reports as ENBW·fs/n.
    fs, nperseg = 48000.0, 2048
    x = ph.signals.noise_signal(fs, 1.0, seed=6)
    psd = ph.signals.power_spectral_density(x, fs, nperseg=nperseg)
    metrics = ph.signals.window_metrics("hann", nperseg)
    assert psd.resolution_bandwidth == pytest.approx(metrics.enbw_hz(fs), rel=1e-12)


def test_parametric_windows_are_accepted() -> None:
    res = ph.signals.window_metrics(("kaiser", 8.6), 512)
    assert res.enbw_bins > 1.5
    assert res.highest_sidelobe_db < -60.0
    assert res.n == 512
    assert res.taps.size == 512


def test_window_metrics_result_is_frozen() -> None:
    res = ph.signals.window_metrics("hann", N)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.enbw_bins = 2.0  # type: ignore[misc]


def test_window_metrics_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Unknown window"):
        ph.signals.window_metrics("not-a-window", N)
    with pytest.raises(ValueError, match=r"'n' must be at least \d+ samples"):
        ph.signals.window_metrics("hann", 8)


def test_window_metrics_rejects_taps_of_another_length() -> None:
    """Taps that disagree with the declared ``n`` are a different window.

    Every figure of merit was computed from ``n`` samples and the waveform
    panel draws the taps against a sample index counted out of ``n``, so a
    mismatch used to stop inside matplotlib about first dimensions, naming
    neither the field nor the result.
    """
    res = ph.signals.window_metrics("hann", N)
    with pytest.raises(
        ValueError, match=r"WindowMetricsResult: 'n' \(1034\), 'taps' \(1024\)"
    ):
        dataclasses.replace(res, n=1034)


def test_window_metrics_plot_two_panels_and_single_axes() -> None:
    res = ph.signals.window_metrics("hann", N)
    axes = res.plot(linewidth=2)
    assert isinstance(axes, np.ndarray)
    assert axes.size == 2
    assert "Harris 1978" in axes[0].get_title()
    assert any(line.get_linewidth() == 2.0 for line in axes[1].lines)
    plt.close("all")

    _fig, ax = plt.subplots()
    out = res.plot(ax=ax, color="red")
    assert out is ax
    red = plt.matplotlib.colors.to_rgba("red")
    assert any(
        plt.matplotlib.colors.to_rgba(line.get_color()) == red for line in ax.lines
    )
    plt.close("all")

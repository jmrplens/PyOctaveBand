#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the deterministic colored-noise generator.

The oracle is the closed-form spectral slope of each color: the PSD follows
``Gxx(f) ∝ f^α`` so a log2-frequency regression of the Welch estimate must
return ``10·lg(2)·α`` dB per octave (0, ±3.01, ±6.02) to within a few
hundredths of a dB, since the frequency-domain shaping is exact bin by bin.
"""

from __future__ import annotations

import numpy as np
import pytest

import phonometry as ph

FS = 48000.0

#: (color, exact PSD slope in dB/octave = 10·lg(2)·α).
_SLOPES = [
    ("white", 0.0),
    ("pink", -10.0 * np.log10(2.0)),
    ("red", -20.0 * np.log10(2.0)),
    ("blue", 10.0 * np.log10(2.0)),
    ("violet", 20.0 * np.log10(2.0)),
]


def _measured_slope(color: str) -> float:
    """Welch-PSD regression slope in dB per octave over 20 Hz - 20 kHz."""
    x = ph.signals.noise_signal(FS, 40.0, color=color, seed=3)  # type: ignore[arg-type]
    res = ph.signals.power_spectral_density(x, FS, nperseg=8192)
    band = (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
    slope = np.polyfit(
        np.log2(res.frequencies[band]), 10.0 * np.log10(res.psd[band]), 1
    )[0]
    return float(slope)


@pytest.mark.parametrize(("color", "expected"), _SLOPES)
def test_spectral_slope_matches_exact_power_law(color: str, expected: float) -> None:
    # Three-decade regression on a 40 s record: the exact frequency-domain
    # shaping pins the slope to within a few hundredths of a dB/octave
    # (residual scatter is the 1/sqrt(nd) random error of the PSD estimate).
    assert _measured_slope(color) == pytest.approx(expected, abs=0.05)


def test_same_seed_is_bit_reproducible() -> None:
    a = ph.signals.noise_signal(FS, 1.0, color="pink", seed=42)
    b = ph.signals.noise_signal(FS, 1.0, color="pink", seed=42)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ() -> None:
    a = ph.signals.noise_signal(FS, 1.0, color="pink", seed=1)
    b = ph.signals.noise_signal(FS, 1.0, color="pink", seed=2)
    assert not np.array_equal(a, b)


def test_rms_is_exact_and_mean_zero() -> None:
    for color in ("white", "pink", "red", "blue", "violet"):
        x = ph.signals.noise_signal(FS, 2.0, color=color, rms=0.25, seed=5)  # type: ignore[arg-type]
        assert float(np.sqrt(np.mean(x * x))) == pytest.approx(0.25, rel=1e-12)
        assert float(np.mean(x)) == pytest.approx(0.0, abs=1e-12)


def test_length_and_dtype() -> None:
    x = ph.signals.noise_signal(FS, 0.5, seed=1)
    assert x.shape == (24000,)
    assert x.dtype == np.float64


def test_white_is_gaussian_like() -> None:
    x = ph.signals.noise_signal(FS, 10.0, color="white", seed=6)
    # Kurtosis of a Gaussian is 3; a crude but effective distribution check.
    kurt = float(np.mean(x**4) / np.mean(x**2) ** 2)
    assert kurt == pytest.approx(3.0, abs=0.1)


def test_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match=r"'fs' must be a positive, finite number"):
        ph.signals.noise_signal(0.0, 1.0)
    with pytest.raises(
        ValueError, match=r"'seconds' must be a positive, finite number"
    ):
        ph.signals.noise_signal(FS, -1.0)
    with pytest.raises(
        ValueError, match=r"'fs'\*'seconds' must give at least .* samples"
    ):
        ph.signals.noise_signal(FS, 1e-5)
    with pytest.raises(ValueError, match="'color'"):
        ph.signals.noise_signal(FS, 1.0, color="brown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'rms'"):
        ph.signals.noise_signal(FS, 1.0, rms=0.0)

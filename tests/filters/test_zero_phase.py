#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for zero-phase (sosfiltfilt) filtering in OctaveFilterBank."""

import numpy as np
import pytest

from phonometry import filters

FS = 48000


def test_zero_phase_no_group_delay() -> None:
    """A pulse filtered zero-phase keeps its energy centered at the pulse time."""
    bank = filters.OctaveFilterBank(fs=FS, fraction=1, limits=[800, 1200])
    x = np.zeros(FS)
    center = FS // 2
    x[center] = 1.0

    _, _, bands_zp = bank.filter(
        x, sigbands=True, zero_phase=True, calculate_level=False
    )
    yb = bands_zp[0]
    centroid = int(np.sum(np.arange(len(yb)) * yb**2) / np.sum(yb**2))
    assert abs(centroid - center) < FS // 1000  # within 1 ms


def test_zero_phase_doubles_attenuation() -> None:
    """filtfilt applies the filter twice: stopband attenuation doubles.

    Measured on the steady-state middle segment: the whole-signal RMS is
    dominated by the filter onset transient, which masks the difference.
    """
    bank = filters.OctaveFilterBank(fs=FS, fraction=1, limits=[800, 1200])
    t = np.arange(FS) / FS
    x = np.sin(2 * np.pi * 4000 * t)  # far out of band

    _, _, bands_fwd = bank.filter(x, sigbands=True, calculate_level=False)
    _, _, bands_zp = bank.filter(
        x, sigbands=True, calculate_level=False, zero_phase=True
    )
    mid = slice(FS // 4, 3 * FS // 4)
    rms_fwd = np.sqrt(np.mean(bands_fwd[0][mid] ** 2))
    rms_zp = np.sqrt(np.mean(bands_zp[0][mid] ** 2))
    assert 20 * np.log10(rms_zp / rms_fwd) < -20


def test_zero_phase_rejects_stateful() -> None:
    bank = filters.OctaveFilterBank(
        fs=FS,
        design=filters.FilterDesign(resample=False),
        block_processing=filters.BlockProcessing(stateful=True),
    )
    silence = np.zeros(FS)
    with pytest.raises(ValueError, match="zero_phase"):
        bank.filter(silence, zero_phase=True)


def test_zero_phase_passband_level_matches() -> None:
    """In-band level must match forward filtering (0 dB passband both ways)."""
    bank = filters.OctaveFilterBank(fs=FS, fraction=1, limits=[800, 1200])
    t = np.arange(FS * 2) / FS
    x = np.sin(2 * np.pi * 1000 * t)
    spl_fwd, _ = bank.filter(x)
    spl_zp, _ = bank.filter(x, zero_phase=True)
    assert spl_zp[0] == pytest.approx(spl_fwd[0], abs=0.1)


def test_zero_phase_broadband_band_narrowing() -> None:
    """Forward-backward filtering narrows the effective passband, lowering the
    measured broadband band level ~0.2-0.3 dB per band (a pure center tone,
    tested above, does not exercise this). Characterizes the documented bias.
    """
    bank = filters.OctaveFilterBank(fs=FS, fraction=1, limits=[100, 8000])
    x = np.random.default_rng(7).standard_normal(FS * 4)
    spl_fwd, _ = bank.filter(x)
    spl_zp, _ = bank.filter(x, zero_phase=True)
    delta = spl_zp - spl_fwd
    # Every band shifts down by a small, bounded amount (never up).
    assert np.all(delta < 0.0)
    assert np.all(delta > -0.5)
    assert -0.35 < float(np.mean(delta)) < -0.15


def test_zero_phase_short_signal_does_not_crash() -> None:
    """Heavily decimated bands can be shorter than sosfiltfilt's default padlen."""
    bank = filters.OctaveFilterBank(fs=FS, fraction=1, limits=[100, 200])
    x = np.random.default_rng(3).standard_normal(4000)  # ~23 samples after decimation
    spl, _freq = bank.filter(x, zero_phase=True)
    assert np.all(np.isfinite(spl))


def test_spectrogram_supports_zero_phase() -> None:
    bank = filters.OctaveFilterBank(fs=FS, fraction=1, limits=[800, 1200])
    x = np.random.default_rng(4).standard_normal(FS)
    levels_zp, _, _times = bank.spectrogram(x, window_time=0.125, zero_phase=True)
    levels_fwd, _, _ = bank.spectrogram(x, window_time=0.125, zero_phase=False)
    assert levels_zp.shape == levels_fwd.shape
    assert np.all(np.isfinite(levels_zp))

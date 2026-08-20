#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for OctaveFilterBank.spectrogram() short-time band levels."""

import numpy as np
import pytest

from phonometry import filters

FS = 48000


def test_spectrogram_shapes_and_times() -> None:
    """Output shapes follow (bands, frames) and times are window centers."""
    bank = filters.OctaveFilterBank(fs=FS, fraction=1, limits=[100, 5000])
    x = np.random.default_rng(0).standard_normal(FS * 2)
    levels, freq, times = bank.spectrogram(x, window_time=0.25, overlap=0.5)

    hop = 0.25 * 0.5
    expected_frames = int((2.0 - 0.25) / hop) + 1
    assert levels.shape == (bank.num_bands, expected_frames)
    assert len(freq) == bank.num_bands
    assert times.shape == (expected_frames,)
    np.testing.assert_allclose(times[1] - times[0], hop, rtol=1e-6)


def test_spectrogram_detects_level_step() -> None:
    """A tone whose amplitude doubles mid-signal shows a ~6 dB step in its band."""
    bank = filters.OctaveFilterBank(fs=FS, fraction=1, limits=[800, 1200])
    t = np.arange(FS * 2) / FS
    amp = np.where(t < 1.0, 0.5, 1.0)
    x = amp * np.sin(2 * np.pi * 1000 * t)
    levels, freq, times = bank.spectrogram(x, window_time=0.125, overlap=0.0)

    band = int(np.argmin(np.abs(np.asarray(freq) - 1000)))
    first_half = levels[band, times < 0.9].mean()
    second_half = levels[band, times > 1.1].mean()
    assert second_half - first_half == pytest.approx(6.02, abs=0.3)


def test_spectrogram_multichannel() -> None:
    bank = filters.OctaveFilterBank(fs=FS, fraction=1, limits=[100, 5000])
    x = np.random.default_rng(1).standard_normal((3, FS))
    levels, _, times = bank.spectrogram(x, window_time=0.125, overlap=0.5)
    assert levels.ndim == 3
    assert levels.shape[0] == 3
    assert levels.shape[1] == bank.num_bands
    assert levels.shape[2] == times.shape[0]


def test_spectrogram_rejects_stateful() -> None:
    bank = filters.OctaveFilterBank(
        fs=FS,
        design=filters.FilterDesign(resample=False),
        block_processing=filters.BlockProcessing(stateful=True),
    )
    silence = np.zeros(FS)
    with pytest.raises(ValueError, match="stateful"):
        bank.spectrogram(silence)


def test_spectrogram_invalid_params_raise() -> None:
    bank = filters.OctaveFilterBank(fs=FS, fraction=1, limits=[100, 5000])
    # the signals are built outside the raises blocks, so each block holds
    # exactly the one call whose exception is under test
    one_second = np.zeros(FS)
    shorter_than_the_window = np.zeros(1000)
    with pytest.raises(ValueError, match="overlap"):
        bank.spectrogram(one_second, overlap=1.0)
    with pytest.raises(ValueError, match="window_time"):
        bank.spectrogram(shorter_than_the_window, window_time=1.0)

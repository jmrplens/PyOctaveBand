#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for :mod:`phonometry.speech.objective_intelligibility` (STOI / ESTOI).

STOI (Taal et al. 2011) and ESTOI (Jensen & Taal 2016) are validated three
ways: exact degenerate cases (a degraded signal equal to the clean one scores
1; an uncorrelated signal scores low), monotonicity with the signal-to-noise
ratio, and an external cross-check against ``pystoi`` (skipped when it is not
installed) which reproduces the authors' MATLAB implementation. ``pystoi`` is
a test-only contrast and never a runtime dependency: the algorithm here is a
clean-room implementation from the two papers.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import itertools

import numpy as np
import pytest

from phonometry import speech
from phonometry.speech import objective_intelligibility as oi

FS = oi.SAMPLE_RATE  # 10 kHz: the internal rate, so the contrast skips resampling.


def _speech_like(seed: int, seconds: float = 3.0) -> np.ndarray:
    """A deterministic speech-like signal: AM-modulated formant-ish tones."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    sig = np.zeros_like(t)
    for f0 in (200.0, 400.0, 700.0, 1100.0, 1800.0, 2600.0):
        depth = 0.5 * (
            1.0
            + np.sin(
                2.0 * np.pi * rng.uniform(2.0, 6.0) * t + rng.uniform(0.0, 2.0 * np.pi)
            )
        )
        sig += depth * np.sin(2.0 * np.pi * f0 * t + rng.uniform(0.0, 2.0 * np.pi))
    return sig


def _add_noise(clean: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(clean.size)
    p_clean = np.sqrt(np.mean(clean**2))
    p_noise = np.sqrt(np.mean(noise**2))
    gain = p_clean / (p_noise * 10.0 ** (snr_db / 20.0))
    return clean + gain * noise


@pytest.mark.parametrize("extended", [False, True])
def test_identical_signals_score_one(extended: bool) -> None:
    # A degraded signal equal to the clean reference is a perfect correlation.
    x = _speech_like(1)
    assert speech.stoi(x, x, FS, extended=extended).value == pytest.approx(
        1.0, abs=1e-9
    )


@pytest.mark.parametrize("extended", [False, True])
def test_uncorrelated_noise_scores_low(extended: bool) -> None:
    # An unrelated signal has no envelope correlation with the clean speech.
    x = _speech_like(1)
    y = _speech_like(999)  # independent, same statistics
    value = speech.stoi(x, y, FS, extended=extended).value
    assert value < 0.3


@pytest.mark.parametrize("extended", [False, True])
def test_monotonic_with_snr(extended: bool) -> None:
    # Higher SNR must not lower the index (monotonic relation with quality).
    x = _speech_like(2)
    values = [
        speech.stoi(x, _add_noise(x, snr, seed=10), FS, extended=extended).value
        for snr in (-15.0, -5.0, 5.0, 15.0, 25.0)
    ]
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(values))
    assert values[-1] > values[0] + 0.2  # a clear spread, not a flat line


def test_scale_invariance() -> None:
    # A global gain on the degraded signal is normalised out (Taal Eq. 3).
    x = _speech_like(3)
    y = _add_noise(x, 5.0, seed=11)
    assert speech.stoi(x, y, FS).value == pytest.approx(
        speech.stoi(x, 7.3 * y, FS).value, abs=1e-9
    )


def test_result_fields_and_plot() -> None:
    x = _speech_like(4)
    y = _add_noise(x, 8.0, seed=12)
    res = speech.stoi(x, y, FS)
    assert res.extended is False
    assert res.sample_rate == FS
    assert res.band_frequencies.shape == (15,)
    assert res.band_scores is not None
    assert res.band_scores.shape == (15,)
    assert res.band_frequencies[0] == pytest.approx(150.0)
    res.plot()  # STOI: per-band bars
    ext = speech.stoi(x, y, FS, extended=True)
    assert ext.band_scores is None
    assert ext.segment_scores.ndim == 1
    ext.plot()  # ESTOI: per-segment line


def test_resampling_path_runs() -> None:
    # A non-10-kHz input is resampled internally; identical inputs still score 1.
    t = np.arange(int(3.0 * 16000)) / 16000
    s = np.sin(2 * np.pi * 300 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    assert speech.stoi(s, s, 16000).value == pytest.approx(1.0, abs=1e-9)


def test_input_validation() -> None:
    x = _speech_like(6)
    shorter = x[:-10]
    two_d = x.reshape(2, -1)
    with_nan = x.copy()
    with_nan[0] = np.nan
    empty = np.array([])

    with pytest.raises(ValueError, match="equal length"):
        speech.stoi(x, shorter, FS)
    with pytest.raises(ValueError, match="1-D"):
        speech.stoi(two_d, two_d, FS)
    with pytest.raises(ValueError, match="positive"):
        speech.stoi(x, x, 0)
    with pytest.raises(ValueError, match="finite"):
        speech.stoi(x, with_nan, FS)
    with pytest.raises(ValueError, match="non-empty"):
        speech.stoi(empty, empty, FS)


def test_too_short_signal_raises() -> None:
    # Under ~0.4 s of active speech there are fewer than 30 frames to segment.
    short = _speech_like(7, seconds=0.2)
    with pytest.raises(ValueError, match="30"):
        speech.stoi(short, short, FS)


# --- External cross-check against pystoi (test-only, skipped if absent) ----


@pytest.mark.parametrize("extended", [False, True])
@pytest.mark.parametrize("snr_db", [20.0, 10.0, 0.0, -10.0])
def test_matches_pystoi_reference(extended: bool, snr_db: float) -> None:
    pystoi = pytest.importorskip("pystoi")
    x = _speech_like(1)
    y = _add_noise(x, snr_db, seed=10)
    # At FS = 10 kHz both implementations skip resampling, isolating the
    # correlation core; the clean-room result matches the reference to machine
    # precision (the papers' tolerance vs the MATLAB original is 1e-3, so this
    # is two orders tighter and would catch any real regression).
    mine = speech.stoi(x, y, FS, extended=extended).value
    theirs = float(pystoi.stoi(x, y, FS, extended=extended))
    assert mine == pytest.approx(theirs, abs=1e-6)

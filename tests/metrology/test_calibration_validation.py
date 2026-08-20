#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Calibration-tone stability validation (IEC 60942:2017).

IEC 60942:2017 (EN IEC 60942:2018) 5.3.3: the short-term level fluctuation is
measured with time-weighting F over 60 s, sampling at least 30 times; the
absolute value of the difference between each of the maximum and minimum
levels and the mean level shall not exceed the Table 2 (p. 16) acceptance
limits. Class 1: 0.20 dB in 31,5-63 Hz, 0.10 dB in >63-<160 Hz, 0.07 dB from
160 Hz up (classes LS/2 are only specified in 160-1250 Hz: 0.03/0.15 dB).
"""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

from phonometry import metrology

FS = 48000


def _cal_tone(seconds: float = 5.0, am_depth: float = 0.0) -> np.ndarray:
    """94 dB-style 1 kHz calibrator tone, optionally amplitude-modulated."""
    t = np.arange(int(FS * seconds)) / FS
    am = 1.0 + am_depth * np.sin(2 * np.pi * 2.0 * t)
    return 0.5 * am * np.sin(2 * np.pi * 1000 * t)


@contextmanager
def _assert_no_calibration_warning() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", metrology.CalibrationWarning)
        yield


def test_stable_tone_no_warning() -> None:
    with _assert_no_calibration_warning():
        factor = metrology.sensitivity(_cal_tone(), fs=FS)
    assert factor > 0


def test_unstable_tone_warns() -> None:
    """5% AM -> ~0.4 dB fluctuation, far beyond the 0.10 dB class 1 limit."""
    with pytest.warns(metrology.CalibrationWarning, match="fluctuation"):
        metrology.sensitivity(_cal_tone(am_depth=0.05), fs=FS)


def test_validation_can_be_disabled() -> None:
    with _assert_no_calibration_warning():
        metrology.sensitivity(_cal_tone(am_depth=0.05), fs=FS, validate=False)


def test_validation_needs_fs_and_is_skipped_without_it() -> None:
    """`validate=True` alone buys nothing: the check is gated on `fs`.

    The same recording that warns with `fs` passes silently without it, and
    still returns the same factor. The guides now state that precondition
    on the `validate` row instead of only three lines above it.
    """
    unstable = _cal_tone(am_depth=0.05)
    with pytest.warns(metrology.CalibrationWarning, match="fluctuation"):
        with_fs = metrology.sensitivity(unstable, fs=FS, validate=True)
    with _assert_no_calibration_warning():
        without_fs = metrology.sensitivity(unstable, validate=True)
    assert without_fs == pytest.approx(with_fs, rel=1e-12)


def test_custom_fluctuation_limit() -> None:
    with pytest.warns(metrology.CalibrationWarning):
        metrology.sensitivity(_cal_tone(am_depth=0.05), fs=FS, max_fluctuation_db=0.1)
    with _assert_no_calibration_warning():
        metrology.sensitivity(_cal_tone(am_depth=0.05), fs=FS, max_fluctuation_db=1.0)


def test_empty_reference_raises() -> None:
    empty = np.array([])
    with pytest.raises(ValueError, match="empty"):
        metrology.sensitivity(empty, fs=FS)


def test_too_short_recording_warns() -> None:
    with pytest.warns(metrology.CalibrationWarning, match="shorter than 2 s"):
        metrology.sensitivity(_cal_tone(seconds=1.0), fs=FS)


def test_multichannel_stable_at_different_levels_no_warning() -> None:
    """Per-channel stability: a level offset between channels is not fluctuation."""
    x = np.stack([_cal_tone(), 0.5 * _cal_tone()])
    with _assert_no_calibration_warning():
        metrology.sensitivity(x, fs=FS)


def test_default_limit_follows_iec_60942_2017_table2() -> None:
    """~0.13 dB fluctuation: over the 0.07 dB class 1 limit at 1 kHz, but
    within the 0.20 dB limit for a 31.5-63 Hz nominal frequency (Table 2)."""
    x = _cal_tone(am_depth=0.015)
    with pytest.warns(metrology.CalibrationWarning, match="Table 2"):
        metrology.sensitivity(x, fs=FS)
    with _assert_no_calibration_warning():
        metrology.sensitivity(x, fs=FS, frequency=50.0)


def test_asymmetric_fluctuation_uses_max_min_vs_mean() -> None:
    """A one-sided dip must be judged by |min - mean|, not (max - min) / 2."""
    x = _cal_tone(seconds=5.0)
    dip = np.ones_like(x)
    # 200 ms one-sided level dip of ~0.2 dB in the settled region
    i0, i1 = int(3.0 * FS), int(3.2 * FS)
    dip[i0:i1] = 10 ** (-0.2 / 20)
    with pytest.warns(metrology.CalibrationWarning, match="fluctuation"):
        metrology.sensitivity(x * dip, fs=FS)


def test_narrowband_rejects_broadband_noise() -> None:
    """A noisy calibrator take biases the broadband-RMS factor low by
    ``-10*lg(1 + 1/SNR)``; the narrowband tone estimator recovers it."""
    tone = _cal_tone(seconds=3.0)
    tone_power = float(np.mean(tone**2))
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(tone.size)
    # 10 dB SNR broadband noise: broadband RMS inflates ~0.42 dB.
    noise *= np.sqrt(tone_power / 10 ** (10 / 10)) / np.sqrt(np.mean(noise**2))
    noisy = tone + noise

    clean = metrology.sensitivity(tone, fs=FS, validate=False)
    broadband = metrology.sensitivity(noisy, fs=FS, validate=False)
    narrow = metrology.sensitivity(noisy, fs=FS, validate=False, narrowband=True)

    broadband_bias_db = 20 * np.log10(broadband / clean)
    narrow_bias_db = 20 * np.log10(narrow / clean)
    assert broadband_bias_db < -0.3  # broadband under-reads the sensitivity
    assert abs(narrow_bias_db) < 0.05  # narrowband stays within 0.05 dB


def test_narrowband_locks_to_off_nominal_tone() -> None:
    """A calibrator a few Hz off 1 kHz is still estimated exactly."""
    t = np.arange(int(FS * 3.0)) / FS
    tone = 0.5 * np.sin(2 * np.pi * 1003.0 * t)
    clean = metrology.sensitivity(tone, fs=FS, validate=False)
    narrow = metrology.sensitivity(tone, fs=FS, validate=False, narrowband=True)
    assert 20 * np.log10(narrow / clean) == pytest.approx(0.0, abs=0.01)


def test_narrowband_requires_fs() -> None:
    tone = _cal_tone()
    with pytest.raises(ValueError, match="requires 'fs'"):
        metrology.sensitivity(tone, narrowband=True)


def test_table2_row_boundaries() -> None:
    """IEC 60942:2017 Table 2: 160 Hz belongs to the 0.07 dB row and 63 Hz
    to the 0.20 dB row; the open interval between them gets 0.10 dB."""
    from phonometry.metrology.calibration import _class1_fluctuation_limit

    assert _class1_fluctuation_limit(63.0) == pytest.approx(0.20)
    assert _class1_fluctuation_limit(100.0) == pytest.approx(0.10)
    assert _class1_fluctuation_limit(160.0) == pytest.approx(0.07)
    assert _class1_fluctuation_limit(1000.0) == pytest.approx(0.07)


def test_sensitivity_rejects_non_finite_samples() -> None:
    """NaN samples propagated silently to a NaN factor before; now rejected."""
    import numpy as np
    import pytest

    sig = np.sin(2 * np.pi * 1000.0 * np.arange(4800) / 48000.0)
    sig[100] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        metrology.sensitivity(sig, fs=48000)
    sig[100] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        metrology.sensitivity(sig, fs=48000)

#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for underwater reference levels (ISO 18405 / ISO 18406 primitives).

Every quantity is anchored on a synthetic signal of known amplitude and
duration, so the level follows in closed form from the 1 µPa / 1 µPa²·s
references.
"""

from __future__ import annotations

import numpy as np
import pytest
import reference_data as ref

from phonometry import (
    in_air_to_underwater_spl,
    peak_sound_pressure_level,
    sound_exposure_level,
    sound_pressure_level,
    underwater_to_in_air_spl,
)

FS = 48000


def _tone(freq: float, seconds: float, amplitude: float) -> np.ndarray:
    t = np.arange(round(seconds * FS)) / FS
    return amplitude * np.sin(2 * np.pi * freq * t)


def test_spl_of_known_tone() -> None:
    # RMS of a sine of amplitude A is A/sqrt(2); SPL = 20 lg((A/sqrt2)/1e-6).
    amp = 2.0  # Pa
    x = _tone(500.0, 1.0, amp)
    expected = 20.0 * np.log10((amp / np.sqrt(2.0)) / 1e-6)
    assert sound_pressure_level(x) == pytest.approx(expected, rel=1e-9)


def test_sel_of_known_tone() -> None:
    # SEL of a stationary tone over T seconds = SPL + 10 lg(T).
    amp, seconds = 1.0, 2.0
    x = _tone(500.0, seconds, amp)
    spl = 20.0 * np.log10((amp / np.sqrt(2.0)) / 1e-6)
    expected = spl + 10.0 * np.log10(seconds)
    assert sound_exposure_level(x, FS) == pytest.approx(expected, rel=1e-6)


def test_peak_level_of_known_tone() -> None:
    amp = 3.0
    x = _tone(500.0, 0.5, amp)
    expected = 20.0 * np.log10(amp / 1e-6)
    assert peak_sound_pressure_level(x) == pytest.approx(expected, rel=1e-6)


def test_reference_conversion_round_trip() -> None:
    # 20 lg(20e-6/1e-6) = 26.0206 dB.
    assert underwater_to_in_air_spl(120.0) == pytest.approx(120.0 - ref.UW_REFERENCE_OFFSET_DB)
    assert in_air_to_underwater_spl(94.0) == pytest.approx(94.0 + ref.UW_REFERENCE_OFFSET_DB)
    assert in_air_to_underwater_spl(underwater_to_in_air_spl(100.0)) == pytest.approx(100.0)


def test_rejects_invalid_signal() -> None:
    two_dimensional = np.zeros((2, 2))
    with_nan = np.array([np.nan, 1.0])
    valid_tone = _tone(500.0, 0.1, 1.0)
    silence = np.zeros(100)
    with pytest.raises(ValueError):
        sound_pressure_level(two_dimensional)
    with pytest.raises(ValueError):
        sound_pressure_level(with_nan)
    with pytest.raises(ValueError):
        sound_exposure_level(valid_tone, 0.0)
    with pytest.raises(ValueError):
        sound_pressure_level(silence)  # no energy

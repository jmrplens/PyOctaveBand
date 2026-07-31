#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
ECMA-418-2:2025 (Sottek Hearing Model) roughness conformance tests.

The primary oracle is the standard's own calibration (Clause 7, intro and
7.1.7): a 1 kHz carrier, 100 % amplitude-modulated (m = 1) at 70 Hz and a
sound pressure level of 60 dB SPL yields 1 asper. That is the only reference
value the standard tabulates for roughness. The remaining checks are the
behaviours the standard states qualitatively (roughness peaks near a 70 Hz
modulation rate and falls off toward low and high rates, grows with modulation
depth, an unmodulated tone and silence give ~0) plus result-structure guards.
"""

import numpy as np
import pytest

from phonometry import EcmaRoughness, roughness_ecma

FS = 48000
P0 = 2e-5


def _am_tone(
    fc: float,
    fmod: float,
    depth: float,
    level_db: float,
    seconds: float = 2.0,
) -> np.ndarray:
    """Amplitude-modulated tone at an OVERALL RMS level (pressure in Pa).

    Clause 7 states the calibration level as the sound pressure level of the
    signal, i.e. the overall RMS level of the modulated waveform.
    """
    t = np.arange(int(FS * seconds)) / FS
    x = (1.0 + depth * np.cos(2.0 * np.pi * fmod * t)) * np.sin(
        2.0 * np.pi * fc * t
    )
    return np.asarray(
        x * (P0 * 10.0 ** (level_db / 20.0)) / np.sqrt(np.mean(x**2))
    )


def _tone(fc: float, level_db: float, seconds: float = 2.0) -> np.ndarray:
    """Unmodulated tone at ``level_db`` RMS."""
    return _am_tone(fc, 0.0, 0.0, level_db, seconds)


@pytest.fixture(scope="module")
def ref_calibration() -> EcmaRoughness:
    """The 1 kHz / 70 Hz / m=1 / overall 60 dB calibration signal (1 asper)."""
    return roughness_ecma(_am_tone(1000.0, 70.0, 1.0, 60.0), FS)


# --------------------------------------------------------------------------
# Primary reference value (Clause 7 calibration)
# --------------------------------------------------------------------------


def test_calibration_1khz_70hz_is_one_asper(ref_calibration: EcmaRoughness) -> None:
    """The Clause 7 reference signal computes 1 asper.

    A 1 kHz carrier, 100 % amplitude-modulated at 70 Hz, with an overall
    sound pressure level of 60 dB SPL is defined as 1 asper; footnote 34
    allows c_R to be adjusted by at most +/-0.25 %. With the tabulated
    c_R = 0.0180685 (not reverse-fit) and the Clause 5.1.2 Formula (1)
    fade-in applied in the front-end, this chain computes 0.99990 asper.
    """
    assert ref_calibration.roughness == pytest.approx(1.0, abs=0.01)


def test_calibration_constant_is_the_tabulated_c_r() -> None:
    # Formula 104 tabulates c_R = 0.0180685; the implementation must use the
    # standard's value verbatim (shared oracle in tests/reference_data.py).
    from reference_data import ECMA418_2_ROUGHNESS_C_R

    from phonometry.psychoacoustics.roughness_ecma import _C_R

    assert _C_R == ECMA418_2_ROUGHNESS_C_R


# --------------------------------------------------------------------------
# Standard's qualitative behaviours
# --------------------------------------------------------------------------


def test_unmodulated_tone_is_near_zero() -> None:
    result = roughness_ecma(_tone(1000.0, 60.0), FS)
    assert result.roughness < 0.1


def test_silence_is_zero() -> None:
    result = roughness_ecma(np.zeros(FS), FS)
    assert result.roughness == pytest.approx(0.0, abs=1e-6)
    assert np.all(result.specific_roughness == 0.0)


def test_peaks_near_70hz_modulation(ref_calibration: EcmaRoughness) -> None:
    # Roughness is maximal around 70 Hz modulation and falls off toward slow
    # (< 20 Hz) and fast (> 200 Hz) modulation (Clause 7 intro, Annex C). The
    # 70 Hz reference is the module calibration signal, so reuse its result
    # rather than recompute the same chain.
    r70 = ref_calibration.roughness
    r10 = roughness_ecma(_am_tone(1000.0, 10.0, 1.0, 60.0), FS).roughness
    r300 = roughness_ecma(_am_tone(1000.0, 300.0, 1.0, 60.0), FS).roughness
    assert r70 > r10
    assert r70 > r300


def test_grows_with_modulation_depth(ref_calibration: EcmaRoughness) -> None:
    # The full-depth 70 Hz signal is the module calibration signal; reuse it.
    r_full = ref_calibration.roughness
    r_half = roughness_ecma(_am_tone(1000.0, 70.0, 0.5, 60.0), FS).roughness
    assert r_full > r_half > 0.05


# --------------------------------------------------------------------------
# Result structure and API guards
# --------------------------------------------------------------------------


def test_result_structure(ref_calibration: EcmaRoughness) -> None:
    res = ref_calibration
    assert res.specific_roughness.shape == res.bark.shape == (53,)
    assert res.roughness_vs_time.shape == res.time.shape
    assert res.specific_roughness_vs_time.shape == (
        res.time.size,
        res.bark.size,
    )
    assert np.all(res.specific_roughness >= 0.0)
    assert np.all(res.roughness_vs_time >= 0.0)
    assert res.field == "free"


def test_invalid_field_raises() -> None:
    with pytest.raises(ValueError):
        roughness_ecma(_tone(1000.0, 60.0, 0.5), FS, field="bogus")


def test_empty_signal_raises() -> None:
    with pytest.raises(ValueError):
        roughness_ecma(np.array([]), FS)


def test_deterministic(ref_calibration: EcmaRoughness) -> None:
    again = roughness_ecma(_am_tone(1000.0, 70.0, 1.0, 60.0), FS)
    assert again.roughness == pytest.approx(ref_calibration.roughness, abs=1e-9)


def test_free_and_diffuse_differ() -> None:
    sig = _am_tone(1000.0, 70.0, 1.0, 60.0)
    free = roughness_ecma(sig, FS, field="free").roughness
    diffuse = roughness_ecma(sig, FS, field="diffuse").roughness
    assert free != diffuse


def test_time_dependent_pchip_batch_matches_per_band_bitwise() -> None:
    # _time_dependent interpolates all 53 bands with one axis-0 pchip; that
    # must stay bit-identical to the per-band interpolator loop it replaced.
    from scipy.interpolate import PchipInterpolator

    from phonometry.psychoacoustics.roughness_ecma import _CBF, _R_S50

    rng = np.random.default_rng(20260715)
    block_times = np.arange(9) * (4096.0 / FS)
    a_lz = np.abs(rng.standard_normal((block_times.size, _CBF)))
    grid = np.arange(int(np.floor(block_times[-1] * _R_S50)) + 1) / _R_S50
    batched = PchipInterpolator(block_times, a_lz, axis=0)(grid)
    per_band = np.column_stack(
        [
            PchipInterpolator(block_times, a_lz[:, band])(grid)
            for band in range(_CBF)
        ]
    )
    assert np.array_equal(batched, per_band)

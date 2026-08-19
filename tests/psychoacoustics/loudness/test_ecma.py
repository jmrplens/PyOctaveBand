#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
ECMA-418-2:2025 (Sottek Hearing Model) loudness conformance tests.

The primary oracle is the standard's own calibration (Clause 5.1.8): a
1 kHz sinusoid at 40 dB SPL yields 1 sone_HMS via the Clause 8 method. The
standard ships no reference WAVs and its Annex A is graphical only, so the
remaining checks are internal consistency properties the standard implies
(monotonicity in level, silence, field/resampling handling, structure).
"""

import numpy as np
import pytest

from phonometry import psychoacoustics

FS = 48000


def _tone(freq: float, level_db: float, seconds: float = 1.2) -> np.ndarray:
    """Pure tone at an SPL (dB re 20 uPa), pressure in pascals."""
    t = np.arange(int(FS * seconds)) / FS
    amp = np.sqrt(2.0) * 2e-5 * 10 ** (level_db / 20.0)
    return amp * np.sin(2.0 * np.pi * freq * t)


@pytest.fixture(scope="module")
def ref_1k_40() -> psychoacoustics.EcmaLoudness:
    """The calibration signal result, computed once for the module."""
    return psychoacoustics.loudness_ecma(_tone(1000.0, 40.0), FS)


# --------------------------------------------------------------------------
# Primary reference value (Clause 5.1.8 calibration)
# --------------------------------------------------------------------------


@pytest.mark.xdist_group("ecma-loudness-ref")
def test_calibration_1khz_40db_is_one_sone(
    ref_1k_40: psychoacoustics.EcmaLoudness,
) -> None:
    # c_N is defined so the 1 kHz / 40 dB tone gives 1 sone_HMS. With the
    # full Clause 6.2.3 band averaging this chain computes 0.9845 sone_HMS;
    # the -1.55 % residual (outside the +/-0.25 % c_N allowance) comes from
    # the mandated averaging around the block-size-boundary bands and is
    # documented in the module docstring -- c_N stays the verbatim constant.
    assert ref_1k_40.loudness == pytest.approx(1.0, abs=0.03)
    # Regression pin of the deterministic chain value (not a standard target).
    assert ref_1k_40.loudness == pytest.approx(0.9845, abs=0.005)


def test_calibration_constant_is_the_tabulated_c_n() -> None:
    # Clause 5.1.8 tabulates c_N = 0.0211964; the implementation must use the
    # standard's value verbatim (shared oracle in tests/reference_data/).
    from reference_data import ECMA418_2_LOUDNESS_C_N

    from phonometry.psychoacoustics.loudness.ecma import _C_N

    assert _C_N == ECMA418_2_LOUDNESS_C_N


# --------------------------------------------------------------------------
# Internal cross-checks
# --------------------------------------------------------------------------


def test_monotonic_in_level() -> None:
    # Monotonicity is level-driven, not duration-driven: 0.6 s segments and
    # three levels suffice (the 0.6 s / 40 dB anchor also holds 1 sone within
    # 3 % in the CI conformance check, which uses the same duration).
    values = [
        psychoacoustics.loudness_ecma(
            _tone(1000.0, lvl, seconds=0.6), FS
        ).loudness
        for lvl in (20, 40, 80)
    ]
    assert values[0] < values[1] < values[2]
    # 40 dB is the 1-sone anchor; higher levels are clearly louder.
    assert values[1] == pytest.approx(1.0, abs=0.03)
    assert values[2] > 5.0


def test_silence_is_zero() -> None:
    # Pure-property check: zero in, zero out at any length past the
    # transient-discard window, so 0.6 s is enough.
    result = psychoacoustics.loudness_ecma(np.zeros(int(FS * 0.6)), FS)
    assert result.loudness == 0.0
    assert np.all(result.specific_loudness == 0.0)


def test_subthreshold_tone_is_inaudible() -> None:
    # A 1 kHz tone at -10 dB SPL is well below the threshold in quiet.
    result = psychoacoustics.loudness_ecma(
        _tone(1000.0, -10.0, seconds=0.6), FS
    )
    assert result.loudness < 0.01


@pytest.mark.xdist_group("ecma-loudness-ref")
def test_specific_loudness_peaks_near_tone(
    ref_1k_40: psychoacoustics.EcmaLoudness,
) -> None:
    peak_band = int(np.argmax(ref_1k_40.specific_loudness))
    assert ref_1k_40.centre_frequencies[peak_band] == pytest.approx(1000.0, rel=0.15)


# --------------------------------------------------------------------------
# Field handling and resampling
# --------------------------------------------------------------------------


def test_free_and_diffuse_fields_differ() -> None:
    # Property check (ear-filter difference), not a calibration: 0.6 s is
    # enough for a stable value in both fields.
    x = _tone(1000.0, 60.0, seconds=0.6)
    free = psychoacoustics.loudness_ecma(x, FS, field="free").loudness
    diffuse = psychoacoustics.loudness_ecma(x, FS, field="diffuse").loudness
    # Both plausible loudspeaker-range values, but the ear filter differs.
    assert free > 1.0
    assert diffuse > 1.0
    assert free != diffuse


def test_resampling_matches_native_rate() -> None:
    fs_alt = 44100
    t = np.arange(int(fs_alt * 0.6)) / fs_alt
    amp = np.sqrt(2.0) * 2e-5 * 10 ** (40.0 / 20.0)
    x = amp * np.sin(2.0 * np.pi * 1000.0 * t)
    resampled = psychoacoustics.loudness_ecma(x, fs_alt).loudness
    assert resampled == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------
# Result structure and validation
# --------------------------------------------------------------------------


@pytest.mark.xdist_group("ecma-loudness-ref")
def test_result_structure(ref_1k_40: psychoacoustics.EcmaLoudness) -> None:
    assert ref_1k_40.specific_loudness.shape == (53,)
    assert ref_1k_40.bark.shape == (53,)
    assert ref_1k_40.bark[0] == pytest.approx(0.5)
    assert ref_1k_40.bark[-1] == pytest.approx(26.5)
    assert ref_1k_40.centre_frequencies.shape == (53,)
    assert ref_1k_40.time.shape == ref_1k_40.loudness_vs_time.shape
    assert ref_1k_40.field == "free"
    # Time-dependent loudness is sampled at 187.5 Hz (Clause 6.2.6).
    dt = np.diff(ref_1k_40.time)
    assert np.allclose(dt, 1.0 / 187.5)


def test_invalid_field() -> None:
    tone = _tone(1000.0, 40.0, seconds=0.5)
    with pytest.raises(ValueError):
        psychoacoustics.loudness_ecma(tone, FS, field="reverberant")


def test_invalid_fs() -> None:
    tone = _tone(1000.0, 40.0, seconds=0.5)
    with pytest.raises(ValueError):
        psychoacoustics.loudness_ecma(tone, 0.0)


def test_empty_signal() -> None:
    empty = np.array([])
    with pytest.raises(ValueError):
        psychoacoustics.loudness_ecma(empty, FS)


@pytest.mark.xdist_group("ecma-loudness-ref")
def test_plot_smoke(ref_1k_40: psychoacoustics.EcmaLoudness) -> None:
    import matplotlib

    matplotlib.use("Agg")
    axes = ref_1k_40.plot()
    assert axes.shape == (2,)


# --------------------------------------------------------------------------
# Vectorized-kernel equivalence (bitwise) against retained reference loops
# --------------------------------------------------------------------------


def _reference_band_acf(rect: np.ndarray, s_b: int) -> np.ndarray:
    """Per-lag loop form of the unbiased normalized ACF (Formulae 27-29).

    Retained reference for the column-vectorized ``_band_acf``: the
    production kernel must reproduce this loop bit for bit.
    """
    from phonometry.psychoacoustics.loudness.ecma import _EPS

    n_fft = 2 * s_b
    spec = np.fft.fft(rect, n=n_fft, axis=1)
    unscaled = np.real(np.fft.ifft(np.abs(spec) ** 2, axis=1))
    energy = rect**2
    csum = np.cumsum(energy[:, ::-1], axis=1)[:, ::-1]
    m_max = (3 * s_b) // 4
    out = np.zeros_like(unscaled)
    total = csum[:, 0:1]
    for m in range(m_max):
        left = total[:, 0] - (csum[:, s_b - m] if (s_b - m) < s_b else 0.0)
        right = csum[:, m] if m < s_b else 0.0
        denom = np.sqrt(left * right + _EPS)
        out[:, m] = unscaled[:, m] / denom
    return out


def _short_band_signals() -> tuple[list, int]:
    """Clause 5 front-end band-pass signals for a short two-tone signal."""
    from scipy import signal as sp_signal

    from phonometry.psychoacoustics.loudness.ecma import (
        _CBF,
        _auditory_bandpass,
        _ear_filter_sos,
        _preprocess,
    )

    t = np.arange(int(FS * 0.15)) / FS
    x = 0.02 * np.sin(2.0 * np.pi * 1000.0 * t) + 0.01 * np.sin(
        2.0 * np.pi * 4000.0 * t
    )
    padded, _, n_new = _preprocess(x)
    p_om = sp_signal.sosfilt(_ear_filter_sos("free"), padded)
    return [_auditory_bandpass(p_om, band) for band in range(_CBF)], n_new


def test_band_acf_matches_reference_loop_bitwise() -> None:
    # The vectorized lag normalization must be bit-identical to the per-lag
    # reference loop for all four block sizes (8192/4096/2048/1024).
    from phonometry.psychoacoustics.loudness.ecma import (
        _S_B,
        _band_acf,
        _segment_bs,
    )

    p_bands, n_new = _short_band_signals()
    for band in (0, 10, 20, 30, 52):
        s_b = int(_S_B[band])
        seg = _segment_bs(p_bands[band], s_b, n_new)
        rect = np.where(seg > 0.0, seg, 0.0)
        got = _band_acf(rect, s_b, rect**2)
        ref = _reference_band_acf(rect, s_b)
        assert np.array_equal(got, ref), band


def test_average_bands_matches_stacked_mean_bitwise() -> None:
    # The sequential neighbour accumulation of _average_bands_full must be
    # bit-identical to the stacked np.mean formulation it replaced.
    import importlib

    le = importlib.import_module("phonometry.psychoacoustics.loudness.ecma")

    p_bands, n_new = _short_band_signals()
    got = le._average_bands_full(p_bands, n_new)

    cache: dict = {}
    for band in range(le._CBF):
        block_size = int(le._S_B[band])
        n_b = le._N_B_BY_SB[block_size]
        native = le._scaled_acf_at(p_bands, band, block_size, n_new, cache)
        if n_b == 0:
            ref = native
        elif band == 0:
            other = le._scaled_acf_at(p_bands, 1, block_size, n_new, cache)
            ref = 0.5 * (native + other)
        else:
            reach = min(n_b, band, le._CBF - 1 - band)
            if reach == 0:
                ref = native
            else:
                stack = [
                    le._scaled_acf_at(p_bands, band + off, block_size, n_new, cache)
                    for off in range(-reach, reach + 1)
                ]
                ref = np.mean(np.stack(stack, axis=0), axis=0)
        assert np.array_equal(got[band], ref), band

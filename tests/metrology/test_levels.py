#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for integrated and statistical sound levels (Leq, LAeq, LN).
"""

import numpy as np
import pytest

from phonometry import laeq, leq

FS = 48000


def _tone(f0: float, seconds: float = 1.0, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(FS * seconds)) / FS
    return amp * np.sin(2 * np.pi * f0 * t)


def test_leq_sine_matches_rms() -> None:
    """Leq of a 1 Pa amplitude sine = 20*log10((1/sqrt2)/20u) = 90.97 dB."""
    x = _tone(1000)
    assert leq(x) == pytest.approx(90.97, abs=0.05)


def test_leq_dbfs() -> None:
    """RMS of a full-scale sine is -3.01 dBFS."""
    x = _tone(1000)
    assert leq(x, dbfs=True) == pytest.approx(-3.01, abs=0.05)


def test_leq_multichannel_returns_per_channel() -> None:
    x = np.stack([_tone(1000), 0.5 * _tone(1000)])
    out = leq(x)
    assert out.shape == (2,)
    assert out[0] - out[1] == pytest.approx(6.02, abs=0.05)


def test_leq_calibration_factor() -> None:
    x = _tone(1000)
    assert leq(x, calibration_factor=10.0) == pytest.approx(90.97 + 20.0, abs=0.05)


def test_laeq_1khz_equals_leq() -> None:
    """A-weighting is 0 dB at 1 kHz, so LAeq == Leq there."""
    x = _tone(1000, seconds=2.0)
    assert laeq(x, FS) == pytest.approx(leq(x), abs=0.3)


def test_laeq_100hz_attenuated() -> None:
    """A-weighting at 100 Hz is about -19.1 dB."""
    x = _tone(100, seconds=2.0)
    assert laeq(x, FS) - leq(x) == pytest.approx(-19.1, abs=0.5)


def test_ln_levels_constant_signal_all_equal() -> None:
    """For a steady tone, L10 == L90 and L50 == Leq (within envelope ripple)."""
    from phonometry import ln_levels

    x = _tone(1000, seconds=3.0)
    out = ln_levels(x, FS, n=(10, 50, 90))
    assert set(out.keys()) == {10, 50, 90}
    # 5*tau attack skip settles the F integrator to 99.3%, so a steady tone's
    # L10-L90 spread collapses to ~0.01 dB (was ~0.15 dB at the old 2*tau).
    assert out[10] == pytest.approx(out[90], abs=0.05)
    assert out[50] == pytest.approx(90.97, abs=0.3)


def test_ln_levels_ordering() -> None:
    """L10 (exceeded 10% of time) >= L50 >= L90 for a fluctuating signal."""
    from phonometry import ln_levels

    rng = np.random.default_rng(0)
    x = rng.standard_normal(FS * 3) * np.linspace(0.1, 1.0, FS * 3)
    out = ln_levels(x, FS)
    assert out[10] >= out[50] >= out[90]


def test_ln_levels_weighting_a() -> None:
    from phonometry import ln_levels

    x = _tone(100, seconds=3.0)
    unweighted = ln_levels(x, FS, n=(50,))[50]
    weighted = ln_levels(x, FS, n=(50,), weighting="A")[50]
    assert weighted - unweighted == pytest.approx(-19.1, abs=0.5)


def test_ln_levels_invalid_percentile_raises() -> None:
    from phonometry import ln_levels

    with pytest.raises(ValueError, match="between 0 and 100"):
        ln_levels(_tone(1000), FS, n=(0,))


def test_ln_levels_multichannel() -> None:
    from phonometry import ln_levels

    x = np.stack([_tone(1000, 2.0), 0.5 * _tone(1000, 2.0)])
    out = ln_levels(x, FS, n=(50,))
    assert out[50].shape == (2,)
    assert out[50][0] - out[50][1] == pytest.approx(6.02, abs=0.2)


def test_leq_dbfs_ignores_calibration_factor() -> None:
    """dBFS is relative to digital full scale (consistent with OctaveFilterBank)."""
    x = _tone(1000)
    assert leq(x, calibration_factor=10.0, dbfs=True) == pytest.approx(
        leq(x, dbfs=True), abs=1e-12
    )


def test_leq_empty_signal_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        leq(np.array([]))


def test_leq_nonpositive_calibration_raises() -> None:
    with pytest.raises(ValueError, match="calibration_factor"):
        leq(_tone(1000), calibration_factor=-1.0)


def test_ln_levels_empty_signal_raises() -> None:
    from phonometry import ln_levels

    with pytest.raises(ValueError, match="empty"):
        ln_levels(np.array([]), FS)


# ---------------------------------------------------------------------------
# LCpeak — IEC 61672-1:2013 §5.13
# ---------------------------------------------------------------------------


def _faded(x: np.ndarray, ramp: float = 0.05) -> np.ndarray:
    """Fade a tone in/out so the filter onset transient does not add overshoot."""
    n = int(FS * ramp)
    window = np.ones_like(x)
    window[:n] = np.hanning(2 * n)[:n]
    window[-n:] = np.hanning(2 * n)[n:]
    return x * window


def test_lc_peak_steady_1khz() -> None:
    """C weighting is ~0 dB at 1 kHz: LCpeak of a steady sine = 20*log10(A/p0)."""
    from phonometry import lc_peak

    x = _faded(_tone(1000, seconds=1.0, amp=1.0))
    assert lc_peak(x, FS) == pytest.approx(20 * np.log10(1.0 / 2e-5), abs=0.15)


def test_lc_peak_exceeds_lc_by_crest_factor() -> None:
    """For a steady sine, LCpeak - LC = 20*log10(sqrt(2)) = 3.01 dB."""
    from phonometry import lc_peak, leq
    from phonometry.metrology.parametric_filters import weighting_filter

    # 10 ms ramps: enough to avoid the onset click without biasing the RMS
    x = _faded(_tone(1000, seconds=1.0), ramp=0.01)
    lc = leq(weighting_filter(x, FS, "C"))
    assert lc_peak(x, FS) - lc == pytest.approx(3.01, abs=0.2)


def test_lc_peak_multichannel_and_dbfs() -> None:
    from phonometry import lc_peak

    x = np.stack([_faded(_tone(1000)), 0.5 * _faded(_tone(1000))])
    out = lc_peak(x, FS)
    assert out.shape == (2,)
    assert out[0] - out[1] == pytest.approx(6.02, abs=0.1)
    # dBFS: peak 1.0 -> 0 dBFS, calibration must not apply
    assert lc_peak(_faded(_tone(1000)), FS, calibration_factor=10.0, dbfs=True) == pytest.approx(0.0, abs=0.15)


@pytest.mark.parametrize(
    "cycles,freq,ref,tol",
    [
        # BS EN 61672-1:2013 Table 5 (standard page 27): reference differences
        # LCpeak - LC and class 1 acceptance limits. Test frequencies are the
        # EXACT one-third-octave frequencies (Annex D), not nominal.
        (1.0, 10 ** 1.5, 2.5, 2.0),     # one cycle, 31.5 Hz nominal
        (1.0, 10 ** 2.7, 3.5, 1.0),     # one cycle, 500 Hz nominal
        (1.0, 10 ** 3.9, 3.4, 2.0),     # one cycle, 8 kHz nominal
        (0.5, 10 ** 2.7, 2.4, 1.0),     # positive half cycle, 500 Hz
        (-0.5, 10 ** 2.7, 2.4, 1.0),    # negative half cycle, 500 Hz
    ],
)
def test_lc_peak_iec_table5(cycles: float, freq: float, ref: float, tol: float) -> None:
    """One-cycle / half-cycle bursts must reproduce Table 5 within class 1."""
    from phonometry import lc_peak, leq
    from phonometry.metrology.parametric_filters import weighting_filter

    fs = 96000
    t = np.arange(int(fs * 1.0)) / fs
    steady = np.sin(2 * np.pi * freq * t)
    lc_steady = leq(weighting_filter(steady, fs, "C"))

    n = round(abs(cycles) * fs / freq)  # starts and stops on zero crossings
    sign = 1.0 if cycles > 0 else -1.0
    burst = np.zeros(int(fs * 1.0))
    start = len(burst) // 2
    tt = np.arange(n) / fs
    burst[start:start + n] = sign * np.sin(2 * np.pi * freq * tt)

    diff = lc_peak(burst, fs) - lc_steady
    assert diff == pytest.approx(ref, abs=tol), f"{cycles} cycles @ {freq:.0f} Hz: {diff:.2f} dB"


@pytest.mark.parametrize(
    "cycles,freq,ref,tol",
    [
        # Same BS EN 61672-1:2013 Table 5 cases as above, at a field-typical
        # 48 kHz (audit N2 I-1: the standard test ran only at 96 kHz, masking
        # the inter-sample peak loss). Must still reproduce Table 5 within
        # class 1 with the oversampled peak detection.
        (1.0, 10 ** 1.5, 2.5, 2.0),     # one cycle, 31.5 Hz nominal
        (1.0, 10 ** 2.7, 3.5, 1.0),     # one cycle, 500 Hz nominal
        (1.0, 10 ** 3.9, 3.4, 2.0),     # one cycle, 8 kHz nominal
        (0.5, 10 ** 2.7, 2.4, 1.0),     # positive half cycle, 500 Hz
        (-0.5, 10 ** 2.7, 2.4, 1.0),    # negative half cycle, 500 Hz
    ],
)
def test_lc_peak_iec_table5_48k(cycles: float, freq: float, ref: float, tol: float) -> None:
    """Table 5 reference differences must also hold at fs = 48 kHz."""
    from phonometry import lc_peak, leq
    from phonometry.metrology.parametric_filters import weighting_filter

    fs = 48000
    t = np.arange(int(fs * 1.0)) / fs
    steady = np.sin(2 * np.pi * freq * t)
    lc_steady = leq(weighting_filter(steady, fs, "C"))

    n = round(abs(cycles) * fs / freq)  # starts and stops on zero crossings
    sign = 1.0 if cycles > 0 else -1.0
    burst = np.zeros(int(fs * 1.0))
    start = len(burst) // 2
    tt = np.arange(n) / fs
    burst[start:start + n] = sign * np.sin(2 * np.pi * freq * tt)

    diff = lc_peak(burst, fs) - lc_steady
    assert diff == pytest.approx(ref, abs=tol), f"{cycles} cycles @ {freq:.0f} Hz: {diff:.2f} dB"


def _lcpeak_analytic_steady(x: np.ndarray, fs: int) -> float:
    """Analytic LCpeak of a steady tone: C-weighted steady RMS + 3.01 dB crest.

    The crest factor of a pure sinusoid is exactly 20*log10(sqrt(2)) = 3.01 dB,
    verified independent of the C-weighting gain. Measured from a transient-free
    middle window so it isolates the peak-detection accuracy.
    """
    from phonometry.metrology.parametric_filters import weighting_filter

    w = weighting_filter(x, fs, "C")
    mid = w[int(0.4 * w.shape[-1]):int(0.6 * w.shape[-1])]
    rms = np.sqrt(np.mean(mid ** 2))
    return float(20 * np.log10(rms / 2e-5) + 3.01)


def test_lc_peak_recovers_inter_sample_peak_8k_48k() -> None:
    """LCpeak must catch the inter-sample peak of a sustained 8 kHz tone at 48 kHz.

    8 kHz at 48 kHz is exactly 6.0 samples/cycle, so the sampling phase is
    locked and the on-grid maximum consistently under-reads the true peak
    (audit N2 I-1: up to -1.15 dB worst-case over phase, -0.69 dB at phase 0).
    Oversampled peak detection recovers the analytic sinusoid crest.
    """
    from phonometry import lc_peak

    x = _faded(_tone(8000, seconds=1.0))
    err = lc_peak(x, FS) - _lcpeak_analytic_steady(x, FS)
    assert abs(err) < 0.5, f"LCpeak under-reads by {err:+.3f} dB (inter-sample loss)"


def test_lc_peak_oversample_keyword_controls_recovery() -> None:
    """oversample=1 reproduces the legacy on-grid under-read; the default fixes it."""
    from phonometry import lc_peak

    x = _faded(_tone(8000, seconds=1.0))
    ref = _lcpeak_analytic_steady(x, FS)
    legacy_err = lc_peak(x, FS, oversample=1) - ref
    fixed_err = lc_peak(x, FS) - ref
    assert legacy_err < -0.5          # legacy on-grid detection under-reads
    assert abs(fixed_err) < abs(legacy_err)   # default oversampling recovers it


# ---------------------------------------------------------------------------
# SEL / LAE — sound exposure level
# ---------------------------------------------------------------------------


def test_sel_steady_signal_normalizes_to_one_second() -> None:
    """SEL = Leq + 10*log10(T / 1 s) for a steady signal of duration T."""
    from phonometry import leq, sel

    x = _tone(1000, seconds=4.0)
    assert sel(x, FS) == pytest.approx(leq(x) + 10 * np.log10(4.0), abs=1e-6)


def test_sel_one_second_equals_leq() -> None:
    from phonometry import leq, sel

    x = _tone(1000, seconds=1.0)
    assert sel(x, FS) == pytest.approx(leq(x), abs=1e-9)


def test_sel_a_weighted() -> None:
    from phonometry import laeq, sel

    x = _tone(1000, seconds=2.0)
    assert sel(x, FS, weighting="A") == pytest.approx(laeq(x, FS) + 10 * np.log10(2.0), abs=0.05)


# ---------------------------------------------------------------------------
# Sound exposure / dose — IEC 61252 (BS EN 61252:1995 §3.1-3.3, Annex A)
# ---------------------------------------------------------------------------


def _tone_at_level(level_db: float, seconds: float = 2.0, f0: float = 1000.0) -> np.ndarray:
    """1 kHz tone whose A-weighted level equals level_db (A(1 kHz) = 0 dB)."""
    rms = 2e-5 * 10 ** (level_db / 20)
    t = np.arange(int(FS * seconds)) / FS
    return np.sqrt(2) * rms * np.sin(2 * np.pi * f0 * t)


def test_sound_exposure_anchor_90db_8h_is_3p2_pa2h() -> None:
    """BS EN 61252:1995 Annex A / §3.3 NOTE 4: 3.2 Pa²h <-> exactly 90 dB."""
    from phonometry import sound_exposure

    x = _tone_at_level(90.0)
    assert sound_exposure(x, FS, duration_hours=8.0) == pytest.approx(3.2, rel=0.01)


def test_lex_8h_anchor_90db() -> None:
    from phonometry import lex_8h

    x = _tone_at_level(90.0)
    assert lex_8h(x, FS, duration_hours=8.0) == pytest.approx(90.0, abs=0.05)


def test_lex_8h_half_workday_subtracts_3db() -> None:
    """LEX,8h = LAeq,T + 10*log10(T/8h): a 4 h exposure at 90 dB -> 86.99 dB."""
    from phonometry import lex_8h

    x = _tone_at_level(90.0)
    assert lex_8h(x, FS, duration_hours=4.0) == pytest.approx(90.0 + 10 * np.log10(4 / 8), abs=0.05)


def test_sound_exposure_1_pa2h_is_nearly_85db() -> None:
    """§3.3 NOTE 4: 1 Pa²h corresponds to a LEX,8h of nearly 85 dB (84.95)."""
    from phonometry import lex_8h, sound_exposure

    x = _tone_at_level(84.9485)
    assert sound_exposure(x, FS, duration_hours=8.0) == pytest.approx(1.0, rel=0.01)
    assert lex_8h(x, FS, duration_hours=8.0) == pytest.approx(84.95, abs=0.05)


def test_sound_exposure_defaults_to_recording_duration() -> None:
    """Without duration_hours, x IS the whole event: E = integral over len(x)."""
    from phonometry import sound_exposure

    x = _tone_at_level(90.0, seconds=2.0)
    expected = (2e-5 * 10 ** (90 / 20)) ** 2 * (2.0 / 3600.0)  # Pa² * hours
    assert sound_exposure(x, FS) == pytest.approx(expected, rel=0.01)


def test_sel_invalid_fs_raises() -> None:
    from phonometry import sel

    with pytest.raises(ValueError, match="fs"):
        sel(_tone(1000), 0)


def test_sound_exposure_rejects_nonpositive_duration() -> None:
    from phonometry import lex_8h, sound_exposure

    with pytest.raises(ValueError, match="duration_hours"):
        sound_exposure(_tone_at_level(90.0), FS, duration_hours=0)
    with pytest.raises(ValueError, match="duration_hours"):
        lex_8h(_tone_at_level(90.0), FS, duration_hours=-1.0)

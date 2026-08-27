#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ITU-R BS.1770-5 / EBU R 128 programme loudness and true peak.

Validated against the synthesizable EBU Tech 3341 Table 1 'minimum
requirements' test signals (loudness tolerance +/-0.1 LU, true peak
+0.2/-0.4 dB), the EBU Tech 3342 Table 1 loudness-range signals (+/-1 LU)
and the Recommendation's own 997 Hz / -3.01 LKFS anchor. Cases 7-8 of Tech
3341 and 5-6 of Tech 3342 use authentic programme material distributed by
the EBU and cannot be synthesized, so they are not reproduced here.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from reference_data import (
    BS1770_ANCHOR_997_LKFS,
    EBU_TECH3341_CASE6_EXPECTED,
    EBU_TECH3341_CASE6_LEVELS,
    EBU_TECH3341_INTEGRATED_CASES,
    EBU_TECH3341_TOL_LU,
    EBU_TECH3341_TP_OFFSET_EXPECTED,
    EBU_TECH3341_TP_TOL_DOWN,
    EBU_TECH3341_TP_TOL_UP,
    EBU_TECH3341_TRUE_PEAK_CASES,
    EBU_TECH3342_LRA_CASES,
    EBU_TECH3342_TOL_LU,
)
from scipy import signal as sg

from phonometry.broadcast import (
    DEFAULT_CHANNEL_WEIGHTS,
    KWeightingResponse,
    ProgramLoudnessResult,
    channel_weight,
    integrated_loudness,
    k_weighting,
    k_weighting_coefficients,
    k_weighting_response,
    loudness_range,
    program_loudness,
    true_peak_level,
)

FS = 48000


def _sine(level_dbfs: float, duration: float, freq: float = 1000.0) -> np.ndarray:
    """A sine with per-sample peak level ``level_dbfs`` (dB re full scale)."""
    t = np.arange(round(duration * FS)) / FS
    return 10.0 ** (level_dbfs / 20.0) * np.sin(2.0 * np.pi * freq * t)


def _stereo(x: np.ndarray) -> np.ndarray:
    """The signal applied in phase to both channels simultaneously."""
    return np.vstack([x, x])


def _steps(segments: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Concatenated stereo 1 kHz tone steps (level dBFS, duration s)."""
    return _stereo(np.concatenate([_sine(lvl, dur) for lvl, dur in segments]))


def _tapered(x: np.ndarray, fs: float, fade: float = 0.01) -> np.ndarray:
    """Apply the 10 ms fade-in/fade-out that Tech 3341 asks for."""
    n = round(fade * fs)
    out = x.copy()
    out[:n] *= np.linspace(0.0, 1.0, n)
    out[-n:] *= np.linspace(1.0, 0.0, n)
    return out


# ---------------------------------------------------------------------------
# BS.1770-5 Annex 1: the normative 997 Hz anchor and the K-weighting filter.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel", [0, 1, 2])
def test_annex1_997_hz_anchor(channel: int) -> None:
    """A 0 dB FS 997 Hz sine on L, C or R reads -3.01 LKFS (Annex 1 text)."""
    x = np.zeros((5, 20 * FS))
    x[channel] = _sine(0.0, 20.0, freq=997.0)
    assert integrated_loudness(x, FS) == pytest.approx(
        BS1770_ANCHOR_997_LKFS, abs=0.005
    )


def test_k_weighting_tables_verbatim_at_48k() -> None:
    """At 48 kHz the Table 1 and Table 2 coefficients are returned exactly."""
    (b1, a1), (b2, a2) = k_weighting_coefficients(48000)
    assert b1.tolist() == [1.53512485958697, -2.69169618940638, 1.19839281085285]
    assert a1.tolist() == [1.0, -1.69065929318241, 0.73248077421585]
    assert b2.tolist() == [1.0, -2.0, 1.0]
    assert a2.tolist() == [1.0, -1.99004745483398, 0.99007225036621]


def test_k_weighting_redesign_round_trip() -> None:
    """The analog round trip reproduces the 48 kHz tables to double precision."""
    from phonometry.broadcast.program_loudness import (
        _STAGE1_A48,
        _STAGE1_B48,
        _STAGE2_A48,
        _STAGE2_B48,
        _redesign_biquad,
    )

    for b48, a48 in ((_STAGE1_B48, _STAGE1_A48), (_STAGE2_B48, _STAGE2_A48)):
        b, a = _redesign_biquad(b48, a48, 48000.0)
        np.testing.assert_allclose(b, b48, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a, a48, rtol=1e-12, atol=1e-12)


def _k_response_db(fs: float, freqs: np.ndarray) -> np.ndarray:
    (b1, a1), (b2, a2) = k_weighting_coefficients(fs)
    h = sg.freqz(b1, a1, worN=freqs, fs=fs)[1] * sg.freqz(b2, a2, worN=freqs, fs=fs)[1]
    return np.asarray(20.0 * np.log10(np.abs(h)))


@pytest.mark.parametrize("fs", [44100.0, 96000.0, 192000.0])
def test_k_weighting_response_matches_48k_specification(fs: float) -> None:
    """Other rates preserve the 48 kHz response (Annex 1 requirement).

    The residual is bilinear warping of the shelf transition region (the
    48 kHz reference is itself slightly warped towards its own Nyquist):
    about 0.016 dB worst case, far inside the +/-0.1 LU system tolerance.
    """
    freqs = np.logspace(np.log10(20.0), np.log10(20000.0), 200)
    dev = _k_response_db(fs, freqs) - _k_response_db(48000.0, freqs)
    assert np.max(np.abs(dev)) < 0.02


def test_k_weighting_shape() -> None:
    """The pre-filter is a ~+4 dB HF shelf plus a low-frequency roll-off."""
    freqs = np.array([25.0, 997.0, 10000.0])
    resp = _k_response_db(48000.0, freqs)
    assert resp[0] < -9.0  # RLB high-pass attenuates the lows
    assert resp[1] == pytest.approx(0.691, abs=0.05)  # the Formula 2 constant
    assert resp[2] == pytest.approx(4.0, abs=0.1)  # spherical-head shelf


def test_k_weighting_rejects_low_rate_and_empty() -> None:
    # Below 16 kHz the bilinear warping breaks the +/-0.1 LU tolerance (the
    # 997 Hz anchor drifts to about -2.89 LKFS at 8 kHz), so such rates raise.
    for fs in (4000.0, 8000.0):
        with pytest.raises(ValueError, match="fs >= 16000"):
            k_weighting_coefficients(fs)
    empty = np.empty(0)
    with pytest.raises(ValueError, match="empty"):
        k_weighting(empty, FS)


def test_k_weighting_response_matches_coefficients() -> None:
    """The result carries the exact freqz magnitudes of the Table 1-2 biquads.

    Anchors against the same +4 dB shelf plateau and 997 Hz gain the loudness
    formula assumes: the shelf tends to +4 dB, the combined response passes
    through ~0.69 dB near 1 kHz (the ``-0.691`` Formula 2 constant) and the RLB
    high-pass attenuates the lows.
    """
    res = k_weighting_response(48000.0, frequencies=[25.0, 997.0, 10000.0, 20000.0])
    assert isinstance(res, KWeightingResponse)
    # The stages sum to the combined magnitude, and reproduce the freqz oracle.
    np.testing.assert_allclose(res.magnitude_db, res.shelf_db + res.highpass_db)
    np.testing.assert_allclose(
        res.magnitude_db, _k_response_db(48000.0, res.frequencies), rtol=0, atol=1e-12
    )
    assert res.shelf_db[3] == pytest.approx(4.0, abs=0.01)  # HF shelf plateau
    assert res.magnitude_db[1] == pytest.approx(0.691, abs=0.05)  # Formula 2 gain
    assert res.magnitude_db[0] < -9.0  # RLB high-pass attenuates the lows
    assert res.fs == 48000.0


def test_k_weighting_response_default_grid() -> None:
    """The default grid is log-spaced from 10 Hz to the Nyquist rate."""
    res = k_weighting_response(96000.0, n=256)
    assert res.frequencies.size == 256
    assert res.frequencies[0] == pytest.approx(10.0)
    assert res.frequencies[-1] == pytest.approx(48000.0)


def test_k_weighting_response_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="fs >= 16000"):
        k_weighting_response(8000.0)
    with pytest.raises(ValueError, match="positive integer"):
        k_weighting_response(48000.0, n=0)
    with pytest.raises(ValueError, match="cannot be empty"):
        k_weighting_response(48000.0, frequencies=[])
    with pytest.raises(ValueError, match=r"\(0, fs/2\]"):
        k_weighting_response(48000.0, frequencies=[30000.0])


def test_k_weighting_response_rejects_curve_off_the_grid() -> None:
    """A curve of another length is refused where the response is built."""
    res = k_weighting_response(48000.0, n=64)
    short = res.shelf_db[:-1]
    with pytest.raises(ValueError, match="'shelf_db'"):
        replace(res, shelf_db=short)


def test_k_weighting_response_plot() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    res = k_weighting_response()
    ax = res.plot()
    assert ax.get_ylabel() == "Magnitude [dB]"
    assert "BS.1770" in ax.get_title()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("combined" in s for s in labels)
    # Spanish localises the title; unknown languages are rejected.
    assert "BS.1770" in res.plot(language="es").get_title()
    # Passing an existing axes draws on it and returns that same axes.
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext
    # kwargs reach the combined-curve line.
    styled = res.plot(color="magenta", linewidth=3.0)
    assert any(
        line.get_color() == "magenta" and line.get_linewidth() == 3.0
        for line in styled.get_lines()
    )
    with pytest.raises(ValueError, match="Unknown language"):
        res.plot(language="fr")
    plt.close("all")


# ---------------------------------------------------------------------------
# EBU Tech 3341 Table 1, cases 1-6: integrated loudness.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "segments", "expected"),
    EBU_TECH3341_INTEGRATED_CASES,
    ids=[f"case{c}" for c, _, _ in EBU_TECH3341_INTEGRATED_CASES],
)
def test_tech3341_integrated(
    case: int, segments: tuple[tuple[float, float], ...], expected: float
) -> None:
    assert integrated_loudness(_steps(segments), FS) == pytest.approx(
        expected, abs=EBU_TECH3341_TOL_LU
    )


def test_tech3341_case6_five_channels() -> None:
    """5.0 sine with per-channel levels; exercises the Table 3 weights."""
    x = np.vstack([_sine(lvl, 20.0) for lvl in EBU_TECH3341_CASE6_LEVELS])
    assert integrated_loudness(x, FS) == pytest.approx(
        EBU_TECH3341_CASE6_EXPECTED, abs=EBU_TECH3341_TOL_LU
    )


def test_tech3341_case1_momentary_short_term() -> None:
    """Case 1 expects M, S and I all at -23.0 LUFS."""
    res = program_loudness(_steps(((-23.0, 20.0),)), FS)
    for value in (res.integrated, res.max_momentary, res.max_short_term):
        assert value == pytest.approx(-23.0, abs=EBU_TECH3341_TOL_LU)


def test_tech3341_case2_momentary_short_term() -> None:
    res = program_loudness(_steps(((-33.0, 20.0),)), FS)
    for value in (res.integrated, res.max_momentary, res.max_short_term):
        assert value == pytest.approx(-33.0, abs=EBU_TECH3341_TOL_LU)


# ---------------------------------------------------------------------------
# EBU Tech 3341 Table 1, cases 9-14: momentary / short-term dynamics.
# ---------------------------------------------------------------------------


def test_tech3341_case9_short_term_constant() -> None:
    """(1.34 s at -20; 1.66 s at -30) x 5: S = -23.0 constant after 3 s."""
    x = _stereo(
        np.concatenate([np.concatenate([_sine(-20.0, 1.34), _sine(-30.0, 1.66)])] * 5)
    )
    res = program_loudness(x, FS)
    settled = res.short_term[res.short_term_time >= 3.0]
    np.testing.assert_allclose(settled, -23.0, atol=EBU_TECH3341_TOL_LU)


def test_tech3341_case12_momentary_constant() -> None:
    """(0.18 s at -20; 0.22 s at -30) x 25: M = -23.0 constant after 1 s."""
    x = _stereo(
        np.concatenate([np.concatenate([_sine(-20.0, 0.18), _sine(-30.0, 0.22)])] * 25)
    )
    res = program_loudness(x, FS)
    settled = res.momentary[res.momentary_time >= 1.0]
    np.testing.assert_allclose(settled, -23.0, atol=EBU_TECH3341_TOL_LU)


@pytest.mark.parametrize("i", range(20))
def test_tech3341_case10_max_short_term_per_segment(i: int) -> None:
    """i*0.15 s silence + 3 s at -23 + 1 s silence: Max S = -23.0 each."""
    x = _stereo(
        np.concatenate([np.zeros(int(i * 0.15 * FS)), _sine(-23.0, 3.0), np.zeros(FS)])
    )
    res = program_loudness(x, FS)
    assert res.max_short_term == pytest.approx(-23.0, abs=EBU_TECH3341_TOL_LU)


@pytest.mark.parametrize("i", range(20))
def test_tech3341_case13_max_momentary_per_segment(i: int) -> None:
    """i*20 ms silence + 400 ms at -23 + 1 s silence: Max M = -23.0 each."""
    x = _stereo(
        np.concatenate([np.zeros(int(i * 0.02 * FS)), _sine(-23.0, 0.4), np.zeros(FS)])
    )
    res = program_loudness(x, FS)
    assert res.max_momentary == pytest.approx(-23.0, abs=EBU_TECH3341_TOL_LU)


def test_tech3341_case11_successive_max_short_term() -> None:
    """20 tones stepping -38..-19 dBFS: running Max S tracks each step."""
    x = _stereo(
        np.concatenate(
            [
                np.concatenate(
                    [
                        np.zeros(int(i * 0.15 * FS)),
                        _sine(-38.0 + i, 3.0),
                        np.zeros(round((3.0 - i * 0.15) * FS)),
                    ]
                )
                for i in range(20)
            ]
        )
    )
    res = program_loudness(x, FS)
    for i in range(20):
        upto = res.short_term[res.short_term_time <= (i + 1) * 6.0 + 1e-9]
        assert float(np.max(upto)) == pytest.approx(-38.0 + i, abs=EBU_TECH3341_TOL_LU)


def test_tech3341_case14_successive_max_momentary() -> None:
    """20 tones of 400 ms stepping -38..-19 dBFS: running Max M tracks."""
    x = _stereo(
        np.concatenate(
            [
                np.concatenate(
                    [
                        np.zeros(int(i * 0.02 * FS)),
                        _sine(-38.0 + i, 0.4),
                        np.zeros(round((0.4 - i * 0.02) * FS)),
                    ]
                )
                for i in range(20)
            ]
        )
    )
    res = program_loudness(x, FS)
    for i in range(20):
        upto = res.momentary[res.momentary_time <= (i + 1) * 0.8 + 1e-9]
        assert float(np.max(upto)) == pytest.approx(-38.0 + i, abs=EBU_TECH3341_TOL_LU)


# ---------------------------------------------------------------------------
# EBU Tech 3341 Table 1, cases 15-23: true peak.
# ---------------------------------------------------------------------------


def _assert_true_peak(measured: float, expected: float) -> None:
    """Assert the official asymmetric tolerance (+0.2/-0.4 dB)."""
    assert expected - EBU_TECH3341_TP_TOL_DOWN <= measured
    assert measured <= expected + EBU_TECH3341_TP_TOL_UP


@pytest.mark.parametrize(
    ("case", "freq_ratio", "amplitude", "phase_deg", "expected"),
    EBU_TECH3341_TRUE_PEAK_CASES,
    ids=[f"case{c}" for c, *_ in EBU_TECH3341_TRUE_PEAK_CASES],
)
def test_tech3341_true_peak_tones(
    case: int, freq_ratio: float, amplitude: float, phase_deg: float, expected: float
) -> None:
    t = np.arange(FS) / FS
    x = amplitude * np.sin(2.0 * np.pi * (freq_ratio * FS) * t + np.deg2rad(phase_deg))
    tp = true_peak_level(_stereo(_tapered(x, FS)), FS)
    _assert_true_peak(float(np.max(tp)), expected)


@pytest.mark.parametrize("offset", range(4), ids=[f"case{20 + o}" for o in range(4)])
def test_tech3341_true_peak_intersample_offsets(offset: int) -> None:
    """Cases 20-23: one fs/4 period inside an fs/6 tone, decimated with a
    0-3 sample offset at the 4 fs synthesis rate; all read 0.0 dBTP.
    """
    fs4 = 4 * FS
    n = round(1.0 * fs4)
    frequency = np.full(n, FS / 6.0)
    amplitude = np.full(n, 0.5)
    mid = n // 2
    period = round(fs4 / (FS / 4.0))
    frequency[mid : mid + period] = FS / 4.0
    amplitude[mid : mid + period] = 1.0
    # Phase-continuous synthesis at 4 fs, tapered as the table asks.
    x4 = _tapered(amplitude * np.sin(2.0 * np.pi * np.cumsum(frequency) / fs4), fs4)
    # Anti-alias lowpass, then decimate with the given offset.
    taps = sg.firwin(1023, 0.9 / 4.0)
    x = sg.lfilter(taps, 1.0, x4)[511 + offset :: 4]
    tp = true_peak_level(_stereo(x), FS)
    _assert_true_peak(float(np.max(tp)), EBU_TECH3341_TP_OFFSET_EXPECTED)


def test_true_peak_under_read_bound() -> None:
    """The worst-phase under-read obeys 20 lg cos(pi fnorm / n) (Annex 2
    Attachment 1) at fnorm = 0.25 with the default 4x oversampling.
    """
    t = np.arange(FS) / FS
    worst = 0.0
    for phase in np.linspace(0.0, np.pi / 2.0, 9):
        x = np.sin(2.0 * np.pi * (FS / 4.0) * t + phase)
        worst = min(worst, float(true_peak_level(_tapered(x, FS), FS)))
    bound = 20.0 * np.log10(np.cos(np.pi * 0.25 / 4.0))
    # Within the geometric bound plus a small interpolation-ripple margin.
    assert worst >= bound - 0.05
    # And without oversampling the same tone can under-read by ~3 dB.
    x = np.sin(2.0 * np.pi * (FS / 4.0) * t + np.pi / 4.0)
    raw = float(true_peak_level(x, FS, oversample=1))
    assert raw == pytest.approx(20.0 * np.log10(np.cos(np.pi / 4.0)), abs=0.01)


def test_true_peak_defaults_and_validation() -> None:
    x = _sine(-6.0, 0.5)
    # Explicit 4x equals the default at 48 kHz.
    assert true_peak_level(x, FS) == true_peak_level(x, FS, oversample=4)
    with pytest.raises(ValueError, match="oversample"):
        true_peak_level(x, FS, oversample=0)
    with pytest.raises(ValueError, match="oversample"):
        true_peak_level(x, FS, oversample=True)  # type: ignore[arg-type]
    empty = np.empty(0)
    with pytest.raises(ValueError, match="empty"):
        true_peak_level(empty, FS)
    # Silence: -inf dBTP, no runtime warnings leak.
    assert true_peak_level(np.zeros(1000), FS) == float("-inf")


# ---------------------------------------------------------------------------
# EBU Tech 3342 Table 1: loudness range.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "levels", "expected"),
    EBU_TECH3342_LRA_CASES,
    ids=[f"case{c}" for c, _, _ in EBU_TECH3342_LRA_CASES],
)
def test_tech3342_loudness_range(
    case: int, levels: tuple[float, ...], expected: float
) -> None:
    x = _steps(tuple((lvl, 20.0) for lvl in levels))
    res = program_loudness(x, FS)
    assert res.loudness_range == pytest.approx(expected, abs=EBU_TECH3342_TOL_LU)


def test_tech3342_repetition_invariance() -> None:
    """Repeating a test signal in full leaves the expected LRA unchanged."""
    x = _steps(((-20.0, 20.0), (-30.0, 20.0)))
    res = program_loudness(np.hstack([x, x]), FS)
    assert res.loudness_range == pytest.approx(10.0, abs=EBU_TECH3342_TOL_LU)


def test_loudness_range_reference_percentile_indexing() -> None:
    """loudness_range follows the Tech 3342 reference implementation on a
    hand-computable short-term vector (no gating active).
    """
    stl = np.linspace(-40.0, -20.0, 21)  # 1 LU steps; mean power ~ -24.2
    # Relative threshold ~ -44.2 gates nothing; indices round(20*p/100):
    # low = stl[2] = -38, high = stl[19] = -21.
    assert loudness_range(stl) == pytest.approx(17.0, abs=1e-12)


def test_loudness_range_gates_silence() -> None:
    stl = np.concatenate([np.full(50, -90.0), np.full(50, -23.0)])
    # The -90 LUFS readings fall below the absolute gate and do not widen LRA.
    assert loudness_range(stl) == pytest.approx(0.0, abs=1e-12)
    assert loudness_range(np.full(10, -80.0)) == 0.0


# ---------------------------------------------------------------------------
# Channel weights (Annex 1 Table 3, Annex 3 Tables 4-5).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("azimuth", "elevation", "expected"),
    [
        (0.0, 0.0, 1.0),  # M+000
        (30.0, 0.0, 1.0),  # M+030
        (-30.0, 0.0, 1.0),  # M-030
        (60.0, 0.0, 1.41),  # M+060
        (90.0, 0.0, 1.41),  # M+090
        (110.0, 0.0, 1.41),  # M+110
        (120.0, 0.0, 1.41),  # M+120 (boundary)
        (135.0, 0.0, 1.0),  # M+135
        (180.0, 0.0, 1.0),  # M+180
        (250.0, 0.0, 1.41),  # wraps to |theta| = 110
        (110.0, 35.0, 1.0),  # U+110 (upper layer)
        (90.0, 30.0, 1.0),  # elevation boundary: |phi| < 30 required
        (0.0, -15.0, 1.0),  # B+000 (bottom layer)
        (45.0, 30.0, 1.0),  # U+045
    ],
)
def test_annex3_table4_channel_weight(
    azimuth: float, elevation: float, expected: float
) -> None:
    assert channel_weight(azimuth, elevation) == pytest.approx(expected)


def test_channel_weight_arrays_and_validation() -> None:
    w = channel_weight([0.0, 90.0, 135.0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(w, [1.0, 1.41, 1.0])
    with pytest.raises(ValueError, match="azimuth"):
        channel_weight(400.0)
    with pytest.raises(ValueError, match="azimuth"):
        channel_weight(0.0, 95.0)


def test_lfe_channel_excluded_from_measurement() -> None:
    """A full-scale LFE channel must not change the 5.1 loudness."""
    front = _sine(-23.0, 5.0)
    silent = np.zeros_like(front)
    without_lfe = np.vstack([front, front, silent, silent, silent, silent])
    with_lfe = without_lfe.copy()
    with_lfe[3] = _sine(0.0, 5.0, freq=60.0)
    assert integrated_loudness(with_lfe, FS) == pytest.approx(
        integrated_loudness(without_lfe, FS), abs=1e-9
    )
    assert DEFAULT_CHANNEL_WEIGHTS[6][3] == 0.0


def test_default_weights_and_errors() -> None:
    x = np.zeros((3, FS))
    with pytest.raises(ValueError, match="no default channel weighting"):
        integrated_loudness(x, FS)
    with pytest.raises(ValueError, match="one entry per channel"):
        integrated_loudness(x, FS, weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        integrated_loudness(x, FS, weights=[1.0, -1.0, 1.0])
    # Explicit Annex 3 weights on a 3-channel bed work. Each channel holds
    # half the power of its peak level, so three coherent channels at
    # -23 dBFS sum to 3/2 the stereo reference: -23 + 10 lg 1.5.
    x3 = np.vstack([_sine(-23.0, 5.0)] * 3)
    w = channel_weight([0.0, 30.0, -30.0], [0.0, 0.0, 0.0])
    value = integrated_loudness(x3, FS, weights=np.asarray(w))
    assert value == pytest.approx(-23.0 + 10 * np.log10(1.5), abs=0.1)


# ---------------------------------------------------------------------------
# Gating behaviour and edge cases.
# ---------------------------------------------------------------------------


def test_relative_gate_drops_quiet_passages() -> None:
    """The -10 LU relative gate ignores a long quiet tail (Formulae 5-7)."""
    x = _steps(((-23.0, 10.0), (-50.0, 30.0)))
    res = program_loudness(x, FS)
    # Ungated, the quiet tail would drag the mean far below -23.
    assert res.integrated == pytest.approx(-23.0, abs=EBU_TECH3341_TOL_LU)
    assert res.relative_threshold < -30.0


def test_silence_yields_minus_infinity() -> None:
    res = program_loudness(np.zeros((2, 5 * FS)), FS)
    assert res.integrated == float("-inf")
    assert res.true_peak == float("-inf")
    assert res.loudness_range == 0.0


def test_short_signal_has_no_blocks() -> None:
    res = program_loudness(_stereo(_sine(-23.0, 0.2)), FS)
    assert res.integrated == float("-inf")
    assert res.momentary.size == 0
    assert res.max_momentary == float("-inf")
    assert np.isfinite(res.true_peak)


def test_result_fields_and_series_geometry() -> None:
    x = _steps(((-23.0, 10.0),))
    res = program_loudness(x, FS)
    assert isinstance(res, ProgramLoudnessResult)
    assert res.fs == FS
    np.testing.assert_allclose(res.channel_weights, [1.0, 1.0])
    # Momentary at 100 Hz from t=0.4 s; short-term at 10 Hz from t=3 s.
    assert res.momentary_time[0] == pytest.approx(0.4)
    assert res.momentary_time[1] - res.momentary_time[0] == pytest.approx(0.01)
    assert res.short_term_time[0] == pytest.approx(3.0)
    assert res.short_term_time[1] - res.short_term_time[0] == pytest.approx(0.1)
    assert res.true_peak_per_channel.shape == (2,)
    assert res.true_peak == pytest.approx(-23.0, abs=0.1)
    assert res.lra_low <= res.lra_high
    assert res.loudness_range == pytest.approx(res.lra_high - res.lra_low)
    # A frozen result: attributes cannot be reassigned.
    with pytest.raises(AttributeError):
        res.integrated = 0.0  # type: ignore[misc]


def test_result_rejects_weights_that_miss_a_channel() -> None:
    """The channel axis is the one no reader downstream would flag.

    A weight vector shorter than the channel count renders a complete fiche
    without a word, headlining a true peak for a channel the programme
    loudness never weighted, so the pair is checked where the result is built.
    """
    x = np.vstack([_sine(lvl, 5.0) for lvl in EBU_TECH3341_CASE6_LEVELS])
    res = program_loudness(x, FS)
    assert res.true_peak_per_channel.shape == res.channel_weights.shape
    short = res.channel_weights[:-1]
    with pytest.raises(ValueError, match="'channel_weights'"):
        replace(res, channel_weights=short)


def test_result_rejects_a_maximum_its_own_series_denies() -> None:
    """The headline maxima are the maxima of the series stored beside them.

    No reader recomputes them: the fiche prints each as an informational row
    above a graph of the very readings that would deny it, and the ``bext``
    writer stamps them into a broadcast WAV.
    """
    res = program_loudness(_steps(((-23.0, 10.0),)), FS)
    assert res.max_momentary == pytest.approx(float(np.max(res.momentary)))
    with pytest.raises(ValueError, match="'max_momentary'"):
        replace(res, max_momentary=0.0)
    with pytest.raises(ValueError, match="'max_short_term'"):
        replace(res, max_short_term=0.0)


def test_result_rejects_a_maximum_over_an_empty_series() -> None:
    """An excerpt too short to fill one window holds no maximum at all."""
    res = program_loudness(_stereo(_sine(-23.0, 0.2)), FS)
    assert res.momentary.size == 0
    with pytest.raises(ValueError, match="'max_momentary'"):
        replace(res, max_momentary=-23.0)


def test_maximum_passes_over_readings_left_undetermined() -> None:
    """A window of digital silence reads -inf and the maximum steps over it.

    EBU Tech 3341 leaves both sliding windows ungated, so the -inf readings of
    a silent passage stay in the series; only a wholly silent programme leaves
    the maximum itself undetermined, and the guard accepts that -inf too.
    """
    lead_in = _stereo(np.concatenate([np.zeros(2 * FS), _sine(-23.0, 5.0)]))
    res = program_loudness(lead_in, FS)
    assert np.any(np.isneginf(res.momentary))
    assert res.max_momentary == pytest.approx(-23.0, abs=0.1)
    silent = program_loudness(np.zeros((2, 5 * FS)), FS)
    assert bool(np.all(np.isneginf(silent.momentary)))
    assert silent.max_momentary == float("-inf")
    # A caller who recomputed the maximum along another path is not refused
    # over the last bit of floating point.
    nudged = replace(res, max_momentary=res.max_momentary + 1e-12)
    assert nudged.max_momentary != res.max_momentary


def test_program_loudness_matches_integrated_loudness() -> None:
    x = _steps(((-26.0, 5.0), (-20.0, 5.0)))
    assert program_loudness(x, FS).integrated == pytest.approx(
        integrated_loudness(x, FS), abs=1e-12
    )


def test_mono_input_accepted_as_1d() -> None:
    x = _sine(-23.0, 5.0)
    # Mono: one channel at weight 1.0 carries half the power of the in-phase
    # stereo reference, i.e. 3.01 LU less (the Annex 1 anchor scaled down).
    expected = -23.0 + BS1770_ANCHOR_997_LKFS
    assert integrated_loudness(x, FS) == pytest.approx(expected, abs=0.1)


def test_other_sample_rates_agree_with_48k() -> None:
    """The whole chain holds at 44.1 kHz (redesigned K-weighting)."""
    fs = 44100
    t = np.arange(int(20 * fs)) / fs
    x = 10.0 ** (-23.0 / 20.0) * np.sin(2.0 * np.pi * 1000.0 * t)
    res = program_loudness(np.vstack([x, x]), fs)
    assert res.integrated == pytest.approx(-23.0, abs=EBU_TECH3341_TOL_LU)
    assert res.true_peak == pytest.approx(-23.0, abs=0.2)


def test_program_loudness_validation() -> None:
    empty = np.empty((2, 0))
    tone = _stereo(_sine(-23.0, 1.0))
    with pytest.raises(ValueError, match="empty"):
        program_loudness(empty, FS)
    with pytest.raises(ValueError, match="momentary_step"):
        program_loudness(tone, FS, momentary_step=0.0)
    with pytest.raises(ValueError, match="short_term_step"):
        program_loudness(tone, FS, short_term_step=-1.0)
    # A NaN-poisoned programme must not silently measure as digital silence.
    poisoned = _stereo(_sine(-23.0, 1.0))
    poisoned[0, 100] = np.nan
    with pytest.raises(ValueError, match="finite"):
        program_loudness(poisoned, FS)
    with pytest.raises(ValueError, match="finite"):
        integrated_loudness(poisoned, FS)


def test_update_rate_and_weight_validation() -> None:
    tone = _stereo(_sine(-23.0, 1.0))
    with pytest.raises(ValueError, match="10 Hz"):
        program_loudness(tone, FS, momentary_step=0.2)
    with pytest.raises(ValueError, match="10 Hz"):
        program_loudness(tone, FS, short_term_step=0.5)
    with pytest.raises(ValueError, match="finite"):
        program_loudness(tone, FS, weights=[1.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        channel_weight(np.nan)
    with pytest.raises(ValueError, match="finite"):
        channel_weight(30.0, np.inf)


def test_chunked_inter_sample_peak_matches_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Long programmes are scanned in bounded chunks; the margin far exceeds
    # the interpolator's FIR support, so the piecewise maximum must equal the
    # whole-signal computation exactly.
    from phonometry._internal import peaks

    rng = np.random.default_rng(1770)
    x = np.vstack([rng.standard_normal(20_000), rng.standard_normal(20_000)])
    direct = true_peak_level(x, FS)
    monkeypatch.setattr(peaks, "_CHUNK_SAMPLES", 4096)
    chunked = true_peak_level(x, FS)
    np.testing.assert_array_equal(direct, chunked)


def test_plot_smoke() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    res = program_loudness(_steps(((-30.0, 4.0), (-23.0, 4.0))), FS)
    ax = res.plot()
    assert ax.get_ylabel() == "Loudness [LUFS]"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Integrated" in s for s in labels)
    assert any("LRA" in s for s in labels)
    plt.close("all")

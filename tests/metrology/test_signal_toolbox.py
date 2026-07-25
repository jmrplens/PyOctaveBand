#  Copyright (c) 2026. Jose M. Requena-Plens
"""Tests for the signal toolbox: tone bursts, resampling, fractional delay.

Clean-room oracles, independent of the implementation:

* IEC 60268-1:1985 Clause A2.1 defines the tone burst: it starts at the
  zero crossing of the tone and consists of an integral number of full
  periods. The sample counts of the standard's own test cases (Table AII
  burst durations, Clause A2.2 repetition rates) are hand-computed here,
  and the energy of an integral-period sine gate has the closed form
  ``A²·N/2`` exactly.
* The resampler's anti-alias filter must meet the spec it was designed
  to: passband deviation and stopband leakage bounded by the Kaiser
  ripple ``δ = 10^(-A/20)``, measured on the returned taps themselves.
* A fractional delay multiplies each spectral component's phase by
  ``e^{-j2πfD/fs}``; on a bin-centered tone in circular mode this is
  exact to machine precision, and an integer delay in linear mode reduces
  to an exact sample shift.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phonometry as ph

FS = 48000.0


# ---------------------------------------------------------------------------
# IEC 60268-1 tone bursts
# ---------------------------------------------------------------------------

#: IEC 60268-1 Table AII burst durations at the Clause A2.1 tone frequency
#: of 5 kHz: (duration in ms, full periods). Samples at 48 kHz follow as
#: 48·ms by hand.
_TABLE_AII_BURSTS = [
    (1, 5), (2, 10), (5, 25), (10, 50),
    (20, 100), (50, 250), (100, 500), (200, 1000),
]


@pytest.mark.parametrize(("ms", "cycles"), _TABLE_AII_BURSTS)
def test_table_aii_burst_sample_counts(ms: int, cycles: int) -> None:
    # 5 kHz tone: 5 periods per ms, 48 samples per ms at 48 kHz.
    res = ph.tone_burst(FS, 5000.0, cycles)
    assert res.burst_samples == 48 * ms
    assert res.burst_seconds == pytest.approx(ms / 1000.0)
    assert res.signal.size == res.burst_samples


def test_burst_starts_at_zero_crossing_positive_going() -> None:
    res = ph.tone_burst(FS, 5000.0, 25)
    assert res.signal[0] == 0.0
    assert res.signal[1] > 0.0


def test_integral_periods_energy_closed_form() -> None:
    # Sum of sin² over an integral number of full periods sampled at an
    # integer number of samples per group of periods is exactly N/2.
    amplitude = 0.8
    res = ph.tone_burst(FS, 5000.0, 25, amplitude=amplitude)
    n = res.burst_samples
    energy = float(np.sum(res.signal**2))
    assert energy == pytest.approx(amplitude**2 * n / 2.0, rel=1e-12)
    rms_on = float(np.sqrt(np.mean(res.signal[:n] ** 2)))
    assert rms_on == pytest.approx(amplitude / np.sqrt(2.0), rel=1e-12)


def test_envelope_is_rectangular_gate() -> None:
    res = ph.tone_burst(
        FS, 5000.0, 25, amplitude=2.0, pre_silence=0.01, post_silence=0.02
    )
    n_pre, n_on = round(0.01 * FS), res.burst_samples
    assert res.onset_sample == n_pre
    assert np.all(res.envelope[:n_pre] == 0.0)
    assert np.all(res.envelope[n_pre:n_pre + n_on] == 2.0)
    assert np.all(res.envelope[n_pre + n_on:] == 0.0)
    assert np.all(res.signal[:n_pre] == 0.0)
    assert np.all(res.signal[n_pre + n_on:] == 0.0)
    assert np.all(np.abs(res.signal) <= res.envelope)
    assert res.signal.size == n_pre + n_on + round(0.02 * FS)


def test_repetitive_burst_train_clause_a22() -> None:
    # Clause A2.2: 5 ms bursts of 5 kHz tone; 10 bursts per second gives a
    # 4800-sample period at 48 kHz and a 5 % duty cycle.
    res = ph.tone_burst(FS, 5000.0, 25, repetitions=3, repetition_rate=10.0)
    assert res.burst_samples == 240
    assert res.period_samples == 4800
    assert res.duty_cycle == pytest.approx(0.05)
    assert res.signal.size == 3 * 4800
    period, n_on = 4800, 240
    first = res.signal[:period]
    for k in range(1, 3):
        np.testing.assert_array_equal(
            res.signal[k * period:(k + 1) * period], first
        )
    assert np.all(first[n_on:] == 0.0)


@pytest.mark.parametrize("rate", [2.0, 10.0, 100.0])
def test_table_aiii_repetition_rates_fit(rate: float) -> None:
    res = ph.tone_burst(FS, 5000.0, 25, repetitions=2, repetition_rate=rate)
    assert res.period_samples == round(FS / rate)


def test_tone_burst_incommensurate_frequency_warns() -> None:
    """997 Hz at 48 kHz: 10 periods span 481.44 samples, gated at 481.

    The 'integral number of full periods' of Clause A2.1 is sample-exact
    only for commensurate f/fs; otherwise a warning quantifies the
    residual step at the gate edge.
    """
    with pytest.warns(ph.PhonometryWarning, match="incommensurate") as record:
        res = ph.tone_burst(FS, 997.0, 10)
    assert res.burst_samples == 481
    # The warning quantifies the exact span and the residual step at the
    # gate edge, sin(2*pi*997*481/48000) ~ -0.058 (a commensurate burst
    # would close exactly at the zero crossing).
    message = str(record[0].message)
    assert "481.444" in message
    assert "-0.058" in message


def test_tone_burst_invalid_config_raises_before_warning() -> None:
    """An invalid repetition setup raises even for incommensurate bursts.

    Under a warnings-as-errors filter the incommensurate warning must not
    pre-empt the configuration ValueError.
    """
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", ph.PhonometryWarning)
        with pytest.raises(ValueError, match="repetition_rate"):
            ph.tone_burst(FS, 997.0, 10, repetitions=2)


def test_tone_burst_commensurate_frequency_does_not_warn() -> None:
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", ph.PhonometryWarning)
        res = ph.tone_burst(FS, 5000.0, 25)  # 240 samples exactly
    assert res.burst_samples == 240


def test_tone_burst_result_is_frozen() -> None:
    res = ph.tone_burst(FS, 5000.0, 25)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.cycles = 3  # type: ignore[misc]


def test_tone_burst_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ph.tone_burst(FS, 5000.0, 0)
    with pytest.raises(ValueError, match="Nyquist"):
        ph.tone_burst(FS, 24000.0, 10)
    with pytest.raises(ValueError, match="repetition_rate"):
        ph.tone_burst(FS, 5000.0, 25, repetitions=2)
    with pytest.raises(ValueError, match="does not fit"):
        ph.tone_burst(FS, 5000.0, 25, repetition_rate=300.0)
    with pytest.raises(ValueError, match="non-negative"):
        ph.tone_burst(FS, 5000.0, 25, pre_silence=-1.0)
    with pytest.raises(ValueError, match="positive, finite"):
        ph.tone_burst(FS, 5000.0, 25, amplitude=0.0)


def test_tone_burst_plot_waveform_and_envelope() -> None:
    res = ph.tone_burst(FS, 5000.0, 25, repetitions=2, repetition_rate=10.0)
    ax = res.plot(linewidth=2)
    assert any(line.get_linewidth() == 2.0 for line in ax.lines)
    assert "IEC 60268-1" in ax.get_title()
    assert "10/s" in ax.get_title()
    plt.close("all")
    ax = ph.tone_burst(FS, 5000.0, 25).plot(color="red")
    red = plt.matplotlib.colors.to_rgba("red")
    assert any(
        plt.matplotlib.colors.to_rgba(line.get_color()) == red
        for line in ax.lines
    )
    plt.close("all")


# ---------------------------------------------------------------------------
# Resampling with an explicit anti-alias specification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fs", "fs_new", "atten", "tw"),
    [
        (48000.0, 32000.0, 100.0, 0.1),
        (44100.0, 48000.0, 80.0, 0.2),
        (48000.0, 96000.0, 120.0, 0.1),
    ],
)
def test_designed_filter_meets_its_own_spec(
    fs: float, fs_new: float, atten: float, tw: float
) -> None:
    # The oracle is the design spec itself, measured on the returned taps:
    # passband flat within δ and stopband below -A dB with δ = 10^(-A/20).
    from scipy import signal as sp_signal

    x = ph.noise_signal(fs, 0.2, seed=5)
    res = ph.resample_signal(
        x, fs, fs_new, stopband_attenuation_db=atten, transition_width=tw
    )
    fs_up = res.original_fs * res.up
    freqs, h = sp_signal.freqz(res.filter_taps, worN=1 << 19, fs=fs_up)
    mag = np.abs(h)
    delta = 10.0 ** (-atten / 20.0)
    passband = freqs <= res.passband_edge_hz
    stopband = freqs >= res.stopband_edge_hz
    assert float(np.max(np.abs(mag[passband] - 1.0))) <= delta
    assert float(np.max(mag[stopband])) <= delta
    assert res.stopband_edge_hz == pytest.approx(min(fs, fs_new) / 2.0)
    assert res.passband_edge_hz == pytest.approx((1.0 - tw) * min(fs, fs_new) / 2.0)


def test_resampled_tone_matches_analytic_tone() -> None:
    # A passband tone resampled 48 kHz -> 32 kHz must equal the analytic
    # tone at the new rate within the design ripple δ (edges excluded).
    fs, fs_new, atten = 48000.0, 32000.0, 120.0
    t = np.arange(int(fs * 0.5)) / fs
    tone = np.sin(2.0 * np.pi * 1000.0 * t)
    res = ph.resample_signal(tone, fs, fs_new, stopband_attenuation_db=atten)
    t_new = np.arange(res.signal.size) / fs_new
    exact = np.sin(2.0 * np.pi * 1000.0 * t_new)
    edge = res.n_taps // res.up + 1
    err = np.max(np.abs(res.signal[edge:-edge] - exact[edge:-edge]))
    assert err <= 10.0 ** (-atten / 20.0)


def test_rational_ratio_and_output_length() -> None:
    x = ph.noise_signal(44100.0, 0.1, seed=2)
    res = ph.resample_signal(x, 44100.0, 48000.0)
    assert (res.up, res.down) == (160, 147)
    assert res.signal.size == int(np.ceil(x.size * 160 / 147))
    assert res.fs == 48000.0
    assert res.original_fs == 44100.0


def test_identity_ratio_returns_copy_without_filtering() -> None:
    x = ph.noise_signal(FS, 0.05, seed=3)
    res = ph.resample_signal(x, FS, FS)
    np.testing.assert_array_equal(res.signal, x)
    assert (res.up, res.down) == (1, 1)
    assert res.n_taps == 1


def test_irrational_ratio_raises() -> None:
    x = ph.noise_signal(FS, 0.05, seed=4)
    fs_irrational = FS * np.sqrt(2.0)
    with pytest.raises(ValueError, match="rational"):
        ph.resample_signal(x, FS, fs_irrational)


def test_resample_invalid_parameters() -> None:
    x = ph.noise_signal(FS, 0.05, seed=4)
    two_dimensional = np.zeros((4, 4))
    with pytest.raises(ValueError, match="at least 30"):
        ph.resample_signal(x, FS, 32000.0, stopband_attenuation_db=10.0)
    with pytest.raises(ValueError, match="transition_width"):
        ph.resample_signal(x, FS, 32000.0, transition_width=0.9)
    with pytest.raises(ValueError, match="one-dimensional"):
        ph.resample_signal(two_dimensional, FS, 32000.0)


# ---------------------------------------------------------------------------
# Fractional delay
# ---------------------------------------------------------------------------


def test_circular_delay_of_bin_centered_tone_is_machine_exact() -> None:
    n, delay = 4096, 3.37
    k = 300
    f = k * FS / n
    m = np.arange(n)
    x = np.cos(2.0 * np.pi * f * m / FS + 0.4)
    y = ph.fractional_delay(x, delay, mode="circular")
    exact = np.cos(2.0 * np.pi * f * (m - delay) / FS + 0.4)
    np.testing.assert_allclose(y, exact, atol=1e-11)


def test_phase_slope_is_exactly_minus_2pi_f_d_over_fs() -> None:
    # Multi-tone record on exact bins: after a delay of D samples every
    # component's phase must change by -2π·f·D/fs, wrapped, to machine
    # precision.
    n, delay = 4096, 7.25
    bins = np.array([50, 300, 901, 1500])
    m = np.arange(n)
    x = np.zeros(n)
    for k in bins:
        x += np.cos(2.0 * np.pi * k * m / n + 0.1 * k)
    y = ph.fractional_delay(x, delay, mode="circular")
    spectrum_x = np.fft.rfft(x)
    spectrum_y = np.fft.rfft(y)
    measured = np.angle(spectrum_y[bins] / spectrum_x[bins])
    freqs = bins * FS / n
    expected = -2.0 * np.pi * freqs * delay / FS
    wrapped = (expected + np.pi) % (2.0 * np.pi) - np.pi
    np.testing.assert_allclose(measured, wrapped, atol=1e-10)


def test_linear_integer_delay_is_exact_sample_shift() -> None:
    x = ph.noise_signal(FS, 0.1, seed=7)
    y = ph.fractional_delay(x, 5.0)
    expected = np.concatenate([np.zeros(5), x[:-5]])
    np.testing.assert_allclose(y, expected, atol=1e-13)


def test_negative_delay_advances() -> None:
    x = ph.noise_signal(FS, 0.1, seed=8)
    y = ph.fractional_delay(x, -3.0)
    expected = np.concatenate([x[3:], np.zeros(3)])
    np.testing.assert_allclose(y, expected, atol=1e-13)


def test_linear_mode_is_bit_identical_to_alignment_kernel() -> None:
    # align_impulse_responses removes delays with the same kernel; the
    # public function must reproduce it bit for bit (advance = -delay).
    from phonometry.metrology.correlation import _fractional_advance

    x = ph.noise_signal(FS, 0.1, seed=9)
    shift = 4.6180339887
    np.testing.assert_array_equal(
        ph.fractional_delay(x, -shift), _fractional_advance(x, shift)
    )


def test_circular_roundtrip_recovers_record() -> None:
    # Odd length: the rfft has no Nyquist bin, so the circular ramp is
    # exactly invertible for any record.
    x = ph.noise_signal(FS, 0.05, seed=10)[:2399]
    y = ph.fractional_delay(
        ph.fractional_delay(x, 2.5, mode="circular"), -2.5, mode="circular"
    )
    np.testing.assert_allclose(y, x, atol=1e-12)


def test_fractional_delay_invalid_inputs() -> None:
    x = ph.noise_signal(FS, 0.05, seed=11)
    full_length = float(x.size)
    two_dimensional = np.zeros((2, 8))
    with pytest.raises(ValueError, match="mode"):
        ph.fractional_delay(x, 1.0, mode="wrap")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="magnitude"):
        ph.fractional_delay(x, full_length)
    with pytest.raises(ValueError, match="finite"):
        ph.fractional_delay(x, np.nan)
    with pytest.raises(ValueError, match="one-dimensional"):
        ph.fractional_delay(two_dimensional, 1.0)

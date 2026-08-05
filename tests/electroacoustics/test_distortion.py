#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for electroacoustic distortion metrics (IEC 60268-3 / AES17).

Every metric has an exact analytic oracle: a signal synthesised with known
harmonic or intermodulation amplitudes reproduces the closed-form ratio. Tones
are placed on FFT bins (coherent sampling) so the FFT reads their amplitudes
without leakage.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import reference_data as ref

from phonometry import (
    HarmonicDistortionResult,
    difference_frequency_distortion,
    dynamic_intermodulation_distortion,
    dynamic_range,
    harmonic_analysis,
    harmonic_distortion,
    idle_channel_noise,
    modulation_distortion,
    sinad,
    thd,
    thd_plus_noise,
    total_difference_frequency_distortion,
    weighted_thd,
)

FS = 48000
N = 48000  # 1 s -> 1 Hz bin resolution, every test tone lands on a bin.
_T = np.arange(N) / FS


def _tone(freq: float, amp: float = 1.0) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * _T)


def _harmonic_signal() -> np.ndarray:
    a1, a2, a3, a4 = ref.DISTORTION_HARMONICS
    return (
        _tone(1000.0, a1)
        + _tone(2000.0, a2)
        + _tone(3000.0, a3)
        + _tone(4000.0, a4)
    )


def test_thd_f_matches_closed_form() -> None:
    x = _harmonic_signal()
    assert thd(x, FS, 1000.0, kind="F") == pytest.approx(ref.DISTORTION_THD_F, rel=1e-6)


def test_thd_r_matches_closed_form() -> None:
    x = _harmonic_signal()
    assert thd(x, FS, 1000.0, kind="R") == pytest.approx(ref.DISTORTION_THD_R, rel=1e-6)


def test_thd_auto_detects_fundamental() -> None:
    x = _harmonic_signal()
    assert thd(x, FS) == pytest.approx(ref.DISTORTION_THD_F, rel=1e-6)


def test_nth_order_harmonic_distortion() -> None:
    x = _harmonic_signal()
    assert harmonic_distortion(x, FS, 1000.0, 2) == pytest.approx(
        ref.DISTORTION_D2, rel=1e-6
    )


def test_harmonic_distortion_rejects_order_below_two() -> None:
    with pytest.raises(ValueError):
        harmonic_distortion(_tone(1000.0), FS, 1000.0, 1)


def test_thd_plus_noise_recovers_noise_floor() -> None:
    # Fundamental (RMS 1/sqrt2) + white noise of RMS 0.01; the notch removes
    # the fundamental, so THD+N ~ in-band noise_rms / total_rms with the
    # AES17 measurement bandwidth (20 Hz - 20 kHz) applied to both.
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(N) * 0.01
    x = _tone(1000.0) + noise
    band_fraction = (20000.0 - 20.0) / (FS / 2.0)
    expected = 0.01 * np.sqrt(band_fraction) / np.sqrt(0.5)
    # The notch also removes a small noise band, so the measured value is a
    # little below the broadband ratio.
    assert thd_plus_noise(x, FS, 1000.0) == pytest.approx(expected, rel=0.1)
    # bandwidth=None restores the full-Nyquist measurement.
    full = thd_plus_noise(x, FS, 1000.0, bandwidth=None)
    assert full == pytest.approx(0.01 / np.sqrt(0.5), rel=0.1)
    assert full > thd_plus_noise(x, FS, 1000.0)


def test_thd_plus_noise_aes17_bandwidth_at_high_fs() -> None:
    # AES17 5.2.5/6.3.1: at fs = 192 kHz the out-of-band noise (20-96 kHz)
    # must not count. Full-Nyquist reads ~+6.8 dB above the band-limited
    # value for white noise (10*lg(95980/19980)).
    fs = 192000
    t = np.arange(fs) / fs
    rng = np.random.default_rng(7)
    x = np.sin(2 * np.pi * 1000.0 * t) + rng.standard_normal(fs) * 0.01
    banded = thd_plus_noise(x, fs, 1000.0)
    full = thd_plus_noise(x, fs, 1000.0, bandwidth=None)
    expected_banded = 0.01 * np.sqrt((20000.0 - 20.0) / (fs / 2.0)) / np.sqrt(0.5)
    assert banded == pytest.approx(expected_banded, rel=0.1)
    gain_db = 20.0 * np.log10(full / banded)
    assert gain_db == pytest.approx(
        10.0 * np.log10((fs / 2.0 - 20.0) / (20000.0 - 20.0)), abs=1.0
    )


def test_thd_plus_noise_dc_offset_not_counted() -> None:
    # A DC offset is outside the 20 Hz - 20 kHz measurement bandwidth and
    # must not be counted as noise (it previously produced THD+N = 0.577
    # for a 0.5 offset on a unit tone).
    x = _tone(1000.0) + 0.5
    assert thd_plus_noise(x, FS, 1000.0) < 1e-3


def test_sinad_is_negative_thd_plus_noise_db() -> None:
    rng = np.random.default_rng(1)
    x = _tone(1000.0) + rng.standard_normal(N) * 0.01
    thdn_db = thd_plus_noise(x, FS, 1000.0, as_db=True)
    assert sinad(x, FS, 1000.0) == pytest.approx(-thdn_db)


def test_thd_plus_noise_pure_tone_is_tiny() -> None:
    # A pure fundamental has no distortion or noise; THD+N ~ 0.
    assert thd_plus_noise(_tone(1000.0), FS, 1000.0) < 1e-3


def test_thd_plus_noise_rejects_q_out_of_range() -> None:
    with pytest.raises(ValueError):
        thd_plus_noise(_tone(1000.0), FS, 1000.0, notch_q=5.0)


def test_weighted_thd_attenuates_low_harmonics() -> None:
    # A-weighting attenuates a 100 Hz fundamental's 200 Hz harmonic relative to
    # the unweighted residual, so weighted THD < the plain harmonic ratio.
    x = _tone(100.0) + _tone(200.0, 0.1)
    plain = thd(x, FS, 100.0, kind="R")
    weighted = weighted_thd(x, FS, 100.0, weighting="A")
    assert 0.0 < weighted < plain


def test_weighted_thd_rejects_bad_notch_q() -> None:
    with pytest.raises(ValueError):
        weighted_thd(_tone(1000.0), FS, 1000.0, notch_q=5.0)


def test_thd_plus_noise_rejects_fundamental_above_nyquist() -> None:
    # A fundamental at/above fs/2 cannot be notched (iirnotch would fail).
    with pytest.raises(ValueError):
        thd_plus_noise(_tone(1000.0), FS, FS / 2.0)


def test_modulation_distortion_iec_per_order() -> None:
    # IEC 60268-3 14.12.7.2 g)-h): carrier f_high = 8 kHz (amp 0.25),
    # modulator f_low = 250 Hz; 2nd-order sidebands at f_high +/- f_low
    # (0.02 each), 3rd-order at f_high +/- 2 f_low (0.01 each). Per-order
    # values are ARITHMETIC sideband sums over the f_high amplitude:
    # d_m,2 = (0.02+0.02)/0.25 = 0.16, d_m,3 = (0.01+0.01)/0.25 = 0.08.
    fl, fh, ah = 250.0, 8000.0, 0.25
    x = (
        _tone(fl)
        + _tone(fh, ah)
        + _tone(fh + fl, 0.02)
        + _tone(fh - fl, 0.02)
        + _tone(fh + 2 * fl, 0.01)
        + _tone(fh - 2 * fl, 0.01)
    )
    res = modulation_distortion(x, FS, fl, fh)
    assert res.d2 == pytest.approx(0.16, rel=1e-6)
    assert res.d3 == pytest.approx(0.08, rel=1e-6)
    # The SMPTE-analyzer combined RMS convention is kept, explicitly labelled.
    smpte = math.sqrt(0.02**2 + 0.02**2 + 0.01**2 + 0.01**2) / ah
    assert res.smpte == pytest.approx(smpte, rel=1e-6)


def test_modulation_distortion_carries_sideband_spectrum() -> None:
    # The result carries the measured carrier and sideband amplitudes (the
    # data behind .plot()), in ascending frequency order:
    # f2-2f1, f2-f1, f2+f1, f2+2f1.
    fl, fh, ah = 250.0, 8000.0, 0.25
    x = (
        _tone(fl)
        + _tone(fh, ah)
        + _tone(fh + fl, 0.02)
        + _tone(fh - fl, 0.02)
        + _tone(fh + 2 * fl, 0.01)
        + _tone(fh - 2 * fl, 0.01)
    )
    res = modulation_distortion(x, FS, fl, fh)
    assert res.f_low == fl and res.f_high == fh
    assert res.carrier_amplitude == pytest.approx(ah, rel=1e-6)
    np.testing.assert_allclose(
        res.sideband_frequencies,
        [fh - 2 * fl, fh - fl, fh + fl, fh + 2 * fl],
    )
    np.testing.assert_allclose(
        res.sideband_amplitudes, [0.01, 0.02, 0.02, 0.01], rtol=1e-6
    )


def test_modulation_distortion_plot_marks_carrier_and_sidebands() -> None:
    import matplotlib.pyplot as plt

    from phonometry import ModulationDistortionResult

    fl, fh = 250.0, 8000.0
    x = (
        _tone(fl)
        + _tone(fh, 0.25)
        + _tone(fh + fl, 0.02)
        + _tone(fh - fl, 0.02)
        + _tone(fh + 2 * fl, 0.01)
        + _tone(fh - 2 * fl, 0.01)
    )
    res = modulation_distortion(x, FS, fl, fh)
    ax = res.plot()
    # The carrier marker sits at (f_high, 0 dB); the four sidebands read
    # 20*lg(a_s / carrier) re the carrier.
    xs = np.concatenate([line.get_xdata() for line in ax.lines])
    assert fh in xs and fh - 2 * fl in xs and fh + 2 * fl in xs
    sb_line = ax.lines[-1]
    np.testing.assert_allclose(
        sb_line.get_ydata(),
        20.0 * np.log10(np.array([0.01, 0.02, 0.02, 0.01]) / 0.25),
        atol=1e-6,
    )
    assert "d₂" in ax.get_title() and "SMPTE" in ax.get_title()
    plt.close("all")
    # Spanish labels and the unknown-language rejection.
    ax_es = res.plot(language="es")
    assert ax_es.get_ylabel() == "Nivel respecto a la portadora [dB]"
    plt.close("all")
    with pytest.raises(ValueError):
        res.plot(language="xx")
    # A hand-built result without the spectral fields cannot be plotted.
    bare = ModulationDistortionResult(d2=0.1, d3=0.05, smpte=0.08)
    with pytest.raises(ValueError, match="no sideband spectrum"):
        bare.plot()
    plt.close("all")


def test_difference_frequency_distortion_iec() -> None:
    # IEC 60268-3 14.12.8.1: equal tones f1 = 13 kHz, f2 = 14 kHz (amp 0.5);
    # 2nd-order product at f2 - f1 = 1 kHz (0.03); 3rd-order at 2f1 - f2 and
    # 2f2 - f1 (0.02 each). The reference is U_2,ref = 2*U_2,f2 (the sum of
    # both tone amplitudes = 1.0) and the 3rd order sums ARITHMETICALLY:
    # d_d,2 = 0.03/1.0, d_d,3 = (0.02+0.02)/1.0.
    f1, f2 = 13000.0, 14000.0
    x = (
        _tone(f1, 0.5)
        + _tone(f2, 0.5)
        + _tone(f2 - f1, 0.03)
        + _tone(2 * f1 - f2, 0.02)
        + _tone(2 * f2 - f1, 0.02)
    )
    assert difference_frequency_distortion(x, FS, f1, f2, order=2) == pytest.approx(
        0.03, rel=1e-6
    )
    assert difference_frequency_distortion(x, FS, f1, f2, order=3) == pytest.approx(
        0.04, rel=1e-6
    )


def test_total_difference_frequency_distortion_iec() -> None:
    # IEC 60268-3 14.12.10: the standard tones f1 = 8 kHz, f2 = 11.95 kHz
    # (f0 = 4 kHz, delta = 50 Hz), products U' at f2-f1 = 3950 Hz (0.02) and
    # 2f1-f2 = 4050 Hz (0.03), tones 0.5 each. Only these two in-band
    # products enter, rms-summed over the tone-amplitude sum:
    # d_TDFD = sqrt(0.02^2 + 0.03^2) / (0.5 + 0.5) = sqrt(0.0013).
    f1, f2 = 8000.0, 11950.0
    x = (
        _tone(f1, 0.5)
        + _tone(f2, 0.5)
        + _tone(f2 - f1, 0.02)
        + _tone(2 * f1 - f2, 0.03)
        + _tone(2 * f2 - f1, 0.05)  # out-of-band product: must NOT count
    )
    expected = 0.03605551275463989  # sqrt(0.0013), exact
    assert total_difference_frequency_distortion(x, FS) == pytest.approx(
        expected, rel=1e-6
    )
    # Explicit tone arguments give the same value as the standard defaults.
    assert total_difference_frequency_distortion(x, FS, f1, f2) == pytest.approx(
        expected, rel=1e-6
    )


def test_difference_frequency_clean_octave_tones_read_zero() -> None:
    # Search-window hygiene: with octave-spaced clean tones the old
    # half-difference window latched a primary tone and reported d2 = 1.0.
    # A clean signal must read zero for both orders and both spacings.
    for f1, f2 in ((1000.0, 2000.0), (1000.0, 3000.0)):
        x = _tone(f1, 0.5) + _tone(f2, 0.5)
        # Numerical floor only (window leakage of the FFT at ~1e-16).
        assert difference_frequency_distortion(x, FS, f1, f2, order=2) < 1e-12
        assert difference_frequency_distortion(x, FS, f1, f2, order=3) < 1e-12


def test_difference_frequency_dc_offset_not_counted() -> None:
    # 2f1 - f2 <= 0 clamps to zero and a DC offset must not leak into d3
    # (the old window at negative/zero product frequencies included bin 0).
    f1, f2 = 1000.0, 2500.0
    x = _tone(f1, 0.5) + _tone(f2, 0.5) + 0.5
    assert difference_frequency_distortion(x, FS, f1, f2, order=3) < 1e-12
    assert difference_frequency_distortion(x, FS, f1, f2, order=2) < 1e-12


def test_difference_frequency_rejects_bad_args() -> None:
    tone = _tone(1000.0)
    with pytest.raises(ValueError):
        difference_frequency_distortion(tone, FS, 2000.0, 1000.0)
    with pytest.raises(ValueError):
        difference_frequency_distortion(tone, FS, 1000.0, 2000.0, order=4)
    with pytest.raises(ValueError):
        total_difference_frequency_distortion(tone, FS, 2000.0, 1000.0)


def test_dynamic_intermodulation_distortion() -> None:
    # IEC 60268-3 Table 2: the standard 15 kHz / 3.15 kHz signal has NINE
    # difference products at |k*3150 - 15000| < 15000 for k = 1..9, spanning
    # 750 Hz (k=5) to 13350 Hz (k=9). Enumerated here from the standard, not
    # from the implementation, so an off-by-one in the loop bound is caught.
    fsine, fsq = 15000.0, 3150.0
    comps = sorted(
        round(abs(k * fsq - fsine), 6) for k in range(1, 10) if abs(k * fsq - fsine) < fsine
    )
    assert comps == pytest.approx(
        [750.0, 2400.0, 3900.0, 5550.0, 7050.0, 8700.0, 10200.0, 11850.0, 13350.0]
    )
    amps = [0.01 * (i + 1) for i in range(len(comps))]
    # The captured DIM signal ALSO carries a strong 3.15 kHz square-wave
    # fundamental (0.8 here; in the standard signal the square wave dominates
    # the sine 4:1 peak-to-peak); it must not be mistaken for a product even
    # though it sits 750 Hz from the k=4 (2400) and k=6 (3900) products.
    # DIM = sqrt(sum products^2) / sine_amp, excluding the 3150 tone.
    x = _tone(fsine) + _tone(fsq, 0.8)
    for c, a in zip(comps, amps):
        x = x + _tone(c, a)
    expected = math.sqrt(sum(a**2 for a in amps))
    assert dynamic_intermodulation_distortion(x, FS) == pytest.approx(
        expected, rel=1e-6
    )


def test_harmonic_analysis_bundle_and_plot() -> None:
    res = harmonic_analysis(_harmonic_signal(), FS, 1000.0)
    assert isinstance(res, HarmonicDistortionResult)
    assert res.fundamental == pytest.approx(1000.0)
    assert res.thd_f == pytest.approx(ref.DISTORTION_THD_F, rel=1e-6)
    assert res.thd_r == pytest.approx(ref.DISTORTION_THD_R, rel=1e-6)
    assert res.harmonic_amplitudes[0] == pytest.approx(1.0, rel=1e-6)
    ax = res.plot()
    assert ax is not None


@pytest.mark.parametrize(
    "func", [thd, thd_plus_noise, lambda s, fs: sinad(s, fs)]
)
def test_rejects_non_finite_signal(func) -> None:  # type: ignore[no-untyped-def]
    bad = np.array([np.nan] * 100)
    with pytest.raises(ValueError):
        func(bad, FS)


def test_rejects_bad_fs_and_kind() -> None:
    with pytest.raises(ValueError):
        thd(_tone(1000.0), 0.0)
    with pytest.raises(ValueError):
        thd(_tone(1000.0), FS, 1000.0, kind="X")  # type: ignore[arg-type]


def test_rejects_too_short_signal() -> None:
    with pytest.raises(ValueError):
        thd(np.array([0.0, 1.0, 0.0]), FS)


# ---------------------------------------------------------------------------
# AES17 notch, weighting and clipped-sine oracles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [1.2, 2.0, 3.0])
def test_notch_effective_q_matches_request(q: float) -> None:
    # AES17-2015 5.2.8 defines Q on the APPLIED (combined) response. filtfilt
    # squares the magnitude, so the single-pass design is sharpened by
    # sqrt(1 + sqrt(2)); the effective Q (f0 over the -3 dB width of the
    # squared response) must equal the request. Without the compensation a
    # nominal 2.0 acted as ~1.29 and 1.2 fell outside the AES17 range.
    from scipy import signal as sp

    from phonometry.electroacoustics.distortion import _FILTFILT_NOTCH_Q_FACTOR

    f0 = 997.0
    b, a = sp.iirnotch(f0, q * _FILTFILT_NOTCH_Q_FACTOR, FS)
    freqs = np.linspace(f0 - 800.0, f0 + 800.0, 200001)
    _, h = sp.freqz(b, a, worN=freqs, fs=FS)
    applied = np.abs(h) ** 2  # zero-phase filtering applies |H| twice
    minus3 = 10.0 ** (-3.0 / 20.0)
    inside = freqs[applied <= minus3]
    q_eff = f0 / (inside.max() - inside.min())
    assert q_eff == pytest.approx(q, rel=0.02)


def test_itu_r_468_weighting_table_values() -> None:
    # ITU-R BS.468-4 Table 1 nominal response, including the +12.2 dB peak
    # at 6.3 kHz; interpolation between mask points is linear in dB over
    # log-frequency per the recommendation's own tolerance rule.
    from phonometry import itu_r_468_weighting

    pins = {
        31.5: -29.9, 200.0: -13.8, 1000.0: 0.0, 2000.0: 5.6, 5000.0: 11.7,
        6300.0: 12.2, 10000.0: 8.1, 20000.0: -22.2, 31500.0: -42.7,
    }
    for f, db in pins.items():
        assert itu_r_468_weighting([f])[0] == pytest.approx(db, abs=1e-9)
    # The 6.3 kHz table point is the curve's maximum.
    grid = np.geomspace(20.0, 31500.0, 20000)
    curve = itu_r_468_weighting(grid)
    assert float(np.max(curve)) == pytest.approx(12.2, abs=1e-3)
    assert grid[int(np.argmax(curve))] == pytest.approx(6300.0, rel=1e-3)
    # DC blocks completely; negative/non-finite frequencies are rejected.
    assert itu_r_468_weighting([0.0])[0] == -np.inf
    with pytest.raises(ValueError):
        itu_r_468_weighting([-100.0])


def test_itu_r_468_matches_aes17_ccir_rms_table() -> None:
    # AES17-2015 Table 1 prints the same curve shifted by -5.63 dB (the
    # CCIR-RMS filter, unity at 2 kHz); cross-check a few rows within the
    # 0.05 dB rounding of the AES17 print.
    from phonometry import itu_r_468_weighting

    aes17 = {63.0: -29.5, 3150.0: 3.4, 8000.0: 5.8, 12500.0: -5.6}
    for f, db in aes17.items():
        assert itu_r_468_weighting([f])[0] - 5.63 == pytest.approx(db, abs=0.05)


def test_weighted_thd_468_emphasises_6khz_products() -> None:
    # IEC 60268-3 14.12.11 requires the IEC 60268-1 (ITU-R 468) network: a
    # distortion product near the +12.2 dB peak is emphasised accordingly,
    # where A-weighting leaves it nearly unchanged (~+0.1 dB).
    x = _tone(100.0) + _tone(6300.0, 0.01)
    w468 = weighted_thd(x, FS, 100.0)  # default weighting="468"
    assert w468 == pytest.approx(0.01 * 10.0 ** (12.2 / 20.0), rel=0.02)
    w_a = weighted_thd(x, FS, 100.0, weighting="A")
    assert w468 / w_a == pytest.approx(10.0 ** (12.2 / 20.0), rel=0.05)
    with pytest.raises(ValueError):
        weighted_thd(x, FS, 100.0, weighting="B")  # type: ignore[arg-type]


def test_thd_clipped_sine_fourier_oracle() -> None:
    # Oracle: unit sine symmetrically clipped at 0.7 (48 samples/period);
    # odd-harmonic amplitudes and THD_F from an independent single-period
    # Fourier series of the sampled waveform (reference_data/).
    x = np.clip(_tone(1000.0), -0.7, 0.7)
    assert thd(x, FS, 1000.0) == pytest.approx(ref.CLIPPED_SINE_THD_F, rel=1e-9)
    res = harmonic_analysis(x, FS, 1000.0)
    b = res.harmonic_amplitudes
    assert b[0] == pytest.approx(ref.CLIPPED_SINE_B1, rel=1e-9)
    assert b[2] == pytest.approx(ref.CLIPPED_SINE_B3, rel=1e-9)
    assert b[4] == pytest.approx(ref.CLIPPED_SINE_B5, rel=1e-9)
    assert b[6] == pytest.approx(ref.CLIPPED_SINE_B7, rel=1e-9)
    assert b[8] == pytest.approx(ref.CLIPPED_SINE_B9, rel=1e-9)
    # Even harmonics of a symmetric clip vanish.
    assert b[1] < 1e-12 and b[3] < 1e-12


def test_thd_raises_when_no_harmonic_below_nyquist() -> None:
    # A 20 kHz fundamental at fs = 48 kHz has its 2nd harmonic above
    # Nyquist: the THD is undefined and must raise, not return 0.
    tone_hf = _tone(20000.0)
    with pytest.raises(ValueError, match="Nyquist"):
        thd(tone_hf, FS, 20000.0)


def test_dim_full_signal_regression() -> None:
    # End-to-end DIM oracle: the standard test signal is synthesised at
    # 1.536 MHz (3.15 kHz square through a single-pole 30 kHz low-pass,
    # 4:1 peak-to-peak against the 15 kHz sine), passed through the weak
    # nonlinearity y = x + 0.01 x^2 + 0.002 x^3, FIR-decimated x8 to
    # 192 kHz, and the module value is checked against an exact-bin FFT
    # oracle on the same capture.
    from scipy import signal as sp

    fs_hi = 1_536_000
    t = np.arange(fs_hi) / fs_hi
    square = np.sign(np.sin(2 * np.pi * 3150.0 * t))
    b, a = sp.butter(1, 30000.0, fs=fs_hi)
    sq = sp.lfilter(b, a, square)
    sine = np.sin(2 * np.pi * 15000.0 * t)
    sq *= 4.0 * np.ptp(sine) / np.ptp(sq)
    x = sq + sine
    y = x + 0.01 * x**2 + 0.002 * x**3
    fs = 192000
    y192 = sp.resample_poly(y, 1, 8)
    # Exact-bin oracle: 1 s at 192 kHz -> 1 Hz bins; both tones and all
    # Table 2 products land on bins (rectangular window, no leakage).
    spec = np.abs(np.fft.rfft(y192)) * 2.0 / y192.size
    products = [750, 2400, 3900, 5550, 7050, 8700, 10200, 11850, 13350]
    oracle = float(np.sqrt(sum(spec[p] ** 2 for p in products)) / spec[15000])
    value = dynamic_intermodulation_distortion(y192, fs)
    assert abs(value - oracle) < 1e-3
    assert oracle > 0.01  # the nonlinearity produces measurable DIM


# --------------------------------------------------------------------------- #
# AES17-2015 6.4 noise measurements (dynamic range, idle channel noise)
# --------------------------------------------------------------------------- #

def _dbfs_sine_amplitude(dbfs: float) -> float:
    """Peak amplitude of a sine at ``dbfs`` dBFS (0 dBFS = full-scale sine)."""
    return 10.0 ** (dbfs / 20.0)


def test_idle_channel_noise_1khz_offset_closed_form() -> None:
    # AES17 6.4.2: the CCIR-RMS weighting (5.2.7) is the 468 curve with a flat
    # -5.63 dB offset, and the 468 curve is 0 dB at 1 kHz. A 1 kHz sine at
    # -20 dBFS therefore reads exactly -20 - 5.63 = -25.63 dBFS CCIR-RMS.
    sig = _dbfs_sine_amplitude(-20.0) * _tone(1000.0)
    assert idle_channel_noise(sig, FS) == pytest.approx(-25.63, abs=1e-3)


def test_idle_channel_noise_2khz_unity() -> None:
    # The CCIR-RMS filter is unity at 2 kHz (by construction), so a 2 kHz sine
    # reads its own dBFS to within the 0.03 dB the -5.63 print rounds off.
    sig = _dbfs_sine_amplitude(-30.0) * _tone(2000.0)
    assert idle_channel_noise(sig, FS) == pytest.approx(-30.0, abs=0.05)


def test_idle_channel_noise_scales_with_level() -> None:
    # Doubling the idle level raises the reported dBFS by 20*log10(2) = 6.02 dB.
    base = _dbfs_sine_amplitude(-40.0) * _tone(3150.0)
    lo = idle_channel_noise(base, FS)
    hi = idle_channel_noise(2.0 * base, FS)
    assert hi - lo == pytest.approx(20.0 * math.log10(2.0), abs=1e-6)


def test_idle_channel_noise_digital_zero_is_minus_inf() -> None:
    # Digital zero (silence, 3.14) carries no energy: the level is -inf dBFS.
    assert idle_channel_noise(np.zeros(N), FS) == -math.inf


def test_dynamic_range_closed_form_residual_at_2khz() -> None:
    # AES17 6.4.1: a 997 Hz tone at -60 dBFS with a lone -40 dBFS residual at
    # 2 kHz (where the CCIR-RMS filter is unity and the 997 Hz notch is
    # negligible). DR = ratio of the full-scale sine to the -40 dBFS residual
    # = 40 dB, to within the small notch/weighting corrections.
    sig = _dbfs_sine_amplitude(-60.0) * _tone(997.0)
    sig = sig + _dbfs_sine_amplitude(-40.0) * _tone(2000.0)
    # The 997 Hz notch trims a little of the 2 kHz residual, lifting DR ~0.4 dB
    # above the ideal 40 dB.
    assert dynamic_range(sig, FS, 997.0) == pytest.approx(40.0, abs=0.6)


def test_dynamic_range_residual_at_1khz_uses_ccir_offset() -> None:
    # A lone -40 dBFS residual at 1 kHz is weighted down by the 5.63 dB
    # CCIR-RMS offset, so the dynamic range reads 40 + 5.63 = 45.63 dB.
    sig = _dbfs_sine_amplitude(-60.0) * _tone(997.0)
    sig = sig + _dbfs_sine_amplitude(-40.0) * _tone(1000.0)
    # The 997 Hz notch also attenuates the nearby 1 kHz residual, lifting DR a
    # little; check it clears the unweighted 40 dB by at least the offset.
    assert dynamic_range(sig, FS, 997.0) > 45.0


def test_dynamic_range_monotonic_with_noise() -> None:
    # More residual noise lowers the dynamic range.
    rng = np.random.default_rng(4)
    tone = _dbfs_sine_amplitude(-60.0) * _tone(997.0)
    noise = rng.standard_normal(N)
    noise /= math.sqrt(float(np.mean(noise**2)))
    dr_quiet = dynamic_range(tone + 1e-4 * noise, FS, 997.0)
    dr_noisy = dynamic_range(tone + 1e-3 * noise, FS, 997.0)
    assert dr_quiet > dr_noisy
    # A 10x louder noise floor costs 20 dB of dynamic range.
    assert dr_quiet - dr_noisy == pytest.approx(20.0, abs=1.0)


def test_dynamic_range_full_scale_reference() -> None:
    # Halving the full-scale reference halves the numerator: DR drops 6.02 dB.
    sig = _dbfs_sine_amplitude(-60.0) * _tone(997.0)
    sig = sig + _dbfs_sine_amplitude(-40.0) * _tone(2000.0)
    dr1 = dynamic_range(sig, FS, 997.0, full_scale=1.0)
    dr2 = dynamic_range(sig, FS, 997.0, full_scale=0.5)
    assert dr1 - dr2 == pytest.approx(20.0 * math.log10(2.0), abs=1e-6)


def test_dynamic_range_rejects_bad_notch_q_and_full_scale() -> None:
    sig = _dbfs_sine_amplitude(-60.0) * _tone(997.0) + 1e-3 * _tone(2000.0)
    with pytest.raises(ValueError):
        dynamic_range(sig, FS, 997.0, notch_q=0.5)
    with pytest.raises(ValueError):
        dynamic_range(sig, FS, 997.0, full_scale=0.0)
    with pytest.raises(ValueError):
        idle_channel_noise(sig, FS, full_scale=-1.0)

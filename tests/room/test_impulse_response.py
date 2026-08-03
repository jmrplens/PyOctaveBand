#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for ISO 18233:2006 impulse-response acquisition.

Validation strategy (closed-form, not self-consistency):
- Ideal chain (recorded == excitation) -> band-limited delta at t=0.
- Known IIR system -> recovered in-band magnitude equals the analytic
  filter response (scipy.signal.freqz) within 0.1 dB (ISO 18233 5.4/B.5).
- Added noise -> IR peak-to-floor ratio grows ~3 dB per doubling of the
  sweep duration (ISO 18233 B.6).
- MLS: circular autocorrelation is L at lag 0 and -1 elsewhere with a flat
  magnitude spectrum (ISO 18233 A.1); a known IIR system is recovered in
  band within 0.1 dB via circular cross-correlation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import signal

from phonometry import (
    ImpulseResponseResult,
    impulse_response,
    inverse_filter,
    mls_impulse_response,
    mls_signal,
    plot_excitation,
    room_parameters,
    sweep_signal,
)
from phonometry.room.impulse_response import _MLS_TAPS

FS = 48000


def _band_response_db(b: np.ndarray, a: np.ndarray, ir: np.ndarray, fs: int,
                      f_lo: float, f_hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (estimated, true) magnitudes in dB on the rfft grid, in band."""
    n = ir.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    h_est = np.fft.rfft(ir)
    _, h_true = signal.freqz(b, a, worN=freqs, fs=fs)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    est_db = 20.0 * np.log10(np.abs(h_est[mask]))
    true_db = 20.0 * np.log10(np.abs(h_true[mask]))
    return est_db, true_db


# --------------------------------------------------------------------------
# Sweep generation
# --------------------------------------------------------------------------
def test_sweep_length_and_bounds() -> None:
    x = sweep_signal(FS, 20.0, 20000.0, 1.0)
    assert x.size == FS
    assert np.max(np.abs(x)) <= 1.0 + 1e-12


def test_sweep_instantaneous_frequency_is_exponential() -> None:
    # Zero crossings of an exponential sweep should follow f(t)=f1*(f2/f1)^(t/T).
    fs, f1, f2, secs = 48000, 100.0, 10000.0, 2.0
    x = sweep_signal(fs, f1, f2, secs, fade=0.0)
    # Analytic phase; verify start/end instantaneous frequency via finite diff.
    # Use the analytic model directly: compare against the closed-form phase.
    t = np.arange(x.size) / fs
    ts = x.size / fs
    k = 2.0 * np.pi * f1 * ts / np.log(f2 / f1)
    phase = k * ((f2 / f1) ** (t / ts) - 1.0)
    assert np.allclose(x, np.sin(phase), atol=1e-9)


# --------------------------------------------------------------------------
# Ideal chain -> delta
# --------------------------------------------------------------------------
def test_sweep_ideal_chain_is_delta() -> None:
    # Near-full-band sweep -> the recovered IR is a clean band-limited delta.
    x = sweep_signal(FS, 5.0, 23800.0, 2.0, fade=0.005)
    ir = impulse_response(x, x, FS, regularization=1e-8)
    assert int(np.argmax(np.abs(ir))) == 0
    peak = np.abs(ir[0])
    sidelobe = np.abs(ir[20:]).max()
    assert 20.0 * np.log10(sidelobe / peak) < -40.0


def test_farina_inverse_filter_compresses_sweep() -> None:
    f1, f2, secs = 50.0, 20000.0, 1.0
    x = sweep_signal(FS, f1, f2, secs, fade=0.02)
    inv = inverse_filter(FS, f1, f2, secs, fade=0.02)
    y = signal.fftconvolve(x, inv)
    peak = np.abs(y).max()
    ipk = int(np.argmax(np.abs(y)))
    # Everything more than 1 ms away from the compression peak is a sidelobe.
    guard = FS // 1000
    mask = np.ones(y.size, dtype=bool)
    mask[max(0, ipk - guard):ipk + guard] = False
    assert 20.0 * np.log10(np.abs(y[mask]).max() / peak) < -30.0


# --------------------------------------------------------------------------
# Known IIR system -> recovered response
# --------------------------------------------------------------------------
def test_sweep_recovers_iir_response_in_band() -> None:
    b, a = signal.butter(4, [200.0, 2000.0], btype="band", fs=FS)
    x = sweep_signal(FS, 20.0, 20000.0, 2.0)
    y = signal.lfilter(b, a, x)
    ir = impulse_response(y, x, FS, length=16384)
    est_db, true_db = _band_response_db(b, a, ir, FS, 300.0, 1500.0)
    assert np.max(np.abs(est_db - true_db)) < 0.1


def test_farina_method_recovers_iir_response_in_band() -> None:
    b, a = signal.butter(4, [200.0, 2000.0], btype="band", fs=FS)
    f1, f2, secs = 20.0, 20000.0, 2.0
    x = sweep_signal(FS, f1, f2, secs)
    y = signal.lfilter(b, a, x)
    ir = impulse_response(y, x, FS, method="farina", f_range=(f1, f2), length=16384)
    est_db, true_db = _band_response_db(b, a, ir, FS, 300.0, 1500.0)
    # Farina inverse-filter whitening is approximate; use a looser band bound.
    assert np.max(np.abs(est_db - true_db)) < 0.5


def test_farina_rejects_zero_padded_reference() -> None:
    """A reference zero-padded to the recording length (the correct input
    for method='spectral') silently rebuilds a wrong inverse filter for
    Farina; it must raise instead. The unpadded sweep is unaffected."""
    b, a = signal.butter(4, [200.0, 2000.0], btype="band", fs=FS)
    f1, f2, secs = 20.0, 20000.0, 2.0
    x = sweep_signal(FS, f1, f2, secs)
    y = signal.lfilter(b, a, x)
    # Pad the reference to the recording length, as one would for spectral.
    n = y.size + x.size - 1
    x_padded = np.concatenate([x, np.zeros(n - x.size)])
    with pytest.raises(ValueError, match="zero-pad|unpadded|spectral"):
        impulse_response(y, x_padded, FS, method="farina", f_range=(f1, f2))
    # The unpadded sweep still works unchanged.
    ir = impulse_response(y, x, FS, method="farina", f_range=(f1, f2), length=16384)
    assert np.isfinite(ir).all()


# --------------------------------------------------------------------------
# Harmonic separation (Farina): distortion products at negative times
# --------------------------------------------------------------------------
def test_harmonic_products_appear_at_negative_time() -> None:
    f1, f2, secs = 50.0, 12000.0, 3.0
    x = sweep_signal(FS, f1, f2, secs)
    # Memoryless quadratic non-linearity -> strong 2nd-harmonic product.
    y = x + 0.3 * x**2
    full = impulse_response(y, x, FS, return_full=True)
    n = full.size

    # The 2nd harmonic of an ESS arrives ahead of the linear IR by
    # dt = T*ln(2)/ln(f2/f1) (Farina 2000); in the wrapped deconvolution this
    # negative time sits near index n - dt*fs.
    advance = round(secs * np.log(2.0) / np.log(f2 / f1) * FS)
    predicted = n - advance
    window = np.abs(full[predicted - 500:predicted + 500])
    harmonic_peak = window.max()
    peak_idx = predicted - 500 + int(np.argmax(window))
    assert abs(peak_idx - predicted) < 50

    floor = np.sqrt(np.mean(full[n // 2:predicted - 5000] ** 2))
    assert harmonic_peak > 15.0 * floor

    # The default (trimmed) IR is causal, so it excludes the harmonic product.
    causal = impulse_response(y, x, FS)
    assert int(np.argmax(np.abs(causal))) == 0
    assert causal.size == x.size


# --------------------------------------------------------------------------
# Noise -> SNR improves with sweep duration (~3 dB/doubling)
# --------------------------------------------------------------------------
def test_sweep_snr_improves_3db_per_doubling() -> None:
    rng = np.random.default_rng(12345)
    f1, f2, noise_amp = 20.0, 20000.0, 2e-3

    def snr_for(seconds: float) -> float:
        x = sweep_signal(FS, f1, f2, seconds)
        noise = rng.standard_normal(x.size) * noise_amp
        ir = impulse_response(x + noise, x, FS, length=x.size)
        peak = np.abs(ir).max()
        lo, hi = ir.size // 4, 3 * ir.size // 4
        floor = np.sqrt(np.mean(ir[lo:hi] ** 2))
        return 20.0 * np.log10(peak / floor)

    snr_1 = snr_for(1.0)
    snr_2 = snr_for(2.0)
    snr_4 = snr_for(4.0)
    assert snr_2 - snr_1 == pytest.approx(3.0, abs=1.0)
    assert snr_4 - snr_2 == pytest.approx(3.0, abs=1.0)


# --------------------------------------------------------------------------
# MLS
# --------------------------------------------------------------------------
def test_mls_length_and_levels() -> None:
    for order in (4, 8, 12, 15):
        a = mls_signal(order)
        assert a.size == 2**order - 1
        assert set(np.unique(a)).issubset({-1.0, 1.0})
        # bipolar sum of a maximum-length sequence is exactly -1
        assert round(a.sum()) == -1


def test_mls_taps_cover_supported_orders() -> None:
    for order, taps in _MLS_TAPS.items():
        assert max(taps) == order  # highest tap must be the register length
        assert min(taps) >= 1


def test_all_mls_orders_are_primitive() -> None:
    # Every tap set must yield a true maximum-length sequence: circular
    # autocorrelation == length at lag 0 and -1 elsewhere (ISO 18233 A.1).
    for order in _MLS_TAPS:
        a = mls_signal(order)
        length = a.size
        autocorr = np.fft.irfft(np.abs(np.fft.rfft(a)) ** 2, length)
        assert autocorr[0] == pytest.approx(length, rel=1e-9)
        assert np.max(np.abs(autocorr[1:] + 1.0)) < 1e-6


def test_mls_autocorrelation_is_delta() -> None:
    order = 12
    a = mls_signal(order)
    length = a.size
    spec = np.fft.rfft(a)
    autocorr = np.fft.irfft(np.abs(spec) ** 2, length)
    assert autocorr[0] == pytest.approx(length, abs=1e-6)
    assert np.allclose(autocorr[1:], -1.0, atol=1e-6)
    # Flat magnitude spectrum away from DC.
    mag = np.abs(spec[1:])
    assert mag.std() / mag.mean() < 1e-9


def test_mls_ideal_chain_is_delta() -> None:
    order = 12
    a = mls_signal(order)
    ir = mls_impulse_response(a, a)
    assert int(np.argmax(np.abs(ir))) == 0
    assert ir[0] == pytest.approx(1.0, abs=1e-3)
    assert np.max(np.abs(ir[1:])) < 2e-3


def test_mls_recovers_iir_response_in_band() -> None:
    order = 14
    a = mls_signal(order)
    length = a.size
    b, coef_a = signal.butter(4, [200.0, 2000.0], btype="band", fs=FS)
    # Periodic excitation -> steady-state period == circular convolution.
    x = np.tile(a, 3)
    y = signal.lfilter(b, coef_a, x)
    y_period = y[-length:]
    ir = mls_impulse_response(y_period, a)
    freqs = np.fft.rfftfreq(length, d=1.0 / FS)
    h_est = np.fft.rfft(ir)
    _, h_true = signal.freqz(b, coef_a, worN=freqs, fs=FS)
    mask = (freqs >= 300.0) & (freqs <= 1500.0)
    est_db = 20.0 * np.log10(np.abs(h_est[mask]))
    true_db = 20.0 * np.log10(np.abs(h_true[mask]))
    assert np.max(np.abs(est_db - true_db)) < 0.1


def test_mls_short_period_aliasing_warns() -> None:
    """An IR longer than one MLS period folds back circularly (A.1); the
    recovered IR keeps undecayed energy at the period end, which is warned."""
    order = 12  # L = 4095 samples (~85 ms at 48 kHz)
    a = mls_signal(order)
    length = a.size
    t = np.arange(int(0.3 * FS)) / FS  # 0.3 s IR >> one period
    h = np.exp(-6.9078 * t / 0.3) * np.random.default_rng(0).standard_normal(t.size)
    h[0] += 2.0
    hh = np.zeros(length)
    for i in range(h.size):
        hh[i % length] += h[i]  # circular wrap into one period
    y = np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(hh)))
    with pytest.warns(UserWarning, match="aliases"):
        mls_impulse_response(y, a)


def test_mls_well_fitting_ir_does_not_warn() -> None:
    """A system IR that decays within one period must not warn."""
    order = 14  # L = 16383 samples (~340 ms)
    a = mls_signal(order)
    length = a.size
    b, coef_a = signal.butter(4, [200.0, 2000.0], btype="band", fs=FS)
    y = signal.lfilter(b, coef_a, np.tile(a, 3))[-length:]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mls_impulse_response(y, a)


def test_mls_snr_improves_with_averaging() -> None:
    order = 14
    a = mls_signal(order)
    length = a.size
    rng = np.random.default_rng(7)
    noise_amp = 0.5

    def snr_for(n_avg: int) -> float:
        # Average n_avg noisy periods before recovery (ISO 18233 6.3.6).
        acc = np.zeros(length)
        for _ in range(n_avg):
            acc += a + rng.standard_normal(length) * noise_amp
        recorded = acc / n_avg
        ir = mls_impulse_response(recorded, a)
        peak = np.abs(ir).max()
        floor = np.sqrt(np.mean(ir[length // 4:length // 2] ** 2))
        return 20.0 * np.log10(peak / floor)

    snr_1 = snr_for(1)
    snr_4 = snr_for(4)
    # 4x averaging -> +6 dB (3 dB per doubling, ISO 18233 6.3.6).
    assert snr_4 - snr_1 == pytest.approx(6.0, abs=1.5)


def test_mls_multi_period_recording_averages_internally() -> None:
    """A >1-period recording exercises the internal reshape/mean averaging
    branch and recovers a known filter, beating a single noisy period."""
    order = 14
    a = mls_signal(order)
    length = a.size
    b = np.array([0.6, 0.25, 0.1, 0.05])  # known short FIR filter
    reps = 4
    rng = np.random.default_rng(2026)
    # Steady-state periodic response: filter reps+1 tiles and drop the first
    # (transient) period so every remaining period is the circular response.
    x = np.tile(a, reps + 1)
    y = signal.lfilter(b, 1.0, x)[length:]
    noisy = y + rng.standard_normal(y.size) * 0.3
    assert noisy.size == reps * length  # multi-period recording

    ir_multi = mls_impulse_response(noisy, a)  # internal averaging over reps
    ir_single = mls_impulse_response(noisy[:length], a)  # one noisy period

    err_multi = float(np.max(np.abs(ir_multi[: b.size] - b)))
    err_single = float(np.max(np.abs(ir_single[: b.size] - b)))
    assert err_multi < err_single  # averaging lowers the error
    assert err_multi < 0.02  # recovers the filter within a tight tolerance


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="f2 must be greater than f1"):
        sweep_signal(FS, 100.0, 100.0, 1.0)  # f1 == f2
    with pytest.raises(ValueError, match="f2 must be greater than f1"):
        sweep_signal(FS, 100.0, 50.0, 1.0)  # f1 > f2
    with pytest.raises(
        ValueError, match="must not exceed the Nyquist frequency"
    ):
        sweep_signal(FS, 20.0, 30000.0, 1.0)  # f2 > Nyquist
    with pytest.raises(ValueError, match="order must be one of"):
        mls_signal(1)  # order too small
    with pytest.raises(ValueError, match="order must be one of"):
        mls_signal(99)  # unsupported order
    with pytest.raises(ValueError, match="unknown method"):
        impulse_response(np.zeros(10), np.zeros(10), FS, method="bogus")
    with pytest.raises(ValueError, match="requires f_range"):
        impulse_response(np.zeros(10), np.zeros(10), FS, method="farina")  # no f_range
    with pytest.raises(
        ValueError,
        match="recorded length must be a positive multiple of the MLS length",
    ):
        mls_impulse_response(np.zeros(10), mls_signal(4))  # not a multiple of L


def test_inverse_filter_empty_band_raises() -> None:
    # A very short sweep over a razor-thin band yields no in-band FFT bin:
    # the median of an empty slice would be NaN, so this must raise instead.
    with pytest.raises(
        ValueError, match="no frequency bin falls within the sweep band"
    ):
        inverse_filter(FS, 1000.0, 1000.5, 0.001)
    # The error must propagate through the Farina deconvolution path too.
    # The reference must be non-zero, or the zero-padding guard fires first.
    with pytest.raises(
        ValueError, match="no frequency bin falls within the sweep band"
    ):
        impulse_response(
            np.zeros(48), np.ones(48), FS,
            method="farina", f_range=(1000.0, 1000.5),
        )


def test_impulse_response_rejects_bad_shapes() -> None:
    good = np.zeros(10)
    with pytest.raises(
        ValueError, match="'recorded' and 'reference' must be one-dimensional"
    ):
        impulse_response(np.zeros((2, 5)), good, FS)  # 2-D recorded
    with pytest.raises(
        ValueError, match="'recorded' and 'reference' must be one-dimensional"
    ):
        impulse_response(good, np.zeros((2, 5)), FS)  # 2-D reference
    with pytest.raises(
        ValueError, match="'recorded' and 'reference' must be non-empty"
    ):
        impulse_response(np.zeros(0), good, FS)  # empty recorded
    with pytest.raises(
        ValueError, match="'recorded' and 'reference' must be non-empty"
    ):
        impulse_response(good, np.zeros(0), FS)  # empty reference


def test_mls_impulse_response_rejects_bad_shapes() -> None:
    seq = mls_signal(4)
    good = np.tile(seq, 2)
    with pytest.raises(
        ValueError, match="'recorded' and 'mls' must be one-dimensional"
    ):
        mls_impulse_response(np.zeros((2, seq.size)), seq)  # 2-D recorded
    with pytest.raises(
        ValueError, match="'recorded' and 'mls' must be one-dimensional"
    ):
        mls_impulse_response(good, np.zeros((2, seq.size)))  # 2-D mls
    with pytest.raises(
        ValueError, match="'recorded' and 'mls' must be non-empty"
    ):
        mls_impulse_response(np.zeros(0), seq)  # empty recorded
    with pytest.raises(
        ValueError, match="'recorded' and 'mls' must be non-empty"
    ):
        mls_impulse_response(good, np.zeros(0))  # empty mls


# --------------------------------------------------------------------------
# Result object: backward compatibility (drop-in for the raw IR array)
# --------------------------------------------------------------------------
def _synthetic_recording() -> tuple[np.ndarray, np.ndarray]:
    sweep = sweep_signal(FS, 20.0, 20000.0, 1.0)
    system = np.zeros(FS)
    system[100] = 1.0
    system[2000] = 0.4
    return signal.fftconvolve(sweep, system), sweep


def test_impulse_response_returns_result_object() -> None:
    rec, sweep = _synthetic_recording()
    ir = impulse_response(rec, sweep, FS)
    assert isinstance(ir, ImpulseResponseResult)
    assert ir.fs == FS
    assert ir.method == "spectral"
    # Array protocol: np.asarray yields the raw IR and equals the .ir field.
    arr = np.asarray(ir)
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, ir.ir)
    # Drop-in array surface used across the codebase.
    assert ir.size == ir.ir.size == len(ir)
    assert ir.shape == ir.ir.shape
    assert ir.ndim == 1
    assert np.array_equal(ir[100:110], ir.ir[100:110])
    assert int(np.argmax(np.abs(ir))) == 100


def test_impulse_response_asarray_matches_untrimmed_math() -> None:
    """np.asarray(result) must equal the array the old API returned."""
    rec, sweep = _synthetic_recording()
    ir = impulse_response(rec, sweep, FS, length=4096)
    n = rec.size + sweep.size - 1
    spec_x = np.fft.rfft(sweep, n=n)
    spec_y = np.fft.rfft(rec, n=n)
    power = np.abs(spec_x) ** 2
    reg = 1e-6 * float(power.max())
    full = np.fft.irfft(spec_x.conj() * spec_y / (power + reg), n=n)
    assert np.allclose(np.asarray(ir), full[:4096])


def test_room_parameters_accepts_result_object() -> None:
    """room_parameters(result, fs) must work exactly like room_parameters(arr, fs)."""
    rec, sweep = _synthetic_recording()
    ir = impulse_response(rec, sweep, FS)
    from_result = room_parameters(ir, FS, limits=None)
    from_array = room_parameters(np.asarray(ir), FS, limits=None)
    assert np.allclose(from_result.edt, from_array.edt, equal_nan=True)
    assert np.allclose(from_result.t30, from_array.t30, equal_nan=True)


def test_mls_result_metadata() -> None:
    a = mls_signal(14)
    ir = mls_impulse_response(np.tile(a, 2)[: a.size], a, fs=FS)
    assert isinstance(ir, ImpulseResponseResult)
    assert ir.method == "mls"
    assert ir.fs == FS
    # fs is optional; omitting it leaves the axis unitful-agnostic (None).
    assert mls_impulse_response(a, a).fs is None


# --------------------------------------------------------------------------
# Plotting (Agg backend): impulse response and excitation signals
# --------------------------------------------------------------------------
def test_impulse_response_plot_returns_axes() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    rec, sweep = _synthetic_recording()
    ir = impulse_response(rec, sweep, FS)
    axes = ir.plot()
    assert isinstance(axes, np.ndarray) and axes.size == 2
    assert all(isinstance(a, Axes) for a in axes)

    _, single = matplotlib.pyplot.subplots()
    returned = ir.plot(ax=single)
    assert returned is single
    matplotlib.pyplot.close("all")


def test_plot_excitation_returns_axes() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    sweep = sweep_signal(FS, 20.0, 20000.0, 0.5)
    axes = plot_excitation(sweep, FS, kind="sweep")
    assert isinstance(axes, np.ndarray) and axes.size == 2

    mls = mls_signal(12).astype(float)
    axes_m = plot_excitation(mls, FS, kind="mls")
    assert isinstance(axes_m, np.ndarray) and axes_m.size == 2
    assert all(isinstance(a, Axes) for a in axes_m)
    matplotlib.pyplot.close("all")


def test_plot_excitation_short_and_empty_guards() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A short sweep must not crash the spectrogram (nperseg capped by length).
    short = sweep_signal(FS, 100.0, 4000.0, 0.02)
    axes = plot_excitation(short, FS, kind="sweep")
    assert axes.size == 2
    plt.close("all")

    # Empty inputs raise a clear error rather than a cryptic reduction failure.
    with pytest.raises(ValueError, match="empty"):
        plot_excitation(np.array([]), FS, kind="mls")
    with pytest.raises(ValueError, match="empty"):
        ImpulseResponseResult(ir=np.array([]), fs=FS, method="spectral").plot()

#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for cepstral analysis (cepstrum, liftering, echo detection).

Clean-room oracles, hand-derived from the definitions in the sources:

**Echo rahmonics.** For ``x[n] = delta[n] + a*delta[n-d]`` (one in-record
echo of reflection coefficient ``a`` at delay ``d`` samples) the DFT is
exactly ``X[k] = 1 + a*exp(-j*2*pi*k*d/N)``. With ``|a| < 1`` the complex
logarithm expands into the absolutely convergent Mercator series

``ln(1 + a*e^{-j*theta}) = sum_{n>=1} (-1)^{n+1} (a^n/n) e^{-j*n*theta}``,

whose inverse DFT is a spike train: the complex cepstrum carries exactly
``(-1)^{n+1} a^n / n`` at quefrency ``n*d`` (mod N). Taking real parts,
``ln|X|`` carries the same coefficients on ``cos`` terms, so its inverse
DFT splits each spike in half between ``n*d`` and ``N - n*d``: the real
cepstrum reads ``a/2`` at the delay and the power cepstrum (``ln|X|^2 =
2*ln|X|``) reads ``a`` -- the reflection coefficient itself. The partial
sums of the rahmonic amplitudes converge to ``ln(1 + a)``, the value of
the log spectrum where the echo interferes fully constructively. All of
these are closed forms with no free parameter, pinned to near machine
precision below.

**Liftering.** For the same signal the log spectrum is pure ripple of
period ``1/d``: a lowpass lifter cutting below ``d`` must return a flat
0 dB envelope, a highpass lifter must keep the ripple, whose exact
extrema are ``20*log10(1 +/- a)`` dB, and the two liftered spectra must
add up to the original log spectrum exactly (the split is linear).

**Complex-cepstrum round trip.** ``CepstrumResult.invert`` must undo the
forward transform (including the removed linear phase) to machine
precision on a mixed-phase, delayed record.

**Minimum-phase refactor.** ``phonometry.minimum_phase`` now folds the
real cepstrum through the shared ``_fold_causal`` core; a frozen verbatim
copy of the pre-refactor implementation must agree *bit for bit* on fixed
responses.
"""

from __future__ import annotations

import dataclasses

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import signal as sp_signal

import phonometry as ph

FS = 8192.0
N = 4096
DELAY = 313
ALPHA = 0.4


def _impulse_echo(n: int = N, d: int = DELAY, a: float = ALPHA) -> np.ndarray:
    x = np.zeros(n, dtype=np.float64)
    x[0] = 1.0
    x[d] = a
    return x


# ---------------------------------------------------------------------------
# Rahmonic closed forms (hand-derived series, see module docstring)
# ---------------------------------------------------------------------------


def test_power_cepstrum_echo_rahmonics_are_exact() -> None:
    res = ph.signals.cepstrum(_impulse_echo(), FS, kind="power")
    assert res.kind == "power"
    for n in (1, 2, 3, 4):
        expected = (-1.0) ** (n + 1) * ALPHA**n / n
        assert res.cepstrum[n * DELAY] == pytest.approx(expected, abs=1e-12)
    # The rahmonic amplitude partial sums converge to ln(1 + a).
    total = sum((-1.0) ** (n + 1) * ALPHA**n / n for n in range(1, 60))
    assert total == pytest.approx(np.log1p(ALPHA), abs=1e-15)


def test_real_cepstrum_is_half_the_power_cepstrum() -> None:
    x = _impulse_echo()
    real = ph.signals.cepstrum(x, FS, kind="real")
    assert real.cepstrum[DELAY] == pytest.approx(ALPHA / 2.0, abs=1e-12)
    # ...and mirrored onto the negative quefrencies (ln|X| is even).
    assert real.cepstrum[N - DELAY] == pytest.approx(ALPHA / 2.0, abs=1e-12)


def test_complex_cepstrum_echo_rahmonics_are_exact() -> None:
    res = ph.signals.cepstrum(_impulse_echo(), FS, kind="complex")
    # delta + a*delta_d with a < 1 is minimum phase: no linear phase removed.
    assert res.linear_phase_samples == 0
    for n in (1, 2, 3):
        expected = (-1.0) ** (n + 1) * ALPHA**n / n
        assert res.cepstrum[n * DELAY] == pytest.approx(expected, abs=1e-12)
    # A minimum-phase signal has a (numerically) causal complex cepstrum:
    # nothing at the negative quefrencies where the real cepstrum mirrors.
    assert abs(res.cepstrum[N - DELAY]) < 1e-12


def test_echo_on_broadband_noise_source() -> None:
    """The spike survives a non-trivial source: x = s + a*s(t-t0), circular."""
    s = ph.signals.noise_signal(FS, N / FS, color="white", seed=20260721)
    x = s + ALPHA * np.roll(s, DELAY)
    res = ph.signals.cepstrum(x, FS, kind="power")
    # ln X = ln S + ln(1 + a e^{-j theta}) exactly (circular shift), so the
    # echo spike still reads the reflection coefficient; the noise source
    # spreads a small random floor over all quefrencies.
    assert res.cepstrum[DELAY] == pytest.approx(ALPHA, abs=0.02)


def test_quefrency_axis_and_shapes() -> None:
    res = ph.signals.cepstrum(_impulse_echo(), FS)
    assert res.quefrencies.size == res.cepstrum.size == res.nfft == N
    assert res.quefrencies[0] == 0.0
    assert res.quefrencies[1] == pytest.approx(1.0 / FS)
    assert res.fs == FS


def test_odd_record_pads_to_even_nfft() -> None:
    res = ph.signals.cepstrum(np.random.default_rng(7).standard_normal(4095), FS)
    assert res.nfft == 4096


# ---------------------------------------------------------------------------
# Complex cepstrum: invertibility (homomorphic round trip)
# ---------------------------------------------------------------------------


def _mixed_phase_delayed() -> np.ndarray:
    """A delayed band-limited pulse: linear phase plus mixed-phase content."""
    x = np.zeros(2048, dtype=np.float64)
    impulse = np.zeros(256)
    impulse[0] = 1.0
    b, a = sp_signal.butter(2, 0.3)
    x[37 : 37 + 256] = sp_signal.lfilter(b, a, impulse)
    return x


def test_complex_cepstrum_round_trip_is_exact() -> None:
    x = _mixed_phase_delayed()
    res = ph.signals.cepstrum(x, FS, kind="complex")
    assert res.linear_phase_samples != 0  # the delay was detected and removed
    np.testing.assert_allclose(res.invert(), x, atol=1e-12)


def test_only_the_complex_cepstrum_inverts() -> None:
    res = ph.signals.cepstrum(_impulse_echo(), FS, kind="power")
    with pytest.raises(ValueError, match=r"Only the complex cepstrum is invertible"):
        res.invert()


# ---------------------------------------------------------------------------
# Echo detection
# ---------------------------------------------------------------------------


def test_echo_detection_reads_delay_and_reflection_exactly() -> None:
    res = ph.signals.echo_detection(_impulse_echo(), FS)
    assert res.delay_samples == DELAY
    assert res.delay == pytest.approx(DELAY / FS)
    assert res.reflection_coefficient == pytest.approx(ALPHA, abs=1e-12)


def test_echo_detection_on_noise_source() -> None:
    s = ph.signals.noise_signal(FS, N / FS, color="white", seed=42)
    x = s + 0.25 * np.roll(s, 200)
    res = ph.signals.echo_detection(x, FS, min_quefrency=0.005)
    assert res.delay_samples == 200
    assert res.reflection_coefficient == pytest.approx(0.25, abs=0.02)


def test_echo_detection_negative_reflection_exact() -> None:
    """An inverting echo (a < 0) is a *negative* rahmonic of height a.

    The Mercator series holds for either sign of ``a``: the first rahmonic
    of ``delta[n] - 0.4*delta[n-d]`` is exactly ``-0.4`` at quefrency
    ``d``. Peak-picking on ``|cepstrum|`` must find the true delay and
    report the signed coefficient.
    """
    res = ph.signals.echo_detection(_impulse_echo(a=-ALPHA), FS)
    assert res.delay_samples == DELAY
    assert res.delay == pytest.approx(DELAY / FS)
    assert res.reflection_coefficient == pytest.approx(-ALPHA, abs=1e-12)


def test_echo_detection_negative_reflection_on_noise_source() -> None:
    s = ph.signals.noise_signal(FS, N / FS, color="white", seed=42)
    x = s - 0.25 * np.roll(s, 200)
    res = ph.signals.echo_detection(x, FS, min_quefrency=0.005)
    assert res.delay_samples == 200
    assert res.reflection_coefficient == pytest.approx(-0.25, abs=0.02)


def test_echo_detection_off_sample_delay_is_interpolated() -> None:
    """A delay midway between samples: quadratic interpolation locates it.

    The echo is synthesised circularly by an exact half-sample phase ramp,
    so the true delay is 313.5 samples. The rahmonic splits across the
    neighbouring bins: the interpolated ``delay`` recovers the fraction,
    while the coefficient at the peak sample underestimates ``a`` (the
    documented bin-splitting caveat).
    """
    rng = np.random.default_rng(7)
    s = rng.standard_normal(N)
    true_delay = DELAY + 0.5
    ramp = np.exp(-2j * np.pi * np.fft.rfftfreq(N) * true_delay)
    x = s + ALPHA * np.fft.irfft(np.fft.rfft(s) * ramp, N)

    res = ph.signals.echo_detection(x, FS)
    assert res.delay_samples in (DELAY, DELAY + 1)
    assert res.delay * FS == pytest.approx(true_delay, abs=0.15)
    # Bin splitting: the peak-sample coefficient reads low but keeps sign.
    assert 0.15 < res.reflection_coefficient < ALPHA


def test_echo_detection_search_band_is_respected() -> None:
    # Restrict the band away from the true echo: the peak reported must
    # come from inside the band.
    res = ph.signals.echo_detection(
        _impulse_echo(),
        FS,
        min_quefrency=400.0 / FS,
        max_quefrency=1000.0 / FS,
    )
    assert 400 <= res.delay_samples <= 1000
    assert res.search_range == (400.0 / FS, 1000.0 / FS)


def test_echo_detection_rejects_bad_band() -> None:
    x = _impulse_echo()
    with pytest.raises(ValueError, match=r"quefrency search band must satisfy"):
        ph.signals.echo_detection(x, FS, min_quefrency=0.1, max_quefrency=0.05)
    with pytest.raises(ValueError, match=r"quefrency search band must satisfy"):
        ph.signals.echo_detection(x, FS, max_quefrency=1.0)


# ---------------------------------------------------------------------------
# Liftering
# ---------------------------------------------------------------------------


def test_lowpass_lifter_removes_the_echo_ripple() -> None:
    """Below the echo quefrency there is only the flat 0 dB source."""
    res = ph.signals.lifter(
        _impulse_echo(), FS, cutoff=(DELAY - 50) / FS, mode="lowpass"
    )
    assert float(np.max(np.abs(res.liftered_db))) < 1e-4


def test_highpass_lifter_keeps_the_ripple_extrema() -> None:
    """ln|1 + a e^{-j theta}| swings between ln(1+a) and ln(1-a)."""
    res = ph.signals.lifter(
        _impulse_echo(), FS, cutoff=(DELAY - 50) / FS, mode="highpass"
    )
    assert float(np.max(res.liftered_db)) == pytest.approx(
        20.0 * np.log10(1.0 + ALPHA), abs=1e-4
    )
    assert float(np.min(res.liftered_db)) == pytest.approx(
        20.0 * np.log10(1.0 - ALPHA), abs=1e-4
    )


def test_lifter_modes_are_exactly_complementary() -> None:
    x = ph.signals.noise_signal(FS, 0.5, color="pink", seed=3)
    low = ph.signals.lifter(x, FS, cutoff=0.002, mode="lowpass")
    high = ph.signals.lifter(x, FS, cutoff=0.002, mode="highpass")
    np.testing.assert_allclose(
        low.liftered_db + high.liftered_db, low.spectrum_db, atol=1e-9
    )


def test_lifter_validates_inputs() -> None:
    x = _impulse_echo()
    with pytest.raises(ValueError, match=r"'mode' must be 'lowpass'"):
        ph.signals.lifter(x, FS, cutoff=0.01, mode="bandpass")
    with pytest.raises(ValueError, match=r"'cutoff' must resolve to"):
        ph.signals.lifter(x, FS, cutoff=10.0)  # beyond nfft/2
    with pytest.raises(ValueError, match=r"'cutoff' must resolve to"):
        ph.signals.lifter(x, FS, cutoff=1e-9)  # below one sample


# ---------------------------------------------------------------------------
# Validation and result invariants
# ---------------------------------------------------------------------------


def test_cepstrum_validates_inputs() -> None:
    x = _impulse_echo()
    with pytest.raises(ValueError, match=r"'kind' must be one of"):
        ph.signals.cepstrum(x, FS, kind="cheese")
    with pytest.raises(ValueError, match=r"'nfft' must be at least the record length"):
        ph.signals.cepstrum(x, FS, nfft=N - 2)
    with pytest.raises(ValueError, match=r"'nfft' must be even"):
        ph.signals.cepstrum(x, FS, nfft=N + 1)
    with pytest.raises(ValueError, match=r"'fs' must be a positive, finite number"):
        ph.signals.cepstrum(x, 0.0)
    silence = np.zeros(N)
    with pytest.raises(ValueError, match=r"'x' must not be identically zero"):
        ph.signals.cepstrum(silence, FS)


def test_results_are_frozen() -> None:
    res = ph.signals.cepstrum(_impulse_echo(), FS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.kind = "real"  # type: ignore[misc]
    echo = ph.signals.echo_detection(_impulse_echo(), FS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        echo.delay = 0.0  # type: ignore[misc]


def test_a_cepstrum_result_refuses_a_cepstrum_short_of_its_quefrency_axis() -> None:
    """The half a plot never draws is where a slipped pair hides.

    :meth:`CepstrumResult.plot` cuts both arrays at ``nfft // 2 + 1``, so a
    disagreement past that midpoint never reaches the figure: the curve comes
    out complete and ordinary, and the samples that were never paired with a
    quefrency are simply not there. That pair is what an echo delay is read
    off, so it is pinned when the result is built instead.
    """
    res = ph.signals.cepstrum(_impulse_echo(), FS)
    short = res.cepstrum[:-1]
    with pytest.raises(ValueError, match=r"'cepstrum'.*one value per quefrency"):
        dataclasses.replace(res, cepstrum=short)


def test_a_lifter_result_refuses_a_quefrency_axis_short_of_its_cepstrum() -> None:
    """The lifter's cepstrum panel is cut at the same midpoint, and as blind.

    A quefrency axis one sample short of the cepstrum still covers the whole
    ``nfft // 2 + 1`` slice the panel draws, so the figure is complete and the
    mismatch is invisible in it.
    """
    res = ph.signals.lifter(_impulse_echo(), FS, cutoff=(DELAY - 50) / FS)
    short = res.quefrencies[:-1]
    with pytest.raises(ValueError, match=r"'quefrencies'.*one value per quefrency"):
        dataclasses.replace(res, quefrencies=short)


def test_a_lifter_result_names_the_spectrum_column_that_slipped() -> None:
    """The overlay raises about two shapes; the result raises about a field.

    Handing ``Axes.plot`` a curve of a different length from the frequency
    axis is refused outright, in either direction, but the message names the
    shapes it could not reconcile rather than the column they came from, and
    only once someone draws. The check moves the refusal to construction,
    where the column still has its name.
    """
    res = ph.signals.lifter(_impulse_echo(), FS, cutoff=(DELAY - 50) / FS)
    short = res.liftered_db[:-1]
    with pytest.raises(ValueError, match=r"'liftered_db'.*one value per frequency"):
        dataclasses.replace(res, liftered_db=short)


def test_an_echo_detection_result_refuses_a_cepstrum_short_of_its_axis() -> None:
    """The detection reports a place, and the two axes are what pin it.

    :attr:`delay_samples` is a position in the cepstrum and :attr:`delay` a
    time on the quefrency axis; the report is the claim that those are the
    same place. The plot slices at ``nfft // 2 + 1`` like the others, so a
    pair that disagrees past the midpoint is drawn without complaint.
    """
    res = ph.signals.echo_detection(_impulse_echo(), FS)
    short = res.cepstrum[:-1]
    with pytest.raises(ValueError, match=r"'cepstrum'.*one value per quefrency"):
        dataclasses.replace(res, cepstrum=short)


# ---------------------------------------------------------------------------
# Minimum-phase refactor: bit-exact against the pre-refactor implementation
# ---------------------------------------------------------------------------


def _old_minimum_phase(response: np.ndarray, *, oversample: int = 8) -> np.ndarray:
    """Verbatim frozen copy of the pre-refactor ``minimum_phase`` body.

    Only the private helpers it called (validation and the trigonometric
    oversampler, both untouched by the refactor) are used from the live
    module; the cepstral folding block is the literal pre-refactor code.
    """
    from phonometry.signals.phase import (
        _MAGNITUDE_FLOOR,
        _trig_oversample,
        _validate_oversample,
        _validate_response,
    )

    resp = _validate_response(response)
    factor = _validate_oversample(oversample)
    magnitude = np.abs(resp)
    floor = float(np.max(magnitude)) * _MAGNITUDE_FLOOR

    bins = resp.size
    nfft = 2 * (bins - 1) * factor
    mag_fine = _trig_oversample(magnitude, factor)
    log_fine = np.log(np.maximum(mag_fine, floor))
    cepstrum = np.fft.irfft(log_fine, nfft)
    folded = np.zeros(nfft, dtype=np.float64)
    folded[0] = cepstrum[0]
    folded[1 : nfft // 2] = 2.0 * cepstrum[1 : nfft // 2]
    folded[nfft // 2] = cepstrum[nfft // 2]
    phase = np.imag(np.fft.rfft(folded))[::factor]
    return np.asarray(magnitude * np.exp(1j * phase), dtype=np.complex128)


def _fixed_responses() -> list[np.ndarray]:
    """Deterministic responses exercising smooth, sharp and noisy magnitudes."""
    w = np.linspace(0.0, np.pi, 513)
    # RBJ peaking biquad (strictly minimum phase).
    w0, q, a_lin = 0.3 * np.pi, 1.5, 10.0 ** (6.0 / 40.0)
    alpha = np.sin(w0) / (2.0 * q)
    b = [1.0 + alpha * a_lin, -2.0 * np.cos(w0), 1.0 - alpha * a_lin]
    a = [1.0 + alpha / a_lin, -2.0 * np.cos(w0), 1.0 - alpha / a_lin]
    _, biquad = sp_signal.freqz(b, a, worN=w)
    # A band-pass-like magnitude with exact zeros at both grid ends.
    bandpass = np.sin(w) * (1.0 + 0.5 * np.cos(3.0 * w))
    # A seeded rough magnitude (measurement-like).
    rng = np.random.default_rng(20260721)
    rough = np.exp(0.3 * rng.standard_normal(w.size)) + 0.1
    return [
        np.asarray(biquad, dtype=np.complex128),
        bandpass.astype(np.float64),
        rough.astype(np.float64),
    ]


@pytest.mark.parametrize("oversample", [1, 4, 8])
def test_minimum_phase_refactor_is_bit_exact(oversample: int) -> None:
    """The shared-core refactor changes no output bit on any fixed input."""
    for response in _fixed_responses():
        old = _old_minimum_phase(response, oversample=oversample)
        new = ph.signals.minimum_phase(response, oversample=oversample)
        assert old.dtype == new.dtype
        assert old.shape == new.shape
        assert np.array_equal(old.view(np.float64), new.view(np.float64)), (
            "minimum_phase output changed bitwise after the cepstrum refactor"
        )


def test_phase_decomposition_uses_the_shared_core_bit_exactly() -> None:
    response = _fixed_responses()[0]
    old = _old_minimum_phase(response)
    res = ph.signals.phase_decomposition(response, FS)
    assert np.array_equal(
        old.view(np.float64), res.minimum_phase_response.view(np.float64)
    )


# ---------------------------------------------------------------------------
# .plot() renderers
# ---------------------------------------------------------------------------


def test_cepstrum_plot_single_axes_and_external_ax() -> None:
    res = ph.signals.cepstrum(_impulse_echo(), FS)
    ax = res.plot()
    assert ax.get_title() != ""
    assert len(ax.lines) == 1
    # Only the unambiguous first half of the quefrency axis is drawn.
    assert ax.lines[0].get_xdata().size == N // 2 + 1
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext
    plt.close("all")


def test_echo_detection_plot_marks_the_peak() -> None:
    res = ph.signals.echo_detection(_impulse_echo(), FS)
    ax = res.plot()
    marker = ax.lines[-1]
    assert marker.get_xdata()[0] == pytest.approx(1e3 * res.delay)
    assert marker.get_ydata()[0] == pytest.approx(res.reflection_coefficient)
    plt.close("all")


def test_lifter_plot_two_panels_and_external_ax() -> None:
    res = ph.signals.lifter(_impulse_echo(), FS, cutoff=(DELAY - 50) / FS)
    axes = res.plot()
    assert len(axes) == 2
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext
    plt.close("all")


def test_plot_rejects_unknown_language() -> None:
    res = ph.signals.cepstrum(_impulse_echo(), FS)
    with pytest.raises(ValueError, match=r"Unknown language"):
        res.plot(language="xx")
    plt.close("all")

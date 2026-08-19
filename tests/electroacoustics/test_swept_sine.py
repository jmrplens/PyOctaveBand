#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for swept-sine harmonic separation (Farina 2000 / Novak 2015).

The oracle is the memoryless polynomial nonlinearity, whose harmonic
responses follow exactly from the Chebyshev identities: driving
``y = x + a2·x² + a3·x³`` with a unit sweep gives ``H1 = 1 + 3a3/4``,
``|H2| = a2/2`` at phase ``-π/2``, ``|H3| = a3/4`` at phase ``π``, and
``THD = √((a2/2)² + (a3/4)²) / (1 + 3a3/4)``. The same THD measured tone
by tone with :func:`phonometry.thd` cross-checks the separator against
the classical analyzer. The linear case pins the THD floor, a
delay-plus-gain post-filter checks the Hammerstein composition, and the
synchronized-sweep phases are asserted where the Farina sweep's are
documented as excitation-dependent.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phonometry as ph

FS = 48000
F1, F2, SECONDS = 20.0, 6000.0, 4.0
A2, A3 = 0.1, 0.2

#: Chebyshev closed forms for y = x + a2 x^2 + a3 x^3 driven by sin.
H1_EXPECTED = 1.0 + 3.0 * A3 / 4.0
H2_EXPECTED = A2 / 2.0
H3_EXPECTED = A3 / 4.0
THD_EXPECTED = float(np.hypot(H2_EXPECTED, H3_EXPECTED) / H1_EXPECTED)


def _polynomial_recording(sweep: np.ndarray) -> np.ndarray:
    return sweep + A2 * sweep**2 + A3 * sweep**3


def _analyze(**kwargs: object) -> ph.SweptSineDistortionResult:
    x = ph.synchronized_sweep_signal(FS, F1, F2, SECONDS)
    return ph.swept_sine_distortion(
        _polynomial_recording(x), FS, f1=F1, f2=F2, seconds=SECONDS, n_harmonics=3, **kwargs
    )


def _at(freqs: np.ndarray, values: np.ndarray, f: float) -> complex:
    return complex(values[int(np.argmin(np.abs(freqs - f)))])


# ---------------------------------------------------------------------------
# Synchronized sweep generation (Novak Eqs. 47/49)
# ---------------------------------------------------------------------------


def test_synchronized_sweep_rate_is_integer_periods_of_f1() -> None:
    """L·f1 must be an integer (the Eq. 49 rounding)."""
    x = ph.synchronized_sweep_signal(FS, F1, F2, SECONDS)
    rate = round(F1 * SECONDS / np.log(F2 / F1)) / F1
    assert rate * F1 == round(rate * F1)
    assert x.size == int(np.ceil(rate * np.log(F2 / F1) * FS))


def test_synchronized_sweep_time_shift_equals_harmonic() -> None:
    """Delaying by L·ln(n) equals generating the n-th harmonic (Eq. 18)."""
    rate = round(F1 * SECONDS / np.log(F2 / F1)) / F1
    t = np.arange(int(np.ceil(rate * np.log(F2 / F1) * FS))) / FS
    phase = 2.0 * np.pi * F1 * rate * np.exp(t / rate)
    n = 3
    shifted = np.sin(2.0 * np.pi * F1 * rate * np.exp((t + rate * np.log(n)) / rate))
    np.testing.assert_allclose(shifted, np.sin(n * phase), atol=1e-9)


def test_synchronized_sweep_starts_at_zero_phase() -> None:
    x = ph.synchronized_sweep_signal(FS, F1, F2, SECONDS)
    assert abs(x[0]) < 1e-9


def test_synchronized_sweep_band_and_duration_validation() -> None:
    with pytest.raises(ValueError, match="f2"):
        ph.synchronized_sweep_signal(FS, 100.0, 50.0, SECONDS)
    with pytest.raises(ValueError, match="Nyquist"):
        ph.synchronized_sweep_signal(FS, 100.0, 30000.0, SECONDS)
    with pytest.raises(ValueError, match="synchronize"):
        ph.synchronized_sweep_signal(FS, 1.0, 1000.0, 0.5)
    with pytest.raises(ValueError, match="fade"):
        ph.synchronized_sweep_signal(FS, F1, F2, SECONDS, fade=0.7)


def test_synchronized_sweep_rejects_bad_amplitude() -> None:
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="amplitude"):
            ph.synchronized_sweep_signal(FS, F1, F2, SECONDS, amplitude=bad)


def test_empty_thd_grid_raises_actionably() -> None:
    """f1 above fs/4: every order-2 product would exceed Nyquist."""
    f1, f2, seconds = 13000.0, 20000.0, 0.5
    x = ph.synchronized_sweep_signal(FS, f1, f2, seconds)
    with pytest.raises(ValueError, match="order-2 product"):
        ph.swept_sine_distortion(x, FS, f1=f1, f2=f2, seconds=seconds, n_harmonics=2)


def test_synchronized_sweep_amplitude_and_fade() -> None:
    x = ph.synchronized_sweep_signal(FS, F1, F2, SECONDS, amplitude=0.25)
    assert 0.24 < np.max(np.abs(x)) <= 0.25 + 1e-12
    faded = ph.synchronized_sweep_signal(FS, F1, F2, SECONDS, fade=0.05)
    assert abs(faded[-1]) < 1e-3  # the fade-out lands on the last samples


# ---------------------------------------------------------------------------
# Chebyshev oracle: magnitudes, phases and THD(f)
# ---------------------------------------------------------------------------


def test_polynomial_harmonic_magnitudes_synchronized() -> None:
    """|H1|, |H2|, |H3| match the Chebyshev closed forms to 2e-3 relative."""
    res = _analyze()
    h1 = abs(_at(res.frequencies, res.harmonic_responses[0], 1000.0))
    h2 = abs(_at(res.frequencies, res.harmonic_responses[1], 2000.0))
    h3 = abs(_at(res.frequencies, res.harmonic_responses[2], 3000.0))
    assert h1 == pytest.approx(H1_EXPECTED, rel=2e-3)
    assert h2 == pytest.approx(H2_EXPECTED, rel=2e-3)
    assert h3 == pytest.approx(H3_EXPECTED, rel=2e-3)


def test_polynomial_harmonic_phases_synchronized() -> None:
    """H2 sits at -π/2 and H3 at ±π: the Novak phase coherence."""
    res = _analyze()
    p2 = np.angle(_at(res.frequencies, res.harmonic_responses[1], 2000.0))
    p3 = np.angle(_at(res.frequencies, res.harmonic_responses[2], 3000.0))
    assert p2 == pytest.approx(-np.pi / 2.0, abs=5e-3)
    assert abs(p3) == pytest.approx(np.pi, abs=5e-3)


def test_polynomial_phase_coherent_across_band() -> None:
    """The H3 phase stays on ±π across the band, not just at one bin."""
    res = _analyze()
    band = (res.frequencies > 500.0) & (res.frequencies < 12000.0)
    phases = np.abs(np.angle(res.harmonic_responses[2][band]))
    np.testing.assert_allclose(phases, np.pi, atol=2e-2)


def test_polynomial_thd_matches_closed_form() -> None:
    res = _analyze()
    idx = int(np.argmin(np.abs(res.thd_frequencies - 1000.0)))
    assert res.thd[idx] == pytest.approx(THD_EXPECTED, rel=1e-3)


def test_thd_flat_for_memoryless_polynomial() -> None:
    """The polynomial THD is frequency-independent over the interior band."""
    res = _analyze()
    band = (res.thd_frequencies > 200.0) & (res.thd_frequencies < 1800.0)
    np.testing.assert_allclose(res.thd[band], THD_EXPECTED, rtol=5e-3)


def test_sweep_thd_matches_classical_tone_analyzer() -> None:
    """Same nonlinearity, same orders: sweep separator vs phonometry.thd."""
    t = np.arange(FS) / FS
    tone = np.sin(2.0 * np.pi * 1000.0 * t)
    classical = ph.thd(_polynomial_recording(tone), FS, 1000.0, n_harmonics=3)
    res = _analyze()
    idx = int(np.argmin(np.abs(res.thd_frequencies - 1000.0)))
    assert classical == pytest.approx(THD_EXPECTED, rel=1e-6)
    assert res.thd[idx] == pytest.approx(classical, rel=1e-3)


def test_linear_system_thd_is_noise_floor() -> None:
    """A purely linear path leaves only the deconvolution residue."""
    x = ph.synchronized_sweep_signal(FS, F1, F2, SECONDS)
    res = ph.swept_sine_distortion(0.5 * x, FS, f1=F1, f2=F2, seconds=SECONDS, n_harmonics=3)
    h1 = abs(_at(res.frequencies, res.harmonic_responses[0], 1000.0))
    assert h1 == pytest.approx(0.5, rel=1e-3)
    band = (res.thd_frequencies > 100.0) & (res.thd_frequencies < 2000.0)
    assert float(np.max(res.thd[band])) < 1e-3


def test_hammerstein_composition_delay_and_gain() -> None:
    """A linear post-filter (gain+delay) scales every harmonic response."""
    x = ph.synchronized_sweep_signal(FS, F1, F2, SECONDS)
    rec = np.concatenate(
        [np.zeros(100), 0.5 * _polynomial_recording(x), np.zeros(4096)]
    )
    res = ph.swept_sine_distortion(rec, FS, f1=F1, f2=F2, seconds=SECONDS, n_harmonics=3)
    h1 = abs(_at(res.frequencies, res.harmonic_responses[0], 1000.0))
    h3 = abs(_at(res.frequencies, res.harmonic_responses[2], 3000.0))
    assert h1 == pytest.approx(0.5 * H1_EXPECTED, rel=2e-3)
    assert h3 == pytest.approx(0.5 * H3_EXPECTED, rel=2e-3)
    # The THD is a ratio: the linear post-filter cancels out of it.
    idx = int(np.argmin(np.abs(res.thd_frequencies - 1000.0)))
    assert res.thd[idx] == pytest.approx(THD_EXPECTED, rel=2e-3)


def test_delays_follow_l_ln_n() -> None:
    res = _analyze()
    expected = res.rate * np.log(np.arange(1, 4))
    np.testing.assert_allclose(res.delays, expected, rtol=1e-12)
    assert res.n_harmonics == 3


# ---------------------------------------------------------------------------
# Farina (non-synchronized ESS) path
# ---------------------------------------------------------------------------


def _analyze_farina() -> ph.SweptSineDistortionResult:
    x = ph.sweep_signal(FS, F1, F2, SECONDS)
    rec = np.concatenate([_polynomial_recording(x), np.zeros(4096)])
    return ph.swept_sine_distortion(rec, FS, f1=F1, f2=F2, seconds=SECONDS, method="farina", n_harmonics=3
    )


def test_farina_magnitudes_match_chebyshev() -> None:
    res = _analyze_farina()
    assert res.method == "farina"
    h1 = abs(_at(res.frequencies, res.harmonic_responses[0], 1000.0))
    h2 = abs(_at(res.frequencies, res.harmonic_responses[1], 2000.0))
    h3 = abs(_at(res.frequencies, res.harmonic_responses[2], 3000.0))
    assert h1 == pytest.approx(H1_EXPECTED, rel=3e-3)
    assert h2 == pytest.approx(H2_EXPECTED, rel=3e-3)
    assert h3 == pytest.approx(H3_EXPECTED, rel=3e-3)


def test_farina_thd_matches_closed_form() -> None:
    res = _analyze_farina()
    idx = int(np.argmin(np.abs(res.thd_frequencies - 1000.0)))
    assert res.thd[idx] == pytest.approx(THD_EXPECTED, rel=2e-3)


def test_farina_band_is_capped_at_f2() -> None:
    """The inverse filter stops at f2, so H2 carries no signal above it."""
    res = _analyze_farina()
    beyond = (res.frequencies > 1.5 * F2) & (res.frequencies < 2.0 * F2)
    assert float(np.max(np.abs(res.harmonic_responses[1][beyond]))) < 0.1 * H2_EXPECTED
    # THD grid must stop where the order-2 product leaves the band.
    assert res.thd_frequencies[-1] <= F2 / 2.0 + res.frequencies[1]


def test_synchronized_band_extends_beyond_f2() -> None:
    """The analytic deconvolution keeps H2 valid up to 2·f2 (Novak Fig. 6)."""
    res = _analyze()
    beyond = (res.frequencies > 1.5 * F2) & (res.frequencies < 1.9 * F2)
    h2_beyond = np.abs(res.harmonic_responses[1][beyond])
    # The abrupt sweep stop at f2 leaves some ripple up here; the level
    # is still the Chebyshev value, not a filtered-out band.
    assert float(np.mean(h2_beyond)) == pytest.approx(H2_EXPECTED, rel=1e-2)
    np.testing.assert_allclose(h2_beyond, H2_EXPECTED, rtol=0.15)


# ---------------------------------------------------------------------------
# Amplitude referencing, DC removal, validation
# ---------------------------------------------------------------------------


def test_amplitude_referencing_recovers_unit_gain() -> None:
    amp = 0.25
    x = ph.synchronized_sweep_signal(FS, F1, F2, SECONDS, amplitude=amp)
    res = ph.swept_sine_distortion(
        x.copy(), FS, f1=F1, f2=F2, seconds=SECONDS, n_harmonics=2, amplitude=amp
    )
    h1 = abs(_at(res.frequencies, res.harmonic_responses[0], 1000.0))
    assert h1 == pytest.approx(1.0, rel=1e-3)


def test_dc_offset_is_removed_by_default() -> None:
    x = ph.synchronized_sweep_signal(FS, F1, F2, SECONDS)
    res_clean = ph.swept_sine_distortion(x, FS, f1=F1, f2=F2, seconds=SECONDS, n_harmonics=2)
    res_dc = ph.swept_sine_distortion(x + 0.5, FS, f1=F1, f2=F2, seconds=SECONDS, n_harmonics=2)
    h_clean = abs(_at(res_clean.frequencies, res_clean.harmonic_responses[0], 1000.0))
    h_dc = abs(_at(res_dc.frequencies, res_dc.harmonic_responses[0], 1000.0))
    assert h_dc == pytest.approx(h_clean, rel=1e-6)


def test_validation_errors() -> None:
    x = ph.synchronized_sweep_signal(FS, F1, F2, SECONDS)
    rec = _polynomial_recording(x)
    two_dimensional = rec.reshape(2, -1)
    with pytest.raises(ValueError, match="one-dimensional"):
        ph.swept_sine_distortion(two_dimensional, FS, f1=F1, f2=F2, seconds=SECONDS)
    not_finite = np.full(rec.size, np.nan)
    with pytest.raises(ValueError, match="finite"):
        ph.swept_sine_distortion(not_finite, FS, f1=F1, f2=F2, seconds=SECONDS)
    with pytest.raises(ValueError, match="method"):
        ph.swept_sine_distortion(rec, FS, f1=F1, f2=F2, seconds=SECONDS, method="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_harmonics"):
        ph.swept_sine_distortion(rec, FS, f1=F1, f2=F2, seconds=SECONDS, n_harmonics=1)
    with pytest.raises(ValueError, match="whole sweep"):
        ph.swept_sine_distortion(rec[: rec.size // 2], FS, f1=F1, f2=F2, seconds=SECONDS)
    with pytest.raises(ValueError, match="overlap"):
        ph.swept_sine_distortion(rec, FS, f1=F1, f2=F2, seconds=SECONDS, ir_length=1 << 16)
    with pytest.raises(ValueError, match="Lengthen the sweep"):
        # So many orders that the two closest arrivals are < 16 samples
        # apart and no usable window exists.
        ph.swept_sine_distortion(rec, FS, f1=F1, f2=F2, seconds=SECONDS, n_harmonics=2500)


def test_farina_rejects_orders_beyond_the_sweep_ratio() -> None:
    """Orders arriving beyond the anticausal span must raise, not wrap.

    With f2/f1 = 2 the order-3 arrival L*ln(3) exceeds the sweep duration:
    in the linear Farina deconvolution its window would wrap onto the
    causal linear response and read back a spurious copy of H1.
    """
    fs, f1, f2, seconds = 48000, 1000.0, 2000.0, 1.0
    x = ph.sweep_signal(fs, f1, f2, seconds)
    rec = np.concatenate([x, np.zeros(4096)])
    with pytest.raises(ValueError, match="anticausal"):
        ph.swept_sine_distortion(rec, fs, f1=f1, f2=f2, seconds=seconds, method="farina", n_harmonics=3,
            ir_length=1024,
        )
    # The synchronized path handles the same request: absent orders read
    # only the deconvolution floor, never a copy of the linear response.
    xs = ph.synchronized_sweep_signal(fs, f1, f2, seconds)
    res = ph.swept_sine_distortion(xs, fs, f1=f1, f2=f2, seconds=seconds, n_harmonics=3, ir_length=1024
    )
    band = (res.frequencies > 1.2 * f1) & (res.frequencies < 1.8 * f2)
    assert float(np.max(np.abs(res.harmonic_responses[2][band]))) < 0.05


def test_farina_window_overrun_blames_the_window_not_the_ratio() -> None:
    """When the order fits but its window does not, the message says so."""
    fs, f1, seconds = 48000, 500.0, 1.0
    f2 = f1 * float(np.e)  # ratio e: the order-2 arrival itself fits
    x = ph.sweep_signal(fs, f1, f2, seconds)
    rec = np.concatenate([x, np.zeros(65536)])
    with pytest.raises(ValueError, match="window around the order-2 arrival"):
        ph.swept_sine_distortion(rec, fs, f1=f1, f2=f2, seconds=seconds, method="farina", n_harmonics=2,
            ir_length=32768,
        )


def test_explicit_ir_length_below_minimum_has_its_own_message() -> None:
    x = ph.synchronized_sweep_signal(FS, F1, F2, SECONDS)
    with pytest.raises(ValueError, match="'ir_length' must be at least"):
        ph.swept_sine_distortion(x, FS, f1=F1, f2=F2, seconds=SECONDS, ir_length=8)


def test_ir_length_explicit_and_windows_centered() -> None:
    x = ph.synchronized_sweep_signal(FS, F1, F2, SECONDS)
    res = ph.swept_sine_distortion(
        _polynomial_recording(x), FS, f1=F1, f2=F2, seconds=SECONDS,
        n_harmonics=3, ir_length=2048,
    )
    assert res.harmonic_irs.shape == (3, 2048)
    # Each memoryless harmonic IR peaks at its window centre (the
    # band-limited pulse can lean a couple of samples).
    for row in res.harmonic_irs:
        assert abs(int(np.argmax(np.abs(row))) - 1024) <= 3


# ---------------------------------------------------------------------------
# .plot() renderer
# ---------------------------------------------------------------------------


def test_plot_two_panels_and_external_ax() -> None:
    res = _analyze()
    axes = res.plot()
    assert len(axes) == 2
    plt.close("all")
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext
    plt.close("all")

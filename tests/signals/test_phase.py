#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the phase utilities (minimum phase, group delay, excess phase).

Oracles: a strictly minimum-phase biquad (an RBJ peaking EQ, all poles
and zeros inside the unit circle) whose true phase the cepstral
reconstruction must match to the documented 1e-12 rad on an adequate
grid; the exact closed-form group delay of a first-order allpass,
``τ_g(ω) = (1-a²)/(1+2a·cosω+a²)`` samples; a pure fractional delay whose
excess phase is exactly ``-ωD`` and whose excess group delay is the
constant ``D``; and the invariances (a minimum-phase response has zero
excess phase; ``minimum_phase`` is idempotent and phase-blind).
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

N_BINS = 4097
FS = 48000.0

#: Documented reconstruction tolerance on a strictly minimum-phase biquad
#: sampled on a dense (4097-bin) grid.
MINPHASE_TOL_RAD = 1e-12


def _peaking_biquad() -> tuple[np.ndarray, np.ndarray]:
    """RBJ peaking EQ, +6 dB at 0.3π, Q=1.5: strictly minimum phase."""
    w0, q, gain_db = 0.3 * np.pi, 1.5, 6.0
    a_lin = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q)
    b = np.array([1.0 + alpha * a_lin, -2.0 * np.cos(w0), 1.0 - alpha * a_lin])
    a = np.array([1.0 + alpha / a_lin, -2.0 * np.cos(w0), 1.0 - alpha / a_lin])
    return b, a


def _digital_grid(n: int = N_BINS) -> np.ndarray:
    """Normalized angular frequency ω in rad/sample, DC to Nyquist inclusive."""
    return np.linspace(0.0, np.pi, n)


def _biquad_response(n: int = N_BINS) -> np.ndarray:
    b, a = _peaking_biquad()
    # Evaluate on the rfft layout (Nyquist included) the utilities expect.
    _, response = sp_signal.freqz(b, a, worN=_digital_grid(n))
    return np.asarray(response, dtype=np.complex128)


# ---------------------------------------------------------------------------
# minimum_phase: cepstral reconstruction
# ---------------------------------------------------------------------------


def test_biquad_is_strictly_minimum_phase() -> None:
    b, a = _peaking_biquad()
    assert np.all(np.abs(np.roots(b)) < 1.0)
    assert np.all(np.abs(np.roots(a)) < 1.0)


def test_minimum_phase_reconstructs_biquad_phase() -> None:
    """On a strictly min-phase response the phase comes back from |H| alone."""
    response = _biquad_response()
    rec = ph.signals.minimum_phase(response)
    err = np.abs(np.angle(rec) - np.angle(response))
    assert float(np.max(err)) < MINPHASE_TOL_RAD


def test_minimum_phase_preserves_magnitude_exactly() -> None:
    response = _biquad_response()
    rec = ph.signals.minimum_phase(response)
    # |mag * exp(j*phi)| only differs from mag by the 1-ulp rounding of
    # the complex modulus.
    np.testing.assert_allclose(np.abs(rec), np.abs(response), rtol=1e-14)


def test_minimum_phase_ignores_input_phase() -> None:
    """Magnitude array and phase-scrambled response give the same output."""
    response = _biquad_response()
    from_mag = ph.signals.minimum_phase(np.abs(response))
    scrambled = response * np.exp(-1j * _digital_grid() * 11.5)
    from_scrambled = ph.signals.minimum_phase(scrambled)
    np.testing.assert_allclose(from_mag, from_scrambled, atol=1e-12)


def test_minimum_phase_is_idempotent() -> None:
    response = _biquad_response()
    once = ph.signals.minimum_phase(response)
    twice = ph.signals.minimum_phase(once)
    np.testing.assert_allclose(np.angle(twice), np.angle(once), atol=1e-12)


def test_minimum_phase_flat_magnitude_gives_zero_phase() -> None:
    rec = ph.signals.minimum_phase(np.ones(129))
    np.testing.assert_allclose(np.angle(rec), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(rec), 1.0, atol=1e-12)


def test_minimum_phase_oversample_helps_near_circle_zeros() -> None:
    """A deep in-circle notch on a coarse grid: oversampling cuts the error.

    The log-magnitude cepstrum of a zero at radius 0.995 decays slowly, so
    the 513-bin grid aliases it; the trigonometric oversampling roughly
    halves the reconstruction error (0.031 -> 0.014 rad pinned).
    """
    radius, w0 = 0.995, 0.4 * np.pi
    b = np.array([1.0, -2.0 * radius * np.cos(w0), radius**2])
    a = np.array([1.0, -np.cos(w0), 0.25])
    _, response = sp_signal.freqz(b, a, worN=_digital_grid(513))
    err = []
    for oversample in (1, 8):
        rec = ph.signals.minimum_phase(response, oversample=oversample)
        err.append(float(np.max(np.abs(np.angle(rec) - np.angle(response)))))
    assert err[0] > 0.02
    assert err[1] < 0.6 * err[0]


# ---------------------------------------------------------------------------
# group_delay: closed forms
# ---------------------------------------------------------------------------


def test_allpass_group_delay_closed_form() -> None:
    """First-order allpass: τ_g = (1-a²)/(1+2a·cosω+a²) samples, exact."""
    a_coef = 0.5
    omega = _digital_grid()
    _, response = sp_signal.freqz([a_coef, 1.0], [1.0, a_coef], worN=omega)
    tau = ph.signals.group_delay(response, FS)  # seconds
    expected = (1.0 - a_coef**2) / (1.0 + 2.0 * a_coef * np.cos(omega) + a_coef**2) / FS
    # Central differences on this grid resolve the closed form to about
    # 1e-5 samples.
    np.testing.assert_allclose(tau, expected, atol=1e-5 / FS)


def test_pure_delay_group_delay_is_constant() -> None:
    delay_samples = 12.25
    response = np.exp(-1j * _digital_grid() * delay_samples)
    tau = ph.signals.group_delay(response, FS)
    np.testing.assert_allclose(tau, delay_samples / FS, rtol=1e-9)


def test_group_delay_matches_scipy_on_biquad() -> None:
    b, a = _peaking_biquad()
    response = _biquad_response()
    _, tau_ref = sp_signal.group_delay((b, a), w=_digital_grid())
    tau = ph.signals.group_delay(response, FS) * FS  # to samples
    np.testing.assert_allclose(tau[3:-3], tau_ref[3:-3], atol=1e-3)


# ---------------------------------------------------------------------------
# excess_phase and phase_decomposition
# ---------------------------------------------------------------------------


def test_excess_phase_of_minimum_phase_system_is_zero() -> None:
    response = _biquad_response()
    excess = ph.signals.excess_phase(response)
    assert float(np.max(np.abs(excess))) < MINPHASE_TOL_RAD


def test_excess_phase_of_delayed_biquad_is_minus_omega_d() -> None:
    delay_samples = 7.25
    omega = _digital_grid()
    delayed = _biquad_response() * np.exp(-1j * omega * delay_samples)
    excess = ph.signals.excess_phase(delayed)
    np.testing.assert_allclose(excess, -omega * delay_samples, atol=1e-9)


def test_phase_decomposition_bundles_consistently() -> None:
    delay_samples = 7.25
    omega = _digital_grid()
    delayed = _biquad_response() * np.exp(-1j * omega * delay_samples)
    res = ph.signals.phase_decomposition(delayed, FS)
    np.testing.assert_allclose(
        res.excess_phase, res.phase - res.minimum_phase, atol=0.0
    )
    np.testing.assert_allclose(res.magnitude, np.abs(delayed))
    assert res.frequencies[0] == 0.0
    assert res.frequencies[-1] == pytest.approx(FS / 2.0)
    # The all-pass group delay is the constant D (away from the grid ends).
    inner = slice(10, -10)
    np.testing.assert_allclose(
        res.excess_group_delay[inner], delay_samples / FS, rtol=5e-3
    )
    # The reconstructed minimum-phase response carries |H| with phi_min.
    np.testing.assert_allclose(
        np.angle(res.minimum_phase_response),
        np.angle(_biquad_response()),
        atol=MINPHASE_TOL_RAD,
    )


def test_phase_referenced_to_dc() -> None:
    """The measured phase is unwrapped from a zero-DC reference."""
    response = -1.0 * _biquad_response()  # polarity flip: arg = pi at DC
    res = ph.signals.phase_decomposition(response, FS)
    assert res.phase[0] == 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_errors() -> None:
    good = _biquad_response()
    two_dimensional = good.reshape(1, -1)
    with pytest.raises(ValueError, match="one-dimensional"):
        ph.signals.minimum_phase(two_dimensional)
    too_short = good[:4]
    with pytest.raises(ValueError, match="at least"):
        ph.signals.minimum_phase(too_short)
    not_finite = np.full(64, np.nan + 0j)
    with pytest.raises(ValueError, match="finite"):
        ph.signals.minimum_phase(not_finite)
    all_zero = np.zeros(64)
    with pytest.raises(ValueError, match="identically zero"):
        ph.signals.minimum_phase(all_zero)
    with pytest.raises(ValueError, match="oversample"):
        ph.signals.minimum_phase(good, oversample=0)
    with pytest.raises(ValueError, match="fs"):
        ph.signals.group_delay(good, -1.0)
    with pytest.raises(ValueError, match="fs"):
        ph.signals.phase_decomposition(good, 0.0)


def test_decomposition_parts_must_lie_on_one_grid() -> None:
    """A part off the response grid is refused when built, not when drawn.

    The plot masks ``frequencies`` for DC and applies that one mask to seven
    of the eight arrays, so a length disagreement surfaces only as numpy's
    complaint about a boolean index and two axis sizes, naming neither field.
    Nothing reads ``minimum_phase_response`` at all. And an extra axis counts
    one row per bin, so it passes every length check and reaches the plot as
    a second curve on its panel, doubling that curve's legend entry.
    """
    good = ph.signals.phase_decomposition(_biquad_response(), FS)
    per_bin = "one value per frequency bin"
    fields = (
        "frequencies",
        "magnitude",
        "phase",
        "minimum_phase",
        "excess_phase",
        "group_delay",
        "excess_group_delay",
        "minimum_phase_response",
    )
    for field in fields:
        value = np.asarray(getattr(good, field))
        cases = (
            (value[:-1], per_bin),
            (np.append(value, value[-1]), per_bin),
            (np.column_stack([value] * 2), "must have one axis"),
        )
        for wrong, fragment in cases:
            with pytest.raises(ValueError, match=rf"'{field}'.*{fragment}"):
                dataclasses.replace(good, **{field: wrong})


def test_magnitude_zeros_are_floored_not_fatal() -> None:
    """A response with an exact zero still returns finite phase."""
    response = _biquad_response()
    response[response.size // 2] = 0.0
    rec = ph.signals.minimum_phase(response)
    assert np.all(np.isfinite(rec))


# ---------------------------------------------------------------------------
# .plot() renderer
# ---------------------------------------------------------------------------


def test_plot_three_panels_and_external_ax() -> None:
    res = ph.signals.phase_decomposition(_biquad_response(), FS)
    axes = res.plot()
    assert len(axes) == 3
    plt.close("all")
    _fig, ext = plt.subplots()
    assert res.plot(ax=ext) is ext
    plt.close("all")

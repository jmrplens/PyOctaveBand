#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for complementary Golay-pair impulse-response acquisition.

Validation strategy (closed-form, not self-consistency):
- The complementary property is an algebraic identity: the sum of the two
  periodic autocorrelations equals ``2L`` at zero lag and *exactly* zero
  at every other lag (Havelock 2008, Part I Ch. 6 (Xiang), Eq. (2)) --
  asserted at machine precision for every supported order.
- The recovery chain (Xiang Eq. (4)) applied to a known synthetic system
  (a pure delay + gain, and an RBJ-style biquad) returns its impulse
  response to machine precision in the noiseless case.
- Uncorrelated noise is attenuated by synchronous averaging while the
  deterministic recovery stays exact.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from phonometry import room
from phonometry.room.impulse_response import ImpulseResponseWarning


def _periodic_response(code: np.ndarray, b: np.ndarray, a: np.ndarray,
                       periods: int = 3) -> np.ndarray:
    """Steady-state periodic system response: keep the last ``periods``."""
    warmup = 2
    x = np.tile(code, periods + warmup)
    y = signal.lfilter(b, a, x)
    return y[warmup * code.size:]


# --------------------------------------------------------------------------
# Pair generation and the complementary identity
# --------------------------------------------------------------------------
def test_golay_pair_first_orders_match_recursion() -> None:
    a1, b1 = room.golay_pair(1)
    assert np.array_equal(a1, [1.0, 1.0])
    assert np.array_equal(b1, [1.0, -1.0])
    a2, b2 = room.golay_pair(2)
    # Xiang's worked example: a2 = ++.+-, b2 = ++-+
    assert np.array_equal(a2, [1.0, 1.0, 1.0, -1.0])
    assert np.array_equal(b2, [1.0, 1.0, -1.0, 1.0])


@pytest.mark.parametrize("order", range(1, 15))
def test_complementary_autocorrelation_identity(order: int) -> None:
    """sum of periodic autocorrelations == 2L*delta, machine exact."""
    a, b = room.golay_pair(order)
    length = 1 << order
    assert a.size == length
    assert b.size == length
    acorr = np.fft.irfft(
        np.abs(np.fft.rfft(a)) ** 2 + np.abs(np.fft.rfft(b)) ** 2, length
    )
    assert acorr[0] == pytest.approx(2.0 * length, abs=1e-9)
    # The sidelobes are algebraically zero; only FFT rounding remains.
    assert np.max(np.abs(acorr[1:])) < 2.0 * length * 1e-14


def test_golay_pair_is_bipolar() -> None:
    a, b = room.golay_pair(10)
    assert set(np.unique(a)) == {-1.0, 1.0}
    assert set(np.unique(b)) == {-1.0, 1.0}


@pytest.mark.parametrize("order", [0, -3, 23])
def test_golay_pair_rejects_bad_order(order: int) -> None:
    with pytest.raises(ValueError, match="order"):
        room.golay_pair(order)


# --------------------------------------------------------------------------
# Impulse-response recovery (Xiang Eq. (4))
# --------------------------------------------------------------------------
def test_recovers_delay_and_gain_to_machine_precision() -> None:
    pair = room.golay_pair(8)
    a, b = pair
    gain, delay = 0.75, 37
    rec_a = gain * np.roll(a, delay)
    rec_b = gain * np.roll(b, delay)
    res = room.golay_impulse_response(rec_a, rec_b, pair)
    assert isinstance(res, room.ImpulseResponseResult)
    assert res.method == "golay"
    expected = np.zeros(a.size)
    expected[delay] = gain
    np.testing.assert_allclose(res.ir, expected, atol=1e-13)


def test_recovers_biquad_to_machine_precision() -> None:
    # An RBJ-style peaking biquad (any short IIR works: the recovery is
    # circular, so the IR must decay within one code period).
    b, a = signal.iirpeak(0.12, Q=4.0)
    pair = room.golay_pair(12)
    rec_a = _periodic_response(pair[0], b, a)
    rec_b = _periodic_response(pair[1], b, a)
    res = room.golay_impulse_response(rec_a, rec_b, pair, fs=48000)
    imp = np.zeros(pair[0].size)
    imp[0] = 1.0
    true_ir = signal.lfilter(b, a, imp)
    np.testing.assert_allclose(res.ir, true_ir, atol=1e-12)
    assert res.fs == 48000


def test_multi_period_recordings_average_noise_down() -> None:
    pair = room.golay_pair(10)
    length = pair[0].size
    rng = np.random.default_rng(7)
    delay = 11
    periods = 8
    noise_a = 0.05 * rng.standard_normal(periods * length)
    noise_b = 0.05 * rng.standard_normal(periods * length)
    rec_a = np.tile(np.roll(pair[0], delay), periods) + noise_a
    rec_b = np.tile(np.roll(pair[1], delay), periods) + noise_b
    res = room.golay_impulse_response(rec_a, rec_b, pair)
    single = room.golay_impulse_response(
        rec_a[:length], rec_b[:length], pair
    )
    expected = np.zeros(length)
    expected[delay] = 1.0
    err_avg = float(np.sqrt(np.mean((res.ir - expected) ** 2)))
    err_one = float(np.sqrt(np.mean((single.ir - expected) ** 2)))
    # 8 synchronous averages: noise RMS falls by sqrt(8) ~ 2.83.
    assert err_avg < err_one / 2.0


def test_length_trims_and_extends_periodically() -> None:
    pair = room.golay_pair(6)
    rec_a, rec_b = pair
    short = room.golay_impulse_response(rec_a, rec_b, pair, length=10)
    assert short.size == 10
    long = room.golay_impulse_response(rec_a, rec_b, pair, length=100)
    assert long.size == 100
    np.testing.assert_allclose(long.ir[64:74], short.ir[:10], atol=1e-15)


def test_alias_warning_when_ir_longer_than_period() -> None:
    # A slowly decaying system relative to a short pair aliases circularly.
    b, a = signal.butter(2, 0.01)
    pair = room.golay_pair(6)
    rec_a = _periodic_response(pair[0], b, a, periods=1)
    rec_b = _periodic_response(pair[1], b, a, periods=1)
    with pytest.warns(ImpulseResponseWarning, match="Golay"):
        room.golay_impulse_response(rec_a, rec_b, pair)


def test_input_validation() -> None:
    pair = room.golay_pair(4)
    valid = np.zeros(16)
    non_multiple = np.zeros(10)
    empty = np.zeros(0)
    truncated_pair = (pair[0], pair[1][:8])
    two_dim = np.zeros((2, 16))
    with pytest.raises(ValueError, match="multiple"):
        room.golay_impulse_response(non_multiple, valid, pair)
    with pytest.raises(ValueError, match="multiple"):
        room.golay_impulse_response(valid, empty, pair)
    with pytest.raises(ValueError, match="equal-length"):
        room.golay_impulse_response(valid, valid, truncated_pair)
    with pytest.raises(ValueError, match="one-dimensional"):
        room.golay_impulse_response(two_dim, valid, pair)


def test_result_behaves_like_array() -> None:
    pair = room.golay_pair(5)
    res = room.golay_impulse_response(pair[0], pair[1], pair)
    assert len(res) == 32
    assert np.asarray(res).shape == (32,)
    assert res[0] == pytest.approx(1.0)

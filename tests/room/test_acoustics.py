#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO 3382-1:2009 / ISO 3382-2:2008 room acoustic parameters.

Validation strategy (closed forms, not self-consistency). For a purely
exponential energy decay p^2(t) = exp(-a*t) with a = 6*ln(10)/T:

- The Schroeder curve is an exact straight line L(t) = -60*t/T dB, so
  EDT = T20 = T30 = T for any evaluation range (ISO 3382-1, 5.3.3 / A.2.2).
- C_te = 10*lg(exp(a*te) - 1) dB (from ISO 3382-1 Eq. A.10),
  e.g. C80 = 10*lg(exp(13.8155*0.08/T) - 1).
- D50 = 1 - exp(-a*0.05)  (Eq. A.11), with C50 = 10*lg(D50/(1-D50))
  (Eq. A.12).
- Ts = 1/a = T/13.8155  (centre of gravity of exp(-a*t), Eq. A.13).

Tolerances are chosen far below the ISO 3382-1 Table A.1 JNDs
(EDT: rel. 5 %; C80: 1 dB; D50: 0.05; Ts: 10 ms).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phonometry import room

FS = 48000
#: 6*ln(10): decay-rate constant so that exp(-A*t/T) falls 60 dB in T seconds.
A60 = 6.0 * np.log(10.0)


def exponential_ir(t60: float, seconds: float, fs: int = FS) -> np.ndarray:
    """Pressure IR whose energy envelope is exactly exp(-A60*t/t60)."""
    t = np.arange(round(seconds * fs)) / fs
    return np.asarray(np.exp(-0.5 * A60 * t / t60))


def multitone_ir(
    t60: float, seconds: float, freqs: list[float], fs: int = FS
) -> np.ndarray:
    """One sine carrier per octave band, all sharing the same exponential
    energy envelope, so every band-filtered envelope is exp(-A60*t/t60).
    """
    t = np.arange(round(seconds * fs)) / fs
    env = np.exp(-0.5 * A60 * t / t60)
    out = np.zeros_like(t)
    for f in freqs:
        out += np.sin(2.0 * np.pi * f * t)
    return out * env


def c_te_closed_form(t60: float, te: float) -> float:
    """C_te for a pure exponential decay: 10*lg(exp(A60*te/T) - 1) dB."""
    return float(10.0 * np.log10(np.exp(A60 * te / t60) - 1.0))


# --------------------------------------------------------------------------
# decay_curve (Schroeder backward integration, ISO 3382-1 5.3.3)
# --------------------------------------------------------------------------
def test_decay_curve_is_linear_for_exponential() -> None:
    # L(t) = -60*t/T exactly for an exponential decay (5.3.3, Eq. 1).
    t60 = 1.0
    ir = exponential_ir(t60, 3.0)
    t, level = room.decay_curve(ir, FS)
    assert level[0] == pytest.approx(0.0, abs=1e-9)
    # Compare over the first 40 dB of decay.
    mask = level >= -40.0
    np.testing.assert_allclose(level[mask], -60.0 * t[mask] / t60, atol=0.1)


def test_decay_curve_monotone_nonincreasing() -> None:
    rng = np.random.default_rng(7)
    ir = exponential_ir(0.7, 2.0) * rng.standard_normal(int(2.0 * FS))
    _, level = room.decay_curve(ir, FS)
    assert np.all(np.diff(level) <= 1e-12)


def test_decay_curve_single_band() -> None:
    ir = multitone_ir(0.8, 2.5, [500.0])
    t, level = room.decay_curve(ir, FS, band=500.0)
    assert t.shape == level.shape
    assert level[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.isfinite(level))


# --------------------------------------------------------------------------
# Reverberation times: closed form for pure exponential decay
# --------------------------------------------------------------------------
@pytest.mark.parametrize("t60", [0.5, 1.0, 2.0])
def test_rt_exponential_broadband(t60: float) -> None:
    # T20 = T30 = EDT = T within +/-1 % (far below the Table A.1 JND of
    # rel. 5 % for EDT / reverberation).
    ir = exponential_ir(t60, 3.0 * t60)
    res = room.room_parameters(ir, FS, limits=None)
    assert res.frequency is None
    assert res.edt[0] == pytest.approx(t60, rel=0.01)
    assert res.t20[0] == pytest.approx(t60, rel=0.01)
    assert res.t30[0] == pytest.approx(t60, rel=0.01)
    assert bool(res.edt_valid[0])
    assert bool(res.t20_valid[0])
    assert bool(res.t30_valid[0])
    # Perfectly straight decay: curvature ~ 0 % (ISO 3382-2, B.3).
    assert abs(res.curvature[0]) < 2.0


def test_rt_exponential_per_octave_band() -> None:
    # Each octave band sees the same exponential envelope: T within +/-1 %.
    t60 = 1.0
    centers = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
    ir = multitone_ir(t60, 3.0, centers)
    res = room.room_parameters(ir, FS)
    assert res.frequency is not None
    assert len(res.frequency) == 6
    np.testing.assert_allclose(res.t20, t60, rtol=0.01)
    np.testing.assert_allclose(res.t30, t60, rtol=0.01)
    np.testing.assert_allclose(res.edt, t60, rtol=0.01)
    assert bool(np.all(res.t30_valid))
    # Band-filter group delay smears some early energy past 80 ms, so the
    # per-band C80 sits below the closed form; the smearing is worst in the
    # 125 Hz band (~0.58 dB) and smaller above. 0.7 dB (below the 1 dB
    # Table A.1 JND, was a full 1.0) locks the current accuracy in.
    np.testing.assert_allclose(res.c80, c_te_closed_form(t60, 0.080), atol=0.7)


def test_zero_phase_reduces_short_decay_group_delay_bias() -> None:
    """ISO 3382-2:2008 Clause 7.3 NOTE: time-reversed (zero-phase) octave
    filtering removes the filter group delay, which otherwise inflates the
    apparent decay time of a short low-frequency decay. On a deterministic
    125 Hz damped sinusoid (T = 0.2 s) the zero-phase T20/T30 sit closer to
    the true value than the causal fit.
    """
    t60, fc = 0.2, 125.89254118
    n = int(1.2 * FS)
    t = np.arange(n) / FS
    # Amplitude decays A60/2 (energy decays A60, i.e. 60 dB) over t60.
    ir = np.exp(-0.5 * A60 * t / t60) * np.sin(2 * np.pi * fc * t)
    causal = room.room_parameters(ir, FS, limits=(112.0, 140.0), fraction=1)
    zp = room.room_parameters(
        ir, FS, limits=(112.0, 140.0), fraction=1, zero_phase=True
    )
    for arr in (causal.t20, causal.t30, zp.t20, zp.t30):
        assert np.isfinite(arr[0])
    # Both over-estimate the short decay (positive group-delay bias); the
    # zero-phase estimate is strictly closer to the true T.
    assert t60 <= zp.t20[0] < causal.t20[0]
    assert t60 <= zp.t30[0] < causal.t30[0]


# --------------------------------------------------------------------------
# Energy parameters: closed forms (ISO 3382-1, A.2.3)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("t60", [0.5, 1.0, 2.0])
def test_clarity_closed_form(t60: float) -> None:
    # C80 = 10*lg(exp(13.8155*0.08/T) - 1) dB; tolerance << 1 dB JND.
    ir = exponential_ir(t60, 3.0 * t60)
    res = room.room_parameters(ir, FS, limits=None)
    assert res.c80[0] == pytest.approx(c_te_closed_form(t60, 0.080), abs=0.05)
    assert res.c50[0] == pytest.approx(c_te_closed_form(t60, 0.050), abs=0.05)


@pytest.mark.parametrize("t60", [0.5, 1.0, 2.0])
def test_definition_and_centre_time_closed_form(t60: float) -> None:
    ir = exponential_ir(t60, 3.0 * t60)
    res = room.room_parameters(ir, FS, limits=None)
    # D50 = 1 - exp(-13.8155*0.05/T); tolerance << 0.05 JND.
    d50_expected = 1.0 - np.exp(-A60 * 0.05 / t60)
    assert res.d50[0] == pytest.approx(d50_expected, abs=0.005)
    # Ts = T/13.8155 seconds; tolerance 0.5 ms << 10 ms JND.
    assert res.ts[0] == pytest.approx(t60 / A60, abs=5e-4)


def test_c50_d50_exact_relation() -> None:
    # C50 = 10*lg(D50/(1 - D50)) must hold exactly (Eq. A.12).
    ir = exponential_ir(1.2, 3.5)
    res = room.room_parameters(ir, FS, limits=None)
    d50 = float(res.d50[0])
    assert res.c50[0] == pytest.approx(10.0 * np.log10(d50 / (1.0 - d50)), abs=1e-9)


def test_energy_parameters_shift_invariant() -> None:
    # t = 0 is the start of the direct sound (A.2.1): prepending silence
    # must not change the energy parameters.
    t60 = 1.0
    ir = exponential_ir(t60, 3.0)
    shifted = np.concatenate([np.zeros(int(0.05 * FS)), ir])
    res_a = room.room_parameters(ir, FS, limits=None)
    res_b = room.room_parameters(shifted, FS, limits=None)
    assert res_b.c80[0] == pytest.approx(float(res_a.c80[0]), abs=0.05)
    assert res_b.ts[0] == pytest.approx(float(res_a.ts[0]), abs=5e-4)
    assert res_b.t30[0] == pytest.approx(float(res_a.t30[0]), rel=0.01)


# --------------------------------------------------------------------------
# Double-slope decay: EDT != T30 with the correct direction
# --------------------------------------------------------------------------
def test_double_slope_edt_below_t30() -> None:
    # Fast early decay (T = 0.4 s) with a weak slow tail (T = 2.0 s,
    # -20 dB): the initial 0/-10 dB slope is steep (EDT ~ fast decay)
    # while -5/-35 dB includes the shallow tail, so EDT < T20 < T30.
    fs = FS
    t = np.arange(int(4.0 * fs)) / fs
    p2 = np.exp(-A60 * t / 0.4) + 1e-2 * np.exp(-A60 * t / 2.0)
    ir = np.sqrt(p2)
    res = room.room_parameters(ir, fs, limits=None)
    assert np.isfinite(res.edt[0])
    assert np.isfinite(res.t30[0])
    assert res.edt[0] < res.t20[0] < res.t30[0]
    # Strong curvature indicator (ISO 3382-2, B.3: C > 10 % is suspicious).
    assert res.curvature[0] > 10.0


# --------------------------------------------------------------------------
# Background noise: validity flags (ISO 3382-1, 5.3.3 / 5.2.1)
# --------------------------------------------------------------------------
def test_noise_flips_validity_flags() -> None:
    # Noise floor ~30 dB below the IR maximum: EDT needs 25 dB (10+15),
    # T20 needs 35 dB (20+15) and T30 needs 45 dB (30+15), so only EDT
    # stays valid (5.3.3: noise at least evaluation range + 15 dB below
    # the maximum of the impulse response).
    rng = np.random.default_rng(11)
    ir = exponential_ir(1.0, 3.0)
    noisy = ir + 10.0 ** (-30.0 / 20.0) * rng.standard_normal(ir.size)
    res = room.room_parameters(noisy, FS, limits=None)
    assert bool(res.edt_valid[0])
    assert not bool(res.t20_valid[0])
    assert not bool(res.t30_valid[0])
    # Reported dynamic range should sit near the constructed 30 dB.
    assert res.dynamic_range[0] == pytest.approx(30.0, abs=3.0)


def test_decay_validity_thresholds_account_for_tail_compensation() -> None:
    """The T20/T30 validity flags must require more dynamic range than the
    bare ISO 3382-1 +15 dB rule (old thresholds 35/45 dB peak-to-noise).
    That rule is derived for finite forward integration with no tail
    compensation (C = 0, which under-estimates T); this module compensates
    the truncated tail (Schroeder Eq. (3), C != 0), whose residual bias is
    positive, so a flagged-valid T20 at dyn=35 / T30 at dyn=45 exceeds the
    5 % JND (ISO 3382-2 Table A.1). Flags must tighten to ~46 dB (T20) and
    ~54 dB (T30), where the bias falls below the JND.
    """

    def noisy_exp(sigma: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        ir = exponential_ir(1.0, 3.0)
        return ir + sigma * rng.standard_normal(ir.size)

    # dyn ~ 40 dB: above the OLD T20 threshold (35 dB, would flag valid),
    # below the NEW one (46 dB) -> must now be flagged INVALID.
    res = room.room_parameters(noisy_exp(0.010, 0), FS, limits=None)
    assert res.dynamic_range[0] == pytest.approx(40.0, abs=2.0)
    assert np.isfinite(res.t20[0])
    assert not bool(res.t20_valid[0])

    # dyn ~ 52 dB: T20 now valid (>= 46) with bias < JND; T30 still below its
    # NEW threshold (54 dB) though above the OLD one (45 dB) -> T30 invalid.
    res = room.room_parameters(noisy_exp(0.0025, 0), FS, limits=None)
    assert res.dynamic_range[0] == pytest.approx(52.0, abs=2.0)
    assert bool(res.t20_valid[0])
    assert abs(res.t20[0] - 1.0) < 0.05
    assert np.isfinite(res.t30[0])
    assert not bool(res.t30_valid[0])

    # dyn ~ 56 dB: T30 now valid (>= 54) with bias < JND.
    res = room.room_parameters(noisy_exp(0.0016, 0), FS, limits=None)
    assert res.dynamic_range[0] == pytest.approx(56.0, abs=2.0)
    assert bool(res.t30_valid[0])
    assert abs(res.t30[0] - 1.0) < 0.05


def test_clean_ir_flags_all_valid() -> None:
    ir = exponential_ir(1.0, 3.0)
    res = room.room_parameters(ir, FS, limits=None)
    assert bool(res.edt_valid[0] and res.t20_valid[0] and res.t30_valid[0])
    assert res.dynamic_range[0] > 45.0


def test_short_ir_yields_nan_and_invalid() -> None:
    # 0.3 s of a T = 2 s decay only covers ~9 dB: T30 (needs -35 dB) is
    # not evaluable.
    ir = exponential_ir(2.0, 0.3)
    res = room.room_parameters(ir, FS, limits=None)
    assert np.isnan(res.t30[0])
    assert not bool(res.t30_valid[0])


# --------------------------------------------------------------------------
# Band handling and input validation
# --------------------------------------------------------------------------
def test_third_octave_bands_option() -> None:
    rng = np.random.default_rng(3)
    ir = exponential_ir(0.8, 2.0) * rng.standard_normal(int(2.0 * FS))
    res = room.room_parameters(ir, FS, limits=(100.0, 5000.0), fraction=3)
    assert res.frequency is not None
    assert len(res.frequency) == 18  # 100 Hz..5 kHz one-third octaves
    assert res.t20.shape == (18,)
    assert isinstance(res, room.RoomAcousticsResult)


def test_default_bands_are_octaves_125_to_4k() -> None:
    rng = np.random.default_rng(5)
    ir = exponential_ir(0.8, 2.0) * rng.standard_normal(int(2.0 * FS))
    res = room.room_parameters(ir, FS)
    assert res.frequency is not None
    np.testing.assert_allclose(
        res.frequency,
        [
            125.89254117941672,
            251.188643150958,
            501.18723362727235,
            1000.0,
            1995.2623149688795,
            3981.0717055349724,
        ],
    )


def test_rejects_bad_input() -> None:
    two_dimensional = np.zeros((2, 100))
    silent = np.zeros(100)
    decaying = exponential_ir(1.0, 1.0)
    with pytest.raises(ValueError, match="impulse response must be one-dimensional"):
        room.room_parameters(two_dimensional, FS)  # 2D
    with pytest.raises(ValueError, match="'ir' is silent"):
        room.room_parameters(silent, FS)  # silent
    with pytest.raises(ValueError, match="'fs' must be positive"):
        room.room_parameters(decaying, 0)  # bad fs
    with pytest.raises(ValueError, match="'ir' is silent"):
        room.decay_curve(silent, FS)  # silent


def test_constant_noise_no_decay_is_finite_and_invalid() -> None:
    """A near-constant-noise 'IR' with no decay must not overflow.

    A barely-negative fitted slope (e.g. -1e-16 dB/s) would otherwise make
    the tail-compensation terms overflow to inf; the -1e-7 dB/s cutoff in
    _truncation forces a graceful no-truncation return with invalid decays.
    """
    rng = np.random.default_rng(0)
    ir = rng.standard_normal(FS)  # stationary noise, no decay envelope
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = room.room_parameters(ir, FS, limits=None)
    # No decay is measurable -> the T-parameters are NaN and flagged invalid.
    assert not bool(res.edt_valid[0])
    assert not bool(res.t20_valid[0])
    assert not bool(res.t30_valid[0])
    # Nothing overflowed to +/-inf (energy metrics stay finite).
    for arr in (res.edt, res.t20, res.t30, res.c50, res.c80, res.d50, res.ts):
        assert not np.any(np.isinf(arr))


def test_band_plot_uses_nominal_frequency_labels() -> None:
    """The per-band decay-time plot labels bars with nominal band centres.

    The categorical octave axis must read the IEC 61260 nominal centres
    (125, 250, 500, 1k, 2k, 4k), matching the nominal frequency table a
    report prints, not the exact base-ten filter centres (125.89..., 1.99526k).
    """
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    ir = multitone_ir(1.0, 3.0, [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    res = room.room_parameters(ir, FS)
    fig, ax = plt.subplots()
    try:
        res.plot(ax=ax)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
    finally:
        plt.close(fig)
    assert labels == ["125", "250", "500", "1k", "2k", "4k"]

#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the regularized spectral inversion (Kirkeby & Nelson 1999,
Mueller & Massarani 2001).

Validation strategy (closed-form, not self-consistency):
- In-band the equalized magnitude ``|H*H_inv|`` equals
  ``|H|**2 / (|H|**2 + eps)``: unity within the analytic residue
  ``eps / (|H|**2 + eps)``, checked bin by bin against the designed
  regularization profile.
- Out-of-band the filter gain is bounded by the analytic maximum of
  ``x / (x**2 + eps)``, i.e. ``1 / (2*sqrt(eps))`` -- the Kirkeby cap that
  replaces Mueller & Massarani's explicit band-pass.
- Applying the filter to the response it was designed from returns a
  delayed band-limited unit pulse; ``apply`` removes the delay.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import signal

from phonometry import room, signals

FS = 48000.0


def _biquad_ir(n: int = 512) -> np.ndarray:
    """A gentle peaking response: unity plus an RBJ-style bell at 2 kHz."""
    b, a = signal.iirpeak(2000.0, Q=2.0, fs=FS)
    imp = np.zeros(n)
    imp[0] = 1.0
    return imp + 0.5 * signal.lfilter(b, a, imp)  # ~ +3.5 dB bump


def _bandpass_ir(n: int = 1024) -> np.ndarray:
    """A loudspeaker-like band-pass (100 Hz - 8 kHz) IR."""
    b, a = signal.butter(2, [100.0, 8000.0], btype="bandpass", fs=FS)
    imp = np.zeros(n)
    imp[0] = 1.0
    return signal.lfilter(b, a, imp)


# --------------------------------------------------------------------------
# The Kirkeby closed forms
# --------------------------------------------------------------------------
def test_in_band_product_is_unity_within_regularization_bound() -> None:
    h = _bandpass_ir()
    reg = 1e-6
    res = signals.regularized_inverse_filter(
        h, FS, f_range=(200.0, 4000.0), regularization_inside=reg
    )
    product = np.abs(res.response_spectrum * res.spectrum)
    band = (res.frequencies >= 200.0) & (res.frequencies <= 4000.0)
    power = np.abs(res.response_spectrum) ** 2
    # |H*Hinv| = |H|^2/(|H|^2 + eps): the deviation from 1 is exactly
    # eps/(|H|^2 + eps), bin by bin (machine precision).
    residue = res.regularization[band] / (power[band] + res.regularization[band])
    np.testing.assert_allclose(1.0 - product[band], residue, atol=1e-12)
    # And the global closed-form bound: eps_in = reg * max|H|^2.
    bound = reg * float(np.max(power)) / float(np.min(power[band]))
    assert float(np.max(1.0 - product[band])) <= bound
    assert res.flatness_db <= 20.0 * np.log10(1.0 / (1.0 - bound))


def test_out_of_band_gain_is_capped_by_the_analytic_maximum() -> None:
    h = _bandpass_ir()
    reg_out = 1.0
    res = signals.regularized_inverse_filter(
        h, FS, f_range=(200.0, 4000.0), regularization_outside=reg_out
    )
    ratio = 2.0 ** (1.0 / 3.0)
    outside = (res.frequencies > 0.0) & (
        (res.frequencies < 200.0 / ratio) | (res.frequencies > 4000.0 * ratio)
    )
    gain = np.abs(res.spectrum[outside])
    peak = float(np.max(np.abs(res.response_spectrum)))
    # max of x/(x^2 + eps) is 1/(2*sqrt(eps)); eps = reg_out * peak^2.
    cap = 1.0 / (2.0 * np.sqrt(reg_out) * peak)
    assert float(np.max(gain)) <= cap + 1e-15
    assert res.max_gain_db <= 20.0 * np.log10(cap * peak) + 1e-9


def test_flatness_reports_the_worst_in_band_deviation() -> None:
    h = _biquad_ir()
    res = signals.regularized_inverse_filter(h, FS, f_range=(500.0, 8000.0))
    product = np.abs(res.response_spectrum * res.spectrum)
    band = (res.frequencies >= 500.0) & (res.frequencies <= 8000.0)
    worst = float(np.max(np.abs(20.0 * np.log10(product[band]))))
    assert res.flatness_db == pytest.approx(worst, abs=1e-12)
    assert res.flatness_db < 0.01  # a gentle response inverts almost exactly


def test_stronger_regularization_trades_flatness_for_smaller_gain() -> None:
    h = _bandpass_ir()
    gentle = signals.regularized_inverse_filter(
        h, FS, f_range=(200.0, 4000.0), regularization_inside=1e-8
    )
    strong = signals.regularized_inverse_filter(
        h, FS, f_range=(200.0, 4000.0), regularization_inside=1e-2
    )
    assert gentle.flatness_db < strong.flatness_db


# --------------------------------------------------------------------------
# Time-domain behaviour
# --------------------------------------------------------------------------
def test_apply_equalizes_the_designed_response_to_a_pulse() -> None:
    h = _bandpass_ir()
    res = signals.regularized_inverse_filter(h, FS, f_range=(200.0, 4000.0))
    # The raw convolution concentrates into a band-limited pulse at the
    # modeling delay: > 95 % of the energy within +/- 128 samples.
    full = signal.fftconvolve(h, res.inverse)
    assert int(np.argmax(np.abs(full))) == res.delay
    window = slice(res.delay - 128, res.delay + 128)
    concentration = float(np.sum(full[window] ** 2) / np.sum(full**2))
    assert concentration > 0.95
    # apply() removes the delay: same pulse, aligned at sample 0.
    pulse = res.apply(h)
    assert pulse.size == h.size
    assert int(np.argmax(np.abs(pulse))) == 0
    assert pulse[0] == pytest.approx(float(full[res.delay]), abs=1e-12)


def test_delay_defaults_to_half_the_block_and_is_stored() -> None:
    h = _biquad_ir(200)
    res = signals.regularized_inverse_filter(h, FS, f_range=(500.0, 8000.0))
    assert res.size == 512  # next pow2 of 2*200
    assert res.delay == 256
    custom = signals.regularized_inverse_filter(
        h, FS, f_range=(500.0, 8000.0), n_fft=1024, delay=100
    )
    assert custom.size == 1024
    assert custom.delay == 100


def test_fs_is_taken_from_a_result_object() -> None:
    sweep = room.sweep_signal(int(FS), 50.0, 20000.0, 0.5)
    rec = np.concatenate([sweep, np.zeros(2048)])
    ir = room.impulse_response(
        rec, np.concatenate([sweep, np.zeros(2048)]), int(FS), length=2048
    )
    res = signals.regularized_inverse_filter(ir, f_range=(200.0, 10000.0))
    assert isinstance(res, signals.InverseFilterResult)
    assert res.fs == FS
    bare = np.asarray(ir)
    with pytest.raises(ValueError, match="fs"):
        signals.regularized_inverse_filter(bare, f_range=(200.0, 10000.0))


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def test_rejects_bad_inputs() -> None:
    h = _biquad_ir()
    with pytest.raises(ValueError, match="f_range"):
        signals.regularized_inverse_filter(h, FS, f_range=(0.0, 4000.0))
    with pytest.raises(ValueError, match="f_range"):
        signals.regularized_inverse_filter(h, FS, f_range=(4000.0, 200.0))
    with pytest.raises(ValueError, match="Nyquist"):
        signals.regularized_inverse_filter(h, FS, f_range=(200.0, 40000.0))
    with pytest.raises(ValueError, match="regularization"):
        signals.regularized_inverse_filter(
            h, FS, f_range=(200.0, 4000.0), regularization_inside=0.0
        )
    with pytest.raises(ValueError, match="transition"):
        signals.regularized_inverse_filter(
            h, FS, f_range=(200.0, 4000.0), transition_octaves=-1.0
        )
    with pytest.raises(ValueError, match="n_fft"):
        signals.regularized_inverse_filter(h, FS, f_range=(200.0, 4000.0), n_fft=16)
    with pytest.raises(ValueError, match="delay"):
        signals.regularized_inverse_filter(h, FS, f_range=(200.0, 4000.0), delay=-1)
    two_dim = np.zeros((2, 8))
    with pytest.raises(ValueError, match="one-dimensional"):
        signals.regularized_inverse_filter(two_dim, FS, f_range=(200.0, 4000.0))
    with_nan = np.array([1.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        signals.regularized_inverse_filter(with_nan, FS, f_range=(200.0, 4000.0))
    all_zero = np.zeros(64)
    with pytest.raises(ValueError, match="zero"):
        signals.regularized_inverse_filter(all_zero, FS, f_range=(200.0, 4000.0))


def test_design_must_lie_on_one_frequency_grid() -> None:
    """A design off its own grid is refused when built, not when drawn.

    ``plot_inverse_filter`` masks every magnitude with the positive-frequency
    mask taken from ``frequencies``, so a mismatched ``frequencies``,
    ``spectrum`` or ``response_spectrum`` already stops the figure there --
    ``IndexError`` from the boolean index, or ``ValueError`` from the
    product, naming neither field. ``regularization`` is the silent one: no
    renderer opens it, and a profile short, long or a single bin wide draws
    the whole figure without a word. An extra axis is silent too, and
    doubles the ``size`` the result reports.
    """
    res = signals.regularized_inverse_filter(_biquad_ir(), FS, f_range=(500.0, 8000.0))
    per_bin = "one value per frequency"
    one_axis = "must have one axis"
    cases = (
        ("regularization", res.regularization[:-1], per_bin),
        ("regularization", np.append(res.regularization, 1.0), per_bin),
        ("regularization", res.regularization[:1], per_bin),
        ("frequencies", res.frequencies[:-1], per_bin),
        ("spectrum", res.spectrum[:1], per_bin),
        ("response_spectrum", res.response_spectrum[:-1], per_bin),
        ("inverse", np.column_stack([res.inverse] * 2), one_axis),
        ("regularization", np.column_stack([res.regularization] * 2), one_axis),
    )
    for field, value, fragment in cases:
        with pytest.raises(ValueError, match=rf"'{field}'.*{fragment}"):
            dataclasses.replace(res, **{field: value})


def test_flatness_must_restate_the_equalized_magnitude() -> None:
    """The figure title cannot claim a ripple the drawn curve never shows.

    ``plot_inverse_filter`` prints ``flatness_db`` in the title and draws
    ``|H*H_inv|`` on the axes below it, so before this guard a variant built
    with ``dataclasses.replace`` titled the figure "flatness 6.00 dB" over a
    curve that never left 0.00 dB.
    """
    res = signals.regularized_inverse_filter(_biquad_ir(), FS, f_range=(500.0, 8000.0))
    with pytest.raises(ValueError, match=r"'flatness_db' must be the largest"):
        dataclasses.replace(res, flatness_db=6.0)


def test_flatness_must_follow_the_spectra_it_summarises() -> None:
    """Restating a spectrum restates the flatness, or the pair is refused."""
    res = signals.regularized_inverse_filter(_biquad_ir(), FS, f_range=(500.0, 8000.0))
    band = np.flatnonzero((res.frequencies >= 500.0) & (res.frequencies <= 8000.0))
    notched = res.response_spectrum.copy()
    notched[band[0]] = 0.0  # a null the inversion cannot fill: |H*H_inv| = 0
    with pytest.raises(ValueError, match=r"'flatness_db' must be the largest"):
        dataclasses.replace(res, response_spectrum=notched)
    # The same null with the flatness it implies is a reading, not a defect:
    # infinitely far from 0 dB is what that band is, and it is accepted.
    with_null = dataclasses.replace(res, response_spectrum=notched, flatness_db=np.inf)
    assert with_null.flatness_db == np.inf


def test_flatness_needs_a_band_to_be_worst_over() -> None:
    """A band between two bins summarises nothing, and is refused."""
    res = signals.regularized_inverse_filter(_biquad_ir(), FS, f_range=(500.0, 8000.0))
    step = FS / res.size  # 46.875 Hz: no bin falls in (100, 101)
    empty = (2.0 * step + 1.0, 3.0 * step - 1.0)
    with pytest.raises(ValueError, match=r"'f_range' must select at least one bin"):
        dataclasses.replace(res, f_range=empty)


def test_max_gain_is_not_determined_by_the_stored_design() -> None:
    """Two genuine designs agree in every stored field and differ in gain.

    ``max_gain_db`` is the achieved gain outside the band *padded by*
    ``transition_octaves``, and the padding is an argument the result does not
    keep. On a 128-bin grid (375 Hz apart) with the band reaching Nyquist, a
    one-octave padding puts the lower edge exactly on the first positive bin
    and a half-octave padding puts it just above: the profile is the same
    array either way, while that bin falls outside one maximum and inside the
    other. Nothing computable from the stored fields separates ``-inf`` from
    ``-6.13 dB`` here, which is why ``__post_init__`` claims nothing about it.
    """
    b, a = signal.butter(2, 300.0, btype="highpass", fs=FS)
    imp = np.zeros(128)
    imp[0] = 1.0
    h = signal.lfilter(b, a, imp)
    design = {"f_range": (750.0, FS / 2.0), "n_fft": 128}
    wide = signals.regularized_inverse_filter(h, FS, transition_octaves=1.0, **design)
    narrow = signals.regularized_inverse_filter(h, FS, transition_octaves=0.5, **design)
    for field in (
        "inverse",
        "frequencies",
        "spectrum",
        "response_spectrum",
        "regularization",
    ):
        assert np.array_equal(getattr(wide, field), getattr(narrow, field)), field
    assert (wide.f_range, wide.delay, wide.fs) == (
        narrow.f_range,
        narrow.delay,
        narrow.fs,
    )
    assert wide.flatness_db == narrow.flatness_db
    assert wide.max_gain_db == -np.inf
    assert narrow.max_gain_db == pytest.approx(-6.133314, abs=1e-6)


def test_apply_rejects_non_1d() -> None:
    h = _biquad_ir()
    res = signals.regularized_inverse_filter(h, FS, f_range=(500.0, 8000.0))
    two_dim = np.zeros((2, 4))
    with pytest.raises(ValueError, match="one-dimensional"):
        res.apply(two_dim)

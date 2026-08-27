#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ECMA-418-2:2025 (Sottek Hearing Model) fluctuation-strength conformance tests.

The primary oracle is the standard's own calibration (Clause 9, intro and
9.1.11): a 1 kHz carrier, 100 % amplitude-modulated (m = 1) at 4 Hz and a
sound pressure level of 60 dB SPL yields 1 vacil_HMS. That is the only
reference value Clause 9 tabulates (it has no verification annex). The HSA
core is additionally anchored with closed-form synthetic envelopes whose
line-pair amplitudes it must recover to machine precision, and the remaining
checks are the behaviours the standard states qualitatively (fluctuation
strength peaks near a 4 Hz modulation rate and falls off toward slower and
faster rates, vanishes in the roughness domain near 70 Hz, grows with
modulation depth, an unmodulated tone and silence give ~0) plus
result-structure guards.
"""

import dataclasses
import importlib

import numpy as np
import pytest

from phonometry import psychoacoustics

_fse = importlib.import_module(
    "phonometry.psychoacoustics.quality.fluctuation_strength_ecma"
)

FS = 48000
P0 = 2e-5


def _am_tone(
    fc: float,
    fmod: float,
    depth: float,
    level_db: float,
    seconds: float = 2.5,
) -> np.ndarray:
    """Amplitude-modulated tone at an OVERALL RMS level (pressure in Pa).

    Clause 9 states the calibration level as the sound pressure level of the
    signal, i.e. the overall RMS level of the modulated waveform (same
    convention as the Clause 7 roughness calibration).
    """
    t = np.arange(int(FS * seconds)) / FS
    x = (1.0 + depth * np.cos(2.0 * np.pi * fmod * t)) * np.sin(2.0 * np.pi * fc * t)
    return np.asarray(x * (P0 * 10.0 ** (level_db / 20.0)) / np.sqrt(np.mean(x**2)))


def _tone(fc: float, level_db: float, seconds: float = 2.5) -> np.ndarray:
    """Unmodulated tone at ``level_db`` RMS."""
    return _am_tone(fc, 0.0, 0.0, level_db, seconds)


@pytest.fixture(scope="module")
def ref_calibration() -> psychoacoustics.EcmaFluctuationStrength:
    """The 1 kHz / 4 Hz / m=1 / overall 60 dB calibration signal (1 vacil_HMS)."""
    return psychoacoustics.fluctuation_strength_ecma(
        _am_tone(1000.0, 4.0, 1.0, 60.0, 5.0), FS
    )


# --------------------------------------------------------------------------
# Primary reference value (Clause 9 calibration)
# --------------------------------------------------------------------------


@pytest.mark.xdist_group("ecma-fluctuation-ref")
def test_calibration_1khz_4hz_is_one_vacil(
    ref_calibration: psychoacoustics.EcmaFluctuationStrength,
) -> None:
    """The Clause 9 reference signal computes 1 vacil_HMS.

    A 1 kHz carrier, 100 % amplitude-modulated at 4 Hz, with an overall
    sound pressure level of 60 dB SPL is defined as 1 vacil_HMS; footnote 47
    allows c_F to be adjusted by at most +/-0.25 %. With the tabulated
    c_F = 0.003840572 (not reverse-fit) this chain computes 0.9931 vacil_HMS
    for the 5 s fixture signal (0.9957 at 8 s, converged 0.9958 by 12 s).
    """
    assert ref_calibration.fluctuation_strength == pytest.approx(1.0, abs=0.01)


def test_calibration_constant_is_the_tabulated_c_f() -> None:
    # Formula 163 tabulates c_F = 0.003840572; the implementation must use
    # the standard's value verbatim (shared oracle in tests/reference_data/).
    from reference_data import ECMA418_2_FLUCTUATION_C_F

    assert _fse._C_F == ECMA418_2_FLUCTUATION_C_F


def test_prominence_threshold_is_the_clause_9_2_value() -> None:
    from reference_data import ECMA418_2_PROMINENT_FLUCTUATION_VACIL

    from phonometry.psychoacoustics import (
        fluctuation_strength_ecma as _pkg_func,  # noqa: F401 - import guard
    )

    assert (
        _fse.PROMINENT_FLUCTUATION_STRENGTH_VACIL
        == ECMA418_2_PROMINENT_FLUCTUATION_VACIL
    )


# --------------------------------------------------------------------------
# HSA core: closed-form synthetic anchors (Clause 9.1.4)
# --------------------------------------------------------------------------


def test_hsa_recovers_noiseless_line_pairs_exactly() -> None:
    """The HSA recovers constant part, amplitudes and phases exactly.

    Clause 9.1.4 claims "theoretically infinite resolution for signals
    without noise"; for a windowed envelope made of a constant plus two
    cosines at off-bin rates the least-squares fit must return the exact
    parameters. This also pins the Formula (127) kernel-phase reading
    recorded in docs/ERRATA.md (with the printed 2*pi phase the fit cannot
    reproduce the spectrum at all).
    """
    sb = _fse._SB_TILDE
    n = np.arange(sb)
    n_zb, n_ze = 64, 100
    p0, a1, ph1, f1 = 0.02, 0.013, 0.7, 4.37
    a2, ph2, f2 = 0.004, -1.2, 9.1
    env = (
        p0
        + a1 * np.cos(2.0 * np.pi * f1 * n / _fse._R_TILDE + ph1)
        + a2 * np.cos(2.0 * np.pi * f2 * n / _fse._R_TILDE + ph2)
    )
    window = np.zeros(sb)
    window[n_zb : sb - n_ze] = 1.0
    p_spec = np.fft.rfft(env * window)[: _fse._N_K]
    phi = np.abs(p_spec) ** 2

    x, err = _fse._hsa_solve(p_spec, phi, np.array([f1, f2]), n_zb, n_ze)
    assert x[0] == pytest.approx(p0, rel=1e-9)
    assert 2.0 * np.hypot(x[1], x[2]) == pytest.approx(a1, rel=1e-9)
    assert np.arctan2(x[2], x[1]) == pytest.approx(ph1, rel=1e-6)
    assert 2.0 * np.hypot(x[3], x[4]) == pytest.approx(a2, rel=1e-9)
    assert np.arctan2(x[4], x[3]) == pytest.approx(ph2, rel=1e-6)
    assert abs(err) < 1e-9  # exact fit: the residual error vanishes


def test_hsa_window_kernel_matches_the_dft_of_the_window() -> None:
    # Formula (127) with the pi phase reading must equal the DFT of the
    # rectangular analysis window modulated to +f_c (the model the HSA fits).
    n_zb, n_ze, f_c = 80, 200, 3.3
    n = np.arange(_fse._SB_TILDE)
    window = np.zeros(_fse._SB_TILDE)
    window[n_zb : _fse._SB_TILDE - n_ze] = 1.0
    direct = np.fft.fft(np.exp(1j * 2.0 * np.pi * f_c * n / _fse._R_TILDE) * window)[
        : _fse._N_K
    ]
    kernel = _fse._window_kernel(f_c, n_zb, n_ze, _fse._K_ARR)
    assert np.allclose(kernel, direct, atol=1e-6)


def test_hsa_single_pair_matches_general_solver() -> None:
    # The closed-form Cramer path (Formulae 136-142) must agree with the
    # general normal-equations solver (Formulae 130-135) for one pair.
    sb = _fse._SB_TILDE
    n = np.arange(sb)
    n_zb, n_ze = 64, 64
    env = 0.01 + 0.006 * np.cos(2.0 * np.pi * 5.1 * n / _fse._R_TILDE)
    window = np.zeros(sb)
    window[n_zb : sb - n_ze] = 1.0
    p_spec = np.fft.rfft(env * window)[: _fse._N_K]
    phi = np.abs(p_spec) ** 2
    w0 = _fse._window_kernel(0.0, n_zb, n_ze, _fse._K_ARR)
    for f in (0.3, 1.7, 5.1, 11.0, 30.0):
        x_gen, e_gen = _fse._hsa_solve(p_spec, phi, np.array([f]), n_zb, n_ze)
        x_single, e_single = _fse._hsa_single(p_spec, phi, f, n_zb, n_ze, w0)
        assert np.allclose(x_gen, x_single, rtol=1e-9, atol=1e-14)
        assert e_gen == pytest.approx(e_single, rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------
# Envelope-dependent analysis window (Clause 9.1.3)
# --------------------------------------------------------------------------


def _synthetic_envelope() -> np.ndarray:
    """A clearly modulated 2048-sample envelope (passes every 9.1.3 check)."""
    n = np.arange(_fse._SB_TILDE)
    return np.asarray(
        0.01 * (1.0 + 0.5 * np.sin(2.0 * np.pi * 4.0 * n / _fse._R_TILDE))
    )


def test_window_default_guards_for_a_full_envelope() -> None:
    # A loud, fully modulated envelope keeps the initial 64-sample Hilbert
    # guards on both sides (Clause 9.1.3.1).
    assert _fse._analysis_window(_synthetic_envelope()) == (64, 64)


def test_window_trims_to_the_longer_side_of_a_quieter_period() -> None:
    # A > 320-sample quieter period (Clause 9.1.3.4) splits the block; the
    # longer part (here: the left) is kept and the window ends 64 guard
    # samples before the quieter period begins (Clause 9.1.3.5).
    env = _synthetic_envelope()
    env[1050:1550] = 0.0
    result = _fse._analysis_window(env)
    assert result is not None
    n_zb, n_ze = result
    assert n_zb == 64
    n2 = _fse._SB_TILDE - 1 - n_ze
    # The smoothed quieter period begins near sample 1066 (the length-33
    # median blurs the edge by ~16 samples); the window must stop one guard
    # length before it.
    assert 950 <= n2 <= 1010


def test_window_rejects_an_unmodulated_envelope() -> None:
    # Clause 9.1.3.6: the relative standard deviation of the linear
    # regression must be at least 0.1 %; a constant envelope fails and the
    # whole block is a quieter period.
    assert _fse._analysis_window(np.full(_fse._SB_TILDE, 0.01)) is None


def test_window_rejects_a_quiet_block() -> None:
    # Clause 9.1.3.2: peak smoothed envelope at or below 5e-6 Pa.
    assert _fse._analysis_window(np.full(_fse._SB_TILDE, 1e-6)) is None


# --------------------------------------------------------------------------
# Standard's qualitative behaviours
# --------------------------------------------------------------------------


@pytest.mark.xdist_group("ecma-fluctuation-ref")
def test_peaks_near_4hz_modulation(
    ref_calibration: psychoacoustics.EcmaFluctuationStrength,
) -> None:
    # Fluctuation strength is maximal around 4 Hz modulation and falls off
    # toward slower and faster rates (Clause 9 intro). The 4 Hz reference is
    # the module calibration signal, so reuse its result.
    f4 = ref_calibration.fluctuation_strength
    f1 = psychoacoustics.fluctuation_strength_ecma(
        _am_tone(1000.0, 1.0, 1.0, 60.0), FS
    ).fluctuation_strength
    f16 = psychoacoustics.fluctuation_strength_ecma(
        _am_tone(1000.0, 16.0, 1.0, 60.0), FS
    ).fluctuation_strength
    assert f4 > f1
    assert f4 > f16


def test_roughness_domain_modulation_scores_near_zero() -> None:
    # 70 Hz modulation is the roughness reference; fluctuation strength
    # covers slow variations (typically below 20 Hz, Clause 9 intro) and
    # must be negligible there.
    f70 = psychoacoustics.fluctuation_strength_ecma(
        _am_tone(1000.0, 70.0, 1.0, 60.0), FS
    ).fluctuation_strength
    assert f70 < 0.05


@pytest.mark.xdist_group("ecma-fluctuation-ref")
def test_grows_with_modulation_depth(
    ref_calibration: psychoacoustics.EcmaFluctuationStrength,
) -> None:
    f_full = ref_calibration.fluctuation_strength
    f_half = psychoacoustics.fluctuation_strength_ecma(
        _am_tone(1000.0, 4.0, 0.5, 60.0), FS
    ).fluctuation_strength
    assert f_full > f_half > 0.05


def test_unmodulated_tone_is_near_zero() -> None:
    result = psychoacoustics.fluctuation_strength_ecma(_tone(1000.0, 60.0), FS)
    assert result.fluctuation_strength < 0.1


def test_silence_is_zero() -> None:
    result = psychoacoustics.fluctuation_strength_ecma(np.zeros(FS), FS)
    assert result.fluctuation_strength == pytest.approx(0.0, abs=1e-9)
    assert np.all(result.specific_fluctuation_strength == 0.0)


# --------------------------------------------------------------------------
# Result structure and API guards
# --------------------------------------------------------------------------


@pytest.mark.xdist_group("ecma-fluctuation-ref")
def test_result_structure(
    ref_calibration: psychoacoustics.EcmaFluctuationStrength,
) -> None:
    res = ref_calibration
    assert res.specific_fluctuation_strength.shape == res.bark.shape == (53,)
    assert res.fluctuation_strength_vs_time.shape == res.time.shape
    assert res.specific_fluctuation_strength_vs_time.shape == (
        res.time.size,
        res.bark.size,
    )
    assert np.all(res.specific_fluctuation_strength >= 0.0)
    assert np.all(res.fluctuation_strength_vs_time >= 0.0)
    assert res.field == "free"


@pytest.mark.parametrize("trim", [True, False])
@pytest.mark.xdist_group("ecma-fluctuation-ref")
def test_a_trace_off_its_time_axis_is_refused(
    ref_calibration: psychoacoustics.EcmaFluctuationStrength, trim: bool
) -> None:
    """The time panel is one ``plot`` call pairing ``time`` with F(l50).

    A trace of another length reaches matplotlib, which raises about x and y
    differing in their first dimension and names neither field, from a call
    several frames below the one the caller made. Short and long fail the
    same way, which is the only mismatch here that never passes silently.
    """
    trace = ref_calibration.fluctuation_strength_vs_time
    wrong = trace[:-1] if trim else np.append(trace, trace[-1])
    with pytest.raises(
        ValueError, match=rf"'fluctuation_strength_vs_time' \({wrong.size}\)"
    ):
        dataclasses.replace(ref_calibration, fluctuation_strength_vs_time=wrong)


@pytest.mark.xdist_group("ecma-fluctuation-ref")
def test_a_specific_table_short_on_both_axes_is_refused(
    ref_calibration: psychoacoustics.EcmaFluctuationStrength,
) -> None:
    """The one mismatch the heatmap draws instead of refusing.

    A table one entry short along time *and* along the bands is exactly the
    shape ``pcolormesh`` expects once ``shading="auto"`` reads the two axes as
    cell corners, so the full figure draws a complete heatmap spanning the
    whole time and Bark_HMS range, one cell coarser than the run it claims to
    show, without a word from anywhere. Short on one axis only is refused by
    ``pcolormesh``, but only when the caller asked for the whole figure.
    """
    table = ref_calibration.specific_fluctuation_strength_vs_time
    with pytest.raises(
        ValueError, match=r"'specific_fluctuation_strength_vs_time \(axis 1\)'"
    ):
        dataclasses.replace(
            ref_calibration, specific_fluctuation_strength_vs_time=table[:-1, :-1]
        )


@pytest.mark.xdist_group("ecma-fluctuation-ref")
def test_a_specific_table_that_lost_its_band_axis_is_refused(
    ref_calibration: psychoacoustics.EcmaFluctuationStrength,
) -> None:
    """Counting the frames says nothing about how many axes carry them.

    Averaged down to one axis the table still has one entry per 50 Hz frame,
    so every length agrees and the time panel is drawn unharmed; only
    ``pcolormesh`` notices, unpacking the shape into two names and reporting
    that it got one value -- and only for the two-panel figure, since a
    caller-supplied ``ax`` never reaches the heatmap at all.
    """
    flat = ref_calibration.specific_fluctuation_strength_vs_time.mean(axis=1)
    with pytest.raises(
        ValueError, match="'specific_fluctuation_strength_vs_time' must have 2 axes"
    ):
        dataclasses.replace(ref_calibration, specific_fluctuation_strength_vs_time=flat)


@pytest.mark.parametrize(
    "field", ["bark", "centre_frequencies", "specific_fluctuation_strength"]
)
@pytest.mark.xdist_group("ecma-fluctuation-ref")
def test_a_band_axis_off_the_filter_bank_is_refused(
    ref_calibration: psychoacoustics.EcmaFluctuationStrength, field: str
) -> None:
    """The 53 bands are the standard's filter bank, however long the signal.

    ``bark`` is the heatmap's y axis, and the other two name and summarise
    those same bands for the caller. A ``bark`` of another length makes
    ``pcolormesh`` refuse the whole table when the full figure is drawn, and
    nothing at all when the caller supplies an ``ax``: only the time panel is
    drawn then, so the disagreement goes nowhere and ``plot`` returns as if
    all were well.
    """
    column = np.asarray(getattr(ref_calibration, field))
    with pytest.raises(ValueError, match=rf"'{field}' \({column.size - 1}\)"):
        dataclasses.replace(ref_calibration, **{field: column[:-1]})


@pytest.mark.xdist_group("ecma-fluctuation-ref")
def test_a_time_axis_with_a_second_axis_is_refused(
    ref_calibration: psychoacoustics.EcmaFluctuationStrength,
) -> None:
    """A column of times counts the frames correctly and is still not an axis.

    ``plot`` broadcasts it against the trace without complaint; the shaded
    fill underneath is what stops, with ``'x' is not 1-dimensional`` -- naming
    matplotlib's own parameter, not the field that carried the extra axis.
    """
    with pytest.raises(ValueError, match="'time' must have one axis"):
        dataclasses.replace(ref_calibration, time=ref_calibration.time[:, None])


def test_invalid_field_raises() -> None:
    sig = _tone(1000.0, 60.0, 1.0)
    with pytest.raises(ValueError, match="field must be"):
        psychoacoustics.fluctuation_strength_ecma(sig, FS, field="bogus")


def test_empty_signal_raises() -> None:
    empty = np.array([])
    with pytest.raises(ValueError, match="signal must not be empty"):
        psychoacoustics.fluctuation_strength_ecma(empty, FS)


@pytest.mark.parametrize("bad_fs", [0.0, -48000.0, float("nan"), float("inf")])
def test_invalid_fs_raises(bad_fs: float) -> None:
    sig = _tone(1000.0, 60.0, 1.0)
    with pytest.raises(ValueError, match="'fs' must be positive"):
        psychoacoustics.fluctuation_strength_ecma(sig, bad_fs)


def test_non_finite_signal_raises() -> None:
    sig = _tone(1000.0, 60.0, 1.0)
    sig[100] = np.nan
    with pytest.raises(
        ValueError, match="'signal_in' must contain only finite samples"
    ):
        psychoacoustics.fluctuation_strength_ecma(sig, FS)


def test_resample_length_clamps_to_one_sample() -> None:
    # 4 samples at 10 MHz resample to round(4 * 48000 / 1e7) = 0 samples
    # without the clamp; the clamped 1-sample signal must process cleanly.
    res = psychoacoustics.fluctuation_strength_ecma(np.zeros(4), 1.0e7)
    assert res.fluctuation_strength == 0.0


@pytest.mark.xdist_group("ecma-fluctuation-ref")
def test_deterministic(
    ref_calibration: psychoacoustics.EcmaFluctuationStrength,
) -> None:
    again = psychoacoustics.fluctuation_strength_ecma(
        _am_tone(1000.0, 4.0, 1.0, 60.0, 5.0), FS
    )
    assert again.fluctuation_strength == pytest.approx(
        ref_calibration.fluctuation_strength, abs=1e-9
    )


def test_free_and_diffuse_differ() -> None:
    sig = _am_tone(1000.0, 4.0, 1.0, 60.0)
    free = psychoacoustics.fluctuation_strength_ecma(sig, FS, field="free")
    diffuse = psychoacoustics.fluctuation_strength_ecma(sig, FS, field="diffuse")
    assert free.fluctuation_strength != diffuse.fluctuation_strength


@pytest.mark.xdist_group("ecma-fluctuation-ref")
def test_plot_smoke(
    ref_calibration: psychoacoustics.EcmaFluctuationStrength,
) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    axes = ref_calibration.plot()
    assert axes.shape == (2,)
    single = ref_calibration.plot(ax=axes[0])
    assert single is axes[0]


@pytest.mark.xdist_group("ecma-fluctuation-ref")
def test_plot_accepts_matplotlib_color_alias(
    ref_calibration: psychoacoustics.EcmaFluctuationStrength,
) -> None:
    # ``c=`` is matplotlib's alias for ``color=``; the renderer must not
    # inject the canonical name alongside it (that raises a TypeError).
    import matplotlib as mpl

    mpl.use("Agg")
    axes = ref_calibration.plot(c="#123456")
    assert axes[0].lines[0].get_color() == "#123456"

#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for integrated and statistical sound levels (Leq, LAeq, LN)."""

import inspect
import pathlib
import re
import string

import numpy as np
import pytest

from phonometry import signals

FS = 48000


def _tone(f0: float, seconds: float = 1.0, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(FS * seconds)) / FS
    return amp * np.sin(2 * np.pi * f0 * t)


def test_leq_sine_matches_rms() -> None:
    """Leq of a 1 Pa amplitude sine = 20*log10((1/sqrt2)/20u) = 90.97 dB."""
    x = _tone(1000)
    assert signals.leq(x) == pytest.approx(90.97, abs=0.05)


def test_leq_dbfs() -> None:
    """RMS of a full-scale sine is -3.01 dBFS."""
    x = _tone(1000)
    assert signals.leq(x, dbfs=True) == pytest.approx(-3.01, abs=0.05)


def test_leq_dbfs_zero_reference_is_rms_one_not_a_full_scale_sine() -> None:
    """0 dBFS is RMS 1.0, not the AES17 full-scale sine.

    Pins the convention the guides quote. Under AES17 a full-scale sine reads
    0 dBFS; here it reads -3.01, and it takes a sine of amplitude sqrt(2)
    (RMS 1.0) to reach 0. Anything else is a 3.01 dB systematic offset.
    """
    assert signals.leq(_tone(1000, amp=np.sqrt(2.0)), dbfs=True) == pytest.approx(
        0.0, abs=1e-6
    )
    assert signals.leq(_tone(1000), dbfs=True) == pytest.approx(-3.0103, abs=1e-3)
    # Any RMS-1.0 waveform hits 0 dBFS: the reference is RMS, not a shape.
    assert signals.leq(np.ones(FS), dbfs=True) == pytest.approx(0.0, abs=1e-9)


def test_leq_multichannel_returns_per_channel() -> None:
    x = np.stack([_tone(1000), 0.5 * _tone(1000)])
    out = signals.leq(x)
    assert out.shape == (2,)
    assert out[0] - out[1] == pytest.approx(6.02, abs=0.05)


def test_leq_calibration_factor() -> None:
    x = _tone(1000)
    assert signals.leq(x, calibration_factor=10.0) == pytest.approx(
        90.97 + 20.0, abs=0.05
    )


def test_laeq_1khz_equals_leq() -> None:
    """A-weighting is 0 dB at 1 kHz, so LAeq == Leq there."""
    x = _tone(1000, seconds=2.0)
    assert signals.laeq(x, FS) == pytest.approx(signals.leq(x), abs=0.3)


def test_laeq_100hz_attenuated() -> None:
    """A-weighting at 100 Hz is about -19.1 dB."""
    x = _tone(100, seconds=2.0)
    assert signals.laeq(x, FS) - signals.leq(x) == pytest.approx(-19.1, abs=0.5)


def test_ln_levels_constant_signal_all_equal() -> None:
    """For a steady tone, L10 == L90 and L50 == Leq (within envelope ripple)."""
    x = _tone(1000, seconds=3.0)
    out = signals.ln_levels(x, FS, n=(10, 50, 90))
    assert set(out.keys()) == {10, 50, 90}
    # 5*tau attack skip settles the F integrator to 99.3%, so a steady tone's
    # L10-L90 spread collapses to ~0.01 dB (was ~0.15 dB at the old 2*tau).
    assert out[10] == pytest.approx(out[90], abs=0.05)
    assert out[50] == pytest.approx(90.97, abs=0.3)


def test_ln_levels_ordering() -> None:
    """L10 (exceeded 10% of time) >= L50 >= L90 for a fluctuating signal."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(FS * 3) * np.linspace(0.1, 1.0, FS * 3)
    out = signals.ln_levels(x, FS)
    assert out[10] >= out[50] >= out[90]


def test_ln_levels_weighting_a() -> None:

    x = _tone(100, seconds=3.0)
    unweighted = signals.ln_levels(x, FS, n=(50,))[50]
    weighted = signals.ln_levels(x, FS, n=(50,), weighting="A")[50]
    assert weighted - unweighted == pytest.approx(-19.1, abs=0.5)


def test_ln_levels_invalid_percentile_raises() -> None:

    tone = _tone(1000)
    with pytest.raises(ValueError, match="between 0 and 100"):
        signals.ln_levels(tone, FS, n=(0,))


def test_ln_levels_multichannel() -> None:

    x = np.stack([_tone(1000, 2.0), 0.5 * _tone(1000, 2.0)])
    out = signals.ln_levels(x, FS, n=(50,))
    assert out[50].shape == (2,)
    assert out[50][0] - out[50][1] == pytest.approx(6.02, abs=0.2)


def test_leq_dbfs_ignores_calibration_factor() -> None:
    """dBFS is relative to digital full scale (consistent with OctaveFilterBank)."""
    x = _tone(1000)
    assert signals.leq(x, calibration_factor=10.0, dbfs=True) == pytest.approx(
        signals.leq(x, dbfs=True), abs=1e-12
    )


def test_leq_empty_signal_raises() -> None:
    empty = np.array([])
    with pytest.raises(ValueError, match="empty"):
        signals.leq(empty)


def test_leq_nonpositive_calibration_raises() -> None:
    tone = _tone(1000)
    with pytest.raises(ValueError, match="calibration_factor"):
        signals.leq(tone, calibration_factor=-1.0)


def test_ln_levels_empty_signal_raises() -> None:

    empty = np.array([])
    with pytest.raises(ValueError, match="empty"):
        signals.ln_levels(empty, FS)


# ---------------------------------------------------------------------------
# LCpeak — IEC 61672-1:2013 §5.13
# ---------------------------------------------------------------------------


def _faded(x: np.ndarray, ramp: float = 0.05) -> np.ndarray:
    """Fade a tone in/out so the filter onset transient does not add overshoot."""
    n = int(FS * ramp)
    window = np.ones_like(x)
    window[:n] = np.hanning(2 * n)[:n]
    window[-n:] = np.hanning(2 * n)[n:]
    return x * window


def test_lc_peak_steady_1khz() -> None:
    """C weighting is ~0 dB at 1 kHz: LCpeak of a steady sine = 20*log10(A/p0)."""
    x = _faded(_tone(1000, seconds=1.0, amp=1.0))
    assert signals.lc_peak(x, FS) == pytest.approx(20 * np.log10(1.0 / 2e-5), abs=0.15)


def test_lc_peak_exceeds_lc_by_crest_factor() -> None:
    """For a steady sine, LCpeak - LC = 20*log10(sqrt(2)) = 3.01 dB."""
    from phonometry.filters.weighting import weighting_filter

    # 10 ms ramps: enough to avoid the onset click without biasing the RMS
    x = _faded(_tone(1000, seconds=1.0), ramp=0.01)
    lc = signals.leq(weighting_filter(x, FS, "C"))
    assert signals.lc_peak(x, FS) - lc == pytest.approx(3.01, abs=0.2)


def test_lc_peak_multichannel_and_dbfs() -> None:

    x = np.stack([_faded(_tone(1000)), 0.5 * _faded(_tone(1000))])
    out = signals.lc_peak(x, FS)
    assert out.shape == (2,)
    assert out[0] - out[1] == pytest.approx(6.02, abs=0.1)
    # dBFS: peak 1.0 -> 0 dBFS, calibration must not apply
    assert signals.lc_peak(
        _faded(_tone(1000)), FS, calibration_factor=10.0, dbfs=True
    ) == pytest.approx(0.0, abs=0.15)


@pytest.mark.parametrize(
    ("cycles", "freq", "ref", "tol"),
    [
        # BS EN 61672-1:2013 Table 5 (standard page 27): reference differences
        # LCpeak - LC and class 1 acceptance limits. Test frequencies are the
        # EXACT one-third-octave frequencies (Annex D), not nominal.
        (1.0, 10**1.5, 2.5, 2.0),  # one cycle, 31.5 Hz nominal
        (1.0, 10**2.7, 3.5, 1.0),  # one cycle, 500 Hz nominal
        (1.0, 10**3.9, 3.4, 2.0),  # one cycle, 8 kHz nominal
        (0.5, 10**2.7, 2.4, 1.0),  # positive half cycle, 500 Hz
        (-0.5, 10**2.7, 2.4, 1.0),  # negative half cycle, 500 Hz
    ],
)
def test_lc_peak_iec_table5(cycles: float, freq: float, ref: float, tol: float) -> None:
    """One-cycle / half-cycle bursts must reproduce Table 5 within class 1."""
    from phonometry.filters.weighting import weighting_filter

    fs = 96000
    t = np.arange(int(fs * 1.0)) / fs
    steady = np.sin(2 * np.pi * freq * t)
    lc_steady = signals.leq(weighting_filter(steady, fs, "C"))

    n = round(abs(cycles) * fs / freq)  # starts and stops on zero crossings
    sign = 1.0 if cycles > 0 else -1.0
    burst = np.zeros(int(fs * 1.0))
    start = len(burst) // 2
    tt = np.arange(n) / fs
    burst[start : start + n] = sign * np.sin(2 * np.pi * freq * tt)

    diff = signals.lc_peak(burst, fs) - lc_steady
    assert diff == pytest.approx(ref, abs=tol), (
        f"{cycles} cycles @ {freq:.0f} Hz: {diff:.2f} dB"
    )


@pytest.mark.parametrize(
    ("cycles", "freq", "ref", "tol"),
    [
        # Same BS EN 61672-1:2013 Table 5 cases as above, at a field-typical
        # 48 kHz (audit N2 I-1: the standard test ran only at 96 kHz, masking
        # the inter-sample peak loss). Must still reproduce Table 5 within
        # class 1 with the oversampled peak detection.
        (1.0, 10**1.5, 2.5, 2.0),  # one cycle, 31.5 Hz nominal
        (1.0, 10**2.7, 3.5, 1.0),  # one cycle, 500 Hz nominal
        (1.0, 10**3.9, 3.4, 2.0),  # one cycle, 8 kHz nominal
        (0.5, 10**2.7, 2.4, 1.0),  # positive half cycle, 500 Hz
        (-0.5, 10**2.7, 2.4, 1.0),  # negative half cycle, 500 Hz
    ],
)
def test_lc_peak_iec_table5_48k(
    cycles: float, freq: float, ref: float, tol: float
) -> None:
    """Table 5 reference differences must also hold at fs = 48 kHz."""
    from phonometry.filters.weighting import weighting_filter

    fs = 48000
    t = np.arange(int(fs * 1.0)) / fs
    steady = np.sin(2 * np.pi * freq * t)
    lc_steady = signals.leq(weighting_filter(steady, fs, "C"))

    n = round(abs(cycles) * fs / freq)  # starts and stops on zero crossings
    sign = 1.0 if cycles > 0 else -1.0
    burst = np.zeros(int(fs * 1.0))
    start = len(burst) // 2
    tt = np.arange(n) / fs
    burst[start : start + n] = sign * np.sin(2 * np.pi * freq * tt)

    diff = signals.lc_peak(burst, fs) - lc_steady
    assert diff == pytest.approx(ref, abs=tol), (
        f"{cycles} cycles @ {freq:.0f} Hz: {diff:.2f} dB"
    )


def _lcpeak_analytic_steady(x: np.ndarray, fs: int) -> float:
    """Analytic LCpeak of a steady tone: C-weighted steady RMS + 3.01 dB crest.

    The crest factor of a pure sinusoid is exactly 20*log10(sqrt(2)) = 3.01 dB,
    verified independent of the C-weighting gain. Measured from a transient-free
    middle window so it isolates the peak-detection accuracy.
    """
    from phonometry.filters.weighting import weighting_filter

    w = weighting_filter(x, fs, "C")
    mid = w[int(0.4 * w.shape[-1]) : int(0.6 * w.shape[-1])]
    rms = np.sqrt(np.mean(mid**2))
    return float(20 * np.log10(rms / 2e-5) + 3.01)


def test_lc_peak_recovers_inter_sample_peak_8k_48k() -> None:
    """LCpeak must catch the inter-sample peak of a sustained 8 kHz tone at 48 kHz.

    8 kHz at 48 kHz is exactly 6.0 samples/cycle, so the sampling phase is
    locked and the on-grid maximum consistently under-reads the true peak
    (audit N2 I-1: up to -1.15 dB worst-case over phase, -0.69 dB at phase 0).
    Oversampled peak detection recovers the analytic sinusoid crest.
    """
    x = _faded(_tone(8000, seconds=1.0))
    err = signals.lc_peak(x, FS) - _lcpeak_analytic_steady(x, FS)
    assert abs(err) < 0.5, f"LCpeak under-reads by {err:+.3f} dB (inter-sample loss)"


def test_lc_peak_oversample_keyword_controls_recovery() -> None:
    """oversample=1 reproduces the legacy on-grid under-read; the default fixes it."""
    x = _faded(_tone(8000, seconds=1.0))
    ref = _lcpeak_analytic_steady(x, FS)
    legacy_err = signals.lc_peak(x, FS, oversample=1) - ref
    fixed_err = signals.lc_peak(x, FS) - ref
    assert legacy_err < -0.5  # legacy on-grid detection under-reads
    assert abs(fixed_err) < abs(legacy_err)  # default oversampling recovers it


# ---------------------------------------------------------------------------
# SEL / LAE — sound exposure level
# ---------------------------------------------------------------------------


def test_sel_steady_signal_normalizes_to_one_second() -> None:
    """SEL = Leq + 10*log10(T / 1 s) for a steady signal of duration T."""
    x = _tone(1000, seconds=4.0)
    assert signals.sel(x, FS) == pytest.approx(
        signals.leq(x) + 10 * np.log10(4.0), abs=1e-6
    )


def test_sel_one_second_equals_leq() -> None:

    x = _tone(1000, seconds=1.0)
    assert signals.sel(x, FS) == pytest.approx(signals.leq(x), abs=1e-9)


def test_sel_a_weighted() -> None:

    x = _tone(1000, seconds=2.0)
    assert signals.sel(x, FS, weighting="A") == pytest.approx(
        signals.laeq(x, FS) + 10 * np.log10(2.0), abs=0.05
    )


# ---------------------------------------------------------------------------
# Sound exposure / dose — IEC 61252 (BS EN 61252:1995 §3.1-3.3, Annex A)
# ---------------------------------------------------------------------------


def _tone_at_level(
    level_db: float, seconds: float = 2.0, f0: float = 1000.0
) -> np.ndarray:
    """1 kHz tone whose A-weighted level equals level_db (A(1 kHz) = 0 dB)."""
    rms = 2e-5 * 10 ** (level_db / 20)
    t = np.arange(int(FS * seconds)) / FS
    return np.sqrt(2) * rms * np.sin(2 * np.pi * f0 * t)


def test_sound_exposure_anchor_90db_8h_is_3p2_pa2h() -> None:
    """BS EN 61252:1995 Annex A / §3.3 NOTE 4: 3.2 Pa²h <-> exactly 90 dB."""
    x = _tone_at_level(90.0)
    assert signals.sound_exposure(x, FS, duration_hours=8.0) == pytest.approx(
        3.2, rel=0.01
    )


def test_lex_8h_anchor_90db() -> None:

    x = _tone_at_level(90.0)
    assert signals.lex_8h(x, FS, duration_hours=8.0) == pytest.approx(90.0, abs=0.05)


def test_lex_8h_half_workday_subtracts_3db() -> None:
    """LEX,8h = LAeq,T + 10*log10(T/8h): a 4 h exposure at 90 dB -> 86.99 dB."""
    x = _tone_at_level(90.0)
    assert signals.lex_8h(x, FS, duration_hours=4.0) == pytest.approx(
        90.0 + 10 * np.log10(4 / 8), abs=0.05
    )


def test_sound_exposure_1_pa2h_is_nearly_85db() -> None:
    """§3.3 NOTE 4: 1 Pa²h corresponds to a LEX,8h of nearly 85 dB (84.95)."""
    x = _tone_at_level(84.9485)
    assert signals.sound_exposure(x, FS, duration_hours=8.0) == pytest.approx(
        1.0, rel=0.01
    )
    assert signals.lex_8h(x, FS, duration_hours=8.0) == pytest.approx(84.95, abs=0.05)


def test_sound_exposure_defaults_to_recording_duration() -> None:
    """Without duration_hours, x IS the whole event: E = integral over len(x)."""
    x = _tone_at_level(90.0, seconds=2.0)
    expected = (2e-5 * 10 ** (90 / 20)) ** 2 * (2.0 / 3600.0)  # Pa² * hours
    assert signals.sound_exposure(x, FS) == pytest.approx(expected, rel=0.01)


def test_sel_invalid_fs_raises() -> None:

    tone = _tone(1000)
    with pytest.raises(ValueError, match="fs"):
        signals.sel(tone, 0)


def test_sound_exposure_rejects_nonpositive_duration() -> None:

    tone = _tone_at_level(90.0)
    with pytest.raises(ValueError, match="duration_hours"):
        signals.sound_exposure(tone, FS, duration_hours=0)
    with pytest.raises(ValueError, match="duration_hours"):
        signals.lex_8h(tone, FS, duration_hours=-1.0)


# ---------------------------------------------------------------------------
# The guide's parameter tables must not overstate what they list
# ---------------------------------------------------------------------------

#: The levels guide exists three times over: the plain-markdown edition under
#: ``docs/`` that GitHub and the llms artifacts read, and the two published
#: site editions. They are written by hand, one per language, so a table
#: drifts in one copy at a time; every check below runs on all three.
_LEVELS_GUIDE_COPIES = (
    "docs/signals/levels/levels.md",
    "site/src/content/docs/signals/levels/levels.mdx",
    "site/src/content/docs/es/signals/levels/levels.mdx",
)


def _levels_guide(relative_path: str) -> str:
    """The text of one copy of the levels guide."""
    path = pathlib.Path(__file__).resolve().parents[2] / relative_path
    assert path.is_file(), f"missing guide copy: {relative_path}"
    return path.read_text(encoding="utf-8")


#: A first-column cell of the "Peak / event / dose parameters" table, e.g.
#: ``| `sel(x, fs, weighting=None, ...)` | ...``.
_SIGNATURE_CELL = re.compile(r"^\|\s*`([a-z_0-9]+)\((.*?)\)`\s*\|")


def _documented_signatures(markdown: str) -> dict[str, list[str]]:
    """Function name -> the parameter tokens its guide row spells out."""
    found: dict[str, list[str]] = {}
    for line in markdown.splitlines():
        match = _SIGNATURE_CELL.match(line.strip())
        if match is None:
            continue
        params = [tok.strip() for tok in match.group(2).split(",") if tok.strip()]
        found[match.group(1)] = params
    return found


def _real_signature_tokens(func: object) -> list[str]:
    """The full parameter list of ``func`` rendered the way the guide writes it."""
    tokens = []
    for name, param in inspect.signature(func).parameters.items():  # type: ignore[arg-type]
        if param.default is inspect.Parameter.empty:
            tokens.append(name)
        else:
            tokens.append(f"{name}={param.default!r}")
    return tokens


@pytest.mark.parametrize("relative_path", _LEVELS_GUIDE_COPIES)
def test_guide_signature_cells_match_the_code(relative_path: str) -> None:
    """Rows that omit parameters must say so with an explicit ellipsis.

    Three rows of the peak/event/dose table elide their tail with ``...``,
    which makes a row *without* one read as the complete signature. This
    checks both readings: an elided row must be a true prefix of the real
    signature, and a plain row must list it in full.

    Signatures are code, not prose, so they read the same in every copy and
    the same check applies to each: the published site editions are the ones
    a reader actually meets.
    """
    import phonometry
    from phonometry.signals import levels as levels_module

    documented = _documented_signatures(_levels_guide(relative_path))
    checked = 0
    for name, params in documented.items():
        func = getattr(levels_module, name, None) or getattr(phonometry, name, None)
        if func is None or not callable(func):
            continue
        real = _real_signature_tokens(func)
        checked += 1
        if params and params[-1] == "...":
            listed = params[:-1]
            assert listed == real[: len(listed)], (name, listed, real)
            assert len(listed) < len(real), f"{name}: '...' elides nothing"
        else:
            assert params == real, (name, params, real)
    assert checked >= 4, f"expected the peak/event/dose rows, checked {checked}"


#: The ``weighting`` row of the LN parameter table, whose values column names
#: the curves. Curve letters are identifiers, so they are the same in both
#: languages even though the prose around them is not.
_WEIGHTING_ROW = re.compile(r"^\|\s*`weighting`\s*\|")
_QUOTED_CURVE = re.compile(r"`'([A-Za-z]+)'`")


def _accepted_weighting_curves() -> set[str]:
    """Every curve `weighting_filter` really takes, found by asking it."""
    from phonometry import filters

    silence = np.zeros(256)
    accepted = set()
    for candidate in [*string.ascii_uppercase, "AU"]:
        try:
            filters.weighting_filter(silence, FS, curve=candidate)
        except ValueError:
            continue
        accepted.add(candidate)
    return accepted


@pytest.mark.parametrize("relative_path", _LEVELS_GUIDE_COPIES)
def test_guide_weighting_row_lists_every_curve_the_code_takes(
    relative_path: str,
) -> None:
    """The LN table's `weighting` row must name the real accepted set.

    It used to name four of the seven curves, which reads as a closed list
    and hides 'B', 'D' and 'AU'. Listing one that does not exist would be
    the same defect the other way round, so this compares the two sets.
    """
    rows = [
        line
        for line in _levels_guide(relative_path).splitlines()
        if _WEIGHTING_ROW.match(line.strip())
    ]
    assert len(rows) == 1, f"expected one `weighting` row, found {len(rows)}"
    columns = rows[0].strip().strip("|").split("|")
    assert len(columns) == 5, columns
    documented = set(_QUOTED_CURVE.findall(columns[3]))
    assert documented == _accepted_weighting_curves(), sorted(documented)


@pytest.mark.parametrize("curve", ["A", "B", "C", "D", "G", "AU", "Z"])
def test_ln_levels_and_sel_accept_every_weighting_filter_curve(curve: str) -> None:
    """Both functions forward `weighting` straight to `weighting_filter`.

    The docstrings used to advertise only 'A', 'C', 'Z' and the guides only
    'A', 'C', 'G', 'Z'; the real accepted set is whatever `weighting_filter`
    takes. Only an unknown letter raises.
    """
    x = _tone(1000, seconds=2.0)
    assert np.isfinite(signals.ln_levels(x, FS, n=(50,), weighting=curve)[50])
    assert np.isfinite(signals.sel(x, FS, weighting=curve))


def test_ln_levels_and_sel_reject_an_unknown_weighting() -> None:

    x = _tone(1000)
    with pytest.raises(ValueError, match="Weighting curve"):
        signals.ln_levels(x, FS, weighting="Q")
    with pytest.raises(ValueError, match="Weighting curve"):
        signals.sel(x, FS, weighting="Q")


# ---------------------------------------------------------------------------
# Signal overloads: every public function here takes a phonometry.io Signal
# in place of the bare (x, fs) pair. Every equality below is exact (==, not
# approx): the overload must resolve to the identical bare-array call, never
# to a nearby number.
# ---------------------------------------------------------------------------


def test_leq_takes_the_signals_own_calibration() -> None:
    from phonometry.io import Signal

    x = _tone(1000)
    sig = Signal(x, FS, calibration_factor=2.0)
    assert signals.leq(sig) == signals.leq(x, calibration_factor=2.0)


def test_explicit_calibration_beats_the_signals() -> None:
    """The documented precedence: the caller knows more than the object."""
    from phonometry.io import Signal

    x = _tone(1000)
    sig = Signal(x, FS, calibration_factor=2.0)
    assert signals.leq(sig, calibration_factor=10.0) == signals.leq(
        x, calibration_factor=10.0
    )


def test_uncalibrated_signal_levels_are_digital_unit_levels() -> None:
    from phonometry.io import Signal

    x = _tone(1000)
    assert signals.leq(Signal(x, FS)) == signals.leq(x)


def test_dbfs_ignores_the_signals_calibration() -> None:
    """dBFS is relative to digital full scale whatever the object carries."""
    from phonometry.io import Signal

    x = _tone(1000)
    assert signals.leq(
        Signal(x, FS, calibration_factor=123.0), dbfs=True
    ) == signals.leq(x, dbfs=True)


def test_fs_functions_take_the_signals_rate_and_calibration() -> None:
    from phonometry.io import Signal

    x = _tone(1000, seconds=2.0)
    sig = Signal(x, FS, calibration_factor=0.5)
    assert signals.lc_peak(sig) == signals.lc_peak(x, FS, calibration_factor=0.5)
    assert signals.sel(sig, weighting="A") == signals.sel(
        x, FS, weighting="A", calibration_factor=0.5
    )
    assert signals.ln_levels(sig, weighting="A") == signals.ln_levels(
        x, FS, weighting="A", calibration_factor=0.5
    )


def test_a_conflicting_fs_is_refused_a_matching_one_is_not() -> None:
    from phonometry.io import Signal

    sig = Signal(_tone(1000), FS)
    with pytest.raises(ValueError, match="conflicts with the Signal's own fs"):
        signals.lc_peak(sig, FS + 1)
    # The same number twice is agreement, not a conflict.
    assert signals.lc_peak(sig, FS) == signals.lc_peak(sig)


def test_a_bare_array_still_requires_fs() -> None:

    x = _tone(1000)
    for func in (
        signals.ln_levels,
        signals.sel,
        signals.lc_peak,
        signals.laeq,
        signals.sound_exposure,
        signals.lex_8h,
    ):
        with pytest.raises(ValueError, match="fs is required"):
            func(x)


def test_exposure_functions_take_the_signals_rate_and_calibration() -> None:
    """laeq / sound_exposure / lex_8h honour the object like their siblings.

    These three sit in the same module and the same docs table as the
    functions above; a Signal accepted here but with its calibration
    silently dropped (np.asarray sees only the samples) computed a level
    ~10 dB off that looked perfectly plausible. All of them or none.
    """
    from phonometry.io import Signal

    x = _tone(1000, seconds=2.0)
    sig = Signal(x, FS, calibration_factor=3.5)
    assert signals.laeq(sig) == signals.laeq(x, FS, calibration_factor=3.5)
    assert signals.sound_exposure(sig) == signals.sound_exposure(
        x, FS, calibration_factor=3.5
    )
    assert signals.sound_exposure(sig, duration_hours=8.0) == signals.sound_exposure(
        x, FS, duration_hours=8.0, calibration_factor=3.5
    )
    assert signals.lex_8h(sig) == signals.lex_8h(x, FS, calibration_factor=3.5)


def test_exposure_functions_refuse_a_conflicting_fs() -> None:
    from phonometry.io import Signal

    sig = Signal(_tone(1000), FS)
    for func in (signals.laeq, signals.sound_exposure, signals.lex_8h):
        with pytest.raises(ValueError, match="conflicts with the Signal's own fs"):
            func(sig, FS + 1)


def test_multichannel_signal_returns_per_channel_levels() -> None:
    from phonometry.io import Signal

    x = np.stack([_tone(1000), 0.5 * _tone(1000)])
    out = signals.leq(Signal(x, FS, calibration_factor=2.0))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)
    assert out.tolist() == np.asarray(signals.leq(x, calibration_factor=2.0)).tolist()

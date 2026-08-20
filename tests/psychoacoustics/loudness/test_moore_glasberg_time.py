#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
ISO 532-3:2023 (Moore-Glasberg-Schlittenlacher) time-varying loudness tests.

The algorithmic reference for steady tones is Annex C.1 of the standard, which
tabulates the maximum long-term loudness (sone) and loudness level (phon) of
5-second tones - the same numbers the model must reach for a long steady tone
(clause 7.9: the loudness of sounds up to ~5 s is predicted by the maximum of
the long-term loudness).  A 1 kHz tone at 40 dB SPL, binaural, free field, is
1.000 sone / 40 phon by definition of the sone (clause 7.3, the spectral
calibration is fixed to this anchor).

The remaining checks are the temporal properties the two-stage attack/release
integration (clauses 7.6/7.9) implies: the short-term loudness reacts faster
than the long-term loudness, release is slower than attack, a steady tone gives
a flat trace at the ISO 532-2 stationary value, silence gives zero, and the
percentile / plotting / validation plumbing behaves.

The synthetic-tone values of Annex B (WAV-file signals 2/6-8) are not
reproduced here: those waveform files are not distributed with this
implementation, and the tabulated 250 Hz / 80 dB value (69.3 phon) sits about
5 phon below the ISO 226 equal-loudness contour, whereas this model reproduces
that contour (and agrees with the validated ISO 532-2 stationary module) to
within ~1.5 phon.  Annex C.1 is therefore the tone oracle used.
"""

import numpy as np
import pytest

from phonometry import psychoacoustics
from phonometry.psychoacoustics.loudness.moore_glasberg_time import _ALPHA_AL, _ALPHA_RL

FS = 32000.0  # the Annex B/C sampling rate; keeps reference tones on FFT bins


def _tone(frequency: float, level_db: float, duration: float = 1.3) -> np.ndarray:
    """Calibrated pure tone: sound pressure in pascals at ``level_db`` dB SPL."""
    t = np.arange(round(duration * FS)) / FS
    p_rms = 2e-5 * 10.0 ** (level_db / 20.0)
    return np.sqrt(2.0) * p_rms * np.sin(2.0 * np.pi * frequency * t)


@pytest.fixture(scope="module")
def mg_tone():
    """Memoised steady-tone loudness, shared across the Annex C.1 checks.

    Several parametrised suites analyse the *same* (frequency, level, field,
    presentation) 1.3 s tone but assert different columns of the one result
    object: ``test_annex_c1_other_tones`` pins the phon value while
    ``test_annex_c1_sone_values`` pins the sone value of the identical tone,
    and the 1 kHz / 40 dB anchor is shared by three checks. Each analysis is a
    full time-varying loudness chain (~2 s), so computing it once per distinct
    input and reusing the (read-only) result object removes ~8 redundant
    recomputations without touching any oracle or tolerance. The returned
    ``MooreGlasbergTimeVaryingLoudness`` is never mutated by callers.
    """
    cache: dict[tuple, psychoacoustics.MooreGlasbergTimeVaryingLoudness] = {}

    def get(
        frequency: float,
        level_db: float,
        *,
        field: str = "free",
        presentation: str = "binaural",
    ) -> psychoacoustics.MooreGlasbergTimeVaryingLoudness:
        key = (frequency, level_db, field, presentation)
        if key not in cache:
            cache[key] = psychoacoustics.loudness_moore_glasberg_time(
                _tone(frequency, level_db), FS, field=field, presentation=presentation
            )
        return cache[key]

    return get


# --------------------------------------------------------------------------
# Definitional anchor (clause 7.3 / Annex C.1)
# --------------------------------------------------------------------------


def test_anchor_1khz_40db_is_one_sone(mg_tone) -> None:
    """A 1 kHz tone at 40 dB SPL, binaural, free field -> 1.000 sone / 40 phon."""
    res = mg_tone(1000.0, 40.0)
    assert res.n_max == pytest.approx(1.0, abs=0.02)
    assert res.loudness_level_max == pytest.approx(40.0, abs=0.3)


@pytest.mark.parametrize(
    ("fs", "expected_n_max"),
    [(32000.0, 0.9918), (44100.0, 0.9926), (48000.0, 0.9900)],
)
def test_anchor_holds_at_native_sample_rates(fs: float, expected_n_max: float) -> None:
    """The Annex C.1 anchor at the documented native-fs conformance mode.

    The module processes at the native sampling rate instead of the
    clause 5 32 kHz conversion (see the module docstring); this pins the
    anchor n_max at the three common rates (0.3 % cross-rate spread, far
    inside the 2.8 phon expanded uncertainty of clause 8) so a change to
    the spectral plan or calibration shows up here. A 0.5 s tone is used:
    the long-term smoothing has not fully settled there, which makes the
    pin sensitive to the temporal chain as well; at 1.3 s the same anchor
    reads 1.0000/1.0000/0.9982.
    """
    t = np.arange(round(0.5 * fs)) / fs
    x = np.sqrt(2.0) * 2e-5 * 10.0 ** (40.0 / 20.0) * np.sin(2.0 * np.pi * 1000.0 * t)
    res = psychoacoustics.loudness_moore_glasberg_time(x, fs)
    assert res.n_max == pytest.approx(expected_n_max, abs=0.003)


# --------------------------------------------------------------------------
# Annex C.1 - 1 kHz tone loudness vs level (primary tone oracle)
# --------------------------------------------------------------------------

_C1_1KHZ = [
    (10.0, 0.025, 10.0),
    (20.0, 0.14, 20.0),
    (30.0, 0.42, 30.0),
    (40.0, 1.00, 40.0),
    (50.0, 2.1, 50.0),
    (60.0, 4.1, 60.0),
    (70.0, 7.9, 70.0),
    (80.0, 15.4, 80.0),
]


@pytest.mark.parametrize("level_db, sone, phon", _C1_1KHZ)
def test_annex_c1_1khz_tone(mg_tone, level_db: float, sone: float, phon: float) -> None:
    """1 kHz tone 10-80 dB reaches the Annex C.1 peak long-term loudness."""
    res = mg_tone(1000.0, level_db)
    # Loudness level (phon) is the linear-in-perception comparison; within the
    # standard's expanded uncertainty (2.8 phon, clause 8).
    assert res.loudness_level_max == pytest.approx(phon, abs=1.0)


# --------------------------------------------------------------------------
# Annex C.1 - other frequencies / presentations
# --------------------------------------------------------------------------

_C1_OTHER = [
    # frequency, level, field, presentation, expected phon (Annex C.1)
    (3000.0, 20.0, "free", "binaural", 28.8),
    (3000.0, 40.0, "free", "binaural", 49.0),
    (3000.0, 60.0, "free", "binaural", 68.6),
    (100.0, 50.0, "free", "binaural", 28.2),
    (4000.0, 40.0, "free", "binaural", 48.9),  # Annex B signal 4
    (1000.0, 20.0, "eardrum", "monaural", 14.8),
    (1000.0, 40.0, "eardrum", "monaural", 32.7),
    (1000.0, 60.0, "eardrum", "monaural", 51.4),
    (1000.0, 80.0, "eardrum", "monaural", 71.2),
]


@pytest.mark.parametrize("frequency, level_db, field, presentation, phon", _C1_OTHER)
def test_annex_c1_other_tones(
    mg_tone,
    frequency: float,
    level_db: float,
    field: str,
    presentation: str,
    phon: float,
) -> None:
    """3 kHz / 100 Hz / 4 kHz and monaural earphone tones match Annex C.1."""
    res = mg_tone(frequency, level_db, field=field, presentation=presentation)
    assert res.loudness_level_max == pytest.approx(phon, abs=1.2)


# Annex C.1 sone columns (previously only the phon values were pinned).
_C1_SONE = [
    # frequency, field, presentation, level -> printed peak LTL in sone
    (3000.0, "free", "binaural", 20.0, 0.37),
    (3000.0, "free", "binaural", 40.0, 1.9),
    (3000.0, "free", "binaural", 60.0, 7.2),
    (3000.0, "free", "binaural", 80.0, 26.4),
    (1000.0, "eardrum", "monaural", 20.0, 0.06),
    (1000.0, "eardrum", "monaural", 40.0, 0.54),
    (1000.0, "eardrum", "monaural", 60.0, 2.3),
    (1000.0, "eardrum", "monaural", 80.0, 8.6),
]


@pytest.mark.parametrize("frequency, field, presentation, level_db, sone", _C1_SONE)
def test_annex_c1_sone_values(
    mg_tone,
    frequency: float,
    field: str,
    presentation: str,
    level_db: float,
    sone: float,
) -> None:
    """Annex C.1.2/C.1.3 peak long-term loudness in sone.

    Values print to two decimals; the absolute floor covers their rounding
    half-width at the quiet end.
    """
    res = mg_tone(frequency, level_db, field=field, presentation=presentation)
    assert res.n_max == pytest.approx(sone, rel=0.02, abs=0.005)


# --------------------------------------------------------------------------
# Annex C.3 - multi-tone complexes (pure tones, fully reproducible)
# --------------------------------------------------------------------------

_C3_CASES = [
    # component frequencies, level per tone, printed sone, printed phon
    ([1500.0, 1600.0, 1700.0], 60.0, 6.36, 66.7),
    ([1000.0, 1600.0, 2400.0], 60.0, 12.8, 77.3),
    ([float(f) for f in range(100, 1001, 100)], 30.0, 1.88, 48.5),
]


@pytest.mark.parametrize("freqs, level_db, sone, phon", _C3_CASES)
def test_annex_c3_multi_tone(
    freqs: list, level_db: float, sone: float, phon: float
) -> None:
    """Annex C.3 multi-tone complexes reproduce the printed peak LTL.

    Computed values sit within 2.5 % (sone) / 0.4 phon of the print (the
    close-tone complex C.3.1 is the worst case); tolerances hold those
    margins with a little headroom, well inside the 2.8 phon expanded
    uncertainty.
    """
    x = np.sum([_tone(f, level_db) for f in freqs], axis=0)
    res = psychoacoustics.loudness_moore_glasberg_time(x, FS)
    assert res.n_max == pytest.approx(sone, rel=0.04)
    assert res.loudness_level_max == pytest.approx(phon, abs=0.5)


# --------------------------------------------------------------------------
# Cross-checks against the ISO 532-2 stationary result
# --------------------------------------------------------------------------


def test_steady_matches_iso532_2_stationary() -> None:
    """A long steady tone converges to the ISO 532-2 stationary loudness."""

    stationary = psychoacoustics.loudness_moore_glasberg_from_spectrum([(1000.0, 60.0)])
    res = psychoacoustics.loudness_moore_glasberg_time(_tone(1000.0, 60.0), FS)
    # Both models share the excitation/specific-loudness machinery (different
    # calibration constants); agree well within the standard's uncertainty.
    assert res.n_max == pytest.approx(stationary.loudness, rel=0.05)


# --------------------------------------------------------------------------
# Temporal behaviour (clauses 7.6, 7.9)
# --------------------------------------------------------------------------


def test_steady_tone_gives_flat_trace() -> None:
    """A steady tone gives an essentially flat long-term-loudness trace."""
    res = psychoacoustics.loudness_moore_glasberg_time(
        _tone(1000.0, 50.0, duration=1.5), FS
    )
    steady = res.long_term_loudness[res.time > 0.8]
    assert steady.std() < 0.02
    assert res.long_term_loudness[-1] == pytest.approx(res.n_max, rel=1e-3)


def test_short_term_reacts_faster_than_long_term() -> None:
    """Short-term loudness rises faster and peaks higher than long-term."""
    res = psychoacoustics.loudness_moore_glasberg_time(_tone(1000.0, 60.0), FS)
    stl, ltl, time = (
        res.short_term_loudness,
        res.long_term_loudness,
        res.time,
    )
    half = 0.5 * ltl[-1]
    t_stl = time[np.argmax(stl >= half)]
    t_ltl = time[np.argmax(ltl >= half)]
    assert t_stl < t_ltl  # short-term attack is faster
    assert stl.max() > ltl.max()  # short-term is more responsive


def test_release_is_slower_than_attack() -> None:
    """After a burst offset the short-term loudness decays far faster than LTL.

    Release time constants (~30 ms STL, ~750 ms LTL) exceed the attack ones,
    and the long-term averager is deliberately sluggish (clause 7.9).
    """
    burst = np.concatenate([_tone(1000.0, 60.0, duration=0.3), np.zeros(int(0.7 * FS))])
    res = psychoacoustics.loudness_moore_glasberg_time(burst, FS)
    stl, ltl, time = (
        res.short_term_loudness,
        res.long_term_loudness,
        res.time,
    )
    offset = int(np.argmin(np.abs(time - 0.3)))

    def half_decay(trace: np.ndarray) -> float:
        v0 = trace[offset]
        j = int(np.argmax(trace[offset:] <= 0.5 * v0))
        return float(time[offset + j] - time[offset]) if j > 0 else np.inf

    assert half_decay(stl) < half_decay(ltl)
    assert half_decay(ltl) > 0.2  # long-term release is sluggish


def test_silence_is_zero() -> None:
    """Silence yields a zero loudness trace and zero peak."""
    res = psychoacoustics.loudness_moore_glasberg_time(np.zeros(int(0.3 * FS)), FS)
    assert res.n_max == 0.0
    assert np.all(res.short_term_loudness == 0.0)
    assert np.all(res.long_term_loudness == 0.0)
    assert res.loudness_level_max == 0.0


def test_louder_tone_is_louder() -> None:
    """Peak long-term loudness increases monotonically with level.

    0.7 s segments: monotonicity in level is duration-independent once the
    long-term averager has settled (attack ~100 ms).
    """
    values = [
        psychoacoustics.loudness_moore_glasberg_time(
            _tone(1000.0, lvl, duration=0.7), FS
        ).n_max
        for lvl in (30.0, 50.0, 70.0)
    ]
    assert values[0] < values[1] < values[2]


# --------------------------------------------------------------------------
# Result plumbing, percentiles and validation
# --------------------------------------------------------------------------


def test_result_fields_and_percentiles() -> None:
    """The result exposes consistent traces, percentiles and metadata."""
    res = psychoacoustics.loudness_moore_glasberg_time(
        _tone(1000.0, 60.0), FS, percentiles=(5.0, 50.0, 95.0)
    )
    assert isinstance(res, psychoacoustics.MooreGlasbergTimeVaryingLoudness)
    n = res.time.size
    assert res.short_term_loudness.shape == (n,)
    assert res.long_term_loudness.shape == (n,)
    assert res.short_term_loudness_level.shape == (n,)
    assert res.long_term_loudness_level.shape == (n,)
    assert set(res.percentiles) == {5.0, 50.0, 95.0}
    # Higher exceedance fraction -> lower level.
    assert res.percentiles[5.0] >= res.percentiles[50.0] >= res.percentiles[95.0]
    assert res.n_max >= res.percentiles[5.0]
    assert res.field == "free"
    assert res.presentation == "binaural"


def test_diotic_equals_binaural_and_exceeds_monaural() -> None:
    """Binaural (diotic) loudness exceeds the single-ear monaural loudness.

    0.7 s: the binaural-vs-monaural gain is a spectral property, not a
    duration effect.
    """
    tone = _tone(1000.0, 60.0, duration=0.7)
    binaural = psychoacoustics.loudness_moore_glasberg_time(
        tone, FS, presentation="binaural"
    )
    monaural = psychoacoustics.loudness_moore_glasberg_time(
        tone, FS, field="eardrum", presentation="monaural"
    )
    assert binaural.n_max > monaural.n_max


def test_stereo_input_accepted() -> None:
    """A two-channel (n, 2) signal is accepted as the two ear signals.

    0.7 s: the (n, 2)-input/diotic equivalence is exact at any length.
    """
    tone = _tone(1000.0, 60.0, duration=0.7)
    stereo = np.column_stack([tone, tone])
    res = psychoacoustics.loudness_moore_glasberg_time(stereo, FS)
    mono = psychoacoustics.loudness_moore_glasberg_time(tone, FS)
    # Identical channels == diotic presentation.
    assert res.n_max == pytest.approx(mono.n_max, rel=1e-6)


def _sum_then_agc(stl: np.ndarray) -> np.ndarray:
    """The old (non-conformant) sum-then-AGC long-term loudness of a trace.

    Runs the clause-7.9 attack/release averager on the *binaural sum* of the
    short-term loudness.  For dichotic input this differs from the standard's
    per-ear-then-sum result because the averager is nonlinear.
    """
    ltl = np.zeros_like(stl)
    state = 0.0
    for k, s in enumerate(stl):
        if s > state:
            state = _ALPHA_AL * s + (1.0 - _ALPHA_AL) * state
        else:
            state = _ALPHA_RL * s + (1.0 - _ALPHA_RL) * state
        ltl[k] = state
    return ltl


def test_dichotic_ltl_is_per_ear_then_summed() -> None:
    """Clause 7.9: long-term loudness is smoothed per ear, then summed.

    For a dichotic transient (a different envelope at each ear) the conformant
    per-ear-then-sum long-term loudness must differ from smoothing the binaural
    sum (the old behaviour); for a diotic input the two are provably identical.
    """
    # Halved (0.5 s) dichotic transient: the per-ear-vs-summed AGC difference
    # is ~1.16 sone here, > 10x the 0.1 threshold, so the shorter signal keeps
    # the same discriminating power at half the cost.
    left = np.concatenate([_tone(1000.0, 75.0, 0.15), np.zeros(int(0.35 * FS))])
    right = np.concatenate(
        [np.zeros(int(0.15 * FS)), _tone(1000.0, 55.0, 0.2), np.zeros(int(0.15 * FS))]
    )
    n = min(left.size, right.size)
    res = psychoacoustics.loudness_moore_glasberg_time(
        np.column_stack([left[:n], right[:n]]), FS
    )
    old = _sum_then_agc(res.short_term_loudness)
    # (a) The fix changes the dichotic result substantially.
    assert np.max(np.abs(res.long_term_loudness - old)) > 0.1

    # (b) A diotic input is byte-identical to the old sum-then-AGC (regression);
    # the identity is algebraic, so 0.7 s is as strong as any length.
    tone = _tone(1000.0, 60.0, duration=0.7)
    diotic = psychoacoustics.loudness_moore_glasberg_time(
        np.column_stack([tone, tone]), FS
    )
    assert np.array_equal(
        diotic.long_term_loudness, _sum_then_agc(diotic.short_term_loudness)
    )


def test_anchor_oracle_is_byte_identical() -> None:
    """Regression pin: the diotic 1 kHz / 40 dB anchor is unchanged by the fix.

    The dichotic-path fix must leave every diotic/monaural oracle bit-for-bit
    identical; this pins the definitional anchor to full float precision.

    NOTE: the pinned number is this implementation's own output, NOT a
    conformance oracle from the standard (those are the Annex C values
    asserted elsewhere in this file). It only detects unintended numerical
    drift; update it deliberately when the chain legitimately changes.
    """
    res = psychoacoustics.loudness_moore_glasberg_time(_tone(1000.0, 40.0), FS)
    assert res.n_max == pytest.approx(1.0000044713237626, abs=1e-12)
    assert res.loudness_level_max == pytest.approx(40.00005907615314, abs=1e-9)


def test_low_freq_band_not_truncated_across_sample_rates() -> None:
    """The 64 ms (20-80 Hz) window must not be truncated at 44.1/48 kHz.

    At 44.1/48 kHz the 64 ms segment is 2822/3072 samples, longer than the
    nominal 2048-point FFT.  A fixed 2048-point transform truncates the Hann
    window tail while its ``sum_w2`` still normalises the full window, so the
    20-80 Hz band under-reads by ~0.75 dB.  With a per-window FFT length >= the
    segment length the recovered low-frequency band level is consistent with
    the 32 kHz case (where the 64 ms window is exactly 2048 samples).
    """
    import importlib

    _mgt_mod = importlib.import_module(
        "phonometry.psychoacoustics.loudness.moore_glasberg_time"
    )

    def _band_level(fs: float) -> float:
        p_rms = 2e-5 * 10.0 ** (60.0 / 20.0)
        t = np.arange(round(1.0 * fs)) / fs
        sig = np.sqrt(2.0) * p_rms * np.sin(2.0 * np.pi * 50.0 * t)
        comp_f, plans, perm = _mgt_mod._spectral_plan(fs)
        levels = _mgt_mod._frame_levels(sig, sig.size // 2, plans, comp_f.size)[perm]
        band = (comp_f >= 20.0) & (comp_f < 80.0)
        return float(10.0 * np.log10(np.sum(10.0 ** (levels[band] / 10.0))))

    ref = _band_level(32000.0)  # exact 2048-sample window, no zero-pad/truncate
    for fs in (44100.0, 48000.0):
        assert _band_level(fs) == pytest.approx(ref, abs=0.05)


def test_invalid_inputs_raise() -> None:
    """Invalid field, presentation, sampling rate and signal raise ValueError."""
    tone = _tone(1000.0, 40.0, duration=0.1)
    with pytest.raises(ValueError, match="field must be one of"):
        psychoacoustics.loudness_moore_glasberg_time(tone, FS, field="bogus")
    with pytest.raises(ValueError, match="presentation must be one of"):
        psychoacoustics.loudness_moore_glasberg_time(tone, FS, presentation="bogus")
    with pytest.raises(ValueError, match="'fs' must be a positive sampling rate"):
        psychoacoustics.loudness_moore_glasberg_time(tone, -1.0)
    empty = np.array([])
    with pytest.raises(ValueError, match="Input signal cannot be empty"):
        psychoacoustics.loudness_moore_glasberg_time(empty, FS)
    with_nan = np.array([1.0, np.nan, 2.0])
    with pytest.raises(
        ValueError, match="Input signal must contain only finite values"
    ):
        psychoacoustics.loudness_moore_glasberg_time(with_nan, FS)


def test_plot_smoke() -> None:
    """The lazy .plot() renders without error when matplotlib is present."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    res = psychoacoustics.loudness_moore_glasberg_time(
        _tone(1000.0, 60.0, duration=0.4), FS
    )
    ax = res.plot()
    assert ax.get_ylabel() == "Loudness [sone]"


def test_three_channel_input_is_rejected() -> None:
    # Mono and two-channel are supported; anything higher-dimensional used to
    # be silently flattened and must be rejected.
    import numpy as np
    import pytest

    from phonometry.psychoacoustics import loudness_moore_glasberg_time

    cube = np.zeros((2, 2, 2400))
    with pytest.raises(ValueError, match="mono .1-D. or two-channel"):
        loudness_moore_glasberg_time(cube, 32000.0)

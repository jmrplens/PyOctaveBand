#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Speech Transmission Index per IEC 60268-16:2020 (Edition 5).

Validation vectors:
- Weighting-factor test (Ed.5 A.2.2): band pairs with TI=1 give exact STI
  values 0,127 / 0,279 / 0,398 / 0,531 / 0,486 / 0,302 (tol 0,001);
  analytically STI = alpha_k + alpha_{k+1} - beta_k.
- Filter-bank phase test pairs (Ed.5 A.3.1.2): uniform m <-> STI mapping,
  e.g. (0,5; 0,5), (0,11182; 0,2), tol +/-0,01.
- Indirect method (Ed.5 C.3.3): exponential-decay IRs vs the analytic
  Schroeder MTF m(F) = 1/sqrt(1 + (2 pi F T/13,8)^2).
- Auditory masking control points (Ed.5 Table A.2): L=60 -> -35 dB;
  65 -> -29,9; 80 -> -19,8; 100 -> -10.
- Annex F qualification bands: edges 0,36-0,76 in 0,04 steps, U..A+.
"""

import warnings
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pytest
from reference_data import (
    IEC60268_16_ANNEX_M_ART_DB,
    IEC60268_16_ANNEX_M_EFFECTIVE_SNR,
    IEC60268_16_ANNEX_M_INTENSITY_SCALE,
    IEC60268_16_ANNEX_M_INTENSITY_THRESHOLD,
    IEC60268_16_ANNEX_M_MEASURED_AMBIENT,
    IEC60268_16_ANNEX_M_MEASURED_COMBINED_ADJUSTMENT,
    IEC60268_16_ANNEX_M_MEASURED_COMBINED_LEVEL,
    IEC60268_16_ANNEX_M_MEASURED_INTENSITY,
    IEC60268_16_ANNEX_M_MEASURED_INTENSITY_MASKING,
    IEC60268_16_ANNEX_M_MEASURED_LEVEL,
    IEC60268_16_ANNEX_M_MEASURED_MASKING_DB,
    IEC60268_16_ANNEX_M_MEASURED_MASKING_MILLI,
    IEC60268_16_ANNEX_M_MEASURED_MASKING_THRESHOLD_ADJUSTMENT,
    IEC60268_16_ANNEX_M_MEASURED_MTF,
    IEC60268_16_ANNEX_M_MEASURED_NOISE_ADJUSTMENT,
    IEC60268_16_ANNEX_M_MEASURED_NOISE_TRANSFER,
    IEC60268_16_ANNEX_M_MEASURED_SNR,
    IEC60268_16_ANNEX_M_MTI,
    IEC60268_16_ANNEX_M_OPERATIONAL_AMBIENT,
    IEC60268_16_ANNEX_M_OPERATIONAL_COMBINED_ADJUSTMENT,
    IEC60268_16_ANNEX_M_OPERATIONAL_COMBINED_LEVEL,
    IEC60268_16_ANNEX_M_OPERATIONAL_INTENSITY,
    IEC60268_16_ANNEX_M_OPERATIONAL_INTENSITY_MASKING,
    IEC60268_16_ANNEX_M_OPERATIONAL_LEVEL,
    IEC60268_16_ANNEX_M_OPERATIONAL_MASKING_DB,
    IEC60268_16_ANNEX_M_OPERATIONAL_MASKING_MILLI,
    IEC60268_16_ANNEX_M_OPERATIONAL_MASKING_THRESHOLD_TRANSFER,
    IEC60268_16_ANNEX_M_OPERATIONAL_MTF,
    IEC60268_16_ANNEX_M_OPERATIONAL_NOISE_TRANSFER,
    IEC60268_16_ANNEX_M_OPERATIONAL_SNR,
    IEC60268_16_ANNEX_M_SOURCE_MTF,
    IEC60268_16_ANNEX_M_STI,
)

from phonometry import speech
from phonometry.speech.sti import (
    _ALPHA_MALE,
    _ART_DB,
    _BETA_MALE,
    _MOD_FREQS,
    _NUM_BANDS,
    _RATING_EDGES,
    _RATING_LETTERS,
    STIResult,
    STIWarning,
    _level_correction,
    _LevelCorrection,
    _masking_amdb,
    _rating,
    _sti_from_mtf,
    _truncated_mtf,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

FS = 48000


def _uniform_mtf(m: float) -> np.ndarray:
    return np.full((_NUM_BANDS, _MOD_FREQS.size), m)


def _decay_ir(t60: float, fs: int, seed: int = 0) -> np.ndarray:
    """Noise-carrier impulse response with exponential energy decay:
    p(t) ~ e^(-13.8 t / T60), i.e. -60 dB at t = T60.
    """
    rng = np.random.default_rng(seed)
    n = int(2.0 * t60 * fs)
    t = np.arange(n) / fs
    return rng.standard_normal(n) * np.exp(-3.0 * np.log(10.0) * t / t60)


def _analytic_decay_sti(t60: float) -> float:
    """Expected STI from the closed-form Schroeder MTF of an exponential
    decay, identical in all bands: m(F) = 1/sqrt(1 + (2 pi F T/13,8)^2).
    """
    m = 1.0 / np.sqrt(1.0 + (2.0 * np.pi * _MOD_FREQS * t60 / 13.8) ** 2)
    return _sti_from_mtf(np.tile(m, (_NUM_BANDS, 1))).sti


# ---------------------------------------------------------------------------
# Final formula: weighting/redundancy factors (Ed.5 A.2.2 verification test)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bands", "expected"),
    [
        ((0, 1), 0.127),  # 125 + 250 Hz
        ((1, 2), 0.279),  # 250 + 500 Hz
        ((2, 3), 0.398),  # 500 + 1000 Hz
        ((3, 4), 0.531),  # 1000 + 2000 Hz
        ((4, 5), 0.486),  # 2000 + 4000 Hz
        ((5, 6), 0.302),  # 4000 + 8000 Hz
    ],
)
def test_weighting_factor_pairs(bands: tuple[int, int], expected: float) -> None:
    # m=1 -> SNR_eff clipped to +15 -> TI=1; m=0 -> -15 -> TI=0.
    mtf = _uniform_mtf(0.0)
    mtf[list(bands), :] = 1.0
    result = _sti_from_mtf(mtf)
    assert result.sti == pytest.approx(expected, abs=0.001)
    # Analytic identity for adjacent pairs: alpha_k + alpha_{k+1} - beta_k.
    k = bands[0]
    assert result.sti == pytest.approx(
        _ALPHA_MALE[k] + _ALPHA_MALE[k + 1] - _BETA_MALE[k], abs=1e-12
    )


@pytest.mark.parametrize(
    ("m", "expected_sti"),
    [
        (0.0, 0.0),
        (0.059351, 0.1),
        (0.11182, 0.2),
        (0.20076, 0.3),
        (0.33386, 0.4),
        (0.5, 0.5),
        (0.66614, 0.6),
        (0.79924, 0.7),
        (0.88818, 0.8),
        (0.94065, 0.9),
        (1.0, 1.0),
    ],
)
def test_m_to_sti_mapping(m: float, expected_sti: float) -> None:
    # Ed.5 A.3.1.2 pairs: uniform m across all bands and modulation
    # frequencies maps to the given STI (tol +/-0,01).
    result = _sti_from_mtf(_uniform_mtf(m))
    assert result.sti == pytest.approx(expected_sti, abs=0.01)


def test_alpha_beta_artifact_truncated_to_one() -> None:
    # Ed.4 Table A.3 NOTE (= Ed.5 Table A.1): with the 250 Hz band knocked
    # out (TI=0) and all other bands at TI=1 the raw formula gives 1,036
    # (sum(alpha) - alpha_1 - sum(beta) + beta_0 + beta_1); it must be
    # truncated to 1,0.
    mtf = _uniform_mtf(1.0)
    mtf[1, :] = 0.0
    raw = (
        _ALPHA_MALE.sum()
        - _ALPHA_MALE[1]
        - _BETA_MALE.sum()
        + _BETA_MALE[0]
        + _BETA_MALE[1]
    )
    assert raw == pytest.approx(1.036, abs=1e-3)
    result = _sti_from_mtf(mtf)
    assert result.sti == 1.0
    assert result.rating == "A+"


# ---------------------------------------------------------------------------
# Indirect method: impulse responses
# ---------------------------------------------------------------------------


def test_delta_impulse_response_is_perfect_transmission() -> None:
    ir = np.zeros(FS // 2)
    ir[100] = 1.0
    result = speech.sti_from_impulse_response(ir, FS)
    # The residual error is the analysis filter bank's own MTF; the
    # standard allows < 0,01 STI systematic error (Ed.4 A.5.1.2).
    assert result.sti == pytest.approx(1.0, abs=0.01)
    assert result.rating == "A+"
    assert result.mtf.shape == (7, 14)
    assert result.mti.shape == (7,)
    assert result.band_levels is None
    assert isinstance(result, speech.STIResult)


def test_exponential_decay_matches_analytic_schroeder_mtf() -> None:
    fs = 24000
    stis = []
    for t60 in (0.5, 1.0, 2.0, 4.0):
        got = speech.sti_from_impulse_response(_decay_ir(t60, fs), fs).sti
        assert got == pytest.approx(_analytic_decay_sti(t60), abs=0.01)
        stis.append(got)
    # Monotonic: longer reverberation always degrades intelligibility.
    assert all(a > b for a, b in pairwise(stis))


def test_snr_degradation_on_impulse_response() -> None:
    fs = 24000
    ir = _decay_ir(1.0, fs)
    plain = speech.sti_from_impulse_response(ir, fs)
    high_snr = speech.sti_from_impulse_response(ir, fs, snr=30.0)
    zero_snr = speech.sti_from_impulse_response(ir, fs, snr=0.0)
    # +30 dB SNR: m factor 1/(1+10^-3) = 0,999 -> no visible change.
    assert high_snr.sti == pytest.approx(plain.sti, abs=0.01)
    # 0 dB SNR: every m halved -> STI drops markedly.
    np.testing.assert_allclose(zero_snr.mtf, plain.mtf / 2.0, rtol=1e-12)
    assert zero_snr.sti < plain.sti - 0.1
    # Per-band SNR vector is accepted and equals the scalar case.
    vec = speech.sti_from_impulse_response(ir, fs, snr=np.zeros(7))
    assert vec.sti == pytest.approx(zero_snr.sti, abs=1e-12)


def test_level_corrections_reduce_sti() -> None:
    fs = 24000
    ir = _decay_ir(1.0, fs)
    plain = speech.sti_from_impulse_response(ir, fs)
    # Comfortable speech levels: masking/threshold effects are small.
    comfortable = speech.sti_from_impulse_response(
        ir, fs, level=[62, 62, 59, 53, 47, 41, 35]
    )
    # Very quiet speech: the absolute reception threshold dominates.
    quiet = speech.sti_from_impulse_response(ir, fs, level=[20, 20, 17, 11, 5, -1, -7])
    assert comfortable.sti <= plain.sti
    assert quiet.sti < comfortable.sti - 0.05
    assert comfortable.band_levels is not None
    # Ambient noise at the listener degrades further.
    noisy = speech.sti_from_impulse_response(
        ir, fs, level=[62, 62, 59, 53, 47, 41, 35], ambient=[55] * 7
    )
    assert noisy.sti < comfortable.sti


# ---------------------------------------------------------------------------
# Auditory masking function (Ed.5 Table A.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "expected"),
    [(60.0, -35.0), (65.0, -29.9), (80.0, -19.8), (100.0, -10.0)],
)
def test_masking_amdb_control_points(level: float, expected: float) -> None:
    assert _masking_amdb(level) == pytest.approx(expected, abs=1e-9)


def test_masking_amdb_is_vectorized_and_continuous() -> None:
    levels = np.array([62.999, 63.0, 66.999, 67.0, 99.999, 100.0, 120.0])
    out = _masking_amdb(levels)
    assert out.shape == levels.shape
    # The 63 and 67 dB joints are continuous; at 100 dB the table has a
    # deliberate 0,2 dB step onto the -10 dB plateau.
    assert out[0] == pytest.approx(out[1], abs=1e-2)
    assert out[2] == pytest.approx(out[3], abs=1e-2)
    assert out[4] == pytest.approx(-9.8, abs=1e-2)
    assert out[5] == -10.0
    assert out[6] == -10.0


# ---------------------------------------------------------------------------
# STIPA: direct method and test-signal generator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stipa_18s_seed1234() -> np.ndarray:
    """The 18 s seed-1234 STIPA test signal, generated once for the module."""
    return speech.stipa_signal(FS, seconds=18.0, seed=1234)


def test_stipa_loopback_ideal_channel(stipa_18s_seed1234: np.ndarray) -> None:
    x = stipa_18s_seed1234
    result = speech.stipa(x, FS)
    # Ideal loopback recovers STI 0.998 and min MTF 0.945 at 18 s; lock those
    # in (was >= 0.95 / > 0.9, several x looser than the achieved accuracy).
    assert result.sti >= 0.99
    assert result.rating == "A+"
    assert result.mtf.shape == (7, 2)
    assert np.all(result.mtf > 0.93)


def test_stipa_short_recording_warns(stipa_18s_seed1234: np.ndarray) -> None:
    """A recording shorter than the recommended 15 s biases the recovered
    modulation depths (and STI) low; stipa should warn (IEC 60268-16 STIPA
    practice recommends 15 s to 25 s).
    """
    short = speech.stipa_signal(FS, seconds=5.0, seed=1234)
    with pytest.warns(
        UserWarning, match=r"STIPA recording is .* shorter than the recommended"
    ):
        speech.stipa(short, FS)
    # No warning at the recommended length.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        speech.stipa(stipa_18s_seed1234, FS)


def test_stipa_with_noise_is_monotonic(stipa_18s_seed1234: np.ndarray) -> None:
    x = stipa_18s_seed1234
    rng = np.random.default_rng(7)
    rms = float(np.sqrt(np.mean(x**2)))
    stis = []
    for snr_db in (30.0, 10.0, 0.0):
        noise = rng.standard_normal(x.size) * rms * 10.0 ** (-snr_db / 20.0)
        stis.append(speech.stipa(x + noise, FS).sti)
    assert all(a > b for a, b in pairwise(stis))
    assert stis[-1] < 0.7  # 0 dB broadband SNR is clearly degraded


def test_stipa_reference_normalization() -> None:
    x = speech.stipa_signal(FS, seconds=18.0, seed=99)
    with_nominal = speech.stipa(x, FS)
    with_reference = speech.stipa(0.25 * x, FS, reference=x)
    # Loop-back against the emitted signal itself: m = 1 in every band
    # (gain does not affect modulation depths).
    assert with_reference.sti == pytest.approx(1.0, abs=1e-6)
    assert with_nominal.sti == pytest.approx(with_reference.sti, abs=0.05)


def test_stipa_signal_properties() -> None:
    seconds = 18.0
    x = speech.stipa_signal(FS, seconds=seconds, seed=0)
    assert x.shape == (int(seconds * FS),)
    # Default normalization: RMS = 0,1 digital units.
    assert float(np.sqrt(np.mean(x**2))) == pytest.approx(0.1, rel=1e-9)
    # Crest factor of the STIPA signal: expected around 12-14 dB (A.4).
    crest_db = 20.0 * np.log10(np.max(np.abs(x)) / np.sqrt(np.mean(x**2)))
    assert 9.0 < crest_db < 16.0
    # Calibrated output: overall level in dB re 20 uPa.
    x_cal = speech.stipa_signal(FS, seconds=6.0, level_db=74.0, seed=0)
    level = 20.0 * np.log10(np.sqrt(np.mean(x_cal**2)) / 2e-5)
    assert level == pytest.approx(74.0, abs=1e-9)
    # Reproducible for a fixed seed.
    np.testing.assert_array_equal(x, speech.stipa_signal(FS, seconds=seconds, seed=0))


# ---------------------------------------------------------------------------
# Annex F qualification rating
# ---------------------------------------------------------------------------


def test_rating_letters_from_band_edges() -> None:
    assert _rating(0.74) == "A"
    # Compute the expected letter from the Annex F edges and assert the
    # helper is consistent across the whole scale, including boundaries.
    edges = np.asarray(_RATING_EDGES)
    for sti in np.arange(0.0, 1.001, 0.01):
        expected = _RATING_LETTERS[int(np.searchsorted(edges, sti, side="right"))]
        assert _rating(float(sti)) == expected
    # Band centres (Annex F): 0,38 -> J ... 0,74 -> A; extremes U / A+.
    assert _rating(0.30) == "U"
    assert _rating(0.38) == "J"
    assert _rating(0.50) == "G"
    assert _rating(0.62) == "D"
    assert _rating(0.80) == "A+"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_inputs_raise() -> None:
    ir = np.zeros(FS // 4)
    ir[10] = 1.0
    # the signals are built outside the raises blocks, so each block holds
    # exactly the one call whose exception is under test
    two_dimensional_ir = np.zeros((2, 1000))
    silent_ir = np.zeros(FS // 4)

    with pytest.raises(
        ValueError, match=r"sti_from_impulse_response expects a 1D impulse response"
    ):
        speech.sti_from_impulse_response(two_dimensional_ir, FS)
    with pytest.raises(ValueError, match=r"Sample rate 'fs' must be positive"):
        speech.sti_from_impulse_response(ir, -1)
    with pytest.raises(ValueError, match="8 kHz octave band"):
        speech.sti_from_impulse_response(ir, 16000)
    with pytest.raises(ValueError, match=r"Impulse response 'ir' is silent"):
        speech.sti_from_impulse_response(silent_ir, FS)
    with pytest.raises(
        ValueError, match=r"'level' must contain exactly .* octave-band values"
    ):
        speech.sti_from_impulse_response(ir, FS, level=[60.0, 60.0, 60.0])
    with pytest.raises(ValueError, match="'snr' must be a scalar or a vector"):
        speech.sti_from_impulse_response(ir, FS, snr=[10.0, 10.0])
    with pytest.raises(ValueError, match="requires the speech octave-band levels"):
        speech.sti_from_impulse_response(ir, FS, ambient=[40.0] * 7)
    with pytest.raises(
        ValueError, match=r"Provide either 'snr' or 'ambient' noise levels, not both"
    ):
        speech.sti_from_impulse_response(
            ir, FS, snr=10.0, level=[60.0] * 7, ambient=[40.0] * 7
        )

    two_dimensional_signal = np.zeros((2, FS))
    with pytest.raises(ValueError, match=r"stipa expects a 1D signal"):
        speech.stipa(two_dimensional_signal, FS)
    half_second_clip = speech.stipa_signal(FS, seconds=18.0, seed=3)[: FS // 2]
    # The 0.5 s clip also triggers the (correct) sub-15 s STIPA warning;
    # silence it so the test output stays clean while asserting the error.
    # simplefilter has to run inside catch_warnings, before the call, so the
    # two blocks stay nested rather than combined.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(
            ValueError,
            match=r"Signal too short for STIPA: it must contain at least one full period",
        ):
            speech.stipa(half_second_clip, FS)
    with pytest.warns(STIWarning) as tone_warnings:  # noqa: PT031 - the warns block records the whole STIWarning family while running
        # A pure tone leaves other octave bands empty: those bands read
        # m = 0 (TI = 0) with a warning rather than a hard error, so the
        # IEC 60268-16 C.4.2 verification signals (energy in only two
        # bands) remain measurable. The 4 s clip also (correctly) raises
        # the sub-15 s STIPA and the m > 1.3 advisories; recording the
        # whole STIWarning family keeps them out of the run summary (a
        # leaked warning is re-materialised on the pytest-xdist controller
        # by importing its module, which races the phonometry import).
        t = np.arange(4 * FS) / FS
        res_tone = speech.stipa(np.sin(2 * np.pi * 1000.0 * t), FS)
    assert any("No energy in octave band" in str(w.message) for w in tone_warnings)
    # The 1 kHz band carries the tone (unmodulated: m ~ 0); at least one
    # dead band integrates to non-positive envelope energy and is pinned
    # to exactly m = 0 instead of raising.
    assert np.any(res_tone.mtf == 0.0)
    assert np.all(res_tone.mtf[3] < 0.01)

    with pytest.raises(ValueError, match=r"'seconds' must be positive"):
        speech.stipa_signal(FS, seconds=0.0)
    with pytest.raises(
        ValueError, match=r"Sample rate 'fs' must be .* half-octave carrier"
    ):
        speech.stipa_signal(8000)

    wrong_shape_mtf = np.full((3, 14), 0.5)
    negative_mtf = _uniform_mtf(-0.1)
    with pytest.raises(ValueError, match=r"'mtf' must have shape"):
        _sti_from_mtf(wrong_shape_mtf)
    with pytest.raises(ValueError, match=r"Modulation transfer values must be finite"):
        _sti_from_mtf(negative_mtf)


def test_mtf_above_1_3_warns_and_truncates() -> None:
    over_unity = _uniform_mtf(1.4)
    with pytest.warns(
        UserWarning, match=r"Modulation transfer values above .* detected"
    ):
        result = _sti_from_mtf(over_unity)
    assert result.sti == 1.0


# ---------------------------------------------------------------------------
# Ed.5 C.3.2 - modulation-depth test, direct method (end-to-end oracle)
# ---------------------------------------------------------------------------

# Expected STI for the Ed.5 Formula (C.1) test signal at modulation scale
# m = 0,0, 0,1, ... 1,0 (Ed.5 Table C.2 staircase), tolerance +/-0,05.
_C32_STI_STAIRCASE = [0.0, 0.18, 0.30, 0.38, 0.44, 0.50, 0.56, 0.62, 0.70, 0.82, 1.0]


def _c32_signal(m: float, fs: int, seconds: float) -> np.ndarray:
    """Ed.5 Formula (C.1) verification signal with sinusoidal carriers.

    A_k(t) = g_k sin(2 pi fc_k t) sqrt(0,5 (1 + 0,55 m (sin 2 pi f1_k t
    - sin 2 pi f2_k t))), with the Table B.1 modulation pairs and the
    A.6.1 male spectrum g_k; the values here restate the standard, not
    the implementation.
    """
    centers = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    f1 = [1.60, 1.00, 0.63, 2.00, 1.25, 0.80, 2.50]
    f2 = [8.00, 5.00, 3.15, 10.00, 6.25, 4.00, 12.50]
    levels_db = [-2.5, 0.5, 0.0, -6.0, -12.0, -18.0, -24.0]
    t = np.arange(round(seconds * fs)) / fs
    x = np.zeros(t.size)
    for fc, fa, fb, level in zip(centers, f1, f2, levels_db, strict=True):
        envelope = 0.5 * (
            1.0 + 0.55 * m * (np.sin(2 * np.pi * fa * t) - np.sin(2 * np.pi * fb * t))
        )
        x += (
            10.0 ** (level / 20.0)
            * np.sin(2 * np.pi * fc * t)
            * np.sqrt(np.maximum(envelope, 0.0))
        )
    return x


@pytest.mark.parametrize("i", range(len(_C32_STI_STAIRCASE)))
def test_stipa_direct_method_modulation_depth_staircase(i: int) -> None:
    """Ed.5 C.3.2: the full stipa() audio path (octave bank, intensity
    envelopes, sine/cosine correlation, TI chain) must reproduce the
    published STI staircase for the Formula (C.1) signal within +/-0,05.
    """
    m = i / 10.0
    x = _c32_signal(m, FS, seconds=16.0)
    res = speech.stipa(x, FS)
    assert res.sti == pytest.approx(_C32_STI_STAIRCASE[i], abs=0.05)


# ---------------------------------------------------------------------------
# Annex M: adjusting a measured result to occupancy noise and other speech
# levels. Oracle: IEC 60268-16 Ed.4 (2011), Annex M, Table M.1 "Example
# calculation" (printed pp. 64-66), which walks the four steps of the
# procedure over one measurement and prints every intermediate on the way.
#
# Two tolerances recur below and both come from the same place: the step 1
# matrix is printed to three decimals, so every cell of it carries up to
# 0,0005 of hidden rounding. Propagated through the adjustment that is one
# unit in the last printed place of an MTF cell (0,001), and, because
# d/dm of 10 lg(m/(1-m)) reaches ~220 dB per unit m at the m ~ 0,98 of the
# first modulation frequencies, up to ~0,11 dB of an effective SNR.
# ---------------------------------------------------------------------------
_ANNEX_M_MTF_TOL = 0.001
_ANNEX_M_SNR_TOL = 0.08


def _annex_m_matrix(rows: tuple[tuple[float, ...], ...]) -> np.ndarray:
    """One printed MTF block as a (7 bands, 14 modulation frequencies) array.

    Table M.1 prints the modulation frequencies down the page and the octave
    bands across it, the transpose of the library's matrix.
    """
    matrix = np.asarray(rows, dtype=float).T
    assert matrix.shape == (_NUM_BANDS, _MOD_FREQS.size)
    return matrix


def _annex_m_measured_correction() -> _LevelCorrection:
    """The A.5.3 correction of the measurement condition (step 2 of Table M.1)."""
    return _level_correction(
        np.asarray(IEC60268_16_ANNEX_M_MEASURED_LEVEL, dtype=float),
        np.asarray(IEC60268_16_ANNEX_M_MEASURED_AMBIENT, dtype=float),
    )


def _annex_m_operational_correction() -> _LevelCorrection:
    """The A.5.3 correction of the operational condition (step 3 of Table M.1)."""
    return _level_correction(
        np.asarray(IEC60268_16_ANNEX_M_OPERATIONAL_LEVEL, dtype=float),
        np.asarray(IEC60268_16_ANNEX_M_OPERATIONAL_AMBIENT, dtype=float),
    )


def _annex_m_result() -> speech.STIResult:
    """The adjusted result of Table M.1, from step 1 to step 4."""
    return speech.sti_adjusted_for_levels(
        _annex_m_matrix(IEC60268_16_ANNEX_M_MEASURED_MTF),
        measured_level=IEC60268_16_ANNEX_M_MEASURED_LEVEL,
        measured_ambient=IEC60268_16_ANNEX_M_MEASURED_AMBIENT,
        operational_level=IEC60268_16_ANNEX_M_OPERATIONAL_LEVEL,
        operational_ambient=IEC60268_16_ANNEX_M_OPERATIONAL_AMBIENT,
    )


def _masked_bands(printed: tuple[float | None, ...]) -> tuple[np.ndarray, np.ndarray]:
    """The bands of a printed row that carry a value, and those values.

    The auditory masking rows print "not applicable" in the 125 Hz band,
    which has no band below it to be masked by.
    """
    bands = np.array([k for k, cell in enumerate(printed) if cell is not None])
    values = np.array([cell for cell in printed if cell is not None], dtype=float)
    return bands, values


def test_annex_m_step2_prints_the_measurement_correction() -> None:
    """Every printed intermediate of Table M.1 step 2, band by band.

    Step 2 removes the background noise, the auditory masking and the
    reception threshold that the measurement condition put into the matrix,
    and the annex prints the whole chain: the signal-to-noise ratio, m_k(f)
    for noise only and its reciprocal, the combined speech and noise level,
    the auditory masking factor in dB and as amf x 1000, the combined
    squared sound pressure I_k, I_am,k, the absolute reception threshold and
    I_rt,k, the masking and threshold adjustment, and the combined
    adjustment. Reproducing them one at a time is what tells a chain that
    lands on the right STI through two compensating errors from one that is
    right all the way down.
    """
    correction = _annex_m_measured_correction()

    assert correction.snr == pytest.approx(IEC60268_16_ANNEX_M_MEASURED_SNR, abs=0.005)
    assert correction.noise_transfer == pytest.approx(
        IEC60268_16_ANNEX_M_MEASURED_NOISE_TRANSFER, abs=0.0005
    )
    assert 1.0 / correction.noise_transfer == pytest.approx(
        IEC60268_16_ANNEX_M_MEASURED_NOISE_ADJUSTMENT, abs=0.0005
    )
    assert correction.combined_level == pytest.approx(
        IEC60268_16_ANNEX_M_MEASURED_COMBINED_LEVEL, abs=0.005
    )

    bands, printed_masking = _masked_bands(IEC60268_16_ANNEX_M_MEASURED_MASKING_DB)
    assert correction.masking_db[bands] == pytest.approx(printed_masking, abs=0.05)
    assert np.isneginf(correction.masking_db[0])
    bands, printed_milli = _masked_bands(IEC60268_16_ANNEX_M_MEASURED_MASKING_MILLI)
    assert 1000.0 * 10.0 ** (correction.masking_db[bands] / 10.0) == pytest.approx(
        printed_milli, rel=0.005
    )

    assert correction.intensity / IEC60268_16_ANNEX_M_INTENSITY_SCALE == pytest.approx(
        IEC60268_16_ANNEX_M_MEASURED_INTENSITY, rel=0.005
    )
    assert correction.intensity_masking[1:] == pytest.approx(
        IEC60268_16_ANNEX_M_MEASURED_INTENSITY_MASKING[1:], rel=0.005
    )
    assert correction.intensity_masking[0] == 0.0
    np.testing.assert_array_equal(_ART_DB, IEC60268_16_ANNEX_M_ART_DB)
    # The I_rt,k row is printed to two figures in four of its seven cells
    # (4,5 for 10^0,65 = 4,4668), which is what sets this tolerance.
    assert correction.intensity_threshold == pytest.approx(
        IEC60268_16_ANNEX_M_INTENSITY_THRESHOLD, rel=0.01
    )

    assert 1.0 / correction.masking_threshold_transfer == pytest.approx(
        IEC60268_16_ANNEX_M_MEASURED_MASKING_THRESHOLD_ADJUSTMENT, abs=0.0005
    )
    assert 1.0 / correction.factor == pytest.approx(
        IEC60268_16_ANNEX_M_MEASURED_COMBINED_ADJUSTMENT, abs=0.0005
    )


def test_annex_m_step2_recovers_the_printed_source_matrix() -> None:
    """The measured matrix divided by the measurement correction (step 2).

    The result is the modulation transfer of the transmission channel
    alone, printed in the annex as the "adjusted MTF matrix without noise,
    masking and threshold". All 98 cells land within one unit in the last
    printed place.
    """
    measured = _truncated_mtf(_annex_m_matrix(IEC60268_16_ANNEX_M_MEASURED_MTF))
    source = measured / _annex_m_measured_correction().factor[:, np.newaxis]
    np.testing.assert_allclose(
        source,
        _annex_m_matrix(IEC60268_16_ANNEX_M_SOURCE_MTF),
        atol=_ANNEX_M_MTF_TOL,
        rtol=0.0,
    )


def test_annex_m_step3_prints_the_operational_correction() -> None:
    """Every printed intermediate of Table M.1 step 3, band by band.

    The same chain as step 2 at the operational speech and occupancy-noise
    levels, printed as transfer factors rather than as their reciprocals
    because here the correction is applied instead of undone. The 250 Hz
    cell of the I_am,k row is the one printed value of the whole table that
    does not reproduce; it is asserted separately below against the quantity
    it names, and recorded in docs/ERRATA.md.
    """
    correction = _annex_m_operational_correction()

    assert correction.snr == pytest.approx(
        IEC60268_16_ANNEX_M_OPERATIONAL_SNR, abs=0.005
    )
    assert correction.noise_transfer == pytest.approx(
        IEC60268_16_ANNEX_M_OPERATIONAL_NOISE_TRANSFER, abs=0.0005
    )
    assert correction.combined_level == pytest.approx(
        IEC60268_16_ANNEX_M_OPERATIONAL_COMBINED_LEVEL, abs=0.05
    )

    bands, printed_masking = _masked_bands(IEC60268_16_ANNEX_M_OPERATIONAL_MASKING_DB)
    assert correction.masking_db[bands] == pytest.approx(printed_masking, abs=0.05)
    bands, printed_milli = _masked_bands(IEC60268_16_ANNEX_M_OPERATIONAL_MASKING_MILLI)
    assert 1000.0 * 10.0 ** (correction.masking_db[bands] / 10.0) == pytest.approx(
        printed_milli, rel=0.005
    )

    assert correction.intensity / IEC60268_16_ANNEX_M_INTENSITY_SCALE == pytest.approx(
        IEC60268_16_ANNEX_M_OPERATIONAL_INTENSITY, rel=0.005
    )
    assert correction.intensity_masking[2:] == pytest.approx(
        IEC60268_16_ANNEX_M_OPERATIONAL_INTENSITY_MASKING[2:], rel=0.005
    )
    assert correction.intensity_threshold == pytest.approx(
        IEC60268_16_ANNEX_M_INTENSITY_THRESHOLD, rel=0.01
    )

    assert correction.masking_threshold_transfer == pytest.approx(
        IEC60268_16_ANNEX_M_OPERATIONAL_MASKING_THRESHOLD_TRANSFER, abs=0.0005
    )
    assert correction.factor == pytest.approx(
        IEC60268_16_ANNEX_M_OPERATIONAL_COMBINED_ADJUSTMENT, abs=0.0005
    )


def test_annex_m_step3_masking_intensity_at_250_hz_is_the_printed_erratum() -> None:
    """The one printed cell of Table M.1 that names a value it is not.

    I_am,k at 250 Hz of step 3 is amf x I_k of the 125 Hz band below it,
    2 858 804 on the printed levels, which the table prints as 2 850 000.
    The 500 Hz cell beside it, 2 852 252, prints as 2 850 000 correctly, and
    step 2 prints the two neighbours apart (508 000 and 507 000), so the
    defect is this cell rather than the annex's rounding. It moves the
    masking and threshold correction of the band by 44 parts in a million
    and changes no printed result. See docs/ERRATA.md.
    """
    correction = _annex_m_operational_correction()
    assert correction.intensity_masking[1] == pytest.approx(2_858_803.8, abs=1.0)
    assert correction.intensity_masking[1] != pytest.approx(
        IEC60268_16_ANNEX_M_OPERATIONAL_INTENSITY_MASKING[1], rel=0.001
    )


def test_annex_m_adjustment_reproduces_the_worked_example() -> None:
    """The whole of Table M.1: measured matrix and levels in, STI out.

    Step 3's printed matrix, step 4a's effective SNRs, step 4c's per-band
    MTI row and the STI, from the step 1 matrix and the four printed level
    spectra. The MTI row is checked at the annex's own two decimals rather
    than with a tolerance on top of them, which would accept the
    neighbouring cent.
    """
    result = _annex_m_result()

    np.testing.assert_allclose(
        result.mtf,
        _annex_m_matrix(IEC60268_16_ANNEX_M_OPERATIONAL_MTF),
        atol=_ANNEX_M_MTF_TOL,
        rtol=0.0,
    )
    with np.errstate(divide="ignore"):
        effective_snr = 10.0 * np.log10(result.mtf / (1.0 - result.mtf))
    np.testing.assert_allclose(
        effective_snr,
        _annex_m_matrix(IEC60268_16_ANNEX_M_EFFECTIVE_SNR),
        atol=_ANNEX_M_SNR_TOL,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        np.round(result.mti, 2), np.asarray(IEC60268_16_ANNEX_M_MTI, dtype=float)
    )
    assert result.sti == pytest.approx(IEC60268_16_ANNEX_M_STI, abs=0.005)
    assert round(result.sti, 2) == IEC60268_16_ANNEX_M_STI
    assert result.band_levels == pytest.approx(IEC60268_16_ANNEX_M_OPERATIONAL_LEVEL)
    assert result.ambient_levels == pytest.approx(
        IEC60268_16_ANNEX_M_OPERATIONAL_AMBIENT
    )


def test_annex_m_step4_from_the_printed_effective_snrs() -> None:
    """Step 4b to 4c on the annex's own effective SNRs.

    Inverting step 4a exactly, m = 1/(1 + 10^(-SNR/10)), feeds the printed
    step 4a table straight into the library's +/-15 dB clamp, transmission
    indices, band MTI average and alpha/beta weighting. It pins that tail of
    the chain to the annex without going through the modulation matrix, so a
    defect in the adjustment and a compensating one in the weighting cannot
    both hide here.
    """
    printed_snr = _annex_m_matrix(IEC60268_16_ANNEX_M_EFFECTIVE_SNR)
    result = _sti_from_mtf(1.0 / (1.0 + 10.0 ** (-printed_snr / 10.0)))
    np.testing.assert_array_equal(
        np.round(result.mti, 2), np.asarray(IEC60268_16_ANNEX_M_MTI, dtype=float)
    )
    assert round(result.sti, 2) == IEC60268_16_ANNEX_M_STI


def test_annex_m_forward_chain_on_the_printed_source_matrix() -> None:
    """The forward A.5.3 chain agrees with the adjustment on step 3.

    Applying the operational levels to the printed step 2 matrix through the
    ordinary measurement entry point must give the same index as running the
    adjustment from step 1, because both multiply by the one correction of
    the module. Any second implementation of the masking and threshold model
    would show up as a difference here.
    """
    forward = _sti_from_mtf(
        _annex_m_matrix(IEC60268_16_ANNEX_M_SOURCE_MTF),
        level=np.asarray(IEC60268_16_ANNEX_M_OPERATIONAL_LEVEL, dtype=float),
        ambient=np.asarray(IEC60268_16_ANNEX_M_OPERATIONAL_AMBIENT, dtype=float),
    )
    assert round(forward.sti, 2) == IEC60268_16_ANNEX_M_STI
    assert forward.sti == pytest.approx(_annex_m_result().sti, abs=0.002)


def test_annex_m_adjustment_to_the_same_levels_is_the_measurement() -> None:
    """Adjusting to the condition already measured returns the measurement.

    Step 2 and step 3 are then the same correction undone and reapplied, so
    the matrix has to come back unchanged: the round trip is what says the
    two directions use one model rather than two that nearly agree.
    """
    measured = _annex_m_matrix(IEC60268_16_ANNEX_M_MEASURED_MTF)
    unmoved = speech.sti_adjusted_for_levels(
        measured,
        measured_level=IEC60268_16_ANNEX_M_MEASURED_LEVEL,
        measured_ambient=IEC60268_16_ANNEX_M_MEASURED_AMBIENT,
        operational_level=IEC60268_16_ANNEX_M_MEASURED_LEVEL,
        operational_ambient=IEC60268_16_ANNEX_M_MEASURED_AMBIENT,
    )
    np.testing.assert_allclose(unmoved.mtf, measured, atol=1e-12, rtol=0.0)


def test_annex_m_adjustment_from_a_measured_result() -> None:
    """``STIResult.adjusted_for_levels`` takes the measured condition itself.

    Closing the annex's own loop: the printed step 2 matrix put back through
    the measurement condition is the printed step 1 matrix, and a result of
    that measurement carries the
    two spectra the adjustment needs. Moving it to the operational condition
    then needs only that condition, and lands on the printed step 3 matrix
    and STI. Passing all four spectra to the module function must give the
    same number.
    """
    measured = _sti_from_mtf(
        _annex_m_matrix(IEC60268_16_ANNEX_M_SOURCE_MTF),
        level=np.asarray(IEC60268_16_ANNEX_M_MEASURED_LEVEL, dtype=float),
        ambient=np.asarray(IEC60268_16_ANNEX_M_MEASURED_AMBIENT, dtype=float),
    )
    np.testing.assert_allclose(
        measured.mtf,
        _annex_m_matrix(IEC60268_16_ANNEX_M_MEASURED_MTF),
        atol=_ANNEX_M_MTF_TOL,
        rtol=0.0,
    )
    occupied = measured.adjusted_for_levels(
        operational_level=IEC60268_16_ANNEX_M_OPERATIONAL_LEVEL,
        operational_ambient=IEC60268_16_ANNEX_M_OPERATIONAL_AMBIENT,
    )
    np.testing.assert_allclose(
        occupied.mtf,
        _annex_m_matrix(IEC60268_16_ANNEX_M_OPERATIONAL_MTF),
        atol=_ANNEX_M_MTF_TOL,
        rtol=0.0,
    )
    assert round(occupied.sti, 2) == IEC60268_16_ANNEX_M_STI
    by_hand = speech.sti_adjusted_for_levels(
        measured.mtf,
        measured_level=IEC60268_16_ANNEX_M_MEASURED_LEVEL,
        measured_ambient=IEC60268_16_ANNEX_M_MEASURED_AMBIENT,
        operational_level=IEC60268_16_ANNEX_M_OPERATIONAL_LEVEL,
        operational_ambient=IEC60268_16_ANNEX_M_OPERATIONAL_AMBIENT,
    )
    assert occupied.sti == pytest.approx(by_hand.sti, abs=1e-12)


def test_annex_m_needs_the_levels_the_measurement_was_corrected_with() -> None:
    """A result without band levels has nothing to undo.

    Its matrix never had the masking, threshold and noise of a measurement
    condition applied, so dividing them out would remove what was never
    there and hand back a lower STI with no warning.
    """
    result = _sti_from_mtf(_annex_m_matrix(IEC60268_16_ANNEX_M_SOURCE_MTF))
    assert result.band_levels is None
    with pytest.raises(ValueError, match="needs the speech band levels"):
        result.adjusted_for_levels(
            operational_level=IEC60268_16_ANNEX_M_OPERATIONAL_LEVEL
        )


@pytest.mark.parametrize(
    "name",
    ["measured_level", "measured_ambient", "operational_level", "operational_ambient"],
)
def test_annex_m_adjustment_rejects_a_level_vector_of_the_wrong_length(
    name: str,
) -> None:
    """Each of the four spectra is seven octave bands, named when it is not."""
    measured = _annex_m_matrix(IEC60268_16_ANNEX_M_MEASURED_MTF)
    spectra: dict[str, Sequence[float]] = {
        "measured_level": IEC60268_16_ANNEX_M_MEASURED_LEVEL,
        "operational_level": IEC60268_16_ANNEX_M_OPERATIONAL_LEVEL,
        name: [70.0, 70.0],
    }
    with pytest.raises(
        ValueError, match=rf"'{name}' must contain exactly 7 octave-band"
    ):
        speech.sti_adjusted_for_levels(measured, **spectra)


def test_an_sti_result_refuses_noise_levels_without_speech_levels() -> None:
    """Half of the level pair is not a condition the adjustment can undo.

    Noise levels only enter the chain against a speech spectrum; a result
    holding them alone would let the adjustment divide out a correction that
    was never applied.
    """
    import dataclasses

    result = _sti_from_mtf(_annex_m_matrix(IEC60268_16_ANNEX_M_SOURCE_MTF))
    with pytest.raises(ValueError, match="'ambient_levels' needs the speech"):
        dataclasses.replace(result, ambient_levels=np.full(7, 40.0))


# --------------------------------------------------------------------------
# Per-band quantities that do not run over the band axis
# --------------------------------------------------------------------------
def test_an_sti_result_refuses_an_mti_off_the_band_axis() -> None:
    """The fiche tabulates ``mti`` beside the modulation matrix and the levels.

    All three run over the same seven octave bands, so a per-band column of
    another length is a table whose rows no longer describe one band each.
    """
    import dataclasses

    result = STIResult(
        sti=0.75,
        mti=np.full(7, 0.75),
        mtf=np.full((7, 2), 0.75),
        band_levels=np.full(7, 60.0),
        rating="B",
    )
    with pytest.raises(ValueError, match=r"'mti' \(6\)"):
        dataclasses.replace(result, mti=result.mti[:-1])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_an_sti_result_refuses_a_non_finite_index(bad: float) -> None:
    """No measurement can return an STI that is not a number.

    Every path runs through the modulation-transfer validation, which refuses
    a non-finite matrix, and the chain from there is arithmetic on values
    clipped into [0, 1]. One built by hand used to crash the boxed statement
    of the fiche as "cannot convert float NaN to integer", from inside the
    display rounder, naming neither the field nor the result type.
    """
    import dataclasses

    result = STIResult(
        sti=0.75,
        mti=np.full(7, 0.75),
        mtf=np.full((7, 2), 0.75),
        band_levels=np.full(7, 60.0),
        rating="B",
    )
    with pytest.raises(ValueError, match="'sti' must be finite"):
        dataclasses.replace(result, sti=bad)


def test_an_sti_result_refuses_a_bare_number_for_a_band_column() -> None:
    """One number is not seven bands, however few axes it has.

    A field with no axis at all is exempt from the band pin, for the entry
    points elsewhere in the library that take a single frequency and answer
    with bare numbers throughout. An STI result is never one of those: its
    columns run over the seven octave bands, so a lone number among them is
    left to the band count, which reports it by name.
    """
    import dataclasses

    ir = np.zeros(FS // 2)
    ir[100] = 1.0
    result = speech.sti_from_impulse_response(ir, FS)
    with pytest.raises(ValueError, match="'mti' must have one axis; got 0"):
        dataclasses.replace(result, mti=0.75)

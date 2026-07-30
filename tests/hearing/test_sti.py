#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Speech Transmission Index per IEC 60268-16:2020 (Edition 5).

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

import numpy as np
import pytest
from reference_data import (
    IEC60268_16_ANNEX_M_AMBIENT,
    IEC60268_16_ANNEX_M_LEVEL,
    IEC60268_16_ANNEX_M_MTF,
    IEC60268_16_ANNEX_M_MTI,
    IEC60268_16_ANNEX_M_STI,
)

from phonometry import STIResult, sti_from_impulse_response, stipa, stipa_signal
from phonometry.hearing.sti import (
    _ALPHA_MALE,
    _BETA_MALE,
    _MOD_FREQS,
    _NUM_BANDS,
    _RATING_EDGES,
    _RATING_LETTERS,
    STIWarning,
    _masking_amdb,
    _rating,
    _sti_from_mtf,
)

FS = 48000


def _uniform_mtf(m: float) -> np.ndarray:
    return np.full((_NUM_BANDS, _MOD_FREQS.size), m)


def _decay_ir(t60: float, fs: int, seed: int = 0) -> np.ndarray:
    """Noise-carrier impulse response with exponential energy decay:
    p(t) ~ e^(-13.8 t / T60), i.e. -60 dB at t = T60."""
    rng = np.random.default_rng(seed)
    n = int(2.0 * t60 * fs)
    t = np.arange(n) / fs
    return rng.standard_normal(n) * np.exp(-3.0 * np.log(10.0) * t / t60)


def _analytic_decay_sti(t60: float) -> float:
    """Expected STI from the closed-form Schroeder MTF of an exponential
    decay, identical in all bands: m(F) = 1/sqrt(1 + (2 pi F T/13,8)^2)."""
    m = 1.0 / np.sqrt(1.0 + (2.0 * np.pi * _MOD_FREQS * t60 / 13.8) ** 2)
    return _sti_from_mtf(np.tile(m, (_NUM_BANDS, 1))).sti


# ---------------------------------------------------------------------------
# Final formula: weighting/redundancy factors (Ed.5 A.2.2 verification test)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bands, expected",
    [
        ((0, 1), 0.127),  # 125 + 250 Hz
        ((1, 2), 0.279),  # 250 + 500 Hz
        ((2, 3), 0.398),  # 500 + 1000 Hz
        ((3, 4), 0.531),  # 1000 + 2000 Hz
        ((4, 5), 0.486),  # 2000 + 4000 Hz
        ((5, 6), 0.302),  # 4000 + 8000 Hz
    ],
)
def test_weighting_factor_pairs(bands, expected):
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
    "m, expected_sti",
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
def test_m_to_sti_mapping(m, expected_sti):
    # Ed.5 A.3.1.2 pairs: uniform m across all bands and modulation
    # frequencies maps to the given STI (tol +/-0,01).
    result = _sti_from_mtf(_uniform_mtf(m))
    assert result.sti == pytest.approx(expected_sti, abs=0.01)


def test_alpha_beta_artifact_truncated_to_one():
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

def test_delta_impulse_response_is_perfect_transmission():
    ir = np.zeros(FS // 2)
    ir[100] = 1.0
    result = sti_from_impulse_response(ir, FS)
    # The residual error is the analysis filter bank's own MTF; the
    # standard allows < 0,01 STI systematic error (Ed.4 A.5.1.2).
    assert result.sti == pytest.approx(1.0, abs=0.01)
    assert result.rating == "A+"
    assert result.mtf.shape == (7, 14)
    assert result.mti.shape == (7,)
    assert result.band_levels is None
    assert isinstance(result, STIResult)


def test_exponential_decay_matches_analytic_schroeder_mtf():
    fs = 24000
    stis = []
    for t60 in (0.5, 1.0, 2.0, 4.0):
        got = sti_from_impulse_response(_decay_ir(t60, fs), fs).sti
        assert got == pytest.approx(_analytic_decay_sti(t60), abs=0.01)
        stis.append(got)
    # Monotonic: longer reverberation always degrades intelligibility.
    assert all(a > b for a, b in pairwise(stis))


def test_snr_degradation_on_impulse_response():
    fs = 24000
    ir = _decay_ir(1.0, fs)
    plain = sti_from_impulse_response(ir, fs)
    high_snr = sti_from_impulse_response(ir, fs, snr=30.0)
    zero_snr = sti_from_impulse_response(ir, fs, snr=0.0)
    # +30 dB SNR: m factor 1/(1+10^-3) = 0,999 -> no visible change.
    assert high_snr.sti == pytest.approx(plain.sti, abs=0.01)
    # 0 dB SNR: every m halved -> STI drops markedly.
    np.testing.assert_allclose(zero_snr.mtf, plain.mtf / 2.0, rtol=1e-12)
    assert zero_snr.sti < plain.sti - 0.1
    # Per-band SNR vector is accepted and equals the scalar case.
    vec = sti_from_impulse_response(ir, fs, snr=np.zeros(7))
    assert vec.sti == pytest.approx(zero_snr.sti, abs=1e-12)


def test_level_corrections_reduce_sti():
    fs = 24000
    ir = _decay_ir(1.0, fs)
    plain = sti_from_impulse_response(ir, fs)
    # Comfortable speech levels: masking/threshold effects are small.
    comfortable = sti_from_impulse_response(ir, fs, level=[62, 62, 59, 53, 47, 41, 35])
    # Very quiet speech: the absolute reception threshold dominates.
    quiet = sti_from_impulse_response(ir, fs, level=[20, 20, 17, 11, 5, -1, -7])
    assert comfortable.sti <= plain.sti
    assert quiet.sti < comfortable.sti - 0.05
    assert comfortable.band_levels is not None
    # Ambient noise at the listener degrades further.
    noisy = sti_from_impulse_response(
        ir, fs, level=[62, 62, 59, 53, 47, 41, 35], ambient=[55] * 7
    )
    assert noisy.sti < comfortable.sti


# ---------------------------------------------------------------------------
# Auditory masking function (Ed.5 Table A.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "level, expected",
    [(60.0, -35.0), (65.0, -29.9), (80.0, -19.8), (100.0, -10.0)],
)
def test_masking_amdb_control_points(level, expected):
    assert _masking_amdb(level) == pytest.approx(expected, abs=1e-9)


def test_masking_amdb_is_vectorized_and_continuous():
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
    return stipa_signal(FS, seconds=18.0, seed=1234)


def test_stipa_loopback_ideal_channel(stipa_18s_seed1234: np.ndarray):
    x = stipa_18s_seed1234
    result = stipa(x, FS)
    # Ideal loopback recovers STI 0.998 and min MTF 0.945 at 18 s; lock those
    # in (was >= 0.95 / > 0.9, several x looser than the achieved accuracy).
    assert result.sti >= 0.99
    assert result.rating == "A+"
    assert result.mtf.shape == (7, 2)
    assert np.all(result.mtf > 0.93)


def test_stipa_short_recording_warns(stipa_18s_seed1234: np.ndarray):
    """A recording shorter than the recommended 15 s biases the recovered
    modulation depths (and STI) low; stipa should warn (IEC 60268-16 STIPA
    practice recommends 15 s to 25 s)."""
    short = stipa_signal(FS, seconds=5.0, seed=1234)
    with pytest.warns(UserWarning, match="15"):
        stipa(short, FS)
    # No warning at the recommended length.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stipa(stipa_18s_seed1234, FS)


def test_stipa_with_noise_is_monotonic(stipa_18s_seed1234: np.ndarray):
    x = stipa_18s_seed1234
    rng = np.random.default_rng(7)
    rms = float(np.sqrt(np.mean(x**2)))
    stis = []
    for snr_db in (30.0, 10.0, 0.0):
        noise = rng.standard_normal(x.size) * rms * 10.0 ** (-snr_db / 20.0)
        stis.append(stipa(x + noise, FS).sti)
    assert all(a > b for a, b in pairwise(stis))
    assert stis[-1] < 0.7  # 0 dB broadband SNR is clearly degraded


def test_stipa_reference_normalization():
    x = stipa_signal(FS, seconds=18.0, seed=99)
    with_nominal = stipa(x, FS)
    with_reference = stipa(0.25 * x, FS, reference=x)
    # Loop-back against the emitted signal itself: m = 1 in every band
    # (gain does not affect modulation depths).
    assert with_reference.sti == pytest.approx(1.0, abs=1e-6)
    assert with_nominal.sti == pytest.approx(with_reference.sti, abs=0.05)


def test_stipa_signal_properties():
    seconds = 18.0
    x = stipa_signal(FS, seconds=seconds, seed=0)
    assert x.shape == (int(seconds * FS),)
    # Default normalization: RMS = 0,1 digital units.
    assert float(np.sqrt(np.mean(x**2))) == pytest.approx(0.1, rel=1e-9)
    # Crest factor of the STIPA signal: expected around 12-14 dB (A.4).
    crest_db = 20.0 * np.log10(np.max(np.abs(x)) / np.sqrt(np.mean(x**2)))
    assert 9.0 < crest_db < 16.0
    # Calibrated output: overall level in dB re 20 uPa.
    x_cal = stipa_signal(FS, seconds=6.0, level_db=74.0, seed=0)
    level = 20.0 * np.log10(np.sqrt(np.mean(x_cal**2)) / 2e-5)
    assert level == pytest.approx(74.0, abs=1e-9)
    # Reproducible for a fixed seed.
    np.testing.assert_array_equal(x, stipa_signal(FS, seconds=seconds, seed=0))


# ---------------------------------------------------------------------------
# Annex F qualification rating
# ---------------------------------------------------------------------------

def test_rating_letters_from_band_edges():
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

def test_invalid_inputs_raise():
    ir = np.zeros(FS // 4)
    ir[10] = 1.0

    with pytest.raises(ValueError, match="1D"):
        sti_from_impulse_response(np.zeros((2, 1000)), FS)
    with pytest.raises(ValueError, match="positive"):
        sti_from_impulse_response(ir, -1)
    with pytest.raises(ValueError, match="8 kHz octave band"):
        sti_from_impulse_response(ir, 16000)
    with pytest.raises(ValueError, match="silent"):
        sti_from_impulse_response(np.zeros(FS // 4), FS)
    with pytest.raises(ValueError, match="7 octave-band values"):
        sti_from_impulse_response(ir, FS, level=[60.0, 60.0, 60.0])
    with pytest.raises(ValueError, match="scalar or a vector"):
        sti_from_impulse_response(ir, FS, snr=[10.0, 10.0])
    with pytest.raises(ValueError, match="requires the speech octave-band levels"):
        sti_from_impulse_response(ir, FS, ambient=[40.0] * 7)
    with pytest.raises(ValueError, match="not both"):
        sti_from_impulse_response(ir, FS, snr=10.0, level=[60.0] * 7, ambient=[40.0] * 7)

    with pytest.raises(ValueError, match="1D"):
        stipa(np.zeros((2, FS)), FS)
    with pytest.raises(ValueError, match="too short"):  # noqa: SIM117 - simplefilter must run inside catch_warnings before the call
        # The 0.5 s clip also triggers the (correct) sub-15 s STIPA warning;
        # silence it so the test output stays clean while asserting the error.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            stipa(stipa_signal(FS, seconds=18.0, seed=3)[: FS // 2], FS)
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
        res_tone = stipa(np.sin(2 * np.pi * 1000.0 * t), FS)
    assert any("No energy in octave band" in str(w.message) for w in tone_warnings)
    # The 1 kHz band carries the tone (unmodulated: m ~ 0); at least one
    # dead band integrates to non-positive envelope energy and is pinned
    # to exactly m = 0 instead of raising.
    assert np.any(res_tone.mtf == 0.0)
    assert np.all(res_tone.mtf[3] < 0.01)

    with pytest.raises(ValueError, match="positive"):
        stipa_signal(FS, seconds=0.0)
    with pytest.raises(ValueError, match="fs"):
        stipa_signal(8000)

    with pytest.raises(ValueError, match="shape"):
        _sti_from_mtf(np.full((3, 14), 0.5))
    with pytest.raises(ValueError, match="finite"):
        _sti_from_mtf(_uniform_mtf(-0.1))


def test_mtf_above_1_3_warns_and_truncates():
    with pytest.warns(UserWarning, match="1.3"):
        result = _sti_from_mtf(_uniform_mtf(1.4))
    assert result.sti == 1.0


# ---------------------------------------------------------------------------
# Ed.5 C.3.2 - modulation-depth test, direct method (end-to-end oracle)
# ---------------------------------------------------------------------------

# Expected STI for the Ed.5 Formula (C.1) test signal at modulation scale
# m = 0,0, 0,1, ... 1,0 (Ed.5 Table C.2 staircase), tolerance +/-0,05.
_C32_STI_STAIRCASE = [0.0, 0.18, 0.30, 0.38, 0.44, 0.50, 0.56, 0.62,
                      0.70, 0.82, 1.0]


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
    for fc, fa, fb, level in zip(centers, f1, f2, levels_db):
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
    published STI staircase for the Formula (C.1) signal within +/-0,05."""
    m = i / 10.0
    x = _c32_signal(m, FS, seconds=16.0)
    res = stipa(x, FS)
    assert res.sti == pytest.approx(_C32_STI_STAIRCASE[i], abs=0.05)


def test_annex_m_full_sti_worked_example() -> None:
    """Reproduce the IEC 60268-16 Annex M full-STI worked example.

    The annex prints the adjusted MTF matrix before noise, masking and
    threshold are applied, together with the operational speech and ambient
    noise spectra; combining them must give back its printed MTI row (step 4c)
    and STI. This is an independent oracle for the whole A.5.3 to A.5.6 chain:
    auditory masking, the reception threshold, the SNR clamp, the per-band MTI
    average and the alpha/beta band weighting.
    """
    mtf = np.asarray(IEC60268_16_ANNEX_M_MTF, dtype=float).T  # bands x mod freqs
    assert mtf.shape == (_NUM_BANDS, _MOD_FREQS.size)
    result = _sti_from_mtf(
        mtf,
        level=np.asarray(IEC60268_16_ANNEX_M_LEVEL, dtype=float),
        ambient=np.asarray(IEC60268_16_ANNEX_M_AMBIENT, dtype=float),
    )
    # Exact at the annex's own two decimals: a tolerance of 0,01 on top of the
    # rounding would accept the neighbouring cent (0,75 for a printed 0,76).
    np.testing.assert_array_equal(
        np.round(result.mti, 2),
        np.asarray(IEC60268_16_ANNEX_M_MTI, dtype=float),
    )
    assert result.sti == pytest.approx(IEC60268_16_ANNEX_M_STI, abs=0.005)
    assert round(result.sti, 2) == IEC60268_16_ANNEX_M_STI

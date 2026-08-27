#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the parametric-EQ biquads (RBJ Audio EQ Cookbook).

Clean-room oracles: closed-form consequences of the cookbook's own
formulas (Bristow-Johnson, *Audio EQ Cookbook*, W3C Working Group Note,
https://www.w3.org/TR/audio-eq-cookbook/), independent of the module:

* One complete coefficient set is hand-computed digit by digit in
  ``test_peaking_hand_computed_coefficients`` and pinned.
* Exact response anchors: the peaking gain at ``f0`` is exactly ``G`` dB
  and exactly 0 dB at DC and Nyquist; shelves reach exactly ``G`` dB at
  the shelved end and 0 dB at the other; the constant-0dB band-pass peaks
  at exactly 0 dB; low/high-pass read exactly ``Q`` at ``f0``; the notch
  transmits nothing at ``f0``; the all-pass has unit magnitude everywhere.
  (Derivations in the individual tests.)
* The cookbook's Q definition: the midpoint-gain (``G/2`` dB) edge
  frequencies of the digital peaking filter satisfy the closed form
  ``cos(w) - cos(w0) = ±alpha·sin(w)``, and the BW and slope
  parameterizations reduce to the stated Q relations coefficient by
  coefficient.
* Corroboration (not the oracle): ``scipy.signal.iirpeak`` matches the
  constant-0dB band-pass and ``scipy.signal.iirnotch`` the notch;
  ``scipy.signal.butter`` order 2 matches low/high-pass at
  ``Q = 1/sqrt(2)``.
"""

from __future__ import annotations

import dataclasses

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import optimize, signal

import phonometry as ph

FS = 48000.0

_GAINLESS_TYPES = (
    "lowpass",
    "highpass",
    "bandpass",
    "bandpass_skirt",
    "notch",
    "allpass",
)
_ALL_TYPES = ("peaking", "lowshelf", "highshelf") + _GAINLESS_TYPES


def _gain_db_at(sos: np.ndarray, freqs: list[float], fs: float = FS) -> np.ndarray:
    """Cascade gain in dB at the given frequencies (machine-exact eval)."""
    _, h = signal.sosfreqz(sos, worN=np.asarray(freqs, dtype=np.float64), fs=fs)
    return 20 * np.log10(np.abs(h))


# ---------------------------------------------------------------------------
# Hand-computed coefficient example (the cookbook recipe, worked by hand)
# ---------------------------------------------------------------------------


def test_peaking_hand_computed_coefficients() -> None:
    """Peaking EQ at fs=48 kHz, f0=1 kHz, Q=0.707, G=+6 dB, worked by hand.

    Cookbook intermediate variables (30-digit arithmetic, rounded to 17
    significant digits):

        w0     = 2*pi*1000/48000 = pi/24     = 0.13089969389957472
        cos w0                               = 0.99144486137381041
        sin w0                               = 0.13052619222005159
        A      = 10^(6/40) = 10^0.15         = 1.4125375446227543
        alpha  = sin(w0) / (2*0.707)         = 0.092309895488013855
        alpha*A                              = 0.13039119311702216
        alpha/A                              = 0.065350401367679762

    Cookbook peaking coefficients (unnormalized):

        b0 = 1 + alpha*A  = 1.1303911931170222
        b1 = -2*cos w0    = -1.9828897227476208
        b2 = 1 - alpha*A  = 0.86960880688297784
        a0 = 1 + alpha/A  = 1.0653504013676798
        a1 = -2*cos w0    = -1.9828897227476208
        a2 = 1 - alpha/A  = 0.93464959863232024

    Normalized by a0 (the SOS row):

        b0/a0 = 1.0610510792184844
        b1/a0 = -1.8612559024730444
        b2/a0 = 0.81626552706657641
        a1/a0 = -1.8612559024730444
        a2/a0 = 0.87731660628506083
    """
    eq = ph.filters.ParametricEQ(
        FS, ph.filters.EQSection("peaking", 1000.0, gain_db=6.0, q=0.707)
    )
    expected = np.array(
        [
            1.0610510792184844,
            -1.8612559024730444,
            0.81626552706657641,
            1.0,
            -1.8612559024730444,
            0.87731660628506083,
        ]
    )
    assert eq.sos.shape == (1, 6)
    np.testing.assert_allclose(eq.sos[0], expected, rtol=0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Exact response anchors (closed forms of the cookbook coefficients)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gain_db", [-12.0, -6.0, 3.0, 6.0, 12.0])
@pytest.mark.parametrize("f0", [100.0, 1000.0, 8000.0])
def test_peaking_gain_exact_at_f0_and_unity_at_ends(f0: float, gain_db: float) -> None:
    """Peaking: exactly G dB at f0, exactly 0 dB at DC and Nyquist.

    At w0 the numerator and denominator reduce to 2j*alpha*A*sin(w0) and
    2j*(alpha/A)*sin(w0), so H(e^{jw0}) = A^2 = 10^(G/20) exactly. At z=1
    and z=-1 the b and a sums are both 2*(1 -/+ cos w0), so |H| = 1.
    """
    eq = ph.filters.ParametricEQ(
        FS, ph.filters.EQSection("peaking", f0, gain_db=gain_db, q=2.0)
    )
    at_f0, at_dc, at_nyq = _gain_db_at(eq.sos, [f0, 0.0, FS / 2])
    assert at_f0 == pytest.approx(gain_db, abs=1e-10)
    assert at_dc == pytest.approx(0.0, abs=1e-10)
    assert at_nyq == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("gain_db", [-9.0, 6.0])
def test_shelves_exact_gain_at_ends(gain_db: float) -> None:
    """Shelves: exactly G dB at the shelved end, exactly 0 dB at the other.

    For the low shelf the coefficient sums give H(1) = 4A^2(1-cos w0) /
    (4(1-cos w0)) = A^2 and H(-1) = 4A(1+cos w0) / (4A(1+cos w0)) = 1;
    the high shelf mirrors them.
    """
    low = ph.filters.ParametricEQ(
        FS, ph.filters.EQSection("lowshelf", 500.0, gain_db=gain_db)
    )
    high = ph.filters.ParametricEQ(
        FS, ph.filters.EQSection("highshelf", 500.0, gain_db=gain_db)
    )
    low_dc, low_nyq = _gain_db_at(low.sos, [0.0, FS / 2])
    high_dc, high_nyq = _gain_db_at(high.sos, [0.0, FS / 2])
    assert low_dc == pytest.approx(gain_db, abs=1e-10)
    assert low_nyq == pytest.approx(0.0, abs=1e-10)
    assert high_dc == pytest.approx(0.0, abs=1e-10)
    assert high_nyq == pytest.approx(gain_db, abs=1e-10)


def test_shelf_midpoint_gain_at_f0() -> None:
    """Both shelves pass exactly G/2 dB at f0 (the midpoint-gain frequency).

    At w0 the shelf transfer function reduces to a ratio whose magnitude is
    A (the square root of the shelf gain A^2), for every S.
    """
    for filter_type in ("lowshelf", "highshelf"):
        eq = ph.filters.ParametricEQ(
            FS,
            ph.filters.EQSection(filter_type, 1000.0, gain_db=8.0, slope=1.0),
        )
        (at_f0,) = _gain_db_at(eq.sos, [1000.0])
        assert at_f0 == pytest.approx(4.0, abs=1e-10)


@pytest.mark.parametrize("q", [0.5, 1.0 / np.sqrt(2.0), 2.0])
def test_lowpass_highpass_gain_q_at_f0(q: float) -> None:
    """LPF/HPF: |H(f0)| = Q exactly, and the correct 0 dB / blocked ends.

    The analog prototype H(s) = 1/(s^2 + s/Q + 1) reads |H(j)| = Q at the
    corner; the cookbook's bilinear design prewarps so the digital response
    keeps that value exactly at f0.
    """
    lp = ph.filters.ParametricEQ(FS, ph.filters.EQSection("lowpass", 1000.0, q=q))
    hp = ph.filters.ParametricEQ(FS, ph.filters.EQSection("highpass", 1000.0, q=q))
    q_db = 20 * np.log10(q)
    (lp_f0,) = _gain_db_at(lp.sos, [1000.0])
    (hp_f0,) = _gain_db_at(hp.sos, [1000.0])
    assert lp_f0 == pytest.approx(q_db, abs=1e-10)
    assert hp_f0 == pytest.approx(q_db, abs=1e-10)
    lp_dc, lp_nyq = _gain_db_at(lp.sos, [0.0, FS / 2])
    hp_dc, hp_nyq = _gain_db_at(hp.sos, [0.0, FS / 2])
    assert lp_dc == pytest.approx(0.0, abs=1e-10)
    assert hp_nyq == pytest.approx(0.0, abs=1e-10)
    assert lp_nyq < -250.0  # double zero at Nyquist
    assert hp_dc < -250.0  # double zero at DC


def test_bandpass_notch_allpass_anchors() -> None:
    """Band-pass peaks at 0 dB (skirt variant at Q), notch nulls, all-pass is flat."""
    (bp_f0,) = _gain_db_at(
        ph.filters.ParametricEQ(
            FS, ph.filters.EQSection("bandpass", 1000.0, q=3.0)
        ).sos,
        [1000.0],
    )
    assert bp_f0 == pytest.approx(0.0, abs=1e-10)

    (skirt_f0,) = _gain_db_at(
        ph.filters.ParametricEQ(
            FS, ph.filters.EQSection("bandpass_skirt", 1000.0, q=3.0)
        ).sos,
        [1000.0],
    )
    assert skirt_f0 == pytest.approx(20 * np.log10(3.0), abs=1e-10)

    notch = ph.filters.ParametricEQ(FS, ph.filters.EQSection("notch", 1000.0, q=5.0))
    (notch_f0,) = _gain_db_at(notch.sos, [1000.0])
    assert notch_f0 < -250.0
    notch_dc, notch_nyq = _gain_db_at(notch.sos, [0.0, FS / 2])
    assert notch_dc == pytest.approx(0.0, abs=1e-10)
    assert notch_nyq == pytest.approx(0.0, abs=1e-10)

    ap = ph.filters.ParametricEQ(FS, ph.filters.EQSection("allpass", 1000.0, q=0.9))
    freqs = np.linspace(1.0, FS / 2, 512)
    _, h = signal.sosfreqz(ap.sos, worN=freqs, fs=FS)
    np.testing.assert_allclose(np.abs(h), 1.0, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# The cookbook's Q / BW / S definitions
# ---------------------------------------------------------------------------


def test_peaking_half_gain_edges_match_q_definition() -> None:
    """The G/2 dB edges satisfy cos(w) - cos(w0) = ±alpha*sin(w).

    Factoring e^{-jw} from numerator and denominator of the peaking
    H(e^{jw}) leaves H = [(cos w - cos w0) + j*alpha*A*sin w] /
    [(cos w - cos w0) + j*(alpha/A)*sin w]. Setting |H|^2 = A^2 (the
    midpoint gain the cookbook defines the bandwidth between) forces
    (cos w - cos w0)^2 = (alpha*sin w)^2 for any A != 1: the closed form
    for both edges, verified here on edges located numerically from the
    response alone.
    """
    f0, q, gain_db = 1000.0, 1.5, 6.0
    eq = ph.filters.ParametricEQ(
        FS, ph.filters.EQSection("peaking", f0, gain_db=gain_db, q=q)
    )
    w0 = 2 * np.pi * f0 / FS
    alpha = np.sin(w0) / (2 * q)
    half_gain = gain_db / 2

    def offset(f: float) -> float:
        return float(_gain_db_at(eq.sos, [f])[0]) - half_gain

    f_lower = optimize.brentq(offset, 100.0, f0 - 1e-6)
    f_upper = optimize.brentq(offset, f0 + 1e-6, 10000.0)
    for f_edge, sign in ((f_lower, 1.0), (f_upper, -1.0)):
        w = 2 * np.pi * f_edge / FS
        assert np.cos(w) - np.cos(w0) == pytest.approx(
            sign * alpha * np.sin(w), abs=1e-9
        )


def test_peaking_half_gain_bandwidth_matches_q_in_analog_limit() -> None:
    """Far below Nyquist the G/2 dB bandwidth approaches f0/Q."""
    f0, q = 200.0, 2.0  # f0/fs ~ 0.4 %: negligible warping
    eq = ph.filters.ParametricEQ(
        FS, ph.filters.EQSection("peaking", f0, gain_db=6.0, q=q)
    )

    def offset(f: float) -> float:
        return float(_gain_db_at(eq.sos, [f])[0]) - 3.0

    f_lower = optimize.brentq(offset, 20.0, f0 - 1e-6)
    f_upper = optimize.brentq(offset, f0 + 1e-6, 2000.0)
    # The residual is the bilinear warping, O((pi*f0/fs)^2) ~ 2e-4 here.
    assert f_upper - f_lower == pytest.approx(f0 / q, rel=5e-4)


def test_bw_parameterization_reduces_to_q_relation() -> None:
    """BW gives the coefficients of Q = 1/(2*sinh(ln2/2 * BW * w0/sin(w0))).

    The cookbook states the digital-domain relation
    1/Q = 2*sinh(ln(2)/2 * BW * w0/sin(w0)); designing from BW must
    therefore reproduce the Q design exactly, coefficient by coefficient.
    """
    f0, bw = 1000.0, 1.0
    w0 = 2 * np.pi * f0 / FS
    q = 1.0 / (2 * np.sinh(np.log(2.0) / 2 * bw * w0 / np.sin(w0)))
    for filter_type in ("peaking", "bandpass", "notch"):
        gain = 6.0 if filter_type == "peaking" else 0.0
        from_bw = ph.filters.ParametricEQ(
            FS, ph.filters.EQSection(filter_type, f0, gain_db=gain, bw=bw)
        )
        from_q = ph.filters.ParametricEQ(
            FS, ph.filters.EQSection(filter_type, f0, gain_db=gain, q=q)
        )
        np.testing.assert_allclose(from_bw.sos, from_q.sos, rtol=0.0, atol=1e-15)


def test_slope_parameterization_reduces_to_q_relation() -> None:
    """S gives the coefficients of 1/Q = sqrt((A + 1/A)(1/S - 1) + 2)."""
    f0, gain_db, slope = 800.0, 9.0, 0.8
    big_a = 10.0 ** (gain_db / 40.0)
    q = 1.0 / np.sqrt((big_a + 1 / big_a) * (1 / slope - 1) + 2)
    for filter_type in ("lowshelf", "highshelf"):
        from_s = ph.filters.ParametricEQ(
            FS,
            ph.filters.EQSection(filter_type, f0, gain_db=gain_db, slope=slope),
        )
        from_q = ph.filters.ParametricEQ(
            FS, ph.filters.EQSection(filter_type, f0, gain_db=gain_db, q=q)
        )
        np.testing.assert_allclose(from_s.sos, from_q.sos, rtol=0.0, atol=1e-15)


def test_default_q_is_butterworth() -> None:
    """With no q/bw/slope the section designs at Q = 1/sqrt(2) (-3.01 dB)."""
    eq = ph.filters.ParametricEQ(FS, ph.filters.EQSection("lowpass", 1000.0))
    explicit = ph.filters.ParametricEQ(
        FS, ph.filters.EQSection("lowpass", 1000.0, q=1.0 / np.sqrt(2.0))
    )
    np.testing.assert_allclose(eq.sos, explicit.sos, rtol=0.0, atol=1e-16)
    (at_f0,) = _gain_db_at(eq.sos, [1000.0])
    assert at_f0 == pytest.approx(20 * np.log10(1 / np.sqrt(2)), abs=1e-10)


# ---------------------------------------------------------------------------
# Corroboration against scipy (independent implementations, not the oracle)
# ---------------------------------------------------------------------------


def test_bandpass_matches_scipy_iirpeak() -> None:
    """Constant-0dB band-pass magnitude matches scipy.signal.iirpeak.

    scipy maps Q to the -3 dB bandwidth through a tan() prewarp where the
    cookbook uses sin(w0)/(2Q) directly, so the two designs coincide only
    up to the warping residual: agreement is corroborative (within
    0.03 dB across the band here), not coefficient-exact.
    """
    f0, q = 1000.0, 5.0
    eq = ph.filters.ParametricEQ(FS, ph.filters.EQSection("bandpass", f0, q=q))
    b, a = signal.iirpeak(f0, q, fs=FS)
    freqs = np.logspace(1, np.log10(FS / 2 * 0.99), 512)
    _, h_eq = signal.sosfreqz(eq.sos, worN=freqs, fs=FS)
    _, h_sp = signal.freqz(b, a, worN=freqs, fs=FS)
    np.testing.assert_allclose(np.abs(h_eq), np.abs(h_sp), rtol=0.0, atol=5e-3)


def test_notch_matches_scipy_iirnotch() -> None:
    """Notch magnitude matches scipy.signal.iirnotch (same caveat as iirpeak)."""
    f0, q = 1000.0, 5.0
    eq = ph.filters.ParametricEQ(FS, ph.filters.EQSection("notch", f0, q=q))
    b, a = signal.iirnotch(f0, q, fs=FS)
    freqs = np.logspace(1, np.log10(FS / 2 * 0.99), 512)
    _, h_eq = signal.sosfreqz(eq.sos, worN=freqs, fs=FS)
    _, h_sp = signal.freqz(b, a, worN=freqs, fs=FS)
    np.testing.assert_allclose(np.abs(h_eq), np.abs(h_sp), rtol=0.0, atol=5e-3)


def test_butterworth_alignment_matches_scipy_butter() -> None:
    """LPF/HPF at Q = 1/sqrt(2) match a 2nd-order scipy Butterworth."""
    f0 = 1000.0
    freqs = np.logspace(1, np.log10(FS / 2 * 0.99), 512)
    for filter_type, btype in (("lowpass", "low"), ("highpass", "high")):
        eq = ph.filters.ParametricEQ(
            FS, ph.filters.EQSection(filter_type, f0, q=1.0 / np.sqrt(2.0))
        )
        sos_ref = signal.butter(2, f0, btype=btype, output="sos", fs=FS)
        _, h_eq = signal.sosfreqz(eq.sos, worN=freqs, fs=FS)
        _, h_ref = signal.sosfreqz(sos_ref, worN=freqs, fs=FS)
        np.testing.assert_allclose(np.abs(h_eq), np.abs(h_ref), rtol=1e-7, atol=1e-9)


# ---------------------------------------------------------------------------
# Cascading, filtering and block processing
# ---------------------------------------------------------------------------


def _three_sections() -> list[ph.filters.EQSection]:
    return [
        ph.filters.EQSection("lowshelf", 100.0, gain_db=4.0, slope=1.0),
        ph.filters.EQSection("peaking", 1000.0, gain_db=-6.0, bw=1.0),
        ph.filters.EQSection("highshelf", 8000.0, gain_db=3.0),
    ]


def test_cascade_magnitude_is_sum_of_sections() -> None:
    """The cascade magnitude in dB is the sum of the section magnitudes."""
    res = ph.filters.ParametricEQ(FS, _three_sections()).response(n_points=256)
    np.testing.assert_allclose(
        res.magnitude_db,
        res.section_magnitude_db.sum(axis=0),
        rtol=0.0,
        atol=1e-9,
    )
    assert res.sos.shape == (3, 6)
    assert len(res.sections) == 3


def test_filter_equals_sosfilt_and_convenience_wrapper() -> None:
    """filter() is the SOS cascade; parametric_eq() is the same one-shot."""
    rng = np.random.default_rng(20260721)
    x = rng.standard_normal(4096)
    eq = ph.filters.ParametricEQ(FS, _three_sections())
    expected = signal.sosfilt(eq.sos, x)
    np.testing.assert_allclose(eq.filter(x), expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        ph.filters.parametric_eq(x, FS, sections=_three_sections()),
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_stateful_block_processing_equals_one_shot() -> None:
    """Concatenated stateful blocks equal the single continuous call."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(8192)
    one_shot = ph.filters.ParametricEQ(FS, _three_sections()).filter(x)
    eq = ph.filters.ParametricEQ(FS, _three_sections(), stateful=True)
    blocks = [eq.filter(x[i : i + 1024]) for i in range(0, len(x), 1024)]
    np.testing.assert_allclose(np.concatenate(blocks), one_shot, rtol=0.0, atol=1e-12)


def test_multichannel_filtering() -> None:
    """2D [channels, samples] input filters each channel independently."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((3, 2048))
    eq = ph.filters.ParametricEQ(FS, _three_sections())
    y = eq.filter(x)
    assert y.shape == x.shape
    for ch in range(3):
        np.testing.assert_allclose(y[ch], eq.filter(x[ch]), rtol=0.0, atol=0.0)
    eq_st = ph.filters.ParametricEQ(FS, _three_sections(), stateful=True)
    y_blocks = np.concatenate(
        [eq_st.filter(x[:, :1024]), eq_st.filter(x[:, 1024:])], axis=-1
    )
    np.testing.assert_allclose(y_blocks, y, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("filter_type", _ALL_TYPES)
def test_every_type_designs_and_is_stable(filter_type: str) -> None:
    """Every cookbook type yields a stable, normalized SOS section."""
    gain = 6.0 if filter_type in ("peaking", "lowshelf", "highshelf") else 0.0
    eq = ph.filters.ParametricEQ(
        FS, ph.filters.EQSection(filter_type, 1000.0, gain_db=gain)
    )
    assert eq.sos.shape == (1, 6)
    assert eq.sos[0, 3] == pytest.approx(1.0, abs=0.0)  # normalized a0
    poles = np.roots(eq.sos[0, 3:])
    assert np.all(np.abs(poles) < 1.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_unknown_type_and_bad_frequencies() -> None:
    with pytest.raises(ValueError, match=r"Unknown filter_type 'bell'"):
        ph.filters.EQSection("bell", 1000.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"'f0' must be positive"):
        ph.filters.EQSection("peaking", 0.0)
    at_nyquist = ph.filters.EQSection("peaking", FS / 2, gain_db=3.0)
    with pytest.raises(
        ValueError, match=r"f0 = .* must sit below the Nyquist frequency"
    ):
        ph.filters.ParametricEQ(FS, at_nyquist)
    section = ph.filters.EQSection("peaking", 1000.0, gain_db=3.0)
    with pytest.raises(ValueError, match=r"'fs' must be positive"):
        ph.filters.ParametricEQ(0, section)


def test_rejects_conflicting_or_misplaced_parameters() -> None:
    with pytest.raises(ValueError, match=r"Give only one of 'q', 'bw' and 'slope'"):
        ph.filters.EQSection("peaking", 1000.0, gain_db=3.0, q=1.0, bw=1.0)
    with pytest.raises(ValueError, match=r"'slope' applies to the shelving types"):
        ph.filters.EQSection("peaking", 1000.0, gain_db=3.0, slope=1.0)
    with pytest.raises(ValueError, match=r"'bw' applies to"):
        ph.filters.EQSection("lowshelf", 1000.0, gain_db=3.0, bw=1.0)
    with pytest.raises(ValueError, match=r"'gain_db' applies to"):
        ph.filters.EQSection("notch", 1000.0, gain_db=3.0)
    with pytest.raises(ValueError, match=r"'q' must be positive"):
        ph.filters.EQSection("peaking", 1000.0, gain_db=3.0, q=-1.0)
    with pytest.raises(ValueError, match="at least one EQSection"):
        ph.filters.ParametricEQ(FS, [])


@pytest.mark.parametrize(
    ("kwargs", "offender"),
    [
        ({"f0": float("nan"), "gain_db": 3.0}, "f0"),
        ({"f0": float("inf"), "gain_db": 3.0}, "f0"),
        ({"f0": 1000.0, "gain_db": float("nan")}, "gain_db"),
        ({"f0": 1000.0, "gain_db": float("inf")}, "gain_db"),
        ({"f0": 1000.0, "gain_db": 3.0, "q": float("nan")}, "q"),
        ({"f0": 1000.0, "gain_db": 3.0, "bw": float("inf")}, "bw"),
    ],
)
def test_rejects_non_finite_parameters(kwargs: dict, offender: str) -> None:
    with pytest.raises(ValueError, match=rf"'{offender}' must be finite"):
        ph.filters.EQSection("peaking", **kwargs)


@pytest.mark.parametrize("gain_db", [1e308, -1e308])
@pytest.mark.parametrize(
    ("filter_type", "kwargs"),
    [("peaking", {}), ("lowshelf", {"slope": 1.0})],
)
def test_extreme_finite_gain_raises_value_error(
    gain_db: float, filter_type: str, kwargs: dict
) -> None:
    # 10^(gain_db/40) overflows for huge positive gains and underflows to
    # zero (then divides by zero in the designers) for huge negative ones;
    # both used to leak raw OverflowError/ZeroDivisionError.
    with pytest.raises(
        ValueError, match=r"'gain_db' = .* is outside the representable range"
    ):
        ph.filters.EQSection(filter_type, 1000.0, gain_db=gain_db, **kwargs)


def test_shelf_slope_beyond_gain_bound_raises_informatively() -> None:
    # A = 10^(9/40): the alpha recipe is real only below
    # (A + 1/A)/(A + 1/A - 2) = 8.2869...; beyond it the raw math would
    # fail with a bare "math domain error".
    with pytest.raises(ValueError, match=r"Shelf slope 'slope' = .* is too steep"):
        ph.filters.EQSection("lowshelf", 1000.0, gain_db=9.0, slope=12.0)
    # Just below the bound the design succeeds and is strictly stable.
    section = ph.filters.EQSection("lowshelf", 1000.0, gain_db=9.0, slope=8.0)
    sos = ph.filters.ParametricEQ(FS, section).sos
    assert np.max(np.abs(np.roots(sos[0, 3:]))) < 1.0
    # Gain 0 dB (A = 1): every positive slope is admissible.
    ph.filters.EQSection("lowshelf", 1000.0, gain_db=0.0, slope=50.0)


def test_bw_overflow_near_nyquist_raises_value_error() -> None:
    # The w0/sin(w0) warping factor diverges towards Nyquist: this bw/f0
    # pair used to escape as a raw OverflowError from math.sinh.
    section = ph.filters.EQSection("notch", 23999.0, bw=1.0)
    with pytest.raises(ValueError, match=r"Bandwidth 'bw' = .* is too wide"):
        ph.filters.ParametricEQ(FS, section)


def test_bw_marginally_unstable_section_raises_value_error() -> None:
    # Large but finite alpha: the rounded a2 lands exactly at -1.0 (poles
    # on the unit circle) with no arithmetic error to flag it.
    for f0, bw in ((23800.0, 1.0), (23900.0, 6.0)):
        section = ph.filters.EQSection("notch", f0, bw=bw)
        with pytest.raises(ValueError, match="not strictly stable"):
            ph.filters.ParametricEQ(FS, section)


def test_extreme_but_valid_sections_still_design_stably() -> None:
    for filter_type, f0, kwargs in (
        ("peaking", 23990.0, {"gain_db": 40.0, "q": 100.0}),
        ("lowpass", 23999.0, {"q": 100.0}),
        ("highpass", 0.01, {"q": 0.001}),
        ("allpass", 12000.0, {"q": 1e-6}),
    ):
        sos = ph.filters.ParametricEQ(
            FS, ph.filters.EQSection(filter_type, f0, **kwargs)
        ).sos
        assert np.all(np.isfinite(sos))
        assert np.max(np.abs(np.roots(sos[0, 3:]))) < 1.0


def test_response_grid_validation() -> None:
    eq = ph.filters.ParametricEQ(
        FS, ph.filters.EQSection("peaking", 1000.0, gain_db=3.0)
    )
    with pytest.raises(ValueError, match=r"'n_points' must be at least"):
        eq.response(n_points=1)
    with pytest.raises(ValueError, match=r"Need 0 < f_min < f_max <= fs/2"):
        eq.response(f_min=0.0)
    with pytest.raises(ValueError, match=r"Need 0 < f_min < f_max <= fs/2"):
        eq.response(f_min=100.0, f_max=50.0)
    with pytest.raises(ValueError, match=r"Need 0 < f_min < f_max <= fs/2"):
        eq.response(f_max=FS)


def test_response_rejects_cascade_short_of_its_sections() -> None:
    """Two rows of curves for three sections is not a response."""
    res = ph.filters.ParametricEQ(FS, _three_sections()).response(n_points=128)
    short = res.section_magnitude_db[:2]
    with pytest.raises(ValueError, match=r"'section_magnitude_db' \(2\)"):
        dataclasses.replace(res, section_magnitude_db=short)


def test_response_rejects_cascade_taller_than_its_sections() -> None:
    """The silent direction: the plot loops over ``sections`` and drops the rest."""
    res = ph.filters.ParametricEQ(FS, _three_sections()).response(n_points=128)
    tall = np.vstack([res.section_magnitude_db, res.section_magnitude_db[:1]])
    with pytest.raises(ValueError, match=r"'section_magnitude_db' \(4\)"):
        dataclasses.replace(res, section_magnitude_db=tall)


def test_response_rejects_section_curves_off_the_evaluation_grid() -> None:
    """The grid axis is pinned on its own: the section axis still agrees."""
    res = ph.filters.ParametricEQ(FS, _three_sections()).response(n_points=128)
    short_grid = res.section_magnitude_db[:, :-1]
    with pytest.raises(ValueError, match=r"'section_magnitude_db \(axis 1\)'"):
        dataclasses.replace(res, section_magnitude_db=short_grid)


def test_response_rejects_sos_block_short_of_its_sections() -> None:
    """The cascade that leaves here to be filtered with must be the drawn one."""
    res = ph.filters.ParametricEQ(FS, _three_sections()).response(n_points=128)
    short_sos = res.sos[:2]
    with pytest.raises(ValueError, match=r"'sos' \(2\)"):
        dataclasses.replace(res, sos=short_sos)


def test_response_rejects_section_curves_with_an_extra_axis() -> None:
    """Both counts agree on a (sections, frequencies, 1) block; the rank pin does not."""
    res = ph.filters.ParametricEQ(FS, _three_sections()).response(n_points=128)
    boxed = res.section_magnitude_db[:, :, None]
    with pytest.raises(ValueError, match="'section_magnitude_db' must have 2 axes"):
        dataclasses.replace(res, section_magnitude_db=boxed)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plot_two_panel_and_single_axes() -> None:
    res = ph.filters.ParametricEQ(FS, _three_sections()).response(n_points=128)
    axes = res.plot()
    assert axes.shape == (2,)
    assert "Audio EQ Cookbook" in axes[0].get_title()
    assert axes[1].get_ylabel() == "Phase [deg]"
    plt.close("all")

    _, ax = plt.subplots()
    out = res.plot(ax=ax)
    assert out is ax
    assert ax.get_ylabel() == "Magnitude [dB]"
    plt.close("all")


def test_plot_rejects_unknown_language() -> None:
    res = ph.filters.ParametricEQ(
        FS, ph.filters.EQSection("peaking", 1000.0, gain_db=3.0)
    ).response(n_points=64)
    with pytest.raises(ValueError, match=r"Unknown language 'xx'"):
        res.plot(language="xx")
    plt.close("all")

#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Clean-room oracles for the B, AU and D frequency weightings.

B is pinned to ANSI S1.4-1983: the Table IV design goals (B column), the
Table V tolerance masks and the Appendix C analytic form. AU is pinned to
IEC 61012:1990: the Table 2 poles must reproduce every Table 1 nominal value,
and the realized filter must stay within the Table 1 separate-unit
tolerances, with the subclause 2.2 explicit AU values above 20 kHz. D has no
surviving standard text (IEC 537:1976 is withdrawn), so its rational transfer
function is cross-checked against two independent implementations - librosa's
closed form (constants transcribed in ``reference_data``) and SQAT's
zeros/poles (identical to ours by construction) - and against the tabulated
IEC 537 curve republished in NASA CR-3406 Table SLD-I.

The masks transcribed inside :mod:`phonometry.filters.compliance` are
pinned to the independent ``reference_data`` copies shared with the CI
conformance report, so a typo in either surfaces.
"""

import math

import numpy as np
import pytest
from reference_data import (
    ANSIS14_TABLE4_B,
    ANSIS14_TABLE5,
    IEC537_NASA_TABLE_SLD1,
    IEC61012_AU_HF,
    IEC61012_TABLE1,
    IEC61012_TABLE2_POLES_HZ,
    IEC61672_TABLE3,
    LIBROSA_D_WEIGHTING_CONSTS,
)
from scipy import signal as sg

from phonometry import filters
from phonometry.filters.weighting_compliance import (
    _ANSI_S14_TABLE4_B,
    _ANSI_S14_TABLE5_12,
    _IEC61012_AU_HF,
    _IEC61012_TABLE1,
    _U_POLES_HZ,
    _analytic_weighting_db,
)


def _exact(freq: float) -> float:
    """Exact base-10 frequency behind a nominal one-third-octave label."""
    return float(10.0 ** (round(10.0 * math.log10(freq)) / 10.0))


def _response_db(
    wf: filters.WeightingFilter, freqs: "list[float] | np.ndarray"
) -> np.ndarray:
    """Relative response of the designed SOS in dB, normalized to 1 kHz."""
    fs_proc = wf.fs * wf._oversample
    worn = np.concatenate([np.asarray(freqs, dtype=float), [1000.0]])
    _, h = sg.sosfreqz(wf.sos, worN=worn, fs=fs_proc)
    gain = 20.0 * np.log10(np.abs(h))
    return np.asarray(gain[:-1] - gain[-1], dtype=np.float64)


# ---------------------------------------------------------------------------
# Mask transcriptions pinned to the independent reference_data copies
# ---------------------------------------------------------------------------


def test_b_masks_match_reference_data() -> None:
    """The module's ANSI S1.4-1983 tables equal the reference_data copies."""
    assert len(_ANSI_S14_TABLE4_B) == len(ANSIS14_TABLE4_B) == 34
    for row, ref_row in zip(_ANSI_S14_TABLE4_B, ANSIS14_TABLE4_B):
        assert row == pytest.approx(ref_row), f"Table IV mismatch at {ref_row[0]} Hz"
    # Table V Types 1/2 (columns 3-6 of the reference table).
    assert len(_ANSI_S14_TABLE5_12) == len(ANSIS14_TABLE5) == 34
    for row, ref_row in zip(_ANSI_S14_TABLE5_12, ANSIS14_TABLE5):
        assert row[0] == ref_row[0]
        assert row[1:] == pytest.approx(ref_row[3:]), f"Table V mismatch at {ref_row[0]} Hz"


def test_au_masks_match_reference_data() -> None:
    """The module's IEC 61012:1990 tables equal the reference_data copies."""
    assert len(_IEC61012_TABLE1) == len(IEC61012_TABLE1) == 37
    for row, ref_row in zip(_IEC61012_TABLE1, IEC61012_TABLE1):
        assert row == pytest.approx(ref_row), f"Table 1 mismatch at {ref_row[0]} Hz"
    assert _IEC61012_AU_HF == pytest.approx(IEC61012_AU_HF)
    poles = [(p.real, p.imag) for p in _U_POLES_HZ]
    assert poles == pytest.approx(IEC61012_TABLE2_POLES_HZ)


# ---------------------------------------------------------------------------
# B weighting vs ANSI S1.4-1983
# ---------------------------------------------------------------------------


def test_b_pins_ansi_table4_values() -> None:
    """Pin the realized B response to Table IV at 31.5 Hz, 1 kHz and 8 kHz."""
    wf = filters.WeightingFilter(96000, "B")
    resp = _response_db(wf, [_exact(31.5), 1000.0, _exact(8000.0)])
    assert resp[0] == pytest.approx(-17.1, abs=0.06)
    assert resp[1] == 0.0  # exact by 1 kHz normalization
    assert resp[2] == pytest.approx(-2.9, abs=0.1)


def test_b_realized_within_type0_at_every_table4_row() -> None:
    """The realized B filter meets the strictest Table V mask (Type 0)."""
    wf = filters.WeightingFilter(96000, "B")
    freqs = [_exact(row[0]) for row in ANSIS14_TABLE4_B]
    resp = _response_db(wf, freqs)
    for (freq, goal), (_, up0, lo0, *_), got in zip(
        ANSIS14_TABLE4_B, ANSIS14_TABLE5, resp
    ):
        dev = got - goal
        assert lo0 <= dev <= up0, f"B outside Type 0 at {freq} Hz: {dev:+.3f} dB"


def test_b_appendix_c_analytic_reproduces_table4() -> None:
    """The Appendix C closed form reproduces every Table IV B value.

    W_B = 10 lg(K2 f^2 / (f^2 + f5^2)) + W_C (Formula C2) evaluated at the
    exact base-10 frequencies rounds to the printed 0.1 dB column, so the
    analytic design goal used by the between-nominals sweep is anchored to
    the standard's own table (0.05 dB = the table's rounding half-step).
    """
    freqs = np.array([_exact(row[0]) for row in ANSIS14_TABLE4_B])
    analytic = _analytic_weighting_db("B", freqs)
    for (freq, goal), got in zip(ANSIS14_TABLE4_B, analytic):
        assert got == pytest.approx(goal, abs=0.05), f"{freq} Hz"


def test_b_sits_between_a_and_c_at_low_frequency() -> None:
    """Historical sanity: B (70-phon heritage) lies between A and C below 1 kHz."""
    a_nom = {row[0]: row[1] for row in IEC61672_TABLE3}
    c_nom = {row[0]: row[2] for row in IEC61672_TABLE3}
    for freq, b_val in ANSIS14_TABLE4_B:
        if freq < 800:
            assert a_nom[freq] < b_val < c_nom[freq] + 1e-9, f"{freq} Hz"


def test_b_verifier_reaches_type1() -> None:
    """verify_weighting_class attests the ANSI Type 1 mask for B."""
    result = filters.verify_weighting_class(
        filters.WeightingFilter(48000, "B")
    )
    assert result["overall_class"] == 1
    assert result["range_limited"] is False
    assert all(b["margin_class1_db"] >= 0 for b in result["bands"])


# ---------------------------------------------------------------------------
# AU weighting vs IEC 61012:1990
# ---------------------------------------------------------------------------


def test_au_table2_poles_reproduce_table1_nominals() -> None:
    """The Table 2 poles reproduce every Table 1 nominal U value within 0.06 dB.

    This anchors the pole realization to the standard's own response table
    independently of any filter design code. The roll-off rows (12.5-40 kHz,
    printed to 0.1 dB) agree within the 0.05 dB rounding half-step; the
    pass-band rows print a nominal 0 while the 1 kHz-normalized pole
    response sits up to 0.054 dB above it (the reference itself absorbs the
    onset of the U droop), well inside the +/-1 dB tolerance there.
    """
    poles = np.array([complex(re, im) for re, im in IEC61012_TABLE2_POLES_HZ])

    def u_db(freq: float) -> float:
        den = np.prod(1j * freq - poles)
        return float(20.0 * math.log10(1.0 / abs(den)))

    ref_1k = u_db(1000.0)
    for freq, u_nom, _, _ in IEC61012_TABLE1:
        got = u_db(_exact(freq)) - ref_1k
        assert got == pytest.approx(u_nom, abs=0.06), f"{freq} Hz"


def test_au_pins_nominal_values() -> None:
    """Pin the realized AU response at 31.5 Hz, 1 kHz, 8 kHz and the
    subclause 2.2 explicit values at 25/31.5/40 kHz (nominal A + U)."""
    wf = filters.WeightingFilter(96000, "AU")
    freqs = [_exact(f) for f in (31.5, 1000.0, 8000.0, 25000.0, 31500.0, 40000.0)]
    resp = _response_db(wf, freqs)
    assert resp[0] == pytest.approx(-39.4, abs=0.06)
    assert resp[1] == 0.0  # exact by 1 kHz normalization (zero tolerance row)
    assert resp[2] == pytest.approx(-1.1, abs=0.06)
    # Above 20 kHz the bilinear design attenuates faster than nominal (the
    # deviation is negative); IEC 61012 allows -6/-10/-inf dB there.
    assert -56.0 <= resp[3] <= -50.0 + 3.0
    assert -75.4 <= resp[4] <= -65.4 + 3.0
    assert resp[5] <= -81.1 + 3.0


def test_au_realized_within_table1_tolerances() -> None:
    """The realized AU filter meets the Table 1 separate-unit mask at 96 kHz."""
    a_nom = {row[0]: row[1] for row in IEC61672_TABLE3}
    wf = filters.WeightingFilter(96000, "AU")
    rows = [r for r in IEC61012_TABLE1 if _exact(r[0]) < 48000.0 and r[0] != 1000]
    resp = _response_db(wf, [_exact(r[0]) for r in rows])
    for (freq, u_nom, up, lo), got in zip(rows, resp):
        goal = IEC61012_AU_HF.get(freq, a_nom.get(freq, 0.0) + u_nom)
        assert lo <= got - goal <= up, f"AU outside Table 1 at {freq} Hz"


def test_au_matches_a_weighting_through_the_audio_band() -> None:
    """Below 8 kHz the AU filter is the A filter (U contributes ~0 dB).

    The two designs run at different internal oversampled rates (AU targets
    288 kHz for the U roll-off, A targets 144 kHz), so their bilinear
    residuals differ by up to ~0.1 dB near the top of the compared band.
    """
    au = filters.WeightingFilter(48000, "AU")
    a = filters.WeightingFilter(48000, "A")
    freqs = np.geomspace(20.0, 8000.0, 200)
    assert _response_db(au, freqs) == pytest.approx(
        _response_db(a, freqs), abs=0.12
    )


def test_au_verifier_flags_range_limited_at_48k() -> None:
    """At fs = 48 kHz the 25-40 kHz Table 1 rows cannot be demonstrated."""
    result = filters.verify_weighting_class(
        filters.WeightingFilter(48000, "AU")
    )
    assert result["overall_class"] == 1
    assert result["range_limited"] is True
    result_full = filters.verify_weighting_class(
        filters.WeightingFilter(96000, "AU")
    )
    assert result_full["overall_class"] == 1
    assert result_full["range_limited"] is False
    # Single tolerance set: both verdict slots carry the same margin.
    for band in result_full["bands"]:
        assert band["margin_class1_db"] == band["margin_class2_db"]


# ---------------------------------------------------------------------------
# D weighting vs independent implementations and the published curve
# ---------------------------------------------------------------------------


def _librosa_d_weighting_db(freqs: np.ndarray) -> np.ndarray:
    """librosa's closed-form D weighting (independent oracle).

    Transcribed from ``librosa.D_weighting`` (librosa/core/convert.py, ISC
    license) via the constants pinned in ``reference_data``; librosa works
    from squared constants in a log10 sum, an entirely different formulation
    from the zpk cascade under test.
    """
    f_sq = np.asarray(freqs, dtype=float) ** 2
    c = np.array(LIBROSA_D_WEIGHTING_CONSTS) ** 2
    return 20.0 * (
        0.5 * np.log10(f_sq)
        - np.log10(c[0])
        + 0.5
        * (
            np.log10((c[1] - f_sq) ** 2 + c[2] * f_sq)
            - np.log10((c[3] - f_sq) ** 2 + c[4] * f_sq)
            - np.log10(c[5] + f_sq)
            - np.log10(c[6] + f_sq)
        )
    )


def test_d_matches_librosa_closed_form() -> None:
    """The realized D filter agrees with librosa's independent closed form.

    The s-domain transfer functions agree within 0.002 dB from 10 Hz to
    20 kHz; the realized digital filter adds only the documented bilinear
    residual near Nyquist, so the agreement bound tightens with frequency.
    """
    wf = filters.WeightingFilter(96000, "D")
    freqs = np.geomspace(10.0, 20000.0, 400)
    ours = _response_db(wf, freqs)
    theirs = _librosa_d_weighting_db(freqs) - _librosa_d_weighting_db(
        np.array([1000.0])
    )
    low = freqs <= 8000.0
    assert ours[low] == pytest.approx(theirs[low], abs=0.08)
    # Near 20 kHz the bilinear warping of the realized filter (designed at
    # 192 kHz for fs = 96 kHz) reaches ~0.33 dB against the analytic form.
    assert ours == pytest.approx(theirs, abs=0.4)


def test_d_pins_published_table_values() -> None:
    """Pin the realized D response to NASA CR-3406 Table SLD-I (IEC 537).

    The table starts at 50 Hz, so the pins are 50 Hz, 1 kHz and 8 kHz; 1 kHz
    is exact by normalization. Every other row must agree within 0.2 dB
    except 1600/2500 Hz, where the published table itself departs from the
    rational transfer function by 0.15/0.28 dB (see the module docstring of
    ``weighting``).
    """
    wf = filters.WeightingFilter(48000, "D")
    freqs = [float(row[0]) for row in IEC537_NASA_TABLE_SLD1]
    resp = _response_db(wf, freqs)
    table = {row[0]: row[1] for row in IEC537_NASA_TABLE_SLD1}
    got = dict(zip(table, resp))
    assert got[50] == pytest.approx(-12.8, abs=0.1)
    assert got[1000] == 0.0
    assert got[8000] == pytest.approx(5.5, abs=0.2)
    for freq, value in table.items():
        bound = 0.45 if freq in (1600, 2500) else 0.2
        assert got[freq] == pytest.approx(value, abs=bound), f"{freq} Hz"


def test_d_peak_sits_in_the_perceived_noisiness_region() -> None:
    """The D curve peaks near 3.15 kHz at about +11.5 dB (Table SLD-I)."""
    wf = filters.WeightingFilter(96000, "D")
    freqs = np.geomspace(2000.0, 5000.0, 300)
    resp = _response_db(wf, freqs)
    peak = float(freqs[int(np.argmax(resp))])
    assert 2800.0 <= peak <= 3600.0
    assert float(np.max(resp)) == pytest.approx(11.55, abs=0.15)


def test_d_and_g_are_rejected_by_the_class_verifier() -> None:
    """No surviving tolerance tables: the class verifier refuses D (and G)."""
    for curve in ("D", "G"):
        weighting = filters.WeightingFilter(48000, curve)
        with pytest.raises(ValueError, match="'A', 'B', 'C', 'AU' or 'Z'"):
            filters.verify_weighting_class(weighting)


# ---------------------------------------------------------------------------
# Filtering behavior shared with the other curves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("curve", ["B", "D", "AU"])
def test_new_curves_filter_time_domain_tone(curve: str) -> None:
    """A filtered 1 kHz tone keeps its RMS (0 dB at 1 kHz by definition)."""
    fs = 48000
    t = np.arange(fs) / fs
    x = np.sin(2 * np.pi * 1000.0 * t)
    y = filters.WeightingFilter(fs, curve).filter(x)
    skip = fs // 5  # discard the onset transient
    gain_db = 20 * np.log10(np.std(y[skip:]) / np.std(x[skip:]))
    assert gain_db == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("curve", ["B", "D", "AU"])
def test_new_curves_support_stateful_block_processing(curve: str) -> None:
    """Stateful block outputs equal the continuous (plain-design) result."""
    fs = 48000
    rng = np.random.default_rng(7)
    x = rng.standard_normal(fs // 2)
    continuous = filters.WeightingFilter(
        fs, curve, high_accuracy=False
    ).filter(x)
    wf = filters.WeightingFilter(fs, curve, stateful=True)
    blocks = np.concatenate([wf.filter(part) for part in np.split(x, 4)])
    assert blocks == pytest.approx(continuous, abs=1e-12)

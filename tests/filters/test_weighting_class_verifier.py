#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the IEC 61672-1:2013 frequency-weighting class verifier
(``verify_weighting_class`` / ``weighting_class_limits``).

The design-goal responses and class 1 limits in the verifier's Table 3
transcription are cross-checked against the independent ``reference_data``
copy shared with the CI conformance report, so a typo in either surfaces.
"""

import math

import pytest
from reference_data import IEC61672_TABLE3 as TABLE3

from phonometry import (
    WeightingFilter,
    verify_weighting_class,
    weighting_class_limits,
)
from phonometry.filters.weighting_compliance import (
    _WEIGHTING_TABLE3,
)


def test_masks_match_reference_data() -> None:
    """The module's Table 3 (freq, A, C, class1 upper/lower) equals reference_data."""
    assert len(_WEIGHTING_TABLE3) == len(TABLE3) == 34
    for row, ref in zip(_WEIGHTING_TABLE3, TABLE3):
        assert row[:5] == pytest.approx(ref, nan_ok=False), f"mismatch at {ref[0]} Hz"


def test_weighting_class_limits_shape_and_known_values() -> None:
    freqs, lower1, upper1 = weighting_class_limits(1)
    _, lower2, upper2 = weighting_class_limits(2)
    assert len(freqs) == 34
    i1k = freqs.tolist().index(1000.0)
    # 1 kHz reference point: +-0.7 (class 1), +-1.0 (class 2).
    assert (lower1[i1k], upper1[i1k]) == (-0.7, 0.7)
    assert (lower2[i1k], upper2[i1k]) == (-1.0, 1.0)
    # One-sided limits at the extreme bands (lower = -inf).
    i10 = freqs.tolist().index(10.0)
    assert math.isinf(lower1[i10])
    assert lower1[i10] < 0
    assert upper1[i10] == 3.0
    # Asymmetric class 1 limit at 16 kHz.
    i16k = freqs.tolist().index(16000.0)
    assert (lower1[i16k], upper1[i16k]) == (-16.0, 2.5)


@pytest.mark.parametrize("fs", [48000, 96000])
@pytest.mark.parametrize("curve", ["A", "C", "Z"])
def test_high_accuracy_weightings_are_class1(fs: int, curve: str) -> None:
    """A/C/Z high-accuracy filters meet class 1 at every in-range frequency."""
    result = verify_weighting_class(WeightingFilter(fs, curve))
    assert result["overall_class"] == 1
    assert all(b["class"] == 1 for b in result["bands"])
    assert all(b["margin_class1_db"] >= 0 for b in result["bands"])


def test_z_weighting_zero_deviation() -> None:
    """Z is a flat bypass: zero deviation and full class-1 margin everywhere."""
    result = verify_weighting_class(WeightingFilter(48000, "Z"))
    assert all(b["deviation_db"] == 0.0 for b in result["bands"])
    assert result["overall_class"] == 1


def test_band_dict_keys_and_deviation_sign() -> None:
    result = verify_weighting_class(WeightingFilter(48000, "A"))
    band = result["bands"][0]
    assert set(band) == {
        "freq",
        "class",
        "deviation_db",
        "margin_class1_db",
        "margin_class2_db",
    }
    # A weighting strongly attenuates 10 Hz; relative to 1 kHz the design goal
    # is -70.4 dB, so a compliant filter's deviation stays small.
    assert abs(band["deviation_db"]) < 3.0


def test_frequencies_above_nyquist_are_dropped() -> None:
    """Only Table 3 rows whose exact frequency is below fs/2 are evaluated.

    At fs = 16 kHz the "8 kHz" row IS evaluated (its exact base-10 frequency,
    7 943.3 Hz, is below Nyquist) while 10 kHz and above are dropped.
    """
    result = verify_weighting_class(WeightingFilter(16000, "A"))
    assert max(b["freq"] for b in result["bands"]) == 8000.0


def test_low_fs_verdict_is_flagged_range_limited() -> None:
    """Dropping Table 3 rows that carry finite lower limits (class 1 has them
    up to 16 kHz) must flag the verdict as range-limited: a 16 kHz-sampled
    system cannot demonstrate full class-1 conformance over 10 Hz-20 kHz."""
    result = verify_weighting_class(WeightingFilter(16000, "A"))
    assert result["range_limited"] is True
    result_full = verify_weighting_class(WeightingFilter(48000, "A"))
    assert result_full["range_limited"] is False


def test_deviation_evaluated_at_exact_base10_frequency() -> None:
    """The SOS is evaluated at the exact base-10 frequency behind each nominal
    label (Table 3 NOTE; IEC 61672-3 subclause 13.3), not at the label itself:
    the reported deviation at "16 kHz" equals the response at 15 848.9 Hz
    minus the Table 3 design goal."""
    import numpy as np
    from scipy import signal as sg

    wf = WeightingFilter(96000, "A")
    result = verify_weighting_class(wf)
    band = next(b for b in result["bands"] if b["freq"] == 16000.0)
    exact = 10.0 ** (np.round(10.0 * np.log10(16000.0)) / 10.0)  # 15848.93 Hz
    fs_proc = wf.fs * wf._oversample
    _, h = sg.sosfreqz(wf.sos, worN=np.array([exact, 1000.0]), fs=fs_proc)
    response = 20.0 * np.log10(np.abs(h))
    deviation = (response[0] - response[1]) - (-6.6)  # Table 3 A @ 16 kHz
    assert band["deviation_db"] == pytest.approx(deviation, abs=1e-9)


def test_notch_between_nominals_fails_the_sweep() -> None:
    """Adversarial 5.5.7 case: an iirnotch at 900 Hz (between the 800 and
    1000 Hz nominals) leaves every nominal-frequency verdict at class 1 but
    must fail the between-nominals sweep, so no class can be assigned."""
    import numpy as np
    from scipy import signal as sg

    wf = WeightingFilter(48000, "A")
    fs_proc = wf.fs * wf._oversample
    b, a = sg.iirnotch(900.0, 30.0, fs=fs_proc)
    wf.sos = np.vstack([wf.sos, sg.tf2sos(b, a)])
    result = verify_weighting_class(wf)
    assert all(bd["class"] == 1 for bd in result["bands"])
    assert result["between_nominals"]["margin_class1_db"] < 0.0
    assert result["between_nominals"]["margin_class2_db"] < 0.0
    assert 800.0 < result["between_nominals"]["worst_freq"] < 1000.0
    assert result["overall_class"] is None


def test_sweep_result_reported_for_compliant_filter() -> None:
    result = verify_weighting_class(WeightingFilter(48000, "A"))
    between = result["between_nominals"]
    assert set(between) == {"worst_freq", "margin_class1_db", "margin_class2_db"}
    assert between["margin_class1_db"] >= 0.0
    wf = WeightingFilter(48000, "A")
    with pytest.raises(ValueError, match="sweep_points"):
        verify_weighting_class(wf, sweep_points=8)


@pytest.mark.parametrize("fs,expected", [(32000, 2), (24000, 2)])
def test_plain_bilinear_degrades_to_class2(fs: int, expected: int) -> None:
    """Without oversampling the bilinear warping fails class 1 near Nyquist."""
    result = verify_weighting_class(WeightingFilter(fs, "A", high_accuracy=False))
    assert result["overall_class"] == expected
    assert any(b["class"] != 1 for b in result["bands"])


def test_invalid_class_raises() -> None:
    with pytest.raises(ValueError):
        weighting_class_limits(0)
    with pytest.raises(ValueError):
        weighting_class_limits(3)


def _tone_gain_db(wf: WeightingFilter, fs: int, f0: float) -> float:
    """Steady-state RMS gain of *wf* at f0 (time-domain, independent method)."""
    import numpy as np

    duration = max(0.5, 12 / f0)
    t = np.arange(int(fs * duration)) / fs
    x = np.sin(2 * np.pi * f0 * t)
    y = wf.filter(x)
    n0 = int(0.2 * fs)  # skip the transient
    return float(20 * np.log10(np.std(y[n0:]) / np.std(x[n0:])))


def test_deviation_matches_independent_tone_measurement() -> None:
    """Cross-check the sosfreqz deviations against a time-domain tone RMS.

    ``verify_weighting_class`` reads the designed SOS in the frequency domain;
    this recomputes the deviation from a filtered pure tone (relative to the
    1 kHz tone, matching the verifier's normalization), an independent method.
    They must agree to within the RMS-estimation error.
    """
    fs = 48000
    wf = WeightingFilter(fs, "A")  # stateless, so it can be reused per tone
    bands = {b["freq"]: b["deviation_db"] for b in verify_weighting_class(wf)["bands"]}
    ref_1k = _tone_gain_db(wf, fs, 1000.0)
    for f0 in (63.0, 250.0, 1000.0, 4000.0, 8000.0):
        measured = _tone_gain_db(wf, fs, f0) - ref_1k
        design = {row[0]: row[1] for row in _WEIGHTING_TABLE3}[f0]
        assert bands[f0] == pytest.approx(measured - design, abs=0.15), f"{f0} Hz"


def test_response_is_deterministic() -> None:
    """The verifier reads the designed SOS, so repeated runs are identical."""
    a = verify_weighting_class(WeightingFilter(48000, "A"))
    b = verify_weighting_class(WeightingFilter(48000, "A"))
    assert [x["deviation_db"] for x in a["bands"]] == [
        x["deviation_db"] for x in b["bands"]
    ]

#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the IEC 61672-1:2013 frequency-weighting class verifier
(``verify_weighting_class`` / ``weighting_class_limits``).

The design-goal responses and class 1 limits in the verifier's Table 3
transcription are cross-checked against the independent ``reference_data``
copy shared with the CI conformance report, so a typo in either surfaces.
"""

import inspect
import math

import pytest
from reference_data import IEC61672_TABLE3 as TABLE3

from phonometry import filters
from phonometry.filters.weighting_compliance import (
    _WEIGHTING_TABLE3,
)


def test_masks_match_reference_data() -> None:
    """The module's Table 3 (freq, A, C, class1 upper/lower) equals reference_data."""
    assert len(_WEIGHTING_TABLE3) == len(TABLE3) == 34
    for row, ref in zip(_WEIGHTING_TABLE3, TABLE3, strict=True):
        assert row[:5] == pytest.approx(ref, nan_ok=False), f"mismatch at {ref[0]} Hz"


def test_weighting_class_limits_shape_and_known_values() -> None:
    freqs, lower1, upper1 = filters.weighting_class_limits(1)
    _, lower2, upper2 = filters.weighting_class_limits(2)
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


# The class each default (high-accuracy) filter actually earns, per curve and
# sample rate. Oversampling fixes the bilinear warping but not the anti-alias
# transition band the resampling stages put on the input Nyquist frequency, so
# at 16 and 32 kHz the top Table 3 row below Nyquist is judged inside it: A
# lands 12.0 dB low at 16 kHz (no class) and 16.2 dB low at 32 kHz (class 2),
# C lands 13.7 dB low at 16 kHz (no class) and 15.3 dB low at 32 kHz, still
# inside its -16.0 dB class 1 limit. Z is a bypass and is always class 1.
_EXPECTED_CLASS: dict[tuple[str, int], int | None] = {
    ("A", 16000): None,
    ("A", 32000): 2,
    ("A", 48000): 1,
    ("A", 96000): 1,
    ("C", 16000): None,
    ("C", 32000): 1,
    ("C", 48000): 1,
    ("C", 96000): 1,
    ("Z", 16000): 1,
    ("Z", 32000): 1,
    ("Z", 48000): 1,
    ("Z", 96000): 1,
}


@pytest.mark.parametrize("fs", [16000, 32000, 48000, 96000])
@pytest.mark.parametrize("curve", ["A", "C", "Z"])
def test_high_accuracy_weighting_verdicts(fs: int, curve: str) -> None:
    """The verdict of each default A/C/Z filter, at four sample rates."""
    expected = _EXPECTED_CLASS[(curve, fs)]
    result = filters.verify_weighting_class(filters.WeightingFilter(fs, curve))
    assert result["overall_class"] == expected
    if expected == 1:
        assert all(b["class"] == 1 for b in result["bands"])
        assert all(b["margin_class1_db"] >= 0 for b in result["bands"])
    else:
        # Exactly one row, the one nearest Nyquist, is what costs the class.
        off = [b for b in result["bands"] if b["class"] != 1]
        assert len(off) == 1
        assert off[0]["freq"] == max(b["freq"] for b in result["bands"])


def test_high_accuracy_docstring_states_the_32_khz_verdict_per_curve() -> None:
    """The ``high_accuracy`` paragraph serves every curve, so a verdict stated
    in it without naming one reads as a verdict for all of them, and at 32 kHz
    the curves do not agree: the same -16.0 dB class 1 lower limit that the A
    response misses the C response clears. Each verdict the paragraph states
    for that rate must therefore name its curve and match what the verifier
    measures for it, so the prose cannot drift from the code.
    """
    doc = inspect.getdoc(filters.WeightingFilter.__init__) or ""
    assert ":param high_accuracy:" in doc
    # ``high_accuracy`` is the last parameter, so the rest of the docstring is
    # its paragraph; collapse the wrapping so phrases survive a line break.
    paragraph = " ".join(doc.split(":param high_accuracy:", 1)[1].split())

    verdicts = {
        curve: filters.verify_weighting_class(filters.WeightingFilter(32000, curve))[
            "overall_class"
        ]
        for curve in ("A", "C")
    }
    # The premise of the check: at 32 kHz the verdict depends on the curve.
    assert verdicts["A"] != verdicts["C"]
    for curve, verdict in verdicts.items():
        assert f"{verdict} for {curve}" in paragraph, (
            f"the 32 kHz verdict of the {curve} curve is class {verdict}, which "
            f"the high_accuracy paragraph does not state for it: {paragraph}"
        )
    # At 16 kHz the two curves do agree, so one unqualified verdict is honest.
    assert all(
        filters.verify_weighting_class(filters.WeightingFilter(16000, curve))[
            "overall_class"
        ]
        is None
        for curve in ("A", "C")
    )


def test_z_weighting_zero_deviation() -> None:
    """Z is a flat bypass: zero deviation and full class-1 margin everywhere."""
    result = filters.verify_weighting_class(filters.WeightingFilter(48000, "Z"))
    assert all(b["deviation_db"] == 0.0 for b in result["bands"])
    assert result["overall_class"] == 1


def test_band_dict_keys_and_deviation_sign() -> None:
    result = filters.verify_weighting_class(filters.WeightingFilter(48000, "A"))
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
    result = filters.verify_weighting_class(filters.WeightingFilter(16000, "A"))
    assert max(b["freq"] for b in result["bands"]) == 8000.0


def test_low_fs_verdict_is_flagged_range_limited() -> None:
    """Dropping Table 3 rows that carry finite lower limits (class 1 has them
    up to 16 kHz) must flag the verdict as range-limited: a 16 kHz-sampled
    system cannot demonstrate full class-1 conformance over 10 Hz-20 kHz.
    """
    result = filters.verify_weighting_class(filters.WeightingFilter(16000, "A"))
    assert result["range_limited"] is True
    result_full = filters.verify_weighting_class(filters.WeightingFilter(48000, "A"))
    assert result_full["range_limited"] is False


def test_deviation_evaluated_at_exact_base10_frequency() -> None:
    """The response is taken at the exact base-10 frequency behind each
    nominal label (Table 3 NOTE; IEC 61672-3 subclause 13.3), not at the
    label itself: the reported deviation at "16 kHz" follows a tone at
    15 848.9 Hz and not one at 16 000 Hz, which the A curve puts about
    0.1 dB lower.
    """
    wf = filters.WeightingFilter(96000, "A")
    band = next(
        b for b in filters.verify_weighting_class(wf)["bands"] if b["freq"] == 16000.0
    )
    ref_1k = _tone_gain_db(wf, 96000, 1000.0)
    at_exact = _tone_gain_db(wf, 96000, 15848.93192) - ref_1k - (-6.6)
    at_label = _tone_gain_db(wf, 96000, 16000.0) - ref_1k - (-6.6)
    assert band["deviation_db"] == pytest.approx(at_exact, abs=0.02)
    assert abs(band["deviation_db"] - at_label) > 0.05


@pytest.mark.parametrize(
    ("fs", "label"), [(16000, 8000.0), (32000, 16000.0), (48000, 16000.0)]
)
def test_verdict_measures_the_resampled_path(fs: int, label: float) -> None:
    """The row nearest Nyquist is judged on what ``filter()`` really does.

    With the default ``high_accuracy`` the sections are reached through an
    interpolation and a decimation stage, whose anti-alias filter has its
    transition band on the input Nyquist frequency. Reading the response
    from the sections alone put the 7 943.3 Hz row of a 16 kHz system
    0.08 dB off its design goal when a tone through ``filter()`` measures
    12.0 dB off, and certified class 1 for a filter that meets neither
    class there.
    """
    wf = filters.WeightingFilter(fs, "A")
    band = next(
        b for b in filters.verify_weighting_class(wf)["bands"] if b["freq"] == label
    )
    exact = 10.0 ** (round(10.0 * math.log10(label)) / 10.0)
    design = {row[0]: row[1] for row in _WEIGHTING_TABLE3}[label]
    measured = _tone_gain_db(wf, fs, exact) - _tone_gain_db(wf, fs, 1000.0) - design
    assert band["deviation_db"] == pytest.approx(measured, abs=0.05)


def test_notch_between_nominals_fails_the_sweep() -> None:
    """Adversarial 5.5.7 case: an iirnotch at 900 Hz (between the 800 and
    1000 Hz nominals) leaves every nominal-frequency verdict at class 1 but
    must fail the between-nominals sweep, so no class can be assigned.
    """
    import numpy as np
    from scipy import signal as sg

    wf = filters.WeightingFilter(48000, "A")
    fs_proc = wf.fs * wf._oversample
    b, a = sg.iirnotch(900.0, 30.0, fs=fs_proc)
    wf.sos = np.vstack([wf.sos, sg.tf2sos(b, a)])
    result = filters.verify_weighting_class(wf)
    assert all(bd["class"] == 1 for bd in result["bands"])
    assert result["between_nominals"]["margin_class1_db"] < 0.0
    assert result["between_nominals"]["margin_class2_db"] < 0.0
    assert 800.0 < result["between_nominals"]["worst_freq"] < 1000.0
    assert result["overall_class"] is None


def test_sweep_result_reported_for_compliant_filter() -> None:
    result = filters.verify_weighting_class(filters.WeightingFilter(48000, "A"))
    between = result["between_nominals"]
    assert set(between) == {"worst_freq", "margin_class1_db", "margin_class2_db"}
    assert between["margin_class1_db"] >= 0.0
    wf = filters.WeightingFilter(48000, "A")
    with pytest.raises(ValueError, match=r"'sweep_points' must be at least"):
        filters.verify_weighting_class(wf, sweep_points=8)


@pytest.mark.parametrize(("fs", "expected"), [(32000, 2), (24000, 2)])
def test_plain_bilinear_degrades_to_class2(fs: int, expected: int) -> None:
    """Without oversampling the bilinear warping fails class 1 near Nyquist."""
    result = filters.verify_weighting_class(
        filters.WeightingFilter(fs, "A", high_accuracy=False)
    )
    assert result["overall_class"] == expected
    assert any(b["class"] != 1 for b in result["bands"])


def test_invalid_class_raises() -> None:
    with pytest.raises(ValueError, match="weighting_class must be"):
        filters.weighting_class_limits(0)
    with pytest.raises(ValueError, match="weighting_class must be"):
        filters.weighting_class_limits(3)


def _tone_gain_db(wf: filters.WeightingFilter, fs: int, f0: float) -> float:
    """Steady-state RMS gain of *wf* at f0 (time-domain, independent method)."""
    import numpy as np

    duration = max(0.5, 12 / f0)
    t = np.arange(int(fs * duration)) / fs
    x = np.sin(2 * np.pi * f0 * t)
    y = wf.filter(x)
    n0 = int(0.2 * fs)  # skip the transient
    return float(20 * np.log10(np.std(y[n0:]) / np.std(x[n0:])))


def test_deviation_matches_independent_tone_measurement() -> None:
    """Cross-check the closed-form deviations against a time-domain tone RMS.

    ``verify_weighting_class`` computes the response of the whole filter path
    in the frequency domain; this recomputes the deviation from a filtered
    pure tone (relative to the 1 kHz tone, matching the verifier's
    normalization), an independent method. They must agree to within the
    RMS-estimation error.
    """
    fs = 48000
    wf = filters.WeightingFilter(fs, "A")  # stateless, so it can be reused per tone
    bands = {
        b["freq"]: b["deviation_db"]
        for b in filters.verify_weighting_class(wf)["bands"]
    }
    ref_1k = _tone_gain_db(wf, fs, 1000.0)
    for f0 in (63.0, 250.0, 1000.0, 4000.0, 8000.0):
        measured = _tone_gain_db(wf, fs, f0) - ref_1k
        design = {row[0]: row[1] for row in _WEIGHTING_TABLE3}[f0]
        assert bands[f0] == pytest.approx(measured - design, abs=0.15), f"{f0} Hz"


def test_response_is_deterministic() -> None:
    """The response is computed in closed form, so repeated runs are identical."""
    a = filters.verify_weighting_class(filters.WeightingFilter(48000, "A"))
    b = filters.verify_weighting_class(filters.WeightingFilter(48000, "A"))
    assert [x["deviation_db"] for x in a["bands"]] == [
        x["deviation_db"] for x in b["bands"]
    ]

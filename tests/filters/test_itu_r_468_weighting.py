#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ITU-R BS.468-4 psophometric weighting as a time-domain filter.

Clause 1 makes the passive network of Fig. 1a the primitive -- "the nominal
response curve of the weighting network is given in Fig. 1b which is the
theoretical response of the passive network shown in Fig. 1a" -- and Table 1
"the values of this response at various frequencies". So there are two
separate things to check, and this module keeps them apart:

* the **analog prototype** against the 21 printed rows, which is the oracle
  the Recommendation offers (``reference_data.ITU_R_468_TABLE1``); and
* the **realised digital path** against the Table 1 tolerance column, at
  three sample rates, which is the mask a piece of measuring equipment has to
  sit inside.

One part of that mask cannot be read literally, and the departure is made
explicit below rather than assumed: the printed 0 at 6 300 Hz gets a
substitute budget, because nothing satisfies a zero-width tolerance. Nothing
else is excused. Every Table 1 row below the Nyquist frequency is checked at
every rate, including 32 kHz, which the resampled path this design replaced
could not reach: its anti-alias transition band sat on the input Nyquist
frequency and put the 20 kHz row 2.1 dB below the printed cell at 44.1 kHz,
against a +/-2.0 dB tolerance, so that row had to be dropped.
"""

from __future__ import annotations

import numpy as np
import pytest
import reference_data as ref
from scipy import signal as sp_signal

from phonometry import filters
from phonometry.electroacoustics import itu_r_468_weighting
from phonometry.filters.weighting import (
    _FIT_SECTIONS,
    _ITU_R_468_LADDER,
    _itu_r_468_prototype,
    _runtime_frequency_response,
)

#: Substitute for the 0 dB tolerance Table 1 prints at 6 300 Hz. The printed
#: value is unsatisfiable: the +/-1 % component tolerance the same figure
#: blesses already moves the peak by up to 0.11 dB, and AES17-2015 Table 1
#: quietly replaces the row with +/-0,01 dB. The fitted design lands within
#: 0.017 dB of the nominal peak at every rate here, so this stays the
#: library's own budget: the 0,1 dB quantum the table is printed to.
_PEAK_ROW_BUDGET_DB = ref.ITU_R_468_TABLE1_ROUNDING_DB


def _realised_response_db(fs: int, frequencies: np.ndarray) -> np.ndarray:
    """Response of the whole filter path, in dB re its own 1 kHz value."""
    wf = filters.WeightingFilter(fs, "468")
    h = _runtime_frequency_response(wf, frequencies)
    h_ref = _runtime_frequency_response(wf, np.array([1000.0]))[0]
    return np.asarray(20.0 * np.log10(np.abs(h) / abs(h_ref)))


def _demonstrable_rows(fs: int) -> list[tuple[float, float, float]]:
    """Table 1 rows this rate can carry, i.e. every one below Nyquist."""
    return [row for row in ref.ITU_R_468_TABLE1 if row[0] < fs / 2.0]


def _row_budget(tolerance: float) -> float:
    """A Table 1 tolerance a test can actually hold the filter to.

    Every row but one is the printed value; 6 300 Hz is the substitution
    :data:`_PEAK_ROW_BUDGET_DB` explains.
    """
    return tolerance if tolerance > 0.0 else _PEAK_ROW_BUDGET_DB


def test_ladder_is_the_seven_component_values_of_figure_1a() -> None:
    """The constants a reader checks against the printed figure.

    The point of building the prototype from components rather than from
    poles is that Fig. 1a can be read by eye and ``-23615.535214+36379.9j``
    cannot, so the transcription itself is worth pinning: five shunt
    capacitors, two series inductors and the one series capacitor, in ladder
    order, left to right.
    """
    assert _ITU_R_468_LADDER == (
        ("shunt_c", 13.85e-9),
        ("series_l", 12.88e-3),
        ("shunt_c", 26.82e-9),
        ("series_c", 33.06e-9),
        ("shunt_c", 9.21e-9),
        ("series_l", 26.49e-3),
        ("shunt_c", 31.47e-9),
    )


def test_prototype_is_one_zero_at_the_origin_and_six_stable_poles() -> None:
    """Seven reactive elements, order six, and nothing in the right half plane.

    ``C2``, ``C3`` and ``C4`` form a capacitive loop (two to ground with one
    between the nodes), so one of the seven states is dependent and the
    denominator is degree six. The network is passive and lossless-terminated,
    so every pole must sit strictly left of the imaginary axis, and the
    element values are real, so the pole set must be its own mirror image in
    that axis.
    """
    poles, gain = _itu_r_468_prototype()
    assert len(poles) == 6
    assert all(pole.real < 0.0 for pole in poles)
    assert gain > 0.0
    # Two real poles and two conjugate pairs, per the ladder's structure.
    real_poles = [pole for pole in poles if abs(pole.imag) < 1.0]
    assert len(real_poles) == 2
    ordered = sorted(poles, key=lambda v: (v.real, v.imag))
    mirrored = sorted((v.conjugate() for v in poles), key=lambda v: (v.real, v.imag))
    # 1e-6 rad/s against poles of order 1e5 is a relative 1e-11: tight enough
    # to catch an unpaired root, loose enough to survive another LAPACK build.
    assert np.allclose(ordered, mirrored, rtol=0.0, atol=1e-6)


def test_prototype_reproduces_all_21_rows_of_table_1() -> None:
    """The oracle the Recommendation offers, at the printed frequencies.

    Table 1 is the nominal curve rounded to 0,1 dB, so agreement to the
    rounding quantum is the strongest statement the printed table supports -
    and it is an independent one, because the network was evaluated from its
    component values while the table was transcribed from a different page.
    """
    freqs = np.array([row[0] for row in ref.ITU_R_468_TABLE1])
    printed = np.array([row[1] for row in ref.ITU_R_468_TABLE1])
    error = itu_r_468_weighting(freqs) - printed
    assert np.max(np.abs(error)) <= ref.ITU_R_468_NETWORK_VS_TABLE1_DB
    assert float(np.sqrt(np.mean(error**2))) == pytest.approx(0.0264, abs=0.001)


@pytest.mark.parametrize("fs", [32000, 44100, 48000, 96000])
def test_realised_filter_sits_inside_the_table_1_mask(fs: int) -> None:
    """Table 1 column 3, at every row below the Nyquist frequency.

    The mask constrains the measuring equipment's departure from the nominal
    curve, so the comparison is against the printed cell the reader has, with
    the one documented departure from a literal reading: the 6 300 Hz row's
    unsatisfiable 0 is replaced. 32 kHz is in the list because the resampled
    path could not carry it -- its 14 kHz top row sat inside the anti-alias
    transition band -- and the fitted design reads 0.03 dB there against a
    +/-0.2 dB tolerance.
    """
    rows = _demonstrable_rows(fs)
    freqs = np.array([row[0] for row in rows])
    realised = _realised_response_db(fs, freqs)
    for (frequency, printed, tolerance), value in zip(rows, realised, strict=True):
        limit = _row_budget(tolerance)
        assert abs(value - printed) < limit, (
            f"{frequency:g} Hz: {value - printed:+.4f} dB against +/-{limit} dB"
        )


@pytest.mark.parametrize("fs", [32000, 44100, 48000, 96000])
def test_realised_filter_tracks_the_network_far_inside_the_mask(fs: int) -> None:
    """The digital path against the nominal curve, not against its rounding.

    The mask above is what conformance asks for; a digital implementation
    should be held to something far tighter, and the natural way to say so is
    as a fraction of each row's own budget, which stays comparable across
    rates while the absolute departure does not. The resampled path this
    design replaced spent 100 % of the 20 kHz row's budget at 44.1 kHz and had
    to have that row excused; the fit spends at most half of any row's budget
    at any rate here. The binding row is 6 300 Hz at 32 kHz, and it binds
    because its budget is the tightest in the table by a factor of four -- the
    0.05 dB substitute for an unsatisfiable printed zero -- not because the
    departure there is large: 0.024 dB, the smallest absolute departure of any
    binding row in this file.
    """
    rows = _demonstrable_rows(fs)
    freqs = np.array([row[0] for row in rows])
    budget = np.array([_row_budget(row[2]) for row in rows])
    departure = _realised_response_db(fs, freqs) - itu_r_468_weighting(freqs)
    assert np.max(np.abs(departure) / budget) < 0.5


def test_the_20_khz_row_at_44_1_khz_is_no_longer_a_problem() -> None:
    """The row the resampled path had to be excused from, measured.

    At 44.1 kHz the 20 kHz row sits at 0.907 of the Nyquist frequency. The
    interpolate / filter / decimate path put it 2.10 dB below the printed cell
    against a +/-2.0 dB tolerance -- of which the two anti-alias passes were
    -1.66 dB and the sections only -0.49 -- so the row had to be dropped from
    the mask test. The fit controls the band to 0.995 of Nyquist, so the row is
    inside what the design covers and reads within a fiftieth of its budget.
    """
    at_20k = np.array([20000.0])
    printed = next(row[1] for row in ref.ITU_R_468_TABLE1 if row[0] == 20000.0)

    assert 20000.0 in [row[0] for row in _demonstrable_rows(44100)]
    deviation = _realised_response_db(44100, at_20k)[0] - printed
    assert abs(deviation) < 0.05, f"{deviation:+.4f} dB against +/-2.0 dB"

    # The sections are the whole path now, so measuring them separately must
    # give the same number rather than a better one.
    wf = filters.WeightingFilter(44100, "468")
    _, h = sp_signal.sosfreqz(wf.sos, worN=at_20k, fs=44100)
    _, h_ref = sp_signal.sosfreqz(wf.sos, worN=np.array([1000.0]), fs=44100)
    sections_only = 20.0 * np.log10(abs(h[0]) / abs(h_ref[0])) - printed
    assert sections_only == pytest.approx(deviation, abs=1e-12)


def test_the_468_curve_keeps_the_network_order() -> None:
    """Three biquads, the six poles of Fig. 1a, and no fourth section.

    A fourth section is worth buying only when the prototype's own order is
    what limits the fit. It is not here: the residual is set by the magnitude
    of a real-coefficient filter having zero slope at the Nyquist frequency
    while this curve is still falling at about -30 dB/octave, and a fourth
    section moves the worst deviation at 48 kHz only from 0.102 to 0.083 dB.
    So the realised filter carries exactly the network's degree.
    """
    assert _FIT_SECTIONS["468"] == 3
    assert filters.WeightingFilter(48000, "468").sos.shape == (3, 6)
    assert len(_itu_r_468_prototype()[0]) == 6


def test_weighting_filter_emphasises_the_6_khz_band_of_a_record() -> None:
    """End to end: what the filter does to a waveform, which is the point.

    The tree could weight a spectrum before this curve became a filter but
    could not weight a signal. Two equal tones in, one at the 1 kHz reference
    and one at the 6.3 kHz peak, and the ratio of their amplitudes out must be
    the +12.2 dB the network puts between them.
    """
    fs = 48000
    t = np.arange(fs) / fs
    x = np.sin(2 * np.pi * 1000.0 * t) + np.sin(2 * np.pi * 6300.0 * t)
    y = np.asarray(filters.weighting_filter(x, fs, curve="468"))
    spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    freqs = np.fft.rfftfreq(y.size, 1.0 / fs)
    at_1k = spectrum[int(np.argmin(np.abs(freqs - 1000.0)))]
    at_6k3 = spectrum[int(np.argmin(np.abs(freqs - 6300.0)))]
    assert 20.0 * np.log10(at_6k3 / at_1k) == pytest.approx(12.2167, abs=0.05)


def test_stateful_468_is_now_available() -> None:
    """The refusal was about the resampler, and the resampler is gone.

    Block processing used to imply ``high_accuracy=False``, which for this
    curve meant the plain bilinear design and 23 dB of error at 16 kHz, so it
    was refused rather than shipped. The fitted design is a plain cascade at
    the input rate, so block processing costs this curve nothing -- and must
    reproduce a single call exactly.
    """
    fs = 48000
    x = np.random.default_rng(468).standard_normal(fs)
    blocks = filters.WeightingFilter(fs, "468", stateful=True)
    stitched = np.concatenate([blocks.filter(part) for part in np.split(x, 5)])
    np.testing.assert_array_equal(
        stitched, np.asarray(filters.WeightingFilter(fs, "468").filter(x))
    )


def test_plain_bilinear_468_is_refused() -> None:
    """``high_accuracy=False`` is the design the Recommendation cannot grade.

    Every other curve degrades gracefully and is documented rather than
    refused, because IEC 61672-1 grades A and C into classes. BS.468-4 prints
    one mask and no lower grade, so there is nothing honest to return here.
    """
    with pytest.raises(ValueError, match=r"'468' needs the fitted design"):
        filters.WeightingFilter(48000, "468", high_accuracy=False)


def test_verify_weighting_class_still_refuses_468() -> None:
    """It returns an IEC 61672-1 performance class, and 468 has none.

    Table 1 gives this curve a tolerance mask but no graded classes, exactly
    like D and G, so it stays outside the class verifier rather than being
    given a class it does not have.
    """
    wf = filters.WeightingFilter(48000, "468")
    with pytest.raises(ValueError, match=r"Weighting curve must be .*'AU' or 'Z'"):
        filters.verify_weighting_class(wf)

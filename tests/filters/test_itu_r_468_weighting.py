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

Two parts of that mask cannot be read literally, and both departures are
made explicit below rather than assumed: the printed 0 at 6 300 Hz gets a
substitute budget, because nothing satisfies a zero-width tolerance, and any
row inside the resampler's anti-alias transition band near the input Nyquist
frequency is dropped, because every curve in this module rolls off there.
The dropped row is not simply forgotten: a test of its own pins what it
costs and where the cost comes from.
"""

from __future__ import annotations

import numpy as np
import pytest
import reference_data as ref
from scipy import signal as sp_signal

from phonometry import filters
from phonometry.electroacoustics import itu_r_468_weighting
from phonometry.filters.weighting import (
    _ITU_R_468_LADDER,
    _itu_r_468_prototype,
    _runtime_frequency_response,
)

#: Above this fraction of the input Nyquist frequency the ``resample_poly``
#: anti-alias FIR of the high-accuracy path dominates, whatever the design
#: rate: its cutoff sits on ``fs / 2``, so the transition band straddles it,
#: and the signal crosses it twice. Measured on this curve at 44.1 kHz, the
#: 20 kHz row (0.907 of Nyquist) reads -2.13 dB against the network, of which
#: the two FIR passes are -1.66 dB and the sections only -0.49 dB. That -1.66
#: does not move with the oversample factor: the tap count grows as 20 L while
#: the normalised cutoff falls as 1/L, so the transition band in hertz is
#: invariant (-3 dB at 21.17 kHz for every L from 2 to 16). Raising L improves
#: the row, from -10.4 dB at x2 to -1.7 dB at x16, and every decibel of that is
#: the sections decompressing; the factor is already at the module's cap of 8
#: here, so there is no headroom left to spend. The A curve loses 2.25 dB at
#: the same point for the same reason, so this is the module's shared range
#: limit and not a property of this curve.
_DEMONSTRABLE_FRACTION_OF_NYQUIST = 0.9

#: Substitute for the 0 dB tolerance Table 1 prints at 6 300 Hz. The printed
#: value is unsatisfiable: the +/-1 % component tolerance the same figure
#: blesses already moves the peak by up to 0.11 dB, and AES17-2015 Table 1
#: quietly replaces the row with +/-0,01 dB. Even that is out of reach for a
#: resampled digital path, which lands 0.010 to 0.013 dB from the nominal
#: peak depending on the rate, so this is the library's own budget: the
#: 0,1 dB quantum the table is printed to.
_PEAK_ROW_BUDGET_DB = ref.ITU_R_468_TABLE1_ROUNDING_DB


def _realised_response_db(fs: int, frequencies: np.ndarray) -> np.ndarray:
    """Response of the whole filter path, in dB re its own 1 kHz value.

    The high-accuracy path does not apply its sections to the input directly,
    so the sections alone are not the filter; this measures the interpolate /
    filter / decimate cascade the caller actually gets.
    """
    wf = filters.WeightingFilter(fs, "468")
    h = _runtime_frequency_response(wf, frequencies)
    h_ref = _runtime_frequency_response(wf, np.array([1000.0]))[0]
    return np.asarray(20.0 * np.log10(np.abs(h) / abs(h_ref)))


def _demonstrable_rows(fs: int) -> list[tuple[float, float, float]]:
    """Table 1 rows that lie below this rate's anti-alias transition band."""
    ceiling = _DEMONSTRABLE_FRACTION_OF_NYQUIST * fs / 2.0
    return [row for row in ref.ITU_R_468_TABLE1 if row[0] < ceiling]


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


@pytest.mark.parametrize("fs", [44100, 48000, 96000])
def test_realised_filter_sits_inside_the_table_1_mask(fs: int) -> None:
    """Table 1 column 3, at every row this rate can demonstrate.

    The mask constrains the measuring equipment's departure from the nominal
    curve, so the comparison is against the printed cell the reader has, with
    the two documented departures from a literal reading: the 6 300 Hz row's
    unsatisfiable 0 is replaced, and the rows inside the anti-alias transition
    band are dropped rather than held to a tolerance no rate can meet there.
    """
    rows = _demonstrable_rows(fs)
    freqs = np.array([row[0] for row in rows])
    realised = _realised_response_db(fs, freqs)
    for (frequency, printed, tolerance), value in zip(rows, realised, strict=True):
        limit = _row_budget(tolerance)
        assert abs(value - printed) < limit, (
            f"{frequency:g} Hz: {value - printed:+.4f} dB against +/-{limit} dB"
        )


@pytest.mark.parametrize("fs", [44100, 48000, 96000])
def test_realised_filter_tracks_the_network_far_inside_the_mask(fs: int) -> None:
    """The digital path against the nominal curve, not against its rounding.

    The mask above is what conformance asks for; a digital implementation
    should be held to something far tighter, and the natural way to say so is
    as a fraction of each row's own budget, which stays comparable across
    rates while the absolute departure does not (it reaches 0.97 dB at the
    31.5 kHz row, inside a +/-2.8 dB one). This is what the 384 kHz design
    target buys: it spends 16 % of the 16 kHz row's +/-1.6 dB at 48 kHz, where
    the module's 144 kHz default would read 1.99 dB low and be outside it.
    """
    rows = _demonstrable_rows(fs)
    freqs = np.array([row[0] for row in rows])
    budget = np.array([_row_budget(row[2]) for row in rows])
    departure = _realised_response_db(fs, freqs) - itu_r_468_weighting(freqs)
    assert np.max(np.abs(departure) / budget) < 0.4


def test_the_20_khz_row_is_out_of_reach_at_44_1_khz_but_not_at_48() -> None:
    """The exclusion above, stated as a measurement instead of an assumption.

    A row nobody checks is a place to hide a regression, so the reason for
    dropping it is pinned here. At 44.1 kHz the 20 kHz row sits at 0.91 of
    Nyquist, inside the resampler's anti-alias transition band, and the
    realised response falls about 2.1 dB below the printed cell against a
    +/-2.0 dB tolerance. At 48 kHz the same row is at 0.83 and costs 0.4 dB.
    The deficit belongs to the resampling path and not to the design: the
    second-order sections alone are 0.5 dB low there, and a higher design
    rate would make it worse, because the anti-alias FIR keeps its cutoff at
    the input Nyquist frequency and only gets sharper.
    """
    at_20k = np.array([20000.0])
    printed = next(row[1] for row in ref.ITU_R_468_TABLE1 if row[0] == 20000.0)

    assert 20000.0 not in [row[0] for row in _demonstrable_rows(44100)]
    deficit_44k1 = _realised_response_db(44100, at_20k)[0] - printed
    assert deficit_44k1 == pytest.approx(-2.10, abs=0.05)

    assert 20000.0 in [row[0] for row in _demonstrable_rows(48000)]
    assert _realised_response_db(48000, at_20k)[0] - printed == pytest.approx(
        -0.40, abs=0.05
    )

    # The sections alone, without the interpolate / decimate pair around them.
    wf = filters.WeightingFilter(44100, "468")
    _, h = sp_signal.sosfreqz(wf.sos, worN=at_20k, fs=44100 * wf._oversample)
    _, h_ref = sp_signal.sosfreqz(
        wf.sos, worN=np.array([1000.0]), fs=44100 * wf._oversample
    )
    sections_only = 20.0 * np.log10(abs(h[0]) / abs(h_ref[0])) - printed
    assert sections_only == pytest.approx(-0.49, abs=0.05)
    assert sections_only > deficit_44k1 + 1.0


def test_design_target_is_384_khz_not_the_module_default() -> None:
    """A -30 dB/octave skirt needs eight times oversampling at 48 kHz.

    Bilinear frequency compression grows quadratically with ``f / f_design``,
    so the 144 kHz target sized for the A and C curves is not enough here.
    The factor is capped at the module's existing 8, which 44.1 kHz reaches:
    the target would ask for 9 and the cap holds it at 352.8 kHz, close
    enough that only the row inside the transition band suffers.
    """
    assert filters.WeightingFilter(48000, "468")._oversample == 8
    assert filters.WeightingFilter(44100, "468")._oversample == 8
    assert filters.WeightingFilter(96000, "468")._oversample == 4
    assert filters.WeightingFilter(384000, "468")._oversample == 1


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


def test_stateful_468_is_refused() -> None:
    """Block processing runs the plain design at the input rate: 23 dB out.

    Every other curve degrades gracefully and is documented rather than
    refused, because IEC 61672-1 grades A and C into classes. BS.468-4 prints
    one mask and no lower grade, so there is nothing honest to return here.
    """
    with pytest.raises(ValueError, match=r"'468' needs the oversampled design"):
        filters.WeightingFilter(48000, "468", stateful=True)


def test_plain_bilinear_468_is_refused() -> None:
    """``high_accuracy=False`` is the same filter stateful mode would build."""
    with pytest.raises(ValueError, match=r"'468' needs the oversampled design"):
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

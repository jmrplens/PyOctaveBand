#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ITU-R BS.468-4 clause 2: the psophometric quasi-peak programme-level meter.

Clause 2 prints no time constant, no rise time and no transfer function. It
specifies the detector through eleven acceptance windows (Tables 2 and 3),
one absolute calibration (clause 2.6) and four small requirements that a
float64 chain satisfies by construction. This file is organised the same
way, and each group says which of the three it is:

* **Conformance.** The eleven windows, run at five sample rates. This is the
  whole conformance statement the Recommendation authorises for a detector.
* **A self-imposed regression bound.** How close the readings sit to the
  *reference* column, which the standard does not require them to match, and
  how little they move between sample rates.
* **True by construction.** Clauses 2.1 (attenuators), 2.3 (overload
  linearity), 2.4 (reversibility) and 2.5 (overswing). Rectification,
  first-order recursions and a maximum are positively homogeneous and cannot
  overshoot a monotone step, so these pass exactly rather than nearly. They
  are kept as live tests because the first three would catch a chain that
  stopped being homogeneous and the fourth would catch a second-order
  reading device added for realism.

The windows themselves are transcribed in ``tests/reference_data/broadcast``
from the rendered page, independently of the copy the module carries.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from reference_data import (
    BS468_BURST_HZ,
    BS468_CALIBRATION_V,
    BS468_OVERLOAD_BURST_MS,
    BS468_OVERLOAD_RANGE_DB,
    BS468_OVERLOAD_TOL_DB,
    BS468_OVERSWING_TOL_DB,
    BS468_REVERSIBILITY_TOL_DB,
    BS468_TABLE2_SINGLE_BURSTS,
    BS468_TABLE3_BURST_TRAINS,
)

from phonometry.broadcast import (
    BS468_BALLISTICS,
    DBQPS_REFERENCE,
    QuasiPeakBallistics,
    QuasiPeakResult,
    quasi_peak_meter,
    verify_quasi_peak_dynamics,
)
from phonometry.io import Signal
from phonometry.signals import tone_burst

FS = 48000.0

#: The rates the ballistics are checked over. 32 kHz is the lowest broadcast
#: rate, 192 kHz the highest this library designs the network at, and
#: 44.1 kHz is the one where the 25-cycle burst is not sample-exact.
RATES = (32000.0, 44100.0, FS, 96000.0, 192000.0)

#: Self-imposed, not from BS.468-4: how far a reading may sit from the
#: *reference* reading printed in Tables 2 and 3. The worst of the eleven is
#: 0.140 dB, at 10 bursts per second. The reference is printed to two
#: significant figures on nine of the eleven cells, so its own quantum is
#: 0.027 to 0.108 dB and nothing tighter than this would be meaningful.
REFERENCE_BOUND_DB = 0.15

#: Self-imposed: how much of its own acceptance window a reading may spend on
#: moving between sample rates. The worst of the eleven is 4.22 %, at 100
#: bursts per second, whose window is the narrowest in the document.
RATE_SPREAD_FRACTION = 0.05


def _steady_sine(rms: float, fs: float, seconds: float = 3.0) -> np.ndarray:
    """A settled sine of *rms* volts at 1 kHz, the clause 2.6 stimulus."""
    t = np.arange(int(round(seconds * fs))) / fs
    return math.sqrt(2.0) * rms * np.sin(2.0 * math.pi * 1000.0 * t)


@pytest.fixture(scope="module")
def dynamics() -> dict[float, dict[str, object]]:
    """The eleven windows at every rate, computed once for the whole module.

    44.1 kHz warns from ``tone_burst``: 25 cycles of 5 kHz span 220.5 samples
    there, so the gate cannot close on the tone's final zero crossing. The
    warning is correct and the consequence measures 0.006 dB against a
    2.626 dB window, which
    :func:`test_a_reading_barely_moves_between_sample_rates` is what checks.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*incommensurate with fs.*")
        return {fs: verify_quasi_peak_dynamics(fs) for fs in RATES}


# ---------------------------------------------------------------------------
# Conformance: the eleven acceptance windows of Tables 2 and 3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("duration_ms", "lower", "upper"),
    [(d, lo, hi) for d, _cycles, lo, _ref, hi in BS468_TABLE2_SINGLE_BURSTS],
    ids=[f"{d:g}ms" for d, *_ in BS468_TABLE2_SINGLE_BURSTS],
)
def test_a_single_burst_lands_inside_its_table_2_window(
    dynamics: dict[float, dict[str, object]],
    duration_ms: float,
    lower: float,
    upper: float,
) -> None:
    """Table 2: one 5 kHz burst of an integral number of periods, weighted."""
    rows = dynamics[FS]["stimuli"]
    assert isinstance(rows, list)
    row = next(r for r in rows if r["stimulus"] == f"{duration_ms:g} ms")
    assert lower <= row["reading_percent"] <= upper


@pytest.mark.parametrize(
    ("rate", "lower", "upper"),
    [(r, lo, hi) for r, lo, _ref, hi in BS468_TABLE3_BURST_TRAINS],
    ids=[f"{r:g}per_s" for r, *_ in BS468_TABLE3_BURST_TRAINS],
)
def test_a_burst_train_lands_inside_its_table_3_window(
    dynamics: dict[float, dict[str, object]],
    rate: float,
    lower: float,
    upper: float,
) -> None:
    """Table 3: 5 ms bursts at 2, 10 and 100 per second, the decay constraint."""
    rows = dynamics[FS]["stimuli"]
    assert isinstance(rows, list)
    row = next(r for r in rows if r["stimulus"] == f"{rate:g} bursts/s")
    assert lower <= row["reading_percent"] <= upper


@pytest.mark.parametrize("fs", RATES, ids=[f"{f:g}Hz" for f in RATES])
def test_every_window_is_met_at_every_sample_rate(
    dynamics: dict[float, dict[str, object]], fs: float
) -> None:
    """The verifier's own verdict, at each rate, with the margin it reports."""
    report = dynamics[fs]
    assert report["passed"]
    assert report["fs"] == fs
    margin = report["worst_margin_db"]
    assert isinstance(margin, float)
    assert margin > 0.0


def test_the_verifier_reports_all_eleven_stimuli(
    dynamics: dict[float, dict[str, object]],
) -> None:
    """Eight single bursts and three trains, each with its printed window."""
    rows = dynamics[FS]["stimuli"]
    assert isinstance(rows, list)
    assert len(rows) == len(BS468_TABLE2_SINGLE_BURSTS) + len(BS468_TABLE3_BURST_TRAINS)
    printed = {
        f"{d:g} ms": (lo, ref, hi)
        for d, _cycles, lo, ref, hi in BS468_TABLE2_SINGLE_BURSTS
    }
    printed |= {
        f"{r:g} bursts/s": (lo, ref, hi) for r, lo, ref, hi in BS468_TABLE3_BURST_TRAINS
    }
    for row in rows:
        lower, reference, upper = printed[row["stimulus"]]
        assert (
            row["lower_percent"],
            row["reference_percent"],
            row["upper_percent"],
        ) == (
            lower,
            reference,
            upper,
        )


# ---------------------------------------------------------------------------
# Self-imposed regression bounds: the reference column and the sample rate
# ---------------------------------------------------------------------------


def test_the_readings_stay_near_the_printed_reference_column(
    dynamics: dict[float, dict[str, object]],
) -> None:
    """Not conformance: BS.468-4 requires the window, not the reference.

    Pinned anyway, because the windows are 0.5 to 4.0 dB wide and a refit
    that moved a reading by a decibel would still pass every one of them.
    """
    worst = dynamics[FS]["worst_deviation_db"]
    assert isinstance(worst, float)
    assert worst < REFERENCE_BOUND_DB


@pytest.mark.parametrize(
    "stimulus",
    [f"{d:g} ms" for d, *_ in BS468_TABLE2_SINGLE_BURSTS]
    + [f"{r:g} bursts/s" for r, *_ in BS468_TABLE3_BURST_TRAINS],
)
def test_a_reading_barely_moves_between_sample_rates(
    dynamics: dict[float, dict[str, object]], stimulus: str
) -> None:
    """The property the whole discretisation rests on, per stimulus.

    A peak follower on raw rectified samples reads the largest *sample*, not
    the largest value of the waveform, and that deficit is rate dependent (a
    5 kHz sine can be read 0.47 dB low at 48 kHz). This chain never takes
    such a peak: the 1.41 ms charge spans fourteen half-periods of the
    rectified carrier, so the ripple is gone before any maximum is taken.
    """
    readings = []
    window_db = 0.0
    for fs in RATES:
        rows = dynamics[fs]["stimuli"]
        assert isinstance(rows, list)
        row = next(r for r in rows if r["stimulus"] == stimulus)
        readings.append(row["reading_percent"])
        window_db = 20.0 * math.log10(row["upper_percent"] / row["lower_percent"])
    spread_db = 20.0 * math.log10(max(readings) / min(readings))
    assert spread_db < RATE_SPREAD_FRACTION * window_db


# ---------------------------------------------------------------------------
# Clause 2.6: the one absolute statement in clause 2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fs", RATES, ids=[f"{f:g}Hz" for f in RATES])
def test_a_steady_1_khz_sine_at_0_775_volts_reads_0_dbqps(fs: float) -> None:
    """Clause 2.6, the sentence that fixes the scale of the whole instrument.

    It is an equality, not a tolerance, so it is checked as one: the scale
    factor is measured per rate rather than frozen at 48 kHz, which is what
    keeps this exact at 32 and 192 kHz too.
    """
    result = quasi_peak_meter(_steady_sine(BS468_CALIBRATION_V, fs), fs)
    assert result.reading == pytest.approx(BS468_CALIBRATION_V, abs=1e-12)
    assert result.level_db == pytest.approx(0.0, abs=1e-9)
    assert result.level_unit == "dBqps"


def test_the_detector_reads_the_rms_of_a_steady_sine_not_its_peak() -> None:
    """The consequence of clause 2.6 an implementer trips over first."""
    peak = 2.0
    result = quasi_peak_meter(_steady_sine(peak / math.sqrt(2.0), FS), FS)
    assert result.reading == pytest.approx(peak / math.sqrt(2.0), rel=1e-9)


# ---------------------------------------------------------------------------
# True by construction: clauses 2.1, 2.3, 2.4 and 2.5
# ---------------------------------------------------------------------------


def test_scaling_the_input_scales_the_reading_by_the_same_factor() -> None:
    """Clause 2.1's two attenuator variants, which cannot disagree here.

    The tests are run "without adjusting the attenuators" and again with them
    reset per duration. A chain of rectification, first-order recursions and
    a maximum is positively homogeneous, so the two variants are the same
    measurement and "80 % of full scale" constrains the bench, not the
    quantity. That is also why this module models no full scale at all.
    """
    burst = tone_burst(FS, BS468_BURST_HZ, 250, post_silence=0.5).signal
    plain = quasi_peak_meter(burst, FS).reading
    scaled = quasi_peak_meter(3.7 * burst, FS).reading
    assert scaled == pytest.approx(3.7 * plain, rel=1e-12)


def test_the_reading_tracks_over_20_db_of_short_bursts() -> None:
    """Clause 2.3: isolated 0.6 ms bursts stepped down 20 dB, within +-1 dB.

    The only clause that exercises a burst shorter than Table 2's shortest,
    and the only one that reaches three whole carrier periods. In float64 it
    passes on the same homogeneity as clause 2.1, so the residual is rounding
    and the +-1 dB allowance is spent to fourteen decimal places.

    Three periods of 5 kHz span 28.8 samples at 48 kHz, the one stimulus in
    the Recommendation whose gate cannot close on a zero crossing there
    (every Table 2 and Table 3 duration is a multiple of five periods and so
    is sample-exact). ``tone_burst`` says so, and it makes no difference to a
    test that only compares scaled copies of the very same burst.
    """
    cycles = round(BS468_OVERLOAD_BURST_MS * 1e-3 * BS468_BURST_HZ)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*incommensurate with fs.*")
        burst = tone_burst(FS, BS468_BURST_HZ, cycles, post_silence=0.5).signal
    top = quasi_peak_meter(burst, FS).level_db
    for step_db in np.linspace(0.0, BS468_OVERLOAD_RANGE_DB, 5):
        attenuated = quasi_peak_meter(10.0 ** (-step_db / 20.0) * burst, FS).level_db
        assert abs((top - attenuated) - step_db) < BS468_OVERLOAD_TOL_DB
        assert (top - attenuated) == pytest.approx(step_db, abs=1e-12)


def test_reversing_the_polarity_leaves_the_reading_unchanged() -> None:
    """Clause 2.4: 1 ms d.c. pulses at 100 per second, in the unweighted mode.

    The only test the Recommendation runs without the weighting network, and
    for the right reason: the network has a zero at the origin and would kill
    a d.c. pulse train. Full-wave rectification as ``abs(x)`` makes the
    difference exactly zero rather than merely under the 0.5 dB limit.
    """
    n = int(round(0.5 * FS))
    pulses = np.zeros(n)
    period = int(round(FS / 100.0))
    width = int(round(1e-3 * FS))
    for start in range(0, n - period, period):
        pulses[start : start + width] = 0.8
    forward = quasi_peak_meter(pulses, FS, weighted=False).level_db
    reversed_ = quasi_peak_meter(-pulses, FS, weighted=False).level_db
    assert abs(forward - reversed_) < BS468_REVERSIBILITY_TOL_DB
    assert forward == reversed_


def test_a_suddenly_applied_tone_does_not_overswing() -> None:
    """Clause 2.5: less than 0.3 dB of momentary excess reading.

    A cascade of first-order elements driven by a monotone step cannot
    overshoot, so the whole allowance goes unspent. The test earns its keep
    the day anyone gives the reading device a second order for realism.
    """
    tone = _steady_sine(BS468_CALIBRATION_V, FS)
    result = quasi_peak_meter(np.concatenate([np.zeros(int(FS // 10)), tone]), FS)
    # The settled indication is the needle over the last second, ripple and
    # all; the momentary reading is its largest excursion over the whole
    # record, transient included. Anything the transient adds shows up as the
    # difference between the two.
    settled = float(result.trace[-int(FS) :].max())
    excess_db = 20.0 * math.log10(result.reading / settled)
    assert excess_db < BS468_OVERSWING_TOL_DB
    assert excess_db == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# The result object
# ---------------------------------------------------------------------------


def test_the_trace_is_the_needle_over_the_record_the_caller_gave() -> None:
    """One sample of trace per sample of record, and the reading is its peak.

    The record is padded with silence before the weighting network and
    un-padded after, so the trace the caller gets back covers their record
    and nothing else.
    """
    burst = tone_burst(FS, BS468_BURST_HZ, 500, post_silence=0.5).signal
    result = quasi_peak_meter(burst, FS)
    assert result.trace.shape == burst.shape
    assert result.time.size == burst.size
    assert result.reading == float(result.trace.max())
    assert result.fs == FS
    assert result.weighted


def test_the_reading_does_not_depend_on_where_the_record_ends() -> None:
    """What the padding before the weighting network is there to guarantee.

    The polyphase resampler's last output sample sits on an incomplete
    window and comes out 5.1 % above the settled peak of a weighted 5 kHz
    tone. Cutting a settled record at four consecutive samples therefore
    hands the detector four different final values, and the reading must not
    be able to tell: it is a measurement of the record, not of its last
    sample. The 1.41 ms charge already absorbs almost all of that on its own,
    which is why the tolerance here is tight rather than generous.
    """
    t = np.arange(int(3.0 * FS)) / FS
    tone = np.sin(2.0 * math.pi * BS468_BURST_HZ * t)
    cut = tone.size - 4
    readings = [quasi_peak_meter(tone[: cut + k], FS).reading for k in range(4)]
    assert max(readings) - min(readings) < 1e-7 * min(readings)


def test_switching_out_the_weighting_network_changes_the_reading() -> None:
    """Clause 2's preamble: every test but 2.4 runs through the network."""
    burst = tone_burst(FS, BS468_BURST_HZ, 25, post_silence=0.5).signal
    weighted = quasi_peak_meter(burst, FS)
    unweighted = quasi_peak_meter(burst, FS, weighted=False)
    assert weighted.weighted
    assert not unweighted.weighted
    # The network is +11.7 dB at 5 kHz, so the weighted reading is far larger.
    assert weighted.reading > 3.0 * unweighted.reading


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        (DBQPS_REFERENCE, "dBqps"),
        (2e-5, "dB re 20 uPa"),
        (1.0, "dB re 1"),
    ],
)
def test_the_level_names_the_scale_it_is_on(reference: float, expected: str) -> None:
    """Only 0.775 V earns the name dBqps; clause 3 gives no other scale one."""
    result = quasi_peak_meter(
        _steady_sine(0.775, FS, seconds=1.0), FS, reference=reference
    )
    assert result.level_unit == expected


def test_a_silent_record_reads_minus_infinity() -> None:
    """No excursion at all is a level of ``-inf``, not a division by zero."""
    result = quasi_peak_meter(np.zeros(4800), FS)
    assert result.reading == 0.0
    assert result.level_db == -math.inf


def test_the_result_is_frozen() -> None:
    """A measurement is a record of what happened, not a mutable buffer."""
    result = quasi_peak_meter(_steady_sine(0.775, FS, seconds=0.5), FS)
    with pytest.raises(FrozenInstanceError):
        result.reading = 1.0  # type: ignore[misc]


def test_the_plot_draws_the_trace_and_marks_the_reading() -> None:
    """The figure is the point of the result: a scalar hides the ballistics."""
    burst = tone_burst(FS, BS468_BURST_HZ, 250, post_silence=0.5).signal
    result = quasi_peak_meter(burst, FS)
    ax = result.plot()
    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Quasi-peak reading"
    assert ax.get_title() == "Quasi-peak reading (ITU-R BS.468-4, weighted)"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(label.startswith("Reading") and "dBqps" in label for label in labels)
    ax_es = result.plot(language="es")
    assert ax_es.get_xlabel() == "Tiempo [s]"
    assert ax_es.get_title() == "Lectura cuasipico (UIT-R BS.468-4, ponderada)"
    labels_es = [t.get_text() for t in ax_es.get_legend().get_texts()]
    assert any(label.startswith("Lectura ") for label in labels_es)


def test_the_plot_refuses_a_language_it_cannot_render() -> None:
    """The label table has two entries and says so."""
    result = quasi_peak_meter(_steady_sine(0.775, FS, seconds=0.5), FS)
    with pytest.raises(ValueError, match=r"Unknown language 'fr'"):
        result.plot(language="fr")


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_an_empty_record_is_refused() -> None:
    """A meter with nothing to read has no reading, not a reading of zero."""
    with pytest.raises(ValueError, match=r"'x' cannot be empty"):
        quasi_peak_meter(np.array([]), FS)


def test_a_multichannel_record_is_refused() -> None:
    """One needle, one channel: mixing them would invent a measurement."""
    with pytest.raises(ValueError, match=r"x must be a 1-D time series"):
        quasi_peak_meter(np.zeros((2, 4800)), FS)


def test_a_bare_array_still_needs_its_rate() -> None:
    """The ballistics are in seconds; an array on its own has no time axis."""
    with pytest.raises(ValueError, match=r"fs is required when 'x' is a bare array"):
        quasi_peak_meter(np.zeros(4800))


def test_a_non_positive_rate_is_refused() -> None:
    """A rate of zero divides the time constants by nothing."""
    with pytest.raises(ValueError, match=r"'fs' must be positive"):
        quasi_peak_meter(np.zeros(4800), 0.0)


def test_a_non_positive_reference_is_refused() -> None:
    """A level is a ratio; the denominator has to be a level to divide by."""
    with pytest.raises(ValueError, match=r"'reference' must be positive"):
        quasi_peak_meter(np.zeros(4800), FS, reference=0.0)


def test_the_verifier_refuses_a_non_positive_rate() -> None:
    """The stimuli are built at this rate before anything is measured."""
    with pytest.raises(ValueError, match=r"'fs' must be positive"):
        verify_quasi_peak_dynamics(-1.0)


def test_a_result_whose_reading_is_not_its_trace_peak_is_refused() -> None:
    """The reading rule is the maximum of the needle, so it cannot disagree."""
    result = quasi_peak_meter(_steady_sine(0.775, FS, seconds=0.5), FS)
    with pytest.raises(ValueError, match=r"'reading' is .* but 'trace' peaks at"):
        replace(result, reading=result.reading * 2.0)


def test_a_two_dimensional_trace_is_refused() -> None:
    """The needle is one series; an extra axis would reach the plot intact."""
    with pytest.raises(
        ValueError, match=r"QuasiPeakResult: 'trace' must have one axis"
    ):
        QuasiPeakResult(
            reading=1.0,
            level_db=0.0,
            reference=DBQPS_REFERENCE,
            trace=np.ones((2, 3)),
            fs=FS,
            weighted=True,
        )


# ---------------------------------------------------------------------------
# The Signal contract
# ---------------------------------------------------------------------------


def test_an_uncalibrated_signal_reads_exactly_like_the_bare_array() -> None:
    """With no factor to apply, the object and the array are one recording."""
    record = _steady_sine(0.775, FS, seconds=0.5)
    from_signal = quasi_peak_meter(Signal(record, int(FS)))
    from_array = quasi_peak_meter(record, FS)
    assert from_signal.reading == from_array.reading
    assert from_signal.level_db == from_array.level_db


def test_a_calibrated_signal_is_read_in_pascals() -> None:
    """The library's one rule holds here: the factor scales the samples."""
    record = _steady_sine(0.775, FS, seconds=0.5)
    factor = 5.0
    from_signal = quasi_peak_meter(
        Signal(record, int(FS), calibration_factor=factor), reference=2e-5
    )
    from_array = quasi_peak_meter(factor * record, FS, reference=2e-5)
    assert from_signal.reading == from_array.reading
    assert from_signal.level_db == from_array.level_db


def test_a_calibrated_signal_needs_a_reference_of_its_own() -> None:
    """0.775 V is not a pressure, so it cannot become one by default."""
    record = _steady_sine(0.775, FS, seconds=0.5)
    with pytest.raises(ValueError, match=r"'reference' is required for a calibrated"):
        quasi_peak_meter(Signal(record, int(FS), calibration_factor=5.0))


def test_a_signal_refuses_a_rate_that_disagrees_with_its_own() -> None:
    """Exempt from nothing about the rate: the recording knows it."""
    record = _steady_sine(0.775, FS, seconds=0.5)
    with pytest.raises(ValueError, match=r"conflicts with the Signal's own fs"):
        quasi_peak_meter(Signal(record, int(FS)), 44100)


# ---------------------------------------------------------------------------
# The ballistics constant
# ---------------------------------------------------------------------------


def test_the_shipped_ballistics_are_the_three_fitted_time_scales() -> None:
    """A fit to Tables 2 and 3; these numbers are nowhere in BS.468-4."""
    assert BS468_BALLISTICS.charge == pytest.approx(1.4096e-3)
    assert BS468_BALLISTICS.discharge == pytest.approx(293.20e-3)
    assert BS468_BALLISTICS.reading_device == pytest.approx(139.99e-3)


@pytest.mark.parametrize(
    "field",
    ["charge", "discharge", "reading_device"],
    ids=["charge", "discharge", "rd"],
)
def test_a_non_positive_time_constant_is_refused(field: str) -> None:
    """Seconds, and a positive number of them."""
    with pytest.raises(ValueError, match=rf"'{field}' must be positive"):
        replace(BS468_BALLISTICS, **{field: 0.0})


def test_a_non_finite_time_constant_is_refused() -> None:
    """An infinite constant freezes the stage it belongs to."""
    with pytest.raises(ValueError, match=r"'discharge'"):
        replace(BS468_BALLISTICS, discharge=math.inf)


def test_a_stage_that_falls_faster_than_it_rises_is_refused() -> None:
    """Swap the two and the peak rectifier becomes a valley follower."""
    with pytest.raises(
        ValueError, match=r"'charge' .* must be shorter than 'discharge'"
    ):
        QuasiPeakBallistics(charge=0.5, discharge=0.001, reading_device=0.14)

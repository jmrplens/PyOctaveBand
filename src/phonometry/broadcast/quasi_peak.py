#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Psophometric quasi-peak programme-level meter (ITU-R BS.468-4 clause 2).

ITU-R BS.468-4 specifies the instrument that measures audio-frequency noise
voltage in sound broadcasting as two parts. Clause 1 is the weighting
network, realised elsewhere in this library
(:func:`phonometry.filters.weighting_filter` with ``curve="468"``, from the
Fig. 1a ladder). Clause 2 is the detector this module implements: a
**quasi-peak** value method, whose reading is reported in **dBqps**
(clause 3).

**Clause 2 prints no dynamics.** It gives no time constant, no rise time, no
decay law, no transfer function and no differential equation. The words
"time constant" occur once in the whole Recommendation, in an informative
Note that offers "two peak rectifier circuits of different time-constants
connected in tandem" as *a possible arrangement* after full-wave
rectification, and the preamble says the performance "may be realized in a
variety of ways". What clause 2 does give is eleven acceptance windows:
Table 2 reads a single 5 kHz tone burst at eight durations from 1 ms to
200 ms, Table 3 reads a train of 5 ms bursts at 2, 10 and 100 per second,
and each cell carries a reference reading with a lower and an upper limit,
expressed as a percentage of the reading the same tone gives steadily. Those
eleven windows are the entire specification of the dynamics, and they are
what :func:`verify_quasi_peak_dynamics` checks.

The one absolute statement clause 2 does make is clause 2.6: a steady 1 kHz
sine at 0.775 V r.m.s. shall read 0.775 V, that is 0 dBqps. So the detector
reads the **r.m.s.** of a steady sine, not its peak, and that fixes the scale
of the whole instrument (NOTE 2 to clause 1: "The whole instrument is
calibrated at 1 kHz"). Here that scale is not a stored number but a
measurement: :func:`quasi_peak_meter` runs its own chain on a 1 kHz sine of
1 V r.m.s. at the caller's sample rate and divides by what it reads, so
clause 2.6 holds exactly at every rate rather than at the one rate a frozen
constant would have been fitted at.

What this module chooses, because the Recommendation does not
--------------------------------------------------------------

Six decisions, none of them in BS.468-4, all of them inseparable: change any
one and :data:`BS468_BALLISTICS` has to be refitted.

* **Full-wave rectification** as :math:`|x|`, following the Note's own
  wording. An analytic-signal envelope also meets all eleven windows when it
  is refitted from scratch, with a three times larger worst-case residual.
* **A peak rectifier** as a one-pole recursion with different rise and fall
  rates, the same asymmetric kernel the impulse time weighting uses
  (:func:`phonometry.filters.weighting.time_weighting`), discharging toward
  the input rather than toward zero. Refitting toward zero instead moves the
  worst residual by 0.002 dB, so the discharge law is not load-bearing.
* **One symmetric reading device** after it, first order. Clause 2.5 treats
  "the reading device" as a separate object, and a peak rectifier alone
  cannot meet the tables: its best fit is 4.3 dB out at the worst point and
  leaves only 4 of the 11 readings inside their windows.
* **The reading is the maximum of the reading device's output over the
  record**, which is what an operator writes down from a pointer. The record
  therefore has to include the decay: a burst that ends at the last sample
  is read before the needle has finished rising.
* **The reference for Tables 2 and 3** is the same chain fed a steady 5 kHz
  tone through the same network, so both tables are ratios and every
  absolute factor, the network's +11.7 dB at 5 kHz included, cancels.
* **The ballistics run at the caller's rate** on the record
  :func:`phonometry.filters.weighting_filter` returns. Measured across 32,
  44.1, 48, 96 and 192 kHz, no reading moves by more than 0.080 dB, or 4.2 %
  of its own window.

What no oracle can check
------------------------

The tables use one carrier, one waveform, one amplitude, eight durations and
three repetition rates, and that is the whole specification. There is no
reference implementation, no tabulated impulse response and no worked
example in BS.468-4, in IEC 60268-1:1985 Appendix A or in either 1988
amendment. So the **response to noise**, which is the instrument's entire
purpose, is unverifiable: two conforming detectors will disagree on noise,
on speech and on programme material by an amount nothing in the document
bounds. So is the relationship between a dB(468) quasi-peak figure and a
dB(A) r.m.s. one, and so is everything outside the tabulated stimuli
(bursts shorter than 0.6 ms or longer than 200 ms, carriers other than
5 kHz, anything that is not a gated sine).

Four clauses are computable and pass **by construction** rather than by
measurement, because rectification, first-order recursions and a maximum are
all positively homogeneous and cannot overshoot a monotone step: clause 2.1's
two attenuator variants, clause 2.3's +-1 dB over 20 dB of 0.6 ms bursts,
clause 2.4's 0.5 dB reversibility (:math:`|-x| = |x|`, so the difference is
exactly zero) and clause 2.5's 0.3 dB overswing cap. They are properties of
the arithmetic, not evidence about the design. The rest of clauses 2.3, 2.6
and 2.7 has nothing to compute at all: overload capacity above full scale,
the law of a logarithmic stage, a calibrated scale range of at least 20 dB,
an input impedance of at least 20 kOhm and a 600 Ohm termination are
instrument hardware, and "80 % of full scale" constrains where a real needle
sits during the test rather than the quantity, which is why this module does
not model a full scale at all.

**There is no** :meth:`~QuasiPeakResult.report` **fiche.** BS.468-4
prescribes no report format, no declaration form and no worked example; the
conformity evidence is the eleven windows, and its home is
``docs/CONFORMANCE.md`` through :func:`verify_quasi_peak_dynamics`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import (
    require_1d_signal,
    require_finite_fields,
    require_positive,
    require_ranks,
)
from ..filters.weighting import weighting_filter
from ..io._resolve import SignalInput, resolve_fs, resolve_samples
from ..io._signal import Signal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

try:
    from numba import jit as _numba_jit
except ImportError:  # pragma: no cover - depends on install extras
    # unused-ignore: with numba absent its import is Any and the ignore is
    # unnecessary; with numba installed the assignment needs it.
    _numba_jit = None  # type: ignore[assignment, unused-ignore]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Reference voltage of the dBqps scale (clause 2.6), in volts: the r.m.s.
#: value of the 1 kHz sine that shall read 0 dB.
DBQPS_REFERENCE = 0.775

#: Reference sound pressure, in pascals. BS.468-4 has none: it is here only
#: so that :attr:`QuasiPeakResult.level_unit` can name the scale a caller
#: gets by reading a pressure record against 20 uPa, which is a quasi-peak
#: sound pressure level and is not dBqps.
_P_REF = 20e-6

#: Carrier of the clause 2.1 and 2.2 stimuli, in Hz.
_BURST_HZ = 5000.0

#: Calibration tone of clause 2.6, in Hz.
_CALIBRATION_HZ = 1000.0

#: Length of every synthesised steady tone (the clause 2.6 calibration and
#: the Tables 2 and 3 reference reading), in seconds. The slowest element of
#: the chain settles in about 1.5 s; at 3 s the reading has converged to
#: within a part in :math:`10^9`, which is 9e-9 dB.
_SETTLING_SECONDS = 3.0

#: Silence appended after a clause 2.1 burst so the reading device can finish
#: rising, in seconds. Measured: the maximum is already reached within 0.3 s
#: of the end of the longest burst.
_BURST_TAIL_SECONDS = 0.5

#: Silence appended to a record before the weighting network runs, in
#: seconds. ``weighting_filter`` reaches its sections through an
#: interpolation and a decimation, and the polyphase resampler's last output
#: sample sits on an incomplete window: on a 5 kHz tone at 48 kHz that one
#: sample comes out 5.1 % above the settled peak of the weighted waveform.
#: The padding moves it into a region where the network has decayed (its
#: impulse response is 40 dB down within 0.32 ms) and the padded tail is
#: dropped before the detector runs, so the reading is a property of the
#: record rather than of where the resampler's window ran out.
#:
#: As the chain stands the repair is worth about 1e-6 dB, because the
#: 1.41 ms charge of :data:`BS468_BALLISTICS` spans fourteen half-periods of
#: the rectified 5 kHz carrier and one sample cannot move it: the same
#: reason the sampled-peak worry does not bite anywhere else here. It is
#: done at source anyway, so that the reading never depends on a resampler
#: edge and no later change to the charge constant can bring the artefact
#: back.
_WEIGHTING_TAIL_SECONDS = 0.005


@dataclass(frozen=True)
class QuasiPeakBallistics:
    """Time constants of the quasi-peak chain, in seconds.

    :ivar charge: Rise time constant of the peak rectifier.
    :ivar discharge: Fall time constant of the peak rectifier.
    :ivar reading_device: Time constant of the symmetric first-order reading
        device that follows it (clause 2.5's "reading device").
    """

    charge: float
    discharge: float
    reading_device: float

    def __post_init__(self) -> None:
        """Reject constants that do not describe a peak rectifier.

        A time constant is a positive finite number of seconds, and a *peak*
        rectifier is the one that follows a rise faster than a fall. Swap the
        two and the same recursion becomes a valley follower whose reading
        tracks the quietest passage of the record: an object that still runs,
        still returns a trace, and measures the opposite of what the class
        name says.

        :raises ValueError: if any constant is not positive and finite, or if
            the charge is not shorter than the discharge.
        """
        require_positive(self.charge, "charge")
        require_positive(self.discharge, "discharge")
        require_positive(self.reading_device, "reading_device")
        require_finite_fields(self, "charge", "discharge", "reading_device")
        if self.charge >= self.discharge:
            msg = (
                f"'charge' ({self.charge} s) must be shorter than 'discharge' "
                f"({self.discharge} s): a stage that falls faster than it "
                "rises follows valleys, not peaks."
            )
            raise ValueError(msg)


#: **A fit to the eleven reference readings of ITU-R BS.468-4 Tables 2 and 3;
#: these three numbers appear nowhere in the Recommendation.** Clause 2
#: prints no time constant at all (see the module docstring), so the
#: dynamics were solved for rather than quoted, by Nelder-Mead over
#: :math:`\log_{10}(\tau)` from 48 independent starts at 48 kHz.
#:
#: How well the tables pin them down depends on the question asked, so the
#: question is stated. Holding the other two at the values above and moving
#: one until a window fails, all eleven are met from 1.014 to 1.894 ms of
#: charge (a factor of 1.87), 233 to 367 ms of discharge (1.58) and 96 to
#: 200 ms of reading device (2.09). Re-optimising the other two at each step
#: roughly doubles each of those. So the reading device, not the charge, is
#: the constant the tables pin down least, and two conforming instruments may
#: differ by a factor of two in it. No reading this module produces for any
#: signal other than a gated 5 kHz sine can be tighter than that.
#:
#: Three is the model order the tables support. A fourth constant (a second
#: peak rectifier, the informative Note's own arrangement) buys 0.011 dB and
#: is unidentified over a factor of 15.9; a fifth buys 0.036 dB at the worst
#: point, concentrated on the two cells whose printed reference is quantised
#: to +-0.064 and +-0.056 dB, and its two middle constants move by factors of
#: 3.0 and 2.1 under a change in the objective's last decimal. The extra
#: constants fit the rounding of the table.
BS468_BALLISTICS = QuasiPeakBallistics(
    charge=1.4096e-3,
    discharge=293.20e-3,
    reading_device=139.99e-3,
)

#: The eight single-burst rows of Table 2 (printed p. 4): burst duration in
#: ms, full 5 kHz periods in the burst, then the lower limit, the reference
#: reading and the upper limit as a percentage of the steady reading. The
#: percentage rows are primary and the dB rows are derived from them: 22 of
#: Table 2's 24 cells agree to within 0.050 dB, the rounding of a
#: two-figure percentage, and the one that does not is a defect of the dB
#: cell (see ``docs/ERRATA.md``).
_TABLE_2: tuple[tuple[float, int, float, float, float], ...] = (
    (1.0, 5, 13.5, 17.0, 21.4),
    (2.0, 10, 22.4, 26.6, 31.6),
    (5.0, 25, 34.0, 40.0, 46.0),
    (10.0, 50, 41.0, 48.0, 55.0),
    (20.0, 100, 44.0, 52.0, 60.0),
    (50.0, 250, 50.0, 59.0, 68.0),
    (100.0, 500, 58.0, 68.0, 78.0),
    (200.0, 1000, 68.0, 80.0, 92.0),
)

#: The three repetitive-burst rows of Table 3 (printed p. 4): bursts per
#: second of a 5 ms (25-cycle) 5 kHz burst, then the lower limit, the
#: reference reading and the upper limit, again as a percentage of the steady
#: reading. The 100 per second upper limit is exactly 100 %: the train "may
#: reach but not exceed the steady reading", which is what settles the
#: percentages as fractions of the steady reading rather than of full scale.
_TABLE_3: tuple[tuple[float, float, float, float], ...] = (
    (2.0, 43.0, 48.0, 53.0),
    (10.0, 72.0, 77.0, 82.0),
    (100.0, 94.0, 97.0, 100.0),
)

#: Full 5 kHz periods in the 5 ms burst Table 3 repeats.
_TABLE_3_CYCLES = 25


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


def _quasi_peak_kernel_py(
    rectified: np.ndarray,
    alpha_charge: float,
    alpha_discharge: float,
    alpha_reading: float,
) -> np.ndarray:
    """Peak rectifier and reading device in one pass (jitted when numba is present)."""
    trace = np.empty_like(rectified)
    peak = 0.0
    reading = 0.0
    for i in range(rectified.shape[0]):
        value = rectified[i]
        alpha = alpha_charge if value > peak else alpha_discharge
        peak += alpha * (value - peak)
        reading += alpha_reading * (peak - reading)
        trace[i] = reading
    return trace


if _numba_jit is not None:
    _apply_quasi_peak_kernel = _numba_jit(nopython=True, cache=True)(
        _quasi_peak_kernel_py
    )
else:  # pragma: no cover - exercised only without numba installed
    _apply_quasi_peak_kernel = _quasi_peak_kernel_py


def _alpha(time_constant: float, fs: float) -> float:
    r"""One-pole coefficient of a time constant at a sample rate.

    :math:`\alpha = 1 - e^{-T/\tau}` with :math:`T = 1/f_\mathrm{s}`, spelled
    through ``expm1`` so that the very small exponents of a long constant at
    a high rate keep their significant figures.

    :param time_constant: The time constant, in seconds.
    :param fs: Sample rate, in Hz.
    :return: The per-sample coefficient, in ``(0, 1)``.
    """
    return -math.expm1(-1.0 / (fs * time_constant))


def _weighted_record(samples: np.ndarray, fs: float, *, weighted: bool) -> np.ndarray:
    """The record the detector rectifies, weighted or not.

    Clause 2's preamble runs every test but 2.4 through the weighting
    network. The padding is the resampler-edge repair described at
    :data:`_WEIGHTING_TAIL_SECONDS`; it is left on so the caller can drop it
    after the detector has run, which is what keeps the returned trace the
    same length as the record the caller handed in.

    :param samples: The record, float64.
    :param fs: Sample rate, in Hz. Rounded for the network, which is
        designed at an integer rate.
    :param weighted: Whether the ITU-R BS.468-4 network is in the path.
    :return: The record to rectify, with the padding still on it when
        weighted.
    """
    if not weighted:
        return samples
    pad = np.zeros(int(round(_WEIGHTING_TAIL_SECONDS * fs)))
    padded = np.concatenate([samples, pad])
    return np.asarray(weighting_filter(padded, round(fs), "468"), dtype=np.float64)


def _detector_trace(samples: np.ndarray, fs: float, *, weighted: bool) -> np.ndarray:
    """The reading device's output over the record, before calibration.

    :param samples: The record, float64.
    :param fs: Sample rate, in Hz.
    :param weighted: Whether the ITU-R BS.468-4 network is in the path.
    :return: The trace, as long as *samples*, in the units of *samples*.
    """
    rectified = np.abs(_weighted_record(samples, fs, weighted=weighted))
    trace = _apply_quasi_peak_kernel(
        rectified,
        _alpha(BS468_BALLISTICS.charge, fs),
        _alpha(BS468_BALLISTICS.discharge, fs),
        _alpha(BS468_BALLISTICS.reading_device, fs),
    )
    return np.asarray(trace[: samples.shape[-1]])


def _steady_tone(frequency: float, amplitude: float, fs: float) -> np.ndarray:
    """A sine of peak *amplitude* at *frequency*, long enough for the chain to settle.

    The amplitude is a *peak* amplitude, as
    :func:`~phonometry.signals.tone_burst` uses, so that the steady tone and
    the bursts Tables 2 and 3 compare against it are the same tone.

    :param frequency: Tone frequency, in Hz.
    :param amplitude: Peak amplitude.
    :param fs: Sample rate, in Hz.
    :return: The tone, :data:`_SETTLING_SECONDS` long.
    """
    n = int(round(_SETTLING_SECONDS * fs))
    t = np.arange(n) / fs
    return amplitude * np.sin(2.0 * math.pi * frequency * t)


@lru_cache(maxsize=32)
def _calibration_factor(fs: float) -> float:
    """The scale factor clause 2.6 fixes, at this sample rate.

    Clause 2.6: a steady 1 kHz sine at 0.775 V r.m.s. shall read 0.775 V. The
    chain reads a fixed multiple of the peak of a steady sine, 0.9816 of it at
    48 kHz, where clause 2.6 asks for the r.m.s. value, 0.7071 of it; one
    division settles the difference. Measured over 32 to 192 kHz the multiple
    moves by 0.0141 dB, which is why it is computed per rate instead of
    frozen: clause 2.6 is an equality, not a tolerance.

    The tone goes through the weighting network, where it sees the network's
    0 dB at 1 kHz. NOTE 2 to clause 1 calibrates the whole instrument at
    1 kHz, so the unweighted mode of clause 2.4 shares this same scale, as a
    real instrument with one scale and a switch would.

    :param fs: Sample rate, in Hz.
    :return: The factor that turns a raw reading into volts.
    """
    unit_rms = _steady_tone(_CALIBRATION_HZ, math.sqrt(2.0), fs)
    return 1.0 / float(_detector_trace(unit_rms, fs, weighted=True).max())


@dataclass(frozen=True)
class QuasiPeakResult:
    """A quasi-peak reading of one record (ITU-R BS.468-4 clause 2).

    :ivar reading: The quasi-peak reading: the largest value :attr:`trace`
        reaches, on the scale clause 2.6 fixes, in the record's own unit.
    :ivar level_db: ``20 lg(reading / reference)``. This is **dBqps**
        (clause 3) when :attr:`reference` is the 0.775 V clause 2.6 names,
        and a level against whatever else the caller referred it to
        otherwise; :attr:`level_unit` is the one place that says which.
    :ivar reference: The reference :attr:`level_db` is taken against, in the
        record's own unit.
    :ivar trace: The reading device's output over the record, sample for
        sample with it, in the record's own unit. This is the needle, and the
        shape of its rise and decay is the whole content of clause 2.
    :ivar fs: Sample rate of the record, Hz.
    :ivar weighted: Whether the ITU-R BS.468-4 weighting network was in the
        path. Clause 2's preamble runs every test but the clause 2.4
        reversibility check through it.
    """

    reading: float
    level_db: float
    reference: float
    trace: np.ndarray
    fs: float
    weighted: bool

    def __post_init__(self) -> None:
        """Reject a reading that its own trace does not support.

        The reading is not an independent number: clause 2's reading rule
        makes it the largest excursion of the needle, so a stored value that
        is not the maximum of :attr:`trace` is a measurement disagreeing with
        the graph beneath it. Nothing downstream recomputes it -- the plot
        draws the trace and annotates the stored scalar, and the level is
        derived from the scalar alone -- so the disagreement would print as a
        marker sitting off the curve, with no error anywhere.

        :raises ValueError: if the trace is not a series, or if the reading
            is not the maximum of the trace.
        """
        require_ranks(self, trace=1)
        peak = float(np.max(self.trace)) if self.trace.size else 0.0
        if not math.isclose(self.reading, peak, rel_tol=0.0, abs_tol=1e-12):
            msg = (
                f"'reading' is {self.reading!r} but 'trace' peaks at {peak!r}: "
                "the quasi-peak reading is the largest excursion of the trace."
            )
            raise ValueError(msg)

    @property
    def time(self) -> np.ndarray:
        """Time of each sample of :attr:`trace`, in seconds."""
        return np.arange(self.trace.size) / self.fs

    @property
    def level_unit(self) -> str:
        """The name of the scale :attr:`level_db` is on.

        The one place that answers whether the level may be called
        **dBqps**, which clause 3 defines only against the 0.775 V of
        clause 2.6. Anything else is named by its own reference instead,
        because BS.468-4 gives no other scale a name -- a quasi-peak sound
        pressure level read against 20 uPa is a perfectly good number and is
        not dBqps.

        The name is read off the *reference*, which is the only unit-bearing
        thing the result holds: the samples arrive in whatever unit they were
        already in and the library never learns which. A caller who refers a
        voltage record to 20 uV rather than to 20 uPa gets the pascal name;
        nothing here can tell those two apart, and no other reference is
        close enough to either to collide.
        """
        if math.isclose(self.reference, DBQPS_REFERENCE):
            return "dBqps"
        if math.isclose(self.reference, _P_REF):
            return "dB re 20 uPa"
        return f"dB re {self.reference:g}"

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the quasi-peak trace over time, with the reading marked.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the trace ``plot`` call.
        :return: The axes.
        """
        from .._i18n import check_language
        from .._plot.broadcast import plot_quasi_peak

        check_language(language)
        return plot_quasi_peak(self, ax=ax, language=language, **kwargs)


def _resolve_reference(x: SignalInput, reference: float | None) -> float:
    """The reference of the level, refusing the one case that has no default.

    A calibrated :class:`~phonometry.io.Signal` presents its samples in
    pascals, as everywhere else in this library. 0.775 V is not a pressure, so
    it cannot quietly become the reference of a pascal reading: the level would
    come out ``20 lg(p / 0.775)``, a number with no meaning, under a name
    (dBqps) that clause 3 gives only to voltages. Nothing in BS.468-4 supplies
    a pressure reference to put there instead, so the caller does.

    What is refused is the *guess*, not the value. An explicitly passed
    reference is taken as given, 0.775 included: it is the caller saying what
    their reading is relative to, and a library that overrode that would be
    second-guessing a deliberate statement rather than declining to invent one.
    The two are not the same act, and only the first is this guard's business.

    :param x: The signal argument, as the caller passed it.
    :param reference: The explicitly passed reference, or ``None``.
    :return: The reference to use, in the record's own unit.
    :raises ValueError: if a calibrated Signal arrives without one.
    """
    if reference is not None:
        return require_positive(reference, "reference")
    if isinstance(x, Signal) and x.calibration_factor is not None:
        msg = (
            "'reference' is required for a calibrated Signal: its samples are "
            "in pascals and the 0.775 V of ITU-R BS.468-4 clause 2.6 is not a "
            "pressure, so the level would be dBqps in name only. Pass "
            "reference=2e-5 for a quasi-peak sound pressure level re 20 uPa."
        )
        raise ValueError(msg)
    return DBQPS_REFERENCE


def quasi_peak_meter(
    x: SignalInput,
    fs: float | None = None,
    *,
    weighted: bool = True,
    reference: float | None = None,
) -> QuasiPeakResult:
    """Read a record with the ITU-R BS.468-4 quasi-peak meter.

    Runs clause 2's detector over the whole record and returns the largest
    excursion of the reading device, on the scale clause 2.6 fixes. The
    reading rule is the maximum over the record, so the record must include
    the decay of whatever it is measuring: a burst that ends at the last
    sample is read before the needle has finished rising.

    The chain and everything it chooses that the Recommendation does not are
    described in the module docstring; the time constants are
    :data:`BS468_BALLISTICS`, a fit to Tables 2 and 3 rather than a quoted
    value, and :func:`verify_quasi_peak_dynamics` is what checks them.

    **Units.** The detector itself is unit-agnostic: rectification,
    first-order recursions and a maximum carry whatever unit the record
    arrived in, and the eleven acceptance windows are ratios in which every
    unit cancels. Only the *level* needs a unit, and clause 2.6 gives
    BS.468-4 exactly one, the volt. A bare array or an uncalibrated
    :class:`~phonometry.io.Signal` is therefore read against 0.775 V and its
    ``level_db`` is dBqps. A **calibrated** Signal is analysed in pascals,
    as everywhere else in this library, and then has to be given a
    *reference* of its own: 0.775 V is not a pressure, and BS.468-4 offers
    no pressure to put in its place. ``result.level_unit`` names whichever
    scale came out.

    :param x: The record: a 1-D array, or a :class:`phonometry.io.Signal`.
        Multichannel input is refused rather than mixed, because a
        quasi-peak reading is one needle over one channel.
    :param fs: Sample rate, Hz. Required for a bare array; a
        :class:`~phonometry.io.Signal` brings its own, and an explicit value
        that disagrees with it raises instead of silently winning.
    :param weighted: Whether to run the record through the clause 1
        weighting network first (default ``True``, which is what clause 2's
        preamble requires of every test but the clause 2.4 reversibility
        check).
    :param reference: The reference of ``level_db``, in the record's own
        unit. ``None`` (the default) is the 0.775 V of clause 2.6, and is
        refused for a calibrated Signal, whose samples are pascals.
    :return: A :class:`QuasiPeakResult`.
    :raises ValueError: If the record is empty or not 1-D, if *fs* is not
        positive, if *reference* is not positive, or if a calibrated Signal
        arrives without one.
    """
    fs = require_positive(resolve_fs(x, fs), "fs")
    reference = _resolve_reference(x, reference)
    samples = require_1d_signal(resolve_samples(x), "x")
    if samples.size == 0:
        msg = "Input signal 'x' cannot be empty."
        raise ValueError(msg)

    trace = _detector_trace(samples, fs, weighted=weighted) * _calibration_factor(fs)
    reading = float(trace.max())
    level_db = 20.0 * math.log10(reading / reference) if reading > 0.0 else -math.inf
    return QuasiPeakResult(
        reading=reading,
        level_db=level_db,
        reference=float(reference),
        trace=trace,
        fs=float(fs),
        weighted=weighted,
    )


# ---------------------------------------------------------------------------
# Conformance against the eleven acceptance windows
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _steady_reference(fs: float) -> float:
    """The chain's reading for the steady 5 kHz tone Tables 2 and 3 divide by."""
    tone = _steady_tone(_BURST_HZ, 1.0, fs)
    return float(_detector_trace(tone, fs, weighted=True).max())


def _burst_percent(
    fs: float, cycles: int, repetitions: int, rate: float | None
) -> float:
    """Read one clause 2.1 or 2.2 stimulus, as a percentage of the steady reading.

    :param fs: Sample rate, in Hz.
    :param cycles: Full 5 kHz periods in each burst.
    :param repetitions: Number of bursts.
    :param rate: Bursts per second, or ``None`` for a single burst.
    :return: The reading, in percent of the steady 5 kHz reading.
    """
    from ..signals.test_signals import tone_burst

    burst = tone_burst(
        fs,
        _BURST_HZ,
        cycles,
        repetitions=repetitions,
        repetition_rate=rate,
        post_silence=_BURST_TAIL_SECONDS,
    )
    reading = float(_detector_trace(burst.signal, fs, weighted=True).max())
    return 100.0 * reading / _steady_reference(fs)


def _window_row(
    stimulus: str,
    table: int,
    percent: float,
    lower: float,
    reference: float,
    upper: float,
) -> dict[str, Any]:
    """One acceptance window, its reading and the margins.

    :param stimulus: Label of the stimulus ("5 ms", "10 bursts/s").
    :param table: 2 for a single burst, 3 for a repetitive train.
    :param percent: The reading, in percent of the steady reading.
    :param lower: Lower limit of the window, in percent.
    :param reference: Printed reference reading, in percent.
    :param upper: Upper limit of the window, in percent.
    :return: The row, with ``margin_db`` positive when the reading is inside
        its window and ``deviation_db`` measured against the printed
        reference reading.
    """
    return {
        "stimulus": stimulus,
        "table": table,
        "reading_percent": percent,
        "lower_percent": lower,
        "reference_percent": reference,
        "upper_percent": upper,
        "deviation_db": 20.0 * math.log10(percent / reference),
        "margin_db": min(
            20.0 * math.log10(percent / lower), 20.0 * math.log10(upper / percent)
        ),
    }


def verify_quasi_peak_dynamics(fs: float = 48000.0) -> dict[str, Any]:
    """Check the detector against the eleven acceptance windows of clause 2.

    Runs the clause 2.1 and 2.2 stimuli exactly as they are specified: a
    5 kHz sine starting at a zero crossing and lasting an integral number of
    full periods (at 5 kHz one period is 0.2 ms, so the eight tabulated
    durations are 5, 10, 25, 50, 100, 250, 500 and 1000 cycles), through the
    weighting network, and reads each one against the same chain's steady
    reading for the same tone. Table 3 repeats the 5 ms burst at 2, 10 and
    100 per second.

    This is the whole conformance statement BS.468-4 authorises for a
    detector, and it is an acceptance region rather than a set of values:
    every reading inside a window conforms, and the Recommendation tolerates
    about 2.5 dB of disagreement between two conforming instruments on a
    single burst. ``deviation_db`` is reported against the printed reference
    reading as well, but that column is a *self-imposed* regression bound,
    not conformance: the reference is printed to two significant figures on
    nine of the eleven cells, so its own quantum is 0.027 to 0.108 dB.

    The stimulus amplitude does not appear anywhere: both tables are ratios
    to the steady reading at the same amplitude, and a chain built from
    rectification, first-order recursions and a maximum is exactly
    homogeneous, so clause 2.1's two attenuator variants give the same
    answer by construction.

    :param fs: Sample rate to run the stimuli at, in Hz. At 44.1 kHz the
        25-cycle burst is not sample-exact (it spans 220.5 samples) and
        :func:`~phonometry.signals.tone_burst` warns; the consequence
        measures 0.006 dB against a 2.626 dB window.
    :return: Dict with ``fs``, ``passed`` (every reading inside its window),
        ``worst_margin_db`` (the smallest margin over the eleven, negative
        when one is outside), ``worst_deviation_db`` (the largest departure
        from a printed reference reading) and ``stimuli``: eleven rows of
        ``{"stimulus", "table", "reading_percent", "lower_percent",
        "reference_percent", "upper_percent", "deviation_db", "margin_db"}``.
    :raises ValueError: If *fs* is not positive.
    """
    fs = require_positive(fs, "fs")
    rows = [
        _window_row(
            f"{duration:g} ms",
            2,
            _burst_percent(fs, cycles, 1, None),
            lower,
            reference,
            upper,
        )
        for duration, cycles, lower, reference, upper in _TABLE_2
    ]
    rows += [
        _window_row(
            f"{rate:g} bursts/s",
            3,
            _burst_percent(
                fs,
                _TABLE_3_CYCLES,
                int(round(_SETTLING_SECONDS * rate)) + 1,
                rate,
            ),
            lower,
            reference,
            upper,
        )
        for rate, lower, reference, upper in _TABLE_3
    ]
    return {
        "fs": float(fs),
        "passed": all(row["margin_db"] >= 0.0 for row in rows),
        "worst_margin_db": min(row["margin_db"] for row in rows),
        "worst_deviation_db": max(abs(row["deviation_db"]) for row in rows),
        "stimuli": rows,
    }

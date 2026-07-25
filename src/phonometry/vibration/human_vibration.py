#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Human exposure to whole-body and hand-transmitted vibration.

The measurement chain of the ISO human-vibration family is implemented from the
standards' own analog definitions, clean-room:

* **ISO 8041-1:2017** - the authoritative *master* definition of every
  frequency weighting.  A single cascade ``H(s) = Hh(s)*Hl(s)*Ht(s)*Hs(s)``
  (Formula (5)) of second-order band-limiting Butterworth sections
  (Formulae (1)/(2)), an acceleration-velocity transition (Formula (3)) and an
  upward step (Formula (4)) realises all nine weightings from the one Table 3
  parameter set: ``Wb, Wc, Wd, We, Wf, Wh, Wj, Wk, Wm``.  The tabulated
  design-goal factors of Annex B (Tables B.1-B.9) are reproduced to their
  four-significant-figure precision.

* **ISO 2631-1:1997** - whole-body vibration: the weighted r.m.s. acceleration
  ``a_w`` (Eq. (1)/(9)), the vibration total value ``a_v`` with axis
  multiplying factors ``k`` (Eq. (10)), the running r.m.s. and maximum
  transient vibration value ``MTVV`` (Eqs. (2)-(4)), the vibration dose value
  ``VDV`` (Eq. (5)), the crest factor (6.2.1) and the energy-equivalent
  magnitude relations of Annex B (Eqs. (B.1)-(B.3)).

* **ISO 2631-2:2003** - whole-body vibration in buildings: the direction-
  independent weighting ``Wm`` (Annex A).

* **ISO 2631-4:2001** - ride comfort in fixed-guideway (rail) transport: the
  vertical weighting ``Wb`` (Annex A) with the ``Wk`` axis multiplying
  factors.  ``Wb`` is realised here from its exact ISO 8041-1 Table 3
  parameters (of which ISO 2631-4 Table A.1 is the two-decimal rounding).

* **ISO 5349-1:2001 / ISO 5349-2:2001** - hand-transmitted vibration: the
  vibration total value ``a_hv`` (5349-1 Eq. (1)), the daily exposure ``A(8)``
  for single and multiple operations (5349-1 Eqs. (2)/(3); 5349-2 Eqs. (1)-(3))
  and the vibration-white-finger dose relation of 5349-1 Annex C (Eq. (C.1)).

* **Directive 2002/44/EC** - the daily exposure action and limit values
  (Article 3) that ISO does not fix: hand-arm ``A(8)`` EAV ``2,5`` /
  ELV ``5`` m/s2; whole-body ``A(8)`` EAV ``0,5`` / ELV ``1,15`` m/s2 (or VDV
  EAV ``9,1`` / ELV ``21`` m/s^1,75).  The hand-arm ``A(8)`` is based on the
  ISO 5349-1 vector total ``a_hv`` (Annex, Part A, point 1); the whole-body
  ``A(8)`` is based on the *highest* of the frequency-weighted axis values
  ``1,4*a_wx``, ``1,4*a_wy``, ``a_wz`` (Annex, Part B, point 1; see
  :func:`wbv_exposure_basis`), not on the ISO 2631-1 Eq. (10) vector total.

The band (spectrum) method and the exposure arithmetic carry the standards'
worked-example oracles; the time-domain metrics operate on a weighted
acceleration signal, which :func:`apply_weighting` produces from a raw record
by applying the exact analog response of ISO 8041-1 in the frequency domain.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal as sig

from .._internal.types import Real
from .._internal.warnings import PhonometryWarning

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

Complex = NDArray[np.complex128]

__all__ = [
    "HAV_EAV_A8",
    "HAV_ELV_A8",
    "REFERENCE_ACCELERATION",
    "REFERENCE_DURATION_S",
    "WBV_EAV_A8",
    "WBV_EAV_VDV",
    "WBV_ELV_A8",
    "WBV_ELV_VDV",
    "WEIGHTING_NAMES",
    "DailyVibrationExposure",
    "ExposureAssessment",
    "HumanVibrationWarning",
    "WeightedSpectrum",
    "WeightingResponse",
    "apply_weighting",
    "combine_partial_exposures",
    "crest_factor",
    "daily_exposure",
    "daily_vibration_exposure",
    "energy_equivalent_acceleration",
    "exposure_assessment",
    "frequency_weighting",
    "hav_daily_exposure",
    "hav_vwf_lifetime_years",
    "motion_sickness_dose_value",
    "mtvv",
    "partial_exposure",
    "running_rms",
    "vibration_dose_value",
    "vibration_total_value",
    "wbv_exposure_basis",
    "weighted_acceleration",
    "weighting_factors",
]

# ---------------------------------------------------------------------------
# Reference constants.
# ---------------------------------------------------------------------------
#: Reference acceleration ``a0 = 10^-6 m/s2`` for vibration levels
#: (ISO 8041-1:2017, 3.1.2.2, after ISO 1683).
REFERENCE_ACCELERATION = 1e-6

#: Reference duration ``T0 = 8 h = 28 800 s`` of the daily exposure ``A(8)``
#: (ISO 5349-1:2001, 3.2; ISO 2631-1:1997, B.1).
REFERENCE_DURATION_S = 28800.0

#: Daily hand-arm exposure action value ``A(8) = 2,5 m/s2`` (Directive
#: 2002/44/EC, Article 3(1)(a)).
HAV_EAV_A8 = 2.5
#: Daily hand-arm exposure limit value ``A(8) = 5 m/s2`` (Directive
#: 2002/44/EC, Article 3(1)(b)).
HAV_ELV_A8 = 5.0
#: Daily whole-body exposure action value ``A(8) = 0,5 m/s2`` (Directive
#: 2002/44/EC, Article 3(2)(a)).
WBV_EAV_A8 = 0.5
#: Daily whole-body exposure limit value ``A(8) = 1,15 m/s2`` (Directive
#: 2002/44/EC, Article 3(2)(b)).
WBV_ELV_A8 = 1.15
#: Whole-body VDV exposure action value ``9,1 m/s^1,75`` (Directive
#: 2002/44/EC, Article 3(2)(a), alternative dose metric).
WBV_EAV_VDV = 9.1
#: Whole-body VDV exposure limit value ``21 m/s^1,75`` (Directive
#: 2002/44/EC, Article 3(2)(b), alternative dose metric).
WBV_ELV_VDV = 21.0

_Q_BUTTERWORTH = 1.0 / math.sqrt(2.0)


class HumanVibrationWarning(PhonometryWarning):
    """Advisory for out-of-range human-vibration measurement conditions."""


# ---------------------------------------------------------------------------
# Frequency weightings (ISO 8041-1:2017, Table 3 + Formulae (1)-(5)).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _WParams:
    """Master parameters of one ISO 8041-1 Table 3 weighting (Hz / gain)."""

    f1: float  # band-limiting high-pass corner
    q1: float
    f2: float  # band-limiting low-pass corner
    q2: float
    f3: float  # a-v transition zero (``inf`` -> absent)
    f4: float  # a-v transition pole (``inf`` -> absent)
    q4: float
    f5: float  # upward-step zero (``inf`` -> step absent)
    q5: float
    f6: float  # upward-step pole (``inf`` -> step absent)
    q6: float
    k: float  # overall gain K


_INF = math.inf

#: ISO 8041-1:2017, Table 3 - the master parameters of the nine frequency
#: weightings.  ``Q1 = Q2 = 1/sqrt(2)`` (band-limiting Butterworth) throughout;
#: ``inf`` corners collapse the corresponding stage to unity (Table 3 NOTEs).
#: ``Wh`` uses the exact ISO 8041-1 values (Table 3, NOTE 2), of which the
#: ISO 5349-1 Table A.1 figures are the five-significant-figure rounding.
_WEIGHTINGS: dict[str, _WParams] = {
    # Whole-body, vertical z (ISO 2631-4); K = 1,024.
    "Wb": _WParams(0.4, _Q_BUTTERWORTH, 100.0, _Q_BUTTERWORTH,
                   16.0, 16.0, 0.55, 2.5, 0.9, 4.0, 0.95, 1.024),
    # Whole-body, horizontal x seat-back (ISO 2631-1).
    "Wc": _WParams(0.4, _Q_BUTTERWORTH, 100.0, _Q_BUTTERWORTH,
                   8.0, 8.0, 0.63, _INF, 1.0, _INF, 1.0, 1.0),
    # Whole-body, horizontal x/y (ISO 2631-1).
    "Wd": _WParams(0.4, _Q_BUTTERWORTH, 100.0, _Q_BUTTERWORTH,
                   2.0, 2.0, 0.63, _INF, 1.0, _INF, 1.0, 1.0),
    # Whole-body, rotational (ISO 2631-1).
    "We": _WParams(0.4, _Q_BUTTERWORTH, 100.0, _Q_BUTTERWORTH,
                   1.0, 1.0, 0.63, _INF, 1.0, _INF, 1.0, 1.0),
    # Whole-body, motion sickness, vertical z (ISO 2631-1).
    "Wf": _WParams(0.08, _Q_BUTTERWORTH, 0.63, _Q_BUTTERWORTH,
                   _INF, 0.25, 0.86, 0.0625, 0.80, 0.10, 0.80, 1.0),
    # Hand-arm, all directions (ISO 5349-1); exact ISO 8041-1 corners.
    "Wh": _WParams(10.0**0.8, _Q_BUTTERWORTH, 10.0**3.1, _Q_BUTTERWORTH,
                   100.0 / (2.0 * math.pi), 100.0 / (2.0 * math.pi), 0.64,
                   _INF, 1.0, _INF, 1.0, 1.0),
    # Whole-body, vertical head, recumbent x (ISO 2631-1).
    "Wj": _WParams(0.4, _Q_BUTTERWORTH, 100.0, _Q_BUTTERWORTH,
                   _INF, _INF, 1.0, 3.75, 0.91, 5.32, 0.91, 1.0),
    # Whole-body, vertical z, the principal ISO 2631-1 weighting.
    "Wk": _WParams(0.4, _Q_BUTTERWORTH, 100.0, _Q_BUTTERWORTH,
                   12.5, 12.5, 0.63, 2.37, 0.91, 3.35, 0.91, 1.0),
    # Whole-body in buildings, all directions (ISO 2631-2).
    "Wm": _WParams(10.0**-0.1, _Q_BUTTERWORTH, 100.0, _Q_BUTTERWORTH,
                   1.0 / (0.028 * 2.0 * math.pi), 1.0 / (0.028 * 2.0 * math.pi),
                   0.5, _INF, 1.0, _INF, 1.0, 1.0),
}

#: The nine ISO 8041-1 frequency-weighting names, in Table 3 order.
WEIGHTING_NAMES: tuple[str, ...] = tuple(_WEIGHTINGS)


def _params(name: str) -> _WParams:
    """Return the Table 3 parameters for ``name`` or raise ``ValueError``."""
    try:
        return _WEIGHTINGS[name]
    except KeyError:
        raise ValueError(
            f"Unknown weighting {name!r}; choose from {', '.join(WEIGHTING_NAMES)}."
        ) from None


def _weighting_response(name: str, freq: Real) -> Complex:
    """Complex ``H(j*2*pi*f)`` of weighting ``name`` (ISO 8041-1, Formula (5)).

    The four cascaded stages of Formulae (1)-(4) are evaluated directly.  A
    corner at infinity collapses its stage to unity, exactly as the Table 3
    NOTEs prescribe.  ``f <= 0`` returns ``0`` (the high-pass blocks DC).
    """
    p = _params(name)
    out = np.zeros(freq.shape, dtype=np.complex128)
    positive = freq > 0.0
    if not np.any(positive):
        return out
    s = 1j * 2.0 * math.pi * freq[positive].astype(np.float64)

    w1 = 2.0 * math.pi * p.f1
    w2 = 2.0 * math.pi * p.f2
    # Formula (1) high-pass and Formula (2) low-pass band limiting.
    hh = 1.0 / (1.0 + w1 / (p.q1 * s) + (w1 / s) ** 2)
    hl = 1.0 / (1.0 + s / (p.q2 * w2) + (s / w2) ** 2)

    # Formula (3) acceleration-velocity transition, gain K.
    ones = np.ones_like(s)
    num_t = ones if math.isinf(p.f3) else 1.0 + s / (2.0 * math.pi * p.f3)
    if math.isinf(p.f4):
        den_t = ones
    else:
        w4 = 2.0 * math.pi * p.f4
        den_t = 1.0 + s / (p.q4 * w4) + (s / w4) ** 2
    ht = p.k * num_t / den_t

    # Formula (4) upward step.
    if math.isinf(p.f5) or math.isinf(p.f6):
        hs = ones
    else:
        w5 = 2.0 * math.pi * p.f5
        w6 = 2.0 * math.pi * p.f6
        hs = (1.0 + s / (p.q5 * w5) + (s / w5) ** 2) / (
            1.0 + s / (p.q6 * w6) + (s / w6) ** 2
        ) * (w5 / w6) ** 2

    out[positive] = np.asarray(hh * hl * ht * hs, dtype=np.complex128)
    return out


@dataclass(frozen=True)
class WeightingResponse:
    """A frequency-weighting magnitude response (ISO 8041-1, Formula (5)).

    :ivar name: Weighting name (one of :data:`WEIGHTING_NAMES`).
    :ivar frequencies: Frequencies at which the response was evaluated, in Hz.
    :ivar response: Complex weighting ``H(j*2*pi*f)`` per frequency.
    :ivar magnitude: Weighting factor ``|H|`` per frequency.
    :ivar magnitude_db: ``20*log10(|H|)`` per frequency, in decibels.
    """

    name: str
    frequencies: Real
    response: Complex
    magnitude: Real
    magnitude_db: Real

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the weighting factor (dB) versus frequency.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.
        """
        from .._i18n import check_language
        from .._plot.vibration import plot_vibration_weighting

        return plot_vibration_weighting(self, ax=ax, language=check_language(language), **kwargs)


def frequency_weighting(name: str, frequencies: ArrayLike) -> WeightingResponse:
    """Frequency-weighting response ``H(f)`` (ISO 8041-1:2017, Formula (5)).

    Evaluates the overall weighting - band limiting, acceleration-velocity
    transition and upward step - of weighting ``name`` at ``frequencies``.

    :param name: Weighting name (one of :data:`WEIGHTING_NAMES`).
    :param frequencies: Frequencies at which to evaluate, in hertz (> 0).
    :return: A :class:`WeightingResponse` with ``.plot()``.
    :raises ValueError: if ``name`` is unknown or ``frequencies`` is empty.
    """
    freq = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if freq.ndim != 1 or freq.size == 0:
        raise ValueError("'frequencies' must be a non-empty 1-D array.")
    resp = _weighting_response(name, freq)
    mag = np.abs(resp)
    with np.errstate(divide="ignore"):
        mag_db = 20.0 * np.log10(mag)
    return WeightingResponse(
        name=name,
        frequencies=freq,
        response=resp,
        magnitude=mag,
        magnitude_db=mag_db,
    )


def weighting_factors(name: str, frequencies: ArrayLike) -> Real:
    """Weighting factors ``|H(f)|`` of weighting ``name`` (ISO 8041-1).

    Convenience wrapper over :func:`frequency_weighting` returning only the
    magnitude array (the ``W_i`` of ISO 2631-1 Eq. (9) / ISO 5349-1 Eq. (A.1)).

    :param name: Weighting name (one of :data:`WEIGHTING_NAMES`).
    :param frequencies: Band centre frequencies, in hertz.
    :return: Weighting factor per frequency.
    """
    return frequency_weighting(name, frequencies).magnitude


def apply_weighting(signal: ArrayLike, fs: float, name: str) -> Real:
    """Apply frequency weighting ``name`` to a time signal (ISO 8041-1).

    The exact analog response :func:`frequency_weighting` is applied in the
    frequency domain (real FFT), so the weighted signal reproduces both the
    magnitude and phase of the standard's cascade without bilinear warping.
    The multiplication is circular, so the record wraps at its ends; apply it
    to a record long enough (or pre-tapered) that the boundary transient is
    negligible, as for any block frequency-domain filtering.

    :param signal: Unweighted acceleration time history (1-D), in m/s2.
    :param fs: Sampling frequency, in hertz (> 0).
    :param name: Weighting name (one of :data:`WEIGHTING_NAMES`).
    :return: The frequency-weighted acceleration signal, same length as input.
    :raises ValueError: if ``signal`` is not 1-D, ``fs`` is not positive, or
        ``name`` is unknown.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("'signal' must be a non-empty 1-D array.")
    fs = _positive_fs(fs)
    n = x.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs).astype(np.float64)
    resp = _weighting_response(name, freqs)
    weighted = np.fft.irfft(np.fft.rfft(x) * resp, n=n)
    return np.asarray(weighted, dtype=np.float64)


# ---------------------------------------------------------------------------
# Band (spectrum) method (ISO 2631-1 Eq. (9); ISO 5349-1 Eq. (A.1)).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeightedSpectrum:
    """A weighted one-third-octave acceleration spectrum and its ``a_w``.

    :ivar frequencies: Band centre frequencies, in hertz.
    :ivar band_accelerations: Unweighted r.m.s. acceleration per band, in m/s2.
    :ivar weighting_name: Weighting applied (one of :data:`WEIGHTING_NAMES`).
    :ivar weighting_factors: Weighting factor ``W_i`` per band.
    :ivar weighted: Weighted band contribution ``W_i*a_i``, in m/s2.
    :ivar overall: Overall weighted r.m.s. acceleration ``a_w``, in m/s2.
    """

    frequencies: Real
    band_accelerations: Real
    weighting_name: str
    weighting_factors: Real
    weighted: Real
    overall: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the unweighted and weighted band spectra with ``a_w``.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.
        """
        from .._i18n import check_language
        from .._plot.vibration import plot_weighted_spectrum

        return plot_weighted_spectrum(self, ax=ax, language=check_language(language), **kwargs)


def weighted_acceleration(
    band_accelerations: ArrayLike,
    frequencies: ArrayLike,
    weighting: str,
) -> WeightedSpectrum:
    """Weighted r.m.s. acceleration from a band spectrum (ISO 2631-1 Eq. (9)).

    ``a_w = sqrt( sum_i (W_i * a_i)^2 )`` with the per-band weighting factors
    ``W_i`` of ISO 8041-1 evaluated at the band centres (ISO 5349-1 Eq. (A.1)
    is the identical construction for the hand-arm weighting ``Wh``).

    .. note::
        The weightings are evaluated at exactly the frequencies you pass. The
        ISO tables (ISO 8041-1 Annex B, ISO 2631-1 Table 3, ISO 5349-1
        Table A.2) tabulate ``W_i`` at the *true* one-third-octave centres
        ``10^(n/10)`` Hz (6.31, 7.943, 15.85, ...), not at the nominal band
        labels (6.3, 8, 16, ...). Pass true centres when comparing against
        the tabulated factors; a nominal label (e.g. 16.0 instead of 15.85)
        gives a slightly different ``W_i``.

    :param band_accelerations: r.m.s. acceleration ``a_i`` per band, in m/s2.
    :param frequencies: Band centre frequencies, in hertz (true
        one-third-octave centres ``10^(n/10)`` for table-conformant band
        values).
    :param weighting: Weighting name (one of :data:`WEIGHTING_NAMES`).
    :return: A :class:`WeightedSpectrum` with ``.plot()``.
    :raises ValueError: if the inputs differ in length or are empty.
    """
    accel = np.atleast_1d(np.asarray(band_accelerations, dtype=np.float64))
    freq = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if accel.ndim != 1 or accel.size == 0 or accel.shape != freq.shape:
        raise ValueError(
            "'band_accelerations' and 'frequencies' must be non-empty, 1-D and "
            "equal-length."
        )
    factors = weighting_factors(weighting, freq)
    weighted = factors * accel
    overall = float(np.sqrt(np.sum(weighted**2)))
    return WeightedSpectrum(
        frequencies=freq,
        band_accelerations=accel,
        weighting_name=weighting,
        weighting_factors=factors,
        weighted=weighted,
        overall=overall,
    )


# ---------------------------------------------------------------------------
# Time-domain metrics (ISO 2631-1 clause 6; ISO 8041-1 clause 3.1.2).
# ---------------------------------------------------------------------------
def _weighted_signal(signal: ArrayLike) -> Real:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("'signal' must be a non-empty 1-D array.")
    return x


def _positive_fs(fs: float) -> float:
    """Return ``fs`` as a positive, finite float or raise ``ValueError``."""
    fs = float(fs)
    if not math.isfinite(fs) or fs <= 0.0:
        raise ValueError("'fs' must be a positive, finite sampling frequency.")
    return fs


def running_rms(
    signal: ArrayLike,
    fs: float,
    *,
    integration_time: float = 1.0,
    method: str = "linear",
) -> Real:
    """Running r.m.s. of a weighted signal (ISO 2631-1 Eqs. (2)/(3)).

    :param signal: Frequency-weighted acceleration signal (1-D), in m/s2.
    :param fs: Sampling frequency, in hertz.
    :param integration_time: Averaging time ``tau``, in seconds (default 1 s,
        the "slow" constant of ISO 2631-1 6.3.1).
    :param method: ``"linear"`` (Eq. (2), a sliding rectangular average) or
        ``"exponential"`` (Eq. (3), a single-pole average).
    :return: The running r.m.s. value ``a_w(t0)`` per sample, in m/s2.
    :raises ValueError: for a bad signal, non-positive ``fs``/``tau`` or an
        unknown ``method``.
    """
    x = _weighted_signal(signal)
    fs = _positive_fs(fs)
    tau = float(integration_time)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("'integration_time' must be positive and finite.")
    power = x**2
    if method == "linear":
        window = max(1, round(tau * fs))
        # Causal sliding mean over the trailing ``tau`` seconds, O(n) via a
        # prefix sum. The divisor is always the full window, so the samples
        # before ``t = tau`` behave as if the record were zero-padded at the
        # front: the running r.m.s. ramps up from zero over the first ``tau``
        # seconds, like an integrating meter switched on at ``t = 0``.
        csum = np.concatenate(([0.0], np.cumsum(power)))
        lo = np.maximum(0, np.arange(power.size) + 1 - window)
        mean_power = (csum[1:] - csum[lo]) / window
    elif method == "exponential":
        # Single-pole IIR average: y[i] = (1-alpha) y[i-1] + alpha x[i].
        alpha = 1.0 - math.exp(-1.0 / (tau * fs))
        mean_power = sig.lfilter([alpha], [1.0, -(1.0 - alpha)], power)
    else:
        raise ValueError("'method' must be 'linear' or 'exponential'.")
    return np.sqrt(mean_power)


def mtvv(signal: ArrayLike, fs: float, *, integration_time: float = 1.0) -> float:
    """Maximum transient vibration value (ISO 2631-1 Eq. (4)).

    ``MTVV = max a_w(t0)``, the peak of the 1 s running r.m.s. value.

    :param signal: Frequency-weighted acceleration signal (1-D), in m/s2.
    :param fs: Sampling frequency, in hertz.
    :param integration_time: Running-r.m.s. averaging time, in seconds (1 s).
    :return: The MTVV, in m/s2.
    """
    return float(
        np.max(running_rms(signal, fs, integration_time=integration_time))
    )


def vibration_dose_value(signal: ArrayLike, fs: float) -> float:
    """Vibration dose value ``VDV`` (ISO 2631-1 Eq. (5)).

    ``VDV = ( integral a_w(t)^4 dt )^(1/4)``, in m/s^1,75; more sensitive to
    peaks than the r.m.s. value.

    :param signal: Frequency-weighted acceleration signal (1-D), in m/s2.
    :param fs: Sampling frequency, in hertz.
    :return: The VDV, in m/s^1,75.
    :raises ValueError: for a bad signal or non-positive ``fs``.
    """
    x = _weighted_signal(signal)
    fs = _positive_fs(fs)
    return float(np.sum(x**4 / fs) ** 0.25)


def motion_sickness_dose_value(signal: ArrayLike, fs: float) -> float:
    """Motion sickness dose value ``MSDV`` (ISO 2631-1 clause 9; 8041-1 3.1.2.5).

    ``MSDV = ( integral a_w(t)^2 dt )^(1/2)``, in m/s^1,5; the ``Wf``-weighted
    signal is the intended input.

    :param signal: Frequency-weighted acceleration signal (1-D), in m/s2.
    :param fs: Sampling frequency, in hertz.
    :return: The MSDV, in m/s^1,5.
    :raises ValueError: for a bad signal or non-positive ``fs``.
    """
    x = _weighted_signal(signal)
    fs = _positive_fs(fs)
    return float(np.sum(x**2 / fs) ** 0.5)


def crest_factor(signal: ArrayLike) -> float:
    """Crest factor of a weighted signal (ISO 2631-1 clause 6.2.1).

    The modulus of the ratio of the peak weighted acceleration to its r.m.s.
    value.  ISO 2631-1 6.2.2 deems the basic (r.m.s.) method adequate for a
    crest factor up to 9.

    Emits a :class:`HumanVibrationWarning` when the crest factor exceeds 9,
    the threshold above which ISO 2631-1 6.2.2 deems the basic method
    inadequate.

    :param signal: Frequency-weighted acceleration signal (1-D), in m/s2.
    :return: The crest factor (dimensionless); ``0`` for an all-zero signal.
    """
    x = _weighted_signal(signal)
    rms = float(np.sqrt(np.mean(x**2)))
    # rms is a root-mean-square, so it is >= 0; <= 0 means an all-zero signal.
    if rms <= 0.0:
        return 0.0
    cf = float(np.max(np.abs(x)) / rms)
    if cf > 9.0:
        warnings.warn(
            f"Crest factor {cf:.1f} exceeds 9; the basic r.m.s. method may be "
            "inadequate (ISO 2631-1, 6.2.2) - consider the VDV or running "
            "r.m.s. dose measures.",
            HumanVibrationWarning,
            stacklevel=2,
        )
    return cf


# ---------------------------------------------------------------------------
# Vector sum and daily exposure (ISO 2631-1 Eq. (10); ISO 5349-1 Eqs. (1)-(3)).
# ---------------------------------------------------------------------------
def vibration_total_value(
    components: ArrayLike,
    *,
    k: ArrayLike | None = None,
) -> float:
    """Vibration total value ``a_v`` / ``a_hv`` (ISO 2631-1 Eq. (10)).

    ``a_v = sqrt( sum_j k_j^2 * a_wj^2 )`` over the (up to three) axis-weighted
    r.m.s. accelerations.  With ``k = None`` the unweighted vector sum of
    ISO 5349-1 Eq. (1) (``a_hv``, all ``k = 1``) is returned.

    :param components: Axis-weighted r.m.s. accelerations ``a_wj``, in m/s2.
    :param k: Optional per-axis multiplying factors ``k_j`` (ISO 2631-1 7.2.3);
        ``None`` uses unity for every axis.
    :return: The vibration total value, in m/s2.
    :raises ValueError: if ``components`` is empty or ``k`` differs in length.
    """
    comp = np.atleast_1d(np.asarray(components, dtype=np.float64))
    if comp.ndim != 1 or comp.size == 0:
        raise ValueError("'components' must be a non-empty 1-D array.")
    if k is None:
        factors = np.ones_like(comp)
    else:
        factors = np.atleast_1d(np.asarray(k, dtype=np.float64))
        if factors.shape != comp.shape:
            raise ValueError("'k' must have the same length as 'components'.")
    return float(np.sqrt(np.sum((factors * comp) ** 2)))


def wbv_exposure_basis(a_wx: float, a_wy: float, a_wz: float) -> float:
    """Whole-body exposure basis of Directive 2002/44/EC (Annex, Part B).

    ``max(1,4*a_wx, 1,4*a_wy, a_wz)`` - the Directive bases the whole-body
    daily exposure ``A(8)`` on the *highest* of the frequency-weighted axis
    values ``1,4*a_wx``, ``1,4*a_wy``, ``a_wz`` for a seated or standing
    worker (Annex, Part B, point 1, with the ISO 2631-1 clause 7.2.3
    multiplying factors), **not** on the ISO 2631-1 Eq. (10) vector total
    ``a_v``.  Feed the returned dominant-axis value to
    :func:`daily_vibration_exposure` (``kind="wbv"``) for a
    Directive-conforming whole-body assessment; the hand-arm basis is the
    vector total ``a_hv`` instead (Annex, Part A, point 1).

    :param a_wx: ``Wd``-weighted r.m.s. acceleration on the x axis, in m/s2.
    :param a_wy: ``Wd``-weighted r.m.s. acceleration on the y axis, in m/s2.
    :param a_wz: ``Wk``-weighted r.m.s. acceleration on the z axis, in m/s2.
    :return: The dominant-axis value ``max(1,4*a_wx, 1,4*a_wy, a_wz)``, in m/s2.
    :raises ValueError: for a negative or non-finite axis value.
    """
    values = (float(a_wx), float(a_wy), float(a_wz))
    if any(not math.isfinite(v) or v < 0.0 for v in values):
        raise ValueError("The axis r.m.s. accelerations must be non-negative.")
    return max(1.4 * values[0], 1.4 * values[1], values[2])


def daily_exposure(total_value: float, duration_s: float) -> float:
    """Daily exposure ``A(8)`` for one operation (ISO 5349-1 Eq. (2)).

    ``A(8) = a_hv * sqrt(T / T0)`` with ``T0 = 8 h``.  The identical form gives
    the whole-body ``A(8)`` of Directive 2002/44/EC, whose magnitude is the
    Annex Part B dominant-axis value (see :func:`wbv_exposure_basis`) rather
    than a vector total.

    :param total_value: Vibration total value ``a_hv`` (or ``a_v``), in m/s2.
    :param duration_s: Daily exposure duration ``T``, in seconds.
    :return: The daily exposure ``A(8)``, in m/s2.
    :raises ValueError: for a negative magnitude or duration.
    """
    a = float(total_value)
    t = float(duration_s)
    if a < 0.0 or t < 0.0:
        raise ValueError("'total_value' and 'duration_s' must be non-negative.")
    return a * math.sqrt(t / REFERENCE_DURATION_S)


#: Alias for :func:`daily_exposure` reading naturally in a hand-arm context.
partial_exposure = daily_exposure


def combine_partial_exposures(partials: ArrayLike) -> float:
    """Combine partial exposures into ``A(8)`` (ISO 5349-1/-2 Eq. (3)).

    ``A(8) = sqrt( sum_i A_i(8)^2 )``.

    :param partials: Partial exposures ``A_i(8)``, in m/s2.
    :return: The combined daily exposure ``A(8)``, in m/s2.
    :raises ValueError: if ``partials`` is empty.
    """
    parts = np.atleast_1d(np.asarray(partials, dtype=np.float64))
    if parts.ndim != 1 or parts.size == 0:
        raise ValueError("'partials' must be a non-empty 1-D array.")
    return float(np.sqrt(np.sum(parts**2)))


def hav_daily_exposure(
    total_values: ArrayLike,
    durations_s: ArrayLike,
) -> float:
    """Daily exposure ``A(8)`` for several operations (ISO 5349-1 Eq. (3)).

    ``A(8) = sqrt( (1/T0) * sum_i a_hvi^2 * T_i )``.

    :param total_values: Vibration total value ``a_hvi`` per operation, in m/s2.
    :param durations_s: Duration ``T_i`` per operation, in seconds.
    :return: The daily exposure ``A(8)``, in m/s2.
    :raises ValueError: if the inputs differ in length, are empty or negative.
    """
    ahv = np.atleast_1d(np.asarray(total_values, dtype=np.float64))
    t = np.atleast_1d(np.asarray(durations_s, dtype=np.float64))
    if ahv.ndim != 1 or ahv.size == 0 or ahv.shape != t.shape:
        raise ValueError(
            "'total_values' and 'durations_s' must be non-empty, 1-D and "
            "equal-length."
        )
    if np.any(ahv < 0.0) or np.any(t < 0.0):
        raise ValueError("'total_values' and 'durations_s' must be non-negative.")
    return float(np.sqrt(np.sum(ahv**2 * t) / REFERENCE_DURATION_S))


def energy_equivalent_acceleration(
    magnitudes: ArrayLike,
    durations_s: ArrayLike,
) -> float:
    """Energy-equivalent weighted acceleration (ISO 2631-1 Eq. (B.3)).

    ``a_w,e = sqrt( sum a_wi^2 * T_i / sum T_i )``.

    :param magnitudes: Weighted r.m.s. magnitudes ``a_wi``, in m/s2.
    :param durations_s: Duration ``T_i`` per period, in seconds.
    :return: The energy-equivalent magnitude ``a_w,e``, in m/s2.
    :raises ValueError: if the inputs differ in length, are empty, or the total
        duration is zero.
    """
    a = np.atleast_1d(np.asarray(magnitudes, dtype=np.float64))
    t = np.atleast_1d(np.asarray(durations_s, dtype=np.float64))
    if a.ndim != 1 or a.size == 0 or a.shape != t.shape:
        raise ValueError(
            "'magnitudes' and 'durations_s' must be non-empty, 1-D and "
            "equal-length."
        )
    if np.any(t < 0.0):
        raise ValueError("'durations_s' must be non-negative.")
    total = float(np.sum(t))
    if total <= 0.0:
        raise ValueError("The total duration must be positive.")
    return float(np.sqrt(np.sum(a**2 * t) / total))


def hav_vwf_lifetime_years(a8: float) -> float:
    """Years to 10 % vibration-white-finger prevalence (ISO 5349-1 Eq. (C.1)).

    ``Dy = 31,8 * A(8)^(-1,06)`` - the group-mean lifetime exposure that
    produces finger blanching in 10 % of an exposed group (informative
    Annex C).

    :param a8: Daily vibration exposure ``A(8)``, in m/s2 (> 0).
    :return: The lifetime exposure duration ``Dy``, in years.
    :raises ValueError: if ``a8`` is not positive.
    """
    a = float(a8)
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("'a8' must be a positive daily exposure.")
    return float(31.8 * a**-1.06)


# ---------------------------------------------------------------------------
# Exposure assessment against Directive 2002/44/EC action / limit values.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExposureAssessment:
    """A daily exposure assessed against the Directive 2002/44/EC values.

    :ivar value: The assessed daily exposure ``A(8)`` (or VDV), in its unit.
    :ivar kind: ``"hav"`` or ``"wbv"``.
    :ivar metric: ``"a8"`` (m/s2) or ``"vdv"`` (m/s^1,75).
    :ivar action_value: The exposure action value (EAV).
    :ivar limit_value: The exposure limit value (ELV).
    :ivar exceeds_action: Whether ``value`` reaches or exceeds the EAV.
    :ivar exceeds_limit: Whether ``value`` reaches or exceeds the ELV.
    :ivar zone: ``"below action"``, ``"action"`` (EAV<=value<ELV) or
        ``"limit"`` (value>=ELV).
    """

    value: float
    kind: str
    metric: str
    action_value: float
    limit_value: float
    exceeds_action: bool
    exceeds_limit: bool
    zone: str


def exposure_assessment(
    value: float,
    *,
    kind: str,
    metric: str = "a8",
) -> ExposureAssessment:
    """Assess a daily exposure against Directive 2002/44/EC (Article 3).

    :param value: The daily exposure ``A(8)`` in m/s2 (``metric="a8"``) or the
        vibration dose value in m/s^1,75 (``metric="vdv"``, whole-body only).
    :param kind: ``"hav"`` (hand-arm) or ``"wbv"`` (whole-body).
    :param metric: ``"a8"`` (default) or ``"vdv"``.
    :return: An :class:`ExposureAssessment`.
    :raises ValueError: for an unknown ``kind``/``metric`` combination or a
        negative ``value``.
    """
    v = float(value)
    if v < 0.0:
        raise ValueError("'value' must be non-negative.")
    if kind == "hav" and metric == "a8":
        eav, elv = HAV_EAV_A8, HAV_ELV_A8
    elif kind == "wbv" and metric == "a8":
        eav, elv = WBV_EAV_A8, WBV_ELV_A8
    elif kind == "wbv" and metric == "vdv":
        eav, elv = WBV_EAV_VDV, WBV_ELV_VDV
    else:
        raise ValueError(
            "kind must be 'hav' or 'wbv'; metric 'a8' (both) or 'vdv' (wbv only)."
        )
    exceeds_action = v >= eav
    exceeds_limit = v >= elv
    if exceeds_limit:
        zone = "limit"
    elif exceeds_action:
        zone = "action"
    else:
        zone = "below action"
    return ExposureAssessment(
        value=v,
        kind=kind,
        metric=metric,
        action_value=eav,
        limit_value=elv,
        exceeds_action=exceeds_action,
        exceeds_limit=exceeds_limit,
        zone=zone,
    )


@dataclass(frozen=True)
class DailyVibrationExposure:
    """A daily exposure built from several operations, with its assessment.

    :ivar a8: The daily exposure ``A(8)``, in m/s2.
    :ivar labels: A label per operation.
    :ivar total_values: Vibration total value ``a_hvi`` per operation, in m/s2.
    :ivar durations_s: Duration ``T_i`` per operation, in seconds.
    :ivar partials: Partial exposure ``A_i(8)`` per operation, in m/s2.
    :ivar assessment: The :class:`ExposureAssessment` of ``a8``.
    """

    a8: float
    labels: tuple[str, ...]
    total_values: Real
    durations_s: Real
    partials: Real
    assessment: ExposureAssessment

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the partial exposures against the EAV / ELV thresholds.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.
        """
        from .._i18n import check_language
        from .._plot.vibration import plot_daily_exposure

        return plot_daily_exposure(self, ax=ax, language=check_language(language), **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a daily vibration exposure assessment fiche to a PDF.

        Writes a one-page assessment sheet laid out like the hand-arm /
        whole-body vibration exposure calculators of the occupational-hygiene
        practice: the standard-basis line naming the applied ISO method
        (ISO 5349-1/-2:2001 for hand-transmitted vibration, ISO 2631-1:1997 for
        whole-body vibration) and the Directive 2002/44/EC it is assessed
        against, an optional metadata header (company, operator/worker,
        workplace, instrumentation, calibration), the per-operation exposure
        analysis (the vibration magnitude - the hand-arm vector total ``a_hv``
        or the whole-body Directive Part B dominant-axis value ``a_w,max`` -
        the daily exposure time ``T_i`` and
        the partial exposure ``A_i(8)`` of ISO 5349-1/-2 Eq. (2), closed by the
        daily total and the combined ``A(8)`` of Eq. (3)), the per-operation
        contribution chart, the boxed ``A(8)`` with its exposure zone, an
        assessment table against the exposure action value (EAV) and exposure
        limit value (ELV) of Directive 2002/44/EC (Article 3), a PASS/FAIL
        verdict against the limit value, and a footer identity/disclaimer block.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header identity (``client`` is the company, ``specimen`` the
            operator/worker, ``test_room`` the workplace) plus the
            ``instrumentation`` and ``calibration`` free-text fields and the
            footer identity.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When True, the operations table adds each operation's
            share of the daily vibration energy ``A_i(8)^2 / A(8)^2``.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
        :raises ImportError: If reportlab or matplotlib is not installed. The
            fiche always embeds the contribution chart, so both are required
            (``pip install "phonometry[report,plot]"``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.human_vibration import render_human_vibration_report

        return render_human_vibration_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def daily_vibration_exposure(
    total_values: ArrayLike,
    durations_s: ArrayLike,
    *,
    kind: str,
    labels: list[str] | tuple[str, ...] | None = None,
) -> DailyVibrationExposure:
    """Daily exposure from several operations, assessed (ISO 5349 + Directive).

    Combines the operations via the partial exposures of ISO 5349-1/-2
    Eqs. (2)/(3) and assesses the resulting ``A(8)`` against the Directive
    2002/44/EC action and limit values for ``kind``.

    The Directive fixes the per-operation magnitude each kind must be fed
    with (Annex, points 1): for ``kind="hav"`` the ISO 5349-1 Eq. (1) vector
    total ``a_hv`` (Part A); for ``kind="wbv"`` the *highest* frequency-
    weighted axis value ``max(1,4*a_wx, 1,4*a_wy, a_wz)`` that
    :func:`wbv_exposure_basis` returns (Part B), **not** the ISO 2631-1
    Eq. (10) vector total ``a_v``.

    :param total_values: Per-operation vibration magnitude, in m/s2: the
        hand-arm vibration total value ``a_hv,i`` (``kind="hav"``) or the
        Directive Part B dominant-axis value of :func:`wbv_exposure_basis`
        (``kind="wbv"``).
    :param durations_s: Duration ``T_i`` per operation, in seconds.
    :param kind: ``"hav"`` or ``"wbv"`` (selects the EAV/ELV).
    :param labels: Optional operation labels; defaults to ``op 1``, ``op 2``, ...
    :return: A :class:`DailyVibrationExposure` with ``.plot()``.
    :raises ValueError: if the inputs differ in length or ``labels`` mismatches.
    """
    ahv = np.atleast_1d(np.asarray(total_values, dtype=np.float64))
    t = np.atleast_1d(np.asarray(durations_s, dtype=np.float64))
    if ahv.ndim != 1 or ahv.size == 0 or ahv.shape != t.shape:
        raise ValueError(
            "'total_values' and 'durations_s' must be non-empty, 1-D and "
            "equal-length."
        )
    if labels is None:
        labels = tuple(f"op {i + 1}" for i in range(ahv.size))
    else:
        labels = tuple(labels)
        if len(labels) != ahv.size:
            raise ValueError("'labels' must match the number of operations.")
    partials = np.array(
        [daily_exposure(float(a), float(dt)) for a, dt in zip(ahv, t)],
        dtype=np.float64,
    )
    a8 = combine_partial_exposures(partials)
    return DailyVibrationExposure(
        a8=a8,
        labels=labels,
        total_values=ahv,
        durations_s=t,
        partials=partials,
        assessment=exposure_assessment(a8, kind=kind, metric="a8"),
    )

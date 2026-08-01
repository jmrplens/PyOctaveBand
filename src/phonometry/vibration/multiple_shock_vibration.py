#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Whole-body vibration containing multiple shocks (ISO 2631-5:2018).

Implements the normative Clause 5 spinal-response model and the Annex C
assessment of adverse health effects for the vertical (``z``) axis.

The 2018 edition is vertical-axis only by design: clause 4 (delineation,
item a) neglects the ``x`` and ``y`` contributions to spinal compression, the
seat-to-spine transfer function of clause 5.2 is the vertical seat-to-lumbar
response, and the Annex C stress conversion :math:`m_z` is the vertical one.
The horizontal spinal model of the withdrawn 2004 edition is not reproduced.
Assess horizontal whole-body exposure with the ISO 2631-1 metrics in this
domain instead: the weighted r.m.s. acceleration
(:func:`~phonometry.vibration.weighted_acceleration`) and the vibration dose
value (:func:`~phonometry.vibration.vibration_dose_value`).

A seat-to-spine transfer function :math:`H(\omega)` (clause 5.2, Formula 1)
maps the measured seat acceleration :math:`a_z(t)` to the spinal response
acceleration

.. math::

   A_z(t) = F^{-1}[H(\omega) \, F[a_z(t)]] \tag{Formula 2}

The standard assumes a *conditioned* input: :math:`H` has unity
transmissibility at 0 Hz, so any DC offset in the record (e.g. the gravity
component of a non-AC-coupled accelerometer) passes straight into
:math:`A_z(t)` and corrupts the response peaks; remove the mean (high-pass)
before processing. The acceleration dose is

.. math::

   D_z = 1.07 \left( \sum_i A_{z,i}^6 \right)^{1/6} \tag{Formula 3}

over the positive response peaks, scaled to a daily dose

.. math::

   D_{zd} = D_z \, (t_d/t_m)^{1/6} \tag{Formula 4/5}

Annex C turns the daily dose into an injury risk: the daily compressive
stress :math:`S_d` (Formula C.1), the age-cumulated stress variable
:math:`R` (Formulae C.3/C.4) and the Weibull probability of lumbar injury
:math:`P` (Formula C.5, Table C.1):

.. math::

   S_d = m_z D_{zd} \tag{Formula C.1}

   R = \left[ \sum_i \left(
   S_d N^{1/6} / (S_{u,i} - S_\text{stat})
   \right)^6 \right]^{1/6} \tag{Formulae C.3/C.4}

   P = 1 - \exp\left(-(R/\alpha)^\beta\right) \tag{Formula C.5}

The Annex A / Annex E model (intervertebral compressive forces via a
finite-element model distributed by ISO) is not reproducible from the standard
text and is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._internal.types import as_float_or_array
from .._internal.validation import require_1d_signal, require_choice, require_positive
from ..hearing.threshold import SEXES as _SEXES

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

from numpy.typing import ArrayLike

# ---------------------------------------------------------------------------
# Normative constants (Clause 5).
# ---------------------------------------------------------------------------

#: Seat-to-spine transfer function complex zero ``(w1, zeta1)`` (Formula 1).
_TRANSFER_ZERO: tuple[float, float] = (34.0, 0.35)

#: Seat-to-spine transfer function complex poles ``(wk, zetak)`` (Formula 1).
_TRANSFER_POLES: tuple[tuple[float, float], ...] = (
    (31.0, 0.21),
    (230.0, 0.88),
    (260.0, 0.80),
    (320.0, 0.40),
    (380.0, 0.75),
    (420.0, 0.65),
)

#: Amplitude response factor of the spine in the dose (Formula 3).
DOSE_AMPLITUDE_FACTOR = 1.07

#: Palmgren-Miner fatigue exponent of the dose (Formula 3).
DOSE_EXPONENT = 6.0

# ---------------------------------------------------------------------------
# Annex C constants (health-effect assessment).
# ---------------------------------------------------------------------------

#: Acceleration gravity used for the static stress (clause C, ``9.81 m/s2``).
GRAVITY = 9.81

#: Stress conversion ``mz`` (MPa per m/s2) by sex (Formula C.2).
MZ_MALE = 0.029
MZ_FEMALE = 0.025

#: Ultimate lumbar strength intercept ``6.75 MPa`` (Formula C.4).
ULTIMATE_STRENGTH_INTERCEPT = 6.75

#: Age slope ``Sage`` of the ultimate strength by sex, MPa/year (Formula C.4).
STRENGTH_AGE_SLOPE_MALE = 0.052
STRENGTH_AGE_SLOPE_FEMALE = 0.039

#: Weibull coefficients ``(alpha, beta)`` of the injury model by sex (Table C.1).
WEIBULL_MALE: tuple[float, float] = (1.613, 2.799)
WEIBULL_FEMALE: tuple[float, float] = (0.959, 3.709)

#: R values for 10 %, 50 % and 90 % risk of injury by sex (Table C.2).
RISK_THRESHOLDS_MALE: tuple[float, float, float] = (0.72, 1.42, 2.17)
RISK_THRESHOLDS_FEMALE: tuple[float, float, float] = (0.52, 0.87, 1.20)


def _mz_for_sex(sex: str) -> float:
    return MZ_MALE if sex == "male" else MZ_FEMALE


# ---------------------------------------------------------------------------
# Clause 5: spinal response and acceleration dose.
# ---------------------------------------------------------------------------


def seat_to_spine_transfer(frequencies: ArrayLike) -> np.ndarray:
    r"""Seat-to-spine transfer function :math:`H(\omega)` (clause 5.2,
    Formula 1).

    A single complex zero and six complex poles map the seat acceleration to
    the vertical spinal response; the transmissibility is unity at 0 Hz.

    :param frequencies: Frequencies at which to evaluate ``H``, in hertz.
    :return: The complex frequency response, aligned with ``frequencies``.
    """
    freq = np.asarray(frequencies, dtype=np.float64)
    jw = 2.0j * np.pi * freq

    w0, z0 = _TRANSFER_ZERO
    response = 1.0 + 2.0 * z0 * (jw / w0) + (jw / w0) ** 2
    for wk, zk in _TRANSFER_POLES:
        response = response / (1.0 + 2.0 * zk * (jw / wk) + (jw / wk) ** 2)
    return np.asarray(response, dtype=np.complex128)


def spinal_response(acceleration: ArrayLike, fs: float) -> np.ndarray:
    r"""Vertical spinal response :math:`A_z(t)` (clause 5.2, Formula 2).

    Applies the seat-to-spine transfer function to the measured conditioned
    seat acceleration in the frequency domain and returns the time-domain
    response by the inverse transform.

    The input must be **conditioned (DC-removed)**: the transfer function is
    unity at 0 Hz by design (clause 5.2), so a DC offset (e.g. the 1 g
    gravity component of a DC-coupled accelerometer) is passed unattenuated
    and produces a spurious constant shift in :math:`A_z(t)` that corrupts
    the positive response peaks of the dose. Subtract the mean (or
    high-pass) of :math:`a_z(t)` before calling.

    :param acceleration: Measured, conditioned (zero-mean) vertical seat
        acceleration :math:`a_z(t)`, m/s2.
    :param fs: Sampling frequency, in hertz.
    :return: The spinal response acceleration :math:`A_z(t)`, m/s2, same
        length.
    """
    fs = require_positive(fs, "fs")
    az = np.asarray(acceleration, dtype=np.float64)
    if az.ndim != 1:
        raise ValueError("acceleration must be a 1-D time series.")
    if az.size == 0:
        raise ValueError("acceleration must not be empty.")
    spectrum = np.fft.rfft(az)
    freq = np.fft.rfftfreq(az.size, d=1.0 / fs)
    response = np.fft.irfft(spectrum * seat_to_spine_transfer(freq), n=az.size)
    return np.asarray(response, dtype=np.float64)


def response_peaks(response: ArrayLike) -> np.ndarray:
    r"""Positive response peaks :math:`A_{z,i}` (clause 5.3).

    A peak is the maximum value of the response between two consecutive zero
    crossings; only positive peaks are counted.

    :param response: The spinal response acceleration :math:`A_z(t)`.
    :return: The positive peak values, in the order they occur.
    """
    sig = require_1d_signal(response, name="response")
    if sig.size == 0:
        return np.array([], dtype=np.float64)
    # Bracket the positive samples with False sentinels so the first and last
    # runs are closed, then take the maximum of each positive run.
    positive = np.empty(sig.size + 2, dtype=bool)
    positive[0] = positive[-1] = False
    positive[1:-1] = sig > 0.0
    starts = np.where(~positive[:-1] & positive[1:])[0]
    if starts.size == 0:
        return np.array([], dtype=np.float64)
    return np.maximum.reduceat(sig, starts)


def dose_from_peaks(peaks: ArrayLike) -> float:
    r"""Acceleration dose :math:`D_z` from response peaks (clause 5.3,
    Formula 3).

    :param peaks: The positive response peaks :math:`A_{z,i}`, m/s2.
    :return: The acceleration dose
        :math:`D_z = 1.07 \left( \sum A_{z,i}^6 \right)^{1/6}`, m/s2.
    """
    az = np.asarray(peaks, dtype=np.float64).ravel()
    if az.size == 0:
        return 0.0
    total = float(np.sum(az**DOSE_EXPONENT))
    return float(DOSE_AMPLITUDE_FACTOR * total ** (1.0 / DOSE_EXPONENT))


def acceleration_dose(acceleration: ArrayLike, fs: float) -> float:
    r"""Acceleration dose :math:`D_z` from a seat acceleration time history.

    Filters the acceleration through the seat-to-spine transfer function
    (Formula 2), takes the positive response peaks and combines them by
    Formula 3. The input must be conditioned (DC-removed); see
    :func:`spinal_response`.

    :param acceleration: Measured, conditioned (zero-mean) vertical seat
        acceleration :math:`a_z(t)`, m/s2.
    :param fs: Sampling frequency, in hertz.
    :return: The acceleration dose :math:`D_z`, m/s2.
    """
    return dose_from_peaks(response_peaks(spinal_response(acceleration, fs)))


def daily_dose(dose: float, exposure_time: float, measurement_time: float) -> float:
    r"""Daily acceleration dose :math:`D_{zd}` (clause 5.3, Formula 4).

    :param dose: The measured acceleration dose :math:`D_z`, m/s2.
    :param exposure_time: Daily exposure period :math:`t_d` (any time
        unit).
    :param measurement_time: Period :math:`t_m` over which :math:`D_z`
        was measured (same unit as ``exposure_time``).
    :return: The daily dose :math:`D_{zd} = D_z (t_d/t_m)^{1/6}`, m/s2.
    """
    if not measurement_time > 0.0 or not exposure_time > 0.0:
        raise ValueError("exposure_time and measurement_time must be positive.")
    return float(dose * (exposure_time / measurement_time) ** (1.0 / DOSE_EXPONENT))


def daily_dose_multi(
    doses: ArrayLike, exposure_times: ArrayLike, measurement_times: ArrayLike
) -> float:
    r"""Daily dose from several exposure conditions (clause 5.3, Formula 5).

    :param doses: Acceleration dose :math:`D_{z,j}` of each condition,
        m/s2.
    :param exposure_times: Daily exposure duration :math:`t_{d,j}` of
        each condition.
    :param measurement_times: Measurement duration :math:`t_{m,j}` of
        each condition.
    :return: The combined daily dose
        :math:`D_{zd} = \left[ \sum_j D_{z,j}^6 \,
        (t_{d,j}/t_{m,j}) \right]^{1/6}`, m/s2.
    """
    dz = np.asarray(doses, dtype=np.float64).ravel()
    td = np.asarray(exposure_times, dtype=np.float64).ravel()
    tm = np.asarray(measurement_times, dtype=np.float64).ravel()
    if not dz.shape == td.shape == tm.shape:
        raise ValueError("doses, exposure_times and measurement_times must match.")
    if dz.size == 0:
        raise ValueError("at least one condition is required.")
    if not np.all(tm > 0.0) or not np.all(td > 0.0):
        raise ValueError("exposure_times and measurement_times must be positive.")
    total = float(np.sum(dz**DOSE_EXPONENT * (td / tm)))
    return float(total ** (1.0 / DOSE_EXPONENT))


# ---------------------------------------------------------------------------
# Annex C: assessment of adverse health effects.
# ---------------------------------------------------------------------------


def compression_dose(daily_dose_value: float, *, mz: float = MZ_MALE) -> float:
    r"""Daily compressive stress :math:`S_d` (Annex C, Formula C.1).

    :param daily_dose_value: The daily acceleration dose :math:`D_{zd}`,
        m/s2.
    :param mz: Stress conversion :math:`m_z` (MPa per m/s2); default the
        82 kg male value :data:`MZ_MALE`. See :data:`MZ_FEMALE`.
    :return: The daily compressive stress :math:`S_d = m_z D_{zd}`, MPa.
    """
    return mz * daily_dose_value


def static_stress(mz: float = MZ_MALE) -> float:
    r"""Static compressive stress :math:`S_\text{stat} = m_z \cdot 9.81`
    (Annex C), MPa."""
    return mz * GRAVITY


def ultimate_strength(age: ArrayLike, *, sex: Literal["male", "female"] = "male") -> np.ndarray:
    r"""Ultimate lumbar strength :math:`S_u` at an age (Annex C,
    Formula C.4).

    :param age: Age :math:`b + i`, in years.
    :param sex: ``"male"`` or ``"female"`` (sets the age slope
        :math:`S_\text{age}`).
    :return: The ultimate strength
        :math:`S_u = 6.75 - S_\text{age} (b+i)`, MPa.
    """
    require_choice(sex, "sex", _SEXES)
    slope = STRENGTH_AGE_SLOPE_MALE if sex == "male" else STRENGTH_AGE_SLOPE_FEMALE
    years = np.asarray(age, dtype=np.float64)
    return ULTIMATE_STRENGTH_INTERCEPT - slope * years


def injury_risk(
    daily_compression: float,
    *,
    start_age: float,
    years: int,
    days_per_year: float,
    sex: Literal["male", "female"] = "male",
    mz: float | None = None,
) -> float:
    r"""Cumulative injury stress variable :math:`R` (Annex C, Formula C.3).

    Accumulates the daily compressive stress over the exposure years, each year
    weighted by the reducing ultimate strength of the ageing spine.

    :param daily_compression: The daily compressive stress :math:`S_d`,
        MPa.
    :param start_age: Age :math:`b` at which the exposure started, in
        years.
    :param years: Number of exposure years ``n``.
    :param days_per_year: Number of exposure days per year ``N``.
    :param sex: ``"male"`` or ``"female"``.
    :param mz: Stress conversion for the static stress
        :math:`S_\text{stat} = m_z \cdot 9.81`; defaults to the
        sex-specific value.
    :return: The stress variable :math:`R`.
    :raises ValueError: if ``years`` is not positive or the spine strength
        is exhausted (:math:`S_u - S_\text{stat} \le 0`) within the
        exposure period.
    """
    require_choice(sex, "sex", _SEXES)
    if years <= 0:
        raise ValueError("years must be a positive integer.")
    if not days_per_year > 0.0:
        raise ValueError("days_per_year must be positive.")
    mz = _mz_for_sex(sex) if mz is None else mz
    s_stat = static_stress(mz)
    ages = start_age + np.arange(years, dtype=np.float64)
    strength = ultimate_strength(ages, sex=sex) - s_stat
    if np.any(strength <= 0.0):
        raise ValueError(
            "ultimate strength minus static stress is non-positive within the "
            "exposure period; the age range exceeds the model's validity."
        )
    numerator = daily_compression * days_per_year ** (1.0 / DOSE_EXPONENT)
    total = float(np.sum((numerator / strength) ** DOSE_EXPONENT))
    return float(total ** (1.0 / DOSE_EXPONENT))


def injury_probability(risk: ArrayLike, *, sex: Literal["male", "female"] = "male") -> np.ndarray | float:
    r"""Probability of lumbar injury :math:`P(R)` (Annex C, Formula C.5).

    :param risk: The stress variable :math:`R` (see :func:`injury_risk`);
        scalar or array-like.
    :param sex: ``"male"`` or ``"female"`` (sets the Weibull coefficients).
    :return: The injury probability
        :math:`P = 1 - \exp(-(R/\alpha)^\beta)` in 0-1; a float for a
        scalar input, otherwise an array. Negative :math:`R` gives 0.
    """
    require_choice(sex, "sex", _SEXES)
    alpha, beta = WEIBULL_MALE if sex == "male" else WEIBULL_FEMALE
    r = np.asarray(risk, dtype=np.float64)
    prob = 1.0 - np.exp(-((np.maximum(r, 0.0) / alpha) ** beta))
    return as_float_or_array(prob)


def _risk_thresholds(sex: str) -> tuple[float, float, float]:
    return RISK_THRESHOLDS_MALE if sex == "male" else RISK_THRESHOLDS_FEMALE


@dataclass(frozen=True)
class MultipleShockResult:
    r"""Multiple-shock health assessment (ISO 2631-5:2018, Clause 5 +
    Annex C).

    :ivar sex: ``"male"`` or ``"female"``.
    :ivar acceleration_dose: The acceleration dose :math:`D_z`, m/s2.
    :ivar daily_dose: The daily acceleration dose :math:`D_{zd}`, m/s2.
    :ivar compression_dose: The daily compressive stress :math:`S_d`, MPa.
    :ivar risk: The cumulative stress variable :math:`R`.
    :ivar probability: The probability of lumbar injury :math:`P(R)` in
        0-1.
    :ivar start_age: Age at which the exposure started, in years.
    :ivar years: Number of exposure years.
    :ivar days_per_year: Number of exposure days per year.
    :ivar peaks: The positive response peaks :math:`A_{z,i}` used for the
        dose, m/s2.
    :ivar risk_thresholds: The :math:`R` values for 10 %, 50 % and 90 %
        risk of injury for this sex (Table C.2).
    """

    sex: Literal["male", "female"]
    acceleration_dose: float
    daily_dose: float
    compression_dose: float
    risk: float
    probability: float
    start_age: float
    years: int
    days_per_year: float
    peaks: np.ndarray
    risk_thresholds: tuple[float, float, float]

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the injury-probability curve with this assessment's
        :math:`R`.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.vibration import plot_multiple_shock

        return plot_multiple_shock(self, ax=ax, language=check_language(language), **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a whole-body multiple-shock health-risk fiche to a PDF.

        Writes a one-page health-risk assessment sheet for whole-body vibration
        containing multiple shocks (ISO 2631-5:2018): the standard-basis line
        (Clause 5 spinal response and Annex C risk model), an optional metadata
        header (client, subject, workplace/vehicle, instrumentation,
        calibration), the exposure-scenario grid (subject sex, the age
        ``b`` at which the exposure started, the number of exposure years
        ``n``, the number of exposure days per year ``N`` and the number
        of counted response shocks), the dose-and-stress analysis table
        (the acceleration dose :math:`D_z` of Formula 3, the daily dose
        :math:`D_{zd}` of Formula 4, the daily compressive stress
        :math:`S_d` of Formula C.1, the cumulative stress variable
        :math:`R` of Formula C.3 and the probability of lumbar injury
        :math:`P` of Formula C.5), the injury-probability chart, the boxed
        :math:`R` and :math:`P` with the Annex C risk classification, a
        classification table against the Table C.2 risk levels with a zone
        row, and a footer identity/disclaimer block.

        The Annex C classification is informative (ISO 2631-5:2018 defines no
        exposure limit), so the fiche carries a risk-band zone row rather than a
        PASS/FAIL verdict: :math:`R` is placed among the Table C.2 stress
        variables for 10 / 50 / 90 % risk of injury (low / moderate /
        high / very high probability of an adverse health effect), the
        moderate band matching the Annex C worked example.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header identity (``client``, ``specimen`` the subject,
            ``test_room`` the workplace or vehicle) plus the ``instrumentation``
            and ``calibration`` free-text fields and the footer identity.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform ``.report()`` signature; the
            fiche has one stacked body layout, so it has no effect.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
        :raises ImportError: If reportlab or matplotlib is not installed. The
            fiche always embeds the injury-probability chart, so both are
            required (``pip install "phonometry[report,plot]"``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.iso2631_5 import render_iso2631_5_report

        return render_iso2631_5_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def multiple_shock_assessment(
    acceleration: ArrayLike,
    fs: float,
    *,
    start_age: float,
    years: int,
    days_per_year: float,
    exposure_time: float | None = None,
    measurement_time: float | None = None,
    sex: Literal["male", "female"] = "male",
    mz: float | None = None,
) -> MultipleShockResult:
    r"""Full multiple-shock assessment from a seat acceleration time
    history.

    Chains the Clause 5 dose and the Annex C risk: spinal response
    (Formula 2), acceleration dose (Formula 3), daily dose (Formula 4),
    compressive stress (C.1), stress variable :math:`R` (C.3) and injury
    probability (C.5). The input must be conditioned (DC-removed); see
    :func:`spinal_response`.

    The model is vertical-axis only (clause 4a of the 2018 edition); for
    horizontal whole-body exposure use the ISO 2631-1 metrics in this domain
    (:func:`~phonometry.vibration.weighted_acceleration`,
    :func:`~phonometry.vibration.vibration_dose_value`).

    :param acceleration: Measured, conditioned (zero-mean) vertical seat
        acceleration :math:`a_z(t)`, m/s2.
    :param fs: Sampling frequency, in hertz.
    :param start_age: Age ``b`` at which the exposure started, in years.
    :param years: Number of exposure years ``n``.
    :param days_per_year: Number of exposure days per year ``N``.
    :param exposure_time: Daily exposure period :math:`t_d`; when given
        with ``measurement_time`` the dose is scaled to a daily dose
        (Formula 4), otherwise the measured dose is taken as the daily
        dose.
    :param measurement_time: Period :math:`t_m` over which the record was
        measured.
    :param sex: ``"male"`` or ``"female"``.
    :param mz: Stress conversion :math:`m_z` (MPa per m/s2); defaults to
        the sex-specific value.
    :return: The :class:`MultipleShockResult`.
    """
    require_choice(sex, "sex", _SEXES)
    if (exposure_time is None) != (measurement_time is None):
        raise ValueError(
            "provide both exposure_time and measurement_time to scale to a "
            "daily dose, or neither."
        )
    mz = _mz_for_sex(sex) if mz is None else mz
    peaks = response_peaks(spinal_response(acceleration, fs))
    dz = dose_from_peaks(peaks)
    if exposure_time is not None and measurement_time is not None:
        dzd = daily_dose(dz, exposure_time, measurement_time)
    else:
        dzd = dz
    sd = compression_dose(dzd, mz=mz)
    r = injury_risk(
        sd,
        start_age=start_age,
        years=years,
        days_per_year=days_per_year,
        sex=sex,
        mz=mz,
    )
    prob = float(injury_probability(r, sex=sex))
    return MultipleShockResult(
        sex=sex,
        acceleration_dose=dz,
        daily_dose=dzd,
        compression_dose=sd,
        risk=r,
        probability=prob,
        start_age=start_age,
        years=years,
        days_per_year=days_per_year,
        peaks=peaks,
        risk_thresholds=_risk_thresholds(sex),
    )

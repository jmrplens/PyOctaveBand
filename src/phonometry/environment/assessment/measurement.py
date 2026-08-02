#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Determination of environmental-noise sound pressure levels (ISO 1996-2:2017).

The measurement companion of the ISO 1996-1 descriptors in
:mod:`phonometry.environment`. ISO 1996-2 covers *how* the levels that feed
those descriptors are obtained: the tonal adjustment for prominent tones, the
residual-noise correction, and the measurement-uncertainty budget.

**Tonal audibility (engineering method, ISO 1996-2:2007 Annex C).** From the
energy-summed tone level ``Lpt`` (Formula (C.1)) and the masking-noise level
``Lpn`` in the critical band around the tone, the tonal audibility above the
masking threshold is
:math:`\Delta L_{ta} = L_{pt} - L_{pn} + 2 + \log_{10}[1 + (f_c/502)^{2.5}]` dB
(Formula (C.3)), and the
tonal adjustment is the piecewise function :math:`K_t = 0` for
:math:`\Delta L_{ta} < 4`,
:math:`K_t = \Delta L_{ta} - 4` for :math:`4 \le \Delta L_{ta} \le 10` and
:math:`K_t = 6` for :math:`\Delta L_{ta} > 10`
(Formulae (C.4)–(C.6)). The critical bandwidth is 100 Hz for centre
frequencies up to 500 Hz and 20 % of the centre frequency above (Table C.1).
:func:`assess_tonal_audibility` returns both in a plottable result. (The 2017
edition defers the full engineering method to ISO/PAS 20065; the detailed,
self-contained algorithm implemented here is the 2007/2009 Annex C one.)

**Survey method (ISO 1996-2:2017 Annex K).**
:func:`tonal_seeking_survey` flags a one-third-octave band that exceeds *both*
neighbours by 15 dB (25–125 Hz), 8 dB (160–400 Hz) or 5 dB (500–10 000 Hz).

**Mean-audibility route (ISO 1996-2:2017 Table J.1).**
:func:`tonal_adjustment_from_mean_audibility` maps the ISO/PAS 20065 mean
audibility ``ΔL`` to ``Kt`` (0–6 dB).

**Residual-noise correction (Clause 10.4).**
:math:`L = 10 \log_{10}(10^{L'/10} - 10^{L_{res}/10})` (Formula (16)); with a
residual
within 3 dB of the measured level no correction is allowed; the
*uncorrected* measured level ``L'`` is then the reportable value, as an upper
bound of the specific sound. :func:`gaussian_residual_level` estimates the
residual from percentile levels (Annex I, Formulae (I.1)/(I.2)).

**Measurement uncertainty (Clause 4, Annex F).**
:math:`u = \sqrt{\sum (c_j \cdot u_j)^2}`
(Formula (2)) expanded by :math:`k = 2` (95 %) or :math:`k = 1.3` (80 %). The
residual-correction sensitivity coefficients (Formulae (F.7)/(F.8)) and the
repeated-measurement standard uncertainty (Formulae (17)–(20)) are provided.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.warnings import PhonometryWarning

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Masking-threshold reference frequency in Formula (C.3), in Hz.
_MASKING_REFERENCE_FREQUENCY = 502.0
#: Exponent of the masking-threshold term in Formula (C.3).
_MASKING_EXPONENT = 2.5
#: Constant offset (dB) of the masking-threshold term in Formula (C.3).
_MASKING_OFFSET = 2.0

#: Tonal-audibility breakpoints (dB) of the adjustment (Formulae (C.4)–(C.6)).
_KT_LOW = 4.0
_KT_HIGH = 10.0
#: Maximum tonal adjustment ``Kt`` (dB).
_KT_MAX = 6.0

#: Critical-band centre-frequency split (Table C.1): 100 Hz up to 500 Hz.
_CRITICAL_BAND_SPLIT = 500.0
_CRITICAL_BAND_LOW_WIDTH = 100.0
_CRITICAL_BAND_HIGH_FRACTION = 0.20

#: Coverage factors k (Clause 4): 95 % (k = 2) and 80 % (k = 1.3).
_COVERAGE_FACTORS = {0.95: 2.0, 0.80: 1.3}

#: Gaussian residual-level constant and divisors (Annex I, (I.1)/(I.2)).
_GAUSS_CONSTANT = 0.115
_GAUSS_DIVISOR_L90 = 1.28
_GAUSS_DIVISOR_L95 = 1.65


class EnvironmentalMeasurementWarning(PhonometryWarning):
    """Warning for unreliable environmental-noise determinations."""


def _positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"'{name}' must be a positive, finite number.")
    return scalar


def _finite(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"'{name}' must be finite.")
    return scalar


# --------------------------------------------------------------------------- #
# Tonal audibility -- engineering method (ISO 1996-2:2007 Annex C)
# --------------------------------------------------------------------------- #
def critical_bandwidth(centre_frequency: float) -> float:
    """Critical bandwidth around a tone (ISO 1996-2 Annex C, Table C.1).

    100 Hz for a centre frequency up to 500 Hz, 20 % of the centre frequency
    above 500 Hz.

    :param centre_frequency: Critical-band centre frequency ``fc``, in Hz.
    :return: Critical bandwidth, in Hz.
    :raises ValueError: If ``centre_frequency`` is not positive/finite.
    """
    fc = _positive(centre_frequency, "centre_frequency")
    if fc <= _CRITICAL_BAND_SPLIT:
        return _CRITICAL_BAND_LOW_WIDTH
    return _CRITICAL_BAND_HIGH_FRACTION * fc


def tonal_audibility(
    tone_level: float, masking_noise_level: float, centre_frequency: float
) -> float:
    r"""Tonal audibility above the masking threshold (Formula (C.3)).

    :math:`\Delta L_{ta} = L_{pt} - L_{pn} + 2 + \log_{10}[1 + (f_c/502)^{2.5}]`
    dB.

    :param tone_level: Energy-summed tone level ``Lpt`` in the critical band,
        in dB (see Formula (C.1)).
    :param masking_noise_level: Masking-noise level ``Lpn`` in the critical
        band, in dB (see Formula (C.2)/(C.11)).
    :param centre_frequency: Critical-band centre frequency ``fc``, in Hz.
    :return: Tonal audibility ``ΔLta``, in dB (dB above the masking threshold).
    :raises ValueError: If ``centre_frequency`` is not positive/finite or the
        levels are not finite.
    """
    lpt = _finite(tone_level, "tone_level")
    lpn = _finite(masking_noise_level, "masking_noise_level")
    fc = _positive(centre_frequency, "centre_frequency")
    masking_threshold = _MASKING_OFFSET + np.log10(
        1.0 + (fc / _MASKING_REFERENCE_FREQUENCY) ** _MASKING_EXPONENT
    )
    return float(lpt - lpn + masking_threshold)


def tonal_adjustment(audibility: float) -> float:
    r"""Tonal adjustment ``Kt`` from the audibility (Formulae (C.4)–(C.6)).

    :math:`K_t = 0` for :math:`\Delta L_{ta} < 4`,
    :math:`K_t = \Delta L_{ta} - 4` for :math:`4 \le \Delta L_{ta} \le 10`
    and
    :math:`K_t = 6` for :math:`\Delta L_{ta} > 10`. ``Kt`` is not restricted
    to integers.

    :param audibility: Tonal audibility ``ΔLta``, in dB.
    :return: Tonal adjustment ``Kt``, in dB (0 to 6).
    :raises ValueError: If ``audibility`` is not finite.
    """
    delta = _finite(audibility, "audibility")
    if delta < _KT_LOW:
        return 0.0
    if delta > _KT_HIGH:
        return _KT_MAX
    return float(delta - _KT_LOW)


@dataclass(frozen=True)
class TonalAssessmentResult:
    """Tonal-audibility assessment of a tone in noise (ISO 1996-2 Annex C).

    :ivar tone_level: Energy-summed tone level ``Lpt``, in dB.
    :ivar masking_noise_level: Masking-noise level ``Lpn``, in dB.
    :ivar centre_frequency: Critical-band centre frequency ``fc``, in Hz.
    :ivar critical_bandwidth: Critical bandwidth, in Hz (Table C.1).
    :ivar audibility: Tonal audibility ``ΔLta``, in dB (Formula (C.3)).
    :ivar adjustment: Tonal adjustment ``Kt``, in dB (Formulae (C.4)–(C.6)).
    """

    tone_level: float
    masking_noise_level: float
    centre_frequency: float
    critical_bandwidth: float
    audibility: float
    adjustment: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the ``Kt(ΔLta)`` adjustment curve with this tone marked."""
        from ..._i18n import check_language
        from ..._plot.environment import plot_tonal_adjustment

        return plot_tonal_adjustment(self, ax=ax, language=check_language(language), **kwargs)


def assess_tonal_audibility(
    tone_level: float, masking_noise_level: float, centre_frequency: float
) -> TonalAssessmentResult:
    """Assess a tone's audibility and adjustment (ISO 1996-2 Annex C).

    Combines :func:`tonal_audibility` and :func:`tonal_adjustment` with the
    :func:`critical_bandwidth` into a plottable result.

    :param tone_level: Energy-summed tone level ``Lpt``, in dB.
    :param masking_noise_level: Masking-noise level ``Lpn``, in dB.
    :param centre_frequency: Critical-band centre frequency ``fc``, in Hz.
    :return: A :class:`TonalAssessmentResult`.
    """
    audibility = tonal_audibility(tone_level, masking_noise_level, centre_frequency)
    return TonalAssessmentResult(
        tone_level=float(tone_level),
        masking_noise_level=float(masking_noise_level),
        centre_frequency=float(centre_frequency),
        critical_bandwidth=critical_bandwidth(centre_frequency),
        audibility=audibility,
        adjustment=tonal_adjustment(audibility),
    )


#: ISO 1996-2:2017 Table J.1 -- (upper ΔL limit inclusive, Kt) rows.
_TABLE_J1 = ((0.0, 0), (2.0, 1), (4.0, 2), (6.0, 3), (9.0, 4), (12.0, 5))


def tonal_adjustment_from_mean_audibility(
    mean_audibility: float, *, coarse: bool = False
) -> int:
    r"""Tonal adjustment ``Kt`` from the mean audibility ``ΔL`` (Table J.1).

    The ISO 1996-2:2017 route that maps the ISO/PAS 20065 mean audibility
    ``ΔL`` to an integer adjustment. With ``coarse=True`` the 3-dB-step
    alternative applies (:math:`\Delta L \le 2` → 0,
    :math:`2 < \Delta L \le 9` → 3, :math:`\Delta L > 9` → 6).

    :param mean_audibility: Mean audibility ``ΔL``, in dB.
    :param coarse: Use the coarse 3-dB-step mapping instead of Table J.1.
    :return: Tonal adjustment ``Kt``, in dB (integer, 0 to 6).
    :raises ValueError: If ``mean_audibility`` is not finite.
    """
    delta = _finite(mean_audibility, "mean_audibility")
    if coarse:
        if delta <= 2.0:
            return 0
        if delta <= 9.0:
            return 3
        return 6
    for upper, adjustment in _TABLE_J1:
        if delta <= upper:
            return adjustment
    return 6


# --------------------------------------------------------------------------- #
# Survey / simplified tonal method (ISO 1996-2:2017 Annex K)
# --------------------------------------------------------------------------- #
def tonal_seeking_survey(
    levels: Sequence[float] | np.ndarray,
    frequencies: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Flag prominent tones by the one-third-octave survey method (Annex K).

    A band is flagged when it exceeds *both* adjacent one-third-octave bands by
    the level difference for its range: 15 dB (25–125 Hz), 8 dB (160–400 Hz),
    5 dB (500–10 000 Hz). The two end bands (no pair of neighbours) are never
    flagged.

    .. note::
        Annex K defines the thresholds for 25 Hz to 10 kHz only. Bands
        supplied outside that span are extrapolated with the nearest rule
        (15 dB below 25 Hz, 5 dB above 10 kHz), outside the standard.

    :param levels: One-third-octave-band time-average levels, in dB.
    :param frequencies: The band centre frequencies, in Hz (same length).
    :return: Boolean array, ``True`` where a prominent tone is present.
    :raises ValueError: If the inputs are empty, non-finite, or differ in
        length.
    """
    lev = np.asarray(levels, dtype=np.float64)
    freq = np.asarray(frequencies, dtype=np.float64)
    if lev.ndim != 1 or freq.ndim != 1:
        raise ValueError("'levels' and 'frequencies' must be one-dimensional.")
    if lev.size != freq.size:
        raise ValueError("'levels' and 'frequencies' must share their length.")
    if lev.size < 3:
        raise ValueError("Need at least three bands to test the neighbours.")
    if not (np.all(np.isfinite(lev)) and np.all(np.isfinite(freq))):
        raise ValueError("'levels' and 'frequencies' must be finite.")
    flags = np.zeros(lev.size, dtype=np.bool_)
    for i in range(1, lev.size - 1):
        threshold = _survey_threshold(freq[i])
        margin = lev[i] - max(lev[i - 1], lev[i + 1])
        flags[i] = margin >= threshold
    return flags


def _survey_threshold(frequency: float) -> float:
    """The survey level-difference threshold for a band centre (Annex K)."""
    if frequency <= 125.0:
        return 15.0
    if frequency <= 400.0:
        return 8.0
    return 5.0


# --------------------------------------------------------------------------- #
# Residual-noise correction (Clause 10.4, Annex I)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResidualCorrectionResult:
    """Residual-noise-corrected level (ISO 1996-2:2017 Clause 10.4).

    :ivar corrected_level: The corrected level ``L`` (Formula (16)), in dB.
        When ``reliable`` is ``False`` the standard allows *no* correction;
        this value is then informative only (it estimates the source from
        below) and must not be reported as the result.
    :ivar reportable_upper_bound: The *measured* level ``L'``, in dB. When the
        margin is 3 dB or less, §10.4 permits reporting the measured level as
        an upper bound of the specific sound level; this field carries that
        reportable value.
    :ivar margin: ``L' − Lres``, in dB (measured minus residual).
    :ivar reliable: ``True`` when the residual is more than 3 dB below the
        measured level; ``False`` when no correction is allowed and only the
        uncorrected ``L'`` may be reported, as an upper bound.
    """

    corrected_level: float
    reportable_upper_bound: float
    margin: float
    reliable: bool


def residual_sound_correction(
    measured_level: float, residual_level: float
) -> ResidualCorrectionResult:
    r"""Correct a measured level for residual sound (Formula (16)).

    :math:`L = 10 \log_{10}(10^{L'/10} - 10^{L_{res}/10})`. When the residual is
    within 3 dB
    of the measured level, §10.4 allows **no** correction: the *uncorrected*
    measured level ``L'`` is the reportable value, as an upper bound of the
    specific sound (the corrected value would understate reliability, being
    the lower-side estimate). The result is then flagged ``reliable = False``
    and an :class:`EnvironmentalMeasurementWarning` is issued; report
    ``reportable_upper_bound`` (= ``L'``), not ``corrected_level``.

    :param measured_level: Measured level ``L'`` including residual, in dB.
    :param residual_level: Residual (background) level ``Lres``, in dB.
    :return: A :class:`ResidualCorrectionResult`.
    :raises ValueError: If the levels are not finite or the residual is not
        below the measured level.
    """
    import warnings

    lp = _finite(measured_level, "measured_level")
    lres = _finite(residual_level, "residual_level")
    margin = lp - lres
    if margin <= 0.0:
        raise ValueError(
            "'residual_level' must be below 'measured_level' to correct."
        )
    corrected = 10.0 * np.log10(10.0 ** (lp / 10.0) - 10.0 ** (lres / 10.0))
    reliable = margin > 3.0
    if not reliable:
        warnings.warn(
            "Residual is within 3 dB of the measured level; no correction is "
            "allowed and only the uncorrected measured level may be reported, "
            "as an upper bound (ISO 1996-2:2017, 10.4). Use "
            "reportable_upper_bound, not corrected_level.",
            EnvironmentalMeasurementWarning,
            stacklevel=2,
        )
    return ResidualCorrectionResult(
        corrected_level=float(corrected),
        reportable_upper_bound=float(lp),
        margin=float(margin),
        reliable=reliable,
    )


def gaussian_residual_level(
    l50: float, *, l90: float | None = None, l95: float | None = None
) -> float:
    r"""Estimate the residual equivalent level from percentiles (Annex I).

    :math:`L_{eq} = L_{50} + 0.115 \cdot ((L_{50} - L_{90})/1.28)^2`
    (Formula (I.1)) or, with ``L95``,
    :math:`L_{eq} = L_{50} + 0.115 \cdot ((L_{50} - L_{95})/1.65)^2`
    (Formula (I.2)). Supply exactly
    one of ``l90`` / ``l95``.

    :param l50: Median level ``L50``, in dB.
    :param l90: Level exceeded 90 % of the time ``L90``, in dB.
    :param l95: Level exceeded 95 % of the time ``L95``, in dB.
    :return: The estimated Gaussian residual equivalent level, in dB.
    :raises ValueError: If not exactly one of ``l90`` / ``l95`` is given, the
        inputs are not finite, or the percentile ordering is inverted
        (``L90``/``L95`` cannot exceed ``L50``, almost certainly swapped
        arguments, which the squared spread would otherwise hide).
    """
    median = _finite(l50, "l50")
    if (l90 is None) == (l95 is None):
        raise ValueError("Supply exactly one of 'l90' or 'l95'.")
    if l90 is not None:
        spread = median - _finite(l90, "l90")
        divisor = _GAUSS_DIVISOR_L90
        name = "l90"
    elif l95 is not None:
        spread = median - _finite(l95, "l95")
        divisor = _GAUSS_DIVISOR_L95
        name = "l95"
    else:  # unreachable: the check above guarantees exactly one is supplied
        raise ValueError("Supply exactly one of 'l90' or 'l95'.")
    if spread < 0.0:
        raise ValueError(
            f"'{name}' exceeds 'l50': the exceedance percentiles satisfy "
            "L50 >= L90 >= L95, so the arguments look swapped."
        )
    return float(median + _GAUSS_CONSTANT * (spread / divisor) ** 2)


# --------------------------------------------------------------------------- #
# Measurement uncertainty (Clause 4, Annex F)
# --------------------------------------------------------------------------- #
def combined_standard_uncertainty(
    contributions: Sequence[float] | Sequence[tuple[float, float]] | np.ndarray,
) -> float:
    r"""Combined standard uncertainty (Formula (2)).

    :math:`u = \sqrt{\sum (c_j \cdot u_j)^2}`.

    :param contributions: Either the per-component products
        :math:`c_j \cdot u_j` (dB), or
        ``(uⱼ, cⱼ)`` pairs whose product is formed. Independent inputs are
        assumed (no covariance term).
    :return: The combined standard uncertainty, in dB.
    :raises ValueError: If ``contributions`` is empty or non-finite.
    """
    items = list(contributions)
    if not items:
        raise ValueError("'contributions' must not be empty.")
    products = []
    for item in items:
        if isinstance(item, (tuple, list, np.ndarray)):
            pair = np.asarray(item, dtype=np.float64)
            if pair.shape != (2,):
                raise ValueError("Each pair must be (uncertainty, sensitivity).")
            products.append(pair[0] * pair[1])
        else:
            products.append(float(item))
    arr = np.asarray(products, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("'contributions' must be finite.")
    return float(np.sqrt(np.sum(arr**2)))


def expanded_uncertainty(
    standard_uncertainty: float, *, confidence: float = 0.95
) -> float:
    r"""Expanded uncertainty :math:`U = k \cdot u` (Clause 4).

    Coverage factor :math:`k = 2` for 95 % or :math:`k = 1.3` for 80 %.

    :param standard_uncertainty: Combined standard uncertainty ``u``, in dB.
    :param confidence: Coverage probability (0.95 or 0.80).
    :return: The expanded uncertainty ``U``, in dB.
    :raises ValueError: If ``u`` is negative/non-finite or ``confidence`` is
        not one of the tabulated values.
    """
    u = _finite(standard_uncertainty, "standard_uncertainty")
    if u < 0.0:
        raise ValueError("'standard_uncertainty' must be non-negative.")
    if confidence not in _COVERAGE_FACTORS:
        raise ValueError(
            f"'confidence' must be one of {sorted(_COVERAGE_FACTORS)}."
        )
    return float(_COVERAGE_FACTORS[confidence] * u)


def residual_correction_uncertainty(
    measured_level: float,
    residual_level: float,
    measured_uncertainty: float,
    residual_uncertainty: float,
) -> float:
    r"""Uncertainty of the residual-corrected level (Formulae (F.7)–(F.9)).

    With :math:`m = 10^{-0.1(L' - L_{res})}`, the sensitivity coefficients
    are
    :math:`c_{L'} = 1/(1 - m)` and :math:`c_{res} = -m/(1 - m)`, and
    :math:`u_L = \sqrt{c_{L'}^2 \cdot u_{L'}^2 + c_{res}^2 \cdot u_{res}^2}`.

    :param measured_level: Measured level ``L'``, in dB.
    :param residual_level: Residual level ``Lres``, in dB.
    :param measured_uncertainty: Standard uncertainty of ``L'``, in dB.
    :param residual_uncertainty: Standard uncertainty of ``Lres``, in dB.
    :return: The combined standard uncertainty of the corrected level, in dB.
    :raises ValueError: If the residual is not below the measured level or an
        uncertainty is negative/non-finite.
    """
    lp = _finite(measured_level, "measured_level")
    lres = _finite(residual_level, "residual_level")
    u_lp = _finite(measured_uncertainty, "measured_uncertainty")
    u_res = _finite(residual_uncertainty, "residual_uncertainty")
    if lp - lres <= 0.0:
        raise ValueError("'residual_level' must be below 'measured_level'.")
    if u_lp < 0.0 or u_res < 0.0:
        raise ValueError("Uncertainties must be non-negative.")
    m = 10.0 ** (-0.1 * (lp - lres))
    c_lp = 1.0 / (1.0 - m)
    c_res = -m / (1.0 - m)
    return float(np.sqrt((c_lp * u_lp) ** 2 + (c_res * u_res) ** 2))


#: Level spread (max - min, dB) beyond which the Formula (20) Note 2
#: approximation is flagged as unreliable.
_LEVEL_SPREAD_WARNING_DB = 3.0


@dataclass(frozen=True)
class RepeatedMeasurementResult:
    r"""Energy-mean level and its uncertainty from repeats (Formulae (17)–(20)).

    :ivar mean_level: Energy-mean level
        :math:`L_k = 10 \log_{10}((1/N) \cdot \sum 10^{0.1 \cdot L_i})`, dB
        (Formula (18)).
    :ivar standard_uncertainty: Standard uncertainty ``uk`` by the primary
        route, Formulae (17)+(19): the sample standard deviation ``sk`` of the
        energy values :math:`10^{0.1 \cdot L_i}` mapped back to level,
        :math:`u_k = 10 \log_{10}(10^{0.1 \cdot L_k} + s_k) - L_k`, in dB.
    :ivar approximate_uncertainty: The Note 2 substitute (Formula (20)),
        :math:`\sqrt{\sum (L_i - L_k)^2/(N - 1)}`, in dB; valid only when the
        spread of the
        ``Li`` is small; it grossly inflates for spread levels.
    :ivar n: Number of measurements.
    """

    mean_level: float
    standard_uncertainty: float
    approximate_uncertainty: float
    n: int


def uncertainty_from_repeated_measurements(
    levels: Sequence[float] | np.ndarray,
) -> RepeatedMeasurementResult:
    r"""Energy mean and its uncertainty from repeated levels (Formulae (17)–(20)).

    :math:`L_k = 10 \log_{10}((1/N) \cdot \sum 10^{0.1 \cdot L_i})` (Formula (18)).
    The standard
    uncertainty follows the primary §10.5 route: the sample standard
    deviation ``sk`` of the energy values :math:`10^{0.1 \cdot L_i}`
    (Formula (17))
    propagated back to level,
    :math:`u_k = 10 \log_{10}(10^{0.1 \cdot L_k} + s_k) - L_k`
    (Formula (19)). The Note 2 level-domain approximation
    :math:`\sqrt{\sum (L_i - L_k)^2/(N - 1)}` (Formula (20)) is also
    reported as
    ``approximate_uncertainty``; it is valid only "if the difference between
    different Li is small", so a spread above 3 dB triggers an
    :class:`EnvironmentalMeasurementWarning` (e.g. [50, 60, 70] dB gives
    3.94 dB by Formulae (17)+(19) but 12.18 dB by Formula (20)).

    :param levels: The repeated measured levels ``Li``, in dB (at least two).
    :return: A :class:`RepeatedMeasurementResult`.
    :raises ValueError: If fewer than two finite levels are given.
    """
    import warnings

    lev = np.asarray(levels, dtype=np.float64)
    if lev.ndim != 1 or lev.size < 2:
        raise ValueError("'levels' must be a 1-D array of at least two values.")
    if not np.all(np.isfinite(lev)):
        raise ValueError("'levels' must be finite.")
    n = int(lev.size)
    energies = 10.0 ** (0.1 * lev)
    mean_energy = float(np.mean(energies))
    mean_level = 10.0 * np.log10(mean_energy)
    sk = float(np.sqrt(np.sum((energies - mean_energy) ** 2) / (n - 1)))
    uk = 10.0 * np.log10(mean_energy + sk) - mean_level
    uk_approx = float(np.sqrt(np.sum((lev - mean_level) ** 2) / (n - 1)))
    if float(np.max(lev) - np.min(lev)) > _LEVEL_SPREAD_WARNING_DB:
        warnings.warn(
            "The repeated levels spread by more than 3 dB: the Formula (20) "
            "approximation (approximate_uncertainty) is unreliable there; "
            "use the primary Formulae (17)+(19) standard_uncertainty "
            "(ISO 1996-2:2017, 10.5 Note 2).",
            EnvironmentalMeasurementWarning,
            stacklevel=2,
        )
    return RepeatedMeasurementResult(
        mean_level=float(mean_level),
        standard_uncertainty=float(uk),
        approximate_uncertainty=uk_approx,
        n=n,
    )

#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Age-related hearing threshold (ISO 7029:2017) and audiometric reference zero
(ISO 389-7:2005).

Implements the statistical distribution of the hearing threshold of an
otologically normal population as a function of age and sex (ISO 7029:2017),
and the reference threshold of hearing under free-field and diffuse-field
listening (ISO 389-7:2005, Table 1), over the audiometric frequencies from
125 Hz to 8000 Hz.

ISO 7029 gives the median threshold deviation from the value at age 18 as
:math:`dH_\mathrm{md} = a \, (\mathrm{age} - 18)^b` (clause 4.2, Table 1) and the
spread around the
median as two half-Gaussian standard deviations ``su`` (worse than median) and
``sl`` (better than median), each a fifth-degree polynomial in ``age - 18``
(clause 4.3, Tables 2-5). A population fractile is obtained by shifting the
median by the standard-normal quantile times the appropriate spread
(clause 4.4 / Annex A).

The noise-induced permanent threshold shift of ISO 1999 (which combines a noise
component with this age component) is not part of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from numpy.typing import ArrayLike

# ---------------------------------------------------------------------------
# Normative constants.
# ---------------------------------------------------------------------------

#: Audiometric frequencies, in hertz (ISO 7029 Table 1 / ISO 389-7 Table 1).
AUDIOMETRIC_FREQUENCIES: np.ndarray = np.array(
    [125.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0,
     6000.0, 8000.0],
    dtype=np.float64,
)

#: Median coefficients ``(a, b)`` of ``dHmd = a*(age-18)**b`` (ISO 7029 Table 1).
_MEDIAN_MALE: np.ndarray = np.array(
    [[2.50e-6, 3.841], [1.39e-4, 2.832], [4.59e-4, 2.537], [5.70e-4, 2.512],
     [7.02e-4, 2.494], [1.09e-3, 2.446], [1.56e-3, 2.404], [2.54e-3, 2.350],
     [3.40e-3, 2.325], [4.53e-3, 2.315], [5.06e-3, 2.328]],
    dtype=np.float64,
)
_MEDIAN_FEMALE: np.ndarray = np.array(
    [[6.16e-4, 2.451], [3.98e-4, 2.568], [2.61e-4, 2.708], [2.25e-4, 2.775],
     [2.21e-4, 2.805], [2.53e-4, 2.813], [3.12e-4, 2.792], [4.88e-4, 2.728],
     [7.37e-4, 2.660], [1.47e-3, 2.539], [2.53e-3, 2.439]],
    dtype=np.float64,
)

#: Upper-spread ``su`` polynomial coefficients c0..c5 (ISO 7029 Table 2, male).
_SU_MALE: np.ndarray = np.array(
    [[4.63, 0.645, -8.85e-2, 3.69e-3, -5.98e-5, 3.39e-7],
     [5.27, 0.710, -9.13e-2, 3.64e-3, -5.74e-5, 3.22e-7],
     [4.98, 0.751, -9.20e-2, 3.68e-3, -5.84e-5, 3.28e-7],
     [4.65, 0.733, -8.81e-2, 3.59e-3, -5.76e-5, 3.24e-7],
     [4.42, 0.714, -8.54e-2, 3.57e-3, -5.82e-5, 3.29e-7],
     [4.14, 0.679, -8.04e-2, 3.52e-3, -5.89e-5, 3.35e-7],
     [4.10, 0.632, -7.53e-2, 3.46e-3, -5.94e-5, 3.40e-7],
     [4.29, 0.530, -6.28e-2, 3.09e-3, -5.37e-5, 2.95e-7],
     [4.68, 0.455, -5.52e-2, 2.95e-3, -5.30e-5, 2.92e-7],
     [5.61, 0.363, -4.72e-2, 2.92e-3, -5.58e-5, 3.12e-7],
     [6.62, 0.291, -4.16e-2, 2.92e-3, -5.85e-5, 3.33e-7]],
    dtype=np.float64,
)
#: Lower-spread ``sl`` polynomial coefficients c0..c5 (ISO 7029 Table 3, male).
_SL_MALE: np.ndarray = np.array(
    [[3.34, 0.131, -2.02e-2, 1.12e-3, -2.28e-5, 1.57e-7],
     [3.32, 0.230, -2.54e-2, 1.20e-3, -2.27e-5, 1.46e-7],
     [3.43, 0.362, -4.11e-2, 1.87e-3, -3.44e-5, 2.21e-7],
     [3.60, 0.384, -4.43e-2, 1.98e-3, -3.55e-5, 2.22e-7],
     [3.77, 0.363, -4.19e-2, 1.82e-3, -3.14e-5, 1.89e-7],
     [3.93, 0.365, -4.22e-2, 1.79e-3, -2.96e-5, 1.70e-7],
     [4.01, 0.387, -4.47e-2, 1.87e-3, -3.02e-5, 1.69e-7],
     [4.11, 0.405, -4.56e-2, 1.86e-3, -2.83e-5, 1.46e-7],
     [4.09, 0.439, -4.78e-2, 1.92e-3, -2.84e-5, 1.40e-7],
     [4.01, 0.497, -4.97e-2, 1.93e-3, -2.67e-5, 1.18e-7],
     [3.90, 0.559, -5.62e-2, 2.40e-3, -3.92e-5, 2.29e-7]],
    dtype=np.float64,
)
#: Upper-spread ``su`` polynomial coefficients c0..c5 (ISO 7029 Table 4, female).
_SU_FEMALE: np.ndarray = np.array(
    [[5.05, 0.400, -4.60e-2, 1.73e-3, -2.75e-5, 1.71e-7],
     [5.01, 0.481, -4.88e-2, 1.80e-3, -2.81e-5, 1.67e-7],
     [4.68, 0.510, -5.16e-2, 1.95e-3, -3.07e-5, 1.77e-7],
     [4.45, 0.511, -5.25e-2, 2.03e-3, -3.18e-5, 1.81e-7],
     [4.34, 0.492, -5.15e-2, 2.03e-3, -3.18e-5, 1.78e-7],
     [4.23, 0.479, -5.12e-2, 2.07e-3, -3.26e-5, 1.80e-7],
     [4.26, 0.456, -4.91e-2, 2.01e-3, -3.16e-5, 1.70e-7],
     [4.36, 0.476, -5.15e-2, 2.19e-3, -3.51e-5, 1.91e-7],
     [4.61, 0.477, -5.07e-2, 2.19e-3, -3.51e-5, 1.88e-7],
     [5.22, 0.483, -4.83e-2, 2.13e-3, -3.39e-5, 1.74e-7],
     [5.84, 0.516, -4.89e-2, 2.18e-3, -3.49e-5, 1.77e-7]],
    dtype=np.float64,
)
#: Lower-spread ``sl`` polynomial coefficients c0..c5 (ISO 7029 Table 5, female).
_SL_FEMALE: np.ndarray = np.array(
    [[3.64, 0.047, 2.28e-3, -6.68e-5, -8.72e-7, 2.30e-8],
     [3.11, 0.226, -7.71e-3, 9.83e-5, -7.11e-7, 9.02e-9],
     [2.98, 0.338, -1.74e-2, 3.53e-4, -2.78e-6, 1.01e-8],
     [3.03, 0.378, -2.24e-2, 5.20e-4, -4.60e-6, 1.50e-8],
     [3.15, 0.382, -2.39e-2, 5.62e-4, -4.60e-6, 9.87e-9],
     [3.32, 0.387, -2.64e-2, 6.71e-4, -5.76e-6, 1.07e-8],
     [3.47, 0.392, -2.84e-2, 7.79e-4, -7.35e-6, 1.71e-8],
     [3.69, 0.392, -2.96e-2, 8.44e-4, -7.55e-6, 8.16e-9],
     [3.84, 0.402, -3.17e-2, 9.99e-4, -1.08e-5, 2.99e-8],
     [4.04, 0.403, -3.15e-2, 1.06e-3, -1.21e-5, 3.44e-8],
     [4.15, 0.413, -3.01e-2, 1.00e-3, -9.97e-6, 8.74e-9]],
    dtype=np.float64,
)

_MEDIAN = {"male": _MEDIAN_MALE, "female": _MEDIAN_FEMALE}
_SU = {"male": _SU_MALE, "female": _SU_FEMALE}
_SL = {"male": _SL_MALE, "female": _SL_FEMALE}
SEXES: tuple[str, ...] = ("male", "female")

#: Reference threshold of hearing, in dB, free-field then diffuse-field, at
#: :data:`AUDIOMETRIC_FREQUENCIES` (ISO 389-7:2005 Table 1).
_REFERENCE_FREE: np.ndarray = np.array(
    [22.1, 11.4, 4.4, 2.4, 2.4, 2.4, -1.3, -5.8, -5.4, 4.3, 12.6],
    dtype=np.float64,
)
_REFERENCE_DIFFUSE: np.ndarray = np.array(
    [22.1, 11.4, 3.8, 1.2, 0.8, 1.0, -1.5, -4.0, -3.8, 1.4, 6.8],
    dtype=np.float64,
)
_REFERENCE = {"free-field": _REFERENCE_FREE, "diffuse-field": _REFERENCE_DIFFUSE}
FIELDS: tuple[str, ...] = ("free-field", "diffuse-field")

_REFERENCE_AGE = 18.0  # lower age limit of the ISO 7029 formulae.


@dataclass(frozen=True)
class AgeThresholdResult:
    """Age-related hearing threshold distribution (ISO 7029:2017).

    All arrays are in dB and aligned with :data:`AUDIOMETRIC_FREQUENCIES`.

    :ivar age: Listener age, in years.
    :ivar sex: ``"male"`` or ``"female"``.
    :ivar fractile: Population fractile of ``threshold`` (0-1).
    :ivar frequencies: Audiometric frequencies, in hertz.
    :ivar median: Median threshold deviation from age 18 (clause 4.2).
    :ivar spread_upper: Upper half-Gaussian standard deviation ``su``.
    :ivar spread_lower: Lower half-Gaussian standard deviation ``sl``.
    :ivar threshold: Threshold deviation at ``fractile`` (clause 4.4).
    """

    age: float
    sex: str
    fractile: float
    frequencies: np.ndarray
    median: np.ndarray
    spread_upper: np.ndarray
    spread_lower: np.ndarray
    threshold: np.ndarray

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the median threshold with the fractile band over frequency.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.hearing import plot_age_threshold

        return plot_age_threshold(self, ax=ax, language=check_language(language), **kwargs)


def _select(values: np.ndarray, frequencies: ArrayLike | None) -> np.ndarray:
    """Return ``values`` for a requested frequency subset, or all of them."""
    if frequencies is None:
        return values.copy()
    fr = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    idx = []
    for f in fr:
        matches = np.isclose(AUDIOMETRIC_FREQUENCIES, f, rtol=1e-3)
        if not matches.any():
            raise ValueError(
                f"frequency {f} Hz is not an ISO 7029 audiometric frequency "
                f"(125 Hz - 8000 Hz)."
            )
        idx.append(int(np.argmax(matches)))
    return values[idx]


def _spread(coeffs: np.ndarray, u: float) -> np.ndarray:
    """Evaluate the degree-5 spread polynomials (ascending powers) at ``u``."""
    powers = u ** np.arange(coeffs.shape[1], dtype=np.float64)
    return coeffs @ powers


def age_threshold(
    age: float,
    sex: Literal["male", "female"] = "male",
    fractile: float = 0.5,
    frequencies: ArrayLike | None = None,
) -> AgeThresholdResult:
    """Age-related hearing threshold distribution (ISO 7029:2017).

    Returns, per audiometric frequency, the median threshold deviation from the
    value at age 18 (clause 4.2), the upper/lower half-Gaussian spreads
    (clause 4.3) and the threshold at the requested population ``fractile``
    (clause 4.4): ``median + z * spread`` where ``z`` is the standard-normal
    quantile of ``fractile`` and the spread is the upper one for ``z >= 0``
    (worse than the median) or the lower one otherwise.

    :param age: Listener age, in years (must be at least 18). The standard's
        formulae are established up to 80 years for frequencies at or below
        2000 Hz and up to 70 years above; ages beyond that extrapolate.
    :param sex: ``"male"`` or ``"female"``.
    :param fractile: Population fractile in the open interval (0, 1); ``0.5``
        gives the median.
    :param frequencies: Optional subset of the audiometric frequencies, in
        hertz; ``None`` uses all eleven (125 Hz - 8000 Hz).
    :return: An :class:`AgeThresholdResult` with the distribution and ``.plot()``.
    :raises ValueError: for an age below 18, an unknown sex, a fractile outside
        (0, 1), or an unknown frequency.
    """
    if age < _REFERENCE_AGE:
        raise ValueError(
            f"age must be at least {_REFERENCE_AGE:.0f} years (the ISO 7029 "
            f"lower limit); got {age}."
        )
    if sex not in _MEDIAN:
        raise ValueError(f"sex must be one of {SEXES}; got {sex!r}.")
    if not 0.0 < fractile < 1.0:
        raise ValueError(f"fractile must be in (0, 1); got {fractile}.")

    from scipy.special import ndtri  # standard-normal quantile

    u = float(age) - _REFERENCE_AGE
    med_coeffs = _MEDIAN[sex]
    median = med_coeffs[:, 0] * u ** med_coeffs[:, 1]
    su = _spread(_SU[sex], u)
    sl = _spread(_SL[sex], u)

    z = float(ndtri(fractile))
    threshold = median + z * (su if z >= 0.0 else sl)

    return AgeThresholdResult(
        age=float(age),
        sex=sex,
        fractile=float(fractile),
        frequencies=_select(AUDIOMETRIC_FREQUENCIES, frequencies),
        median=_select(median, frequencies),
        spread_upper=_select(su, frequencies),
        spread_lower=_select(sl, frequencies),
        threshold=_select(threshold, frequencies),
    )


def reference_threshold(
    field: str = "free-field", frequencies: ArrayLike | None = None
) -> np.ndarray:
    """Reference threshold of hearing (ISO 389-7:2005, Table 1).

    The sound pressure level, in dB, that corresponds to the audiometric zero
    (0 dB HL) under the given listening condition, at the audiometric
    frequencies.

    :param field: ``"free-field"`` (frontal incidence) or ``"diffuse-field"``.
    :param frequencies: Optional subset of the audiometric frequencies, in
        hertz; ``None`` uses all eleven (125 Hz - 8000 Hz).
    :return: The reference threshold, in dB, aligned with the frequencies.
    :raises ValueError: for an unknown field or frequency.
    """
    if field not in _REFERENCE:
        raise ValueError(f"field must be one of {FIELDS}; got {field!r}.")
    return _select(_REFERENCE[field], frequencies)

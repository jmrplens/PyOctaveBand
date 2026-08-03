#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Marine-mammal hearing thresholds (group audiograms and the orca audiogram).

Two independent published descriptions of how well a marine mammal hears:

* :func:`group_audiogram` -- the **group audiogram** of Southall et al. (2019),
  a four-parameter band-pass fit (their Equation 1, after Finneran 2016)

  .. math::
      T(f) = T_0 + A \log_{10}\!\left(1 + \frac{F_1}{f}\right) + (f/F_2)^{B}

  with the group parameters of their Table 2 (absolute thresholds) and Table 3
  (normalised to 0 dB at best sensitivity).
* :func:`orca_audiogram` -- the killer-whale (*Orcinus orca*) audiogram of
  Wensveen & Van Roij (2007) as printed in Ainslie, *Principles of Sonar
  Performance Modelling* (Springer 2010), Equation (11.159), a **three-branch**
  power law over 0.5 to 80 kHz fitted to the measurements of Hall & Johnson
  (1972) and Szymanski et al. (1999).

Thresholds are sound pressure levels in dB re 1 µPa under water and dB re
20 µPa in air (the two in-air carnivore groups ``"PCA"`` and ``"OCA"``).

.. note::
    Southall et al. publish **no fitted audiogram for low-frequency (LF)
    cetaceans**: no audiometric data exist for them. The article gives
    :math:`A = 20` dB/decade, :math:`B = 3.2`, :math:`F_2 = 9.4` kHz and
    :math:`T_0 = 53.2` dB (0.8 dB normalised) in prose but never prints
    :math:`F_1`, only the criterion used to choose it. The group is
    therefore absent from
    :data:`AUDIOGRAM_GROUPS` rather than reconstructed by guesswork.

Group codes follow Southall et al.: ``HF`` and ``VHF`` cetaceans, ``SI``
sirenians, ``PCW``/``OCW`` phocid and otariid carnivores in water and
``PCA``/``OCA`` the same in air. Beware that NMFS (2018) calls the Southall
``HF`` group ``MF`` and the Southall ``VHF`` group ``HF``; see
:mod:`phonometry.underwater.bioacoustics.weighting`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray


@dataclass(frozen=True)
class AudiogramParameters:
    r"""Group-audiogram fit parameters (Southall et al. 2019, Tables 2 and 3).

    :ivar group: Hearing-group code.
    :ivar t0: Vertical position ``T0``, in dB.
    :ivar f1_khz: Low-frequency inflection ``F1``, in kHz.
    :ivar f2_khz: High-frequency inflection ``F2``, in kHz.
    :ivar a: Low-frequency slope parameter ``A``, in dB/decade.
    :ivar b: High-frequency exponent ``B``.
    :ivar r_squared: Goodness of fit :math:`R^2` reported with the row.
    :ivar in_air: Whether the group's reference pressure is 20 µPa (in air).
    """

    group: str
    t0: float
    f1_khz: float
    f2_khz: float
    a: float
    b: float
    r_squared: float
    in_air: bool


#: Southall et al. (2019) Table 2, printed p. 144: estimated group audiograms
#: fitted to the original (absolute) median behavioural thresholds.
_AUDIOGRAM_ORIGINAL: dict[str, AudiogramParameters] = {
    "HF": AudiogramParameters("HF", 46.2, 25.9, 47.8, 35.5, 3.56, 0.977, False),
    "VHF": AudiogramParameters("VHF", 46.4, 7.57, 126.0, 42.3, 17.1, 0.968, False),
    "SI": AudiogramParameters("SI", -40.4, 3990.0, 3.8, 37.3, 1.7, 0.982, False),
    "PCW": AudiogramParameters("PCW", 43.7, 10.2, 3.97, 20.1, 1.41, 0.907, False),
    "OCW": AudiogramParameters("OCW", 63.1, 3.06, 11.8, 30.1, 3.23, 0.939, False),
    "PCA": AudiogramParameters("PCA", -110.0, 5.56, 1.02e-6, 69.1, 0.289, 0.973, True),
    "OCA": AudiogramParameters("OCA", 6.24, 1.54, 8.24, 55.6, 2.76, 0.978, True),
}

#: Southall et al. (2019) Table 3, printed p. 144: the same fits on thresholds
#: normalised so that the frequency of best sensitivity sits at 0 dB.
_AUDIOGRAM_NORMALIZED: dict[str, AudiogramParameters] = {
    "HF": AudiogramParameters("HF", 3.61, 12.7, 64.4, 31.8, 4.5, 0.960, False),
    "VHF": AudiogramParameters("VHF", 2.48, 9.68, 126.0, 40.1, 17.0, 0.969, False),
    "SI": AudiogramParameters("SI", -109.0, 5590.0, 2.62, 38.1, 1.53, 0.963, False),
    "PCW": AudiogramParameters("PCW", -39.6, 368.0, 2.21, 20.5, 1.23, 0.907, False),
    "OCW": AudiogramParameters("OCW", 2.36, 0.366, 12.8, 73.5, 3.4, 0.958, False),
    "PCA": AudiogramParameters("PCA", -71.3, 4.8, 6.33e-5, 63.0, 0.364, 0.975, True),
    "OCA": AudiogramParameters("OCA", -1.55, 1.6, 8.66, 54.9, 2.91, 0.968, True),
}

#: Hearing groups with a published group audiogram (Southall et al. 2019).
AUDIOGRAM_GROUPS: tuple[str, ...] = tuple(_AUDIOGRAM_ORIGINAL)

#: Southall et al. (2019) Table 4, printed p. 148: frequency of best hearing
#: ``f0`` (kHz) from the original and from the normalised fits.
BEST_HEARING_FREQUENCY_KHZ: dict[str, tuple[float, float]] = {
    "HF": (55.0, 58.0),
    "VHF": (105.0, 105.0),
    "SI": (16.0, 12.0),
    "PCW": (8.6, 13.0),
    "OCW": (12.0, 10.0),
    "PCA": (2.3, 2.3),
    "OCA": (10.0, 10.0),
}

#: Validity range of Ainslie's orca audiogram, Equation (11.159), in kHz.
ORCA_AUDIOGRAM_RANGE_KHZ = (0.5, 80.0)
#: Branch boundaries of Equation (11.159), in kHz.
_ORCA_BREAKS = (11.3, 46.2)


def audiogram_parameters(group: str, *, normalized: bool = False) -> AudiogramParameters:
    """Fit parameters of a published group audiogram.

    :param group: Hearing-group code, one of :data:`AUDIOGRAM_GROUPS`
        (case-insensitive).
    :param normalized: Return the Table 3 normalised fit instead of the Table 2
        absolute one.
    :return: The :class:`AudiogramParameters` for that group.
    :raises ValueError: If the group has no published audiogram.
    """
    key = str(group).strip().upper()
    table = _AUDIOGRAM_NORMALIZED if normalized else _AUDIOGRAM_ORIGINAL
    if key not in table:
        extra = (
            " Southall et al. publish no fitted audiogram for LF cetaceans"
            " (the F1 parameter is never printed)." if key == "LF" else ""
        )
        raise ValueError(
            f"'group' must be one of {AUDIOGRAM_GROUPS}, got {group!r}.{extra}"
        )
    return table[key]


@dataclass(frozen=True)
class AudiogramResult:
    """Hearing threshold versus frequency.

    :ivar frequencies: Frequencies, in Hz.
    :ivar threshold: Hearing threshold at each frequency, in dB re 1 µPa
        (under water) or dB re 20 µPa (in air).
    :ivar group: Hearing-group code, or ``"orca"`` for the species audiogram.
    :ivar source: Short citation of the fit used.
    :ivar in_air: Whether the reference pressure is 20 µPa.
    :ivar best_frequency: Frequency of the minimum threshold on the evaluated
        grid, in Hz.
    :ivar best_threshold: The minimum threshold on the evaluated grid, in dB.
    """

    frequencies: NDArray[np.float64]
    threshold: NDArray[np.float64]
    group: str
    source: str
    in_air: bool
    best_frequency: float
    best_threshold: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the hearing threshold versus frequency."""
        from ..._i18n import check_language
        from ..._plot.underwater import plot_marine_mammal_audiogram

        return plot_marine_mammal_audiogram(self, ax=ax, language=check_language(language), **kwargs)


def _frequency_khz(frequency_hz: NDArray[np.float64] | list[float] | float) -> NDArray[np.float64]:
    f = np.atleast_1d(np.asarray(frequency_hz, dtype=np.float64))
    if f.size == 0 or not np.all(np.isfinite(f)):
        raise ValueError("'frequency_hz' must be finite and non-empty.")
    if np.any(f <= 0.0):
        raise ValueError("'frequency_hz' must be strictly positive.")
    return f / 1000.0


def _bundle(
    freq_hz: NDArray[np.float64], threshold: NDArray[np.float64], group: str,
    source: str, in_air: bool,
) -> AudiogramResult:
    best = int(np.argmin(threshold))
    return AudiogramResult(
        frequencies=freq_hz,
        threshold=threshold,
        group=group,
        source=source,
        in_air=in_air,
        best_frequency=float(freq_hz[best]),
        best_threshold=float(threshold[best]),
    )


def group_audiogram(
    frequency_hz: NDArray[np.float64] | list[float] | float,
    group: str,
    *,
    normalized: bool = False,
) -> AudiogramResult:
    r"""Marine-mammal group audiogram (Southall et al. 2019, Equation 1).

    :math:`T(f) = T_0 + A \log_{10}(1 + F_1/f) + (f/F_2)^B`, with ``f`` in
    kilohertz and the group parameters of Table 2 (``normalized=False``) or
    Table 3 (``normalized=True``).

    :param frequency_hz: Frequency or frequencies, in Hz (strictly positive).
    :param group: Hearing-group code, one of :data:`AUDIOGRAM_GROUPS`.
    :param normalized: Use the normalised fit (0 dB at best sensitivity).
    :return: An :class:`AudiogramResult`.
    :raises ValueError: If the group is unknown or a frequency is invalid.
    """
    params = audiogram_parameters(group, normalized=normalized)
    f_khz = _frequency_khz(frequency_hz)
    threshold = (
        params.t0
        + params.a * np.log10(1.0 + params.f1_khz / f_khz)
        + (f_khz / params.f2_khz) ** params.b
    )
    table = "Table 3 (normalized)" if normalized else "Table 2 (original)"
    return _bundle(
        f_khz * 1000.0, np.asarray(threshold, dtype=np.float64), params.group,
        f"Southall et al. (2019) Eq. (1), {table}", params.in_air,
    )


def orca_audiogram(
    frequency_hz: NDArray[np.float64] | list[float] | float,
) -> AudiogramResult:
    r"""Killer-whale hearing threshold (Ainslie 2010, Equation 11.159).

    A three-branch fit in :math:`F = f/(1~\text{kHz})`, valid over 0.5 to
    80 kHz:

    * :math:`445.2 F^{-0.05401} - 344.3` for :math:`0.5 \le F < 11.3`,
    * :math:`242.9 F^{-0.7578} + 0.5643 F^{1.076}` for
      :math:`11.3 \le F < 46.2`,
    * :math:`2.792 F^{0.7537} - 2.064` for :math:`46.2 \le F \le 80`.

    The published check points are the minimum, 39.0 dB re 1 µPa at 22.6 kHz
    (second branch), and 51.2 dB re 1 µPa at 50 kHz -- the latter **needs the
    third branch**; evaluating the second one there returns 50.5 dB instead.

    :param frequency_hz: Frequency or frequencies, in Hz, within
        :data:`ORCA_AUDIOGRAM_RANGE_KHZ` scaled to hertz.
    :return: An :class:`AudiogramResult` in dB re 1 µPa.
    :raises ValueError: If a frequency falls outside the fitted range.
    """
    f_khz = _frequency_khz(frequency_hz)
    lo, hi = ORCA_AUDIOGRAM_RANGE_KHZ
    # A relative slack of 1e-9 keeps a grid built on the exact end points (e.g.
    # np.logspace) from tripping on floating-point representation error.
    if np.any(f_khz < lo * (1.0 - 1e-9)) or np.any(f_khz > hi * (1.0 + 1e-9)):
        raise ValueError(
            f"'frequency_hz' must lie within {lo * 1e3:g}-{hi * 1e3:g} Hz,"
            " the range over which Equation (11.159) is fitted."
        )
    b1, b2 = _ORCA_BREAKS
    threshold = np.where(
        f_khz < b1,
        445.2 * f_khz ** (-0.05401) - 344.3,
        np.where(
            f_khz < b2,
            242.9 * f_khz ** (-0.7578) + 0.5643 * f_khz**1.076,
            2.792 * f_khz**0.7537 - 2.064,
        ),
    )
    return _bundle(
        f_khz * 1000.0, np.asarray(threshold, dtype=np.float64), "orca",
        "Wensveen & Van Roij (2007) via Ainslie (2010) Eq. (11.159)", False,
    )

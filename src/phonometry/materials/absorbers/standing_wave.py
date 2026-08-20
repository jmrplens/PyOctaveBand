#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Standing-wave-ratio method for normal-incidence absorption and impedance.

**BS EN ISO 10534-1:2001**, the probe-traverse method of the impedance tube: a
loudspeaker drives one pure tone at a time, a probe microphone traverses the
tube and reads the maximum and minimum sound pressure levels of the standing
wave the specimen sets up, and the position of the first pressure minimum
fixes the phase. The reflection magnitude, phase, absorption coefficient and
normalised impedance follow from those two readings alone (Clause 5,
Eqs. (12)-(26)).

That is the whole method, and it is why it is one module: it consumes a
standing-wave ratio and a distance, and needs neither a measured transfer
function nor the complex wavenumber and air properties every broadband
reduction in the tube goes through. The quantities are the same
normal-incidence quantities the two-microphone method reports -
:math:`\alpha = 1 - \lvert r\rvert^2` is Eq. (9) here and Eq. (18) there -
measured one frequency at a time; see
:mod:`~phonometry.materials.absorbers.impedance_tube`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from ..._internal.types import Real

Complex = NDArray[np.complex128]

__all__ = [
    "standing_wave_absorption",
    "standing_wave_normalized_impedance",
    "standing_wave_ratio_from_level",
    "standing_wave_reflection",
    "standing_wave_reflection_magnitude",
]


def standing_wave_ratio_from_level(level_difference: ArrayLike) -> Real:
    r"""Standing-wave ratio from a level difference (ISO 10534-1, Eq. (15)).

    :math:`s = 10^{\Delta L / 20}` with
    :math:`\Delta L = L_{\max} - L_{\min}` in decibels.

    :param level_difference: Level difference
        :math:`\Delta L = L_{\max} - L_{\min}`, in dB.
    :return: Standing-wave ratio ``s`` (>= 1).
    """
    dl = np.asarray(level_difference, dtype=np.float64)
    if np.any(dl < 0.0):
        raise ValueError("'level_difference' must be non-negative.")
    return np.asarray(10.0 ** (dl / 20.0), dtype=np.float64)


def _check_swr(swr: NDArray[np.float64]) -> None:
    if np.any(swr < 1.0):
        raise ValueError("Standing-wave ratio 's' must be >= 1.")


def standing_wave_reflection_magnitude(swr: ArrayLike) -> Real:
    r"""Reflection magnitude from the standing-wave ratio (ISO 10534-1, Eq. (14)).

    :math:`|r| = (s - 1) / (s + 1)`.

    :param swr: Standing-wave ratio ``s`` (>= 1).
    :return: Reflection magnitude ``|r|`` in ``[0, 1]``.
    """
    s = np.asarray(swr, dtype=np.float64)
    _check_swr(s)
    return np.asarray((s - 1.0) / (s + 1.0), dtype=np.float64)


def standing_wave_absorption(swr: ArrayLike) -> Real:
    r"""Absorption coefficient from the standing-wave ratio (ISO 10534-1).

    Combining :math:`\alpha = 1 - |r|^2` (Eq. (9)) with
    :math:`|r| = (s - 1)/(s + 1)` (Eq. (14)) gives
    :math:`\alpha = 4s/(s + 1)^2`.

    :param swr: Standing-wave ratio ``s`` (>= 1).
    :return: Absorption coefficient ``alpha`` in ``[0, 1]``.
    """
    s = np.asarray(swr, dtype=np.float64)
    _check_swr(s)
    return np.asarray(4.0 * s / (s + 1.0) ** 2, dtype=np.float64)


def _standing_wave_phase(
    first_min_distance: NDArray[np.float64], wavelength: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Reflection phase from the first minimum (ISO 10534-1, Eq. (20))."""
    if np.any(wavelength <= 0.0):
        raise ValueError("'wavelength' must be positive.")
    if np.any(first_min_distance < 0.0):
        raise ValueError("'first_min_distance' must be non-negative.")
    return np.asarray(
        np.pi * (4.0 * first_min_distance / wavelength - 1.0), dtype=np.float64
    )


def standing_wave_reflection(
    swr: ArrayLike, first_min_distance: ArrayLike, wavelength: ArrayLike
) -> Complex:
    r"""Complex reflection factor from the standing wave (ISO 10534-1, Eqs. (17)-(23)).

    :math:`r = |r| e^{j\phi}` with :math:`|r| = (s - 1)/(s + 1)` (Eq. (14))
    and the phase at the first pressure minimum
    :math:`\phi = \pi (4 x_{\text{min},1} / \lambda_0 - 1)` (Eq. (20)).

    :param swr: Standing-wave ratio ``s`` (>= 1).
    :param first_min_distance: Distance ``x_min1`` from the reference plane to
        the first pressure minimum (toward the source), in metres.
    :param wavelength: Wavelength ``lambda0``, in metres (Eq. (27)).
    :return: Complex reflection factor ``r``.
    """
    magnitude = standing_wave_reflection_magnitude(swr)
    phase = _standing_wave_phase(
        np.asarray(first_min_distance, dtype=np.float64),
        np.asarray(wavelength, dtype=np.float64),
    )
    return np.asarray(magnitude * np.exp(1j * phase), dtype=np.complex128)


def standing_wave_normalized_impedance(
    swr: ArrayLike, first_min_distance: ArrayLike, wavelength: ArrayLike
) -> Complex:
    r"""Normalised impedance from the standing wave (ISO 10534-1, Eqs. (24)-(26)).

    :math:`z = Z/Z_0 = (1 + r)/(1 - r)`; the real/imaginary split is
    Eqs. (25)/(26).

    :param swr: Standing-wave ratio ``s`` (>= 1).
    :param first_min_distance: Distance ``x_min1`` to the first minimum, in metres.
    :param wavelength: Wavelength ``lambda0``, in metres.
    :return: Normalised surface impedance ``z`` (complex).
    """
    r = standing_wave_reflection(swr, first_min_distance, wavelength)
    return np.asarray((1.0 + r) / (1.0 - r), dtype=np.complex128)

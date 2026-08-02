#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Sound transmission through slits, holes and apertures (Hopkins 2007, Sound
Insulation, Section 4.3.10; Gomperts 1964; Wilson & Soroka 1965).

Air paths are the real limit on the sound insulation of an otherwise heavy
construction: a small slit or hole caps the achievable sound reduction index no
matter how massive the wall. This module predicts the transmission coefficient
``tau`` of the two canonical apertures and combines them with the surrounding
wall into a composite sound reduction index, the practical answer to "why do I
never reach the catalogue ``Rw``".

**Straight-edged slit (Hopkins Eq. 4.99, Gomperts).** With the acoustic
wavenumber :math:`k = 2\pi f / c_0`, :math:`K = k w` (``w`` slit width),
:math:`X = d / w` (``d`` slit depth) and the end correction
:math:`e = (1/\pi)(\ln(8/K) - 0.57722)`:

.. math::

   \tau = \frac{m K \cos^2(Ke)}
   {2 n^2 \left[ \frac{\sin^2(KX + 2Ke)}{\cos^2(Ke)}
   + \frac{K^2}{2 n^2} \left( 1 + \cos(KX) \cos(KX + 2Ke) \right) \right]}

where :math:`m = 8` (diffuse field) or ``4`` (normal incidence), and
:math:`n = 1` (slit in the middle of a plate) or ``0.5`` (slit along an
edge). The model assumes an inviscid air path; maxima in ``tau`` (dips in
``R``) occur at the resonances :math:`d + 2e = z\lambda/2` (Eq. 4.101,
:math:`z = 1, 2, 3, \ldots`).

**Circular aperture (Hopkins Eq. 4.102, Wilson & Soroka).** With the piston
radiation resistance :math:`R_0(2ka) = 1 - 2 J_1(2ka)/(2ka)` and reactance
:math:`X_0(2ka) = 2 H_1(2ka)/(2ka)` (:math:`J_1` Bessel, :math:`H_1` Struve;
radius ``a``, depth ``d``):

.. math::

   \tau = \frac{4 R_0}
   {4 R_0^2 \left[ \cos(kd) - X_0 \sin(kd) \right]^2
   + \left[ \left( R_0^2 - X_0^2 + 1 \right) \sin(kd)
   + 2 X_0 \cos(kd) \right]^2}

**Composite (Hopkins Eq. 4.92).** For elements of area :math:`S_n` and sound
reduction index :math:`R_n` the resultant is the area-weighted energy sum

.. math::

   R = -10 \log_{10}\!\left[ \frac{1}{\sum S_n} \sum S_n \, 10^{-R_n/10} \right]

so a bare opening (:math:`R = 0`, :math:`\tau = 1`) of relative area
:math:`S_a/S` caps the composite at :math:`10 \log_{10}(S / S_a)`. This is the same
energetic combination used by the EN 12354-3/-4 facade model of
:mod:`phonometry.building.prediction.facade`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import j1, struve

from ..._internal.validation import require_choice, require_positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Default speed of sound in air ``c0``, m/s (20 degC).
_SPEED_OF_SOUND: float = 343.0
#: Euler-Mascheroni constant in the slit end correction (Hopkins Eq. 4.100).
_EULER_GAMMA: float = 0.57722
#: Incident-field constant ``m`` (Hopkins Eq. 4.99).
_FIELD_M: dict[str, float] = {"diffuse": 8.0, "normal": 4.0}
#: Slit-position constant ``n`` (Hopkins Eq. 4.99).
_POSITION_N: dict[str, float] = {"mid": 1.0, "edge": 0.5}

__all__ = [
    "ApertureTransmissionResult",
    "circular_aperture_transmission_coefficient",
    "composite_transmission_loss",
    "slit_resonance_frequencies",
    "slit_transmission_coefficient",
    "transmission_loss_from_coefficient",
]


def transmission_loss_from_coefficient(tau: ArrayLike) -> np.ndarray:
    r"""Sound reduction index :math:`R = -10 \log_{10}(\tau)` from a transmission
    coefficient.

    :param tau: Transmission coefficient(s) ``tau`` (> 0). Values above 1 (a
        resonating aperture that transmits more than the incident intensity)
        give a negative ``R``.
    :return: The sound reduction index ``R``, in dB.
    :raises ValueError: for a non-positive coefficient.
    """
    t = np.asarray(tau, dtype=np.float64)
    if np.any(t <= 0.0):
        raise ValueError("'tau' must be positive.")
    return np.asarray(-10.0 * np.log10(t), dtype=np.float64)


@dataclass(frozen=True)
class ApertureTransmissionResult:
    """Transmission through a slit or circular aperture (Hopkins 4.3.10).

    :ivar frequencies: Band centre frequencies, in hertz.
    :ivar transmission_coefficient: Transmission coefficient ``tau`` per band.
    :ivar kind: ``"slit"`` or ``"circular"``.
    :ivar width: Slit width, in metres, retained (with ``depth``) so
        :meth:`plot_geometry` can draw the section; appended after the
        original fields and ``None`` for circular apertures or hand-built
        results.
    :ivar radius: Circular-aperture radius, in metres, or ``None``.
    :ivar depth: Wall thickness, in metres, or ``None``.
    """

    frequencies: np.ndarray
    transmission_coefficient: np.ndarray
    kind: str
    width: float | None = None
    radius: float | None = None
    depth: float | None = None

    @property
    def transmission_loss(self) -> np.ndarray:
        r"""Aperture sound reduction index :math:`R = -10 \log_{10}(\tau)` per
        band, dB."""
        return transmission_loss_from_coefficient(self.transmission_coefficient)

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the aperture sound reduction index ``R(f)``.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_aperture_transmission

        check_language(language)
        return plot_aperture_transmission(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the wall-aperture cross-section to scale.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the result does not retain its geometry.
        """
        from ..._i18n import check_language
        from ..._plot.geometry import plot_aperture_result_geometry

        check_language(language)
        return plot_aperture_result_geometry(
            self, ax=ax, language=language, **kwargs
        )


def _slit_end_correction(k_big: np.ndarray) -> np.ndarray:
    r"""End correction :math:`e = (1/\pi)(\ln(8/K) - \gamma)` (Hopkins
    Eq. 4.100)."""
    return np.asarray(
        (np.log(8.0 / k_big) - _EULER_GAMMA) / np.pi, dtype=np.float64
    )


def slit_transmission_coefficient(
    frequency: ArrayLike,
    width: float,
    depth: float,
    *,
    field: str = "diffuse",
    position: str = "mid",
    speed_of_sound: float = _SPEED_OF_SOUND,
) -> ApertureTransmissionResult:
    r"""Transmission coefficient of a straight-edged slit (Hopkins Eq. 4.99).

    :param frequency: Band centre frequencies ``f``, in hertz (array, > 0).
    :param width: Slit width ``w``, in m (> 0).
    :param depth: Slit depth ``d`` (wall thickness across the slit), in m (> 0).
    :param field: ``"diffuse"`` (:math:`m = 8`) or ``"normal"``
        (:math:`m = 4`).
    :param position: ``"mid"`` (:math:`n = 1`) or ``"edge"``
        (:math:`n = 0.5`).
    :param speed_of_sound: Speed of sound in air ``c0`` (Default: 343 m/s).
    :return: An :class:`ApertureTransmissionResult` (kind ``"slit"``).
    :raises ValueError: for a non-positive input or unknown field/position.
    """
    w = require_positive(width, "width")
    d = require_positive(depth, "depth")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    field = require_choice(field, "field", tuple(_FIELD_M))
    position = require_choice(position, "position", tuple(_POSITION_N))
    f = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    if f.ndim != 1 or f.size == 0:
        raise ValueError("'frequency' must be a non-empty 1-D array.")
    if np.any(f <= 0.0):
        raise ValueError("'frequency' must be positive.")

    m = _FIELD_M[field]
    n = _POSITION_N[position]
    k = 2.0 * np.pi * f / c0
    k_big = k * w
    x = d / w
    e = _slit_end_correction(k_big)
    cos_ke = np.cos(k_big * e)
    kx = k_big * x
    kx_2ke = kx + 2.0 * k_big * e
    # Numerator and denominator are both multiplied by cos^2(Ke) relative to
    # the printed Eq. 4.99, an identity that removes the division by cos(Ke)
    # (which vanishes at Ke = pi/2 + n*pi) and keeps tau finite there.
    numerator = m * k_big * cos_ke**4
    denominator = 2.0 * n**2 * (
        np.sin(kx_2ke) ** 2
        + (k_big**2 / (2.0 * n**2)) * cos_ke**2
        * (1.0 + np.cos(kx) * np.cos(kx_2ke))
    )
    tau = numerator / denominator
    return ApertureTransmissionResult(
        frequencies=f,
        transmission_coefficient=np.asarray(tau, dtype=np.float64),
        kind="slit",
        width=float(width),
        depth=float(depth),
    )


def slit_resonance_frequencies(
    depth: float,
    width: float,
    *,
    orders: int = 3,
    speed_of_sound: float = _SPEED_OF_SOUND,
) -> np.ndarray:
    r"""Slit resonance frequencies :math:`d + 2e = z\lambda/2` (Hopkins
    Eq. 4.101).

    Maxima in the transmission coefficient (dips in ``R``) occur where the
    effective slit depth is a half-wavelength multiple. Solved iteratively
    because the end correction ``e`` depends weakly on frequency.

    :param depth: Slit depth ``d``, in m (> 0).
    :param width: Slit width ``w``, in m (> 0).
    :param orders: Number of resonance orders ``z = 1..orders`` (>= 1).
    :param speed_of_sound: Speed of sound in air ``c0`` (Default: 343 m/s).
    :return: The resonance frequencies (Hz), one per order.
    :raises ValueError: for a non-positive input, ``orders < 1``, or a slit so
        wide relative to its depth that the effective depth :math:`d + 2e` is
        non-positive (no resonance exists; ``width`` must be much less than the
        wavelength).
    """
    d = require_positive(depth, "depth")
    w = require_positive(width, "width")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    if orders < 1:
        raise ValueError("'orders' must be at least 1.")
    out = np.empty(orders, dtype=np.float64)
    for i in range(orders):
        z = i + 1
        # Fixed-point iteration on f: f = z c0 / (2 (d + 2 e(f))).
        f = z * c0 / (2.0 * d)
        for _ in range(50):
            k_big = (2.0 * np.pi * f / c0) * w
            e = (np.log(8.0 / k_big) - _EULER_GAMMA) / np.pi
            effective_depth = d + 2.0 * e * w
            if effective_depth <= 0.0:
                raise ValueError(
                    "slit too wide relative to its depth: the effective depth "
                    "d + 2e is non-positive, so no resonance exists (the "
                    "Gomperts slit model requires 'width' << wavelength)."
                )
            f_new = z * c0 / (2.0 * effective_depth)
            if abs(f_new - f) < 1e-9 * f:
                f = f_new
                break
            f = f_new
        out[i] = f
    return out


def circular_aperture_transmission_coefficient(
    frequency: ArrayLike,
    radius: float,
    depth: float,
    *,
    speed_of_sound: float = _SPEED_OF_SOUND,
) -> ApertureTransmissionResult:
    """Transmission coefficient of a circular aperture (Hopkins Eq. 4.102).

    :param frequency: Band centre frequencies ``f``, in hertz (array, > 0).
    :param radius: Aperture radius ``a``, in m (> 0).
    :param depth: Aperture depth ``d`` (wall thickness), in m (> 0).
    :param speed_of_sound: Speed of sound in air ``c0`` (Default: 343 m/s).
    :return: An :class:`ApertureTransmissionResult` (kind ``"circular"``).
    :raises ValueError: for a non-positive input.
    """
    a = require_positive(radius, "radius")
    d = require_positive(depth, "depth")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    f = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    if f.ndim != 1 or f.size == 0:
        raise ValueError("'frequency' must be a non-empty 1-D array.")
    if np.any(f <= 0.0):
        raise ValueError("'frequency' must be positive.")

    k = 2.0 * np.pi * f / c0
    x = 2.0 * k * a
    r0 = 1.0 - 2.0 * j1(x) / x
    x0 = 2.0 * struve(1, x) / x
    kd = k * d
    term_a = 4.0 * r0**2 * (np.cos(kd) - x0 * np.sin(kd)) ** 2
    term_b = (
        (r0**2 - x0**2 + 1.0) * np.sin(kd) + 2.0 * x0 * np.cos(kd)
    ) ** 2
    tau = 4.0 * r0 / (term_a + term_b)
    return ApertureTransmissionResult(
        frequencies=f,
        transmission_coefficient=np.asarray(tau, dtype=np.float64),
        kind="circular",
        radius=float(radius),
        depth=float(depth),
    )


def composite_transmission_loss(
    areas: ArrayLike, reduction_indices: ArrayLike
) -> np.ndarray:
    r"""Composite sound reduction index of parallel elements (Hopkins
    Eq. 4.92).

    The area-weighted energy combination of ``N`` elements (wall, window,
    slit, open aperture ...) sharing a partition:

    .. math::

       R = -10 \log_{10}\!\left[ \frac{1}{\sum S_n} \sum S_n \, 10^{-R_n/10}
       \right]

    A bare opening enters with :math:`R = 0` (:math:`\tau = 1`).

    :param areas: Element areas ``S_n``, in m^2 (1-D, length ``N``, all > 0).
    :param reduction_indices: Element sound reduction indices ``R_n``, in dB.
        Either a 1-D array of length ``N`` (one value per element) or a 2-D
        array of shape ``(N, M)`` (``N`` elements over ``M`` bands).
    :return: The composite ``R``: a scalar array for 1-D input, or one value
        per band (length ``M``) for 2-D input.
    :raises ValueError: for a non-positive area, mismatched shapes, or an
        empty element set.
    """
    s = np.atleast_1d(np.asarray(areas, dtype=np.float64))
    if s.ndim != 1 or s.size == 0:
        raise ValueError("'areas' must be a non-empty 1-D array.")
    if np.any(s <= 0.0) or not np.all(np.isfinite(s)):
        raise ValueError("'areas' must be positive and finite.")
    r = np.asarray(reduction_indices, dtype=np.float64)
    if r.ndim == 1:
        if r.shape[0] != s.shape[0]:
            raise ValueError("'reduction_indices' length must match 'areas'.")
        tau = 10.0 ** (-r / 10.0)
        total_1d = float(np.sum(s * tau) / np.sum(s))
        return np.asarray(-10.0 * np.log10(total_1d), dtype=np.float64)
    if r.ndim == 2:
        if r.shape[0] != s.shape[0]:
            raise ValueError(
                "'reduction_indices' first axis must match 'areas'."
            )
        tau = 10.0 ** (-r / 10.0)
        total_2d = np.sum(s[:, None] * tau, axis=0) / np.sum(s)
        return np.asarray(-10.0 * np.log10(total_2d), dtype=np.float64)
    raise ValueError("'reduction_indices' must be 1-D or 2-D.")

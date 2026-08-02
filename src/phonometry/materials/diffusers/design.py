#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Far-field polar response and diffusion coefficient predicted from a diffuser design.

Where :mod:`phonometry.materials.diffusers.scattering_diffusion` evaluates the *measured*
directional diffusion coefficient of ISO 17497-2 from a set of reflected
sound-pressure levels, this module *predicts* those levels from the physical
design of a Schroeder phase-grating diffuser (well-depth sequence and well
width), so the diffusion coefficient can be estimated before a sample is built.

The prediction is the standard single-plane Fraunhofer (Fourier) far-field model
of a phase-grating surface, as set out in Cox and D'Antonio, *Acoustic Absorbers
and Diffusers* (3rd ed., CRC Press, 2017):

* Each well of depth ``d_n`` behaves, at normal incidence and with a rigid
  bottom, as a locally reacting patch of pressure reflection coefficient
  :math:`R_n = e^{-2jkd_n}` (Chapter 10; the phase change of a wave travelling
  down and back up the well). An arbitrary complex reflection coefficient per
  well is accepted as well, so admittance or resonator-loaded surfaces computed
  elsewhere can be fed in directly.
* The scattered pressure at reflection angle :math:`\theta` for a source at
  incidence :math:`\psi` is the sum over the wells of the periodic surface
  (Chapter 5, Equation (5.8), and Chapter 9, Equation (9.32)):
  :math:`p(\theta) = F(\theta) \sum_n R_n
  e^{jkx_n(\sin\psi + \sin\theta)}`,
  with ``x_n`` the centre of the ``n``-th well and :math:`k = 2\pi f/c`. The
  optional prefactor :math:`F(\theta)` collects the single-well aperture
  directivity :math:`\operatorname{sinc}(kw(\sin\psi + \sin\theta)/2)` (the
  Fourier transform of one well of width ``w``) and the Kirchhoff obliquity
  factor :math:`(1 + \cos\theta)/2` of Equation (9.32); both are absorbed
  into the constant ``A`` of Equation (5.8) and can be switched off to
  recover the plain Fourier form.

The polar levels :math:`L_i = 20 \log_{10}|p(\theta_i)|` are then passed to the
ISO 17497-2 autocorrelation diffusion coefficient of
:func:`~phonometry.materials.diffusers.scattering_diffusion.directional_diffusion_coefficient`,
and normalised against the same-footprint flat reference with
:func:`~phonometry.materials.diffusers.scattering_diffusion.normalized_diffusion_coefficient`
(Formula (7)), exactly as a measurement would be reduced.

For a quadratic residue diffuser the depth sequence follows from the prime
generator ``N`` and the design frequency ``f_0`` (Cox and D'Antonio,
Equations (10.2) and (10.3)):

.. math::

   s_n = n^2 \bmod N, \qquad
   d_n = \frac{s_n \lambda_0}{2N}, \qquad
   \lambda_0 = c / f_0.

The model is a far-field approximation and, like every Fourier diffuser model,
loses accuracy at low frequencies, at grazing angles and for surfaces with
strong absorption; it is meant for design estimation, not for replacing an
ISO 17497-2 measurement. A resonator-loaded "metadiffuser" slit surface is not
modelled here: only its per-well reflection coefficient would change, so such a
surface can be predicted by supplying the complex ``reflection`` sequence
directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..._internal.types import Real
from .scattering_diffusion import (
    DiffusionSpectrum,
    diffusion_spectrum,
    directional_diffusion_coefficient,
    normalized_diffusion_coefficient,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

#: Complex-valued complex128 array alias (per-well reflection coefficients).
Complex = NDArray[np.complex128]


__all__ = [
    "DEFAULT_POLAR_ANGLES",
    "DiffuserPolarResponse",
    "predict_diffuser_polar_response",
    "predicted_diffusion_spectrum",
    "qrd_well_depths",
    "quadratic_residue_sequence",
]

#: Reference speed of sound in air used by the design helpers when none is
#: supplied, in metres per second (20 degC, after ISO 9613-1).
_C_DEFAULT = 343.0

#: Default single-plane receiver angles of ISO 17497-2: a semicircle from
#: -90 deg to +90 deg in 5 deg steps (Clause 6.3, Figure 4), in degrees.
DEFAULT_POLAR_ANGLES: tuple[int, ...] = tuple(range(-90, 91, 5))


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------
def _positive_scalar(value: float, name: str) -> float:
    """Return ``value`` as a positive, finite float or raise ``ValueError``."""
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"'{name}' must be a positive, finite number.")
    return v


def _prime_generator(prime: int) -> int:
    """Return ``prime`` as an odd prime >= 3 or raise ``ValueError``."""
    n = int(prime)
    if n != prime:
        raise ValueError("'prime' must be an integer.")
    if n < 3 or n % 2 == 0:
        raise ValueError("'prime' must be an odd prime >= 3.")
    for factor in range(3, math.isqrt(n) + 1, 2):
        if n % factor == 0:
            raise ValueError(f"'prime' must be prime; {n} is divisible by {factor}.")
    return n


# ---------------------------------------------------------------------------
# Quadratic residue diffuser geometry (Cox and D'Antonio, Eqs. (10.2)/(10.3)).
# ---------------------------------------------------------------------------
def quadratic_residue_sequence(prime: int) -> Real:
    r"""Quadratic residue sequence ``s_n`` (Cox and D'Antonio, Eq. (10.2)).

    :math:`s_n = n^2 \bmod N` for :math:`n = 0, 1, \ldots, N - 1`, the least
    non-negative remainders, where ``N`` is the (odd) prime generator and
    number of wells per period. For :math:`N = 7` this returns
    ``[0, 1, 4, 2, 2, 4, 1]``.

    :param prime: Prime generator ``N`` (odd prime, at least 3).
    :return: The sequence ``s_n`` as an integer-valued float array of length
        ``N``.
    :raises ValueError: if ``prime`` is not an odd prime of at least 3.
    """
    n = _prime_generator(prime)
    indices = np.arange(n, dtype=np.int64)
    return np.asarray((indices**2) % n, dtype=np.float64)


def qrd_well_depths(
    prime: int,
    design_frequency: float,
    *,
    speed_of_sound: float = _C_DEFAULT,
) -> Real:
    r"""Well depths of a quadratic residue diffuser (Cox and D'Antonio, Eq. (10.3)).

    :math:`d_n = s_n \lambda_0 / (2N)` with the quadratic residue sequence
    ``s_n`` of :func:`quadratic_residue_sequence` and the design wavelength
    :math:`\lambda_0 = c / f_0`. The depths span 0 to roughly
    :math:`\lambda_0 / 2`.

    :param prime: Prime generator ``N`` (odd prime, at least 3).
    :param design_frequency: Design frequency ``f_0``, in hertz, normally the
        lower frequency limit of the device.
    :param speed_of_sound: Speed of sound ``c``, in metres per second; defaults
        to 343 m/s.
    :return: Well depths ``d_n``, in metres, one per well of a single period.
    :raises ValueError: if ``prime`` is not an odd prime, or if
        ``design_frequency`` or ``speed_of_sound`` is not positive.
    """
    n = _prime_generator(prime)
    f0 = _positive_scalar(design_frequency, "design_frequency")
    c = _positive_scalar(speed_of_sound, "speed_of_sound")
    lambda0 = c / f0
    s = quadratic_residue_sequence(n)
    return np.asarray(s * lambda0 / (2.0 * n), dtype=np.float64)


# ---------------------------------------------------------------------------
# Far-field Fraunhofer model of a periodic phase-grating surface.
# ---------------------------------------------------------------------------
def _scattered_pressure(
    reflection: Complex,
    well_width: float,
    frequency: float,
    angles_deg: Real,
    *,
    source_angle: float,
    periods: int,
    speed_of_sound: float,
    include_aperture: bool,
    include_obliquity: bool,
) -> Complex:
    r"""Complex far-field scattered pressure of a periodic surface (arbitrary units).

    Implements
    :math:`p(\theta) = F(\theta) \sum_n R_n e^{jkx_n(\sin\psi + \sin\theta)}`
    (Cox and D'Antonio, Eqs. (5.8)/(9.32)) for ``periods`` repetitions of the
    single-period reflection sequence ``reflection``. The result is defined only
    up to the overall constant ``A``; only relative levels matter for the
    diffusion coefficient.
    """
    r_period = np.asarray(reflection, dtype=np.complex128)
    r_all = np.tile(r_period, periods)
    m = r_all.size
    k = 2.0 * np.pi * frequency / speed_of_sound
    # Well centres, symmetric about the origin (a global phase offset would not
    # change |p|, but a symmetric array keeps a normal-incidence response even).
    centres = (np.arange(m, dtype=np.float64) - (m - 1) / 2.0) * well_width
    theta = np.radians(angles_deg)
    psi = math.radians(source_angle)
    spatial = np.sin(theta) + math.sin(psi)  # (A,)
    phase = np.exp(1j * k * centres[None, :] * spatial[:, None])  # (A, M)
    pressure = phase @ r_all  # (A,)
    if include_aperture:
        # Fourier transform of one well of width w: the single-well directivity.
        pressure = pressure * np.sinc(k * well_width * spatial / (2.0 * np.pi))
    if include_obliquity:
        # Kirchhoff obliquity factor. Cox and D'Antonio Eq. (9.32) print the
        # normal-incidence form (1 + cos theta) / 2; for an oblique source the
        # Kirchhoff-approximation factor generalises to the symmetric
        # (cos theta + cos psi) / 2, which reduces to Eq. (9.32) at psi = 0.
        pressure = pressure * (np.cos(theta) + math.cos(psi)) / 2.0
    return np.asarray(pressure, dtype=np.complex128)


def _polar_levels(pressure: Complex) -> Real:
    r"""Reflected levels :math:`L_i = 20 \log_{10}|p_i|` (dB), floored finitely.

    The levels are referenced to the peak magnitude so the strongest receiver is
    at 0 dB; the reference is immaterial to the diffusion coefficient, which is a
    ratio of energies. A magnitude of exactly zero maps to a large negative
    finite level rather than ``-inf`` so the coefficient stays defined.
    """
    magnitude = np.abs(np.asarray(pressure, dtype=np.complex128))
    peak = float(np.max(magnitude))
    if peak <= 0.0:
        raise ValueError(
            "The predicted polar response carries no energy; check the inputs."
        )
    ratio = magnitude / peak
    floor = 1e-10  # -200 dB relative to the peak.
    return np.asarray(20.0 * np.log10(np.maximum(ratio, floor)), dtype=np.float64)


@dataclass(frozen=True)
class DiffuserPolarResponse:
    """A predicted far-field polar response of a diffuser at one frequency.

    :ivar frequency: Frequency of the prediction, in hertz.
    :ivar angles: Receiver reflection angles, in degrees.
    :ivar levels: Predicted reflected sound-pressure level at each angle, in
        decibels, referenced to the peak of the response (peak at 0 dB).
    :ivar coefficient: Directional diffusion coefficient ``d_theta`` of the
        predicted response (ISO 17497-2, Formula (5)).
    :ivar source_angle: Angle of incidence ``psi`` of the source, in degrees.
    :ivar well_width: Well width ``w`` of the predicted surface, in metres,
        always retained by the predictor (with ``periods``) so
        :meth:`plot_geometry` can draw the well profile; appended after the
        original fields and ``None`` only for hand-built responses.
    :ivar depths: Well depths ``d_n`` of one period, in metres, when the
        response was predicted from depths; ``None`` otherwise (explicit
        ``reflection`` surfaces have no drawable well profile).
    :ivar periods: Number of repeated periods of the prediction.
    """

    frequency: float
    angles: Real
    levels: Real
    coefficient: float
    source_angle: float = 0.0
    well_width: float | None = None
    depths: Real | None = None
    periods: int | None = None

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the predicted polar response with the diffusion coefficient annotated.

        Requires matplotlib (``pip install phonometry[plot]``); returns the polar
        :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_diffuser_polar_response

        check_language(language)
        return plot_diffuser_polar_response(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the well profile of the predicted surface to scale.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the response does not retain its well
            geometry (hand-built, or predicted from an explicit
            ``reflection`` sequence).
        """
        from ..._i18n import check_language
        from ..._plot.geometry import plot_diffuser_geometry

        check_language(language)
        return plot_diffuser_geometry(self, ax=ax, language=language, **kwargs)


def _resolve_reflection(
    depths: ArrayLike | None,
    reflection: ArrayLike | None,
    frequency: float,
    speed_of_sound: float,
) -> Complex:
    r"""Return the per-well reflection coefficient sequence at ``frequency``.

    Exactly one of ``depths`` (rigid-bottom wells,
    :math:`R_n = e^{-2jkd_n}`) or ``reflection`` (an explicit complex
    sequence) must be given.
    """
    if (depths is None) == (reflection is None):
        raise ValueError(
            "Provide exactly one of 'depths' or 'reflection'."
        )
    if reflection is not None:
        r = np.atleast_1d(np.asarray(reflection, dtype=np.complex128))
        if r.ndim != 1 or r.size < 2:
            raise ValueError("'reflection' must be a 1-D sequence of at least two wells.")
        if not np.all(np.isfinite(r)):
            raise ValueError("'reflection' values must be finite.")
        return r
    d = np.atleast_1d(np.asarray(depths, dtype=np.float64))
    if d.ndim != 1 or d.size < 2:
        raise ValueError("'depths' must be a 1-D sequence of at least two wells.")
    if not np.all(np.isfinite(d)) or np.any(d < 0.0):
        raise ValueError("'depths' values must be finite and non-negative.")
    k = 2.0 * np.pi * frequency / speed_of_sound
    return np.asarray(np.exp(-2j * k * d), dtype=np.complex128)


def _prepare_geometry(
    well_width: float,
    frequency: float,
    angles: ArrayLike,
    source_angle: float,
    periods: int,
    speed_of_sound: float,
) -> tuple[float, float, Real, float, int, float]:
    """Validate the shared geometry arguments and return them normalised."""
    w = _positive_scalar(well_width, "well_width")
    f = _positive_scalar(frequency, "frequency")
    c = _positive_scalar(speed_of_sound, "speed_of_sound")
    ang = np.atleast_1d(np.asarray(angles, dtype=np.float64))
    if ang.ndim != 1 or ang.size < 2:
        raise ValueError("'angles' must be a 1-D sequence of at least two angles.")
    if not np.all(np.isfinite(ang)):
        raise ValueError("'angles' values must be finite.")
    psi = float(source_angle)
    if not math.isfinite(psi):
        raise ValueError("'source_angle' must be finite.")
    n_periods = int(periods)
    if n_periods != periods or n_periods < 1:
        raise ValueError("'periods' must be an integer of at least 1.")
    return w, f, ang, psi, n_periods, c


def predict_diffuser_polar_response(
    well_width: float,
    frequency: float,
    *,
    depths: ArrayLike | None = None,
    reflection: ArrayLike | None = None,
    angles: ArrayLike = DEFAULT_POLAR_ANGLES,
    source_angle: float = 0.0,
    periods: int = 1,
    speed_of_sound: float = _C_DEFAULT,
    include_aperture: bool = True,
    include_obliquity: bool = True,
) -> DiffuserPolarResponse:
    r"""Predict the far-field polar response of a diffuser (Cox and D'Antonio, Eq. (5.8)).

    Evaluates the single-plane Fraunhofer scattered pressure of a periodic
    phase-grating surface and reduces it to the ISO 17497-2 directional
    diffusion coefficient. Supply the surface either as rigid-bottom well
    ``depths`` (:math:`R_n = e^{-2jkd_n}`) or as an explicit per-well complex
    ``reflection`` sequence (for admittance or resonator-loaded surfaces); give
    exactly one.

    :param well_width: Well width ``w``, in metres (uniform across wells).
    :param frequency: Frequency of the prediction ``f``, in hertz.
    :param depths: Well depths ``d_n`` of one period, in metres (1-D, at least
        two wells); mutually exclusive with ``reflection``.
    :param reflection: Explicit per-well complex pressure reflection coefficient
        of one period (1-D, at least two wells); mutually exclusive with
        ``depths``.
    :param angles: Receiver reflection angles ``theta``, in degrees; defaults to
        the ISO 17497-2 semicircle :data:`DEFAULT_POLAR_ANGLES`.
    :param source_angle: Angle of incidence ``psi`` of the source, in degrees
        (0 = normal incidence).
    :param periods: Number of repetitions ``N_p`` of the single period; the
        grating lobes that define a Schroeder diffuser require ``periods >= 2``.
    :param speed_of_sound: Speed of sound ``c``, in metres per second.
    :param include_aperture: Include the single-well aperture directivity
        :math:`\operatorname{sinc}(kw(\sin\psi + \sin\theta)/2)` of
        Eq. (9.32); defaults to ``True``.
    :param include_obliquity: Include the Kirchhoff obliquity factor
        :math:`(\cos\theta + \cos\psi)/2`, the oblique-source generalisation
        of the normal-incidence :math:`(1 + \cos\theta)/2` of Eq. (9.32);
        defaults to ``True``.
    :return: A :class:`DiffuserPolarResponse` with the per-angle levels, the
        directional diffusion coefficient and ``.plot()``.
    :raises ValueError: for invalid geometry, or if not exactly one of ``depths``
        and ``reflection`` is supplied.
    """
    w, f, ang, psi, n_periods, c = _prepare_geometry(
        well_width, frequency, angles, source_angle, periods, speed_of_sound
    )
    r_period = _resolve_reflection(depths, reflection, f, c)
    pressure = _scattered_pressure(
        r_period, w, f, ang,
        source_angle=psi, periods=n_periods, speed_of_sound=c,
        include_aperture=include_aperture, include_obliquity=include_obliquity,
    )
    levels = _polar_levels(pressure)
    coefficient = directional_diffusion_coefficient(levels)
    return DiffuserPolarResponse(
        frequency=f,
        angles=ang,
        levels=levels,
        coefficient=coefficient,
        source_angle=psi,
        well_width=w,
        depths=(
            np.asarray(depths, dtype=np.float64) if depths is not None
            else None
        ),
        periods=n_periods,
    )


def predicted_diffusion_spectrum(
    well_width: float,
    frequencies: ArrayLike,
    *,
    depths: ArrayLike | None = None,
    reflection_of: Any | None = None,
    angles: ArrayLike = DEFAULT_POLAR_ANGLES,
    source_angle: float = 0.0,
    periods: int = 1,
    speed_of_sound: float = _C_DEFAULT,
    include_aperture: bool = True,
    include_obliquity: bool = True,
    normalize: bool = True,
) -> DiffusionSpectrum:
    """Predicted diffusion-coefficient spectrum ``d(f)`` of a diffuser design.

    Predicts the far-field polar response at each frequency with
    :func:`predict_diffuser_polar_response`, forms the ISO 17497-2 directional
    diffusion coefficient band by band, and (by default) normalises it against a
    same-footprint flat reference surface (all wells of zero depth) with
    :func:`~phonometry.materials.diffusers.scattering_diffusion.normalized_diffusion_coefficient`
    (Formula (7)). The result reuses
    :class:`~phonometry.materials.diffusers.scattering_diffusion.DiffusionSpectrum`, so it
    plots and reports like a measured spectrum.

    Only the rigid-bottom well ``depths`` path supports the automatic flat
    reference; an explicit frequency-dependent reflection model is out of scope
    for this helper (build the spectrum yourself from
    :func:`predict_diffuser_polar_response` in that case).

    :param well_width: Well width ``w``, in metres.
    :param frequencies: Frequencies of the spectrum, in hertz (1-D).
    :param depths: Well depths ``d_n`` of one period, in metres (1-D, at least
        two wells).
    :param reflection_of: Reserved for future frequency-dependent reflection
        models; must be ``None``.
    :param angles: Receiver reflection angles ``theta``, in degrees; defaults to
        :data:`DEFAULT_POLAR_ANGLES`.
    :param source_angle: Angle of incidence ``psi``, in degrees.
    :param periods: Number of repetitions ``N_p`` of the single period.
    :param speed_of_sound: Speed of sound ``c``, in metres per second.
    :param include_aperture: Include the single-well aperture directivity.
    :param include_obliquity: Include the Kirchhoff obliquity factor.
    :param normalize: If ``True`` (default), also compute the normalised
        coefficient ``d_n`` against the flat reference.
    :return: A :class:`DiffusionSpectrum` carrying the raw diffusion ``d(f)`` and,
        when ``normalize`` is ``True``, the normalised ``d_n(f)``.
    :raises ValueError: for invalid geometry, an empty ``frequencies`` array, or
        if ``reflection_of`` is not ``None``.
    """
    if reflection_of is not None:
        raise ValueError(
            "'reflection_of' is reserved for a future frequency-dependent "
            "reflection model and must be None; build the spectrum from "
            "predict_diffuser_polar_response for an explicit reflection model."
        )
    if depths is None:
        raise ValueError("'depths' is required for a predicted diffusion spectrum.")
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("'frequencies' must be a non-empty 1-D sequence.")
    d_period = np.atleast_1d(np.asarray(depths, dtype=np.float64))
    flat = np.zeros_like(d_period)

    raw = np.empty(freqs.size, dtype=np.float64)
    norm = np.empty(freqs.size, dtype=np.float64) if normalize else None
    for i, f in enumerate(freqs):
        surface = predict_diffuser_polar_response(
            well_width, float(f), depths=d_period, angles=angles,
            source_angle=source_angle, periods=periods,
            speed_of_sound=speed_of_sound, include_aperture=include_aperture,
            include_obliquity=include_obliquity,
        )
        raw[i] = surface.coefficient
        if norm is not None:
            reference = predict_diffuser_polar_response(
                well_width, float(f), depths=flat, angles=angles,
                source_angle=source_angle, periods=periods,
                speed_of_sound=speed_of_sound, include_aperture=include_aperture,
                include_obliquity=include_obliquity,
            )
            norm[i] = float(
                normalized_diffusion_coefficient(
                    surface.coefficient, reference.coefficient
                )
            )
    return diffusion_spectrum(freqs, raw, normalized=norm)

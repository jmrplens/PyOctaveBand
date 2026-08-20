#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""2D near-to-far-field (NTFF) transformation over a closed contour.

Given the steady-state pressure and outward normal velocity phasors on a
closed contour that encloses a scatterer (or any source region), the
exterior field is fully determined by the Kirchhoff-Helmholtz boundary
integral. In two dimensions, with the :math:`e^{+j \omega t}` time
convention
used throughout the library, the free-space Green function is

.. math::

   G(R) = -(j / 4)\, H_0^{(2)}(k R),

with :math:`H_0^{(2)}` the Hankel function of the second kind (outgoing
waves) and
``R`` the source-observer distance, and the exterior representation reads

.. math::

   p(r) = \oint_S \left[ p(r')\, dG/dn'
   + j \omega \rho\, v_n(r')\, G \right] dl',

where ``n'`` is the outward normal of the contour ``S`` and the momentum
equation :math:`dp/dn = -j \omega \rho v_n` eliminated the pressure
gradient. The
normal derivative of the Green function brings in :math:`H_1^{(2)}`:
:math:`dG/dR = (j k / 4) H_1^{(2)}(k R)`. This is the same construction
full-wave
solvers use to report far-field scattering patterns from near-field data
(e.g. the finite-element polar responses of Jimenez, Cox, Romero-Garcia
and Groby, *Metadiffusers: Deep-subwavelength sound diffusers*, Sci. Rep.
7, 5389, 2017); the classical background is Williams, *Fourier Acoustics*
(Academic Press, 1999), chapter 8.

:func:`far_field_from_contour` evaluates that integral for phasors captured
by :meth:`~phonometry.simulation.FDTD2D.add_contour_probe` (or assembled by
hand into a :class:`ContourPhasors`), either in the true far-field limit,
where
:math:`H_0^{(2)}(kR) \to \sqrt{2 / (\pi k R)}\, e^{-j (k R - \pi / 4)}`
turns the
integral into an angular pattern :math:`F(\theta)` with
:math:`p(r, \theta) \to F(\theta)\, e^{-j k r} / \sqrt{r}`, or at a finite
observation radius with the exact Hankel kernels.

Because the integral representation is source-free inside ``S`` for any
field whose sources lie outside the contour, the incident wave of a
scattering run contributes (analytically) nothing to the exterior integral:
transforming total-field phasors already yields the scattered far field.
Subtracting a reference run without the scatterer
(:meth:`ContourPhasors.subtract`) removes the residual of that cancellation
caused by grid dispersion; on the validation scenes the residual stays
below 0.01 dB, so the subtraction is optional in practice.

Validated against analytic oracles: the omnidirectional pattern and the
absolute level of a monopole line source reconstructed from an enclosing
contour, the null pattern of an antiphase source pair, the extinction of a
contour that does not enclose the source, and the far-field polar response
of meshed Schroeder-type diffusers against the library's Fraunhofer model
(``tests/simulation/test_fdtd_ntff.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import hankel2

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ContourPhasors",
    "far_field_from_contour",
]


@dataclass(frozen=True)
class ContourPhasors:
    r"""Steady-state ``p`` and ``v_n`` phasors sampled on a closed contour.

    The phasors follow the library's :math:`e^{+j \omega t}` convention:
    :math:`p(t) = \operatorname{Re}\{p \, e^{+j \omega t}\}`.
    ``normals`` point outward
    (away from the enclosed region) and ``normal_velocity`` is the particle
    velocity component along them. Instances are produced by
    :meth:`~phonometry.simulation.FDTD2D.add_contour_probe` probes; they can
    also be assembled by hand from any solver's near field.

    :ivar frequency: Frequency of the phasors [Hz].
    :ivar positions: Sample positions ``(x, y)`` [m], shape ``(n, 2)``.
    :ivar normals: Outward unit normals, shape ``(n, 2)``.
    :ivar pressure: Complex pressure phasor at each sample [Pa].
    :ivar normal_velocity: Complex outward normal velocity phasor [m/s].
    :ivar segment: Contour length carried by each sample ``dl`` [m].
    """

    frequency: float
    positions: NDArray[np.float64]
    normals: NDArray[np.float64]
    pressure: NDArray[np.complex128]
    normal_velocity: NDArray[np.complex128]
    segment: float

    def __post_init__(self) -> None:
        """Check positive scalars, matched shapes and finite samples; coerce dtypes."""
        if not np.isfinite(self.frequency) or self.frequency <= 0.0:
            msg = "frequency must be positive and finite"
            raise ValueError(msg)
        if not np.isfinite(self.segment) or self.segment <= 0.0:
            msg = "segment must be positive and finite"
            raise ValueError(msg)
        pos = np.asarray(self.positions, dtype=np.float64)
        nrm = np.asarray(self.normals, dtype=np.float64)
        p = np.asarray(self.pressure, dtype=np.complex128)
        v = np.asarray(self.normal_velocity, dtype=np.complex128)
        n = p.shape[0] if p.ndim == 1 else -1
        if n < 1 or pos.shape != (n, 2) or nrm.shape != (n, 2) or v.shape != (n,):
            msg = (
                "positions and normals must have shape (n, 2) and pressure "
                "and normal_velocity shape (n,) for the same n >= 1"
            )
            raise ValueError(msg)
        if not (
            np.all(np.isfinite(pos))
            and np.all(np.isfinite(nrm))
            and np.all(np.isfinite(p))
            and np.all(np.isfinite(v))
        ):
            msg = "contour phasor arrays must be finite"
            raise ValueError(msg)
        # Store the converted arrays so hand-built instances (lists, mixed
        # dtypes) behave exactly like probe-built ones downstream.
        object.__setattr__(self, "positions", pos)
        object.__setattr__(self, "normals", nrm)
        object.__setattr__(self, "pressure", p)
        object.__setattr__(self, "normal_velocity", v)

    def subtract(self, reference: ContourPhasors) -> ContourPhasors:
        """Phasor difference on the same contour: ``self - reference``.

        The standard way to isolate the scattered field of a plane-wave
        scattering run: capture the same contour in a second, otherwise
        identical simulation without the scatterer and subtract its
        (incident-only) phasors. Geometry and frequency must match.
        """
        if not np.isclose(self.frequency, reference.frequency):
            msg = "cannot subtract phasors at different frequencies"
            raise ValueError(msg)
        if self.positions.shape != reference.positions.shape or not (
            np.allclose(self.positions, reference.positions)
            and np.allclose(self.normals, reference.normals)
        ):
            msg = "cannot subtract phasors sampled on different contours"
            raise ValueError(msg)
        return ContourPhasors(
            frequency=self.frequency,
            positions=self.positions,
            normals=self.normals,
            pressure=np.asarray(self.pressure - reference.pressure),
            normal_velocity=np.asarray(
                self.normal_velocity - reference.normal_velocity
            ),
            segment=self.segment,
        )


def far_field_from_contour(
    contour: ContourPhasors,
    angles: ArrayLike,
    *,
    distance: float | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
    speed_of_sound: float = 343.0,
    air_density: float = 1.2,
) -> NDArray[np.complex128]:
    r"""Exterior field of a closed contour: the 2D Kirchhoff-Helmholtz integral.

    Evaluates, for each observation angle ``a`` (degrees, measured from the
    ``+x`` axis towards ``+y`` of the grid coordinates, so the unit
    direction is :math:`u = (\cos a, \sin a)`),

    .. math::

       p(r) = \oint_S \left[ p\, dG/dn'
       + j \omega \rho\, v_n G \right] dl'

    with the outgoing 2D free-space Green function
    :math:`G = -(j/4) H_0^{(2)}(k R)` and its normal derivative through
    :math:`dG/dR = (j k / 4) H_1^{(2)}(k R)` (:math:`e^{+j \omega t}`
    convention,
    :func:`scipy.special.hankel2`).

    With ``distance=None`` (the default) the far-field limit
    :math:`H_0^{(2)}(kR) \to \sqrt{2 / (\pi k R)}\, e^{-j (k R - \pi/4)}`
    is taken
    analytically and the returned complex pattern :math:`F(a)` is the
    relative
    far-field amplitude defined by

    .. math::

       p(r, a) \to F(a)\, e^{-j k r} / \sqrt{r}
       \quad \text{as } r \to \infty,

    with ``r`` measured from ``origin`` (the phase reference; magnitudes at
    infinity do not depend on it). With a finite ``distance`` the exact
    Hankel kernels are used instead and the return value is the complex
    pressure [Pa] on the circle of that radius around ``origin`` (which must
    lie outside the contour samples).

    For radiating problems the contour must enclose every source of the
    field being transformed. For scattering runs (scatterer enclosed,
    incident source outside) the incident field integrates to nothing, so
    total-field phasors transform directly into the scattered far field;
    subtracting a no-scatterer reference (:meth:`ContourPhasors.subtract`)
    is optional numerical cleanup of the grid-dispersion residual of that
    cancellation (below 0.01 dB on the validation scenes).

    :param contour: The contour phasors (from
        :meth:`~phonometry.simulation.FDTD2D.add_contour_probe`, or hand
        built).
    :param angles: Observation angles [degrees], 1D. ``0`` is ``+x``,
        ``90`` is ``+y`` of the coordinates ``contour.positions`` live in.
    :param distance: Observation radius [m] for the exact evaluation, or
        ``None`` (default) for the far-field pattern.
    :param origin: Phase-reference point ``(x, y)`` [m]; the centre of the
        observation circle when ``distance`` is given. Default the grid
        origin.
    :param speed_of_sound: Speed of sound ``c`` [m/s] (default 343).
    :param air_density: Density ``rho`` [kg/m3] (default 1.2).
    :return: Complex array, one value per angle: the far-field pattern
        ``F(a)`` (units Pa sqrt(m)), or the pressure [Pa] at ``distance``.
    :raises ValueError: For invalid angles, a non-positive ``distance``, or
        an observation circle that does not clear the contour.
    """
    if not isinstance(contour, ContourPhasors):
        msg = "contour must be a ContourPhasors"
        raise TypeError(msg)
    ang = np.atleast_1d(np.asarray(angles, dtype=np.float64))
    if ang.ndim != 1 or ang.size == 0 or not np.all(np.isfinite(ang)):
        msg = "angles must be a non-empty 1D sequence of finite values in degrees"
        raise ValueError(msg)
    c = float(speed_of_sound)
    rho = float(air_density)
    if not np.isfinite(c) or c <= 0.0:
        msg = "speed_of_sound must be positive and finite"
        raise ValueError(msg)
    if not np.isfinite(rho) or rho <= 0.0:
        msg = "air_density must be positive and finite"
        raise ValueError(msg)
    omega = 2.0 * np.pi * contour.frequency
    k = omega / c
    ox, oy = float(origin[0]), float(origin[1])
    if not (np.isfinite(ox) and np.isfinite(oy)):
        msg = "origin must be finite"
        raise ValueError(msg)
    pos = contour.positions - np.asarray([ox, oy])
    rad = np.radians(ang)
    u = np.stack((np.cos(rad), np.sin(rad)), axis=1)  # (A, 2)
    monopole = 1j * omega * rho * contour.normal_velocity  # j w rho v_n
    if distance is None:
        # Far-field limit: dG/dn' -> (j k u.n') G and the common cylindrical
        # spreading exp(-j k r)/sqrt(r) is factored out of both kernels.
        prefactor = -0.25j * np.sqrt(2.0 / (np.pi * k)) * np.exp(0.25j * np.pi)
        kernel = np.exp(1j * k * (u @ pos.T))  # (A, n)
        integrand = 1j * k * (u @ contour.normals.T) * contour.pressure + monopole
        return np.asarray(
            prefactor * contour.segment * np.sum(kernel * integrand, axis=1),
            dtype=np.complex128,
        )
    d = float(distance)
    if not np.isfinite(d) or d <= 0.0:
        msg = "distance must be positive and finite"
        raise ValueError(msg)
    reach = float(np.max(np.hypot(pos[:, 0], pos[:, 1])))
    if d <= reach:
        msg = (
            f"distance {d:g} m does not clear the contour (farthest sample "
            f"{reach:g} m from origin); the representation is exterior only"
        )
        raise ValueError(msg)
    obs = d * u  # (A, 2)
    sep = obs[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (A, n, 2)
    dist = np.hypot(sep[..., 0], sep[..., 1])  # (A, n)
    green = -0.25j * hankel2(0, k * dist)
    # dG/dn' = dG/dR * (r' - r).n' / R with dG/dR = (j k / 4) H1(2)(k R).
    dgdn = (
        0.25j
        * k
        * hankel2(1, k * dist)
        * np.sum(-sep * contour.normals[np.newaxis, :, :], axis=2)
        / dist
    )
    return np.asarray(
        contour.segment * np.sum(contour.pressure * dgdn + monopole * green, axis=1),
        dtype=np.complex128,
    )

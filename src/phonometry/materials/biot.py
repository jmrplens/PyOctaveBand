#  Copyright (c) 2026. Jose M. Requena-Plens
"""Biot poroelastic layers: the three waves and the 6x6 transfer matrix.

An equivalent fluid replaces a porous material by a single wave travelling in
the pores while the skeleton stands still (rigid frame) or is merely dragged
along by the pore fluid (limp frame, see
:func:`~phonometry.materials.porous_absorber.limp_frame`). Neither can carry a
wave in the skeleton itself, so neither can produce a frame resonance. The Biot
theory can: it treats the frame as an elastic solid coupled to the pore fluid
through a potential coupling coefficient and an inertial coupling coefficient,
and it predicts **three** waves in an isotropic porous layer, two compressional
and one shear (Allard & Atalla, *Propagation of Sound in Porous Media* 2e,
chapter 6).

This module implements that theory in the ``e^{+j w t}`` convention of the rest
of the package, exactly as printed:

* **Elastic coefficients** ``P``, ``Q`` and ``R`` for the usual case of a frame
  built from a material far stiffer than the frame itself (``Ks -> inf``),
  Eqs. (6.26)-(6.29) printed pp. 116-117: ``R = phi Kf``,
  ``Q = (1 - phi) Kf``, ``P = 4N/3 + Kb + (1 - phi)^2 Kf / phi`` with the frame
  bulk modulus ``Kb = 2N(1 + nu)/(3(1 - 2 nu))``.
* **Modified densities** ``rho11``, ``rho12``, ``rho22`` (Eq. (6.56), printed
  p. 120) built from the inertial coupling ``rho_a = phi rho0 (a_inf - 1)`` and
  the visco-inertial term ``j sigma phi^2 G(w) / w``. The latter is *not*
  re-derived here: it is read back from the equivalent-fluid model handed in,
  through the identity ``rho22 = phi^2 rho_eq`` stated on printed p. 253, so a
  Biot layer and a rigid-frame layer built from the same
  :class:`~phonometry.materials.porous_absorber.PorousMediumResult` share one
  visco-thermal description by construction.
* **The two compressional waves** as the eigenvalues of Eq. (6.65)
  (Eqs. (6.67)-(6.69), printed p. 121), the **shear wave** of Eq. (6.83), and
  the fluid-to-frame velocity ratios ``mu1``, ``mu2`` (Eq. (6.71)) and ``mu3``
  (Eq. (6.84)).
* **The closed-form surface impedance** of a hard-backed layer at normal
  incidence, Eqs. (6.107)-(6.108) printed p. 128, in
  :func:`biot_surface_impedance`, and the ``lambda/4`` frame resonance it
  develops, Eq. (6.110) printed p. 129, in
  :func:`frame_quarter_wave_resonance`.
* **The 6x6 layer matrix** of chapter 11: the field vector
  ``[v1s, v3s, v3f, s33s, s13s, s33f]`` (Eq. (11.26)) expressed through the
  wave-amplitude matrix ``[Gamma(x3)]`` of Table 11.1 (printed p. 252), from
  which :func:`poroelastic_transfer_matrix` returns
  ``[T] = [Gamma(-h)][Gamma(0)]^-1`` (Eq. (11.34)).

The layer is used through
:class:`~phonometry.materials.porous_absorber.PoroelasticLayer` inside
:func:`~phonometry.materials.porous_absorber.layered_absorber`, which switches
to the global-matrix assembly of Sect. 11.5 as soon as a poroelastic layer is
present, with the coupling matrices of Sect. 11.4 (``[Ipf]``/``[Jpf]``
Eq. (11.73), ``[Ipp]`` Eq. (11.67)) and the hard-wall conditions ``[Y]`` of
Eq. (11.81).

**On oracles.** Allard & Atalla publish no table of computed surface
impedances, so the model is anchored on closed forms and on exact limits: the
rigid-frame limit reproduces the already-anchored Johnson-Champoux-Allard
equivalent fluid, the limp limit reproduces
:func:`~phonometry.materials.porous_absorber.limp_frame`, and the chapter 11
assembly reproduces the chapter 6 closed form Eq. (6.107) to machine precision.
The book does print three *output* digits for the fully specified glass wool of
Table 6.1, and all three are reproduced: the airborne branch changes from
``(delta1, mu1)`` to ``(delta2, mu2)`` at 495 Hz (printed p. 124),
``|mu_a| > 40`` above 50 Hz (printed p. 124) and the impedance peak of a
5,6 cm layer sits at 860 Hz (printed p. 129). See ``tests/materials/test_biot.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._internal.types import Real
from .._internal.validation import require_positive

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

    from .porous_absorber import PorousMediumResult

Complex = NDArray[np.complex128]

__all__ = [
    "BiotWavesResult",
    "biot_surface_impedance",
    "biot_waves",
    "frame_bulk_modulus",
    "frame_elastic_coefficient",
    "frame_quarter_wave_resonance",
    "poroelastic_transfer_matrix",
]

#: Shared validation message for open porosities.
_POROSITY_MESSAGE = "'porosity' must not exceed 1."


def _require_shear_modulus(value: complex, name: str) -> complex:
    """Validate a complex shear modulus ``N = N'(1 + j eta)``."""
    n = complex(value)
    if not np.isfinite(n):
        raise ValueError(f"'{name}' must be finite.")
    if n.real <= 0.0:
        raise ValueError(f"'{name}' must have a positive real part.")
    if n.imag < 0.0:
        raise ValueError(
            f"'{name}' must have a non-negative imaginary part in the "
            "e^{+j w t} convention (a lossy frame has Im(N) >= 0)."
        )
    return n


def _require_poisson_ratio(value: float) -> float:
    """Validate a Poisson coefficient of an isotropic frame."""
    nu = float(value)
    if not -1.0 < nu < 0.5:
        raise ValueError("'poisson_ratio' must satisfy -1 < nu < 0,5.")
    return nu


def frame_bulk_modulus(shear_modulus: complex, poisson_ratio: float) -> complex:
    """Bulk modulus ``Kb`` of the frame in vacuum from ``N`` and ``nu``.

    ``Kb = 2 N (nu + 1) / (3 (1 - 2 nu))`` (Allard & Atalla 2e, Eq. (6.29),
    printed p. 116). ``Kb`` is the quantity the jacketed "gedanken experiment"
    of Eq. (6.7) measures, and the one the limp-frame rules of thumb of
    :func:`~phonometry.materials.porous_absorber.limp_frame_applicable` compare
    with the bulk modulus of the pore fluid.

    :param shear_modulus: Complex shear modulus ``N`` of the frame, in Pa
        (``Im(N) >= 0`` for a lossy frame in the ``e^{+j w t}`` convention).
    :param poisson_ratio: Poisson coefficient ``nu`` of the frame
        (``-1 < nu < 0,5``).
    :return: The complex bulk modulus ``Kb`` of the frame in vacuum, in Pa.
    :raises ValueError: for a non-positive or non-finite ``N``, a negative
        ``Im(N)`` or a ``nu`` outside ``(-1, 0,5)``.
    """
    n = _require_shear_modulus(shear_modulus, "shear_modulus")
    nu = _require_poisson_ratio(poisson_ratio)
    return 2.0 * n * (nu + 1.0) / (3.0 * (1.0 - 2.0 * nu))


def frame_elastic_coefficient(
    shear_modulus: complex, poisson_ratio: float
) -> complex:
    """Longitudinal elastic coefficient ``Kc`` of the frame in vacuum.

    ``Kc = lambda + 2 mu = Kb + 4 N / 3 = 2 (1 - nu) N / (1 - 2 nu)``
    (Allard & Atalla 2e, Eqs. (1.76) and (6.111), printed pp. 12 and 130).
    A compressional wave in the frame *in vacuum* travels at
    ``sqrt(Kc / rho1)``, which is what sets the ``lambda/4`` frame resonance of
    :func:`frame_quarter_wave_resonance`.

    :param shear_modulus: Complex shear modulus ``N`` of the frame, in Pa.
    :param poisson_ratio: Poisson coefficient ``nu`` of the frame.
    :return: The complex elastic coefficient ``Kc``, in Pa.
    :raises ValueError: as :func:`frame_bulk_modulus`.
    """
    n = _require_shear_modulus(shear_modulus, "shear_modulus")
    nu = _require_poisson_ratio(poisson_ratio)
    return 2.0 * (1.0 - nu) * n / (1.0 - 2.0 * nu)


def frame_quarter_wave_resonance(
    thickness: float,
    *,
    shear_modulus: complex,
    poisson_ratio: float,
    frame_density: float,
) -> float:
    """Quarter-wavelength resonance of the frame-borne wave (closed form).

    A porous layer glued to a rigid wall holds the frame still at the wall and
    free at the front face, so the frame-borne compressional wave resonates
    where ``l Re(delta_b) = pi/2`` (Allard & Atalla 2e, Eq. (6.109), printed
    p. 129). Since ``delta_b`` stays close to the frame-in-vacuum wavenumber
    ``omega sqrt(rho1 / Kc)`` of Eq. (6.88), the resonance sits at

    ``fr = (1 / 4 l) sqrt(Re(Kc) / rho1)``

    (Eq. (6.110)). This is the frequency at which the peak that no
    equivalent-fluid model can produce appears in the surface impedance of
    :func:`biot_surface_impedance`; Eq. (6.110) is an approximation to it
    (``delta_b`` is not exactly the in-vacuum wavenumber), so the peak lands a
    few per cent above ``fr``.

    :param thickness: Layer thickness ``l``, in metres (> 0).
    :param shear_modulus: Complex shear modulus ``N`` of the frame, in Pa.
    :param poisson_ratio: Poisson coefficient ``nu`` of the frame.
    :param frame_density: Bulk density of the frame ``rho1``, in kg/m3 (> 0):
        the mass of solid per unit volume of material.
    :return: The resonance frequency ``fr``, in hertz.
    :raises ValueError: for a non-positive thickness or frame density, or an
        invalid ``N`` / ``nu``.
    """
    length = require_positive(thickness, "thickness")
    rho1 = require_positive(frame_density, "frame_density")
    k_c = frame_elastic_coefficient(shear_modulus, poisson_ratio)
    return float(np.sqrt(k_c.real / rho1) / (4.0 * length))


@dataclass(frozen=True)
class BiotWavesResult:
    """The three Biot waves of an isotropic air-saturated porous material.

    All arrays share the shape of ``frequency``. ``compressional_wavenumber_1``
    and ``compressional_wavenumber_2`` are ``delta1`` and ``delta2`` of
    Eqs. (6.67)-(6.68) (the branch with ``-sqrt(Delta)`` first, as printed,
    with ``sqrt(Delta)`` taken on the root with non-positive real part so that
    the numbering matches the book's own example), ``shear_wavenumber`` is
    ``delta3`` of Eq. (6.83), all in rad/m and all taken on the root with
    non-negative real part. ``velocity_ratio_1``,
    ``velocity_ratio_2`` and ``velocity_ratio_3`` are the ratios ``mu`` of the
    fluid displacement over the frame displacement (Eqs. (6.71) and (6.84)).
    ``elastic_p``, ``elastic_q`` and ``elastic_r`` are the Biot elastic
    coefficients and ``density_11``, ``density_12``, ``density_22`` the
    modified densities of Eq. (6.56).

    The **airborne** wave is the one whose ``|mu|`` is the larger (the pore
    fluid moves far more than the frame); the **frame-borne** wave is the
    other. Which of ``delta1`` / ``delta2`` plays which role swaps with
    frequency, so use the ``airborne_*`` and ``frame_borne_*`` properties
    rather than the numbered ones.
    """

    frequency: Real
    porosity: float
    tortuosity: float
    frame_density: float
    shear_modulus: complex
    poisson_ratio: float
    elastic_p: Complex
    elastic_q: Complex
    elastic_r: Complex
    density_11: Complex
    density_12: Complex
    density_22: Complex
    compressional_wavenumber_1: Complex
    compressional_wavenumber_2: Complex
    shear_wavenumber: Complex
    velocity_ratio_1: Complex
    velocity_ratio_2: Complex
    velocity_ratio_3: Complex

    @property
    def airborne_is_second(self) -> NDArray[np.bool_]:
        """Whether the airborne wave is ``(delta2, mu2)`` at each frequency."""
        return np.asarray(
            np.abs(self.velocity_ratio_2) >= np.abs(self.velocity_ratio_1),
            dtype=np.bool_,
        )

    @property
    def airborne_wavenumber(self) -> Complex:
        """Wavenumber ``delta_a`` of the airborne compressional wave, rad/m."""
        return np.asarray(
            np.where(
                self.airborne_is_second,
                self.compressional_wavenumber_2,
                self.compressional_wavenumber_1,
            ),
            dtype=np.complex128,
        )

    @property
    def frame_borne_wavenumber(self) -> Complex:
        """Wavenumber ``delta_b`` of the frame-borne wave, in rad/m."""
        return np.asarray(
            np.where(
                self.airborne_is_second,
                self.compressional_wavenumber_1,
                self.compressional_wavenumber_2,
            ),
            dtype=np.complex128,
        )

    @property
    def airborne_velocity_ratio(self) -> Complex:
        """Fluid-over-frame velocity ratio ``mu_a`` of the airborne wave."""
        return np.asarray(
            np.where(
                self.airborne_is_second,
                self.velocity_ratio_2,
                self.velocity_ratio_1,
            ),
            dtype=np.complex128,
        )

    @property
    def frame_borne_velocity_ratio(self) -> Complex:
        """Fluid-over-frame velocity ratio ``mu_b`` of the frame-borne wave."""
        return np.asarray(
            np.where(
                self.airborne_is_second,
                self.velocity_ratio_1,
                self.velocity_ratio_2,
            ),
            dtype=np.complex128,
        )

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the three Biot wavenumbers against frequency.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.materials import plot_biot_waves

        check_language(language)
        return plot_biot_waves(self, ax=ax, language=language, **kwargs)


def _positive_real_root(value: Complex) -> Complex:
    """Square root on the branch with a non-negative real part.

    Allard & Atalla state the convention explicitly for the ``x3`` components
    of the wave-number vectors (printed p. 247, below Eq. (11.21)): "The square
    root symbol represents the root which yields a positive real part."
    """
    root = np.sqrt(np.asarray(value, dtype=np.complex128))
    return np.asarray(np.where(root.real < 0.0, -root, root), dtype=np.complex128)


def biot_waves(
    medium: PorousMediumResult,
    *,
    porosity: float,
    tortuosity: float,
    frame_density: float,
    shear_modulus: complex,
    poisson_ratio: float = 0.0,
) -> BiotWavesResult:
    """The two compressional waves and the shear wave of a Biot layer.

    *medium* supplies the visco-thermal description of the pore fluid, through
    the two identities of Allard & Atalla printed p. 253,
    ``rho22 = phi^2 rho_eq`` and ``R = phi^2 K_eq``, where ``rho_eq`` and
    ``K_eq`` are the surface-normalised effective density and bulk modulus that
    every model in :mod:`~phonometry.materials.porous_absorber` returns. Pass
    the rigid-frame :func:`~phonometry.materials.porous_absorber.johnson_champoux_allard`
    result: the frame motion is the Biot model's business, not the equivalent
    fluid's, so the *rigid-frame* effective density is the correct input (a
    limp-corrected one would count the frame inertia twice).

    From it, with the inertial coupling ``rho_a = phi rho0 (a_inf - 1)``
    (Eq. (6.44)):

    * ``rho22 = phi rho0 + rho_a - j sigma phi^2 G(w)/w``, ``rho12 = -rho_a +
      j sigma phi^2 G(w)/w`` and ``rho11 = rho1 + rho_a - j sigma phi^2 G(w)/w``
      (Eq. (6.56), printed p. 120), the visco-inertial term being recovered as
      ``phi rho0 a_inf - rho22``;
    * ``Kf = phi K_eq``, then ``R = phi Kf``, ``Q = (1 - phi) Kf`` and
      ``P = 4N/3 + Kb + (1 - phi)^2 Kf / phi`` (Eqs. (6.26)-(6.28));
    * ``delta1^2`` and ``delta2^2`` from Eqs. (6.67)-(6.69), ``delta3^2`` from
      Eq. (6.83), ``mu1``, ``mu2`` from Eq. (6.71) and ``mu3`` from Eq. (6.84).

    :param medium: Rigid-frame :class:`~phonometry.materials.porous_absorber.PorousMediumResult`
        for the pore fluid, evaluated on the frequency vector of interest.
    :param porosity: Open porosity ``phi`` (0 < phi <= 1).
    :param tortuosity: High-frequency tortuosity ``a_inf`` (>= 1).
    :param frame_density: Bulk density of the frame ``rho1``, in kg/m3 (> 0).
    :param shear_modulus: Complex shear modulus ``N`` of the frame, in Pa
        (``Im(N) >= 0``; a structural loss factor ``eta`` gives
        ``N = N'(1 + j eta)``).
    :param poisson_ratio: Poisson coefficient ``nu`` of the frame (Default: 0,
        the value Allard & Atalla use for their glass wool).
    :return: A :class:`BiotWavesResult`.
    :raises ValueError: for an out-of-range porosity, tortuosity, frame
        density, shear modulus or Poisson coefficient.
    """
    phi = require_positive(porosity, "porosity")
    if phi > 1.0:
        raise ValueError(_POROSITY_MESSAGE)
    alpha_inf = require_positive(tortuosity, "tortuosity")
    if alpha_inf < 1.0:
        raise ValueError("'tortuosity' must be >= 1.")
    rho1 = require_positive(frame_density, "frame_density")
    n_mod = _require_shear_modulus(shear_modulus, "shear_modulus")
    nu = _require_poisson_ratio(poisson_ratio)

    f = np.asarray(medium.frequency, dtype=np.float64)
    omega = 2.0 * np.pi * f
    rho0 = medium.air_density
    rho_eq = np.asarray(medium.effective_density, dtype=np.complex128)
    k_eq = np.asarray(medium.bulk_modulus, dtype=np.complex128)

    # Modified densities, A&A Eq. (6.56). The visco-inertial term
    # b = j sigma phi^2 G(w)/w is read back from rho22 = phi^2 rho_eq.
    rho_a = phi * rho0 * (alpha_inf - 1.0)
    rho22 = phi**2 * rho_eq
    coupling = phi * rho0 * alpha_inf - rho22
    rho12 = coupling - rho_a
    rho11 = rho1 + rho_a - coupling

    # Elastic coefficients, A&A Eqs. (6.26)-(6.29) with Ks -> inf.
    k_f = phi * k_eq
    k_b = frame_bulk_modulus(n_mod, nu)
    r_coef = phi * k_f
    q_coef = (1.0 - phi) * k_f
    p_coef = 4.0 * n_mod / 3.0 + k_b + (1.0 - phi) ** 2 * k_f / phi

    # The two compressional waves, A&A Eqs. (6.67)-(6.69). The book does not
    # state the branch of sqrt(Delta); the root with non-positive real part is
    # the one that reproduces its printed identification of the airborne wave
    # with (delta1, mu1) below 495 Hz and with (delta2, mu2) above it for the
    # Table 6.1 glass wool (printed p. 124). Only the *numbering* of the two
    # roots depends on this choice; every predicted quantity is invariant.
    trace = p_coef * rho22 + r_coef * rho11 - 2.0 * q_coef * rho12
    det_m = p_coef * r_coef - q_coef**2
    discriminant = -np.sqrt(
        trace**2 - 4.0 * det_m * (rho11 * rho22 - rho12**2)
    )
    delta1_sq = omega**2 * (trace - discriminant) / (2.0 * det_m)
    delta2_sq = omega**2 * (trace + discriminant) / (2.0 * det_m)
    # The shear wave, A&A Eq. (6.83).
    delta3_sq = omega**2 * (rho11 * rho22 - rho12**2) / (n_mod * rho22)

    # Velocity ratios, A&A Eqs. (6.71) and (6.84).
    mu1 = (p_coef * delta1_sq - omega**2 * rho11) / (
        omega**2 * rho12 - q_coef * delta1_sq
    )
    mu2 = (p_coef * delta2_sq - omega**2 * rho11) / (
        omega**2 * rho12 - q_coef * delta2_sq
    )
    mu3 = -rho12 / rho22

    return BiotWavesResult(
        frequency=f,
        porosity=phi,
        tortuosity=alpha_inf,
        frame_density=rho1,
        shear_modulus=n_mod,
        poisson_ratio=nu,
        elastic_p=np.asarray(p_coef, dtype=np.complex128),
        elastic_q=np.asarray(q_coef, dtype=np.complex128),
        elastic_r=np.asarray(r_coef, dtype=np.complex128),
        density_11=np.asarray(rho11, dtype=np.complex128),
        density_12=np.asarray(rho12, dtype=np.complex128),
        density_22=np.asarray(rho22, dtype=np.complex128),
        compressional_wavenumber_1=_positive_real_root(delta1_sq),
        compressional_wavenumber_2=_positive_real_root(delta2_sq),
        shear_wavenumber=_positive_real_root(delta3_sq),
        velocity_ratio_1=np.asarray(mu1, dtype=np.complex128),
        velocity_ratio_2=np.asarray(mu2, dtype=np.complex128),
        velocity_ratio_3=np.asarray(mu3, dtype=np.complex128),
    )


def biot_surface_impedance(waves: BiotWavesResult, thickness: float) -> Complex:
    """Surface impedance of a hard-backed Biot layer at normal incidence.

    The closed form of Allard & Atalla 2e, Eqs. (6.107)-(6.108) (printed
    p. 128), obtained by writing the four compressional-wave amplitudes against
    the three boundary conditions of a layer *glued* to an impervious rigid
    wall (zero frame and fluid velocity at the wall, Eq. (6.95); continuity of
    pressure and of the total normal stress at the free face, Eqs. (6.97)-(6.99);
    conservation of the volume flow, Eq. (6.100)):

    ``Z = -j (Z1s Z2f mu2 - Z2s Z1f mu1) / D``

    ``D = (1 - phi + phi mu2)[Z1s - (1 - phi) Z1f mu1] tan(delta2 l)
    + (1 - phi + phi mu1)[(1 - phi) Z2f mu2 - Z2s] tan(delta1 l)``

    with the four characteristic impedances of Eqs. (6.74)-(6.77),
    ``Zif = (R + Q/mui) deltai / (phi w)`` and ``Zis = (P + Q mui) deltai / w``.

    This is an independent derivation of the same physics as the chapter 11
    transfer-matrix assembly reached through
    :class:`~phonometry.materials.porous_absorber.PoroelasticLayer`; the two
    agree to machine precision, which is one of the anchors of this module.
    Unlike the transfer matrix, it is restricted to normal incidence, to a
    single layer and to a glued rigid backing.

    :param waves: The :class:`BiotWavesResult` of the material.
    :param thickness: Layer thickness ``l``, in metres (> 0).
    :return: The complex surface impedance ``Z``, in Pa s/m.
    :raises ValueError: for a non-positive thickness.
    """
    length = require_positive(thickness, "thickness")
    phi = waves.porosity
    omega = 2.0 * np.pi * np.asarray(waves.frequency, dtype=np.float64)
    d1 = waves.compressional_wavenumber_1
    d2 = waves.compressional_wavenumber_2
    mu1 = waves.velocity_ratio_1
    mu2 = waves.velocity_ratio_2
    p_coef, q_coef, r_coef = waves.elastic_p, waves.elastic_q, waves.elastic_r

    z1f = (r_coef + q_coef / mu1) * d1 / (phi * omega)
    z2f = (r_coef + q_coef / mu2) * d2 / (phi * omega)
    z1s = (p_coef + q_coef * mu1) * d1 / omega
    z2s = (p_coef + q_coef * mu2) * d2 / omega
    denominator = (1.0 - phi + phi * mu2) * (
        z1s - (1.0 - phi) * z1f * mu1
    ) * np.tan(d2 * length) + (1.0 - phi + phi * mu1) * (
        (1.0 - phi) * z2f * mu2 - z2s
    ) * np.tan(d1 * length)
    return np.asarray(
        -1j * (z1s * z2f * mu2 - z2s * z1f * mu1) / denominator,
        dtype=np.complex128,
    )


def _gamma(
    waves: BiotWavesResult, x3: float, transverse_wavenumber: Complex
) -> Complex:
    """The matrix ``[Gamma(x3)]`` of Allard & Atalla Table 11.1.

    Columns are the coefficients of the amplitude vector
    ``[(A1 + A1'), (A1 - A1'), (A2 + A2'), (A2 - A2'), (A3 + A3'), (A3 - A3')]``
    (Eq. (11.30)); rows are the six field variables of Eq. (11.26),
    ``[v1s, v3s, v3f, s33s, s13s, s33f]``. Returned with shape
    ``(len(frequency), 6, 6)``.

    Table 11.1 prints ``k_{i3}`` in its first two columns where the ``mu1``,
    ``D1`` and ``E1`` of the same columns make ``k_{13}`` the only consistent
    reading; that reading is implemented here.
    """
    omega = 2.0 * np.pi * np.asarray(waves.frequency, dtype=np.float64)
    k_t = np.asarray(transverse_wavenumber, dtype=np.complex128)
    n_mod = waves.shear_modulus
    p_coef, q_coef, r_coef = waves.elastic_p, waves.elastic_q, waves.elastic_r
    mu1, mu2, mu3 = (
        waves.velocity_ratio_1,
        waves.velocity_ratio_2,
        waves.velocity_ratio_3,
    )
    kt2 = k_t**2
    # x3 components of the three wave-number vectors, A&A Eq. (11.21).
    k13 = _positive_real_root(waves.compressional_wavenumber_1**2 - kt2)
    k23 = _positive_real_root(waves.compressional_wavenumber_2**2 - kt2)
    k33 = _positive_real_root(waves.shear_wavenumber**2 - kt2)

    # A&A Eqs. (11.49)-(11.50).
    d1 = (p_coef + q_coef * mu1) * (kt2 + k13**2) - 2.0 * n_mod * kt2
    d2 = (p_coef + q_coef * mu2) * (kt2 + k23**2) - 2.0 * n_mod * kt2
    e1 = (r_coef * mu1 + q_coef) * (kt2 + k13**2)
    e2 = (r_coef * mu2 + q_coef) * (kt2 + k23**2)

    c1, s1 = np.cos(k13 * x3), np.sin(k13 * x3)
    c2, s2 = np.cos(k23 * x3), np.sin(k23 * x3)
    c3, s3 = np.cos(k33 * x3), np.sin(k33 * x3)
    zero = np.zeros_like(c1)
    shear = n_mod * (k33**2 - kt2)
    rows = (
        (
            omega * k_t * c1, -1j * omega * k_t * s1,
            omega * k_t * c2, -1j * omega * k_t * s2,
            1j * omega * k33 * s3, -omega * k33 * c3,
        ),
        (
            -1j * omega * k13 * s1, omega * k13 * c1,
            -1j * omega * k23 * s2, omega * k23 * c2,
            omega * k_t * c3, -1j * omega * k_t * s3,
        ),
        (
            -1j * omega * k13 * mu1 * s1, omega * k13 * mu1 * c1,
            -1j * omega * k23 * mu2 * s2, omega * k23 * mu2 * c2,
            omega * k_t * mu3 * c3, -1j * omega * k_t * mu3 * s3,
        ),
        (
            -d1 * c1, 1j * d1 * s1,
            -d2 * c2, 1j * d2 * s2,
            2j * n_mod * k33 * k_t * s3, -2.0 * n_mod * k33 * k_t * c3,
        ),
        (
            2j * n_mod * k_t * k13 * s1, -2.0 * n_mod * k_t * k13 * c1,
            2j * n_mod * k_t * k23 * s2, -2.0 * n_mod * k_t * k23 * c2,
            shear * c3, -1j * shear * s3,
        ),
        (-e1 * c1, 1j * e1 * s1, -e2 * c2, 1j * e2 * s2, zero, zero),
    )
    gamma = np.array(
        [[np.broadcast_to(entry, omega.shape) for entry in row] for row in rows],
        dtype=np.complex128,
    )
    return np.asarray(np.moveaxis(gamma, -1, 0), dtype=np.complex128)


def poroelastic_transfer_matrix(
    waves: BiotWavesResult,
    thickness: float,
    *,
    transverse_wavenumber: ArrayLike = 0.0,
) -> Complex:
    """The 6x6 transfer matrix ``[T p]`` of a Biot poroelastic layer.

    ``[T p] = [Gamma(-h)][Gamma(0)]^-1`` (Allard & Atalla 2e, Eq. (11.34),
    printed p. 249), relating the field vector
    ``[v1s, v3s, v3f, s33s, s13s, s33f]`` (Eq. (11.26)) just inside the front
    face of the layer to the same vector just inside its back face,
    ``V(M) = [T p] V(M')``. The wave-amplitude matrix ``[Gamma]`` of Table 11.1
    behind it is built by :func:`_gamma`.

    The layer solver of
    :func:`~phonometry.materials.porous_absorber.layered_absorber` does *not*
    use this matrix: it assembles the wave amplitudes directly, which avoids
    inverting ``[Gamma(0)]`` and is far better conditioned for a very soft or a
    very thick frame. The matrix is exposed because it is the object chapter 11
    is written around and the one an external multilayer chain expects.

    :param waves: The :class:`BiotWavesResult` of the material.
    :param thickness: Layer thickness ``h``, in metres (> 0).
    :param transverse_wavenumber: In-plane wavenumber ``kt = k sin(theta)``,
        in rad/m (Default: 0, normal incidence). Scalar or one value per
        frequency.
    :return: The transfer matrix with shape ``(len(frequency), 6, 6)``.
    :raises ValueError: for a non-positive thickness.
    """
    h = require_positive(thickness, "thickness")
    k_t = np.asarray(transverse_wavenumber, dtype=np.complex128)
    front = _gamma(waves, -h, k_t)
    back = _gamma(waves, 0.0, k_t)
    return np.asarray(front @ np.linalg.inv(back), dtype=np.complex128)


# ---------------------------------------------------------------------------
# Global-matrix assembly (Allard & Atalla Sects. 11.4-11.6)
# ---------------------------------------------------------------------------
def _porous_fluid_matrices(porosity: float) -> tuple[Complex, Complex]:
    """``[Ipf]`` and ``[Jpf]`` of A&A Eq. (11.73), printed p. 259.

    Continuity at a porous-fluid boundary (Eq. (11.72)): volume-flow
    conservation, ``s33s = -(1 - phi) p``, ``s13s = 0`` and ``s33f = -phi p``.
    """
    phi = porosity
    i_pf = np.zeros((4, 6), dtype=np.complex128)
    i_pf[0, 1] = 1.0 - phi
    i_pf[0, 2] = phi
    i_pf[1, 3] = 1.0
    i_pf[2, 4] = 1.0
    i_pf[3, 5] = 1.0
    j_pf = np.zeros((4, 2), dtype=np.complex128)
    j_pf[0, 1] = -1.0
    j_pf[1, 0] = 1.0 - phi
    j_pf[3, 0] = phi
    return i_pf, j_pf


def _porous_porous_matrix(left: float, right: float) -> Complex:
    """``[Ipp]`` of A&A Eq. (11.67), printed p. 258, for bonded frames."""
    i_pp = np.eye(6, dtype=np.complex128)
    i_pp[2, 1] = 1.0 - right / left
    i_pp[2, 2] = right / left
    i_pp[3, 5] = 1.0 - left / right
    i_pp[5, 5] = left / right
    return i_pp


@dataclass(frozen=True)
class _Block:
    """One solved layer block of the global assembly.

    ``kind`` is ``"fluid"`` (two field variables ``[p, v3]``) or
    ``"poroelastic"`` (six field variables). ``unknowns`` is the number of
    unknowns the block contributes: the back-face field vector for a fluid
    block, the six wave amplitudes for a poroelastic one. ``front`` and
    ``back`` map those unknowns onto the field vector at each face.
    """

    kind: str
    front: Complex
    back: Complex
    porosity: float

    @property
    def unknowns(self) -> int:
        return int(self.front.shape[-1])


def _fluid_block(transfer_matrix: Complex) -> _Block:
    """A two-variable block from its ``(nf, 2, 2)`` chain matrix."""
    identity = np.broadcast_to(
        np.eye(2, dtype=np.complex128), transfer_matrix.shape
    )
    return _Block("fluid", transfer_matrix, np.asarray(identity), 1.0)


def _poroelastic_block(
    waves: BiotWavesResult, thickness: float, transverse_wavenumber: ArrayLike
) -> _Block:
    """A six-variable block, parameterised by its wave amplitudes."""
    h = require_positive(thickness, "thickness")
    k_t = np.asarray(transverse_wavenumber, dtype=np.complex128)
    return _Block(
        "poroelastic",
        _gamma(waves, -h, k_t),
        _gamma(waves, 0.0, k_t),
        waves.porosity,
    )


def _interface(left: _Block | None, right: _Block) -> tuple[Complex, Complex]:
    """Coupling matrices ``([I], [J])`` between two adjacent media.

    *left* is ``None`` for the free air on the excitation side, which is a
    fluid. A&A Sect. 11.4: two media of the same nature couple through the
    identity (fluid) or through ``[Ipp]`` (porous, Eq. (11.67)); a porous and a
    fluid medium couple through ``[Ipf]``/``[Jpf]`` (Eq. (11.73)), with the two
    matrices interchanged for a fluid-porous boundary.
    """
    left_kind = "fluid" if left is None else left.kind
    if left_kind == "fluid" and right.kind == "fluid":
        return np.eye(2, dtype=np.complex128), -np.eye(2, dtype=np.complex128)
    if left_kind == "poroelastic" and right.kind == "poroelastic":
        left_phi = 1.0 if left is None else left.porosity
        return (
            np.eye(6, dtype=np.complex128),
            -_porous_porous_matrix(left_phi, right.porosity),
        )
    if left_kind == "poroelastic":
        i_pf, j_pf = _porous_fluid_matrices(left.porosity)  # type: ignore[union-attr]
        return i_pf, j_pf
    i_pf, j_pf = _porous_fluid_matrices(right.porosity)
    return j_pf, i_pf


def _wall_matrix(block: _Block) -> Complex:
    """``[Y]`` of A&A Eq. (11.81): zero velocity against a hard wall."""
    if block.kind == "fluid":
        return np.array([[0.0, 1.0]], dtype=np.complex128)
    return np.eye(6, dtype=np.complex128)[:3]


def _identity_map(n_freq: int) -> Complex:
    """The 2x2 identity broadcast over *n_freq* frequencies."""
    return np.asarray(
        np.broadcast_to(np.eye(2, dtype=np.complex128), (n_freq, 2, 2)),
        dtype=np.complex128,
    )


def _stack_surface_impedance(
    blocks: list[_Block], termination: Complex | None
) -> Complex:
    """Surface impedance of a stratified medium by the global-matrix method.

    Assembles the linear system ``[D] V = 0`` of Allard & Atalla Eqs. (11.78)
    to (11.85) and solves it with the incident pressure fixed to one
    (Sect. 11.6.1, Eq. (11.97)), so that ``Zs = 1 / V1(1)``. Unlike the printed
    determinant form Eq. (11.88) this needs a single linear solve per
    frequency, and unlike the transfer-matrix form of
    :func:`poroelastic_transfer_matrix` it never inverts ``[Gamma(0)]``.

    :param blocks: The layer blocks, from the incidence side inwards.
    :param termination: ``None`` for a hard wall (Eq. (11.81)), otherwise the
        impedance ``p / v3`` seen at the back face, one value per frequency
        (Eq. (11.84) with ``ZB / cos(theta)``).
    :return: The complex surface impedance per frequency, in Pa s/m.
    """
    n_freq = int(blocks[0].front.shape[0])
    last = blocks[-1]
    # A semi-infinite fluid behind a poroelastic layer needs its own two
    # variables (A&A Eq. (11.85)); behind a fluid layer the impedance closes
    # the system directly.
    trailing_fluid = termination is not None and last.kind != "fluid"
    sizes = [2, *(block.unknowns for block in blocks)]
    if trailing_fluid:
        sizes.append(2)
    offsets = [int(value) for value in np.cumsum([0, *sizes])]

    couplings: list[tuple[int, int, Complex, Complex]] = []
    previous: _Block | None = None
    for index, block in enumerate(blocks):
        i_mat, j_mat = _interface(previous, block)
        couplings.append((index, index + 1, i_mat, j_mat))
        previous = block
    if trailing_fluid:
        i_pf, j_pf = _porous_fluid_matrices(last.porosity)
        couplings.append((len(blocks), len(blocks) + 1, i_pf, j_pf))

    equations = sum(int(i_mat.shape[0]) for _, _, i_mat, _ in couplings)
    closing = _wall_matrix(last).shape[0] if termination is None else 1
    equations += int(closing)

    matrix = np.zeros((n_freq, equations, offsets[-1]), dtype=np.complex128)
    row = 0
    for left_slot, right_slot, i_mat, j_mat in couplings:
        height = int(i_mat.shape[0])
        left_map = (
            _identity_map(n_freq)
            if left_slot == 0
            else blocks[left_slot - 1].back
        )
        matrix[:, row : row + height, offsets[left_slot] : offsets[left_slot + 1]] = (
            i_mat @ left_map
        )
        if right_slot <= len(blocks):
            right_map = blocks[right_slot - 1].front
            block_rows = j_mat @ right_map
        else:
            block_rows = np.broadcast_to(j_mat, (n_freq, *j_mat.shape))
        matrix[
            :, row : row + height, offsets[right_slot] : offsets[right_slot + 1]
        ] = block_rows
        row += height

    if termination is None:
        wall = _wall_matrix(last)
        height = int(wall.shape[0])
        matrix[:, row : row + height, offsets[-2] : offsets[-1]] = wall @ last.back
    else:
        # p(B) = Z_L v3(B), A&A Eq. (11.84).
        z_l = np.asarray(termination, dtype=np.complex128)
        closer = np.zeros((n_freq, 1, 2), dtype=np.complex128)
        closer[:, 0, 0] = -1.0
        closer[:, 0, 1] = z_l
        tail = _identity_map(n_freq) if trailing_fluid else last.back
        matrix[:, row : row + 1, offsets[-2] : offsets[-1]] = closer @ tail

    lhs = matrix[:, :, 1:]
    rhs = -matrix[:, :, 0]
    # Row-equilibrate: the six-variable rows mix velocities with stresses,
    # whose magnitudes differ by many orders of magnitude.
    scale = np.max(np.abs(lhs), axis=2)
    scale = np.where(scale > 0.0, scale, 1.0)
    solution = np.linalg.solve(
        lhs / scale[:, :, None], (rhs / scale)[:, :, None]
    )
    velocity = solution[:, 0, 0]
    finite = velocity != 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(
            np.where(finite, 1.0 / np.where(finite, velocity, 1.0), np.inf + 0j),
            dtype=np.complex128,
        )

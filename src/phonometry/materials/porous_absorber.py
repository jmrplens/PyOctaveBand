#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Porous-material models and multilayer absorber prediction.

Three complementary building blocks, all in the ``e^{+j w t}`` time
convention with the forward wave carried by ``e^{-j k x}`` (so a passive
medium has ``Im(k) < 0``):

* **Equivalent-fluid models** for the characteristic impedance ``Zc`` and the
  complex wavenumber ``k`` of a rigid-frame porous material:

  - the one-parameter **Delany-Bazley** power law in the absorber variable
    ``X = rho0 f / sigma`` (Mechel, *Formulas of Acoustics* 2e, Sect. G.11
    Eqs. (1)-(2); Bies, Hansen & Howard, *Engineering Noise Control* 5e,
    Appendix D Eqs. (D.22)-(D.23) and Table D.1; Hopkins, *Sound Insulation*,
    Eqs. (1.171)-(1.174)), stated valid for ``0.01 < X < 1.0`` and porosity
    close to one. Table D.1 also provides coefficient sets fitted to
    polyester (Garai & Pompoli 2005) and to foams (Dunn & Davern 1986,
    Wu 1988), exposed here as presets.
  - the **Miki** modification, regressed on the same Delany-Bazley data under
    a positive-real (passivity) constraint so the model stays well behaved
    below the fit range (Miki 1990, *J. Acoust. Soc. Jpn (E)* 11(1),
    Eqs. (30)-(34), in the variable ``f / sigma``).
  - the five-parameter **Johnson-Champoux-Allard (JCA)** semi-phenomenological
    model with flow resistivity, porosity, tortuosity and the viscous/thermal
    characteristic lengths (Cox & D'Antonio, *Acoustic Absorbers and
    Diffusers* 3e, Eqs. (6.19)-(6.25); Attenborough & Van Renterghem,
    *Predicting Outdoor Sound* 2e, Eqs. (5.13)-(5.14)). The returned
    equivalent-fluid density and bulk modulus are the surface-normalised
    quantities (they absorb the porosity), so ``Zc = sqrt(rho_e K_e)`` and
    ``k = w sqrt(rho_e / K_e)`` hold for every model.
  - the **limp-frame** correction of any of the three rigid-frame models
    (Allard & Atalla, *Propagation of Sound in Porous Media* 2e, Sect. 11.3.4,
    Eqs. (11.53)-(11.55), printed pp. 251-253): a light frame is dragged along
    by the pore fluid, so its inertia has to be carried by the equivalent
    fluid. Only the effective density changes; the bulk modulus is the
    rigid-frame one. See :func:`limp_frame` and
    :func:`decoupling_frequency`.

* **Transfer-matrix multilayer prediction**: each fluid layer contributes
  ``[[cos(kx d), j Zx sin(kx d)], [j sin(kx d)/Zx, cos(kx d)]]`` with the
  in-depth wavenumber ``kx = sqrt(k^2 - k0^2 sin^2 theta)`` from Snell's law
  and ``Zx = Zc k / kx`` (Cox & D'Antonio Eqs. (2.29)-(2.32); Bies
  Eq. (D.83); equivalent to the layer-recursion of Bies Eq. (D.95) and
  Mechel Sect. D.4). Thin resonant sheets (perforated plate, microperforated
  plate, limp membrane) enter as series transfer impedances
  ``[[1, z],[0, 1]]``. The stack is closed by a rigid wall, by free air or
  by an arbitrary termination impedance, giving the surface impedance, the
  oblique reflection factor and ``alpha(theta)``. This same layer transfer
  matrix underlies the critically-coupled perfect-absorber designs of Jiménez,
  Groby, Pagneux & Romero-García (2017, *Applied Sciences* 7(6), 618,
  doi:10.3390/app7060618) and, for a rigidly-backed high-porosity layer,
  Jiménez, Romero-García & Groby (2018, *Acta Acustica united with Acustica*
  104(3), 396-409, doi:10.3813/AAA.919183), where the critical-coupling
  condition on the surface impedance yields total single-frequency absorption.

* **Resonant sheets and random incidence**: the perforated-plate impedance
  uses the end-corrected air-plug mass and the visco-thermal surface
  resistance (Cox & D'Antonio Eqs. (7.6)/(7.12)/(7.21), end-correction
  variants of Table 7.1); the microperforated plate follows Maa's exact
  short-tube impedance (Maa 1998, *J. Acoust. Soc. Am.* 104(5), Eq. (2),
  with the Eq. (5) end corrections; reproduced as Cox & D'Antonio
  Eqs. (7.33)-(7.35) and built on the same Bessel kernel as Mechel
  Sect. G.3); the membrane is the limp surface
  mass ``j w m`` (Cox & D'Antonio Eq. (7.14); Bies Eq. (D.96)). The
  random-incidence (Paris) integral follows Mechel Sect. D.5 Eqs. (9)-(10),
  with the closed form for locally reacting surfaces implemented in
  :func:`statistical_absorption` (its maximum over passive impedances is the
  published 0.951).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

from .._internal.types import Real
from .._internal.validation import (
    require_choice,
    require_non_negative,
    require_positive,
    require_positive_array,
)
from .._internal.warnings import PhonometryWarning

if TYPE_CHECKING:
    from matplotlib.axes import Axes

Complex = NDArray[np.complex128]

#: Default speed of sound in air, in m/s (20 degC).
_SPEED_OF_SOUND = 343.0
#: Default air density, in kg/m3 (Bies 5e Appendix D: 1,205 at 20 degC).
_AIR_DENSITY = 1.205
#: Default dynamic viscosity of air, in Pa s (Cox & D'Antonio Eq. (7.13)).
_AIR_VISCOSITY = 1.84e-5
#: Default Prandtl number of air at 20 degC (``eta c_p / kappa``).
_PRANDTL_NUMBER = 0.71
#: Default ratio of specific heats of air.
_HEAT_CAPACITY_RATIO = 1.4
#: Default atmospheric pressure, in Pa.
_ATMOSPHERIC_PRESSURE = 101325.0

#: Shared validation message for fractional open areas.
_OPEN_AREA_MESSAGE = "'open_area' must not exceed 1."
#: Shared validation message for open porosities.
_POROSITY_MESSAGE = "'porosity' must not exceed 1."

#: Delany-Bazley power-law coefficient presets ``(C1..C8)`` from Bies 5e
#: Appendix D, Table D.1: ``Zc = rho c (1 + C1 X^-C2 - j C3 X^-C4)`` and
#: ``k = (w/c)(1 + C5 X^-C6 - j C7 X^-C8)`` with ``X = rho f / sigma``.
DELANY_BAZLEY_COEFFICIENTS: Mapping[str, tuple[float, ...]] = {
    # Rockwool / fibreglass (Delany & Bazley 1970).
    "delany_bazley": (0.0571, 0.754, 0.087, 0.732, 0.0978, 0.700, 0.189, 0.595),
    # Polyester (Garai & Pompoli 2005).
    "garai_pompoli": (0.078, 0.623, 0.074, 0.660, 0.159, 0.571, 0.121, 0.530),
    # Polyurethane foam of low flow resistivity (Dunn & Davern 1986).
    "dunn_davern": (0.114, 0.369, 0.0985, 0.758, 0.168, 0.715, 0.136, 0.491),
    # Porous plastic foams of medium flow resistivity (Wu 1988).
    "wu": (0.212, 0.455, 0.105, 0.607, 0.163, 0.592, 0.188, 0.544),
}

#: Stated validity range of the Delany-Bazley regression in ``X = rho f/sigma``
#: (Hopkins Eq. (1.174); Cox & D'Antonio Sect. 6.5.1).
DELANY_BAZLEY_VALIDITY = (0.01, 1.0)
#: Fit range of the Miki regression in ``f/sigma`` (Miki 1990, Sect. 4.1: the
#: Delany-Bazley data below ``f/sigma = 0.01`` are extrapolation).
MIKI_VALIDITY = (0.01, 1.0)
#: Published upper limits on ``|K_c / K_f|`` (frame-in-vacuum bulk modulus over
#: pore-fluid bulk modulus) below which the limp-frame equivalent fluid of
#: :func:`limp_frame` may be used, from Allard & Atalla 2e printed pp. 253-254:
#: ``"beranek"`` is Beranek's (1947) original 0,05 and ``"doutres"`` the 0,2
#: to which Doutres et al. (2007) relaxed it with their frame structural
#: interaction criterion (which, with ``K_f`` approximated by the isothermal
#: value of air ``P0 = 101,3 kPa``, is the book's "lower than 20 kPa").
LIMP_FRAME_CRITERIA: Mapping[str, float] = {"beranek": 0.05, "doutres": 0.2}

__all__ = [
    "DELANY_BAZLEY_COEFFICIENTS",
    "DELANY_BAZLEY_VALIDITY",
    "LIMP_FRAME_CRITERIA",
    "MIKI_VALIDITY",
    "AirLayer",
    "DiffuseFieldAbsorptionResult",
    "LayeredAbsorberResult",
    "MembraneLayer",
    "MicroperforatedPlateLayer",
    "PerforatedPlateLayer",
    "PoroelasticLayer",
    "PorousAbsorberWarning",
    "PorousLayer",
    "PorousMediumResult",
    "decoupling_frequency",
    "delany_bazley",
    "diffuse_field_absorption",
    "helmholtz_resonance_frequency",
    "johnson_champoux_allard",
    "layered_absorber",
    "limp_frame",
    "limp_frame_applicable",
    "membrane_impedance",
    "membrane_resonance_frequency",
    "microperforated_plate_impedance",
    "miki",
    "perforated_plate_impedance",
    "perforation_end_correction",
    "statistical_absorption",
]


class PorousAbsorberWarning(PhonometryWarning):
    """Advisory for porous-model use outside the published fit range."""


# ---------------------------------------------------------------------------
# Equivalent-fluid models of porous materials
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PorousMediumResult:
    """Equivalent-fluid characterisation of a porous material.

    All arrays share the shape of ``frequency``. ``characteristic_impedance``
    is the complex characteristic impedance ``Zc`` in Pa s/m as seen from the
    material surface, ``wavenumber`` the complex wavenumber ``k`` in rad/m
    (``Im(k) < 0`` for the ``e^{+j w t}`` convention),
    ``effective_density = Zc k / w`` and ``bulk_modulus = Zc w / k`` the
    surface-normalised equivalent-fluid density and bulk modulus, so that
    ``Zc = sqrt(rho_e K_e)`` and ``k = w sqrt(rho_e / K_e)`` for every model.
    """

    frequency: Real
    characteristic_impedance: Complex
    wavenumber: Complex
    effective_density: Complex
    bulk_modulus: Complex
    model: str
    flow_resistivity: float
    speed_of_sound: float
    air_density: float

    @property
    def normalized_impedance(self) -> Complex:
        """Characteristic impedance normalised by ``rho c`` of air."""
        rc = self.air_density * self.speed_of_sound
        return np.asarray(self.characteristic_impedance / rc, dtype=np.complex128)

    @property
    def normalized_wavenumber(self) -> Complex:
        """Wavenumber normalised by the free-air wavenumber ``k0 = w / c``."""
        k0 = 2.0 * np.pi * self.frequency / self.speed_of_sound
        return np.asarray(self.wavenumber / k0, dtype=np.complex128)

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the normalised ``Zc`` and ``k`` components against frequency.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.materials import plot_porous_medium

        check_language(language)
        return plot_porous_medium(self, ax=ax, language=language, **kwargs)


def _medium_from_zc_k(
    f: Real,
    zc: Complex,
    k: Complex,
    *,
    model: str,
    flow_resistivity: float,
    speed_of_sound: float,
    air_density: float,
) -> PorousMediumResult:
    """Package ``(Zc, k)`` into a :class:`PorousMediumResult`."""
    omega = 2.0 * np.pi * f
    return PorousMediumResult(
        frequency=f,
        characteristic_impedance=zc,
        wavenumber=k,
        effective_density=np.asarray(zc * k / omega, dtype=np.complex128),
        bulk_modulus=np.asarray(zc * omega / k, dtype=np.complex128),
        model=model,
        flow_resistivity=flow_resistivity,
        speed_of_sound=speed_of_sound,
        air_density=air_density,
    )


def _warn_fit_range(
    x: Real, limits: tuple[float, float], variable: str, model: str
) -> None:
    """Warn once when *x* leaves the published fit range of *model*."""
    lo, hi = limits
    if bool(np.any(x < lo)) or bool(np.any(x > hi)):
        warnings.warn(
            f"{model}: {variable} outside the published fit range "
            f"[{lo:g}, {hi:g}]; the regression is an extrapolation there.",
            PorousAbsorberWarning,
            stacklevel=3,
        )


def delany_bazley(
    frequency: ArrayLike,
    flow_resistivity: float,
    *,
    coefficients: str | tuple[float, ...] = "delany_bazley",
    speed_of_sound: float = _SPEED_OF_SOUND,
    air_density: float = _AIR_DENSITY,
) -> PorousMediumResult:
    """Delany-Bazley one-parameter porous model (power laws in ``X``).

    ``Zc = rho c (1 + C1 X^-C2 - j C3 X^-C4)`` and
    ``k = (w/c)(1 + C5 X^-C6 - j C7 X^-C8)`` with ``X = rho f / sigma``
    (Mechel 2e Sect. G.11 Eqs. (1)-(2); Bies 5e Eqs. (D.22)-(D.23) with the
    Table D.1 coefficients; Hopkins Eqs. (1.171)-(1.173)). A
    :class:`PorousAbsorberWarning` is raised when any ``X`` leaves the stated
    ``0.01 < X < 1.0`` validity range (Hopkins Eq. (1.174)); the values are
    still returned.

    :param frequency: Frequency vector ``f``, in hertz.
    :param flow_resistivity: Airflow resistivity ``sigma``, in Pa s/m2.
    :param coefficients: Preset name from :data:`DELANY_BAZLEY_COEFFICIENTS`
        (``"delany_bazley"`` rockwool/fibreglass default, ``"garai_pompoli"``
        polyester, ``"dunn_davern"`` / ``"wu"`` foams) or an explicit
        ``(C1..C8)`` tuple.
    :param speed_of_sound: Speed of sound ``c`` in air, in m/s.
    :param air_density: Air density ``rho``, in kg/m3.
    :return: A :class:`PorousMediumResult`.
    """
    f = require_positive_array(frequency, "frequency")
    sigma = require_positive(flow_resistivity, "flow_resistivity")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    rho0 = require_positive(air_density, "air_density")
    if isinstance(coefficients, str):
        try:
            coeffs = DELANY_BAZLEY_COEFFICIENTS[coefficients]
        except KeyError:
            options = ", ".join(sorted(DELANY_BAZLEY_COEFFICIENTS))
            raise ValueError(
                f"unknown coefficient preset {coefficients!r}; "
                f"options: {options}."
            ) from None
        model = f"delany_bazley[{coefficients}]"
    else:
        coeffs = tuple(float(v) for v in coefficients)
        model = "delany_bazley[custom]"
    if len(coeffs) != 8:
        raise ValueError("'coefficients' must provide exactly 8 values C1..C8.")
    c1, c2, c3, c4, c5, c6, c7, c8 = coeffs
    x = np.asarray(rho0 * f / sigma, dtype=np.float64)
    _warn_fit_range(x, DELANY_BAZLEY_VALIDITY, "X = rho f / sigma", "Delany-Bazley")
    zc = rho0 * c0 * (1.0 + c1 * x**-c2 - 1j * c3 * x**-c4)
    k = (2.0 * np.pi * f / c0) * (1.0 + c5 * x**-c6 - 1j * c7 * x**-c8)
    return _medium_from_zc_k(
        f,
        np.asarray(zc, dtype=np.complex128),
        np.asarray(k, dtype=np.complex128),
        model=model,
        flow_resistivity=sigma,
        speed_of_sound=c0,
        air_density=rho0,
    )


def miki(
    frequency: ArrayLike,
    flow_resistivity: float,
    *,
    speed_of_sound: float = _SPEED_OF_SOUND,
    air_density: float = _AIR_DENSITY,
) -> PorousMediumResult:
    """Miki (1990) positive-real modification of the Delany-Bazley model.

    In the variable ``Y = f / sigma`` (Miki 1990, Eqs. (30)-(34)):
    ``Zc = rho c (1 + 0.070 Y^-0.632 - j 0.107 Y^-0.632)`` and, from the
    propagation constant ``gamma = alpha + j beta`` via ``k = beta - j alpha``,
    ``k = (w/c)(1 + 0.109 Y^-0.618 - j 0.160 Y^-0.618)``. The regression was
    constrained to be positive real, so the surface impedance of a
    hard-backed layer keeps a non-negative real part even below the
    Delany-Bazley range; a :class:`PorousAbsorberWarning` still flags
    ``Y`` outside the fit range ``0.01 < f/sigma < 1.0`` (paper Sect. 4.1).

    :param frequency: Frequency vector ``f``, in hertz.
    :param flow_resistivity: Airflow resistivity ``sigma``, in Pa s/m2.
    :param speed_of_sound: Speed of sound ``c`` in air, in m/s.
    :param air_density: Air density ``rho``, in kg/m3.
    :return: A :class:`PorousMediumResult`.
    """
    f = require_positive_array(frequency, "frequency")
    sigma = require_positive(flow_resistivity, "flow_resistivity")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    rho0 = require_positive(air_density, "air_density")
    y = np.asarray(f / sigma, dtype=np.float64)
    _warn_fit_range(y, MIKI_VALIDITY, "f / sigma", "Miki")
    zc = rho0 * c0 * (1.0 + 0.070 * y**-0.632 - 1j * 0.107 * y**-0.632)
    k = (2.0 * np.pi * f / c0) * (1.0 + 0.109 * y**-0.618 - 1j * 0.160 * y**-0.618)
    return _medium_from_zc_k(
        f,
        np.asarray(zc, dtype=np.complex128),
        np.asarray(k, dtype=np.complex128),
        model="miki",
        flow_resistivity=sigma,
        speed_of_sound=c0,
        air_density=rho0,
    )


def johnson_champoux_allard(
    frequency: ArrayLike,
    flow_resistivity: float,
    *,
    porosity: float,
    tortuosity: float,
    viscous_length: float,
    thermal_length: float,
    speed_of_sound: float = _SPEED_OF_SOUND,
    air_density: float = _AIR_DENSITY,
    viscosity: float = _AIR_VISCOSITY,
    prandtl_number: float = _PRANDTL_NUMBER,
    heat_capacity_ratio: float = _HEAT_CAPACITY_RATIO,
    atmospheric_pressure: float = _ATMOSPHERIC_PRESSURE,
) -> PorousMediumResult:
    """Johnson-Champoux-Allard five-parameter rigid-frame model.

    Effective density (Cox & D'Antonio 3e, Eq. (6.19)):

    ``rho_e = (T rho / phi) [1 + (sigma phi / (j w rho T))
    sqrt(1 + 4 j T^2 eta rho w / (sigma^2 L^2 phi^2))]``

    and effective bulk modulus (Eq. (6.20)):

    ``K_e = (gamma P0 / phi) / (gamma - (gamma - 1) [1 +
    (8 eta / (j L'^2 Pr w rho)) sqrt(1 + j rho w Pr L'^2 / (16 eta))]^-1)``

    with tortuosity ``T``, porosity ``phi``, viscous/thermal characteristic
    lengths ``L`` / ``L'``; then ``Zc = sqrt(K_e rho_e)`` and
    ``k = w sqrt(rho_e / K_e)`` (Eqs. (6.24)-(6.25)). Both quantities are
    surface-normalised (the ``1/phi`` factors are included). The model has
    the exact limits ``j w rho_e -> sigma`` as ``w -> 0`` and
    ``rho_e -> (T rho / phi)(1 + (1 - j) delta_v / L)`` as ``w -> inf``
    (Johnson et al. 1987), pinned in the tests.

    :param frequency: Frequency vector ``f``, in hertz.
    :param flow_resistivity: Airflow resistivity ``sigma``, in Pa s/m2.
    :param porosity: Open porosity ``phi`` (0 < phi <= 1).
    :param tortuosity: High-frequency tortuosity ``T = alpha_inf`` (>= 1).
    :param viscous_length: Viscous characteristic length ``L``, in metres.
    :param thermal_length: Thermal characteristic length ``L'``, in metres
        (physically ``L' >= L``).
    :param speed_of_sound: Speed of sound ``c`` in air, in m/s.
    :param air_density: Air density ``rho``, in kg/m3.
    :param viscosity: Dynamic viscosity ``eta`` of air, in Pa s.
    :param prandtl_number: Prandtl number ``Pr`` of air.
    :param heat_capacity_ratio: Ratio of specific heats ``gamma``.
    :param atmospheric_pressure: Static pressure ``P0``, in Pa.
    :return: A :class:`PorousMediumResult`.
    """
    f = require_positive_array(frequency, "frequency")
    sigma = require_positive(flow_resistivity, "flow_resistivity")
    phi = require_positive(porosity, "porosity")
    if phi > 1.0:
        raise ValueError(_POROSITY_MESSAGE)
    t_inf = require_positive(tortuosity, "tortuosity")
    if t_inf < 1.0:
        raise ValueError("'tortuosity' must be >= 1.")
    lam_v = require_positive(viscous_length, "viscous_length")
    lam_t = require_positive(thermal_length, "thermal_length")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    rho0 = require_positive(air_density, "air_density")
    eta = require_positive(viscosity, "viscosity")
    pr = require_positive(prandtl_number, "prandtl_number")
    gamma = require_positive(heat_capacity_ratio, "heat_capacity_ratio")
    p0 = require_positive(atmospheric_pressure, "atmospheric_pressure")

    omega = 2.0 * np.pi * f
    # Effective density, Cox & D'Antonio Eq. (6.19).
    g_v = np.sqrt(
        1.0
        + 4.0j * t_inf**2 * eta * rho0 * omega / (sigma**2 * lam_v**2 * phi**2)
    )
    rho_e = (t_inf * rho0 / phi) * (
        1.0 + sigma * phi / (1j * omega * rho0 * t_inf) * g_v
    )
    # Effective bulk modulus, Cox & D'Antonio Eq. (6.20).
    g_t = np.sqrt(1.0 + 1j * rho0 * omega * pr * lam_t**2 / (16.0 * eta))
    inner = 1.0 + 8.0 * eta / (1j * lam_t**2 * pr * omega * rho0) * g_t
    k_e = (gamma * p0 / phi) / (gamma - (gamma - 1.0) / inner)
    zc = np.sqrt(k_e * rho_e)
    k = omega * np.sqrt(rho_e / k_e)
    return PorousMediumResult(
        frequency=f,
        characteristic_impedance=np.asarray(zc, dtype=np.complex128),
        wavenumber=np.asarray(k, dtype=np.complex128),
        effective_density=np.asarray(rho_e, dtype=np.complex128),
        bulk_modulus=np.asarray(k_e, dtype=np.complex128),
        model="johnson_champoux_allard",
        flow_resistivity=sigma,
        speed_of_sound=c0,
        air_density=rho0,
    )


def decoupling_frequency(
    flow_resistivity: float, *, porosity: float, frame_density: float
) -> float:
    """Zwikker-Kosten decoupling frequency ``Fd`` of a porous frame.

    ``Fd = sigma phi**2 / (2 pi rho1)`` (Allard & Atalla 2e, Sect. 11.3.4,
    printed p. 251; the same closed form as their Eq. (6.90), printed p. 126).
    Above ``Fd`` the visco-inertial coupling between the pore fluid and the
    frame is too weak for the acoustic wave to shake the frame, so the
    rigid-frame equivalent fluid of :func:`johnson_champoux_allard` applies;
    below it the frame moves and the limp correction of :func:`limp_frame`
    matters.

    :param flow_resistivity: Airflow resistivity ``sigma``, in Pa s/m2 (> 0).
    :param porosity: Open porosity ``phi`` (0 < phi <= 1).
    :param frame_density: Bulk density of the frame ``rho1``, in kg/m3 (> 0):
        the mass of solid per unit volume of material, i.e. the density of the
        sample as weighed, not the density of the material the fibres are
        made of.
    :return: The decoupling frequency ``Fd``, in hertz.
    :raises ValueError: for a non-positive input or a porosity above 1.
    """
    sigma = require_positive(flow_resistivity, "flow_resistivity")
    phi = require_positive(porosity, "porosity")
    if phi > 1.0:
        raise ValueError(_POROSITY_MESSAGE)
    rho1 = require_positive(frame_density, "frame_density")
    return float(sigma * phi**2 / (2.0 * np.pi * rho1))


def limp_frame_applicable(
    frame_bulk_modulus: float,
    *,
    criterion: str = "doutres",
    fluid_bulk_modulus: float = _ATMOSPHERIC_PRESSURE,
) -> bool:
    """Whether the limp-frame model may be used, by published rule of thumb.

    Both published criteria compare the bulk modulus of the frame *in vacuum*
    ``K_c`` with that of the fluid in the pores ``K_f`` (Allard & Atalla 2e,
    printed pp. 253-254): Beranek (1947) requires ``|K_c/K_f| < 0.05``, and the
    frame structural interaction study of Doutres et al. (2007) relaxes it to
    ``|K_c/K_f| < 0.2``. With ``K_f`` taken as the isothermal bulk modulus of
    air, ``P0 = 101,3 kPa``, the relaxed criterion is the book's statement that
    "the limp model is applicable for materials having a bulk modulus lower
    than 20 kPa". Neither criterion accounts for boundary or mounting
    conditions, and the book notes that a thin light foam decoupled from a
    vibrating structure by an air gap behaves limply well above the limit.

    :param frame_bulk_modulus: Bulk modulus of the frame in vacuum ``K_c``, in
        Pa (>= 0; pass ``abs(K_c)`` for a complex modulus).
    :param criterion: Key into :data:`LIMP_FRAME_CRITERIA`, ``"doutres"``
        (Default, 0,2) or ``"beranek"`` (0,05).
    :param fluid_bulk_modulus: Bulk modulus of the pore fluid ``K_f``, in Pa
        (Default: 101 325, the isothermal value for air).
    :return: ``True`` when ``|K_c/K_f|`` does not exceed the threshold.
    :raises ValueError: for a negative modulus or an unknown criterion.
    """
    key = require_choice(criterion, "criterion", tuple(LIMP_FRAME_CRITERIA))
    k_c = require_non_negative(frame_bulk_modulus, "frame_bulk_modulus")
    k_f = require_positive(fluid_bulk_modulus, "fluid_bulk_modulus")
    return bool(k_c / k_f <= LIMP_FRAME_CRITERIA[key])


def limp_frame(
    medium: PorousMediumResult,
    frame_density: float,
    *,
    porosity: float = 1.0,
) -> PorousMediumResult:
    """Limp-frame correction of a rigid-frame equivalent fluid (A&A 11.3.4).

    A light frame (aeronautic-grade fibreglass, felts, screens) is dragged
    along by the pore fluid instead of standing still, and the rigid-frame
    models of :func:`delany_bazley`, :func:`miki` and
    :func:`johnson_champoux_allard` have no way to carry that inertia.
    Neglecting the stiffness of the frame altogether in the Biot mixed
    pressure-displacement formulation leaves an equivalent fluid with the same
    bulk modulus and a corrected effective density (Allard & Atalla 2e,
    Eqs. (11.53)-(11.55), printed pp. 252-253, after Panneton 2007):

    ``rho_limp = (rho_t rho_eq - rho0**2) / (rho_t + rho_eq - 2 rho0)``

    with ``rho_eq`` the rigid-frame effective density of *medium*, ``rho0`` the
    density of the pore fluid and ``rho_t = rho1 + phi rho0`` the apparent
    total density of the material. What anchors this expression is the printed
    equation itself, transcribed term by term; Allard & Atalla tabulate no
    computed limp density anywhere, so there are no published digits to check
    against. The book also states two exact limits in prose, and both are
    verified, but they are weaker than they look: neither pins the ``rho0**2``
    and ``2 rho0`` terms, since a sign-flipped variant of Eq. (11.55) satisfies
    both of them (and even the ``1/rho1`` decay of the heavy-frame residual).
    They corroborate the transcription rather than determine it:

    * **heavy frame**: as ``rho1 -> inf`` the correction vanishes and the
      rigid-frame result is recovered (the book's own reading of Eq. (11.55));
    * **low frequency**: since ``rho_eq -> sigma / (j w)`` as ``w -> 0``
      (Eq. (5.37)), ``rho_limp -> rho_t``, a finite real density, where the
      rigid-frame model diverges. The rigid frame forbids rigid-body motion of
      the sample; the limp one allows it, which is why the two differ mainly
      at low frequency and why the limp model is the right one for an
      unconstrained sample in an impedance tube.

    The corrected medium is a drop-in
    :class:`PorousMediumResult`, so it can be handed to
    :class:`PorousLayer` inside :func:`layered_absorber` exactly like the
    rigid-frame one.

    Use :func:`decoupling_frequency` to see where the frame stops following the
    fluid and :func:`limp_frame_applicable` for the published bulk-modulus
    rule of thumb on when the frame may be treated as limp at all.

    :param medium: A rigid-frame :class:`PorousMediumResult` (its
        ``effective_density`` is ``rho_eq`` and its ``bulk_modulus`` is kept).
    :param frame_density: Bulk density of the frame ``rho1``, in kg/m3 (> 0).
    :param porosity: Open porosity ``phi`` (0 < phi <= 1, Default: 1,0, the
        high-porosity assumption of the one-parameter models).
    :return: A :class:`PorousMediumResult` with model
        ``"limp_frame(<base model>)"``.
    :raises ValueError: for a non-positive input or a porosity above 1.
    """
    rho1 = require_positive(frame_density, "frame_density")
    phi = require_positive(porosity, "porosity")
    if phi > 1.0:
        raise ValueError(_POROSITY_MESSAGE)
    rho0 = medium.air_density
    rho_eq = np.asarray(medium.effective_density, dtype=np.complex128)
    k_e = np.asarray(medium.bulk_modulus, dtype=np.complex128)
    # Apparent total density of the limp medium, A&A Eq. (11.55).
    rho_t = rho1 + phi * rho0
    rho_limp = (rho_t * rho_eq - rho0**2) / (rho_t + rho_eq - 2.0 * rho0)
    omega = 2.0 * np.pi * medium.frequency
    zc = np.sqrt(k_e * rho_limp)
    k = omega * np.sqrt(rho_limp / k_e)
    return PorousMediumResult(
        frequency=medium.frequency,
        characteristic_impedance=np.asarray(zc, dtype=np.complex128),
        wavenumber=np.asarray(k, dtype=np.complex128),
        effective_density=np.asarray(rho_limp, dtype=np.complex128),
        bulk_modulus=k_e,
        model=f"limp_frame({medium.model})",
        flow_resistivity=medium.flow_resistivity,
        speed_of_sound=medium.speed_of_sound,
        air_density=rho0,
    )


# ---------------------------------------------------------------------------
# Resonant sheet impedances
# ---------------------------------------------------------------------------
def perforation_end_correction(open_area: float) -> float:
    """End-correction factor ``delta`` of a circular perforation.

    ``delta = 0.85 (1 - 1.47 eps^1/2 + 0.47 eps^3/2)`` - the Fok-function
    interaction correction for circular holes (Cox & D'Antonio 3e, Table 7.1,
    Nesterov row; no open-area limit). Each orifice end adds ``delta a`` of
    air-plug length, and ``delta -> 0.85`` for an isolated hole.

    :param open_area: Fractional open area ``eps`` of the sheet (0..1).
    :return: End-correction factor ``delta`` (dimensionless, per end).
    """
    eps = require_positive(open_area, "open_area")
    if eps > 1.0:
        raise ValueError(_OPEN_AREA_MESSAGE)
    return float(0.85 * (1.0 - 1.47 * eps**0.5 + 0.47 * eps**1.5))


def perforated_plate_impedance(
    frequency: ArrayLike,
    *,
    thickness: float,
    hole_radius: float,
    open_area: float,
    end_correction: float | None = None,
    air_density: float = _AIR_DENSITY,
    viscosity: float = _AIR_VISCOSITY,
) -> Complex:
    """Transfer impedance of a rigid perforated plate with circular holes.

    Acoustic mass with both end corrections and the boundary-layer term
    (Cox & D'Antonio 3e, Eq. (7.6)):

    ``m = (rho/eps)[t + 2 delta a + sqrt(8 nu / w)(1 + t/(2a))]``

    and visco-thermal surface resistance (Eq. (7.12)):

    ``r = (rho/eps) sqrt(8 nu w) (1 + t/(2a))``,

    giving ``z = r + j w m`` (the series impedance added on top of the
    backing, Eq. (7.21)). Assumes hole radii well above the boundary-layer
    thickness; use :func:`microperforated_plate_impedance` for submillimetre
    holes.

    :param frequency: Frequency vector ``f``, in hertz.
    :param thickness: Plate thickness ``t``, in metres.
    :param hole_radius: Hole radius ``a``, in metres.
    :param open_area: Fractional open area ``eps`` (0..1).
    :param end_correction: End-correction factor ``delta`` per end; default
        :func:`perforation_end_correction` of ``eps``.
    :param air_density: Air density ``rho``, in kg/m3.
    :param viscosity: Dynamic viscosity ``eta`` of air, in Pa s.
    :return: Complex transfer impedance ``z``, in Pa s/m.
    """
    f = require_positive_array(frequency, "frequency")
    t = require_positive(thickness, "thickness")
    a = require_positive(hole_radius, "hole_radius")
    eps = require_positive(open_area, "open_area")
    if eps > 1.0:
        raise ValueError(_OPEN_AREA_MESSAGE)
    rho0 = require_positive(air_density, "air_density")
    eta = require_positive(viscosity, "viscosity")
    delta = (
        perforation_end_correction(eps)
        if end_correction is None
        else require_non_negative(end_correction, "end_correction")
    )
    omega = 2.0 * np.pi * f
    nu = eta / rho0
    edge = 1.0 + t / (2.0 * a)
    mass = (rho0 / eps) * (t + 2.0 * delta * a + np.sqrt(8.0 * nu / omega) * edge)
    resistance = (rho0 / eps) * np.sqrt(8.0 * nu * omega) * edge
    return np.asarray(resistance + 1j * omega * mass, dtype=np.complex128)


def microperforated_plate_impedance(
    frequency: ArrayLike,
    *,
    thickness: float,
    hole_radius: float,
    open_area: float,
    end_correction: float = 0.85,
    air_density: float = _AIR_DENSITY,
    viscosity: float = _AIR_VISCOSITY,
) -> Complex:
    """Transfer impedance of a microperforated plate (Maa's exact model).

    The specific impedance of one submillimetre hole is the exact short-tube
    result (Maa 1998, Eq. (2); reproduced as Cox & D'Antonio 3e Eq. (7.33)
    and the same Bessel kernel as Mechel 2e Sect. G.3):

    ``z1 = j w rho t [1 - (2 / (x sqrt(-j))) J1(x sqrt(-j)) / J0(x sqrt(-j))]^-1``

    with the perforate constant ``x = a sqrt(rho w / eta)``. Dividing by the
    open area and adding Maa's Eq. (5) end corrections - the Rayleigh/Ingard
    surface resistance ``sqrt(2 w rho eta) / (2 eps)`` and the piston
    end-correction reactance ``j w rho (2 delta a) / eps`` (``0.85 d`` total
    for the default ``delta = 0.85`` per end) - gives the sheet transfer
    impedance (Cox & D'Antonio Eq. (7.35)).

    :param frequency: Frequency vector ``f``, in hertz.
    :param thickness: Plate thickness ``t``, in metres.
    :param hole_radius: Hole radius ``a``, in metres (submillimetre for a
        genuine microperforated design).
    :param open_area: Fractional open area ``eps`` (0..1).
    :param end_correction: End-correction factor ``delta`` per end
        (default 0.85, the isolated-orifice value used by Maa).
    :param air_density: Air density ``rho``, in kg/m3.
    :param viscosity: Dynamic viscosity ``eta`` of air, in Pa s.
    :return: Complex transfer impedance ``z``, in Pa s/m.
    """
    f = require_positive_array(frequency, "frequency")
    t = require_positive(thickness, "thickness")
    a = require_positive(hole_radius, "hole_radius")
    eps = require_positive(open_area, "open_area")
    if eps > 1.0:
        raise ValueError(_OPEN_AREA_MESSAGE)
    delta = require_non_negative(end_correction, "end_correction")
    rho0 = require_positive(air_density, "air_density")
    eta = require_positive(viscosity, "viscosity")
    omega = 2.0 * np.pi * f
    arg = a * np.sqrt(rho0 * omega / eta) * np.sqrt(-1j)
    bessel_ratio = special.jv(1, arg) / special.jv(0, arg)
    z_hole = 1j * omega * rho0 * t / (1.0 - 2.0 * bessel_ratio / arg)
    z = (
        z_hole / eps
        + np.sqrt(2.0 * omega * rho0 * eta) / (2.0 * eps)
        + 1j * omega * rho0 * 2.0 * delta * a / eps
    )
    return np.asarray(z, dtype=np.complex128)


def membrane_impedance(
    frequency: ArrayLike,
    *,
    surface_density: float,
    resistance: float = 0.0,
) -> Complex:
    """Transfer impedance of a limp impervious membrane.

    ``z = r + j w m`` - the surface-mass reactance (Cox & D'Antonio 3e,
    Eq. (7.14); Bies 5e Eq. (D.96)) plus an optional empirical resistance
    for the internal/fixing losses.

    :param frequency: Frequency vector ``f``, in hertz.
    :param surface_density: Mass per unit area ``m``, in kg/m2.
    :param resistance: Series flow resistance ``r``, in Pa s/m (default 0).
    :return: Complex transfer impedance ``z``, in Pa s/m.
    """
    f = require_positive_array(frequency, "frequency")
    m = require_positive(surface_density, "surface_density")
    r = require_non_negative(resistance, "resistance")
    return np.asarray(r + 1j * 2.0 * np.pi * f * m, dtype=np.complex128)


def helmholtz_resonance_frequency(
    *,
    cavity_depth: float,
    plate_thickness: float,
    hole_radius: float,
    open_area: float,
    end_correction: float | None = None,
    speed_of_sound: float = _SPEED_OF_SOUND,
) -> float:
    """Resonance of a perforated sheet over a shallow cavity (closed form).

    ``f0 = (c / 2 pi) sqrt(eps / (t' d))`` with the end-corrected plug length
    ``t' = t + 2 delta a`` (Cox & D'Antonio 3e, Eqs. (7.4)/(7.6), valid for
    ``k d << 1``).

    :param cavity_depth: Cavity depth ``d``, in metres.
    :param plate_thickness: Plate thickness ``t``, in metres.
    :param hole_radius: Hole radius ``a``, in metres.
    :param open_area: Fractional open area ``eps`` (0..1).
    :param end_correction: End-correction factor ``delta`` per end; default
        :func:`perforation_end_correction` of ``eps``.
    :param speed_of_sound: Speed of sound ``c`` in air, in m/s.
    :return: Resonance frequency ``f0``, in hertz.
    """
    d = require_positive(cavity_depth, "cavity_depth")
    t = require_positive(plate_thickness, "plate_thickness")
    a = require_positive(hole_radius, "hole_radius")
    eps = require_positive(open_area, "open_area")
    if eps > 1.0:
        raise ValueError(_OPEN_AREA_MESSAGE)
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    delta = (
        perforation_end_correction(eps)
        if end_correction is None
        else require_non_negative(end_correction, "end_correction")
    )
    t_eff = t + 2.0 * delta * a
    return float(c0 / (2.0 * np.pi) * np.sqrt(eps / (t_eff * d)))


def membrane_resonance_frequency(
    *,
    surface_density: float,
    cavity_depth: float,
    isothermal: bool = False,
    speed_of_sound: float = _SPEED_OF_SOUND,
    air_density: float = _AIR_DENSITY,
) -> float:
    """Mass-spring resonance of a membrane over a shallow cavity.

    ``f0 = (1 / 2 pi) sqrt(rho c^2 / (m d))`` for an adiabatic air spring -
    numerically the classical ``f0 = 60 / sqrt(m d)`` (Cox & D'Antonio 3e,
    Eq. (7.9)). With ``isothermal=True`` the spring stiffness drops by
    ``gamma``, giving ``~50 / sqrt(m d)`` (Eq. (7.10)), the porous-filled
    cavity case below about 500 Hz.

    :param surface_density: Membrane mass per unit area ``m``, in kg/m2.
    :param cavity_depth: Cavity depth ``d``, in metres.
    :param isothermal: Use the isothermal air-spring stiffness.
    :param speed_of_sound: Speed of sound ``c`` in air, in m/s.
    :param air_density: Air density ``rho``, in kg/m3.
    :return: Resonance frequency ``f0``, in hertz.
    """
    m = require_positive(surface_density, "surface_density")
    d = require_positive(cavity_depth, "cavity_depth")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    rho0 = require_positive(air_density, "air_density")
    stiffness = rho0 * c0**2 / d
    if isothermal:
        stiffness /= _HEAT_CAPACITY_RATIO
    return float(np.sqrt(stiffness / m) / (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Declarative layers and the transfer-matrix solver
# ---------------------------------------------------------------------------
class _DrawableLayer:
    """Shared geometry drawing for the layer dataclasses.

    ``plot()`` draws the layer as a one-layer stack cross-section, to scale,
    against the rigid backing; a full stack is drawn by
    :func:`~phonometry.materials.plot_absorber_stack` or by
    :meth:`LayeredAbsorberResult.plot_geometry`.
    """

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw this layer's cross-section to scale (dimensioned).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.geometry import plot_absorber_stack

        check_language(language)
        return plot_absorber_stack(
            [cast("Layer", self)], ax=ax, language=language, **kwargs
        )


@dataclass(frozen=True)
class AirLayer(_DrawableLayer):
    """A plain air gap of ``thickness`` metres inside the stack."""

    thickness: float


@dataclass(frozen=True)
class PorousLayer(_DrawableLayer):
    """A porous layer of ``thickness`` metres described by *medium*.

    ``medium`` is a :class:`PorousMediumResult` (from :func:`delany_bazley`,
    :func:`miki`, :func:`johnson_champoux_allard`, or built directly from
    measured ``Zc``/``k`` data) evaluated on the same frequency vector that
    is passed to :func:`layered_absorber`.
    """

    thickness: float
    medium: PorousMediumResult


@dataclass(frozen=True)
class PerforatedPlateLayer(_DrawableLayer):
    """A rigid perforated plate (see :func:`perforated_plate_impedance`)."""

    thickness: float
    hole_radius: float
    open_area: float
    end_correction: float | None = None


@dataclass(frozen=True)
class MicroperforatedPlateLayer(_DrawableLayer):
    """A microperforated plate (see :func:`microperforated_plate_impedance`)."""

    thickness: float
    hole_radius: float
    open_area: float
    end_correction: float = 0.85


@dataclass(frozen=True)
class MembraneLayer(_DrawableLayer):
    """A limp impervious membrane (see :func:`membrane_impedance`)."""

    surface_density: float
    resistance: float = 0.0


@dataclass(frozen=True)
class PoroelasticLayer(_DrawableLayer):
    """A porous layer whose frame is elastic (full Biot theory).

    Where :class:`PorousLayer` collapses the material into a single wave in an
    equivalent fluid, this layer carries the three Biot waves of Allard &
    Atalla 2e chapter 6 - two compressional and one shear - so the frame can
    resonate. It is the only layer type that reproduces the quarter-wavelength
    frame resonance of :func:`~phonometry.materials.biot.frame_quarter_wave_resonance`,
    and the only one for which an air gap behind the layer, a bonded backing or
    an oblique angle change the frame motion rather than only the pore fluid.

    ``medium`` is the **rigid-frame** equivalent fluid of the pores (normally a
    :func:`johnson_champoux_allard` result on the solver's frequency vector):
    the frame inertia is added by the Biot model itself, so a limp-corrected
    medium would count it twice. The remaining fields describe the frame.

    Adding one of these to a stack switches :func:`layered_absorber` to the
    global-matrix assembly of Allard & Atalla Sect. 11.5. Two adjacent
    poroelastic layers are coupled as *bonded* frames (their Eq. (11.67)); a
    sheet layer next to a poroelastic layer is coupled as a free, mechanically
    decoupled screen (air on both sides, their Sect. 11.3.6).
    """

    thickness: float
    medium: PorousMediumResult
    porosity: float
    tortuosity: float
    frame_density: float
    shear_modulus: complex
    poisson_ratio: float = 0.0


Layer = (
    AirLayer
    | PorousLayer
    | PerforatedPlateLayer
    | MicroperforatedPlateLayer
    | MembraneLayer
    | PoroelasticLayer
)


@dataclass(frozen=True)
class LayeredAbsorberResult:
    """Oblique-incidence prediction of a layered absorber.

    All arrays share the shape of ``frequency``. ``surface_impedance`` is the
    specific impedance ``Zs = p / u_n`` at the front face (may be ``inf``
    for a lossless-sheet stack over a rigid wall), ``reflection`` the complex
    plane-wave reflection factor ``R(theta)``, ``absorption`` the coefficient
    ``alpha(theta) = 1 - |R|^2`` and ``transfer_matrix`` the total chain
    matrix with shape ``(2, 2, len(frequency))`` (unimodular: every layer is
    reciprocal).

    ``layers`` retains the layer sequence the stack was solved with (front
    layer first) so :meth:`plot_geometry` can draw the cross-section; it is
    appended after the original fields and defaults to ``None`` for
    hand-built results.
    """

    frequency: Real
    angle: float
    surface_impedance: Complex
    normalized_impedance: Complex
    reflection: Complex
    absorption: Real
    transfer_matrix: Complex
    layers: tuple[Layer, ...] | None = None

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the absorption spectrum ``alpha(f)`` with ``|R|`` overlaid.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.materials import plot_layered_absorber

        check_language(language)
        return plot_layered_absorber(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the solved stack cross-section to scale (dimensioned).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the result does not retain its ``layers``.
        """
        from .._i18n import check_language
        from .._plot.geometry import plot_layered_absorber_geometry

        check_language(language)
        return plot_layered_absorber_geometry(
            self, ax=ax, language=language, **kwargs
        )


@dataclass(frozen=True)
class DiffuseFieldAbsorptionResult:
    """Random-incidence (Paris-integral) absorption of a layered absorber.

    ``absorption`` is ``alpha_dif(f)`` from Mechel 2e Sect. D.5 Eq. (9):
    the plane-wave ``alpha(theta)`` weighted by ``cos(theta) sin(theta)`` and
    normalised by ``sin^2(theta_limit)``.
    """

    frequency: Real
    absorption: Real
    angle_limit: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the random-incidence absorption spectrum ``alpha_dif(f)``.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.materials import plot_diffuse_field_absorption

        check_language(language)
        return plot_diffuse_field_absorption(self, ax=ax, language=language, **kwargs)


def _sheet_impedance(
    layer: Layer,
    f: Real,
    *,
    air_density: float,
    viscosity: float,
) -> Complex:
    """Series transfer impedance of a sheet layer on the grid *f*."""
    if isinstance(layer, PerforatedPlateLayer):
        return perforated_plate_impedance(
            f,
            thickness=layer.thickness,
            hole_radius=layer.hole_radius,
            open_area=layer.open_area,
            end_correction=layer.end_correction,
            air_density=air_density,
            viscosity=viscosity,
        )
    if isinstance(layer, MicroperforatedPlateLayer):
        return microperforated_plate_impedance(
            f,
            thickness=layer.thickness,
            hole_radius=layer.hole_radius,
            open_area=layer.open_area,
            end_correction=layer.end_correction,
            air_density=air_density,
            viscosity=viscosity,
        )
    if isinstance(layer, MembraneLayer):
        return membrane_impedance(
            f,
            surface_density=layer.surface_density,
            resistance=layer.resistance,
        )
    raise TypeError(f"not a sheet layer: {layer!r}")  # pragma: no cover


def _fluid_layer_terms(
    zc: Complex, k: Complex, thickness: float, k0_sin2: Real
) -> tuple[Complex, Complex]:
    """In-depth impedance ``Zx`` and phase ``kx d`` of an oblique fluid layer.

    ``kx = sqrt(k^2 - k0^2 sin^2 theta)`` (Snell's law, Cox & D'Antonio 3e
    Eq. (2.30)) and the in-depth wave impedance ``Zx = zc k / kx``; the
    layer chain matrix (Eq. (2.29)) is built from ``cos``/``sin`` of
    ``kx d`` with these two terms.
    """
    kx = np.sqrt(k * k - k0_sin2)
    # Passive decay: keep the branch with non-positive imaginary part.
    kx = np.where(kx.imag > 0.0, -kx, kx)
    zx = zc * k / kx
    return (
        np.asarray(zx, dtype=np.complex128),
        np.asarray(kx * thickness, dtype=np.complex128),
    )


def _porous_layer_term(
    layer: PorousLayer, f: Real, k0_sin2: Real
) -> tuple[Complex, Complex] | None:
    """``(Zx, kx d)`` of a porous layer, or ``None`` when zero-thickness."""
    d = require_non_negative(layer.thickness, "PorousLayer.thickness")
    if d <= 0.0:
        return None
    medium = layer.medium
    _check_medium_grid(medium, f, "PorousLayer")
    return _fluid_layer_terms(
        np.asarray(medium.characteristic_impedance, dtype=np.complex128),
        np.asarray(medium.wavenumber, dtype=np.complex128),
        d,
        k0_sin2,
    )


def _layer_terms(
    layers: list[Layer] | tuple[Layer, ...],
    f: Real,
    *,
    k0: Real,
    k0_sin2: Real,
    rc: float,
    rho0: float,
    viscosity: float,
) -> list[tuple[str, Complex, Complex]]:
    """Evaluate each layer once: fluid layers as ``(Zx, kx d)``, sheets as z.

    Zero-thickness fluid layers contribute the identity matrix and are
    skipped (``require_non_negative`` guarantees ``d >= 0``, so the strict
    ``d > 0`` test keeps exactly the non-degenerate layers).
    """
    terms: list[tuple[str, Complex, Complex]] = []
    for layer in layers:
        if isinstance(layer, AirLayer):
            d = require_non_negative(layer.thickness, "AirLayer.thickness")
            if d > 0.0:
                zc = np.full(f.shape, rc, dtype=np.complex128)
                k = np.asarray(k0, dtype=np.complex128)
                terms.append(("fluid", *_fluid_layer_terms(zc, k, d, k0_sin2)))
        elif isinstance(layer, PorousLayer):
            term = _porous_layer_term(layer, f, k0_sin2)
            if term is not None:
                terms.append(("fluid", *term))
        else:
            z = _sheet_impedance(
                layer,
                f,
                air_density=rho0,
                viscosity=viscosity,
            )
            terms.append(("sheet", z, z))
    return terms


def _termination_admittance(
    termination: str | complex | ArrayLike,
    f: Real,
    *,
    cos_t: float,
    rc: float,
) -> Complex:
    """Admittance ``G = u/p`` at the termination face of the stack."""
    zl_arr = _termination_impedance(termination, f, cos_t=cos_t, rc=rc)
    if zl_arr is None:
        return np.zeros_like(f, dtype=np.complex128)
    return np.asarray(np.ones_like(f) / zl_arr, dtype=np.complex128)


def _termination_impedance(
    termination: str | complex | ArrayLike,
    f: Real,
    *,
    cos_t: float,
    rc: float,
) -> Complex | None:
    """Impedance ``p/v3`` closing the stack, or ``None`` for a hard wall.

    The global-matrix assembly of Allard & Atalla Sect. 11.5 needs the
    termination as an impedance (their Eq. (11.84)); the admittance the
    recursion of :func:`_surface_admittance` consumes is its reciprocal, with
    the hard wall the one case that has no finite impedance.
    """
    if isinstance(termination, str):
        if termination == "rigid":
            return None
        if termination == "free":
            return np.full(f.shape, rc / cos_t, dtype=np.complex128)
        raise ValueError(
            "'termination' must be 'rigid', 'free' or a complex impedance."
        )
    zl_arr = np.asarray(termination, dtype=np.complex128)
    if zl_arr.ndim > 0 and zl_arr.shape != f.shape:
        raise ValueError(
            "'termination' impedance array must be scalar or match the "
            f"frequency vector length ({f.size}), got {zl_arr.size}."
        )
    if not np.all(np.abs(zl_arr) > 0.0):
        raise ValueError("'termination' impedance must be non-zero.")
    return np.asarray(np.broadcast_to(zl_arr, f.shape), dtype=np.complex128)


def _surface_admittance(
    terms: list[tuple[str, Complex, Complex]], g: Complex
) -> Complex:
    """Back-to-front admittance recursion from the termination admittance.

    Stable: ``tan`` saturates where the chain-matrix entries would overflow.
    """
    for kind, a, b in reversed(terms):
        if kind == "fluid":
            zx, kxd = a, b
            t = np.tan(kxd)
            g = (g + 1j * t / zx) / (1.0 + 1j * zx * t * g)
        else:
            g = g / (1.0 + a * g)
    return g


def _chain_matrix(terms: list[tuple[str, Complex, Complex]], f: Real) -> Complex:
    """Raw front-to-back chain-matrix product of the evaluated layers.

    Informational; may overflow for extremely attenuating layers while the
    admittance recursion stays finite.
    """
    ones = np.ones_like(f, dtype=np.complex128)
    zeros = np.zeros_like(f, dtype=np.complex128)
    t11, t12, t21, t22 = ones, zeros, zeros, ones
    with np.errstate(over="ignore", invalid="ignore"):
        for kind, a, b in terms:
            if kind == "fluid":
                zx, kxd = a, b
                cos_l, sin_l = np.cos(kxd), np.sin(kxd)
                m = (cos_l, 1j * zx * sin_l, 1j * sin_l / zx, cos_l)
            else:
                m = (ones, a, zeros, ones)
            m11, m12, m21, m22 = m
            t11, t12, t21, t22 = (
                t11 * m11 + t12 * m21,
                t11 * m12 + t12 * m22,
                t21 * m11 + t22 * m21,
                t21 * m12 + t22 * m22,
            )
    return np.asarray([[t11, t12], [t21, t22]], dtype=np.complex128)


def _check_medium_grid(medium: PorousMediumResult, f: Real, owner: str) -> None:
    """Reject a medium evaluated on a different frequency vector."""
    if not np.array_equal(np.asarray(medium.frequency), f):
        raise ValueError(
            f"{owner}.medium was evaluated on a different frequency "
            "vector; rebuild the medium on the solver grid."
        )


def _split_fluid_run(
    terms: list[tuple[str, Complex, Complex]], budget: float, limit: int
) -> list[list[tuple[str, Complex, Complex]]]:
    """Group a fluid run into chain blocks of at most *budget* nepers.

    A fluid run that attenuates by ``b`` nepers has chain-matrix entries of
    order ``e^b`` while the same block's back face is the identity, so the
    assembled system of Allard & Atalla Sect. 11.6 holds rows differing by
    ``e^b`` and the elimination of the block loses about ``b / ln(10)``
    digits; past ``b ~ 710`` the entries overflow float64 outright. The split
    is algebraically exact, because a homogeneous fluid layer of phase
    ``kx d`` is the product of ``m`` layers of phase ``kx d / m``.

    Returns the run unchanged, as a single group, whenever it stays inside
    the budget, so ordinary stacks keep the exact chain product they had.
    Sheet layers carry no attenuation and never force a split of their own.

    :raises ValueError: when the run would need more than *limit* blocks.
    """
    parts: list[tuple[str, Complex, Complex, float]] = []
    attenuation = 0.0
    for kind, a, b in terms:
        if kind != "fluid":
            parts.append((kind, a, b, 0.0))
            continue
        loss = float(np.max(np.abs(np.imag(b))))
        attenuation += loss
        pieces = max(1, int(np.ceil(loss / budget)))
        if pieces == 1:
            parts.append((kind, a, b, loss))
        else:
            parts.extend([(kind, a, b / pieces, loss / pieces)] * pieces)
    if attenuation > budget * limit:
        raise ValueError(
            f"the fluid layers of the stack attenuate by {attenuation:.0f} "
            f"nepers, which the global-matrix assembly cannot resolve in "
            f"{limit} blocks. Reduce their thickness: nothing behind such a "
            "run contributes to the surface impedance."
        )

    groups: list[list[tuple[str, Complex, Complex]]] = []
    current: list[tuple[str, Complex, Complex]] = []
    total = 0.0
    for kind, a, b, loss in parts:
        if current and total + loss > budget:
            groups.append(current)
            current, total = [], 0.0
        current.append((kind, a, b))
        total += loss
    if current:
        groups.append(current)
    return groups


def _stack_blocks(
    layers: list[Layer] | tuple[Layer, ...],
    f: Real,
    *,
    k0: Real,
    k0_sin2: Real,
    rc: float,
    rho0: float,
    viscosity: float,
    transverse_wavenumber: Real,
) -> list[Any]:
    """Split a stack into fluid blocks and poroelastic blocks.

    Consecutive fluid and sheet layers collapse into two-variable blocks
    carrying their chain-matrix product; each :class:`PoroelasticLayer` becomes
    a six-variable block of Allard & Atalla Sect. 11.3.3. A run or a layer
    that attenuates by more than ``biot._BLOCK_NEPERS`` is cut into several
    blocks first, which the global matrix handles exactly as it handles
    adjacent fluid layers and bonded halves of one poroelastic material.
    """
    from . import biot

    blocks: list[Any] = []
    pending: list[Layer] = []

    def flush() -> None:
        if not pending:
            return
        terms = _layer_terms(
            pending, f, k0=k0, k0_sin2=k0_sin2, rc=rc, rho0=rho0,
            viscosity=viscosity,
        )
        groups = _split_fluid_run(
            terms, biot._BLOCK_NEPERS, biot._MAX_BLOCKS
        )
        for group in groups:
            chain = _chain_matrix(group, f)
            blocks.append(biot._fluid_block(np.moveaxis(chain, -1, 0)))
        pending.clear()

    for layer in layers:
        if isinstance(layer, PoroelasticLayer):
            thickness = require_non_negative(
                layer.thickness, "PoroelasticLayer.thickness"
            )
            if thickness <= 0.0:
                continue
            _check_medium_grid(layer.medium, f, "PoroelasticLayer")
            flush()
            waves = biot.biot_waves(
                layer.medium,
                porosity=layer.porosity,
                tortuosity=layer.tortuosity,
                frame_density=layer.frame_density,
                shear_modulus=layer.shear_modulus,
                poisson_ratio=layer.poisson_ratio,
            )
            blocks.extend(
                biot._poroelastic_blocks(waves, thickness, transverse_wavenumber)
            )
        else:
            pending.append(layer)
    flush()
    return blocks


def layered_absorber(
    frequency: ArrayLike,
    layers: list[Layer] | tuple[Layer, ...],
    *,
    angle: float = 0.0,
    termination: str | complex | ArrayLike = "rigid",
    speed_of_sound: float = _SPEED_OF_SOUND,
    air_density: float = _AIR_DENSITY,
    viscosity: float = _AIR_VISCOSITY,
) -> LayeredAbsorberResult:
    """Transfer-matrix prediction of a layered absorber at one angle.

    The *layers* list is ordered from the sound-incidence side towards the
    *termination*. Fluid layers (:class:`AirLayer`, :class:`PorousLayer`)
    contribute the oblique chain matrix of Cox & D'Antonio 3e Eq. (2.29)
    (equivalently the impedance recursion of Bies 5e Eq. (D.95) and the
    scheme of Mechel 2e Sect. D.4); sheet layers (:class:`PerforatedPlateLayer`,
    :class:`MicroperforatedPlateLayer`, :class:`MembraneLayer`) enter as
    locally reacting series impedances. The chain is closed by a rigid wall
    (``termination="rigid"``), by radiation into free air behind
    (``termination="free"``, ``Z_L = rho c / cos(theta)``) or by an arbitrary
    complex impedance. The reflection factor is
    ``R = (Zs cos(theta) - rho c) / (Zs cos(theta) + rho c)`` and
    ``alpha = 1 - |R|^2`` (Mechel 2e Sect. D.3 Eq. (2)).

    ``Zs``, ``R`` and ``alpha`` are evaluated with the numerically robust
    admittance recursion (algebraically identical to the chain product but
    immune to the ``e^{|Im(kx)| d}`` overflow of the raw matrix entries for
    extremely attenuating layers); the raw chain matrix is still returned in
    ``transfer_matrix`` and may overflow in such extreme cases.

    :param frequency: Frequency vector ``f``, in hertz.
    :param layers: Layer stack from the incidence side to the termination.
    :param angle: Polar angle of incidence ``theta``, in radians
        (``0 <= theta < pi/2 - 1e-6``; grazing incidence is excluded).
    :param termination: ``"rigid"`` (default), ``"free"``, or a non-zero
        complex impedance (scalar or per-frequency array), in Pa s/m.
    :param speed_of_sound: Speed of sound ``c`` in air, in m/s.
    :param air_density: Air density ``rho``, in kg/m3.
    :param viscosity: Dynamic viscosity of air, in Pa s (sheet layers).
    :return: A :class:`LayeredAbsorberResult`.
    """
    f = require_positive_array(frequency, "frequency")
    if not layers:
        raise ValueError("'layers' must contain at least one layer.")
    theta = float(angle)
    # The last ~3e-8 rad below pi/2 round sin(theta)**2 to 1.0, driving the
    # in-depth wavenumber of an air layer to exactly zero (inf * 0 = nan in
    # the recursion); reject effectively grazing input with a clear error.
    if not 0.0 <= theta < np.pi / 2.0 - 1e-6:
        raise ValueError("'angle' must satisfy 0 <= angle < pi/2 - 1e-6.")
    c0 = require_positive(speed_of_sound, "speed_of_sound")
    rho0 = require_positive(air_density, "air_density")
    require_positive(viscosity, "viscosity")

    k0 = 2.0 * np.pi * f / c0
    k0_sin2 = np.asarray((k0 * np.sin(theta)) ** 2, dtype=np.float64)
    cos_t = float(np.cos(theta))
    rc = rho0 * c0

    if any(isinstance(layer, PoroelasticLayer) for layer in layers):
        from . import biot

        blocks = _stack_blocks(
            layers, f, k0=k0, k0_sin2=k0_sin2, rc=rc, rho0=rho0,
            viscosity=viscosity,
            transverse_wavenumber=np.asarray(k0 * np.sin(theta)),
        )
        if not blocks:
            raise ValueError("'layers' must contain at least one layer.")
        zs = biot._stack_surface_impedance(
            blocks, _termination_impedance(termination, f, cos_t=cos_t, rc=rc)
        )
        finite = np.isfinite(zs)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(finite, 1.0 / np.where(finite, zs, 1.0), 0.0 + 0j)
        tm = np.full((2, 2, f.size), np.nan + 0j, dtype=np.complex128)
    else:
        terms = _layer_terms(
            layers, f, k0=k0, k0_sin2=k0_sin2, rc=rc, rho0=rho0,
            viscosity=viscosity,
        )
        g = _surface_admittance(
            terms, _termination_admittance(termination, f, cos_t=cos_t, rc=rc)
        )
        # G = 0 (lossless stack over a rigid wall) maps to an infinite surface
        # impedance; everywhere else Zs = 1/G with a safe denominator.
        nonzero = np.abs(g) > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            zs = np.where(nonzero, 1.0 / np.where(nonzero, g, 1.0), np.inf + 0j)
        tm = _chain_matrix(terms, f)

    r = (cos_t - rc * g) / (cos_t + rc * g)
    alpha = 1.0 - np.abs(r) ** 2
    return LayeredAbsorberResult(
        frequency=f,
        angle=theta,
        surface_impedance=np.asarray(zs, dtype=np.complex128),
        normalized_impedance=np.asarray(zs / rc, dtype=np.complex128),
        reflection=np.asarray(r, dtype=np.complex128),
        absorption=np.asarray(alpha, dtype=np.float64),
        transfer_matrix=tm,
        layers=tuple(layers),
    )


def diffuse_field_absorption(
    frequency: ArrayLike,
    layers: list[Layer] | tuple[Layer, ...],
    *,
    angle_limit: float = np.pi / 2.0,
    quadrature_points: int = 64,
    termination: str | complex | ArrayLike = "rigid",
    speed_of_sound: float = _SPEED_OF_SOUND,
    air_density: float = _AIR_DENSITY,
    viscosity: float = _AIR_VISCOSITY,
) -> DiffuseFieldAbsorptionResult:
    """Random-incidence absorption by the Paris integral (Mechel Sect. D.5).

    ``alpha_dif = (2 / sin^2 theta_lim) * int_0^theta_lim alpha(theta)
    cos(theta) sin(theta) d(theta)`` (Mechel 2e Sect. D.5 Eq. (9)), evaluated
    with fixed-order Gauss-Legendre quadrature over the bulk-reacting
    ``alpha(theta)`` of :func:`layered_absorber` (Sect. D.6 notes the bulk
    integral generally must be evaluated numerically). Some references
    truncate the integral at 75-87 degrees instead of 90 (Sect. D.5); set
    ``angle_limit`` accordingly.

    :param frequency: Frequency vector ``f``, in hertz.
    :param layers: Layer stack, as in :func:`layered_absorber`.
    :param angle_limit: Upper integration angle ``theta_lim``, in radians
        (0 < theta_lim <= pi/2; default pi/2).
    :param quadrature_points: Gauss-Legendre order (default 64).
    :param termination: As in :func:`layered_absorber`.
    :param speed_of_sound: Speed of sound ``c`` in air, in m/s.
    :param air_density: Air density ``rho``, in kg/m3.
    :param viscosity: Dynamic viscosity of air, in Pa s.
    :return: A :class:`DiffuseFieldAbsorptionResult`.
    """
    f = require_positive_array(frequency, "frequency")
    lim = float(angle_limit)
    if not 0.0 < lim <= np.pi / 2.0:
        raise ValueError("'angle_limit' must satisfy 0 < angle_limit <= pi/2.")
    n = int(quadrature_points)
    if n < 2:
        raise ValueError("'quadrature_points' must be at least 2.")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    theta = 0.5 * lim * (nodes + 1.0)
    w = 0.5 * lim * weights
    total = np.zeros_like(f, dtype=np.float64)
    for th, wt in zip(theta, w):
        res = layered_absorber(
            f,
            layers,
            angle=float(th),
            termination=termination,
            speed_of_sound=speed_of_sound,
            air_density=air_density,
            viscosity=viscosity,
        )
        total += wt * res.absorption * np.cos(th) * np.sin(th)
    alpha_dif = 2.0 * total / np.sin(lim) ** 2
    return DiffuseFieldAbsorptionResult(
        frequency=f,
        absorption=np.asarray(alpha_dif, dtype=np.float64),
        angle_limit=lim,
    )


def statistical_absorption(
    normalized_impedance: ArrayLike,
    *,
    angle_limit: float = np.pi / 2.0,
) -> Real:
    """Closed-form Paris integral for a locally reacting plane.

    With the normalised surface admittance ``Z0 G = g1 + j g2 = 1/z``
    (Mechel 2e Sect. D.5 Eq. (10)):

    ``alpha_dif = (8 g1 / sin^2 T) [1 - cos T
    + ((g1^2 - g2^2)/g2)(arctan((1 + g1)/g2) - arctan((g1 + cos T)/g2))
    + g1 ln((g1^2 + g2^2 + 2 g1 cos T + cos^2 T)/(1 + g1^2 + g2^2 + 2 g1))]``

    reducing for ``T = pi/2`` to Eq. (4) and, for real admittance, to the
    printed ``g2 = 0`` special case. The maximum over passive impedances is
    0.951 (the published bound for locally reacting absorbers, Sect. D.5).

    :param normalized_impedance: Normalised surface impedance
        ``z = Zs / (rho c)`` (complex scalar or array), with ``Re(z) > 0``.
    :param angle_limit: Upper integration angle ``theta_lim``, in radians
        (0 < theta_lim <= pi/2; default pi/2).
    :return: Statistical absorption coefficient ``alpha_dif``.
    """
    z = np.asarray(normalized_impedance, dtype=np.complex128)
    if np.any(z.real <= 0.0):
        raise ValueError("'normalized_impedance' must have a positive real part.")
    lim = float(angle_limit)
    if not 0.0 < lim <= np.pi / 2.0:
        raise ValueError("'angle_limit' must satisfy 0 < angle_limit <= pi/2.")
    g = 1.0 / z
    g1 = g.real
    g2 = g.imag
    cos_t = np.cos(lim)
    sin2_t = np.sin(lim) ** 2
    log_term = np.log(
        (g1**2 + g2**2 + 2.0 * g1 * cos_t + cos_t**2)
        / (1.0 + g1**2 + g2**2 + 2.0 * g1)
    )
    # Mechel prints (g1^2 - g2^2)/g2 * [arctan((1+g1)/g2) -
    # arctan((g1+cosT)/g2)], which cancels catastrophically as g2 -> 0
    # (a difference of two values near +-pi/2 amplified by 1/g2). With
    # a = 1 + g1 > 0 and b = g1 + cosT > 0 the identity
    # arctan(a/g2) - arctan(b/g2) = arctan(g2 (a - b) / (g2^2 + a b))
    # (valid because (a/g2)(b/g2) > 0) evaluates the same quantity
    # stably for every non-zero g2. Expanding arctan(x/g2) about
    # g2 = 0 (arctan(x/g2) = sgn(g2) pi/2 - g2/x + O(g2^3)) gives the
    # exact limit of the whole term,
    # g1^2 (1 - cos T) / ((g1 + cos T)(1 + g1)), with an O(g2^2)
    # truncation error - far below double precision at the switch
    # threshold, while the direct form is stable for every larger |g2|.
    a = 1.0 + g1
    b = g1 + cos_t
    near_real = np.abs(g2) < 1e-30
    g2_safe = np.where(near_real, 1.0, g2)
    atan_term = np.where(
        near_real,
        g1**2 * (1.0 - cos_t) / (b * a),
        (g1**2 - g2_safe**2)
        / g2_safe
        * np.arctan(g2_safe * (a - b) / (g2_safe**2 + a * b)),
    )
    alpha = 8.0 * g1 / sin2_t * (1.0 - cos_t + atan_term + g1 * log_term)
    return np.asarray(alpha, dtype=np.float64)

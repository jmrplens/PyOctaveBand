#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Porous-material models and resonant sheet impedances.

Two complementary building blocks, all in the :math:`e^{+j \omega t}`
time convention with the forward wave carried by :math:`e^{-j k x}` (so a
passive medium has :math:`\operatorname{Im}(k) < 0`):

* **Equivalent-fluid models** for the characteristic impedance ``Zc`` and the
  complex wavenumber ``k`` of a rigid-frame porous material:

  - the one-parameter **Delany-Bazley** power law in the absorber variable
    :math:`X = \rho_0 f / \sigma` (Mechel, *Formulas of Acoustics* 2e,
    Sect. G.11 Eqs. (1)-(2); Bies, Hansen & Howard, *Engineering Noise
    Control* 5e, Appendix D Eqs. (D.22)-(D.23) and Table D.1; Hopkins,
    *Sound Insulation*, Eqs. (1.171)-(1.174)), stated valid for
    :math:`0.01 < X < 1.0` and porosity close to one. Table D.1 also
    provides coefficient sets fitted to polyester (Garai & Pompoli 2005)
    and to foams (Dunn & Davern 1986, Wu 1988), exposed here as presets.
  - the **Miki** modification, regressed on the same Delany-Bazley data under
    a positive-real (passivity) constraint so the model stays well behaved
    below the fit range (Miki 1990, *J. Acoust. Soc. Jpn (E)* 11(1),
    Eqs. (30)-(34), in the variable :math:`f / \sigma`).
  - the five-parameter **Johnson-Champoux-Allard (JCA)** semi-phenomenological
    model with flow resistivity, porosity, tortuosity and the viscous/thermal
    characteristic lengths (Cox & D'Antonio, *Acoustic Absorbers and
    Diffusers* 3e, Eqs. (6.19)-(6.25); Attenborough & Van Renterghem,
    *Predicting Outdoor Sound* 2e, Eqs. (5.13)-(5.14)). The returned
    equivalent-fluid density and bulk modulus are the surface-normalised
    quantities (they absorb the porosity), so
    :math:`Z_\mathrm{c} = \sqrt{\rho_\mathrm{e} K_\mathrm{e}}` and
    :math:`k = \omega \sqrt{\rho_\mathrm{e} / K_\mathrm{e}}` hold for every model.
  - the **limp-frame** correction of any of the three rigid-frame models
    (Allard & Atalla, *Propagation of Sound in Porous Media* 2e, Sect. 11.3.4,
    Eqs. (11.53)-(11.55), printed pp. 251-253): a light frame is dragged along
    by the pore fluid, so its inertia has to be carried by the equivalent
    fluid. Only the effective density changes; the bulk modulus is the
    rigid-frame one. See :func:`limp_frame` and
    :func:`decoupling_frequency`.

* **Resonant sheets**: the perforated-plate impedance
  uses the end-corrected air-plug mass and the visco-thermal surface
  resistance (Cox & D'Antonio Eqs. (7.6)/(7.12)/(7.21), end-correction
  variants of Table 7.1); the microperforated plate follows Maa's exact
  short-tube impedance (Maa 1998, *J. Acoust. Soc. Am.* 104(5), Eq. (2),
  with the Eq. (5) end corrections; reproduced as Cox & D'Antonio
  Eqs. (7.33)-(7.35) and built on the same Bessel kernel as Mechel
  Sect. G.3); the membrane is the limp surface
  mass :math:`j \omega m` (Cox & D'Antonio Eq. (7.14); Bies Eq. (D.96)).
  Each sheet is closed by the shallow-cavity resonance it is designed
  around, :func:`helmholtz_resonance_frequency` for a perforate and
  :func:`membrane_resonance_frequency` for a membrane.

The air all of them propagate through is described by :class:`Fluid`,
which carries the six quantities a visco-thermal model can need (speed of
sound, density, viscosity, Prandtl number, ratio of specific heats and static
pressure) with the values these models were published with. The narrow-channel
models of :mod:`~phonometry.materials.absorbers.slow_sound` and
:mod:`~phonometry.materials.diffusers.metadiffuser` take it as a single
argument.

These are the elements a multilayer absorber is assembled from; declaring a
stack of them and solving it with the transfer matrix is the subject of
:mod:`~phonometry.materials.absorbers.layered`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

from ..._internal.validation import (
    require_choice,
    require_non_negative,
    require_positive,
    require_positive_array,
)
from ..._internal.warnings import PhonometryWarning
from ...fluids import Fluid

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.axes import Axes

    from ..._internal.types import Real

Complex = NDArray[np.complex128]

#: Default speed of sound in fluid, in m/s (20 degC).
_SPEED_OF_SOUND = 343.0
#: Default air density, in kg/m3 (Bies 5e Appendix D: 1,205 at 20 degC).
_AIR_DENSITY = 1.205
#: Default dynamic viscosity of fluid, in Pa s (Cox & D'Antonio Eq. (7.13)).
_AIR_VISCOSITY = 1.84e-5
#: Default Prandtl number of air at 20 degC (``eta c_p / kappa``).
_PRANDTL_NUMBER = 0.71
#: Default ratio of specific heats of fluid.
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

#: Count of Delany-Bazley power-law coefficients ``C1..C8`` an explicit
#: ``coefficients`` tuple must supply: four for ``Zc`` and four for ``k``
#: (Mechel 2e Sect. G.11 Eqs. (1)-(2)), the length of the preset tuples above.
_DELANY_BAZLEY_COEFFICIENT_COUNT = 8

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
    "PUBLISHED_AIR",
    "PorousAbsorberWarning",
    "PorousMediumResult",
    "decoupling_frequency",
    "delany_bazley",
    "helmholtz_resonance_frequency",
    "johnson_champoux_allard",
    "limp_frame",
    "limp_frame_applicable",
    "membrane_impedance",
    "membrane_resonance_frequency",
    "microperforated_plate_impedance",
    "miki",
    "perforated_plate_impedance",
    "perforation_end_correction",
]


class PorousAbsorberWarning(PhonometryWarning):
    """Advisory for porous-model use outside the published fit range."""


# ---------------------------------------------------------------------------
# The air the models propagate through
# ---------------------------------------------------------------------------
#: The air the visco-thermal models were **published** with, not a measurement
#: of anyone's fluid. Johnson, Champoux and Allard fitted their closed forms
#: against these six values, so they are constants of the model in the same way
#: its exponents are: substituting better physics does not correct an error, it
#: changes the model. Most visibly, the Prandtl number here is 0,71 while air at
#: this state has 0,728, and that alone moves the characteristic impedance and
#: the wavenumber by 1,5 parts in a thousand.
#:
#: Frozen, shared by every model rather than rebuilt, and it stays in this
#: module because it belongs to these models and to nothing else.
PUBLISHED_AIR = Fluid(
    temperature_c=20.0,
    static_pressure_pa=_ATMOSPHERIC_PRESSURE,
    composition={},
    model="Johnson-Champoux-Allard published constants (dry air at 20 degC)",
    validity="",
    properties={
        "speed_of_sound": _SPEED_OF_SOUND,
        "density": _AIR_DENSITY,
        "viscosity": _AIR_VISCOSITY,
        "heat_capacity_ratio": _HEAT_CAPACITY_RATIO,
        # Carried rather than derived: this is the model's fitted constant, and
        # Fluid.prandtl_number would close eta/(rho alpha_t) from an air that
        # does not have it.
        "prandtl_number": _PRANDTL_NUMBER,
    },
)


# ---------------------------------------------------------------------------
# Equivalent-fluid models of porous materials
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PorousMediumResult:
    r"""Equivalent-fluid characterisation of a porous material.

    All arrays share the shape of ``frequency``. ``characteristic_impedance``
    is the complex characteristic impedance ``Zc`` in Pa s/m as seen from the
    material surface, ``wavenumber`` the complex wavenumber ``k`` in rad/m
    (:math:`\operatorname{Im}(k) < 0` for the :math:`e^{+j \omega t}`
    convention), ``effective_density`` :math:`= Z_\mathrm{c} k / \omega` and
    ``bulk_modulus`` :math:`= Z_\mathrm{c} \omega / k` the surface-normalised
    equivalent-fluid density and bulk modulus, so that
    :math:`Z_\mathrm{c} = \sqrt{\rho_\mathrm{e} K_\mathrm{e}}` and
    :math:`k = \omega \sqrt{\rho_\mathrm{e} / K_\mathrm{e}}` for every model.
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
        r"""Characteristic impedance normalised by :math:`\rho c` of fluid."""
        rc = self.air_density * self.speed_of_sound
        return np.asarray(self.characteristic_impedance / rc, dtype=np.complex128)

    @property
    def normalized_wavenumber(self) -> Complex:
        r"""Wavenumber normalised by the free-air wavenumber
        :math:`k_0 = \omega / c`.
        """
        k0 = 2.0 * np.pi * self.frequency / self.speed_of_sound
        return np.asarray(self.wavenumber / k0, dtype=np.complex128)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the normalised ``Zc`` and ``k`` components against frequency.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_porous_medium

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
    fluid: Fluid = PUBLISHED_AIR,
) -> PorousMediumResult:
    r"""Delany-Bazley one-parameter porous model (power laws in ``X``).

    :math:`Z_\mathrm{c} = \rho c (1 + C_1 X^{-C_2} - j C_3 X^{-C_4})` and
    :math:`k = (\omega/c)(1 + C_5 X^{-C_6} - j C_7 X^{-C_8})` with
    :math:`X = \rho f / \sigma`
    (Mechel 2e Sect. G.11 Eqs. (1)-(2); Bies 5e Eqs. (D.22)-(D.23) with the
    Table D.1 coefficients; Hopkins Eqs. (1.171)-(1.173)). A
    :class:`PorousAbsorberWarning` is raised when any ``X`` leaves the stated
    :math:`0.01 < X < 1.0` validity range (Hopkins Eq. (1.174)); the values
    are still returned.

    :param frequency: Frequency vector ``f``, in hertz.
    :param flow_resistivity: Airflow resistivity ``sigma``, in Pa s/m2.
    :param coefficients: Preset name from :data:`DELANY_BAZLEY_COEFFICIENTS`
        (``"delany_bazley"`` rockwool/fibreglass default, ``"garai_pompoli"``
        polyester, ``"dunn_davern"`` / ``"wu"`` foams) or an explicit
        ``(C1..C8)`` tuple.
    :param speed_of_sound: Speed of sound ``c`` in fluid, in m/s.
    :param air_density: Air density ``rho``, in kg/m3.
    :return: A :class:`PorousMediumResult`.
    """
    f = require_positive_array(frequency, "frequency")
    sigma = require_positive(flow_resistivity, "flow_resistivity")
    c0 = fluid.speed_of_sound
    rho0 = fluid.density
    if isinstance(coefficients, str):
        try:
            coeffs = DELANY_BAZLEY_COEFFICIENTS[coefficients]
        except KeyError:
            options = ", ".join(sorted(DELANY_BAZLEY_COEFFICIENTS))
            msg = f"unknown coefficient preset {coefficients!r}; options: {options}."
            raise ValueError(msg) from None
        model = f"delany_bazley[{coefficients}]"
    else:
        coeffs = tuple(float(v) for v in coefficients)
        model = "delany_bazley[custom]"
    if len(coeffs) != _DELANY_BAZLEY_COEFFICIENT_COUNT:
        msg = "'coefficients' must provide exactly 8 values C1..C8."
        raise ValueError(msg)
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
    fluid: Fluid = PUBLISHED_AIR,
) -> PorousMediumResult:
    r"""Miki (1990) positive-real modification of the Delany-Bazley model.

    In the variable :math:`Y = f / \sigma` (Miki 1990, Eqs. (30)-(34)):
    :math:`Z_\mathrm{c} = \rho c (1 + 0.070 Y^{-0.632} - j 0.107 Y^{-0.632})` and,
    from the propagation constant :math:`\gamma = \alpha + j \beta` via
    :math:`k = \beta - j \alpha`,
    :math:`k = (\omega/c)(1 + 0.109 Y^{-0.618} - j 0.160 Y^{-0.618})`. The
    regression was constrained to be positive real, so the surface impedance
    of a hard-backed layer keeps a non-negative real part even below the
    Delany-Bazley range; a :class:`PorousAbsorberWarning` still flags
    ``Y`` outside the fit range :math:`0.01 < f/\sigma < 1.0` (paper
    Sect. 4.1).

    :param frequency: Frequency vector ``f``, in hertz.
    :param flow_resistivity: Airflow resistivity ``sigma``, in Pa s/m2.
    :param speed_of_sound: Speed of sound ``c`` in fluid, in m/s.
    :param air_density: Air density ``rho``, in kg/m3.
    :return: A :class:`PorousMediumResult`.
    """
    f = require_positive_array(frequency, "frequency")
    sigma = require_positive(flow_resistivity, "flow_resistivity")
    c0 = fluid.speed_of_sound
    rho0 = fluid.density
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
    fluid: Fluid = PUBLISHED_AIR,
) -> PorousMediumResult:
    r"""Johnson-Champoux-Allard five-parameter rigid-frame model.

    Effective density (Cox & D'Antonio 3e, Eq. (6.19)):

    .. math::

       \rho_\mathrm{e} = \frac{T \rho}{\phi} \left[1
       + \frac{\sigma \phi}{j \omega \rho T}
       \sqrt{1 + \frac{4 j T^2 \eta \rho \omega}{\sigma^2 L^2 \phi^2}}
       \right]

    and effective bulk modulus (Eq. (6.20)):

    .. math::

       K_\mathrm{e} = \frac{\gamma P_0 / \phi}{\gamma - (\gamma - 1) \left[1
       + \frac{8 \eta}{j {L'}^2 \mathrm{Pr}\, \omega \rho}
       \sqrt{1 + \frac{j \rho \omega \mathrm{Pr}\, {L'}^2}{16 \eta}}
       \right]^{-1}}

    with tortuosity ``T``, porosity ``phi``, viscous/thermal characteristic
    lengths ``L`` / ``L'``; then :math:`Z_\mathrm{c} = \sqrt{K_\mathrm{e} \rho_\mathrm{e}}` and
    :math:`k = \omega \sqrt{\rho_\mathrm{e} / K_\mathrm{e}}` (Eqs. (6.24)-(6.25)). Both
    quantities are surface-normalised (the :math:`1/\phi` factors are
    included). The model has the exact limits
    :math:`j \omega \rho_\mathrm{e} \to \sigma` as :math:`\omega \to 0` and
    :math:`\rho_\mathrm{e} \to (T \rho / \phi)(1 + (1 - j) \delta_v / L)` as
    :math:`\omega \to \infty` (Johnson et al. 1987), pinned in the tests.

    :param frequency: Frequency vector ``f``, in hertz.
    :param flow_resistivity: Airflow resistivity ``sigma``, in Pa s/m2.
    :param porosity: Open porosity ``phi`` (0 < phi <= 1).
    :param tortuosity: High-frequency tortuosity :math:`T = \alpha_\infty`
        (>= 1).
    :param viscous_length: Viscous characteristic length ``L``, in metres.
    :param thermal_length: Thermal characteristic length ``L'``, in metres
        (physically :math:`L' \ge L`).
    :param speed_of_sound: Speed of sound ``c`` in fluid, in m/s.
    :param air_density: Air density ``rho``, in kg/m3.
    :param viscosity: Dynamic viscosity ``eta`` of fluid, in Pa s.
    :param prandtl_number: Prandtl number ``Pr`` of fluid.
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
        msg = "'tortuosity' must be >= 1."
        raise ValueError(msg)
    lam_v = require_positive(viscous_length, "viscous_length")
    lam_t = require_positive(thermal_length, "thermal_length")
    c0 = fluid.speed_of_sound
    rho0 = fluid.density
    eta = fluid.viscosity
    pr = fluid.prandtl_number
    gamma = fluid.heat_capacity_ratio
    p0 = fluid.static_pressure_pa

    omega = 2.0 * np.pi * f
    # Effective density, Cox & D'Antonio Eq. (6.19).
    g_v = np.sqrt(
        1.0 + 4.0j * t_inf**2 * eta * rho0 * omega / (sigma**2 * lam_v**2 * phi**2)
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
    r"""Zwikker-Kosten decoupling frequency ``Fd`` of a porous frame.

    :math:`F_\mathrm{d} = \sigma \phi^2 / (2 \pi \rho_1)` (Allard & Atalla 2e,
    Sect. 11.3.4, printed p. 251; the same closed form as their Eq. (6.90),
    printed p. 126).
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
    r"""Whether the limp-frame model may be used, by published rule of thumb.

    Both published criteria compare the bulk modulus of the frame *in vacuum*
    ``K_c`` with that of the fluid in the pores ``K_f`` (Allard & Atalla 2e,
    printed pp. 253-254): Beranek (1947) requires
    :math:`\lvert K_c/K_\mathrm{f} \rvert < 0.05`, and the frame structural
    interaction study of Doutres et al. (2007) relaxes it to
    :math:`\lvert K_c/K_\mathrm{f} \rvert < 0.2`. With ``K_f`` taken as the
    isothermal bulk modulus of fluid, :math:`P_0 = 101.3` kPa, the relaxed
    criterion is the book's statement that
    "the limp model is applicable for materials having a bulk modulus lower
    than 20 kPa". Neither criterion accounts for boundary or mounting
    conditions, and the book notes that a thin light foam decoupled from a
    vibrating structure by an air gap behaves limply well above the limit.

    :param frame_bulk_modulus: Bulk modulus of the frame in vacuum ``K_c``, in
        Pa (>= 0; pass ``abs(K_c)`` for a complex modulus).
    :param criterion: Key into :data:`LIMP_FRAME_CRITERIA`, ``"doutres"``
        (Default, 0,2) or ``"beranek"`` (0,05).
    :param fluid_bulk_modulus: Bulk modulus of the pore fluid ``K_f``, in Pa
        (Default: 101 325, the isothermal value for fluid).
    :return: ``True`` when :math:`\lvert K_c/K_\mathrm{f} \rvert` does not exceed
        the threshold.
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
    r"""Limp-frame correction of a rigid-frame equivalent fluid (A&A 11.3.4).

    A light frame (aeronautic-grade fibreglass, felts, screens) is dragged
    along by the pore fluid instead of standing still, and the rigid-frame
    models of :func:`delany_bazley`, :func:`miki` and
    :func:`johnson_champoux_allard` have no way to carry that inertia.
    Neglecting the stiffness of the frame altogether in the Biot mixed
    pressure-displacement formulation leaves an equivalent fluid with the same
    bulk modulus and a corrected effective density (Allard & Atalla 2e,
    Eqs. (11.53)-(11.55), printed pp. 252-253, after Panneton 2007):

    .. math::

       \rho_{\mathrm{limp}} =
       \frac{\rho_\mathrm{t} \rho_{\mathrm{eq}} - \rho_0^2}
       {\rho_\mathrm{t} + \rho_{\mathrm{eq}} - 2 \rho_0}

    with ``rho_eq`` the rigid-frame effective density of *medium*, ``rho0``
    the density of the pore fluid and :math:`\rho_\mathrm{t} = \rho_1 + \phi \rho_0`
    the apparent total density of the material. What anchors this
    expression is the printed
    equation itself, transcribed term by term; Allard & Atalla tabulate no
    computed limp density anywhere, so there are no published digits to check
    against. The book also states two exact limits in prose, and both are
    verified, but they are weaker than they look: neither pins the
    :math:`\rho_0^2` and :math:`2 \rho_0` terms, since a sign-flipped
    variant of Eq. (11.55) satisfies both of them (and even the
    :math:`1/\rho_1` decay of the heavy-frame residual).
    They corroborate the transcription rather than determine it:

    * **heavy frame**: as :math:`\rho_1 \to \infty` the correction vanishes
      and the rigid-frame result is recovered (the book's own reading of
      Eq. (11.55));
    * **low frequency**: since
      :math:`\rho_{\mathrm{eq}} \to \sigma / (j \omega)` as
      :math:`\omega \to 0` (Eq. (5.37)),
      :math:`\rho_{\mathrm{limp}} \to \rho_\mathrm{t}`, a finite real density, where
      the rigid-frame model diverges. The rigid frame forbids rigid-body
      motion of the sample; the limp one allows it, which is why the two
      differ mainly
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
    r"""End-correction factor ``delta`` of a circular perforation.

    The Fok-function interaction correction for circular holes (Cox &
    D'Antonio 3e, Table 7.1, Nesterov row; no open-area limit):

    .. math::

       \delta = 0.85 (1 - 1.47 \varepsilon^{1/2} + 0.47 \varepsilon^{3/2})

    Each orifice end adds :math:`\delta a` of air-plug length, and
    :math:`\delta \to 0.85` for an isolated hole.

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
    fluid: Fluid = PUBLISHED_AIR,
) -> Complex:
    r"""Transfer impedance of a rigid perforated plate with circular holes.

    Acoustic mass with both end corrections and the boundary-layer term
    (Cox & D'Antonio 3e, Eq. (7.6)):

    .. math::

       m = \frac{\rho}{\varepsilon} \left[t + 2 \delta a
       + \sqrt{\frac{8 \nu}{\omega}} \left(1 + \frac{t}{2a}\right)\right]

    and visco-thermal surface resistance (Eq. (7.12)):

    .. math::

       r = \frac{\rho}{\varepsilon} \sqrt{8 \nu \omega}
       \left(1 + \frac{t}{2a}\right)

    giving :math:`z = r + j \omega m` (the series impedance added on top of
    the backing, Eq. (7.21)). Assumes hole radii well above the boundary-layer
    thickness; use :func:`microperforated_plate_impedance` for submillimetre
    holes.

    :param frequency: Frequency vector ``f``, in hertz.
    :param thickness: Plate thickness ``t``, in metres.
    :param hole_radius: Hole radius ``a``, in metres.
    :param open_area: Fractional open area ``eps`` (0..1).
    :param end_correction: End-correction factor ``delta`` per end; default
        :func:`perforation_end_correction` of ``eps``.
    :param air_density: Air density ``rho``, in kg/m3.
    :param viscosity: Dynamic viscosity ``eta`` of fluid, in Pa s.
    :return: Complex transfer impedance ``z``, in Pa s/m.
    """
    f = require_positive_array(frequency, "frequency")
    t = require_positive(thickness, "thickness")
    a = require_positive(hole_radius, "hole_radius")
    eps = require_positive(open_area, "open_area")
    if eps > 1.0:
        raise ValueError(_OPEN_AREA_MESSAGE)
    rho0 = fluid.density
    eta = fluid.viscosity
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
    fluid: Fluid = PUBLISHED_AIR,
) -> Complex:
    r"""Transfer impedance of a microperforated plate (Maa's exact model).

    The specific impedance of one submillimetre hole is the exact short-tube
    result (Maa 1998, Eq. (2); reproduced as Cox & D'Antonio 3e Eq. (7.33)
    and the same Bessel kernel as Mechel 2e Sect. G.3):

    .. math::

       z_1 = j \omega \rho t \left[1 - \frac{2}{x \sqrt{-j}}
       \frac{J_1(x \sqrt{-j})}{J_0(x \sqrt{-j})}\right]^{-1}

    with the perforate constant :math:`x = a \sqrt{\rho \omega / \eta}`.
    Dividing by the open area and adding Maa's Eq. (5) end corrections - the
    Rayleigh/Ingard surface resistance
    :math:`\sqrt{2 \omega \rho \eta} / (2 \varepsilon)` and the piston
    end-correction reactance
    :math:`j \omega \rho (2 \delta a) / \varepsilon` (:math:`0.85 d` total
    for the default :math:`\delta = 0.85` per end) - gives the sheet
    transfer impedance (Cox & D'Antonio Eq. (7.35)).

    :param frequency: Frequency vector ``f``, in hertz.
    :param thickness: Plate thickness ``t``, in metres.
    :param hole_radius: Hole radius ``a``, in metres (submillimetre for a
        genuine microperforated design).
    :param open_area: Fractional open area ``eps`` (0..1).
    :param end_correction: End-correction factor ``delta`` per end
        (default 0.85, the isolated-orifice value used by Maa).
    :param air_density: Air density ``rho``, in kg/m3.
    :param viscosity: Dynamic viscosity ``eta`` of fluid, in Pa s.
    :return: Complex transfer impedance ``z``, in Pa s/m.
    """
    f = require_positive_array(frequency, "frequency")
    t = require_positive(thickness, "thickness")
    a = require_positive(hole_radius, "hole_radius")
    eps = require_positive(open_area, "open_area")
    if eps > 1.0:
        raise ValueError(_OPEN_AREA_MESSAGE)
    delta = require_non_negative(end_correction, "end_correction")
    rho0 = fluid.density
    eta = fluid.viscosity
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
    r"""Transfer impedance of a limp impervious membrane.

    :math:`z = r + j \omega m` - the surface-mass reactance (Cox & D'Antonio
    3e, Eq. (7.14); Bies 5e Eq. (D.96)) plus an optional empirical
    resistance for the internal/fixing losses.

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
    r"""Resonance of a perforated sheet over a shallow cavity (closed form).

    :math:`f_0 = (c / 2 \pi) \sqrt{\varepsilon / (t' d)}` with the
    end-corrected plug length :math:`t' = t + 2 \delta a` (Cox & D'Antonio
    3e, Eqs. (7.4)/(7.6), valid for :math:`k d \ll 1`).

    :param cavity_depth: Cavity depth ``d``, in metres.
    :param plate_thickness: Plate thickness ``t``, in metres.
    :param hole_radius: Hole radius ``a``, in metres.
    :param open_area: Fractional open area ``eps`` (0..1).
    :param end_correction: End-correction factor ``delta`` per end; default
        :func:`perforation_end_correction` of ``eps``.
    :param speed_of_sound: Speed of sound ``c`` in fluid, in m/s.
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
    fluid: Fluid = PUBLISHED_AIR,
) -> float:
    r"""Mass-spring resonance of a membrane over a shallow cavity.

    :math:`f_0 = (1 / 2 \pi) \sqrt{\rho c^2 / (m d)}` for an adiabatic air
    spring - numerically the classical :math:`f_0 = 60 / \sqrt{m d}` (Cox &
    D'Antonio 3e, Eq. (7.9)). With ``isothermal=True`` the spring stiffness
    drops by ``gamma``, giving :math:`\sim 50 / \sqrt{m d}` (Eq. (7.10)),
    the porous-filled cavity case below about 500 Hz.

    :param surface_density: Membrane mass per unit area ``m``, in kg/m2.
    :param cavity_depth: Cavity depth ``d``, in metres.
    :param isothermal: Use the isothermal air-spring stiffness.
    :param speed_of_sound: Speed of sound ``c`` in fluid, in m/s.
    :param air_density: Air density ``rho``, in kg/m3.
    :return: Resonance frequency ``f0``, in hertz.
    """
    m = require_positive(surface_density, "surface_density")
    d = require_positive(cavity_depth, "cavity_depth")
    c0 = fluid.speed_of_sound
    rho0 = fluid.density
    stiffness = rho0 * c0**2 / d
    if isothermal:
        stiffness /= _HEAT_CAPACITY_RATIO
    return float(np.sqrt(stiffness / m) / (2.0 * np.pi))
